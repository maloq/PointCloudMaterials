"""Relaxed-anchor supervision of a jointly trained temporal MACE transformer."""

from datetime import datetime
import json
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np
from sklearn.decomposition import PCA
import torch
from torch import nn

from src.data_utils.mace_history import Histories, prepare
from src.data_utils.temporal_campaign import write_json
from src.models.encoders.mace_temporal import PretrainedMACETemporalEncoder
from src.training_methods.mace_logging import flatten_metrics, save_checkpoint, start_wandb
from src.training_methods.mace_objective import make_scheduler, variance_covariance


class TemporalLearner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = PretrainedMACETemporalEncoder(**cfg['encoder'])
        self.tda = nn.Sequential(nn.Linear(self.encoder.invariant_dim, 256), nn.SiLU(),
                                 nn.Linear(256, cfg['tda_components']))


def objective(model, z, target, settings):
    variance, covariance = variance_covariance(z)
    tda = (model.tda(z) - target).square().mean()
    loss = settings['tda'] * tda + settings['variance'] * variance + settings['covariance'] * covariance
    return loss, dict(loss=loss, tda_mse=tda, variance_penalty=variance, covariance_penalty=covariance)


def encode(model, x, material, microbatch):
    return torch.cat([model.encoder(x[start:start+microbatch], material[start:start+microbatch])
                      for start in range(0, len(x), microbatch)])


def cached_step(model, batch, cfg):
    """Full-batch covariance with exact microbatch replay through MACE and attention."""
    x, target, material = batch
    size = cfg['microbatch_size']
    with torch.no_grad():
        z = encode(model, x, material, size)
    z.requires_grad_(True)
    loss, parts = objective(model, z, target, cfg['loss'])
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Nonfinite temporal objective for history batch {tuple(x.shape)}")
    loss.backward()
    for start in range(0, len(x), size):
        actual = model.encoder(x[start:start+size], material[start:start+size])
        actual.backward(z.grad[start:start+size])
    return {name: float(value.detach()) for name, value in parts.items()}


def fit_scaling(model, data, cfg, out):
    targets = data.raw_targets[data.indices['train'], 0]
    pca = PCA(n_components=cfg['tda_components'], svd_solver='full').fit(targets)
    scaling = dict(tda_mean=pca.mean_.astype(np.float32), tda_components=pca.components_.astype(np.float32),
                   tda_std=np.maximum(np.sqrt(pca.explained_variance_), 1e-5).astype(np.float32))
    rows = np.random.default_rng(cfg['seed']).choice(data.indices['train'], cfg['scaling_anchors'], replace=False)
    features = []
    with torch.no_grad():
        for start in range(0, len(rows), cfg['microbatch_size']):
            selected = torch.as_tensor(rows[start:start+cfg['microbatch_size']], device='cuda')
            x = data.clouds[selected, 0].float()
            m = data.material[selected].repeat_interleave(x.shape[1])
            features.append(model.encoder.mace.raw_features(x.flatten(0, 1), m).cpu().numpy())
    features = np.concatenate(features)
    scaling.update(feature_mean=features.mean(0), feature_std=np.maximum(features.std(0), 0.01))
    for name in ('feature_mean', 'feature_std'):
        getattr(model.encoder.mace, name).copy_(torch.from_numpy(scaling[name]))
    np.savez(out / 'scaling.npz', **scaling)
    write_json(out / 'scaling.json', dict(fit_split='train', tda_fit_anchors=len(targets),
        feature_fit_histories=len(rows), tda_explained_variance=float(pca.explained_variance_ratio_.sum())))
    data.set_scaling(scaling)
    return scaling


@torch.no_grad()
def validate(model, data, cfg):
    model.eval()
    rows = data.indices['val']
    x, target, material = data.get(rows)
    z = encode(model, x, material, cfg['microbatch_size'])
    _, parts = objective(model, z, target, cfg['loss'])
    result = {key: float(value) for key, value in parts.items()}
    prediction = model.tda(z)
    result['by_material'] = {name: dict(tda_mse=float((prediction[material == i] - target[material == i]).square().mean()))
                             for i, name in enumerate(('Al', 'Mg', 'Ta'))}
    return result


def preflight(model, data, cfg, out):
    rows = np.concatenate([data.indices['train'][data.materials[data.indices['train']] == m][:2] for m in range(3)])
    batch = data.get(rows)
    model.train()
    model.zero_grad(set_to_none=True)
    test_cfg = dict(cfg, microbatch_size=2)
    cached_step(model, batch, test_cfg)
    expected = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    z = model.encoder(batch[0], batch[2])
    loss, _ = objective(model, z, batch[1], cfg['loss'])
    loss.backward()
    error = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, expected[name], rtol=3e-3, atol=1e-4, msg=name)
            error = max(error, float((parameter.grad - expected[name]).abs().max()))
    model.zero_grad(set_to_none=True)
    model.eval()
    x = batch[0][:1].clone().requires_grad_(True)
    model.tda(model.encoder(x, batch[2][:1])).square().mean().backward()
    by_frame = x.grad.abs().sum(dim=(0, 2, 3))
    if not torch.isfinite(by_frame).all() or not (by_frame > 0).all():
        raise AssertionError(f"History frame gradients missing/nonfinite: {by_frame}")
    model.zero_grad(set_to_none=True)
    write_json(out / 'preflight.json', dict(state='passed', replay_gradient_max_error=error,
        coordinate_gradient_l1_by_frame=by_frame.tolist(), optimizer_updates_retained=0))


def train(model, data, cfg, out, wandb_run):
    if (out / 'initial.pt').exists():
        raise FileExistsError(f"Training would overwrite {out}; choose a new experiment output.")
    rng = np.random.default_rng(cfg['seed'])
    rows = data.indices['train']
    steps_per_epoch = (len(rows) + cfg['batch_size'] - 1) // cfg['batch_size']
    optimizer = torch.optim.AdamW([
        dict(params=[p for p in model.encoder.parameters() if p.requires_grad], lr=cfg['learning_rate']),
        dict(params=model.tda.parameters(), lr=cfg['head_learning_rate']),
    ], weight_decay=cfg['weight_decay'], fused=True)
    scheduler = make_scheduler(optimizer, cfg, steps_per_epoch)
    initial = validate(model, data, cfg)
    torch.save(dict(model=model.state_dict(), config=cfg, validation=initial), out / 'initial.pt')
    write_json(out / 'initial_validation.json', initial)
    wandb_run.log({'training_step': 0, **flatten_metrics('validation', initial)})
    best = float('inf')
    best_epoch = None
    step = seen = 0
    started = time.monotonic()
    with (out / 'training.jsonl').open('w', buffering=1) as log:
        for epoch in range(1, cfg['epochs'] + 1):
            model.train()
            order = rng.permutation(rows)
            totals = {}
            epoch_start = time.monotonic()
            for start in range(0, len(order), cfg['batch_size']):
                selected = order[start:start+cfg['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                parts = cached_step(model, data.get(selected), cfg)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                step += 1
                seen += len(selected)
                for key, value in parts.items():
                    totals[key] = totals.get(key, 0.0) + value * len(selected)
                if step % 5 == 0:
                    status = dict(state='training', epoch=epoch, step=step, total_steps=steps_per_epoch * cfg['epochs'],
                        elapsed_seconds=time.monotonic()-started, metrics=parts, wandb_url=wandb_run.url)
                    write_json(out / 'status.json', status)
                    print('TRAIN', json.dumps(status), flush=True)
                    wandb_run.log({'training_step': step, **flatten_metrics('train', dict(parts, epoch=epoch, gradient_norm=float(norm)))})
                if time.monotonic() - started > cfg['max_training_seconds']:
                    save_checkpoint(out / 'last.pt', model, optimizer, cfg, epoch, step, seen, initial)
                    raise TimeoutError('Temporal training exceeded its configured budget; partial checkpoint retained.')
            validation = validate(model, data, cfg)
            record = dict(epoch=epoch, step=step, train={k: v/len(rows) for k, v in totals.items()},
                          validation=validation, seconds=time.monotonic()-epoch_start)
            log.write(json.dumps(record, allow_nan=False)+'\n')
            print('VALIDATION', json.dumps(record), flush=True)
            wandb_run.log({'training_step': step, **flatten_metrics('validation', validation)})
            save_checkpoint(out / 'last.pt', model, optimizer, cfg, epoch, step, seen, validation)
            # Selection follows the scientific target; regularization is a training constraint.
            if validation['tda_mse'] < best:
                best, best_epoch = validation['tda_mse'], epoch
                torch.save(dict(model=model.state_dict(), config=cfg, epoch=epoch, step=step,
                                validation=validation), out / 'best.pt')
    write_json(out / 'training_summary.json', dict(state='complete', epochs=cfg['epochs'], steps=step,
        partial_epoch=False, anchor_exposures=seen, frame_exposures=seen*len(cfg['encoder']['frame_offsets_ps']),
        seconds=time.monotonic()-started, best_epoch=best_epoch, best_tda_mse=best,
        selection='minimum held-out relaxed-anchor TDA MSE', initial_validation=initial))


def run(cfg, stage):
    out = Path(cfg['output'])
    out.mkdir(parents=True, exist_ok=True)
    if stage in ('train', 'all') and (out / 'initial.pt').exists():
        raise FileExistsError(f"Training artifacts already exist in {out}; choose a fresh output.")
    if cfg['encoder']['dropout'] != 0:
        raise ValueError('Exact temporal gradient replay requires transformer dropout=0.')
    torch.set_num_threads(cfg['cpu_threads'])
    torch.manual_seed(cfg['seed'])
    def interrupted(signum, frame):
        raise InterruptedError(f'Temporal run interrupted by signal {signum}.')
    for sig in (signal.SIGTERM, signal.SIGALRM):
        signal.signal(sig, interrupted)
    seconds = int(datetime.fromisoformat(cfg['deadline']).timestamp() - time.time())
    if seconds <= 0:
        raise TimeoutError(f"Allocation safety deadline has passed: {cfg['deadline']}")
    signal.alarm(seconds)
    wandb_run = None
    from src.experiment_runner.tracking import tracked_run
    try:
        with tracked_run(out, kind='training' if stage in ('train', 'all') else 'analysis',
                         configs=[Path(sys.argv[sys.argv.index('--config')+1])], command=[sys.executable, *sys.argv]):
            if stage in ('prepare', 'all'):
                write_json(out / 'status.json', dict(state='preparing_histories'))
                prepare(cfg)
            if stage in ('train', 'preflight', 'all'):
                write_json(out / 'status.json', dict(state='scaling_and_preflight'))
                model = TemporalLearner(cfg).cuda()
                data = Histories(cfg)
                fit_scaling(model, data, cfg, out)
                preflight(model, data, cfg, out)
                if stage in ('train', 'all'):
                    wandb_run = start_wandb(cfg, out)
                    train(model, data, cfg, out, wandb_run)
                    wandb_run.finish()
                    wandb_run = None
                del model, data
                torch.cuda.empty_cache()
            if stage in ('analysis', 'all'):
                write_json(out / 'status.json', dict(state='analysis'))
                from src.analysis.mace_temporal import analyze
                analyze(cfg)
            write_json(out / 'status.json', dict(state='complete', stage=stage, completed_at=datetime.now().astimezone().isoformat()))
    except BaseException:
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
        write_json(out / 'status.json', dict(state='failed', traceback=traceback.format_exc()))
        raise
    finally:
        signal.alarm(0)
