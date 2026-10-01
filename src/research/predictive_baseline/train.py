"""One-GPU fits, exact epoch-boundary resume, and local execution gates."""
import json
import math
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.execution import allocation_deadline
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_rows
from src.experiment_runner.wandb_tracking import online_training
from src.research.encoder_context.geometry import graph
from .data import load, output
from .model import Predictor, initialize


def save(path, state):
    temporary = path.with_suffix('.building')
    torch.save(state, temporary)
    temporary.replace(path)


@torch.no_grad()
def predict(model, positions, batch, *, embeddings=False):
    model.eval()
    device = next(model.parameters()).device
    predictions, states = [], []
    for start in range(0, len(positions), batch):
        x = torch.tensor(positions[start:start+batch], device=device)
        z = model.encode(x)
        predictions.append(model.head(z).cpu().numpy())
        if embeddings:
            states.append(z.cpu().numpy())
    pred = np.concatenate(predictions)
    if not np.isfinite(pred).all():
        raise FloatingPointError('Nonfinite model prediction')
    return (pred, np.concatenate(states)) if embeddings else pred


def score(model, data, ids, batch):
    pred = predict(model, data['positions'][ids], batch)
    error = ((pred - data['target'][ids])**2).sum(-1)
    return float(np.average(error, weights=data['weights'][ids]))


def preflight(c):
    data, manifest = load(c)
    device = torch.device('cuda')
    torch.set_num_threads(4)
    torch.manual_seed(c['fit_seeds'][0])
    batch = c['training']['batch_size']
    if c['training']['microbatch'] != batch or batch != 256:
        raise ValueError('This protocol uses a full 256-observation microbatch')
    model = Predictor(c).to(device)
    initialize(model, data, c, device)
    ids = np.flatnonzero(data['roles'] == 'train')[:batch]
    x = torch.tensor(data['positions'][ids], device=device)
    target = torch.tensor(data['target'][ids], device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=c['training']['encoder_lr'])
    torch.cuda.reset_peak_memory_stats()
    durations = []
    for _ in range(4):
        model.train()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        began = time.monotonic()
        loss = (model(x) - target).square().sum(-1).mean()
        loss.backward()
        gradient = sum(float(p.grad.square().sum()) for p in model.encoder.parameters() if p.grad is not None)**.5
        if not math.isfinite(float(loss.detach())) or not math.isfinite(gradient) or gradient <= 0:
            raise FloatingPointError(f'Invalid encoder gradient/loss: {gradient}, {loss}')
        opt.step()
        torch.cuda.synchronize()
        durations.append(time.monotonic() - began)
    # This execution gate is local and never opens a W&B training run.
    model.eval()
    with torch.no_grad():
        original = model(x[:8])
        rotation, _ = torch.linalg.qr(torch.randn(3, 3, device=device))
        rotated = model(x[:8] @ rotation)
        invariant_error = float((rotated-original).abs().max())
        g = graph(x[:8], model.encoder)
        if not torch.equal(g['attrs'], (x[:8].norm(dim=-1) < 8).flatten().float()[:, None]):
            raise ValueError('Actual atom attributes violate geometry-only contract')
    if invariant_error > 1e-3:
        raise ValueError(f'Rotation gate failed: {invariant_error}')
    result = dict(state='complete', dataset=manifest['identity'], device=torch.cuda.get_device_name(),
        batch_size=batch, microbatch=batch, parameters=sum(p.numel() for p in model.parameters()),
        encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
        encoder_gradient_norm=gradient, rotation_max_abs=invariant_error,
        seconds_per_update=float(np.median(durations[1:])),
        peak_gpu_memory_GiB=torch.cuda.max_memory_allocated()/2**30,
        updates_per_epoch=math.ceil((data['roles'] == 'train').sum()/batch),
        online_runs_created=0)
    write_json(output(c) / 'technical/preflight.json', result)
    print(json.dumps(result), flush=True)
    return result


def fit(c, index):
    data, manifest = load(c)
    gate = json.loads((output(c) / 'technical/preflight.json').read_text())
    if gate['state'] != 'complete' or gate['dataset'] != manifest['identity']:
        raise ValueError('Matching numerical preflight is required before training')
    contract = check_metric_docs(family='predictive_baseline')['predictive_baseline']
    seed = c['fit_seeds'][index]
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device('cuda')
    cfg = c['training']
    batch = cfg['batch_size']
    if cfg['microbatch'] != batch or cfg['precision'] != 'float32':
        raise ValueError('Unimplemented microbatch/precision protocol')
    root = output(c) / 'analyses' / f'joint-seed-{seed}'
    tech = root / 'technical'
    tech.mkdir(parents=True, exist_ok=True)
    binding = dict(config=c, dataset=manifest['identity'], seed=seed, implementation=contract)
    identity = digest(binding)
    write_json(tech / 'identity.json', dict(identity=identity, binding=binding))
    model = Predictor(c).to(device)
    optimizer = torch.optim.AdamW([
        dict(params=model.encoder.parameters(), lr=cfg['encoder_lr']),
        dict(params=model.head.parameters(), lr=cfg['head_lr'])], weight_decay=cfg['weight_decay'])
    train_ids = np.flatnonzero(data['roles'] == 'train')
    selection = np.flatnonzero(data['roles'] == 'selection')
    probability = data['weights'][train_ids] / data['weights'][train_ids].sum()
    steps = math.ceil(len(train_ids) / batch)
    target = torch.tensor(np.asarray(data['target']), device=device)
    positions = torch.tensor(np.asarray(data['positions']), device=device)
    history, best, best_epoch, start = [], math.inf, 0, 1
    if (tech / 'last.pt').exists():
        last = torch.load(tech / 'last.pt', map_location=device, weights_only=False)
        if last['identity'] != identity:
            raise ValueError('Resume config, input or implementation changed')
        model.load_state_dict(last['model'])
        optimizer.load_state_dict(last['optimizer'])
        rng.bit_generator.state = last['rng']
        torch.set_rng_state(last['torch_rng'].cpu())
        torch.cuda.set_rng_state(last['cuda_rng'].cpu())
        history, best, best_epoch, start = last['history'], last['best'], last['best_epoch'], last['epoch']+1
    else:
        initialize(model, data, c, device)
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGUSR1, lambda *_: stop.__setitem__(0, True))
    deadline = allocation_deadline(reserve_seconds=300)
    run_id = 'pb274-' + identity[:16]
    with online_training(c['wandb'], run_id=run_id, name=f'Predictive future 274 | MACE128 | {seed}',
            config=binding, folder=tech, receipt_path=tech/'wandb.json', job_type='encoder',
            group='predictive-baseline-al480-274', receipt_fields=dict(identity=identity)) as run:
        if start == 1:
            best = score(model, data, selection, batch)
            save(tech/'best.pt', dict(identity=identity, model=model.state_dict(), epoch=0,
                selection_feature_error=best, config=c, dataset=manifest['identity']))
        for epoch in range(start, cfg['epochs']+1):
            if epoch-best_epoch > cfg['patience']:
                break
            if stop[0] or time.time() > deadline:
                raise TimeoutError('Allocation ending; resume the saved epoch boundary with the same frozen code/config')
            began = time.monotonic()
            ids = rng.choice(train_ids, steps*batch, replace=True, p=probability)
            losses = []
            model.train()
            for offset in range(0, len(ids), batch):
                ix = torch.tensor(ids[offset:offset+batch], device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = (model(positions[ix]) - target[ix]).square().sum(-1).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite training loss at epoch {epoch}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
            validation = score(model, data, selection, batch)
            if validation < best:
                best, best_epoch = validation, epoch
                save(tech/'best.pt', dict(identity=identity, model=model.state_dict(), epoch=epoch,
                    selection_feature_error=best, config=c, dataset=manifest['identity']))
            row = dict(epoch=epoch, train_feature_error=float(np.mean(losses)),
                       selection_feature_error=validation, selection_feature_nll=.5*(validation+target.shape[1]*math.log(2*math.pi)),
                       best_epoch=best_epoch, epoch_seconds=time.monotonic()-began)
            history.append(row)
            save(tech/'last.pt', dict(identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
                epoch=epoch, best=best, best_epoch=best_epoch, history=history, rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()))
            write_json(tech/'progress.json', dict(state='running', **row, best=best, seed=seed))
            run.log({k:v for k,v in row.items() if k != 'epoch_seconds'}, step=epoch)
            print(json.dumps(row), flush=True)
        chosen = torch.load(tech/'best.pt', map_location=device, weights_only=False)
        model.load_state_dict(chosen['model'])
        prediction, z = predict(model, data['positions'], batch, embeddings=True)
        np.savez_compressed(tech/'predictions.npz', prediction=prediction, z=z,
                            parent=data['parent'], atom_ids=data['atom_ids'])
        # All stability evaluations are post-selection and cannot choose a model.
        eval_ids = np.flatnonzero(data['roles'] == 'test')
        audit_rng = np.random.default_rng(c['target']['seed'])
        noise_rows = []
        for sigma in c['noise_levels_A']:
            xyz = data['positions'][eval_ids].copy()
            noise = audit_rng.normal(scale=sigma, size=xyz.shape).astype(np.float32)
            xyz += noise - noise[:, :1]
            perturbed = predict(model, xyz, batch)
            sensitivity = ((perturbed-prediction[eval_ids])**2).sum(1)
            noise_rows.append(dict(noise_A=sigma, feature_change=float(np.average(sensitivity, weights=data['weights'][eval_ids]))))
        write_metric_rows(history, root, family='predictive_baseline', name='learning')
        write_metric_rows(noise_rows, root, family='predictive_baseline', name='noise-response')
        run.summary.update(dict(selected_epoch=best_epoch, selected_feature_error=best,
                               target_identity=manifest['identity'], checkpoint_sha256=sha(tech/'best.pt')))
    write_json(tech/'complete.json', dict(state='complete', seed=seed, identity=identity, selected_epoch=best_epoch,
        selection_feature_error=best, epochs=len(history), checkpoint_sha256=sha(tech/'best.pt'),
        predictions_sha256=sha(tech/'predictions.npz')))
