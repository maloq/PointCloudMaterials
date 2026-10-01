"""Joint scientific fits online; matched frozen diagnostic heads remain local."""
from contextlib import nullcontext
import json
import math
import signal
import time
import numpy as np
import torch

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.checkpoints import TrainingState
from src.experiment_runner.execution import allocation_deadline
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_rows
from src.experiment_runner.wandb_tracking import online_training
from src.research.predictive_baseline.model import initialize
from src.research.predictive_baseline.train import save, predict
from .common import FAMILY, load, features, folder, output, cache, read
from .model import Forecast


def score(model, data, x, ids, batch):
    prediction = predict(model, x[ids], batch)
    error = ((prediction[:, :18]-data['target'][ids, :18])**2).sum(-1)
    return float(np.average(error, weights=data['weights'][ids]))


def fit(c, arm):
    data, manifest = load(c)
    binding = read(output(c)/'technical/binding.json')
    gate = read(output(c)/'technical/preflight.json')
    if gate['state'] != 'complete' or gate['binding'] != binding['identity']:
        raise ValueError('Follow-up numerical gate missing or changed')
    contract = check_metric_docs(family=FAMILY)[FAMILY]
    torch.set_num_threads(4)
    seed = arm['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    joint = arm['source'] == 'joint'
    device = torch.device('cuda' if joint else 'cpu')
    x = features(c, data, manifest, arm['source'])
    root = folder(c, arm)
    tech = root/'technical'
    tech.mkdir(parents=True, exist_ok=True)
    identity = digest(dict(binding=binding['identity'], arm=arm, implementation=contract))
    if (tech/'complete.json').exists():
        done = read(tech/'complete.json')
        if done['identity'] != identity or sha(tech/'predictions.npz') != done['predictions_sha256']:
            raise ValueError('Completed fit identity changed')
        return done
    model = Forecast(c, arm, data, x).to(device)
    cfg = c['training']
    batch = cfg['batch_size']
    if cfg['microbatch'] != batch or cfg['feature_arithmetic'] != 'float64':
        raise ValueError('Full batch and double feature arithmetic required')
    parameters = [dict(params=model.head.parameters(), lr=cfg['head_lr'])]
    if joint:
        parameters.insert(0, dict(params=model.encoder.parameters(), lr=cfg['encoder_lr']))
    optimizer = torch.optim.AdamW(parameters, weight_decay=cfg['weight_decay'])
    train_ids = np.flatnonzero(data['roles'] == 'train')
    selection = np.flatnonzero(data['roles'] == 'selection')
    probability = data['weights'][train_ids]/data['weights'][train_ids].sum()
    steps = math.ceil(len(train_ids)/batch)
    dimensions = model.head.dimensions
    target = torch.tensor(np.asarray(data['target'][:, :dimensions]), device=device, dtype=torch.float64)
    inputs = torch.tensor(x, device=device, dtype=torch.float32)
    history, best, best_epoch, start = [], math.inf, 0, 1
    if (tech/'last.pt').exists():
        last = TrainingState.read(tech/'last.pt', identity=identity, device=device)
        last.restore(model, optimizer, rng, restore_torch_rng=joint)
        last = last.payload
        history, best, best_epoch, start = last['history'], last['best'], last['best_epoch'], last['epoch']+1
    elif joint:
        initialize(model, data, c, device)
    context = dict(identity=identity, arm=arm, config=c, target_identity=manifest['identity'],
                   implementation=contract, parameters=sum(p.numel() for p in model.parameters()),
                   selector=c['selector'], feature_arithmetic='float64', network_precision='float32')
    write_json(tech/'identity.json', context)
    stopped = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    signal.signal(signal.SIGUSR1, lambda *_: stopped.__setitem__(0, True))
    deadline = allocation_deadline(reserve_seconds=120)
    tracking = online_training(c['wandb'], run_id='pbf-' + identity[:16],
        name=f'Future baseline | {arm["target"]} | {arm["variance"]} | {seed}', config=context,
        folder=tech, receipt_path=tech/'wandb.json', job_type='encoder',
        group='predictive-head-target-followup', receipt_fields=dict(identity=identity)) if joint else nullcontext(None)
    with tracking as run:
        if start == 1:
            best = score(model, data, x, selection, batch)
            save(tech/'best.pt', dict(identity=identity, model=model.state_dict(), arm=arm, config=c,
                epoch=0, selection_moment_error=best, target_identity=manifest['identity']))
        for epoch in range(start, cfg['epochs']+1):
            if epoch-best_epoch > cfg['patience']:
                break
            if stopped[0] or time.time() > deadline:
                raise TimeoutError('Resume the same frozen arm from the last completed epoch')
            began = time.monotonic()
            ids = rng.choice(train_ids, steps*batch, replace=True, p=probability)
            losses = []
            model.train()
            for begin in range(0, len(ids), batch):
                ix = torch.tensor(ids[begin:begin+batch], device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = (model(inputs[ix])-target[ix]).square().sum(-1).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite training feature error: {arm}, epoch {epoch}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
            validation = score(model, data, x, selection, batch)
            if validation < best:
                best, best_epoch = validation, epoch
                save(tech/'best.pt', dict(identity=identity, model=model.state_dict(), arm=arm, config=c,
                    epoch=epoch, selection_moment_error=best, target_identity=manifest['identity']))
            row = dict(epoch=epoch, train_active_feature_error=float(np.mean(losses)),
                selection_moment_error=validation, selection_moment_nll=.5*(validation+18*math.log(2*math.pi)),
                best_epoch=best_epoch, epoch_seconds=time.monotonic()-began)
            history.append(row)
            TrainingState.capture(model, optimizer, rng, identity=identity, capture_torch_rng=joint,
                history=history, epoch=epoch, best=best, best_epoch=best_epoch).save(tech/'last.pt')
            write_json(tech/'progress.json', dict(state='running', arm=arm, **row))
            if run is not None:
                run.log({k:v for k,v in row.items() if k != 'epoch_seconds'}, step=epoch)
            if epoch % 10 == 0 or epoch == 1:
                print(json.dumps(dict(arm=arm, **row)), flush=True)
        chosen = torch.load(tech/'best.pt', map_location=device, weights_only=False)
        model.load_state_dict(chosen['model'])
        prediction, z = predict(model, x, batch, embeddings=True)
        raw = prediction[:, :18]/model.head.scale[:18].cpu().numpy()+model.head.center[:18].cpu().numpy()
        minimum_variance = float(np.min(raw[:, 9:18]-raw[:, :9]**2))
        if arm['variance'] == 'nonnegative' and minimum_variance < 0:
            raise FloatingPointError(f'Constrained output violates second moments: {minimum_variance}')
        np.savez_compressed(tech/'predictions.npz', prediction=prediction,
            **(dict(z=z) if joint else {}), parent=data['parent'], atom_ids=data['atom_ids'])
        write_metric_rows(history, root, family=FAMILY, name='learning')
        # Additional readouts are local and frozen, including those associated with an online fit.
        if joint:
            from .evaluate import readouts
            readouts(c, arm, data, z, root)
        if run is not None:
            run.summary.update(dict(selected_epoch=best_epoch, selected_moment_error=best,
                minimum_normalized_variance=minimum_variance, checkpoint_sha256=sha(tech/'best.pt')))
    record = dict(state='complete', identity=identity, target_identity=manifest['identity'], arm=arm,
        epochs=len(history), selected_epoch=best_epoch, selection_moment_error=best,
        minimum_normalized_variance=minimum_variance, predictions_sha256=sha(tech/'predictions.npz'),
        checkpoint_sha256=sha(tech/'best.pt'), online_training=joint)
    write_json(tech/'complete.json', record)
    write_json(tech/'progress.json', record)
    return record


def preflight(c):
    data, manifest = load(c)
    torch.set_num_threads(4)
    ids = np.flatnonzero(data['roles'] == 'train')[:c['training']['batch_size']]
    records = []
    for target in c['target_arms']:
        for variance in c['variance_arms']:
            arm = dict(source='joint', target=target, variance=variance, seed=c['fit_seeds'][0])
            torch.manual_seed(arm['seed'])
            model = Forecast(c, arm, data, data['positions']).cuda()
            initialize(model, data, c, torch.device('cuda'))
            model.train()
            x = torch.tensor(data['positions'][ids], device='cuda')
            y = torch.tensor(data['target'][ids, :model.head.dimensions], device='cuda', dtype=torch.float64)
            torch.cuda.reset_peak_memory_stats()
            began = time.monotonic()
            loss = (model(x)-y).square().sum(-1).mean()
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 10., error_if_nonfinite=True))
            if norm <= 0 or not torch.isfinite(loss):
                raise FloatingPointError(f'Invalid joint gradient: {arm}')
            rff_gradient = float(model.head.network[-1].weight.grad[18:].abs().max())
            if (target == 'moments' and rff_gradient != 0) or (target == 'full' and rff_gradient == 0):
                raise ValueError(f'Target ablation gradient mismatch: {arm}, {rff_gradient}')
            model.eval()
            with torch.no_grad():
                pred = model(x[:8])
                rotation, _ = torch.linalg.qr(torch.randn(3, 3, device='cuda'))
                error = float((pred-model(x[:8]@rotation)).abs().max())
                raw = pred[:, :18]/model.head.scale[:18]+model.head.center[:18]
                minimum = float((raw[:, 9:18]-raw[:, :9].square()).min())
            if error > 1e-3 or (variance == 'nonnegative' and minimum < 0):
                raise ValueError(f'Rotation or positive variance gate failed: {arm}')
            torch.cuda.synchronize()
            records.append(dict(arm=arm, finite_loss=float(loss.detach()), encoder_gradient_norm=norm,
                rotation_max_abs=error, minimum_normalized_variance=minimum, rff_gradient=rff_gradient,
                seconds=time.monotonic()-began, peak_memory_GiB=torch.cuda.max_memory_allocated()/2**30))
            del model, x, y, loss
            torch.cuda.empty_cache()
    # Exercise the actual 256-wide frozen MM-TDA export and constrained head on CPU.
    z = features(c, data, manifest, 'mm_tda_block_direct_full')
    arm = dict(source='mm_tda_block_direct_full', target='full', variance='nonnegative', seed=c['fit_seeds'][0])
    model = Forecast(c, arm, data, z)
    loss = (model(torch.tensor(z[ids]))-torch.tensor(data['target'][ids])).square().mean()
    loss.backward()
    if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError('MM-TDA local head gate failed')
    record = dict(state='complete', binding=read(output(c)/'technical/binding.json')['identity'],
        device=torch.cuda.get_device_name(), batch=256, microbatch=256, checks=records,
        mm_shape=list(z.shape), online_runs_created=0)
    write_json(output(c)/'technical/preflight.json', record)
    print(json.dumps(record), flush=True)
