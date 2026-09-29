"""Full-pass, distributed rich-descriptor learning and detached resumption."""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.metric_docs import check_metric_docs
from src.project_runtime.paths import resolve_path, REPO
from src.research.crystal_vector import parallel
from src.research.crystal_vector.model import vcreg
from src.research.crystal_vector.train import compile_model, calibrate, deadline
from src.research.supervised_onset.tracking import tracked_run, update_training_summary
from .control_train import ControlData, export, FAMILIES
from .models import ControlMACE
from .data import config


def setup(c):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(c['seed'])
    torch.cuda.manual_seed_all(c['seed'])
    device = torch.device('cuda', torch.cuda.current_device())
    data = ControlData(c, c['data'], device)
    model = ControlMACE(c['encoder_config'], c, data.outputs).to(device)
    model.output_mask.copy_(torch.as_tensor(data.active, device=device))
    calibrate(model, data, c)
    with torch.no_grad():
        model.readout[-1].weight.mul_(0.01)
        model.readout[-1].bias.zero_()
    return data, model


def probe(c):
    """Local-only numerical/memory measurement; never creates a scientific fit."""
    root = resolve_path(c['output']) / 'technical'
    root.mkdir(parents=True, exist_ok=True)
    data, model = setup(c)
    compile_model(model, data, c)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=0.0, fused=True)
    search = c['batch_search']
    total = torch.cuda.get_device_properties(0).total_memory
    budget = total * search['memory_fraction'] - search['headroom_bytes']
    ids = np.random.default_rng(c['seed']).permutation(data.split['train'])
    records = []

    def measure(n):
        opt.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        b = out = loss = None
        try:
            b = data.batch(ids[:n])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(b)
            if (
                out['state'].shape != (n, 256)
                or out['z'].shape[-1] != 256
                or len(model.encoder.interactions) != 3
            ):
                raise ValueError('Requested 256/3/256 architecture is not the executed model')
            loss = (
                data.loss(out['prediction'], b['target']).mean()
                + vcreg(out, c['regularization'])[0]
            )
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), c['training']['gradient_clip'], error_if_nonfinite=True
            )
            layer_grads = [
                math.sqrt(
                    sum(
                        float(p.grad.square().sum())
                        for p in layer.parameters()
                        if p.grad is not None
                    )
                )
                for layer in model.encoder.interactions
            ]
            if not all(v > 0 and math.isfinite(v) for v in layer_grads):
                raise ValueError(f'Missing spatial-layer gradients: {layer_grads}')
            opt.step()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            record = dict(
                per_gpu_batch=n,
                finite=True,
                loss=float(loss.detach()),
                gradient_norm=float(norm),
                interaction_gradient_norms=layer_grads,
                peak_allocated_bytes=peak,
                within_budget=peak <= budget,
                seconds=time.time() - started,
            )
        except torch.OutOfMemoryError:
            record = dict(
                per_gpu_batch=n,
                finite=None,
                within_budget=False,
                out_of_memory=True,
                seconds=time.time() - started,
            )
        finally:
            del b, out, loss
            opt.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
        records.append(record)
        print(json.dumps(record), flush=True)
        write_json(
            root / 'batch-probe-progress.json', dict(config_sha256=digest(c), records=records)
        )
        return record['within_budget']

    lo = 0
    hi = search['initial_per_gpu']
    while measure(hi):
        lo = hi
        hi *= 2
        if hi > len(ids):
            raise ValueError('Batch search exceeded training population before memory bound')
    unit = search['increment']
    while hi - lo > unit:
        mid = ((hi + lo) // (2 * unit)) * unit
        if mid <= lo:
            break
        if measure(mid):
            lo = mid
        else:
            hi = mid
    if lo == 0:
        raise RuntimeError('No requested batch fits the declared memory reserve')
    # Recheck the accepted boundary with a different real training permutation.
    ids = np.random.default_rng(c['seed'] + 1).permutation(data.split['train'])
    if not measure(lo):
        raise RuntimeError('Boundary batch failed independent geometry check; revise batch plan')
    global_batch = lo * c['runtime']['gpus']
    steps = math.ceil(len(data.split['train']) / global_batch)
    result = dict(
        config_sha256=digest(c),
        dataset_identity=data.manifest['identity'],
        gpu=torch.cuda.get_device_name(0),
        total_memory_bytes=total,
        memory_budget_bytes=budget,
        minimum_device_memory_bytes=total,
        per_gpu_batch=lo,
        global_batch=global_batch,
        world_size=c['runtime']['gpus'],
        train_rows=len(data.split['train']),
        updates_per_epoch=steps,
        total_updates=steps * c['training']['epochs'],
        epochs=c['training']['epochs'],
        encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
        total_parameters=sum(p.numel() for p in model.parameters()),
        records=records,
        online_runs_created=0,
    )
    write_json(root / 'batch-plan.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}), flush=True)


def learning_rate(update, steps, c):
    t = c['training']
    warm = t['warmup_epochs'] * steps
    total = t['epochs'] * steps
    if update < warm:
        return t['max_lr'] * (update + 1) / warm
    phase = (update - warm) / max(1, total - warm - 1)
    return t['min_lr'] + 0.5 * (t['max_lr'] - t['min_lr']) * (1 + math.cos(math.pi * phase))


@torch.no_grad()
def validate(model, data, per_gpu, rank, world):
    model.eval()
    ids = data.split['selection'][rank::world]
    weights = data.weights['selection'][rank::world]
    totals = np.zeros(5)
    families = np.array([v['family'] for v in data.columns])
    masks = [torch.as_tensor((families == f) & data.active, device=data.device) for f in FAMILIES]
    for start in range(0, len(ids), per_gpu):
        b = data.batch(ids[start : start + per_gpu])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(b)
        squared = (out['prediction'] - b['target']).square()
        values = torch.stack(
            [data.loss(out['prediction'], b['target'])] + [squared[:, m].mean(1) for m in masks], 1
        )
        totals += (
            values.double().cpu().numpy() * weights[start : start + len(b['target']), None]
        ).sum(0)
    result = parallel.sum_values(totals)
    return dict(
        zip(('target_nll',) + tuple(f'{f}_standardized_mse' for f in FAMILIES), result.tolist())
    )


def train(c):
    rank, world = parallel.initialize()
    root = resolve_path(c['output'])
    tech = root / 'technical'
    if (root / 'analyses/prediction-v1/technical/complete.json').exists():
        if world > 1:
            dist.destroy_process_group()
        return
    plan = config(tech / 'batch-plan.json')
    if plan['config_sha256'] != digest(c) or plan['world_size'] != world:
        raise ValueError('Batch protocol or world size changed')
    if (
        torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        < plan['minimum_device_memory_bytes']
    ):
        raise RuntimeError('Resume GPU has less VRAM than the frozen batch plan')
    per_gpu = plan['per_gpu_batch']
    batch_size = plan['global_batch']
    steps = plan['updates_per_epoch']
    data, model = setup(c)
    if data.manifest['identity'] != plan['dataset_identity']:
        raise ValueError('Training cohort changed')
    binding = dict(
        config=c,
        architecture=model.architecture,
        batch_plan_sha256=sha(tech / 'batch-plan.json'),
        data=data.manifest['identity'],
        implementation=check_metric_docs(family=c['metric_family'])[c['metric_family']]['files'],
    )
    identity = digest(binding)
    study = SimpleNamespace(root=root, technical=tech, identity=identity, config=c)
    if (tech / 'identity.json').exists() and config(tech / 'identity.json') != binding:
        raise ValueError('Run identity changed')
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=c['training']['max_lr'],
        weight_decay=c['training']['weight_decay'],
        fused=True,
    )
    epoch = step = update = 0
    best = float('inf')
    if (tech / 'last.pt').exists():
        saved = torch.load(tech / 'last.pt', map_location=data.device, weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('Resume checkpoint identity mismatch')
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        epoch, step, update, best = (saved[k] for k in ('epoch', 'step', 'update', 'best'))
        torch.set_rng_state(saved['torch_rng'].cpu())
        torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    parallel.broadcast_model(model)
    compile_model(model, data, c)
    if rank == 0:
        write_json(tech / 'identity.json', binding)
        np.savez(
            tech / 'target-standardization.npz',
            mean=data.mean,
            scale=data.scale,
            active=data.active,
        )
        write_json(
            tech / 'prediction-context.json',
            dict(
                encoder=dict(
                    geometry_only=True,
                    channels=256,
                    spatial_message_passing_layers=3,
                    embedding=256,
                    radius_A=8,
                    edge_cutoff_A=5,
                    nearest_candidates=80,
                    halo=False,
                    species=False,
                    conditions=[],
                    history=False,
                    motion=False,
                ),
                predictor=dict(
                    shared_patch_encoder=True,
                    patches=25,
                    context_width=256,
                    vector_message_layers=2,
                    latent=256,
                    inputs=[
                        'patch scalars and equivariant vectors',
                        'relative patch-center offsets',
                    ],
                    conditions=[],
                ),
                decoder=dict(
                    input='one 256-D invariant context embedding', outputs=3536, hidden=512
                ),
                relaxed=False,
                initialization='scratch',
                training_only_teacher='fixed rich descriptors computed from the same observed geometry',
                cohort='existing raw crystal-free Al64 cohort; no new label-based selection',
                fixed_dataset=c['fixed_dataset'],
                sampling='each training row once per epoch; population weights in likelihood; VCReg on uniform shuffled batches',
                batch_size=batch_size,
                per_gpu_batch=per_gpu,
                world_size=world,
            ),
        )

    def save(path):
        if rank:
            return
        tmp = path.with_suffix('.building.pt')
        torch.save(
            dict(
                identity=identity,
                architecture=model.architecture,
                model=model.state_dict(),
                encoder=model.encoder.state_dict(),
                encoder_config=c['encoder_config'],
                config=c,
                optimizer=opt.state_dict(),
                epoch=epoch,
                step=step,
                update=update,
                best=best,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(),
            ),
            tmp,
        )
        tmp.replace(path)

    stop = deadline(c)
    requested = [False]
    signal.signal(signal.SIGUSR1, lambda *_: requested.__setitem__(0, True))
    n = len(data.split['train'])
    weights = torch.as_tensor(data.rows['weights'], device=data.device, dtype=torch.float32)
    weights = weights / float(data.rows['weights'][data.split['train']].sum())
    tracking = (
        tracked_run(study, 'fit', job_type='encoder') if rank == 0 else parallel.local_tracking()
    )
    finished = False
    with tracking as log:
        if rank == 0:
            log.summary.update(
                dict(
                    batch_size=batch_size,
                    per_gpu_batch=per_gpu,
                    training_rows=n,
                    epochs_requested=c['training']['epochs'],
                    updates_per_epoch=steps,
                    parameters=plan['total_parameters'],
                    encoder_parameters=plan['encoder_parameters'],
                    checkpoint_selector='validation family-balanced descriptor Gaussian NLL',
                    peak_learning_rate=c['training']['max_lr'],
                    data_relaxed=False,
                )
            )
        while epoch < c['training']['epochs']:
            order = np.random.default_rng(np.random.SeedSequence([c['seed'], epoch])).permutation(
                data.split['train']
            )
            model.train()
            started = time.time()
            while step < steps:
                if parallel.stop_requested(requested[0] or time.time() > stop):
                    save(tech / 'last.pt')
                    if rank == 0:
                        write_json(
                            tech / 'state.json',
                            dict(state='checkpointed', epoch=epoch, step=step, update=update),
                        )
                    break
                global_ids = order[step * batch_size : (step + 1) * batch_size]
                ids = global_ids[rank::world]
                b = data.batch(ids)
                opt.zero_grad(set_to_none=True)
                lr = learning_rate(update, steps, c)
                for group in opt.param_groups:
                    group['lr'] = lr
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    out = model(b)
                w = weights[torch.as_tensor(ids, device=data.device)] * n
                nll = (data.loss(out['prediction'], b['target']) * w).sum() * (
                    world / len(global_ids)
                )
                reg, stats = vcreg(out, c['regularization'])
                reg = reg * min((update + 1) / (steps * c['regularization']['warmup_epochs']), 1.0)
                loss = nll + reg
                if parallel.stop_requested(not bool(torch.isfinite(loss))):
                    raise FloatingPointError(f'Nonfinite loss at update {update}')
                loss.backward()
                parallel.average_gradients(model)
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), c['training']['gradient_clip'], error_if_nonfinite=True
                )
                opt.step()
                step += 1
                update += 1
                if update == 1 or update % 8 == 0 or step == steps:
                    means = parallel.sum_values(
                        [float(nll.detach()) / world, float(reg.detach()) / world]
                    )
                    if rank == 0:
                        record = dict(
                            optimizer_update=update,
                            epoch=epoch + step / steps,
                            **{
                                'train/target_nll': float(means[0]),
                                'train/vcreg': float(means[1]),
                                'train/learning_rate': lr,
                                'train/gradient_norm': float(norm),
                                'train/contexts_seen': epoch * n + min(step * batch_size, n),
                            },
                        )
                        log.log(record)
                        with (tech / 'training.jsonl').open('a') as f:
                            f.write(json.dumps(record) + '\n')
                if rank == 0:
                    write_json(
                        tech / 'state.json',
                        dict(
                            state='training',
                            epoch=epoch,
                            step=step,
                            update=update,
                            total_epochs=c['training']['epochs'],
                        ),
                    )
                if update == 1 or update % c['training']['save_every_updates'] == 0:
                    save(tech / 'last.pt')
                del b, out, loss, nll, reg, w, stats
            if step < steps:
                break
            scores = validate(model, data, per_gpu, rank, world)
            epoch += 1
            step = 0
            if not all(math.isfinite(v) for v in scores.values()):
                raise FloatingPointError('Nonfinite validation descriptor score')
            if scores['target_nll'] < best:
                best = scores['target_nll']
                save(tech / 'best.pt')
                if rank == 0:
                    log.summary['checkpoint/selected_epoch'] = epoch
            save(tech / 'last.pt')
            if epoch % 5 == 0:
                save(tech / f'epoch-{epoch:02d}.pt')
            if rank == 0:
                record = dict(
                    optimizer_update=update,
                    epoch=epoch,
                    seconds=time.time() - started,
                    **{'validation/' + k: v for k, v in scores.items()},
                )
                log.log({key: value for key, value in record.items() if key != 'seconds'})
                with (tech / 'validation.jsonl').open('a') as f:
                    f.write(json.dumps(record) + '\n')
                print(json.dumps(record), flush=True)
        finished = epoch == c['training']['epochs']
        if finished and rank == 0:
            write_json(
                tech / 'complete.json',
                dict(
                    identity=identity,
                    epochs=epoch,
                    updates=update,
                    training_context_visits=epoch * n,
                    best_sha256=sha(tech / 'best.pt'),
                    last_sha256=sha(tech / 'last.pt'),
                ),
            )
            log.summary['epochs_completed'] = epoch
    if world > 1:
        dist.destroy_process_group()
    if finished and rank == 0:
        # Give the complete, immutable-row export its own allocation if fitting
        # finishes near the wall limit. An interrupted export is safely rewritten.
        if time.time() > stop - 3600:
            write_json(
                tech / 'state.json',
                dict(
                    state='checkpointed',
                    phase='evaluation_pending',
                    epoch=epoch,
                    step=step,
                    update=update,
                ),
            )
            return
        model.load_state_dict(
            torch.load(tech / 'best.pt', map_location=data.device, weights_only=False)['model']
        )
        summary = export(model, data, dict(c, batch_size=per_gpu), study)
        update_training_summary(study, 'fit', summary, evaluation='rich-descriptor-full-epochs')
        write_json(tech / 'state.json', dict(state='complete', epochs=epoch, updates=update))


def worker(path, requeue):
    c = config(path)
    root = resolve_path(c['output']) / 'technical'
    cmd = [
        sys.executable,
        '-u',
        '-m',
        'torch.distributed.run',
        '--standalone',
        '--nproc_per_node',
        str(c['runtime']['gpus']),
        '-m',
        'src.research.liquid_predictability.rich_encoder',
        'train',
        '--config',
        str(path),
    ]
    subprocess.run(cmd, check=True)
    if not (root.parent / 'analyses/prediction-v1/technical/complete.json').exists():
        if not requeue:
            return
        if config(root / 'state.json')['state'] != 'checkpointed':
            raise RuntimeError('Training stopped without a resumable checkpoint')
        write_json(
            root / f'requeue-{time.time_ns()}.json',
            dict(job=os.environ['SLURM_JOB_ID'], state=config(root / 'state.json')),
        )
        subprocess.run(['scontrol', 'requeue', os.environ['SLURM_JOB_ID']], check=True)


def submit(path, allocation=None):
    c = config(path)
    root = resolve_path(c['output']) / 'technical'
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'launch.json').exists():
        raise ValueError('Run already submitted; use its frozen worker command')
    check_metric_docs(family=c['metric_family'])
    if config(root / 'batch-plan.json')['config_sha256'] != digest(c):
        raise ValueError('Run a numerical batch measurement for this recipe first')
    code = root / 'code'
    source = Path(__file__).resolve().parents[3]
    for name in ('src', 'docs/metrics', 'configs/liquid_predictability'):
        shutil.copytree(
            source / name,
            code / name,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.nbc', '*.nbi'),
        )
    write_json(code / 'config.json', c)
    env = dict(
        PCM_PROJECT_ROOT=str(REPO),
        OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
    )
    cmd = [
        sys.executable,
        '-u',
        '-m',
        'src.research.liquid_predictability.rich_encoder',
        'worker',
        '--config',
        str(code / 'config.json'),
    ]
    script = root / 'worker.sbatch'
    r = c['runtime']
    script.write_text(
        '\n'.join(
            [
                '#!/bin/bash',
                f'#SBATCH --job-name={c["name"]}',
                f'#SBATCH --partition={r["partition"]}',
                '#SBATCH --nodes=1',
                '#SBATCH --ntasks=1',
                f'#SBATCH --cpus-per-task={r["cpu_threads"]}',
                f'#SBATCH --gpus={r["gpus"]}',
                f'#SBATCH --mem={r["memory_GB"]}G',
                f'#SBATCH --time={r["walltime"]}',
                '#SBATCH --requeue',
                f'#SBATCH --output={root}/worker-%j.log',
                'set -euo pipefail',
                'ulimit -n 4096',
                'cd ' + shlex.quote(str(code)),
                'exec env '
                + shlex.join([f'{k}={v}' for k, v in env.items()])
                + ' '
                + shlex.join(cmd + ['--requeue']),
                '',
            ]
        )
    )
    args = (
        ['sbatch', '--parsable']
        + (['--dependency=afterany:' + allocation] if allocation else [])
        + [str(script)]
    )
    jid = subprocess.check_output(args, text=True).strip().split(';')[0]
    receipt = dict(
        job=jid,
        code=str(code),
        config_sha256=sha(Path(path)),
        batch_plan_sha256=sha(root / 'batch-plan.json'),
        allocation=allocation,
    )
    write_json(root / 'launch.json', receipt)
    if allocation:
        if os.environ.get('SLURM_JOB_ID') != allocation:
            raise ValueError('Local start must run inside the specified allocation')
        localenv = dict(os.environ, **env)
        with (root / 'local-worker.log').open('a') as f:
            p = subprocess.Popen(
                cmd,
                cwd=code,
                env=localenv,
                stdout=f,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        receipt['local_pid'] = p.pid
        write_json(root / 'launch.json', receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['probe', 'submit', 'worker', 'train'])
    p.add_argument('--config', required=True)
    p.add_argument('--allocation')
    p.add_argument('--requeue', action='store_true')
    a = p.parse_args()
    c = config(a.config)
    if a.stage == 'probe':
        probe(c)
    elif a.stage == 'submit':
        print(json.dumps(submit(a.config, a.allocation), indent=2))
    elif a.stage == 'worker':
        worker(a.config, a.requeue)
    else:
        train(c)


if __name__ == '__main__':
    main()
