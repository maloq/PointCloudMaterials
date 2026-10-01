"""Frozen, resumable Slurm execution of the shooting-law diagnostic study."""
import argparse
import json
import os
import subprocess
from pathlib import Path
import sys

from src.data.fixed_cohort.protocol import sha, write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs
from .common import read, bind, plan, folder, result


def preflight(c):
    """Compute one actual parent/branch and verify its numerical consumers."""
    import numpy as np
    import torch
    from .data import prepare
    from .fit import Head
    prepare(c, 0, shots=[0])
    root = folder(c) / 'parents/000'
    with np.load(root / 'parent.npz') as a:
        observed = a['descriptors']
        weight = a['weights']
        positions = a['positions']
    with np.load(root / 'shot-00.npz') as a:
        future = a['future'].reshape(len(observed), 1, -1)
        events = a['event']
    if not (np.isfinite(observed).all() and np.isfinite(future).all() and np.isclose(weight.sum(), 1)):
        raise ValueError('Invalid real shooting targets or inclusion weights')
    if not np.all(positions[:, 0] == 0) or future.shape[2] != 24:
        raise ValueError('Actual tensor contract changed')
    x = torch.as_tensor((observed - observed.mean(0)) / np.maximum(observed.std(0), 1e-5))
    y = torch.as_tensor((future - future.mean(0)) / np.maximum(future.std(0), 1e-5))
    model = Head(x.shape[1], y.shape[-1], c['mixtures'], c['hidden'])
    loss = model.loss(x, y).mean()
    loss.backward()
    if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError('Nonfinite real-data mixture likelihood/gradient')
    valid = events >= 0
    event_model = Head(x.shape[1], 0, c['mixtures'], c['hidden'])
    event_loss = event_model.loss(x[valid], torch.as_tensor(events[valid, None])).mean()
    event_loss.backward()
    if not torch.isfinite(event_loss):
        raise FloatingPointError('Nonfinite real event likelihood')
    receipt = dict(passed=True, plan_identity=plan(c)['identity'], config=c,
        descriptor_dimensions=x.shape[1], target_dimensions=y.shape[-1], observations=len(x),
        path_nll=float(loss.detach()), event_nll=float(event_loss.detach()),
        producer_sha256=sha(Path(__file__)), data_producer_sha256=sha(Path(__file__).with_name('data.py')),
        input_contract='Geometry-only actual tensors; metadata never passed to bank/model',
        tracking='local numerical verification; no W&B')
    write_json(result(c) / 'technical/preflight.json', receipt)
    return receipt


def submit(config_path):
    c = read(config_path)
    p = bind(c)
    check_metric_docs(family='shooting_laws')
    tech = result(c) / 'technical'
    checked = read(tech / 'preflight.json')
    if not checked['passed'] or checked['plan_identity'] != p['identity'] or checked['data_producer_sha256'] != sha(Path(__file__).with_name('data.py')):
        raise ValueError('Missing current real-data preflight')
    if (tech / 'launch.json').exists():
        raise ValueError('Already submitted; resume recorded frozen stages')
    repo = Path(__file__).resolve().parents[3]
    bundle = ExecutionBundle.freeze(repo, tech / 'code', c, directories=('src', 'docs/metrics'))
    receipt = dict(protocol=c['protocol'], plan_identity=p['identity'], jobs={}, code=str(bundle.root),
                   tracking='All fits are local frozen diagnostic probes; no new scientific encoder training')
    env = dict(PCM_PROJECT_ROOT=str(repo), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OVITO_THREAD_COUNT='1', NUMBA_NUM_THREADS='1', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    q = SlurmQueue(tech, bundle, 'src.research.shooting_laws.queue', env, tech / 'launch.json', receipt, 'LAW')
    with q.submission():
        prepare = q.submit('prepare', ['--array=0-7', '--cpus-per-task=2', '--mem=8G', '--time=12:00:00'],
            command_stage='group', arguments=('--worker-stage', 'prepare', '--count', '40', '--groups', '8'))
        sealed = q.submit('seal', ['--cpus-per-task=2', '--mem=8G', '--time=00:30:00'], 'afterok:' + prepare)
        encoded = q.submit('encode', ['--array=0-1%1', '--gpus=1', '--cpus-per-task=2', '--mem=12G', '--time=02:00:00'], 'afterok:' + sealed, partition=c['gpu_partition'])
        fits = len(c['arms']) * len(c['fit_seeds'])
        fitted = q.submit('fit', ['--array=0-2', '--cpus-per-task=4', '--mem=12G', '--time=08:00:00'], 'afterok:' + encoded,
            command_stage='group', arguments=('--worker-stage', 'fit', '--count', str(fits), '--groups', '3'))
        q.submit('collect', ['--cpus-per-task=4', '--mem=16G', '--time=04:00:00'], 'afterok:' + fitted)
    return receipt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('bind', 'preflight', 'submit', 'prepare', 'seal', 'encode', 'fit', 'collect', 'group'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--index', type=int)
    parser.add_argument('--worker-stage', choices=('prepare', 'fit'))
    parser.add_argument('--worker-module', choices=('queue', 'diagnostics'), default='queue')
    parser.add_argument('--count', type=int)
    parser.add_argument('--groups', type=int)
    args = parser.parse_args()
    c = read(args.config)
    index = args.index if args.index is not None else int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
    if args.stage == 'group':
        if not args.worker_stage or not args.count or not args.groups or not 0 <= index < args.groups:
            raise ValueError('A group requires a worker stage, positive count/groups and valid index')
        for item in range(index, args.count, args.groups):
            print(f'{args.worker_module}/{args.worker_stage}: starting index {item}', flush=True)
            subprocess.run([sys.executable, '-u', '-m', 'src.research.shooting_laws.' + args.worker_module,
                args.worker_stage, '--config', args.config, '--index', str(item)], check=True)
        return
    if args.stage == 'bind':
        print(json.dumps(dict(identity=bind(c)['identity'])))
        return
    if args.stage == 'submit':
        print(json.dumps(submit(args.config), indent=2))
        return
    state = result(c) / 'technical' / f'{args.stage}-{index}.json'
    with recorded_stage(state, job=os.environ.get('SLURM_JOB_ID')) as record:
        if args.stage == 'preflight':
            record.update(result=preflight(c))
        elif args.stage == 'prepare':
            from .data import prepare
            prepare(c, index)
        elif args.stage == 'seal':
            from .data import seal
            record.update(result=seal(c))
        elif args.stage == 'encode':
            from .features import encode
            encode(c, index)
        elif args.stage == 'fit':
            from .fit import fit
            fit(c, index)
        elif args.stage == 'collect':
            from .evaluate import collect
            collect(c)


if __name__ == '__main__':
    main()
