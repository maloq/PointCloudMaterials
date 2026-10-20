"""Detached two-GPU queue with independent stage receipts and bounded recovery."""
import argparse
import fcntl
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, allocation_deadline
from src.experiment_runner.metric_docs import check_metric_docs
from src.research.ni_phase_sim.campaign import environment
from src.research.ni_phase_sim.history import stage
from .data import FAMILY, prepare, root, read, sha, write_json

MODULE = 'src.research.ni_latent_flow.queue'
REPO = Path(__file__).resolve().parents[3]


def launch(c):
    technical = root(c)/'technical'
    if (technical/'submission.json').exists():
        raise FileExistsError('Queue already submitted; inspect its launch receipt')
    if c.get('workflow') == 'validation':
        from .validation import prepare as prepare_validation
        plan = prepare_validation(c)
    else:
        plan = prepare(c)
    check_metric_docs(family=c.get('metric_family', FAMILY))
    bundle = ExecutionBundle.freeze(REPO, technical/'source', c,
        directories=['src', 'configs', 'docs/metrics'],
        files=[(REPO/'machine.local.yaml', 'machine.local.yaml')])
    write_json(technical/'source-files.json', {str(f.relative_to(bundle.root)):sha(f)
        for f in bundle.root.rglob('*') if f.is_file()})
    devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    devices = devices.split(',') if devices else [str(i) for i in range(torch.cuda.device_count())]
    if len(devices) < c['runtime']['workers']:
        raise RuntimeError('The declared local workers require two allocated GPUs')
    started = time.time()
    receipt = dict(identity=plan['identity'], started=started,
        deadline=started+c['runtime']['hours']*3600, models=len(c['arms']),
        origin_allocation=os.environ.get('SLURM_JOB_ID'), workers=[],
        independent_of_chat=True, source=str(bundle.root))
    write_json(technical/'submission.json', receipt)
    write_json(root(c)/'run.json', dict(question='ni_latent_flow', run=root(c).name,
        status='running', models=len(c['arms']), seeds=c['model_seeds'], started=started,
        deadline=receipt['deadline'], local_allocation=receipt['origin_allocation']))
    if c.get('workflow') != 'validation':
        command = [sys.executable, '-u', '-m', MODULE, 'control-worker', '--config', str(bundle.config_path)]
        log = technical/'controls-worker.log'
        with log.open('a') as stream:
            process = subprocess.Popen(command, cwd=bundle.root,
                env=dict(os.environ, **environment(bundle), CUDA_VISIBLE_DEVICES=''),
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        receipt['controls'] = dict(pid=process.pid, command=command, log=str(log))
        write_json(technical/'submission.json', receipt)
    for lane in range(c['runtime']['workers']):
        command = [sys.executable, '-u', '-m', MODULE, 'worker', '--config',
                   str(bundle.config_path), '--lane', str(lane)]
        log = technical/f'worker-{lane}.log'
        with log.open('a') as stream:
            process = subprocess.Popen(command, cwd=bundle.root,
                env=dict(os.environ, **environment(bundle), CUDA_VISIBLE_DEVICES=devices[lane]),
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True)
        receipt['workers'].append(dict(lane=lane, pid=process.pid, device=devices[lane],
            allocation=os.environ.get('SLURM_JOB_ID'), command=command, log=str(log)))
        write_json(technical/'submission.json', receipt)
    print(receipt, flush=True)


def tasks(c):
    if c.get('workflow') == 'validation':
        # Complete the declared fits independently of potentially long readouts.
        for arm in c['arms']:
            yield arm, 'train', 'complete.json', None
        for arm in c['reference_arms']:
            yield arm, 'kinetics', 'kinetics-complete.json', None
        for arm in c['arms']:
            if arm.get('stages') == ['representation']:
                yield arm, 'kinetics', 'kinetics-complete.json', 'complete.json'
                continue
            yield arm, 'diagnostics', 'diagnostics-complete.json', 'complete.json'
            yield arm, 'evaluate', 'flow-evaluation-complete.json', 'complete.json'
        return
    yield dict(name='controls'), 'controls', 'controls-complete.json', None
    for arm in c['arms']:
        yield arm, 'train', 'complete.json', None
        yield arm, 'evaluate', 'flow-evaluation-complete.json', 'complete.json'
        yield arm, 'births', 'birth-evaluation-complete.json', 'complete.json'


def remaining(c):
    return [dict(arm=arm['name'], stage=name,
                 failed=(root(c)/'analyses'/arm['name']/'technical'/f'{name}-failed.json').exists())
        for arm,name,marker,_ in tasks(c)
        if not (root(c)/'analyses'/arm['name']/'technical'/marker).exists()]


def busy(c):
    for arm in [*c['arms'], *c.get('reference_arms', []), dict(name='controls')]:
        path = root(c)/'analyses'/arm['name']/'technical/worker.lock'
        if path.exists():
            with path.open('a') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    return True
    return False


def continuation(c, bundle, lane, deadline):
    path = root(c)/'technical'/f'continuation-{lane}.json'
    if path.exists() or deadline-time.time() < 900:
        return
    receipt = dict(jobs={}, deadline=deadline,
        parent_allocation=os.environ.get('SLURM_JOB_ID'), reason='Resume fixed budget before original queue deadline')
    queue = SlurmQueue(root(c)/'technical', bundle, MODULE, environment(bundle),
                       path, receipt, 'ni-flow')
    with queue.submission():
        queue.submit(f'continue-{lane}', ['--gres=gpu:1', '--cpus-per-task=4', '--mem=64G',
            f'--time={math.ceil((deadline-time.time())/60)}'], partition=c['runtime']['partitions'],
            command_stage='worker', arguments=['--lane', str(lane)])


def worker(c, bundle, lane):
    technical = root(c)/'technical'
    global_deadline = read(technical/'submission.json')['deadline']
    deadline = min(global_deadline, allocation_deadline(reserve_seconds=180))
    write_json(technical/f'worker-start-{lane}-{time.time_ns()}.json', dict(
        allocation=os.environ.get('SLURM_JOB_ID'), gpu=torch.cuda.get_device_name(),
        deadline=deadline, started=time.time()))
    paused = False
    while time.time() < deadline-600:
        progress = False
        for arm, name, marker, prerequisite in tasks(c):
            folder = root(c)/'analyses'/arm['name']/'technical'
            if prerequisite and not (folder/prerequisite).exists():
                continue
            result = stage(c, bundle, lane, deadline, arm, name, marker,
                           module=MODULE, hard_deadline=True)
            if result == 'paused':
                paused = True
                break
            if result is not None:
                progress = True
                break
        if paused:
            break
        if progress:
            continue
        if busy(c):
            time.sleep(15)
            continue
        break
    pending = remaining(c)
    write_json(technical/f'worker-exit-{lane}-{time.time_ns()}.json', dict(
        pending=pending, finished=time.time(), paused=paused))
    if pending and (paused or time.time() >= deadline-600) and deadline < global_deadline-600:
        continuation(c, bundle, lane, global_deadline)
    if not busy(c):
        with (technical/'comparison.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            comparison_complete = False
            try:
                if c.get('workflow') == 'validation':
                    from .validation import compare
                    compare(c)
                else:
                    from .assessment import compare
                    compare(c)
                comparison_complete = True
            except Exception:
                write_json(technical/f'comparison-failed-{time.time_ns()}.json', dict(traceback=traceback.format_exc()))
            write_json(technical/'queue-finished.json', dict(
                state='complete' if not pending and comparison_complete else 'partial',
                pending=pending, comparison_complete=comparison_complete, finished=time.time()))
            metadata = read(root(c)/'run.json')
            metadata.update(status='complete' if not pending and comparison_complete else 'partial',
                completed_models=sum((root(c)/'analyses'/a['name']/'technical/complete.json').exists() for a in c['arms']),
                pending=pending, comparison_complete=comparison_complete, finished=time.time())
            write_json(root(c)/'run.json', metadata)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['prepare','launch','worker','control-worker','train','evaluate','births','controls','compare','kinetics','diagnostics'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--name')
    parser.add_argument('--lane', type=int, default=0)
    parser.add_argument('--deadline', type=float, default=float('inf'))
    args = parser.parse_args()
    c = read(args.config)
    torch.set_num_threads(1)
    if args.stage == 'prepare':
        if c.get('workflow') == 'validation':
            from .validation import prepare as prepare_validation
            prepare_validation(c)
        else:
            prepare(c)
    elif args.stage == 'launch':
        launch(c)
    elif args.stage == 'worker':
        worker(c, ExecutionBundle(Path(args.config).resolve().parent), args.lane)
    elif args.stage == 'control-worker':
        deadline = min(read(root(c)/'technical/submission.json')['deadline'],
                       allocation_deadline(reserve_seconds=180))
        stage(c, ExecutionBundle(Path(args.config).resolve().parent), 'controls', deadline,
              dict(name='controls'), 'controls', 'controls-complete.json', module=MODULE,
              hard_deadline=True)
    elif args.stage == 'compare':
        if c.get('workflow') == 'validation':
            from .validation import compare
        else:
            from .assessment import compare
        compare(c)
    elif args.stage == 'controls':
        from .assessment import controls
        if not controls(c, args.deadline):
            raise SystemExit(75)
    else:
        arm = next(a for a in [*c['arms'], *c.get('reference_arms', [])] if a['name'] == args.name)
        if args.stage == 'train':
            from .learning import fit
            done = fit(c, arm, args.deadline, family=c.get('metric_family', FAMILY))
        elif args.stage == 'evaluate':
            from .assessment import evaluate
            done = evaluate(c, arm, args.deadline, family=c.get('metric_family', FAMILY))
        elif args.stage in ('kinetics', 'diagnostics'):
            from . import validation
            done = getattr(validation, args.stage)(c, arm, args.deadline)
        else:
            from .assessment import evaluate_births
            done = evaluate_births(c, arm, args.deadline)
        if not done:
            raise SystemExit(75)


if __name__ == '__main__':
    main()
