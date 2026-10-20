"""Detached fixed-budget native-time kinetic fits and independent diagnostics."""
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

MODULE = 'src.research.ni_kinetic_latent.queue'
REPO = Path(__file__).resolve().parents[3]


def launch(c):
    plan = prepare(c)
    check_metric_docs(family=FAMILY)
    technical = root(c)/'technical'
    bundle = ExecutionBundle.freeze(REPO, technical/'source', c,
        directories=['src', 'configs', 'docs/metrics'],
        files=[(REPO/'machine.local.yaml', 'machine.local.yaml')])
    write_json(technical/'source-files.json', {str(f.relative_to(bundle.root)): sha(f)
        for f in bundle.root.rglob('*') if f.is_file()})
    started = time.time()
    receipt = dict(jobs={}, started=started, deadline=started+c['runtime']['hours']*3600,
        origin_allocation=os.environ.get('SLURM_JOB_ID'), workers=c['runtime']['workers'],
        independent_of_chat=True, models=len(c['arms']), identity=plan['identity'])
    write_json(technical/'submission.json', receipt)
    queue = SlurmQueue(technical, bundle, MODULE, environment(bundle),
        technical/'submission.json', receipt, 'ni-kinetic')
    command = queue.command('worker', '--lane', '0')
    log = technical/'local-worker.log'
    with log.open('a') as stream:
        process = subprocess.Popen(command, cwd=bundle.root, env=dict(os.environ, **environment(bundle)),
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    receipt['local'] = dict(pid=process.pid, command=command, log=str(log),
                            allocation=os.environ.get('SLURM_JOB_ID'))
    write_json(technical/'submission.json', receipt)
    with queue.submission():
        queue.submit('annotations', [f'--cpus-per-task={c["runtime"]["annotation_workers"]}',
            '--mem=32G', f'--time={c["runtime"]["hours"]:02d}:00:00'],
            partition='CPU', command_stage='annotation-worker')
        for lane in range(1, c['runtime']['workers']):
            queue.submit(f'lane-{lane}', ['--gres=gpu:1', '--cpus-per-task=4', '--mem=48G',
                f'--time={c["runtime"]["hours"]:02d}:00:00'],
                partition=c['runtime']['partitions'], command_stage='worker',
                arguments=['--lane', str(lane)])
    metadata = read(root(c)/'run.json') if (root(c)/'run.json').exists() else {}
    metadata.update(status='running', started=started, deadline=receipt['deadline'],
                    remote_jobs=receipt['jobs'], local_allocation=receipt['origin_allocation'])
    write_json(root(c)/'run.json', metadata)
    print(receipt, flush=True)


def tasks(c):
    for arm in c['arms']:
        yield arm, 'train', 'complete.json', None
        yield arm, 'births', 'birth-evaluation-complete.json', 'complete.json'
        yield arm, 'evaluate', 'evaluation-complete.json', 'complete.json'
    yield dict(name='controls'), 'controls', 'controls-complete.json', None


def remaining(c):
    return [dict(arm=arm['name'], stage=name,
                 failed=(root(c)/'analyses'/arm['name']/'technical'/f'{name}-failed.json').exists())
            for arm, name, marker, _ in tasks(c)
            if not (root(c)/'analyses'/arm['name']/'technical'/marker).exists()]


def busy(c):
    for name in [a['name'] for a in c['arms']]+['controls', 'annotations']:
        path = root(c)/'analyses'/name/'technical/worker.lock'
        if path.exists():
            with path.open('a') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    return True
    return False


def continuation(c, bundle, lane, deadline, annotation=False):
    path = root(c)/'technical'/f'continuation-{lane}.json'
    if path.exists() or deadline-time.time() < 900:
        return
    receipt = dict(jobs={}, deadline=deadline, parent_allocation=os.environ.get('SLURM_JOB_ID'),
                   reason='Resume within the original absolute deadline')
    queue = SlurmQueue(root(c)/'technical', bundle, MODULE, environment(bundle), path, receipt, 'ni-kinetic')
    with queue.submission():
        options = ['--cpus-per-task=4', '--mem=48G', f'--time={math.ceil((deadline-time.time())/60)}']
        queue.submit(f'continue-{lane}', options if annotation else ['--gres=gpu:1', *options],
            partition='CPU' if annotation else c['runtime']['partitions'],
            command_stage='annotation-worker' if annotation else 'worker',
            arguments=[] if annotation else ['--lane', str(lane)])


def annotation_worker(c, bundle):
    global_deadline = read(root(c)/'technical/submission.json')['deadline']
    deadline = min(global_deadline, allocation_deadline(reserve_seconds=180))
    result = stage(c, bundle, 'annotations', deadline, dict(name='annotations'),
        'annotate', 'annotation-complete.json', module=MODULE, hard_deadline=True)
    if result == 'paused' and deadline < global_deadline-600:
        continuation(c, bundle, 'annotations', global_deadline, annotation=True)


def finish(c, bundle, deadline):
    technical = root(c)/'technical'
    with (technical/'comparison.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return
        if ((technical/'queue-finished.json').exists() and
                read(technical/'queue-finished.json')['state'] == 'complete'):
            return
        try:
            command = [sys.executable, '-u', '-m', MODULE, 'compare', '--config', str(bundle.config_path)]
            with (technical/f'comparison-{time.time_ns()}.log').open('a') as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True,
                               timeout=max(1, deadline-time.time()))
        except Exception:
            write_json(technical/f'comparison-failed-{time.time_ns()}.json', dict(traceback=traceback.format_exc()))
            return
        pending = remaining(c)
        record = dict(state='complete' if not pending else 'partial', finished=time.time(),
                      models=len(c['arms']), pending=pending)
        write_json(technical/'queue-finished.json', record)
        metadata = read(root(c)/'run.json')
        metadata.update(status=record['state'], finished=record['finished'], pending=pending,
            completed_models=sum((root(c)/'analyses'/a['name']/'technical/complete.json').exists() for a in c['arms']),
            generation_failures=[a['name'] for a in c['arms']
                if (root(c)/'analyses'/a['name']/'technical/generation-failure.json').exists()])
        write_json(root(c)/'run.json', metadata)
        if not pending:
            write_json(technical/'complete.json', record)


def worker(c, bundle, lane):
    technical = root(c)/'technical'
    global_deadline = read(technical/'submission.json')['deadline']
    deadline = min(global_deadline, allocation_deadline(reserve_seconds=180))
    write_json(technical/f'worker-start-{lane}-{time.time_ns()}.json', dict(
        allocation=os.environ.get('SLURM_JOB_ID'), gpu=torch.cuda.get_device_name(),
        deadline=deadline, global_deadline=global_deadline, started=time.time()))
    paused = False
    while time.time() < deadline-600:
        progress = False
        for arm, name, marker, prerequisite in tasks(c):
            folder = root(c)/'analyses'/arm['name']/'technical'
            if prerequisite is not None and not (folder/prerequisite).exists():
                continue
            if name == 'evaluate' and not (technical/'labels-complete.json').exists():
                continue
            result = stage(c, bundle, lane, deadline, arm, name, marker, module=MODULE, hard_deadline=True)
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
        annotation_pending = (not (technical/'labels-complete.json').exists() and
            not (root(c)/'analyses/annotations/technical/annotate-failed.json').exists())
        if busy(c) or (remaining(c) and annotation_pending):
            write_json(technical/f'worker-wait-{lane}.json', dict(
                waiting_for='other workers or independent annotations', updated=time.time()))
            time.sleep(15)
            continue
        break
    pending = remaining(c)
    time_limited = paused or time.time() >= deadline-600
    write_json(technical/f'worker-exit-{lane}-{time.time_ns()}.json', dict(pending=pending,
        time_limited=time_limited, finished=time.time(), allocation=os.environ.get('SLURM_JOB_ID')))
    if pending and time_limited and deadline < global_deadline-600:
        continuation(c, bundle, lane, global_deadline)
        return
    if not busy(c):
        finish(c, bundle, deadline)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['prepare', 'benchmark', 'launch', 'worker',
        'annotation-worker', 'annotate', 'train', 'evaluate', 'births', 'controls', 'compare'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--name')
    parser.add_argument('--lane', type=int, default=0)
    parser.add_argument('--deadline', type=float, default=float('inf'))
    args = parser.parse_args()
    c = read(args.config)
    torch.set_num_threads(1)
    if args.stage == 'prepare':
        prepare(c)
        return
    if args.stage == 'launch':
        launch(c)
        return
    if args.stage in ('worker', 'annotation-worker'):
        bundle = ExecutionBundle(Path(args.config).resolve().parent)
        if args.stage == 'worker':
            worker(c, bundle, args.lane)
        else:
            annotation_worker(c, bundle)
        return
    if args.stage == 'benchmark':
        from .learning import benchmark
        benchmark(c)
        return
    if args.stage == 'compare':
        from .births import compare
        compare(c)
        return
    if args.stage == 'annotate':
        from .data import annotate
        done = annotate(c, args.deadline)
        if done:
            write_json(root(c)/'analyses/annotations/technical/annotation-complete.json', dict(state='complete'))
    elif args.stage == 'controls':
        from .births import controls
        done = controls(c, args.deadline)
    else:
        arm = next(a for a in c['arms'] if a['name'] == args.name)
        if args.stage == 'train':
            from .learning import fit
            done = fit(c, arm, args.deadline)
        elif args.stage == 'evaluate':
            from .assessment import evaluate
            done = evaluate(c, arm, args.deadline)
        else:
            from .births import evaluate
            done = evaluate(c, arm, args.deadline)
    if not done:
        raise SystemExit(75)


if __name__ == '__main__':
    main()
