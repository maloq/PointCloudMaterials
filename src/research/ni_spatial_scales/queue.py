"""Detached radius comparison: shared collection, fixed fits, independent tests."""
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
from .data import FAMILY, prepare, root, cache, read, sha, write_json, resolve_path

MODULE = 'src.research.ni_spatial_scales.queue'
REPO = Path(__file__).resolve().parents[3]


def launch(c):
    plan = prepare(c)
    check_metric_docs(family=FAMILY)
    technical = root(c)/'technical'
    write_json(technical/'prediction-context.json', dict(
        encoder=dict(inputs='central244 plus87 regional measurements; local arm appends87 zeros',
            local_support='original nearest80, original measurement cutoffs',
            regional_support='complete periodic ball R=8/12/20 Angstrom, no count cap, center excluded',
            geometry='unrelaxed observed Q', motion='observed relative V; local17 and regional13',
            history=1, conditions=[], species=False, exported_dimension=32),
        predictor=dict(inputs='last four32-dimensional latent states, native0.1ps spacing',
            history_ps=.3, conditions=[], output='central244 measurements',
            future_context='none supplied; latent state evolves autonomously'),
        teacher=dict(values='saved native NPT observations', derivatives=False,
            training_targets='central244 reconstruction and lagged latent kinetics; no phase labels'),
        split='preserved training ancestry; all former nontraining roles pooled as test',
        checkpoint='fixed final12epochs per stage; no test selection or calibration'))
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
    metadata = dict(question='ni_spatial_scales', run='radius-20261015-v1',
        status='running', models=len(c['arms']), seeds=c['model_seeds'], scales=['local', 'r8', 'r12', 'r20'],
        started=started, deadline=receipt['deadline'], remote_jobs=receipt['jobs'],
        local_allocation=receipt['origin_allocation'], context_counts=plan['context_counts'])
    write_json(root(c)/'run.json', metadata)
    queue = SlurmQueue(technical, bundle, MODULE, environment(bundle),
                       technical/'submission.json', receipt, 'ni-scales')
    command = queue.command('worker', '--lane', '0')
    log = technical/'local-worker.log'
    with log.open('a') as stream:
        process = subprocess.Popen(command, cwd=bundle.root, env=dict(os.environ, **environment(bundle)),
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    receipt['local'] = dict(pid=process.pid, command=command, log=str(log),
                           allocation=os.environ.get('SLURM_JOB_ID'))
    write_json(technical/'submission.json', receipt)
    with queue.submission():
        for lane in range(1, c['runtime']['workers']):
            queue.submit(f'lane-{lane}', ['--gres=gpu:1', '--cpus-per-task=4', '--mem=48G',
                f'--time={c["runtime"]["hours"]:02d}:00:00'], partition=c['runtime']['partitions'],
                command_stage='worker', arguments=['--lane', str(lane)])
    write_json(root(c)/'run.json', metadata)
    print(receipt, flush=True)


def tasks(c, p):
    # Zero-context controls use the already sealed local bank immediately. Their
    # scientific fits do not depend on chat review or expensive context assembly.
    for arm in c['arms']:
        if arm['scale'] == 'local':
            yield arm, 'train', 'complete.json', root(c)/'analyses'/arm['name']/'technical', True
    for job in p['context_jobs']:
        yield dict(name=job['key']), 'collect', 'collected.json', root(c)/'technical/collection'/job['key'], True
    ready = all((cache(c)/'context-sources'/j['key']/'complete.json').exists() for j in p['context_jobs'])
    yield dict(name='assembly'), 'assemble', 'assembled.json', root(c)/'technical/assembly', ready
    labels_ready = (resolve_path(c['base_run'])/'technical/labels-complete.json').exists()
    context_ready = (root(c)/'technical/context-complete.json').exists()
    for arm in c['arms']:
        folder = root(c)/'analyses'/arm['name']/'technical'
        if arm['scale'] != 'local':
            yield arm, 'train', 'complete.json', folder, context_ready
    for arm in c['arms']:
        folder = root(c)/'analyses'/arm['name']/'technical'
        fitted = (folder/'complete.json').exists()
        yield arm, 'births', 'birth-evaluation-complete.json', folder, fitted
        yield arm, 'evaluate', 'evaluation-complete.json', folder, fitted and labels_ready
    yield dict(name='controls'), 'controls', 'controls-complete.json', root(c)/'analyses/controls/technical', True


def remaining(c, p):
    return [dict(arm=a['name'], stage=name, ready=ready,
                 failed=(folder/f'{name}-failed.json').exists())
            for a, name, marker, folder, ready in tasks(c, p) if not (folder/marker).exists()]


def busy(c, p):
    for folder in {task[3] for task in tasks(c, p)}:
        path = folder/'worker.lock'
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
    receipt = dict(jobs={}, deadline=deadline, parent_allocation=os.environ.get('SLURM_JOB_ID'),
                   reason='Resume within the original absolute deadline')
    queue = SlurmQueue(root(c)/'technical', bundle, MODULE, environment(bundle), path, receipt, 'ni-scales')
    with queue.submission():
        queue.submit(f'continue-{lane}', ['--gres=gpu:1', '--cpus-per-task=4', '--mem=48G',
            f'--time={math.ceil((deadline-time.time())/60)}'], partition=c['runtime']['partitions'],
            command_stage='worker', arguments=['--lane', str(lane)])


def finish(c, p, bundle, deadline):
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
        pending = remaining(c, p)
        record = dict(state='complete' if not pending else 'partial', finished=time.time(), pending=pending)
        write_json(technical/'queue-finished.json', record)
        if not pending:
            write_json(technical/'complete.json', record)
        metadata = read(root(c)/'run.json')
        metadata.update(status=record['state'], finished=record['finished'], pending=pending,
            completed_models=sum((root(c)/'analyses'/a['name']/'technical/complete.json').exists() for a in c['arms']),
            generation_failures=[a['name'] for a in c['arms']
                if (root(c)/'analyses'/a['name']/'technical/generation-failure.json').exists()])
        write_json(root(c)/'run.json', metadata)


def worker(c, bundle, lane):
    technical = root(c)/'technical'
    p = read(technical/'plan.json')
    global_deadline = read(technical/'submission.json')['deadline']
    deadline = min(global_deadline, allocation_deadline(reserve_seconds=180))
    write_json(technical/f'worker-start-{lane}-{time.time_ns()}.json', dict(
        allocation=os.environ.get('SLURM_JOB_ID'), gpu=torch.cuda.get_device_name(),
        deadline=deadline, global_deadline=global_deadline, started=time.time()))
    paused = False
    while time.time() < deadline-600:
        progress = False
        for arm, name, marker, folder, ready in tasks(c, p):
            if not ready:
                continue
            result = stage(c, bundle, lane, deadline, arm, name, marker,
                           module=MODULE, hard_deadline=True, folder=folder)
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
        pending = remaining(c, p)
        base = resolve_path(c['base_run'])
        annotation_pending = (not (base/'technical/labels-complete.json').exists() and
            not (base/'analyses/annotations/technical/annotate-failed.json').exists())
        if busy(c, p) or (annotation_pending and any(t['stage'] == 'evaluate' for t in pending)):
            write_json(technical/f'worker-wait-{lane}.json', dict(
                waiting_for='other workers or base annotations', updated=time.time()))
            time.sleep(15)
            continue
        break
    pending = remaining(c, p)
    time_limited = paused or time.time() >= deadline-600
    write_json(technical/f'worker-exit-{lane}-{time.time_ns()}.json', dict(pending=pending,
        time_limited=time_limited, finished=time.time(), allocation=os.environ.get('SLURM_JOB_ID')))
    if pending and time_limited and deadline < global_deadline-600:
        continuation(c, bundle, lane, global_deadline)
        return
    if not busy(c, p):
        finish(c, p, bundle, deadline)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['prepare', 'benchmark', 'launch', 'worker',
        'collect', 'assemble', 'train', 'evaluate', 'births', 'controls', 'compare'])
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
    if args.stage == 'worker':
        worker(c, ExecutionBundle(Path(args.config).resolve().parent), args.lane)
        return
    if args.stage == 'benchmark':
        from .data import benchmark
        benchmark(c)
        return
    if args.stage == 'compare':
        from .births import compare
        compare(c)
        return
    if args.stage == 'collect':
        from .data import prepare_source
        done = prepare_source(c, args.name, deadline=args.deadline)
        if done:
            write_json(root(c)/'technical/collection'/args.name/'collected.json',
                       dict(state='complete', cache_receipt_sha256=sha(cache(c)/'context-sources'/args.name/'complete.json')))
    elif args.stage == 'assemble':
        from .data import assemble
        assemble(c)
        write_json(root(c)/'technical/assembly/assembled.json', dict(state='complete'))
        done = True
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
