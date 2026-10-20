"""One absolute twelve-hour queue for measured Ni latent dynamics."""
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
from .campaign import environment
from .data import read, root, cache, sha, write_json
from .history import stage
from .measurement_data import prepare, collect

MODULE = 'src.research.ni_phase_sim.measurement_queue'
REPO = Path(__file__).resolve().parents[3]


def launch(c):
    p = prepare(c); technical = root(c)/'technical'
    bundle = ExecutionBundle.freeze(REPO, technical/'source', c, directories=['src', 'configs', 'docs/metrics'],
        files=[(REPO/'machine.local.yaml', 'machine.local.yaml')])
    write_json(technical/'source-files.json', {str(f.relative_to(bundle.root)): sha(f) for f in bundle.root.rglob('*') if f.is_file()})
    started = time.time(); receipt = dict(jobs={}, started=started, deadline=started+c['runtime']['hours']*3600,
        origin_allocation=os.environ.get('SLURM_JOB_ID'), workers=c['runtime']['workers'], independent_of_chat=True,
        models=len(c['arms']), frozen_diagnostic_seeds=c['model_seeds'])
    write_json(technical/'submission.json', receipt)
    queue = SlurmQueue(technical, bundle, MODULE, environment(bundle), technical/'submission.json', receipt, 'ni-measured')
    command = queue.command('worker', '--lane', '0')
    log = technical/'local-worker.log'
    with log.open('a') as stream:
        process = subprocess.Popen(command, cwd=bundle.root, env=dict(os.environ, **environment(bundle)),
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    receipt['local'] = dict(pid=process.pid, command=command, log=str(log), allocation=os.environ.get('SLURM_JOB_ID'))
    write_json(technical/'submission.json', receipt)
    with queue.submission():
        for lane in range(1, c['runtime']['workers']):
            queue.submit(f'lane-{lane}', ['--gres=gpu:1', '--cpus-per-task=4', '--mem=64G',
                f'--time={c["runtime"]["hours"]:02d}:00:00'], partition=c['runtime']['partitions'],
                command_stage='worker', arguments=['--lane', str(lane)])
    print(receipt, flush=True)


def tasks(c):
    for seed in c['model_seeds']:
        for arm in [a for a in c['arms'] if a['seed'] == seed]:
            yield arm, 'train', 'complete.json', None
            yield arm, 'evaluate', 'measurement-evaluation-complete.json', 'complete.json'
            yield arm, 'births', 'birth-evaluation-complete.json', 'complete.json'
        yield dict(name=f'diagnostic-s{seed}', seed=seed), 'diagnose', 'diagnostic-complete.json', None


def remaining(c):
    return [dict(arm=a['name'], stage=name) for a, name, marker, _ in tasks(c)
            if not (root(c)/'analyses'/a['name']/'technical'/marker).exists()]


def continue_local(c, bundle, lane, global_deadline):
    path = root(c)/'technical'/f'continuation-{lane}.json'
    if path.exists() or global_deadline-time.time() < 900:
        return
    minutes = math.ceil((global_deadline-time.time())/60)
    receipt = dict(jobs={}, parent_allocation=os.environ.get('SLURM_JOB_ID'), lane=lane,
                   deadline=global_deadline, reason='Continue within original absolute twelve-hour cap')
    queue = SlurmQueue(root(c)/'technical', bundle, MODULE, environment(bundle), path, receipt, 'ni-measured')
    with queue.submission():
        queue.submit(f'continue-{lane}', ['--gres=gpu:1', '--cpus-per-task=4', '--mem=64G', f'--time={minutes}'],
            partition=c['runtime']['partitions'], command_stage='worker', arguments=['--lane', str(lane)])


def worker(c, bundle, lane):
    technical = root(c)/'technical'; global_deadline = read(technical/'submission.json')['deadline']
    deadline = min(global_deadline, allocation_deadline(reserve_seconds=180))
    write_json(technical/f'worker-start-{lane}-{time.time_ns()}.json', dict(
        allocation=os.environ.get('SLURM_JOB_ID'), gpu=torch.cuda.get_device_name(), deadline=deadline,
        global_deadline=global_deadline, started=time.time()))
    paused = False
    while time.time() < deadline-600:
        progress = False
        for arm, name, marker, prerequisite in tasks(c):
            folder = root(c)/'analyses'/arm['name']/'technical'
            if prerequisite is not None and not (folder/prerequisite).exists():
                continue
            result = stage(c, bundle, lane, deadline, arm, name, marker, module=MODULE, hard_deadline=True)
            if result == 'paused':
                paused = True; break
            if result is not None:
                progress = True; break
        if paused or not progress:
            break
    pending = remaining(c)
    time_limited = paused or time.time() >= deadline-600
    write_json(technical/f'worker-exit-{lane}-{time.time_ns()}.json', dict(pending=pending,
        time_limited=time_limited, finished=time.time(), allocation=os.environ.get('SLURM_JOB_ID')))
    # The current interactive allocation can finish earlier than this queue's deadline.
    if pending and time_limited and deadline < global_deadline-600:
        continue_local(c, bundle, lane, global_deadline)
        return
    if pending and not time_limited:
        return
    torch.cuda.empty_cache()
    with (technical/'comparison.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return
        try:
            command = [sys.executable, '-u', '-m', MODULE, 'compare', '--config', str(bundle.config_path)]
            with (technical/f'comparison-{time.time_ns()}.log').open('a') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                               timeout=max(1, global_deadline-time.time()))
        except Exception:
            write_json(technical/f'comparison-failed-{time.time_ns()}.json', dict(traceback=traceback.format_exc()))
    if not remaining(c):
        write_json(technical/'complete.json', dict(state='complete', models=len(c['arms']), diagnostics=3, finished=time.time()))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['collect', 'prepare', 'benchmark', 'launch', 'worker', 'train', 'evaluate', 'births', 'diagnose', 'compare'])
    parser.add_argument('--config', required=True); parser.add_argument('--name'); parser.add_argument('--lane', type=int, default=0)
    parser.add_argument('--deadline', type=float, default=float('inf'))
    args = parser.parse_args(); c = read(args.config); torch.set_num_threads(1)
    if args.stage == 'collect':
        collect(c); return
    if args.stage == 'prepare':
        prepare(c); return
    if args.stage == 'launch':
        launch(c); return
    try:
        if args.stage == 'benchmark':
            from .measurement_learning import benchmark
            benchmark(c); return
        if args.stage == 'worker':
            worker(c, ExecutionBundle(Path(args.config).resolve().parent), args.lane); return
        p = read(root(c)/'technical/plan.json')
        if args.stage == 'compare':
            from .measurement_assessment import compare
            compare(c); return
        if args.stage == 'diagnose':
            from .measurement_diagnostics import diagnose
            done = diagnose(c, p, int(args.name.split('-s')[-1]), args.deadline)
            if done:
                write_json(root(c)/'analyses'/args.name/'technical/diagnostic-complete.json', dict(state='complete'))
        else:
            arm = next(a for a in c['arms'] if a['name'] == args.name)
            if args.stage == 'train':
                from .measurement_learning import fit
                done = fit(c, arm, args.deadline)
            elif args.stage == 'evaluate':
                from .measurement_assessment import evaluate
                done = evaluate(c, arm, args.deadline)
            else:
                from .measurement_assessment import births
                done = births(c, p, arm, args.deadline)
        if not done:
            sys.exit(75)
    except Exception:
        write_json(root(c)/'technical'/f'failure-{args.stage}-{args.lane}-{time.time_ns()}.json', dict(traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
