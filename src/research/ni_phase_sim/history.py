"""Detached matched encoder/decoder history-attention training and evaluation."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import torch
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, allocation_deadline
from .data import read, root, cache, sha, write_json
from .history_data import prepare, collect
from .campaign import environment

MODULE = 'src.research.ni_phase_sim.history'
REPO = Path(__file__).resolve().parents[3]


def launch(c):
    p = prepare(c); technical = root(c)/'technical'
    if read(cache(c)/'complete.json')['identity'] != p['data_identity']:
        raise ValueError('History observations are not ready')
    bundle = ExecutionBundle.freeze(REPO, technical/'source', c, directories=['src', 'configs', 'docs/metrics'],
        files=[(REPO/'machine.local.yaml', 'machine.local.yaml')])
    write_json(technical/'source-files.json', {str(f.relative_to(bundle.root)): sha(f) for f in bundle.root.rglob('*') if f.is_file()})
    receipt = dict(jobs={}, started=time.time(), origin_allocation=os.environ.get('SLURM_JOB_ID'),
                   workers=c['runtime']['workers'], independent_of_chat=True)
    queue = SlurmQueue(technical, bundle, MODULE, environment(bundle), technical/'submission.json', receipt, 'ni-history')
    # Start real work on the available allocation even if remote resources wait.
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


def stage(c, bundle, lane, deadline, arm, name, marker):
    folder = root(c)/'analyses'/arm['name']/'technical'; folder.mkdir(parents=True, exist_ok=True)
    if (folder/marker).exists() or (folder/(name+'-failed.json')).exists():
        return None
    with (folder/'worker.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        if (folder/marker).exists():
            return None
        torch.cuda.empty_cache()
        for attempt in range(c['runtime']['max_attempts']):
            command = [sys.executable, '-u', '-m', MODULE, name, '--config', str(bundle.config_path),
                       '--name', arm['name'], '--deadline', str(deadline)]
            log = folder/f'{name}-{time.time_ns()}.log'
            with log.open('a') as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                record = dict(stage=name, arm=arm['name'], lane=lane, pid=process.pid, attempt=attempt+1,
                              log=str(log), command=command, started=time.time())
                write_json(root(c)/'technical'/f'lane-{lane}.json', record)
                code = process.wait()
            record.update(exit_code=code, finished=time.time())
            write_json(folder/f'{name}-attempt-{time.time_ns()}.json', record)
            if code == 75:
                return 'paused'
            if (folder/marker).exists():
                return 'done'
        write_json(folder/(name+'-failed.json'), record)
        return 'failed'


def continue_lane(c, bundle, lane, continuation):
    if continuation >= c['runtime']['max_continuations']:
        return
    technical = root(c)/'technical'; path = technical/f'continuation-{lane}-{continuation+1}.json'
    if path.exists():
        return
    receipt = dict(jobs={}, parent_allocation=os.environ.get('SLURM_JOB_ID'), lane=lane,
                   continuation=continuation+1, reason='Fixed-budget queue unfinished at allocation deadline')
    queue = SlurmQueue(technical, bundle, MODULE, environment(bundle), path, receipt, 'ni-history')
    with queue.submission():
        queue.submit(f'continue-{lane}-{continuation+1}', ['--gres=gpu:1', '--cpus-per-task=4', '--mem=64G',
            f'--time={c["runtime"]["hours"]:02d}:00:00'], partition=c['runtime']['partitions'], command_stage='worker',
            arguments=['--lane', str(lane), '--continuation', str(continuation+1)])


def worker(c, bundle, lane, continuation):
    from .history_evaluation import controls, compare
    p = read(root(c)/'technical/plan.json'); technical = root(c)/'technical'
    deadline = min(time.time()+c['runtime']['hours']*3600, allocation_deadline(reserve_seconds=180))
    write_json(technical/f'worker-start-{lane}-{continuation}.json', dict(
        allocation=os.environ.get('SLURM_JOB_ID'), gpu=torch.cuda.get_device_name(), deadline=deadline, started=time.time()))
    # One worker prepares readout controls; independent neural fits need not wait.
    if not (technical/'birth-controls-complete.json').exists() and not (technical/'birth-controls-failed.json').exists():
        with (technical/'birth-controls.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                if not (technical/'birth-controls-complete.json').exists():
                    for attempt in range(c['runtime']['max_attempts']):
                        try:
                            controls(c, p)
                        except Exception:
                            error = dict(attempt=attempt+1, traceback=traceback.format_exc())
                            write_json(technical/f'birth-controls-attempt-{attempt+1}.json', error)
                        else:
                            break
                    if not (technical/'birth-controls-complete.json').exists():
                        write_json(technical/'birth-controls-failed.json', error)
    paused = False
    while time.time() < deadline-480:
        progress = False
        for arm in c['arms']:
            folder = root(c)/'analyses'/arm['name']/'technical'
            for name, marker in [('train', 'complete.json'), ('evaluate', 'evaluation-complete.json'), ('births', 'birth-evaluation-complete.json')]:
                if name != 'train' and not (folder/'complete.json').exists():
                    continue
                if name == 'births' and not (technical/'birth-controls-complete.json').exists():
                    continue
                result = stage(c, bundle, lane, deadline, arm, name, marker)
                if result == 'paused':
                    paused = True; break
                if result is not None:
                    progress = True; break
            if progress or paused:
                break
        if paused:
            break
        if progress:
            continue
        busy = False
        for arm in c['arms']:
            path = root(c)/'analyses'/arm['name']/'technical/worker.lock'
            if path.exists():
                with path.open('a') as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        busy = True; break
        if busy:
            time.sleep(15); continue
        break
    with (technical/'comparison.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX); compare(c)
    pending = [a['name'] for a in c['arms'] if not (root(c)/'analyses'/a['name']/'technical/birth-evaluation-complete.json').exists()
               or not (root(c)/'analyses'/a['name']/'technical/evaluation-complete.json').exists()]
    time_limited = paused or time.time() >= deadline-480
    write_json(technical/f'worker-exit-{lane}-{continuation}.json', dict(
        pending=pending, time_limited=time_limited, finished=time.time()))
    if pending and time_limited:
        continue_lane(c, bundle, lane, continuation)
    if not pending:
        write_json(technical/'complete.json', dict(state='complete', models=len(c['arms']), finished=time.time()))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['prepare', 'collect', 'benchmark', 'launch', 'worker', 'train', 'evaluate', 'births', 'compare'])
    parser.add_argument('--config', required=True); parser.add_argument('--name')
    parser.add_argument('--lane', type=int, default=0); parser.add_argument('--continuation', type=int, default=0)
    parser.add_argument('--deadline', type=float, default=float('inf'))
    args = parser.parse_args(); c = read(args.config); torch.set_num_threads(1)
    if args.stage == 'prepare':
        prepare(c); return
    if args.stage == 'launch':
        launch(c); return
    p = read(root(c)/'technical/plan.json'); bundle = ExecutionBundle(Path(args.config).resolve().parent)
    try:
        if args.stage == 'collect':
            collect(c, p)
        elif args.stage == 'benchmark':
            from .history_learning import benchmark
            benchmark(c)
        elif args.stage == 'worker':
            worker(c, bundle, args.lane, args.continuation)
        elif args.stage == 'compare':
            from .history_evaluation import compare
            compare(c)
        else:
            arm = next(a for a in c['arms'] if a['name'] == args.name)
            if args.stage == 'train':
                from .history_learning import fit
                done = fit(c, arm, args.deadline)
            elif args.stage == 'evaluate':
                from .history_evaluation import evaluate
                done = evaluate(c, arm, args.deadline)
            else:
                from .history_evaluation import births
                done = births(c, p, arm, args.deadline)
            if not done:
                sys.exit(75)
    except Exception:
        write_json(root(c)/'technical'/f'failure-{args.stage}-{args.lane}-{time.time_ns()}.json', dict(traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
