"""Detached two-GPU queue with frozen source, resumable fits and isolated failures."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, allocation_deadline
from src.experiment_runner.metric_docs import check_metric_docs
from .data import read, root, cache, prepare, write_json, sha

MODULE = 'src.research.measurement_sim.workflow'
REPO = Path(__file__).resolve().parents[3]


def environment(bundle):
    return dict(PYTHONPATH=str(bundle.root), PCM_PROJECT_ROOT=str(bundle.root),
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OVITO_THREAD_COUNT='1', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
        PATH=os.environ['PATH'], CUDA_HOME=os.environ.get('CUDA_HOME', '/usr/local/cuda'))


def team(c, bundle, *, detach):
    technical = root(c)/'technical'
    deadline = min(time.time()+c['runtime']['hours']*3600, allocation_deadline(reserve_seconds=180))
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    devices = visible.split(',') if visible else [str(i) for i in range(torch.cuda.device_count())]
    if len(devices) < 2: raise RuntimeError('Two allocated GPUs are required for this queue')
    processes = []
    receipt = dict(allocation=os.environ.get('SLURM_JOB_ID'), started=time.time(), deadline=deadline,
                   bundle=str(bundle.root), workers=[])
    for lane in range(2):
        command = [sys.executable, '-u', '-m', MODULE, 'worker', '--config', str(bundle.config_path),
                   '--lane', str(lane), '--deadline', str(deadline)]
        env = dict(os.environ, **environment(bundle), CUDA_VISIBLE_DEVICES=devices[lane])
        log_path = technical/f'worker-{receipt["allocation"]}-{lane}.log'
        with log_path.open('a') as log:
            process = subprocess.Popen(command, cwd=bundle.root, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=detach)
        processes.append(process)
        receipt['workers'].append(dict(pid=process.pid, lane=lane, device=devices[lane], command=command, log=str(log_path)))
    write_json(technical/f'launch-{receipt["allocation"]}.json', receipt)
    print(receipt, flush=True)
    if not detach:
        codes = [p.wait() for p in processes]
        if any(codes): raise RuntimeError(f'Queue worker exit codes: {codes}')


def launch(c):
    plan = prepare(c)
    technical = root(c)/'technical'
    if read(cache(c)/'data.json')['identity'] != plan['identity']:
        raise ValueError('Measurement cache does not belong to this plan')
    if not (technical/'physical-complete.json').exists(): raise RuntimeError('Physical measurements incomplete')
    check_metric_docs(family='measurement_sim')
    bundle = ExecutionBundle.freeze(REPO, technical/'source', c,
        directories=['src', 'configs', 'docs/metrics'], files=[(REPO/'machine.local.yaml', 'machine.local.yaml')])
    write_json(technical/'source-files.json', {str(p.relative_to(bundle.root)):sha(p)
        for p in bundle.root.rglob('*') if p.is_file()})
    team(c, bundle, detach=True)


def continuation(c, bundle):
    """Only if allocation time runs out; one reservation for the two workers."""
    technical = root(c)/'technical'
    with (technical/'continuation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt_path = technical/'continuation.json'
        if receipt_path.exists(): return
        receipt = dict(jobs={}, reason='Unfinished fixed scientific queue at allocation boundary',
                       parent_allocation=os.environ['SLURM_JOB_ID'])
        queue = SlurmQueue(technical, bundle, MODULE, environment(bundle), receipt_path, receipt, 'ni-measurements')
        with queue.submission():
            queue.submit('continuation', ['--gres=gpu:2', '--cpus-per-task=4', '--mem=64G', '--time=08:00:00'],
                dependency='afterany:'+os.environ['SLURM_JOB_ID'],
                partition='RTX6000PRO,A100,A40,L40S', command_stage='team')


def worker(c, lane, deadline, bundle):
    technical = root(c)/'technical'
    # Obtain a matched response/control pair first, then the marginal control;
    # repeat independently for the second initialization.
    order = sorted(c['arms'], key=lambda a:(a['seed'], {'connected_response':0, 'connected':1, 'marginal':2}[a['treatment']]))
    paused = False
    for arm in order:
        folder = root(c)/'analyses'/arm['name']/'technical'; folder.mkdir(parents=True, exist_ok=True)
        with (folder/'worker.lock').open('a') as lock:
            try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: continue
            for stage, marker in [('train', 'complete.json'), ('evaluate', 'evaluation-complete.json')]:
                if (folder/marker).exists(): continue
                if time.time() > deadline-1200: paused = True; break
                if stage == 'evaluate' and not (folder/'complete.json').exists(): break
                success = False
                for attempt in range(1, c['runtime']['max_attempts']+1):
                    record = dict(lane=lane, allocation=os.environ.get('SLURM_JOB_ID'), arm=arm['name'],
                                  stage=stage, attempt=attempt, state='running', started=time.time())
                    command = [sys.executable, '-u', '-m', MODULE, stage, '--config', str(bundle.config_path),
                               '--arm', arm['name'], '--deadline', str(deadline)]
                    log = folder/f'{stage}-{record["allocation"]}-{attempt}.log'
                    with log.open('a') as stream:
                        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                        record.update(pid=process.pid, command=command, log=str(log))
                        write_json(technical/f'lane-{lane}.json', record)
                        code = process.wait()
                    record.update(exit_code=code, finished=time.time(), state='complete' if code == 0 else 'failed')
                    write_json(folder/f'{stage}-attempt-{record["allocation"]}-{attempt}.json', record)
                    write_json(technical/f'lane-{lane}.json', record)
                    if code == 75: paused = True; break
                    if code == 0 and (folder/marker).exists(): success = True; break
                    print('Stage failed; preserving checkpoint and error:', record, flush=True)
                if paused: break
                if not success: break  # Other experiments still run; no change to scientific settings.
            if paused: break
    with (technical/'comparison.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        from .compare import run
        run(c)
    states = {a['name']:{'trained':(root(c)/'analyses'/a['name']/'technical/complete.json').exists(),
                        'evaluated':(root(c)/'analyses'/a['name']/'technical/evaluation-complete.json').exists()}
              for a in c['arms']}
    write_json(technical/f'queue-status-{lane}.json', dict(models=states, paused=paused, updated=time.time()))
    if paused: continuation(c, bundle)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['launch', 'worker', 'team', 'train', 'evaluate'])
    parser.add_argument('--config', required=True); parser.add_argument('--arm')
    parser.add_argument('--lane', type=int, default=0); parser.add_argument('--deadline', type=float, default=float('inf'))
    args = parser.parse_args(); c = read(args.config); torch.set_num_threads(1)
    bundle = ExecutionBundle(Path(args.config).resolve().parent)
    if args.stage == 'launch': launch(c)
    elif args.stage == 'team': team(c, bundle, detach=False)
    elif args.stage == 'worker': worker(c, args.lane, args.deadline, bundle)
    else:
        arm = next(a for a in c['arms'] if a['name'] == args.arm)
        p = read(root(c)/'technical/plan.json')
        if args.stage == 'train':
            from .train import fit
            if not fit(c, p, arm, args.deadline): sys.exit(75)
        else:
            from .evaluate import run
            if not run(c, p, arm, args.deadline): sys.exit(75)


if __name__ == '__main__': main()
