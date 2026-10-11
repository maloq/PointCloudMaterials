"""Run an already-frozen spatial experiment inside the current allocation.

Only the experiment's receipt-owned pending successor is changed. Existing
allocation owners and their simulation processes are never changed or stopped.
Scientific code/config remain in the original execution bundle.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix('.building')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def job_state(job):
    raw = subprocess.check_output(['scontrol', 'show', 'job', str(job), '-o'], text=True)
    return re.search(r'\bJobState=(\S+)', raw).group(1), raw


def command(run, stage, *extra):
    return [sys.executable, '-u', '-m', 'src.research.spatial_architecture.workflow',
            stage, '--config', str(run / 'technical/code/config.json'), *extra]


def launch(run):
    technical = run / 'technical'
    parent = os.environ['SLURM_JOB_ID']
    receipt = read(technical / 'launch.json')
    gpu_count = read(technical / 'code/config.json')['runtime'].get('gpus', 1)
    visible_devices = ','.join(str(i) for i in range(gpu_count))
    if receipt['owner'] != 'Response research chat: spatial MACE/eSEN/TensorNet comparisons, authorized 2026-10-07/08':
        raise ValueError('This run has no matching chat ownership receipt')
    successor = receipt['jobs']['queued']
    state, original = job_state(successor)
    if state != 'PENDING' or f'Command={technical}/queued.sbatch' not in original:
        raise ValueError('Receipt-owned job must still be pending with the recorded command')
    parent_state, parent_record = job_state(parent)
    if parent_state != 'RUNNING':
        raise ValueError('Current allocation is not running')
    scope = Path('/proc/self/cgroup').read_text()
    if f'/job_{parent}/' not in scope:
        raise ValueError('Shell is not contained in the declared allocation')
    if receipt['local'] is not None:
        raise ValueError('A local worker is already recorded')
    local = technical / 'local-allocation'
    local.mkdir(exist_ok=False)
    write(local / 'original-launch.json', receipt)
    shutil.copy2(Path(__file__), local / 'runner.py')
    info = dict(owner=receipt['owner'], allocation=parent, host=os.uname().nodename,
        successor=successor, original_successor_state=original,
        parent_allocation=parent_record, parent_cgroup=scope,
        dependency='afterany:' + parent, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen_code=str(technical / 'code'), started_at=time.time(),
        sharing='Uses the requested GPUs in the current allocation; unrelated jobs and allocation remain unchanged.',
        visible_devices=visible_devices, gpu_count=gpu_count,
        scientific_settings_changed=False)
    write(local / 'launch.json', info)
    subprocess.run(['scontrol', 'update', f'JobId={successor}', f'Dependency=afterany:{parent}'], check=True)
    environment = dict(os.environ, PCM_PROJECT_ROOT=str(Path.cwd()),
        CUDA_VISIBLE_DEVICES=visible_devices, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        CUEQUIVARIANCE_OPS_PARALLEL_THREADS='4', PYTHONWARNINGS='ignore::FutureWarning',
        PYTORCH_ALLOC_CONF='expandable_segments:True', TORCHINDUCTOR_COMPILE_THREADS='2')
    with (local / 'supervisor.log').open('ab') as log:
        process = subprocess.Popen([sys.executable, '-u', str(local / 'runner.py'),
            'supervise', '--run', str(run)], cwd=technical / 'code', env=environment,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    info['supervisor_pid'] = process.pid
    write(local / 'launch.json', info)
    receipt['local'] = dict(pid=process.pid, host=os.uname().nodename, allocation=parent,
        gpu=visible_devices, log=str(local / 'supervisor.log'), receipt=str(local / 'launch.json'))
    receipt['continuation_dependency'] = info['dependency']
    write(technical / 'launch.json', receipt)
    print(json.dumps(info, indent=2), flush=True)


def supervise(run):
    technical = run / 'technical'
    local = technical / 'local-allocation'
    info = read(local / 'launch.json')
    status = dict(state='running', phase='starting', allocation=info['allocation'],
                  supervisor_pid=os.getpid(), started_at=time.time())
    with (local / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            stages = []
            if not (technical / 'probe.json').exists():
                stages.append(('numerical_and_speed_probe', command(run, 'probe'), technical/'probe.log'))
            stages.append(('scientific_training', command(run, 'worker', '--lane', 'local'), local/'training.log'))
            for phase, argv, log_path in stages:
                with log_path.open('ab') as log:
                    child = subprocess.Popen(argv, cwd=technical/'code', stdout=log,
                                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                    status.update(phase=phase, child_pid=child.pid, command=argv, log=str(log_path))
                    write(local / 'state.json', status)
                    result = child.wait()
                if result:
                    raise RuntimeError(f'{phase} exited {result}; see {log_path}')
            complete = technical / 'complete.json'
            if complete.exists() and read(complete)['state']=='complete':
                receipt = read(technical / 'launch.json')
                if receipt['jobs']['queued'] != info['successor']:
                    raise ValueError('Successor ownership changed')
                state, _ = job_state(info['successor'])
                if state == 'PENDING':
                    subprocess.run(['scancel', info['successor']], check=True)
                    status['unused_owned_successor_cancelled'] = info['successor']
                status['state'] = 'complete'
            else:
                status.update(state='checkpointed_for_successor', successor=info['successor'])
        except BaseException:
            status.update(state='failed', traceback=traceback.format_exc())
            raise
        finally:
            status['finished_at'] = time.time()
            write(local / 'state.json', status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['launch', 'supervise'])
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    (launch if args.stage=='launch' else supervise)(args.run.resolve())
