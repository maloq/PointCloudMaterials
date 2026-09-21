"""GPU cell producers sharing the existing CPU queue's task locks and receipts."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import save_json, file_hash
from src.data.relaxed_targets.worker import verify_archive
from src.training_methods.shared_pretraining.queue import deadline_for_job
from .queue import claim, worker
from .prepare import produce


def accelerator_settings(base, profile):
    """Change execution only; retain potential, minimizer and tolerance settings."""
    binary=resolve_path(profile['binary'])
    if file_hash(binary)!=profile['binary_sha256']:
        raise ValueError(f'GPU LAMMPS binary changed: {binary}')
    return dict(base,lammps_command=[str(binary),'-k','on','g','1','-sf','kk',
                '-pk','kokkos','neigh','half','newton','on','gpu/aware','off'],
                accelerator=dict(profile,host=socket.gethostname(),allocation=os.environ.get('SLURM_JOB_ID')))


def wait_for_benchmark(config,backend,deadline,status):
    """Require an actual converged same-input force check before producing data."""
    root=resolve_path(config['benchmark'])/'technical'
    while time.time()<deadline-300:
        receipt=root/'results'/f'{backend}-liquid-520K-r0.json'
        if receipt.exists():
            record=json.loads(receipt.read_text())
            if record['state']!='complete':raise RuntimeError(f'{backend} benchmark failed: {record}')
            ref=np.load(root/'results/cpu-current-liquid-520K-r0-forces.npy')
            actual=np.load(root/'results'/f'{backend}-liquid-520K-r0-forces.npy')
            if ref.shape!=actual.shape or not np.isfinite(actual).all():raise ValueError('Invalid GPU benchmark forces')
            max_error=float(np.abs(actual-ref).max())
            if max_error>config['force_max_error_tolerance']:raise ValueError(f'GPU force mismatch: {max_error}')
            profile=dict(backend=backend,binary=record['hardware']['binary'],binary_sha256=record['binary_sha256'],
                         source_commit=record['hardware']['source_commit'],precision='double',benchmark=str(receipt),
                         initial_force_max_error_eV_per_A=max_error,
                         numerical_protocol='Same generating potential, full fixed periodic box and force tolerance; GPU FIRE can reach different local minima.')
            if file_hash(Path(profile['binary']))!=profile['binary_sha256']:raise ValueError('Benchmark binary changed')
            save_json(status,dict(state='validated',profile=profile));return profile
        save_json(status,dict(state='waiting_for_benchmark',backend=backend));time.sleep(20)
    raise TimeoutError('Allocation ended before a validated GPU benchmark was available')


def run(config,backend,lane,*,handoff=False,wait_benchmark_completion=False):
    torch.set_num_threads(1)
    plan=json.loads(resolve_path(config['plan']).read_text());c=plan['config'];root=resolve_path(c['output'])/'technical'
    release=root/'accelerated';release.mkdir(exist_ok=True);cache=resolve_path(c['cache']);deadline=deadline_for_job()
    status=release/f'{lane}.json';profile=wait_for_benchmark(config,backend,deadline,status)
    actual_gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid','--format=csv,noheader'],text=True).strip()
    if backend.upper() not in actual_gpu:raise ValueError(f'Expected {backend}, got {actual_gpu}')
    profile['gpu']=actual_gpu
    if wait_benchmark_completion:
        done=resolve_path(config['benchmark'])/'technical'/f'status-{backend}.json'
        while time.time()<deadline-300:
            state=json.loads(done.read_text())['state']
            if state in ('complete','completed_with_failures'):break
            if state=='failed':raise RuntimeError(f'Benchmark worker failed: {done}')
            save_json(status,dict(state='waiting_for_benchmark_gpu_release'));time.sleep(20)
        else:raise TimeoutError('Benchmark did not release GPU before deadline')
    completed=0
    # Same shared plan and locks as old frozen CPU workers. Shuffle within each
    # priority to avoid assigning whole temperature/source ranges to a backend.
    rng=np.random.default_rng(config['seed']+sum(lane.encode()))
    tasks=[]
    for priority in (0,1):
        group=[t for t in plan['tasks'] if t['priority']==priority]
        tasks.extend(group[i] for i in rng.permutation(len(group)))
    while time.time()<deadline-300:
        if handoff and (root/'training-ready.json').exists():break
        remaining=[t for t in tasks if not (cache/'cells'/t['id']/'complete.json').exists()]
        if not remaining:break
        # Reserve the complete normal + extended retry budgets, with publication
        # margin, rather than killing a cell midway through its archived retry.
        budget=2400+c['retry_relaxation']['frame_timeout_seconds']+300
        if time.time()+budget>deadline:
            save_json(status,dict(state='walltime_reserve',completed=completed,missing=len(remaining)));return
        acquired_any=False
        for task in remaining:
            if handoff and (root/'training-ready.json').exists():break
            if time.time()+budget>deadline:break
            with claim(root/'locks'/f'cell-{task["id"]}') as acquired:
                if not acquired:continue
                if (cache/'cells'/task['id']/'complete.json').exists():continue
                if (root/'failures'/f'{task["id"]}.json').exists():continue
                acquired_any=True
                save_json(status,dict(state='relaxing',task=task,completed=completed,profile=profile,pid=os.getpid()))
                try:
                    try:result=produce(plan,task,1,accelerator=profile)
                    except RuntimeError:
                        failure=resolve_path(c['archive'])/'failures'/task['id']
                        log=failure/'log.lammps'
                        if not log.exists() or 'Stopping criterion = max iterations' not in log.read_text():raise
                        verify_archive(failure)
                        recovery=dict(name='extended-budget',limits=c['retry_relaxation'],restart_dump=str(failure/'relaxed.dump'),restart_sha256=file_hash(failure/'relaxed.dump'))
                        result=produce(plan,task,1,recovery=recovery,accelerator=profile)
                    completed+=1
                    print(json.dumps(dict(cell=task['id'],backend=backend,seconds=result['relaxation']['seconds'],completed=completed)),flush=True)
                except Exception as exc:
                    save_json(root/'failures'/f'{task["id"]}.json',dict(task=task,backend=backend,error=repr(exc),traceback=traceback.format_exc()))
                    raise
        if list((root/'failures').glob('*.json')):raise RuntimeError('Preparation failure recorded; inspect before proceeding')
        if not acquired_any:time.sleep(20)
    if handoff and time.time()<deadline-300:
        save_json(status,dict(state='handed_to_training',completed=completed,profile=profile))
        worker(plan,lane,resolve_path(config['training_config']))
    else:save_json(status,dict(state='finished',completed=completed,profile=profile))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--backend',choices=['h100','a100','v100'],required=True);p.add_argument('--lane',required=True);p.add_argument('--handoff',action='store_true');p.add_argument('--wait-benchmark-completion',action='store_true');args=p.parse_args()
    config=json.loads(resolve_path(args.config).read_text())
    try:run(config,args.backend,args.lane,handoff=args.handoff,wait_benchmark_completion=args.wait_benchmark_completion)
    except Exception as exc:
        plan=json.loads(resolve_path(config['plan']).read_text())
        save_json(resolve_path(plan['config']['output'])/'technical/accelerated'/f'{args.lane}.json',dict(state='failed',error=repr(exc),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':
    code=0
    try:main()
    except BaseException:traceback.print_exc();code=1
    sys.stdout.flush();sys.stderr.flush();os._exit(code)
