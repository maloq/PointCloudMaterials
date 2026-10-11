"""Resumable two-worker Ni response campaign; all jobs have launch receipts."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from src.experiment_runner.execution import allocation_deadline,ExecutionBundle,SlurmQueue
from src.experiment_runner.metric_docs import check_metric_docs
from src.project_runtime.paths import REPO
from .data import read,root,cache,prepare,observation_job,seal,birth_source,seal_births,write_json,sha


def cpu(c,p):
    from functools import partial
    with ProcessPoolExecutor(max_workers=c['runtime']['cpu_threads']) as pool:
        list(pool.map(partial(observation_job,c,p),p['jobs']))
        list(pool.map(partial(birth_source,c,p),p['sources']))
    seal(c,p);seal_births(c,p)


def worker(c,p,lane,deadline):
    from . import simulator,targets,train,evaluate
    import torch
    torch.set_num_threads(2)
    micro=c['runtime']['microbatch']
    out=root(c)/'technical';lockpath=out/f'worker-{lane}.lock'
    with lockpath.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for i,job in enumerate(p['jobs'][lane::2]):
            simulator.collect(c,p,job,deadline-300)
            write_json(out/f'worker-{lane}.json',dict(state='simulator_bank',done=i+1,total=len(p['jobs'][lane::2]),host=os.uname().nodename,job=os.environ.get('SLURM_JOB_ID')))
        write_json(out/f'bank-lane-{lane}.json',dict(state='complete',plan=p['identity']))
        while not all((out/f'bank-lane-{i}.json').exists() for i in range(2)):
            if time.time()>deadline-600:return False
            time.sleep(10)
        with (out/'targets.lock').open('a+') as teacher_lock:
            fcntl.flock(teacher_lock,fcntl.LOCK_EX)
            targets.prepare(c,p,micro,deadline)
        names=[]
        for arm in c['arms'][lane::2]:
            if not train.run(c,p,arm,deadline,micro):return False
            evaluate.infer(c,p,arm['name'],micro);names.append(arm['name'])
        if lane==0:evaluate.infer(c,p,'spatial',micro);names.append('spatial')
        write_json(out/f'worker-{lane}-complete.json',dict(state='complete',models=names))
        return True


def submit(c):
    check_metric_docs(family='ni_esen_response');prepare(c)
    out=root(c)/'technical';out.mkdir(parents=True,exist_ok=True)
    if (out/'launch.json').exists():raise ValueError('Launch already recorded; use recorded continuation')
    bundle=ExecutionBundle.freeze(REPO,out/'code',c,directories=['src','configs','docs/metrics'],files=[(REPO/'machine.local.yaml','machine.local.yaml')])
    env=dict(PCM_PROJECT_ROOT=str(REPO),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OVITO_THREAD_COUNT='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',PYTORCH_ALLOC_CONF='expandable_segments:True',PYTHONPATH=str(bundle.root),CUDA_HOME='/usr/local/cuda',PATH=str(Path(sys.executable).parent)+':/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin')
    # Preserve configured compiler/toolkit paths; never copy credentials to receipts.
    for name in ['CUDA_HOME','TORCH_CUDA_ARCH_LIST']:
        if name in os.environ:env[name]=os.environ[name]
    receipt=dict(protocol=c['protocol'],started_unix=time.time(),jobs={},owner='current-chat Ni unrelaxed eSEN response request',bundle=str(bundle.root))
    q=SlurmQueue(out,bundle,'src.research.ni_response.workflow',env,out/'launch.json',receipt,'niresp')
    with q.submission():
        cp=q.submit('cpu',['--cpus-per-task=4','--mem=32G','--time=03:00:00'])
        jobs=[]
        for lane in range(2):
            jobs.append(q.submit(f'worker-{lane}',['--gres=gpu:1','--cpus-per-task=4','--mem=48G','--time=24:00:00','--requeue'],dependency='afterok:'+cp,partition=c['runtime']['partitions'],command_stage='worker',arguments=['--lane',str(lane)]))
        q.submit('evaluate',['--cpus-per-task=4','--mem=48G','--time=12:00:00'],dependency='afterok:'+':'.join(jobs))
    return receipt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','cpu','worker','evaluate','submit']);ap.add_argument('--config',required=True);ap.add_argument('--lane',type=int,default=0);a=ap.parse_args();c=read(a.config)
    if a.stage=='submit':print(submit(c));return
    if a.stage=='prepare':print(prepare(c)['identity']);return
    p=read(root(c)/'technical/plan.json');out=root(c)/'technical'
    try:
        if a.stage=='cpu':cpu(c,p)
        elif a.stage=='worker':
            deadline=allocation_deadline(reserve_seconds=120)
            try:complete=worker(c,p,a.lane,deadline)
            except TimeoutError:complete=False
            if not complete:
                job=os.environ['SLURM_JOB_ID']
                launch=read(out/'launch.json')
                if launch['jobs'][f'worker-{a.lane}']!=job:raise ValueError('Cannot requeue an allocation not launched by this campaign')
                write_json(out/f'requeue-{a.lane}.json',dict(job=job,time=time.time(),reason='checkpointed before walltime'))
                subprocess.run(['scontrol','requeue',job],check=True)
        elif a.stage=='evaluate':
            from .evaluate import evaluate
            for name in ['spatial']+[arm['name'] for arm in c['arms']]:evaluate(c,p,name)
    except BaseException:
        write_json(out/f'failed-{a.stage}-{a.lane}.json',dict(error=traceback.format_exc(),time=time.time(),job=os.environ.get('SLURM_JOB_ID')));raise

if __name__=='__main__':main()
