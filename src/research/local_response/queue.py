"""Freeze and detach local-response gates, two-GPU collection and twelve fits."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from src.experiment_runner.execution import ExecutionBundle,SlurmQueue,recorded_stage,allocation_deadline
from src.experiment_runner.metric_docs import check_metric_docs
from .common import FAMILY,root,read,bind,write_json


def gate(c):
    from .gates import run
    path=root(c)/'technical/gate.lock';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return run(c)


def parallel(c,stage):
    devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(devices)!=c['lanes']:raise ValueError(f'Expected{c["lanes"]} allocated GPUs, got {devices}')
    children=[]
    try:
        for lane,device in enumerate(devices):
            log=(root(c)/'technical'/f'{stage}-lane{lane}.log').open('a')
            command=[sys.executable,'-u','-m','src.research.local_response.queue',stage,
                '--config',str(Path('config.json').resolve()),'--lane',str(lane)]
            process=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=device),stdout=log,stderr=subprocess.STDOUT)
            children.append((process,log))
        while any(p.poll() is None for p,_ in children):
            if any(p.poll() not in (None,0) for p,_ in children):raise RuntimeError(f'{stage} GPU lane failed; inspect lane logs')
            time.sleep(5)
        if any(p.returncode!=0 for p,_ in children):raise RuntimeError(f'{stage} lane failed')
    finally:
        for p,log in children:
            if p.poll() is None:p.terminate()
            p.wait();log.close()


def submit(c):
    record=bind(c);check_metric_docs(family=FAMILY)
    import wandb
    if not wandb.Api(timeout=30).viewer:raise RuntimeError('W&B online authentication failed')
    tech=root(c)/'technical';path=tech/'launch.json'
    if path.exists():raise ValueError('Already submitted; inspect recorded jobs before explicit continuation')
    repo=Path(__file__).resolve().parents[3]
    bundle=ExecutionBundle.freeze(repo,tech/'code',c,directories=('src','docs/metrics','configs/fixed_cohort'),
        files=((repo/'configs/simulation/local_response_20261002.json','configs/simulation/local_response_20261002.json'),))
    receipt=dict(identity=record['identity'],jobs={},code=str(bundle.root))
    queue=SlurmQueue(tech,bundle,'src.research.local_response.queue',dict(PCM_PROJECT_ROOT=str(repo),
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',WANDB_MODE='online'),
        path,receipt,'LOCAL-RESP')
    with queue.submission():
        queue.submit('worker',['--gpus=2','--cpus-per-task=8','--mem=96G',f'--time={c["time_limit"]}',
            '--signal=B:TERM@300','--requeue','--exclude=node52'],partition=c['gpu_partition'])
    print(receipt,flush=True)


def execute(c,args):
    if args.stage=='prepare':
        from .data import prepare
        prepare(c)
    elif args.stage=='gate':gate(c)
    elif args.stage=='collect-lane':
        from .collection import collect
        collect(c,args.lane)
    elif args.stage=='fit-lane':
        from .train import fit
        for i,seed in enumerate(c['fit_seeds']):
            if i%c['lanes']==args.lane:
                # Response fit supplies the declared budget for its time control.
                for arm in ['responses8','values8','values32','values8_time']:fit(c,arm,seed)
    elif args.stage=='evaluate':
        from .evaluate import evaluate
        evaluate(c)
    elif args.stage=='worker':
        gate(c)
        # Release the coordinator's gate allocations before launching GPU lanes.
        import gc
        import torch
        gc.collect();torch.cuda.empty_cache()
        parallel(c,'collect-lane')
        from .collection import seal
        seal(c);parallel(c,'fit-lane')
        from .evaluate import evaluate
        evaluate(c)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('stage',choices=('prepare','gate','submit','worker','collect-lane','fit-lane','evaluate'))
    p.add_argument('--config',required=True);p.add_argument('--lane',type=int,default=0)
    args=p.parse_args();c=read(args.config)
    if args.stage=='submit':submit(c);return
    bind(c)
    def stop(signum,frame):raise TimeoutError(f'Scheduler signal{signum}; completed batches/epochs preserved')
    signal.signal(signal.SIGTERM,stop)
    try:
        with recorded_stage(root(c)/'technical'/f'stage-{args.stage}-{args.lane}.json',job=os.environ.get('SLURM_JOB_ID')):
            execute(c,args)
    except TimeoutError:
        job=os.environ.get('SLURM_JOB_ID');restarts=int(os.environ.get('SLURM_RESTART_COUNT','0'))
        if args.stage!='worker' or not job or restarts>=2 or time.time()<allocation_deadline(reserve_seconds=420,job=job):raise
        write_json(root(c)/'technical'/f'requeue-{restarts+1}.json',dict(job=job,reason='allocation ending',maximum_restarts=2))
        signal.signal(signal.SIGTERM,signal.SIG_IGN);subprocess.run(['scontrol','requeue',job],check=True)


if __name__=='__main__':main()
