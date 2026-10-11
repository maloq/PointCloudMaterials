"""Freeze, detach training in the current GPU allocation, and queue forward MD."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, recorded_stage, allocation_deadline
from src.experiment_runner.metric_docs import check_metric_docs
from .common import FAMILY, root, bind, read, write_json, load


def launch(c):
    record=bind(c);check_metric_docs(family=FAMILY)
    import torch
    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count()!=1:
        raise ValueError('Detached training requires one allocated GPU')
    if allocation_deadline(reserve_seconds=0)-time.time()<5*3600:
        raise ValueError('Need at least5 remaining allocation hours for the15-fit worker')
    import wandb
    if not wandb.Api(timeout=30).viewer:raise RuntimeError('W&B online authentication failed')
    tech=root(c)/'technical';path=tech/'launch.json'
    if path.exists():raise ValueError('Already launched; inspect receipts before resuming')
    repo=Path(__file__).resolve().parents[3]
    bundle=ExecutionBundle.freeze(repo,tech/'code',c,directories=('src','docs/metrics','configs/fixed_cohort'))
    env=dict(PCM_PROJECT_ROOT=str(repo),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
             TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',WANDB_MODE='online')
    receipt=dict(identity=record['identity'],jobs={},code=str(bundle.root))
    queue=SlurmQueue(tech,bundle,'src.research.local_response_followup.queue',env,path,receipt,'LOCAL-FOLLOW')
    with queue.submission():
        queue.submit('forward',['--gpus=1','--cpus-per-task=8','--mem=64G',f'--time={c["time_limit"]}',
            '--signal=B:TERM@300','--requeue','--exclude=node52'],partition=c['gpu_partition'])
    with (tech/'training.log').open('a') as log:
        process=subprocess.Popen(queue.command('train'),cwd=bundle.root,env=dict(os.environ,**env),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt['training']=dict(pid=process.pid,job=os.environ['SLURM_JOB_ID'],node=os.uname().nodename,
        log=str(tech/'training.log'),mode='detached inside current allocation')
    write_json(path,receipt);print(receipt,flush=True)


def execute(c,stage):
    if stage=='prepare':
        from .forward import prepare
        data=load(c)
        if data['response_scale']<=0:raise ValueError('Invalid original normalization')
        prepare(c)
    elif stage=='train':
        from .train import fit
        for seed in c['fit_seeds']:
            for arm in c['arms']:fit(c,arm,seed)
        from .evaluate import evaluate
        evaluate(c)
        write_json(root(c)/'technical/training-complete.json',dict(state='complete',fits=len(c['fit_seeds'])*len(c['arms'])))
    elif stage=='forward':
        from .forward import run
        run(c)
    elif stage=='summarize':
        from .forward import summarize
        summarize(c)
    elif stage=='evaluate':
        from .evaluate import evaluate
        evaluate(c)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=['prepare','launch','train','forward','summarize','evaluate'])
    parser.add_argument('--config',required=True);args=parser.parse_args();c=read(args.config)
    if args.stage=='launch':launch(c);return
    bind(c);check_metric_docs(family=FAMILY)
    def stop(signum,frame):raise TimeoutError(f'Scheduler signal{signum}; checkpoints and archived branches preserved')
    signal.signal(signal.SIGTERM,stop)
    tech=root(c)/'technical'
    with (tech/f'{args.stage}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            with recorded_stage(tech/f'stage-{args.stage}.json',job=os.environ.get('SLURM_JOB_ID')):
                execute(c,args.stage)
        except TimeoutError:
            job=os.environ.get('SLURM_JOB_ID');restarts=int(os.environ.get('SLURM_RESTART_COUNT','0'))
            if args.stage!='forward' or not job or restarts>=2 or time.time()<allocation_deadline(reserve_seconds=420,job=job):raise
            write_json(tech/f'requeue-{restarts+1}.json',dict(job=job,reason='allocation ending',maximum_restarts=2))
            signal.signal(signal.SIGTERM,signal.SIG_IGN);subprocess.run(['scontrol','requeue',job],check=True)


if __name__=='__main__':main()
