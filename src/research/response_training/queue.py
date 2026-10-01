"""One resumable GPU lane: fresh simulator labels, nine fits, paired evaluation."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, recorded_stage, allocation_deadline
from src.experiment_runner.metric_docs import check_metric_docs
from .common import FAMILY,root,read,bind,write_json


def submit(c):
    record=bind(c);check_metric_docs(family=FAMILY)
    gate=read(root(c)/'technical/preflight.json')
    if gate['state']!='complete' or gate['identity']!=record['identity']:raise ValueError('Numerical preflight required')
    import wandb
    viewer=wandb.Api(timeout=30).viewer
    if not viewer:raise RuntimeError('Online W&B authentication failed')
    tech=root(c)/'technical';receipt_path=tech/'launch.json'
    if receipt_path.exists():
        receipt=read(receipt_path)
        if receipt['jobs']:raise ValueError('Already queued; inspect the recorded worker before an explicit continuation')
        bundle=ExecutionBundle(Path(receipt['code']))
    else:
        repo=Path(__file__).resolve().parents[3]
        bundle=ExecutionBundle.freeze(repo,tech/'code',c,directories=('src','docs/metrics','configs/response_atlas','configs/simulation'))
        receipt=dict(identity=record['identity'],jobs={},code=str(bundle.root))
    queue=SlurmQueue(tech,bundle,'src.research.response_training.queue',
        dict(PCM_PROJECT_ROOT=str(Path(__file__).resolve().parents[3]),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
             MKL_NUM_THREADS='2',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',WANDB_MODE='online'),receipt_path,receipt,'RESP-TRAIN')
    options=['--gpus=1','--cpus-per-task=4','--mem=64G',f'--time={c["time_limit"]}','--signal=B:TERM@240','--requeue']
    if c['exclude_nodes']:options.append('--exclude='+','.join(c['exclude_nodes']))
    with queue.submission():queue.submit('worker',options,partition=c['gpu_partition'])
    print(receipt,flush=True)


def worker(c):
    from .data import collect
    from .train import fit
    from .evaluate import collect as evaluate
    # Every branch and optimizer epoch can resume. Scientific errors propagate.
    collect(c)
    for index,seed in enumerate(c['fit_seeds']):
        arms=c['arms'][index:]+c['arms'][:index]
        for arm in arms:fit(c,arm,seed)
    evaluate(c)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('prepare','preflight','submit','worker','collect','train','evaluate'))
    parser.add_argument('--config',required=True)
    parser.add_argument('--arm',choices=('values8','responses8','values32'))
    parser.add_argument('--seed',type=int)
    args=parser.parse_args();c=read(args.config)
    if args.stage=='submit':submit(c);return
    def stop(signum,frame):raise TimeoutError(f'Scheduler signal {signum}; resume saved branches/optimizer epochs')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
    bind(c)
    try:
        execute(c,args)
    except TimeoutError:
        job=os.environ.get('SLURM_JOB_ID')
        restarts=int(os.environ.get('SLURM_RESTART_COUNT','0'))
        if (args.stage!='worker' or not job or restarts>=2
                or time.time()<allocation_deadline(reserve_seconds=360,job=job)):
            raise
        # Only actual allocation exhaustion gets a bounded continuation. Model,
        # numerical and network errors outside that window still fail loudly.
        write_json(root(c)/'technical'/f'requeue-{restarts+1}.json',
            dict(job=job,reason='allocation ending; branches and optimizer epochs saved',
                 requested_at=time.time(),next_restart=restarts+1,maximum_restarts=2))
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        subprocess.run(['scontrol','requeue',job],check=True)
        print(f'Requeued {job} for bounded continuation {restarts+1}/2',flush=True)


def execute(c,args):
    with recorded_stage(root(c)/'technical'/f'stage-{args.stage}.json',job=os.environ.get('SLURM_JOB_ID')):
        if args.stage=='prepare':
            from .data import prepare
            prepare(c)
        elif args.stage=='preflight':
            from .preflight import run
            run(c)
        elif args.stage=='worker':worker(c)
        elif args.stage=='collect':
            from .data import collect
            collect(c)
        elif args.stage=='train':
            from .train import fit
            if args.arm is None or args.seed not in c['fit_seeds']:raise ValueError('Declared arm/seed required')
            fit(c,args.arm,args.seed)
        else:
            from .evaluate import collect
            collect(c)


if __name__=='__main__':main()
