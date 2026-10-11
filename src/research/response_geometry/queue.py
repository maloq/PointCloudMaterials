"""Two-GPU detached, resumable response collection and scientific comparisons."""
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
from .common import FAMILY,root,bind,read,write_json


def submit(c):
    record=bind(c);check_metric_docs(family=FAMILY);check_metric_docs(family='response_geometry_clusters')
    from .data import prepare
    prepare(c)
    import wandb
    if not wandb.Api(timeout=30).viewer:raise RuntimeError('W&B authentication failed; no offline fallback')
    tech=root(c)/'technical';path=tech/'launch.json'
    if path.exists():raise FileExistsError(path)
    repo=Path(__file__).resolve().parents[3]
    bundle=ExecutionBundle.freeze(repo,tech/'code',c,directories=('src','docs/metrics'))
    receipt=dict(identity=record['identity'],jobs={},code=str(bundle.root))
    queue=SlurmQueue(tech,bundle,'src.research.response_geometry.queue',dict(PCM_PROJECT_ROOT=str(repo),
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',WANDB_MODE='online',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'),
        path,receipt,'RESP-GEOM')
    with queue.submission():
        queue.submit('worker',['--gpus=2','--cpus-per-task=8','--mem=96G','--time=24:00:00',
            '--signal=B:TERM@420','--requeue','--exclude=node52'],partition=c['gpu_partition'])
    print(receipt,flush=True)


def parallel(c,stage):
    devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(devices)!=2:raise ValueError('Expected two allocated GPUs')
    children=[]
    try:
        for lane,device in enumerate(devices):
            log=(root(c)/'technical'/f'{stage}-lane{lane}.log').open('a')
            command=[sys.executable,'-u','-m','src.research.response_geometry.queue',stage,
                '--config',str(Path('config.json').resolve()),'--lane',str(lane)]
            p=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=device),stdout=log,stderr=subprocess.STDOUT)
            children.append((p,log))
        while any(p.poll()is None for p,_ in children):
            codes=[p.poll()for p,_ in children]
            if 124 in codes:raise TimeoutError('Worker allocation reserve reached')
            if any(code not in [None,0]for code in codes):raise RuntimeError('GPU worker failed; inspect lane logs')
            time.sleep(5)
        if any(p.returncode for p,_ in children):raise RuntimeError('GPU lane failed')
    finally:
        for p,log in children:
            if p.poll()is None:p.terminate()
            try:p.wait(timeout=60)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            log.close()


def smoke(c):
    import torch
    from .data import prepare
    from .train import distribution_distances
    from .common import legacy
    from src.research.local_response.model import Predictor,initialize,responses
    torch.set_num_threads(2);torch.manual_seed(c['seed'])
    states=prepare(c);q=torch.stack([s['q'][:80]for s in states[:8]]).float().cuda()
    v=torch.stack([s['basis']for s in states[:8]]).float().cuda()
    model=Predictor(legacy(c)).cuda();initialize(model,q,4)
    z=model.encode(q[:4]);h=responses(model,q[:4],v[:4],create_graph=True)
    labels=torch.randn(4,8,256,device='cuda')*.01+torch.arange(4,device='cuda')[:,None,None]*.03
    target,scale=distribution_distances(labels)
    d=(z[:,None]-z[None,:]).square().mean(-1);mask=~torch.eye(4,dtype=torch.bool,device='cuda')
    loss=h.square().mean()+(d[mask]-target[mask]).square().mean();loss.backward()
    gradient=float(torch.nn.utils.clip_grad_norm_(model.encoder.parameters(),10,error_if_nonfinite=True))
    if gradient<=0:raise ValueError('Response/distance objective has no encoder gradient')
    write_json(root(c)/'technical/smoke.json',dict(state='complete',encoder_gradient_norm=gradient,
        distance_scale=float(scale),online_runs_created=0))


def execute(c,a):
    if a.stage=='prepare':
        from .data import prepare
        prepare(c)
    elif a.stage=='collect':
        from .collection import collect
        collect(c,a.lane)
    elif a.stage=='fit':
        from .train import fit,clustering
        tasks=[(arm,seed)for seed in c['fit_seeds']for arm in c['arms']]
        for i,(arm,seed)in enumerate(tasks):
            if i%2!=a.lane:continue
            fit(c,arm,seed);clustering(c,arm,seed)
    elif a.stage=='worker':
        from .collection import gate,seal
        import gc,torch
        smoke(c);gate(c);gc.collect();torch.cuda.empty_cache()
        parallel(c,'collect');seal(c);parallel(c,'fit')
        from .clusters import publish
        publish(dict(c['clustering'],output=str(root(c)/'cluster-evaluation')))
        write_json(root(c)/'technical/complete.json',dict(state='complete',fits=len(c['arms'])*len(c['fit_seeds'])))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['prepare','submit','worker','collect','fit']);p.add_argument('--config',required=True);p.add_argument('--lane',type=int,default=0)
    a=p.parse_args();c=read(a.config)
    if a.stage=='submit':submit(c);return
    bind(c);check_metric_docs(family=FAMILY)
    def stop(signum,frame):raise TimeoutError(f'Scheduler signal {signum}; archived work retained')
    signal.signal(signal.SIGTERM,stop)
    tech=root(c)/'technical'
    with (tech/f'{a.stage}-{a.lane}.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            with recorded_stage(tech/f'stage-{a.stage}-{a.lane}.json',job=os.environ.get('SLURM_JOB_ID')):execute(c,a)
        except TimeoutError:
            job=os.environ.get('SLURM_JOB_ID');restarts=int(os.environ.get('SLURM_RESTART_COUNT','0'))
            if a.stage in ['collect','fit']:sys.exit(124)
            if a.stage!='worker' or not job or restarts>=2 or time.time()<allocation_deadline(reserve_seconds=600,job=job):raise
            write_json(tech/f'requeue-{restarts+1}.json',dict(job=job,reason='allocation expiry',maximum_restarts=2))
            signal.signal(signal.SIGTERM,signal.SIG_IGN);subprocess.run(['scontrol','requeue',job],check=True)


if __name__=='__main__':main()
