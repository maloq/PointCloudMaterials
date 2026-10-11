"""Detached GPU workers with shared shot/fit leases and resumable provenance."""
import argparse
import fcntl
import gc
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from src.project_runtime.paths import resolve_path,machine
from src.experiment_runner.execution import ExecutionBundle,SlurmQueue,recorded_stage,allocation_deadline
from src.experiment_runner.metric_docs import check_metric_docs,write_metric_rows
from .common import FAMILY,root,bind,read,write_json,deadline


def smoke(c):
    import torch
    from .data import prepare,velocity
    from .model import Predictor,initialize,derivatives
    torch.set_num_threads(2);torch.manual_seed(c['seed'])
    states=prepare(c)[:8]
    x=torch.stack([torch.cat((s['q'][:80],velocity(c,s,0)[:80]),-1)for s in states]).float().cuda()
    basis=torch.zeros(*x.shape,4,device='cuda')
    for i,s in enumerate(states):
        basis[i,:,:3,:2]=s['basis'].float().cuda()
        basis[i,:,3:,2:]=s['basis'].float().cuda()*c['velocity_scale_A_fs']
    counts=[];records=[]
    for enabled in [False,True]:
        torch.manual_seed(c['seed']);m=Predictor(c,enabled).cuda();initialize(m,x,4);m.eval()
        counts.append(sum(p.numel()for p in m.parameters()))
        h=derivatives(m,x[:2],basis[:2],create_graph=True)
        (m(x[:2]).square().mean()+h.square().mean()).backward()
        grad=float(torch.nn.utils.clip_grad_norm_(m.encoder.parameters(),10,error_if_nonfinite=True))
        if grad<=0:raise ValueError('Response objective does not reach encoder')
        rot,_=torch.linalg.qr(torch.randn(3,3,device='cuda'))
        rotated=torch.cat((x[:2,:,:3]@rot,x[:2,:,3:]@rot),-1)
        with torch.no_grad():
            reference=m(x[:2]);rot_error=float((m(rotated)-reference).abs().max())
            altered=x[:2].clone();altered[:,:,3:]*=-1
            v_effect=float((m(altered)-reference).norm())
        velocity_jvp=float(h[...,2:].detach().abs().max())
        if rot_error>1e-4 or (enabled and v_effect<=1e-9) or (not enabled and (v_effect>1e-5 or velocity_jvp!=0)):
            raise ValueError(f'Input symmetry/velocity gate: {enabled},{rot_error},{v_effect}')
        records.append(dict(velocities=enabled,gradient_norm=grad,rotation_error=rot_error,velocity_effect=v_effect,
                            velocity_jvp_max=velocity_jvp))
        del m;gc.collect();torch.cuda.empty_cache()
    if len(set(counts))!=1:raise ValueError('Q/QV parameter counts differ')
    write_json(root(c)/'technical/student-gate.json',dict(passed=True,parameters=counts[0],checks=records,wandb_runs=0))


def run_worker(c,lane):
    import torch
    from .data import prepare
    from .collection import gate,collect,seal
    from .train import fit
    from .evaluate import clustering
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    tech=root(c)/'technical';stop_at=deadline(c,reserve_seconds=300)
    if c.get('compiler_cache'):
        major,minor=torch.cuda.get_device_capability();cache=resolve_path(c['compiler_cache'])/str(lane)
        os.environ.update(TORCH_CUDA_ARCH_LIST=f'{major}.{minor}',TORCH_EXTENSIONS_DIR=str(cache/'extensions'),
            TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'))
        write_json(tech/f'hardware-{lane}.json',dict(device=torch.cuda.get_device_name(),architecture=f'{major}.{minor}',
            torch=torch.__version__,cuda=torch.version.cuda,host=os.uname().nodename,pid=os.getpid(),
            visible=os.environ.get('CUDA_VISIBLE_DEVICES'),allocation=os.environ.get('SLURM_JOB_ID')))
    with (tech/'prepare.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        prepare(c)
    with (tech/'gate.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not (tech/'student-gate.json').exists():smoke(c)
        gate(c)
    gc.collect();torch.cuda.empty_cache()
    expected=sum(c.get('parent_counts',[90,15,30]))*c['velocity_fields']
    while not (tech/'data-complete.json').exists():
        if time.time()>=stop_at:raise TimeoutError('Worker time budget')
        collect(c,lane)
        with (tech/'seal.lock').open('a+')as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            complete=list(resolve_path(c['simulation_archive']).glob('parent*-velocity*/complete.json'))
            if len(complete)==expected and not (tech/'data-complete.json').exists():seal(c)
        if not (tech/'data-complete.json').exists():time.sleep(20)
    for seed in c['fit_seeds']:
        for arm in c['arms']:
            name=arm['name']+'-'+str(seed)
            with (tech/'locks'/(name+'.lock')).open('a+')as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                write_json(tech/f'fit-{lane}.json',dict(state='running',model=name))
                fit(c,arm,seed);clustering(c,arm,seed);gc.collect();torch.cuda.empty_cache()
    with (tech/'report.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        # Re-read receipts while holding the publisher lock to avoid stale overwrites.
        predictive=[];clusters=[];finished=0
        for seed in c['fit_seeds']:
            for arm in c['arms']:
                name=arm['name']+'-'+str(seed)
                p=root(c)/'analyses'/name/'technical/complete.json';q=root(c)/'cluster-evaluation'/name/'technical/complete.json'
                if p.exists():predictive.extend(dict(model=name,**r)for r in read(p)['metrics'])
                if q.exists():clusters.extend(read(q)['metrics']);finished+=1
        if predictive:
            columns=['model','horizon_fs','selected_update','selection_nll','test_nll','value_corrected_mse',
                'position_response_corrected_mse','velocity_response_corrected_mse','force_free_corrected_mse','shuffled_velocity_corrected_mse']
            write_metric_rows(predictive,root(c)/'analyses/comparison-v1',family=FAMILY,name='predictive',columns=columns)
        if clusters:write_metric_rows(clusters,root(c)/'analyses/comparison-v1',family=FAMILY,name='clustering')
        if finished==len(c['arms'])*len(c['fit_seeds']):
            write_json(tech/'complete.json',dict(state='complete',fits=finished))
    write_json(tech/f'fit-{lane}.json',dict(state='complete'))


def launch(c):
    bind(c);check_metric_docs(family=FAMILY)
    from .data import prepare
    prepare(c)
    import wandb
    if not wandb.Api(timeout=30).viewer:raise RuntimeError('W&B authentication failed; no offline fallback')
    tech=root(c)/'technical';path=tech/'launch.json'
    if path.exists():raise FileExistsError(path)
    if 'execution' in c:
        started=time.time()
        write_json(tech/'budget.json',dict(started_unix=started,
            deadline_unix=started+c['execution']['wall_seconds'],automatic_requeue=False,
            deadline_Paris=datetime.fromtimestamp(started+c['execution']['wall_seconds'],ZoneInfo('Europe/Paris')).isoformat(),
            includes_queue_wait=True,owner='current response/nucleus-recovery chat'))
    repo=Path(__file__).resolve().parents[3]
    bundle=ExecutionBundle.freeze(repo,tech/'code',c,directories=('src','docs/metrics'))
    environment=dict(PCM_PROJECT_ROOT=str(repo),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
        WANDB_MODE='online',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',TZ='Europe/Paris',UCX_TLS='tcp,self')
    if c.get('teacher_backend')=='cuda_meam':
        cuda=machine()['execution'][c.get('cuda_root_key','gpu_meam_cuda_root')]
        environment.update(CUDA_HOME=cuda,PATH=cuda+'/bin:'+str(Path(sys.prefix)/'bin')+':/usr/bin:/bin',
            MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='12.0',TORCHINDUCTOR_COMPILE_THREADS='2')
    receipt=dict(jobs={},code=str(bundle.root),owner='current response/nucleus-recovery chat');queue=SlurmQueue(tech,bundle,'src.research.response_velocity.queue',
        environment,path,receipt,'RESP-QV')
    with queue.submission():
        resources=c.get('gpu_jobs',dict(jobs=1,gpus_per_job=2,cpus_per_job=8,memory='96G'))
        options=[f"--gpus={resources['gpus_per_job']}",f"--cpus-per-task={resources['cpus_per_job']}",
                 '--mem='+resources['memory'],'--signal=B:TERM@420','--exclude=node52']
        if 'execution' in c:
            options+=[f'--time={c["execution"].get("allocation_minutes",180)}','--no-requeue',
                '--deadline='+datetime.fromtimestamp(read(tech/'budget.json')['deadline_unix'],ZoneInfo('Europe/Paris')).strftime('%Y-%m-%dT%H:%M:%S')]
        else:options+=['--time=24:00:00','--requeue']
        for i in range(resources['jobs']):
            name='coordinator' if resources['jobs']==1 else f'coordinator-{i}'
            queue.submit(name,options,partition=c['gpu_partition'],command_stage='coordinator')
    # The user's current allocation contributes immediately, sharing all leases.
    if 'local_gpus' in c:
        for gpu in c['local_gpus']:
            lane=f'helper-{os.environ["SLURM_JOB_ID"]}-{gpu}'
            command=queue.command('worker','--lane',lane)
            with (tech/f'worker-{lane}.log').open('a') as log:
                child=subprocess.Popen(command,cwd=bundle.root,env={**os.environ,**environment,'CUDA_VISIBLE_DEVICES':str(gpu)},
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write_json(tech/f'helper-{lane}-launch.json',dict(pid=child.pid,process_group=child.pid,
                owner='current response/nucleus-recovery chat',allocation=os.environ['SLURM_JOB_ID'],host=os.uname().nodename,
                gpu=gpu,command=command,code=str(bundle.root),budget=read(tech/'budget.json')))
    else:
        log=(tech/'helper.log').open('a')
        command=queue.command('worker','--lane','helper-'+os.environ['SLURM_JOB_ID'])
        child=subprocess.Popen(command,cwd=bundle.root,env=dict(os.environ,**environment),stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
        write_json(tech/'helper-launch.json',dict(pid=child.pid,allocation=os.environ['SLURM_JOB_ID'],command=command,code=str(bundle.root)))
    print(receipt,flush=True)


def coordinator(c):
    devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    expected=c.get('gpu_jobs',dict(gpus_per_job=2))['gpus_per_job']
    if len(devices)!=expected:raise ValueError(f'Expected {expected} allocated GPUs')
    children=[]
    try:
        for i,device in enumerate(devices):
            lane=f"{os.environ['SLURM_JOB_ID']}-{i}"
            log=(root(c)/'technical'/f'worker-{lane}.log').open('a')
            command=[sys.executable,'-u','-m',__package__+'.queue','worker','--config',str(Path('config.json').resolve()),'--lane',lane]
            p=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=device),stdout=log,stderr=subprocess.STDOUT)
            children.append((p,log))
        while True:
            codes=[p.poll()for p,_ in children]
            if 124 in codes:raise TimeoutError('GPU worker reached allocation reserve')
            if any(code not in [None,0]for code in codes):raise RuntimeError('GPU worker failed; inspect logs')
            if all(code==0 for code in codes):break
            time.sleep(5)
    finally:
        for p,log in children:
            if p.poll()is None:p.terminate()
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            log.close()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['prepare','prepare-clustering','smoke','gate','launch','worker','coordinator'])
    p.add_argument('--config',required=True);p.add_argument('--lane',default='local');a=p.parse_args();c=read(a.config)
    bind(c);check_metric_docs(family=FAMILY)
    def stop(signum,frame):raise TimeoutError(f'Scheduler signal{signum}; resumable artifacts retained')
    signal.signal(signal.SIGTERM,stop)
    if 'execution' in c and a.stage in ['worker','coordinator']:
        signal.signal(signal.SIGALRM,stop)
        signal.setitimer(signal.ITIMER_REAL,max(.01,deadline(c,reserve_seconds=0)-time.time()))
    try:
        stage_lane=os.environ.get('SLURM_JOB_ID','local') if a.stage=='coordinator' else a.lane
        with recorded_stage(root(c)/'technical'/f'stage-{a.stage}-{stage_lane}.json',job=os.environ.get('SLURM_JOB_ID')):
            if a.stage=='prepare':
                from .data import prepare
                prepare(c)
            elif a.stage=='prepare-clustering':
                from .native import static_velocities
                from src.research.response_geometry.clusters import references
                arrays,fit,held,_=references(c['clustering'])
                static_velocities(c,arrays,fit,held)
            elif a.stage=='gate':
                from .collection import gate
                gate(c)
            elif a.stage=='worker':run_worker(c,a.lane)
            else:globals()[a.stage](c)
    except TimeoutError:
        job=os.environ.get('SLURM_JOB_ID');restart=int(os.environ.get('SLURM_RESTART_COUNT','0'))
        if 'execution' not in c and a.stage=='coordinator' and job and restart<2 and time.time()>=allocation_deadline(reserve_seconds=600):
            signal.signal(signal.SIGTERM,signal.SIG_IGN);subprocess.run(['scontrol','requeue',job],check=True)
        else:sys.exit(124)


if __name__=='__main__':main()
