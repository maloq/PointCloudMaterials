"""Two resumable GPU lanes: matched quenches, response targets, fits and probes."""
import argparse
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
from src.simulation.eam_gpu.relaxation_service import RelaxationService
from .data import read,root,cache,prepare,assemble,write_json,sha,resolve_path,FAMILY
from . import quench,response,targets,train,evaluate


def barrier(out,name,p,deadline):
    while not all((out/f'{name}-{lane}.json').exists() for lane in [0,1]):
        if time.time()>deadline-750:raise TimeoutError('Continuation at '+name)
        time.sleep(10)
    for lane in [0,1]:
        if read(out/f'{name}-{lane}.json')['identity']!=p['identity']:raise ValueError('Changed lane barrier')


def worker(c,p,lane,deadline):
    import torch
    torch.set_num_threads(2);out=root(c)/'technical';micro=c['runtime']['microbatch']
    with (out/f'worker-{lane}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while subprocess.run(['timeout','-k','1','5','stat',str(resolve_path(c['relaxation_scratch']))],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
            write_json(out/f'worker-{lane}.json',dict(stage='waiting_for_scratch_storage',time=time.time()))
            if time.time()>deadline-1500:return False
            time.sleep(15)
        if lane==0 and not (out/'probe.json').exists():
            if time.time()>deadline-1500:return False
            probe(c,p)
        while not (out/'probe.json').exists():
            if time.time()>deadline-750:return False
            time.sleep(10)
        with RelaxationService() as service:
            tasks=p['native_tasks'][lane::2]
            for i,t in enumerate(tasks):
                if time.time()>deadline-750:return False
                done=quench.native(c,p,t,service)
                write_json(out/f'worker-{lane}.json',dict(stage='matched_native_quenches',done=i+1,total=len(tasks),last=t['key'],seconds=done['seconds'],job=os.environ.get('SLURM_JOB_ID')))
            write_json(out/f'native-complete-{lane}.json',dict(identity=p['identity']))
            barrier(out,'native-complete',p,deadline)
            for i,j in enumerate(p['response_jobs'][lane::2]):
                response.collect(c,p,j,service,deadline)
                write_json(out/f'worker-{lane}.json',dict(stage='paired_response_quenches',done=i+1,total=len(p['response_jobs'][lane::2]),last=j['key']))
            write_json(out/f'response-complete-{lane}.json',dict(identity=p['identity']))
        barrier(out,'response-complete',p,deadline)
        with (out/'assemble.lock').open('a+') as assembly:
            fcntl.flock(assembly,fcntl.LOCK_EX);assemble(c,p)
        targets.prepare_lane(c,p,lane,micro,deadline)
        barrier(cache(c)/'target-packets','lane',p,deadline)
        with (out/'targets.lock').open('a+') as assembly:
            fcntl.flock(assembly,fcntl.LOCK_EX);targets.seal(c,p)
        for arm in c['arms'][lane::2]:
            if not train.run(c,p,arm,deadline,micro):return False
            evaluate.infer(c,p,arm['name'],micro)
        if lane==0:evaluate.infer(c,p,'spatial',micro)
        write_json(out/f'worker-{lane}-complete.json',dict(identity=p['identity'],state='complete'))
        return True


def submit(c,local):
    check_metric_docs(family=FAMILY);p=prepare(c);out=root(c)/'technical'
    if (out/'launch.json').exists():raise ValueError('Already submitted; use receipts')
    bundle=ExecutionBundle.freeze(REPO,out/'code',c,directories=['src','configs','docs/metrics'],files=[(REPO/'machine.local.yaml','machine.local.yaml')])
    env=dict(PCM_PROJECT_ROOT=str(REPO),CUDA_HOME='/usr/local/cuda',PATH=str(Path(sys.executable).parent)+':/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin',PYTHONPATH=str(bundle.root),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OVITO_THREAD_COUNT='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
    env['UCX_TLS']='self,sm,tcp'
    receipt=dict(identity=p['identity'],jobs={},owner='current chat: user-selected original MD with relaxed eSEN observations',bundle=str(bundle.root),local_allocation=os.environ['SLURM_JOB_ID'] if local else None,started=time.time())
    queue=SlurmQueue(out,bundle,'src.research.ni_relaxed_response.workflow',env,out/'launch.json',receipt,'ni-cold')
    with queue.submission():
        jobs=[]
        for lane in [0,1]:
            jobs.append(queue.submit(f'worker-{lane}',['--gres=gpu:1','--cpus-per-task=4','--mem=48G','--time=24:00:00','--requeue'],partition=c['runtime']['partitions'],dependency='afterany:'+receipt['local_allocation'] if local else None,command_stage='worker',arguments=['--lane',str(lane)]))
        evaluation=queue.submit('evaluate',['--cpus-per-task=4','--mem=48G','--time=12:00:00'],dependency='afterok:'+':'.join(jobs))
        parent_evaluation=read(Path(c['parent'])/'technical/launch.json')['jobs']['evaluate']
        queue.submit('compare',['--cpus-per-task=4','--mem=48G','--time=12:00:00'],dependency=f'afterok:{evaluation}:{parent_evaluation}')
    if local:
        processes=[]
        for lane in [0,1]:
            with (out/f'local-worker-{lane}.log').open('a') as log:
                proc=subprocess.Popen(queue.command('worker','--lane',str(lane),'--local'),cwd=bundle.root,env=dict(os.environ,**env,CUDA_VISIBLE_DEVICES=str(lane)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append(proc.pid)
        write_json(out/'local-launch.json',dict(pids=processes,allocation=receipt['local_allocation'],gpus=[0,1],deadline_unix=allocation_deadline(reserve_seconds=120),owner=receipt['owner']))
    return receipt


def probe(c,p):
    import torch
    from src.research.ni_response.model import Motion
    from .data import native_packet
    torch.set_num_threads(2)
    job=p['response_jobs'][0];task=next(t for t in p['native_tasks'] if t['source']==job['source_key'] and t['frame']==job['frame'])
    with RelaxationService() as service:
        quench.native(c,p,task,service);response.collect(c,p,job,service,time.time()+3600)
        md=torch.load(response.folder(c,job)/'restart.pt',weights_only=False,map_location='cpu')
        repeat,_=quench.virtual(c,job['key']+'-repeatability-base',md['future'][0,0].numpy(),md['box'],md['atom_ids'],md['center'],md['neighbors'],service,dict(job=job['key'],stage='independent_repeated_quench_of_same_endpoint'),reuse=False)
    bank=torch.load(response.folder(c,job)/'response.pt',weights_only=False,map_location='cpu')
    model=Motion(c).cuda().eval();model.set_arm(c['arms'][0]);x=bank['state'].cuda();q=bank['future'].flatten(0,1).cuda()
    with torch.no_grad():
        y=model.geometry(q).reshape(c['teacher']['shots'],5,128).mean(0).cpu();z=model.base_encode(x,create_graph=False).cpu()
        repeated=model.geometry(torch.from_numpy(repeat['positions']).float().cuda()*c['length_factor']).cpu()
        original=model.geometry(bank['future'][0,:1].cuda()).cpu()
    eps=c['student_difference']['response_epsilon']
    r=dict(state=bank['state'][:1],input_pairs=torch.stack([bank['state'][1:3],bank['state'][3:5]])[:,:,None],z=z[:1],z_pairs=torch.stack([z[1:3],z[3:5]])[:,:,None],target=y[:1],response=torch.stack([(y[2]-y[1])/(2*eps),(y[4]-y[3])/(2*eps)],-1)[None])
    v,d=targets.short_objective(model,r,c,c['arms'][0],c['seed']);(.1*v+.1*d).backward()
    grads=[v.grad for v in model.parameters() if v.grad is not None]
    if not all(torch.isfinite(g).all() for g in grads):raise FloatingPointError('Nonfinite relaxed response gradient')
    result=dict(value=float(v.detach()),finite_response=float(d.detach()),teacher_response_rms=r['response'].square().mean((0,1)).sqrt().tolist(),gradient_norm=float(torch.cat([g.flatten() for g in grads]).norm()),finite_gradients=True,scientific_updates=0)
    result['repeated_quench_embedding_rms']=float((repeated-original).square().mean().sqrt())
    write_json(root(c)/'technical/probe.json',result);print(result,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','probe','submit','worker','evaluate','compare']);ap.add_argument('--config',required=True);ap.add_argument('--lane',type=int,default=0);ap.add_argument('--local',action='store_true');a=ap.parse_args();c=read(a.config)
    if a.stage=='prepare':print(prepare(c)['summary']);return
    if a.stage=='submit':print(submit(c,a.local));return
    p=read(root(c)/'technical/plan.json');out=root(c)/'technical'
    try:
        if a.stage=='probe':probe(c,p)
        elif a.stage=='worker':
            try:done=worker(c,p,a.lane,allocation_deadline(reserve_seconds=120))
            except TimeoutError:done=False
            if not done:
                write_json(out/f'continuation-{a.lane}.json',dict(state='checkpointed',local=a.local,time=time.time()))
                if not a.local:
                    job=os.environ['SLURM_JOB_ID']
                    if read(out/'launch.json')['jobs'][f'worker-{a.lane}']!=job:raise ValueError('Refuse requeue of unowned job')
                    subprocess.run(['scontrol','requeue',job],check=True)
        elif a.stage=='evaluate':
            for name in ['spatial']+[arm['name'] for arm in c['arms']]:evaluate.evaluate(c,p,name)
        elif a.stage=='compare':
            from .comparison import run
            run(c,p)
    except BaseException:
        write_json(out/f'failed-{a.stage}-{a.lane}.json',dict(error=traceback.format_exc(),time=time.time()));raise

if __name__=='__main__':main()
