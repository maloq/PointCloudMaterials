"""Bounded, detached two-GPU preparation, fitting and independent evaluation."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import torch
from src.experiment_runner.execution import ExecutionBundle,SlurmQueue,allocation_deadline
from src.experiment_runner.metric_docs import check_metric_docs
from src.research.measurement_sim.workflow import environment as base_environment
from .data import read,root,archive,prepare,collect,sha,write_json,FAMILY

MODULE='src.research.ni_phase_sim'
REPO=Path(__file__).resolve().parents[3]


def environment(bundle):
    values=base_environment(bundle)
    values['PATH']=str(Path(sys.executable).parent)+':'+values['CUDA_HOME']+'/bin:'+values['PATH']
    return values


def launch(c):
    p=prepare(c); check_metric_docs(family=FAMILY); technical=root(c)/'technical'
    bundle=ExecutionBundle.freeze(REPO,technical/'source',c,directories=['src','configs','docs/metrics'],
                                  files=[(REPO/'machine.local.yaml','machine.local.yaml')])
    write_json(technical/'source-files.json',{str(f.relative_to(bundle.root)):sha(f) for f in bundle.root.rglob('*') if f.is_file()})
    receipt=dict(jobs={},started=time.time(),models=len(c['arms']),counts=p['counts'],
                 origin_allocation=os.environ.get('SLURM_JOB_ID'),independent_of_chat=True)
    queue=SlurmQueue(technical,bundle,MODULE,environment(bundle),technical/'submission.json',receipt,'ni-phase')
    with queue.submission():
        queue.submit('overnight',['--gres=gpu:2','--cpus-per-task=6','--mem=64G',
                     f'--time={c["runtime"]["hours"]:02d}:00:00'],partition=c['runtime']['partitions'],command_stage='team')
    print(receipt,flush=True)


def team(c,bundle):
    technical=root(c)/'technical'; deadline=allocation_deadline(reserve_seconds=120)
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','0,1').split(',')
    processes=[]; receipt=dict(allocation=os.environ['SLURM_JOB_ID'],started=time.time(),deadline=deadline,workers=[])
    for lane in range(2):
        command=[sys.executable,'-u','-m',MODULE,'worker','--config',str(bundle.config_path),
                 '--lane',str(lane),'--deadline',str(deadline)]
        log=technical/f'worker-{receipt["allocation"]}-{lane}.log'
        with log.open('a') as stream:
            process=subprocess.Popen(command,cwd=bundle.root,env=dict(os.environ,**environment(bundle),CUDA_VISIBLE_DEVICES=devices[lane]),
                stdout=stream,stderr=subprocess.STDOUT)
        processes.append(process); receipt['workers'].append(dict(pid=process.pid,lane=lane,device=devices[lane],log=str(log)))
    write_json(technical/f'launch-{receipt["allocation"]}.json',receipt)
    codes=[p.wait() for p in processes]
    write_json(technical/'team-exit.json',dict(codes=codes,finished=time.time()))
    if any(codes): raise RuntimeError(f'Phase-space worker failures: {codes}')


def ready(c,p,role):
    return all((archive(c)/j['key']/'complete.json').exists() for j in p['jobs'] if j['role']==role)


def collect_one(c,p,lane,deadline,role):
    for job in p['jobs']:
        if job['role']!=role: continue
        folder=archive(c)/job['key']; folder.mkdir(parents=True,exist_ok=True)
        if (folder/'complete.json').exists() or (folder/'failed.json').exists(): continue
        with (folder/'worker.lock').open('a') as lock:
            try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: continue
            if (folder/'complete.json').exists(): continue
            for attempt in range(c['runtime']['max_attempts']):
                record=dict(stage='teacher',job=job['key'],lane=lane,role=role,attempt=attempt+1,started=time.time())
                write_json(root(c)/'technical'/f'lane-{lane}.json',dict(record,state='running'))
                try:
                    done=collect(c,p,job,deadline-300)
                except Exception:
                    record.update(state='failed',traceback=traceback.format_exc(),finished=time.time())
                    write_json(folder/f'attempt-{time.time_ns()}.json',record); print(record['traceback'],flush=True)
                else:
                    print('teacher',job['key'],done['seconds'],flush=True); return True
            write_json(folder/'failed.json',record); return True
    return False


def stage(c,bundle,lane,deadline,arm,name,marker):
    folder=root(c)/'analyses'/arm['name']/'technical'; folder.mkdir(parents=True,exist_ok=True)
    if (folder/marker).exists() or (folder/(name+'-failed.json')).exists(): return None
    with (folder/'worker.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return None
        if (folder/marker).exists(): return None
        for attempt in range(c['runtime']['max_attempts']):
            command=[sys.executable,'-u','-m',MODULE,name,'--config',str(bundle.config_path),
                     '--name',arm['name'],'--deadline',str(deadline)]
            record=dict(stage=name,arm=arm['name'],lane=lane,attempt=attempt+1,started=time.time(),state='running')
            log=folder/f'{name}-{time.time_ns()}.log'; torch.cuda.empty_cache()
            with log.open('a') as stream:
                process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
                record.update(pid=process.pid,log=str(log),command=command)
                write_json(root(c)/'technical'/f'lane-{lane}.json',record); code=process.wait()
            record.update(exit_code=code,finished=time.time(),state='paused' if code==75 else ('complete' if code==0 else 'failed'))
            write_json(folder/f'{name}-attempt-{time.time_ns()}.json',record)
            if code==75: return 'paused'
            if (folder/marker).exists(): return 'done'
        write_json(folder/(name+'-failed.json'),record); return 'failed'


def worker(c,lane,deadline,bundle):
    p=read(root(c)/'technical/plan.json'); paused=False
    while time.time()<deadline-600:
        if not ready(c,p,'train'):
            if collect_one(c,p,lane,deadline,'train'): continue
        if lane==1 and collect_one(c,p,lane,deadline,'test'): continue
        progress=False
        if ready(c,p,'train'):
            for arm in c['arms']:
                folder=root(c)/'analyses'/arm['name']/'technical'
                for name,marker in [('train','complete.json'),('evaluate','evaluation-complete.json')]:
                    if name=='evaluate' and (not (folder/'complete.json').exists() or not ready(c,p,'test')): continue
                    result=stage(c,bundle,lane,deadline,arm,name,marker)
                    if result=='paused': paused=True; break
                    if result is not None: progress=True; break
                if progress or paused: break
        if paused: break
        if progress: continue
        if collect_one(c,p,lane,deadline,'test'): continue
        with (root(c)/'technical/comparison.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            from .assessment import compare
            compare(c)
        incomplete=[a['name'] for a in c['arms'] if not (root(c)/'analyses'/a['name']/'technical/evaluation-complete.json').exists()]
        write_json(root(c)/'technical'/f'queue-status-{lane}.json',dict(pending=incomplete,updated=time.time()))
        if not incomplete: break
        busy=False
        folders=[archive(c)/j['key'] for j in p['jobs']]+[root(c)/'analyses'/a['name']/'technical' for a in c['arms']]
        for folder in folders:
            if not (folder/'worker.lock').exists(): continue
            with (folder/'worker.lock').open('a') as lock:
                try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError: busy=True; break
        if not busy:
            write_json(root(c)/'technical'/f'blocked-{lane}.json',dict(pending=incomplete,reason='Remaining stages have recorded failures'))
            break
        time.sleep(15)
    write_json(root(c)/'technical'/f'worker-exit-{lane}.json',dict(paused=paused,finished=time.time()))


def dispatch(c,args):
    bundle=ExecutionBundle(Path(args.config).resolve().parent)
    if args.stage=='launch': launch(c)
    elif args.stage=='team': team(c,bundle)
    elif args.stage=='worker': worker(c,args.lane,args.deadline,bundle)
    elif args.stage=='compare':
        from .assessment import compare
        compare(c)
    else:
        arm=next(a for a in c['arms'] if a['name']==args.name)
        if args.stage=='train':
            from .learning import fit
            done=fit(c,arm,args.deadline)
        else:
            from .assessment import run
            done=run(c,arm,args.deadline)
        if not done: sys.exit(75)
