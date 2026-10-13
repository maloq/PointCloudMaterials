"""Detached preparation/training queue for density rates and Ni source expansion."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import torch
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, allocation_deadline
from src.experiment_runner.metric_docs import check_metric_docs
from .data import read, root, cache, write_json, sha
from .rate_data import prepare, measure_job, ready, FAMILY
from .workflow import environment as base_environment

MODULE = 'src.research.measurement_sim.rate_campaign'
REPO = Path(__file__).resolve().parents[3]


def environment(bundle):
    value = base_environment(bundle)
    value['PATH'] = str(Path(sys.executable).parent)+':'+value['CUDA_HOME']+'/bin:'+value['PATH']
    return value


def launch(c):
    p=prepare(c); check_metric_docs(family=FAMILY); technical=root(c)/'technical'
    bundle=ExecutionBundle.freeze(REPO,technical/'source',c,directories=['src','configs','docs/metrics'],
                                  files=[(REPO/'machine.local.yaml','machine.local.yaml')])
    write_json(technical/'source-files.json',{str(f.relative_to(bundle.root)):sha(f) for f in bundle.root.rglob('*') if f.is_file()})
    receipt=dict(jobs={},parent_allocation=os.environ['SLURM_JOB_ID'],started=time.time(),
                 models=len(c['arms']),physical_states=len(p['jobs']),test_states=len(p['test']))
    queue=SlurmQueue(technical,bundle,MODULE,environment(bundle),technical/'submission.json',receipt,'ni-rates')
    with queue.submission():
        previous=os.environ['SLURM_JOB_ID']
        for index in range(c['runtime']['continuations']):
            previous=queue.submit(f'continuation-{index+1}',
                ['--gres=gpu:2','--cpus-per-task=6','--mem=64G',f'--time={c["runtime"]["hours"]:02d}:00:00'],
                dependency='afterany:'+previous,partition=c['runtime']['partitions'],command_stage='team')
    team(c,bundle,detach=True)
    print('Submitted density-rate campaign',receipt,flush=True)


def team(c,bundle,*,detach):
    technical=root(c)/'technical'
    deadline=min(time.time()+c['runtime']['hours']*3600,allocation_deadline(reserve_seconds=180))
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    devices=visible.split(',') if visible else [str(i) for i in range(torch.cuda.device_count())]
    if len(devices)<2: raise RuntimeError('Two allocated GPUs required')
    receipt=dict(allocation=os.environ.get('SLURM_JOB_ID'),started=time.time(),deadline=deadline,workers=[])
    processes=[]
    for lane in range(2):
        command=[sys.executable,'-u','-m',MODULE,'worker','--config',str(bundle.config_path),
                 '--lane',str(lane),'--deadline',str(deadline)]
        env=dict(os.environ,**environment(bundle),CUDA_VISIBLE_DEVICES=devices[lane])
        log=technical/f'worker-{receipt["allocation"]}-{lane}.log'
        with log.open('a') as stream:
            process=subprocess.Popen(command,cwd=bundle.root,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=detach)
        processes.append(process)
        receipt['workers'].append(dict(pid=process.pid,lane=lane,device=devices[lane],command=command,log=str(log)))
    write_json(technical/f'launch-{receipt["allocation"]}.json',receipt)
    print(receipt,flush=True)
    if not detach:
        codes=[p.wait() for p in processes]
        if any(codes): raise RuntimeError(f'Queue worker failures: {codes}')


def collect_one(c,p,lane,deadline):
    for job in p['jobs']:
        folder=cache(c)/'physical'/job['key']; folder.mkdir(parents=True,exist_ok=True)
        if (folder/'complete.json').exists() or (folder/'failed.json').exists(): continue
        with (folder/'worker.lock').open('a') as lock:
            try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: continue
            if (folder/'complete.json').exists(): continue
            for attempt in range(c['runtime']['max_attempts']):
                record=dict(stage='physical_preparation',job=job['key'],lane=lane,started=time.time(),
                            allocation=os.environ.get('SLURM_JOB_ID'),attempt=attempt+1)
                write_json(root(c)/'technical'/f'lane-{lane}.json',dict(record,state='running'))
                try:
                    measure_job(c,p,job,deadline)
                except Exception:
                    record.update(state='failed',traceback=traceback.format_exc(),finished=time.time())
                    write_json(folder/f'attempt-{time.time_ns()}.json',record)
                    print(record['traceback'],flush=True)
                else:
                    print('prepared',job['key'],'reused' if job['reuse'] else 'new',flush=True)
                    return True
            write_json(folder/'failed.json',record)
            return True
    return False


def subprocess_stage(c,bundle,lane,deadline,stage,name,folder,marker):
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/marker).exists() or (folder/(stage+'-failed.json')).exists(): return None
    with (folder/'worker.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return None
        if (folder/marker).exists(): return None
        for attempt in range(c['runtime']['max_attempts']):
            command=[sys.executable,'-u','-m',MODULE,stage,'--config',str(bundle.config_path),
                     '--name',name,'--deadline',str(deadline)]
            record=dict(stage=stage,name=name,lane=lane,allocation=os.environ.get('SLURM_JOB_ID'),
                        attempt=attempt+1,started=time.time(),state='running')
            log=folder/f'{stage}-{time.time_ns()}.log'
            torch.cuda.empty_cache()
            with log.open('a') as stream:
                process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
                record.update(pid=process.pid,log=str(log),command=command)
                write_json(root(c)/'technical'/f'lane-{lane}.json',record)
                code=process.wait()
            record.update(exit_code=code,finished=time.time(),state='paused' if code==75 else ('complete' if code==0 else 'failed'))
            write_json(folder/f'{stage}-attempt-{time.time_ns()}.json',record)
            if code==75: return 'paused'
            if code==0 and (folder/marker).exists(): return 'done'
        write_json(folder/(stage+'-failed.json'),record)
        return 'failed'


def worker(c,lane,deadline,bundle):
    p=read(root(c)/'technical/plan.json'); paused=False
    while time.time()<deadline-900:
        # One lane acquires teachers while the other starts fits on ready banks.
        if lane==1 and collect_one(c,p,lane,deadline): continue
        progressed=False
        for arm in c['arms']:
            folder=root(c)/'analyses'/arm['name']/'technical'
            if not ready(c,p,arm['bank']): continue
            for stage,marker in [('train','complete.json'),('evaluate','evaluation-complete.json')]:
                if stage=='evaluate' and (not (folder/'complete.json').exists() or not ready(c,p,arm['bank'],test=True)): continue
                result=subprocess_stage(c,bundle,lane,deadline,stage,arm['name'],folder,marker)
                if result=='paused': paused=True; break
                if result is not None: progressed=True; break
            if progressed or paused: break
        if paused: break
        if progressed: continue
        if collect_one(c,p,lane,deadline): continue
        for bank in p['banks']:
            if not ready(c,p,bank) or not ready(c,p,bank,test=True): continue
            folder=root(c)/'analyses'/('readouts-'+bank)/'technical'
            result=subprocess_stage(c,bundle,lane,deadline,'readouts',bank,folder,'complete.json')
            if result is not None: progressed=True; break
        if progressed: continue
        pending=[a for a in c['arms'] if not (root(c)/'analyses'/a['name']/'technical/evaluation-complete.json').exists()]
        with (root(c)/'technical/comparison.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            from .rate_assessment import compare
            compare(c)
        write_json(root(c)/'technical'/f'queue-status-{lane}.json',dict(pending=[a['name'] for a in pending],updated=time.time()))
        if not pending: break
        busy=False
        folders=[cache(c)/'physical'/j['key'] for j in p['jobs']]
        folders += [root(c)/'analyses'/a['name']/'technical' for a in c['arms']]
        folders += [root(c)/'analyses'/('readouts-'+bank)/'technical' for bank in p['banks']]
        for folder in folders:
            if not (folder/'worker.lock').exists(): continue
            with (folder/'worker.lock').open('a') as lock:
                try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError: busy=True; break
        if not busy:
            write_json(root(c)/'technical'/f'blocked-{lane}.json',dict(
                pending=[a['name'] for a in pending],reason='All remaining tasks have failure receipts or missing failed prerequisites'))
            break
        # Other lane owns the remaining work. No long blocking waits.
        time.sleep(15)
    write_json(root(c)/'technical'/f'worker-exit-{os.environ.get("SLURM_JOB_ID")}-{lane}.json',
               dict(paused=paused or time.time()>=deadline-900,finished=time.time()))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=['prepare','launch','team','worker','train','evaluate','readouts','compare'])
    parser.add_argument('--config',required=True); parser.add_argument('--name'); parser.add_argument('--lane',type=int,default=0)
    parser.add_argument('--deadline',type=float,default=float('inf'))
    args=parser.parse_args(); c=read(args.config); torch.set_num_threads(1)
    bundle=ExecutionBundle(Path(args.config).resolve().parent)
    if args.stage=='prepare': print(prepare(c)['identity'])
    elif args.stage=='launch': launch(c)
    elif args.stage=='team': team(c,bundle,detach=False)
    elif args.stage=='worker': worker(c,args.lane,args.deadline,bundle)
    elif args.stage=='readouts':
        from .rate_assessment import readouts
        readouts(c,args.name)
    elif args.stage=='compare':
        from .rate_assessment import compare
        compare(c)
    else:
        arm=next(a for a in c['arms'] if a['name']==args.name)
        if args.stage=='train':
            from .rate_learning import fit
            done=fit(c,arm,args.deadline)
        else:
            from .rate_assessment import run
            done=run(c,arm,args.deadline)
        if not done: sys.exit(75)


if __name__=='__main__': main()
