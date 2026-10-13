"""Frozen overnight fits, independent diagnostics and pre-submitted continuation."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, allocation_deadline
from src.experiment_runner.metric_docs import check_metric_docs
from .data import read, root, write_json, sha
from .memory import prepare, FAMILY
from .workflow import environment

MODULE='src.research.measurement_sim.campaign'
REPO=Path(__file__).resolve().parents[3]


def launch(c):
    prepare(c);check_metric_docs(family=FAMILY);technical=root(c)/'technical'
    bundle=ExecutionBundle.freeze(REPO,technical/'source',c,directories=['src','configs','docs/metrics'],
        files=[(REPO/'machine.local.yaml','machine.local.yaml')])
    write_json(technical/'source-files.json',{str(p.relative_to(bundle.root)):sha(p) for p in bundle.root.rglob('*') if p.is_file()})
    receipt=dict(jobs={},parent_allocation=os.environ['SLURM_JOB_ID'],started=time.time(),
        scope='15 scientific fits, local response/history readouts, two-worker continuation; no new MD')
    queue=SlurmQueue(technical,bundle,MODULE,environment(bundle),technical/'submission.json',receipt,'ni-meas2')
    with queue.submission():
        queue.submit('assays',['--cpus-per-task=4','--mem=32G','--time=08:00:00'],command_stage='assays')
        queue.submit('continuation',['--gres=gpu:2','--cpus-per-task=4','--mem=64G',
            f'--time={c["runtime"]["continuation_hours"]:02d}:00:00'],
            dependency='afterany:'+os.environ['SLURM_JOB_ID'],partition=c['runtime']['partitions'],command_stage='team')
    team(c,bundle,detach=True)
    print('Submitted',receipt,flush=True)


def team(c,bundle,*,detach):
    technical=root(c)/'technical'; deadline=min(time.time()+c['runtime']['hours']*3600,allocation_deadline(reserve_seconds=180))
    visible=os.environ.get('CUDA_VISIBLE_DEVICES');devices=visible.split(',') if visible else [str(i) for i in range(torch.cuda.device_count())]
    if len(devices)<2:raise RuntimeError('Two allocated GPUs required')
    receipt=dict(allocation=os.environ.get('SLURM_JOB_ID'),started=time.time(),deadline=deadline,bundle=str(bundle.root),workers=[]);processes=[]
    for lane in range(2):
        command=[sys.executable,'-u','-m',MODULE,'worker','--config',str(bundle.config_path),'--lane',str(lane),'--deadline',str(deadline)]
        env=dict(os.environ,**environment(bundle),CUDA_VISIBLE_DEVICES=devices[lane]);log=technical/f'worker-{receipt["allocation"]}-{lane}.log'
        with log.open('a') as stream:process=subprocess.Popen(command,cwd=bundle.root,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=detach)
        processes.append(process);receipt['workers'].append(dict(pid=process.pid,lane=lane,device=devices[lane],command=command,log=str(log)))
    write_json(technical/f'launch-{receipt["allocation"]}.json',receipt);print(receipt,flush=True)
    if not detach:
        codes=[p.wait() for p in processes]
        if any(codes):raise RuntimeError(f'Worker failures: {codes}')


def compare(c):
    with (root(c)/'technical/comparison.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        from .assessment import compare as summarize
        summarize(c)


def worker(c,lane,deadline,bundle):
    technical=root(c)/'technical';paused=False
    for arm in c['arms']:
        folder=root(c)/'analyses'/arm['name']/'technical';folder.mkdir(parents=True,exist_ok=True)
        with (folder/'worker.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            for stage,marker in [('train','complete.json'),('evaluate','evaluation-complete.json')]:
                if (folder/marker).exists():continue
                if time.time()>deadline-1200:paused=True;break
                if stage=='evaluate' and not (folder/'complete.json').exists():break
                success=False
                for attempt in range(1,c['runtime']['max_attempts']+1):
                    record=dict(lane=lane,allocation=os.environ.get('SLURM_JOB_ID'),arm=arm['name'],stage=stage,attempt=attempt,state='running',started=time.time())
                    command=[sys.executable,'-u','-m',MODULE,stage,'--config',str(bundle.config_path),'--arm',arm['name'],'--deadline',str(deadline)]
                    log=folder/f'{stage}-{record["allocation"]}-{attempt}.log'
                    with log.open('a') as stream:
                        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
                        record.update(pid=process.pid,command=command,log=str(log));write_json(technical/f'lane-{lane}.json',record)
                        code=process.wait()
                    record.update(exit_code=code,finished=time.time(),state='complete' if code==0 else ('paused' if code==75 else 'failed'))
                    write_json(folder/f'{stage}-attempt-{record["allocation"]}-{attempt}.json',record);write_json(technical/f'lane-{lane}.json',record)
                    if code==75:paused=True;break
                    if code==0 and (folder/marker).exists():success=True;break
                    print('Stage failure; same-setting retry then continue independent fits:',record,flush=True)
                if paused or not success:break
            if paused:break
        compare(c)
    states={a['name']:{'trained':(root(c)/'analyses'/a['name']/'technical/complete.json').exists(),
        'evaluated':(root(c)/'analyses'/a['name']/'technical/evaluation-complete.json').exists()} for a in c['arms']}
    write_json(technical/f'queue-status-{lane}.json',dict(models=states,paused=paused,updated=time.time()))
    compare(c)


def assays(c,bundle):
    failures=[]
    for stage in ['response-readouts','history-readouts']:
        folder=root(c)/'analyses'/stage/'technical';folder.mkdir(parents=True,exist_ok=True)
        for attempt in range(1,c['runtime']['max_attempts']+1):
            if (folder/'complete.json').exists():break
            log=folder/f'attempt-{attempt}.log'
            with log.open('a') as stream:
                code=subprocess.call([sys.executable,'-u','-m',MODULE,stage,'--config',str(bundle.config_path)],stdout=stream,stderr=subprocess.STDOUT)
            write_json(folder/f'attempt-{attempt}.json',dict(exit_code=code,log=str(log),allocation=os.environ.get('SLURM_JOB_ID')))
        if not (folder/'complete.json').exists():failures.append(stage)
    if failures:raise RuntimeError(f'Incomplete diagnostic stages: {failures}; GPU training remains independent')


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=['launch','team','worker','train','evaluate','assays','response-readouts','history-readouts','compare'])
    parser.add_argument('--config',required=True);parser.add_argument('--arm');parser.add_argument('--lane',type=int,default=0);parser.add_argument('--deadline',type=float,default=float('inf'))
    args=parser.parse_args();c=read(args.config);torch.set_num_threads(4 if args.stage.endswith('readouts') else 1)
    bundle=ExecutionBundle(Path(args.config).resolve().parent)
    if args.stage=='launch':launch(c)
    elif args.stage=='team':team(c,bundle,detach=False)
    elif args.stage=='worker':worker(c,args.lane,args.deadline,bundle)
    elif args.stage=='assays':assays(c,bundle)
    elif args.stage=='compare':compare(c)
    elif args.stage.endswith('readouts'):
        from .assays import response,history
        (response if args.stage=='response-readouts' else history)(c)
    else:
        arm=next(a for a in c['arms'] if a['name']==args.arm)
        if args.stage=='train':
            from .learning import fit
            done=fit(c,arm,args.deadline)
        else:
            from .assessment import run
            done=run(c,arm,args.deadline)
        if not done:sys.exit(75)


if __name__=='__main__':main()
