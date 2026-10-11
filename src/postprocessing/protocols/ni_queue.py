"""Bounded chunk scheduling, admitting only completed immutable Ni trajectories."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from functools import lru_cache
import fcntl
import getpass
import multiprocessing
import os
from pathlib import Path
import subprocess
import time

from src.experiment_runner.artifacts import file_hash as sha, json_digest as digest, write_json
from src.experiment_runner.metric_docs import check_metric_docs
from src.project_runtime.paths import resolve_path
from src.postprocessing import extraction, lineage
from src.postprocessing.io import read
from . import ni


@lru_cache(maxsize=16)
def payload(path, checksum):
    if sha(path)!=checksum:raise ValueError(f'Changed Ni source plan: {path}')
    return read(path)


def execute(work):
    path,checksum,kind,task=work;p=payload(path,checksum)
    return (extraction.extract(p['parent'],task) if kind=='extract'
            else lineage.run(p['tracking'],p['parent'],task))


def source_tasks(p,s):
    chunk=p['parent']['config']['chunk_frames']
    return [dict(source=s['id'],start=start,stop=min(start+chunk,s['frame_count']))
            for start in range(0,s['frame_count'],chunk)]


def active_jobs(jobs):
    if not jobs:return False
    ids=subprocess.check_output(['squeue','--user',getpass.getuser(),'-h','-o','%A'],text=True).split()
    return bool(set(jobs)&set(ids))


def run(c,*,workers,plan_sha256=None,wait_for_jobs=()):
    if workers<1:raise ValueError('Ni workers must be positive')
    if any(not j.isdigit() for j in wait_for_jobs):raise ValueError('MD job IDs must be numeric')
    root=resolve_path(c['analysis']);tech=root/'technical';tech.mkdir(parents=True,exist_ok=True)
    with (tech/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return schedule(c,root,workers,plan_sha256,wait_for_jobs)


def schedule(c,root,workers,plan_sha256,wait_for_jobs):
    campaign_path=resolve_path(c['launch'])/'plan.json';campaign=read(campaign_path)
    if campaign['config']!=c or digest({k:v for k,v in campaign.items() if k!='identity'})!=campaign['identity']:
        raise ValueError('Changed Ni campaign plan')
    records=campaign['records'];tech=root/'technical';archive=resolve_path(c['archive'])
    job=os.environ.get('SLURM_JOB_ID',str(os.getpid()));execution=tech/'executions'/job
    implementation=check_metric_docs(family='ni_births')['ni_births']['files']
    fixed=ni.resume_plan(c,plan_sha256) if plan_sha256 else None
    if fixed is None and (tech/'plans.json').exists():
        raise ValueError('An existing Ni population requires --plan-sha256 to resume')
    inputs={};pending={};outstanding={};tracking={};ready=deque();completed=set();futures={}
    initial_chunks=0;finished_chunks=0;reused_chunks=0

    def admit(s,path,checksum):
        nonlocal initial_chunks
        sid=s['id'];p=payload(str(path),checksum);inputs[sid]=(s,str(path),checksum)
        tasks=source_tasks(p,s);pending[sid]=deque(tasks);outstanding[sid]=len(tasks)
        tracking[sid]=len(p['parent']['config']['detectors'])
        existing=sum((extraction.source_root(p['parent'],s)/f'clusters-{t["start"]:06d}-{t["stop"]:06d}.json').exists() for t in tasks)
        initial_chunks+=existing
        write_json(execution/(sid+'.json'),dict(source=sid,plan=str(path),plan_sha256=checksum,
            parent_identity=p['parent']['identity'],tracking_identity=p['tracking']['identity'],
            chunks=len(tasks),preexisting_chunks=existing,implementation=implementation))

    if fixed is not None:
        for s in fixed['parent']['sources']:
            if 'source_plans' in fixed:
                ref=fixed['source_plans'][s['id']];path=Path(ref['path']);checksum=ref['sha256']
            else:path=tech/'plans.json';checksum=plan_sha256
            admit(s,path,checksum)

    def discover():
        if fixed is not None:return
        for r in records:
            sid=r['id']
            if sid in inputs or not (archive/sid/'complete.json').exists():continue
            path=tech/'source-plans'/f'{sid}.json'
            if path.exists():
                p=ni.resume_payload(c,read(path));s=p['parent']['sources'][0]
                if s['id']!=sid:raise ValueError('Ni source-plan identity differs')
            else:
                s=ni.source(c,r)
                row=dict(source=sid,state='complete',frames=s['frame_count'],temperature_K=r['temperature_K'])
                p=ni.plans(c,[s],[row]);write_json(path,p)
            admit(s,path,sha(path))

    def report(state):
        write_json(tech/'progress.json',dict(state=state,job=job,workers=workers,
            completed_sources=sorted(completed),available_sources=len(inputs),total_sources=len(records),
            finished_chunks=finished_chunks,reused_chunks=reused_chunks,initial_cached_chunks=initial_chunks,
            pending_chunks=sum(len(q) for q in pending.values()),in_flight=len(futures),updated_unix=time.time()))

    write_json(execution/'execution.json',dict(resumed_plan_sha256=plan_sha256,
        campaign_plan_sha256=sha(campaign_path),workers=workers,wait_for_jobs=list(wait_for_jobs),
        implementation=implementation,scheduling='bounded round-robin chunks; tracking after source extraction'))
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        last_discovery=0;sealed=fixed is not None
        while True:
            if not sealed and time.monotonic()-last_discovery>=30:
                discover();last_discovery=time.monotonic()
                sealed=len(inputs)==len(records) or (bool(wait_for_jobs) and not active_jobs(wait_for_jobs))
                if sealed:discover() # final archive check after producers end
                if not sealed and not wait_for_jobs:
                    raise ValueError('Incomplete Ni campaign: supply its --wait-for-jobs to stream completed sources')
            # Keep only 2*workers tasks in flight; rotate sources on every chunk.
            while len(futures)<2*workers:
                if ready:work=ready.popleft()
                else:
                    sid=next((sid for sid,q in pending.items() if q),None)
                    if sid is None:break
                    q=pending.pop(sid);task=q.popleft();pending[sid]=q
                    _,path,checksum=inputs[sid];work=(path,checksum,'extract',task)
                p=payload(work[0],work[1]);task=work[3]
                if work[2]=='extract':
                    source=next(s for s in p['parent']['sources'] if s['id']==task['source'])
                    saved=extraction.source_root(p['parent'],source)/f'clusters-{task["start"]:06d}-{task["stop"]:06d}.json'
                    existed=saved.exists()
                else:existed=False
                futures[pool.submit(execute,work)]=(work,existed)
            if not futures:
                if sealed:break
                report('waiting_for_completed_simulations');time.sleep(30);continue
            done,_=wait(futures,timeout=5,return_when=FIRST_COMPLETED)
            for future in done:
                work,existed=futures.pop(future);future.result();sid=work[3]['source']
                if work[2]=='extract':
                    finished_chunks+=1;reused_chunks+=int(existed);outstanding[sid]-=1
                    if outstanding[sid]==0:
                        p=payload(work[0],work[1])
                        for detector in p['parent']['config']['detectors']:
                            ready.append((work[0],work[1],'track',dict(source=sid,cadence='010',detector=detector)))
                else:
                    tracking[sid]-=1
                    if tracking[sid]==0:completed.add(sid)
            report('running')
    if completed!=set(inputs):raise ValueError('Ni scheduler ended with unfinished tracking')
    if fixed is None:
        coverage=[]
        for r in records:
            sid=r['id'];done=sid in inputs;outcome=archive/sid/'outcome.json'
            state='complete' if done else (read(outcome)['state'] if outcome.exists() else 'not_started')
            coverage.append(dict(source=sid,state=state,frames=inputs[sid][0]['frame_count'] if done else 0,temperature_K=r['temperature_K']))
        fixed=ni.plans(c,[inputs[r['id']][0] for r in records if r['id'] in inputs],coverage)
        fixed['source_plans']={sid:dict(path=path,sha256=checksum) for sid,(_,path,checksum) in inputs.items()}
        write_json(tech/'plans.json',fixed)
    ni.collect(c,fixed);report('complete' if len(inputs)==len(records) else 'partial_campaign_analysed')
