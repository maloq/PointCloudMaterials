"""Frozen CPU preparation and detached GPU lanes with Slurm continuation."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from src.data.fixed_cohort.protocol import sha,write_json
from src.project_runtime.paths import resolve_path
from src.experiment_runner.metric_docs import check_metric_docs
from .data import config,prepare_source,seal

def environment(repo):
    return dict(PCM_PROJECT_ROOT=str(repo),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OVITO_THREAD_COUNT='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',PYTORCH_ALLOC_CONF='expandable_segments:True',TORCHINDUCTOR_COMPILE_THREADS='4')

def launch(path):
    c=config(path);tech=resolve_path(c['output'])/'technical';tech.mkdir(parents=True,exist_ok=True)
    if (tech/'launch.json').exists():raise ValueError('Already submitted; use recorded frozen commands')
    checked=config(tech/'preflight.json')
    if checked['config_sha256']!=sha(Path(path)) or not checked['finite_full_batch']:raise ValueError('Missing exact full-batch preflight')
    check_metric_docs(family='liquid_predictability')
    repo=Path(__file__).resolve().parents[3];code=tech/'code'
    shutil.copytree(repo/'src',code/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(repo/'docs/metrics',code/'docs/metrics');write_json(code/'config.json',c)
    env=environment(repo);allocation=os.environ['SLURM_JOB_ID'];receipt=dict(allocation=allocation,code=str(code),submitted_at=time.time(),jobs={},lanes={})
    prefix=[sys.executable,'-u','-m','src.research.liquid_predictability.queue'];cfg=['--config',str(code/'config.json')]
    def submit(name,stage,options,dependency=None):
        script=tech/f'{name}.sbatch';cmd=prefix+[stage]+cfg
        script.write_text('\n'.join(['#!/bin/bash',f'#SBATCH --job-name=LP-{name}','#SBATCH --nodes=1','#SBATCH --ntasks=1',
            f'#SBATCH --output={tech}/{name}-%A_%a.log',*['#SBATCH '+o for o in options],'set -euo pipefail','ulimit -n 4096',
            'cd '+shlex.quote(str(code)),'exec env '+shlex.join([f'{k}={v}' for k,v in env.items()])+' '+shlex.join(cmd),'']))
        args=['sbatch','--parsable']+(['--dependency='+dependency] if dependency else [])+[str(script)]
        job=subprocess.check_output(args,text=True).strip().split(';')[0];receipt['jobs'][name]=job;write_json(tech/'launch.json',receipt);return job
    count=math.ceil(150/c['queue']['prepare_sources_per_task'])
    prep=submit('prepare','prepare',['--partition=CPU','--cpus-per-task=2','--mem=12G','--time=04:00:00',f'--array=0-{count-1}%8'])
    sealed=submit('seal','seal',['--partition=CPU','--cpus-per-task=2','--mem=16G','--time=00:30:00'],'afterok:'+prep)
    cpu=submit('controls','cpu',['--partition=CPU','--cpus-per-task=4','--mem=24G','--time=08:00:00'],'afterok:'+sealed)
    backup=submit('resume','worker',['--partition='+c['queue']['partition'],'--gpus=1','--cpus-per-task=6','--mem=48G',
        '--time='+c['queue']['training_walltime'],'--array=0-1%2'],'afterany:'+allocation)
    for lane in (0,1):
        cmd=prefix+['worker']+cfg+['--lane',str(lane)]
        receipt['lanes'][str(lane)]=dict(command=cmd,gpu=lane,continuation_job=f'{backup}_{lane}')
    write_json(tech/'launch.json',receipt)
    for lane in (0,1):
        with (tech/f'lane-{lane}.log').open('ab') as stream:
            p=subprocess.Popen(receipt['lanes'][str(lane)]['command'],cwd=code,env=dict(os.environ,**env,CUDA_VISIBLE_DEVICES=str(lane)),
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        receipt['lanes'][str(lane)]['pid']=p.pid
    write_json(tech/'launch.json',receipt)
    return receipt

def worker(path,lane):
    c=config(path);tech=resolve_path(c['output'])/'technical';state=tech/f'lane-{lane}.json';receipt=config(tech/'launch.json')
    with (tech/f'lane-{lane}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while not (resolve_path(c['study_cache'])/'manifest.json').exists():
                status=subprocess.check_output(['scontrol','show','job','-o',receipt['jobs']['seal']],text=True)
                if any(s in status for s in ('JobState=FAILED','JobState=CANCELLED','JobState=TIMEOUT','Reason=DependencyNeverSatisfied')):
                    raise RuntimeError('CPU preparation failed: '+status)
                write_json(state,dict(state='waiting_for_preparation',job=receipt['jobs']['seal']));time.sleep(15)
            if os.environ.get('SLURM_JOB_ID')==receipt['allocation']:
                previous=Path(c['queue']['wait_for']);old=config(previous.parent/'launch.json');pid=old['pid']
                while Path(f'/proc/{pid}').exists():
                    proc=Path(f'/proc/{pid}/stat')
                    if proc.read_text().split(') ',1)[1].split()[0]=='Z':break
                    write_json(state,dict(state='waiting_for_previous_gpu_run',pid=pid));time.sleep(15)
            for arm in [a for a in c['arms'] if a['lane']==lane]:
                done=resolve_path(c['output'])/arm['name']/'analyses/predictability-v1/technical/complete.json'
                if done.exists():continue
                write_json(state,dict(state='running',arm=arm['name']))
                subprocess.run([sys.executable,'-u','-m',__name__.replace('__main__','src.research.liquid_predictability.queue'),
                    'fit','--config',str(path),'--arm',arm['name']],check=True)
                if not done.exists():write_json(state,dict(state='checkpointed',arm=arm['name']));return
            write_json(state,dict(state='complete',finished_at=time.time()))
            continuation=receipt['lanes'][str(lane)]['continuation_job']
            if os.environ.get('SLURM_ARRAY_JOB_ID')!=continuation.split('_')[0]:subprocess.run(['scancel',continuation],check=True)
            maybe_compare(c)
        except BaseException:
            write_json(state,dict(state='failed',traceback=traceback.format_exc()));raise

def maybe_compare(c):
    root=resolve_path(c['output']);done=[root/a['name']/'analyses/predictability-v1/technical/complete.json' for a in c['arms']]
    if not all(p.exists() for p in done):return
    with (root/'technical/comparison.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not (root/'analyses/comparison-v1/technical/complete.json').exists():
            from .evaluate import compare
            compare(c)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['launch','prepare','seal','cpu','worker','fit','profiles','compare','preflight'])
    p.add_argument('--config',required=True);p.add_argument('--arm');p.add_argument('--lane',type=int);p.add_argument('--source',type=int)
    a=p.parse_args();c=config(a.config)
    if a.stage=='launch':print(json.dumps(launch(a.config),indent=2))
    elif a.stage=='prepare':
        plan=config(resolve_path(c['dataset']['root'])/'plan.json');size=c['queue']['prepare_sources_per_task']
        items=[s for s in plan['sources'] if s['id']==a.source] if a.source is not None else plan['sources'][int(os.environ['SLURM_ARRAY_TASK_ID'])*size:(int(os.environ['SLURM_ARRAY_TASK_ID'])+1)*size]
        for item in items:print(json.dumps(prepare_source(c,item['id'])),flush=True)
    elif a.stage=='seal':seal(c)
    elif a.stage=='worker':worker(a.config,a.lane if a.lane is not None else int(os.environ['SLURM_ARRAY_TASK_ID']))
    elif a.stage=='cpu':
        from .evaluate import profiles
        profiles(c)
        for arm in [v for v in c['arms'] if v['lane']=='cpu']:
            subprocess.run([sys.executable,'-u','-m','src.research.liquid_predictability.queue','fit','--config',a.config,'--arm',arm['name']],check=True)
        maybe_compare(c)
    elif a.stage in ('fit','preflight'):
        from .train import run
        print(json.dumps(run(a.config,a.arm,preflight=a.stage=='preflight')),flush=True)
    else:
        from .evaluate import profiles,compare
        (profiles if a.stage=='profiles' else compare)(c)

if __name__=='__main__':main()
