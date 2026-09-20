"""Detached allocation workers, locked fits/probes, and development-only continuations."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import save_json,file_hash
from src.training_methods.shared_pretraining.queue import snapshot,deadline_for_job
from .specs import variants


def tasks(config,code):
    result=[]
    first=['sig-mlp-no-order','sig-mlp-order','vic-mlp-order','epi-mlp-order',
           'sig-linear-order','vic-linear-order','epi-linear-order','none-mlp-order']
    ordered=sorted(variants(config),key=lambda s:first.index(s['name']) if s['name'] in first else len(first))
    for spec in ordered:result.append(dict(type='fit',name=spec['name'],spec=spec))
    for regularizer in ('sigreg','vicreg','epi'):result.append(dict(type='long',name='long-'+regularizer,regularizer=regularizer))
    for spec in variants(config):result.append(dict(type='probe',name=spec['name'],requires=spec['name']))
    for regularizer in ('sigreg','vicreg','epi'):result.append(dict(type='probe',name='long-'+regularizer,requires='long-'+regularizer))
    for name in ('geometry-only-baseline','geometry-baseline','condition-baseline'):result.append(dict(type='baseline',name=name))
    return result


def directory(root,task):return root/('runs' if task['type'] in ('fit','long') else 'crystallization')/task['name']


def ready(config,task,all_tasks,root):
    if task['type'] in ('probe','long'):
        required=[task['requires']] if task['type']=='probe' else [t['name'] for t in all_tasks if t['type']=='fit' and t['spec']['regularizer']==task['regularizer']]
        for name in required:
            p=root/'runs'/name/'status.json'
            if not p.exists() or json.loads(p.read_text())['state']!='complete':return False
    return True


def failed_dependencies(task,all_tasks,root):
    required=([task['requires']] if task['type']=='probe' else
              [t['name'] for t in all_tasks if t['type']=='fit' and t['spec']['regularizer']==task['regularizer']] if task['type']=='long' else [])
    failed=[]
    for name in required:
        path=root/'runs'/name/'status.json'
        if path.exists() and json.loads(path.read_text())['state'] in ('failed','blocked'):
            failed.append(name)
    return failed


def select_long(config,task,all_tasks,root):
    path=directory(root,task)/'selection.json'
    if path.exists():return json.loads(path.read_text())['spec']
    candidates=[]
    for item in all_tasks:
        if item['type']!='fit':continue
        s=item['spec']
        if s['regularizer']!=task['regularizer'] or s['order_weight']==0 or s['initialization']!='warm':continue
        m=json.loads((root/'runs'/s['name']/'metrics.json').read_text())
        score=m['physical']+.25*m['tda']+.25*m['order']+.25*m['future_physical']
        candidates.append((score,s))
    score,chosen=min(candidates,key=lambda row:row[0])
    spec=dict(chosen,name=task['name'],initialization='continuation',updates=config['long_updates'],
        checkpoint=str((root/'runs'/chosen['name']/'best.pt').resolve()),encoder_lr=.00005,head_lr=.0005)
    save_json(path,dict(spec=spec,chosen=chosen['name'],development_score=score,
                       criterion='development physical + .25 TDA + .25 order + .25 future physical; no crystallization test metrics'))
    return spec


def execute(config_path,index):
    config=json.loads(Path(config_path).read_text());root=resolve_path(config['output']).resolve()/'technical'
    all_tasks=json.loads((root/'tasks.json').read_text());task=all_tasks[index];dest=directory(root,task);dest.mkdir(parents=True,exist_ok=True)
    deadline=deadline_for_job()
    if task['type'] in ('fit','long'):
        from .runtime import run
        spec=task['spec'] if task['type']=='fit' else select_long(config,task,all_tasks,root)
        if not run(config,spec,deadline):sys.exit(75)
    else:
        from ..v2.probe import run
        item=dict(name=task['name'],kind='baseline') if task['type']=='baseline' else dict(
            name=task['name'],kind='regularization',checkpoint=str(root/'runs'/task['requires']/'best.pt'),producer_code=str(Path.cwd()))
        if not run(config,item,deadline):sys.exit(75)


def worker(config_path,lane):
    config=json.loads(Path(config_path).read_text());root=resolve_path(config['output']).resolve()/'technical'
    all_tasks=json.loads((root/'tasks.json').read_text());deadline=deadline_for_job();state=root/f'lane-{lane}.json'
    while time.time()<deadline-300:
        unfinished=False;claimed=False
        for index,task in enumerate(all_tasks):
            dest=directory(root,task);dest.mkdir(parents=True,exist_ok=True);status=dest/'status.json'
            if status.exists() and json.loads(status.read_text())['state'] in ('complete','failed','blocked'):continue
            failed=failed_dependencies(task,all_tasks,root)
            if failed:
                save_json(status,dict(state='blocked',failed_dependencies=failed));continue
            unfinished=True
            if not ready(config,task,all_tasks,root):continue
            with (dest/'worker.lock').open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if status.exists() and json.loads(status.read_text())['state'] in ('complete','failed','blocked'):continue
                save_json(state,dict(state='running',task=task['name'],host=os.uname().nodename,pid=os.getpid(),time=time.time()))
                with (dest/'execution.log').open('a') as log:
                    result=subprocess.run([sys.executable,'-u','-m',__package__+'.queue','execute','--config',config_path,'--index',str(index)],stdout=log,stderr=subprocess.STDOUT)
                if result.returncode==75:
                    save_json(state,dict(state='allocation_checkpoint',task=task['name']));return
                if result.returncode:
                    save_json(status,dict(state='failed',exit_code=result.returncode,log=str(dest/'execution.log')))
                claimed=True;break
        if not unfinished:save_json(state,dict(state='complete',time=time.time()));return
        if not claimed:time.sleep(20)
    save_json(state,dict(state='allocation_deadline'))


def coordinate(config_path,lane):
    config=json.loads(Path(config_path).read_text());root=resolve_path(config['output']).resolve()/'technical'
    devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(devices)!=2:raise ValueError(f'Expected a two-GPU allocation, got {devices}')
    children=[]
    for i,device in enumerate(devices):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=device)
        with (root/f'lane-{lane+i}.log').open('a') as log:
            children.append(subprocess.Popen([sys.executable,'-u','-m',__package__+'.queue','worker','--config',config_path,'--lane',str(lane+i)],env=env,stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in children]
    if any(codes):raise RuntimeError(f'Worker failure {codes}')


def submit(config_path):
    config=json.loads(Path(config_path).read_text());root=resolve_path(config['output']).resolve()/'technical'
    if (root/'launches.json').exists():raise FileExistsError('Campaign already submitted')
    # Complete the shared population once before concurrent read-only probes.
    from ..v2.probe import prepare_population
    from .data import Data
    for spec in variants(config):
        if spec['regularizer']=='epi':
            Data(config,spec)
            break
    if not resolve_path(config['warm_checkpoint']).is_file():raise FileNotFoundError(config['warm_checkpoint'])
    prepare_population(config)
    code=snapshot(root);frozen=str(code/config_path);save_json(root/'tasks.json',tasks(config,code))
    env=dict(os.environ,PCM_PROJECT_ROOT=str(code),TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',OPENBLAS_NUM_THREADS='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTORCH_ALLOC_CONF='expandable_segments:True',TORCHINDUCTOR_COMPILE_THREADS='4')
    launches=[]
    for job,lane,cpus in [(1000616,0,20),(1000818,2,16)]:
        cmd=['srun',f'--jobid={job}','--overlap','--exact','--nodes=1','--ntasks=1',f'--cpus-per-task={cpus}','--gres=gpu:2',
            sys.executable,'-u','-m',__package__+'.queue','coordinate','--config',frozen,'--lane',str(lane)]
        with (root/f'coordinator-{lane}.log').open('a') as log:
            p=subprocess.Popen(cmd,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        launches.append(dict(allocation=job,lane_start=lane,pid=p.pid,command=cmd))
        save_json(root/'launches.json',launches)
    from src.experiment_runner.slurm import submit_sbatch
    for lane in range(4,8):
        command=[sys.executable,'-u','-m',__package__+'.queue','worker','--config',frozen,'--lane',str(lane)]
        exports='\n'.join('export '+k+'='+shlex.quote(env[k]) for k in ('PCM_PROJECT_ROOT','TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTORCH_ALLOC_CONF','TORCHINDUCTOR_COMPILE_THREADS'))
        script=f'''#!/bin/bash
#SBATCH --job-name=nj-regularization
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output={root}/coordinator-{lane}.log
set -euo pipefail
cd {shlex.quote(str(code))}
{exports}
{shlex.join(command)}
'''
        job=submit_sbatch(script,root/f'lane-{lane}.sbatch');launches.append(dict(allocation=job,lane_start=lane,submitted_new=True))
        save_json(root/'launches.json',launches)
    print(json.dumps(launches),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['submit','coordinate','worker','execute']);parser.add_argument('--config',required=True);parser.add_argument('--lane',type=int);parser.add_argument('--index',type=int)
    args=parser.parse_args()
    if args.phase=='submit':submit(args.config)
    elif args.phase=='execute':
        # Fits own a dedicated subprocess. PyTorch's shared-tensor resource
        # tracker can hang during interpreter teardown after completed loader
        # work. All checkpoints, logs and W&B are closed inside execute/run;
        # exit here so a completed fit releases its GPU and the queue advances.
        code=0
        try:execute(args.config,args.index)
        except SystemExit as error:code=error.code
        except BaseException:
            traceback.print_exc();code=1
        sys.stdout.flush();sys.stderr.flush();os._exit(code)
    else:{'coordinate':coordinate,'worker':worker}[args.phase](args.config,args.lane)
