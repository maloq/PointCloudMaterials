"""Detached, dependency-aware queue; immutable configs and complete checkpoint resumes."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import save_json,file_hash,digest
from src.training_methods.shared_pretraining.queue import deadline_for_job,snapshot
from src.research.relaxed_encoder.queue import claim
from .specs import encoder_specs,path_specs,choose_encoder,choose_paths


def freeze(c):
    root=resolve_path(c['output'])/'technical';root.mkdir(parents=True,exist_ok=True)
    p=root/'plan.json'
    if p.exists():
        plan=json.loads(p.read_text())
        if plan['config']!=c:raise ValueError('Night queue changed')
        return plan
    from src.research.crystallization_paths.queue import freeze_plan
    paths=freeze_plan(c['path']);save_json(root/'path-plan.json',paths)
    tasks=[dict(kind='encoder',name=s['name'],spec=s) for s in encoder_specs(c)]
    tasks+=[dict(kind='path',name=s['name'],spec=s) for s in path_specs(c)]
    tasks+=[dict(kind='encoder_long',name='selected-long')]
    tasks+=[dict(kind='probe',name=n,requires=('encoder_long' if n=='selected-long' else 'encoder')) for n in ['selected-long',*[s['name'] for s in encoder_specs(c)]]]
    tasks+=[dict(kind='path_long',name=m+'-selected-E36',method=m) for m in ('direct','ar_mse','mixture','diffusion')]
    tasks+=[dict(kind='bridge',name=m+'-new-encoder-E18',method=m) for m in ('direct','ar_mse')]
    plan=dict(config=c,tasks=tasks,identity=digest(c),encoder_checkpoint_sha256=file_hash(resolve_path(c['encoder']['warm_checkpoint'])),
        parent_recipe_sha256=file_hash(resolve_path(c['path']['reference_path_output'])/'technical/promotions.json'),path_identity=paths['identity'])
    save_json(p,plan);return plan


def dependencies(task,tasks):
    if task['kind']=='encoder_long':return [('encoder',t['name']) for t in tasks if t['kind']=='encoder']
    if task['kind']=='probe':return [(task['requires'],task['name'])]
    if task['kind']=='path_long':return [('path',t['name']) for t in tasks if t['kind']=='path']
    if task['kind']=='bridge':return [('probe','selected-long')]
    return []


def ready(task,root,tasks):
    def done(kind,name):
        p=root/'tasks'/f'{kind}--{name}.json'
        return p.exists() and json.loads(p.read_text())['state']=='complete'
    return all(done(kind,name) for kind,name in dependencies(task,tasks))


def short_probes(c,name):
    from src.research.crystallization_information.runtime import standardize,fit
    from src.research.crystallization_information.data import BLOCKS
    root=resolve_path(c['encoder']['output']);folder=root/'technical/crystallization'/name
    diagnostic=resolve_path(c['short_diagnostic'])/'technical/data';pop=dict(np.load(diagnostic/'population.npz'))
    export=np.load(root/'technical/crystallization/population.npz');np.testing.assert_array_equal(pop['rows'],export['rows'])
    z=np.empty((len(pop['source']),128),np.float32)
    for sid in np.unique(pop['source']):z[pop['source']==sid]=np.load(folder/'features'/f'{sid}.npy')
    train=np.flatnonzero(pop['role']=='train');z,zm,zs=standardize(z,train,pop['source'])
    observed,om,oscale=standardize(np.load(diagnostic/'observed.npy'),train,pop['source'])
    np.savez(folder/'short-normalizers.npz',z_mean=zm,z_std=zs,observed_mean=om,observed_std=oscale)
    for variant,readout in [('z','linear'),('z','mlp'),('z+outer_geometry','mlp')]:
        x=np.zeros((len(z),562),np.float32);x[:,:128]=z;x[:,-7:]=pop['condition']
        if variant!='z':columns=BLOCKS['outer_geometry'];x[:,128+columns]=observed[:,columns]
        fit(c['short_probe'],dict(encoder=name,variant=variant,readout=readout,task='hazard'),pop,x,pop['event'],resolve_path(c['output'])/'short_readouts')


def execute(c,task,root):
    deadline=deadline_for_job();kind=task['kind']
    if kind in ('encoder','encoder_long'):
        from src.training_methods.neighborhood_jepa.regularization.runtime import run
        if kind=='encoder':spec=task['spec']
        else:
            selected=choose_encoder(resolve_path(c['encoder']['output']),encoder_specs(c));save_json(root/'encoder-selection.json',selected)
            spec=dict(selected['spec'],name=task['name'],initialization='continuation',updates=c['encoder_long_updates'],
                checkpoint=str(resolve_path(c['encoder']['output'])/'technical/runs'/selected['name']/'best.pt'),encoder_lr=.000025,head_lr=.00025)
        if not run(c['encoder'],spec,deadline):raise SystemExit(75)
    elif kind=='probe':
        from src.training_methods.neighborhood_jepa.v2.probe import run
        item=dict(name=task['name'],kind='regularization',checkpoint=str(resolve_path(c['encoder']['output'])/'technical/runs'/task['name']/'best.pt'),producer_code=str(Path.cwd()))
        if not run(c['encoder'],item,deadline):raise SystemExit(75)
        short_probes(c,task['name'])
    else:
        from src.research.crystallization_transfer.runtime import setup
        from src.research.crystallization_paths.runtime import fit
        from .context import ContextPaths
        setup();plan=json.loads((root/'path-plan.json').read_text())
        if kind=='path':spec=task['spec']
        elif kind=='path_long':
            with (root/'promotion.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                if not (root/'path-promotions.json').exists():save_json(root/'path-promotions.json',choose_paths(resolve_path(c['path']['output']),path_specs(c)))
                spec=next(s for s in json.loads((root/'path-promotions.json').read_text()) if s['name']==task['name'])
        else:
            spec=copy.deepcopy(next(s for s in path_specs(c) if s['method']==task['method'] and s['information_context']=='both'))
            spec.update(name=task['name'],information_context='both_new')
            selected=resolve_path(c['encoder']['output'])/'technical/crystallization/selected-long'
            record=json.loads((selected/'record.json').read_text());spec['extra_encoder_sha256']=record['checkpoint_sha256']
        data=ContextPaths(plan,spec)
        if kind=='bridge':data.load_encoder(selected,resolve_path(c['encoder']['output'])/'technical/crystallization/population.npz')
        save_json(resolve_path(c['path']['output'])/'technical/runs'/spec['name']/'spec.json',spec)
        if not fit(plan,spec,data,deadline):raise SystemExit(75)


def worker(config_path,lane,role):
    c=json.loads(Path(config_path).read_text());plan=freeze(c);root=resolve_path(c['output'])/'technical';deadline=deadline_for_job()
    allowed={'encoder','encoder_long','probe'} if role=='encoder' else {'path','path_long','bridge'}
    tasks=[t for t in plan['tasks'] if t['kind'] in allowed]
    while time.time()<deadline-1800:
        unfinished=False;worked=False
        for i,task in enumerate(tasks):
            status=root/'tasks'/f'{task["kind"]}--{task["name"]}.json'
            if status.exists() and json.loads(status.read_text())['state'] in ('complete','failed','blocked'):continue
            unfinished=True
            failed=[]
            for kind,name in dependencies(task,plan['tasks']):
                p=root/'tasks'/f'{kind}--{name}.json'
                if p.exists() and json.loads(p.read_text())['state'] in ('failed','blocked'):failed.append(str(p))
            if failed:save_json(status,dict(state='blocked',dependencies=failed));continue
            if not ready(task,root,plan['tasks']):continue
            with claim(root/'locks'/status.stem) as acquired:
                if not acquired:continue
                if status.exists() and json.loads(status.read_text())['state']=='complete':continue
                worked=True;save_json(status,dict(state='running',lane=lane,allocation=os.environ['SLURM_JOB_ID']))
                save_json(root/f'lane-{lane}.json',dict(state='running',task=task,pid=os.getpid()))
                log=root/'logs'/f'{status.stem}.log';log.parent.mkdir(exist_ok=True)
                index=plan['tasks'].index(task)
                with log.open('a') as out:r=subprocess.run([sys.executable,'-u','-m',__package__+'.queue','execute','--config',str(config_path),'--index',str(index)],stdout=out,stderr=subprocess.STDOUT)
                save_json(status,dict(state='complete' if r.returncode==0 else 'checkpointed' if r.returncode==75 else 'failed',exit_code=r.returncode,log=str(log),lane=lane))
                if r.returncode==75:return
                from .report import report
                report(c)
        if not unfinished:save_json(root/f'lane-{lane}.json',dict(state='complete'));return
        if not worked:save_json(root/f'lane-{lane}.json',dict(state='waiting_for_dependencies'))
        time.sleep(20)
    save_json(root/f'lane-{lane}.json',dict(state='allocation_deadline'))


def submit(config_path):
    import shlex
    c=json.loads(Path(config_path).read_text());plan=freeze(c);root=resolve_path(c['output'])/'technical'
    if (root/'launches.json').exists():raise FileExistsError('Overnight queue already submitted')
    code=snapshot(root);records=[]
    env=dict(os.environ,PCM_PROJECT_ROOT=str(code),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
    command=[sys.executable,'-u','-m',__package__+'.queue','worker','--config',str(code/config_path)]
    job=c['encoder_allocation'];lane='h100-encoder'
    cmd=['srun',f'--jobid={job}','--overlap','--exact','--nodes=1','--ntasks=1','--cpus-per-task=12','--gres=gpu:1',*command,'--lane',lane,'--role','encoder']
    with (root/f'{lane}.log').open('a') as out:p=subprocess.Popen(cmd,cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    records.append(dict(lane=lane,allocation=job,pid=p.pid));save_json(root/'launches.json',records)
    for i in range(c['path_workers']):
        lane=f'path-{i}';file=root/f'{lane}.sbatch'
        text=f'#!/bin/bash\n#SBATCH --job-name=context-night-{i}\n#SBATCH --partition=RTX6000PRO,H100,L40S\n#SBATCH --gres=gpu:1\n#SBATCH --cpus-per-task=8\n#SBATCH --mem=64G\n#SBATCH --time={c["hours"]}:00:00\n#SBATCH --output={root}/{lane}-%j.log\n'
        text+='set -euo pipefail\ncd '+shlex.quote(str(code))+'\n'+''.join('export '+k+'='+shlex.quote(env[k])+'\n' for k in ('PCM_PROJECT_ROOT','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','PYTORCH_ALLOC_CONF'))
        text+='exec '+shlex.join([*command,'--lane',lane,'--role','path'])+'\n';file.write_text(text)
        job=subprocess.check_output(['sbatch','--parsable',str(file)],text=True).strip();records.append(dict(lane=lane,job=job));save_json(root/'launches.json',records)
    print(json.dumps(records),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','submit','worker','execute','report']);p.add_argument('--config',required=True);p.add_argument('--lane');p.add_argument('--role',choices=['encoder','path']);p.add_argument('--index',type=int);a=p.parse_args();c=json.loads(Path(a.config).read_text())
    if a.stage=='submit':submit(a.config)
    elif a.stage=='worker':worker(Path(a.config).resolve(),a.lane,a.role)
    elif a.stage=='execute':
        plan=freeze(c);execute(c,plan['tasks'][a.index],resolve_path(c['output'])/'technical')
    elif a.stage=='report':
        from .report import report
        report(c)
    else:freeze(c)

if __name__=='__main__':
    code=0
    try:main()
    except SystemExit as e:code=e.code
    except BaseException:traceback.print_exc();code=1
    sys.stdout.flush();sys.stderr.flush();os._exit(code)
