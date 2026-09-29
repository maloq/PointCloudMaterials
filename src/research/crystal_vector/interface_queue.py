"""CPU interface preparation followed by detached joint encoder/context training."""
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

from src.data.fixed_cohort.protocol import sha, write_json
from src.project_runtime.paths import resolve_path
from src.experiment_runner.metric_docs import check_metric_docs
from . import interface


def launch(config_path,gpu):
    """One distributed treatment in the user's current allocation, with a backup."""
    c=json.loads(Path(config_path).read_text());tech=resolve_path(c['output'])/'technical'
    if len(c['variants'])!=1:raise ValueError('Current-allocation launch requires exactly one scientific treatment')
    if (tech/'launch.json').exists():raise ValueError('Already launched')
    checked=json.loads((tech/'preflight.json').read_text())
    if checked['config_sha256']!=sha(Path(config_path)) or not checked['finite_gradients'] or checked['batch']!=c['batch_size']:
        raise ValueError('Local full-batch check must match this configuration')
    if 'prepared_data' in c:
        dataset=resolve_path(c['dataset']['root'])
    else:
        from . import expand
        dataset=expand.make_plan(config_path)
    plan=json.loads((dataset/'plan.json').read_text())
    if 'prepared_data' in c and plan['identity']!=c['prepared_data']['identity']:
        raise ValueError('Prepared cohort identity differs from the declared reused data')
    check_metric_docs(family='crystal_liquid_distance' if c.get('observation_filter')=='liquid_no_visible_crystal' else 'crystal_interface_unseen')
    allocation=os.environ['SLURM_JOB_ID'];repo=Path(__file__).resolve().parents[3];code=tech/'code'
    shutil.copytree(repo/'src',code/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(repo/'docs/metrics',code/'docs/metrics');write_json(code/'config.json',c)
    env=dict(PCM_PROJECT_ROOT=str(repo),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OVITO_THREAD_COUNT='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',PYTORCH_ALLOC_CONF='expandable_segments:True',
        TORCHINDUCTOR_COMPILE_THREADS='4')
    command=[sys.executable,'-u','-m',__package__+'.interface_queue','worker','--config',str(code/'config.json'),'--index','0']
    script=tech/'continuation.sbatch'
    gpus=c['runtime']['gpus']
    job_name='LCD-liquid-resume' if c.get('observation_filter')=='liquid_no_visible_crystal' else 'CIV-unseen-resume'
    script.write_text('\n'.join(['#!/bin/bash','#SBATCH --job-name='+job_name,
        '#SBATCH --partition='+c['queue']['partition'],'#SBATCH --nodes=1','#SBATCH --ntasks=1',
        f'#SBATCH --gpus={gpus}','#SBATCH --cpus-per-task=8','#SBATCH --mem=96G','#SBATCH --array=0',
        '#SBATCH --time='+c['queue']['training_walltime'],f'#SBATCH --output={tech}/continuation-%A_%a.log',
        'set -euo pipefail','ulimit -n 4096','cd '+shlex.quote(str(code)),
        'exec env '+shlex.join([f'{k}={v}' for k,v in env.items()])+' '+shlex.join(command),'']))
    def cpu_job(stage,options,dependency=None):
        path=tech/f'{stage}.sbatch'
        cmd=[sys.executable,'-u','-m',__package__+'.interface_queue',stage,'--config',str(code/'config.json')]
        path.write_text('\n'.join(['#!/bin/bash',f'#SBATCH --job-name=CIV-dense-{stage}',
            '#SBATCH --partition=CPU','#SBATCH --nodes=1','#SBATCH --ntasks=1','#SBATCH --cpus-per-task=2',
            '#SBATCH --mem=12G','#SBATCH --time=06:00:00',f'#SBATCH --output={tech}/{stage}-%A_%a.log',
            *['#SBATCH '+x for x in options],'set -euo pipefail','cd '+shlex.quote(str(code)),
            'exec env '+shlex.join([f'{k}={v}' for k,v in env.items()])+' '+shlex.join(cmd),'']))
        args=['sbatch','--parsable']+(['--dependency='+dependency] if dependency else [])+[str(path)]
        return subprocess.check_output(args,text=True).strip().split(';')[0]
    if 'prepared_data' in c:
        prep=c['prepared_data']['prepare_job'];seal=c['prepared_data']['seal_job']
    else:
        prep=cpu_job('prepare',['--array=0-14%8'])
        seal=cpu_job('seal',[],'afterok:'+prep)
    job=subprocess.check_output(['sbatch','--parsable','--dependency=afterany:'+allocation,str(script)],text=True).strip().split(';')[0]
    receipt=dict(submitted_at=time.time(),allocation=allocation,gpu=gpu,code=str(code),command=command,
        jobs=dict(prepare=prep,seal=seal,continuation=job),dataset_identity=plan['identity'],gpus=gpus)
    write_json(tech/'launch.json',receipt)
    with (tech/'training.log').open('ab') as log:
        p=subprocess.Popen(command,cwd=code,env=dict(os.environ,**env,CUDA_VISIBLE_DEVICES=','.join(str(gpu+i) for i in range(gpus))),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt['pid']=p.pid;write_json(tech/'launch.json',receipt)
    return receipt


def submit(config_path):
    c=json.loads(Path(config_path).read_text());root=resolve_path(c['output']);tech=root/'technical'
    tech.mkdir(parents=True,exist_ok=True)
    if (tech/'launch.json').exists():raise ValueError('Already submitted; use the frozen worker commands to resume')
    checked=json.loads((tech/'preflight.json').read_text())
    if checked['config_sha256']!=sha(Path(config_path)) or checked['batch']!=c['batch_size'] or not checked['finite_gradients']:
        raise ValueError('Missing local full-batch numerical check for this configuration')
    check_metric_docs(family='crystal_interface')
    data=interface.make_plan(config_path);plan=json.loads((data/'plan.json').read_text())
    repo=Path(__file__).resolve().parents[3];code=tech/'code'
    shutil.copytree(repo/'src',code/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(repo/'docs/metrics',code/'docs/metrics')
    write_json(code/'config.json',c)
    env=dict(PCM_PROJECT_ROOT=str(repo),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OVITO_THREAD_COUNT='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',PYTORCH_ALLOC_CONF='expandable_segments:True',
        TORCHINDUCTOR_COMPILE_THREADS='4')
    prefix='exec env '+shlex.join([f'{k}={v}' for k,v in env.items()])+' '+shlex.join([
        sys.executable,'-u','-m',__package__+'.interface_queue'])
    config_arg=' --config '+shlex.quote(str(code/'config.json'))
    def batch(name,stage,options,dependency=None):
        file=tech/f'{name}.sbatch'
        file.write_text('\n'.join(['#!/bin/bash',f'#SBATCH --job-name=CIV-{name}',
            '#SBATCH --nodes=1','#SBATCH --ntasks=1',f'#SBATCH --output={tech}/{name}-%A_%a.log',
            *['#SBATCH '+v for v in options],'set -euo pipefail','ulimit -n 4096',
            'cd '+shlex.quote(str(code)),prefix+' '+stage+config_arg,'']))
        cmd=['sbatch','--parsable']
        if dependency:cmd+=['--dependency='+dependency]
        job=subprocess.check_output(cmd+[str(file)],text=True).strip().split(';')[0]
        return job
    receipt=dict(submitted_at=time.time(),code=str(code),config_sha256=sha(code/'config.json'),
                 dataset_identity=plan['identity'],jobs={})
    # Write each accepted submission immediately, so a scheduler error never loses IDs.
    def record(name,job):
        receipt['jobs'][name]=job;write_json(tech/'launch.json',receipt);return job
    count=math.ceil(len(plan['sources'])/c['queue']['prepare_sources_per_task'])
    prep=record('prepare',batch('prepare','prepare',['--partition=CPU','--cpus-per-task=2','--mem=12G',
        '--time=04:00:00',f'--array=0-{count-1}%{c["queue"]["prepare_parallel_tasks"]}']))
    sealed=record('seal',batch('seal','seal',['--partition=CPU','--cpus-per-task=1','--mem=8G',
        '--time=00:30:00'],'afterok:'+prep))
    options=[f'--partition={c["queue"]["partition"]}','--gpus=1','--cpus-per-task=8','--mem=48G',
        '--time='+c['queue']['training_walltime'],f'--array=0-{len(c["variants"])-1}%{c["queue"]["max_concurrent_gpus"]}']
    training=record('training',batch('training','worker',options,'afterok:'+sealed))
    # A dependent task resumes only an intentional deadline checkpoint. Failed work
    # is left failed with its traceback; successful tasks cancel their backup.
    record('continuation',batch('continuation','resume',options,'afterok:'+training))
    write_json(tech/'state.json',dict(state='submitted',jobs=receipt['jobs']))
    return receipt


def worker(config_path,index,resume=False):
    c=json.loads(Path(config_path).read_text());variant=c['variants'][index]
    tech=resolve_path(c['output'])/'technical';state=tech/f'variant-{index}.json'
    with (tech/f'variant-{index}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if resume:
            previous=json.loads(state.read_text())
            if previous['state']=='complete':return
            if previous['state']!='checkpointed':raise ValueError(f'Refusing to retry failed/non-checkpointed task: {previous}')
        try:
            from .train import run,deadline
            from .evaluate import run as evaluate
            if c.get('expansion') or c.get('prepared_data'):
                manifest=resolve_path(c['dataset']['root'])/'manifest.json'
                launch=json.loads((tech/'launch.json').read_text())
                while not manifest.exists():
                    # The live controller owns dependency state; sacct requires a
                    # separate accounting daemon which may be unavailable here.
                    status=subprocess.check_output(['scontrol','show','job','-o',launch['jobs']['seal']],text=True)
                    if any(word in status for word in ('JobState=FAILED','JobState=CANCELLED','JobState=TIMEOUT',
                            'JobState=OUT_OF_MEMORY','JobState=NODE_FAIL','Reason=DependencyNeverSatisfied')):
                        raise RuntimeError('Expansion failed; inspect preparation logs: '+status)
                    if 'JobState=COMPLETED' in status and not manifest.exists():
                        raise RuntimeError('Sealing completed without publishing its manifest: '+status)
                    write_json(state,dict(state='waiting_for_expanded_data',jobs=launch['jobs']))
                    time.sleep(20)
                if json.loads(manifest.read_text())['identity']!=launch['dataset_identity']:raise ValueError('Wrong expanded dataset')
            write_json(state,dict(state='training',variant=variant,job=os.environ.get('SLURM_JOB_ID')))
            if c['runtime'].get('gpus',1)>1:
                subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',
                    '--nproc-per-node='+str(c['runtime']['gpus']),'-m',__package__+'.queue','train',
                    '--config',config_path,'--variant',variant],check=True)
                complete=(resolve_path(c['output'])/variant/'technical/complete.json').exists()
            else:complete=run(config_path,variant)
            if not complete:
                write_json(state,dict(state='checkpointed',variant=variant));return
            if deadline(c)-time.time()<5400:
                write_json(state,dict(state='checkpointed',variant=variant,next_stage='evaluation'));return
            write_json(state,dict(state='evaluating',variant=variant))
            evaluate(config_path,variant)
            write_json(state,dict(state='complete',variant=variant,finished_at=time.time()))
            if not resume:
                launch=json.loads((tech/'launch.json').read_text())
                subprocess.run(['scancel',launch['jobs']['continuation']+'_'+str(index)],check=True)
        except BaseException:
            write_json(state,dict(state='failed',variant=variant,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['submit','launch','prepare','seal','worker','resume'])
    p.add_argument('--config',required=True);p.add_argument('--index',type=int);p.add_argument('--gpu',type=int,default=0)
    a=p.parse_args();c=json.loads(Path(a.config).read_text())
    if a.stage=='submit':print(json.dumps(submit(a.config),indent=2))
    elif a.stage=='launch':print(json.dumps(launch(a.config,a.gpu),indent=2))
    elif a.stage=='seal':
        if c.get('expansion'):
            from . import expand
            expand.seal(a.config)
        else:interface.seal(a.config)
    else:
        index=a.index if a.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
        if a.stage=='prepare':
            plan=json.loads((resolve_path(c['dataset']['root'])/'plan.json').read_text())
            size=c['queue']['prepare_sources_per_task']
            if c.get('expansion'):
                from . import expand
                producer=expand.prepare_source
            else:producer=interface.prepare_source
            for item in plan['sources'][index*size:(index+1)*size]:producer(a.config,item['id'])
        else:worker(a.config,index,resume=a.stage=='resume')
