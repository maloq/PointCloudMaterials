"""Frozen one-GPU lanes for joint spatial localization and associated evaluation."""
import argparse
import fcntl
import json
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


def worker(config_path,lane):
    c=json.loads(Path(config_path).read_text());tech=resolve_path(c['output'])/'technical'
    state=tech/f'lane-{lane}.json'
    variants={0:['distance_direction_vcreg','distance_only'],1:['distance_direction']}[lane]
    with (tech/f'lane-{lane}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for variant in variants:
                root=resolve_path(c['output'])/variant
                if (root/'technical/complete.json').exists():continue
                write_json(state,dict(state='training',variant=variant,job=os.environ.get('SLURM_JOB_ID')))
                subprocess.run([sys.executable,'-u','-m',__package__+'.queue','train','--config',config_path,'--variant',variant],check=True)
                if not (root/'technical/complete.json').exists():
                    write_json(state,dict(state='checkpointed',variant=variant));return
            for variant in variants:
                write_json(state,dict(state='evaluating',variant=variant,job=os.environ.get('SLURM_JOB_ID')))
                subprocess.run([sys.executable,'-u','-m',__package__+'.queue','evaluate','--config',config_path,'--variant',variant],check=True)
            write_json(state,dict(state='complete',variants=variants,finished_at=time.time()))
            launch=json.loads((tech/'launch.json').read_text())
            continuation=launch['lanes'][str(lane)]['continuation_job']
            if continuation!=os.environ.get('SLURM_JOB_ID'):
                subprocess.run(['scancel',continuation],check=True)
        except BaseException:
            write_json(state,dict(state='failed',traceback=traceback.format_exc()));raise


def launch(config_path):
    c=json.loads(Path(config_path).read_text());root=resolve_path(c['output']);tech=root/'technical'
    if (tech/'launch.json').exists():raise ValueError('Already launched; resume the captured lane command')
    allocation=os.environ.get('SLURM_JOB_ID')
    if not allocation:raise ValueError('Launch inside the intended live two-GPU allocation')
    preflight=json.loads((tech/'preflight.json').read_text())
    if preflight['batch']!=c['batch_size'] or not preflight['finite_gradients'] or preflight['config_sha256']!=sha(Path(config_path)):
        raise ValueError('Missing full-batch verification for this exact configuration')
    manifest=resolve_path(c['dataset']['root'])/'manifest.json'
    if json.loads(manifest.read_text())['state']!='complete':raise ValueError('Data are not sealed')
    check_metric_docs(family='crystal_vector')
    repo=Path(__file__).resolve().parents[3];code=tech/'code'
    shutil.copytree(repo/'src',code/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(repo/'docs/metrics',code/'docs/metrics')
    write_json(code/'config.json',c)
    environment=dict(PCM_PROJECT_ROOT=str(repo),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OVITO_THREAD_COUNT='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',PYTORCH_ALLOC_CONF='expandable_segments:True',
        TORCHINDUCTOR_COMPILE_THREADS='4')
    receipt=dict(allocation=allocation,code=str(code),dataset_manifest_sha256=sha(manifest),
        submitted_at=time.time(),lanes={})
    for lane in (0,1):
        command=[sys.executable,'-u','-m',__package__+'.queue','worker','--config',str(code/'config.json'),'--lane',str(lane)]
        script=tech/f'resume-lane-{lane}.sbatch'
        script.write_text('\n'.join(['#!/bin/bash',f'#SBATCH --job-name=CDV-resume-{lane}',
            '#SBATCH --partition=RTX6000PRO,H100','#SBATCH --nodes=1','#SBATCH --ntasks=1','#SBATCH --gpus=1',
            '#SBATCH --cpus-per-task=8','#SBATCH --mem=48G','#SBATCH --time=08:00:00',
            f'#SBATCH --output={tech}/resume-{lane}-%j.log','set -euo pipefail','ulimit -n 4096',
            'cd '+shlex.quote(str(code)), 'exec env '+shlex.join([f'{k}={v}' for k,v in environment.items()])+' '+shlex.join(command),'']))
        job=subprocess.check_output(['sbatch','--parsable',f'--dependency=afterany:{allocation}',str(script)],text=True).strip().split(';')[0]
        receipt['lanes'][str(lane)]=dict(gpu=lane,command=command,continuation_job=job)
    # Write continuation identities before workers can complete and cancel them.
    write_json(tech/'launch.json',receipt)
    for lane in (0,1):
        env=dict(os.environ,**environment,CUDA_VISIBLE_DEVICES=str(lane))
        with (tech/f'lane-{lane}.log').open('ab') as out:
            p=subprocess.Popen(receipt['lanes'][str(lane)]['command'],cwd=code,env=env,
                stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        receipt['lanes'][str(lane)]['pid']=p.pid
    write_json(tech/'launch.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','launch','worker','train','evaluate'])
    p.add_argument('--config',required=True);p.add_argument('--variant');p.add_argument('--lane',type=int,choices=[0,1])
    a=p.parse_args()
    if a.stage=='prepare':
        from .data import prepare
        prepare(a.config)
    elif a.stage=='launch':print(json.dumps(launch(a.config),indent=2))
    elif a.stage=='worker':worker(a.config,a.lane)
    elif a.stage=='train':
        from .train import run
        run(a.config,a.variant)
    else:
        from .evaluate import run
        run(a.config,a.variant)
