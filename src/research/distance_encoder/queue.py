"""Freeze and submit the matched VCReg training and local-only evaluations."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from src.project_runtime.paths import resolve_path
from src.data.fixed_cohort.protocol import write_json
from src.experiment_runner.metric_docs import check_metric_docs


def submit(path):
    config=json.loads(Path(path).read_text());root=resolve_path(config['output']);tech=root/'technical';tech.mkdir(parents=True,exist_ok=True)
    if (tech/'launch.json').exists():raise ValueError('This run is already submitted; resume the recorded frozen worker instead')
    for family in ('distance_encoder','distance_encoder_local','spatial_distance','spatial_confidence'):check_metric_docs(family=family)
    repo=Path(__file__).resolve().parents[3];code=tech/'code'
    shutil.copytree(repo/'src',code/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(repo/'docs/metrics',code/'docs/metrics');write_json(code/'config.json',config)
    frozen_baseline=code/'baseline.json';shutil.copy2(resolve_path(config['baseline_local_config']),frozen_baseline)
    environment=dict(PCM_PROJECT_ROOT=str(repo),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',PYTORCH_ALLOC_CONF='expandable_segments:True',TORCHINDUCTOR_COMPILE_THREADS='4')
    slurm=config['slurm'];script=tech/'train.sbatch'
    lines=['#!/bin/bash',f'#SBATCH --job-name={config["encoder_name"]}',f'#SBATCH --partition={slurm["partition"]}',
        '#SBATCH --nodes=1','#SBATCH --ntasks=1',f'#SBATCH --gres=gpu:{slurm["gpus"]}',f'#SBATCH --cpus-per-task={slurm["cpus"]}',
        f'#SBATCH --mem={slurm["memory"]}',f'#SBATCH --time={slurm["time"]}',f'#SBATCH --output={tech}/slurm-%j.log',
        'set -euo pipefail','cd '+shlex.quote(str(code)),
        'exec env '+shlex.join([f'{k}={v}' for k,v in environment.items()])+' '+shlex.join([sys.executable,'-u','-m',
            'src.research.distance_encoder.queue','worker','--config',str(code/'config.json')]),'']
    script.write_text('\n'.join(lines));job=subprocess.check_output(['sbatch','--parsable',str(script)],text=True).strip()
    result=dict(job=job,script=str(script),code=str(code),submitted_at=time.time(),name=config['encoder_name'])
    write_json(tech/'launch.json',result);return result


def worker(path):
    c=json.loads(Path(path).read_text());tech=resolve_path(c['output'])/'technical';state=tech/'queue-state.json'
    try:
        write_json(state,dict(state='training',job=os.environ.get('SLURM_JOB_ID')))
        subprocess.run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1',f'--nproc_per_node={c["world_size"]}',
            '-m','src.research.distance_encoder.train','--config',str(path)],check=True)
        if not (tech/'complete.json').exists():raise RuntimeError('Encoder stopped short of the required 12 epochs; checkpoint remains resumable')
        for config in (Path(path).with_name('baseline.json'),Path(path)):
            item=json.loads(config.read_text());root=resolve_path(item['output'])
            if not (root/'technical/complete.json').exists():raise RuntimeError(f'Required completed encoder unavailable: {root}')
            write_json(state,dict(state='local_evaluation',active=item['encoder_name']))
            subprocess.run([sys.executable,'-u','-m','src.research.distance_encoder.local','--config',str(config),
                '--checkpoint',str(root/'technical/best.pt'),'--output',str(root/'local-only'),
                '--name',item['encoder_name']],check=True)
        write_json(state,dict(state='complete',finished_at=time.time()))
    except BaseException:
        write_json(state,dict(state='failed',traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['submit','worker']);p.add_argument('--config',required=True)
    a=p.parse_args();print(json.dumps(submit(a.config) if a.action=='submit' else worker(a.config),indent=2))
