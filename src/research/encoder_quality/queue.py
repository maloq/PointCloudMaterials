"""Snapshot and submit the frozen encoder-evaluation sequence to Slurm."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from .common import load
from .report import collect
from src.research.structural_state.common import sha,write_json


def submit(path,node):
    config=load(path);root=Path(config['output']);technical=root/'technical'
    if (technical/'launch.json').exists():raise FileExistsError('Evaluation already submitted; inspect its recorded job')
    check=json.loads((technical/'checks-scratch-hot.json').read_text())
    if check['run_source_sha256']!=sha(Path(__file__).with_name('run.py')):
        raise ValueError('Preflight does not match the evaluation producer')
    if check['config_sha256']!=sha(path):raise ValueError('Preflight configuration changed')
    if not all(check['checks'][k]['passed'] for k in ('repeat','rotation','permutation','translation_recentered','periodic_image_recentered')):
        raise ValueError('Native consistency checks did not pass')
    if not (Path(config['reference'])/'manifest.json').exists():raise ValueError('Static reference preparation incomplete')
    from src.training_methods.shared_pretraining.queue import snapshot
    code=snapshot(technical)
    frozen=code/Path(path).resolve().relative_to(Path.cwd())
    command=[sys.executable,'-u','-m','src.research.encoder_quality.queue','worker','--config',str(frozen)]
    environment=['PCM_PROJECT_ROOT='+str(Path.cwd()),'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
        'TORCHINDUCTOR_COMPILE_THREADS=4','TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1','PYTORCH_ALLOC_CONF=expandable_segments:True']
    script=technical/'evaluation.sbatch'
    lines=['#!/bin/bash','#SBATCH --job-name=latest-mace-quality','#SBATCH --partition=RTX6000PRO',
        '#SBATCH --gres=gpu:1','#SBATCH --cpus-per-task=4','#SBATCH --mem=24G','#SBATCH --time=12:00:00',
        '#SBATCH --nodelist='+node,'#SBATCH --output='+str(technical/'slurm-%j.log'),
        '#SBATCH --chdir='+str(code),'set -euo pipefail','exec env '+shlex.join(environment)+' '+shlex.join(command),'']
    script.write_text('\n'.join(lines))
    job=subprocess.check_output(['sbatch','--parsable',str(script)],text=True).strip()
    receipt=dict(job=job,node=node,script=str(script),code=str(code),config=str(frozen),models=[m['name'] for m in config['models']],
                 source_checked=True,submitted_at=time.time())
    write_json(technical/'launch.json',receipt);collect(config);return receipt


def worker(path):
    config=load(path);technical=Path(config['output'])/'technical';failures=[]
    for spec in config['models']:
        name=spec['name'];write_json(technical/'queue-state.json',dict(state='running',active=name,failed=failures,updated_at=time.time()))
        with (technical/f'{name}.log').open('ab',buffering=0) as log:
            completed=subprocess.run([sys.executable,'-u','-m','src.research.encoder_quality.run','--config',str(path),'--name',name],
                stdout=log,stderr=subprocess.STDOUT)
        if completed.returncode:failures.append(name)
        collect(config)
    result=dict(state='failed' if failures else 'complete',failed=failures,finished_at=time.time(),job=os.environ.get('SLURM_JOB_ID'))
    write_json(technical/'queue-state.json',result)
    if failures:raise RuntimeError(f'Encoder evaluations failed: {failures}; see per-model logs/receipts')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('action',choices=['submit','worker','report'])
    parser.add_argument('--config',required=True);parser.add_argument('--node',default='node58');args=parser.parse_args()
    result=submit(args.config,args.node) if args.action=='submit' else worker(args.config) if args.action=='worker' else collect(load(args.config))
    print(json.dumps(result,indent=2),flush=True)
