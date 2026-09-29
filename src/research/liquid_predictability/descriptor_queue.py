"""CPU descriptor extraction, GPU boosting and validation-selected comparisons."""
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
from src.data.fixed_cohort.protocol import sha,write_json
from src.project_runtime.paths import resolve_path
from src.experiment_runner.metric_docs import check_metric_docs
from .data import config


def submit(path):
    c=config(path);root=resolve_path(c['output']);tech=root/'technical';tech.mkdir(parents=True,exist_ok=True)
    receipt_path=tech/'launch.json'
    if receipt_path.exists():raise ValueError('Already submitted; use frozen commands and existing source receipts')
    preflight=config(tech/'preflight.json')
    if preflight['config_sha256']!=sha(Path(path)) or not preflight['finite']:raise ValueError('Missing matching local numerical check')
    check_metric_docs(family='liquid_descriptors')
    repo=Path(__file__).resolve().parents[3];code=tech/'code'
    shutil.copytree(repo/'src',code/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.nbc','*.nbi'))
    shutil.copytree(repo/'docs/metrics',code/'docs/metrics')
    destination=code/'configs/liquid_predictability';destination.mkdir(parents=True)
    shutil.copy2(path,destination/Path(path).name);write_json(code/'config.json',c)
    env=dict(PCM_PROJECT_ROOT=str(repo),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
             TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',NUMBA_NUM_THREADS='1')
    command=[sys.executable,'-u','-m','src.research.liquid_predictability.descriptor_queue']
    receipt=dict(submitted_at=time.time(),code=str(code),config_sha256=sha(Path(path)),jobs={})
    def job(stage,options,dependency=None,partition='CPU'):
        cmd=command+[stage,'--config',str(code/'config.json')];script=tech/f'{stage}.sbatch'
        script.write_text('\n'.join(['#!/bin/bash',f'#SBATCH --job-name=LD-{stage}',
            '#SBATCH --partition='+partition,'#SBATCH --nodes=1','#SBATCH --ntasks=1',
            f'#SBATCH --output={tech}/{stage}-%A_%a.log',*['#SBATCH '+v for v in options],
            'set -euo pipefail','ulimit -n 4096','cd '+shlex.quote(str(code)),
            'exec env '+shlex.join([f'{k}={v}' for k,v in env.items()])+' '+shlex.join(cmd),'']))
        args=['sbatch','--parsable']+(['--dependency='+dependency] if dependency else [])+[str(script)]
        ident=subprocess.check_output(args,text=True).strip().split(';')[0]
        receipt['jobs'][stage]=ident;write_json(receipt_path,receipt);return ident
    try:
        if c.get('prepared_launch'):
            original=config(resolve_path(c['prepared_launch']))
            preparation_config=config(Path(original['code'])/'config.json')
            for key in ('cache','parent_config_sha256','preparation'):
                if preparation_config[key]!=c[key]:raise ValueError(f'Reused preparation changed: {key}')
            prep=original['jobs']['prepare'];sealed=original['jobs']['seal']
            receipt['reused_preparation']=dict(launch=c['prepared_launch'],prepare=prep,seal=sealed)
            write_json(receipt_path,receipt)
        else:
            prep=job('prepare',[f'--array=0-{c["preparation"]["tasks"]-1}%{c["preparation"]["tasks"]}',
                f'--cpus-per-task={c["preparation"]["workers"]}','--mem=24G','--time=08:00:00'])
            sealed=job('seal',['--cpus-per-task=2','--mem=24G','--time=01:00:00'],'afterok:'+prep)
        cpu=job('cpu-worker',[f'--cpus-per-task={c["fit_threads"]}','--mem=64G','--time=10:00:00'],'afterok:'+sealed)
        lanes=c['gpu_lanes']
        gpu=job('gpu-worker',[f'--array=0-{lanes-1}%{lanes}','--gpus=1',
            f'--cpus-per-task={c["fit_threads"]}','--mem=64G','--time=06:00:00'],
            'afterok:'+sealed,partition=c['gpu_partition'])
        job('compare',['--cpus-per-task=2','--mem=24G','--time=01:00:00'],'afterok:'+cpu+':'+gpu)
    except BaseException:
        receipt['submission_error']=traceback.format_exc();write_json(receipt_path,receipt)
        raise
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['submit','prepare','seal','cpu-worker','gpu-worker','fit','compare'])
    p.add_argument('--config',required=True);p.add_argument('--index',type=int);p.add_argument('--arm')
    args=p.parse_args();c=config(args.config)
    if args.stage=='submit':print(json.dumps(submit(args.config),indent=2));return
    index=args.index if args.index is not None else int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))
    tech=resolve_path(c['output'])/'technical';state=tech/f'{args.stage}-{args.arm or index}.json'
    try:
        write_json(state,dict(state='running',job=os.environ.get('SLURM_JOB_ID')))
        if args.stage=='prepare':
            from .descriptor_data import prepare
            prepare(c,index)
        elif args.stage=='seal':
            from .descriptor_data import seal
            seal(c)
        elif args.stage in ('cpu-worker','gpu-worker'):
            gpu=args.stage=='gpu-worker'
            arms=[arm for arm in c['arms'] if (arm['model']=='catboost')==gpu]
            if gpu:arms=arms[index::c['gpu_lanes']]
            for arm in arms:
                write_json(state,dict(state='running',arm=arm['name']))
                subprocess.run([sys.executable,'-u','-m','src.research.liquid_predictability.descriptor_queue',
                    'fit','--config',args.config,'--arm',arm['name']],check=True)
        else:
            from .descriptor_fit import fit,compare
            fit(c,args.arm) if args.stage=='fit' else compare(c)
        write_json(state,dict(state='complete',finished_at=time.time()))
    except BaseException:
        write_json(state,dict(state='failed',traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
