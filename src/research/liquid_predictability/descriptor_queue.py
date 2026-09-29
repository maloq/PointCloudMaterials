"""CPU descriptor extraction, GPU boosting and validation-selected comparisons."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from src.data.fixed_cohort.protocol import sha,write_json
from src.project_runtime.paths import resolve_path
from src.experiment_runner.metric_docs import check_metric_docs
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, recorded_stage
from .data import config


def submit(path):
    c=config(path);root=resolve_path(c['output']);tech=root/'technical';tech.mkdir(parents=True,exist_ok=True)
    receipt_path=tech/'launch.json'
    if receipt_path.exists():raise ValueError('Already submitted; use frozen commands and existing source receipts')
    preflight=config(tech/'preflight.json')
    if preflight['config_sha256']!=sha(Path(path)) or not preflight['finite']:raise ValueError('Missing matching local numerical check')
    check_metric_docs(family='liquid_descriptors')
    repo = Path(__file__).resolve().parents[3]
    bundle = ExecutionBundle.freeze(
        repo, tech / 'code', c, directories=('src', 'docs/metrics'),
        files=((path, 'configs/liquid_predictability/' + Path(path).name),),
    )
    code = bundle.root
    env=dict(PCM_PROJECT_ROOT=str(repo),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
             TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',NUMBA_NUM_THREADS='1')
    receipt=dict(submitted_at=time.time(),code=str(code),config_sha256=sha(Path(path)),jobs={})
    queue = SlurmQueue(tech, bundle, 'src.research.liquid_predictability.descriptor_queue',
                       env, receipt_path, receipt, 'LD')
    job = queue.submit
    with queue.submission():
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
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['submit','prepare','seal','cpu-worker','gpu-worker','fit','compare'])
    p.add_argument('--config',required=True);p.add_argument('--index',type=int);p.add_argument('--arm')
    args=p.parse_args();c=config(args.config)
    if args.stage=='submit':print(json.dumps(submit(args.config),indent=2));return
    index=args.index if args.index is not None else int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))
    tech=resolve_path(c['output'])/'technical';state=tech/f'{args.stage}-{args.arm or index}.json'
    with recorded_stage(state, job=os.environ.get('SLURM_JOB_ID')) as progress:
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
                progress.update(arm=arm['name'])
                subprocess.run([sys.executable,'-u','-m','src.research.liquid_predictability.descriptor_queue',
                    'fit','--config',args.config,'--arm',arm['name']],check=True)
        else:
            from .descriptor_fit import fit,compare
            fit(c,args.arm) if args.stage=='fit' else compare(c)


if __name__=='__main__':main()
