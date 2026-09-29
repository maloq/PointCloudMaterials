"""Detached sensitivity, descriptor-learning and paired-relaxation Slurm queue."""
import argparse
import copy
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback

from src.data.fixed_cohort.protocol import sha, write_json
from src.project_runtime.paths import resolve_path, REPO as PROJECT_ROOT
from src.experiment_runner.metric_docs import check_metric_docs
from .data import config


def recipes(c):
    root=resolve_path(c['output']);folder=root/'technical/recipes';folder.mkdir(parents=True,exist_ok=True)
    base=config(resolve_path(c['descriptor_parent']));base.pop('prepared_launch',None);base.pop('backend_change',None)
    names=[]
    for arm in c['signals']:
        dc=copy.deepcopy(base);name=arm['name'];names.append(name)
        dc.update(output=str(root/'descriptors'/name),cache=str(resolve_path(c['cache'])/name),
                  observation=dict(domain='raw',relaxed=False),target_protocol=dict(kind='synthetic',signal=arm),
                  arms=[a for a in dc['arms'] if a['name'] in ('prior','all_catboost_shallow')])
        write_json(folder/f'{name}.json',dc)
    for domain in c['paired_protocols']:
        dc=copy.deepcopy(base);names.append(domain)
        dc.update(output=str(root/'descriptors'/domain),cache=str(resolve_path(c['cache'])/domain),
                  observation=dict(domain=domain,relaxed=domain.startswith('relaxed'),
                    archived_float16=True,matched_common_cohort=True,hot_atom_membership_frozen=True,
                    full_cell_relaxation_has_external_context=domain.startswith('relaxed')),
                  target_protocol='relaxed instantaneous PTM cluster distance' if domain.endswith('newlabels') else 'original established MD crystal distance')
        write_json(folder/f'{domain}.json',dc)
        bc=config(resolve_path(c['baseline_parent']));bc.update(parent_config=str(folder/f'{domain}.json'),
            parent_config_sha256=sha(folder/f'{domain}.json'),output=str(root/'descriptors'/domain/'baselines-v1'))
        write_json(folder/f'{domain}-baselines.json',bc)
    return names


def submit(path):
    from .control_data import freeze
    c=config(path);root=resolve_path(c['output']);tech=root/'technical';tech.mkdir(parents=True,exist_ok=True)
    receipt_file=tech/'launch.json'
    if receipt_file.exists():raise ValueError('Already submitted; resume frozen commands, do not duplicate jobs')
    for family in ('liquid_controls','liquid_descriptors','liquid_descriptor_baselines'):check_metric_docs(family=family)
    freeze(c);cohorts=recipes(c)
    repo=Path(__file__).resolve().parents[3];code=tech/'code'
    shutil.copytree(repo/'src',code/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.nbc','*.nbi'))
    shutil.copytree(repo/'docs/metrics',code/'docs/metrics');shutil.copytree(repo/'configs/liquid_predictability',code/'configs/liquid_predictability')
    write_json(code/'config.json',c)
    receipt=dict(config_sha256=sha(Path(path)),code=str(code),submitted_at=time.time(),jobs={},
                 recipes={str(p):sha(p) for p in folder_files(tech/'recipes')})
    # A dependent production job may submit from a frozen code snapshot. Keep
    # that code immutable while resolving storage/catalog against the project.
    env=dict(PCM_PROJECT_ROOT=str(PROJECT_ROOT),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
             TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',NUMBA_NUM_THREADS='1',OVITO_THREAD_COUNT='1')
    def job(stage,options,after=None,gpu=False):
        command=[sys.executable,'-u','-m','src.research.liquid_predictability.control_queue',stage,'--config',str(code/'config.json')]
        script=tech/f'{stage}.sbatch';script.write_text('\n'.join(['#!/bin/bash',f'#SBATCH --job-name=LC-{stage}',
            '#SBATCH --partition='+(c['gpu_partition'] if gpu else 'CPU'),'#SBATCH --nodes=1','#SBATCH --ntasks=1',
            f'#SBATCH --output={tech}/{stage}-%A_%a.log',*['#SBATCH '+s for s in options],
            'set -euo pipefail','ulimit -n 4096','cd '+shlex.quote(str(code)),
            'exec env '+shlex.join([f'{k}={v}' for k,v in env.items()])+' '+shlex.join(command),'']))
        args=['sbatch','--parsable']+(['--dependency=afterok:'+':'.join(after)] if after else [])+[str(script)]
        ident=subprocess.check_output(args,text=True).strip().split(';')[0];receipt['jobs'][stage]=ident;write_json(receipt_file,receipt);return ident
    try:
        syn=job('synthetic',['--cpus-per-task=4','--mem=48G','--time=02:00:00'])
        prep=job('paired-prepare',[f'--array=0-{c["preparation"]["tasks"]-1}%{c["preparation"]["tasks"]}',
            f'--cpus-per-task={c["preparation"]["workers"]}','--mem=32G','--time=08:00:00'])
        seal=job('paired-seal',['--cpus-per-task=2','--mem=48G','--time=01:00:00'],[prep])
        cpu=job('baselines',[f'--array=0-{len(c["paired_protocols"])-1}%2','--cpus-per-task=8','--mem=64G','--time=04:00:00'],[seal])
        # Run GPU boosting first on one GPU. Subsequent MACE lanes use two GPUs,
        # so this queue never holds more than two training GPUs at once.
        boost=job('boosting',['--gpus=1','--cpus-per-task=8','--mem=64G','--time=04:00:00'],[syn,seal],True)
        check=job('preflight',['--gpus=1','--cpus-per-task=4','--mem=64G','--time=01:00:00'],[boost],True)
        mace=job('mace',['--array=0-1%2','--gpus=1','--cpus-per-task=4','--mem=64G','--time=16:00:00'],[check],True)
        job('report',['--cpus-per-task=4','--mem=48G','--time=02:00:00'],[cpu,boost,mace])
    except BaseException:
        receipt['submission_error']=traceback.format_exc();write_json(receipt_file,receipt);raise
    return receipt


def folder_files(folder):return sorted(folder.glob('*.json'))


def execute(c,stage,index,arm=None):
    root=resolve_path(c['output']);folder=root/'technical/recipes'
    if stage=='synthetic':
        from .control_data import synthetic
        synthetic(c)
    elif stage=='paired-prepare':
        from .control_data import paired_prepare
        paired_prepare(c,index)
    elif stage=='paired-seal':
        from .control_data import paired_seal
        paired_seal(c)
    elif stage=='baselines':
        from threadpoolctl import threadpool_limits
        from .descriptor_fit import fit
        from .descriptor_baselines import fit as baseline_fit
        name=c['paired_protocols'][index];dc=config(folder/f'{name}.json')
        with threadpool_limits(limits=8):
            for a in dc['arms']:
                if a['model']!='catboost':fit(dc,a['name'])
            baseline_fit(config(folder/f'{name}-baselines.json'))
    elif stage=='boosting':
        from .descriptor_fit import fit
        for name in [a['name'] for a in c['signals']]+c['paired_protocols']:
            dc=config(folder/f'{name}.json')
            for a in dc['arms']:
                if a['model']=='catboost' or (name not in c['paired_protocols'] and a['model']=='prior'):
                    fit(dc,a['name'])
    elif stage=='preflight':
        from .control_train import run
        from .descriptor_data import load
        # Verify paired row/target identity separately from any scientific fit.
        dc=[config(folder/f'{d}.json') for d in c['paired_protocols']];datasets=[load(v) for v in dc]
        import numpy as np
        rows=datasets[0][1]
        for _,r,_,_ in datasets[1:]:
            for k in ('ids','role','source','weights'):
                if not np.array_equal(rows[k],r[k]):raise ValueError(f'Unmatched factorial rows: {k}')
        results=[]
        for name in c['preflight_arms']:
            results.append(run(c,name,preflight=True))
            import gc,torch
            gc.collect();torch.cuda.empty_cache()
        write_json(root/'technical/preflight.json',dict(finite=True,results=results,paired_rows=len(rows['ids']),created_online_runs=0))
    elif stage in ('mace','mace-fit'):
        if stage=='mace-fit':
            from .control_train import run
            run(c,arm)
            if not (root/'mace'/arm/'analyses/prediction-v1/technical/complete.json').exists():
                raise RuntimeError(f'Checkpointed before completion: {arm}; resume the recorded frozen command')
        else:
            path=Path(__file__).resolve().parents[3]/'config.json'
            for a in c['mace_arms'][index::2]:
                subprocess.run([sys.executable,'-u','-m','src.research.liquid_predictability.control_queue','mace-fit',
                    '--config',str(path),'--arm',a['name']],check=True)
    elif stage=='report':
        from .descriptor_fit import compare
        from .descriptor_baselines import compare as baselines_compare
        for name in [a['name'] for a in c['signals']]+c['paired_protocols']:
            compare(config(folder/f'{name}.json'))
            if name in c['paired_protocols']:baselines_compare(config(folder/f'{name}-baselines.json'))
        from .control_report import report
        report(c)
    else:raise ValueError(stage)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['submit','freeze','synthetic','paired-prepare','paired-seal','baselines','boosting','preflight','mace','mace-fit','report'])
    p.add_argument('--config',required=True);p.add_argument('--index',type=int);p.add_argument('--arm');a=p.parse_args();c=config(a.config)
    if a.stage=='submit':print(json.dumps(submit(a.config),indent=2));return
    if a.stage=='freeze':
        from .control_data import freeze
        print(json.dumps(freeze(c)['counts']));recipes(c);return
    index=a.index if a.index is not None else int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))
    state=resolve_path(c['output'])/'technical'/f'{a.stage}-{a.arm or index}.json'
    try:
        write_json(state,dict(state='running',started_at=time.time(),job=os.environ.get('SLURM_JOB_ID')))
        execute(c,a.stage,index,a.arm)
        write_json(state,dict(state='complete',finished_at=time.time()))
    except BaseException:
        write_json(state,dict(state='failed',traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
