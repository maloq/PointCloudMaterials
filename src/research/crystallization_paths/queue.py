"""Detached companion queue; original hazard experiments and their frozen code are untouched."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import sys
import time
import traceback
import torch
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import save_json,file_hash
from src.research.crystallization_transfer.data import freeze
from src.research.crystallization_transfer.runtime import setup
from src.training_methods.shared_pretraining.queue import snapshot,deadline_for_job
from src.experiment_runner.slurm import submit_sbatch
from src.experiment_runner.metric_docs import check_metric_docs
from .data import prepare_source,ResidentPaths
from .runtime import fit


def select_context(config):
    """Freeze completed frozen-backbone hyperparameters using selection data only."""
    candidates=[]
    for folder in (resolve_path(config['reference_output'])/'technical/runs').iterdir():
        if not (folder/'status.json').exists():continue
        status=json.loads((folder/'status.json').read_text())
        if status['state']!='complete':continue
        spec=json.loads((folder/'spec.json').read_text())
        if spec['mode']!='frozen' or spec['equivariant']:continue
        candidates.append((status['best_selection_nll'],spec['name'],spec,file_hash(folder/'status.json')))
    if not candidates:raise ValueError('No completed scalar frozen-backbone reference available')
    score,name,spec,sha=min(candidates,key=lambda x:(x[0],x[1]))
    return dict(spec=spec,selection_nll=score,source=name,status_sha256=sha,
        criterion='Lowest completed scalar frozen-encoder selection NLL at submission; test metrics not read')


def tasks(config,reference):
    if 'refinement' in config:
        from .refinement import screens
        return screens(config,reference)
    result=[]
    for epochs in config['epochs']:
        for method in ('direct','ar_mse','ar_gaussian','mixture','diffusion'):
            spec=dict(reference['spec']);spec.update(method=method,name=f'{method}-E{epochs}',mode='frozen',
                training=dict(budget='epochs',epochs=epochs,sources=90,window_fraction=1.))
            result.append(spec)
    return result


def freeze_plan(config):
    plan=freeze(config)
    if 'future_cache_plan' in config:
        path=resolve_path(config['future_cache_plan'])
        if file_hash(path)!=config['future_cache_plan_sha256']:raise ValueError('Reused future-feature plan changed')
        original=json.loads(path.read_text())
        for key in ('sources','checkpoint_sha256','scale','anchors','lags'):
            if plan[key]!=original[key]:raise ValueError(f'Reused future cache differs in {key}')
        if resolve_path(config['future_cache'])!=resolve_path(original['config']['future_cache']):raise ValueError('Reused future cache path differs')
        plan['future_cache_identity']=original['identity']
    return plan


def worker(config,lane):
    setup();plan=freeze_plan(config);root=resolve_path(config['output'])/'technical';deadline=deadline_for_job()
    def state(name,**extra):save_json(root/f'lane-{lane}.json',dict(state=name,lane=lane,pid=os.getpid(),updated_at=time.time(),**extra))
    try:
        model=None;cache=resolve_path(config['future_cache']);cache.mkdir(parents=True,exist_ok=True)
        for source in plan['sources']:
            if time.time()>deadline-600:state('checkpointed',stage='future_center_extraction');return
            folder=cache/str(source['id']);folder.mkdir(exist_ok=True)
            with (folder/'worker.lock').open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                state('preparing_future_centers',source=source['id']);model=prepare_source(plan,source,model)
        del model;torch.cuda.empty_cache()
        while not all((cache/str(s['id'])/'complete.json').exists() for s in plan['sources']):
            state('waiting_for_future_centers')
            for p in root.glob('lane-*.json'):
                if json.loads(p.read_text())['state']=='failed':raise RuntimeError(f'Peer failed: {p}')
            if time.time()>deadline-600:state('checkpointed',stage='future_center_wait');return
            time.sleep(15)
        specs=json.loads((root/'queue.json').read_text());state('loading_resident_timelines');data=ResidentPaths(plan,specs[0])
        while time.time()<deadline-1800:
            if 'refinement' in config:
                from .refinement import expanded_tasks
                specs=expanded_tasks(config,root)
            unfinished=False
            for spec in specs:
                folder=root/'runs'/spec['name'];folder.mkdir(parents=True,exist_ok=True)
                with (folder/'worker.lock').open('a') as lock:
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:unfinished=True;continue
                    status=folder/'status.json'
                    if status.exists():
                        prior=json.loads(status.read_text())
                        if prior['state']=='complete':continue
                        if prior['state']=='failed':raise RuntimeError(f'Fit failed: {spec["name"]}; inspect before resuming')
                    unfinished=True;state('training',variant=spec['name']);save_json(folder/'spec.json',spec)
                    try:complete=fit(plan,spec,data,deadline)
                    except Exception as error:
                        save_json(status,dict(state='failed',error=repr(error),traceback=traceback.format_exc()));raise
                    if not complete:state('checkpointed',variant=spec['name']);return
            if not unfinished:
                if 'refinement' in config:
                    extended=expanded_tasks(config,root)
                    if len(extended)>len(specs):continue
                break
            time.sleep(10)
        state('finished_available_queue')
    except Exception as error:
        state('failed',error=repr(error),traceback=traceback.format_exc());raise


def submit(config_path):
    config=json.loads(Path(config_path).read_text());plan=freeze_plan(config);root=resolve_path(config['output'])/'technical'
    if (root/'submissions.json').exists():raise FileExistsError('Path queue already submitted')
    check_metric_docs(family='crystallization_paths_refinement' if 'refinement' in config else 'crystallization_paths')
    reference=select_context(config);save_json(root/'reference-selection.json',reference)
    save_json(root/'queue.json',tasks(config,reference))
    code=snapshot(root);records=[]
    for lane in range(config['workers']):
        command=[sys.executable,'-u','-m','src.research.crystallization_paths.queue','worker',
            '--config',str(code/config_path),'--lane',str(lane)]
        script='\n'.join(['#!/bin/bash',f'#SBATCH --job-name=crystal-paths-{lane}',
            f'#SBATCH --partition={config["partition"]}','#SBATCH --gres=gpu:1','#SBATCH --cpus-per-task=8',
            '#SBATCH --mem=64G',f'#SBATCH --time={config["hours"]}:00:00',
            f'#SBATCH --output={root}/lane-{lane}-%j.log','#SBATCH --signal=B:USR1@300',
            'set -euo pipefail','cd '+shlex.quote(str(code)),
            'export PCM_PROJECT_ROOT='+shlex.quote(str(code)),
            'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1',
            'export PYTORCH_ALLOC_CONF=expandable_segments:True',
            'exec '+shlex.join(command),''])
        job=submit_sbatch(script,root/f'lane-{lane}.sbatch');records.append(dict(lane=lane,job_id=job))
        save_json(root/'submissions.json',records)
    print(json.dumps(records,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['submit','worker'])
    parser.add_argument('--config',required=True);parser.add_argument('--lane',type=int)
    args=parser.parse_args()
    if args.command=='submit':submit(args.config)
    else:worker(json.loads(Path(args.config).read_text()),args.lane)
