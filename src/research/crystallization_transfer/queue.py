"""Four independent GPU lanes, shared immutable data and lock-claimed fit queue."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import torch
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import save_json,file_hash
from src.training_methods.shared_pretraining.queue import deadline_for_job,snapshot
from .data import freeze,prepare_source
from .runtime import setup,extract,fit


def variants(config=None):
    if config is not None and 'refinement' in config:
        from .refinement import variants as refined_variants
        return refined_variants(config['refinement'])
    if config is not None and 'scaling' in config:
        from .scaling import variants as scaling_variants
        return scaling_variants(config['scaling'])
    items=[]
    def add(mode,history,radius,aggregation,equivariant=False,baseline=None,repeat=False):
        name=f'{mode}-H{history}-R{radius}-{aggregation}'+('-tensor' if equivariant else '')+('-'+baseline if baseline else '')+('-repeat' if repeat else '')
        if any(s['name']==name for s in items):return
        items.append(dict(name=name,mode=mode,history_ps=history,radius_A=radius,aggregation=aggregation,equivariant=equivariant,baseline=baseline,repeat=repeat))
    add('frozen',0,0,'linear',baseline='persistence')
    add('frozen',0,0,'linear',baseline='condition');add('frozen',0,0,'mlp',baseline='descriptor')
    add('frozen',0,0,'linear');add('frozen',0,0,'mlp')
    for mode in ('frozen','finetune','scratch'):
        for history in (0,3,12,48):
            add(mode,history,0,'mlp');add(mode,history,25,'mean');add(mode,history,25,'attention')
    for mode in ('frozen','finetune'):
        for history in (0,3,12,48):add(mode,history,25,'attention',equivariant=True)
    for h in (3,12,48):add('frozen',h,0,'mlp')
    for agg in ('mean','attention'):add('frozen',12,12,agg)
    add('frozen',48,25,'attention',repeat=True)
    return items


def worker(config,lane):
    setup();plan=freeze(config);root=resolve_path(config['output'])/'technical';deadline=deadline_for_job();status=root/f'lane-{lane}.json'
    def state(name,**extra):save_json(status,dict(state=name,lane=lane,pid=os.getpid(),updated_at=time.time(),**extra))
    try:
        if 'reuse_plan' in config:
            previous=resolve_path(config['reuse_plan']).parent/f'lane-{lane}.json'
            state('waiting_for_previous_lane',previous=str(previous))
            while config.get('wait_for_previous_lanes',True):
                prior=json.loads(previous.read_text())
                if prior['state']=='failed':raise RuntimeError(f'Previous lane failed: {previous}: {prior}')
                if prior['state'] in ('finished_available_queue','checkpointed'):
                    try:os.kill(prior['pid'],0)
                    except ProcessLookupError:break
                if time.time()>deadline-1200:state('checkpointed',stage='previous_lane_wait');return
                time.sleep(10)
        else:
            sources=plan['sources'][lane::len(config['allocations'])];state('preparing',sources=len(sources))
            with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
                futures={pool.submit(prepare_source,(s,plan)):s for s in sources};model=None
                for future in as_completed(futures):
                    sid=future.result();source=futures[future];state('extracting',source=sid)
                    model=extract(plan,source,model)
                    if time.time()>deadline-1200:state('checkpointed',stage='preparation');return
            del model;torch.cuda.empty_cache()
        state('waiting_for_shared_release')
        while not all((resolve_path(config['cache'])/str(s['id'])/'features.json').exists() for s in plan['sources']):
            for p in root.glob('lane-*.json'):
                s=json.loads(p.read_text())
                if s['state']=='failed':raise RuntimeError(f'Peer preparation failed: {p}: {s}')
            if time.time()>deadline-1200:state('checkpointed',stage='release_wait');return
            time.sleep(10)
        tasks=variants(config)
        # Shorter remaining H100 allocation prioritizes cheap frozen comparisons.
        if lane!=0 and 'refinement' not in config:
            if 'scaling' in config:
                tasks=sorted(tasks,key=lambda s:(s['training']['epochs']==6,s['mode']=='frozen',s['training']['epochs'],s['equivariant']))
            else:tasks=sorted(tasks,key=lambda s:(s['mode']=='frozen',s['history_ps'],s['equivariant']))
        while time.time()<deadline-1200:
            if 'refinement' in config:
                from .refinement import expanded_tasks
                tasks=expanded_tasks(config,root)
            for spec in tasks:
                if time.time()>deadline-1200:break
                folder=root/'runs'/spec['name'];folder.mkdir(parents=True,exist_ok=True)
                with (folder/'worker.lock').open('a') as lock:
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                    prior=folder/'status.json'
                    if prior.exists():
                        previous=json.loads(prior.read_text())
                        if previous['state']=='complete':continue
                        if previous['state']=='failed':continue
                    state('training',variant=spec['name']);save_json(folder/'spec.json',spec)
                    try:complete=fit(plan,spec,deadline)
                    except Exception as e:
                        save_json(folder/'status.json',dict(state='failed',error=repr(e),traceback=traceback.format_exc()));raise
                    torch.cuda.empty_cache()
                    if not complete:
                        state('checkpointed',variant=spec['name']);return
            if 'refinement' in config:tasks=expanded_tasks(config,root)
            states=[json.loads((root/'runs'/t['name']/'status.json').read_text())['state'] if (root/'runs'/t['name']/'status.json').exists() else 'pending' for t in tasks]
            if all(s in ('complete','failed') for s in states):break
            time.sleep(10)
        state('finished_available_queue',remaining=[t['name'] for t in tasks if not (root/'runs'/t['name']/'status.json').exists() or json.loads((root/'runs'/t['name']/'status.json').read_text())['state']!='complete'])
    except Exception as e:
        state('failed',error=repr(e),traceback=traceback.format_exc());raise


def submit(config_path):
    config=json.loads(Path(config_path).read_text());plan=freeze(config);root=resolve_path(config['output'])/'technical'
    if (root/'submissions.json').exists():raise FileExistsError('Campaign already submitted')
    for path,expected in plan['encoder_files'].items():
        if file_hash(path)!=expected:raise ValueError(f'Encoder changed before submission: {path}')
    from src.experiment_runner.metric_docs import check_metric_docs
    check_metric_docs(family='crystallization_transfer');code=snapshot(root);records=[]
    save_json(root/'queue.json',variants(config))
    for allocation in config['allocations']:
        job=allocation['job'];lane=allocation['lane']
        # --gres allocates both GPUs for node61; explicit local index separates lanes.
        gpu_count=allocation.get('gpu_count',2 if job==999611 else 1)
        cmd=['srun',f'--jobid={job}','--overlap','-N1','-n1',f'--gres=gpu:{gpu_count}',f'--cpus-per-task={allocation["cpus"]}',
            'env',f'CUDA_VISIBLE_DEVICES={allocation["gpu"]}',f'PCM_PROJECT_ROOT={code}',
            'PYTORCH_ALLOC_CONF=expandable_segments:True','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',
            sys.executable,'-u','-m','src.research.crystallization_transfer.queue','worker','--config',str(code/config_path),'--lane',str(lane)]
        with (root/f'lane-{lane}.log').open('ab',buffering=0) as log:
            proc=subprocess.Popen(cmd,cwd=code,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        records.append(dict(**allocation,pid=proc.pid,command=cmd))
    save_json(root/'submissions.json',records);print(json.dumps(records,indent=2))


def submit_continuations(recipe_path):
    """Resume the immutable queue after existing allocations, without changing its protocol."""
    from src.experiment_runner.slurm import submit_sbatch
    recipe=json.loads(Path(recipe_path).read_text());root=resolve_path(recipe['output'])/'technical';code=(root/'code').resolve()
    config=code/recipe['frozen_config']
    if not config.is_file() or not (root/'submissions.json').exists():raise ValueError('Continuation requires a submitted, frozen campaign')
    receipt=root/'continuations.json'
    if receipt.exists():raise FileExistsError('Continuations already submitted')
    records=[]
    for slot in recipe['slots']:
        lane=slot['lane'];script=root/f'continuation-{lane}.sbatch'
        command=['srun','-N1','-n1',f'--cpus-per-task={recipe["cpus"]}','env',f'PCM_PROJECT_ROOT={code}',
            'PYTORCH_ALLOC_CONF=expandable_segments:True','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',
            sys.executable,'-u','-m','src.research.crystallization_transfer.queue','worker','--config',str(config),'--lane',str(lane)]
        # Preserve Slurm's assigned CUDA_VISIBLE_DEVICES for a one-GPU allocation.
        content='\n'.join(['#!/bin/bash',f'#SBATCH --job-name=crystal-adaptive-{lane}',f'#SBATCH --partition={recipe["partition"]}',
            '#SBATCH --gres=gpu:1',f'#SBATCH --cpus-per-task={recipe["cpus"]}',f'#SBATCH --mem={recipe["memory"]}',
            f'#SBATCH --time={recipe["hours"]}:00:00',f'#SBATCH --dependency=afterany:{slot["after_job"]}',
            f'#SBATCH --output={root}/continuation-{lane}-%j.log','#SBATCH --signal=USR1@300','set -euo pipefail',
            'cd '+shlex.quote(str(code)),'exec '+shlex.join(command),''])
        job=submit_sbatch(content,script);records.append(dict(**slot,job_id=job,script=str(script),hours=recipe['hours'],partition=recipe['partition']))
        save_json(receipt,records)
    print(json.dumps(records,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['submit','worker','continue']);p.add_argument('--config',required=True);p.add_argument('--lane',type=int)
    a=p.parse_args()
    if a.command=='submit':submit(a.config)
    elif a.command=='continue':submit_continuations(a.config)
    else:worker(json.loads(Path(a.config).read_text()),a.lane)
