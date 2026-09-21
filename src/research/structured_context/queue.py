"""Shared extraction queue, eight matched fits, and allocation-aware continuations."""
import argparse
import copy
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
import torch
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import save_json,file_hash,digest
from src.training_methods.shared_pretraining.queue import snapshot,deadline_for_job
from src.research.crystallization_transfer.runtime import setup
from src.research.crystallization_paths.runtime import fit
from .extract import encoders,source
from .data import StructuredPaths


def freeze(config):
    root=resolve_path(config['output'])/'technical';root.mkdir(parents=True,exist_ok=True)
    if (root/'plan.json').exists():
        plan=json.loads((root/'plan.json').read_text())
        if plan['structured_config']!=config:raise ValueError('Structured campaign configuration changed')
        return plan
    oldpath=resolve_path(config['source_plan']);plan=json.loads(oldpath.read_text())
    parent=oldpath.parent/'parent.pt'
    if file_hash(parent)!=plan['checkpoint_sha256']:raise ValueError('Original MACE checkpoint changed')
    shutil.copy2(parent,root/'parent.pt')
    gatr=resolve_path(config['gatr_checkpoint'])
    if file_hash(gatr)!=config['gatr_checkpoint_sha256']:raise ValueError('Requested GATr checkpoint changed')
    shutil.copy2(gatr,root/'gatr.pt');saved=torch.load(gatr,map_location='cpu',weights_only=False)
    provenance=json.loads((resolve_path(config['gatr_static_analysis'])/'technical/encoder/provenance.json').read_text())
    if provenance['source_sha256']!=config['gatr_checkpoint_sha256'] or saved['step']!=3072:raise ValueError('Not the requested GATr analysis encoder')
    code=subprocess.check_output(['git','show',config['gatr_producer_commit']+':src/models/encoders/structural.py'])
    (root/'gatr-producer.py').write_bytes(code)
    if file_hash(root/'gatr-producer.py')!=saved['identity']['implementation']['files']['src/models/encoders/structural.py']:
        raise ValueError('Historical producer source hash differs')
    release=json.loads((resolve_path(saved['identity']['config']['release'])/'manifest.json').read_text())
    held={s['lineage'] for s in plan['sources'] if s.get('validation_role',s['split']) in ('test','calibration')}
    exposed={s['lineage'] for s in release['sources'] if s['split'] in ('train','selection') and 'lineage' in s}
    if held&exposed:raise ValueError(f'GATr pretraining overlap: {held&exposed}')
    if saved['scales']['Al']!=plan['scale']:raise ValueError('Material normalization differs')
    save_json(root/'encoder-provenance.json',dict(mace=plan['checkpoint_sha256'],gatr=config['gatr_checkpoint_sha256'],
        gatr_step=3072,gatr_producer_sha256=file_hash(root/'gatr-producer.py'),protected_overlap=[],
        local_radius_A=dict(mace=8*plan['scale']/9.192189,gatr=17*plan['scale']/9.192189)))
    plan['structured_config']=config;plan['structured_identity']=digest(dict(config=config,source_plan=file_hash(oldpath)))
    plan['config']=dict(plan['config'],output=config['output'],batch_size=config['batch_size'],seed=config['seed'],
        selection_per_source=config['selection_per_source'],selection_samples=config['selection_samples'],evaluation_samples=config['evaluation_samples'])
    plan['identity']=digest(plan);save_json(root/'plan.json',plan)
    tasks=[]
    for backbone in ('mace','gatr'):
        for reference in json.loads(resolve_path(config['reference_specs']).read_text()):
            spec=copy.deepcopy(reference)
            for key in ('initial_checkpoint','initial_checkpoint_sha256'):spec.pop(key)
            spec.update(context_layout='cuboctahedral_v1',encoder=backbone,shell_radii_A=config['shell_radii_A'],
                history_ps=config['history_ps'],information_context=config['information_context'],head_lr=config['head_lr'],depth=config['depth'])
            spec['training']['epochs']=config['epochs'];spec['name']=f'{backbone}-{spec["method"]}-symmetric-E{config["epochs"]}'
            tasks.append(spec)
    save_json(root/'queue.json',tasks);return plan


def report(config):
    root=resolve_path(config['output']);lines=['# Symmetric context: MACE and GATr','',
        '25 structured query slots; fixed source split; one seed; predictor heads trained from scratch with frozen backbones.',
        'GATr retains the requested original 16.87 A local support; MACE retains 7.94 A. Compare shared physical/event targets, not raw cross-encoder latent errors.','',
        '| Fit | State | Selected update | Brier | 12 ps AP | Physical MSE |','|---|---|---:|---:|---:|---:|']
    for spec in json.loads((root/'technical/queue.json').read_text()):
        p=root/'technical/runs'/spec['name'];status=json.loads((p/'status.json').read_text()) if (p/'status.json').exists() else {'state':'pending'}
        if (p/'metrics.json').exists():
            m=json.loads((p/'metrics.json').read_text());ap=m['short_horizon']['classification']['12.0']['average_precision']
            values=f'{m["training"]["selected_step"]} | {m["dense_integrated_brier"]:.5f} | {ap:.5f} | {m["path"]["standardized_mse_physical"]["all_times"]:.5f}'
        else:values=' | | | '
        lines.append(f'| {spec["name"]} | {status["state"]} | {values} |')
    lines+=['','All forecasts remain open-loop to 96 ps. Short-horizon timing, misses, calibration and sampled-center spatial scores are in each fit’s metrics.json. ',
        'The stencil is fixed in the simulation-box frame. Cubic rotations permute slots; arbitrary rotations require transporting the query frame. Real atom assignments are approximate and their exact offsets are inputs.']
    save=root/'RESULTS.md';temporary=save.with_suffix('.tmp-'+str(os.getpid()));temporary.write_text('\n'.join(lines)+'\n');temporary.replace(save)


def worker(config,lane):
    setup();plan=freeze(config);root=resolve_path(config['output'])/'technical';deadline=deadline_for_job()
    cache=resolve_path(config['context_cache']);cache.mkdir(parents=True,exist_ok=True)
    def state(value,**kwargs):save_json(root/f'lane-{lane}.json',dict(state=value,updated_at=time.time(),pid=os.getpid(),**kwargs))
    try:
        models=None
        while not all((cache/str(s['id'])/'complete.json').exists() for s in plan['sources']):
            claimed=False
            for item in plan['sources']:
                if time.time()>deadline-600:state('checkpointed',stage='extraction');return
                folder=cache/str(item['id']);folder.mkdir(exist_ok=True)
                with (folder/'worker.lock').open('a') as lock:
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                    if (folder/'complete.json').exists():continue
                    claimed=True
                    if models is None:models=encoders(plan)
                    state('extracting',source=item['id'])
                    if not source(plan,item,models,deadline,lambda f:state('extracting',source=item['id'],frames=f)):
                        state('checkpointed',stage='extraction',source=item['id']);return
            if not claimed:
                for p in root.glob('lane-*.json'):
                    if json.loads(p.read_text())['state']=='failed':raise RuntimeError(f'Extraction peer failed: {p}')
                state('waiting_for_features');time.sleep(20)
        del models;torch.cuda.empty_cache()
        tasks=json.loads((root/'queue.json').read_text());data=None;backbone=None
        while True:
            remaining=False;claimed=False
            for spec in tasks:
                folder=root/'runs'/spec['name'];folder.mkdir(parents=True,exist_ok=True)
                status=folder/'status.json'
                if status.exists():
                    previous=json.loads(status.read_text())
                    if previous['state']=='complete':continue
                    if previous['state']=='failed':raise RuntimeError(f'Previous fit failed: {spec["name"]}')
                remaining=True
                if time.time()>deadline-1800:state('checkpointed',stage='training');return
                with (folder/'worker.lock').open('a') as lock:
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                    # A peer may have completed between the first check and lock.
                    if status.exists() and json.loads(status.read_text())['state']=='complete':continue
                    claimed=True
                    if backbone!=spec['encoder']:
                        del data;torch.cuda.empty_cache();state('loading_timelines',encoder=spec['encoder'])
                        data=StructuredPaths(plan,spec);backbone=spec['encoder']
                    state('training',fit=spec['name']);save_json(folder/'spec.json',spec)
                    try:complete=fit(plan,spec,data,deadline)
                    except Exception as error:
                        save_json(status,dict(state='failed',error=repr(error),traceback=traceback.format_exc()));raise
                    report(config)
                    if not complete:state('checkpointed',fit=spec['name']);return
            if not remaining:state('complete');report(config);return
            if not claimed:state('waiting_for_fits');time.sleep(20)
    except Exception as error:
        state('failed',error=repr(error),traceback=traceback.format_exc());raise


def submit(config_path):
    config=json.loads(resolve_path(config_path).read_text());freeze(config);root=resolve_path(config['output'])/'technical'
    if (root/'launches.json').exists():raise FileExistsError('Already submitted; use worker for an exact continuation')
    validation=json.loads((root/'validation.json').read_text())
    if not validation['passed'] or len(validation['pipeline_checks'])!=2:raise ValueError('Structured-context preflight did not pass')
    code=snapshot(root);records=[]
    environment=dict(os.environ,PCM_PROJECT_ROOT=str(code),TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
    for lane in range(config['new_workers']+1):
        command=[sys.executable,'-u','-m','src.research.structured_context.queue','worker','--config',str(code/config_path),'--lane',str(lane)]
        if lane==0:
            job=config['allocation']
            if str(job)!=os.environ.get('SLURM_JOB_ID'):raise ValueError('Existing allocation must be the current allocation')
            launch=['srun',f'--jobid={job}','--overlap','--exact','-N1','-n1','-c6','--gres=gpu:1',*command]
            with (root/'current-h100.log').open('a') as log:
                proc=subprocess.Popen(launch,cwd=code,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            records.append(dict(lane=lane,job=str(job),pid=proc.pid,existing=True))
        else:
            script=root/f'lane-{lane}.sbatch'
            lines=['#!/bin/bash',f'#SBATCH --job-name=symmetric-context-{lane}',f'#SBATCH --partition={config["partition"]}',
                '#SBATCH --gres=gpu:1','#SBATCH --cpus-per-task=6','#SBATCH --mem=48G',f'#SBATCH --time={config["hours"]}:00:00',
                f'#SBATCH --output={root}/lane-{lane}-%j.log','set -euo pipefail','cd '+shlex.quote(str(code))]
            for k in ('PCM_PROJECT_ROOT','TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTORCH_ALLOC_CONF'):
                lines.append('export '+k+'='+shlex.quote(environment[k]))
            lines.append('exec '+shlex.join(command));script.write_text('\n'.join(lines)+'\n')
            job=subprocess.check_output(['sbatch','--parsable',str(script)],text=True).strip();records.append(dict(lane=lane,job=job,existing=False))
        save_json(root/'launches.json',records)
    report(config);print(json.dumps(records),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','verify','verify-pipeline','submit','worker','report']);p.add_argument('--config',required=True);p.add_argument('--lane',default='manual');a=p.parse_args()
    config=json.loads(resolve_path(a.config).read_text())
    if a.stage=='freeze':freeze(config)
    elif a.stage=='verify':
        from .verify import verify
        verify(freeze(config))
    elif a.stage=='verify-pipeline':
        from .verify import verify_pipeline
        verify_pipeline(freeze(config))
    elif a.stage=='submit':submit(a.config)
    elif a.stage=='worker':worker(config,a.lane)
    else:report(config)


if __name__=='__main__':main()
