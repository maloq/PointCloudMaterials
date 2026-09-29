"""Resume completed mechanism encoders with process-isolated evaluations."""
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
from types import SimpleNamespace

from src.project_runtime.paths import resolve_path
from src.research.structural_state.common import sha, write_json
from .workflow import load, bind, quality_config, evaluate_checkpoint, evaluate_checkpoint_isolated, adapt


def checkpoint(request):
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    r=json.loads(Path(request).read_text())
    c=r['config'];output=Path(r['output']);path=Path(r['path'])
    if r['displacement_only']:
        from .rearrangement import evaluate
        q=quality_config(c,output)
        spec=dict(name=r['name'],checkpoint=str(path),checkpoint_sha256=sha(path),kind=r['kind'],domain='hot',
            producer=str(Path(__file__).resolve().parents[3]),seed=r['seed'])
        write_json(output/'technical/displacement-spec.json',spec)
        evaluate(c,q,spec,output,r['device'])
    else:
        evaluate_checkpoint(c,path,r['kind'],r['name'],output,r['device'],r['seed'])


def source_study(c,seed):
    return SimpleNamespace(root=resolve_path(c['recovery_source'])/'runs'/str(seed)/'R1')


def completed_assays(roots,relative,name):
    quality=displacement=None
    for root in roots:
        folder=resolve_path(root)/relative
        done=folder/'technical/evaluations'/name/'complete.json'
        if done.exists():
            if sha(done.with_name('metrics.json'))!=json.loads(done.read_text())['metrics_sha256']:
                raise ValueError(f'Completed metrics changed: {done}')
            quality=done
        path=folder/'analyses/displacement/technical/metrics.json'
        if path.exists():displacement=path
    return quality,displacement


def worker(config,stage,index):
    from src.training_methods.shared_pretraining.queue import deadline_for_job
    c=load(config);root=resolve_path(c['output']);old=resolve_path(c['recovery_source'])
    state=root/f'technical/{stage}-{index}.json'
    with state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            bind(c,root)
            write_json(state,dict(state='running',job=os.environ['SLURM_JOB_ID'],stage=stage,index=index))
            if stage=='alignment':
                seed=c['seeds'][index//3];arm=c['arms'][index%3]
                for epoch in (24,12,18,8,4,0):
                    if time.time()>deadline_for_job()-600:raise TimeoutError('Checkpoint evaluation awaits another allocation')
                    name=f'{arm["name"]}-epoch{epoch:03d}'
                    relative=Path(f'analyses/alignment/{seed}/{arm["name"]}/epoch-{epoch:03d}')
                    dest=root/relative
                    saved,displacement=completed_assays(c['recovery_results'],relative,name)
                    if saved is not None and displacement is not None:
                        continue
                    if (dest/'technical/recovery-complete.json').exists():continue
                    path=old/f'runs/{seed}/{arm["name"]}/pretraining/{arm["method"]}/technical/epoch-{epoch:03d}.pt'
                    if sha(path)!=json.loads(path.with_suffix('.json').read_text())['sha256']:
                        raise ValueError(f'Checkpoint changed: {path}')
                    evaluate_checkpoint_isolated(c,path,'initial' if epoch==0 else 'pretrained',name,dest,'cuda',seed,
                        displacement_only=saved is not None)
                    write_json(dest/'technical/recovery-complete.json',dict(state='complete',checkpoint_sha256=sha(path),
                        reused_quality=str(saved.parent) if saved is not None else None))
            elif stage=='adaptation':
                for mode in ('frozen','finetune','scratch'):
                    subprocess.run([sys.executable,'-u','-m',__package__+'.recovery','adapt-mode',
                        '--config',str(config),'--index',str(index),'--mode',mode],check=True)
            else:raise ValueError(stage)
            write_json(state,dict(state='complete',stage=stage,index=index,finished_at=time.time()))
        except BaseException as error:
            write_json(state,dict(state='failed',error=repr(error),traceback=traceback.format_exc()))
            raise


def prepare(config,output,*,previous_recovery=None):
    """Validate saved endpoints, reuse assays and audit actual MD spacing."""
    import numpy as np
    from src.data.fixed_cohort.dataset import read_release
    from src.project_runtime.paths import dataset_path
    c=load(config);old=resolve_path(c['output']);root=resolve_path(output)
    if root.exists():raise FileExistsError(root)
    results=[str(old)]
    if previous_recovery is not None:
        previous=resolve_path(previous_recovery)
        prior=load(previous/'technical/config.json')
        scientific={k:v for k,v in prior.items() if k not in ('output','recovery_source','recovery_results')}
        if scientific!={k:v for k,v in c.items() if k!='output'} or resolve_path(prior['recovery_source'])!=old:
            raise ValueError(f'Previous recovery belongs to a different experiment: {previous}')
        results.append(str(previous))
    records=[];quality=displacement=0
    for seed in c['seeds']:
        for arm in c['arms']:
            folder=old/f'runs/{seed}/{arm["name"]}/pretraining/{arm["method"]}/technical'
            complete=json.loads((folder/'complete.json').read_text())
            if complete['epochs']!=24:raise ValueError(f'Incomplete encoder: {folder}')
            for epoch in c['pretraining']['checkpoint_epochs']:
                path=folder/f'epoch-{epoch:03d}.pt';checksum=sha(path)
                if checksum!=json.loads(path.with_suffix('.json').read_text())['sha256']:raise ValueError(path)
                relative=Path(f'analyses/alignment/{seed}/{arm["name"]}/epoch-{epoch:03d}')
                done,displacement_done=completed_assays(results,relative,f'{arm["name"]}-epoch{epoch:03d}')
                if done is not None:
                    metric=json.loads(done.with_name('metrics.json').read_text())
                    if metric['model']['checkpoint_sha256']!=checksum:raise ValueError(f'Wrong evaluated checkpoint: {done}')
                    quality+=1
                has_displacement=displacement_done is not None
                displacement+=has_displacement
                records.append(dict(seed=seed,arm=arm['name'],epoch=epoch,checkpoint=str(path),sha256=checksum,
                    quality_complete=done is not None,displacement_complete=has_displacement,
                    reused_quality=str(done) if done is not None else None,
                    reused_displacement=str(displacement_done) if has_displacement else None))
    # Keep the trained architecture, inference producer and experiment templates exact.
    repo=Path(__file__).resolve().parents[3];frozen=old/'technical/training-revision-v2/code'
    same=['src/models/encoders/spatial_mace.py','src/models/encoders/mace_backend.py',
        'src/research/supervised_onset/model.py','src/research/encoder_context/geometry.py',
        c['quality_template'],c['supervised_template']]
    for name in same:
        if sha(repo/name)!=sha(frozen/name):raise ValueError(f'Recovery changes inference or scientific template: {name}')
    _,release=read_release(c['fixed_dataset']['root']);timeline=[]
    pairs=old/'technical/rearrangement-inputs'
    for row in json.loads((pairs/'complete.json').read_text())['sources']:
        source=next(s for s in release['sources'] if s['id']==row['source'])
        path=dataset_path(source['dataset'])/source['relative_trajectory_path']
        if sha(path/'manifest.json')!=row['manifest_sha256']:raise ValueError(f'Manifest changed: {path}')
        steps=np.load(path/'timesteps.npy',mmap_mode='r');frames=np.asarray(row['frames'])
        intervals=(steps[frames+1]-steps[frames])*source['timestep_fs']/1000
        np.testing.assert_allclose(intervals,.75,atol=1e-12,rtol=0)
        timeline.append(dict(source=source['id'],frames=row['frames'],intervals_ps=intervals.tolist(),
            timestep_fs=source['timestep_fs'],timesteps_sha256=sha(path/'timesteps.npy')))
    c.update(output=str(root),recovery_source=str(old),recovery_results=results)
    root.mkdir(parents=True)
    shutil.copytree(pairs,root/'technical/rearrangement-inputs')
    write_json(root/'technical/timeline-audit.json',dict(sources=timeline,actual_lag_ps=.75))
    write_json(root/'technical/recovery-inputs.json',dict(checkpoints=records,completed_quality=quality,
        completed_displacement=displacement,encoder_fits_complete=9,adaptation_fits_complete=0,
        unchanged_producers={name:sha(repo/name) for name in same},source=str(old)))
    write_json(root/'technical/config.json',c)
    return c


def submit(config,output,*,exclude_nodes=None,previous_recovery=None):
    from src.training_methods.shared_pretraining.queue import snapshot
    from src.experiment_runner.metric_docs import check_metric_docs
    for family in ('encoder_mechanisms','encoder_quality','supervised_onset'):
        check_metric_docs(family=family)
    c=prepare(config,output,previous_recovery=previous_recovery);root=resolve_path(c['output']);technical=root/'technical'
    code=snapshot(technical);bind(c,root);records=[]
    for stage,array,hours,dependency in [('alignment','0-8%2',8,None),('adaptation','0-2%2',24,'alignment')]:
        script=technical/f'{stage}.sbatch'
        lines=['#!/bin/bash',f'#SBATCH --job-name=mechanisms-resume-{stage}',
            '#SBATCH --partition=RTX6000PRO','#SBATCH --gres=gpu:1',
            '#SBATCH --cpus-per-task=4','#SBATCH --mem=32G',f'#SBATCH --time={hours}:00:00',
            f'#SBATCH --array={array}',f'#SBATCH --chdir={code}',f'#SBATCH --output={technical}/{stage}-%A_%a.log']
        if exclude_nodes:lines.append('#SBATCH --exclude='+exclude_nodes)
        if dependency:lines.append('#SBATCH --dependency=afterok:'+records[0]['job'])
        env=['PCM_PROJECT_ROOT='+str(code),'OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',
            'TORCHINDUCTOR_COMPILE_THREADS=4','PYTORCH_ALLOC_CONF=expandable_segments:True','TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1']
        command=[sys.executable,'-u','-m',__package__+'.recovery','worker','--config',str(technical/'config.json'),'--stage',stage]
        script.write_text('\n'.join(lines+['set -euo pipefail','exec env '+shlex.join(env)+' '+shlex.join(command),'']))
        job=subprocess.check_output(['sbatch','--parsable',str(script)],text=True).strip()
        records.append(dict(stage=stage,job=job,array=array,script=str(script),exclude_nodes=exclude_nodes))
        write_json(technical/'launch.json',dict(state='submitting',code=str(code),jobs=records))
    receipt=dict(state='submitted',code=str(code),jobs=records,source=c['recovery_source'],submitted_at=time.time())
    write_json(technical/'launch.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('action',choices=['submit','worker','checkpoint','adapt-mode'])
    p.add_argument('--config');p.add_argument('--output');p.add_argument('--request');p.add_argument('--exclude-nodes')
    p.add_argument('--previous-recovery')
    p.add_argument('--stage',choices=['alignment','adaptation']);p.add_argument('--index',type=int)
    p.add_argument('--mode',choices=['frozen','finetune','scratch']);a=p.parse_args()
    if a.action=='checkpoint':checkpoint(a.request)
    elif a.action=='submit':print(json.dumps(submit(a.config,a.output,exclude_nodes=a.exclude_nodes,
        previous_recovery=a.previous_recovery),indent=2))
    elif a.action=='worker':worker(a.config,a.stage,a.index if a.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID']))
    else:
        import torch
        from src.training_methods.shared_pretraining.queue import deadline_for_job
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        c=load(a.config);seed=c['seeds'][a.index]
        adapt(c,seed,source_study(c,seed),'cuda',deadline_for_job(),mode=a.mode)


if __name__=='__main__':main()
