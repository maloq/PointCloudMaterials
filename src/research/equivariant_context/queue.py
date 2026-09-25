"""Prepare/check without training; explicit submission of dependency-ordered jobs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
from src.project_runtime.paths import resolve_path
from src.research.structural_state.common import write_json, sha
from .common import Study


def stages(config):
    result=[]
    for domain in config['domains']:
        result.append(dict(name=f'prepare-{domain}',stage='features',domain=domain,variant=None,
            dependencies=[],walltime=config['slurm']['feature_walltime']))
        for variant in config['variants']:
            result.append(dict(name=f'{domain}-{variant}',stage='predictor',domain=domain,variant=variant,
                dependencies=[f'prepare-{domain}'],walltime=config['slurm']['predictor_walltime']))
    return result


def prepare(study):
    from .data import inventory
    from .model import ContextPredictor, context_fields
    from src.research.supervised_onset.common import Study as BaseStudy
    study.bind()
    for domain,path in study.config['base_configs'].items():
        base=BaseStudy(resolve_path(path));c=base.config
        if (resolve_path(c['cache'])!=resolve_path(study.config['population_cache'])
            or c['seed']!=study.config['seed'] or len(c['arms'])!=1 or c['arms'][0]['input']!=domain
            or c['encoder']['channels']!=128 or c['encoder']['code_dim']!=128
            or c.get('fixed_dataset')!=study.config.get('fixed_dataset')
            or any(c['training'][k]!=study.config[k] for k in ('batch_size','microbatch'))):
            raise ValueError(f'Base encoder is not matched to the declared context cohort: {path}')
    available=inventory(study.config)
    write_json(study.technical/'inventory.json',available)
    counts={v:sum(p.numel() for p in ContextPredictor(v,**study.config['predictor']).parameters())
            for v in study.config['variants']}
    record=dict(state='prepared_not_submitted',identity=study.identity,stages=stages(study.config),
        inventory_sha256=sha(study.technical/'inventory.json'),
        predictor_parameters=counts,shared_encoder_parameters=634496,rows=available['rows'],
        extraction_runtime=study.config['extraction'],
        sources=available['source_count'],frames=available['frames'],new_simulations=0,
        prediction_context=dict(encoder='same 128-channel/128-export MACE for all patches within each domain',
            patches=25,stencil_radii_A=[10,20],query_max_offset_A=4,patch_radius_A=8,
            maximum_patch_atoms=80,edge_cutoff_A=5,message_passing_layers=2,halo=False,
            maximum_geometric_reach_A=32,input_frames=1,velocities=False,external_inputs=[],
            relaxed_input='same-time converged full-cell quench, no future information',
            fields='MACE l=1,2 center and smooth pool; hierarchy additionally current-coordinate l=4,6 bonds',
            predictor_fields={v:list(context_fields(v))+['nominal'] for v in study.config['variants']},
            atom_padding='disconnected zero-weight nodes; physical edges and normalization unchanged',
            aggregation='shared patch focal-event head then learned mixture of six event/survival probabilities',
            origin='focal target specified by relative geometry; no index embedding or focal residual',
            selection='source-weighted predictive likelihood; AP3/AP6 reporting only'))
    write_json(study.technical/'plan.json',record)
    return record


def check(study,device):
    """Current hot/cold real-data inference and gradient checks; never fit."""
    import numpy as np
    import torch
    from .data import population,parent_plan,paired_frame,patch_features
    from .model import ContextPredictor
    from src.research.supervised_onset.model import CapacityEncoder
    study.bind();pop=population(study.config);parent=parent_plan(study.config)
    sid=int(pop['source'][0]);frame=int(pop['frame'][0]);source=next(s for s in parent['sources'] if s['id']==sid)
    rows=np.flatnonzero((pop['source']==sid)&(pop['frame']==frame))[:2]
    views,inverse,atoms=paired_frame(source,frame,pop['atom'][rows],parent)
    if 'fixed_dataset' in study.config:
        for domain in study.config['domains']:
            folder=resolve_path(study.config['population_cache'])
            positions=np.load(folder/f'{domain}-positions.npy',mmap_mode='r')
            offsets=np.load(folder/f'{domain}-offsets.npy',mmap_mode='r')
            for i,row in enumerate(rows):
                np.testing.assert_array_equal(views[domain]['patches'][inverse[i,0]],
                    positions[offsets[row]:offsets[row+1]],
                    err_msg=f'Fixed-cohort/context focal geometry differs: {domain}, row {row}')
    base=json.loads(resolve_path(study.config['base_configs']['hot']).read_text())
    manifest=json.loads((resolve_path(study.config['population_cache'])/'manifest.json').read_text())
    ec=base['encoder']|{k:manifest['domains']['hot'][k] for k in ('d0','n_ref')}
    if device=='cpu':ec.update(backend='e3nn',layout='mul_ir',conv_fusion=False)
    encoder=CapacityEncoder(**ec).to(device).eval();results=[]
    from .data import stencil
    for domain in study.config['domains']:
        values=patch_features(encoder,views[domain]['patches'],device)
        batch={k:torch.as_tensor(v[inverse],device=device) for k,v in values.items()}
        batch.update(actual=torch.as_tensor(views[domain]['actual'],device=device),
            nominal=torch.as_tensor(stencil(),device=device)[None].expand(len(rows),-1,-1))
        for variant in study.config['variants']:
            model=ContextPredictor(variant,**study.config['predictor']).to(device)
            # Full configured batch via repeated observations checks shapes and
            # memory; there is no optimizer and no parameter update.
            repeated={k:v.repeat((study.config['batch_size']//len(rows),)+(1,)*(v.ndim-1)) for k,v in batch.items()}
            out=model(repeated);loss=-out[:,5].mean();loss.backward()
            if not torch.isfinite(out).all() or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError(f'Nonfinite or missing predictor gradients: {domain}/{variant}')
            torch.testing.assert_close(out.exp().sum(-1),torch.ones(len(out),device=device),atol=1e-5,rtol=1e-5)
            results.append(dict(domain=domain,variant=variant,checked_batch=len(out),finite_gradients=True))
            del model,out,loss,repeated
    receipt=dict(passed=True,identity=study.identity,purpose='random-weight correctness checks, no training',
        source=sid,frame=frame,rows=rows.tolist(),query_atom_ids=atoms.tolist(),checks=results,wandb_runs=0)
    write_json(study.technical/'checks.json',receipt)
    return receipt


def script(study,job,code,dependency_ids):
    cfg=study.config['slurm'];q=shlex.quote
    config=code/study.path.relative_to(Path.cwd().resolve())
    command=[sys.executable,'-u','-m','src.research.equivariant_context.queue','worker','--config',str(config),
        '--stage',job['stage'],'--domain',job['domain']]
    if job['variant'] is not None:command+=['--variant',job['variant']]
    lines=['#!/bin/bash',f'#SBATCH --job-name=eqctx-{job["name"]}',
        f'#SBATCH --partition={cfg["partitions"]}','#SBATCH --gres=gpu:1',
        f'#SBATCH --cpus-per-task={cfg["cpus"]}',f'#SBATCH --mem={cfg["memory"]}',
        f'#SBATCH --time={job["walltime"]}',f'#SBATCH --output={study.technical}/logs/{job["name"]}-%j.log',
        '#SBATCH --kill-on-invalid-dep=yes']
    if dependency_ids:lines.append('#SBATCH --dependency=afterok:'+':'.join(dependency_ids))
    lines.append(f'#SBATCH --nodelist={cfg["node"]}')
    lines+=['set -euo pipefail',f'cd {q(str(code))}',f'export PCM_PROJECT_ROOT={q(str(code))}',
        'export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
        'export PYTORCH_ALLOC_CONF=expandable_segments:True','ulimit -n 65536',shlex.join(command)]
    return '\n'.join(lines)+'\n'


def submit(study):
    from src.training_methods.shared_pretraining.queue import snapshot
    from src.experiment_runner.slurm import submit_sbatch
    study.bind();receipt=json.loads((study.technical/'checks.json').read_text())
    if not receipt['passed'] or receipt['identity']!=study.identity:raise ValueError('Checks do not match current source/config/data')
    plan=json.loads((study.technical/'plan.json').read_text())
    if plan['identity']!=study.identity or plan['inventory_sha256']!=sha(study.technical/'inventory.json'):
        raise ValueError('Prepared data inventory differs from checked study')
    path=study.technical/'submissions.json'
    if path.exists():raise FileExistsError('Already submitted, including partial submission; inspect recorded jobs')
    code=snapshot(study.technical)
    jobs={};write_json(path,dict(identity=study.identity,state='submitting',jobs=jobs))
    (study.technical/'logs').mkdir(exist_ok=True)
    for job in stages(study.config):
        dependencies=[jobs[n] for n in job['dependencies']]
        jobs[job['name']]=submit_sbatch(script(study,job,code,dependencies),study.technical/'slurm'/f'{job["name"]}.sh')
        write_json(path,dict(identity=study.identity,state='submitting',jobs=jobs))
    write_json(path,dict(identity=study.identity,state='submitted',jobs=jobs))
    return jobs


def worker(study,stage,domain,variant,device):
    import torch
    from src.training_methods.shared_pretraining.queue import deadline_for_job
    study.bind();deadline=deadline_for_job()
    name=f'prepare-{domain}' if stage=='features' else f'{domain}-{variant}'
    with (study.technical/f'{name}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=study.technical/f'{name}-state.json'
        try:
            write_json(path,dict(state='running',job=os.environ['SLURM_JOB_ID']))
            if stage=='features':
                from src.research.supervised_onset.common import Study as BaseStudy
                from src.research.supervised_onset.data import prepare as prepare_base,Corpus
                from src.research.supervised_onset.train import fit,make_banks
                from .data import extract
                base=BaseStudy(resolve_path(study.config['base_configs'][domain]));prepare_base(base);base.bind()
                arm=base.config['arms'][0]['name'];run=base.technical/'runs'/arm
                state=run/'training-state.json'
                done=json.loads(state.read_text()) if state.exists() else None
                if done is None or done['updates']<study.config['encoder_updates']:
                    corpus=Corpus(base);banks=make_banks(base,corpus,device)
                    done=fit(base,corpus,banks,arm,until=deadline-600,max_updates=study.config['encoder_updates'],device=device)
                    del banks,corpus;torch.cuda.empty_cache()
                if done['updates']!=study.config['encoder_updates']:
                    raise TimeoutError('Encoder fixed update budget incomplete; resume features job')
                extract(study,domain,run/'best.pt',device,deadline)
            else:
                from .train import fit
                fit(study,domain,variant,device,deadline)
            write_json(path,dict(state='complete',identity=study.identity,job=os.environ['SLURM_JOB_ID']))
        except Exception as error:
            write_json(path,dict(state='checkpointed' if isinstance(error,TimeoutError) else 'failed',
                error=repr(error),traceback=traceback.format_exc()))
            raise


def launch(study):
    """Two allocated Slurm tasks, with one distinct GPU per sequential run."""
    from src.training_methods.shared_pretraining.queue import snapshot
    from src.research.supervised_onset.common import Study as BaseStudy
    study.bind();cfg=study.config['slurm']
    checked=json.loads((study.technical/'checks.json').read_text())
    plan=json.loads((study.technical/'plan.json').read_text())
    if not checked['passed'] or checked['identity']!=study.identity or plan['identity']!=study.identity:
        raise ValueError('Current source/config does not match prepared and checked study')
    for path in study.config['base_configs'].values():
        base=BaseStudy(resolve_path(path));base.bind()
        check=json.loads((base.technical/'preflight.json').read_text())
        if not check['passed'] or check['identity']!=base.identity:
            raise ValueError(f'Encoder preflight mismatch: {base.path}')
    job=json.loads(subprocess.check_output(['scontrol','show','job',cfg['allocation'],'--json'],text=True))['jobs'][0]
    if job['nodes']!=cfg['node'] or os.uname().nodename!=cfg['node']:
        raise ValueError('Launch must use the declared current node/allocation')
    end=job['end_time']['number'];seconds=int(end-time.time()-300)
    if seconds<3600:raise ValueError('Less than one hour remains for the queue')
    receipt=study.technical/'launch.json'
    if receipt.exists():raise FileExistsError('Already launched; inspect launch and lane-state receipts')
    code=snapshot(study.technical)
    config=code/study.path.relative_to(Path.cwd().resolve())
    command=['srun','--jobid='+cfg['allocation'],'--overlap','--exact','--nodes=1','--ntasks=2',
        '--cpus-per-task='+str(cfg['cpus']),'--gpus-per-task=1','--gpu-bind=single:1','--mem=64G',
        '--kill-on-bad-exit=0','--time='+str(seconds//60),'--job-name=eqctx-b512','--unbuffered',
        sys.executable,'-u','-m','src.research.equivariant_context.queue','lane','--config',str(config)]
    env=dict(os.environ,PCM_PROJECT_ROOT=str(code),TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
    with (study.technical/'allocation.log').open('ab',buffering=0) as log:
        process=subprocess.Popen(command,cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(state='launched',identity=study.identity,allocation=cfg['allocation'],node=cfg['node'],
        gpus_per_run=1,tasks=2,batch_size=study.config['batch_size'],microbatch=study.config['microbatch'],
        launcher_pid=process.pid,command=command,code=str(code),config=str(config))
    write_json(receipt,record);return record


def lane(study):
    """One shared encoder preparation and four independent predictor fits/GPU."""
    import torch
    import resource
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE,(min(65536,hard),hard))
    task=int(os.environ['SLURM_PROCID']);domain=study.config['domains'][task]
    if torch.cuda.device_count()!=1:raise ValueError('Each task must see exactly one Slurm-bound GPU')
    study.bind();state=study.technical/f'lane-{domain}.json'
    binding=dict(domain=domain,task=task,step=os.environ['SLURM_STEP_ID'],job=os.environ['SLURM_JOB_ID'],
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),visible=os.environ['CUDA_VISIBLE_DEVICES'])
    jobs=[j for j in stages(study.config) if j['domain']==domain]
    completed=[]
    try:
        for job in jobs:
            write_json(state,dict(binding,state='running',stage=job['name'],completed=completed))
            command=[sys.executable,'-u','-m','src.research.equivariant_context.queue','worker',
                '--config',str(study.path),'--stage',job['stage'],'--domain',domain]
            if job['variant'] is not None:command+=['--variant',job['variant']]
            with (study.technical/f'{job["name"]}.log').open('ab',buffering=0) as log:
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
            completed.append(job['name'])
        write_json(state,dict(binding,state='complete',completed=completed))
    except Exception as error:
        write_json(state,dict(binding,state='failed',stage=job['name'],completed=completed,error=repr(error)))
        raise
    return binding


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('action',choices=('prepare','check','submit','launch','lane','worker','collect','sync-tracking'))
    parser.add_argument('--config',required=True);parser.add_argument('--device',default='cuda')
    parser.add_argument('--stage',choices=('features','predictor'));parser.add_argument('--domain',choices=('hot','cold'))
    parser.add_argument('--variant');args=parser.parse_args()
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    study=Study(args.config)
    if args.action=='prepare':result=prepare(study)
    elif args.action=='check':result=check(study,args.device)
    elif args.action=='submit':result=submit(study)
    elif args.action=='launch':result=launch(study)
    elif args.action=='lane':result=lane(study)
    elif args.action=='sync-tracking':
        from .tracking import sync_completed
        result=sync_completed(study)
    elif args.action=='worker':
        if args.stage is None or args.domain is None or (args.stage=='predictor' and args.variant not in study.config['variants']):
            parser.error('worker requires a stage/domain and a declared predictor variant')
        result=worker(study,args.stage,args.domain,args.variant,args.device)
    else:
        from .train import collect
        study.bind();result=collect(study)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
