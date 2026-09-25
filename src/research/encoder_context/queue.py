"""Frozen, resumable two-GPU encoder/context/evaluation campaign."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from src.project_runtime.paths import resolve_path
from src.research.structural_state.common import write_json,sha
from .common import Study


def context_study(study,method):
    from src.research.equivariant_context.common import Study as ContextStudy
    return ContextStudy(study.path.parent/method/'comparison.json')


def prepare(study):
    from src.research.equivariant_context.queue import prepare as prepare_context
    from src.research.supervised_onset.common import Study as BaseStudy
    study.bind()
    for method in study.config['methods']:
        context=context_study(study,method);prepare_context(context)
        for path in context.config['base_configs'].values():BaseStudy(resolve_path(path)).bind()
    record=dict(identity=study.identity,methods=study.config['methods'],domains=['hot','cold'],
        encoder_fits=8,structural_initializations=3,context_fits=16,
        source_roles=dict(train=90,selection=15,calibration=15,test=30),
        observation='current geometry; no velocity, history, temperature or explicit time inputs',
        inputs=dict(encoder='same native width-128/export-128 MACE on every 8 A patch; 80 candidates; 5 A edges; two blocks; no halo',
            predictors='25 patches at 0/10/20 A; scalar plus equivariant fields and relative geometry; no central bypass',
            structural_teacher='VICReg/Epi use paired same-time observed/relaxed geometry; Epi adds frozen random geometric reservoir',
            physical='24 radial counts, 2 weighted counts, 6 bond-order powers, from current geometry'),
        epochs=dict(structural=12,supervised=study.config['supervised_epochs'],predictors=study.config['predictor_epochs']),
        evaluation=dict(populations=['all64','legacy16'],lags_ps=[.75],relaxed_dense_available=False,
            metrics=['event NLL','AP3/AP6','raw/calibrated Brier and log loss','recall/FPR','linear/MLP probes',
                     'physical retention','dataset and movement spectra','normalized coordinate-noise response']),
        objective='NLL with validation-only checkpoint selection after twelve full epochs; AP diagnostic only')
    write_json(study.technical/'plan.json',record);return record


def check(study,device):
    import numpy as np
    import torch
    from torch import nn
    from src.research.supervised_onset.model import CapacityEncoder
    from src.models.encoders.spatial_mace import compile_spatial_encoder
    from src.data.fixed_cohort.dataset import StructuralDataset
    from src.research.mace_epi.objective import Objective
    from .geometry import graph,physical_targets
    study.bind();c=study.config;n=c['batch_size']
    dataset=StructuralDataset(c['fixed_dataset']['root'],'train',paired=True)
    ids=np.linspace(0,len(dataset)-1,n,dtype=int);pairs=[dataset[i] for i in ids]
    hot=torch.as_tensor(np.stack([v['inputs']['positions'] for v in pairs]),device=device)
    cold=torch.as_tensor(np.stack([v['teacher']['positions'] for v in pairs]),device=device)
    results=[]
    for method in ('physical','vicreg','epi_variance'):
        torch.manual_seed(c['seed']);model=CapacityEncoder(**c['encoder'],d0=2.8,n_ref=80.).to(device)
        if c['runtime']['compile']:compile_spatial_encoder(model,graph(hot,model))
        z=model(graph(hot,model))
        if method=='physical':
            head=nn.Linear(128,32).to(device);loss=(head(z)-physical_targets(hot)).square().mean()
        else:
            z2=model(graph(cold,model));objective=Objective('epi-variance' if method=='epi_variance' else 'vicreg',.1).to(device)
            loss,_=objective(None,torch.stack((z,z2),1).reshape(-1,128),dict(index=ids,reservoir=torch.randn(n,2,64,device=device)))
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError(f'Invalid structural preflight gradient: {method}')
        if model.center_embedding.weight.grad.norm()==0:raise ValueError('Encoder receives no structural gradient')
        results.append(dict(method=method,batch=n,loss=float(loss.detach()),finite_encoder_gradients=True))
        del model,z,loss
    # Existing production checks validate the actual focal geometry and both context heads.
    from src.research.equivariant_context.queue import check as context_check
    context_check(context_study(study,'scratch'),device)
    record=dict(passed=True,identity=study.identity,structural=results,wandb_runs=0)
    write_json(study.technical/'checks.json',record);return record


def worker(study,stage,method,domain,device):
    from src.training_methods.shared_pretraining.queue import deadline_for_job
    import torch
    study.bind();deadline=deadline_for_job();key='-'.join(x for x in (stage,method,domain) if x)
    with (study.technical/f'{key}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=study.technical/f'{key}-state.json'
        try:
            write_json(state,dict(state='running',identity=study.identity,stage=stage,method=method,domain=domain))
            if stage=='pretrain':
                from .pretrain import run
                run(study,method,device,deadline)
            elif stage=='dense':
                from .dense import prepare as prepare_dense
                prepare_dense(study)
            elif stage=='pipeline':
                from src.research.equivariant_context.queue import worker as context_worker
                from .evaluate import run as evaluate
                context=context_study(study,method);context.bind()
                context_worker(context,'features',domain,None,device)
                for variant in context.config['variants']:
                    context_worker(context,'predictor',domain,variant,device)
                evaluate(study,context,domain,device,deadline)
            else:raise ValueError(stage)
            write_json(state,dict(state='complete',identity=study.identity,stage=stage,method=method,domain=domain))
        except Exception as error:
            write_json(state,dict(state='checkpointed' if isinstance(error,TimeoutError) else 'failed',
                identity=study.identity,error=repr(error),traceback=traceback.format_exc()))
            raise


def lane(study):
    import torch,resource
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE);resource.setrlimit(resource.RLIMIT_NOFILE,(min(hard,65536),hard))
    if torch.cuda.device_count()!=1:raise ValueError('Require one Slurm-bound GPU per lane')
    task=int(os.environ['SLURM_PROCID']);study.bind();done=[]
    binding=dict(task=task,job=os.environ['SLURM_JOB_ID'],step=os.environ['SLURM_STEP_ID'],
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
    jobs=[]
    if task==0:jobs.append(dict(stage='dense'))
    for method in study.config['lanes'][task]:
        if method!='scratch':jobs.append(dict(stage='pretrain',method=method))
        for domain in ('hot','cold'):jobs.append(dict(stage='pipeline',method=method,domain=domain))
    for job in jobs:
        name='-'.join(job.values());state=study.technical/f'{name}-state.json'
        if state.exists() and json.loads(state.read_text()).get('state')=='complete':done.append(name);continue
        write_json(study.technical/f'lane-{task}.json',dict(binding,state='running',active=name,completed=done))
        # Both lanes may need the same deterministic dense evaluation dataset.
        if job['stage']=='pipeline' and job['domain']=='hot':
            while not (study.cache/'dense-observed/manifest.json').exists():
                dense=study.technical/'dense-state.json'
                if dense.exists() and json.loads(dense.read_text()).get('state')=='failed':raise RuntimeError('Dense input preparation failed')
                from src.training_methods.shared_pretraining.queue import deadline_for_job
                if time.time()>deadline_for_job()-600:raise TimeoutError('Waiting for dense evaluation dataset')
                time.sleep(30)
        command=[sys.executable,'-u','-m','src.research.encoder_context.queue','worker','--config',str(study.path)]
        for key,value in job.items():command += ['--'+key,value]
        with (study.technical/f'{name}.log').open('ab',buffering=0) as log:
            process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if process.returncode:
            write_json(study.technical/f'lane-{task}.json',dict(binding,state='failed',active=name,completed=done))
            raise RuntimeError(f'Failed stage {name}; see its saved error and log')
        done.append(name)
    write_json(study.technical/f'lane-{task}.json',dict(binding,state='complete',completed=done))
    # The last finished lane collects only after both independent pipelines finish.
    with (study.technical/'report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if all((study.technical/f'lane-{i}.json').exists() and
               json.loads((study.technical/f'lane-{i}.json').read_text()).get('state')=='complete' for i in (0,1)):
            from .report import collect
            collect(study)


def launch(study):
    from src.training_methods.shared_pretraining.queue import snapshot
    study.bind();checked=json.loads((study.technical/'checks.json').read_text())
    if not checked['passed'] or checked['identity']!=study.identity:raise ValueError('Preflight differs from launch')
    if (study.technical/'launch.json').exists():raise FileExistsError('Already launched; inspect resume state')
    c=study.config['slurm'];job=json.loads(subprocess.check_output(['scontrol','show','job',c['allocation'],'--json'],text=True))['jobs'][0]
    if job['nodes']!=c['node'] or os.uname().nodename!=c['node']:raise ValueError('Wrong allocated node')
    seconds=int(job['end_time']['number']-time.time()-300)
    if seconds<3600:raise ValueError('Less than one hour remains')
    code=snapshot(study.technical)
    config=code/study.path.relative_to(Path.cwd().resolve())
    command=['srun','--jobid='+c['allocation'],'--overlap','--exact','--nodes=1','--ntasks=2','--cpus-per-task=8',
        '--gpus-per-task=1','--gpu-bind=single:1','--mem=64G','--kill-on-bad-exit=0','--time='+str(seconds//60),
        '--job-name=encoder-context-epochs',sys.executable,'-u','-m','src.research.encoder_context.queue','lane','--config',str(config)]
    env=dict(os.environ,PCM_PROJECT_ROOT=str(code),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',PYTORCH_ALLOC_CONF='expandable_segments:True',TORCHINDUCTOR_COMPILE_THREADS='4')
    with (study.technical/'allocation.log').open('ab',buffering=0) as log:
        process=subprocess.Popen(command,cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(identity=study.identity,pid=process.pid,command=command,code=str(code),config=str(config),
        allocation=c['allocation'],gpus=2,detached=True)
    write_json(study.technical/'launch.json',record);return record


def main():
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=['prepare','check','launch','lane','worker','report'])
    p.add_argument('--config',required=True);p.add_argument('--stage');p.add_argument('--method');p.add_argument('--domain')
    p.add_argument('--device',default='cuda');args=p.parse_args();study=Study(args.config)
    if args.action=='prepare':result=prepare(study)
    elif args.action=='check':result=check(study,args.device)
    elif args.action=='launch':result=launch(study)
    elif args.action=='lane':result=lane(study)
    elif args.action=='worker':result=worker(study,args.stage,args.method,args.domain,args.device)
    else:
        from .report import collect
        study.bind();result=collect(study)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
