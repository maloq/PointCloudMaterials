"""Two-GPU, three-hour scale runs and their frozen crystallization follow-up.

The explicit preflight is a separate phase. It freezes an update count before
scientific training starts; the trainer never changes its budget or LR schedule.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import save_json,file_hash,digest
from src.data.structural_pretraining.batches import move
from src.training_methods.shared_pretraining.compilation import compile_encoder
from src.training_methods.shared_pretraining.queue import snapshot,deadline_for_job
from .contracts import variants
from .data import Data,Batches,loader
from .model import Model
from .objective import Objective
from .parallel import ParallelEncoder,configure_host


def specification(config):
    spec=variants(config)[-1]
    spec['name']=config['name']
    return spec


def profile(config):
    """Fresh disposable weights; no measured updates become scientific training."""
    configure_host()
    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    spec=specification(config)
    data=Data(resolve_path(config['cache']),spec)
    model=Model(config['encoder_channels']).cuda()
    model.encoder.geometry_scales.copy_(torch.tensor(data.manifest['geometry_scales'],device='cuda'))
    objective=Objective(data.manifest,spec).cuda()
    sampler=Batches(data,config['batch_size'],config['seed'],0,10)
    stream=iter(loader(data,sampler,config['microbatch'],config['loader_workers']))
    packed,target=next(stream)
    if config['compile']:compile_encoder(model.encoder,move(packed[0],'cuda'),config['precision'])
    parallel=ParallelEncoder(model,packed[0],config['precision'],config['compile'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0002)
    timings=[]
    for step in range(10):
        started=time.monotonic()
        if step:
            packed,target=next(stream)
        optimizer.zero_grad(set_to_none=True)
        parallel.step(model,objective,packed,move(target,'cuda'))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        parallel.synchronize()
        if step>=4:timings.append(time.monotonic()-started)
    # Include actual single-device development evaluation overhead in the estimate.
    from .runtime import evaluate,fixed_baselines
    tick=time.monotonic()
    evaluate(model,objective,data,config,fixed_baselines(data))
    validation_seconds=time.monotonic()-tick
    step_seconds=float(np.median(timings))
    effective=step_seconds+validation_seconds/config['evaluate_every']
    epoch_updates=math.ceil(data.train_size/config['batch_size'])
    epochs=math.floor((config['target_training_seconds']-300)/(effective*epoch_updates))
    if epochs<config['minimum_epochs']:
        raise RuntimeError(f'Three-hour budget yields only {epochs} epochs; minimum {config["minimum_epochs"]}. '
                           f'Observed {step_seconds:.2f}s/update. Inspect and size a new run explicitly.')
    updates=epochs*epoch_updates
    resolved=dict(config,updates=updates,epochs=epochs,train_anchors=data.train_size,
        epoch_definition='sampled anchor draws / training anchor population; batches sampled without replacement within update')
    root=resolve_path(config['output'])/'technical'
    measurement=dict(step_seconds=step_seconds,validation_seconds=validation_seconds,timed_updates=timings,
        parameter_count=sum(p.numel() for p in model.encoder.parameters()),
        model_parameters=sum(p.numel() for p in model.parameters()),
        gpu=[torch.cuda.get_device_name(i) for i in range(2)],
        peak_allocated_GiB=[torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)],
        predicted_training_seconds=updates*effective,updates=updates,epochs=epochs,
        data_identity=data.manifest['identity'],config_sha256=digest(config))
    save_json(root/'preflight.json',measurement)
    path=root/'resolved-config.json'
    if path.exists():raise FileExistsError('Scientific update budget is already frozen')
    save_json(path,resolved)
    parallel.close()
    print(json.dumps(measurement),flush=True)


def fit(config):
    from .runtime import run
    root=resolve_path(config['output'])/'technical'
    allocation_deadline=deadline_for_job()
    budget_deadline=min(allocation_deadline,time.time()+config['target_training_seconds']+300)
    save_json(root/'execution.json',dict(state='training',started_at=time.time(),deadline=budget_deadline,
        allocated_job=os.environ['SLURM_JOB_ID'],host=os.uname().nodename))
    complete=run(config,specification(config),budget_deadline)
    save_json(root/'execution.json',dict(state='complete' if complete else 'checkpointed',updated_at=time.time()))
    if not complete:raise RuntimeError('Training budget exhausted before planned updates; checkpoint retained')


def followup(config):
    from .probe import run,report
    spec=specification(config)
    code=Path.cwd().resolve()
    item=dict(name=config['name'],kind='v2_large',
        checkpoint=str((resolve_path(config['output'])/'technical/runs'/spec['name']/'best.pt').resolve()),
        producer_code=str(code))
    if not run(config,item,deadline_for_job()):raise RuntimeError('Frozen crystallization probe checkpointed at allocation deadline')
    report(config)


def existing_workers_finished(config):
    root=Path(config['previous_queue'])/'technical'
    for lane in config['previous_lanes']:
        path=root/f'lane-{lane}.json'
        if not path.exists():return False
        state=json.loads(path.read_text())
        if state['state'] not in ('complete','failed','checkpointed','allocation_deadline'):return False
        pid=state.get('pid')
        if pid is not None:
            try:os.kill(pid,0)
            except ProcessLookupError:pass
            else:return False
    return True


def coordinate(config_path):
    config=json.loads(Path(config_path).read_text())
    root=resolve_path(config['output'])/'technical'
    try:
        deadline=deadline_for_job()
        while not existing_workers_finished(config):
            save_json(root/'launcher-status.json',dict(state='waiting_for_previous_queue',host=os.uname().nodename,time=time.time()))
            if time.time()>deadline-config['target_training_seconds']-900:
                raise TimeoutError('Insufficient allocated time to start the requested three-hour run after earlier queue')
            time.sleep(30)
        data_path=resolve_path(config['cache'])/'manifest.json'
        if not data_path.exists() or json.loads(data_path.read_text())['state']!='complete':
            raise ValueError('Expanded data was not completed before launch')
        resolved=root/'resolved-config.json'
        if not resolved.exists():
            save_json(root/'launcher-status.json',dict(state='preflight',time=time.time()))
            with (root/'preflight.log').open('a') as log:
                subprocess.run([sys.executable,'-u','-m',__package__+'.large','profile','--config',config_path],check=True,stdout=log,stderr=subprocess.STDOUT)
        resolved_config=json.loads(resolved.read_text())
        run_status=root/'runs'/resolved_config['name']/'status.json'
        if not run_status.exists() or json.loads(run_status.read_text())['state']!='complete':
            save_json(root/'launcher-status.json',dict(state='training',time=time.time()))
            with (root/'training.log').open('a') as log:
                subprocess.run([sys.executable,'-u','-m',__package__+'.large','fit','--config',str(resolved)],check=True,stdout=log,stderr=subprocess.STDOUT)
        save_json(root/'launcher-status.json',dict(state='crystallization',time=time.time()))
        # Evaluation is single-GPU; the paired training used both allocated devices.
        device=os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]
        with (root/'crystallization.log').open('a') as log:
            subprocess.run([sys.executable,'-u','-m',__package__+'.large','probe','--config',str(resolved)],check=True,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=device),stdout=log,stderr=subprocess.STDOUT)
        save_json(root/'launcher-status.json',dict(state='complete',time=time.time()))
    except Exception as error:
        save_json(root/'launcher-status.json',dict(state='failed',error=repr(error),traceback=traceback.format_exc()))
        raise


def submit(campaign_path):
    campaign=json.loads(Path(campaign_path).read_text())
    root=resolve_path(campaign['output'])/'technical'
    if (root/'launches.json').exists():raise FileExistsError('Large campaign already submitted')
    code=snapshot(root)
    launches=[]
    env=dict(os.environ,PCM_PROJECT_ROOT=str(code),TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
    for item in campaign['runs']:
        config_path=code/item['config']
        config=json.loads(config_path.read_text())
        dest=resolve_path(config['output'])/'technical';dest.mkdir(parents=True,exist_ok=True)
        command=['srun','--jobid='+str(item['allocation']),'--overlap','--exact','--nodes=1','--ntasks=1',
            '--cpus-per-task='+str(item['cpus']),'--gres=gpu:2',sys.executable,'-u','-m',__package__+'.large',
            'coordinate','--config',str(config_path)]
        with (dest/'coordinator.log').open('a') as log:
            process=subprocess.Popen(command,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        record=dict(config=str(config_path),code=str(code),pid=process.pid,allocation=item['allocation'],command=command)
        save_json(dest/'launch.json',record);launches.append(record)
    save_json(root/'launches.json',launches)
    print(json.dumps(launches),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['submit','coordinate','profile','fit','probe'])
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    if args.phase=='submit':submit(args.config)
    elif args.phase=='coordinate':coordinate(args.config)
    else:
        config=json.loads(Path(args.config).read_text())
        {'profile':profile,'fit':fit,'probe':followup}[args.phase](config)
