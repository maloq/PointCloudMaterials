"""Test first, snapshot source, then submit a bounded two-slot Slurm array."""
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
import numpy as np
import torch

from src.training_methods.shared_pretraining.queue import snapshot,deadline_for_job
from src.research.structural_state.model import calibrate_heads
from .common import Study,write_json,sha


def preflight(study,device):
    from .train import setup,encode,step,validate
    study.prepare()
    receipts=[]
    # Exercise every actual arm including rebuilt noisy banks, both readouts,
    # physical/event losses, and the full 827-example ranking replay.
    for arm in study.config['arms']:
        model,bank,noisy,corpus,target,scalers,conditions,risk,sources,ec=setup(study,arm['name'],device)
        chunk=study.config['training']['microbatch'];fit=corpus.split['fit']
        with torch.no_grad():
            pooled=encode(model,bank,fit,chunk,True)
            model.encoder.pooled_mean.copy_(pooled.mean(0));model.encoder.pooled_scale.copy_(pooled.std(0,correction=0).clamp_min(1e-5))
            z=encode(model,bank,fit,chunk);variance=float(z.var(0,correction=0).mean())
            calibrate_heads(model,z,{k:v[fit] for k,v in target.items()},1.,10.)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        record=step(model,bank,noisy,corpus,target,conditions,risk,sources,arm,study.config,
            np.random.default_rng(12),study.config['training']['ranking_every']-1,variance)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        if not np.isfinite(record['loss']) or model.encoder.center_embedding.weight.grad.norm()<=0:
            raise ValueError(f'Invalid gradient preflight: {arm["name"]}')
        optimizer.step();torch.cuda.synchronize()
        elapsed=time.monotonic()-started
        val=validate(model,bank,corpus,target,conditions,risk,sources,arm,chunk,horizon_ps=study.config['primary_horizon_ps'])
        receipts.append(dict(arm=arm['name'],production_batch=study.config['training']['batch_size'],
            full_ranking_step_seconds=elapsed,peak_GiB=torch.cuda.max_memory_allocated()/2**30,
            loss=record,gradient_norm=float(norm),validation=val))
        print(json.dumps(receipts[-1]),flush=True)
        del model,bank,noisy,optimizer,z,pooled;torch.cuda.empty_cache()
    study.bind()
    write_json(study.technical/'preflight.json',dict(passed=True,identity=study.identity,
        gpu=torch.cuda.get_device_name(),arms=receipts))


def submit(study):
    study.bind();root=study.technical
    receipt=json.loads((root/'preflight.json').read_text())
    if not receipt['passed'] or receipt['identity']!=study.identity:raise ValueError('Exact-code production preflight required')
    if (root/'launch.json').exists():raise FileExistsError('Already submitted: inspect launch.json')
    code=snapshot(root)
    cfg=study.config['slurm'];config=code/study.config_path.relative_to(Path.cwd().resolve())
    env=['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',
         'PYTORCH_ALLOC_CONF=expandable_segments:True',f'PCM_PROJECT_ROOT={code}']
    def script(action,dependency=None):
        gpu=action=='worker'
        lines=['#!/bin/bash',f'#SBATCH --job-name={cfg.get("job_name","robust-onset")}-{action}',
            f'#SBATCH --partition={cfg["partitions"] if gpu else "CPU"}',
            f'#SBATCH --cpus-per-task={cfg["cpus"]}',f'#SBATCH --mem={cfg["memory_GiB"] if gpu else 12}G',
            f'#SBATCH --time={cfg["hours"] if gpu else 1:02d}:00:00',f'#SBATCH --chdir={code}',
            f'#SBATCH --output={root}/{action}-%A_%a.log']
        if gpu:lines+=['#SBATCH --gres=gpu:1',f'#SBATCH --array=0-{len(study.config["arms"])-1}%{cfg["concurrent"]}']
        if dependency:lines += [f'#SBATCH --dependency=afterany:{dependency}']
        command=[sys.executable,'-u','-m',study.queue_module,action,'--config',str(config)]
        return '\n'.join(lines+['set -euo pipefail','exec env '+' '.join(shlex.quote(v) for v in env)+' '+shlex.join(command),''])
    worker=root/'worker.sbatch';worker.write_text(script('worker'))
    job=subprocess.check_output(['sbatch','--parsable',str(worker)],text=True).strip().split(';')[0]
    launch=dict(state='array_submitted',array_job=job,identity=study.identity,code=str(code),
        arms=[a['name'] for a in study.config['arms']],concurrent=cfg['concurrent'],hours_per_fit=cfg['hours'])
    write_json(root/'launch.json',launch)
    collector=root/'collect.sbatch';collector.write_text(script('collect',job))
    launch['collector_job']=subprocess.check_output(['sbatch','--parsable',str(collector)],text=True).strip().split(';')[0]
    launch['state']='submitted';write_json(root/'launch.json',launch);print(json.dumps(launch,indent=2),flush=True)


def worker(study,name,device):
    from .train import train
    from .evaluate import run
    study.bind();status=study.technical/f'{name}-status.json';deadline=deadline_for_job()
    with (study.technical/f'{name}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stage='train'
        try:
            write_json(status,dict(state='running',stage=stage,job=os.environ['SLURM_JOB_ID'],identity=study.identity))
            if not train(study,name,device,deadline):
                write_json(status,dict(state='checkpointed',stage=stage,identity=study.identity));return 75
            torch.cuda.empty_cache();stage='evaluation'
            write_json(status,dict(state='running',stage=stage,identity=study.identity))
            run(study,name,device,deadline)
            write_json(status,dict(state='complete',identity=study.identity))
        except Exception as error:
            write_json(status,dict(state='checkpointed' if isinstance(error,TimeoutError) else 'failed',
                stage=stage,error=repr(error),traceback=traceback.format_exc(),identity=study.identity))
            raise
    return 0


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=['prepare','preflight','submit','worker','collect'])
    p.add_argument('--config',required=True);p.add_argument('--arm');p.add_argument('--device',default='cuda')
    args=p.parse_args();study=Study(args.config)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.action=='prepare':
        from .data import prepare
        prepare(study)
    elif args.action=='preflight':preflight(study,args.device)
    elif args.action=='submit':submit(study)
    elif args.action=='worker':
        name=args.arm or study.config['arms'][int(os.environ['SLURM_ARRAY_TASK_ID'])]['name']
        return worker(study,name,args.device)
    else:
        from .evaluate import collect
        study.bind();collect(study)
    return 0


if __name__=='__main__':sys.exit(main())
