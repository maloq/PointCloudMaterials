"""Detached one-GPU material fine-tuning and matched parent evaluations."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback

from src.data.fixed_cohort.protocol import sha,write_json
from src.experiment_runner.metric_docs import check_metric_docs
from src.project_runtime.paths import resolve_path


def submit(path,allocation,gpu):
    c=json.loads(Path(path).read_text());root=resolve_path(c['output']);tech=root/'technical'
    tech.mkdir(parents=True,exist_ok=True)
    if (tech/'launch.json').exists():raise ValueError('Already submitted; resume the frozen worker')
    if sha(resolve_path(c['initial_encoder']))!=c['initial_encoder_sha256']:raise ValueError('Parent checkpoint changed')
    if c['world_size']!=1 or c['history']['mode']!='real':raise ValueError('Material queue uses one GPU per real-history fit')
    if os.environ.get('SLURM_JOB_ID')!=str(allocation):raise ValueError('Launch from inside the declared live Slurm allocation')
    check_metric_docs(family=c['metric_family'])
    if c['material_finetune']['material']=='Ta':
        from .material_data import make_plan
        make_plan(path)
    repo=Path(__file__).resolve().parents[3];code=tech/'code'
    shutil.copytree(repo/'src',code/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(repo/'docs/metrics',code/'docs/metrics');write_json(code/'config.json',c)
    env=dict(os.environ,PCM_PROJECT_ROOT=str(repo),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OVITO_THREAD_COUNT='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',PYTORCH_ALLOC_CONF='expandable_segments:True',TORCHINDUCTOR_COMPILE_THREADS='4')
    cpu={}
    if c['material_finetune']['material']=='Ta':
        exports={k:env[k] for k in ('PCM_PROJECT_ROOT','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OVITO_THREAD_COUNT','TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD')}
        command='exec env '+shlex.join([f'{k}={v}' for k,v in exports.items()])+ ' '+shlex.join([sys.executable,'-u','-m','src.research.distance_encoder.material_data'])
        for stage in ('source','seal'):
            script=tech/f'ta-{stage}.sbatch'
            lines=['#!/bin/bash','set -euo pipefail']
            directives=[f'#SBATCH --job-name=CD-Ta-{stage}','#SBATCH --partition=CPU','#SBATCH --nodes=1','#SBATCH --ntasks=1',
                '#SBATCH --cpus-per-task=2','#SBATCH --mem=24G','#SBATCH --time=08:00:00',f'#SBATCH --output={tech}/ta-{stage}-%A-%a.log']
            if stage=='source':directives.append('#SBATCH --array=0-3')
            lines=[lines[0],*directives,*lines[1:],'ulimit -n 4096','cd '+shlex.quote(str(code)),
                command+' '+stage+' --config '+shlex.quote(str(code/'config.json'))+(' --index "$SLURM_ARRAY_TASK_ID"' if stage=='source' else ''),'']
            script.write_text('\n'.join(lines))
            dependency=[] if stage=='source' else [f'--dependency=afterok:{cpu["source"]}']
            cpu[stage]=subprocess.check_output(['sbatch','--parsable',*dependency,str(script)],text=True).strip()
        write_json(tech/'preparation-launch.json',cpu)
    # The interactive step already owns the allocation's GPUs. A nested exclusive
    # step would wait forever; pin each detached process inside that live step.
    env['CUDA_VISIBLE_DEVICES']=str(gpu)
    command=[sys.executable,'-u','-m','src.research.distance_encoder.material_queue','worker','--config',str(code/'config.json')]
    with (tech/'worker.log').open('ab') as out:
        process=subprocess.Popen(command,cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(allocation=allocation,gpu=gpu,pid=process.pid,code=str(code),config=str(code/'config.json'),cpu_jobs=cpu,
        command=command,submitted_at=time.time(),parent_checkpoint_sha256=c['initial_encoder_sha256'])
    write_json(tech/'launch.json',receipt);return receipt


def worker(path):
    c=json.loads(Path(path).read_text());root=resolve_path(c['output']);tech=root/'technical';state=tech/'queue-state.json'
    try:
        if c['material_finetune']['material']=='Ta':
            cache=resolve_path(c['ta_evaluation']['root']);jobs=json.loads((tech/'preparation-launch.json').read_text())
            while not (cache/'manifest.json').exists():
                status=subprocess.check_output(['scontrol','show','job',jobs['seal'],'-o'],text=True)
                if any(x in status for x in ('DependencyNeverSatisfied','JobState=FAILED','JobState=CANCELLED','JobState=TIMEOUT','JobState=OUT_OF_MEMORY','JobState=NODE_FAIL')):
                    raise RuntimeError(f'Ta preparation failed: {status}; inspect {tech}/ta-source logs')
                write_json(state,dict(state='waiting_for_ta_evaluation_data',cpu_jobs=jobs));time.sleep(30)
        write_json(state,dict(state='training',material=c['material_finetune']['material'],job=os.environ.get('SLURM_JOB_ID'),step=os.environ.get('SLURM_STEP_ID')))
        if not (tech/'complete.json').exists():
            subprocess.run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=1',
                '-m','src.research.distance_encoder.train','--config',path],check=True)
        if not (tech/'complete.json').exists():raise RuntimeError('Training checkpointed before completion; resume before evaluation')
        write_json(state,dict(state='evaluating',material=c['material_finetune']['material']))
        if c['material_finetune']['material']=='Al':
            subprocess.run([sys.executable,'-u','-m','src.research.distance_encoder.history_evaluate','evaluate','--config',path,
                '--checkpoint',str(tech/'best.pt'),'--output',str(root/'front')],check=True)
        subprocess.run([sys.executable,'-u','-m','src.research.distance_encoder.material_evaluate','--config',path],check=True)
        write_json(state,dict(state='complete',finished_at=time.time()))
    except BaseException:
        write_json(state,dict(state='failed',traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['submit','worker']);p.add_argument('--config',required=True);p.add_argument('--allocation');p.add_argument('--gpu',type=int,choices=[0,1])
    a=p.parse_args()
    if a.stage=='submit' and (not a.allocation or a.gpu is None):p.error('submit requires --allocation and --gpu')
    print(json.dumps(submit(a.config,a.allocation,a.gpu) if a.stage=='submit' else worker(a.config),indent=2))
