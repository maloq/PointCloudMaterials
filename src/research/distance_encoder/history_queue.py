"""Detached matched history/control fits and spatial-front evaluations."""
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

from src.data.fixed_cohort.protocol import write_json, sha
from src.project_runtime.paths import resolve_path
from src.experiment_runner.metric_docs import check_metric_docs


def submit(path):
    c=json.loads(Path(path).read_text());root=resolve_path(c['output']);tech=root/'technical';tech.mkdir(parents=True,exist_ok=True)
    if (tech/'launch.json').exists():raise ValueError('Already submitted: resume the frozen worker')
    check_metric_docs(family=c.get('metric_family','distance_encoder_history'))
    dense=c['protocol']=='joint_mace_distance_dense_history_v1'
    if dense:
        from .dense_history import make_plan
        make_plan(path)
    repo=Path(__file__).resolve().parents[3];code=tech/'code'
    shutil.copytree(repo/'src',code/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(repo/'docs/metrics',code/'docs/metrics')
    write_json(code/'config.json',c)
    frozen=[]
    for mode in ('real','repeated_current'):
        item=dict(c,output=str(root/mode),history=dict(c['history'],mode=mode),
            encoder_name=c['encoder_name'] if mode=='real' else c['encoder_name']+'-repeat')
        item['wandb']=dict(c['wandb'],display_name=item['encoder_name']+' | current crystal distance')
        dest=code/(mode+'.json');write_json(dest,item);frozen.append(str(dest))
    env=dict(PCM_PROJECT_ROOT=str(repo),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OVITO_THREAD_COUNT='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',PYTORCH_ALLOC_CONF='expandable_segments:True',TORCHINDUCTOR_COMPILE_THREADS='4')
    prefix=['#!/bin/bash','set -euo pipefail','ulimit -n 4096','cd '+shlex.quote(str(code))]
    command='exec env '+shlex.join([f'{k}={v}' for k,v in env.items()])+' '+shlex.join([sys.executable,'-u','-m','src.research.distance_encoder.history_queue'])
    cpu_job=None
    if c['history_evaluation']['protocol']=='native_spatial_front':
        cpu=tech/'prepare.sbatch';cpu.write_text('\n'.join([prefix[0],f'#SBATCH --job-name={c["encoder_name"]}-eval-data',
            '#SBATCH --partition=CPU','#SBATCH --nodes=1','#SBATCH --ntasks=1',f'#SBATCH --cpus-per-task={c["history_evaluation"]["workers"]}',
            '#SBATCH --mem=32G','#SBATCH --time=06:00:00',f'#SBATCH --output={tech}/prepare-%j.log',*prefix[1:],
            command+' prepare --config '+shlex.quote(str(code/'real.json')),'']))
        cpu_job=subprocess.check_output(['sbatch','--parsable',str(cpu)],text=True).strip()
        write_json(tech/'prepare-launch.json',dict(job=cpu_job,script=str(cpu)))
    elif c['history_evaluation']['protocol']!='external_distance':raise ValueError('Unknown history evaluation protocol')
    preparation={}
    if dense:
        module='exec env '+shlex.join([f'{k}={v}' for k,v in env.items()])+' '+shlex.join([sys.executable,'-u','-m','src.research.distance_encoder.dense_history'])
        array=tech/'dense-prepare.sbatch'
        array.write_text('\n'.join([prefix[0],'#SBATCH --job-name=CD-D6-coordinates','#SBATCH --partition=CPU',
            '#SBATCH --nodes=1','#SBATCH --ntasks=1',f'#SBATCH --array=0-{c["dense_history"]["lanes"]-1}',
            f'#SBATCH --cpus-per-task={c["dense_history"]["workers"]}',f'#SBATCH --mem={c["dense_history"]["memory"]}','#SBATCH --time=12:00:00',
            f'#SBATCH --output={tech}/dense-%A-%a.log',*prefix[1:],
            module+' prepare --config '+shlex.quote(str(code/'real.json'))+' --lane "$SLURM_ARRAY_TASK_ID"','']))
        array_job=subprocess.check_output(['sbatch','--parsable',str(array)],text=True).strip()
        seal=tech/'dense-seal.sbatch'
        seal.write_text('\n'.join([prefix[0],'#SBATCH --job-name=CD-D6-seal','#SBATCH --partition=CPU',
            '#SBATCH --nodes=1','#SBATCH --ntasks=1','#SBATCH --cpus-per-task=1','#SBATCH --mem=4G','#SBATCH --time=02:00:00',
            f'#SBATCH --output={tech}/seal-%j.log',*prefix[1:],module+' seal --config '+shlex.quote(str(code/'real.json')),'']))
        seal_job=subprocess.check_output(['sbatch','--parsable',f'--dependency=afterok:{array_job}',str(seal)],text=True).strip()
        preparation=dict(dense_array_job=array_job,dense_seal_job=seal_job)
        write_json(tech/'dense-launch.json',preparation)
    slurm=c['slurm'];script=tech/'train.sbatch'
    script.write_text('\n'.join([prefix[0],f'#SBATCH --job-name={c["encoder_name"]}',f'#SBATCH --partition={slurm["partition"]}',
        '#SBATCH --nodes=1','#SBATCH --ntasks=1',f'#SBATCH --gres=gpu:{slurm["gpus"]}',f'#SBATCH --cpus-per-task={slurm["cpus"]}',
        f'#SBATCH --mem={slurm["memory"]}',f'#SBATCH --time={slurm["time"]}',f'#SBATCH --output={tech}/slurm-%j.log',*prefix[1:],
        command+' worker --config '+shlex.quote(str(code/'config.json')),'']))
    args=['sbatch','--parsable']
    if dense:args.append('--dependency=afterok:'+preparation['dense_seal_job'])
    job=subprocess.check_output([*args,str(script)],text=True).strip()
    record=dict(job=job,prepare_job=cpu_job,code=str(code),script=str(script),configs=frozen,submitted_at=time.time(),
        initial_checkpoint_sha256=sha(resolve_path(c['initial_encoder'])),**preparation)
    write_json(tech/'launch.json',record);return record


def prepare(path):
    from .history_evaluate import prepare as make_history
    c=json.loads(Path(path).read_text());cache=resolve_path(c['history_evaluation']['geometry_cache'])
    try:make_history(path)
    except BaseException:
        write_json(cache/'state.json',dict(state='failed',traceback=traceback.format_exc()));raise


def worker(path):
    c=json.loads(Path(path).read_text());tech=resolve_path(c['output'])/'technical';state=tech/'queue-state.json'
    try:
        for mode in ('real','repeated_current'):
            config=Path(path).with_name(mode+'.json');arm=json.loads(config.read_text());root=resolve_path(arm['output'])
            write_json(state,dict(state='training',active=mode,job=os.environ.get('SLURM_JOB_ID')))
            if not (root/'technical/complete.json').exists():
                subprocess.run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1',f'--nproc_per_node={c["world_size"]}',
                    '-m','src.research.distance_encoder.train','--config',str(config)],check=True)
            if not (root/'technical/complete.json').exists():raise RuntimeError(f'{mode} incomplete; resume checkpoint before evaluation')
        if c['history_evaluation']['protocol']=='external_distance':
            for mode in ('real','repeated_current'):
                write_json(state,dict(state='external_distance_evaluation',active=mode))
                subprocess.run([sys.executable,'-u','-m','src.research.distance_encoder.dense_evaluate',
                    '--config',str(Path(path).with_name(mode+'.json'))],check=True)
            write_json(state,dict(state='complete',finished_at=time.time()));return
        cache=resolve_path(c['history_evaluation']['geometry_cache'])
        while not (cache/'manifest.json').exists():
            if (cache/'state.json').exists() and json.loads((cache/'state.json').read_text())['state']=='failed':
                raise RuntimeError(f'History evaluation preparation failed: {cache}/state.json')
            write_json(state,dict(state='waiting_for_evaluation_geometry'));time.sleep(15)
        for mode in ('baseline','real','repeated_current'):
            config=Path(path).with_name(('real' if mode=='baseline' else mode)+'.json')
            root=resolve_path(c['output'])/mode
            checkpoint=resolve_path(c['initial_encoder']) if mode=='baseline' else root/'technical/best.pt'
            write_json(state,dict(state='front_evaluation',active=mode))
            command=[sys.executable,'-u','-m','src.research.distance_encoder.history_evaluate','evaluate','--config',str(config),
                '--checkpoint',str(checkpoint),'--output',str(root/'front')]
            if mode=='baseline':command.append('--baseline')
            subprocess.run(command,check=True)
        write_json(state,dict(state='complete',finished_at=time.time()))
    except BaseException:
        write_json(state,dict(state='failed',traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['submit','worker','prepare']);p.add_argument('--config',required=True)
    a=p.parse_args();print(json.dumps(dict(submit=submit,worker=worker,prepare=prepare)[a.action](a.config),indent=2))
