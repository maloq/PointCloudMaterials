"""Concrete local starts and dependent Slurm continuations of the shared study."""
import argparse
from datetime import datetime,timezone
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
from src.data.structural_pretraining.prepare import save_json,file_hash
from src.project_runtime.paths import resolve_path
from src.experiment_runner.slurm import submit_sbatch


def deadline_for_job():
    job=os.environ['SLURM_JOB_ID']
    result=subprocess.run(['scontrol','show','job',job,'--json'],check=True,text=True,capture_output=True)
    value=json.loads(result.stdout)['jobs'][0]['end_time']
    end=value['number'] if isinstance(value,dict) else value
    if not isinstance(end,(float,int)) or end<time.time():raise ValueError(f'Invalid Slurm end time: {value}')
    return end-300


def execute_stage(config_path,phase,deadline):
    config=json.loads(Path(config_path).read_text());root=resolve_path(config['output']);status=root/'technical/status.json'
    if status.exists():
        previous=json.loads(status.read_text())
        if previous['state']=='failed':raise RuntimeError(f'Previous scientific stage failed; inspect before retrying: {status}')
    if phase=='analysis':
        from .analysis import run
        return run(config,deadline)
    from .runtime import run
    return run(config,deadline)


def run_pipeline(paths,deadline,final_slot=False):
    for phase in ('structural','causal','analysis'):
        if phase not in paths:continue
        if time.time()>deadline-300:return False
        try:
            complete=execute_stage(paths[phase],phase,deadline)
        except TimeoutError:
            if final_slot:raise
            return False
        except Exception as error:
            if phase=='analysis':
                config=json.loads(Path(paths[phase]).read_text());status=resolve_path(config['output'])/'technical/status.json'
                save_json(status,dict(state='failed',error=repr(error),traceback=traceback.format_exc()))
            raise
        if not complete:
            if final_slot:raise RuntimeError('Final allocated continuation ended before the requested update budget')
            return False
        import torch
        torch.cuda.empty_cache()
    return True


def worker(args):
    paths=json.loads(Path(args.configs).read_text()) if args.configs else {args.phase:args.config}
    return run_pipeline(paths,deadline_for_job(),args.final_slot)


def serial(plan_path,deadline_utc):
    """Run complete variant pipelines on one externally allocated GPU, without Slurm."""
    end=datetime.fromisoformat(deadline_utc)
    if end.tzinfo is None or end.timestamp()<=time.time():
        raise ValueError('An explicit future deadline with a UTC offset is required')
    plan=json.loads(Path(plan_path).read_text());root=resolve_path(plan['output'])/'technical'
    root.mkdir(parents=True,exist_ok=True)
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        saved=root/'queue-plan.json'
        if saved.exists() and json.loads(saved.read_text())!=plan:
            raise ValueError(f'Queue plan changed: {saved}')
        save_json(saved,plan)
        def status(state,**extra):
            save_json(root/'queue-state.json',dict(state=state,pid=os.getpid(),deadline_utc=end.isoformat(),
                updated_at=datetime.now(timezone.utc).isoformat(),**extra))
        try:
            for item in plan['runs']:
                status('running',variant=item['name'])
                if not run_pipeline(item['configs'],end.timestamp()):
                    status('checkpointed',variant=item['name']);return False
            status('complete');return True
        except Exception as error:
            status('failed',error=repr(error),traceback=traceback.format_exc());raise


def snapshot(root):
    """Freeze executable code so later workspace edits cannot alter queued jobs."""
    dest=root/'code'
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir(parents=True)
    for folder in ['src','configs','docs/metrics']:
        for p in Path(folder).rglob('*'):
            if p.is_file() and p.suffix in ('.py','.json','.yaml','.yml','.md') and '__pycache__' not in p.parts:
                target=dest/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    shutil.copy2('machine.local.yaml',dest/'machine.local.yaml')
    (dest/'output').symlink_to(Path('output').resolve(),target_is_directory=True)
    for folder in ['datasets','data']:
        (dest/folder).symlink_to(Path(folder).resolve(),target_is_directory=True)
    save_json(root/'code-files.json',{str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*.py') if 'output' not in p.relative_to(dest).parts})
    return dest.resolve()


def submit(plan_path):
    plan=json.loads(Path(plan_path).read_text());root=resolve_path(plan['output'])/'technical';root.mkdir(parents=True,exist_ok=True)
    if (root/'submissions.json').exists():raise FileExistsError('Campaign already submitted; inspect recorded jobs')
    code=snapshot(root);python=sys.executable;records=[]
    environment=['PYTORCH_ALLOC_CONF=expandable_segments:True','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',f'PCM_PROJECT_ROOT={code}']
    verification='''from pathlib import Path
from src.experiment_runner.metric_docs import check_metric_docs
from src.project_runtime.paths import resolve_path
from src.training_methods.shared_pretraining import runtime
assert Path(runtime.__file__).resolve().is_relative_to(Path.cwd())
check_metric_docs(family="shared_pretraining")
assert (resolve_path("${storage:cache}/shared-causal-38400-20260918")/"manifest.json").is_file()
assert (resolve_path("${storage:cache}/structural_pretraining/broad-250k-v2-20260917")/"manifest.json").is_file()
print("Frozen code and data paths verified",flush=True)
'''
    subprocess.run(['env',*environment,python,'-c',verification],cwd=code,check=True)
    def batch(name,phase,config,dependency,hours,final):
        log=root/f'{name}-{phase}-{len(records)}';log.mkdir(exist_ok=True)
        command=[python,'-u','-m','src.training_methods.shared_pretraining.queue','worker','--phase',phase,'--config',str(code/config)]
        if final:command.append('--final-slot')
        script='\n'.join(['#!/bin/bash',f'#SBATCH --job-name=shared-{name}-{phase}',f'#SBATCH --partition={plan["partition"]}',
          '#SBATCH --gres=gpu:1','#SBATCH --cpus-per-task=8','#SBATCH --mem=96G',f'#SBATCH --time={hours}:00:00',
          f'#SBATCH --output={log}/%j.log',f'#SBATCH --chdir={code}',
          *([f'#SBATCH --dependency={dependency}'] if dependency else []),'#SBATCH --signal=USR1@300','set -euo pipefail',
          'exec env '+' '.join(shlex.quote(v) for v in environment)+' '+shlex.join(command),''])
        job=submit_sbatch(script,log/'job.sbatch')
        records.append(dict(kind='sbatch',name=name,phase=phase,job_id=job,dependency=dependency,script=str(log/'job.sbatch'),config=str(code/config)))
        save_json(root/'submissions.json',dict(code=str(code),records=records,plan=plan));return job
    for item in plan['runs']:
        name=item['name'];allocation=item['allocation'];configs=item['configs']
        if allocation is not None:
            paths=root/f'{name}-pipeline.json';save_json(paths,{p:str(code/v) for p,v in configs.items()})
            command=['srun',f'--jobid={allocation}','--overlap','-N1','-n1','--cpus-per-task=4','env',*environment,python,'-u','-m',
                'src.training_methods.shared_pretraining.queue','worker','--configs',str(paths)]
            with (root/f'{name}-local.log').open('ab',buffering=0) as log:
                process=subprocess.Popen(command,cwd=code,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            records.append(dict(kind='local',name=name,phase='pipeline',allocation=allocation,pid=process.pid,command=command))
            dependency=f'afterany:{allocation}'
        else:dependency=None
        for slot in range(plan['structural_slots']):
            job=batch(name,'structural',configs['structural'],dependency,plan['structural_hours'],slot==plan['structural_slots']-1)
            dependency=f'afterany:{job}'
        causal=batch(name,'causal',configs['causal'],f'afterok:{job}',plan['causal_hours'],True)
        batch(name,'analysis',configs['analysis'],f'afterok:{causal}',plan['analysis_hours'],True)
    save_json(root/'submissions.json',dict(submitted_at=datetime.now(timezone.utc).isoformat(),code=str(code),records=records,plan=plan))
    print(json.dumps(records,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('submit');s.add_argument('--plan',required=True)
    r=sub.add_parser('serial');r.add_argument('--plan',required=True);r.add_argument('--deadline-utc',required=True)
    w=sub.add_parser('worker');w.add_argument('--phase',choices=['structural','causal','analysis']);w.add_argument('--config');w.add_argument('--configs');w.add_argument('--final-slot',action='store_true')
    args=p.parse_args()
    if args.command=='submit':submit(args.plan)
    elif args.command=='serial':serial(args.plan,args.deadline_utc)
    else:worker(args)
