"""Bounded GPU continuation campaign; reuses the maintained recovery integrators."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np

from src.data.fixed_cohort.protocol import sha, write_json
from src.experiment_runner.execution import ExecutionBundle, allocation_deadline, recorded_stage
from src.project_runtime.paths import REPO, machine, resolve_path
from src.research.nucleus_recovery import roots, read
from . import data, analysis

MODULE='src.research.nucleus_controls.queue'


def prepare(c):
    plan=data.prepare(c);out,_,_=roots(c)
    for case in plan['cases']:
        path=out/'technical/reference'/f'{case["key"]}.json'
        if not path.exists():
            result=analysis.reference(c,case)
            write_json(path,result)
            print('reference checked',case['key'],result['any_new_nucleus_12ps'],result['seed_terminal_status'],flush=True)
    analysis.summarize(c)
    return plan


def run_branch(c,plan,task,case,models,deadline):
    from src.research import nucleus_recovery_md as md
    _,scratch,archive_root=roots(c);state=case['state']
    root=scratch/'branches'/task['name'];archive=archive_root/'branches'/task['name']
    root.mkdir(parents=True,exist_ok=True);archive.mkdir(parents=True,exist_ok=True)
    (root/'technical').mkdir(exist_ok=True)
    for key,path in c['meam_files'].items():
        if sha(resolve_path(path))!=plan['potential_sha256'][key]:raise ValueError('Pinned MEAM file changed')
    initial_path=Path(state['initial_path'])
    if sha(initial_path)!=state['initial_sha256']:raise ValueError('Saved starting state changed')
    with np.load(initial_path) as values:initial={k:values[k].copy() for k in values.files}
    provenance=dict(task=task,case=case['key'],cohort=case['cohort'],plan_identity=plan['identity'],
        original_source=case['raw'],original_frame=state['start_frame'],initial_sha256=state['initial_sha256'],
        encoder_input=None,predictor_input=None,physical_temperature_K=task['temperature_K'],
        states='float32 Q,p,noise; float32 forces; float64 scalar energy reduction' if task['integrator']=='baoab'
            else 'native original-settings NPT float64 integration; float32 GPU force arithmetic; bath state reset',
        velocity_source='actual simultaneous saved velocities, angstrom/ps; no velocity redraw',
        pid=os.getpid(),host=os.uname().nodename,allocation=os.environ.get('SLURM_JOB_ID'))
    write_json(root/'input.json',provenance);md.archive_file(root/'input.json',archive/'input.json')
    if not (root/'integration-complete.json').exists():
        observations=md.arrays(root,state['stop_frame']-state['start_frame']+1,case['atom_count'])
        runner=md.baoab if task['integrator']=='baoab' else md.npt
        if not runner(c,plan,task,state,case,root,archive,initial,observations,models,deadline):return False
        write_json(root/'integration-complete.json',dict(task=task['name'],plan_identity=plan['identity'],completed_unix=time.time()))
        del observations
    if not (root/'result.json').exists():analysis.analyze(c,plan,task,case,root)
    # Existing publisher accepts already measured branch records; its fallback
    # birth-recovery analyzer is therefore never used for these control cohorts.
    md.publish(c,plan,task,state,case,root,archive,initial)
    print('completed',task['name'],json.dumps(read(root/'result.json')),flush=True)
    return True


def worker(c,lane,*,summarizer=analysis.summarize,runner=run_branch,completion=None):
    import torch
    out,_,archive=roots(c);plan=read(out/'technical/plan.json');budget=read(out/'technical/budget.json')
    deadline=min(budget['hard_end_unix']-c['allocation_reserve_seconds'],
        allocation_deadline(reserve_seconds=c['allocation_reserve_seconds']))
    if time.time()>=deadline:
        write_json(out/f'technical/lane-{lane}.json',dict(state='budget_expired_before_start'));return
    torch.set_num_threads(1)
    major,minor=torch.cuda.get_device_capability();architecture=f'{major}.{minor}'
    cache=resolve_path(c['compiler_cache'])/str(os.environ.get('SLURM_JOB_ID','local'))/f'gpu-{lane}'
    os.environ.update(TORCH_CUDA_ARCH_LIST=architecture,TORCH_EXTENSIONS_DIR=str(cache/'extensions'),
                      TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'))
    write_json(out/f'technical/hardware-{lane}.json',dict(device=torch.cuda.get_device_name(),architecture=architecture,
        torch=torch.__version__,cuda=torch.version.cuda,host=os.uname().nodename,pid=os.getpid(),
        cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),allocation=os.environ.get('SLURM_JOB_ID')))
    models={}
    for task in plan['tasks']:
        if time.time()>=deadline:
            write_json(out/f'technical/lane-{lane}.json',dict(state='budget_checkpointed'));return
        technical=out/'technical/tasks'/task['name'];technical.mkdir(parents=True,exist_ok=True)
        with (technical/'lease.lock').open('w') as lease:
            try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if (completion(c,plan,task) if completion is not None else
                    (archive/'branches'/task['name']/'complete.json').exists()):continue
            case=next(s for s in plan['cases'] if s['key']==task['case'])
            with recorded_stage(technical/'stage.json',task=task['name'],lane=lane,allocation=os.environ.get('SLURM_JOB_ID')) as stage:
                if not runner(c,plan,task,case,models,deadline):
                    stage.update(phase='checkpointed',reason='campaign/allocation deadline')
                    write_json(out/f'technical/lane-{lane}.json',dict(state='budget_checkpointed',task=task['name']));return
            summarizer(c)
    write_json(out/f'technical/lane-{lane}.json',dict(state='queue_drained'))


def bounded(c,config,*,module=MODULE,summarizer=analysis.summarize,lane_ids=None):
    out,_,_=roots(c);budget=read(out/'technical/budget.json')
    hard_stop=min(budget['hard_end_unix']-120,allocation_deadline(reserve_seconds=120))
    if time.time()>=hard_stop:
        write_json(out/'technical/execution.json',dict(state='expired_before_start',budget=budget));return
    visible=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    lane_ids=list(range(c['lanes'])) if lane_ids is None else lane_ids
    if len(visible)!=len(lane_ids):raise ValueError('Visible GPUs differ from the declared worker lanes')
    owned=[];receipt=dict(state='running',allocation=os.environ.get('SLURM_JOB_ID'),host=os.uname().nodename,
        started_unix=time.time(),budget=budget,workers=[])
    receipt_path=out/f'technical/execution-{lane_ids[0]}.json' if module!=MODULE else out/'technical/execution.json'
    for lane,gpu in zip(lane_ids,visible):
        env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu}
        with (out/f'technical/lane-{lane}.log').open('a') as stream:
            p=subprocess.Popen([sys.executable,'-u','-m',module,'worker','--config',str(config),'--lane',str(lane)],
                cwd=Path(__file__).resolve().parents[3],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        owned.append(p);receipt['workers'].append(dict(pid=p.pid,process_group=p.pid,gpu=gpu,lane=lane))
    write_json(receipt_path,receipt)
    failed_tasks=[]
    try:
        while any(p.poll() is None for p in owned) and time.time()<hard_stop:
            # MPI finalization can hang after Python records a failed stage.
            # Surface that failure promptly instead of waiting for walltime.
            failed_tasks=[p.parent.name for p in (out/'technical/tasks').glob('*/stage.json')
                if (s:=read(p)).get('allocation')==receipt['allocation'] and s.get('state')=='failed'
                and s.get('started_at',0)>=receipt['started_unix']]
            if failed_tasks:break
            time.sleep(5)
    finally:
        # Only process groups created immediately above are owned by this driver.
        for p in owned:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        end=time.time()+30
        while any(p.poll() is None for p in owned) and time.time()<end:time.sleep(1)
        for p in owned:
            if p.poll() is None:os.killpg(p.pid,signal.SIGKILL)
        for p in owned:p.wait()
        progress=summarizer(c)
        receipt.update(state='complete' if progress['completed']==progress['planned'] else 'partial',
            finished_unix=time.time(),returncodes=[p.returncode for p in owned],progress=progress)
        write_json(receipt_path,receipt)
    if failed_tasks or any(p.returncode not in (0,-signal.SIGTERM) for p in owned):
        raise RuntimeError('A control worker failed; see its preserved stage/log: '+', '.join(failed_tasks))


def submit(c):
    prepare(c);out,_,_=roots(c);receipt_path=out/'technical/submission.json'
    if receipt_path.exists():raise FileExistsError('Controls already submitted; inspect the recorded owned job')
    code=out/'technical/code'
    if not code.exists():
        ExecutionBundle.freeze(REPO,code,c,directories=('src','docs/metrics','configs'),
            files=[(REPO/'machine.local.yaml','machine.local.yaml'),(REPO/'scripts/convert_trajectory.py','scripts/convert_trajectory.py')])
    elif read(code/'config.json')!=c:raise ValueError('Frozen config differs')
    start=time.time();hard_end=start+c['maximum_wall_hours']*3600
    budget=dict(submitted_unix=start,hard_end_unix=hard_end,maximum_wall_hours=c['maximum_wall_hours'],
        deadline_Paris=datetime.fromtimestamp(hard_end,ZoneInfo('Europe/Paris')).isoformat(),
        includes_queue_wait=True,automatic_extension=False,allocation_walltime='08:00:00')
    write_json(out/'technical/budget.json',budget)
    cuda=machine()['execution']['gpu_meam_portable_cuda_root']
    environment=dict(PCM_PROJECT_ROOT=str(REPO),CUDA_HOME=cuda,MAX_JOBS='2',TORCHINDUCTOR_COMPILE_THREADS='2',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OVITO_THREAD_COUNT='1',QT_QPA_PLATFORM='offscreen',UCX_TLS='tcp,self',
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',PATH=cuda+'/bin:'+str(Path(sys.executable).parent)+':/usr/bin:/bin',TZ='Europe/Paris')
    deadline=datetime.fromtimestamp(hard_end,ZoneInfo('Europe/Paris')).strftime('%Y-%m-%dT%H:%M:%S')
    command=[sys.executable,'-u','-m',MODULE,'bounded','--config',str(code/'config.json')]
    script=out/'technical/controls.sbatch'
    script.write_text('\n'.join(['#!/bin/bash','#SBATCH --job-name=AL-nucleus-controls',
        '#SBATCH --partition='+c['partitions'],'#SBATCH --nodes=1','#SBATCH --ntasks=1',
        '#SBATCH --cpus-per-task=8','#SBATCH --mem=48G',f'#SBATCH --gres=gpu:{c["lanes"]}',
        '#SBATCH --time=08:00:00','#SBATCH --deadline='+deadline,'#SBATCH --no-requeue',
        '#SBATCH --output='+str(out/'technical/scheduler-%j.log'),'set -euo pipefail',
        'cd '+shlex.quote(str(code)),'exec env '+shlex.join([f'{k}={v}' for k,v in environment.items()]+command),'']))
    job=subprocess.check_output(['sbatch','--parsable',str(script)],text=True,env={**os.environ,'TZ':'Europe/Paris'}).strip().split(';')[0]
    if not job.isdigit():raise ValueError('Unexpected scheduler submission response: '+job)
    receipt=dict(job=job,owner='current response/nucleus-recovery chat',script=str(script),code=str(code),
        budget=budget,config=c,submitted_unix=start)
    write_json(receipt_path,receipt);print(json.dumps(receipt,indent=2))
    return receipt


def assist(c, frozen_code, gpus, partitions=None, exclude=None):
    """Add Slurm workers to an existing fixed plan without extending its budget."""
    if gpus<1:raise ValueError('At least one additional GPU is required')
    code=Path(frozen_code).resolve()
    if read(code/'config.json')!=c:raise ValueError('Helper must use the existing frozen recipe')
    out,_,_=roots(c);budget=read(out/'technical/budget.json');plan=read(out/'technical/plan.json')
    remaining=int((budget['hard_end_unix']-time.time())//60)-2
    minutes=min(120,remaining)
    if minutes<30:raise ValueError('Insufficient original campaign budget for another allocation')
    stamp=datetime.now(ZoneInfo('Europe/Paris')).strftime('%Y%m%dT%H%M%S')
    directory=out/'technical'/('helpers-'+stamp);directory.mkdir()
    cuda=machine()['execution']['gpu_meam_portable_cuda_root']
    environment=dict(PCM_PROJECT_ROOT=str(REPO),CUDA_HOME=cuda,MAX_JOBS='2',TORCHINDUCTOR_COMPILE_THREADS='2',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OVITO_THREAD_COUNT='1',
        QT_QPA_PLATFORM='offscreen',UCX_TLS='tcp,self',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
        PATH=cuda+'/bin:'+str(Path(sys.executable).parent)+':/usr/bin:/bin',TZ='Europe/Paris')
    if 'PCM_MACHINE_CONFIG' in os.environ:
        environment['PCM_MACHINE_CONFIG']=str(Path(os.environ['PCM_MACHINE_CONFIG']).resolve())
    deadline=datetime.fromtimestamp(budget['hard_end_unix'],ZoneInfo('Europe/Paris')).strftime('%Y-%m-%dT%H:%M:%S')
    command=[sys.executable,'-u','-m',MODULE,'worker','--config',str(code/'config.json')]
    script=directory/'helpers.sbatch'
    script.write_text('\n'.join(['#!/bin/bash','#SBATCH --job-name=AL-nucleus-helper',
        '#SBATCH --partition='+(partitions or c['partitions']),
        *(['#SBATCH --exclude='+exclude] if exclude else []),'#SBATCH --nodes=1','#SBATCH --ntasks=1',
        '#SBATCH --cpus-per-task=4','#SBATCH --mem=24G','#SBATCH --gres=gpu:1',
        f'#SBATCH --array=0-{gpus-1}',f'#SBATCH --time={minutes}',
        '#SBATCH --deadline='+deadline,'#SBATCH --no-requeue',
        '#SBATCH --output='+str(directory/'worker-%A_%a.log'),'set -euo pipefail',
        'cd '+shlex.quote(str(code)),
        # Unique worker IDs keep hardware/exit receipts separate from node60 lanes.
        'exec env '+shlex.join([f'{k}={v}' for k,v in environment.items()]+command)
            +' --lane "$((SLURM_ARRAY_JOB_ID + SLURM_ARRAY_TASK_ID))"','']))
    job=subprocess.check_output(['sbatch','--parsable',str(script)],text=True,
        env={**os.environ,'TZ':'Europe/Paris'}).strip().split(';')[0]
    if not job.isdigit():raise ValueError('Unexpected helper submission response: '+job)
    receipt=dict(job=job,array_tasks=list(range(gpus)),owner='current response/nucleus-recovery chat',
        submitted_unix=time.time(),script=str(script),code=str(code),budget=budget,
        allocation_minutes=minutes,plan_identity=plan['identity'],planned_continuations=len(plan['tasks']),
        scientific_plan_changed=False,partitions=partitions or c['partitions'],exclude=exclude,
        submission_source_sha256=sha(Path(__file__)),environment=environment)
    write_json(directory/'submission.json',receipt)
    print(json.dumps(receipt,indent=2));return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','submit','assist','bounded','worker','summarize'])
    parser.add_argument('--config',required=True);parser.add_argument('--lane',type=int)
    parser.add_argument('--frozen-code',type=Path);parser.add_argument('--gpus',type=int,default=4)
    parser.add_argument('--partitions');parser.add_argument('--exclude')
    args=parser.parse_args();c=read(args.config)
    if args.stage=='prepare':prepare(c)
    elif args.stage=='submit':submit(c)
    elif args.stage=='assist':
        if args.frozen_code is None:parser.error('assist requires --frozen-code')
        assist(c,args.frozen_code,args.gpus,args.partitions,args.exclude)
    elif args.stage=='bounded':bounded(c,args.config)
    elif args.stage=='worker':worker(c,args.lane)
    else:print(json.dumps(analysis.summarize(c),indent=2))


if __name__=='__main__':main()
