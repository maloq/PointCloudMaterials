"""Frozen, leased two-GPU collection/training with a shared campaign deadline."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo
from src.project_runtime.paths import machine
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, allocation_deadline
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_rows
from .data import FAMILY, read, root, cache, prepare, prepare_job, seal, digest, sha, write_json, resolve_path

OWNER='current response/nucleus-recovery chat; continuous-motion implementation'
MODULE='src.research.continuous_motion.workflow'


def prepare_run(c):
    """Reuse an immutable data campaign for an explicitly versioned fit budget."""
    if 'reuse_campaign' not in c:
        return prepare(c)
    upstream=resolve_path(c['reuse_campaign']);plan=read(upstream/'technical/plan.json')
    if plan['identity']!=c['reuse_plan_identity']:
        raise ValueError('Reused data-plan identity changed')
    old=plan['config']
    for key in ('seed','source_identity','fixed_identity','encoder','model','teacher','arms','fit_seeds',
                'horizons_ps','anchor_fractions','inner_selection_fraction','cache','simulation_archive','simulation_scratch'):
        if c[key]!=old[key]:
            raise ValueError(f'Budget continuation changes scientific input {key}')
    if {k:v for k,v in c['training'].items() if k!='updates'} != {k:v for k,v in old['training'].items() if k!='updates'}:
        raise ValueError('Budget continuation changes training other than update count')
    if c['training']['updates']<=old['training']['updates']:
        raise ValueError('Continuation must increase the update budget')
    tech=root(c)/'technical'
    for filename in ('plan.json','prediction-context.json','data-complete.json'):
        source=upstream/'technical'/filename;target=tech/filename
        value=read(source)
        if target.exists() and read(target)!=value:
            raise ValueError(f'Changed reused data receipt: {target}')
        write_json(target,value)
    write_json(tech/'reuse.json',dict(upstream=str(upstream),data_plan_identity=plan['identity'],
        data_complete_sha256=sha(upstream/'technical/data-complete.json'),
        previous_updates=old['training']['updates'],new_updates=c['training']['updates'],
        changed='training duration and execution resources only; original data receipts and completed simulator shots reused'))
    return plan


@contextmanager
def lease(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle,fcntl.LOCK_UN)


def report(c):
    tech=root(c)/'technical';plan=read(tech/'plan.json')
    rows=[];fits=[]
    for seed in c['fit_seeds']:
        for arm in c['arms']:
            name=f'{arm}-{seed}';folder=root(c)/'analyses'/name/'technical'
            if (folder/'complete.json').exists():
                done=read(folder/'complete.json')
                rows.extend(dict(arm=arm,seed=seed,**r) for r in done['metrics'])
                fits.append(dict(arm=arm,seed=seed,state='complete'))
            else:
                status=read(folder/'progress.json') if (folder/'progress.json').exists() else dict(state='queued')
                fits.append(dict(arm=arm,seed=seed,**status))
    status=dict(observations=len(list((cache(c)/'observations').glob('*/complete.json'))),
                teachers=len(list(resolve_path(c['simulation_archive']).glob('*/complete.json'))),
                expected=len(plan['jobs']),fits=fits,
                exclusions=plan['exclusions'],updated_unix=time.time())
    with lease(tech/'report.lock') as acquired:
        if acquired:
            write_json(tech/'status.json',status)
            if rows:
                # Combined publication retains the campaign's frozen contract;
                # each fit links its own actual calculation implementation.
                source=Path(read(tech/'launch.json')['code'])/'src/experiment_runner/metric_docs.py'
                spec=importlib.util.spec_from_file_location('src.experiment_runner._motion_reference_metrics',source)
                reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
                columns=['arm','seed','track','horizon_ps','rows','ancestors','nll','nll_lower','nll_upper',
                         'mean_mse','mean_mse_lower','mean_mse_upper','ballistic_mean_mse',
                         'persistence_mse','marginal_nll','coverage90']
                reference.write_metric_rows(rows,root(c)/'analyses/comparison-v1',family=FAMILY,name='predictive',columns=columns)
            if all(x['state']=='complete' for x in fits):
                write_json(tech/'complete.json',dict(state='complete',fits=len(fits),finished_unix=time.time()))
    return status


def worker(c,lane,*,runtime_microbatch=None,execution_profile=None):
    import torch
    from . import teacher
    from .train import fit
    torch.set_num_threads(2)
    tech=root(c)/'technical';plan=read(tech/'plan.json');budget=read(tech/'budget.json')
    deadline=min(budget['deadline_unix'],allocation_deadline(reserve_seconds=600))
    if (tech/'complete.json').exists() or time.time()>=deadline:
        return
    major,minor=torch.cuda.get_device_capability()
    compiler=resolve_path(c['compiler_cache'])/lane
    os.environ.update(TORCH_CUDA_ARCH_LIST=f'{major}.{minor}',TORCH_EXTENSIONS_DIR=str(compiler/'extensions'),
                      TORCHINDUCTOR_CACHE_DIR=str(compiler/'inductor'))
    write_json(tech/f'hardware-{lane}.json',dict(host=os.uname().nodename,pid=os.getpid(),
        device=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        deadline_unix=deadline,visible=os.environ.get('CUDA_VISIBLE_DEVICES'),allocation=os.environ.get('SLURM_JOB_ID')))
    progress=tech/f'worker-{lane}.json'
    def update(**fields):
        write_json(progress,dict(owner=OWNER,lane=lane,pid=os.getpid(),updated_unix=time.time(),**fields))
    try:
        while not (tech/'data-complete.json').exists():
            worked=False
            for job in plan['jobs']:
                if time.time()>=deadline-300:
                    raise TimeoutError('Allocation/campaign reserve reached during collection')
                observation=cache(c)/'observations'/job['key']/'complete.json'
                continuation=teacher.archive(c,job)/'complete.json'
                if observation.exists() and continuation.exists():
                    continue
                with lease(tech/'locks'/f"collect-{job['key']}.lock") as acquired:
                    if not acquired:
                        continue
                    update(state='collecting',job=job['key'])
                    prepare_job(c,plan,job)
                    try:
                        done=teacher.collect(c,plan,job,deadline-120)
                    except BaseException as error:
                        failure=dict(owner=OWNER,job=job,plan_identity=plan['identity'],error=repr(error),
                            traceback=traceback.format_exc(),failed_unix=time.time(),
                            restart='recompute this bounded100fs teacher; completed teachers preserved')
                        write_json(resolve_path(c['simulation_scratch'])/job['key']/'failure.json',failure)
                        write_json(teacher.archive(c,job)/'failure.json',failure)
                        raise
                    print('collected',job['key'],'seconds',round(done['seconds'],2),flush=True)
                    worked=True
            with lease(tech/'seal.lock') as acquired:
                if acquired and not (tech/'data-complete.json').exists():
                    if all((cache(c)/'observations'/j['key']/'complete.json').exists() and
                           (teacher.archive(c,j)/'complete.json').exists() for j in plan['jobs']):
                        seal(c,plan)
            report(c)
            if not worked and not (tech/'data-complete.json').exists():
                time.sleep(10)
        gc.collect();torch.cuda.empty_cache()
        for seed in c['fit_seeds']:
            for arm in c['arms']:
                name=f'{arm}-{seed}'
                if (root(c)/'analyses'/name/'technical/complete.json').exists():
                    continue
                if time.time()>=deadline-300:
                    raise TimeoutError('Allocation/campaign reserve reached before fit')
                with lease(tech/'locks'/f'fit-{name}.lock') as acquired:
                    if not acquired:
                        continue
                    update(state='training',fit=name)
                    fit(c,plan,arm,seed,deadline,runtime_microbatch=runtime_microbatch,
                        execution_profile=execution_profile)
                    gc.collect();torch.cuda.empty_cache();report(c)
        report(c);update(state='finished_available_work')
    except BaseException as error:
        update(state='checkpointed_deadline' if isinstance(error,TimeoutError) else 'failed',
               error=repr(error),traceback=traceback.format_exc())
        raise


def environment(c,repo):
    cuda=machine()['execution'][c['cuda_root_key']]
    return dict(PCM_PROJECT_ROOT=str(repo),CUDA_HOME=cuda,
        PATH=cuda+'/bin:'+str(Path(sys.prefix)/'bin')+':/usr/bin:/bin',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',MAX_JOBS='2',TORCHINDUCTOR_COMPILE_THREADS='2',
        WANDB_MODE='online',TZ='Europe/Paris',UCX_TLS='tcp,self')


def launch(c):
    check_metric_docs(family=FAMILY);prepare_run(c)
    import wandb
    if not wandb.Api(timeout=30).viewer:
        raise RuntimeError('W&B online authentication failed; no offline fallback')
    tech=root(c)/'technical'
    if (tech/'launch.json').exists():
        raise FileExistsError('Already launched; use resume-local with the frozen bundle')
    repo=Path(__file__).resolve().parents[3]
    bundle=ExecutionBundle.freeze(repo,tech/'code',c,directories=('src','docs/metrics'))
    files={str(p.relative_to(bundle.root)):sha(p) for p in (bundle.root/'src').rglob('*')
           if p.is_file() and p.suffix in ('.py','.cu','.h','.cpp')}
    files['docs/metrics/continuous_motion.md']=sha(bundle.root/'docs/metrics/continuous_motion.md')
    write_json(tech/'code-identity.json',dict(identity=digest(files),files=files))
    started=time.time();end=started+c['wall_seconds']
    budget=dict(started_unix=started,deadline_unix=end,includes_queue_wait=True,owner=OWNER,
        deadline_Paris=datetime.fromtimestamp(end,ZoneInfo('Europe/Paris')).isoformat(),wall_seconds=c['wall_seconds'])
    write_json(tech/'budget.json',budget)
    receipt=dict(owner=OWNER,jobs={},code=str(bundle.root),host=os.uname().nodename,
                 allocation=os.environ.get('SLURM_JOB_ID'),budget=budget)
    write_json(tech/'launch.json',receipt)
    resume_local(c)
    queue=SlurmQueue(tech,bundle,MODULE,environment(c,repo),tech/'launch.json',receipt,'CMOTION')
    if c.get('additional_gpu_jobs',0):
        remaining=math.floor((end-time.time())/60)-10
        with queue.submission():
            for index in range(c['additional_gpu_jobs']):
                queue.submit(f'parallel-{index}',['--gpus=2','--cpus-per-task=8','--mem=64G',
                    f'--time={remaining}','--time-min=120','--no-requeue',
                    '--deadline='+datetime.fromtimestamp(end,ZoneInfo('Europe/Paris')).strftime('%Y-%m-%dT%H:%M:%S')],
                    partition=c['gpu_partition'],command_stage='coordinator')
    if c.get('handoff_after_current_allocation'):
        # Dependency observes the shared allocation; it never changes/stops it.
        allocation=os.environ['SLURM_JOB_ID']
        remaining=max(1,math.floor((end-allocation_deadline(reserve_seconds=0))/60)-10)
        with queue.submission():
            queue.submit('handoff',['--gpus=2','--cpus-per-task=8','--mem=64G',f'--time={remaining}',
                '--time-min=60','--no-requeue','--exclude=node52',
                '--deadline='+datetime.fromtimestamp(end,ZoneInfo('Europe/Paris')).strftime('%Y-%m-%dT%H:%M:%S')],
                dependency='afterany:'+allocation,partition='RTX6000PRO',command_stage='coordinator')
    print(json.dumps(receipt,indent=2),flush=True)


def resume_local(c):
    tech=root(c)/'technical';saved=read(tech/'launch.json');bundle=Path(saved['code'])
    # Only recorded workers in this workflow are considered. A live worker is
    # never replaced or signaled, including when its role changes mid-campaign.
    if time.time()>=saved['budget']['deadline_unix']:
        raise TimeoutError('Original campaign deadline expired')
    repo=Path(os.environ.get('PCM_PROJECT_ROOT',Path(__file__).resolve().parents[3]))
    for gpu in c['local_gpus']:
        lane=f"helper-{os.environ['SLURM_JOB_ID']}-{gpu}"
        launch_path=tech/f'launch-{lane}.json'
        if launch_path.exists():
            previous=read(launch_path);pid=previous['pid']
            commandline=Path(f'/proc/{pid}/cmdline')
            if commandline.exists() and MODULE.encode() in commandline.read_bytes():
                continue
        command=[sys.executable,'-u','-m',MODULE,'worker','--config',str(bundle/'config.json'),'--lane',lane]
        with (tech/f'worker-{lane}.log').open('a') as log:
            process=subprocess.Popen(command,cwd=bundle,env={**os.environ,**environment(c,repo),
                'CUDA_VISIBLE_DEVICES':gpu},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(launch_path,dict(owner=OWNER,host=os.uname().nodename,allocation=os.environ['SLURM_JOB_ID'],
            pid=process.pid,process_group=process.pid,gpu=gpu,command=command,started_unix=time.time()))


def coordinator(c):
    if (root(c)/'technical/complete.json').exists():
        return
    devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(devices)!=2:
        raise ValueError('Handoff requires exactly two allocated GPUs')
    children=[]
    for slot,device in enumerate(devices):
        lane=f"allocation-{os.environ['SLURM_JOB_ID']}-{slot}"
        command=[sys.executable,'-u','-m',MODULE,'worker','--config',str(Path('config.json').resolve()),'--lane',lane]
        with (root(c)/'technical'/f'worker-{lane}.log').open('a') as log:
            child=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=device),stdout=log,stderr=subprocess.STDOUT)
        children.append(child)
        write_json(root(c)/'technical'/f'launch-{lane}.json',dict(owner=OWNER,pid=child.pid,
            allocation=os.environ['SLURM_JOB_ID'],host=os.uname().nodename,command=command,gpu=device))
    codes=[p.wait() for p in children]
    if any(code not in (0,124) for code in codes):
        raise RuntimeError(f'Handoff worker failure: {codes}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','launch','resume-local','worker','coordinator','status'])
    parser.add_argument('--config',required=True);parser.add_argument('--lane')
    args=parser.parse_args();c=read(args.config)
    if args.stage=='prepare':
        p=prepare_run(c);print(dict(identity=p['identity'],sources=len(p['sources']),jobs=len(p['jobs'])))
    elif args.stage=='launch':launch(c)
    elif args.stage=='resume-local':resume_local(c)
    elif args.stage=='coordinator':coordinator(c)
    elif args.stage=='status':print(json.dumps(report(c),indent=2))
    elif args.stage=='worker':
        if not args.lane:raise ValueError('Worker requires lane')
        try:worker(c,args.lane)
        except TimeoutError as error:
            print(str(error),flush=True);sys.exit(124)


if __name__=='__main__':
    main()
