"""Owned, frozen training tasks with resumable spatial-to-motion dependencies."""
import argparse
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from src.experiment_runner.execution import ExecutionBundle,SlurmQueue,allocation_deadline,recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs
from src.project_runtime.paths import REPO
from .data import FAMILY,read,root,cache,digest,sha,write_json,resolve_path,data_identity

MODULE='src.research.structural_response.workflow'
OWNER='current response/nucleus-recovery chat; user authorized12-epoch spatial starts and derivative motion overnight on2026-10-07'


@contextmanager
def lease(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            yield False;return
        try:yield True
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


def environment():
    return dict(PCM_PROJECT_ROOT=str(REPO),TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OVITO_THREAD_COUNT='1',CUEQUIVARIANCE_OPS_PARALLEL_THREADS='4',PYTHONWARNINGS='ignore::FutureWarning',PYTORCH_ALLOC_CONF='expandable_segments:True')


def bind(c,require_probe=True):
    check_metric_docs(family=FAMILY)
    done=read(cache(c)/'complete.json')
    if done['data_identity']!=data_identity(c):raise ValueError('Prepared data does not match recipe')
    if any(sha(cache(c)/f)!=h for f,h in done['files'].items()):raise ValueError('Prepared data checksum changed')
    if require_probe:
        probe=read(root(c)/'technical'/('probe.json' if 'spatial_initialization' in c else 'spatial-probe.json'))
        if not probe['passed'] or probe['encoder']!=c['encoder']:raise ValueError('Neural numerical check missing or backend changed')
        if 'spatial_initialization' in c and (probe['config_identity']!=digest(c) or probe['model_sha256']!=sha(Path(__file__).with_name('model.py'))):
            raise ValueError('Motion numerical check does not match the current recipe and derivative implementation')
        if c.get('neural_derivative')=='centered_student_v1' and (probe['fast_sha256']!=sha(Path(__file__).with_name('fast.py')) or probe['training_sha256']!=sha(Path(__file__).with_name('train.py'))):
            raise ValueError('Fast student numerical check does not match training implementation')
    if 'spatial_initialization' in c:
        from .train import spatial_start
        for method in c['methods']:spatial_start(c,method)
    if 'birth_evaluation' in c:
        from .birth_data import prepare as prepare_births
        birth_cohort=prepare_births(c)
        if c.get('neural_derivative')=='centered_student_v1':
            inputs=read(resolve_path(c['birth_evaluation']['cache'])/'complete.json')
            if inputs['cohort']!=birth_cohort['identity'] or inputs['neighborhood_filter'] is not False:
                raise ValueError('Birth inputs do not match the unfiltered catalogue contract')
        if c['response_training']['replay_weight']!=0 or c['response_training']['replay_every']!=0:
            raise ValueError('New motion revision excludes broad structural replay')
    identity=dict(config=c,data_sha256=sha(cache(c)/'complete.json'),implementations=check_metric_docs(family=FAMILY)[FAMILY]['files'])
    identity['identity']=digest(identity);p=root(c)/'technical/plan.json'
    if p.exists() and read(p)!=identity:raise ValueError('Changed frozen campaign; choose another revision')
    write_json(p,identity)
    fast=c.get('neural_derivative')=='centered_student_v1'
    write_json(root(c)/'technical/prediction-context.json',dict(
        encoder=dict(positions='unrelaxed nearest80, physical Al-equivalent coordinates, centered; no halo',channels=128,embedding=128,correlation=3,max_ell=3,interactions=2,cutoff_A=5,support_A=8,motion='centered student secant approximating0.1ps*DE(Q)[V]' if fast else 'none for structural E; motion state appends0.1ps*DE(Q)[V]',conditions=[],species='one constant channel plus center flag',relaxation=False),
        spatial_teacher=dict(bond_order=40,tda=258,other_descriptors=False,crystal_labels=False,masked='20% removal outside nearest12 neighbors; EMA complete-view embedding; no temporal target'),
        motion_predictor=dict(inputs='structural128 + directional-motion128',conditions=[],time='integration interval only',targets='24 fixed future descriptors and128 frozen spatial-teacher coordinates at0.1/0.5/3/12ps'),
        derivative_teacher=dict(bank='reused176 full-cell FastAl response banks; no new MD',duration_ps=.1,shots=2,position_direction_rms_A=.01,velocity_direction_rms_A_ps=.1,physical_integration='float64; forcesfloat32; recorded fixed-box BAOAB versus saved NPT futures remain separate',latent_teacher='fixed pretrained structural encoder; no moving target in motion fits'),
        precision=dict(spatial_and_latent_dynamics='BF16 autocast',geometry_normalization_loss_optimizer='float32',motion_encoder_JVP='float32 centered student differences; fused cuEq forwards and ordinary parameter backward' if fast else 'float32 exact reverse-over-reverse AD with ordinary PyTorch contraction polynomials',microbatch=c['response_training']['microbatch'],smaller_gpu_microbatch=c['runtime']['small_motion_microbatch'],effective_batch=256,nested_neural_derivatives='none through MACE in fast student' if fast else 'float32',physical_bank='original float64 states/tangents retained'),
        student_difference=c.get('student_difference'),
        spatial_initialization=c.get('spatial_initialization'),
        split='Original Al64 roles. Spatial fitting only original training ancestors and external train-only data. Spatial export fixed epoch12; validation diagnostic only. Motion selector uses inner original-training ancestors. Original selection/calibration/test pooled for final future evaluation.',
        stages='Two reusable spatial checkpoints, then matched value/derivative x cached-backbone-adapter/full-MACE motion fits.' if fast else 'Two reusable spatial checkpoints, then matched value/derivative x frozen/joint motion fits.',
        spatial_replay=dict(weight=c['response_training']['replay_weight'],interval=c['response_training']['replay_every'],
            broad_dataset_opened=c['response_training']['replay_weight']>0),
        birth_evaluation=c.get('birth_evaluation'),
        scope='One initialization seed; exploratory motion utility, not a seed-replicated claim;12 complete spatial epochs mandatory.'))
    return identity


def worker(c,kind,method,lane,micro):
    import torch
    from .train import spatial,motion,tech,signature,spatial_start
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    deadline=allocation_deadline(reserve_seconds=180)
    launch=root(c)/'technical'/f'worker-{lane}-{os.getpid()}.json'
    write_json(launch,dict(owner=OWNER,kind=kind,method=method,pid=os.getpid(),allocation=os.environ.get('SLURM_JOB_ID'),host=os.uname().nodename,visible=os.environ.get('CUDA_VISIBLE_DEVICES'),microbatch=micro,started_unix=time.time(),deadline_unix=deadline,command=sys.argv))
    # A reservation can precede the local numerical check. If the interactive
    # allocation ended mid-check, finish it here before scientific training.
    probe_path=root(c)/'technical'/('spatial-probe.json' if kind=='spatial' else 'probe.json')
    while not probe_path.exists():
        with lease(root(c)/'technical/probe.lock') as acquired:
            if acquired:
                if not probe_path.exists():
                    if c.get('neural_derivative')=='centered_student_v1':
                        from .fast_probe import run
                    else:
                        from .probe import run
                    run(c,spatial_only=kind=='spatial')
            else:
                if time.time()>deadline-600:raise TimeoutError('Numerical preparation did not finish')
                time.sleep(5)
    probe=read(probe_path)
    if not probe['passed'] or probe['encoder']!=c['encoder']:raise ValueError('Invalid numerical preparation')
    tasks=[('spatial-'+method,None)] if kind=='spatial' else [(method+'-'+a['name'],a) for a in c['response_arms']]
    if kind=='motion':
        # The nested live-encoder derivative is the largest graph even for one
        # response record. Keep it on >=64 GiB devices; smaller GPUs still run
        # both frozen controls and the joint value control at half microbatch.
        large=torch.cuda.get_device_properties(0).total_memory>=64*2**30
        fast=c.get('neural_derivative')=='centered_student_v1'
        tasks=[t for t in tasks if fast or large or not (t[1]['derivative'] and not t[1]['frozen'])]
        if large:tasks.sort(key=lambda t:not (t[1]['derivative'] and not t[1]['frozen']))
        elif fast:tasks.sort(key=lambda t:not t[1]['frozen'])
    if kind=='motion' and not (spatial_start(c,method)/'complete.json').exists():raise RuntimeError('Spatial12-epoch dependency is incomplete')
    while time.time()<deadline-300:
        pending=False;worked=False
        for name,arm in tasks:
            p=tech(c,name)
            if (p/'complete.json').exists():
                if read(p/'complete.json')['identity']!=signature(c,name):raise ValueError('Changed completed fit')
                continue
            pending=True
            with lease(p/'fit.lock') as acquired:
                if not acquired:continue
                if (p/'complete.json').exists():continue
                with recorded_stage(p/('execution-'+str(os.getpid())+'.json'),owner=OWNER,task=name,lane=lane,microbatch=micro):
                    finished=spatial(c,method,deadline,micro) if arm is None else motion(c,method,arm,deadline,micro)
                worked=True
                if not finished:
                    # Only this workflow's batch job may requeue itself. Local
                    # workers never change the user's containing allocation.
                    if not lane.startswith('local-'):
                        job=os.environ['SLURM_JOB_ID']
                        owned=read(root(c)/'technical/launch.json')['jobs']
                        if job not in owned.values():
                            raise RuntimeError('Cannot establish self-requeue ownership')
                        write_json(p/'requeue.json',dict(job=job,owner=OWNER,time=time.time()))
                        subprocess.run(['scontrol','requeue',job],check=True)
                    return
                torch.cuda.empty_cache()
        if not pending:return
        if not worked:return


def start_local(c):
    bind(c);p=root(c)/'technical';receipt=p/'launch.json';record=read(receipt)
    current=os.environ.get('SLURM_JOB_ID')
    if not current:raise RuntimeError('Local starts require the current allocation')
    if any(r['allocation']==current for r in record['local_workers']):raise RuntimeError('Local workers already launched in this allocation')
    if allocation_deadline(reserve_seconds=300)-time.time()<=360:
        print('Current allocation too short for useful fitting; queued workers will start from initialization.',flush=True);return
    bundle=ExecutionBundle(Path(record['code']));env=environment()
    motion_only='spatial_initialization' in c
    for i,method in enumerate(c['methods'][:len(c['runtime']['local_gpus'])]):
        kind='motion' if motion_only else 'spatial';micro=c['response_training']['microbatch'] if motion_only else 256
        command=[sys.executable,'-u','-m',MODULE,'worker','--config',str(bundle.config_path),'--kind',kind,'--method',method,'--lane','local-'+str(i),'--microbatch',str(micro)]
        log=(p/f'local-{i}.log').open('a')
        child=subprocess.Popen(command,cwd=bundle.root,env=dict(os.environ,**env,CUDA_VISIBLE_DEVICES=c['runtime']['local_gpus'][i]),stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
        record['local_workers'].append(dict(pid=child.pid,process_group=child.pid,method=method,gpu=c['runtime']['local_gpus'][i],allocation=current,command=command));write_json(receipt,record)
    print(record['local_workers'],flush=True)


def launch(c,reserve_only=False):
    bind(c,require_probe=not reserve_only);p=root(c)/'technical';receipt=p/'launch.json'
    if receipt.exists():raise FileExistsError('Campaign already launched')
    dependencies=check_metric_docs(family=FAMILY)[FAMILY]['files']
    extra_files=[(REPO/name,name) for name in dependencies
                 if not name.startswith(('src/','docs/metrics/'))]
    bundle=ExecutionBundle.freeze(REPO,p/'code',c,directories=('src','docs/metrics'),files=extra_files)
    subprocess.run([sys.executable,'-c',
        "from src.experiment_runner.metric_docs import check_metric_docs; check_metric_docs(family='structural_response')"],
        cwd=bundle.root,check=True)
    record=dict(owner=OWNER,code=str(bundle.root),jobs={},local_workers=[],data_sha256=sha(cache(c)/'complete.json'),started_unix=time.time())
    write_json(receipt,record);env=environment()
    queue=SlurmQueue(p,bundle,MODULE,env,receipt,record,'SRESP')
    current=os.environ.get('SLURM_JOB_ID')
    with queue.submission():
        for i,method in enumerate(c['methods']):
            task=None
            if 'spatial_initialization' not in c:
                task=queue.submit('spatial-'+str(i),['--gpus=1','--cpus-per-task=4','--mem=32G','--time=24:00:00','--requeue'],
                    dependency='afterany:'+current if current else None,partition='RTX6000PRO,H100',command_stage='worker',arguments=('--kind','spatial','--method',method,'--lane','spatial-'+str(i),'--microbatch','256'))
            # Each worker leases an unfinished fit; smaller GPU queues are separate.
            for j,partition in enumerate(c['runtime']['partitions']):
                dependency='afterok:'+task if task else None
                # One successor for each local lane prevents an unfinished local
                # fit being stranded after the containing allocation expires.
                if task is None and current and i<len(c['runtime']['local_gpus']) and j==0:dependency='afterany:'+current
                extra=['--exclude='+','.join(c['runtime']['exclude_nodes'])] if c['runtime'].get('exclude_nodes') else []
                queue.submit('motion-'+str(i)+'-'+str(j),['--gpus=1',f"--cpus-per-task={c['runtime']['motion_cpu_threads']}",f"--mem={c['runtime']['motion_memory_GB']}G",'--time=24:00:00','--requeue',*extra],
                    dependency=dependency,partition=partition,command_stage='worker',arguments=('--kind','motion','--method',method,'--lane','motion-'+str(i)+'-'+str(j),'--microbatch',str(c['response_training']['microbatch'] if j==0 else c['runtime']['small_motion_microbatch'])))
    print(record,flush=True)
    if not reserve_only and current:start_local(c)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['prepare','prepare-births','birth-inputs','probe','launch','reserve','start-local','worker']);p.add_argument('--config',required=True)
    p.add_argument('--kind',choices=['spatial','motion']);p.add_argument('--method');p.add_argument('--lane');p.add_argument('--microbatch',type=int,default=256);a=p.parse_args();c=read(a.config)
    if a.stage=='prepare':
        from .data import prepare
        prepare(c)
    elif a.stage=='prepare-births':
        from .birth_data import prepare
        result=prepare(c);print(dict(identity=result['identity'],**result['counts']),flush=True)
    elif a.stage=='birth-inputs':
        from .birth import prepare_observations
        print(prepare_observations(c),flush=True)
    elif a.stage=='probe':
        if c.get('neural_derivative')=='centered_student_v1':
            from .fast_probe import run
        else:
            from .probe import run
        run(c)
    elif a.stage in ('launch','reserve'):launch(c,reserve_only=a.stage=='reserve')
    elif a.stage=='start-local':start_local(c)
    else:worker(c,a.kind,a.method,a.lane,a.microbatch)


if __name__=='__main__':main()
