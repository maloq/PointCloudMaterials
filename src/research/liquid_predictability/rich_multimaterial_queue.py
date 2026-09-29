"""Freeze and submit raw multi-material descriptors and a fixed-epoch fit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from src.data.fixed_cohort.protocol import digest,sha,write_json
from src.experiment_runner.metric_docs import check_metric_docs
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue
from src.project_runtime.paths import resolve_path,REPO
from .data import config


def submit(path):
    c=config(path);tech=resolve_path(c['output'])/'technical';tech.mkdir(parents=True,exist_ok=True)
    launch=tech/'launch.json'
    if launch.exists():raise ValueError('Already submitted; use the recorded frozen jobs')
    check_metric_docs(family=c['metric_family'])
    preflight=config(tech/'preflight.json')
    if preflight['config_sha256']!=sha(Path(path)) or not preflight['finite']:raise ValueError('Run the matching local descriptor check first')
    repo = Path(__file__).resolve().parents[3]
    bundle = ExecutionBundle.freeze(
        repo, tech / 'code', c,
        directories=('src', 'docs/metrics', 'configs/liquid_predictability'),
    )
    code = bundle.root
    env=dict(PCM_PROJECT_ROOT=str(REPO),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        NUMBA_NUM_THREADS='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    receipt=dict(submitted_at=time.time(),code=str(code),config_sha256=sha(Path(path)),
        dataset_plan_sha256=sha(resolve_path(c['cache'])/'plan.json'),jobs={})
    queue = SlurmQueue(tech, bundle, 'src.research.liquid_predictability.rich_multimaterial_queue',
                       env, launch, receipt, 'MM-RD')
    def job(stage, options, dependency=None, partition='CPU'):
        return queue.submit(stage, options,
                            'afterok:' + dependency if dependency else None, partition)
    p=c['preparation'];r=c['runtime']
    with queue.submission():
        sealed=None
        if 'prepared_identity' in c:
            verify_prepared(c)
            receipt['reused_dataset_identity']=c['prepared_identity']
        else:
            prepared=job('prepare',[f'--array=0-{p["lanes"]-1}%{p["lanes"]}',f'--cpus-per-task={p["workers"]}',
                '--mem=16G',f'--time={p["walltime"]}'])
            sealed=job('seal',['--cpus-per-task=2','--mem=16G','--time=02:00:00'],prepared)
        profiled=None
        if (tech/'batch-candidate.json').exists():
            candidate=config(tech/'batch-candidate.json');check=config(tech/'optimization-check.json')
            if candidate['config_sha256']!=digest(c) or not check['finite']:
                raise ValueError('Existing numerical measurement does not match this recipe')
            receipt['batch_candidate_sha256']=sha(tech/'batch-candidate.json')
        else:
            profiled=job('probe',['--gpus=1','--cpus-per-task=4','--mem=64G','--time=02:00:00'],sealed,r['partition'])
        sized=job('subset',['--cpus-per-task=2','--mem=16G','--time=02:00:00'],profiled or sealed)
        job('train-worker',[f'--gpus={r["gpus"]}',f'--cpus-per-task={r["cpu_threads"]}',
            f'--mem={r["memory_GB"]}G',f'--time={r["walltime"]}'],sized,r['partition'])
    return receipt


def verify_prepared(c):
    """Reuse the sealed release, retaining its original producer identity."""
    root=resolve_path(c['cache']);m=config(root/'manifest.json');p=config(root/'plan.json')
    if m['state']!='complete' or m['identity']!=c['prepared_identity'] or p['identity']!=m['identity']:
        raise ValueError('Requested prepared descriptor release is incomplete or has another identity')
    if sha(root/'plan.json')!=m['plan_sha256'] or sha(root/'standardization.npz')!=m['standardization_sha256']:
        raise ValueError('Sealed descriptor release changed')
    if (p['structural_identity']!=c['structural_dataset']['identity'] or
        p['fixed_identity']!=c['fixed_dataset']['identity'] or
        p['normalization']!=c['structural_dataset']['normalization']):
        raise ValueError('Prepared geometry/ancestry contract differs from requested training')
    return m


def preflight(c,path):
    """Real descriptor integrity checks only; no tests directory or online run."""
    import numpy as np
    if 'prepared_identity' in c:
        m=verify_prepared(c)
        write_json(resolve_path(c['output'])/'technical/preflight.json',dict(
            config_sha256=sha(Path(path)),plan_identity=m['identity'],finite=True,
            reused_sealed_release=True,manifest_sha256=sha(resolve_path(c['cache'])/'manifest.json'),
            counts=m['counts'],online_runs_created=0))
        print(json.dumps(dict(reused_sealed_release=True,identity=m['identity'],counts=m['counts'])),flush=True)
        return
    from .rich_multimaterial_data import plan,prepare_task
    from .descriptors import patch_descriptors
    p=plan(c);samples=[];seen=set()
    for t in p['tasks']:
        key=t['role'],t['material']
        if key in seen:continue
        seen.add(key);raw=np.load(resolve_path(t['input']),mmap_mode='r').reshape(-1,80,3)
        ids=np.asarray(t['row_indices'][:8],np.int64) if 'row_indices' in t else np.arange(min(8,len(raw)))
        start=time.monotonic()
        for row in ids:
            values,names=patch_descriptors(np.asarray(raw[row],np.float32)*np.float32(t['factor']))
            if names!=[v['name'] for v in p['columns']] or not np.isfinite(values).all():raise ValueError('Multimaterial descriptor check failed')
        samples.append(dict(role=t['role'],material=t['material'],rows=len(ids),seconds=time.monotonic()-start,outputs=len(values)))
    # Exercise resumable shard production on one small real task. It remains
    # ordinary dataset preparation and is reused by the CPU queue.
    t=min((t for t in p['tasks'] if t['role']=='train'),key=lambda t:t['rows'])
    prepared=prepare_task((c,p['identity'],p['columns'],t))
    repeated=prepare_task((c,p['identity'],p['columns'],t))
    if not repeated['reused']:raise ValueError('Prepared shard did not resume by verified identity')
    out=resolve_path(c['output'])/'technical'
    write_json(out/'preflight.json',dict(config_sha256=sha(Path(path)),plan_identity=p['identity'],
        finite=True,samples=samples,prepared_task=prepared,counts=p['counts'],online_runs_created=0))
    print(json.dumps(dict(counts=p['counts'],descriptor_outputs=len(p['columns']),finite=True)),flush=True)


def worker(c,path,stage,gpus=None):
    command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node',str(gpus or c['runtime']['gpus']),
        '-m','src.research.liquid_predictability.rich_multimaterial_queue',stage,'--config',str(path)]
    subprocess.run(command,check=True)
    if stage=='train':
        root=resolve_path(c['output'])
        if not (root/'technical/complete.json').exists():
            if config(root/'technical/state.json')['state']=='checkpointed':return
            raise RuntimeError('Training exited before the requested epoch count without a resumable checkpoint')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['preflight','submit','prepare','seal','probe','subset','train-worker','train',
        'launch','start','local-worker','continue'])
    p.add_argument('--config',required=True);p.add_argument('--lane',type=int);p.add_argument('--allocation')
    a=p.parse_args();c=config(a.config)
    kernel_cache=resolve_path(c['runtime']['kernel_cache'])
    kernel_cache.mkdir(parents=True,exist_ok=True)
    os.environ['CUEQUIVARIANCE_OPS_NVRTC_CACHE_DIR']=str(kernel_cache)
    if a.stage=='start':
        # A detached initial worker verifies the declared batch before opening
        # a scientific online run. Kernel compilation stays outside training.
        from .rich_multimaterial_train import probe,select_subset
        from .rich_multimaterial_handoff import launch
        tech=resolve_path(c['output'])/'technical'
        try:
            write_json(tech/'state.json',dict(state='checking_declared_batch',global_batch=c['training']['batch_size']))
            preflight(c,a.config)
            worker(c,a.config,'probe',1)
            select_subset(c)
            print(json.dumps(launch(a.config,a.allocation),indent=2),flush=True)
        except BaseException:
            write_json(tech/'state.json',dict(state='startup_failed',traceback=traceback.format_exc()));raise
        return
    if a.stage in ('launch','local-worker','continue'):
        from .rich_multimaterial_handoff import launch,execute
        if a.stage=='launch':print(json.dumps(launch(a.config,a.allocation),indent=2))
        else:execute(c,a.config,continuation=a.stage=='continue')
        return
    if a.stage=='submit':print(json.dumps(submit(a.config),indent=2));return
    if a.stage=='preflight':preflight(c,a.config);return
    lane=a.lane if a.lane is not None else int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))
    state=resolve_path(c['output'])/'technical'/f'{a.stage}-{lane}.json'
    primary=os.environ.get('RANK','0')=='0'
    try:
        if primary:write_json(state,dict(state='running',job=os.environ.get('SLURM_JOB_ID')))
        if a.stage in ('prepare','seal'):
            from .rich_multimaterial_data import prepare,seal
            prepare(c,lane) if a.stage=='prepare' else seal(c)
        elif a.stage.endswith('-worker'):worker(c,a.config,a.stage.removesuffix('-worker'))
        else:
            from .rich_multimaterial_train import probe,select_subset,train
            {'probe':probe,'subset':select_subset,'train':train}[a.stage](c)
        if primary:write_json(state,dict(state='complete',finished_at=time.time()))
    except BaseException:
        if primary:write_json(state,dict(state='failed',traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
