"""Start in an existing allocation and transfer one fit to queued GPUs."""
import fcntl
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time
import traceback

from src.data.fixed_cohort.protocol import digest,sha,write_json
from src.experiment_runner.metric_docs import check_metric_docs
from src.project_runtime.paths import REPO,resolve_path
from .data import config
from .rich_multimaterial_queue import verify_prepared,worker


def launch(path,allocation):
    if not allocation or os.environ.get('SLURM_JOB_ID')!=allocation:
        raise ValueError('Immediate start requires the current Slurm allocation ID')
    c=config(path);tech=resolve_path(c['output'])/'technical'
    if (tech/'launch.json').exists():raise ValueError('Run already launched')
    check_metric_docs(family=c['metric_family']);verify_prepared(c)
    if config(tech/'preflight.json')['config_sha256']!=sha(Path(path)):
        raise ValueError('Descriptor verification belongs to another recipe')
    if config(tech/'batch-candidate.json')['config_sha256']!=digest(c):
        raise ValueError('Numerical batch measurement belongs to another recipe')
    if not config(tech/'optimization-check.json')['finite']:
        raise ValueError('Inspect the numerical gradient diagnostic before starting')
    code=tech/'code';source=Path(__file__).resolve().parents[3]
    for name in ('src','docs/metrics','configs/liquid_predictability'):
        shutil.copytree(source/name,code/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.nbc','*.nbi'))
    write_json(code/'config.json',c)
    env=dict(PCM_PROJECT_ROOT=str(REPO),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        NUMBA_NUM_THREADS='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    base=[sys.executable,'-u','-m','src.research.liquid_predictability.rich_multimaterial_queue']
    r=c['runtime'];script=tech/'continue.sbatch'
    script.write_text('\n'.join(['#!/bin/bash','#SBATCH --job-name=MM-RD-continue',
        f'#SBATCH --partition={r["partition"]}','#SBATCH --nodes=1','#SBATCH --ntasks=1',
        f'#SBATCH --gpus={r["continuation_gpus"]}',f'#SBATCH --cpus-per-task={r["cpu_threads"]}',
        f'#SBATCH --mem={r["memory_GB"]}G',f'#SBATCH --time={r["walltime"]}',
        f'#SBATCH --output={tech}/continue-%j.log','set -euo pipefail','ulimit -n 4096',
        'cd '+shlex.quote(str(code)),
        'exec env '+shlex.join([f'{k}={v}' for k,v in env.items()])+' '+
        shlex.join(base+['continue','--config',str(code/'config.json')]),'']))
    job=subprocess.check_output(['sbatch','--parsable',str(script)],text=True).strip().split(';')[0]
    receipt=dict(code=str(code),config_sha256=sha(Path(path)),allocation=allocation,continuation_job=job,
        initial_gpus=r['gpus'],continuation_gpus=r['continuation_gpus'],submitted_at=time.time())
    write_json(tech/'launch.json',receipt)
    try:
        with (tech/'local-worker.log').open('a') as log:
            p=subprocess.Popen(base+['local-worker','--config',str(code/'config.json')],cwd=code,
                env=dict(os.environ,**env),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        receipt['local_pid']=p.pid;write_json(tech/'launch.json',receipt)
        # The outer launcher owns a detached Slurm step. Keep that step alive
        # until its worker exits; ending it early lets Slurm reap the fit.
        print(f'Scientific worker {p.pid}; continuation job {job}',flush=True)
        status=p.wait()
        if status:raise subprocess.CalledProcessError(status,base)
    except BaseException:
        subprocess.run(['scancel',job],check=True);raise
    return receipt


def execute(c,path,*,continuation):
    tech=resolve_path(c['output'])/'technical';done=tech.parent/'analyses/descriptor-v1/technical/complete.json'
    owner=f'{socket.gethostname()}:{os.getpid()}:{os.environ["SLURM_JOB_ID"]}'
    world=c['runtime']['continuation_gpus'] if continuation else c['runtime']['gpus']
    if continuation:
        # Reject incompatible hardware before asking a healthy local fit to stop.
        import torch
        required=config(tech/'batch-candidate.json')['minimum_device_memory_bytes']
        if torch.cuda.device_count()<world or any(
            torch.cuda.get_device_properties(i).total_memory<required for i in range(world)):
            raise RuntimeError('Continuation GPUs do not meet the measured VRAM requirement; current worker was not interrupted')
    os.environ['PCM_RICH_OWNER']=owner
    lease=tech/'worker-lease.json';request=tech/'handoff-request.json';started=time.monotonic()
    with (tech/'fit.lock').open('a') as lock:
        while True:
            if done.exists():return
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if not continuation:raise RuntimeError('Another worker already owns this scientific fit')
                if lease.exists():
                    holder=config(lease)
                    write_json(request,dict(**{'from':holder['owner']},to=owner,requested_at=time.time()))
                if time.monotonic()-started>900:
                    raise TimeoutError('Existing worker did not checkpoint/release within 15 minutes; it was not killed')
                time.sleep(2)
        receipt=dict(owner=owner,gpus=world,job=os.environ['SLURM_JOB_ID'],started_at=time.time(),state='running')
        write_json(lease,receipt)
        try:
            if continuation and (tech/'local-exit.json').exists() and config(tech/'local-exit.json')['state']=='failed':
                raise RuntimeError('Local scientific worker failed; inspect local-exit.json before resuming')
            if done.exists():return
            if not (tech/'batch-plan.json').exists():
                if continuation:raise RuntimeError('No frozen fitting subset from the initial H100 worker')
                from .rich_multimaterial_train import select_subset
                select_subset(c)
            if request.exists() and config(request)['from']==owner:
                receipt['state']='handed_off_before_training';return
            worker(c,path,'train',world)
            receipt['state']='complete' if done.exists() else 'checkpointed'
            if not done.exists():receipt['training_state']=config(tech/'state.json')
            if done.exists() and not continuation:
                job=config(tech/'launch.json')['continuation_job']
                # A completed local fit needs no queued duplicate. A running
                # waiter sees the completion marker and exits on its own.
                status=subprocess.check_output(['squeue','-h','-j',job,'-o','%T'],text=True).strip()
                if status=='PENDING':subprocess.run(['scancel',job],check=True)
        except BaseException:
            receipt.update(state='failed',traceback=traceback.format_exc());raise
        finally:
            receipt['finished_at']=time.time()
            write_json(tech/('continuation-exit.json' if continuation else 'local-exit.json'),receipt)
            lease.unlink()
            fcntl.flock(lock,fcntl.LOCK_UN)
