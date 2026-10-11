"""Frozen, detached CPU/GPU MD workers with numerical gates and publication."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from src.project_runtime.paths import resolve_path
from src.research.local_response.common import read,write_json,sha
from src.experiment_runner.execution import ExecutionBundle,SlurmQueue,recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs
from src.simulation.runtime import lammps_environment


def worker(c,engine,config):
    out=resolve_path(c['output']);tech=out/'technical';archive=resolve_path(c['simulation_archive'])
    deadline=time.monotonic()+3600
    while not (archive/'preparation/complete.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Shared-liquid preparation not complete after one hour')
        time.sleep(20)
    prefix=[str(Path(sys.prefix)/'bin/mpiexec'),'-n',str(c['cpu_ranks'])] if engine=='cpu' else []
    environment=lammps_environment(hide_gpus=engine=='cpu')
    environment.update(HYDRA_BOOTSTRAP='fork',OVITO_THREAD_COUNT='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def command(module,*args):
        subprocess.run(prefix+[sys.executable,'-u','-m',__package__+'.'+module,*args,'--config',str(config)],check=True,env=environment)
    profile_receipt=out/'analyses/profile-v1/technical'/('gpu-gate.json' if engine=='gpu' else 'cpu-complete.json')
    if not profile_receipt.exists():command('profile',engine)
    command('dynamics',engine,'--nve','--duration-ps','1','--name',engine+'-nve-control')
    deadline=time.monotonic()+3600
    while not all((archive/(name+'-nve-control')/'complete.json').exists() for name in ['cpu','gpu']):
        other='cpu' if engine=='gpu' else 'gpu'
        other_stage=tech/f'stage-{other}.json'
        if other_stage.exists() and read(other_stage)['state']=='failed':raise RuntimeError(f'{other} control worker failed')
        if time.monotonic()>deadline:raise TimeoutError('Paired NVE control did not finish within one hour')
        time.sleep(10)
    from .report import conservation_gate,report
    with (tech/'conservation.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not (out/'analyses/conservation-v1/technical/gate.json').exists():conservation_gate(c)
    command('dynamics',engine)
    with (tech/'publication.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);report(c)


def launch(c):
    contracts=check_metric_docs(family='meam_md')
    out=resolve_path(c['output']);tech=out/'technical';tech.mkdir(parents=True,exist_ok=True)
    receipt_path=tech/'launch.json'
    if receipt_path.exists():raise FileExistsError('Existing launch receipt; resume recorded workers rather than duplicate jobs')
    repo=Path(__file__).resolve().parents[3]
    bundle=ExecutionBundle.freeze(repo,tech/'code',c,directories=('src','docs/metrics'),
        files=[(repo/'scripts/convert_trajectory.py','scripts/convert_trajectory.py')])
    write_json(tech/'implementation.json',dict(files=contracts['meam_md']['files'],
        config_sha256=sha(bundle.config_path),code=str(bundle.root)))
    environment=dict(PCM_PROJECT_ROOT=str(repo),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OVITO_THREAD_COUNT='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',HYDRA_BOOTSTRAP='fork')
    receipt=dict(jobs={},code=str(bundle.root));queue=SlurmQueue(tech,bundle,__package__+'.workflow',
        environment,receipt_path,receipt,'MEAM16K')
    with queue.submission():
        queue.submit('cpu',['--cpus-per-task='+str(c['cpu_ranks']),'--mem=32G','--time=08:00:00'])
    command=queue.command('gpu')
    with (tech/'gpu-worker.log').open('a') as log:
        child=subprocess.Popen(command,cwd=bundle.root,env=dict(os.environ,**environment),stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
    write_json(tech/'gpu-launch.json',dict(pid=child.pid,allocation=os.environ.get('SLURM_JOB_ID'),
        command=command,code=str(bundle.root),started_unix=time.time()))
    print(dict(**receipt,gpu_pid=child.pid),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['launch','cpu','gpu']);parser.add_argument('--config',required=True)
    args=parser.parse_args();c=read(args.config)
    check_metric_docs(family='meam_md')
    if args.stage=='launch':launch(c)
    else:
        with recorded_stage(resolve_path(c['output'])/'technical'/f'stage-{args.stage}.json',allocation=os.environ.get('SLURM_JOB_ID')):
            worker(c,args.stage,Path(args.config).resolve())


if __name__=='__main__':main()
