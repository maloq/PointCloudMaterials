"""Freeze and detach the gated CUDA MEAM numerical/MD/response campaign."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from src.experiment_runner.execution import ExecutionBundle,recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs
from src.project_runtime.paths import resolve_path,machine
from src.research.local_response.common import read,write_json,sha


def worker(c,config):
    root=resolve_path(c['output']);tech=root/'technical';archive=resolve_path(c['simulation_archive'])
    check_metric_docs(family='meam_gpu')
    def run(stage,module,*arguments):
        receipt=tech/f'stage-{stage}.json'
        if receipt.exists() and read(receipt)['state']=='complete':return
        with recorded_stage(receipt,allocation=os.environ.get('SLURM_JOB_ID')):
            command=[sys.executable,'-u','-m',__package__+'.'+module,*arguments,'--config',str(config)]
            with (tech/f'{stage}.log').open('a') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    run('periodic','benchmark','periodic')
    run('nve-v2','dynamics','--name',c['nve_name'],'--duration-ps','1','--nve')
    run('conservation','benchmark','nve')
    run('liquid-v2','dynamics','--name',c['production_name'])
    run('response-v2','benchmark','response')
    if not (archive/'native-reference/execution.json').exists():run('native-reference','native','benchmark')
    run('verified-interface-preparation','preparation')
    source=str(archive/'verified-interface-preparation/initial.npz')
    run('verified-interface-mixed','dynamics','--name','verified-interface-mixed-50ps','--initial-file',source,'--duration-ps','50')
    from .report import report
    report(c)
    run('verified-interface-f64','dynamics','--name','verified-interface-f64-50ps','--initial-file',source,'--duration-ps','50','--force-dtype','float64')
    report(c)
    write_json(tech/'campaign-complete.json',dict(state='complete',allocation=os.environ.get('SLURM_JOB_ID')))
    # This allocation was explicitly created as a resource holder for this task.
    (tech/'development-finished').touch()


def launch(c,allocation):
    contracts=check_metric_docs(family='meam_gpu');repo=Path(__file__).resolve().parents[3]
    tech=resolve_path(c['output'])/'technical';receipt=tech/'launch.json'
    if receipt.exists():raise FileExistsError('Use recorded frozen worker to resume; do not duplicate launch')
    bundle=ExecutionBundle.freeze(repo,tech/'code-v2',c,directories=('src','docs/metrics'),files=[
        (repo/'scripts/convert_trajectory.py','scripts/convert_trajectory.py'),
        (repo/c['response_reference_config'],c['response_reference_config'])])
    cuda=machine()['execution']['gpu_meam_cuda_root']
    environment=dict(PCM_PROJECT_ROOT=str(repo),PYTHONPATH=str(bundle.root),TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
        CUDA_HOME=cuda,PATH=cuda+'/bin:'+str(Path(sys.prefix)/'bin')+':/usr/bin:/bin',MAX_JOBS='2',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OVITO_THREAD_COUNT='1',TORCH_CUDA_ARCH_LIST='12.0')
    command=['srun','--jobid='+allocation,'--overlap','--ntasks=1','--cpus-per-task=4','env']+[
        f'{k}={v}' for k,v in environment.items()]+[sys.executable,'-u','-m',__package__+'.workflow','worker','--config',str(bundle.config_path)]
    with (tech/'worker-v2.log').open('a') as log:
        process=subprocess.Popen(command,cwd=bundle.root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(receipt,dict(command=command,pid=process.pid,allocation=allocation,code=str(bundle.root),
        config_sha256=sha(bundle.config_path),implementation=contracts['meam_gpu']['files']))
    print(read(receipt),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['launch','worker']);parser.add_argument('--config',required=True)
    parser.add_argument('--allocation');a=parser.parse_args();c=read(a.config)
    if a.stage=='launch':
        if a.allocation is None:raise ValueError('Explicit development GPU allocation required')
        launch(c,a.allocation)
    else:
        with recorded_stage(resolve_path(c['output'])/'technical/worker-status.json'):
            worker(c,Path(a.config).resolve())


if __name__=='__main__':main()
