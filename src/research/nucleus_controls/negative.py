"""Three-hour supplement using existing FastAl integrators and branch leases."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from src.data.fixed_cohort.protocol import write_json
from src.experiment_runner.execution import ExecutionBundle
from src.project_runtime.paths import REPO, machine
from src.research.nucleus_recovery import read, roots
from . import queue
from .negative_data import prepare
from .negative_analysis import summarize

MODULE='src.research.nucleus_controls.negative'


def launch(c,*,module=MODULE,preparer=prepare,summarizer=summarize):
    helper_hours=c.get('helper_wall_hours',2)
    if not 0<helper_hours<c['maximum_wall_hours']:
        raise ValueError('Helper walltime must fit inside the shared campaign budget')
    preparer(c);summarizer(c);out,_,_=roots(c)
    receipt_path=out/'technical/launch.json'
    if receipt_path.exists():raise FileExistsError('Inspect the existing owned launch before resuming')
    code=out/'technical/code'
    ExecutionBundle.freeze(REPO,code,c,directories=('src','docs/metrics','configs'),
        files=[(REPO/'machine.local.yaml','machine.local.yaml'),(REPO/'scripts/convert_trajectory.py','scripts/convert_trajectory.py')])
    start=time.time();hard_end=start+c['maximum_wall_hours']*3600
    budget=dict(submitted_unix=start,hard_end_unix=hard_end,maximum_wall_hours=c['maximum_wall_hours'],
        deadline_Paris=datetime.fromtimestamp(hard_end,ZoneInfo('Europe/Paris')).isoformat(),
        includes_queue_wait=True,automatic_extension=False)
    write_json(out/'technical/budget.json',budget)
    cuda=machine()['execution']['gpu_meam_portable_cuda_root']
    environment=dict(PCM_PROJECT_ROOT=str(REPO),CUDA_HOME=cuda,MAX_JOBS='2',TORCHINDUCTOR_COMPILE_THREADS='2',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OVITO_THREAD_COUNT='1',
        QT_QPA_PLATFORM='offscreen',UCX_TLS='tcp,self',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
        PATH=cuda+'/bin:'+str(Path(sys.executable).parent)+':/usr/bin:/bin',TZ='Europe/Paris')
    if 'PCM_MACHINE_CONFIG' in os.environ:environment['PCM_MACHINE_CONFIG']=os.environ['PCM_MACHINE_CONFIG']
    # The caller explicitly selects the currently allocated free GPU pair.
    visible=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(visible)!=c['lanes']:raise ValueError('Local launch needs exactly the configured visible GPUs')
    command=[sys.executable,'-u','-m',module,'bounded','--config',str(code/'config.json')]
    with (out/'technical/local-supervisor.log').open('a') as stream:
        p=subprocess.Popen(command,cwd=code,env={**os.environ,**environment},stdout=stream,
            stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(owner='current response/nucleus-recovery chat',started_unix=start,code=str(code),budget=budget,
        local=dict(pid=p.pid,process_group=p.pid,allocation=os.environ.get('SLURM_JOB_ID'),host=os.uname().nodename,gpus=visible),
        environment=environment)
    write_json(receipt_path,receipt)
    # Helpers additionally obey the shared absolute campaign deadline.
    deadline=datetime.fromtimestamp(hard_end,ZoneInfo('Europe/Paris')).strftime('%Y-%m-%dT%H:%M:%S')
    script=out/'technical/helpers.sbatch'
    script.write_text('\n'.join(['#!/bin/bash','#SBATCH --job-name='+c.get('job_name','AL-negative-extra'),
        '#SBATCH --partition='+c['partitions'],'#SBATCH --exclude='+c['exclude_nodes'],
        '#SBATCH --nodes=1','#SBATCH --ntasks=1','#SBATCH --cpus-per-task=4','#SBATCH --mem=24G',
        '#SBATCH --gres=gpu:1',f'#SBATCH --array=0-{c["helper_gpus"]-1}',
        f'#SBATCH --time={round(helper_hours*60)}','#SBATCH --deadline='+deadline,'#SBATCH --no-requeue',
        '#SBATCH --output='+str(out/'technical/helper-%A_%a.log'),'set -euo pipefail',
        'cd '+shlex.quote(str(code)),
        'exec env '+shlex.join([f'{k}={v}' for k,v in environment.items()]+command)
            +' --lane "$((100 + SLURM_ARRAY_TASK_ID))"','']))
    job=subprocess.check_output(['sbatch','--parsable',str(script)],text=True,env={**os.environ,'TZ':'Europe/Paris'}).strip().split(';')[0]
    if not job.isdigit():raise ValueError('Unexpected scheduler response: '+job)
    receipt['helpers']=dict(job=job,array_tasks=list(range(c['helper_gpus'])),script=str(script),allocation_hours=helper_hours)
    write_json(receipt_path,receipt);print(json.dumps(receipt,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','launch','bounded','worker','summarize'])
    parser.add_argument('--config',required=True);parser.add_argument('--lane',type=int,default=0)
    args=parser.parse_args();c=read(args.config)
    if args.stage=='prepare':prepare(c);print(json.dumps(summarize(c),indent=2))
    elif args.stage=='launch':launch(c)
    elif args.stage=='worker':queue.worker(c,args.lane,summarizer=summarize)
    elif args.stage=='bounded':
        lanes=list(range(args.lane,args.lane+len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
        queue.bounded(c,args.config,module=MODULE,summarizer=summarize,lane_ids=lanes)
    else:print(json.dumps(summarize(c),indent=2))


if __name__=='__main__':main()
