"""Move queued Ni sources into an explicitly authorized existing GPU allocation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time
import traceback
from src.experiment_runner.artifacts import file_hash as sha, write_json
from src.project_runtime.paths import resolve_path, REPO


def job_state(job):
    return subprocess.check_output(['squeue','--noheader','--jobs',str(job),'--format=%T'],text=True).strip()


def gpu_idle():
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
    if len(devices)!=1:raise ValueError('Existing-allocation attachment requires one visible allocated GPU')
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if processes:raise RuntimeError(f'The allocated GPU already has compute processes: {processes}')
    return devices[0].strip()


def remote_child(job,info,config,lane,*,pause=False,resume=False):
    """Control only the receipt-owned worker; never signal its MD child."""
    program='''import json,os,signal,sys,time
from pathlib import Path
worker,child,job,config,lane,index,stop=sys.argv[1:]
p=Path('/proc')/worker
args=(p/'cmdline').read_bytes().decode().split('\\0')
assert p.stat().st_uid==os.getuid()
assert 'job_'+job in (p/'cgroup').read_text()
assert 'src.simulation.campaigns.ni_births.workflow' in args and 'worker' in args
assert args[args.index('--config')+1]==config and args[args.index('--lane')+1]==lane
q=Path('/proc')/child
state='absent'
if q.exists():
 text=(q/'status').read_text();state=next(x.split()[1] for x in text.splitlines() if x.startswith('State:'))
 assert next(x.split()[1] for x in text.splitlines() if x.startswith('PPid:'))==worker
 if state!='Z':
  a=(q/'cmdline').read_bytes().decode().split('\\0')
  assert 'source' in a and a[a.index('--index')+1]==index and a[a.index('--config')+1]==config
if stop=='yes':
 os.kill(int(worker),signal.SIGSTOP)
 for _ in range(100):
  if next(x.split()[1] for x in (p/'status').read_text().splitlines() if x.startswith('State:'))=='T':break
  time.sleep(.01)
 else:raise RuntimeError('Worker did not stop')
elif stop=='resume':os.kill(int(worker),signal.SIGCONT)
print(json.dumps(dict(worker_pid=int(worker),source_pid=int(child),child_state=state)))
'''
    command=['srun','--jobid='+job,'--overlap','--exact','--nodes=1','--ntasks=1',
        '--cpus-per-task=1','--gres=none','--mem=1G','--immediate=20',sys.executable,'-c',program,
        str(info['worker_pid']),str(info['source_pid']),job,str(config),str(lane),str(info['source_index']),
        'yes' if pause else ('resume' if resume else 'no')]
    return json.loads(subprocess.check_output(command,text=True))


def attach(c,lane,allocation,drain=None):
    if os.environ.get('SLURM_JOB_ID')!=allocation:raise ValueError('Attach must run inside the user-authorized allocation')
    gpu=gpu_idle();launch=resolve_path(c['launch']);receipt=json.loads((launch/'launch.json').read_text())
    code=Path(receipt['code']);plan=json.loads((launch/'plan.json').read_text());key=f'worker-{lane:02d}'
    if receipt['identity']!=plan['identity'] or json.loads((code/'config.json').read_text())!=c:
        raise ValueError('Ni launch/config identity changed')
    from src.simulation.campaigns.ni_births.validation import reuse
    gpu_name=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip()
    validation=reuse(c,code_root=code,gpu=gpu_name)
    previous=receipt['jobs'][key];analysis=receipt['jobs']['analyse']
    expected='RUNNING' if drain is not None else 'PENDING'
    if job_state(previous)!=expected or job_state(analysis)!='PENDING':raise ValueError('Lane/analysis scheduler state changed')
    assigned=[r['id'] for i,r in enumerate(plan['records']) if i%c['gpu_workers']==lane]
    sources=list(assigned)
    if drain is not None:
        if drain['source_index']%c['gpu_workers']!=lane:raise ValueError('Active source belongs to another lane')
        active=plan['records'][drain['source_index']]['id']
        drain=dict(drain,job=previous,source=active)
        sources=[r['id'] for i,r in enumerate(plan['records']) if i%c['gpu_workers']==lane and i>drain['source_index']]
    if not sources:raise ValueError('Empty Ni lane')
    for source in sources:
        if (resolve_path(c['scratch'])/source/'started.json').exists() or (resolve_path(c['archive'])/source).exists():
            raise ValueError(f'Cannot transfer a started Ni source: {source}')
    folder=launch/f'allocation-worker-{lane:02d}';folder.mkdir(exist_ok=False)
    entry=folder/'entry.py';shutil.copy2(Path(__file__),entry)
    validator=folder/'validation.py';shutil.copy2(Path(__file__).with_name('validation.py'),validator)
    write_json(folder/'validation-reuse.json',validation)
    # Hold our pending reservation before changing dependencies or running its
    # work elsewhere. The borrowed allocation itself is never stopped/changed.
    if drain is None:
        subprocess.run(['scontrol','hold',previous],check=True)
        if job_state(previous)!='PENDING':raise ValueError('Original lane started before it could be held')
    else:
        observed=remote_child(previous,drain,code/'config.json',lane,pause=True)
        write_json(folder/'paused-supervisor.json',dict(**drain,observed=observed,
            note='Only the waiting queue supervisor is stopped; its native MD child continues uninterrupted'))
    settings=json.loads((launch/'machine.json').read_text());cuda=settings['execution']['gpu_meam_portable_cuda_root']
    env=dict(PCM_PROJECT_ROOT=str(REPO),PCM_MACHINE_CONFIG=str(launch/'machine.json'),PYTHONPATH=str(code),
        CUDA_HOME=cuda,PATH=cuda+'/bin:'+str(Path(sys.executable).parent)+':/usr/bin:/bin',UCX_TLS='tcp,self',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MAX_JOBS='2',OVITO_THREAD_COUNT='1',QT_QPA_PLATFORM='offscreen')
    command=['srun','--jobid='+allocation,'--overlap','--exact','--nodes=1','--ntasks=1','--nodelist='+socket.gethostname(),
        '--cpus-per-task=4','--gres=gpu:1','--mem=24G','--job-name=NI-BIRTH-attached-'+str(lane),
        '--chdir='+str(code),'env']+[f'{k}={v}' for k,v in env.items()]+[
        sys.executable,'-u',str(entry),'run','--config',str(code/'config.json'),
        '--lane',str(lane),'--allocation',allocation,'--receipt',str(folder/'launch.json')]
    script=folder/'controller.sbatch'
    script.write_text('\n'.join(['#!/bin/bash','#SBATCH --job-name=NI-BIRTH-attached-controller',
        '#SBATCH --partition=CPU','#SBATCH --nodes=1','#SBATCH --ntasks=1','#SBATCH --cpus-per-task=1',
        '#SBATCH --mem=2G','#SBATCH --time=08:00:00',f'#SBATCH --output={folder}/controller-%j.log',
        'set -euo pipefail','ulimit -n 4096','exec '+shlex.join(command),'']))
    controller=subprocess.check_output(['sbatch','--hold','--parsable',str(script)],text=True).strip().split(';')[0]
    if not controller.isdigit():raise ValueError('Invalid Ni attachment controller job ID')
    attachment=dict(campaign_identity=plan['identity'],parent_launch_sha256=sha(launch/'launch.json'),
        allocation_id=allocation,node=socket.gethostname(),gpu_uuid=gpu,lane=lane,source_ids=sources,
        controller_job=controller,original_worker_job=previous,entry_sha256=sha(entry),
        validator_sha256=sha(validator),validation_reuse_sha256=sha(folder/'validation-reuse.json'),
        drain=drain,assigned_source_ids=assigned,
        created_at=datetime.now(timezone.utc).isoformat(),purpose='Current chat: user explicitly requested "Use this allocation GPU too"',
        physics='Existing frozen Ni producer/config, unchanged',time_limit='Existing allocation end, with native checkpoint reserves')
    write_json(folder/'launch.json',attachment)
    jobs=dict(receipt['jobs'])
    if drain is None:jobs[key]=controller
    else:jobs[f'attached-{lane:02d}']=controller
    dependencies=[jobs[f'worker-{i:02d}'] for i in range(c['gpu_workers'])]
    if drain is not None:dependencies.append(controller)
    dependency='afterany:'+':'.join(dependencies)
    subprocess.run(['scontrol','update','JobId='+analysis,'Dependency='+dependency],check=True)
    # Only our held, never-started GPU reservation is canceled.
    if drain is None:subprocess.run(['scancel',previous],check=True)
    write_json(launch/'dispatch.json',dict(campaign_identity=plan['identity'],parent_launch_sha256=sha(launch/'launch.json'),
        jobs=jobs,attached_workers=[dict(attachment,receipt=str(folder/'launch.json'))],analysis_dependency=dependency))
    subprocess.run(['scontrol','release',controller],check=True)
    print(json.dumps(attachment),flush=True)


def drain_source(c,path,receipt,folder):
    info=receipt['drain'];job=info['job']
    while job_state(job)=='RUNNING':
        observed=remote_child(job,info,path,receipt['lane'])
        write_json(folder/'drain-progress.json',dict(source=info['source'],**observed,checked_at=datetime.now(timezone.utc).isoformat()))
        if observed['child_state'] in ('absent','Z'):break
        time.sleep(30)
    archive=resolve_path(c['archive'])/info['source'];done=archive/'complete.json'
    if done.exists():
        from src.data.trajectories.shooting import ShootingBinaryTrajectory
        result=json.loads(done.read_text());trajectory=archive/'trajectory_binary_float16'
        if result['job']!=job or sha(trajectory/'manifest.json')!=result['trajectory_manifest_sha256']:
            raise ValueError('Drained source completion identity changed')
        ShootingBinaryTrajectory.load(trajectory).verify_checksums()
    else:
        # A native failure still preserves its checkpoints and observation chunks.
        staging=resolve_path(c['scratch'])/info['source']
        shutil.copytree(staging,archive/'failed-staging',dirs_exist_ok=True)
        result=dict(state='failed',id=info['source'],job=job,reason='Native source exited without completion during queue drain')
        write_json(archive/'outcome.json',result)
    write_json(folder/'drained-source.json',result)
    if job_state(job)=='RUNNING':subprocess.run(['scancel',job],check=True)
    return result


def source_queue(c,path,receipt,folder):
    from src.experiment_runner.execution import allocation_deadline
    launch=resolve_path(c['launch']);plan=json.loads((launch/'plan.json').read_text())
    deadline=allocation_deadline(reserve_seconds=600);results=[]
    for index,record in enumerate(plan['records']):
        if record['id'] not in receipt['source_ids']:continue
        if time.time()+600>deadline:
            results.append(dict(id=record['id'],state='not_started',reason='Allocation reserve'));continue
        command=[sys.executable,'-u','-m','src.simulation.campaigns.ni_births.workflow','source',
            '--config',str(path),'--index',str(index),'--deadline',str(deadline)]
        with (launch/(record['id']+'.log')).open('a') as log:
            process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        archive=resolve_path(c['archive'])/record['id']
        if process.returncode:
            failure=dict(id=record['id'],state='failed',exit_code=process.returncode,log=str(launch/(record['id']+'.log')))
            if (archive/'complete.json').exists():
                write_json(archive/'postprocessing-failure.json',failure)
                result=dict(json.loads((archive/'complete.json').read_text()),postprocessing_state='failed')
            else:
                result=failure
                shutil.copytree(resolve_path(c['scratch'])/record['id'],archive/'failed-staging',dirs_exist_ok=True)
            write_json(archive/'outcome.json',result)
        else:result=json.loads((archive/'outcome.json').read_text())
        results.append(result)
        write_json(folder/'source-progress.json',dict(outcomes=results,source_ids=receipt['source_ids']))
    return results


def run(c,path,lane,allocation,receipt_path):
    folder=receipt_path.parent;receipt=json.loads(receipt_path.read_text())
    state=dict(allocation_id=allocation,step_id=os.environ.get('SLURM_STEP_ID'),lane=lane,
        controller_job=receipt['controller_job'],pid=os.getpid(),node=socket.gethostname(),
        started_at=datetime.now(timezone.utc).isoformat(),entry_sha256=sha(Path(__file__)))
    try:
        if os.environ.get('SLURM_JOB_ID')!=allocation or state['node']!=receipt['node']:
            raise ValueError('Ni worker entered a different allocation/node')
        if state['entry_sha256']!=receipt['entry_sha256'] or gpu_idle()!=receipt['gpu_uuid']:
            raise ValueError('Ni attachment entry or allocated GPU changed')
        state.update(state='validating',gpu_uuid=receipt['gpu_uuid']);write_json(folder/'execution.json',state)
        if (sha(folder/'validation.py')!=receipt['validator_sha256']
                or sha(folder/'validation-reuse.json')!=receipt['validation_reuse_sha256']):
            raise ValueError('Ni validation reuse evidence changed')
        # This helper is copied beside entry.py, including when production uses
        # an older frozen producer. No new LAMMPS/CUDA validation is executed.
        from validation import verify
        gpu_name=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip()
        proof=json.loads((folder/'validation-reuse.json').read_text())
        validation=verify(c,proof,code_root=path.parent,gpu=gpu_name)
        state.update(state='running',gpu=gpu_name,audit_sha256=proof['validation_sha256'],validation_state='reused')
        write_json(folder/'execution.json',state)
        with ThreadPoolExecutor(max_workers=1) as pool:
            draining=pool.submit(drain_source,c,path,receipt,folder) if receipt['drain'] is not None else None
            results=source_queue(c,path,receipt,folder)
            if draining is not None:results.append(draining.result())
        result=dict(state='complete' if all(r['state']=='complete' for r in results) else 'incomplete',outcomes=results)
        state.update(state=result['state'],finished_at=datetime.now(timezone.utc).isoformat(),worker_status=result)
        write_json(folder/'execution.json',state)
    except BaseException:
        state.update(state='failed',traceback=traceback.format_exc(),finished_at=datetime.now(timezone.utc).isoformat())
        if receipt['drain'] is not None and not any((resolve_path(c['scratch'])/s/'started.json').exists() for s in receipt['source_ids']):
            info=receipt['drain']
            if job_state(info['job'])=='RUNNING':
                remote_child(info['job'],info,path,lane,resume=True)
                state['original_queue_resumed']=True
        write_json(folder/'execution.json',state)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=['attach','run'])
    p.add_argument('--config',required=True);p.add_argument('--lane',type=int,required=True)
    p.add_argument('--allocation',required=True);p.add_argument('--receipt',type=Path)
    p.add_argument('--worker-pid',type=int);p.add_argument('--source-pid',type=int);p.add_argument('--source-index',type=int)
    a=p.parse_args();path=resolve_path(a.config);c=json.loads(path.read_text())
    if a.action=='attach':
        drain=None if a.worker_pid is None else dict(worker_pid=a.worker_pid,source_pid=a.source_pid,source_index=a.source_index)
        attach(c,a.lane,a.allocation,drain)
    else:run(c,path,a.lane,a.allocation,a.receipt)
