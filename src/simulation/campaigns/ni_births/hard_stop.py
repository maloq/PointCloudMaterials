"""Enforce an explicitly requested Ni hold cap on receipt-owned native workers.

The frozen integrator has no live stop hook. Publish its exact saved checkpoint
prefix before terminating the matching MD child; retain actual stop overshoot
and original worker-exit receipts. Queue supervisors continue their next source.
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from src.experiment_runner.artifacts import file_hash as sha, write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue
from src.project_runtime.paths import REPO, machine, resolve_path
from src.data.trajectories.shooting import ShootingBinaryTrajectory, compose_shooting_binary_trajectories
from src.postprocessing.protocols.ni_queue import active_jobs


def read(path):
    return json.loads(Path(path).read_text())


def remote(job, config, index, action, pid=0):
    program = '''import json,os,signal,sys,time
from pathlib import Path
job,config,index,action,requested=sys.argv[1:]
matches=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or (requested!='0' and p.name!=requested):continue
 try:
  if p.stat().st_uid!=os.getuid():continue
  args=(p/'cmdline').read_bytes().decode().split('\\0')
  if 'job_'+job not in (p/'cgroup').read_text().replace('\\n','/').split('/'):continue
  if 'src.simulation.campaigns.ni_births.workflow' not in args or 'source' not in args:continue
  if args[args.index('--config')+1]!=config or args[args.index('--index')+1]!=index:continue
  matches.append(p)
 except (FileNotFoundError,ProcessLookupError):continue
if not matches:
 print(json.dumps(dict(state='absent')));sys.exit(0)
assert len(matches)==1,matches
p=matches[0];pid=int(p.name)
if action=='pause':
 os.kill(pid,signal.SIGSTOP)
 for _ in range(200):
  state=next(x.split()[1] for x in (p/'status').read_text().splitlines() if x.startswith('State:'))
  if state=='T':break
  time.sleep(.01)
 else:raise RuntimeError('MD process did not pause')
elif action=='finish':
 state=next(x.split()[1] for x in (p/'status').read_text().splitlines() if x.startswith('State:'))
 assert state=='T',state
 os.kill(pid,signal.SIGKILL)
elif action=='resume':os.kill(pid,signal.SIGCONT)
elif action=='inspect':pass
else:raise ValueError(action)
print(json.dumps(dict(state=action,pid=pid,job=job,index=int(index),config=config)))
'''
    command=['srun','--jobid='+job,'--overlap','--exact','--nodes=1','--ntasks=1',
        '--cpus-per-task=1','--gres=none','--mem=1G','--immediate=20',sys.executable,'-c',program,
        job,str(config),str(index),action,str(pid)]
    return json.loads(subprocess.check_output(command,text=True))


def configuration(settings):
    c=read(resolve_path(settings['campaign_config']));path=resolve_path(c['launch'])/'plan.json'
    if sha(path)!=settings['campaign_plan_sha256']:raise ValueError('Hard-stop campaign plan changed')
    plan=read(path)
    if plan['config']!=c:raise ValueError('Original hard-stop configuration changed')
    total=c['cooling_ps']+settings['hold_cap_ps'];cap=round(total/c['timestep_ps'])
    sample=round(c['sample_interval_ps']/c['timestep_ps']);checkpoint=round(c['checkpoint_ps']/c['timestep_ps'])
    if cap%sample or cap%checkpoint or not 0<settings['hold_cap_ps']<c['hold_ps']:
        raise ValueError('Hold cap must shorten the run at an exact observation/checkpoint boundary')
    return c,plan,cap,sample


def reconcile_exit(folder, result):
    """Keep the original nonzero worker receipt as an intentional-stop record."""
    failure=folder/'postprocessing-failure.json'
    if failure.exists():
        saved=folder/'hard-stop/worker-nonzero-exit.json'
        if not saved.exists():shutil.copy2(failure,saved)
        write_json(folder/'outcome.json',dict(result,postprocessing_state='complete',
                   worker_exit_interpretation='Intentional controller stop after verified capped publication',
                   worker_exit_receipt=str(saved)))


def finish_source(settings, record, index):
    c,plan,cap,sample=configuration(settings)
    root=resolve_path(c['scratch'])/record['id'];archive=resolve_path(c['archive'])/record['id']
    certificate=archive/'hard-stop/certificate.json'
    if (archive/'complete.json').exists():
        result=read(archive/'complete.json')
        if certificate.exists():reconcile_exit(archive,result)
        return result
    progress=archive/'progress.json'
    if not progress.exists():return None
    observed=read(progress)
    if observed['phase']=='melt' or observed['time_ps']<cap*c['timestep_ps']:return None
    started=read(root/'started.json');job=started['job']
    launch=resolve_path(settings['launch']);ownership=read(launch/'ownership.json')
    if job not in ownership['producer_jobs']:raise ValueError('Hard stop cannot control a foreign source job')
    checkpoint=archive/'checkpoints'/f'isothermal-{cap:09d}.restart'
    if not checkpoint.exists() or sha(checkpoint)!=sha(root/'checkpoints'/checkpoint.name):
        raise ValueError('Exact native cap checkpoint missing or changed')
    proof=remote(job,settings['campaign_config'],index,'pause')
    if proof['state']=='absent':return None
    control=archive/'hard-stop';control.mkdir(exist_ok=True)
    write_json(control/'paused.json',dict(proof,policy_sha256=sha(launch/'policy.json'),
               paused_at=datetime.now(timezone.utc).isoformat(),progress_before_pause=observed))
    published=False;promoted=False
    try:
        if (archive/'complete.json').exists():
            remote(job,settings['campaign_config'],index,'resume',proof['pid'])
            return read(archive/'complete.json')
        chunks=[];paths=[]
        for path in sorted((root/'chunks').glob('chunk-*/manifest.json')):
            binary=ShootingBinaryTrajectory.load(path.parent)
            if int(binary.timesteps[0])>cap:continue
            binary.verify_checksums();chunks.append(binary);paths.append(path.parent)
        actual=np.concatenate([b.timesteps[b.timesteps<=cap] for b in chunks])
        if not np.array_equal(actual,np.arange(0,cap+1,sample,dtype=np.int64)):
            raise ValueError('Saved Ni chunks do not cover every exact frame through the cap')
        metadata=read(archive/'metadata.json'); original_metadata=dict(metadata)
        structure=list(csv.DictReader((archive/'structure.csv').open()))
        within=[r for r in structure if int(r['step'])<=cap]
        if not within or int(within[-1]['step'])!=cap:raise ValueError('Native structural monitoring misses the cap')
        crossing=next((float(r['time_ps']) for r in within if float(r['crystal_fraction'])>=c['stop_crystal_fraction']),None)
        thermo=list(csv.DictReader((archive/'thermodynamics.csv').open()))
        last_logged=float(thermo[-1]['time_ps'])
        details=dict(hold_cap_ps=settings['hold_cap_ps'],total_cap_ps=cap*c['timestep_ps'],
            requested_at=read(launch/'policy.json')['created_at'],policy_sha256=sha(launch/'policy.json'),
            last_native_logged_time_ps=last_logged,controller_overshoot_ps=max(0,last_logged-cap*c['timestep_ps']),
            native_checkpoint=str(checkpoint),native_checkpoint_sha256=sha(checkpoint),
            first_half_crystal_time_ps=crossing,final_crystal_fraction=float(within[-1]['crystal_fraction']),
            followup_truncated=crossing is not None and cap*c['timestep_ps']-max(c['cooling_ps'],crossing)<c['post_stop_ps'],
            interpretation='Exact saved prefix; absence of an established birth is a valid censored observation, not a failed run')
        metadata.update(state='complete',original_hold_max_ps=c['hold_ps'],hold_max_ps=settings['hold_cap_ps'],hard_stop=details)
        target=archive/'trajectory_binary_float16';replacement=archive/'trajectory_binary_float16_hard_stop'
        compose_shooting_binary_trajectories(chunks,replacement,timesteps=range(0,cap+1,sample),storage_dtype='float16',
            provenance=dict(**metadata,measurement_duration_ps=cap*c['timestep_ps'],termination='hard_hold_cap',
                source_chunks=[dict(path=str(p),manifest_sha256=sha(p/'manifest.json'),conversion=read(p/'conversion.json')) for p in paths]))
        del chunks
        ShootingBinaryTrajectory.load(replacement).verify_checksums()
        if target.exists():target.rename(control/'interrupted-original-publication')
        replacement.rename(target);promoted=True
        for name in ['executed.in','lammps.log']:
            shutil.copy2(root/name,archive/name)
        write_json(control/'original-metadata.json',original_metadata)
        write_json(control/'certificate.json',dict(details,source=record['id'],source_process=proof,
                   controller_job=os.environ.get('SLURM_JOB_ID'),finalized_at=datetime.now(timezone.utc).isoformat()))
        result=dict(state='complete',record=record,frames=cap//sample+1,duration_ps=cap*c['timestep_ps'],
            termination='hard_hold_cap',trajectory_manifest_sha256=sha(target/'manifest.json'),
            seconds=time.time()-started['started_at'],callbacks=None,gpu=None,job=job,hard_stop=details)
        write_json(archive/'metadata.json',metadata);write_json(archive/'outcome.json',result)
        write_json(archive/'complete.json',result);published=True
        # The original queue sees a verified complete source before this child
        # exits, so it advances without copying/marking the capped MD as failed.
        remote(job,settings['campaign_config'],index,'finish',proof['pid'])
        write_json(control/'terminated.json',dict(pid=proof['pid'],job=job,verified_publication=True))
        time.sleep(2);reconcile_exit(archive,result)
        return result
    except BaseException:
        write_json(control/'failure.json',dict(traceback=traceback.format_exc(),published=published))
        if not published:
            if promoted:
                target.rename(control/'uncommitted-capped-publication')
                interrupted=control/'interrupted-original-publication'
                if interrupted.exists():interrupted.rename(target)
                write_json(archive/'metadata.json',original_metadata)
                if (archive/'outcome.json').exists():(archive/'outcome.json').rename(control/'uncommitted-outcome.json')
            remote(job,settings['campaign_config'],index,'resume',proof['pid'])
        raise


def prepare(settings):
    c,plan,cap,_=configuration(settings);launch=resolve_path(settings['launch']);launch.mkdir(parents=True,exist_ok=False)
    parent=resolve_path(c['launch']);original=read(parent/'launch.json');scaled=read(parent/'scale-01/launch.json')
    if original['identity']!=plan['identity'] or scaled['campaign_identity']!=plan['identity']:
        raise ValueError('Hard-stop source ownership changed')
    jobs=[j for name,j in original['jobs'].items() if name.startswith('worker')]
    jobs.extend(j for name,j in scaled['jobs'].items() if name.startswith('worker'))
    write_json(launch/'ownership.json',dict(campaign_identity=plan['identity'],producer_jobs=jobs,
        receipts={str(p):sha(p) for p in [parent/'launch.json',parent/'scale-01/launch.json']}))
    write_json(launch/'policy.json',dict(settings=settings,campaign_identity=plan['identity'],
        total_cap_ps=cap*c['timestep_ps'],created_at=datetime.now(timezone.utc).isoformat(),
        scope='User requested 400 ps at target temperature; 420 ps including cooling. Current-chat Ni sources only. Preserve earlier complete raw histories; all new analysis uses the same capped window.'))


def submit(settings):
    launch=resolve_path(settings['launch']);c,plan,_,_=configuration(settings)
    if (launch/'launch.json').exists():raise FileExistsError('Hard-stop controller already submitted')
    if read(launch/'policy.json')['settings']!=settings:raise ValueError('Hard-stop policy changed')
    code=ExecutionBundle.freeze(REPO,launch/'code',settings,directories=('src','configs','scripts','docs/metrics'))
    host=machine();host['catalog']=str(code.root/'configs/datasets.json');write_json(launch/'machine.json',host)
    env=dict(PCM_PROJECT_ROOT=str(REPO),PCM_MACHINE_CONFIG=str(launch/'machine.json'),PYTHONPATH=str(code.root),
             OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    receipt=dict(campaign_identity=plan['identity'],jobs={},code=str(code.root),
                 purpose='Current chat: apply user-requested 400-ps target-temperature hard cap')
    queue=SlurmQueue(launch,code,'src.simulation.campaigns.ni_births.hard_stop',env,launch/'launch.json',receipt,'NI-HARD-STOP')
    with queue.submission():queue.submit('run',['--cpus-per-task=2','--mem=16G','--time=12:00:00'])
    print(json.dumps(receipt),flush=True)


def run(settings):
    c,plan,_,_=configuration(settings);launch=resolve_path(settings['launch']);ownership=read(launch/'ownership.json')
    for path,checksum in ownership['receipts'].items():
        if sha(path)!=checksum:raise ValueError('Hard-stop job ownership receipt changed')
    with (launch/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            results=[]
            for index,record in enumerate(plan['records']):
                result=finish_source(settings,record,index)
                if result is not None:results.append(dict(source=record['id'],state=result['state'],termination=result.get('termination')))
            live=active_jobs(ownership['producer_jobs'])
            write_json(launch/'status.json',dict(state='running' if live else 'complete',sources=results,
                checked_at=datetime.now(timezone.utc).isoformat(),producer_jobs_active=live))
            if not live:break
            time.sleep(2)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=['prepare','submit','run']);p.add_argument('--config',required=True)
    a=p.parse_args();settings=read(resolve_path(a.config));globals()[a.action](settings)


if __name__=='__main__':main()
