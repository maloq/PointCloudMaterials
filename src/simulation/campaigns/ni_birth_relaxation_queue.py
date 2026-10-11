"""Four-worker shared dispatch over an unchanged, frozen Ni quench producer.

Replacement workers acquire a GPU before retiring their receipt-owned original
lane. Until that handoff completes, its tasks are ineligible for other workers.
"""
import argparse
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

from src.experiment_runner.artifacts import file_hash as sha, json_digest, write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, allocation_deadline
from src.project_runtime.paths import REPO, resolve_path
from src.data.relaxed_targets.worker import lock, publish
from src.simulation.campaigns.ni_birth_relaxation import read, load_plan, produce


def active_jobs():
    output = subprocess.check_output(
        ['squeue', '--noheader', '--user', os.environ['USER'], '--format=%i|%T'], text=True)
    return dict(line.strip().split('|', 1) for line in output.splitlines() if line.strip())


def verify_receipt(path):
    receipt = read(path)
    if sha(receipt['parent_launch']) != receipt['parent_launch_sha256']:
        raise ValueError('Original Ni relaxation ownership receipt changed')
    if sha(receipt['entry']) != receipt['entry_sha256']:
        raise ValueError('Dispatch implementation changed after submission')
    original = read(receipt['parent_launch'])
    if original['identity'] != receipt['identity'] or original['code'] != receipt['code']:
        raise ValueError('Original frozen Ni relaxation identity changed')
    for lane, job in enumerate(receipt['old_workers']):
        if original['jobs'][f'worker-{lane}'] != job:
            raise ValueError('Replacement job is not owned by this relaxation launch')
    return receipt


def preserve_interrupted(plan, lane, folder):
    """Retain partial artifacts; a published native minimum remains reusable."""
    c = plan['config']; tech = resolve_path(c['output'])/'technical'
    records = []
    for task in plan['tasks'][lane::2]:
        if (tech/'cells'/(task['id']+'.json')).exists():
            continue
        work = resolve_path(c['scratch'])/'cells'/task['id']
        archive = resolve_path(c['archive'])/'cells'/task['id']
        record = dict(task=task['id'])
        partial = archive.with_name(archive.name+'.partial')
        if partial.exists():
            retained = resolve_path(c['archive'])/'handoff-partial-publications'/task['id']
            retained.parent.mkdir(parents=True, exist_ok=True)
            partial.rename(retained)
            record['partial_publication'] = str(retained)
        if work.exists() and not archive.exists():
            retained = resolve_path(c['scratch'])/'handoff-interrupted'/task['id']
            retained.parent.mkdir(parents=True, exist_ok=True)
            work.rename(retained)
            destination = resolve_path(c['archive'])/'handoff-interrupted'/task['id']
            publish(retained, destination)
            record['archived_interruption'] = str(destination)
        if len(record)>1:
            records.append(record)
    write_json(folder/f'interrupted-{lane}.json', records)
    return records


def handoff(plan, receipt_path, lane):
    receipt = verify_receipt(receipt_path); folder = Path(receipt_path).parent
    marker = folder/f'handoff-{lane}.json'
    with lock(folder/f'handoff-{lane}.lock') as acquired:
        if not acquired:
            raise RuntimeError('A second worker attempted the same lane handoff')
        if marker.exists():
            return
        job = receipt['old_workers'][lane]
        proof = dict(original_job=job, replacement_job=os.environ['SLURM_JOB_ID'],
                     identity=plan['identity'], started_at=datetime.now(timezone.utc).isoformat())
        running = active_jobs()
        if job in running:
            record = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
            expected = str(Path(receipt['parent_launch']).parent/f'worker-{lane}.sbatch')
            if f'JobName=NI-RELAX-worker-{lane} ' not in record or f'Command={expected} ' not in record:
                raise ValueError('Original Slurm job does not match the owned launch script')
            proof['scheduler_before'] = record
            write_json(folder/f'handoff-intent-{lane}.json', proof)
            subprocess.run(['scancel', job], check=True)
            deadline = time.monotonic()+180
            while job in active_jobs():
                if time.monotonic()>deadline:
                    raise RuntimeError('Owned original worker did not exit; no tasks released')
                time.sleep(2)
        proof['retained_interruptions'] = preserve_interrupted(plan, lane, folder)
        proof.update(state='released', finished_at=datetime.now(timezone.utc).isoformat())
        write_json(marker, proof)


def worker(c, receipt_path, lane):
    receipt=verify_receipt(receipt_path)
    if 'optimized_audit' not in receipt:
        return _worker(c,receipt_path,lane)
    from src.simulation.eam_gpu.relaxation_fast import implementation
    from src.simulation.eam_gpu.relaxation_service import RelaxationService
    if sha(receipt['optimized_audit'])!=receipt['optimized_audit_sha256']:
        raise ValueError('Optimized execution audit changed')
    audit=read(receipt['optimized_audit'])
    if audit['state']!='passed' or audit['implementation']!=implementation():
        raise ValueError('Optimized implementation does not match its passed audit')
    for relative, expected in receipt['execution_files'].items():
        if sha(Path(receipt['execution_code'])/relative)!=expected:
            raise ValueError(f'Frozen optimized producer changed: {relative}')
    with RelaxationService() as runner:
        return _worker(c,receipt_path,lane,runner=runner,execution_settings=audit['settings'])


def _worker(c, receipt_path, lane, *, runner=None, execution_settings=None):
    receipt=verify_receipt(receipt_path); plan=load_plan(c)
    if plan['identity']!=receipt['identity']:
        raise ValueError('Shared queue changed the original scientific plan')
    if os.environ.get('SLURM_JOB_ID')!=receipt['jobs'][f'worker-{lane}']:
        raise ValueError('Worker is outside its recorded allocation')
    folder=Path(receipt_path).parent; tech=resolve_path(c['output'])/'technical'
    if lane<2:
        try:
            handoff(plan,receipt_path,lane)
        except Exception:
            write_json(folder/f'handoff-failure-{lane}.json',dict(error=traceback.format_exc(),job=os.environ['SLURM_JOB_ID']))
            raise
    deadline=allocation_deadline(reserve_seconds=120)
    failed=set();completed=0;started=time.monotonic()
    # Give each worker a preferred quarter; every unlocked task is stealable.
    indexed=list(enumerate(plan['tasks']))
    workers=len(receipt['jobs'])
    ordered=indexed[lane::workers]+[(i,t) for i,t in indexed if i%workers!=lane]
    remaining=ordered
    record=dict(job=os.environ['SLURM_JOB_ID'],host=socket.gethostname(),lane=lane,
                identity=plan['identity'],producer=receipt['code'])
    while remaining:
        if list(folder.glob('handoff-failure-*.json')):
            raise RuntimeError('A recorded lane handoff failed; review it before further dispatch')
        if time.time()+c['relaxation']['frame_timeout_seconds']>=deadline:
            raise RuntimeError('Shared queue reached its allocation deadline')
        allowed={old for old in range(2) if (folder/f'handoff-{old}.json').exists()}
        if not allowed:
            time.sleep(2);continue
        next_pass=[];did_work=False
        for index,task in remaining:
            done=tech/'cells'/(task['id']+'.json')
            if done.exists() or task['id'] in failed:
                continue
            if index%2 not in allowed:
                next_pass.append((index,task));continue
            if time.time()+c['relaxation']['frame_timeout_seconds']>=deadline:
                write_json(folder/f'worker-{lane}.json',dict(record,state='deadline',completed=completed))
                raise RuntimeError('Shared Ni relaxation queue reached its allocation deadline')
            with lock(tech/'locks'/(task['id']+'.lock')) as acquired:
                if not acquired:
                    continue  # Another worker owns this task through publication.
                if done.exists():
                    continue
                begin=time.monotonic()
                write_json(folder/f'worker-{lane}.json',dict(record,state='running',task=task['id'],completed=completed))
                try:
                    result=(produce(plan,task) if runner is None else
                        produce(plan,task,execution_settings=execution_settings,runner=runner))
                    completed+=1;did_work=True
                    print(json.dumps(dict(task=task['id'],completed=completed,
                        seconds=result['seconds'],wall_seconds=time.monotonic()-begin)),flush=True)
                except Exception:
                    failure=dict(task=task,error=traceback.format_exc(),job=record['job'])
                    write_json(tech/'failures'/(task['id']+'.json'),failure)
                    print(json.dumps(failure),flush=True);failed.add(task['id'])
                    if len(failed)>=3:
                        write_json(folder/f'worker-{lane}.json',dict(record,state='failed',completed=completed,failures=sorted(failed)))
                        raise RuntimeError('Three quench failures; stopping this shared worker')
        remaining=next_pass
        if remaining and not did_work:
            # The other original lane can still be computing while its GPU
            # replacement waits in the scheduler. Recheck its release marker.
            time.sleep(5)
    write_json(folder/f'worker-{lane}.json',dict(record,state='drained',completed=completed,
        failures=sorted(failed),seconds=time.monotonic()-started))
    if failed:raise RuntimeError('Failed Ni cells remain in the completion report')


def submit(c, folder, *, optimized_audit=None, workers=4):
    if workers not in (2,4):raise ValueError('The two original lanes require two or four replacements')
    plan=load_plan(c); tech=resolve_path(c['output'])/'technical'
    parent=tech/'launch.json';original=read(parent);code=Path(original['code'])
    if original['identity']!=plan['identity'] or read(code/'config.json')!=c:
        raise ValueError('Original launch/config does not match the scientific plan')
    proof=read(tech/'audit.json')
    if proof['state']!='passed' or proof['identity']!=plan['identity']:
        raise ValueError('Original physical audit has not passed')
    old=[original['jobs'][f'worker-{lane}'] for lane in range(2)]
    states=active_jobs()
    if any(states.get(job)!='RUNNING' for job in old) or states.get(original['jobs']['status'])!='PENDING':
        raise ValueError('Expected two original running workers and a pending status job')
    folder.mkdir(parents=True,exist_ok=False)
    numerical_code=code
    if optimized_audit:
        from src.simulation.eam_gpu.relaxation_fast import implementation
        audit=read(optimized_audit)
        if audit['state']!='passed' or audit['implementation']!=implementation():
            raise ValueError('Current optimized code has no matching passed numerical audit')
        # Bind the new callback/recipe without rewriting the original population,
        # frozen code, settings, or completed results.
        numerical_code=ExecutionBundle.freeze(REPO,folder/'code',c,
            directories=('src','configs','scripts','docs/metrics')).root
    entry=folder/'entry.py';shutil.copy2(Path(__file__),entry)
    host=read(tech/'machine.json');cuda=host['execution']['gpu_meam_portable_cuda_root']
    env=dict(PCM_PROJECT_ROOT=str(numerical_code),PCM_MACHINE_CONFIG=str(tech/'machine.json'),PYTHONPATH=str(numerical_code),
        CUDA_HOME=cuda,PATH=cuda+'/bin:'+str(Path(sys.executable).parent)+':/usr/bin:/bin',
        UCX_TLS='tcp,self',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MAX_JOBS='2')
    receipt=dict(identity=plan['identity'],parent_launch=str(parent),parent_launch_sha256=sha(parent),
        code=str(code),entry=str(entry),entry_sha256=sha(entry),old_workers=old,jobs={},
        created_at=datetime.now(timezone.utc).isoformat(),
        purpose='Current chat: speed up the existing Ni relaxation using four GPUs and shared tasks; unchanged frozen numerical producer',
        completed_at_submission=len(list((tech/'cells').glob('*.json'))),state='preparing')
    if optimized_audit:
        receipt.update(optimized_audit=str(optimized_audit),optimized_audit_sha256=sha(optimized_audit),
            execution_code=str(numerical_code),implementation=audit['implementation'],
            execution_files={relative:sha(numerical_code/relative) for relative in (
                'src/simulation/relaxation.py','src/simulation/eam_gpu/relaxation_service.py',
                'src/simulation/campaigns/ni_birth_relaxation.py')},
            purpose='Current chat: software speedup of fixed-box Ni FIRE, persistent CUDA callback; unchanged population and physical tolerance')
    target=folder/'launch.json';write_json(target,receipt)
    queue=SlurmQueue(folder,ExecutionBundle(numerical_code),'src.simulation.campaigns.ni_birth_relaxation',
                     env,target,receipt,'NI-RELAX-SHARED')
    with queue.submission():
        for lane in range(workers):
            stage=f'worker-{lane}'
            script=queue.render(stage,['--gres=gpu:1','--cpus-per-task=4','--mem=16G',
                '--time=16:00:00','--exclude=node52'],partition=c['partitions'])
            command=[sys.executable,'-u',str(entry),'worker','--config',str(numerical_code/'config.json'),
                     '--receipt',str(target),'--lane',str(lane)]
            script=script.replace(shlex.join(queue.command(stage)),shlex.join(command))
            path=folder/(stage+'.sbatch');path.write_text(script)
            command=['sbatch','--hold','--parsable']
            if lane>=2:command+=['--dependency=afterany:'+old[lane-2]]
            job=subprocess.check_output(command+[str(path)],text=True).strip().split(';')[0]
            if not job.isdigit():raise ValueError('Invalid shared-worker Slurm ID')
            receipt['jobs'][stage]=job;write_json(target,receipt)
        dependency='afterany:'+':'.join(old+list(receipt['jobs'].values()))
        subprocess.run(['scontrol','update','JobId='+original['jobs']['status'],'Dependency='+dependency],check=True)
        receipt['status_job']=original['jobs']['status'];receipt['status_dependency']=dependency
        receipt['state']='submitted';write_json(target,receipt)
        for job in receipt['jobs'].values():subprocess.run(['scontrol','release',job],check=True)
    print(json.dumps(receipt),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('submit','worker'))
    parser.add_argument('--config',required=True)
    parser.add_argument('--folder')
    parser.add_argument('--receipt')
    parser.add_argument('--lane',type=int)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--optimized-audit')
    args=parser.parse_args();c=read(resolve_path(args.config))
    if args.stage=='submit':submit(c,resolve_path(args.folder),
        optimized_audit=Path(args.optimized_audit).resolve() if args.optimized_audit else None,workers=args.workers)
    else:worker(c,args.receipt,args.lane)


if __name__=='__main__':main()
