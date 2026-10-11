"""Redistribute unstarted Ni sources while existing native simulations drain."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import traceback

from src.experiment_runner.artifacts import file_hash as sha, write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue
from src.project_runtime.paths import resolve_path, REPO
from src.simulation.campaigns.ni_births.allocation import (
    job_state, remote_child, drain_source, source_queue,
)


def discover(job, config, lane):
    """Identify a receipt-owned supervisor and its native source process."""
    program = '''import json,os,sys
from pathlib import Path
job,config,lane=sys.argv[1:]
workers=[];sources=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=os.getuid():continue
  args=(p/'cmdline').read_bytes().decode().split('\\0')
  if 'src.simulation.campaigns.ni_births.workflow' not in args:continue
  if 'job_'+job not in (p/'cgroup').read_text():continue
  if '--config' not in args or args[args.index('--config')+1]!=config:continue
  if 'worker' in args and args[args.index('--lane')+1]==lane:workers.append(int(p.name))
  if 'source' in args:
   parent=int(next(x.split()[1] for x in (p/'status').read_text().splitlines() if x.startswith('PPid:')))
   sources.append((int(p.name),parent,int(args[args.index('--index')+1])))
 except (FileNotFoundError,ProcessLookupError):continue
assert len(workers)==1,(job,workers)
children=[x for x in sources if x[1]==workers[0]]
assert len(children)==1,(job,children)
print(json.dumps(dict(worker_pid=workers[0],source_pid=children[0][0],source_index=children[0][2])))
'''
    command = ['srun', '--jobid='+job, '--overlap', '--exact', '--nodes=1',
               '--ntasks=1', '--cpus-per-task=1', '--gres=none', '--mem=1G',
               '--immediate=20', sys.executable, '-c', program, job, str(config), str(lane)]
    return json.loads(subprocess.check_output(command, text=True))


def submit(c, workers, folder):
    launch = resolve_path(c['launch'])
    original = json.loads((launch/'launch.json').read_text())
    plan = json.loads((launch/'plan.json').read_text())
    code = Path(original['code']); config = code/'config.json'
    if original['identity'] != plan['identity'] or json.loads(config.read_text()) != c:
        raise ValueError('Original Ni launch, plan or frozen configuration changed')
    if not 1 <= workers <= len(plan['records']):
        raise ValueError('Invalid additional Ni GPU worker count')
    analysis = original['jobs']['analyse']
    old_jobs = [original['jobs'][f'worker-{i:02d}'] for i in range(c['gpu_workers'])]
    if job_state(analysis) != 'PENDING' or any(job_state(job) != 'RUNNING' for job in old_jobs):
        raise ValueError('Scaling requires the original running lanes and pending analysis')
    if (launch/'dispatch.json').exists():
        raise FileExistsError('Review the existing Ni dispatch before another redistribution')
    folder.mkdir(parents=True, exist_ok=False)
    entry = folder/'entry.py'; shutil.copy2(Path(__file__), entry)
    proof = json.loads((launch/'validation-reuse.json').read_text())
    from src.simulation.campaigns.ni_births.validation import verify
    verify(c, proof, code_root=code)
    write_json(folder/'validation-reuse.json', proof)
    settings = json.loads((launch/'machine.json').read_text())
    cuda = settings['execution']['gpu_meam_portable_cuda_root']
    env = dict(PCM_PROJECT_ROOT=str(REPO), PCM_MACHINE_CONFIG=str(launch/'machine.json'),
               PYTHONPATH=str(code), CUDA_HOME=cuda,
               PATH=cuda+'/bin:'+str(Path(sys.executable).parent)+':/usr/bin:/bin',
               UCX_TLS='tcp,self', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', MAX_JOBS='2', OVITO_THREAD_COUNT='1', QT_QPA_PLATFORM='offscreen')
    receipt = dict(campaign_identity=plan['identity'], parent_launch=str(launch/'launch.json'),
                   parent_launch_sha256=sha(launch/'launch.json'), code=str(code),
                   entry_sha256=sha(entry), validation_sha256=sha(folder/'validation-reuse.json'),
                   created_at=datetime.now(timezone.utc).isoformat(), jobs={}, drains=[],
                   assignments=[], analysis_job=analysis, state='preparing',
                   purpose='Current chat: user requested more GPUs for the existing 36-source Ni campaign')
    target = folder/'launch.json'; write_json(target, receipt)
    queue = SlurmQueue(folder, ExecutionBundle(code), 'src.simulation.campaigns.ni_births.workflow',
                       env, target, receipt, 'NI-BIRTH-SCALE')
    released = False; analysis_held = False
    previous_dependency = 'afterany:'+':'.join(old_jobs)
    try:
        # Reserve new jobs held, before stopping any old queue supervisor.
        for lane in range(workers+1):
            stage = f'worker-{lane:02d}' if lane < workers else 'drain'
            options = ['--cpus-per-task=4', '--mem=24G', '--gres=gpu:1',
                       '--time=08:00:00', '--exclude='+c['exclude_nodes']] if lane < workers else [
                           '--cpus-per-task=2', '--mem=8G', '--time=18:00:00']
            partition = c['partitions'] if lane < workers else 'CPU'
            script = queue.render(stage, options, partition=partition)
            command = [sys.executable, '-u', str(entry), 'worker' if lane < workers else 'drain',
                       '--config', str(config), '--receipt', str(target), '--lane', str(lane)]
            script = script.replace(shlex.join(queue.command(stage)), shlex.join(command))
            script_path = folder/(stage+'.sbatch'); script_path.write_text(script)
            job = subprocess.check_output(['sbatch', '--hold', '--parsable', str(script_path)], text=True).strip().split(';')[0]
            if not job.isdigit(): raise ValueError('Invalid Ni scaling job ID: '+job)
            receipt['jobs'][stage] = job; write_json(target, receipt)
        subprocess.run(['scontrol', 'hold', analysis], check=True); analysis_held = True
        for lane, job in enumerate(old_jobs):
            info = discover(job, config, lane)
            if info['source_index'] % c['gpu_workers'] != lane:
                raise ValueError('Active source is outside its recorded original lane')
            observed = remote_child(job, info, config, lane, pause=True)
            source = plan['records'][info['source_index']]['id']
            receipt['drains'].append(dict(info, job=job, lane=lane, source=source, observed=observed))
            write_json(target, receipt)
        pending = []; retained = []
        for index, record in enumerate(plan['records']):
            drain = receipt['drains'][index % c['gpu_workers']]
            if index <= drain['source_index']:
                if index < drain['source_index'] and not (resolve_path(c['archive'])/record['id']/'outcome.json').exists():
                    raise ValueError('An earlier Ni source has no outcome: '+record['id'])
                retained.append(record['id']); continue
            if ((resolve_path(c['scratch'])/record['id']/'started.json').exists()
                    or (resolve_path(c['archive'])/record['id']).exists()):
                raise ValueError('Cannot transfer a started Ni source: '+record['id'])
            pending.append(record['id'])
        if len(pending) < workers:
            raise ValueError('Fewer unstarted sources than requested workers')
        receipt['retained_source_ids'] = retained
        receipt['assignments'] = [pending[lane::workers] for lane in range(workers)]
        if len(set(retained+pending)) != len(plan['records']):
            raise ValueError('Redistribution changed source coverage')
        dependencies = old_jobs+list(receipt['jobs'].values())
        receipt['analysis_dependency'] = 'afterany:'+':'.join(dependencies)
        subprocess.run(['scontrol', 'update', 'JobId='+analysis,
                        'Dependency='+receipt['analysis_dependency']], check=True)
        receipt.update(state='submitted', transferred_sources=len(pending)); write_json(target, receipt)
        write_json(launch/'dispatch.json', dict(campaign_identity=plan['identity'],
                   expansion_receipt=str(target), jobs=dict(original['jobs'], **{
                       'scale-'+key: value for key, value in receipt['jobs'].items()}),
                   analysis_dependency=receipt['analysis_dependency']))
        for job in receipt['jobs'].values():
            released = True
            subprocess.run(['scontrol', 'release', job], check=True)
        subprocess.run(['scontrol', 'release', analysis], check=True); analysis_held = False
    except BaseException:
        receipt.update(state='submission_failed', traceback=traceback.format_exc())
        if not released:
            for job in receipt['jobs'].values(): subprocess.run(['scancel', job], check=True)
            for info in receipt['drains']:
                remote_child(info['job'], info, config, info['lane'], resume=True)
            subprocess.run(['scontrol', 'update', 'JobId='+analysis, 'Dependency='+previous_dependency], check=True)
            if analysis_held: subprocess.run(['scontrol', 'release', analysis], check=True)
            receipt['original_queues_resumed'] = True
        write_json(target, receipt)
        raise
    print(json.dumps(receipt), flush=True)


def execute(c, config, target, lane, draining):
    receipt = json.loads(target.read_text()); folder = target.parent
    if sha(Path(__file__)) != receipt['entry_sha256'] or sha(Path(receipt['parent_launch'])) != receipt['parent_launch_sha256']:
        raise ValueError('Ni scaling implementation or ownership receipt changed')
    if config != Path(receipt['code'])/'config.json':
        raise ValueError('Ni scaling must execute the original frozen configuration')
    stage = 'drain' if draining else f'worker-{lane:02d}'
    if os.environ['SLURM_JOB_ID'] != receipt['jobs'][stage]:
        raise ValueError('Ni scaling entered a different allocation')
    output = folder/stage; output.mkdir(exist_ok=True)
    try:
        if draining:
            def finish(info):
                dest = output/f'lane-{info["lane"]:02d}'; dest.mkdir(exist_ok=True)
                return drain_source(c, config, dict(drain=info, lane=info['lane']), dest)
            with ThreadPoolExecutor(max_workers=len(receipt['drains'])) as pool:
                results = list(pool.map(finish, receipt['drains']))
        else:
            proof = folder/'validation-reuse.json'
            if sha(proof) != receipt['validation_sha256']:
                raise ValueError('Numerical validation reuse evidence changed')
            from src.simulation.campaigns.ni_births.validation import verify
            verify(c, json.loads(proof.read_text()), code_root=config.parent)
            results = source_queue(c, config, dict(source_ids=receipt['assignments'][lane]), output)
        write_json(output/'status.json', dict(state='complete' if all(r['state']=='complete' for r in results) else 'incomplete', outcomes=results))
    except BaseException:
        write_json(output/'status.json', dict(state='failed', traceback=traceback.format_exc()))
        raise


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('action', choices=['submit', 'worker', 'drain'])
    p.add_argument('--config', required=True); p.add_argument('--workers', type=int)
    p.add_argument('--output', type=Path); p.add_argument('--receipt', type=Path)
    p.add_argument('--lane', type=int, default=0)
    a = p.parse_args(); config = resolve_path(a.config); c = json.loads(config.read_text())
    if a.action == 'submit': submit(c, a.workers, resolve_path(a.output))
    else: execute(c, config, a.receipt, a.lane, a.action == 'drain')


if __name__ == '__main__': main()
