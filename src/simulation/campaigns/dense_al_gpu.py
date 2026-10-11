"""Transfer a bounded set of unstarted dense Al sources to GPU MEAM workers."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback

from src.project_runtime.paths import REPO, load_json, storage_path, machine
from src.project_runtime.transfer import write_json
from .birth_sources import now, sha256, freeze_code, register_entry
from .dense_al import manifest_at, verify_code, sbatch, run_source, publish_record
from .dense_al_half_stop import source_input

BACKEND = 'torch_meam_lammps_iso_npt_v1'


def execute_gpu(directory, timeout_seconds=None):
    start = time.monotonic()
    command = [sys.executable, '-u', '-m', 'src.research.meam_gpu.npt',
               'run', '--directory', str(directory)]
    with (directory/'source.stdout.log').open('x') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                       timeout=timeout_seconds)
    return time.monotonic()-start


def prepare(c):
    launch = Path(c['launch'])
    if (launch/'manifest.json').exists():
        raise FileExistsError('Preserve the existing GPU preparation')
    old_launch = Path(c['source_launch'])
    original = manifest_at(old_launch)
    selected = [r for r in original['runs'] if r['source_id'] in c['source_ids'] and r['sample_interval_steps'] == 50]
    if len(selected) != len(c['source_ids']) or len({r['source_id'] for r in selected}) != len(selected):
        raise ValueError('Require one exact 0.1 ps record for every requested source')
    for r in selected:
        old = Path(original['root'])/r['run_dir']
        if (old/'status.json').exists() or old.is_symlink():
            raise ValueError(f'Only unstarted inputs can transfer: {old}')
        for name, digest in r['input_sha256'].items():
            if sha256(old/name) != digest:
                raise ValueError(f'Original prepared file changed: {old/name}')
    root = Path(c['root'])
    root.mkdir(parents=True)
    launch.mkdir(parents=True, exist_ok=True)
    records = []
    for old_record in selected:
        r = deepcopy(old_record)
        old = Path(original['root'])/r['run_dir']
        directory = root/r['run_dir']
        shutil.copytree(old, directory)
        metadata = json.loads((directory/'input_metadata.json').read_text())
        execution = dict(engine='Differentiable CUDA MEAM force and normal virial, native LAMMPS isotropic NPT',
            force_dtype='float32', integration_dtype='float64', mpi_ranks=1,
            original_mpi_ranks=original['config']['mpi_ranks'], gpu_count=1,
            thermostat_ps=.3, barostat_ps=3., pressure_bar=0.,
            neighbor_policy='Rebuild CUDA support at every step, including box changes and atom reordering',
            stress_scope='Normal virials and scalar pressure; shear stress unavailable',
            restart_policy='Native positions, velocities and NPT extended state retained; GPU external fix must be reinstalled on restart',
            validation=str(launch/'technical/validation-v1/validation.json'),
            ancestry='Same original prepared melt and velocity seed; not an additional independent ancestor')
        r.update(execution_backend=BACKEND, previous_directory=str(old),
                 failure_id=r['run_id']+'-gpu-failed-20261004')
        metadata.update(execution_backend=BACKEND, execution=execution,
            previous_preparation=str(old), gpu_launch=str(launch))
        write_json(directory/'input_metadata.json', metadata)
        (directory/'source.in.lammps').write_text(source_input(original['config'], r, directory))
        r['input_sha256'] = {str(p.relative_to(directory)):sha256(p)
            for p in directory.rglob('*') if p.is_file()}
        records.append(r)
    plan = dict(protocol=original['protocol'], config=original['config'], runs=records,
        root=str(root), launch_root=str(launch), source_launch=str(old_launch),
        source_manifest_sha256=sha256(old_launch/'manifest.json'), execution_backend=BACKEND,
        source_ids=c['source_ids'], created_at=now(), gpu_config=c,
        ancestry_policy='Transferred descendants of existing melts, retaining source IDs and splits')
    write_json(launch/'manifest.json', plan)
    (launch/'manifest.sha256').write_text(sha256(launch/'manifest.json')+'\n')
    (root/'manifest.json').symlink_to(launch/'manifest.json')
    assignments = [[] for _ in range(c['workers'])]
    loads = [0 for _ in assignments]
    for i in sorted(range(len(records)),key=lambda i:records[i]['stopping']['maximum_measurement_steps'],reverse=True):
        lane = min(range(c['workers']),key=loads.__getitem__)
        assignments[lane].append(i)
        loads[lane] += records[i]['stopping']['maximum_measurement_steps']+original['config']['equilibration_steps']
    write_json(launch/'assignments.json', assignments)
    freeze_code(launch)
    register_entry(c['collection_id'], dict(root='simulation_runs', path=str(root.relative_to(storage_path('simulation_runs'))),
        kind='simulation', dependencies=['al-main-half-recovery-20261003','meam-gpu-blackwell-20261004'],
        metadata=dict(materials=['Al'], potential_ids=['al-lee2003-meam'], role='raw_dynamics', classification='research',
            title='Five dense Al sources transferred to CUDA MEAM with native isotropic NPT',
            description='70304 atoms; original 2 fs integration, exact 0.1 ps positions/velocities, confirmed-halfway stopping and peer caps. Float32 GPU force/virial, float64 native integration.',
            ancestry='Five retained original train ancestors; no additional independent melts.',
            execution_backend=BACKEND, evidence=[str(launch/'manifest.json'),str(launch/'submission.json')])))
    return launch


def submit(launch):
    plan = manifest_at(launch)
    c = plan['gpu_config']
    gate_path = launch/'technical/validation-v1/validation.json'
    gate = json.loads(gate_path.read_text())
    if gate['state'] != 'passed':
        raise ValueError('Production requires passed full-cell native NPT gates')
    relative = 'src/research/meam_gpu/npt.py'
    if sha256(launch/'code'/relative) != sha256(launch/'technical/validation-code-v1'/relative):
        raise ValueError('Validated adapter differs from frozen production adapter')
    if (launch/'submission.json').exists():
        raise FileExistsError('Do not duplicate this GPU launch')
    controller = str(c['cpu_controller'])
    state = subprocess.check_output(['squeue','-h','-j',controller,'-o','%i %t %E'], text=True).strip()
    if not state.startswith(controller+' PD '):
        raise ValueError(f'CPU successor must still be pending: {state}')
    dependency = state.split(maxsplit=2)[2]
    if dependency != c['expected_cpu_dependency']:
        raise ValueError(f'CPU dependency changed: {dependency}')
    original_dependency = dependency.split('(')[0].removesuffix('_*')
    code = launch/'code'
    cuda = machine()['execution']['gpu_meam_cuda_root']
    environment = ['PCM_PROJECT_ROOT='+str(REPO),'PYTHONPATH='+str(code),'OMP_NUM_THREADS=1',
        'OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','OVITO_THREAD_COUNT=1','QT_QPA_PLATFORM=offscreen',
        'CUDA_HOME='+cuda,'PATH='+cuda+'/bin:'+str(Path(sys.prefix)/'bin')+':/usr/bin:/bin',
        'TORCH_CUDA_ARCH_LIST=12.0','MAX_JOBS=4']
    module = 'src.simulation.campaigns.dense_al_gpu'
    command = [sys.executable,'-u','-m',module,'worker','--launch',str(launch)]
    script = '\n'.join(['#!/bin/bash','#SBATCH --job-name=al-dense-GPU-MEAM','#SBATCH --partition=RTX6000PRO',
        '#SBATCH --nodes=1','#SBATCH --ntasks=1','#SBATCH --cpus-per-task=4','#SBATCH --gres=gpu:1',
        '#SBATCH --mem=32G',f'#SBATCH --time={c["walltime"]}',f'#SBATCH --array=0-{c["workers"]-1}%{c["workers"]}',
        '#SBATCH --hold','#SBATCH --signal=B:USR1@600',f'#SBATCH --chdir={code}',
        f'#SBATCH --output={launch}/technical/worker-%A_%a.log','set -euo pipefail',
        'exec env '+shlex.join(environment+command)+' --index "$SLURM_ARRAY_TASK_ID"',''])
    # Prevent the existing successor from racing this bounded transfer. Running
    # CPU simulations are never touched. A held GPU job cannot start early.
    subprocess.run(['scontrol','hold',controller],check=True)
    reserved = []
    job = None
    cleanup = None
    try:
        for r in plan['runs']:
            old = Path(r['previous_directory'])
            with (old/'status.json').open('x') as f:
                json.dump(dict(state='gpu_reserved', launch=str(launch), created_at=now()), f)
            reserved.append(old)
        job = sbatch(script,launch/'workers.sbatch')
        cleanup_script = '\n'.join(['#!/bin/bash','#SBATCH --job-name=al-dense-GPU-collect',
            '#SBATCH --partition=CPU','#SBATCH --nodes=1','#SBATCH --ntasks=1',
            '#SBATCH --cpus-per-task=1','#SBATCH --mem=4G','#SBATCH --time=02:00:00',
            f'#SBATCH --dependency=afterany:{job}',f'#SBATCH --chdir={code}',
            f'#SBATCH --output={launch}/technical/collect-%j.log','set -euo pipefail',
            'exec env '+shlex.join(environment+[sys.executable,'-u','-m',module,'reconcile','--launch',str(launch)]),''])
        cleanup = sbatch(cleanup_script,launch/'reconcile.sbatch')
        new_dependency = original_dependency+',afterok:'+cleanup
        subprocess.run(['scontrol','update','JobId='+controller,'Dependency='+new_dependency],check=True)
        receipt = dict(state='submitted', worker_job=job, workers=c['workers'], source_ids=plan['source_ids'],
            submitted_at=now(), cpu_controller=controller, old_dependency=dependency,
            new_dependency=new_dependency, reconciliation_job=cleanup, validation_sha256=sha256(gate_path))
        write_json(launch/'submission.json',receipt)
        subprocess.run(['scontrol','release',job],check=True)
        subprocess.run(['scontrol','release',controller],check=True)
    except BaseException:
        if cleanup is not None:
            subprocess.run(['scancel',cleanup],check=True)
        if job is not None:
            subprocess.run(['scancel',job],check=True)
            subprocess.run(['scontrol','update','JobId='+controller,'Dependency='+original_dependency],check=True)
        for old in reserved:
            (old/'status.json').unlink()
        subprocess.run(['scontrol','release',controller],check=True)
        raise
    return receipt


def complete_handoff(launch, plan, r):
    old = Path(r['previous_directory'])
    if old.is_symlink():
        return
    directory = Path(plan['root'])/r['run_dir']
    if not directory.is_symlink():
        publish_record(directory,r)
    destination = directory.resolve()
    retained = launch/'original-preparations'/r['run_id']
    retained.parent.mkdir(exist_ok=True)
    old.rename(retained)
    old.symlink_to(destination, target_is_directory=True)
    write_json(launch/'technical'/f'complete-{r["source_id"]}.json',dict(
        state='complete', source_id=r['source_id'], destination=str(destination),
        original_preparation=str(retained), completed_at=now()))


def reconcile(launch):
    """Runs after every GPU task has ended, including timeout or node failure."""
    verify_code(launch)
    plan = manifest_at(launch)
    completed, released = [], []
    for r in plan['runs']:
        directory = Path(plan['root'])/r['run_dir']
        status_path = directory/'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else None
        old = Path(r['previous_directory'])
        if status and status['state']=='complete':
            complete_handoff(launch,plan,r)
            completed.append(r['source_id'])
            continue
        if status and not directory.is_symlink():
            write_json(status_path,dict(state='failed', previous_status=status,
                finished_at=now(), reason='GPU worker ended before successful publication'))
            publish_record(directory,r,failed=True)
        reservation = old/'status.json'
        if reservation.exists():
            value = json.loads(reservation.read_text())
            if value['state']!='gpu_reserved' or value['launch']!=str(launch):
                raise ValueError(f'Cannot release a changed CPU reservation: {old}')
            reservation.unlink()
        released.append(r['source_id'])
    write_json(launch/'completion.json',dict(state='complete' if not released else 'partial',
        completed_source_ids=completed, returned_to_cpu_source_ids=released, finished_at=now()))


def worker(launch,index):
    verify_code(launch)
    plan = manifest_at(launch)
    assigned = json.loads((launch/'assignments.json').read_text())[index]
    def stop(signum, frame):
        raise InterruptedError(f'GPU dense worker received signal {signum}')
    for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,stop)
    failures = []
    for i in assigned:
        r = plan['runs'][i]
        old = Path(r['previous_directory'])
        reservation = json.loads((old/'status.json').read_text())
        if reservation['state'] != 'gpu_reserved' or reservation['launch'] != str(launch):
            raise ValueError(f'GPU reservation changed: {old}')
        try:
            run_source(plan,r)
        except BaseException:
            failures.append(r['source_id'])
            write_json(launch/'technical'/f'failure-{r["source_id"]}.json',dict(
                error=traceback.format_exc(), source_id=r['source_id'], recorded_at=now(),
                disposition='Failed GPU attempt archived; original unstarted CPU input released'))
            # The CPU successor sees its unchanged, still-unstarted input.
            (old/'status.json').unlink()
            continue
        complete_handoff(launch,plan,r)
    write_json(launch/'technical'/f'lane-{index}.json',dict(
        state='complete' if not failures else 'completed_with_archived_failures',
        failed_source_ids=failures, finished_at=now()))
    if failures:
        raise RuntimeError(f'GPU attempts failed and were released to the original CPU queue: {failures}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','submit','worker','reconcile'])
    p.add_argument('--config');p.add_argument('--launch',type=Path);p.add_argument('--index',type=int)
    a=p.parse_args()
    if a.action=='prepare':print(prepare(load_json(a.config)))
    elif a.action=='submit':print(json.dumps(submit(a.launch),indent=2))
    elif a.action=='reconcile':reconcile(a.launch)
    else:worker(a.launch,a.index)


if __name__=='__main__':main()
