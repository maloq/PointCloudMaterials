"""Independent Al birth-screen sources with continuous observation and causal stopping."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback

from src.project_runtime.paths import REPO, load_json, machine, resolve_path, storage_path
from src.project_runtime.transfer import write_json, publish_simulation, archive_failed_simulation

PROTOCOL = 'al_spontaneous_birth_sources_v1'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def register_entry(identifier, entry):
    path = resolve_path(machine()['catalog'])
    with path.with_suffix('.json.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = json.loads(path.read_text())
        if identifier in value['datasets']:
            raise ValueError(f'Dataset already registered: {identifier}')
        value['datasets'][identifier] = entry
        write_json(path, value)


def controls(config, temperature):
    return f'''mass 1 26.9815
pair_style meam
pair_coeff * * potential/Lee2003_Al.library.meam Al potential/Lee2003_Al.meam Al
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
timestep {config['timestep_ps']}
fix remove_drift all momentum 100 linear 1 1 1
fix ensemble all npt temp {temperature:g} {temperature:g} 0.3 iso 0 0 3
thermo {config['sample_interval_steps']}
thermo_style custom step temp press vol pe
thermo_modify format float %.16g flush yes lost error
'''


def melt_input(config, record):
    return f'''log melt.lammps.log
units metal
dimension 3
boundary p p p
atom_style atomic
read_data initial_fcc.lammps.data
{controls(config, 1325)}
velocity all create 1325 {record['preparation_seed']} mom yes rot no dist gaussian
restart 25000 melt.restart.1.bin melt.restart.2.bin
run 50000
compute liquid_msd all msd com yes
thermo_style custom step temp press vol pe c_liquid_msd[4]
run 45000
dump melt_qc all custom 500 melt_validation.lammpstrj id type x y z
dump_modify melt_qc first yes sort id format line "%d %d %.17g %.17g %.17g"
run 5000
print "$(c_liquid_msd[4]:%.17g)" file melt_msd_A2.txt screen no
undump melt_qc
write_data prepared_liquid.lammps.data
write_restart melt_final.restart.bin
print "BIRTH_SOURCE_MELT_COMPLETE {record['run_id']}"
'''


def source_input(config, record):
    # One LAMMPS process and unchanged fixes throughout: no chunk-wise resets.
    maximum = config['equilibration_steps'] + config['measurement_steps']
    assess = config['assessment_steps']
    python = shlex.quote(sys.executable)
    return f'''log source.lammps.log
units metal
dimension 3
boundary p p p
atom_style atomic
read_data prepared_liquid.lammps.data
{controls(config, record['temperature_K'])}
velocity all create {record['temperature_K']:g} {record['velocity_seed']} mom yes rot no dist gaussian loop all
dump trajectory all custom {config['sample_interval_steps']} trajectory.lammpstrj id type x y z vx vy vz
dump_modify trajectory first yes sort id format line "%d %d %.17g %.17g %.17g %.17g %.17g %.17g"
restart 5000 source.restart.1.bin source.restart.2.bin
run 0
write_dump all custom assessment.lammpstrj id type x y z modify sort id format line "%d %d %.17g %.17g %.17g"
shell {python} -m src.simulation.campaigns.birth_sources assess --directory . --step 0
include technical/decision_0.lammps
variable cycle loop {maximum // assess}
label source_loop
run {assess}
write_dump all custom assessment.lammpstrj id type x y z modify sort id format line "%d %d %.17g %.17g %.17g"
shell {python} -m src.simulation.campaigns.birth_sources assess --directory . --step $(step:%.0f)
include technical/decision_$(step:%.0f).lammps
if "${{birth_stop}} == 1" then "jump SELF source_complete"
next cycle
jump SELF source_loop
label source_complete
undump trajectory
write_restart final.restart.bin
print "BIRTH_SOURCE_HOLD_COMPLETE {record['run_id']}"
'''


def assess(directory, step):
    from .elemental import structure
    directory = Path(directory)
    config = json.loads((directory/'technical/launch_config.json').read_text())
    result = structure(directory/'assessment.lammpstrj', config['ptm_rmsd_cutoff'])
    metadata = json.loads((directory/'input_metadata.json').read_text())
    if result['atom_count'] != metadata['atom_count']:
        raise ValueError(f'Atom loss during birth source at step {step}: {result}')
    first = None
    if step:
        previous = json.loads((directory/'source_progress.json').read_text())
        if previous['step'] != step-config['assessment_steps']:
            raise ValueError(f'Discontinuous stopping audit: {previous["step"]} -> {step}')
        first = previous['first_fraction_crossing_step']
    if first is None and result['crystal_fraction'] >= config['stop_crystal_fraction']:
        first = step
    maximum = config['equilibration_steps']+config['measurement_steps']
    tail_complete = first is not None and step >= first+config['followup_steps']
    stop = step >= maximum or tail_complete
    result.update(step=step, time_ps=step*config['timestep_ps'],
                  first_fraction_crossing_step=first, stop=stop,
                  stop_reason='fraction_followup_complete' if tail_complete else
                              'duration_limit' if step >= maximum else None)
    write_json(directory/'technical'/f'assessment_{step}.json', result)
    write_json(directory/'source_progress.json', result)
    # Unique include published only after successful assessment: shell failures
    # cannot be mistaken for a negative decision or reuse the previous decision.
    decision = directory/'technical'/f'decision_{step}.lammps'
    with decision.open('x') as handle:
        handle.write(f'variable birth_stop equal {int(stop)}\n')


def validate_melt(directory, atom_count, config):
    import numpy as np
    from ovito.io import import_file
    from ovito.modifiers import (PolyhedralTemplateMatchingModifier,
                                ClusterAnalysisModifier, CoordinationAnalysisModifier)
    pipeline = import_file(str(directory/'melt_validation.lammpstrj'))
    pipeline.modifiers.append(PolyhedralTemplateMatchingModifier(rmsd_cutoff=config['ptm_rmsd_cutoff']))
    frames = []
    for index in range(pipeline.source.num_frames):
        data = pipeline.compute(index)
        types = np.asarray(data.particles['Structure Type'])
        if len(types) != atom_count:
            raise ValueError('Atom count changed during independent melt')
        selected = np.isin(types, [1, 2, 3])
        data.particles_.create_property('Selection', data=selected.astype(np.int32))
        data.apply(ClusterAnalysisModifier(cutoff=3.6, only_selected=True))
        sizes = np.bincount(np.asarray(data.particles['Cluster'], dtype=np.int64))[1:]
        frames.append(dict(step=int(data.attributes['Timestep']),
                           crystalline_fraction=float(selected.mean()),
                           largest_crystalline_cluster_atoms=int(sizes.max(initial=0))))
    if [f['step'] for f in frames] != list(range(95000, 100001, 500)):
        raise ValueError(f'Melt validation timeline differs: {frames}')
    data.apply(CoordinationAnalysisModifier(cutoff=8., number_of_bins=160))
    rdf = data.tables['coordination-rdf'].xy().tolist()
    msd = float((directory/'melt_msd_A2.txt').read_text())
    accepted = (np.isfinite(msd) and msd >= config['minimum_melt_msd_A2'] and
                all(f['crystalline_fraction'] < .01 and
                    f['largest_crystalline_cluster_atoms'] < 64 for f in frames))
    result = dict(accepted=bool(accepted), atom_count=atom_count, frames=frames,
                  displacement_last_150ps_A2=msd, final_rdf=rdf,
                  policy='Last 15 ps: fraction <1%, no >=64-atom PTM cluster; last 150 ps MSD >= declared minimum')
    write_json(directory/'melt_validation.json', result)
    if not accepted:
        raise ValueError(f'Melt preparation failed QC; inspect {directory}/melt_validation.json')


def prepare(config, name):
    from ase.build import bulk
    from ase.io import write
    if config['protocol'] != PROTOCOL or config['timestep_ps'] != .003:
        raise ValueError('Birth sources require the declared 3 fs Al protocol')
    if Path(name).name != name or name in {'.', '..'}:
        raise ValueError('Campaign name must be one directory component')
    maximum = config['equilibration_steps']+config['measurement_steps']
    if (maximum % config['assessment_steps'] or
        config['assessment_steps'] % config['sample_interval_steps'] or
        config['followup_steps'] % config['assessment_steps']):
        raise ValueError('Hold, assessments, follow-up and output must share an exact step grid')
    root = storage_path('simulation_runs')/name
    launch = storage_path('archive')/'simulation-launches'/name
    if root.exists() or launch.exists():
        raise FileExistsError(f'Preserve existing campaign: {root} / {launch}')
    old_seeds, prior_hashes = set(), {}
    for filename in config['excluded_manifests']:
        prior = json.loads(Path(filename).read_text())
        prior_hashes[filename] = sha256(filename)
        for r in prior['runs']:
            old_seeds.update((r['preparation_seed'], r['velocity_seed']))
    records = []
    for replica in range(config['replicas_per_temperature']):
        for temperature in config['temperatures_K']:
            def seed(role):
                value = f'{config["campaign_seed"]}:{temperature:g}:{replica}:{role}'
                return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'little') % 899999999+1
            melt, velocity = seed('melt'), seed('velocity')
            if melt in old_seeds or velocity in old_seeds or melt == velocity:
                raise ValueError('Melt/velocity seed collision in new source manifest')
            old_seeds.update((melt, velocity))
            index = len(records)
            run_id = f'{name}-source{index:03d}-T{temperature:g}'
            records.append(dict(run_id=run_id, run_index=index, run_dir=f'runs/{run_id}',
                                temperature_K=temperature, replica=replica, split='train',
                                root_lineage=f'independent_melt_{melt}', parent_trajectory_id=None,
                                preparation_seed=melt, velocity_seed=velocity))
    for item in config['potential_files']:
        if sha256(item['path']) != item['sha256']:
            raise ValueError(f'Changed potential: {item["path"]}')
    root.mkdir(parents=True)
    launch.mkdir(parents=True)
    atoms = bulk('Al', 'fcc', a=4.05, cubic=True).repeat(config['fcc_repeats'])
    initial = root/'initial_fcc.lammps.data'
    write(initial, atoms, format='lammps-data', atom_style='atomic', specorder=('Al',))
    for record in records:
        directory = root/record['run_dir']
        (directory/'potential').mkdir(parents=True)
        (directory/'technical').mkdir()
        shutil.copy2(initial, directory/initial.name)
        for item in config['potential_files']:
            path = Path(item['path'])
            shutil.copy2(path, directory/'potential'/path.name)
        (directory/'melt.in.lammps').write_text(melt_input(config, record))
        (directory/'source.in.lammps').write_text(source_input(config, record))
        metadata = dict(protocol=PROTOCOL, state='prepared', material='Al', atom_count=len(atoms),
                        timestep_ps=config['timestep_ps'], melt_steps=100000,
                        equilibration_steps=config['equilibration_steps'], max_hold_steps=maximum,
                        sample_interval_steps=config['sample_interval_steps'],
                        observation_start='target-temperature velocity initialization, step zero',
                        conditions_are_model_inputs=False, potential=config['potential_files'],
                        **record)
        write_json(directory/'input_metadata.json', metadata)
        write_json(directory/'technical/launch_config.json', config)
        files = ['initial_fcc.lammps.data', 'melt.in.lammps', 'source.in.lammps',
                 'input_metadata.json', 'technical/launch_config.json',
                 *['potential/'+Path(p['path']).name for p in config['potential_files']]]
        record['input_sha256'] = {file: sha256(directory/file) for file in files}
    manifest = dict(protocol=PROTOCOL, created_at=now(), config=config, runs=records,
                    atom_count=len(atoms), launch_root=str(launch),
                    excluded_manifest_sha256=prior_hashes,
                    scientific_role='development only; outcome-dependent early stopping; no natural-prevalence test set',
                    main_history_policy='History entirely after step 5000; early events remain a separate stratum',
                    storage_policy='Paired verified float16/float32 observations; native restarts preserved')
    write_json(root/'manifest.json', manifest)
    (root/'manifest.sha256').write_text(sha256(root/'manifest.json')+'\n')
    shutil.copy2(root/'manifest.json', launch/'manifest.json')
    shutil.copy2(root/'manifest.sha256', launch/'manifest.sha256')
    for record in records:
        shutil.copy2(root/'manifest.json', root/record['run_dir']/'campaign_manifest.json')
    write_json(root/'status.json', dict(state='prepared', runs=len(records), launch_root=str(launch)))
    register_entry(name, dict(root='simulation_runs', path=name, kind='simulation',
        dependencies=['potentials'], aliases=[], metadata=dict(
            title=f'Al spontaneous births: {len(records)} independent melts',
            materials=['Al'], potential_ids=['al-lee2003-meam'], role='raw_dynamics',
            classification='building', evidence=['${dataset:'+name+'}/manifest.json'],
            lineage=f'{len(records)} fresh independently melted roots; every source is development/train only',
            description='Continuous position/velocity observations from target-temperature initialization; 0.15 ps cadence; early stopping after 10% PTM crystalline plus 12 ps.',
            limitations=['Prepared/running sources are not completed data.',
                         'Outcome-dependent training collection; not a representative fixed-duration evaluation cohort.',
                         'Temperature and age are audit metadata, never model inputs.'])))
    return root


def execute(directory, filename, stdout_name, ranks):
    from .independent_meam_source import _lammps_environment
    command = ['srun', '--mpi=pmi2', '--nodes=1', f'--ntasks={ranks}',
               '--cpus-per-task=1', '--cpu-bind=cores', '--kill-on-bad-exit=1',
               str(Path(sys.prefix)/'bin/lmp'), '-in', filename]
    started = time.monotonic()
    with (directory/stdout_name).open('xb') as output:
        process = subprocess.Popen(command, cwd=directory, env=_lammps_environment(),
                                   stdin=subprocess.DEVNULL, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait()
            if code:
                raise RuntimeError(f'LAMMPS exit {code}; inspect {directory/stdout_name}')
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    return time.monotonic()-started


def run_source(root, manifest, record):
    directory = root/record['run_dir']
    if (directory/'status.json').exists():
        raise FileExistsError(f'Inspect previous simulation attempt before retry: {directory}')
    for filename, digest in record['input_sha256'].items():
        if sha256(directory/filename) != digest:
            raise ValueError(f'Prepared source modified: {directory/filename}')
    config = manifest['config']
    status = dict(state='running', phase='melt', started_at=now(), run_id=record['run_id'],
                  slurm_job_id=os.environ['SLURM_JOB_ID'], host=socket.gethostname())
    write_json(directory/'status.json', status)
    try:
        melt_seconds = execute(directory, 'melt.in.lammps', 'melt.stdout.log', config['mpi_ranks'])
        if f'BIRTH_SOURCE_MELT_COMPLETE {record["run_id"]}' not in (directory/'melt.stdout.log').read_text().splitlines():
            raise RuntimeError('Melt completion marker missing')
        validate_melt(directory, manifest['atom_count'], config)
        status.update(phase='hold', melt_seconds=melt_seconds, updated_at=now())
        write_json(directory/'status.json', status)
        hold_seconds = execute(directory, 'source.in.lammps', 'source.stdout.log', config['mpi_ranks'])
        if f'BIRTH_SOURCE_HOLD_COMPLETE {record["run_id"]}' not in (directory/'source.stdout.log').read_text().splitlines():
            raise RuntimeError('Hold completion marker missing')
        progress = json.loads((directory/'source_progress.json').read_text())
        if not progress['stop']:
            raise ValueError(f'Hold ended without its declared stopping condition: {progress}')
        restarts = {}
        for filename in ('melt_final.restart.bin', 'final.restart.bin'):
            if not (directory/filename).stat().st_size:
                raise ValueError(f'Empty native restart: {directory/filename}')
            restarts[filename] = sha256(directory/filename)
        metadata = json.loads((directory/'input_metadata.json').read_text())
        metadata.update(state='dynamics_complete', completed_steps=progress['step'],
                        stop_reason=progress['stop_reason'],
                        source_sha256=sha256(directory/'trajectory.lammpstrj'),
                        native_restart_sha256=restarts)
        write_json(directory/'metadata.json', metadata)
        status.update(phase='conversion', hold_seconds=hold_seconds, updated_at=now())
        write_json(directory/'status.json', status)
        command = [sys.executable, str(Path(__file__).resolve().parents[3]/'scripts/convert_trajectory.py'),
                   'birth-pair', str(directory), '--delete-source']
        with (directory/'conversion.log').open('x') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        conversion = json.loads((directory/'paired_conversion.json').read_text())
        status.update(state='complete', phase='converted', completed_at=now(),
                      completed_steps=progress['step'], frame_count=conversion['frame_count'],
                      stop_reason=progress['stop_reason'], native_restart_sha256=restarts)
        write_json(directory/'status.json', status)
        write_json(directory/'outcome.json', status)
    except BaseException:
        status.update(state='failed', finished_at=now(), error=traceback.format_exc())
        write_json(directory/'status.json', status)
        archive_failed_simulation(directory, identifier=record['run_id']+'-failed')
        raise
    publish_simulation(directory, identifier=record['run_id'], move=True)


def worker(root, index, count):
    root = resolve_path(root).resolve()
    if sha256(root/'manifest.json') != (root/'manifest.sha256').read_text().strip():
        raise ValueError('Campaign manifest changed')
    manifest = json.loads((root/'manifest.json').read_text())
    launch = Path(manifest['launch_root'])
    inventory = json.loads((launch/'code_sha256.json').read_text())
    if not Path(__file__).resolve().is_relative_to(launch/'code'):
        raise ValueError('Worker must execute the frozen campaign source')
    for filename, digest in inventory.items():
        if sha256(launch/'code'/filename) != digest:
            raise ValueError(f'Frozen code changed: {filename}')
    if count != manifest['config']['workers'] or not 0 <= index < count:
        raise ValueError('Worker assignment differs from frozen campaign')
    if int(os.environ['SLURM_NTASKS']) != manifest['config']['mpi_ranks']:
        raise ValueError('CPU allocation differs from declared MPI ranks')
    def interrupted(signum, frame):
        raise InterruptedError(f'Slurm interrupted birth worker with signal {signum}')
    signal.signal(signal.SIGUSR1, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    done = []
    path = root/'workers'/f'worker-{index}.json'
    for record in manifest['runs'][index::count]:
        write_json(path, dict(state='running', current=record['run_id'], completed=done, updated_at=now()))
        update_status(root, len(manifest['runs']))
        try:
            run_source(root, manifest, record)
        except BaseException:
            write_json(path, dict(state='failed', current=record['run_id'], completed=done,
                                  error=traceback.format_exc(), updated_at=now()))
            update_status(root, len(manifest['runs']))
            raise
        done.append(record['run_id'])
    write_json(path, dict(state='complete', completed=done, updated_at=now()))
    update_status(root, len(manifest['runs']))


def update_status(root, total):
    with (root/'status.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        statuses = [json.loads(p.read_text()) for p in (root/'workers').glob('worker-*.json')]
        done = sum(len(s['completed']) for s in statuses)
        state = 'failed' if any(s['state'] == 'failed' for s in statuses) else \
                'complete' if done == total else 'running'
        write_json(root/'status.json', dict(state=state, completed_runs=done, runs=total, updated_at=now()))


def freeze_code(launch):
    from importlib.metadata import version
    code = launch/'code'
    code.mkdir()
    for name in ('src', 'scripts', 'configs'):
        shutil.copytree(REPO/name, code/name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'), symlinks=True)
    inventory = {str(p.relative_to(code)): sha256(p) for p in code.rglob('*')
                 if p.is_file() and not p.is_symlink()}
    write_json(launch/'code_sha256.json', inventory)
    (launch/'git_commit.txt').write_text(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True))
    (launch/'working_tree.patch').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD', '--binary'], cwd=REPO))
    write_json(launch/'environment.json', dict(python=sys.version, executable=sys.executable,
        lammps=str(Path(sys.prefix)/'bin/lmp'), lammps_sha256=sha256(Path(sys.prefix)/'bin/lmp'),
        packages={name: version(name) for name in ('numpy', 'ase', 'ovito')},
        machine_settings=machine()))
    return code


def submit(root):
    from src.experiment_runner.slurm import submit_sbatch
    root = resolve_path(root).resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    config = manifest['config']
    launch = Path(manifest['launch_root'])
    if (launch/'submission.json').exists():
        raise FileExistsError(f'Campaign already submitted: {launch}')
    code = freeze_code(launch)
    workers = config['workers']
    environment = [f'PCM_PROJECT_ROOT={REPO}', f'PYTHONPATH={code}',
                   'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
                   'QT_QPA_PLATFORM=offscreen']
    command = [sys.executable, '-u', '-m', 'src.simulation.campaigns.birth_sources',
               'worker', '--campaign-root', str(root), '--workers', str(workers)]
    script = '\n'.join(['#!/bin/bash', '#SBATCH --job-name=al-birth-grid',
        '#SBATCH --partition=CPU', '#SBATCH --nodes=1', f'#SBATCH --ntasks={config["mpi_ranks"]}',
        '#SBATCH --cpus-per-task=1', '#SBATCH --mem=48G',
        f'#SBATCH --array=0-{workers-1}%{workers}', f'#SBATCH --time={config["walltime"]}',
        '#SBATCH --signal=B:USR1@600', f'#SBATCH --output={launch}/worker-%A_%a.log',
        f'#SBATCH --chdir={code}', 'set -euo pipefail',
        'exec env '+shlex.join(environment)+' '+shlex.join(command)+' --worker-index "$SLURM_ARRAY_TASK_ID"', ''])
    job = submit_sbatch(script, launch/'workers.sbatch')
    result = dict(job_id=job, submitted_at=now(), workers=workers, sources=len(manifest['runs']),
                  code=str(code), campaign_root=str(root), launch_root=str(launch))
    write_json(launch/'submission.json', result)
    write_json(root/'status.json', dict(state='submitted', **result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--config', required=True); p.add_argument('--run-name', required=True)
    p = sub.add_parser('submit'); p.add_argument('--campaign-root', required=True)
    p = sub.add_parser('worker'); p.add_argument('--campaign-root', required=True)
    p.add_argument('--worker-index', type=int, required=True); p.add_argument('--workers', type=int, required=True)
    p = sub.add_parser('assess'); p.add_argument('--directory', type=Path, required=True); p.add_argument('--step', type=int, required=True)
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        print(prepare(load_json(args.config), args.run_name))
    elif args.action == 'submit':
        print(json.dumps(submit(args.campaign_root), indent=2))
    elif args.action == 'worker':
        worker(args.campaign_root, args.worker_index, args.workers)
    else:
        assess(args.directory, args.step)


if __name__ == '__main__':
    main()
