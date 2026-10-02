"""Rerun the 150 main Al preparations at exact 0.1 ps cadence, preserving ancestry."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import traceback

from src.project_runtime.paths import REPO, dataset_path, load_json, resolve_path, storage_path
from src.project_runtime.transfer import write_json, publish_simulation, archive_failed_simulation
from .birth_sources import now, sha256, register_entry, freeze_code, execute

PROTOCOL = 'al_dense_replay_v1'


def source_input(config, record):
    t = record['temperature_K']
    return f'''log equilibration.lammps.log
units metal
dimension 3
boundary p p p
atom_style atomic
read_data prepared_liquid.lammps.data
mass 1 26.9815
pair_style meam
pair_coeff * * potential/Lee2003_Al.library.meam Al potential/Lee2003_Al.meam Al
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
timestep {config['timestep_ps']}
velocity all create {t:g} {record['velocity_seed']} mom yes rot no dist gaussian loop all
fix remove_drift all momentum {config['momentum_interval_steps']} linear 1 1 1
fix ensemble all npt temp {t:g} {t:g} 0.3 iso 0 0 3
thermo {config['sample_interval_steps']}
thermo_style custom step temp press vol pe
thermo_modify format float %.16g flush yes lost error
run {config['equilibration_steps']}
reset_timestep 0
log measurement.lammps.log
dump trajectory all custom {config['sample_interval_steps']} trajectory.lammpstrj id type x y z vx vy vz
dump_modify trajectory first yes sort id format line "%d %d %.17g %.17g %.17g %.17g %.17g %.17g"
restart {config['restart_interval_steps']} source.restart.1.bin source.restart.2.bin
run 0
run {config['measurement_steps']}
restart 0
undump trajectory
write_restart final.restart.bin
print "DENSE_AL_COMPLETE {record['run_id']}"
'''


def prepare(config):
    if config['protocol'] != PROTOCOL:
        raise ValueError('Wrong dense Al protocol')
    for key, expected in dict(timestep_ps=.002, sample_interval_steps=50, equilibration_steps=7500,
                              measurement_steps=300000, momentum_interval_steps=150).items():
        if config[key] != expected:
            raise ValueError(f'Declared physical protocol changed: {key}')
    name = config['campaign_id']
    if Path(name).name != name:
        raise ValueError('Campaign name must be a single directory')
    root = storage_path('simulation_runs')/name
    launch = storage_path('archive')/'simulation-launches'/name
    if root.exists() or launch.exists():
        raise FileExistsError(f'Preserve existing campaign: {root}, {launch}')
    if min(shutil.disk_usage(storage_path(k)).free for k in ('simulation_runs', 'archive')) < config['required_free_bytes']:
        raise OSError('Insufficient free space for canonical arrays and temporary conversion')
    plan_path = Path(config['source_plan'])
    plan = json.loads(plan_path.read_text())
    if plan['protocol'] != 'fixed_al64_v1' or len(plan['sources']) != 150:
        raise ValueError('Require all 150 ancestors from fixed_al64_v1')
    if Counter(s['role'] for s in plan['sources']) != dict(train=90, selection=15, calibration=15, test=30):
        raise ValueError('Frozen ancestor role counts changed')
    for p in config['potential_files']:
        if sha256(p['path']) != p['sha256']:
            raise ValueError(f'Potential changed: {p}')
    records = []
    for s in sorted(plan['sources'], key=lambda s: s['id']):
        trajectory = dataset_path(s['dataset'])/s['relative_trajectory_path']
        if sha256(trajectory/'manifest.json') != s['manifest_sha256']:
            raise ValueError(f'Parent trajectory identity changed: {trajectory}')
        parent = trajectory.parent
        outcome = json.loads((parent/'outcome.json').read_text())
        if outcome['state'] != 'complete' or s['atom_count'] != 70304:
            raise ValueError(f'Incomplete/wrong parent: {parent}')
        if s['lineage'] != f"independent_melt_{outcome['preparation_seed']}":
            raise ValueError(f'Parent lineage mismatch: {parent}')
        run_id = f'{name}-source{s["id"]:04d}'
        records.append(dict(run_id=run_id, run_dir=f'runs/{run_id}', source_id=s['id'],
            parent_dataset=s['dataset'], parent_directory=str(parent.resolve()),
            parent_manifest_sha256=s['manifest_sha256'], root_lineage=s['lineage'],
            split=s['role'], temperature_K=outcome['temperature_K'],
            preparation_seed=outcome['preparation_seed'], velocity_seed=outcome['velocity_seed'],
            parent_input_sha256={n: sha256(parent/n) for n in
                ('prepared_liquid.lammps.data', 'melt_final.restart.bin', 'source.in.lammps', 'outcome.json')}))
    if len({r['root_lineage'] for r in records}) != 150:
        raise ValueError('Duplicate ancestors in 150-source contract')
    root.mkdir(parents=True); launch.mkdir(parents=True)
    shutil.copy2(plan_path, launch/'parent_plan.json')
    for record in records:
        directory = root/record['run_dir']
        (directory/'potential').mkdir(parents=True)
        parent = Path(record['parent_directory'])
        for name in ('prepared_liquid.lammps.data', 'melt_final.restart.bin'):
            shutil.copy2(parent/name, directory/name)
            if sha256(directory/name) != record['parent_input_sha256'][name]:
                raise ValueError(f'Preparation copy mismatch: {directory/name}')
        shutil.copy2(parent/'source.in.lammps', directory/'parent_source.in.lammps')
        for p in config['potential_files']:
            shutil.copy2(p['path'], directory/'potential'/Path(p['path']).name)
        (directory/'source.in.lammps').write_text(source_input(config, record))
        metadata = dict(protocol=PROTOCOL, state='prepared', material='Al', atom_count=70304,
            timestep_ps=.002, sample_interval_steps=50, measurement_steps=300000,
            equilibration_steps=7500, sampling_ps=.1, measurement_ps=600,
            ensemble='NPT', thermostat_ps=.3, barostat_ps=3., pressure_bar=0.,
            momentum_removal_interval_ps=.3, preparation_reused=True,
            parent_integration_ps=.003, stopping_rule='fixed duration, independent of outcomes', **record)
        write_json(directory/'input_metadata.json', metadata)
        write_json(directory/'technical/launch_config.json', config)
        record['input_sha256'] = {str(p.relative_to(directory)): sha256(p)
                                  for p in directory.rglob('*') if p.is_file()}
    manifest = dict(protocol=PROTOCOL, created_at=now(), config=config, runs=records,
        root=str(root), launch_root=str(launch), parent_plan_sha256=sha256(plan_path),
        parent_release_identity=plan['identity'], roles=dict(Counter(r['split'] for r in records)),
        ancestry_policy='New dynamics sharing all 150 original melts; never independent of their parent trajectories.',
        precision_policy='Verified float16 positions/velocities, float32 boxes, exact IDs/timesteps; native restarts retained.')
    write_json(launch/'manifest.json', manifest)
    (launch/'manifest.sha256').write_text(sha256(launch/'manifest.json')+'\n')
    (root/'manifest.json').symlink_to(launch/'manifest.json')
    write_json(launch/'status.json', dict(state='prepared', completed=0, total=150))
    register_entry(config['campaign_id'], dict(root='simulation_runs', path=config['campaign_id'], kind='simulation',
        dependencies=sorted({r['parent_dataset'] for r in records}|{'potentials'}),
        metadata=dict(materials=['Al'], potential_ids=['al-lee2003-meam'], role='raw_dynamics', classification='research',
            title='Main Al sources rerun at exact 0.1 ps with velocities',
            description='150 prepared-liquid descendants, 70304 atoms, 600 ps measurement, 2 fs integration; frozen ancestor roles retained.',
            ancestry='Same 150 independent melt ancestors as fixed_al64_v1; not 150 additional independent lineages.',
            evidence=[str(launch/'manifest.json'), str(launch/'status.json')])))
    return launch


def manifest_at(launch):
    launch = Path(launch)
    if sha256(launch/'manifest.json') != (launch/'manifest.sha256').read_text().strip():
        raise ValueError('Dense Al campaign manifest changed')
    return json.loads((launch/'manifest.json').read_text())


def verify_code(launch):
    code = launch/'code'
    if not Path(__file__).resolve().is_relative_to(code):
        raise ValueError('Workers must execute frozen campaign code')
    for filename, digest in json.loads((launch/'code_sha256.json').read_text()).items():
        if sha256(code/filename) != digest:
            raise ValueError(f'Frozen code changed: {filename}')


def publish_record(directory, record, failed=False):
    identifier = record.get('failure_id', record['run_id']+'-failed') if failed else record['run_id']
    if failed:
        result = archive_failed_simulation(directory, identifier=identifier)
    else:
        result = publish_simulation(directory, identifier=identifier, move=True)
    from src.project_runtime.paths import machine
    catalog = resolve_path(machine()['catalog'])
    with catalog.with_suffix('.json.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = json.loads(catalog.read_text())
        entry = value['datasets'][identifier]
        entry['dependencies'] = sorted(set(entry['dependencies']+[record['parent_dataset']]))
        metadata = json.loads((Path(result['destination'])/'input_metadata.json').read_text())
        entry['metadata'] = dict(materials=['Al'], potential_ids=['al-lee2003-meam'],
            role='raw_dynamics', classification='incomplete_or_rejected' if failed else 'research',
            ancestry=record['root_lineage'], split=record['split'], source_id=record['source_id'],
            sampling_ps=metadata['sampling_ps'], protocol=metadata['protocol'],
            description=f'Exact {metadata["sampling_ps"]:g} ps sampled descendant of retained Al prepared liquid; 2 fs integration.',
            evidence=[str(Path(result['destination'])/'input_metadata.json')])
        write_json(catalog, value)


def run_source(manifest, record):
    directory = Path(manifest['root'])/record['run_dir']
    if (directory/'status.json').exists():
        raise FileExistsError(f'Inspect existing attempt before rerunning: {directory}')
    for name, digest in record['input_sha256'].items():
        if sha256(directory/name) != digest:
            raise ValueError(f'Prepared input changed: {directory/name}')
    status = dict(state='running', phase='dynamics', started_at=now(),
        slurm_job_id=os.environ['SLURM_JOB_ID'], host=socket.gethostname(), run_id=record['run_id'])
    write_json(directory/'status.json', status)
    try:
        if 'recovery' in record:
            from .dense_al_continuation import recover_dynamics
            elapsed = recover_dynamics(directory, manifest, record)
        else:
            elapsed = execute(directory, 'source.in.lammps', 'source.stdout.log', manifest['config']['mpi_ranks'])
        if f'DENSE_AL_COMPLETE {record["run_id"]}' not in (directory/'source.stdout.log').read_text().splitlines():
            raise ValueError(f'Missing final dynamics marker: {directory}')
        if not (directory/'final.restart.bin').stat().st_size:
            raise ValueError('Missing native final restart')
        metadata = json.loads((directory/'input_metadata.json').read_text())
        contracts = {PROTOCOL: (50, 'dense-al'), 'al_dense_replay_001ps_v1': (5, 'dense-al-001ps')}
        from .dense_al_half_stop import PROTOCOLS as HALF_PROTOCOLS
        contracts.update({p: (interval, 'dense-al-half') for interval,p in HALF_PROTOCOLS.items()})
        interval, conversion = contracts[metadata['protocol']]
        if metadata['sample_interval_steps'] != interval or metadata['timestep_ps'] != .002:
            raise ValueError(f'Input cadence differs from declared protocol: {directory}')
        if metadata['protocol'] in HALF_PROTOCOLS.values():
            termination = json.loads((directory/'technical/stop-monitor/outcome.json').read_text())
            if termination['state'] != 'dynamics_complete' or termination['protocol'] != metadata['protocol']:
                raise ValueError('Missing or mismatched halfway endpoint certificate')
            metadata.update(maximum_measurement_steps=metadata['measurement_steps'],
                measurement_steps=termination['final_step'], measurement_ps=termination['measurement_ps'],
                termination=termination)
            status['termination'] = termination
        expected_frames = metadata['measurement_steps']//interval+1
        metadata.update(state='dynamics_complete', source_sha256=sha256(directory/'trajectory.lammpstrj'))
        if 'recovery' in record:
            metadata['restart_recovery'] = record['recovery']
        write_json(directory/'metadata.json', metadata)
        status.update(phase='conversion', dynamics_seconds=elapsed)
        write_json(directory/'status.json', status)
        command = [sys.executable, str(Path(__file__).resolve().parents[3]/'scripts/convert_trajectory.py'),
                   conversion, str(directory), '--delete-source']
        with (directory/'conversion.log').open('x') as output:
            subprocess.run(command, check=True, stdout=output, stderr=subprocess.STDOUT)
        report = json.loads((directory/'paired_conversion.json').read_text())
        if report['frame_count'] != expected_frames:
            raise ValueError('Wrong dense source frame count')
        if 'recovery' in record:
            from .dense_al_continuation import finish_recovery
            finish_recovery(directory)
        status.update(state='complete', phase='converted', completed_at=now(), frame_count=expected_frames,
            native_restart_sha256={n:sha256(directory/n) for n in ('melt_final.restart.bin','final.restart.bin')})
        write_json(directory/'outcome.json', status)
        write_json(directory/'status.json', status)
    except BaseException:
        status.update(state='failed', finished_at=now(), error=traceback.format_exc())
        write_json(directory/'status.json', status)
        publish_record(directory, record, failed=True)
        raise
    publish_record(directory, record)


def worker(launch, wave, index):
    manifest = manifest_at(launch); verify_code(launch)
    if int(os.environ['SLURM_NTASKS']) != manifest['config']['mpi_ranks']:
        raise ValueError('MPI allocation differs from declared source protocol')
    assignments = json.loads((launch/f'wave-{wave:03d}/assignments.json').read_text())
    def interrupted(signum, frame):
        raise InterruptedError(f'Dense Al worker received signal {signum}')
    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    for i in assignments[index]:
        run_source(manifest, manifest['runs'][i])


def sbatch(script, path):
    path.write_text(script); path.chmod(0o755)
    # Submitting from a GPU allocation must not leak inherited resource options.
    env = {k:v for k,v in os.environ.items() if not k.startswith(('SBATCH_', 'SLURM_'))}
    result = subprocess.run(['sbatch','--parsable',str(path)], env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f'Slurm submission failed: {result.stderr}')
    return result.stdout.strip().split(';')[0]


def submission_slots(config):
    """Use Slurm's quota counter, which differs from expanded pending-array rows."""
    qos = config.get('qos', 'normal')
    state = subprocess.check_output(['scontrol', 'show', 'assoc_mgr', 'flags=qos'], text=True)
    blocks = re.split(r'^QOS=', state, flags=re.MULTILINE)
    block, = [b for b in blocks if b.startswith(qos+'(')]
    pattern = (r'^\s+'+re.escape(os.environ['USER'])+r'\('+str(os.getuid())+
               r'\)\s*\n\s+[^\n]*MaxSubmitJobsPU=(\d+)\((\d+)\)')
    match = re.search(pattern, block, re.MULTILINE)
    if match is None:
        raise RuntimeError(f'Cannot read declared {qos} submission quota for {os.environ["USER"]}')
    maximum, used = map(int, match.groups())
    return dict(qos=qos, maximum=min(maximum, config['max_submitted_jobs']), used=used)


def submit(launch, wave=0):
    manifest = manifest_at(launch); config = manifest['config']; root = Path(manifest['root'])
    if wave == 0:
        if (launch/'code').exists():
            raise FileExistsError('Inspect frozen initial submission before retry')
        freeze_code(launch)
    else:
        verify_code(launch)
    pending = []
    completed = 0
    for i,r in enumerate(manifest['runs']):
        directory = root/r['run_dir']; status = directory/'status.json'
        if not status.exists():
            pending.append(i)
        elif json.loads(status.read_text())['state'] == 'complete' and directory.is_symlink():
            completed += 1
        else:
            raise RuntimeError(f'Unfinished/failed source requires inspection before next wave: {directory}')
    if not pending:
        write_json(launch/'status.json', dict(state='complete', completed=completed, total=len(manifest['runs']), finished_at=now()))
        return dict(state='complete', completed=completed)
    quota = submission_slots(config)
    workers = min(config['workers'], quota['maximum']-quota['used']-1, len(pending))
    if workers < 1:
        raise RuntimeError(f'No Slurm slots for workers plus preservation/successor controller: {quota}')
    # A 0.01 ps source has ten times as much output; reserve a whole lane for it.
    pending.sort(key=lambda i: manifest['runs'][i].get('queue_priority',1))
    capacity = 1 if any(manifest['runs'][i].get('sample_interval_steps') == 5 for i in pending[:workers]) else config['sources_per_worker_wave']
    selected = pending[:workers*capacity]
    assignments = [selected[j::workers] for j in range(workers)]
    wave_root = launch/f'wave-{wave:03d}'; wave_root.mkdir()
    write_json(wave_root/'assignments.json', assignments)
    code = launch/'code'
    env = ['PCM_PROJECT_ROOT='+str(REPO), 'PYTHONPATH='+str(code), 'OMP_NUM_THREADS=1',
           'OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','QT_QPA_PLATFORM=offscreen']
    command = [sys.executable,'-u','-m','src.simulation.campaigns.dense_al']
    base = ['#!/bin/bash','#SBATCH --partition=CPU',f'#SBATCH --qos={quota["qos"]}', '#SBATCH --nodes=1',
            f'#SBATCH --chdir={code}','set -euo pipefail']
    # SBATCH directives must precede executable shell lines.
    worker_name = 'al-dense-001ps' if capacity == 1 and any(manifest['runs'][i].get('sample_interval_steps') == 5 for i in selected) else 'al-dense-010ps'
    worker_lines = base[:-1]+[f'#SBATCH --job-name={worker_name}',f'#SBATCH --ntasks={config["mpi_ranks"]}',
        '#SBATCH --cpus-per-task=1','#SBATCH --mem=64G',f'#SBATCH --time={config["walltime"]}',
        f'#SBATCH --array=0-{workers-1}%{workers}', '#SBATCH --signal=B:USR1@600',
        f'#SBATCH --output={wave_root}/worker-%A_%a.log',base[-1],
        'exec env '+shlex.join(env+command+['worker','--launch',str(launch),'--wave',str(wave)])+' --index "$SLURM_ARRAY_TASK_ID"','']
    job = sbatch('\n'.join(worker_lines), wave_root/'workers.sbatch')
    write_json(wave_root/'submission.json', dict(worker_job=job, workers=workers, selected=selected, submitted_at=now()))
    controller_lines = base[:-1]+['#SBATCH --job-name=al-dense-next','#SBATCH --ntasks=1',
        '#SBATCH --cpus-per-task=1','#SBATCH --mem=4G','#SBATCH --time=12:00:00',
        f'#SBATCH --dependency=afterany:{job}',f'#SBATCH --output={wave_root}/controller-%j.log',base[-1],
        'exec env '+shlex.join(env+command+['collect','--launch',str(launch),'--wave',str(wave)]),'']
    controller = sbatch('\n'.join(controller_lines), wave_root/'controller.sbatch')
    result = dict(state='submitted', wave=wave, worker_job=job, controller_job=controller,
                  workers=workers, selected=len(selected), completed=completed, total=len(manifest['runs']),
                  submission_quota=quota, submitted_at=now())
    write_json(wave_root/'submission.json', result); write_json(launch/'status.json', result)
    return result


def collect(launch, wave):
    manifest = manifest_at(launch); verify_code(launch)
    assignments = json.loads((launch/f'wave-{wave:03d}/assignments.json').read_text())
    failures = []
    for i in [i for lane in assignments for i in lane]:
        record = manifest['runs'][i]; directory = Path(manifest['root'])/record['run_dir']
        path = directory/'status.json'
        if not path.exists():
            failures.append(record['run_id']+' (never started)'); continue
        status = json.loads(path.read_text())
        if status['state'] == 'complete':
            if not directory.is_symlink():
                publish_record(directory, record)
        else:
            failures.append(record['run_id'])
            from src.project_runtime.paths import catalog
            if record.get('failure_id', record['run_id']+'-failed') not in catalog():
                write_json(path, dict(state='failed', previous_status=status, finished_at=now(),
                    reason='Slurm allocation ended before successful conversion/publication'))
                publish_record(directory, record, failed=True)
    if failures:
        write_json(launch/'status.json', dict(state='failed', wave=wave, failures=failures, updated_at=now()))
        raise RuntimeError(f'Preserved failed wave; inspect before continuing: {failures}')
    return submit(launch, wave+1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--config', required=True)
    p = sub.add_parser('prepare-continuation'); p.add_argument('--config', required=True)
    p = sub.add_parser('add-dense-sources'); p.add_argument('--config', required=True)
    p = sub.add_parser('prepare-half-stop'); p.add_argument('--config', required=True)
    for name in ('submit','worker','collect'):
        p = sub.add_parser(name); p.add_argument('--launch', type=Path, required=True)
        if name != 'submit': p.add_argument('--wave', type=int, required=True)
        if name == 'worker': p.add_argument('--index', type=int, required=True)
    args = parser.parse_args(argv)
    if args.action == 'prepare': print(prepare(load_json(args.config)))
    elif args.action == 'prepare-continuation':
        from .dense_al_continuation import prepare_continuation
        print(prepare_continuation(load_json(args.config)))
    elif args.action == 'add-dense-sources':
        from .dense_al_additions import add_dense_sources
        print(json.dumps(add_dense_sources(load_json(args.config)), indent=2))
    elif args.action == 'prepare-half-stop':
        from .dense_al_half_queue import prepare_half_queue
        print(json.dumps(prepare_half_queue(load_json(args.config)), indent=2))
    elif args.action == 'submit': print(json.dumps(submit(args.launch), indent=2))
    elif args.action == 'collect': print(json.dumps(collect(args.launch,args.wave), indent=2))
    else: worker(args.launch,args.wave,args.index)


if __name__ == '__main__':
    main()
