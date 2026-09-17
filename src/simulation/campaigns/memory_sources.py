"""Fresh independent Al melts with paired precision and outcome-blind splits."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import traceback

from ase.build import bulk
from ase.io import write

from src.project_runtime.paths import load_json, storage_path, dataset_path
from src.project_runtime.transfer import write_json, publish_simulation, archive_failed_simulation
from src.data.conversion.memory_pair import PROTOCOL
from . import independent_meam_source as source


def now():
    return datetime.now(timezone.utc).isoformat()


def specifications(config, name):
    if config['protocol'] != PROTOCOL:
        raise ValueError(f"Wrong memory-source protocol: {config['protocol']}")
    if config['measurement_steps'] % config['sample_interval_steps']:
        raise ValueError('Measurement duration must contain an integral number of sample intervals')
    splits = [split for split, count in config['sources_per_temperature'].items() for _ in range(count)]
    if set(splits) != {'train', 'val', 'sealed_test'}:
        raise ValueError('Declare train, val and sealed_test lineages before simulation')
    records = []
    for temperature in config['temperatures_K']:
        for index, split in enumerate(splits):
            def seed(role):
                digest = hashlib.sha256(f"{config['campaign_seed']}:{temperature:g}:{index}:{role}".encode()).digest()
                return int.from_bytes(digest[:8], 'little') % 899_999_999+1
            melt, velocity = seed('melt'), seed('quench')
            run_id = f'{name}-source{len(records):03d}-T{temperature:g}'
            records.append(dict(run_id=run_id, run_index=len(records), run_dir=f'runs/{run_id}',
                root_lineage=f'independent_melt_{melt}', parent_trajectory_id=None,
                temperature_K=temperature, split=split, preparation_seed=melt, velocity_seed=velocity))
    seeds = [r[key] for r in records for key in ('preparation_seed', 'velocity_seed')]
    if len(seeds) != len(set(seeds)):
        raise ValueError('Fresh melt and velocity seeds must be distinct')
    return records


def source_input(config, record):
    """Same NPT transition kernel as the retained source family; denser output only."""
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
timestep 0.003
velocity all create {t:g} {record['velocity_seed']} mom yes rot no dist gaussian loop all
fix remove_drift all momentum 100 linear 1 1 1
fix ensemble all npt temp {t:g} {t:g} 0.3 iso 0 0 3
thermo {config['sample_interval_steps']}
thermo_style custom step temp press vol pe
thermo_modify format float %.16g flush yes
run 5000
reset_timestep 0
log measurement.lammps.log
dump trajectory all custom {config['sample_interval_steps']} trajectory.lammpstrj id type x y z vx vy vz
dump_modify trajectory first yes sort id format line "%d %d %.17g %.17g %.17g %.17g %.17g %.17g"
restart 5000 source.restart.1.bin source.restart.2.bin
run 0
run {config['measurement_steps']}
restart 0
undump trajectory
write_restart final.restart.bin
print "MEMORY_SOURCE_MEASUREMENT_COMPLETE {record['run_id']}"
'''


def prepare(config, name):
    if not name or Path(name).name != name or name in {'.', '..'}:
        raise ValueError('Run name must be a single new directory name')
    records = specifications(config, name)
    old_seeds, prior_manifests = set(), {}
    for identifier in config['excluded_source_datasets']:
        path = dataset_path(identifier)/'manifest.json'
        prior = json.loads(path.read_text())
        old_seeds.update(int(r[key]) for r in prior['runs'] for key in ('preparation_seed', 'velocity_seed'))
        prior_manifests[identifier] = source._sha256_file(path)
    if any(r[key] in old_seeds for r in records for key in ('preparation_seed', 'velocity_seed')):
        raise ValueError('New source seed collides with an existing melt or velocity seed')
    if config['mpi_ranks'] != source.MPI_RANKS:
        raise ValueError(f'This source workflow uses the validated {source.MPI_RANKS}-rank launcher')
    root = storage_path('simulation_runs')/name
    if root.exists():
        raise FileExistsError(f'Preserve existing memory-source campaign: {root}')
    if shutil.disk_usage(root.parent).free < config['required_free_bytes']:
        raise OSError('Insufficient SCRATCH space for paired trajectories and temporary text')
    potentials = [(Path(p['path']), p['sha256']) for p in config['potential_files']]
    for path, expected in potentials:
        if source._sha256_file(path) != expected:
            raise ValueError(f'Potential checksum mismatch: {path}')
    root.mkdir()
    atoms = bulk('Al', 'fcc', a=4.05, cubic=True).repeat((26, 26, 26))
    if len(atoms) != source.EXPECTED_ATOM_COUNT:
        raise ValueError('FCC atom count differs from the retained source protocol')
    initial = root/'initial_fcc.lammps.data'
    write(initial, atoms, format='lammps-data', atom_style='atomic', specorder=('Al',))
    for record in records:
        directory = root/record['run_dir']
        (directory/'potential').mkdir(parents=True)
        (directory/'technical').mkdir()
        shutil.copy2(initial, directory/initial.name)
        for path, _ in potentials:
            shutil.copy2(path, directory/'potential'/path.name)
        melt = source.render_melt_input(record).replace('../../initial_fcc', 'initial_fcc').replace('../../potential/', 'potential/')
        (directory/'melt.in.lammps').write_text(melt)
        (directory/'source.in.lammps').write_text(source_input(config, record))
        metadata = dict(protocol=PROTOCOL, state='prepared', atom_count=len(atoms),
            timestep_ps=.003, melt_steps=100000, equilibration_steps=5000,
            measurement_steps=config['measurement_steps'], sample_interval_steps=config['sample_interval_steps'],
            ensemble='NPT', thermostat='Nose-Hoover', thermostat_ps=.3, barostat_ps=3., pressure_bar=0.,
            momentum_removal_interval_ps=.3, velocity_reset_before_equilibration=True,
            measurement_interventions=[], stopping_rule='fixed duration, independent of outcomes', **record)
        write_json(directory/'input_metadata.json', metadata)
        write_json(directory/'technical/launch_config.json', config)
        files = ['initial_fcc.lammps.data', 'melt.in.lammps', 'source.in.lammps', 'input_metadata.json',
                 *['potential/'+p.name for p, _ in potentials]]
        record['input_sha256'] = {file: source._sha256_file(directory/file) for file in files}
    manifest = dict(schema_version=1, protocol=PROTOCOL, created_at=now(), config=config, runs=records,
        excluded_source_manifest_sha256=prior_manifests,
        initial_fcc_sha256=source._sha256_file(initial), counts=dict(Counter(r['split'] for r in records)),
        sealed_test_policy='No future observables, target distributions or model scores inspected before protocol lock; numerical integrity and melt QC only.',
        precision_policy='Canonical float16 plus required matched float32 audit reference; native restarts retained.',
        simulator=subprocess.check_output([str(Path(sys.prefix)/'bin/lmp'), '-h'],
            env=source._lammps_environment(), text=True).splitlines()[1:4])
    write_json(root/'manifest.json', manifest)
    digest = source._sha256_file(root/'manifest.json')
    (root/'manifest.sha256').write_text(digest+'\n')
    for record in records:
        write_json(root/record['run_dir']/'campaign_manifest.json', manifest)
    write_json(root/'status.json', dict(state='prepared', runs=len(records), manifest_sha256=digest))
    return root


def run_source(root, manifest, record):
    directory = root/record['run_dir']
    status_path = directory/'status.json'
    if status_path.exists():
        raise FileExistsError(f'Inspect existing simulation attempt before retry: {status_path}')
    for relative, digest in record['input_sha256'].items():
        if source._sha256_file(directory/relative) != digest:
            raise ValueError(f'Modified prepared source: {directory/relative}')
    status = dict(state='running', started_at=now(), slurm_job_id=os.environ['SLURM_JOB_ID'],
                  host=socket.gethostname(), pid=os.getpid(), run_id=record['run_id'])
    write_json(status_path, status)
    try:
        melt_seconds = source._run_lammps(directory, 'melt.in.lammps', 'melt.stdout.log', launcher='srun_pmi2')
        steps, positions, cells = source._read_lammps_dump(directory/'melt_validation.lammpstrj')
        if steps.tolist() != [100000]:
            raise ValueError(f'Melt did not reach the declared duration: {steps}')
        # Predeclared preparation QC only; no measurement PTM/outcome evaluation.
        validation = source._liquid_validation(positions[0], cells[0])
        write_json(directory/'melt_validation.json', validation)
        measurement_seconds = source._run_lammps(directory, 'source.in.lammps', 'source.stdout.log', launcher='srun_pmi2')
        marker = f"MEMORY_SOURCE_MEASUREMENT_COMPLETE {record['run_id']}"
        if marker not in (directory/'source.stdout.log').read_text().splitlines():
            raise ValueError(f'Missing completed-dynamics marker: {directory}')
        for file in ('final.restart.bin', 'melt_final.restart.bin'):
            if not (directory/file).is_file() or (directory/file).stat().st_size == 0:
                raise ValueError(f'Missing native restart: {directory/file}')
        metadata = json.loads((directory/'input_metadata.json').read_text())
        metadata.update(state='dynamics_complete', source_sha256=source._sha256_file(directory/'trajectory.lammpstrj'))
        write_json(directory/'metadata.json', metadata)
        command = [sys.executable, '-u', '-m', 'src.data.conversion.cli', 'memory-pair', str(directory)]
        if manifest['config']['delete_verified_source_text']:
            command.append('--delete-source')
        with (directory/'conversion.log').open('x') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        receipt = json.loads((directory/'paired_conversion.json').read_text())
        status.update(state='complete', completed_at=now(), frame_count=receipt['frame_count'],
                      melt_seconds=melt_seconds, measurement_seconds=measurement_seconds,
                      prepared_liquid_sha256=source._sha256_file(directory/'prepared_liquid.lammps.data'),
                      native_restart_sha256={f: source._sha256_file(directory/f)
                          for f in ('melt_final.restart.bin', 'final.restart.bin')},
                      manifest_sha256=source._sha256_file(root/'manifest.json'))
        write_json(directory/'outcome.json', status)
        write_json(status_path, status)
    except BaseException:
        status.update(state='failed', finished_at=now(), error=traceback.format_exc(), partial_artifacts_preserved=True)
        write_json(status_path, status)
        archive_failed_simulation(directory, identifier=record['run_id']+'-failed')
        raise
    publish_simulation(directory, identifier=record['run_id'], move=True)


def worker(root, index, count):
    root = Path(root).resolve()
    if source._sha256_file(root/'manifest.json') != (root/'manifest.sha256').read_text().strip():
        raise ValueError('Immutable simulation manifest changed')
    manifest = json.loads((root/'manifest.json').read_text())
    if int(os.environ.get('SLURM_NTASKS', '0')) != source.MPI_RANKS:
        raise ValueError(f'This worker requires {source.MPI_RANKS} allocated CPU ranks')
    if not 0 <= index < count:
        raise ValueError('Worker index is outside the declared worker count')
    worker_status = root/'workers'/f'worker-{index}.json'
    completed = []
    try:
        for record in manifest['runs'][index::count]:
            write_json(worker_status, dict(state='running', current=record['run_id'], completed=completed, updated_at=now()))
            run_source(root, manifest, record)
            completed.append(record['run_id'])
        write_json(worker_status, dict(state='complete', completed=completed, updated_at=now()))
    except BaseException:
        write_json(worker_status, dict(state='failed', current=record['run_id'], completed=completed,
                                       error=traceback.format_exc(), updated_at=now()))
        raise
    finally:
        with (root/'status.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            statuses = [json.loads(p.read_text()) for p in (root/'workers').glob('worker-*.json')]
            done = sum(len(s['completed']) for s in statuses)
            state = 'failed' if any(s['state'] == 'failed' for s in statuses) else \
                'complete' if done == len(manifest['runs']) else 'running'
            write_json(root/'status.json', dict(state=state, completed_runs=done, runs=len(manifest['runs']), updated_at=now()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--config', required=True); p.add_argument('--run-name', required=True)
    p = sub.add_parser('run-worker'); p.add_argument('--campaign-root', required=True)
    p.add_argument('--worker-index', type=int, required=True); p.add_argument('--workers', type=int, required=True)
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        print(prepare(load_json(args.config), args.run_name))
    else:
        worker(args.campaign_root, args.worker_index, args.workers)


if __name__ == '__main__':
    main()
