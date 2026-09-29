"""Queue replicated Ta position-conditioned shots through the elemental producer."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from src.project_runtime.paths import REPO, load_json, machine, resolve_path, storage_path
from src.project_runtime.transfer import write_json, publish_simulation, archive_failed_simulation
from .elemental import sha256


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(config_path):
    from .birth_sources import register_entry
    config = load_json(config_path)
    base = load_json(config['base_config'])
    if base['protocol'] != 'ta-position-branches' or base['material'] != 'Ta':
        raise ValueError('This queue preserves the archived-position Ta elemental protocol.')
    name = config['campaign_id']
    if Path(name).name != name or name in {'.', '..'}:
        raise ValueError(f'Invalid campaign ID: {name}')
    launch = storage_path('archive') / 'simulation-launches' / name
    if launch.exists():
        raise FileExistsError(launch)
    for potential in base['potential_files']:
        if sha256(potential['path']) != potential['sha256']:
            raise ValueError(f'Potential checksum mismatch: {potential}')
    records, seeds = [], set(config['excluded_velocity_seeds'])
    for index, parent in enumerate(config['parents']):
        source = Path(parent['source'])
        if sha256(source) != parent['source_sha256']:
            raise ValueError(f'Parent checksum mismatch: {source}')
        with source.open() as handle:
            header = [handle.readline().strip() for _ in range(9)]
        if (header[0] != 'ITEM: TIMESTEP' or int(header[1]) != parent['source_step']
                or header[2] != 'ITEM: NUMBER OF ATOMS' or int(header[3]) != parent['atom_count']
                or header[4] != 'ITEM: BOX BOUNDS pp pp pp'
                or header[8] != 'ITEM: ATOMS id type x y z'):
            raise ValueError(f'Parent header differs from recorded Ta observation: {source}: {header}')
        run_id = f'{name}-parent{index:02d}'
        directory = storage_path('simulation_runs') / run_id
        if directory.exists():
            raise FileExistsError(directory)
        branches = []
        for shot in range(config['shots_per_parent']):
            seed_text = f'{config["campaign_seed"]}:{parent["name"]}:{shot}'
            seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], 'little') % 899999999 + 1
            if seed in seeds:
                raise ValueError(f'Velocity seed collision: {seed}')
            seeds.add(seed)
            branches.append(dict(name=f'shot{shot:02d}', source=str(source),
                source_sha256=parent['source_sha256'], source_step=parent['source_step'],
                velocity_seed=seed, parent_id=parent['name'], replica=shot,
                ancestry_group=config['ancestry_group']))
        recipe = dict(base, atom_count=parent['atom_count'], branches=branches)
        filename = launch / 'recipes' / f'parent{index:02d}.json'
        records.append(dict(index=index, run_id=run_id, parent=parent, branches=branches,
                            config_path=str(filename), directory=str(directory)))
        # All input validation precedes creation of a launch directory.
        records[-1]['recipe'] = recipe
    if len({r['parent']['name'] for r in records}) != len(records):
        raise ValueError('Parent names must be unique.')
    launch.mkdir(parents=True)
    for record in records:
        write_json(record['config_path'], record.pop('recipe'))
        record['config_sha256'] = sha256(record['config_path'])
    manifest = dict(schema_version=1, created_at=now(), config=config, runs=records,
                    parent_count=len(records), shot_count=sum(len(r['branches']) for r in records),
                    launch_root=str(launch), material='Ta', potential_files=base['potential_files'],
                    source_generating_potential='unknown for archived positions',
                    shooting_potential='Zhong 2014 Ta EAM',
                    scientific_scope='Finite-horizon position-conditioned outcomes; no committor basins defined.')
    write_json(launch / 'manifest.json', manifest)
    write_json(launch / 'status.json', dict(state='prepared', created_at=now()))
    metadata = dict(title='Ta shooting: six archived parents, four fresh velocity replicas each',
                    materials=['Ta'], role='raw_dynamics', classification='research',
                    potential_ids=['ta-zhong2014-eam'],
                    description='Prepared replicated full-cell 24 ps NPT shots at 1900 K; manifest records current completion separately.',
                    ancestry=config['ancestry_group'],
                    limitations=['Archived parent generation potential and precise common ancestry are unknown.',
                                 'Parents and shots must not be split as independent preparation lineages.',
                                 'Four shots per parent are an exploratory pilot, not a precise committor estimate.'],
                    evidence=[str(launch / 'manifest.json'), str(launch / 'status.json')])
    register_entry(name, dict(root='archive', path=f'simulation-launches/{name}', kind='simulation',
                             dependencies=config['dependencies'], metadata=metadata))
    return manifest


def submit(launch):
    from .birth_sources import freeze_code
    from src.experiment_runner.slurm import submit_sbatch
    launch = resolve_path(launch)
    manifest = json.loads((launch / 'manifest.json').read_text())
    config = manifest['config']
    if (launch / 'submission.json').exists():
        raise FileExistsError(launch / 'submission.json')
    code = freeze_code(launch)
    settings = machine()
    write_json(launch / 'machine.json', settings)
    environment_record = json.loads((launch / 'environment.json').read_text())
    executable = Path(settings['execution']['lammps']).resolve()
    environment_record['execution_lammps'] = str(executable)
    environment_record['execution_lammps_sha256'] = sha256(executable)
    write_json(launch / 'environment.json', environment_record)
    # The elemental provenance recorder expects a Git checkout. This independent
    # artifact checkout represents the frozen bytes; original HEAD/diff are alongside it.
    subprocess.run(['git', 'init', '--quiet', str(code)], check=True)
    subprocess.run(['git', 'add', 'src', 'scripts', 'configs'], cwd=code, check=True)
    subprocess.run(['git', '-c', 'user.name=Simulation snapshot', '-c',
                    'user.email=simulation-snapshot@localhost', 'commit', '--quiet',
                    '-m', 'Frozen Ta position shooting source'], cwd=code, check=True)
    environment = [f'PCM_PROJECT_ROOT={REPO}', f'PYTHONPATH={code}',
                   f'PCM_MACHINE_CONFIG={launch / "machine.json"}',
                   'CONDA_DEFAULT_ENV=pointnet-torch214', 'OMP_NUM_THREADS=1',
                   'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'QT_QPA_PLATFORM=offscreen']
    worker = [sys.executable, '-u', '-m', 'src.simulation.campaigns.position_shooting',
              'worker', '--launch', str(launch)]
    script = '\n'.join(['#!/bin/bash', '#SBATCH --job-name=ta-position-shots',
        f'#SBATCH --partition={config["partition"]}', '#SBATCH --nodes=1',
        f'#SBATCH --ntasks={config["mpi_ranks"]}', '#SBATCH --cpus-per-task=1',
        f'#SBATCH --mem={config["memory"]}', f'#SBATCH --time={config["walltime"]}',
        f'#SBATCH --array=0-{len(manifest["runs"])-1}%{config["concurrent_jobs"]}',
        f'#SBATCH --output={launch}/worker-%A_%a.log', f'#SBATCH --chdir={code}',
        'set -euo pipefail', 'exec env ' + shlex.join(environment) + ' ' + shlex.join(worker)
        + ' --index "$SLURM_ARRAY_TASK_ID"', ''])
    job = submit_sbatch(script, launch / 'workers.sbatch')
    receipt = dict(job_id=job, submitted_at=now(), launch=str(launch), code=str(code),
                   parent_count=len(manifest['runs']), shot_count=manifest['shot_count'])
    # Persist the first successful submission even if submitting preservation fails.
    write_json(launch / 'submission.json', receipt)
    collect = [sys.executable, '-u', '-m', 'src.simulation.campaigns.position_shooting',
               'collect', '--launch', str(launch)]
    script = '\n'.join(['#!/bin/bash', '#SBATCH --job-name=ta-shots-preserve',
        f'#SBATCH --partition={config["partition"]}', '#SBATCH --nodes=1', '#SBATCH --ntasks=1',
        '#SBATCH --cpus-per-task=2', '#SBATCH --mem=8G', '#SBATCH --time=12:00:00',
        f'#SBATCH --dependency=afterany:{job}', f'#SBATCH --output={launch}/preserve-%j.log',
        f'#SBATCH --chdir={code}', 'set -euo pipefail',
        'exec env ' + shlex.join(environment) + ' ' + shlex.join(collect), ''])
    receipt['preservation_job_id'] = submit_sbatch(script, launch / 'preserve.sbatch')
    write_json(launch / 'submission.json', receipt)
    write_json(launch / 'status.json', dict(state='submitted', **receipt))
    return receipt


def worker(launch, index):
    launch = resolve_path(launch)
    manifest = json.loads((launch / 'manifest.json').read_text())
    record = manifest['runs'][index]
    if sha256(record['config_path']) != record['config_sha256']:
        raise ValueError(f'Prepared configuration changed: {record["config_path"]}')
    code = Path(__file__).resolve().parents[3]
    command = [sys.executable, '-u', str(code / 'scripts/run_lammps_campaign.py'),
               'elemental', 'run', '--config', record['config_path'],
               '--run-name', record['run_id'], '--ranks', str(manifest['config']['mpi_ranks'])]
    subprocess.run(command, cwd=code, check=True)


def collect(launch):
    launch = resolve_path(launch)
    manifest = json.loads((launch / 'manifest.json').read_text())
    receipt = json.loads((launch / 'submission.json').read_text())
    # Query the user's queue: squeue -j fails once a completed array is purged.
    queue = subprocess.check_output(['squeue', '-h', '-u', str(os.getuid()),
                                     '-o', '%F %i %T'], text=True)
    active = [line for line in queue.splitlines() if line.split()[0] == str(receipt['job_id'])]
    if active:
        raise RuntimeError(f'Cannot preserve while shooting jobs remain active: {active}')
    rows = []
    for record in manifest['runs']:
        directory = Path(record['directory'])
        if not directory.exists():
            rows.append(dict(run_id=record['run_id'], state='not_started', completed_shots=0))
            continue
        status_path = directory / 'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {'state': 'missing_status'}
        completed = []
        for branch in record['branches']:
            outcome = directory / 'branches' / branch['name'] / 'outcome.json'
            if outcome.exists() and json.loads(outcome.read_text())['state'] == 'complete':
                completed.append(branch['name'])
        if status['state'] == 'complete' and len(completed) != len(record['branches']):
            raise ValueError(f'Completed parent has missing shot outcomes: {directory}: {completed}')
        if status['state'] != 'complete':
            if status_path.exists():
                write_json(directory / 'technical/status_before_preservation.json', status)
            status = dict(state='failed', previous_status=status,
                          reason='Slurm shooting allocation ended without complete parent campaign',
                          recorded_at=now(), completed_branches=completed)
            write_json(status_path, status)
            published_id = record['run_id'] + '-stopped'
            publication = archive_failed_simulation(directory, identifier=published_id)
            destination = publication['destination']
        else:
            published_id = record['run_id']
            if not directory.is_symlink():
                publish_simulation(directory, identifier=published_id, move=True)
            destination = str(directory.resolve())
        catalog_path = resolve_path(machine()['catalog'])
        with catalog_path.with_suffix('.json.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            value = json.loads(catalog_path.read_text())
            entry = value['datasets'][published_id]
            entry['dependencies'] = sorted(set(entry['dependencies'] + manifest['config']['dependencies']))
            entry['metadata'] = dict(materials=['Ta'], potential_ids=['ta-zhong2014-eam'],
                role='raw_dynamics', classification='research',
                title=f'Ta shooting parent {record["parent"]["name"]}: {len(completed)}/{len(record["branches"])} shots complete',
                ancestry=manifest['config']['ancestry_group'], parent=record['parent'],
                evidence=[str(launch / 'manifest.json'), destination + '/status.json'],
                limitations=['Position-conditioned shots share an archived parent; original generating potential is unknown.'])
            write_json(catalog_path, value)
        rows.append(dict(run_id=record['run_id'], parent=record['parent']['name'],
                         state=status['state'], completed_shots=len(completed), directory=destination))
    result = dict(state='complete' if all(r['state'] == 'complete' for r in rows) else 'failed',
                  completed_shots=sum(r['completed_shots'] for r in rows),
                  expected_shots=manifest['shot_count'], parents=rows, collected_at=now())
    write_json(launch / 'status.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--config', type=Path, required=True)
    for action in ('submit', 'worker', 'collect'):
        p = sub.add_parser(action); p.add_argument('--launch', type=Path, required=True)
        if action == 'worker':
            p.add_argument('--index', type=int, required=True)
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        result = prepare(args.config)
    elif args.action == 'submit':
        result = submit(args.launch)
    elif args.action == 'worker':
        worker(args.launch, args.index)
        return
    else:
        result = collect(args.launch)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
