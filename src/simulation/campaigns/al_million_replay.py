"""Recover dense Al velocities by checkpoint-anchored, audited native replay.

The historical trajectory is immutable. Every four-ps replay must match its
native endpoint and every saved position frame before it can enter the export.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

from src.project_runtime.paths import REPO, dataset_path, machine, storage_path
from src.project_runtime.transfer import verified_copy, write_json
from .birth_sources import freeze_code, now, register_entry, sha256
from .dense_al import manifest_at, sbatch, verify_code


def input_text(manifest, index):
    parent = Path(manifest['parent'])
    start = index * 4000
    restart = parent / ('melt/liquid.restart.bin' if start == 0 else f'source/checkpoint.{start}.restart.bin')
    potential = Path(manifest['launch_root']) / 'potential'
    initialize = 'reset_timestep 0\n' if start == 0 else ''
    scale = 'velocity all scale 450\n' if start == 0 else ''
    return f'''units metal
atom_style atomic
processors 3 4 8
read_restart {restart}
{initialize}pair_style meam
pair_coeff * * {potential}/Lee2003_Al.library.meam Al {potential}/Lee2003_Al.meam Al
mass 1 26.9815
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
timestep 0.001
thermo 100
thermo_style custom step time atoms temp press lx ly lz pe ke etotal density
thermo_modify format float %.16g flush yes lost error
{scale}fix ensemble all npt temp 450 450 0.1 iso 0 0 1.0
dump trajectory all custom 100 trajectory.lammpstrj id type x y z vx vy vz
dump_modify trajectory sort id format float %.17g first yes
run 4000
write_restart endpoint.restart.bin
print "REPLAY_SEGMENT_COMPLETE"
'''


def prepare(config):
    identifier = config['campaign_id']
    launch = storage_path('archive') / 'simulation-launches' / identifier
    root = storage_path('simulation_runs') / identifier
    archive = storage_path('archive') / 'simulations' / identifier
    for path in (launch, root, archive):
        if path.exists():
            raise FileExistsError(f'Preserve existing replay: {path}')
    parent = dataset_path(config['parent_dataset']).resolve()
    old = json.loads((parent / 'config.json').read_text())
    expected = dict(atom_count=1000000, timestep_ps=.001, temperature_K=450,
                    thermostat_ps=.1, barostat_ps=1., pressure_bar=0,
                    dump_every_steps=100, assessment_steps=4000, max_source_steps=400000,
                    mass_g_mol=26.9815)
    for key, value in expected.items():
        if old[key] != value:
            raise ValueError(f'Original protocol changed: {key}={old[key]} != {value}')
    if config['mpi_ranks'] != 96 or config['processor_grid'] != [3, 4, 8]:
        raise ValueError('Historical replay requires 96 ranks and the original 3x4x8 grid')
    geometry = parent / 'source/trajectory_binary_float16'
    if not np.array_equal(np.load(geometry / 'timesteps.npy'), np.arange(0, 400001, 100)):
        raise ValueError('Original production timeline is not exact 0.10 ps over 400 ps')
    q = np.load(geometry / 'positions.npy', mmap_mode='r')
    if q.shape != (4001, 1000000, 3) or q.dtype != np.float16:
        raise ValueError(f'Unexpected historical position contract: {q.shape}, {q.dtype}')
    if (not np.array_equal(np.load(geometry/'atom_ids.npy'), np.arange(1, 1000001)) or
        not np.all(np.load(geometry/'atom_types.npy') == 1)):
        raise ValueError('Historical trajectory identities differ from pure Al IDs 1..N')
    paths = [parent/'melt/liquid.restart.bin', *[parent/f'source/checkpoint.{s}.restart.bin'
             for s in range(4000, 400001, 4000)], parent/'config.json', parent/'source/in.lammps',
             geometry/'manifest.json']
    hashes = {str(p): sha256(p) for p in paths}
    log = (parent/'source/log.lammps').read_text()
    if '3 by 4 by 8 MPI processor grid' not in log:
        raise ValueError('Original processor grid could not be established')
    timings = [float(v) for v in re.findall(r'Loop time of ([\d.]+) on 96 procs for 4000 steps', log)]
    execution = machine()['execution']
    # Freeze the native executable and shared-library identity, not just its Python launcher.
    executable = Path(execution['lammps'])
    lammps_dir, = (executable.parent.parent/'lib/python3.12/site-packages').glob('lammps')
    engines = [executable, lammps_dir/'lmp', lammps_dir/'liblammps.so']
    launch.mkdir(parents=True); root.mkdir(parents=True); archive.mkdir(parents=True)
    (launch/'potential').mkdir()
    for item in old['potential_files']:
        p = Path(item['path'])
        if sha256(p) != item['sha256']:
            raise ValueError(f'Changed original potential: {p}')
        shutil.copy2(p, launch/'potential'/p.name)
        hashes[str(launch/'potential'/p.name)] = item['sha256']
    hashes.update({str(p): sha256(p) for p in engines})
    manifest = dict(config=config, parent=str(parent), root=str(root), archive=str(archive),
        launch_root=str(launch), created_at=now(), input_sha256=hashes,
        execution=execution, python=sys.executable, segment_count=100,
        original_median_seconds_per_4ps=float(np.median(timings)),
        protocol='checkpoint_anchored_native_al_velocity_replay_v1',
        material='Al', potential_id='al-lee2003-meam', timestep_ps=.001, observation_ps=.1,
        integration_and_force_precision='native CPU LAMMPS float64',
        ancestry=dict(parent_dataset=config['parent_dataset'], independent_ancestors_added=0,
                      role='train; same million-atom parent used in structural training'),
        interpretation='Approximate historical replay only after every declared gate passes; not bitwise recovery',
        exclusions='300 ps melt history deferred; no new melt, no random velocities, no GPU force substitution')
    (launch/'inputs').mkdir()
    for i in range(100):
        path = launch/'inputs'/f'{i:03d}.lammps'
        path.write_text(input_text(manifest, i)); hashes[str(path)] = sha256(path)
    write_json(launch/'manifest.json', manifest)
    (launch/'manifest.sha256').write_text(sha256(launch/'manifest.json')+'\n')
    shutil.copy2(launch/'manifest.json', archive/'replay_manifest.json')
    write_json(archive/'status.json', dict(state='prepared', accepted_segments=0, expected_segments=100))
    register_entry(identifier, dict(root='archive', path='simulations/'+identifier, kind='simulation',
        state='prepared', dependencies=[config['parent_dataset'], 'potentials'], aliases=[],
        metadata=dict(title='Al million-atom 400 ps velocity recovery', materials=['Al'],
            potential_ids=['al-lee2003-meam'], role='raw_dynamics', classification='building',
            lineage=manifest['ancestry'], observation_ps=.1,
            description='Checkpoint-anchored replay; consumers must require complete status and fidelity gates.',
            evidence=['${dataset:'+identifier+'}/status.json', '${dataset:'+identifier+'}/replay_manifest.json'])))
    return launch


def checkpoint(path):
    from lammps import lammps
    native = lammps(cmdargs=['-screen', 'none', '-log', 'none'])
    try:
        native.command(f'read_restart {path}')
        n = int(native.get_natoms())
        ids = native.numpy.extract_atom('id')[:n].copy()
        order = np.argsort(ids)
        if n != 1000000 or not np.array_equal(ids[order], np.arange(1, n+1)):
            raise ValueError(f'Unexpected native checkpoint identities: {path}')
        low, high, xy, yz, xz, periodic, _ = native.extract_box()
        if any([xy, yz, xz]) or not all(periodic):
            raise ValueError('Replay requires the original orthogonal periodic cell')
        q = native.numpy.extract_atom('x')[:n][order].copy()
        v = native.numpy.extract_atom('v')[:n][order].copy()
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            raise ValueError(f'Nonfinite native checkpoint: {path}')
        return dict(q=q, v=v, low=np.array(low), high=np.array(high),
                    step=int(native.extract_global('ntimestep')))
    finally:
        native.close()


def compare_endpoint(reference_path, replay_path, config):
    reference, replay = checkpoint(reference_path), checkpoint(replay_path)
    if reference['step'] != replay['step']:
        raise ValueError('Reference/replayed native checkpoint timesteps differ')
    box = reference['high']-reference['low']
    dq = replay['q']-reference['q']; dq -= box*np.rint(dq/box)
    dv = replay['v']-reference['v']
    result = dict(step=reference['step'], position_max_A=float(abs(dq).max()),
        position_rms_A=float(np.sqrt(np.mean(dq**2))),
        velocity_max_A_per_ps=float(abs(dv).max()), velocity_rms_A_per_ps=float(np.sqrt(np.mean(dv**2))),
        box_max_A=float(max(abs(reference[k]-replay[k]).max() for k in ['low', 'high'])))
    result['passed'] = bool(result['position_max_A'] <= config['endpoint_position_max_A'] and
        result['velocity_max_A_per_ps'] <= config['endpoint_velocity_max_A_per_ps'] and
        result['box_max_A'] <= config['endpoint_box_max_A'])
    return result


def compare_frames(manifest, dump, index):
    """Check every replayed coordinate against its historical float16 bin.

    The extra allowance covers original text/float32 transforms plus tiny replay
    drift. Quantized agreement alone does not prove identical velocities.
    """
    from src.data.conversion.offline_dump import COLUMNS
    geometry = Path(manifest['parent'])/'source/trajectory_binary_float16'
    old_q = np.load(geometry/'positions.npy', mmap_mode='r')
    old_low = np.load(geometry/'box_low.npy'); old_high = np.load(geometry/'box_high.npy')
    rows = []
    with dump.open() as stream:
        for expected_step in range(index*4000, (index+1)*4000+1, 100):
            header = [stream.readline().strip() for _ in range(9)]
            if (header[0] != 'ITEM: TIMESTEP' or int(header[1]) != expected_step or
                header[2:5] != ['ITEM: NUMBER OF ATOMS', '1000000', 'ITEM: BOX BOUNDS pp pp pp'] or
                tuple(header[8].removeprefix('ITEM: ATOMS ').split()) != COLUMNS[1]):
                raise ValueError(f'Replay dump schema/timeline mismatch: {dump}, {header}')
            values = np.fromstring(''.join(stream.readline() for _ in range(1000000)), sep=' ')
            if values.size != 8000000 or not np.isfinite(values).all():
                raise ValueError(f'Incomplete/nonfinite frame {expected_step}')
            values = values.reshape(-1, 8)
            if not np.array_equal(values[:, 0], np.arange(1, 1000001)) or not np.all(values[:, 1] == 1):
                raise ValueError('Replay dump atom IDs/types changed')
            frame = expected_step//100; box = old_high[frame]-old_low[frame]
            bounds = np.array([np.fromstring(line, sep=' ') for line in header[5:8]])
            box_error = float(max(abs(bounds[:, 0]-old_low[frame]).max(),
                                  abs(bounds[:, 1]-old_high[frame]).max()))
            q = np.mod(values[:, 2:5].astype(np.float32)-old_low[frame], box)
            stored = old_q[frame]
            delta = q-stored.astype(np.float32); delta -= box*np.rint(delta/box)
            allowance = .5*np.spacing(stored).astype(np.float32)+manifest['config']['frame_position_extra_tolerance_A']
            excess = np.maximum(abs(delta)-allowance, 0)
            rows.append(dict(step=expected_step, outside_quantization_envelope=int(np.count_nonzero(excess)),
                             max_excess_A=float(excess.max()), box_error_A=box_error,
                             coordinate_rms_A=float(np.sqrt(np.mean(delta**2)))))
        if stream.read(1):
            raise ValueError('Unexpected extra frames in replay')
    return dict(passed=all(r['outside_quantization_envelope'] == 0 and
        r['box_error_A'] <= manifest['config']['frame_position_extra_tolerance_A'] for r in rows), frames=rows)


def archive_segment(manifest, directory, index):
    archive = Path(manifest['archive'])
    verified_copy(directory, archive/'segments'/f'{index:03d}',
                  archive/'technical/publications'/f'{index:03d}.json', move=True)


def run_segment(manifest, index):
    archive, root, launch = map(Path, (manifest['archive'], manifest['root'], manifest['launch_root']))
    directory = root/'segments'/f'{index:03d}'
    if (archive/'segments'/f'{index:03d}'/'status.json').exists():
        previous = json.loads((archive/'segments'/f'{index:03d}'/'status.json').read_text())
        if previous['state'] != 'accepted':
            raise ValueError(f'Previous replay failed: {directory}; inspect before retrying')
        return previous
    directory.mkdir(parents=True)
    shutil.copy2(launch/'inputs'/f'{index:03d}.lammps', directory/'in.lammps')
    started = time.monotonic()
    execution = manifest['execution']
    command = [s.format(ranks=96) for s in execution['mpi_launcher']]+[
               execution['lammps'], '-in', 'in.lammps', '-log', 'log.lammps']
    env = dict(os.environ, **execution['mpi_environment'])
    write_json(directory/'command.json', dict(command=command, environment=execution['mpi_environment'],
               job_id=os.environ['SLURM_JOB_ID'], host=os.uname().nodename, started_at=now()))
    status = dict(state='running', segment=index, start_step=index*4000, end_step=(index+1)*4000)
    write_json(directory/'status.json', status)
    try:
        with (directory/'stdout.log').open('w') as stream:
            process = subprocess.Popen(command, cwd=directory, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = process.wait(timeout=manifest['config']['segment_timeout_seconds'])
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait()
        if returncode:
            raise RuntimeError(f'LAMMPS failed ({returncode}); inspect {directory}/stdout.log')
        log = (directory/'log.lammps').read_text()
        if '3 by 4 by 8 MPI processor grid' not in log or 'REPLAY_SEGMENT_COMPLETE' not in log:
            raise ValueError('Native replay execution did not confirm the original grid and completion')
        if index and 'fix style: npt, fix ID: ensemble' not in log:
            raise ValueError('Native NPT restart state was not restored')
        native_seconds = time.monotonic()-started
        reference = Path(manifest['parent'])/f'source/checkpoint.{(index+1)*4000}.restart.bin'
        endpoint = compare_endpoint(reference, directory/'endpoint.restart.bin', manifest['config'])
        frames = compare_frames(manifest, directory/'trajectory.lammpstrj', index)
        write_json(directory/'fidelity.json', dict(endpoint=endpoint, observations=frames,
            thresholds=manifest['config'], reference_checkpoint_sha256=sha256(reference)))
        subprocess.run([sys.executable, str(launch/'code/scripts/convert_trajectory.py'), 'offline-dump',
            str(directory/'trajectory.lammpstrj'), '--output', str(directory/'trajectory_binary_float16'),
            '--delete-source'], check=True)
        if not endpoint['passed'] or not frames['passed']:
            raise ValueError(f'Replay diverged beyond declared recovery tolerances at segment {index}; '
                             'retained Q/V is a separate continuation and cannot supply historical labels')
        status.update(state='accepted', native_seconds=native_seconds, seconds=time.monotonic()-started,
                      completed_at=now(), endpoint=endpoint)
        write_json(directory/'status.json', status)
    except BaseException as error:
        status.update(state='failed', reason=str(error), finished_at=now())
        write_json(directory/'status.json', status)
        archive_segment(manifest, directory, index)
        raise
    archive_segment(manifest, directory, index)
    return status


def worker(launch, stage):
    manifest = manifest_at(launch); verify_code(launch)
    if int(os.environ['SLURM_NTASKS']) != 96:
        raise ValueError('A 96-rank native CPU allocation is required')
    for filename, digest in manifest['input_sha256'].items():
        if sha256(filename) != digest:
            raise ValueError(f'Replay input or native executable changed: {filename}')
    def interrupted(signum, frame):
        raise InterruptedError(f'Replay worker received signal {signum}')
    for sig in [signal.SIGUSR1, signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, interrupted)
    archive = Path(manifest['archive'])
    if stage == 'production':
        if json.loads((archive/'audit.json').read_text())['state'] != 'passed':
            raise ValueError('Native replay audit has not passed')
    indices = manifest['config']['audit_segments'] if stage == 'audit' else range(100)
    try:
        for index in indices:
            result = run_segment(manifest, index)
            accepted = [json.loads(p.read_text()) for p in (archive/'segments').glob('*/status.json')]
            good = [r for r in accepted if r['state'] == 'accepted']
            write_json(archive/'status.json', dict(state='running', stage=stage, accepted_segments=len(good),
                expected_segments=100, last_segment=index, updated_at=now(),
                estimated_remaining_compute_hours=(100-len(good))*np.mean([r['seconds'] for r in good])/3600))
            print(json.dumps(result), flush=True)
        if stage == 'audit':
            write_json(archive/'audit.json', dict(state='passed', segments=list(indices), finished_at=now()))
        else:
            assemble(manifest)
    except BaseException as error:
        write_json(archive/f'{stage}.json', dict(state='failed', reason=str(error), finished_at=now()))
        write_json(archive/'status.json', dict(state='failed', stage=stage, reason=str(error), updated_at=now()))
        raise


def assemble(manifest):
    from src.data.trajectories.shooting import ShootingBinaryTrajectory, compose_shooting_binary_trajectories
    archive = Path(manifest['archive'])
    sources = []
    for index in range(100):
        directory = archive/'segments'/f'{index:03d}'
        if json.loads((directory/'status.json').read_text())['state'] != 'accepted':
            raise ValueError(f'Cannot publish a failed or missing replay: {directory}')
        sources.append(ShootingBinaryTrajectory.load(directory/'trajectory_binary_float16'))
    # Prefer the exact starting checkpoint at shared 4-ps boundaries. Reverse
    # lookup order accomplishes this while the requested output remains sorted.
    result = compose_shooting_binary_trajectories(list(reversed(sources)), archive/'trajectory_binary_float16',
        timesteps=range(0, 400001, 100), storage_dtype='float16',
        provenance=dict(protocol=manifest['protocol'], ancestry=manifest['ancestry'], timestep_ps=.001,
            observation_ps=.1, manifest=str(archive/'replay_manifest.json'),
            interpretation=manifest['interpretation'], gates='Every native endpoint and every saved coordinate checked',
            boundary_policy='Exact native restart at every 4 ps; final frame uses audited replay endpoint',
            independent_ancestors_added=0, historical_labels='No automatic relabelling or catalogue substitution'))
    result.verify_checksums()
    np.save(archive/'frame_segment_index.npy', np.minimum(np.arange(4001)//40, 99).astype(np.int32))
    write_json(archive/'status.json', dict(state='complete', accepted_segments=100, frame_count=4001,
        observation_ps=.1, finished_at=now(), trajectory='trajectory_binary_float16',
        ancestry=manifest['ancestry'], interpretation=manifest['interpretation']))
    # Only derived chunk arrays are redundant; retain all native endpoints,
    # inputs, fidelity/quantization receipts and publication checksums.
    for source in sources:
        (source.root/'manifest.json').rename(source.root/'consolidated_manifest.json')
        for filename in source.root.glob('*.npy'):
            filename.unlink()
        write_json(source.root/'assembled.json', dict(state='consolidated', output=str(result.root),
            output_manifest_sha256=sha256(result.root/'manifest.json')))
    catalog = Path(machine()['catalog'])
    if not catalog.is_absolute():
        catalog = REPO/catalog
    with catalog.with_suffix('.json.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        entries = json.loads(catalog.read_text())
        entry = entries['datasets'][manifest['config']['campaign_id']]
        entry['state'] = 'complete'
        entry['metadata']['classification'] = 'verified'
        entry['metadata']['description'] = manifest['interpretation']
        write_json(catalog, entries)


def submit(launch):
    manifest = manifest_at(launch); c = manifest['config']
    if (launch/'submission.json').exists():
        raise FileExistsError(f'Replay already submitted: {launch}')
    code = freeze_code(launch)
    receipt = dict(submitted_at=now(), launch=str(launch), archive=manifest['archive'], jobs={})
    for stage, walltime in [('audit', c['audit_walltime']), ('production', c['production_walltime'])]:
        command = [manifest['python'], '-u', '-m', 'src.simulation.campaigns.al_million_replay',
                   'worker', '--launch', str(launch), '--stage', stage]
        env = [f'PCM_PROJECT_ROOT={REPO}', f'PYTHONPATH={code}', 'OMP_NUM_THREADS=1',
               'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'UCX_TLS=tcp,self']
        dependency = [] if stage == 'audit' else [f'#SBATCH --dependency=afterok:{receipt["jobs"]["audit"]}',
                                                '#SBATCH --kill-on-invalid-dep=yes']
        script = '\n'.join(['#!/bin/bash', f'#SBATCH --job-name=al1m-QV-{stage}',
            f'#SBATCH --partition={c["partition"]}', '#SBATCH --nodes=1', '#SBATCH --ntasks=96',
            '#SBATCH --cpus-per-task=1', f'#SBATCH --mem={c["memory"]}', f'#SBATCH --time={walltime}',
            '#SBATCH --signal=B:USR1@600', f'#SBATCH --output={launch}/{stage}-%j.log',
            f'#SBATCH --chdir={code}', *dependency, 'set -euo pipefail',
            'exec env '+shlex.join(env)+' '+shlex.join(command), ''])
        receipt['jobs'][stage] = sbatch(script, launch/f'{stage}.sbatch')
        write_json(launch/'submission.json', receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=['prepare', 'submit', 'worker'])
    parser.add_argument('--config', type=Path); parser.add_argument('--launch', type=Path)
    parser.add_argument('--stage', choices=['audit', 'production'])
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        print(prepare(json.loads(args.config.read_text())))
    elif args.action == 'submit':
        print(json.dumps(submit(args.launch), indent=2))
    else:
        worker(args.launch, args.stage)


if __name__ == '__main__':
    main()
