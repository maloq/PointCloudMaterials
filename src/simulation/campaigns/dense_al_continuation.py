"""Resume inspected dense Al failures without changing their physical protocol."""
from copy import deepcopy
import hashlib
import io
import json
import mmap
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from src.project_runtime.paths import storage_path
from src.project_runtime.transfer import write_json
from src.simulation.runtime import lammps_environment
from .birth_sources import execute, now, sha256


def restart_info(path):
    output = subprocess.check_output([str(Path(sys.prefix)/'bin/lmp'), '-restart2info',
                                     str(path), '-log', 'none'], text=True,
                                    env=lammps_environment(hide_gpus=True))
    step, = re.findall(r'Current timestep number = (\d+)', output)
    dt, = re.findall(r'Current timestep size = ([^\n]+)', output)
    atoms, = re.findall(r'Atoms\s*=\s*(\d+),', output)
    if float(dt) != .002 or int(atoms) != 70304:
        raise ValueError(f'Wrong native restart protocol: {path}\n{output}')
    return int(step), output


def checkpoint_frame(path, step):
    """Locate the last exact checkpoint frame; do not parse the truncated tail."""
    with Path(path).open('rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        start = data.rfind(f'ITEM: TIMESTEP\n{step}\n'.encode())
        if start < 0:
            raise ValueError(f'No saved checkpoint frame {step} in {path}')
        end = data.find(b'ITEM: TIMESTEP\n', start+1)
        if end < 0:
            end = len(data)
        frame = bytes(data[start:end])
    return frame, end


def check_checkpoint(old_frame, new_frame, step):
    import numpy as np
    def parse(frame):
        header = frame.split(b'\n', 9)[:9]
        if (header[0] != b'ITEM: TIMESTEP' or int(header[1]) != step or
                header[2] != b'ITEM: NUMBER OF ATOMS' or int(header[3]) != 70304 or
                header[4] != b'ITEM: BOX BOUNDS pp pp pp' or
                header[8] != b'ITEM: ATOMS id type x y z vx vy vz'):
            raise ValueError(f'Unexpected checkpoint header at {step}: {header}')
        boxes = np.array([[float(x) for x in line.split()] for line in header[5:8]])
        atoms = np.loadtxt(io.BytesIO(frame), skiprows=9)
        if atoms.shape != (70304, 8) or not np.array_equal(atoms[:,0], np.arange(1,70305)):
            raise ValueError(f'Checkpoint atom identity/shape changed: {atoms.shape}')
        return boxes, atoms
    boxes, old = parse(old_frame); new_boxes, new = parse(new_frame)
    if not np.array_equal(old[:,:2], new[:,:2]):
        raise ValueError('Checkpoint atom IDs/types changed')
    box_error = float(np.max(np.abs(new_boxes-boxes)))
    lengths = boxes[:,1]-boxes[:,0]
    displacement = new[:,2:5]-old[:,2:5]
    displacement -= np.rint(displacement/lengths)*lengths
    position_error = float(np.max(np.abs(displacement)))
    velocity_error = float(np.max(np.abs(new[:,5:]-old[:,5:])))
    if box_error > 1e-9 or position_error > 1e-8 or velocity_error > 1e-10:
        raise ValueError(f'Native checkpoint differs from saved frame: {box_error}, {position_error}, {velocity_error}')
    return dict(step=step, atom_count=70304, box_max_abs_A=box_error,
                position_max_abs_minimum_image_A=position_error,
                velocity_max_abs_A_per_ps=velocity_error)


def recovery_input(config, record):
    # The same fix IDs restore the saved Nose-Hoover extended state. No new
    # velocities, equilibration or timestep reset is allowed here.
    t = record['temperature_K']
    return f'''log measurement.resume.lammps.log
read_restart recovery.restart.bin
mass 1 26.9815
pair_style meam
pair_coeff * * potential/Lee2003_Al.library.meam Al potential/Lee2003_Al.meam Al
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
timestep {config['timestep_ps']}
fix remove_drift all momentum {config['momentum_interval_steps']} linear 1 1 1
fix ensemble all npt temp {t:g} {t:g} 0.3 iso 0 0 3
thermo {config['sample_interval_steps']}
thermo_style custom step temp press vol pe
thermo_modify format float %.16g flush yes lost error
dump trajectory all custom {config['sample_interval_steps']} trajectory.resume.lammpstrj id type x y z vx vy vz
dump_modify trajectory first yes sort id format line "%d %d %.17g %.17g %.17g %.17g %.17g %.17g"
restart {config['restart_interval_steps']} source.restart.1.bin source.restart.2.bin
run 0
run {config['measurement_steps']} upto
restart 0
undump trajectory
write_restart final.restart.bin
print "DENSE_AL_COMPLETE {record['run_id']}"
'''


def prepare_continuation(config):
    from .dense_al import manifest_at
    source_launch = Path(config['source_launch'])
    original = manifest_at(source_launch)
    environment = json.loads((source_launch/'environment.json').read_text())
    if (str(Path(sys.prefix)/'bin/lmp') != environment['lammps'] or
            sha256(environment['lammps']) != environment['lammps_sha256']):
        raise ValueError('Recovery requires the original pointnet-torch214 LAMMPS executable')
    if json.loads((source_launch/'status.json').read_text())['state'] != 'failed':
        raise ValueError('Inspect campaign state before preparing continuation')
    if subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%j'], text=True).count('al-dense'):
        raise ValueError('Another dense Al worker/controller is already scheduled')
    if not (1 <= config['sources_per_worker_wave'] <= 2 and 1 <= config['workers'] <= 4):
        raise ValueError('Continuation requires at most two sources per lane and four lanes')
    identifier = config['continuation_id']
    if Path(identifier).name != identifier or identifier in {'.','..'}:
        raise ValueError('Continuation ID must be a single directory name')
    launch = storage_path('archive')/'simulation-launches'/identifier
    launch.mkdir()
    write_json(launch/'status.json', dict(state='preparing', source_launch=str(source_launch), created_at=now()))
    manifest = deepcopy(original)
    manifest.update(created_at=now(), launch_root=str(launch),
                    continuation_of=dict(launch=str(source_launch), manifest_sha256=sha256(source_launch/'manifest.json')))
    for name in ('sources_per_worker_wave','workers','walltime','qos'):
        manifest['config'][name] = config[name]
    root = Path(manifest['root'])
    changes = []
    complete = 0
    for record in manifest['runs']:
        directory = root/record['run_dir']
        record['failure_id'] = record['run_id']+'-failed-'+identifier
        status_path = directory/'status.json'
        if not status_path.exists():
            continue
        status = json.loads(status_path.read_text())
        if status['state'] == 'complete' and directory.is_symlink():
            if json.loads((directory/'paired_conversion.json').read_text())['frame_count'] != 6001:
                raise ValueError(f'Wrong completed-source frame count: {directory}')
            complete += 1
            continue
        if status['state'] != 'failed' or status['phase'] != 'dynamics':
            raise ValueError(f'Uninspected partial source: {directory}: {status}')
        archived = storage_path('archive')/'simulations'/(record['run_id']+'-failed')
        receipt = json.loads(archived.with_name(archived.name+'.publication.json').read_text())
        if receipt['state'] != 'complete' or Path(receipt['destination']) != archived:
            raise ValueError(f'Failure archive not verified: {archived}')
        for name, digest in record['input_sha256'].items():
            if sha256(directory/name) != digest or sha256(archived/name) != digest:
                raise ValueError(f'Prepared or archived input changed: {name}')
        restarts = []
        for path in sorted(directory.glob('source.restart.*.bin')):
            if sha256(path) != receipt['files'][path.name]['sha256']:
                raise ValueError(f'Native restart differs from failure archive: {path}')
            step, info = restart_info(path)
            restarts.append((step, path, info))
        step, restart, info = max(restarts, key=lambda item:item[0])
        if not (0 < step < 300000 and step % 7500 == 0):
            raise ValueError(f'Unexpected measurement restart step: {restart}: {step}')
        old_frame, prefix_bytes = checkpoint_frame(directory/'trajectory.lammpstrj', step)
        raw_receipt = receipt['files']['trajectory.lammpstrj']
        if (directory/'trajectory.lammpstrj').stat().st_size != raw_receipt['bytes']:
            raise ValueError(f'Partial raw dump size changed: {directory}')
        stage = root/'continuation-staging'/identifier/record['run_id']
        stage.mkdir(parents=True)
        for name in record['input_sha256']:
            (stage/name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(directory/name, stage/name)
        if sha256(directory/'equilibration.lammps.log') != receipt['files']['equilibration.lammps.log']['sha256']:
            raise ValueError(f'Equilibration log differs from archived attempt: {directory}')
        shutil.copy2(directory/'equilibration.lammps.log', stage/'equilibration.lammps.log')
        shutil.copy2(restart, stage/'recovery.restart.bin')
        (stage/'recovery.in.lammps').write_text(recovery_input(manifest['config'],record))
        (stage/'technical/restart_info.txt').write_text(info)
        inspect = recovery_input(manifest['config'],record).split('restart 7500')[0]
        inspect = inspect.replace('log measurement.resume.lammps.log','log none').replace(
            'trajectory.resume.lammpstrj','technical/checkpoint.lammpstrj')+'run 0\nundump trajectory\n'
        (stage/'technical/inspect_restart.in.lammps').write_text(inspect)
        with (stage/'technical/inspect_restart.stdout.log').open('x') as handle:
            subprocess.run([environment['lammps'],'-log','none','-in','technical/inspect_restart.in.lammps'],
                           cwd=stage, env=lammps_environment(hide_gpus=True), stdout=handle,
                           stderr=subprocess.STDOUT, check=True)
        output = (stage/'technical/inspect_restart.stdout.log').read_text()
        if 'fix style: npt, fix ID: ensemble' not in output or 'Unused restart file global fix' in output:
            raise ValueError(f'NPT extended state was not restored: {stage}')
        checkpoint = check_checkpoint(old_frame, (stage/'technical/checkpoint.lammpstrj').read_bytes(), step)
        previous = root/'attempts'/identifier/record['run_id']
        if previous.exists():
            raise FileExistsError(previous)
        record['recovery'] = dict(restart_step=step, restart_time_ps=step*.002,
            original_restart=restart.name, previous_attempt=str(previous), failure_archive=str(archived),
            failure_publication_receipt=str(archived.with_name(archived.name+'.publication.json')),
            partial_dump_sha256=raw_receipt['sha256'], partial_dump_bytes=raw_receipt['bytes'],
            retained_prefix_bytes=prefix_bytes, retained_frame_count=step//50+1,
            checkpoint_frame_sha256=hashlib.sha256(old_frame).hexdigest(), checkpoint_check=checkpoint)
        for name in ('recovery.restart.bin','recovery.in.lammps'):
            record['input_sha256'][name] = sha256(stage/name)
        write_json(stage/'technical/recovery.json', record['recovery'])
        changes.append((directory, previous, stage))
        print(f'Validated {record["run_id"]}: restart {step*.002:g} ps', flush=True)
    write_json(launch/'manifest.json', manifest)
    (launch/'manifest.sha256').write_text(sha256(launch/'manifest.json')+'\n')
    for directory, previous, stage in changes:
        previous.parent.mkdir(parents=True, exist_ok=True)
        directory.rename(previous)
        stage.rename(directory)
    receipt = dict(state='prepared', continuation_launch=str(launch), original_launch=str(source_launch),
        original_manifest_sha256=manifest['continuation_of']['manifest_sha256'], completed=complete,
        recovered=len(changes), never_started=150-complete-len(changes), total=150, prepared_at=now(),
        recoveries=[dict(run_id=r['run_id'], **r['recovery']) for r in manifest['runs'] if 'recovery' in r])
    write_json(launch/'preparation.json', receipt)
    write_json(launch/'status.json', receipt)
    write_json(source_launch/'continuation.json', receipt)
    write_json(root/'continuation.json', receipt)
    return launch


def recover_dynamics(directory, manifest, record):
    recovery = record['recovery']; step = recovery['restart_step']
    elapsed = execute(directory,'recovery.in.lammps','source.stdout.log',manifest['config']['mpi_ranks'])
    output = (directory/'source.stdout.log').read_text()
    if 'fix style: npt, fix ID: ensemble' not in output or 'Unused restart file global fix' in output:
        raise ValueError('Production continuation did not restore the NPT extended state')
    previous = Path(recovery['previous_attempt'])
    old_frame, prefix_end = checkpoint_frame(previous/'trajectory.lammpstrj',step)
    if prefix_end != recovery['retained_prefix_bytes'] or hashlib.sha256(old_frame).hexdigest() != recovery['checkpoint_frame_sha256']:
        raise ValueError('Retained checkpoint boundary changed')
    suffix = directory/'trajectory.resume.lammpstrj'
    new_frame, suffix_start = checkpoint_frame(suffix, step)
    check = check_checkpoint(old_frame, new_frame, step)
    partial_hash = hashlib.sha256(); retained_hash = hashlib.sha256(); suffix_hash = hashlib.sha256()
    building = directory/'trajectory.lammpstrj.building'
    with building.open('xb') as joined:
        with (previous/'trajectory.lammpstrj').open('rb') as source:
            offset = 0
            while block := source.read(16*1024*1024):
                partial_hash.update(block)
                keep = block[:max(0,min(len(block),prefix_end-offset))]
                joined.write(keep); retained_hash.update(keep)
                offset += len(block)
        if offset != recovery['partial_dump_bytes'] or partial_hash.hexdigest() != recovery['partial_dump_sha256']:
            raise ValueError('Partial dump no longer matches its verified failure archive')
        with suffix.open('rb') as source:
            offset = 0
            while block := source.read(16*1024*1024):
                suffix_hash.update(block)
                skip = max(0,min(len(block),suffix_start-offset))
                joined.write(block[skip:]); offset += len(block)
    building.rename(directory/'trajectory.lammpstrj')
    # Keep the original thermodynamic prefix, excluding observations superseded
    # by the restart, then the continuation log. Original run-0 duplicates remain.
    with (directory/'measurement.lammps.log').open('x') as joined:
        for line in (previous/'measurement.lammps.log').read_text().splitlines(keepends=True):
            fields = line.split()
            if len(fields) == 5 and fields[0].isdigit() and int(fields[0]) >= step:
                break
            joined.write(line)
        joined.write((directory/'measurement.resume.lammps.log').read_text())
    write_json(directory/'technical/recovery_join.json', dict(checkpoint_check=check,
        old_partial_sha256=partial_hash.hexdigest(), retained_prefix_sha256=retained_hash.hexdigest(),
        continuation_sha256=suffix_hash.hexdigest(), continuation_bytes=suffix.stat().st_size,
        retained_frame_count=step//50+1, expected_final_frames=6001,
        discarded_duplicate_checkpoint=True, joined_at=now()))
    return elapsed


def finish_recovery(directory):
    # Called only after the maintained converter has verified the joined export.
    audit = json.loads((directory/'technical/recovery_join.json').read_text())
    suffix = directory/'trajectory.resume.lammpstrj'
    if suffix.stat().st_size != audit['continuation_bytes']:
        raise ValueError('Continuation dump changed during conversion')
    suffix.unlink()
    audit.update(continuation_ascii_deleted_after_verified_conversion=True, deleted_at=now())
    write_json(directory/'technical/recovery_join.json',audit)
