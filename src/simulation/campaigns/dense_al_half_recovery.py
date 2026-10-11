"""Recover stopped halfway Al runs from native restarts and verified binary prefixes."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import numpy as np

from src.data.conversion.shooting_text import load_lammps_shooting_frames_for_conversion
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.project_runtime.paths import REPO, load_json, storage_path
from src.project_runtime.transfer import write_json
from src.simulation.runtime import lammps_environment
from .birth_sources import execute, now, register_entry, sha256
from .dense_al_continuation import recovery_input as fixed_recovery_input, restart_info
from .dense_al_half_stop import source_input, PROTOCOLS


def check_checkpoint(directory):
    directory = Path(directory)
    recovery = json.loads((directory/'input_metadata.json').read_text())['binary_prefix_recovery']
    prefix = ShootingBinaryTrajectory.load(recovery['binary_path'])
    if sha256(prefix.root/'manifest.json') != recovery['manifest_sha256']:
        raise ValueError('Retained prefix manifest changed')
    step = recovery['restart_step']
    indices = np.flatnonzero(prefix.timesteps == step)
    if len(indices) != 1:
        raise ValueError('Retained prefix must contain exactly one checkpoint frame')
    index = int(indices[0])
    snapshot = directory/'technical/recovery/checkpoint.lammpstrj'
    frame = load_lammps_shooting_frames_for_conversion(snapshot,timesteps=[step],atom_count=70304)[step]
    for name in ['atom_ids','atom_types','box_low','box_high']:
        expected = getattr(prefix,name)[index] if name.startswith('box_') else getattr(prefix,name)
        if not np.array_equal(getattr(frame,name),expected):
            raise ValueError(f'Native restart {name} differs from retained observation')
    for name in ['positions','velocities']:
        if not np.array_equal(getattr(frame,name).astype(np.float16),getattr(prefix,name)[index]):
            raise ValueError(f'Native restart encoded {name} differs from verified prefix at {step}')
    report = dict(state='passed',step=step,time_ps=step*.002,atoms=70304,
        check='Exact IDs/types/float32 boxes and exact float16 rounding of native positions/velocities',
        snapshot_sha256=sha256(snapshot),prefix_manifest_sha256=recovery['manifest_sha256'],checked_at=now())
    write_json(directory/'technical/recovery/checkpoint_check.json',report)
    (directory/'technical/recovery/checkpoint-ok.lammps').write_text('print "DENSE_RECOVERY_CHECKPOINT_VERIFIED"\n')
    return report


def recovery_input(config, record, directory):
    interval = record['sample_interval_steps']
    prefix = fixed_recovery_input(dict(config,sample_interval_steps=interval),record).split('run 0\n',1)[0]
    prefix = prefix.replace('trajectory.resume.lammpstrj','trajectory.lammpstrj')
    callback = source_input(config,record,directory).split('run 0\n',1)[1]
    checker = shlex.join([sys.executable,'-m','src.simulation.campaigns.dense_al_half_recovery',
                         'check','--directory',str(directory)])
    return prefix+'run 0\n'+'''write_dump all custom technical/recovery/checkpoint.lammpstrj id type x y z vx vy vz modify sort id format line "%d %d %.17g %.17g %.17g %.17g %.17g %.17g"
shell rm -f technical/recovery/checkpoint-ok.lammps
'''+f'shell {checker}\ninclude technical/recovery/checkpoint-ok.lammps\n'+callback


def recover_dynamics(directory, manifest, record):
    prefix = ShootingBinaryTrajectory.load(record['binary_prefix_recovery']['binary_path'])
    prefix.verify_checksums()
    elapsed = execute(directory,'recovery.in.lammps','source.stdout.log',manifest['config']['mpi_ranks'])
    output = (directory/'source.stdout.log').read_text()
    if 'fix style: npt, fix ID: ensemble' not in output or 'Unused restart file global fix' in output:
        raise ValueError('Recovery did not restore the saved NPT extended state')
    if 'DENSE_RECOVERY_CHECKPOINT_VERIFIED' not in output:
        raise ValueError('Production native checkpoint verification did not pass')
    check_checkpoint(directory)
    return elapsed


def prepare(config):
    from .dense_al import manifest_at, publish_record
    previous = Path(config['source_launch'])
    original = manifest_at(previous)
    if original['protocol'] != 'al_dense_half_stop_queue_v1' or len(original['runs']) != 152:
        raise ValueError('Require the existing 150-main plus two-addition halfway queue')
    jobs = subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-o','%i %j'],text=True)
    if 'al-dense' in jobs:
        raise ValueError(f'Existing dense Al job must finish before recovery: {jobs}')
    environment = json.loads((previous/'environment.json').read_text())
    if sha256(Path(sys.prefix)/'bin/lmp') != environment['lammps_sha256']:
        raise ValueError('Recovery requires the original native LAMMPS executable')
    launch = storage_path('archive')/'simulation-launches'/config['launch_id']
    root = Path(config['execution_root'])
    if root.exists() or launch.exists():
        raise FileExistsError('Preserve existing recovery preparation')
    root.mkdir(parents=True);launch.mkdir(parents=True)
    with (root/'write-probe.bin').open('xb') as f:
        f.write(b'\0'*(16*1024*1024));f.flush();os.fsync(f.fileno())
    (root/'write-probe.bin').unlink()
    manifest = deepcopy(original)
    manifest.update(root=str(root),launch_root=str(launch),created_at=now(),
        execution_recovery_of=dict(launch=str(previous),manifest_sha256=sha256(previous/'manifest.json')),
        storage_exception='STORE execution workspace after SCRATCH quota failure; verified publication keeps canonical source IDs')
    complete=0;recoveries=[];unstarted=0
    for record in manifest['runs']:
        old = Path(original['root'])/record['run_dir']
        new = root/'runs'/record['run_id'];new.parent.mkdir(exist_ok=True)
        status = json.loads((old/'status.json').read_text()) if (old/'status.json').exists() else None
        record['run_dir']='runs/'+record['run_id']
        if status and status['state']=='complete':
            if not old.is_symlink():raise ValueError('Completed source was not canonically published')
            new.symlink_to(old.resolve());complete+=1;continue
        if record.get('protocol') not in PROTOCOLS.values():
            raise ValueError(f'Unexpected unfinished protocol: {record["run_id"]}')
        for name,digest in record['input_sha256'].items():
            if sha256(old/name)!=digest:raise ValueError(f'Original prepared input changed: {old/name}')
        new.mkdir()
        for name in ['prepared_liquid.lammps.data','melt_final.restart.bin','parent_source.in.lammps',
                     'potential/Lee2003_Al.library.meam','potential/Lee2003_Al.meam','technical/launch_config.json']:
            target=new/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/name,target)
        metadata = json.loads((old/'input_metadata.json').read_text())
        record['failure_id']=record['run_id']+'-failed-'+config['launch_id']
        record.pop('recovery',None)
        if status:
            archive=Path(config['binary_archive_root'])/old.relative_to(config['old_scratch_root'])/'trajectory.lammpstrj.float16'
            prefix=ShootingBinaryTrajectory.load(archive);prefix.verify_checksums()
            history=json.loads((old/'technical/stop-monitor/history.json').read_text())
            candidates=[]
            for path in old.glob('source.restart.*.bin'):
                step,info=restart_info(path)
                if step<=int(prefix.timesteps[-1]) and any(h['step']==step for h in history):candidates.append((step,path,info))
            if not candidates:raise ValueError(f'No native checkpoint covered by verified observations and monitor: {old}')
            step,restart,info=max(candidates,key=lambda x:x[0])
            recovery=dict(binary_path=str(archive),manifest_sha256=sha256(archive/'manifest.json'),
                original_conversion_sha256=sha256(archive/'conversion.json'),restart_step=step,
                restart_time_ps=step*.002,native_restart_sha256=sha256(restart),
                previous_attempt=str(old),retained_prefix_frames=step//record['sample_interval_steps'],
                policy='Retain old encoded observations strictly before checkpoint; native checkpoint and tail are re-exported; native integration never uses quantized observations')
            record['binary_prefix_recovery']=recovery
            shutil.copy2(restart,new/'recovery.restart.bin')
            folder=new/'technical/stop-monitor';folder.mkdir(parents=True)
            # Re-evaluate the checkpoint once, preserving all prior physical observations.
            write_json(folder/'history.json',[h for h in history if h['step']<step])
            (new/'technical/recovery').mkdir()
            (new/'technical/recovery/restart_info.txt').write_text(info)
            write_json(new/'technical/recovery/previous_status.json',status)
            (new/'recovery.in.lammps').write_text(recovery_input(manifest['config'],record,new))
            metadata.update(binary_prefix_recovery=recovery)
            write_json(new/'input_metadata.json',dict(metadata,run_dir=record['run_dir'],state='prepared'))
            inspection=(new/'recovery.in.lammps').read_text().split('shell rm -f technical/recovery/checkpoint-ok.lammps',1)[0]
            (new/'technical/recovery/inspect.in.lammps').write_text(inspection)
            command=[str(Path(sys.prefix)/'bin/lmp'),'-in','technical/recovery/inspect.in.lammps','-log','none']
            result=subprocess.run(command,cwd=new,env=lammps_environment(hide_gpus=True),capture_output=True,text=True,check=True)
            (new/'technical/recovery/inspect.stdout.log').write_text(result.stdout)
            if 'fix style: npt, fix ID: ensemble' not in result.stdout or 'Unused restart file global fix' in result.stdout:
                raise ValueError('Preparation inspection failed to restore NPT state')
            checkpoint=check_checkpoint(new)
            # Inspection writes a run-0 tail only; preserve its checkpoint report,
            # then let production create its own output without reuse.
            (new/'trajectory.lammpstrj').unlink()
            (new/'measurement.resume.lammps.log').rename(new/'technical/recovery/inspect.measurement.log')
            failure_record=deepcopy(record);failure_record['failure_id']=record['run_id']+'-stopped-20261003'
            write_json(old/'status.json',dict(state='failed',previous_status=status,finished_at=now(),
                reason='SCRATCH quota stopped native dynamics on October 2; recovery from retained native restart',
                binary_prefix_recovery=recovery))
            publish_record(old,failure_record,failed=True)
            recoveries.append(dict(run_id=record['run_id'],checkpoint_check=checkpoint,**recovery))
            print('Recovered preparation',record['run_id'],step*.002,'ps',flush=True)
        else:
            unstarted+=1
            (new/'technical/stop-monitor').mkdir(parents=True)
            (new/'source.in.lammps').write_text(source_input(manifest['config'],record,new))
            write_json(new/'input_metadata.json',dict(metadata,run_dir=record['run_dir'],state='prepared'))
        record['input_sha256']={str(p.relative_to(new)):sha256(p) for p in new.rglob('*') if p.is_file()}
    write_json(launch/'manifest.json',manifest)
    (launch/'manifest.sha256').write_text(sha256(launch/'manifest.json')+'\n')
    receipt=dict(state='prepared',completed=complete,recoveries=recoveries,unstarted=unstarted,total=152,
        source_launch=str(previous),execution_root=str(root),prepared_at=now())
    write_json(launch/'preparation.json',receipt);write_json(launch/'status.json',receipt)
    write_json(previous/'recovery-successor.json',dict(launch=str(launch),prepared_at=now()))
    register_entry(config['launch_id'],dict(root='archive',path=str(root.relative_to(storage_path('archive'))),kind='simulation',
        dependencies=['al-main-half-stop-20261001'],metadata=dict(materials=['Al'],potential_ids=['al-lee2003-meam'],
        role='raw_dynamics',classification='research',title='Al exact 0.1/0.01 ps halfway rerun recovery',
        description='Native recovery of four stopped runs plus124 unstarted descendants;24 completed sources referenced unchanged. Same150 melt ancestors.',
        evidence=[str(launch/'manifest.json'),str(launch/'preparation.json')],storage_exception=manifest['storage_exception'])))
    return launch


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','check'])
    parser.add_argument('--config');parser.add_argument('--directory',type=Path)
    args=parser.parse_args(argv)
    if args.action=='prepare':print(prepare(load_json(args.config)))
    else:print(json.dumps(check_checkpoint(args.directory)))


if __name__=='__main__':main()
