"""Move receipt-owned Ni work off a session GPU using a native NPT restart."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
import torch
from lammps import lammps

from src.experiment_runner.artifacts import file_hash as sha, write_json
from src.experiment_runner.execution import allocation_deadline
from src.project_runtime.paths import resolve_path
from src.data.trajectories.shooting import ShootingBinaryTrajectory, compose_shooting_binary_trajectories
from src.simulation.campaigns.ni_births import md
from src.simulation.campaigns.ni_births.validation import verify
from src.simulation.eam_gpu.npt import install
from src.research.meam_md.dynamics import structural, archive_file


def read(path):
    return json.loads(Path(path).read_text())


def configuration(settings):
    original = read(settings['campaign_config']); plan = read(resolve_path(original['launch'])/'plan.json')
    if (sha(resolve_path(original['launch'])/'plan.json') != settings['campaign_plan_sha256']
            or plan['config'] != original or sha(__file__) != settings['entry_sha256']):
        raise ValueError('Ni handoff source plan or frozen entry changed')
    if settings['hold_ps'] != 400 or original['cooling_ps'] != 20:
        raise ValueError('Ni handoff must retain the user-requested duration policy')
    return original,dict(original,hold_ps=settings['hold_ps']),plan


def append(path,row):
    exists = path.exists()
    with path.open('a') as f:
        writer = csv.DictWriter(f,fieldnames=list(row))
        if not exists: writer.writeheader()
        writer.writerow(row)


def resume(settings,c,record,deadline):
    folder = Path(settings['launch']); prefix = read(folder/'prefix.json')
    cp = prefix['checkpoint']; step0 = cp['step']
    if record != cp['record'] or sha(cp['absolute_path']) != cp['sha256']:
        raise ValueError('Ni native checkpoint or source identity changed')
    archive = resolve_path(c['archive'])/record['id']
    root = resolve_path(c['scratch'])/record['id']/'rtx-resume-01'; root.mkdir(exist_ok=False)
    paths = []
    for item in prefix['chunks']:
        p = Path(item['path'])
        if sha(p/'manifest.json') != item['manifest_sha256'] or sha(p/'conversion.json') != item['conversion_sha256']:
            raise ValueError('Ni archived prefix chunk changed')
        paths.append(p)
    preserved = Path(prefix['archive'])
    for name in ['thermodynamics.csv','structure.csv']:
        rows = list(csv.DictReader((preserved/name).open()))
        kept = [r for r in rows if int(r['step']) <= step0]
        if int(kept[-1]['step']) != step0:raise ValueError('Ni restart boundary missing from monitoring')
        target = archive/name
        with target.open('w') as f:
            writer = csv.DictWriter(f,fieldnames=list(kept[0]));writer.writeheader();writer.writerows(kept)
    structure_rows = list(csv.DictReader((archive/'structure.csv').open()))
    first_stop = next((int(r['step']) for r in structure_rows if float(r['crystal_fraction'])>=c['stop_crystal_fraction']),None)
    metadata = dict(prefix['original_metadata'],hold_max_ps=c['hold_ps'],original_hold_max_ps=500,
        execution_handoff=dict(receipt=str(folder/'launch.json'),entry_sha256=settings['entry_sha256'],
            restart_sha256=cp['sha256'],restart_step=step0,previous_job=settings['previous_job'],
            resumed_job=os.environ['SLURM_JOB_ID'],preserved_prefix=str(preserved),
            contract='Native float64 NPT restart; matching fix integrate state; no quantized restart'))
    write_json(archive/'metadata.json',metadata)
    native = lammps(cmdargs=['-screen','none','-log',str(root/'lammps.log')])
    started = time.monotonic(); callback = None
    try:
        md.commands(native,root,f'''units metal
atom_style atomic
boundary p p p
atom_modify sort 0 0.0
read_restart {cp['absolute_path']}
neighbor 0.6 bin
neigh_modify every 1 delay 0 check yes
timestep {c['timestep_ps']}
thermo 1000
thermo_style custom step time temp press vol pe ke etotal
thermo_modify flush yes lost error''')
        if int(native.get_natoms()) != c['atoms']:raise ValueError('Native Ni restart atom count changed')
        before = md.snapshot(native)
        callback = install(native,resolve_path(c['potential']),root)
        md.ensemble(native,root,c,record['temperature_K'],record['temperature_K'])
        after = md.snapshot(native)
        if int(native.get_thermo('step')) != step0 or any(not np.array_equal(a,b) for a,b in zip(before,after)):
            raise ValueError('Native restart initialization changed the timestep, Q/V or box')
        log = (root/'lammps.log').read_text()
        if 'fix style: npt, fix ID: integrate' not in log or 'Unused restart file global fix info' in log:
            raise ValueError('Matching native NPT fix state was not confirmed in the restart log')
        write_json(folder/'restart-verified.json',dict(source=record['id'],step=step0,
            native_qv_box_unchanged=True,npt_fix_state_restored=True,gpu=torch.cuda.get_device_name(),
            job=os.environ['SLURM_JOB_ID'],checkpoint_sha256=cp['sha256']))
        sample = md.steps(c,c['sample_interval_ps']); cp_steps = md.steps(c,c['checkpoint_ps'])
        maximum = md.steps(c,c['cooling_ps']+c['hold_ps']); values=[]; termination='hard_hold_cap_native'
        for step in range(step0+sample,maximum+1,sample):
            md.commands(native,root,f'run {sample} pre no post no')
            q,v,lo,hi = md.snapshot(native)
            if not np.isfinite(q).all() or not np.isfinite(v).all():raise FloatingPointError(f'Nonfinite resumed Ni step {step}')
            values.append((step,q,v,lo,hi))
            append(archive/'thermodynamics.csv',dict(step=step,time_ps=step*c['timestep_ps'],phase='isothermal',target_temperature_K=record['temperature_K'],**md.thermo(native)))
            if step%md.steps(c,c['structure_ps'])==0:
                observed=structural(q,hi-lo,dict(ptm_rmsd_cutoff=c['ptm_rmsd_cutoff'],cluster_cutoff_A=c['neighbor_cutoff_A']))
                append(archive/'structure.csv',dict(step=step,time_ps=step*c['timestep_ps'],phase='isothermal',**observed))
                if first_stop is None and observed['crystal_fraction']>=c['stop_crystal_fraction']:first_stop=step
            stop=first_stop is not None and step>=max(md.steps(c,c['cooling_ps']),first_stop)+md.steps(c,c['post_stop_ps'])
            timed_out=time.time()>deadline
            if step%cp_steps==0 or stop or timed_out or step==maximum:
                chunks=root/'chunks';chunks.mkdir(exist_ok=True)
                name=f'chunk-{values[0][0]:09d}-{values[-1][0]:09d}';raw=chunks/(name+'.npz');target=chunks/name
                np.savez(raw,positions=np.stack([a[1] for a in values]),velocities=np.stack([a[2] for a in values]),
                    box_low=np.stack([a[3] for a in values]),box_high=np.stack([a[4] for a in values]),
                    timesteps=np.array([a[0] for a in values],np.int64),atom_ids=np.arange(1,c['atoms']+1,dtype=np.int64),
                    atom_types=np.ones(c['atoms'],np.int32),provenance=json.dumps(metadata))
                md.convert_chunk(raw,target);paths.append(target);values=[]
                md.checkpoint(native,root,archive,'isothermal',step,record)
                elapsed=time.monotonic()-started
                write_json(archive/'progress.json',dict(state='running',phase='isothermal',time_ps=step*c['timestep_ps'],
                    maximum_ps=maximum*c['timestep_ps'],elapsed_seconds=elapsed,observed_md_steps=step-step0,
                    measured_steps_per_second=(step-step0)/elapsed,latest_chunk=str(target),
                    crystal_fraction=observed['crystal_fraction'],job=os.environ['SLURM_JOB_ID']))
            if stop:termination='half_crystalline_plus_10ps';break
            if timed_out:termination='walltime_partial';break
        binaries=[ShootingBinaryTrajectory.load(p) for p in paths]
        for b in binaries:b.verify_checksums()
        if not np.array_equal(np.concatenate([b.timesteps for b in binaries]),np.arange(0,step+1,sample)):
            raise ValueError('Ni migrated trajectory has a gap or duplicate checkpoint frame')
        state='partial' if termination=='walltime_partial' else 'complete';metadata['state']=state
        target=archive/'trajectory_binary_float16'
        compose_shooting_binary_trajectories(binaries,target,timesteps=range(0,step+1,sample),storage_dtype='float16',
            provenance=dict(**metadata,measurement_duration_ps=step*c['timestep_ps'],termination=termination,
                source_chunks=[dict(path=str(p),manifest_sha256=sha(p/'manifest.json'),conversion=read(p/'conversion.json')) for p in paths]))
        del binaries
        ShootingBinaryTrajectory.load(target).verify_checksums()
        result=dict(state=state,record=record,frames=step//sample+1,duration_ps=step*c['timestep_ps'],termination=termination,
            trajectory_manifest_sha256=sha(target/'manifest.json'),resumed_seconds=time.monotonic()-started,
            callbacks=callback.calls,gpu=torch.cuda.get_device_name(),job=os.environ['SLURM_JOB_ID'],execution_handoff=metadata['execution_handoff'])
        write_json(archive/'metadata.json',metadata);write_json(archive/'outcome.json',result)
        if state=='complete':write_json(archive/'complete.json',result)
        return result
    finally:
        native.close()
        for name in ['executed.in','lammps.log']:
            if (root/name).exists():archive_file(root/name,archive/name)


def worker(settings,stage):
    original,c,plan=configuration(settings);folder=Path(settings['launch']);receipt=read(folder/'launch.json')
    if os.environ['SLURM_JOB_ID'] != receipt['jobs'][stage]:raise ValueError('Ni handoff entered an unowned allocation')
    verify(c,read(settings['validation']),code_root=Path(settings['campaign_config']).parent)
    record=next(r for r in plan['records'] if r['id']==settings[stage+'_source'])
    write_json(folder/(stage+'-execution.json'),dict(state='running',source=record['id'],job=os.environ['SLURM_JOB_ID'],
        gpu=torch.cuda.get_device_name(),node=os.uname().nodename,entry_sha256=sha(__file__),effective_hold_ps=c['hold_ps']))
    try:
        deadline=allocation_deadline(reserve_seconds=600)
        result=resume(settings,c,record,deadline) if stage=='resume' else md.run(c,record,deadline)
        write_json(folder/(stage+'-complete.json'),result)
        if result['state']!='complete':raise RuntimeError('Ni handoff source did not finish')
    except BaseException:
        write_json(folder/(stage+'-failure.json'),dict(traceback=traceback.format_exc()))
        raise


def gate(settings):
    original,_,plan=configuration(settings);folder=Path(settings['launch'])
    for stage in ['resume','fresh']:
        if read(folder/(stage+'-complete.json'))['state']!='complete':raise ValueError('Ni handoff source incomplete')
    results=[]
    for record in plan['records']:
        archive=resolve_path(original['archive'])/record['id'];done=read(archive/'complete.json')
        if sha(archive/'trajectory_binary_float16/manifest.json')!=done['trajectory_manifest_sha256']:
            raise ValueError('Ni completed trajectory changed before handoff gate')
        results.append(dict(source=record['id'],state=done['state'],duration_ps=done['duration_ps']))
    write_json(folder/'controller/status.json',dict(state='complete',sources=results,
        checked_at=datetime.now(timezone.utc).isoformat(),policy='Native 400-ps hold enforcement on relocated workers; full saved-history analysis'))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['resume','fresh','gate']);p.add_argument('--config',required=True)
    a=p.parse_args();settings=read(a.config)
    if a.stage=='gate':gate(settings)
    else:worker(settings,a.stage)


if __name__=='__main__':main()
