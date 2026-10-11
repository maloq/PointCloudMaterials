"""Native LAMMPS isotropic NPT with the pinned GPU Ni EAM force engine."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from ase.build import bulk
from lammps import lammps
from src.experiment_runner.artifacts import file_hash as sha,write_json
from src.project_runtime.paths import resolve_path
from src.research.meam_md.dynamics import structural,archive_file
from src.data.trajectories.shooting import ShootingBinaryTrajectory,compose_shooting_binary_trajectories
from src.simulation.eam_gpu.npt import install


def steps(c,ps):
    value=ps/c['timestep_ps']
    if abs(value-round(value))>1e-8:raise ValueError('Nonintegral physical interval')
    return round(value)


def commands(l,root,text):
    with (root/'executed.in').open('a') as f:f.write(text+'\n')
    l.commands_string(text)


def start(c,root,record,*,native=False,q=None,box=None):
    root.mkdir(parents=True,exist_ok=True)
    if q is None:
        a=bulk('Ni','fcc',a=c['initial_lattice_A'],cubic=True).repeat((c['fcc_repeats'],)*3)
        q=a.positions;box=a.cell.lengths()
        if len(q)!=c['atoms']:raise ValueError('Ni cell atom count mismatch')
    l=lammps(cmdargs=['-screen','none','-log',str(root/'lammps.log')])
    commands(l,root,f'''units metal
atom_style atomic
boundary p p p
atom_modify sort 0 0.0
region box block 0 {box[0]} 0 {box[1]} 0 {box[2]}
create_box 1 box
mass 1 {c['mass_amu']}
neighbor 0.6 bin
neigh_modify every 1 delay 0 check yes
timestep {c['timestep_ps']}
thermo 1000
thermo_style custom step time temp press vol pe ke etotal
thermo_modify flush yes lost error
''')
    l.create_atoms(len(q),atomid=np.arange(1,len(q)+1,dtype=np.int32),atype=np.ones(len(q),np.int32),x=np.ascontiguousarray(q.flatten()))
    if int(l.get_natoms())!=len(q):raise ValueError('Native atom creation lost atoms')
    potential=resolve_path(c['potential'])
    if sha(potential)!=c['potential_sha256']:raise ValueError('Ni potential changed')
    if native:
        commands(l,root,f'pair_style eam/fs\npair_coeff * * {potential} Ni');callback=None
    else:
        callback=install(l,potential,root)
        with (root/'executed.in').open('a') as f:f.write('# GPU force callback installed: Ni EAM/FS; isotropic virial trace only\n')
    commands(l,root,f"velocity all create {c['melt_temperature_K']} {record['preparation_seed']} mom yes rot no dist gaussian loop all")
    return l,callback


def ensemble(l,root,c,first,last):
    commands(l,root,f"fix integrate all npt temp {first} {last} {c['thermostat_damping_ps']} iso {c['pressure_bar']} {c['pressure_bar']} {c['barostat_damping_ps']}\nrun 0")


def snapshot(l):
    n=int(l.get_natoms());low,high,*_=l.extract_box();box=np.subtract(high,low)
    q=np.array(l.gather_atoms('x',1,3)[:3*n],dtype=np.float64).reshape(n,3)-low
    v=np.array(l.gather_atoms('v',1,3)[:3*n],dtype=np.float64).reshape(n,3)
    return np.remainder(q,box),v,np.asarray(low),np.asarray(high)


def thermo(l):
    return {k:float(l.get_thermo(k)) for k in ['temp','press','vol','pe','ke','etotal']}


def checkpoint(l,root,archive,phase,step,record):
    path=root/'checkpoints'/f'{phase}-{step:09d}.restart';path.parent.mkdir(exist_ok=True)
    l.command(f'write_restart {path}')
    archive_file(path,archive/'checkpoints'/path.name)
    write_json(archive/'checkpoint.json',dict(phase=phase,step=step,path='checkpoints/'+path.name,
        sha256=sha(path),record=record,integration_dtype='native LAMMPS float64',
        resume_contract='Restore matching fix integrate NPT state and reinstall GPU callback; no quantized observations used for restart'))


def audit(c):
    root=resolve_path(c['launch'])/'audit';root.mkdir(parents=True,exist_ok=True)
    a=bulk('Ni','fcc',a=3.6,cubic=True).repeat((c['fcc_repeats'],)*3);rng=np.random.default_rng(261007303)
    q=(a.positions+rng.normal(0,.12,a.positions.shape))%a.cell.lengths()
    results={}
    for mode in ['native','gpu']:
        folder=root/mode;l,cb=start(c,folder,dict(preparation_seed=261007307),native=mode=='native',q=q,box=a.cell.lengths())
        try:
            ensemble(l,folder,c,1370,1370)
            def observe():
                return dict(q=snapshot(l)[0],v=snapshot(l)[1],box=snapshot(l)[3]-snapshot(l)[2],
                    f=np.array(l.gather_atoms('f',1,3)[:3*len(q)]).reshape(-1,3),**thermo(l))
            before=observe();t=time.monotonic();commands(l,folder,'run 100');after=observe()
            results[mode]=dict(before=before,after=after,seconds=time.monotonic()-t)
            if mode=='gpu':
                q0,v0,lo,hi=snapshot(l);commands(l,folder,'run 100');q1,v1,lo1,hi1=snapshot(l)
                path=folder/'audit-qv.npz'
                np.savez(path,positions=np.stack([q0,q1]),velocities=np.stack([v0,v1]),
                    box_low=np.stack([lo,lo1]),box_high=np.stack([hi,hi1]),timesteps=np.array([100,200],np.int64),
                    atom_ids=np.arange(1,len(q)+1,dtype=np.int64),atom_types=np.ones(len(q),np.int32),
                    provenance=json.dumps(dict(protocol='Ni NPT numerical adapter and observation audit',timestep_ps=c['timestep_ps'])))
                convert_chunk(path,folder/'audit-binary')
                b=ShootingBinaryTrajectory.load(folder/'audit-binary')
                if not np.array_equal(b.velocities,np.stack([v0,v1]).astype(np.float16)):raise ValueError('Saved velocities differ')
                from src.postprocessing.descriptors.structure import structure
                from src.postprocessing.protocols.ni import extraction_config
                lengths=b.box_high[-1].astype(float)-b.box_low[-1].astype(float)
                clusters,counts,_=structure(np.mod(b.positions[-1].astype(float),lengths),lengths,extraction_config(c))
                write_json(folder/'structure-audit.json',dict(detectors=4,shape=list(clusters.shape),solid_counts=counts.tolist()))
                compose_shooting_binary_trajectories([b],folder/'audit-composed',timesteps=[100,200],storage_dtype='float16',
                    provenance=dict(protocol='Ni NPT observation publication and NFS cleanup audit',
                        source_manifest_sha256=sha(b.root/'manifest.json'),
                        conversion=json.loads((b.root/'conversion.json').read_text())))
                del b
                shutil.rmtree(folder/'audit-binary')
                ShootingBinaryTrajectory.load(folder/'audit-composed').verify_checksums()
        finally:l.close()
    ref,gpu=results['native'],results['gpu'];delta=ref['after']['q']-gpu['after']['q'];box=gpu['after']['box'];delta-=box*np.rint(delta/box)
    values=dict(force_rms_eV_A=float(np.sqrt(np.mean((ref['before']['f']-gpu['before']['f'])**2))),
        energy_error_eV_atom=abs(ref['before']['pe']-gpu['before']['pe'])/len(q),
        pressure_error_bar=abs(ref['before']['press']-gpu['before']['press']),
        final_position_rms_A=float(np.sqrt(np.mean(delta**2))),
        final_volume_relative_error=abs(ref['after']['vol']/gpu['after']['vol']-1),
        gpu_seconds_per_step=gpu['seconds']/100,native_seconds_per_step=ref['seconds']/100)
    limits=dict(force_rms_eV_A=.001,energy_error_eV_atom=3e-5,pressure_error_bar=2,
                final_position_rms_A=.002,final_volume_relative_error=2e-5)
    passed=all(values[k]<=limit for k,limit in limits.items())
    write_json(root/'validation.json',dict(state='passed' if passed else 'failed',metrics=values,limits=limits,
        atoms=len(q),gpu=torch.cuda.get_device_name(),job=os.environ.get('SLURM_JOB_ID'),
        lammps_version=20250722,production_eligible=False,observation_chunk_cleanup_passed=True,
        implementation_sha256={str(p.relative_to(Path(__file__).parents[4])):sha(p) for p in
            [Path(__file__),Path(__file__).parents[2]/'eam_gpu/npt.py',Path(__file__).parents[3]/'data/conversion/native_qv.py']}))
    if not passed:raise ValueError(f'Ni NPT numerical validation failed: {values}')
    print(json.dumps(values),flush=True)


def convert_chunk(source,target):
    # Use the maintained conversion entry point; stdout is an operational receipt.
    command=[sys.executable,'scripts/convert_trajectory.py','native-qv',str(source),str(target),'--delete-source']
    with target.with_suffix('.conversion.log').open('w') as log:
        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)


def run(c,record,deadline):
    root=resolve_path(c['scratch'])/record['id'];archive=resolve_path(c['archive'])/record['id']
    if (archive/'complete.json').exists():return json.loads((archive/'complete.json').read_text())
    if (root/'started.json').exists():raise FileExistsError(f'Inspect existing native checkpoint before recovery: {root}')
    root.mkdir(parents=True,exist_ok=True);archive.mkdir(parents=True,exist_ok=True)
    metadata=dict(**record,protocol=c['protocol'],material='Ni',potential='ni-mendelev2012-eam',
        potential_sha256=c['potential_sha256'],timestep_ps=c['timestep_ps'],sample_interval_ps=c['sample_interval_ps'],
        ensemble='isotropic NPT',pressure_bar=c['pressure_bar'],atom_count=c['atoms'],
        integration_dtype='float64',force_dtype='float32',observation_dtype='float16',
        native_velocity_units='angstrom_per_ps',cooling_ps=c['cooling_ps'],hold_max_ps=c['hold_ps'],
        root_lineage=record['id'],split=record['role'],state='running')
    write_json(root/'started.json',dict(job=os.environ.get('SLURM_JOB_ID'),started_at=time.time()))
    write_json(archive/'metadata.json',metadata)
    l,callback=start(c,root,record);started=time.monotonic();chunk_paths=[]
    try:
        ensemble(l,root,c,c['melt_temperature_K'],c['melt_temperature_K'])
        cp=steps(c,c['checkpoint_ps']);melt=steps(c,c['melt_ps']);msd_start=steps(c,c['melt_msd_start_ps'])
        for stop in range(cp,melt+1,cp):
            commands(l,root,f'run {cp} pre no post no')
            if stop==msd_start:
                commands(l,root,'compute melt_msd all msd com yes\nthermo_style custom step time temp press vol pe ke etotal c_melt_msd[4]\nrun 0')
            checkpoint(l,root,archive,'melt',stop,record)
            write_json(archive/'progress.json',dict(state='running',phase='melt',time_ps=stop*c['timestep_ps'],elapsed_seconds=time.monotonic()-started))
            if time.time()>deadline:raise TimeoutError('Allocation reserve reached during melting; native restart preserved')
        q,v,lo,hi=snapshot(l)
        quality=structural(q,hi-lo,dict(ptm_rmsd_cutoff=c['ptm_rmsd_cutoff'],cluster_cutoff_A=c['neighbor_cutoff_A']))
        msd=float(l.extract_compute('melt_msd',0,1)[3]);quality.update(msd_A2=msd,**thermo(l))
        write_json(archive/'melt-quality.json',quality)
        if msd<c['melt_min_msd_A2'] or quality['crystal_fraction']>c['melt_max_crystal_fraction'] or quality['largest_cluster_atoms']>=c['melt_max_cluster_atoms']:
            raise ValueError(f'Independent melt failed liquid gate: {quality}')
        commands(l,root,'unfix integrate\nuncompute melt_msd\nthermo_style custom step time temp press vol pe ke etotal\nreset_timestep 0')
        ensemble(l,root,c,c['melt_temperature_K'],record['temperature_K'])
        sample=steps(c,c['sample_interval_ps']);cool=steps(c,c['cooling_ps']);maximum=cool+steps(c,c['hold_ps'])
        values=[];first_stop=None;termination='maximum_duration';last_step=0
        thermo_file=archive/'thermodynamics.csv';structure_file=archive/'structure.csv'
        def append(path,row):
            exists=path.exists()
            with path.open('a') as f:
                writer=csv.DictWriter(f,fieldnames=list(row))
                if not exists:writer.writeheader()
                writer.writerow(row)
        for step in range(0,maximum+1,sample):
            if step:
                if step<=cool:commands(l,root,f'run {sample} start 0 stop {cool} pre no post no')
                else:commands(l,root,f'run {sample} pre no post no')
            q,v,lo,hi=snapshot(l);last_step=step
            if not np.isfinite(q).all() or not np.isfinite(v).all():raise FloatingPointError(f'Nonfinite step {step}')
            values.append((step,q,v,lo,hi))
            phase='cooling' if step<cool else 'isothermal'
            append(thermo_file,dict(step=step,time_ps=step*c['timestep_ps'],phase=phase,target_temperature_K=(c['melt_temperature_K']+(record['temperature_K']-c['melt_temperature_K'])*min(1,step/cool)),**thermo(l)))
            if step%steps(c,c['structure_ps'])==0:
                observed=structural(q,hi-lo,dict(ptm_rmsd_cutoff=c['ptm_rmsd_cutoff'],cluster_cutoff_A=c['neighbor_cutoff_A']))
                append(structure_file,dict(step=step,time_ps=step*c['timestep_ps'],phase=phase,**observed))
                if first_stop is None and observed['crystal_fraction']>=c['stop_crystal_fraction']:first_stop=step
            stop=first_stop is not None and step>=max(cool,first_stop)+steps(c,c['post_stop_ps'])
            timed_out=time.time()>deadline
            if (step and step%cp==0) or stop or timed_out or step==maximum:
                folder=root/'chunks';folder.mkdir(exist_ok=True)
                name=f'chunk-{values[0][0]:09d}-{values[-1][0]:09d}';source=folder/(name+'.npz');target=folder/name
                np.savez(source,positions=np.stack([a[1] for a in values]),velocities=np.stack([a[2] for a in values]),
                    box_low=np.stack([a[3] for a in values]),box_high=np.stack([a[4] for a in values]),
                    timesteps=np.array([a[0] for a in values],np.int64),atom_ids=np.arange(1,c['atoms']+1,dtype=np.int64),
                    atom_types=np.ones(c['atoms'],np.int32),provenance=json.dumps(metadata))
                convert_chunk(source,target);chunk_paths.append(target);values=[]
                checkpoint(l,root,archive,'cooling' if step<=cool else 'isothermal',step,record)
                elapsed=time.monotonic()-started
                write_json(archive/'progress.json',dict(state='running',phase=phase,time_ps=step*c['timestep_ps'],
                    maximum_ps=maximum*c['timestep_ps'],elapsed_seconds=elapsed,observed_md_steps=step+melt,
                    measured_steps_per_second=(step+melt)/elapsed,latest_chunk=str(target),crystal_fraction=observed['crystal_fraction']))
                print(dict(source=record['id'],step=step,phase=phase,elapsed_seconds=elapsed),flush=True)
            if step==cool:
                commands(l,root,'unfix integrate');ensemble(l,root,c,record['temperature_K'],record['temperature_K'])
            if stop:termination='half_crystalline_plus_10ps';break
            if timed_out:termination='walltime_partial';break
        binaries=[ShootingBinaryTrajectory.load(p) for p in chunk_paths]
        trajectory=archive/'trajectory_binary_float16'
        state='partial' if termination=='walltime_partial' else 'complete'
        metadata['state']=state
        compose_shooting_binary_trajectories(binaries,trajectory,timesteps=range(0,last_step+1,sample),storage_dtype='float16',
            provenance=dict(**metadata,measurement_duration_ps=last_step*c['timestep_ps'],termination=termination,
                source_chunks=[dict(path=str(p),manifest_sha256=sha(p/'manifest.json'),conversion=json.loads((p/'conversion.json').read_text())) for p in chunk_paths]))
        # Close the source memmaps before unlinking their files on NFS. Open maps
        # otherwise leave .nfs files that make rmtree fail after successful MD.
        del binaries
        ShootingBinaryTrajectory.load(trajectory).verify_checksums()
        for filename in ['executed.in','lammps.log']:
            archive_file(root/filename,archive/filename)
        write_json(archive/'metadata.json',metadata)
        result=dict(state=state,record=record,frames=last_step//sample+1,duration_ps=last_step*c['timestep_ps'],
            termination=termination,trajectory_manifest_sha256=sha(trajectory/'manifest.json'),seconds=time.monotonic()-started,
            callbacks=callback.calls,gpu=torch.cuda.get_device_name(),job=os.environ.get('SLURM_JOB_ID'))
        write_json(archive/'outcome.json',result)
        if state=='complete':write_json(archive/'complete.json',result)
        # Only this producer's verified intermediate observation chunks are removed.
        for path in chunk_paths:shutil.rmtree(path)
        return result
    finally:
        l.close()
        for filename in ['executed.in','lammps.log']:
            if (root/filename).exists():archive_file(root/filename,archive/filename)
