"""Matched full-cell BAOAB trajectories with native-precision checkpoints.

The CPU arm uses the original LAMMPS MEAM force engine over MPI. Integration
and PCG64 noise are deliberately identical between engines; this is not a
claim that an externally driven CPU arm equals native LAMMPS MD throughput.
"""
import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from ase import units

from src.project_runtime.paths import resolve_path
from src.research.local_response.common import read, write_json, sha, digest


def structural(q,box,c):
    from src.postprocessing.descriptors.ptm import full_labels
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    points=np.remainder(q,box)
    labels=full_labels(points,box,c['ptm_rmsd_cutoff'])
    solid=np.isin(labels,[1,2,3]);selected=points[solid]
    largest=0
    if len(selected):
        edges=cKDTree(selected,boxsize=box).query_pairs(c['cluster_cutoff_A'],output_type='ndarray')
        graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(selected),len(selected))).tocsr()
        _,components=connected_components(graph,directed=False)
        largest=int(np.bincount(components).max())
    return dict(crystal_fraction=float(solid.mean()),fcc_fraction=float((labels==1).mean()),
                hcp_fraction=float((labels==2).mean()),bcc_fraction=float((labels==3).mean()),
                largest_cluster_atoms=largest)


def archive_file(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+'.copying')
    shutil.copy2(source,temporary)
    if sha(source)!=sha(temporary):raise IOError(f'Archive mismatch: {source}')
    temporary.replace(target)


def checkpoint(root,archive,step,q,p,rng,identity):
    target=root/'checkpoints'/f'state-{step:09d}.npz'
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix('.building.npz')
    np.savez(temporary,q=q,p=p,step=step,rng_state=json.dumps(rng.bit_generator.state),identity=identity)
    temporary.replace(target)
    archive_file(target,archive/'checkpoints'/target.name)
    write_json(root/'checkpoint.json',dict(step=step,path=target.name,sha256=sha(target),identity=identity))


def append_csv(path,row):
    exists=path.exists()
    with path.open('a') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(row))
        if not exists:writer.writeheader()
        writer.writerow(row)


def publish_run(root,archive,out):
    """Resume publication independently of an already completed integration."""
    record=read(root/'integration-complete.json')
    binary=root/'trajectory_binary_float32'
    if not (root/'float16_conversion.json').exists():
        command=[sys.executable,'scripts/convert_trajectory.py','temporal-storage',str(binary),'--delete-source']
        with (root/'conversion.log').open('w') as stream:
            subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
    from src.data.trajectories.lammps import TemporalLAMMPSBinaryTrajectory
    TemporalLAMMPSBinaryTrajectory.load(root/'trajectory_binary_float16').verify_checksums()
    for source in (root/'trajectory_binary_float16').iterdir():
        archive_file(source,archive/'trajectory_binary_float16'/source.name)
    archive_file(root/'float16_conversion.json',archive/'float16_conversion.json')
    for filename in ['thermodynamics.csv','structure.csv','integration-complete.json']:
        archive_file(root/filename,archive/filename)
    write_json(archive/'complete.json',record);write_json(out/'technical/progress.json',record)


def run(c,engine,mode='compiled',duration_ps=None,name=None,nve=False):
    from mpi4py import MPI
    comm=MPI.COMM_WORLD
    rank=comm.Get_rank()
    if engine=='gpu' and comm.Get_size()!=1:raise ValueError('One rank owns the GPU cell')
    initial_root=resolve_path(c['simulation_archive'])/'preparation'
    preparation=read(initial_root/'complete.json')
    if sha(initial_root/'initial.npz')!=preparation['initial_sha256']:
        raise ValueError('Prepared liquid changed')
    with np.load(initial_root/'initial.npz') as data:
        initial=data['q'].copy();box=data['box'].copy()
    duration=c['duration_ps'] if duration_ps is None else duration_ps
    name=name or f'{engine}-{mode if engine=="gpu" else "mpi"+str(comm.Get_size())}'
    root=resolve_path(c['simulation_scratch'])/name
    archive=resolve_path(c['simulation_archive'])/name
    out=resolve_path(c['output'])/'analyses'/name
    if rank==0:
        root.mkdir(parents=True,exist_ok=True);(out/'technical').mkdir(parents=True,exist_ok=True)
    comm.Barrier()
    dt=c['timestep_fs']*units.fs
    steps=round(duration*1000/c['timestep_fs'])
    sample=round(c['sample_ps']*1000/c['timestep_fs'])
    cp=round(c['checkpoint_ps']*1000/c['timestep_fs'])
    structure_step=round(c['structure_ps']*1000/c['timestep_fs'])
    if steps%sample:raise ValueError('Duration must end on an observed frame')
    identity=digest(dict(config=c,preparation=preparation['initial_sha256'],engine=engine,
                         mode=mode,duration_ps=duration,nve=nve,ranks=comm.Get_size()))
    if (archive/'complete.json').exists():
        if read(archive/'complete.json')['identity']!=identity:raise ValueError('Completed run differs')
        return
    if (root/'integration-complete.json').exists():
        if read(root/'integration-complete.json')['identity']!=identity:raise ValueError('Finished integration differs')
        if rank==0:publish_run(root,archive,out)
        comm.Barrier()
        return
    q=initial.copy()
    mass=c['mass_amu'];kbt=units.kB*c['temperature_K']
    p=np.random.Generator(np.random.PCG64(c['velocity_seed'])).standard_normal(q.shape)*math.sqrt(mass*kbt)
    p-=p.mean(0)
    rng=np.random.Generator(np.random.PCG64(c['noise_seed']))
    first_step=0
    if (root/'checkpoint.json').exists():
        receipt=read(root/'checkpoint.json');source=root/'checkpoints'/receipt['path']
        if receipt['identity']!=identity or sha(source)!=receipt['sha256']:raise ValueError('Checkpoint differs')
        with np.load(source) as data:
            q=data['q'].copy();p=data['p'].copy();first_step=int(data['step'])
            rng.bit_generator.state=json.loads(str(data['rng_state']))
        if rank==0:
            for filename in ['thermodynamics.csv','structure.csv']:
                path=root/filename
                if path.exists():
                    with path.open() as stream:rows=[r for r in csv.DictReader(stream) if int(r['step'])<=first_step]
                    with path.open('w') as stream:
                        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    files={key:resolve_path(value) for key,value in c['meam_files'].items()}
    positions=None
    if rank==0:
        filename=root/'positions.building.npy'
        positions=np.lib.format.open_memmap(filename,mode='r+' if filename.exists() else 'w+',dtype=np.float32,shape=(steps//sample+1,c['atoms'],3))
        if first_step%sample==0:
            positions[first_step//sample]=np.remainder(np.remainder(q,box).astype(np.float32),box.astype(np.float32))
    started=time.monotonic()
    if engine=='gpu':
        import torch
        from .potential import PeriodicAl
        torch.set_num_threads(1)
        dtype=getattr(torch,c['gpu_dtype'])
        potential=PeriodicAl(files,len(q),box,dtype=dtype,compile_force=mode=='compiled')
        x=torch.as_tensor(q,device='cuda',dtype=dtype)
        momentum=torch.as_tensor(p,device='cuda',dtype=dtype)
        energy,force=potential.force(x)
        # Warm dynamic-size compilation before charging production timing.
        potential.force(x);torch.cuda.synchronize()
        hardware=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,dtype=str(dtype))
    else:
        from .cpu import PeriodicLAMMPS
        potential=PeriodicLAMMPS(files,q,box,comm)
        x,momentum=q,p
        energy,force=potential(x)
        hardware=dict(ranks=comm.Get_size(),lammps_version=potential.lmp.version(),dtype='float64')
    warmup=time.monotonic()-started
    if rank==0:
        write_json(out/'technical/execution.json',dict(identity=identity,config=c,engine=engine,mode=mode,
            nve=nve,duration_ps=duration,hardware=hardware,hostname=os.uname().nodename,
            allocation=os.environ.get('SLURM_JOB_ID'),pid=os.getpid(),warmup_seconds=warmup,starting_step=first_step))
    decay=1. if nve else math.exp(-c['timestep_fs']/c['friction_fs'])
    noise_scale=math.sqrt(mass*kbt*(1-decay*decay))
    dynamics_start=time.monotonic()
    early={0,1,2,5,10,20,50,100,200,500}
    try:
        for step in range(first_step,steps+1):
            if step>first_step:
                momentum=momentum+dt/2*force
                x=x+dt/2*momentum/mass
                if not nve:
                    noise=rng.standard_normal(initial.shape)*noise_scale
                    if engine=='gpu':noise=torch.as_tensor(noise,device='cuda',dtype=dtype)
                    momentum=decay*momentum+noise
                x=x+dt/2*momentum/mass
                if engine=='gpu':energy,force=potential.force(x)
                else:energy,force=potential(x)
                momentum=momentum+dt/2*force
            if step%sample==0 or step%cp==0 or step in early or step==steps:
                if engine=='gpu':
                    q=x.cpu().numpy();p=momentum.cpu().numpy();e=float(energy)
                else:q,p,e=x,momentum,float(energy)
                if not np.isfinite(q).all() or not np.isfinite(p).all() or not np.isfinite(e):
                    raise FloatingPointError(f'Nonfinite dynamics at step{step}')
                if rank==0:
                    if step%sample==0:
                        positions[step//sample]=np.remainder(np.remainder(q,box).astype(np.float32),box.astype(np.float32))
                        centered=p-p.mean(0);ke=float((p*p).sum()/(2*mass))
                        temperature=float((centered*centered).sum()/(mass*(3*len(q)-3)*units.kB))
                        motion=q-initial;motion-=motion.mean(0)
                        row=dict(step=step,time_ps=step*c['timestep_fs']/1000,potential_eV_per_atom=e/len(q),
                            kinetic_eV_per_atom=ke/len(q),total_eV_per_atom=(e+ke)/len(q),temperature_K=temperature,
                            msd_A2=float((motion*motion).sum(-1).mean()))
                        if step>first_step or first_step==0:append_csv(root/'thermodynamics.csv',row)
                    if step%structure_step==0:
                        row=dict(step=step,time_ps=step*c['timestep_fs']/1000,**structural(q,box,c))
                        if step>first_step or first_step==0:append_csv(root/'structure.csv',row)
                    if step%cp==0 or step in early or step==steps:
                        positions.flush();checkpoint(root,archive,step,q,p,rng,identity)
                        elapsed=time.monotonic()-dynamics_start
                        write_json(out/'technical/progress.json',dict(state='running',step=step,total_steps=steps,
                            time_ps=step*c['timestep_fs']/1000,elapsed_seconds=elapsed,
                            steps_per_second=(step-first_step)/elapsed,identity=identity))
                        if step%cp==0 or step==steps:
                            for filename in ['thermodynamics.csv','structure.csv']:
                                archive_file(root/filename,archive/filename)
                            print(dict(engine=engine,mode=mode,step=step,time_ps=step*c['timestep_fs']/1000,
                                       elapsed_seconds=elapsed),flush=True)
        if engine=='gpu':torch.cuda.synchronize()
        elapsed=time.monotonic()-dynamics_start
        if rank==0:
            from src.data.trajectories.lammps import write_temporal_lammps_binary
            positions.flush()
            binary=root/'trajectory_binary_float32'
            write_temporal_lammps_binary(binary,positions=positions,
                timesteps=np.arange(0,steps+1,sample,dtype=np.int64),
                box_low=np.zeros((len(positions),3),np.float32),box_high=np.tile(box,(len(positions),1)).astype(np.float32),
                atom_ids=np.arange(1,len(q)+1,dtype=np.int64),atom_types=np.ones(len(q),np.int32),
                atom_columns=('id','type','x','y','z'),source=dict(protocol=c['protocol'],engine=engine),
                provenance=dict(identity=identity,initial_sha256=preparation['initial_sha256'],timestep_fs=c['timestep_fs'],
                    native_checkpoints='checkpoints/*.npz retain unwrapped integration precision and exact RNG state'),
                consume_positions_file=root/'positions.building.npy')
            del positions
            record=dict(identity=identity,state='complete',steps=steps,duration_ps=duration,
                elapsed_seconds=elapsed,starting_step=first_step,steps_per_second=(steps-first_step)/elapsed,
                warmup_seconds=warmup,hardware=hardware,config=c)
            write_json(root/'integration-complete.json',record)
            publish_run(root,archive,out)
        comm.Barrier()
    except BaseException:
        if rank==0:write_json(out/'technical/failure.json',dict(traceback=traceback.format_exc(),identity=identity))
        raise
    finally:
        if engine=='cpu':potential.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('engine',choices=['cpu','gpu']);parser.add_argument('--config',required=True)
    parser.add_argument('--mode',choices=['eager','compiled'],default='compiled')
    parser.add_argument('--duration-ps',type=float);parser.add_argument('--name');parser.add_argument('--nve',action='store_true')
    args=parser.parse_args()
    try:run(read(args.config),args.engine,args.mode,args.duration_ps,args.name,args.nve)
    except BaseException:
        from mpi4py import MPI
        if MPI.COMM_WORLD.Get_size()>1:
            traceback.print_exc();MPI.COMM_WORLD.Abort(1)
        raise


if __name__=='__main__':main()
