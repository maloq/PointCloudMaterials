"""Float64 BAOAB integration with declared CUDA MEAM force precision.

Reuse the matched 16k liquid and exact PCG64 velocity/noise streams. Native
states stay float64; observation conversion and archival use maintained APIs.
"""
import argparse
import csv
import json
import math
import os
import time
import traceback
import numpy as np
import torch
from ase import units
from src.project_runtime.paths import resolve_path
from src.research.local_response.common import read,write_json,sha,digest
from src.research.meam_md.dynamics import structural,archive_file,checkpoint,append_csv,publish_run
from .potential import FastAl


def run(c,name,force_dtype='float32',duration_ps=None,nve=False,initial_file=None,seed_offset=0):
    root=resolve_path(c['simulation_scratch'])/name
    archive=resolve_path(c['simulation_archive'])/name
    out=resolve_path(c['output'])/'analyses'/name
    root.mkdir(parents=True,exist_ok=True);(out/'technical').mkdir(parents=True,exist_ok=True)
    source=resolve_path(c['ancestor'])/'initial.npz' if initial_file is None else resolve_path(initial_file)
    initial_hash=sha(source)
    if initial_file is None and initial_hash!=c['ancestor_sha256']:raise ValueError('Original liquid changed')
    with np.load(source) as data:initial=data['q'].copy();box=data['box'].copy()
    if len(initial)!=c['atoms'] or initial.dtype!=np.float64:raise ValueError('Native initial state contract differs')
    duration=c['duration_ps'] if duration_ps is None else duration_ps
    dt=c['timestep_fs']*units.fs;steps=round(duration*1000/c['timestep_fs'])
    sample=round(c['sample_ps']*1000/c['timestep_fs']);cp=round(c['checkpoint_ps']*1000/c['timestep_fs'])
    structure_step=round(c['structure_ps']*1000/c['timestep_fs'])
    if steps%sample:raise ValueError('Duration must end on observed frame')
    from pathlib import Path
    package=Path(__file__).parent
    implementation={name:sha(package/name) for name in ('dynamics.py','potential.py','energy.py','topology.cu')}
    identity=digest(dict(config=c,initial=initial_hash,force_dtype=force_dtype,duration=duration,nve=nve,
                         seed_offset=seed_offset,implementation=implementation))
    if (archive/'complete.json').exists():
        if read(archive/'complete.json')['identity']!=identity:raise ValueError('Completed identity differs')
        return
    if (root/'integration-complete.json').exists():
        if read(root/'integration-complete.json')['identity']!=identity:raise ValueError('Integration identity differs')
        publish_run(root,archive,out);return
    mass=c['mass_amu'];kbt=units.kB*c['temperature_K']
    p=np.random.Generator(np.random.PCG64(c['velocity_seed']+seed_offset)).standard_normal(initial.shape)*math.sqrt(mass*kbt)
    p-=p.mean(0);rng=np.random.Generator(np.random.PCG64(c['noise_seed']+seed_offset))
    q=initial.copy();first_step=0
    if (root/'checkpoint.json').exists():
        receipt=read(root/'checkpoint.json');state=root/'checkpoints'/receipt['path']
        if receipt['identity']!=identity or sha(state)!=receipt['sha256']:raise ValueError('Native checkpoint differs')
        with np.load(state) as data:
            q=data['q'].copy();p=data['p'].copy();first_step=int(data['step']);rng.bit_generator.state=json.loads(str(data['rng_state']))
        for filename in ['thermodynamics.csv','structure.csv']:
            path=root/filename
            with path.open() as stream:rows=[r for r in csv.DictReader(stream) if int(r['step'])<=first_step]
            with path.open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    positions_path=root/'positions.building.npy'
    positions=np.lib.format.open_memmap(positions_path,mode='r+' if positions_path.exists() else 'w+',
        dtype=np.float32,shape=(steps//sample+1,len(q),3))
    torch.set_num_threads(1);files={k:resolve_path(v) for k,v in c['meam_files'].items()}
    started=time.monotonic()
    potential=FastAl(files,len(q),box=box,dtype=getattr(torch,force_dtype),compile_tangents=False)
    x=torch.tensor(q,device='cuda',dtype=torch.float64);momentum=torch.tensor(p,device='cuda',dtype=torch.float64)
    energy,force=potential.force(x);potential.force(x);torch.cuda.synchronize()
    warmup=time.monotonic()-started
    hardware=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,force_dtype=force_dtype,
                  state_dtype='float64',energy_reduction_dtype='float64')
    write_json(out/'technical/execution.json',dict(identity=identity,config=c,initial_sha256=initial_hash,
        initial=str(source),nve=nve,duration_ps=duration,hardware=hardware,hostname=os.uname().nodename,
        allocation=os.environ.get('SLURM_JOB_ID'),pid=os.getpid(),warmup_seconds=warmup,
        starting_step=first_step,seed_offset=seed_offset,implementation=implementation))
    decay=1. if nve else math.exp(-c['timestep_fs']/c['friction_fs'])
    scale=math.sqrt(mass*kbt*(1-decay*decay));started=time.monotonic();early={0,1,2,5,10,20,50,100,200,500}
    try:
        for step in range(first_step,steps+1):
            if step>first_step:
                momentum=momentum+dt/2*force;x=x+dt/2*momentum/mass
                if not nve:momentum=decay*momentum+torch.as_tensor(rng.standard_normal(initial.shape)*scale,device='cuda')
                x=x+dt/2*momentum/mass;energy,force=potential.force(x);momentum=momentum+dt/2*force
            if step%sample==0 or step%cp==0 or step in early or step==steps:
                q=x.cpu().numpy();p=momentum.cpu().numpy();e=float(energy)
                if not np.isfinite(q).all() or not np.isfinite(p).all() or not np.isfinite(e):raise FloatingPointError(f'Nonfinite step {step}')
                if step%sample==0:
                    positions[step//sample]=np.remainder(np.remainder(q,box).astype(np.float32),box.astype(np.float32))
                    centered=p-p.mean(0);ke=float((p*p).sum()/(2*mass));motion=q-initial;motion-=motion.mean(0)
                    row=dict(step=step,time_ps=step*c['timestep_fs']/1000,potential_eV_per_atom=e/len(q),
                        kinetic_eV_per_atom=ke/len(q),total_eV_per_atom=(e+ke)/len(q),
                        temperature_K=float((centered*centered).sum()/(mass*(3*len(q)-3)*units.kB)),
                        msd_A2=float((motion*motion).sum(-1).mean()))
                    if step>first_step or first_step==0:append_csv(root/'thermodynamics.csv',row)
                if step%structure_step==0:
                    row=dict(step=step,time_ps=step*c['timestep_fs']/1000,**structural(q,box,c))
                    if step>first_step or first_step==0:append_csv(root/'structure.csv',row)
                if step%cp==0 or step in early or step==steps:
                    positions.flush();checkpoint(root,archive,step,q,p,rng,identity)
                    elapsed=time.monotonic()-started
                    write_json(out/'technical/progress.json',dict(state='running',step=step,total_steps=steps,
                        time_ps=step*c['timestep_fs']/1000,elapsed_seconds=elapsed,steps_per_second=(step-first_step)/elapsed,identity=identity))
                    if step%cp==0 or step==steps:
                        for filename in ['thermodynamics.csv','structure.csv']:archive_file(root/filename,archive/filename)
                        print(dict(name=name,step=step,time_ps=step*c['timestep_fs']/1000,elapsed_seconds=elapsed),flush=True)
        torch.cuda.synchronize();elapsed=time.monotonic()-started
        from src.data.trajectories.lammps import write_temporal_lammps_binary
        positions.flush()
        write_temporal_lammps_binary(root/'trajectory_binary_float32',positions=positions,
            timesteps=np.arange(0,steps+1,sample,dtype=np.int64),box_low=np.zeros((len(positions),3),np.float32),
            box_high=np.tile(box,(len(positions),1)).astype(np.float32),atom_ids=np.arange(1,len(q)+1,dtype=np.int64),
            atom_types=np.ones(len(q),np.int32),atom_columns=('id','type','x','y','z'),
            source=dict(protocol=c['protocol'],engine='CUDA differentiable MEAM'),
            provenance=dict(identity=identity,initial_sha256=initial_hash,timestep_fs=c['timestep_fs'],
                native_checkpoints='checkpoints/*.npz retain float64 integration, momenta, exact RNG state'),consume_positions_file=positions_path)
        del positions
        record=dict(identity=identity,state='complete',steps=steps,duration_ps=duration,elapsed_seconds=elapsed,
            starting_step=first_step,steps_per_second=(steps-first_step)/elapsed,warmup_seconds=warmup,hardware=hardware,config=c)
        write_json(root/'integration-complete.json',record);publish_run(root,archive,out)
    except BaseException:
        write_json(out/'technical/failure.json',dict(identity=identity,traceback=traceback.format_exc()));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--name',required=True);parser.add_argument('--force-dtype',choices=['float32','float64'],default='float32')
    parser.add_argument('--duration-ps',type=float);parser.add_argument('--nve',action='store_true')
    parser.add_argument('--initial-file');parser.add_argument('--seed-offset',type=int,default=0)
    a=parser.parse_args();run(read(a.config),a.name,a.force_dtype,a.duration_ps,a.nve,a.initial_file,a.seed_offset)


if __name__=='__main__':main()
