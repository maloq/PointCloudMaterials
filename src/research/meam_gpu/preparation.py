"""Explicit crystal/liquid interface control using the fast differentiable engine.

One initially FCC slab is held fixed during partial melting; all atoms are
released for production. This is a seeded growth control, never a birth sample.
"""
import argparse
import math
import os
import time
import numpy as np
import torch
from ase import units
from ase.build import bulk
from src.project_runtime.paths import resolve_path
from src.research.local_response.common import read,write_json,sha
from src.research.meam_md.dynamics import archive_file,structural
from .potential import FastAl


def prepare(c):
    root=resolve_path(c['simulation_scratch'])/'verified-interface-preparation';archive=resolve_path(c['simulation_archive'])/'verified-interface-preparation'
    root.mkdir(parents=True,exist_ok=True)
    if (archive/'complete.json').exists():
        if sha(archive/'initial.npz')!=read(archive/'complete.json')['initial_sha256']:raise ValueError('Interface changed')
        return
    atoms=bulk('Al','fcc',a=4.10,cubic=True).repeat((16,16,16));box=atoms.cell.lengths()
    q=torch.tensor(atoms.positions,device='cuda',dtype=torch.float64)
    mobile=(q[:,2]>=box[2]/2)[:,None];fixed=q.clone();mass=c['mass_amu'];dt=units.fs
    files={k:resolve_path(v) for k,v in c['meam_files'].items()}
    model=FastAl(files,len(q),box=box,dtype=torch.float32,compile_tangents=False)
    # Validate mixed arithmetic on this new solid geometry before generating it.
    exact=FastAl(files,len(q),box=box,dtype=torch.float64,compile_tangents=False)
    e,f=model.force(q);e64,f64=exact.force(q)
    gate=dict(energy_per_atom_error=abs(float(e-e64))/len(q),force_rms=float((f-f64).square().mean().sqrt()))
    if gate['energy_per_atom_error']>c['gates']['mixed_energy_eV_per_atom'] or gate['force_rms']>c['gates']['mixed_force_rms_eV_A']:raise ValueError(f'FCC precision gate failed: {gate}')
    del exact
    rng=np.random.Generator(np.random.PCG64(26100461));p=torch.tensor(rng.standard_normal(q.shape),device='cuda')
    p-=p[mobile[:,0]].mean(0);p*=math.sqrt(mass*units.kB*c['interface_melt_temperature_K']);p*=mobile
    decay=math.exp(-1/100);started=time.monotonic();step=0
    for temperature,steps in [(c['interface_melt_temperature_K'],20000)]:
        scale=math.sqrt(mass*units.kB*temperature*(1-decay**2))
        for _ in range(steps):
            p=(p+dt/2*f)*mobile;q=q+dt/2*p/mass
            p=(decay*p+torch.tensor(rng.standard_normal(q.shape)*scale,device='cuda'))*mobile
            q=torch.where(mobile,q+dt/2*p/mass,fixed)
            e,f=model.force(q);p=(p+dt/2*f)*mobile;step+=1
            if step%1000==0:
                if not torch.isfinite(q).all() or not torch.isfinite(p).all():raise FloatingPointError(f'Interface preparation failed at step{step}')
                native=dict(q=q.cpu().numpy(),p=p.cpu().numpy(),box=box,step=step,rng_state=str(rng.bit_generator.state))
                np.savez(root/'state.npz',**native);archive_file(root/'state.npz',archive/'state.npz')
                write_json(root/'progress.json',dict(step=step,temperature_K=temperature,mobile_temperature_K=float(p.square().sum()/(mass*(3*int(mobile.sum())-3)*units.kB)),seconds=time.monotonic()-started))
                print(dict(stage='interface preparation',step=step,seconds=time.monotonic()-started),flush=True)
    displacement=q-fixed
    mobile_msd=float(displacement[mobile[:,0]].square().sum(-1).mean())
    positions=q.cpu().numpy()%box
    metrics=structural(positions,box,c)
    from src.postprocessing.descriptors.ptm import full_labels
    labels=full_labels(positions,box,c['ptm_rmsd_cutoff'])
    mobile_solid=float(np.isin(labels[mobile[:,0].cpu().numpy()],[1,2,3]).mean())
    phase_audit=dict(mobile_msd_A2=mobile_msd,mobile_crystal_fraction=mobile_solid)
    write_json(archive/'phase-audit.json',phase_audit)
    if mobile_msd<4 or mobile_solid>.1:raise ValueError(f'Mobile region did not melt: {phase_audit}')
    if not .2<metrics['crystal_fraction']<.8:raise ValueError(f'Not a crystal-liquid interface: {metrics}')
    np.savez(root/'initial.npz',q=positions,box=box,atom_ids=np.arange(1,len(q)+1,dtype=np.int64))
    archive_file(root/'initial.npz',archive/'initial.npz')
    write_json(archive/'complete.json',dict(initial_sha256=sha(archive/'initial.npz'),structure=metrics,atoms=len(q),box_A=box.tolist(),
        potential='Pinned Lee2003 Al; float32 force / float64 state',gate=gate,allocation=os.environ.get('SLURM_JOB_ID'),
        phase_audit=phase_audit,provenance=f"FCC16x16x16,a4.10A; half frozen;20ps at{c['interface_melt_temperature_K']}K, BAOAB1fs100fs friction,PCG64seed26100461. Diffusion and structure checked before cooling. Fresh450K velocities and all atoms released at production start.",
        interpretation='Seeded growth control, not spontaneous nucleation or an independent liquid melt.'))
    print(read(archive/'complete.json'),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    a=parser.parse_args();torch.set_num_threads(1);prepare(read(a.config))


if __name__=='__main__':main()
