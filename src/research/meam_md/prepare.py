"""Create and verify one shared periodic liquid ancestor using native LAMMPS."""
import argparse
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from src.project_runtime.paths import resolve_path
from src.research.local_response.common import read, write_json, sha
from src.research.response_meam.potential import FILE_HASHES
from src.research.response_force_engines.engines import checked_files
from src.simulation.runtime import lammps_environment


def prepare(c):
    root=resolve_path(c['simulation_scratch'])/'preparation'
    archive=resolve_path(c['simulation_archive'])/'preparation'
    out=resolve_path(c['output'])/'technical'
    root.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    receipt=archive/'complete.json'
    if receipt.exists():
        saved=read(receipt)
        if saved['config']!=c or sha(archive/'initial.npz')!=saved['initial_sha256']:
            raise ValueError('Prepared liquid/config changed')
        return
    spec=dict(files={k:dict(path=v,sha256=FILE_HASHES[k]) for k,v in c['meam_files'].items()})
    files=checked_files(spec)
    nx,ny,nz=c['repetitions']
    dt=c['timestep_fs']/1000
    melt_steps=round(c['melt_ps']/dt);cool_steps=round(c['cool_equilibration_ps']/dt)
    # A native NPT melt establishes a full periodic cell, never a cropped liquid.
    commands=f'''units metal
atom_style atomic
boundary p p p
lattice fcc {c['lattice_constant_A']}
region box block 0 {nx} 0 {ny} 0 {nz}
create_box 1 box
create_atoms 1 box
mass 1 {c['mass_amu']}
pair_style meam
pair_coeff * * {files['library']} Al {files['parameters']} Al
neighbor 0.6 bin
neigh_modify every 1 delay 0 check yes
timestep {dt}
velocity all create {c['melt_temperature_K']} {c['melt_seed']} mom yes rot no dist gaussian
compute displacement all msd com yes
thermo 1000
thermo_style custom step time temp pe ke etotal press vol c_displacement[4]
fix melt all npt temp {c['melt_temperature_K']} {c['melt_temperature_K']} 0.1 iso 0 0 1.0
restart 10000 melt.restart.1 melt.restart.2
run {melt_steps}
write_restart melted.restart
write_data melted.data
unfix melt
velocity all scale {c['temperature_K']}
fix cool all npt temp {c['temperature_K']} {c['temperature_K']} 0.1 iso 0 0 1.0
run {cool_steps}
write_restart prepared.restart
write_data prepared.data
'''
    input_path=root/'prepare.in'
    if input_path.exists() and input_path.read_text()!=commands:
        raise ValueError('Existing preparation input differs')
    input_path.write_text(commands)
    env=lammps_environment(hide_gpus=True)
    env.update(HYDRA_BOOTSTRAP='fork',CUDA_VISIBLE_DEVICES='')
    command=[str(Path(sys.prefix)/'bin/mpiexec'),'-n',str(c['cpu_ranks']),
             str(Path(sys.prefix)/'bin/lmp'),'-in','prepare.in','-log','prepare.log']
    write_json(out/'preparation-launch.json',dict(command=command,cwd=str(root),config=c,
        started_unix=time.time(),allocation=os.environ.get('SLURM_JOB_ID'),input_sha256=sha(input_path)))
    if not (root/'prepared.data').exists():
        with (root/'stdout.log').open('w') as log:
            subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    from lammps import lammps
    lmp=lammps(cmdargs=['-screen','none','-log','none'])
    lmp.commands_list(['units metal','atom_style atomic','atom_modify map array sort 0 0.0',
                       f'read_data {root/"prepared.data"}'])
    atoms=int(lmp.get_natoms())
    lo,hi,*_=lmp.extract_box();box=np.array(hi)-np.array(lo)
    q=np.ctypeslib.as_array(lmp.gather_atoms('x',1,3),shape=(3*atoms,)).reshape(atoms,3).copy()
    q=(q-np.array(lo))%box;lmp.close()
    if atoms!=c['atoms']:raise ValueError(f'Prepared {atoms} atoms')
    from src.postprocessing.descriptors.ptm import full_labels
    labels=full_labels(q,box,c['ptm_rmsd_cutoff'])
    fraction=float(np.isin(labels,[1,2,3]).mean())
    if fraction>=.01:
        raise ValueError(f'Prepared undercooled cell is not liquid: crystal fraction={fraction}')
    np.savez(root/'initial.npz',q=q,box=box,atom_ids=np.arange(1,atoms+1,dtype=np.int64),ptm=labels)
    # Preserve native integration/restart precision, including unsuccessful runs.
    archive.mkdir(parents=True,exist_ok=True)
    for source in root.iterdir():
        if source.is_file():
            target=archive/source.name;shutil.copy2(source,target)
            if sha(source)!=sha(target):raise IOError(f'Preparation archive mismatch: {source}')
    write_json(receipt,dict(config=c,atoms=atoms,crystal_fraction=fraction,box_A=box.tolist(),
        initial_sha256=sha(archive/'initial.npz'),potential_hashes=FILE_HASHES,
        lineage=f'new_meam_fcc16_melt_{c["melt_seed"]}',
        meaning='One native periodic MEAM liquid ancestor shared by CPU/GPU runs; no independent replicas',
        finished_unix=time.time()))
    print(read(receipt),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();prepare(read(args.config))


if __name__=='__main__':main()
