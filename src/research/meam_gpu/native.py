"""Pinned native Kokkos speed reference and explicitly seeded interface preparation."""
import argparse
import os
import subprocess
import time
import numpy as np
from ase import Atoms
from ase.io import write,read as ase_read
from src.project_runtime.paths import resolve_path
from src.research.local_response.common import read,write_json,sha
from src.research.meam_md.dynamics import archive_file,structural


def execute(c,name,commands):
    root=resolve_path(c['simulation_scratch'])/name;root.mkdir(parents=True,exist_ok=True)
    software=resolve_path(c['software_archive'])/'upstream-double';receipt=read(software/'build.json')
    if sha(software/'lmp')!=receipt['binary_sha256']:raise ValueError('Native reference binary differs')
    source=root/'run.in';source.write_text(commands)
    command=[str(software/'lmp'),'-k','on','g','1','-sf','kk','-pk','kokkos','neigh','half','newton','on','-in','run.in','-log','run.log']
    t=time.time()
    with (root/'stdout.log').open('w') as log:
        subprocess.run(command,cwd=root,check=True,stdout=log,stderr=subprocess.STDOUT)
    archive=resolve_path(c['simulation_archive'])/name
    for p in root.iterdir():
        if p.is_file():archive_file(p,archive/p.name)
    write_json(archive/'execution.json',dict(command=command,seconds=time.time()-t,build=receipt,
        input_sha256=sha(source),allocation=os.environ.get('SLURM_JOB_ID'),host=os.uname().nodename))
    return root,archive


def forcefield(c):
    f={k:resolve_path(v) for k,v in c['meam_files'].items()}
    return f'''mass 1 {c['mass_amu']}
pair_style meam
pair_coeff * * {f['library']} Al {f['parameters']} Al
neighbor 0.6 bin
neigh_modify every 1 delay 0 check yes
timestep 0.001
thermo 1000
thermo_style custom step time temp pe ke etotal press vol
'''


def benchmark(c):
    initial=resolve_path(c['ancestor'])/'initial.npz'
    if sha(initial)!=c['ancestor_sha256']:raise ValueError('Initial liquid changed')
    root=resolve_path(c['simulation_scratch'])/'native-reference';root.mkdir(parents=True,exist_ok=True)
    with np.load(initial) as data:q=data['q'];box=data['box']
    write(root/'initial.data',Atoms('Al'*len(q),positions=q,cell=box,pbc=True),format='lammps-data',atom_style='atomic')
    commands='units metal\natom_style atomic\nboundary p p p\nread_data initial.data\n'+forcefield(c)+'''
run 0
write_dump all custom initial.forces id type x y z fx fy fz modify sort id format float %.17g
velocity all create 450 26100423 mom yes rot no dist gaussian
fix integrate all nve
run 100
run 1000
run 1000
run 1000
'''
    execute(c,'native-reference',commands)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['benchmark']);parser.add_argument('--config',required=True)
    a=parser.parse_args();globals()[a.action](read(a.config))


if __name__=='__main__':main()
