"""Reproducible Blackwell CUDA builds and immutable executable publication."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import time

from src.project_runtime.paths import resolve_path,machine
from src.research.local_response.common import read,write_json,sha


def build(c,variant,jobs):
    root=resolve_path(c['software_scratch']);source=root/'lammps'
    out=resolve_path(c['output'])/'technical';out.mkdir(parents=True,exist_ok=True)
    upstream=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    if upstream!=c['upstream_commit']:raise ValueError(f'Unexpected LAMMPS source: {upstream}')
    precision='double'
    archive=resolve_path(c['software_archive'])/variant
    if (archive/'build.json').exists():
        receipt=read(archive/'build.json')
        if sha(archive/'lmp')!=receipt['binary_sha256']:raise ValueError('Archived executable changed')
        return
    dirty=subprocess.check_output(['git','diff','--binary'],cwd=source,text=True)
    if variant=='upstream-double' and dirty:raise ValueError('Upstream benchmark requires unmodified source')
    directory=root/'build-double'
    cuda=Path(machine()['execution']['gpu_meam_cuda_root'])
    environment=dict(os.environ,PATH=str(cuda/'bin')+':/usr/bin:/bin')
    configure=['cmake','-S',str(source/'cmake'),'-B',str(directory),'-D','CMAKE_BUILD_TYPE=Release',
        '-D','CMAKE_CXX_COMPILER='+str(source/'lib/kokkos/bin/nvcc_wrapper'),'-D','BUILD_MPI=OFF',
        '-D','BUILD_OMP=OFF','-D','PKG_MEAM=ON','-D','PKG_KOKKOS=ON','-D','Kokkos_ENABLE_CUDA=ON',
        '-D','Kokkos_ENABLE_SERIAL=ON','-D','Kokkos_ARCH_'+c['cuda_arch']+'=ON',
        '-D','KOKKOS_PREC='+precision,'-D','BUILD_SHARED_LIBS=OFF']
    started=time.time()
    with (out/f'configure-{variant}.log').open('w') as log:subprocess.run(configure,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    command=['cmake','--build',str(directory),'--parallel',str(jobs)]
    with (out/f'build-{variant}.log').open('w') as log:subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    archive.mkdir(parents=True,exist_ok=True)
    for name in ['lmp','CMakeCache.txt']:shutil.copy2(directory/name,archive/name)
    shutil.copy2(source/'LICENSE',archive/'LICENSE')
    diff=subprocess.check_output(['git','diff','--binary'],cwd=source,text=True)
    (archive/'optimization.patch').write_text(diff)
    write_json(archive/'build.json',dict(variant=variant,upstream_commit=upstream,precision=precision,
        architecture=c['cuda_arch'],binary_sha256=sha(archive/'lmp'),patch_sha256=sha(archive/'optimization.patch'),
        configure=configure,build_command=command,started_unix=started,finished_unix=time.time()))
    print(read(archive/'build.json'),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant',choices=['upstream-double'])
    parser.add_argument('--config',required=True);parser.add_argument('--jobs',type=int,default=6)
    args=parser.parse_args();build(read(args.config),args.variant,args.jobs)


if __name__=='__main__':main()
