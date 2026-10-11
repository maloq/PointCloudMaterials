"""Same-cell periodic force correctness and isolated CPU/GPU cost measurements."""
import argparse
import os
import time

import numpy as np

from src.project_runtime.paths import resolve_path
from src.research.local_response.common import read,write_json,sha
from src.experiment_runner.metric_docs import write_metric_rows


def profile(c,engine):
    from mpi4py import MPI
    comm=MPI.COMM_WORLD;rank=comm.Get_rank()
    initial=resolve_path(c['simulation_archive'])/'preparation/initial.npz'
    if sha(initial)!=read(initial.parent/'complete.json')['initial_sha256']:
        raise ValueError('Initial liquid checksum changed')
    with np.load(initial) as data:q=data['q'].copy();box=data['box'].copy()
    files={k:resolve_path(v) for k,v in c['meam_files'].items()}
    out=resolve_path(c['output'])/'analyses/profile-v1'
    from .cpu import PeriodicLAMMPS
    cpu=PeriodicLAMMPS(files,q,box,comm)
    reference_energy,reference_force=cpu(q)
    if engine=='gpu':
        if comm.Get_size()!=1:raise ValueError('GPU profile uses one rank')
        import torch
        from .potential import PeriodicAl
        torch.set_num_threads(1)
        x=torch.tensor(q,device='cuda',dtype=torch.float64)
        variants=[('eager',PeriodicAl(files,len(q),box)),('compiled',PeriodicAl(files,len(q),box,compile_force=True))]
        rows=[]
        for name,potential in variants:
            begin=time.monotonic();energy,force=potential.force(x);torch.cuda.synchronize();warm=time.monotonic()-begin
            energy_error=abs(float(energy)-reference_energy)/len(q)
            force_error=float(np.sqrt(np.mean((force.cpu().numpy()-reference_force)**2)))
            if energy_error>1e-8 or force_error>1e-8:raise ValueError(f'Periodic liquid gate failed: {name},{energy_error},{force_error}')
            for repeat in range(3):
                torch.cuda.synchronize();begin=time.perf_counter()
                for _ in range(50):potential.force(x)
                torch.cuda.synchronize();seconds=(time.perf_counter()-begin)/50
                rows.append(dict(engine='gpu',mode=name,ranks=1,repeat=repeat,atoms=len(q),
                    seconds_per_force=seconds,energy_error_eV_per_atom=energy_error,force_rms_eV_A=force_error,
                    warmup_seconds=warm,hardware=torch.cuda.get_device_name()))
        write_metric_rows(rows,out,family='meam_md',name='gpu-force')
        write_json(out/'technical/gpu-gate.json',dict(passed=True,rows=rows))
        print(rows,flush=True)
    else:
        rows=[]
        for repeat in range(3):
            comm.Barrier();begin=time.perf_counter()
            for _ in range(50):cpu(q)
            seconds=comm.allreduce((time.perf_counter()-begin)/50,op=MPI.MAX)
            if rank==0:rows.append(dict(engine='cpu',mode='external-force',ranks=comm.Get_size(),
                repeat=repeat,atoms=len(q),seconds_per_force=seconds))
        # Separately disclose native-LAMMPS throughput without per-step Python
        # scatter/gather and forced neighbor rebuilds. This control is NVE, not
        # the matched-noise BAOAB experiment used for trajectory divergence.
        cpu.lmp.commands_list(['neigh_modify every 1 delay 0 check yes',
            f'velocity all create {c["temperature_K"]} {c["velocity_seed"]} mom yes rot no dist gaussian',
            f'timestep {c["timestep_fs"]/1000}','fix native all nve','run 100'])
        native=[]
        for repeat in range(3):
            comm.Barrier();begin=time.perf_counter();cpu.lmp.command('run 1000')
            seconds=comm.allreduce(time.perf_counter()-begin,op=MPI.MAX)
            if rank==0:native.append(dict(engine='cpu',mode='native-nve',ranks=comm.Get_size(),
                repeat=repeat,atoms=len(q),seconds_per_step=seconds/1000,steps_per_second=1000/seconds))
        if rank==0:
            for row in rows+native:
                row.update(hostname=os.uname().nodename,cpu_affinity_count=len(os.sched_getaffinity(0)))
            write_metric_rows(rows,out,family='meam_md',name='cpu-force')
            write_metric_rows(native,out,family='meam_md',name='cpu-native')
            write_json(out/'technical/cpu-complete.json',dict(force=rows,native=native))
            print(dict(force=rows,native=native),flush=True)
    cpu.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('engine',choices=['cpu','gpu']);parser.add_argument('--config',required=True)
    args=parser.parse_args();profile(read(args.config),args.engine)


if __name__=='__main__':main()
