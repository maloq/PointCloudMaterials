"""Publish native-precision CPU/GPU comparisons, including incomplete coverage."""
import argparse
import csv

import numpy as np
from ase import units

from src.project_runtime.paths import resolve_path
from src.research.local_response.common import read,write_json
from src.experiment_runner.metric_docs import write_metric_rows


def rows(path):
    with path.open() as stream:
        return [{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]


def paired(c,cpu_name,gpu_name):
    archive=resolve_path(c['simulation_archive'])
    with np.load(archive/'preparation/initial.npz') as data:box=data['box']
    cpu={p.name:p for p in (archive/cpu_name/'checkpoints').glob('state-*.npz')}
    gpu={p.name:p for p in (archive/gpu_name/'checkpoints').glob('state-*.npz')}
    result=[]
    for name in sorted(cpu.keys()&gpu.keys()):
        with np.load(cpu[name]) as a,np.load(gpu[name]) as b:
            step=int(a['step'])
            if step!=int(b['step']):raise ValueError('Different checkpoint steps')
            if step==0 and (not np.array_equal(a['q'],b['q']) or not np.array_equal(a['p'],b['p'])):
                raise ValueError('CPU/GPU did not start from identical positions and momenta')
            if str(a['rng_state'])!=str(b['rng_state']):raise ValueError('Thermostat streams differ')
            d=b['q']-a['q'];wrapped=d-box*np.round(d/box)
            centered=d-d.mean(0)
            v=(b['p']-a['p'])/c['mass_amu']*units.fs*1000
            result.append(dict(step=step,time_ps=step*c['timestep_fs']/1000,
                position_rms_A=float(np.sqrt(np.mean(np.sum(wrapped**2,axis=-1)))),
                unwrapped_com_removed_rms_A=float(np.sqrt(np.mean(np.sum(centered**2,axis=-1)))),
                velocity_rms_A_ps=float(np.sqrt(np.mean(np.sum(v**2,axis=-1))))))
    return result


def conservation_gate(c):
    archive=resolve_path(c['simulation_archive'])
    names=['cpu-nve-control','gpu-nve-control']
    for name in names:
        if not (archive/name/'complete.json').exists():raise FileNotFoundError(f'Incomplete NVE control: {name}')
    errors=[]
    for name in names:
        values=rows(archive/name/'thermodynamics.csv')
        energy=np.array([r['total_eV_per_atom'] for r in values])
        error=float(abs(energy-energy[0]).max())
        errors.append(dict(engine=name,max_energy_excursion_eV_per_atom=error,
            final_energy_change_eV_per_atom=float(energy[-1]-energy[0]),duration_ps=values[-1]['time_ps']))
        if error>5e-4:raise ValueError(f'NVE energy gate failed: {name}, {error}')
    differences=paired(c,*names)
    if not differences or differences[-1]['time_ps']!=1.:raise ValueError('NVE comparison must cover exactly 1 ps')
    if differences[-1]['position_rms_A']>1e-4:raise ValueError(f'Initial trajectory fidelity failed: {differences[-1]}')
    out=resolve_path(c['output'])/'analyses/conservation-v1'
    write_metric_rows(errors,out,family='meam_md',name='energy-conservation')
    write_metric_rows(differences,out,family='meam_md',name='divergence')
    write_json(out/'technical/gate.json',dict(passed=True,energy=errors,divergence=differences[-1]))
    return errors


def rdf(q,box,bins=160,cutoff=10.):
    from scipy.spatial import cKDTree
    q=q%box
    i,j=cKDTree(q,boxsize=box).query_pairs(cutoff,output_type='ndarray').T
    d=q[j]-q[i];d-=box*np.round(d/box)
    counts,edges=np.histogram(np.linalg.norm(d,axis=1),bins=bins,range=(0.,cutoff))
    shell=4*np.pi/3*np.diff(edges**3)
    g=counts*2*np.prod(box)/(len(q)*(len(q)-1)*shell)
    return (edges[:-1]+edges[1:])/2,g


def report(c):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=resolve_path(c['output']);archive=resolve_path(c['simulation_archive'])
    out=root/'analyses/comparison-v1';(out/'plots').mkdir(parents=True,exist_ok=True)
    names=['cpu-mpi'+str(c['cpu_ranks']),'gpu-compiled']
    state={name:read(archive/name/'complete.json') if (archive/name/'complete.json').exists()
           else read(root/'analyses'/name/'technical/progress.json')
           if (root/'analyses'/name/'technical/progress.json').exists() else {'state':'not started'} for name in names}
    differences=paired(c,*names)
    if differences:write_metric_rows(differences,out,family='meam_md',name='divergence')
    thermo=[];structure=[]
    for name in names:
        if (archive/name/'thermodynamics.csv').exists():thermo.extend(dict(engine=name,**r) for r in rows(archive/name/'thermodynamics.csv'))
        if (archive/name/'structure.csv').exists():structure.extend(dict(engine=name,**r) for r in rows(archive/name/'structure.csv'))
    if thermo:write_metric_rows(thermo,out,family='meam_md',name='thermodynamics')
    if structure:write_metric_rows(structure,out,family='meam_md',name='crystalline-structure')
    fig,axes=plt.subplots(3,2,figsize=(12,11),constrained_layout=True)
    if differences:
        nonzero=[r for r in differences if r['time_ps']>0]
        axes[0,0].loglog([r['time_ps'] for r in nonzero],[max(r['position_rms_A'],1e-16) for r in nonzero])
        axes[0,0].set(xlabel='Time (ps)',ylabel='CPU–GPU position RMS (Å)',title='Native precision, minimum-image difference')
    for name in names:
        t=[r for r in thermo if r['engine']==name];s=[r for r in structure if r['engine']==name]
        for ax,key,label in [(axes[0,1],'temperature_K','Temperature (K)'),(axes[1,0],'potential_eV_per_atom','Potential energy (eV/atom)'),(axes[1,1],'msd_A2','COM-removed MSD (Å²)')]:
            if t:ax.plot([r['time_ps'] for r in t],[r[key] for r in t],label=name)
            ax.set(xlabel='Time (ps)',ylabel=label)
        for ax,key,label in [(axes[2,0],'crystal_fraction','PTM crystalline fraction'),(axes[2,1],'largest_cluster_atoms','Largest solid cluster (atoms)')]:
            if s:ax.plot([r['time_ps'] for r in s],[r[key] for r in s],label=name)
            ax.set(xlabel='Time (ps)',ylabel=label)
    for ax in axes.flat:
        ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:ax.legend()
    complete=all(v['state']=='complete' for v in state.values())
    fig.suptitle(f'Al 16,384 atoms · 450 K · matched MEAM · {"complete" if complete else "partial coverage"}')
    fig.savefig(out/'plots/cpu-gpu-comparison.png',dpi=170);plt.close(fig)
    rdf_rows=[]
    if differences:
        step=int(differences[-1]['step'])
        with np.load(archive/'preparation/initial.npz') as initial:
            box=initial['box'];r,g=rdf(initial['q'],box)
        rdf_rows.extend(dict(engine='shared_initial',time_ps=0.,r_A=float(x),g_r=float(y)) for x,y in zip(r,g))
        fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
        ax.plot(r,g,label='Shared initial liquid')
        for name in names:
            with np.load(archive/name/'checkpoints'/f'state-{step:09d}.npz') as data:r,g=rdf(data['q'],box)
            rdf_rows.extend(dict(engine=name,time_ps=step*c['timestep_fs']/1000,r_A=float(x),g_r=float(y)) for x,y in zip(r,g))
            ax.plot(r,g,label=name)
        ax.set(xlabel='Distance (Å)',ylabel='g(r)',title=f'Radial distribution at {step*c["timestep_fs"]/1000:g} ps')
        ax.legend();fig.savefig(out/'plots/radial-distribution.png',dpi=170);plt.close(fig)
        write_metric_rows(rdf_rows,out,family='meam_md',name='radial-distribution')
    timing=[]
    for name,record in state.items():
        if record['state']=='complete':timing.append(dict(engine=name,duration_ps=record['duration_ps'],
            elapsed_seconds=record['elapsed_seconds'],steps_per_second=record['steps_per_second'],warmup_seconds=record['warmup_seconds']))
    if timing:write_metric_rows(timing,out,family='meam_md',name='trajectory-runtime')
    write_json(out/'technical/coverage.json',dict(state=state,complete=complete,common_checkpoints=len(differences)))
    lines=['# Matched CPU/GPU MEAM crystallization','',
        'One new 16,384-atom periodic Al liquid ancestor; 450 K, 200 ps per engine, 1 fs BAOAB steps and identical PCG64 thermostat noise. Float64 integration. No trained encoder or predictor.','',
        '| Engine | State | Simulated time (ps) |','| --- | --- | --- |']
    for name,record in state.items():lines.append(f'| {name} | {record["state"]} | {record.get("duration_ps",record.get("time_ps",0)):g} |')
    lines.extend(['','![Comparison](analyses/comparison-v1/plots/cpu-gpu-comparison.png)','',
        'The CPU arm uses original LAMMPS MEAM forces inside the shared external BAOAB driver. Native LAMMPS NVE throughput is measured separately in `analyses/profile-v1`; it excludes Python scatter/gather per step and uses native neighbor reuse. GPU force profiling compares eager and compiled evaluation on the same liquid cell. Production runtime includes integration, diagnostics and checkpoint IO, excludes setup/compilation and final archive conversion.','',
        'Coordinate divergence is measured from native float64 checkpoints, never from the quantized observations. Chaotic trajectory separation is expected; compare energy conservation, temperature, diffusion, pair structure and crystallization as well. A single common-noise pair cannot establish equivalent nucleation statistics. The largest solid cluster is a structural diagnostic, not a tracked nucleus-birth event.','',
        'Observations: exact 0.1 ps cadence, verified float16 positions and float32 boxes. Native states, momenta and RNG are retained every 1 ps and at early steps. See each run’s archive conversion receipt for measured quantization errors. Structural metrics use native coordinates.'])
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--gate',action='store_true')
    args=parser.parse_args();c=read(args.config)
    print(conservation_gate(c) if args.gate else report(c),flush=True)


if __name__=='__main__':main()
