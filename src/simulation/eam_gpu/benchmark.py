"""Run numerical validation and hardware timings; not a production Ni campaign."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from ase.build import bulk
from lammps import lammps
from .potential import FastNi
from .dynamics import CapturedNVE
from src.experiment_runner.metric_docs import write_metric_rows, write_metric_table

ACCEL=9648.533215665327
KB=8.617333262145e-5


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def reference(q,box,file,*,melt=0,seed=26100711,temperature=2300):
    q=np.asarray(q,dtype=np.float64);n=len(q)
    low=np.zeros(3) if box is not None else q.min(0)-10
    high=np.asarray(box) if box is not None else q.max(0)+10
    l=lammps(cmdargs=['-log','none','-screen','none'])
    try:
        l.commands_string(f'''units metal
atom_style atomic
boundary {'p p p' if box is not None else 'f f f'}
region box block {low[0]} {high[0]} {low[1]} {high[1]} {low[2]} {high[2]}
create_box 1 box
mass 1 58.71
pair_style eam/fs
pair_coeff * * {Path(file).resolve()} Ni
neighbor 0.6 bin
neigh_modify every 1 delay 0 check yes
thermo_style custom step pe press vol
thermo 100000
''')
        l.create_atoms(n,atomid=np.arange(1,n+1,dtype=np.int32),atype=np.ones(n,dtype=np.int32),x=np.ascontiguousarray(q.flatten()))
        if l.get_natoms()!=n:raise ValueError('LAMMPS did not create every reference atom')
        if melt:
            l.commands_string(f'''velocity all create {temperature} {seed} mom yes rot no dist gaussian
fix integrate all nvt temp {temperature} {temperature} 0.1
timestep 0.001
run {melt}
unfix integrate
velocity all set 0 0 0
''')
        l.command('run 0');energy=l.get_thermo('pe');virial=l.get_thermo('press')*l.get_thermo('vol')*3/l.extract_global('nktv2p')
        forces=np.array(l.gather_atoms('f',1,3)[:n*3]).reshape(n,3)
        coordinates=np.array(l.gather_atoms('x',1,3)[:n*3]).reshape(n,3)
        return energy,forces,virial,coordinates
    finally:l.close()


def timing(call,repeats=100):
    for _ in range(5):call()
    torch.cuda.synchronize();samples=[]
    for _ in range(5):
        t=time.perf_counter()
        for _ in range(repeats):call()
        torch.cuda.synchronize();samples.append((time.perf_counter()-t)/repeats)
    return float(np.median(samples))


def engine(c,q,box,dtype):
    return FastNi(c['potential'],len(q),box=box,dtype=dtype,skin=c['skin_A'],expected_sha256=c['potential_sha256'])


def verlet(model,q,v,steps,dt,*,direction=None):
    alpha=ACCEL/model.data.mass
    if direction is None:e,f=model.force(q)
    else:
        dq=direction.clone();dv=torch.zeros_like(dq);f,h=model.force_tangent(q,dq)
    for _ in range(steps):
        v=v+(.5*dt*alpha)*f;q=q+dt*v
        if direction is None:e,f=model.force(q)
        else:
            dv=dv+(.5*dt*alpha)*h;dq=dq+dt*dv
            f,h=model.force_tangent(q,dq);dv=dv+(.5*dt*alpha)*h
        v=v+(.5*dt*alpha)*f
    return (q,v) if direction is None else (q,v,dq)


def numerical(c,out):
    rng=np.random.default_rng(c['seed']);a=bulk('Ni','fcc',a=3.60,cubic=True).repeat((7,7,7));box=a.cell.lengths()
    if len(a)!=c['validation_atoms'] or c['melt_temperature_K']!=2300:
        raise ValueError('This frozen reference preparation requires 1372 atoms and 2300 K')
    crystal=a.positions;distorted=(crystal+rng.normal(0,.12,crystal.shape))%box
    identity={k:c[k] for k in ['potential_sha256','melt_steps','seed']}
    receipt=out/'technical/preparation.json';states=out/'technical/validation_states.npz'
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        import hashlib
        if saved['identity']!=identity or hashlib.sha256(states.read_bytes()).hexdigest()!=saved['sha256']:
            raise ValueError('Changed numerical preparation')
        with np.load(states) as data:liquid=data['liquid']
    else:
        if states.exists():raise FileExistsError('Unrecorded preparation; inspect before reuse')
        print('Preparing 10 ps liquid reference with LAMMPS',flush=True)
        _,_,_,liquid=reference(distorted,box,c['potential'],melt=c['melt_steps'],seed=c['seed'],temperature=c['melt_temperature_K'])
        np.savez(states,crystal=crystal,distorted=distorted,liquid=liquid,box=box)
        import hashlib
        write(receipt,dict(identity=identity,sha256=hashlib.sha256(states.read_bytes()).hexdigest()))
    rows=[]
    for name,q,b in [('fcc',crystal,box),('distorted',distorted,box),('liquid',liquid,box),('open',distorted[:256],None)]:
        e0,f0,w0,_=reference(q,b,c['potential']);x=torch.tensor(q,device='cuda');models={}
        for dtype in (torch.float64,torch.float32):
            label=str(dtype).split('.')[-1];m=engine(c,q,b,dtype);models[dtype]=m;e,f,w,_=m.evaluate(x)
            row=dict(state=name,precision=label,atoms=len(q),energy_error_eV_per_atom=abs(float(e)-e0)/len(q),force_rms_eV_A=float(np.sqrt(np.mean((f.cpu().numpy()-f0)**2))),virial_error_eV_per_atom=abs(float(w)-w0)/len(q))
            for field,key in [('energy_error_eV_per_atom','energy_eV_per_atom'),('force_rms_eV_A','force_rms_eV_A'),('virial_error_eV_per_atom','virial_eV_per_atom')]:
                if row[field]>c['gates'][label+'_'+key]:raise ValueError(f'LAMMPS gate failed: {row}')
            rows.append(row);print(row,flush=True);write(out/'technical/validation.json',rows)
        dq=torch.tensor(rng.normal(size=(*q.shape,2)),device='cuda');dq/=dq.square().mean().sqrt()
        _,h=models[torch.float64].force_tangent(x,dq);_,hm=models[torch.float32].force_tangent(x,dq)
        eps=1e-5;fd=[]
        for k in range(2):
            fp=models[torch.float64].force(x+eps*dq[...,k])[1];fm=models[torch.float64].force(x-eps*dq[...,k])[1];fd.append((fp-fm)/(2*eps))
        fd=torch.stack(fd,-1);relative=float((h-fd).norm()/fd.norm());mixed=float((hm-h).norm()/h.norm())
        if max(relative,mixed)>c['gates']['hvp_relative']:raise ValueError(f'HVP gate failed {name}: FD={relative}, mixed={mixed}')
        write(out/'technical'/f'derivatives-{name}.json',dict(float64_finite_difference_relative=relative,float32_float64_relative=mixed,finite_difference_epsilon_A=eps))
        if name=='open':
            xb=x.flatten()[None].repeat(2,1);db=dq.reshape(1,-1,2).repeat(2,1,1)
            fb,hb=models[torch.float64].force_tangent(xb,db)
            if not torch.allclose(hb[0].reshape_as(h),h,atol=1e-8,rtol=1e-8):raise ValueError('Disconnected replica HVP mismatch')
    x=torch.tensor(liquid,device='cuda');m=engine(c,liquid,box,torch.float32)
    v=torch.tensor(rng.normal(size=liquid.shape)*np.sqrt(KB*1370*ACCEL/m.data.mass),device='cuda');v-=v.mean(0)
    initial=x.clone();vi=v.clone();energy=[];start=time.perf_counter();dt=c['timestep_fs']/1000
    for _ in range(c['nve_steps']//10):
        e=m.force(x)[0];energy.append(float(e+.5*m.data.mass/ACCEL*v.square().sum())/len(x));x,v=verlet(m,x,v,10,dt)
    energy.append(float(m.force(x)[0]+.5*m.data.mass/ACCEL*v.square().sum())/len(x))
    excursion=max(abs(np.array(energy)-energy[0]));nve=dict(steps=c['nve_steps'],dt_ps=dt,excursion_eV_per_atom=excursion,seconds=time.perf_counter()-start,neighbor_rebuilds=m.rebuilds)
    write(out/'technical/nve.json',nve)
    if excursion>c['gates']['nve_excursion_eV_per_atom']:raise ValueError(f'NVE gate failed: {nve}')
    direction=torch.tensor(rng.normal(size=(*initial.shape,1)),device='cuda');direction/=direction.square().mean().sqrt()
    exact=engine(c,liquid,box,torch.float64);_,_,response=verlet(exact,initial,vi,20,dt,direction=direction)
    eps=1e-5;qp,_=verlet(exact,initial+eps*direction[...,0],vi,20,dt);qm,_=verlet(exact,initial-eps*direction[...,0],vi,20,dt)
    finite=(qp-qm)/(2*eps);err=float((response[...,0]-finite).norm()/finite.norm())
    if err>.01:raise ValueError(f'Trajectory response mismatch: {err}')
    write(out/'technical/trajectory-response.json',dict(steps=20,dt_ps=dt,relative_error=err))
    captured=CapturedNVE(engine(c,liquid,box,torch.float32),initial,vi,dt_ps=dt,block_steps=10)
    eq=[]
    for _ in range(c['nve_steps']//10):
        eq.append(float(captured.total_energy())/len(initial));cq,cv=captured.run(10)
    eq.append(float(captured.total_energy())/len(initial))
    ce=max(abs(np.array(eq)-eq[0]));qrms=float((cq-x).square().mean().sqrt())
    write(out/'technical/captured-nve.json',dict(steps=c['nve_steps'],excursion_eV_per_atom=ce,ordinary_path_rms_A=qrms,rebuilds=int(captured.rebuild_count)))
    if ce>c['gates']['nve_excursion_eV_per_atom'] or qrms>0.01:
        raise ValueError(f'Captured NVE mismatch: excursion={ce}, position RMS={qrms}')
    return rows


def comparison(c,out):
    """Same hardware, force API and precision; different Ni/Al physics."""
    from src.research.meam_gpu.potential import FastAl
    from src.project_runtime.paths import resolve_path
    rng=np.random.default_rng(c['seed']);rows=[]
    files={k:resolve_path(v) for k,v in c['meam_comparison_files'].items()}
    for n in c['benchmark_atoms']:
        side=round((n/4)**(1/3));a=bulk('Al','fcc',a=4.05,cubic=True).repeat((side,side,side))
        q=(a.positions+rng.normal(0,.1,a.positions.shape))%a.cell.lengths();x=torch.tensor(q,device='cuda')
        dq=torch.tensor(rng.normal(size=(*q.shape,2)),device='cuda');dq/=dq.square().mean().sqrt()
        m=FastAl(files,n,box=a.cell.lengths(),dtype=torch.float32,skin=c['skin_A'])
        print(f'Compiling/warming FastAl force and two tangents, {n} atoms',flush=True)
        # Called before the captured objects are explicitly released below.
        force=timing(lambda:m.force(x),repeats=20)  # noqa: F821
        tangent=timing(lambda:m.force_tangent(x,dq),repeats=10)  # noqa: F821
        row=dict(atoms=n,material='Al',potential='Lee2003 2NN MEAM',force_seconds=force,force_two_tangents_seconds=tangent)
        rows.append(row);print(row,flush=True);write(out/'technical/fastal-comparison.json',rows)
        del m
    return rows


def export(out):
    """Publish recorded engineering results with their calculation contract."""
    for source,name in [('validation','lammps-validation'),('timing','timing'),('fastal-comparison','fastal-comparison')]:
        path=out/f'technical/{source}.json'
        if path.exists():write_metric_rows(json.loads(path.read_text()),out,family='ni_eam_gpu',name=name)
    paths=list((out/'technical').glob('derivatives-*.json'))
    paths += [out/f'technical/{name}.json' for name in ['nve','captured-nve','trajectory-response']]
    metrics={p.stem:json.loads(p.read_text()) for p in paths if p.exists()}
    write_metric_table(metrics,out,family='ni_eam_gpu',name='dynamics-and-derivatives')


def performance(c,out):
    rng=np.random.default_rng(c['seed']);rows=[]
    for n in c['benchmark_atoms']:
        side=round((n/4)**(1/3));a=bulk('Ni','fcc',a=3.60,cubic=True).repeat((side,side,side));assert len(a)==n
        q=(a.positions+rng.normal(0,.1,a.positions.shape))%a.cell.lengths();x=torch.tensor(q,device='cuda');m=engine(c,q,a.cell.lengths(),torch.float32)
        seconds=timing(lambda:m.force(x));b=m._buffers(m.support(x),0)
        kernels=timing(lambda:b['graph'].replay(),repeats=500)
        dq=torch.tensor(rng.normal(size=(*q.shape,2)),device=x.device);dq/=dq.square().mean().sqrt()
        tangent=timing(lambda:m.force_tangent(x,dq),repeats=50)
        v=torch.tensor(rng.normal(size=q.shape)*np.sqrt(KB*1370*ACCEL/m.data.mass),device='cuda');v-=v.mean(0)
        torch.cuda.synchronize();start=time.perf_counter();verlet(m,x,v,c['benchmark_steps'],c['timestep_fs']/1000);torch.cuda.synchronize();step=(time.perf_counter()-start)/c['benchmark_steps']
        # Exercise construction from a valid reused list whose actual reference
        # differs from q. This must retain the list's original displacement origin.
        m.support(x)
        shifted=x+.05
        captured=CapturedNVE(m,shifted,v,dt_ps=c['timestep_fs']/1000,block_steps=100)
        if not torch.equal(captured.reference,m.reference):raise ValueError('Lost neighbor-list build origin')
        captured.run(100);torch.cuda.synchronize();start=time.perf_counter();captured.run(c['benchmark_steps']);torch.cuda.synchronize()
        graphstep=(time.perf_counter()-start)/c['benchmark_steps']
        row=dict(atoms=n,captured_verlet_step_seconds=graphstep,captured_ns_per_day=(c['timestep_fs']*1e-6)*86400/graphstep,force_seconds=seconds,kernel_graph_seconds=kernels,force_two_tangents_seconds=tangent,verlet_step_seconds=step,ns_per_day=(c['timestep_fs']*1e-6)*86400/step,neighbor_rebuilds=m.rebuilds,captured_neighbor_rebuilds=int(captured.rebuild_count))
        rows.append(row);print(row,flush=True);write(out/'technical/timing.json',rows)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--stage',choices=['all','numerical','performance','comparison','export'],default='all');a=p.parse_args()
    c=json.loads(Path(a.config).read_text());out=Path(c['output']);(out/'technical').mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    if a.stage=='export':export(out);return
    if (c['state_dtype'],c['force_dtype'],c['energy_reduction_dtype'])!=('float64','float32','float64'):
        raise ValueError('This assay requires float64 states/reductions and float32 production forces')
    if c['nve_steps']%10 or c['benchmark_steps']%100:raise ValueError('Assay lengths must cover whole observation/graph blocks')
    write(out/'technical/hardware.json',dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,job=os.environ.get('SLURM_JOB_ID'),host=os.uname().nodename))
    if a.stage in ['all','numerical']:numerical(c,out)
    if a.stage in ['all','performance']:performance(c,out)
    if a.stage in ['all','comparison']:comparison(c,out)
    if a.stage=='all':write(out/'technical/complete.json',dict(state='complete',config=c))


if __name__=='__main__':main()
