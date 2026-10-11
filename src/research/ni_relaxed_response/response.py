"""Matched finite-amplitude responses of quenched endpoints of original MD."""
import math
import time
import shutil
import numpy as np
import torch
from ase import units
from src.simulation.eam_gpu.potential import FastNi
from .data import Source,read,root,resolve_path,sha,digest,write_json,atomic_save,native_packet
from .quench import virtual


def folder(c,job):return root(c)/'technical/physical-bank'/job['key']

def collect_md(c,p,job,deadline):
    out=folder(c,job);out.mkdir(parents=True,exist_ok=True)
    identity=digest([p['identity'],job,'original_MD_common_noise_signed_paths'])
    if (out/'md.json').exists():
        done=read(out/'md.json')
        if done['identity']!=identity or sha(out/'restart.pt')!=done['sha256']:raise ValueError('Changed signed MD bank')
        return torch.load(out/'restart.pt',weights_only=False,map_location='cpu')
    item=next(s for s in p['sources'] if s['key']==job['source_key']);source=Source(item['source'])
    q,v,box=source.frame(job['frame']);center=item['centers'][job['teacher_center']]
    _,neighbors=source.patches(job['frame'],[center]);neighbors=neighbors[0]
    seed=int(digest([c['seed'],job['key'],'teacher'])[:12],16);rng=np.random.default_rng(seed)
    basis=np.zeros((len(q),6,2),dtype=np.float64)
    for k,(start,scale) in enumerate([(0,c['teacher']['position_direction_rms_A']),(3,c['teacher']['velocity_direction_rms_A_ps'])]):
        d=rng.normal(size=(80,3));d-=d.mean(0);d*=scale/np.sqrt(np.mean(np.sum(d*d,-1)));basis[neighbors,start:start+3,k]=d
    eps=c['student_difference']['response_epsilon'];base=np.concatenate((q,v),-1)
    states=np.stack([base,base-eps*basis[...,0],base+eps*basis[...,0],base-eps*basis[...,1],base+eps*basis[...,1]])
    mass=58.71;shots=c['teacher']['shots'];n=len(q)
    x=torch.from_numpy(states[...,:3].copy()).cuda().flatten(1)[None].repeat(shots,1,1).flatten(0,1)
    mom=torch.from_numpy(states[...,3:].copy()).cuda().flatten(1)[None].repeat(shots,1,1).flatten(0,1)*mass/(1000*units.fs)
    potential=FastNi(resolve_path(c['potential_file']),n,box=box,dtype=torch.float32,expected_sha256=c['potential_sha256'])
    generators=[torch.Generator(device='cuda').manual_seed(seed+i+1) for i in range(shots)]
    dt=c['teacher']['timestep_fs']*units.fs;decay=math.exp(-c['teacher']['timestep_fs']/c['teacher']['friction_fs'])
    amplitude=math.sqrt(mass*units.kB*item['source']['temperature_K']*(1-decay**2))
    _,force=potential.force(x);begin=time.monotonic()
    for step in range(round(100/c['teacher']['timestep_fs'])):
        if time.time()>deadline-120:raise TimeoutError('Signed MD preparation continuation')
        mom+=.5*dt*force;x+=.5*dt*mom/mass
        noise=torch.stack([torch.randn(3*n,device='cuda',dtype=torch.float64,generator=g) for g in generators])
        mom=decay*mom+amplitude*noise[:,None,:].expand(-1,5,-1).reshape_as(mom)
        x+=.5*dt*mom/mass;_,force=potential.force(x);mom+=.5*dt*force
    future=torch.cat((x.reshape(shots,5,n,3),mom.reshape(shots,5,n,3)/mass*(1000*units.fs)),-1).cpu()
    if not torch.isfinite(future).all():raise FloatingPointError('Nonfinite original-state Ni continuation')
    result=dict(initial=torch.from_numpy(states),future=future,box=box,atom_ids=source.ids,neighbors=neighbors,center=center,source=item['key'],ancestor=item['source']['ancestry_group'],role=item['role'],job=job['key'],seed=seed)
    working=resolve_path(c['relaxation_scratch'])/'md'/job['key'];working.mkdir(parents=True,exist_ok=True)
    atomic_save(working/'restart.pt',result)
    shutil.copy2(working/'restart.pt',out/'restart.pt')
    if sha(working/'restart.pt')!=sha(out/'restart.pt'):raise IOError('Signed MD restart publication mismatch')
    write_json(out/'md.json',dict(identity=identity,sha256=sha(out/'restart.pt'),seconds=time.monotonic()-begin,state_dtype='float64',force_dtype='float32',noise='same per-shot noise for unperturbed and four signed paths',input='original unrelaxed MD Q,V',storage='float64 MD integration checkpoint for pre-quantization quenches; not a sampled trajectory'))
    return result


def collect(c,p,job,runner,deadline):
    out=folder(c,job)
    if (out/'complete.json').exists():
        done=read(out/'complete.json')
        if sha(out/'response.pt')!=done['sha256']:raise ValueError('Changed quenched response')
        return
    md=collect_md(c,p,job,deadline);base_task=next(t for t in p['native_tasks'] if t['source']==md['source'] and t['frame']==job['frame'])
    base=native_packet(c,base_task);slot=list(base['center_atom_ids']).index(md['center'])
    initial=[];future=[];receipts=[]
    for variant in range(5):
        if time.time()>deadline-750:raise TimeoutError('Quenched initial-state continuation')
        if variant in [0,3,4]:
            q=base['positions'][slot];v=md['initial'][variant,md['neighbors'],3:].numpy()
            initial.append(np.concatenate((q,v),-1));continue
        a,done=virtual(c,job['key']+f'-initial-{variant}',md['initial'][variant].numpy(),md['box'],md['atom_ids'],md['center'],md['neighbors'],runner,dict(job=job['key'],variant=variant,stage='original_MD_perturbed_initial'))
        initial.append(np.concatenate((a['positions'][0],a['velocities'][0]),-1));receipts.append(done)
    for shot in range(c['teacher']['shots']):
        values=[]
        for variant in range(5):
            if time.time()>deadline-750:raise TimeoutError('Quenched endpoint continuation')
            a,done=virtual(c,job['key']+f'-shot-{shot}-endpoint-{variant}',md['future'][shot,variant].numpy(),md['box'],md['atom_ids'],md['center'],md['neighbors'],runner,dict(job=job['key'],variant=variant,shot=shot,stage='original_MD_100fs_endpoint'))
            values.append(a['positions'][0]);receipts.append(done)
        future.append(values)
    packet=dict(state=torch.from_numpy(np.stack(initial)).float()*c['length_factor'],future=torch.from_numpy(np.asarray(future)).float()*c['length_factor'],source=md['source'],ancestor=md['ancestor'],role=md['role'],job=job['key'])
    atomic_save(out/'response.pt',packet);write_json(out/'complete.json',dict(sha256=sha(out/'response.pt'),finite_response_epsilon=c['student_difference']['response_epsilon'],quenches=receipts,exact_minimizer_derivative=False))
