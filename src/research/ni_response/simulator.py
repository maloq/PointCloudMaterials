"""Ni EAM full-cell BAOAB continuations and analytic local tangent observations.

Only the chosen local80 initial Q or V is perturbed. The rest of the full cell
evolves normally. Paths and tangents are native float64 response observables;
they are never reconstructed from the 0.1 ps saved trajectory.
"""
import math
import shutil
import time
import numpy as np
import torch
from ase import units
from src.simulation.eam_gpu.potential import FastNi
from .data import Source, read, root, cache, digest, sha, write_json, atomic_save, resolve_path


def archive(c, job):
    return resolve_path(c['simulation_archive'])/job['key']


def collect(c, plan, job, deadline):
    final = archive(c, job)
    identity = digest([plan['identity'], job, c['teacher']])
    if (final/'complete.json').exists():
        done = read(final/'complete.json')
        if done['identity'] != identity or sha(final/'response.pt') != done['sha256']:
            raise ValueError('Changed simulator response bank')
        return done
    folder = resolve_path(c['simulation_scratch'])/job['key']
    folder.mkdir(parents=True, exist_ok=True)
    item = next(s for s in plan['sources'] if s['key']==job['source_key'])
    source = Source(item['source'])
    q, v, box = source.frame(job['frame'])
    local, indices = source.patches(job['frame'], [item['centers'][job['teacher_center']]])
    indices = indices[0]
    settings = c['teacher']
    if settings['state_dtype']!='float64' or settings['force_dtype']!='float32':
        raise ValueError('New response integration requires f64 states and f32 forces')
    seed = int(digest([c['seed'], job['key'], 'teacher'])[:12], 16)
    direction_rng = np.random.default_rng(seed)
    local_basis = np.zeros((80,6,2), dtype=np.float64)
    for direction, (start, scale) in enumerate([(0,settings['position_direction_rms_A']),
                                               (3,settings['velocity_direction_rms_A_ps'])]):
        value = direction_rng.normal(size=(80,3))
        value -= value.mean(0)
        value *= scale / np.sqrt(np.mean(np.sum(value**2, axis=-1)))
        local_basis[:,start:start+3,direction] = value
    dq = np.zeros((len(q),3,2)); dv = np.zeros_like(dq)
    dq[indices] = local_basis[:,:3]; dv[indices] = local_basis[:,3:]
    # Input geometry is centered, so its input perturbation is centered too.
    local_basis[:,:3] -= local_basis[:1,:3]
    shots = settings['shots']; mass = 58.71
    x = torch.tensor(q,device='cuda',dtype=torch.float64).flatten()[None].repeat(shots,1)
    momentum = torch.tensor(v,device='cuda',dtype=torch.float64).flatten()[None].repeat(shots,1)*mass/(1000*units.fs)
    dx = torch.tensor(dq,device='cuda',dtype=torch.float64).reshape(1,-1,2).repeat(shots,1,1)
    dp = torch.tensor(dv,device='cuda',dtype=torch.float64).reshape(1,-1,2).repeat(shots,1,1)*mass/(1000*units.fs)
    files = {'Ni':resolve_path(c['potential_file'])}
    potential = FastNi(files['Ni'],len(q),box=torch.tensor(box,device='cuda',dtype=torch.float64),dtype=torch.float32,expected_sha256=c['potential_sha256'])
    generators = [torch.Generator(device='cuda').manual_seed(seed+i+1) for i in range(shots)]
    dt = settings['timestep_fs']*units.fs
    decay = math.exp(-settings['timestep_fs']/settings['friction_fs'])
    amplitude = math.sqrt(mass*units.kB*item['source']['temperature_K']*(1-decay**2))
    index = torch.as_tensor(indices,device='cuda'); box_t = torch.as_tensor(box,device='cuda')
    states, tangents = [], []; begin = time.monotonic()
    force, tangent = potential.force_tangent(x,dx)
    for step in range(1,max(settings['save_fs'])//settings['timestep_fs']+1):
        if time.time() >= deadline:
            raise TimeoutError('Continuous-motion teacher budget exhausted')
        momentum += .5*dt*force; dp += .5*dt*tangent
        x += .5*dt*momentum/mass; dx += .5*dt*dp/mass
        noise = torch.stack([torch.randn(x.shape[1],generator=g,device='cuda',dtype=torch.float64) for g in generators])
        momentum = decay*momentum+amplitude*noise; dp *= decay
        x += .5*dt*momentum/mass; dx += .5*dt*dp/mass
        force, tangent = potential.force_tangent(x,dx)
        momentum += .5*dt*force; dp += .5*dt*tangent
        if step*settings['timestep_fs'] in settings['save_fs']:
            xyz = x.reshape(shots,-1,3)[:,index]
            xyz = xyz-xyz[:,:1]; xyz -= box_t*torch.round(xyz/box_t)
            vel = momentum.reshape(shots,-1,3)[:,index]/mass*(1000*units.fs)
            dxyz = dx.reshape(shots,-1,3,2)[:,index]
            dxyz = dxyz-dxyz[:,:1]
            dvel = dp.reshape(shots,-1,3,2)[:,index]/mass*(1000*units.fs)
            states.append(torch.cat((xyz,vel),-1).cpu())
            tangents.append(torch.cat((dxyz,dvel),-2).cpu())
    states, tangents = torch.stack(states,1), torch.stack(tangents,1)
    if not torch.isfinite(states).all() or not torch.isfinite(tangents).all():
        raise FloatingPointError(f'Nonfinite simulator continuation: {job["key"]}')
    elapsed = time.monotonic()-begin
    atomic_save(folder/'response.pt',dict(state=torch.from_numpy(local[0]),basis=torch.from_numpy(local_basis),
        future=states,tangent=tangents,source=item['source']['id'],ancestor=item['source']['ancestry_group'],
        role=item['role'],job=job['key'],center=item['centers'][job['teacher_center']],horizons_fs=settings['save_fs']))
    done = dict(identity=identity,sha256=sha(folder/'response.pt'),seconds=elapsed,seed=seed,
                state_dtype='float64',force_dtype='float32',noise='independent shots; shared noise for each tangent',
                potential_files={k:sha(v) for k,v in files.items()},
                protocol='full-cell fixed-box BAOAB, actual saved Q,V, source T, no relaxation',
                storage='cropped prequantization float64 response observables; no full trajectory archive')
    write_json(folder/'complete.json',done)
    final.mkdir(parents=True,exist_ok=True)
    for filename in ('response.pt','complete.json'):
        shutil.copy2(folder/filename,final/filename)
        if sha(final/filename)!=sha(folder/filename):
            raise IOError('Simulator archive checksum mismatch')
    # Keep scratch records as well until normal storage maintenance.
    return done


def load(c, plan):
    records=[]
    for job in plan['jobs']:
        folder=archive(c,job);done=read(folder/'complete.json')
        if done['identity']!=digest([plan['identity'],job,c['teacher']]) or sha(folder/'response.pt')!=done['sha256']:
            raise ValueError('Unsealed simulator teacher')
        records.append(torch.load(folder/'response.pt',weights_only=False))
    return records
