"""Fixed-identity local observations of full-cell NVE continuations."""
from collections import Counter
from functools import lru_cache
import time
import numpy as np
import torch
from ase import units
from src.data.splits import train_test_role
from src.research.continuous_motion.data import Source, read, atomic_save
from src.experiment_runner.artifacts import file_hash as sha, json_digest as digest, write_json
from src.project_runtime.paths import resolve_path
from src.simulation.eam_gpu.potential import FastNi

FAMILY = 'ni_phase_sim'


def root(c): return resolve_path(c['output'])
def cache(c): return resolve_path(c['cache'])
def archive(c): return resolve_path(c['simulation_archive'])


def prepare(c):
    path = resolve_path(c['anchor_plan'])
    if sha(path) != c['anchor_plan_sha256']:
        raise ValueError('Changed pinned Ni anchor plan')
    parent = read(path)
    keys = set(parent['banks'][c['bank']] + parent['test'])
    jobs = [j for j in parent['jobs'] if j['key'] in keys]
    sources = [s for s in parent['sources'] if s['key'] in {j['source_key'] for j in jobs}]
    roles = {}
    for s in sources:
        role = train_test_role(s['original_role']); ancestor = s['source']['ancestry_group']
        if ancestor in roles and roles[ancestor] != role:
            raise ValueError('Ni ancestor crosses train/test')
        roles[ancestor] = role
    identities = [(j['source_key'], j['frame'], j['center']) for j in jobs]
    if len(set(identities)) != len(identities): raise ValueError('Duplicate physical input identity')
    plan = dict(config=c, jobs=jobs, sources=sources, parent_identity=parent['identity'],columns=parent['columns'],
        counts=dict(Counter(j['role'] for j in jobs)), ancestors=dict(Counter(roles.values())),
        protocol='New full-cell fixed-box NVE; not replay of saved NPT or reused BAOAB',
        input='Center-relative Q and V for fixed initial80 IDs, no extra conditions or history',
        output='Same80 identities, Q,V,acceleration and two initial-perturbation tangents every5fs')
    plan['identity'] = digest(plan)
    destination = root(c)/'technical/plan.json'
    if destination.exists() and read(destination) != plan: raise ValueError('Changed phase-space plan')
    write_json(destination, plan)
    return plan


@lru_cache(maxsize=4)
def source(plan_path, key):
    plan = read(plan_path)
    return Source(next(s['source'] for s in plan['sources'] if s['key'] == key))


def collect(c, p, job, deadline):
    destination = archive(c)/job['key']; marker = destination/'complete.json'
    identity = digest([p['identity'], job])
    if marker.exists():
        done = read(marker)
        if done['identity'] != identity or sha(destination/'path.pt') != done['sha256']:
            raise ValueError('Changed NVE path archive')
        return done
    reader = source(str(root(c)/'technical/plan.json'), job['source_key'])
    q, v, box = reader.frame(job['frame'])
    local, selected = reader.patches(job['frame'], [job['center']]); indices = selected[0]
    rng = np.random.default_rng(int(digest([c['seed'], job['key'], 'directions'])[:12], 16))
    basis = np.zeros((len(q), 6, 2))
    for k, (start, amplitude) in enumerate([(0, .01), (3, .1)]):
        value = rng.normal(size=(80, 3)); value -= value.mean(0)
        value *= amplitude / np.sqrt(np.mean(np.sum(value**2, -1)))
        basis[indices, start:start+3, k] = value
    x = torch.tensor(q, device='cuda', dtype=torch.float64)
    velocity = torch.tensor(v, device='cuda', dtype=torch.float64)
    dx = torch.tensor(basis[:, :3], device='cuda'); dv = torch.tensor(basis[:, 3:], device='cuda')
    index = torch.as_tensor(indices, device='cuda')
    # Keep the initial periodic image: no rank reselection or wrap discontinuity.
    image_offset = x[index]-x[index[:1]]-torch.tensor(local[0, :, :3], device='cuda')
    potential = FastNi(resolve_path(c['potential_file']), len(q), box=box, dtype=torch.float32,
                       expected_sha256=c['potential_sha256'])
    conversion = (1000*units.fs)**2/58.71  # F[eV/A] -> acceleration[A/ps^2].
    h = .001; states=[]; tangents=[]; accelerations=[]
    begin = time.monotonic(); force, dforce = potential.force_tangent(x, dx)
    e0 = float(potential.force(x)[0]) + float((velocity/(1000*units.fs)).square().sum())*58.71/2
    def observe():
        xyz = x[index]-x[index[:1]]-image_offset
        vel = velocity[index]-velocity[index[:1]]
        dxyz = dx[index]-dx[index[:1]]; dvel = dv[index]-dv[index[:1]]
        states.append(torch.cat((xyz,vel),-1).cpu())
        tangents.append(torch.cat((dxyz,dvel),-2).cpu())
        accelerations.append(((force[index]-force[index[:1]])*conversion).cpu())
    observe()
    for step in range(1, 101):
        if time.time() > deadline: raise TimeoutError('NVE teacher deadline')
        velocity += .5*h*conversion*force; dv += .5*h*conversion*dforce
        x += h*velocity; dx += h*dv
        force, dforce = potential.force_tangent(x, dx)
        velocity += .5*h*conversion*force; dv += .5*h*conversion*dforce
        if step % 5 == 0: observe()
    state = torch.stack(states); tangent = torch.stack(tangents)
    if not torch.isfinite(state).all() or not torch.isfinite(tangent).all():
        raise FloatingPointError('Nonfinite NVE observations: '+job['key'])
    e1 = float(potential.force(x)[0]) + float((velocity/(1000*units.fs)).square().sum())*58.71/2
    packet = dict(state=state, tangent=tangent, acceleration=torch.stack(accelerations),
        atom_ids=reader.ids[indices].copy(), times_fs=list(range(0,101,5)), job=job,
        ancestor=reader.source['ancestry_group'], source=reader.source['id'])
    scratch = resolve_path(c['simulation_scratch'])/job['key']; scratch.mkdir(parents=True,exist_ok=True)
    atomic_save(scratch/'path.pt', packet)
    destination.mkdir(parents=True,exist_ok=True)
    import shutil
    shutil.copy2(scratch/'path.pt', destination/'path.pt')
    checksum = sha(scratch/'path.pt')
    if sha(destination/'path.pt') != checksum: raise IOError('NVE publication checksum mismatch')
    done = dict(identity=identity, sha256=checksum, seconds=time.monotonic()-begin,
        energy_change_eV_per_atom=(e1-e0)/len(q), atoms=len(q),
        state_dtype='float64', tangent_dtype='float64', force_dtype='float32',
        integration='1fs velocity Verlet; fixed original box; no thermostat/barostat',
        storage='prequantization response observables; full trajectories are not stored',
        potential_sha256=c['potential_sha256'])
    write_json(scratch/'complete.json',done); write_json(marker,done)
    return done


def load(c, role, device='cuda'):
    p = read(root(c)/'technical/plan.json'); packets=[]
    for job in p['jobs']:
        if job['role'] != role: continue
        folder=archive(c)/job['key']; done=read(folder/'complete.json')
        if done['identity'] != digest([p['identity'],job]) or sha(folder/'path.pt') != done['sha256']:
            raise ValueError('Unsealed NVE teacher')
        packets.append(torch.load(folder/'path.pt',map_location='cpu',weights_only=False))
    state=torch.stack([v['state'] for v in packets]).float().to(device)
    tangent=torch.stack([v['tangent'] for v in packets]).float().to(device)
    scale=state.new_tensor([5.,5.,5.,10.,10.,10.])
    return p,packets,state/scale,tangent/scale[:,None]


def normalization(state,tangent):
    """Train-only ballistic residual scales at each lag; never use test values."""
    scales=[]; response=[]
    for lag in range(1,21):
        first=state[:,:-lag]; future=state[:,lag:]
        ballistic=first.clone(); ballistic[...,:3]+=lag*.005*2*first[...,3:]
        error=(future-ballistic).reshape(-1,80,2,3)
        scales.append(error.square().mean((0,1,3)).sqrt().clamp_min(state.new_tensor([.002,.02])))
        response.append(tangent[:,lag:].reshape(-1,80,2,3,2).square().mean((0,1,3)).clamp_min(1e-8))
    return dict(path_scale=torch.stack(scales), response_power=torch.stack(response))
