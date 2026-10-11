"""Reuse exact original query identities, extending only their moving environment."""
import hashlib
import numpy as np
import torch
from ase import units
from src.data.fixed_cohort.dataset import read_release
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.project_runtime.paths import dataset_path, resolve_path
from .common import root, read, save, sha, write_json


def prepare(c):
    folder = root(c)/'technical'
    if (folder/'parents.json').exists():
        receipt = read(folder/'parents.json')
        if sha(folder/'parents.pt') != receipt['sha256']:
            raise ValueError('Prepared velocity parents changed')
        return torch.load(folder/'parents.pt', weights_only=False)
    if 'prepared_parents_sha256' in c:
        previous=resolve_path(c['prepared_parents_reference'])/'technical'
        receipt=read(previous/'parents.json')
        if (receipt['sha256']!=c['prepared_parents_sha256'] or
                sha(previous/'parents.pt')!=c['prepared_parents_sha256'] or
                receipt['fixed_identity']!=c['fixed_identity']):
            raise ValueError('Prepared periodic parent reference changed')
        states=torch.load(previous/'parents.pt',weights_only=False)
        if 'parent_indices' in c:states=[s for s in states if s['index'] in c['parent_indices']]
        counts=[sum(s['role']==r for s in states)for r in ['train','selection','test']]
        if counts!=c.get('parent_counts',[90,15,30]):raise ValueError('Reused source roles differ')
        if c.get('initial_velocity')=='saved_md':
            from .native import attach_native
            attach_native(c,states)
        save(folder/'parents.pt',states)
        write_json(folder/'parents.json',dict(sha256=sha(folder/'parents.pt'),parents=len(states),
            train=counts[0],selection=counts[1],test=counts[2],environment='full periodic parent',
            fixed_identity=c['fixed_identity'],producer=str(previous/'parents.pt'),
            producer_sha256=c['prepared_parents_sha256']))
        return states
    old = resolve_path(c['reference'])/'technical/parents.pt'
    if sha(old) != read(folder/'binding.json')['parents_sha256']:
        raise ValueError('Reference parents changed')
    states = torch.load(old, weights_only=False)
    if 'parent_indices' in c:
        selected=set(c['parent_indices'])
        states=[state for state in states if state['index'] in selected]
        if len(states)!=len(selected):raise ValueError('Unknown parent index in pilot cohort')
    _, release = read_release(c['fixed_release'])
    if release['identity'] != c['fixed_identity']:
        raise ValueError('Wrong fixed Al64 release')
    sources = {s['id']:s for s in release['sources']}
    for state in states:
        s = sources[state['source']]
        raw = ShootingBinaryTrajectory.load(dataset_path(s['dataset'])/s['relative_trajectory_path'])
        if sha(raw.root/'manifest.json') != state['source_manifest_sha256']:
            raise ValueError('Parent source manifest changed')
        frame = state['frame']
        if hashlib.sha256(np.ascontiguousarray(raw.positions[frame]).tobytes()).hexdigest() != state['source_frame_sha256']:
            raise ValueError('Parent frame changed')
        box = np.asarray(raw.box_high[frame]-raw.box_low[frame], np.float64)
        ids = np.asarray(raw.atom_ids); center = int(np.searchsorted(ids, state['center_atom_id']))
        q = np.asarray(raw.positions[frame], np.float64)-np.asarray(raw.positions[frame,center], np.float64)
        q -= box*np.round(q/box)
        distance = np.linalg.norm(q, axis=1); order = np.lexsort((ids,distance))
        rows = order
        if not np.array_equal(q[rows[:80]],state['q'][:80].numpy()):
            raise ValueError('Original nearest80 observation changed')
        state['q'] = torch.from_numpy(q[rows].copy())
        state['atom_rows'] = torch.from_numpy(rows.copy()); state['atom_ids'] = torch.from_numpy(ids[rows].copy())
        if state['index']%10==0:print('prepared parent',state['index'],'/135',flush=True)
    counts=[sum(s['role']==role for s in states) for role in ['train','selection','test']]
    if counts != c.get('parent_counts',[90,15,30]):
        raise ValueError('Unexpected ancestry roles')
    save(folder/'parents.pt',states)
    write_json(folder/'parents.json',dict(sha256=sha(folder/'parents.pt'),parents=len(states),
        train=counts[0],selection=counts[1],test=counts[2],environment='full periodic parent',fixed_identity=c['fixed_identity']))
    return states


def velocity(c, state, field):
    """Physical A/fs, sampled once in full-cell identity order; never per-shot."""
    if c.get('initial_velocity')=='saved_md':
        if field!=0 or c['velocity_fields']!=1:raise ValueError('Saved MD supplies exactly one velocity field per state')
        return state['v_A_fs'].clone()
    rng = np.random.default_rng(c['velocity_seed']+state['index']*100+field)
    v = rng.normal(size=(state['original_atoms'],3))*np.sqrt(units.kB*450/26.9815385)*units.fs
    v -= v.mean(0,keepdims=True)
    return torch.from_numpy(v[state['atom_rows'].numpy()].copy())


def initial(c, state, field, radius, device='cuda'):
    mask = torch.ones(len(state['q']),dtype=torch.bool) if radius is None else state['q'].norm(dim=-1) < radius
    dtype = getattr(torch,c.get('state_dtype','float64'))
    q = state['q'][mask].to(device=device,dtype=dtype)
    v = velocity(c,state,field)[mask].to(device=device,dtype=dtype)
    basis = torch.zeros(2,len(q),3,4,dtype=dtype,device=device)
    basis[0,:80,:,:2] = state['basis'].to(device=device,dtype=dtype)
    basis[1,:80,:,2:] = state['basis'].to(device=device,dtype=dtype)*c['velocity_scale_A_fs']
    return q,v,basis,state['atom_rows'][mask].to(device)
