"""Fixed-Al64 parents, exact view membership, and independent structure assays."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from src.data.fixed_cohort.dataset import read_release
from src.data.fixed_cohort.protocol import centered, digest, sha, write_json
from src.project_runtime.paths import resolve_path
from src.research.crystallization_origin.extract import raw_source, frame_geometry
from src.research.liquid_predictability.descriptors import _orders, _topology, cna_packet


def load(path):
    c = json.loads(Path(path).read_text())
    if c['protocol'] != 'spatial_vicreg_bias_al64_v1':
        raise ValueError('Wrong spatial VICReg protocol')
    for key in ('output', 'cache', 'fixed_release', 'ptm_audit', 'encoder_recipe', 'feature_cache'):
        c[key] = str(resolve_path(c[key]).resolve())
    return c


def descriptors(points):
    """Every supplied atom is consumed; no hidden radius-8 truncation."""
    x = np.asarray(points, dtype=np.float64)
    if x.shape != (80, 3) or not np.isfinite(x).all() or np.any(x[0]):
        raise ValueError('Descriptor input must be the exact centered 80-atom patch')
    distance = cdist(x, x)
    if np.any(distance[np.triu_indices(80, 1)] <= 1e-8):
        raise ValueError('Duplicate atoms in descriptor patch')
    r = np.linalg.norm(x, axis=1)[1:]
    if np.any(np.diff(r) < -1e-5):
        raise ValueError('Expected distance-ordered center neighbors')
    values = [12/(4*np.pi*r[11]**3/3), 79/(4*np.pi*r[-1]**3/3),
              *np.quantile(r, [0, .1, .25, .5, .75, .9, 1]), *r[:12]]
    names = ['density12', 'density80'] + [f'r_q{i}' for i in (0,10,25,50,75,90,100)]
    names += [f'neighbor_distance_{i}' for i in range(1, 13)]
    radial = np.exp(-.5*((r[:, None]-np.linspace(.5, 8., 24))/.3)**2).sum(0)/80
    values.extend(radial); names += [f'radial_{i}' for i in range(24)]
    for n in (12, 24):
        u = x[1:n+1] / r[:n, None]
        angles = (u @ u.T)[np.triu_indices(n, 1)]
        values.extend(np.histogram(angles, np.linspace(-1.000001,1.000001,13))[0]/len(angles))
        names += [f'angle_{n}_{i}' for i in range(12)]
    order, onames = _orders(x, distance)
    tda, tnames = _topology(x)
    cna, cnames = [], []
    for label, cutoff in (('fixed32',3.2),('fixed36',3.6),('adaptive12',r[:12].mean()*(1+np.sqrt(2))/2)):
        cna.extend(cna_packet(distance, cutoff))
        cnames += [f'{label}_{s}' for s in ('coordination','421','422','444','666','555','544','433','other',
                                         'common_mean','bonds_mean','chain_mean','common_second','bonds_second','chain_second')]
    blocks = [('geometry', values, names), ('bond_order', order, onames),
              ('cna', cna, cnames), ('tda', tda, tnames)]
    y = np.concatenate([np.asarray(v) for _, v, _ in blocks]).astype(np.float32)
    labels = [group+'/'+name for group, _, ns in blocks for name in ns]
    if not np.isfinite(y).all():
        raise FloatingPointError('Nonfinite independent physical descriptors')
    return y, labels


def freeze(config):
    c = load(config)
    fixed, original = read_release(c['fixed_release'])
    if original['identity'] != c['fixed_identity']:
        raise ValueError('Fixed source/sample contract changed')
    cache = Path(c['cache']); cache.mkdir(parents=True, exist_ok=True)
    binding = dict(protocol=c['protocol'], fixed_identity=original['identity'],
                   population_sha256=sha(fixed/'benchmark/population.npz'),
                   geometry=c['geometry'], assay=c['assay'], sources=original['sources'],
                   structural_frames=original['structural']['frames'],
                   potential_sha256=original['potential_sha256'],
                   producer_sha256=sha(Path(__file__)),
                   descriptor_producer_sha256=sha(Path(__file__).parents[1]/'liquid_predictability/descriptors.py'))
    binding['identity'] = digest(binding)
    path = cache/'plan.json'
    with (cache/'plan.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if json.loads(path.read_text()) != binding:
                raise ValueError('View-cache definition changed; use a fresh cache')
        else:
            write_json(path, binding)
    return c, binding


def source(config, sid):
    c, plan = freeze(config)
    item = next(s for s in plan['sources'] if s['id'] == sid)
    folder = Path(c['cache'])/'sources'/str(sid); folder.mkdir(parents=True, exist_ok=True)
    receipt = folder/'complete.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['identity'] != plan['identity'] or any(sha(folder/k) != v for k,v in saved['files'].items()):
            raise ValueError(f'Changed prepared source {sid}')
        return saved
    started = time.monotonic(); raw = raw_source(item)
    rows = np.searchsorted(raw.atom_ids, item['center_atom_ids'])
    if not np.array_equal(raw.atom_ids[rows], item['center_atom_ids']):
        raise ValueError('Fixed anchor identity changed')
    fixed = Path(c['fixed_release'])
    with np.load(fixed/'benchmark/population.npz') as a:
        pop = {k:a[k] for k in ('source','frame','atom','sample_id')}
    original = np.flatnonzero(pop['source'] == sid)
    bench_map = {(int(pop['frame'][j]), int(pop['atom'][j])):int(j) for j in original}
    assay_frames = sorted(set(c['assay']['frames']) | set(pop['frame'][original].tolist()))
    training_frames = plan['structural_frames'] if item['role'] == 'train' else []
    frames = sorted(set(training_frames + assay_frames))
    original_neighbors = {}
    for section in ('benchmark','structural'):
        directory = fixed/section/'sources'/str(sid)
        if section == 'structural' and item['role'] not in ('train','selection'):
            continue
        f = np.load(directory/'frames.npy')
        ids = np.load(directory/'neighbor_ids.npy', mmap_mode='r')
        for j, frame in enumerate(f):
            original_neighbors[int(frame)] = ids[j]
    ptm_root = Path(c['ptm_audit'])/'technical/sources'/str(sid)
    ptm_receipt = json.loads((ptm_root/'ptm-complete.json').read_text())
    if ptm_receipt['source_manifest_sha256'] != item['manifest_sha256']:
        raise ValueError('Physical reference ancestry changed')
    chunk_start = -1; labels = None; checked_chunks = {}
    train_parents, train_views = [], []
    collected = {k:[] for k in ('parents','view_indices','pair_choice','targets','ptm','solid_fraction',
        'visible_A','visible_B','visible_union','visible_augmentation_union','support_fraction','support_fraction_B',
        'crystal_present','distance','progress','null_fields','null_exposure','benchmark_row','frame','atom','source',
        'uniform','coords','box','center_separation','overlap','parent_radius','view_radius_A','view_radius_B')}
    descriptor_names = None
    for frame in frames:
        points, box = frame_geometry(raw, frame); tree = cKDTree(points, boxsize=box)
        candidate = tree.query(points[rows], k=128, workers=1)[1]
        if frame in original_neighbors:
            anchor = np.searchsorted(raw.atom_ids, original_neighbors[frame])
        else:
            anchor = tree.query(points[rows], k=80, workers=1)[1]
        if not np.array_equal(anchor[:,0], rows):
            raise ValueError('Parent centers disagree with fixed anchors')
        indices = np.stack([np.r_[a, p[~np.isin(p,a)][:48]] for a,p in zip(anchor,candidate)])
        if indices.shape != (64,128):
            raise ValueError('Nearest-128 pool does not cover all preserved nearest-80 atoms')
        parent = centered(points, box, rows, indices)
        dist = np.sum((parent[:,None] - parent[:,1:9,None])**2, axis=-1)
        views = np.argsort(dist, axis=-1, kind='stable')[:,:,:80].astype(np.uint8)
        if not np.array_equal(views[:,:,0], np.broadcast_to(np.arange(1,9),(64,8))):
            raise ValueError('Neighbor-view center identity changed')
        if frame in training_frames:
            train_parents.append(parent); train_views.append(views)
        if frame not in assay_frames:
            continue
        begin = frame//32*32; stop = min(801,begin+32)
        if begin != chunk_start:
            name = f'ptm-{begin:04d}-{stop:04d}.npz'
            if sha(ptm_root/name) != ptm_receipt['files'][name]:
                raise ValueError(f'Changed PTM chunk: {sid}/{name}')
            checked_chunks[name] = ptm_receipt['files'][name]
            with np.load(ptm_root/name) as a: labels = a['labels']
            chunk_start = begin
        ptm = labels[frame-begin]; solid = np.isin(ptm,[1,2,3])
        near = tree.query(points, k=15, workers=1)[1][:,1:]
        fraction = solid[near].mean(1)
        core = solid & (fraction >= .8); liquid = ~solid & (fraction <= .1)
        distance = np.full(64,np.inf)
        if solid.any(): distance = cKDTree(points[solid],boxsize=box).query(points[rows])[0]
        progress = np.full(64,np.nan)
        if core.any() and liquid.any():
            progress = (cKDTree(points[core],boxsize=box).query(points[rows])[0]
                        - cKDTree(points[liquid],boxsize=box).query(points[rows])[0])/2
        diffuse = solid.astype(np.float32); exposed = solid.copy()
        null_fields = [diffuse[anchor].mean(1)]; null_exposure = [exposed[anchor].any(1)]
        for step in range(1,5):
            diffuse = .5*diffuse + .5*diffuse[near].mean(1)
            exposed = exposed | exposed[near].any(1)
            if step in (1,2,4):
                null_fields.append(diffuse[anchor].mean(1));null_exposure.append(exposed[anchor].any(1))
        choice = np.random.default_rng(np.random.SeedSequence([c['assay']['seed'],sid,frame])).integers(0,8,64)
        bidx = views[np.arange(64),choice]; b = np.take_along_axis(indices,bidx,axis=1)
        union_mask = np.zeros((64,128),bool); union_mask[:,:80]=True
        for j in range(64): union_mask[j,views[j].ravel()]=True
        target = []
        for x in parent[:,:80]:
            y,names = descriptors(x);target.append(y)
            if descriptor_names is None: descriptor_names=names
            if names != descriptor_names: raise ValueError('Descriptor columns changed')
        shift = parent[np.arange(64),choice+1]
        xB = np.take_along_axis(parent,bidx[:,:,None],axis=1)-shift[:,None]
        value = dict(parents=parent,view_indices=views,pair_choice=choice.astype(np.uint8),targets=np.stack(target),
            ptm=ptm[rows],solid_fraction=fraction[rows],visible_A=solid[anchor].any(1),visible_B=solid[b].any(1),
            visible_union=solid[anchor].any(1)|solid[b].any(1),
            visible_augmentation_union=(solid[indices]&union_mask).any(1),support_fraction=solid[anchor].mean(1),
            support_fraction_B=solid[b].mean(1),crystal_present=np.full(64,solid.any()),distance=distance,
            progress=progress,null_fields=np.stack(null_fields,1),null_exposure=np.stack(null_exposure,1),
            benchmark_row=np.asarray([bench_map.get((frame,int(a)),-1) for a in raw.atom_ids[rows]]),
            frame=np.full(64,frame,np.int32),atom=raw.atom_ids[rows],source=np.full(64,sid,np.int32),
            uniform=np.full(64,frame in c['assay']['frames']),coords=points[rows].astype(np.float32),
            box=np.broadcast_to(box,(64,3)).astype(np.float32),center_separation=np.linalg.norm(shift,axis=1),
            overlap=(bidx<80).sum(1)/80,parent_radius=np.linalg.norm(parent,axis=-1).max(1),
            view_radius_A=np.linalg.norm(parent[:,:80],axis=-1).max(1),view_radius_B=np.linalg.norm(xB,axis=-1).max(1))
        for k,v in value.items():collected[k].append(v)
    if item['role']=='train':
        np.save(folder/'train_parents.npy',np.concatenate(train_parents))
        np.save(folder/'train_views.npy',np.concatenate(train_views))
    values={k:np.concatenate(v) for k,v in collected.items()}
    values['role']=np.full(len(values['source']),item['role'])
    if sorted(values['benchmark_row'][values['benchmark_row']>=0]) != original.tolist():
        raise ValueError('Fixed benchmark rows were dropped or duplicated')
    np.savez(folder/'assay.npz',**values)
    write_json(folder/'descriptors.json',descriptor_names)
    files={p.name:sha(p) for p in folder.iterdir() if p.suffix in ('.npy','.npz','.json') and p.name!='complete.json'}
    result=dict(identity=plan['identity'],source=sid,role=item['role'],files=files,
        training_rows=len(training_frames)*64,assay_rows=len(values['source']),benchmark_rows=len(original),
        ptm_chunks=checked_chunks,seconds=time.monotonic()-started)
    write_json(receipt,result);print(json.dumps({k:v for k,v in result.items() if k not in ('files','ptm_chunks')}),flush=True)
    return result


def seal(config):
    c,plan=freeze(config);root=Path(c['cache']);records=[]
    for item in plan['sources']:
        folder=root/'sources'/str(item['id']);r=json.loads((folder/'complete.json').read_text())
        if r['identity']!=plan['identity'] or any(sha(folder/k)!=v for k,v in r['files'].items()):
            raise ValueError(f'Changed source before sealing {item["id"]}')
        records.append(r)
    arrays={};names=None
    for r in records:
        folder=root/'sources'/str(r['source'])
        columns=json.loads((folder/'descriptors.json').read_text())
        if names is None:names=columns
        if names!=columns:raise ValueError('Descriptor column mismatch')
        with np.load(folder/'assay.npz') as a:
            for k in a.files:arrays.setdefault(k,[]).append(a[k])
    dest=root/'assay';dest.mkdir(exist_ok=True)
    for k,v in arrays.items():np.save(dest/(k+'.npy'),np.concatenate(v))
    write_json(dest/'descriptors.json',names)
    n=sum(r['training_rows'] for r in records)
    if n!=1157760:raise ValueError(f'Incomplete fixed structural population: {n}')
    for key,shape,dtype in [('parents',(n,128,3),'float32'),('views',(n,8,80),'uint8')]:
        out=np.lib.format.open_memmap(root/f'train_{key}.npy',mode='w+',dtype=dtype,shape=shape);offset=0
        for r in records:
            if r['role']!='train':continue
            a=np.load(root/'sources'/str(r['source'])/f'train_{key}.npy',mmap_mode='r')
            out[offset:offset+len(a)]=a;offset+=len(a)
        out.flush();del out
    benchmark=np.load(dest/'benchmark_row.npy');keep=benchmark>=0
    fixed=Path(c['fixed_release'])
    with np.load(fixed/'benchmark/population.npz') as a:count=len(a['source'])
    if not np.array_equal(np.sort(benchmark[keep]),np.arange(count)):
        raise ValueError('All64 population coverage is not exact')
    np.save(dest/'all64_order.npy',np.flatnonzero(keep)[np.argsort(benchmark[keep])])
    legacy=np.load(fixed/'benchmark/legacy_order.npy');np.save(dest/'legacy16_order.npy',np.load(dest/'all64_order.npy')[legacy])
    files={str(p.relative_to(root)):sha(p) for p in list(dest.iterdir())+[root/'train_parents.npy',root/'train_views.npy']}
    write_json(root/'manifest.json',dict(state='complete',identity=plan['identity'],fixed_identity=plan['fixed_identity'],
        training_rows=n,assay_rows=len(benchmark),benchmark_rows=count,sources=records,files=files))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['freeze','source','seal'])
    p.add_argument('--config',required=True);p.add_argument('--source',type=int);a=p.parse_args()
    if a.stage=='source':source(a.config,a.source)
    elif a.stage=='seal':seal(a.config)
    else:freeze(a.config)
