"""Reuse immutable broad spatial rows and exact observed MD futures."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
import torch
from src.data.fixed_cohort.protocol import digest,sha,write_json
from src.project_runtime.paths import resolve_path
from src.research.liquid_predictability.rich_packed import PackedPatches
from src.research.liquid_predictability.rich_multimaterial_data import CachedPatches
from src.research.liquid_predictability.rich_objectives import descriptor_blocks
from src.research.continuous_motion.data import Source,atomic_save

FAMILY='structural_response'
def read(p):return json.loads(Path(p).read_text())
def root(c):return resolve_path(c['output'])
def cache(c):return resolve_path(c['derived_cache'])


def spatial_target_names(c):
    from src.research.liquid_predictability.descriptors import _order_names,_topology_names,_cna_names
    spec=c.get('descriptors',{}).get('selection')
    if spec:
        from src.postprocessing.descriptors.selection import load
        return load(spec)
    names=[f'bond_order/{n}' for n in _order_names()]
    if 'cna' in c.get('descriptors',{}).get('families',[]):names.extend(f'cna/{n}' for n in _cna_names())
    return names+[f'tda/{n}' for n in _topology_names()]


def data_identity(c):
    binding={k:c[k] for k in ('fixed_dataset','cache','structural_dataset','data','loader',
        'prepared_identity','seed','response_campaign','response_identity','horizons_ps')}
    if 'descriptors' in c:binding['descriptors']=c['descriptors']
    return digest(binding)


def spatial_data(c):
    data=PackedPatches(c)
    expected=spatial_target_names(c)
    columns=np.array([v['name'] in expected for v in data.columns])
    ids=np.flatnonzero(columns)
    if [data.columns[i]['name'] for i in ids]!=expected:
        raise ValueError('Changed spatial target schema; bind compact columns with descriptors.selection')
    weights=np.zeros(len(data.columns),np.float32);blocks=descriptor_blocks(data.columns)
    selected_families={data.columns[i]['family'] for i in ids}
    topology=[k for k,v in blocks.items() if k.startswith('tda_') and columns[v].any()]
    for name,indices in blocks.items():
        if name not in selected_families and name not in topology:continue
        active=np.asarray(indices)[data.active[indices]&columns[indices]]
        if not len(active):raise ValueError('Empty descriptor block: '+name)
        weights[active]=1/(len(selected_families)*len(active)*(len(topology) if name in topology else 1))
    return data,ids,weights[ids]


def future_packet(args):
    c,old,job,item=args;folder=cache(c)/'futures';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(job['key']+'.pt');receipt=path.with_suffix('.json')
    identity=digest(dict(plan=old['identity'],job=job))
    if receipt.exists():
        d=read(receipt)
        if d['identity']!=identity or sha(path)!=d['sha256']:raise ValueError('Changed future packet')
        return job['key']
    source=Source(item['source']);future=[]
    for frame in job['future_frames']:
        qv,_=source.patches(frame,item['centers']);future.append(qv[...,:3].astype(np.float32))
    atomic_save(path,torch.from_numpy(np.stack(future,1)))
    write_json(receipt,dict(identity=identity,sha256=sha(path)));return job['key']


def prepare(c):
    dest=cache(c);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():
        done=read(dest/'complete.json')
        if done['data_identity']!=data_identity(c) or any(sha(dest/f)!=h for f,h in done['files'].items()):raise ValueError('Changed prepared collection')
        return done
    data,ids,weights=spatial_data(c)
    held=CachedPatches(c,'selection');held.use_transform(data)
    # Fixed validation observations; no event or phase selection.
    selected=np.sort(np.random.default_rng(c['seed']+71).choice(len(held),4096,replace=False))
    val=held.batch(selected);np.savez(dest/'spatial-selection.npz',positions=val['positions'],target=val['target'][:,ids],rows=selected)
    packed=read(resolve_path(c['loader']['packed_cache'])/'manifest.json')
    fixed=read(resolve_path(c['fixed_dataset']['root'])/'splits.json');roles={s['lineage']:s['role'] for s in fixed['sources']}
    for s in packed['shards']:
        t=s['task']
        if t['role']!='train' or (t['lineage'] in roles and roles[t['lineage']]!='train'):raise ValueError('Held-out ancestry in spatial fit')
    train_anc={s['task']['lineage'] for s in packed['shards']};validation_anc={s['task']['lineage'] for s in held.shards}
    if train_anc&validation_anc:raise ValueError('Spatial ancestry overlap')
    metadata=dict(packed_identity=data.packed_identity,rows=len(data),material_rows=packed['material_rows'],
        sources=packed['source_count'],ancestry_groups=len(train_anc),descriptor_columns=[data.columns[i] for i in ids],
        weights=weights.tolist(),validation_rows=len(selected),selection_rule='fixed final spatial update; validation diagnostic only')
    data.close();held.close()
    campaign=resolve_path(c['response_campaign']);old=read(campaign/'technical/plan.json')
    if old['identity']!=c['response_identity']:raise ValueError('Changed response source inventory')
    motion=torch.load(resolve_path(old['config']['cache'])/'dataset.pt',weights_only=False)
    if sha(resolve_path(old['config']['cache'])/'dataset.pt')!=read(campaign/'technical/data-complete.json')['sha256']:raise ValueError('Changed motion data')
    bysource={s['key']:s for s in old['sources']}
    tasks=[(c,old,j,bysource[j['source_key']]) for j in old['jobs']]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,key in enumerate(pool.map(future_packet,tasks),1):
            if i%16==0:print('saved native futures',i,len(tasks),flush=True)
    packets={j['key']:torch.load(dest/'futures'/(j['key']+'.pt'),weights_only=False) for j in old['jobs']}
    future=torch.stack([packets[r['job']][r['index']] for r in motion['rows']])
    atomic_save(dest/'future-positions.pt',future)
    write_json(dest/'spatial.json',metadata)
    done=dict(config_identity=digest(c),data_identity=data_identity(c),spatial=metadata,motion_rows=len(motion['rows']),response_identity=old['identity'],
        files={f:sha(dest/f) for f in ['spatial-selection.npz','future-positions.pt','spatial.json']})
    write_json(dest/'complete.json',done);write_json(root(c)/'technical/data-complete.json',done)
    return done
