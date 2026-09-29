"""Exact reference weights, existing geometry, and label-side input clearance."""
import json
from pathlib import Path
import numpy as np
from src.data.fixed_cohort.protocol import sha, digest, write_json
from src.project_runtime.paths import resolve_path

ROLES=('train','selection','calibration','test')

def config(path):
    return json.loads(Path(path).read_text())

def population(c):
    root=resolve_path(c['dataset']['root']);manifest=json.loads((root/'manifest.json').read_text())
    if manifest['identity']!=c['prepared_data']['identity'] or sha(root/'plan.json')!=manifest['plan_sha256']:
        raise ValueError('Changed sealed cohort')
    plan=json.loads((root/'plan.json').read_text());rows=[];offset=0
    fields=('source','frame','atom','role','kind','crystal_distance','inside_crystal','crystal_visible_context',
            'distance','visible_context','interface_exists','interface_parent_index','indices','actual')
    for item in manifest['sources']:
        folder=root/'sources'/str(item['source'])
        if sha(folder/'rows.npz')!=item['files']['rows.npz']:raise ValueError(f'Changed rows {folder}')
        with np.load(folder/'rows.npz') as a:r={k:a[k] for k in fields}
        r['local_row']=np.arange(len(r['atom']));r['indices']+=offset
        rows.append(r);offset+=item['patches']
    meta={k:np.concatenate([r[k] for r in rows]) for k in fields+('local_row',)}
    base=np.zeros(len(meta['atom']))
    for role in ROLES:
        for kind in (0,1):
            ids=np.flatnonzero((meta['role']==role)&(meta['kind']==kind))
            sources,inv=np.unique(meta['source'][ids],return_inverse=True)
            den={s['source']:s['population_counts'][str(kind)] for s in manifest['sources']}
            base[ids]=1/(2*len(sources)*np.asarray([den[int(s)] for s in sources])[inv])
    clear=~meta['inside_crystal']&~meta['crystal_visible_context']&np.isfinite(meta['crystal_distance'])
    if not np.array_equal(meta['distance'][clear],meta['crystal_distance'][clear]):raise ValueError('Liquid target changed')
    if meta['visible_context'][clear].any():raise ValueError('Visible interface in clear population')
    if len({s['lineage'] for s in plan['sources']})!=len(plan['sources']):raise ValueError('Sources share ancestry; bootstrap groups must be revised')
    return root,manifest,plan,meta,base

def masks(meta,base,arm,c):
    valid=~meta['inside_crystal']&np.isfinite(meta['crystal_distance'])&(meta['kind']<2)
    valid &= meta['crystal_visible_context'] if arm['population']=='visible_control' else ~meta['crystal_visible_context']
    split={r:np.flatnonzero(valid&(meta['role']==r)) for r in ROLES}
    available=np.unique(meta['source'][split['train']]);rng=np.random.default_rng(c['seed'])
    order=rng.permutation(available);chosen=order[:max(1,int(np.ceil(len(order)*arm['source_fraction'])))];chosen.sort()
    split['train']=split['train'][np.isin(meta['source'][split['train']],chosen)]
    weights={}
    for r,ids in split.items():
        w=base[ids]
        if not len(ids) or w.sum()<=0:raise ValueError(f'Empty declared {r}/{arm}')
        weights[r]=w/w.sum()
    return split,weights,chosen

def feature_summary(x,actual):
    # Identical local descriptors at all patches; radial summaries and a vector
    # gradient norm preserve rotation invariance without a laboratory direction.
    radius=np.linalg.norm(actual,axis=-1);parts=[]
    for lo,hi in ((-1,4),(4,14),(14,24.0001)):
        w=((radius>lo)&(radius<=hi)).astype(float);w/=w.sum(1,keepdims=True)
        mean=np.einsum('np,npf->nf',w,x)
        parts.extend([mean,np.sqrt(np.maximum(np.einsum('np,npf->nf',w,x*x)-mean*mean,0))])
    center=x-x.mean(1,keepdims=True)
    gradient=np.einsum('npf,npd->nfd',center,actual/24)/25
    parts.append(np.linalg.norm(gradient,axis=-1))
    return np.concatenate(parts,axis=-1).astype(np.float32)

def prepare_source(c,sid):
    """One CPU shard. Labels never enter physical feature calculations."""
    import torch
    from scipy.spatial import cKDTree
    from src.research.encoder_context.geometry import physical_targets
    from src.research.crystallization_origin import ancestry
    torch.set_num_threads(2)
    root=resolve_path(c['dataset']['root']);manifest=json.loads((root/'manifest.json').read_text())
    plan=json.loads((root/'plan.json').read_text());item=next(s for s in plan['sources'] if s['id']==sid)
    record=next(s for s in manifest['sources'] if s['source']==sid)
    dest=resolve_path(c['study_cache'])/'sources'/str(sid);dest.mkdir(parents=True,exist_ok=True)
    binding=dict(dataset=manifest['identity'],source=sid,producer=sha(Path(__file__)),physical=sha(Path(physical_targets.__code__.co_filename)))
    identity=digest(binding)
    if (dest/'complete.json').exists():
        done=json.loads((dest/'complete.json').read_text())
        if done['identity']!=identity or sha(dest/'features.npz')!=done['sha256']:raise ValueError(f'Changed preparation {sid}')
        return done
    folder=root/'sources'/str(sid)
    for name,h in record['files'].items():
        if sha(folder/name)!=h:raise ValueError(f'Changed source {sid}/{name}')
    with np.load(folder/'rows.npz') as a:m={k:a[k] for k in a.files}
    ids=np.flatnonzero((m['kind']<2)&~m['inside_crystal']&np.isfinite(m['crystal_distance']))
    bank=np.load(folder/'positions.npy',mmap_mode='r');patches=np.unique(m['indices'][ids]);desc=np.empty((len(patches),32),np.float32)
    for start in range(0,len(patches),1024):
        desc[start:start+1024]=physical_targets(torch.from_numpy(np.array(bank[patches[start:start+1024]]))).numpy()
    x=desc[np.searchsorted(patches,m['indices'][ids])]
    features=feature_summary(x,m['actual'][ids]);clearance=np.full(len(ids),np.nan,np.float32)
    audit=resolve_path(c['dataset']['audit'])/'technical/sources'/str(sid)
    if sha(audit/'graph.npz')!=json.loads((audit/'graph-complete.json').read_text())['sha256']:raise ValueError('Changed crystal graph')
    with np.load(audit/'graph.npz') as a:g={k:a[k] for k in a.files}
    settings=config(resolve_path(c['dataset']['audit_config']));events,roots,_=ancestry.establish(g,settings['lineage']['thresholds'][0])
    access=ancestry.GeometryAccess(settings,item,audit,g);ptm=json.loads((audit/'ptm-complete.json').read_text());checked=set()
    for frame in np.unique(m['frame'][ids]):
        frame=int(frame);chunk=settings['ptm']['chunk_frames'];start=frame//chunk*chunk
        filename=f'ptm-{start:04d}-{min(item["frame_count"],start+chunk):04d}.npz'
        if filename not in checked:
            if sha(audit/filename)!=ptm['files'][filename]:raise ValueError('Changed PTM labels')
            checked.add(filename)
        points,box,dense=access.frame(frame);loc=np.flatnonzero(m['frame'][ids]==frame);rows=ids[loc]
        nodes=np.flatnonzero((g['frame']==frame)&(g['size']>=64))
        known=[n for n in nodes if any(events[r-1]['confirmation_frame']<=frame for r in roots[n])]
        solid=np.isin(dense,known);tree=cKDTree(points[solid],boxsize=box)
        centers=points[np.searchsorted(access.raw.atom_ids,m['atom'][rows])]
        if not np.allclose(tree.query(centers)[0],m['crystal_distance'][rows],atol=1e-4,rtol=1e-6):raise ValueError(f'Target replay changed {sid}/{frame}')
        unique,first,inverse=np.unique(m['indices'][rows].flatten(),return_index=True,return_inverse=True)
        patch_centers=np.mod((centers[:,None]+m['actual'][rows]).reshape(-1,3)[first],box)
        nearest=np.empty(len(unique))
        for begin in range(0,len(unique),256):
            xyz=np.array(bank[unique[begin:begin+256]],dtype=np.float64)
            observed=np.mod(patch_centers[begin:begin+256,None]+xyz,box)
            distance=tree.query(observed.reshape(-1,3),workers=1)[0].reshape(-1,80)
            nearest[begin:begin+len(xyz)]=np.where(np.linalg.norm(xyz,axis=-1)<8,distance,np.inf).min(1)
        clearance[loc]=nearest[inverse].reshape(-1,25).min(1)
        if not np.array_equal(clearance[loc]<.002,m['crystal_visible_context'][rows]):raise ValueError(f'Input visibility replay changed {sid}/{frame}')
    if not np.isfinite(features).all() or not np.isfinite(clearance).all():raise ValueError(f'Nonfinite features {sid}')
    np.savez(dest/'features.npz',local_row=ids,features=features,physical_local=x[:,0],clearance_A=clearance)
    done=dict(identity=identity,source=sid,rows=len(ids),sha256=sha(dest/'features.npz'),binding=binding)
    write_json(dest/'complete.json',done);return done

def seal(c):
    root,manifest,plan,meta,base=population(c);records=[];cache=resolve_path(c['study_cache'])
    for s in manifest['sources']:
        folder=cache/'sources'/str(s['source']);r=json.loads((folder/'complete.json').read_text())
        if sha(folder/'features.npz')!=r['sha256']:raise ValueError('Changed prepared features')
        records.append(r)
    write_json(cache/'manifest.json',dict(dataset_identity=manifest['identity'],sources=records,identity=digest(records)))

def features(c,meta):
    cache=resolve_path(c['study_cache']);manifest=config(cache/'manifest.json')
    x=np.full((len(meta['atom']),224),np.nan,np.float32);physical=np.full((len(x),32),np.nan,np.float32);clearance=np.full(len(x),np.nan,np.float32)
    for s in manifest['sources']:
        file=cache/'sources'/str(s['source'])/'features.npz'
        if sha(file)!=s['sha256']:raise ValueError('Changed descriptors')
        global_ids=np.flatnonzero(meta['source']==s['source'])
        with np.load(file) as a:
            ids=global_ids[a['local_row']];x[ids]=a['features'];physical[ids]=a['physical_local'];clearance[ids]=a['clearance_A']
    return x,physical,clearance
