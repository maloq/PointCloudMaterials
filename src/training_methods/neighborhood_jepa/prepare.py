"""Tracked six-neighbor triplets from registered raw trajectories; fixed held-out ancestry."""
import argparse,json,time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.spatial import cKDTree
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import source_arrays,chart,offsets,file_hash,digest,save_json,ELEMENTS
from src.data.structural_pretraining.support import REFERENCE_RADIUS,OUTER_RADIUS,EDGE_CUTOFF,support_weights
from src.models.encoders.structural import ATOMIC_NUMBERS


def distributed_neighbors(vectors,count,rng):
    """Random first representative, then angular coverage; rotation-equivariant in distribution."""
    if len(vectors)<count:raise ValueError(f'Only {len(vectors)} candidates for {count} queries')
    unit=vectors/np.linalg.norm(vectors,axis=-1,keepdims=True);order=rng.permutation(len(unit))
    selected=[int(order[0])];distance=np.square(unit-unit[selected[0]]).sum(-1)
    for _ in range(count-1):
        distance[selected]=-1;maximum=distance.max()
        # Randomize genuine symmetry ties, rather than imposing a lab-frame axis.
        tied=np.flatnonzero(np.isclose(distance,maximum,rtol=1e-7,atol=1e-9))
        chosen=int(rng.choice(tied));selected.append(chosen)
        distance=np.minimum(distance,np.square(unit-unit[chosen]).sum(-1))
    return np.array(selected)


def plan(config):
    root=resolve_path(config['cache']);path=root/'plan.json'
    parent=resolve_path(config['parent_release']);manifest=json.loads((parent/'manifest.json').read_text())
    identity=digest(dict(config=config,parent_sha256=file_hash(parent/'manifest.json'),protocol='tracked_neighborhood_v1'))
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['identity']!=identity:raise ValueError('Neighborhood data plan changed')
        return saved
    if manifest['state']!='complete' or manifest['tda_coverage']!='all_supervised_views':raise ValueError('Full TDA parent required')
    sources={s['id']:s for s in manifest['sources']};groups=defaultdict(list);selection=[]
    for shard in manifest['shards']:
        if shard['static']:continue
        if shard['task']['split']=='train':groups[(shard['material'],shard['potential'])].append(shard)
        elif shard['task']['split']=='selection':selection.append(shard)
    keys=sorted(groups);weights=np.array([sum(s['anchors'] for s in groups[k]) for k in keys],float);weights/=weights.sum()
    totals=np.floor(weights*config['anchors']).astype(int);totals[0]+=config['anchors']-totals.sum()
    rng=np.random.default_rng(config['seed']);tasks=[]
    for key,total in zip(keys,totals):
        # Round robin across source identities, avoiding concentration in long trajectories.
        by_source=defaultdict(list)
        for shard in groups[key]:by_source[shard['source']].append(shard)
        source_ids=list(by_source);rng.shuffle(source_ids)
        for sid in source_ids:rng.shuffle(by_source[sid])
        ordered=[]
        while any(by_source.values()):
            for sid in source_ids:
                if by_source[sid]:ordered.append(by_source[sid].pop())
        remaining=int(total)
        for shard in ordered:
            count=min(remaining,shard['anchors'],config['anchors_per_shard'])
            rows=rng.choice(shard['anchors'],count,replace=False).tolist()
            tasks.append(dict(shard=shard,rows=rows,source=sources[shard['source']],group=keys.index(key),seed=int(rng.integers(2**31))))
            remaining-=count
            if not remaining:break
        if remaining:raise ValueError(f'Insufficient training anchors for {key}: {remaining}')
    for shard in selection:
        key=(shard['material'],shard['potential'])
        tasks.append(dict(shard=shard,rows=list(range(shard['anchors'])),source=sources[shard['source']],group=keys.index(key),seed=int(rng.integers(2**31))))
    train_lineages={t['source']['lineage'] for t in tasks if t['shard']['task']['split']=='train'}
    selection_lineages={t['source']['lineage'] for t in tasks if t['shard']['task']['split']=='selection'}
    if train_lineages&selection_lineages:raise ValueError('Training/selection ancestry overlap')
    result=dict(identity=identity,config=config,parent_manifest_sha256=file_hash(parent/'manifest.json'),groups=keys,tasks=tasks)
    save_json(path,result);return result


def prepare_task(args):
    root,parent,identity,task,radius=args;root=Path(root);parent=Path(parent)
    record=task['shard'];sid=record['task']['id'];folder=root/'shards'/sid;receipt=folder/'complete.json'
    if receipt.exists():
        r=json.loads(receipt.read_text())
        if r['identity']!=identity:raise ValueError(f'Changed shard {sid}')
        for name,sha in r['hashes'].items():
            if file_hash(folder/f'{name}.npy')!=sha:raise ValueError(f'Corrupt {sid}/{name}')
        return r
    started=time.monotonic();source=task['source'];raw=source_arrays(source);rng=np.random.default_rng(task['seed'])
    original=parent/'shards'/sid
    a={name:np.load(original/f'{name}.npy',mmap_mode='r') for name in ['views','center_ids','physical','tda','tda_valid']}
    rows=np.array(task['rows']);mapped=a['views'][rows];center_ids=a['center_ids'][mapped[:,2]]
    lookup={int(v):i for i,v in enumerate(raw['atom_ids'])};centers=np.array([lookup[int(v)] for v in center_ids])
    frame=record['task']['frame'];scale=record['scale'];factor=REFERENCE_RADIUS/scale
    x,tree,box=chart(raw,frame,False);chosen=[];query=[]
    for center in centers:
        ids=np.array(sorted(tree.query_ball_point(x[center],radius/factor)));ids=ids[ids!=center]
        vec=offsets(x,center,ids,box)*factor;pick=distributed_neighbors(vec,6,rng)
        chosen.append(np.r_[center,ids[pick]]);query.append(np.vstack((np.zeros((1,3)),vec[pick])))
    chosen=np.stack(chosen);query=np.stack(query).astype(np.float32);del tree,x
    positions=[];weights=[];edges=[];ptr=[0];eptr=[0];views=np.empty((len(rows),3,7),np.int32)
    for ti,f in enumerate([frame-1,frame,frame+1]):
        x,tree,box=chart(raw,f,False)
        for row in range(len(rows)):
            for k,center in enumerate(chosen[row]):
                ids=np.array(sorted(tree.query_ball_point(x[center],OUTER_RADIUS/factor)))
                ids=np.r_[center,ids[ids!=center]];local=offsets(x,center,ids,box)*factor
                local=local[np.linalg.norm(local,axis=-1)<OUTER_RADIUS]
                if len(local)<(80 if k==0 else 12):raise ValueError(f'Insufficient support {sid}/{row}/{ti}/{k}: {len(local)}')
                pairs=cKDTree(local).query_pairs(EDGE_CUTOFF,output_type='ndarray');edge=np.concatenate((pairs,pairs[:,::-1]),0).T.astype(np.int32)
                views[row,ti,k]=len(positions);positions.append(local);weights.append(support_weights(local));edges.append(edge)
                ptr.append(ptr[-1]+len(local));eptr.append(eptr[-1]+edge.shape[1])
        del x,tree
    target_views=mapped[:,[2,3]]
    if not a['tda_valid'][target_views].all():raise ValueError(f'Missing center TDA {sid}')
    times=(raw['timesteps'][[frame-1,frame,frame+1]]-raw['timesteps'][frame])*source['timestep_fs']/1000
    if not times[0]<times[1]==0<times[2]:raise ValueError(f'Invalid time grid {sid}: {times}')
    values=dict(positions=np.concatenate(positions).astype(np.float32),weights=np.concatenate(weights).astype(np.float32),
        edges=np.concatenate(edges,1),offsets=np.array(ptr,np.int64),edge_offsets=np.array(eptr,np.int64),views=views,
        query_positions=query,query_atom_ids=raw['atom_ids'][chosen],times=np.array(times,np.float32),
        physical=np.array(a['physical'][target_views]),tda=np.array(a['tda'][target_views]))
    folder.mkdir(parents=True,exist_ok=True);hashes={}
    for name,v in values.items():
        np.save(folder/f'{name}.npy',v,allow_pickle=False);hashes[name]=file_hash(folder/f'{name}.npy')
    result=dict(identity=identity,id=sid,source=source['id'],lineage=source['lineage'],split=record['task']['split'],
        material=record['material'],potential=record['potential'],group=task['group'],species=ATOMIC_NUMBERS.index(ELEMENTS[record['material']]),
        scale=scale,anchors=len(rows),hashes=hashes,seconds=time.monotonic()-started)
    save_json(receipt,result);return result


def build(config):
    p=plan(config);root=resolve_path(config['cache']);parent=resolve_path(config['parent_release'])
    tasks=[(str(root),str(parent),p['identity'],t,config['query_radius']) for t in p['tasks']];receipts=[]
    with ProcessPoolExecutor(max_workers=config['workers']) as pool:
        futures=[pool.submit(prepare_task,t) for t in tasks]
        for f in as_completed(futures):
            receipts.append(f.result());save_json(root/'status.json',dict(state='preparing',complete=len(receipts),total=len(tasks)))
    receipts.sort(key=lambda r:r['id']);norm={}
    for name in ['physical','tda']:
        values=np.concatenate([np.load(root/'shards'/r['id']/f'{name}.npy').reshape(-1,85 if name=='physical' else 144) for r in receipts if r['split']=='train']).astype(np.float64)
        norm[name]=dict(mean=values.mean(0).tolist(),std=np.maximum(values.std(0),1e-4).tolist())
    save_json(root/'manifest.json',dict(state='complete',identity=p['identity'],config=config,groups=p['groups'],shards=receipts,normalization=norm))
    save_json(root/'status.json',dict(state='complete',shards=len(receipts),anchors=sum(r['anchors'] for r in receipts)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    build(json.loads(Path(args.config).read_text()))
