"""Attach causal crystal-distance labels to retained dynamic geometry shards."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import json
import multiprocessing
from pathlib import Path
import os

import numpy as np
from scipy.spatial import cKDTree

from src.data.fixed_cohort.dataset import read_release
from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.data.structural_pretraining.native_dataset import material_factors
from src.project_runtime.paths import resolve_path
from src.research.crystallization_origin import ancestry, external_data
from src.research.crystallization_origin.extract import raw_source, frame_geometry


def plan(config_path):
    c=json.loads(Path(config_path).read_text())
    structural=resolve_path(c['structural_dataset']['root'])
    manifest=json.loads((structural/'manifest.json').read_text())
    structural_plan=json.loads((structural/'plan.json').read_text())
    if manifest['identity']!=c['structural_dataset']['identity'] or sha(structural/'plan.json')!=manifest['plan_sha256']:
        raise ValueError('Structural release changed or incomplete')
    factors=material_factors(structural_plan,c['structural_dataset']['normalization'])
    _,fixed=read_release(c['fixed_dataset']['root'])
    if fixed['identity']!=c['fixed_dataset']['identity']:raise ValueError('Fixed release changed')
    native_config=json.loads(resolve_path(c['labels']['native_config']).read_text())
    external_path=resolve_path(c['labels']['external_plan']);external=json.loads(external_path.read_text())
    fixed_items={p['id']:p for p in fixed['sources']}
    external_map={part['id']:(s,pi) for s in external['sources'] for pi,part in enumerate(s['parts'])}
    sources={};excluded=[]
    for source in structural_plan['sources']:
        if source['kind']!='dynamic':
            excluded.append(dict(source=source['id'],material=source['material'],reason='static input lacks the sustained-reference history'))
            continue
        if 'native_id' in source:
            item=fixed_items[source['native_id']]
            if item['role'] not in ('train','selection') or item['role']!=source['split']:
                raise ValueError(f'Forbidden source role: {source["id"]}')
            folder=resolve_path(native_config['output'])/'technical/sources'/str(item['id'])
            audit_frames={int(f):int(f) for f in source['observed_frames']}
            audit_source=item;kind='native';settings=native_config['lineage'];chunk=native_config['ptm']['chunk_frames']
        else:
            if source['split']!='train':raise ValueError('External families are train-only')
            audit_source,part=external_map[source['id']]
            folder=external_data.source_folder(external,audit_source['id'])
            audit_frames={raw:i for i,(pi,raw) in enumerate(audit_source['frames']) if pi==part}
            kind='external';settings=audit_source['lineage'];chunk=external['config']['chunk_frames']
        threshold=settings['thresholds'][0]
        frames=[int(f) for f in source['observed_frames'] if audit_frames[int(f)]>=threshold['persistence_frames']-1]
        # Initial unconfirmed history is omitted uniformly, without looking at outcomes.
        sources[source['id']]=dict(source=source,audit_source=audit_source,kind=kind,folder=str(folder),
            graph_sha256=sha(folder/'graph.npz'),graph_receipt_sha256=sha(folder/'graph-complete.json'),
            ptm_receipt_sha256=sha(folder/'ptm-complete.json'),settings=settings,chunk_frames=chunk,
            audit_frames={str(f):audit_frames[f] for f in frames},factor=float(factors[source['material']]))
    shards=[s for s in manifest['shards'] if s['task']['source'] in sources and s['task']['observed'] and
            str(s['task']['frame']) in sources[s['task']['source']]['audit_frames']]
    tasks=[]
    for sid,s in sources.items():
        frames=sorted(map(int,s['audit_frames']))
        for start in range(0,len(frames),c['labels']['frames_per_task']):
            tasks.append(dict(source=sid,frames=frames[start:start+c['labels']['frames_per_task']],atoms=s['source']['atom_count']))
    loads=[0]*c['labels']['lanes']
    for task in sorted(tasks,key=lambda t:-t['atoms']*len(t['frames'])):
        lane=int(np.argmin(loads));task['lane']=lane;loads[lane]+=task['atoms']*len(task['frames'])
    result=dict(protocol='causal-distance-multimaterial-v1',structural_root=str(structural),
        structural_identity=manifest['identity'],manifest_sha256=sha(structural/'manifest.json'),
        fixed_identity=fixed['identity'],sources=sources,shards=shards,tasks=tasks,excluded=excluded,
        output=str(resolve_path(c['labels']['root'])),external_plan_sha256=sha(external_path),
        native_config_sha256=sha(resolve_path(c['labels']['native_config'])),
        producer={str(p):sha(p) for p in (Path(__file__),Path(ancestry.__file__))},
        sampling='all retained dynamic structural centers/frames, omitting only initial unconfirmed history',
        target='periodic distance to a >=64-atom component with lineage confirmed by the current frame',
        normalization=c['structural_dataset']['normalization'])
    # Paths in the producer record are stable repository-relative names.
    result['producer']={Path(k).name:v for k,v in result['producer'].items()}
    result['identity']=digest(result)
    root=resolve_path(c['labels']['root']);root.mkdir(parents=True,exist_ok=True)
    target=root/'plan.json'
    if target.exists() and json.loads(target.read_text())!=result:raise ValueError('Distance label release changed; choose a new label root')
    if not target.exists():write_json(target,result)
    counts={}
    for s in shards:
        key=s['task']['split']+'/'+s['material'];counts[key]=counts.get(key,0)+s['rows']
    print(json.dumps(dict(identity=result['identity'],rows=counts,tasks=len(tasks),lanes=len(loads))),flush=True)
    return target


@lru_cache(maxsize=2)
def load_plan(path):return json.loads(Path(path).read_text())


@lru_cache(maxsize=4)
def reference(path,sid):
    p=load_plan(path);s=p['sources'][sid];folder=Path(s['folder'])
    if sha(folder/'graph.npz')!=s['graph_sha256'] or sha(folder/'ptm-complete.json')!=s['ptm_receipt_sha256']:
        raise ValueError(f'Audit changed: {sid}')
    if json.loads((folder/'graph-complete.json').read_text())['sha256']!=s['graph_sha256']:
        raise ValueError('Graph receipt mismatch')
    with np.load(folder/'graph.npz') as a:g={k:a[k] for k in a.files}
    events,roots,_=ancestry.establish(g,s['settings']['thresholds'][0])
    if s['kind']=='native':
        raw=raw_source(s['audit_source']);ids=raw.atom_ids
    else:
        raw=None;part=s['source'];ids=external_data.arrays(part['path'],part['manifest_sha256'])['atom_ids']
    centers=np.asarray(s['source']['center_ids']);rows=np.searchsorted(ids,centers)
    np.testing.assert_array_equal(ids[rows],centers)
    ptm=json.loads((folder/'ptm-complete.json').read_text())
    return g,events,roots,raw,rows,ptm


@lru_cache(maxsize=8)
def labels_file(path,expected):
    if sha(path)!=expected:raise ValueError(f'Changed PTM chunk: {path}')
    with np.load(path) as a:return a['labels']


def task(plan_path,record):
    p=load_plan(plan_path);sid=record['source'];s=p['sources'][sid]
    folder=Path(p['output'])/'sources'/sid;folder.mkdir(parents=True,exist_ok=True)
    g,events,roots,raw,rows,ptm=reference(plan_path,sid)
    out=[]
    for frame in record['frames']:
        dest=folder/f'{frame}.npy';receipt=dest.with_suffix('.json')
        if receipt.exists():
            saved=json.loads(receipt.read_text())
            if saved['identity']!=p['identity'] or saved['sha256']!=sha(dest):raise ValueError(f'Changed labels: {dest}')
            out.append(saved);continue
        af=s['audit_frames'][str(frame)];nodes=np.flatnonzero((g['frame']==af)&(g['size']>=64))
        known=[n for n in nodes if any(events[r-1]['confirmation_frame']<=af for r in roots[n])]
        if not known:
            distance=np.full(len(rows),np.inf,np.float32)
        else:
            if s['kind']=='native':points,box=frame_geometry(raw,frame)
            else:points,box=external_data.geometry(s['audit_source'],af)
            start=af//s['chunk_frames']*s['chunk_frames']
            stop=min(s['audit_source']['frame_count'],start+s['chunk_frames'])
            name=f'ptm-{start:04d}-{stop:04d}.npz';path=Path(s['folder'])/name
            labels=labels_file(str(path),ptm['files'][name])[af-start]
            if int(g['size'][known].sum())==int(g['crystalline_atoms'][af]):
                # Exact shortcut: every PTM-crystalline atom belongs to the known set.
                solid=np.isin(labels,[1,2,3])
            else:
                local,sizes=ancestry.components(points,box,labels,s['settings'])
                np.testing.assert_array_equal(sizes[1:],g['size'][g['frame']==af])
                solid=np.isin(np.where(local>0,local+g['start'][af],0),known)
            if int(solid.sum())!=int(g['size'][known].sum()):raise ValueError(f'Reconstructed reference changed: {sid}/{frame}')
            distance=cKDTree(points[solid],boxsize=box).query(points[rows],workers=1)[0].astype(np.float32)
        if np.isnan(distance).any() or (distance<0).any():raise FloatingPointError('Invalid distance target')
        temporary=dest.with_suffix('.building.npy');np.save(temporary,distance);temporary.replace(dest)
        normalized=distance*s['factor']
        saved=dict(identity=p['identity'],source=sid,frame=frame,rows=len(rows),sha256=sha(dest),
            zero=int((distance==0).sum()),censored64=int((normalized>=64).sum()),within20=int((normalized<=20).sum()))
        write_json(receipt,saved);out.append(saved)
    return dict(source=sid,frames=len(out),rows=sum(r['rows'] for r in out))


def lane(plan_path,lane_id,workers):
    p=load_plan(str(plan_path));tasks=[t for t in p['tasks'] if t['lane']==lane_id]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(task,str(plan_path),t) for t in tasks]
        for i,future in enumerate(as_completed(futures)):
            result=future.result();print(json.dumps(dict(stage='distance-labels',lane=lane_id,completed=i+1,total=len(tasks),**result)),flush=True)
    write_json(Path(p['output'])/f'lane-{lane_id}.json',dict(identity=p['identity'],tasks=len(tasks),state='complete'))


def seal(plan_path):
    p=load_plan(str(plan_path));root=Path(p['output']);records={}
    for s in p['shards']:
        t=s['task'];key=f'{t["source"]}/{t["frame"]}'
        if key not in records:
            path=root/'sources'/t['source']/f'{t["frame"]}.npy'
            receipt=json.loads(path.with_suffix('.json').read_text())
            if receipt['identity']!=p['identity'] or receipt['sha256']!=sha(path):raise ValueError(f'Unsealed target: {key}')
            records[key]=receipt
    counts={}
    for s in p['shards']:
        key=s['task']['split']+'/'+s['material'];counts[key]=counts.get(key,0)+s['rows']
    write_json(root/'manifest.json',dict(state='complete',identity=p['identity'],plan_sha256=sha(plan_path),frames=records,counts=counts))
    print(json.dumps(dict(stage='sealed',counts=counts,frames=len(records))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['plan','lane','seal'])
    parser.add_argument('--config');parser.add_argument('--plan');parser.add_argument('--lane',type=int)
    parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    if args.stage=='plan':plan(args.config)
    elif args.stage=='lane':lane(args.plan,args.lane if args.lane is not None else int(os.environ['SLURM_ARRAY_TASK_ID']),args.workers)
    else:seal(args.plan)


if __name__=='__main__':main()
