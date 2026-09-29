"""Six tracked MD observations at declared cadences, sharing immutable geometry."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch

from src.data.fixed_cohort.protocol import centered, digest, sha, write_json
from src.data.structural_pretraining.prepare import source_arrays, chart
from src.project_runtime.paths import resolve_path


def make_plan(config_path):
    c=json.loads(Path(config_path).read_text());root=resolve_path(c['dense_history']['root'])
    parent_root=resolve_path(c['labels']['root']);parent=json.loads((parent_root/'plan.json').read_text())
    receipt=json.loads((parent_root/'manifest.json').read_text())
    if receipt['state']!='complete' or sha(parent_root/'plan.json')!=receipt['plan_sha256']:
        raise ValueError('Incomplete or changed parent labels')
    if parent['structural_identity']!=c['structural_dataset']['identity'] or parent['fixed_identity']!=c['fixed_dataset']['identity']:
        raise ValueError('Dense-history source identities differ from the declared releases')
    cadence=c['history']['cadence_ps']
    offsets=np.asarray(c['history']['offsets_ps'],dtype=np.float64)
    if cadence<=0 or offsets.shape!=(6,) or not np.allclose(offsets,np.arange(-5,1)*cadence,rtol=0,atol=1e-8):
        raise ValueError('Dense history requires one positive cadence and six uniformly spaced offsets ending at zero')
    sources=[];excluded=[]
    for sid,s in parent['sources'].items():
        if c['dense_history']['source_kind']!='all' and s['kind']!=c['dense_history']['source_kind']:
            excluded.append(dict(id=sid,reason='outside the declared source collection',kind=s['kind']))
            continue
        raw=source_arrays(s['source']);steps=np.asarray(raw['timesteps'])
        times=steps.astype(np.float64)*s['source']['timestep_fs']/1000
        raw_cadence=float(np.median(np.diff(times)))
        if not np.allclose(np.diff(times),raw_cadence,rtol=0,atol=1e-6):raise ValueError(f'Irregular raw timeline: {sid}')
        observed_cadence=c['history']['observed_cadence_ps_by_kind'][s['kind']]
        observed_offsets=np.arange(-5,1)*observed_cadence
        frame_offsets=np.rint(observed_offsets/raw_cadence).astype(int)
        if not np.allclose(frame_offsets*raw_cadence,observed_offsets,rtol=0,atol=1e-6):
            raise ValueError(f'Cadence mismatch for {sid}: declared {observed_cadence} ps is unavailable on the saved {raw_cadence}-ps timeline')
        label_frames=sorted(map(int,s['audit_frames']))
        # Match exactly the existing three-frame, 6-ps arm's anchor population.
        anchors=label_frames[2:]
        if not anchors or min(anchors)+min(frame_offsets)<0:raise ValueError(f'Missing six-frame history for matched anchors: {sid}')
        np.testing.assert_allclose(np.diff(times[label_frames]),3.,rtol=0,atol=1e-6)
        frames=sorted({int(f+k) for f in anchors for k in frame_offsets})
        roles=c['dense_history']['roles_by_lineage']
        role=roles[s['source']['lineage']] if roles else s['source']['split']
        sources.append(dict(id=sid,source=s['source'],kind=s['kind'],factor=s['factor'],cadence_ps=observed_cadence,raw_cadence_ps=raw_cadence,
            frame_offsets=frame_offsets.tolist(),offsets_ps=observed_offsets.tolist(),anchors=anchors,frames=frames,
            centers=len(s['source']['center_ids']),role=role,material=s['source']['material'],
            sequences=len(anchors)*len(s['source']['center_ids']),unique_patches=len(frames)*len(s['source']['center_ids'])))
    if not {'train','selection'}<={s['role'] for s in sources}:raise ValueError('Uniform-cadence release must contain both train and selection sources')
    lineage_roles={}
    for s in sources:
        lineage_roles.setdefault(s['source']['lineage'],set()).add(s['role'])
    if any(len(roles)!=1 for roles in lineage_roles.values()):raise ValueError('An ancestry group crosses history source roles')
    plan=dict(protocol='six_declared_cadence_md_observations_v3',nominal_cadence_ps=cadence,nominal_offsets_ps=offsets.tolist(),
        observed_cadence_ps_by_kind=c['history']['observed_cadence_ps_by_kind'],interpolation=False,
        source_kind=c['dense_history']['source_kind'],roles_by_lineage=c['dense_history']['roles_by_lineage'],excluded_sources=excluded,
        parent_root=str(parent_root),parent_plan_sha256=sha(parent_root/'plan.json'),
        parent_manifest_sha256=sha(parent_root/'manifest.json'),structural_root=parent['structural_root'],sources=sources,
        producer_sha256=sha(__file__),frames=6,coordinate_dtype='float32',normalization=c['structural_dataset']['normalization'],
        anchors='identical to CD-MACE128-H6: omit first two retained 3-ps label frames per source')
    plan['identity']=digest(plan);root.mkdir(parents=True,exist_ok=True);target=root/'plan.json'
    if target.exists() and json.loads(target.read_text())!=plan:raise ValueError('Dense history release changed; use a new root')
    write_json(target,plan)
    return plan


def prepare_source(config_path,sid):
    c=json.loads(Path(config_path).read_text());root=resolve_path(c['dense_history']['root'])
    plan=json.loads((root/'plan.json').read_text());s=next(s for s in plan['sources'] if s['id']==sid)
    parent=json.loads((Path(plan['parent_root'])/'plan.json').read_text())
    folder=root/'sources'/sid;folder.mkdir(parents=True,exist_ok=True);receipt=folder/'complete.json'
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old['identity']!=plan['identity'] or any(sha(folder/n)!=h for n,h in old['files'].items()):
            raise ValueError(f'Changed dense history geometry: {sid}')
        return old
    raw=source_arrays(s['source']);centers=np.asarray(s['source']['center_ids']);rows=np.searchsorted(raw['atom_ids'],centers)
    np.testing.assert_array_equal(raw['atom_ids'][rows],centers)
    shards={}
    for shard in parent['shards']:
        if shard['task']['source']==sid:shards.setdefault(shard['task']['frame'],[]).append(shard)
    dest=folder/'positions.npy';partial=folder/'positions.building.npy'
    shape=(len(s['frames']),s['centers'],80,3)
    progress=folder/'progress.json'
    done=0
    if progress.exists():
        prior=json.loads(progress.read_text())
        if prior['identity']!=plan['identity']:raise ValueError(f'Changed partial dense release: {sid}')
        done=prior['completed_frames']
    if done:
        coordinates=np.lib.format.open_memmap(partial,mode='r+')
        if coordinates.shape!=shape or coordinates.dtype!=np.float32:raise ValueError('Partial coordinate shape/dtype changed')
    else:
        coordinates=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float32,shape=shape)
    for index in range(done,len(s['frames'])):
        frame=s['frames'][index]
        if frame in shards:
            seen=np.zeros(s['centers'],bool)
            for shard in shards[frame]:
                t=shard['task'];origin=Path(plan['structural_root'])/'shards'/t['id']
                for name in ('hot.npy','center_ids.npy'):
                    if sha(origin/name)!=shard['files'][name]:raise ValueError(f'Changed source coordinates: {origin/name}')
                np.testing.assert_array_equal(np.load(origin/'center_ids.npy'),centers[t['start']:t['stop']])
                coordinates[index,t['start']:t['stop']]=np.load(origin/'hot.npy')*s['factor']
                seen[t['start']:t['stop']]=True
            if not seen.all():raise ValueError(f'Incomplete existing center coverage: {sid}/{frame}')
        else:
            points,tree,box=chart(raw,frame,False)
            for first in range(0,len(rows),4096):
                chosen=rows[first:first+4096];neighbors=tree.query(points[chosen],k=80,workers=1)[1]
                np.testing.assert_array_equal(neighbors[:,0],chosen)
                xyz=centered(points,box,chosen,neighbors)*s['factor']
                if not np.isfinite(xyz).all() or np.any(xyz[:,0]):raise ValueError(f'Invalid dense patch: {sid}/{frame}')
                coordinates[index,first:first+len(chosen)]=xyz
            del points,tree,box
        coordinates.flush()
        write_json(progress,dict(identity=plan['identity'],completed_frames=index+1,total_frames=len(s['frames'])))
    del coordinates;partial.replace(dest)
    result=dict(identity=plan['identity'],source=sid,rows=s['sequences'],unique_patches=s['unique_patches'],
        cadence_ps=s['cadence_ps'],raw_cadence_ps=s['raw_cadence_ps'],offsets_ps=s['offsets_ps'],
        interpolated=False,role=s['role'],files={'positions.npy':sha(dest)})
    write_json(receipt,result);return result


def prepare(config_path,lane):
    c=json.loads(Path(config_path).read_text());root=resolve_path(c['dense_history']['root']);plan=json.loads((root/'plan.json').read_text())
    # Greedy atom-frame load balancing, independent of crystal labels.
    loads=[0]*c['dense_history']['lanes'];assign={}
    for source in sorted(plan['sources'],key=lambda s:-s['source']['atom_count']*len(s['frames'])):
        target=int(np.argmin(loads));assign[source['id']]=target;loads[target]+=source['source']['atom_count']*len(source['frames'])
    sources=[s for s in plan['sources'] if assign[s['id']]==lane];results=[]
    with ProcessPoolExecutor(max_workers=c['dense_history']['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[pool.submit(prepare_source,str(Path(config_path).resolve()),s['id']) for s in sources]
        for job in as_completed(jobs):
            result=job.result();results.append(result)
            write_json(root/f'lane-{lane}-state.json',dict(completed=len(results),total=len(sources),last=result['source']))
            print(json.dumps(dict(lane=lane,completed=len(results),total=len(sources),source=result['source'])),flush=True)
    write_json(root/f'lane-{lane}.json',dict(identity=plan['identity'],sources=results))


def seal(config_path):
    c=json.loads(Path(config_path).read_text());root=resolve_path(c['dense_history']['root']);plan=json.loads((root/'plan.json').read_text())
    receipts=[]
    for source in plan['sources']:
        folder=root/'sources'/source['id'];r=json.loads((folder/'complete.json').read_text())
        if r['identity']!=plan['identity'] or any(sha(folder/n)!=h for n,h in r['files'].items()):raise ValueError(f'Invalid dense shard: {folder}')
        receipts.append(r)
    result=dict(state='complete',identity=plan['identity'],plan_sha256=sha(root/'plan.json'),sources=receipts,
        sequences=sum(r['rows'] for r in receipts),unique_patches=sum(r['unique_patches'] for r in receipts))
    write_json(root/'manifest.json',result);return result


class DenseHistoryDataset:
    def __init__(self,c,role,device):
        root=resolve_path(c['dense_history']['root']);manifest=json.loads((root/'manifest.json').read_text());plan=json.loads((root/'plan.json').read_text())
        if manifest['state']!='complete' or sha(root/'plan.json')!=manifest['plan_sha256']:raise ValueError('Dense history release incomplete/changed')
        if digest({k:v for k,v in plan.items() if k!='identity'})!=manifest['identity']:raise ValueError('Dense history identity changed')
        if plan['nominal_cadence_ps']!=c['history']['cadence_ps'] or plan['source_kind']!=c['dense_history']['source_kind']:
            raise ValueError('Dense release differs from configured cadence/source collection')
        if plan['roles_by_lineage']!=c['dense_history']['roles_by_lineage'] or plan['observed_cadence_ps_by_kind']!=c['history']['observed_cadence_ps_by_kind']:
            raise ValueError('History source roles or actual cadence contract changed')
        for s in plan['sources']:
            expected_cadence=c['history']['observed_cadence_ps_by_kind'][s['kind']]
            expected=np.arange(-5,1)*expected_cadence
            if not np.isclose(s['cadence_ps'],expected_cadence,rtol=0,atol=1e-8) or not np.allclose(s['offsets_ps'],expected,rtol=0,atol=1e-8):
                raise ValueError(f'Undeclared history cadence/offsets: {s["id"]}: {s["offsets_ps"]}, expected {expected.tolist()}')
        label_root=resolve_path(c['labels']['root'])
        if sha(label_root/'manifest.json')!=plan['parent_manifest_sha256'] or sha(label_root/'plan.json')!=plan['parent_plan_sha256']:
            raise ValueError('Parent labels changed after dense extraction')
        parent=json.loads((label_root/'plan.json').read_text());labels=json.loads((label_root/'manifest.json').read_text())
        sources=[s for s in plan['sources'] if s['role']==role]
        subset=c.get('dense_source_subset')
        if subset is not None:
            sources=[s for s in sources if s['material']==subset['material'] and s['kind']==subset['kind']]
        if not sources:raise ValueError(f'No dense-history sources for role={role}, subset={subset}')
        n=sum(s['sequences'] for s in sources)
        self.positions=torch.empty((sum(s['unique_patches'] for s in sources),80,3),device=device,dtype=torch.float32)
        lookup={};offset=0
        for count,s in enumerate(sources):
            path=root/'sources'/s['id']/'positions.npy'
            receipt=next(r for r in manifest['sources'] if r['source']==s['id'])
            if sha(path)!=receipt['files']['positions.npy']:raise ValueError(f'Changed dense coordinates: {path}')
            array=np.load(path,mmap_mode='r').reshape(-1,80,3)
            if len(array)!=s['unique_patches'] or array.dtype!=np.float32:raise ValueError('Dense shape/type mismatch')
            for first in range(0,len(array),16384):
                x=np.array(array[first:first+16384]);self.positions[offset+first:offset+first+len(x)]=torch.from_numpy(x).to(device)
            for i,f in enumerate(s['frames']):lookup[s['id'],f]=offset+i*s['centers']
            offset+=len(array)
            print(json.dumps(dict(stage='resident-dense-history',role=role,sources=count+1,total=len(sources),patches=offset)),flush=True)
        source_map={s['id']:s for s in sources};self.source_names=sorted(source_map);self.material_names=sorted({s['material'] for s in sources})
        index=np.empty((n,6),np.int64);distance=np.empty(n,np.float32);material=np.empty(n,np.int8);source_id=np.empty(n,np.int32)
        offset=0;previous=None
        for shard in parent['shards']:
            t=shard['task']
            if t['source'] not in source_map:continue
            s=source_map[t['source']]
            if t['frame'] not in s['anchors']:continue
            key=(t['source'],t['frame'])
            if key!=previous:
                path=label_root/'sources'/t['source']/f'{t["frame"]}.npy'
                if sha(path)!=labels['frames'][f'{t["source"]}/{t["frame"]}']['sha256']:raise ValueError(f'Changed distance labels: {path}')
                target=np.load(path)*s['factor'];previous=key
            take=slice(offset,offset+shard['rows']);centers=np.arange(t['start'],t['stop'])
            index[take]=centers[:,None]+np.asarray([lookup[t['source'],t['frame']+k] for k in s['frame_offsets']])[None]
            distance[take]=target[t['start']:t['stop']];material[take]=self.material_names.index(s['material']);source_id[take]=self.source_names.index(s['id'])
            offset+=shard['rows']
        if offset!=n:raise ValueError('Dense sequence population mismatch')
        groups=material if role=='train' else source_id;names,counts=np.unique(groups,return_counts=True);weight=np.empty(n,np.float32)
        for name,count in zip(names,counts):weight[groups==name]=n/(len(names)*count)
        self.counts={name:int((material==i).sum()) for i,name in enumerate(self.material_names)}
        self.target_counts={}
        for i,name in enumerate(self.material_names):
            d=distance[material==i];self.target_counts[name]=dict(rows=len(d),zero=int((d==0).sum()),within20=int((d<=20).sum()),censored64=int((d>=64).sum()))
        self.index=torch.from_numpy(index).to(device);self.distance=torch.from_numpy(distance).to(device);self.weight=torch.from_numpy(weight).to(device)
        self.n=n;self.identity=manifest['identity'];self.mode=c['history']['mode'];self.role=role
        self.source_ids=source_id;self.material_ids=material
        self.history_metadata=dict(rows=n,material_counts=self.counts,frames=6,anchor_policy=plan['anchors'],source_count=len(sources),
            nominal_cadence_ps=plan['nominal_cadence_ps'],nominal_offsets_ps=plan['nominal_offsets_ps'],source_kind=plan['source_kind'],
            observed_cadence_ps_by_kind=plan['observed_cadence_ps_by_kind'],interpolation=False,
            offsets_ps_by_source={s['id']:s['offsets_ps'] for s in sources},coordinate_storage='shared resident float32 bank; no extra quantization')
        if subset is not None:
            self.history_metadata['source_subset']=subset
            self.identity=digest(dict(parent=manifest['identity'],role=role,source_subset=subset,sources=self.source_names))

    def batch(self,ids):
        indices=self.index[ids]
        if self.mode=='repeated_current':indices=indices[:,-1:]
        return self.positions[indices],self.distance[ids],self.weight[ids]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','prepare','seal']);p.add_argument('--config',required=True);p.add_argument('--lane',type=int)
    a=p.parse_args();result=make_plan(a.config) if a.action=='plan' else prepare(a.config,a.lane) if a.action=='prepare' else seal(a.config)
    if a.action!='plan':print(json.dumps(result))
    else:print(json.dumps(dict(identity=result['identity'],sources=len(result['sources']),patches=sum(s['unique_patches'] for s in result['sources']))))
