"""Material subsets and unseen Ta velocity branches from a known preparation.

The Ta assay holds out trajectories, not ancestry. Its original starting
configuration was observed by the parent encoder; receipts preserve that fact.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from src.data.fixed_cohort.protocol import centered, digest, sha, write_json
from src.data.structural_pretraining.prepare import source_arrays, chart
from src.data.trajectories.lammps import TemporalLAMMPSBinaryTrajectory
from src.project_runtime.paths import resolve_path
from src.research.crystallization_origin import ancestry, external_data, external_audit
from .dense_history import DenseHistoryDataset


def make_plan(config_path):
    c=json.loads(Path(config_path).read_text());settings=c['ta_evaluation']
    root=resolve_path(settings['root']);audit=resolve_path(settings['audit_root'])
    audit_config=json.loads(resolve_path(settings['audit_config']).read_text())
    audit_config['output']=str(audit)
    scales=c['structural_dataset']['normalization']['scales_A']
    parent=json.loads((resolve_path(c['dense_history']['root'])/'plan.json').read_text())
    seen={resolve_path(s['source']['path']).resolve() for s in parent['sources']}
    sources=[];tasks=[];centers=None;identities=None;seeds=[]
    for item in settings['sources']:
        path=resolve_path(item['path']);manifest=json.loads((path/'manifest.json').read_text())
        if path.resolve() in seen:raise ValueError(f'Ta evaluation trajectory was used in parent training: {path}')
        meta=manifest['provenance']['metadata'];origin=meta['origin']
        if meta['material']!='Ta' or origin['source_sha256']!=settings['parent_positions_sha256']:
            raise ValueError(f'Unexpected Ta shooting parent: {path}')
        if origin['velocity_seed'] in seeds:raise ValueError('Evaluation repeats a velocity seed')
        seeds.append(origin['velocity_seed'])
        part=dict(id=item['id'],path=str(path),kind='dynamic',manifest_sha256=sha(path/'manifest.json'))
        raw=source_arrays(part);ids=raw['atom_ids']
        times=(raw['timesteps']-raw['timesteps'][0])*meta['timestep_ps']
        np.testing.assert_allclose(times,np.arange(241)*.1,rtol=0,atol=1e-8)
        if centers is None:
            identities=np.array(ids)
            centers=np.sort(np.random.default_rng(settings['seed']).choice(ids,settings['centers'],replace=False))
        np.testing.assert_array_equal(ids,identities)
        source=dict(id=item['id'],parts=[part],material='Ta',role=item['role'],atom_count=len(ids),
            frame_count=49,frames=[[0,f] for f in range(0,241,5)],times_ps=(np.arange(49)*.5).tolist(),
            cadence_ps=.5,center_atom_ids=centers.tolist(),lineage=external_data.settings(audit_config,'Ta',scales),
            ancestry_group=origin['ancestry_group'],parent_id=origin['parent_id'],velocity_seed=origin['velocity_seed'],
            potential=meta['potential'],raw_timestep_ps=meta['timestep_ps'],
            normalization_factor=scales['Al']/scales['Ta'])
        sources.append(source)
        for start in range(0,49,audit_config['chunk_frames']):
            tasks.append(dict(source=item['id'],start=start,stop=min(start+audit_config['chunk_frames'],49)))
    if [s['role'] for s in sources].count('selection')!=1 or [s['role'] for s in sources].count('test')!=3:
        raise ValueError('Ta pilot requires one selection and three test shooting branches')
    plan=dict(config=audit_config,sources=sources,tasks=tasks,cache_root=str(root),
        anchors=settings['anchor_raw_frames'],offset_frames=[-35,-28,-21,-14,-7,0],
        observed_cadence_ps=.7,history_span_ps=3.5,interpolation=False,
        parent_training_plan_sha256=sha(resolve_path(c['dense_history']['root'])/'plan.json'),
        source_contract=settings,producer_sha256=sha(__file__),
        generalization='unseen velocity branches of one known parent; NOT ancestry-independent or unseen-material transfer')
    plan['identity']=digest(plan)
    for dest in (root/'plan.json',audit/'technical/plan.json'):
        if dest.exists() and json.loads(dest.read_text())!=plan:raise ValueError(f'Changed Ta evaluation release: {dest}')
        write_json(dest,plan)
    return plan


def prepare_source(config_path,index):
    c=json.loads(Path(config_path).read_text());root=resolve_path(c['ta_evaluation']['root'])
    plan=json.loads((root/'plan.json').read_text());source=plan['sources'][index]
    folder=root/'sources'/source['id'];folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'complete.json'
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old['identity']!=plan['identity'] or any(sha(folder/n)!=h for n,h in old['files'].items()):
            raise ValueError(f'Changed Ta evaluation geometry: {folder}')
        return old
    part=source['parts'][0];raw=source_arrays(part)
    # The trajectory producer hashes array contents, excluding NPY headers.
    TemporalLAMMPSBinaryTrajectory.load(part['path']).verify_checksums()
    for task in plan['tasks']:
        if task['source']==source['id']:
            external_data.extract_chunk(plan,task)
            write_json(folder/'state.json',dict(stage='ptm',frames=task['stop'],total=49))
    external_audit.seal_chunks(plan,source)
    graph=external_audit.graph_for_source(plan,source,digest(external_audit.contract(plan)))
    events,roots,_=ancestry.establish(graph,source['lineage']['thresholds'][0])
    access=external_audit.FrameAccess(plan,source,graph)
    anchors=plan['anchors'];offsets=np.asarray(plan['offset_frames']);centers=np.asarray(source['center_atom_ids'])
    center_rows=np.searchsorted(raw['atom_ids'],centers)
    frames=sorted({int(f+k) for f in anchors for k in offsets})
    if min(frames)<0 or max(frames)>=len(raw['positions']):raise ValueError('Unavailable Ta history')
    factor=source['normalization_factor'];n=len(centers)
    bank=np.lib.format.open_memmap(folder/'positions.building.npy',mode='w+',dtype=np.float32,shape=(len(frames),n,80,3))
    for i,frame in enumerate(frames):
        points,tree,box=chart(raw,frame,False)
        for start in range(0,n,2048):
            rows=center_rows[start:start+2048];neighbors=tree.query(points[rows],k=80,workers=1)[1]
            np.testing.assert_array_equal(neighbors[:,0],rows)
            xyz=centered(points,box,rows,neighbors)*factor
            if not np.isfinite(xyz).all() or np.any(xyz[:,0]):raise ValueError('Invalid Ta coordinates')
            bank[i,start:start+len(rows)]=xyz
        bank.flush();del points,tree
        write_json(folder/'state.json',dict(stage='coordinates',frames=i+1,total=len(frames)))
    del bank;(folder/'positions.building.npy').replace(folder/'positions.npy')
    distance=[];indices=[]
    for frame in anchors:
        if frame%5:raise ValueError('Distance anchor absent from 0.5-ps lineage audit')
        af=frame//5;points,box,dense=access.frame(af)
        nodes=np.flatnonzero((graph['frame']==af)&(graph['size']>=64))
        known=[v for v in nodes if any(events[r-1]['confirmation_frame']<=af for r in roots[v])]
        solid=np.isin(dense,known) if known else np.zeros(len(points),bool)
        d=cKDTree(points[solid],boxsize=box).query(points[center_rows],workers=1)[0] if solid.any() else np.full(n,np.inf)
        distance.append((d*factor).astype(np.float32))
        indices.append(np.arange(n)[:,None]+n*np.asarray([frames.index(frame+k) for k in offsets])[None])
    np.savez(folder/'samples.npz',index=np.concatenate(indices),distance=np.concatenate(distance),
        atom=np.tile(centers,len(anchors)),raw_frame=np.repeat(anchors,n))
    result=dict(identity=plan['identity'],source=source['id'],role=source['role'],rows=n*len(anchors),
        cadence_ps=.7,offsets_ps=(offsets*.1).tolist(),files={name:sha(folder/name) for name in ('positions.npy','samples.npz')})
    write_json(receipt,result);write_json(folder/'state.json',dict(state='complete',**result));return result


def seal(config_path):
    c=json.loads(Path(config_path).read_text());root=resolve_path(c['ta_evaluation']['root'])
    plan=json.loads((root/'plan.json').read_text());receipts=[]
    for source in plan['sources']:
        folder=root/'sources'/source['id'];r=json.loads((folder/'complete.json').read_text())
        if r['identity']!=plan['identity'] or any(sha(folder/n)!=h for n,h in r['files'].items()):raise ValueError(f'Unsealed Ta branch: {folder}')
        receipts.append(r)
    write_json(root/'manifest.json',dict(state='complete',identity=plan['identity'],plan_sha256=sha(root/'plan.json'),sources=receipts))


class ShootingHistoryDataset:
    def __init__(self,c,role,device):
        root=resolve_path(c['ta_evaluation']['root']);plan=json.loads((root/'plan.json').read_text())
        manifest=json.loads((root/'manifest.json').read_text())
        if manifest['state']!='complete' or sha(root/'plan.json')!=manifest['plan_sha256'] or digest({k:v for k,v in plan.items() if k!='identity'})!=manifest['identity']:
            raise ValueError('Ta shooting release changed/incomplete')
        if plan['source_contract']!=c['ta_evaluation']:raise ValueError('Ta evaluation config differs from sealed release')
        sources=[s for s in plan['sources'] if s['role']==role]
        if not sources:raise ValueError(f'No shooting branches for {role}')
        banks=[];indices=[];distances=[];ids=[];atoms=[];frames=[];offset=0
        for sid,s in enumerate(sources):
            folder=root/'sources'/s['id'];r=next(r for r in manifest['sources'] if r['source']==s['id'])
            if any(sha(folder/n)!=h for n,h in r['files'].items()):raise ValueError(f'Changed Ta evaluation bank: {folder}')
            bank=np.load(folder/'positions.npy',mmap_mode='r').reshape(-1,80,3)
            banks.append(torch.from_numpy(np.array(bank)).to(device))
            with np.load(folder/'samples.npz') as a:
                indices.append(a['index']+offset);distances.append(a['distance']);ids.extend([sid]*len(a['distance']))
                atoms.append(a['atom']);frames.append(a['raw_frame'])
            offset+=len(bank)
        self.positions=torch.cat(banks);self.index=torch.as_tensor(np.concatenate(indices),device=device)
        self.distance=torch.as_tensor(np.concatenate(distances),device=device);self.n=len(self.distance)
        self.source_ids=np.asarray(ids);self.source_names=[s['id'] for s in sources]
        self.material_names=['Ta'];self.material_ids=np.zeros(self.n,np.int8);self.counts={'Ta':self.n}
        self.atoms=np.concatenate(atoms);self.raw_frames=np.concatenate(frames)
        from src.research.local_predictability.metrics import source_weights
        self.weight=torch.as_tensor((source_weights(self.source_ids)*self.n).astype(np.float32),device=device)
        self.identity=manifest['identity'];self.role=role
        d=self.distance;self.target_counts={'Ta':dict(rows=self.n,zero=int((d==0).sum()),within20=int((d<=20).sum()),censored64=int((d>=64).sum()))}
        self.history_metadata=dict(rows=self.n,source_count=len(sources),source_names=self.source_names,material_counts=self.counts,
            offsets_ps=[-3.5,-2.8,-2.1,-1.4,-.7,0],cadence_ps=.7,interpolation=False,
            generalization=plan['generalization'],ancestry_families=1,dataset_identity=self.identity)

    def batch(self,ids):return self.positions[self.index[ids]],self.distance[ids],self.weight[ids]


def MaterialHistoryDataset(c,role,device):
    if c['material_finetune']['material']=='Ta' and role!='train':return ShootingHistoryDataset(c,role,device)
    return DenseHistoryDataset(c,role,device)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['plan','source','seal']);p.add_argument('--config',required=True);p.add_argument('--index',type=int)
    a=p.parse_args();result=make_plan(a.config) if a.stage=='plan' else prepare_source(a.config,a.index) if a.stage=='source' else seal(a.config)
    print(json.dumps(result if a.stage!='plan' else dict(identity=result['identity'],sources=len(result['sources']))))
