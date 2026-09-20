"""Dense, ancestry-held-out local onset observations, with reusable local graphs."""
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.spatial import cKDTree
import torch
from src.project_runtime.paths import resolve_path, dataset_path
from src.data.structural_pretraining.prepare import file_hash, save_json, digest
from src.data.structural_pretraining.support import SUPPORT, REFERENCE_RADIUS, OUTER_RADIUS, EDGE_CUTOFF, support_weights
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.research.forecast_crystallization.local_metrics import first_sustained_onset, risk_windows


def freeze(config):
    root=resolve_path(config['output']); tech=root/'technical'; tech.mkdir(parents=True,exist_ok=True)
    path=tech/'plan.json'
    if path.exists():
        plan=json.loads(path.read_text())
        if plan['config']!=config: raise ValueError('Crystallization transfer plan changed')
        return plan
    if 'reuse_plan' in config:
        original_path=resolve_path(config['reuse_plan']);original=json.loads(original_path.read_text())
        if config['cache']!=original['config']['cache']:raise ValueError('Reused cache path differs')
        parent=original_path.parent/'parent.pt'
        if file_hash(parent)!=original['checkpoint_sha256']:raise ValueError('Reused parent checkpoint changed')
        shutil.copy2(parent,tech/'parent.pt')
        plan={k:v for k,v in original.items() if k not in ('identity','config')}
        plan.update(config=config,cache_identity=original['identity'],reused_plan_sha256=file_hash(original_path))
        plan['identity']=digest(plan);save_json(path,plan)
        return plan
    release_path=resolve_path(config['assay_release']); release=json.loads(release_path.read_text())
    if release['state']!='complete' or len(release['sources'])!=150: raise ValueError('Require complete 150-source assay')
    gates=json.loads((release_path.parent/'assay_gates.json').read_text())
    if not all(gates[k] for k in ('source_integrity','assay_integrity','coverage')): raise ValueError(gates)
    saved=torch.load(resolve_path(config['checkpoint']),map_location='cpu',weights_only=False)
    identity=saved['identity']
    if identity['observation_support']!=SUPPORT: raise ValueError('Checkpoint local support differs')
    manifest=json.loads((resolve_path(identity['config']['release'])/'manifest.json').read_text())
    fitted={s['lineage'] for s in manifest['sources'] if s['split'] in ('train','selection') and 'lineage' in s}
    sources=release['sources']; held={s['lineage'] for s in sources if s.get('validation_role',s['split']) in ('test','calibration')}
    if fitted & held: raise ValueError(f'Pretraining ancestry leakage: {fitted & held}')
    by_lineage={}
    for s in sources:
        role=s.get('validation_role',s['split'])
        if role not in ('train','selection','calibration','test'): raise ValueError(role)
        if s['lineage'] in by_lineage: raise ValueError('Duplicate source ancestry')
        by_lineage[s['lineage']]=role
    # Verify actual encoder producers, not unrelated evolving runtime code.
    producer=resolve_path(config['producer_code'])
    encoder_files={p:h for p,h in identity['implementation']['files'].items() if p.startswith('src/models/')}
    for p,h in encoder_files.items():
        if file_hash(producer/p)!=h or file_hash(p)!=h: raise ValueError(f'Encoder implementation changed: {p}')
    shutil.copy2(resolve_path(config['checkpoint']),tech/'parent.pt')
    plan=dict(config=config,sources=sources,scale=manifest['scales']['Al'],support=SUPPORT,
        checkpoint_sha256=file_hash(tech/'parent.pt'),checkpoint_step=saved['step'],
        release_sha256=file_hash(release_path),encoder_files=encoder_files,
        anchors=list(range(64,665,config['origin_stride_frames'])),frames=list(range(0,665,4)),
        lags=[1,4,12,32,64,128],history_offsets=[-64,-16,-4,0],
        population='All 150 audited independent Al trajectories; 16 fixed outcome-independent centers/source; natural at-risk origins',
        pretraining_test_and_calibration_overlap=[])
    plan['identity']=digest(plan);save_json(path,plan)
    return plan


def representatives(points, center, tree, box):
    """Three farthest-point representatives per annulus, anchored by nearest atom."""
    candidates=np.asarray(tree.query_ball_point(points[center],25.),dtype=int)
    r=points[candidates]-points[center];r-=box*np.round(r/box);distance=np.linalg.norm(r,axis=1)
    result=[center]
    for lo,hi in [(0.,12.),(12.,25.)]:
        keep=(distance>lo)&(distance<=hi); ids=candidates[keep]; x=r[keep]
        if len(ids)<3: raise ValueError('Insufficient spatial context atoms')
        first=int(np.lexsort((ids,np.linalg.norm(x,axis=1)))[0]);chosen=[first]
        while len(chosen)<3:
            d=np.min(np.sum((x[:,None]-x[chosen][None])**2,axis=-1),axis=1);d[chosen]=-1
            chosen.append(int(np.argmax(d)))
        result.extend(ids[chosen].tolist())
    return np.array(result)


def prepare_source(args):
    source,plan=args; config=plan['config']; sid=source['id'];root=resolve_path(config['cache'])/str(sid)
    receipt=root/'complete.json'
    if receipt.exists():
        record=json.loads(receipt.read_text())
        if record['identity']!=plan['identity']: raise ValueError('Prepared source identity differs')
        return sid
    root.mkdir(parents=True,exist_ok=True)
    rawpath=dataset_path(source['dataset'])/source['relative_trajectory_path']
    if file_hash(rawpath/'manifest.json')!=source['manifest_sha256']:raise ValueError(f'Raw manifest changed: {sid}')
    raw=ShootingBinaryTrajectory.load(rawpath)
    labelpath=resolve_path(config['assay_cache'])/source['shard']
    if file_hash(labelpath)!=source['shard_sha256']:raise ValueError(f'Assay changed: {sid}')
    with np.load(labelpath) as a:
        labels=np.array(a['labels']); ids=np.array(a['atom_ids']);packet=np.array(a['packet']);order=np.array(a['order'])
    np.testing.assert_array_equal(ids,source['center_atom_ids'])
    center_rows=np.searchsorted(raw.atom_ids,ids);np.testing.assert_array_equal(raw.atom_ids[center_rows],ids)
    np.testing.assert_allclose(raw.timesteps*source['timestep_fs']/1000,np.arange(801)*.75,rtol=0,atol=1e-6)
    xs=[];es=[];ptr=[0];eptr=[0];centers=[];coordinates=[];boxes=[];mapping=[]
    factor=REFERENCE_RADIUS/plan['scale']
    for frame in plan['frames']:
        box=(raw.box_high[frame]-raw.box_low[frame]).astype(float)
        points=np.mod(raw.positions[frame].astype(float),box);tree=cKDTree(points,boxsize=box)
        frame_map=[];relative=[]
        for center in center_rows:
            neighbors=representatives(points,center,tree,box)
            rel=points[neighbors]-points[center];rel-=box*np.round(rel/box);relative.append(rel)
            current=[]
            for neighbor in neighbors:
                keep=np.array(tree.query_ball_point(points[neighbor],OUTER_RADIUS/factor),dtype=int)
                # Put center first, then stable atom identity order.
                keep=np.r_[neighbor,np.sort(keep[keep!=neighbor])]
                x=points[keep]-points[neighbor];x-=box*np.round(x/box);x=(x*factor).astype(np.float32)
                x=x[np.linalg.norm(x,axis=1)<OUTER_RADIUS]
                pairs=cKDTree(x).query_pairs(EDGE_CUTOFF,output_type='ndarray')
                edges=np.concatenate((pairs,pairs[:,::-1]),axis=0).T.astype(np.int32)
                current.append(len(xs));xs.append(x);es.append(edges);ptr.append(ptr[-1]+len(x));eptr.append(eptr[-1]+edges.shape[1])
            frame_map.append(current)
        mapping.append(frame_map);coordinates.append(relative);centers.append(points[center_rows]);boxes.append(box)
    crystal=np.isin(labels,[1,2,3]);anchors=np.array(plan['anchors']);onset=first_sustained_onset(crystal,3)
    risk=risk_windows(crystal,onset,anchors,3).T;delay=onset[None,:]-anchors[:,None]
    event=np.where((delay>0)&(delay<=128),np.searchsorted(plan['lags'],delay),6)
    arrays=dict(positions=np.concatenate(xs),edges=np.concatenate(es,axis=1),offsets=np.array(ptr),edge_offsets=np.array(eptr),
        mapping=np.array(mapping),relative=np.array(coordinates,dtype=np.float32),centers=np.array(centers,dtype=np.float32),boxes=np.array(boxes,dtype=np.float32),
        risk=risk,event=event.astype(np.int64),onset=onset,atom_ids=ids,packet=packet,order=order,labels=labels)
    for name,a in arrays.items():np.save(root/f'{name}.npy',a,allow_pickle=False)
    save_json(receipt,dict(identity=plan['identity'],source_id=sid,graphs=len(xs),positions=ptr[-1],eligible=int(risk.sum()),
        source_manifest_sha256=source['manifest_sha256'],assay_sha256=source['shard_sha256'],
        hashes={f'{k}.npy':file_hash(root/f'{k}.npy') for k in arrays}))
    return sid


class Corpus:
    def __init__(self,plan,require_features=True):
        self.plan=plan;self.root=resolve_path(plan['config']['cache']);self.arrays={};self.rows=[];self.splits={k:[] for k in ('train','selection','calibration','test')}
        for source in plan['sources']:
            sid=source['id'];folder=self.root/str(sid)
            r=json.loads((folder/'complete.json').read_text())
            if r['identity']!=plan.get('cache_identity',plan['identity']):raise ValueError(f'Source identity differs: {sid}')
            a={p.stem:np.load(p,mmap_mode='r') for p in folder.glob('*.npy')}
            if require_features:
                rec=json.loads((folder/'features.json').read_text())
                if rec['checkpoint_sha256']!=plan['checkpoint_sha256']:raise ValueError('Feature checkpoint differs')
            self.arrays[sid]=a
            for ai,ci in zip(*np.where(a['risk'])):
                index=len(self.rows);self.rows.append((sid,int(ai),int(ci),source['temperature_K']))
                self.splits[source.get('validation_role',source['split'])].append(index)
        self.source_ids=np.array([r[0] for r in self.rows]);self.events=np.array([self.arrays[s]['event'][a,c] for s,a,c,_ in self.rows])
        self.groups={s:np.flatnonzero((self.source_ids==s)&np.isin(np.arange(len(self.rows)),self.splits['train'])) for s in np.unique(self.source_ids[self.splits['train']])}
        self.groups={s:i for s,i in self.groups.items() if len(i)}

    def sample(self,rng,n):
        sources=rng.choice(list(self.groups),n)
        return [int(rng.choice(self.groups[s])) for s in sources]

    def inputs(self,indices,spec,raw=False):
        offsets={0:[0],3:[-4,0],12:[-16,-4,0],48:[-64,-16,-4,0]}[spec['history_ps']]
        radius=spec['radius_A']
        if not 0<=radius<=25:raise ValueError(f'Context radius {radius} exceeds the cached 25 Å support')
        # Keep the fixed geometry-selected candidate pool across radius comparisons.
        # The head gives keys outside the requested radius exactly zero weight.
        nodes=1 if radius==0 else 7
        features=[];geometry=[];graphs=[];condition=[];events=[];descriptor=[]
        for index in indices:
            sid,ai,ci,temp=self.rows[index];a=self.arrays[sid];anchor=self.plan['anchors'][ai]
            fi=np.array([anchor+(0 if spec.get('repeat',False) else o) for o in offsets])//4
            mapping=a['mapping'][fi,ci,:nodes];relative=a['relative'][fi,ci,:nodes]
            dt=np.broadcast_to(np.array(offsets)*.75,(nodes,len(offsets))).T
            geometry.append(np.concatenate((relative,dt[...,None]),axis=-1).reshape(-1,4))
            if raw:
                graphs.extend(self.graph(sid,int(g)) for g in mapping.ravel())
            else:features.append(np.array(a['features'][mapping]).reshape(-1,416))
            condition.append([*[float(temp==t) for t in (400,450,500,510,520)],anchor*.75/600,(anchor*.75/600)**2])
            events.append(a['event'][ai,ci]);descriptor.append(np.r_[a['packet'][ci,anchor],a['order'][ci,anchor]])
        result=dict(geometry=torch.tensor(np.array(geometry),dtype=torch.float32),condition=torch.tensor(condition,dtype=torch.float32),event=torch.tensor(events,dtype=torch.long),descriptor=torch.tensor(np.array(descriptor),dtype=torch.float32))
        if raw:result['graphs']=graphs
        else:result['features']=torch.from_numpy(np.array(features))
        return result

    def graph(self,sid,g):
        return graph(self.arrays[sid],g,self.plan['scale'])


def graph(a,g,scale):
    lo,hi=a['offsets'][g:g+2];elo,ehi=a['edge_offsets'][g:g+2]
    x=np.array(a['positions'][lo:hi]);edges=np.array(a['edges'][:,elo:ehi],dtype=np.int64)
    return dict(positions=x[None],weights=support_weights(x)[None],center=0,times=np.array([0.],np.float32),species=1,
        log_scale=np.log(scale/REFERENCE_RADIUS),physical=np.zeros(85,np.float32),tda=np.zeros(144,np.float32),tda_valid=False,edges=edges)
