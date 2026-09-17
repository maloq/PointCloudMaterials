"""Fixed Al causal observations and future targets on the existing source split."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import numpy as np
from src.analysis.liquid_structure import persistence_image
from src.data.structural_pretraining.prepare import (REFERENCE_RADIUS,chart,offsets,source_arrays,geometry_packet,save_json,file_hash,digest)
from src.data.structural_pretraining.batches import Release
from src.project_runtime.paths import resolve_path,dataset_path

LAGS=(1,4,12)


def prepare_task(args):
    root,source,task,scale,identity,packet_path=args
    folder=Path(root)/'shards'/task['id'];receipt=folder/'complete.json'
    if receipt.exists():
        value=json.loads(receipt.read_text())
        if value['identity']!=identity:raise ValueError(f'Changed causal shard: {folder}')
        return value
    arrays=source_arrays(source);ids=arrays['atom_ids'];lookup={int(v):i for i,v in enumerate(ids)}
    centers=[lookup[c] for c in source['center_ids']];anchor=task['frame'];radius=17*scale/REFERENCE_RADIUS
    frames=[anchor-2,anchor-1,anchor,anchor+1]
    times=(arrays['timesteps'][frames]-arrays['timesteps'][anchor])*source['timestep_fs']/1000
    np.testing.assert_allclose(times,[-1.5,-.75,0,.75],rtol=0,atol=1e-8)
    pos=[];atomids=[];ptr=[0];mapping=np.full((len(centers),5),-1,np.int32)
    physical=[];tda=[];valid=[];view_centers=[];steps=[]
    future_p=np.empty((len(centers),3,85),np.float32);future_h=np.empty((len(centers),3,144),np.float32)
    for frame in frames+[anchor+4,anchor+12]:
        x,tree,box=chart(arrays,frame,False)
        if min(box)<=4*radius:raise ValueError('Causal observation exceeds periodic chart')
        lag=frame-anchor
        if lag in LAGS:
            delta=(arrays['timesteps'][frame]-arrays['timesteps'][anchor])*source['timestep_fs']/1000
            if abs(delta-lag*.75)>1e-8:raise ValueError('Nonuniform native target time')
        for row,center in enumerate(centers):
            neighbors=np.asarray(sorted(tree.query_ball_point(x[center],radius)),np.int64)
            local=offsets(x,center,neighbors,box)
            p=np.zeros(85,np.float32);h=np.zeros(144,np.float32)
            if lag>=0:
                p=geometry_packet(local)
                d2=np.square(local.astype(np.float64)).sum(-1)
                nearest=np.lexsort((ids[neighbors],d2))[:80]
                if len(nearest)!=80:raise ValueError('Insufficient physical TDA support')
                h=persistence_image(local[nearest])
            if lag in LAGS:
                slot=LAGS.index(lag);future_p[row,slot]=p;future_h[row,slot]=h
            if frame in frames:
                slot=frames.index(frame);mapping[row,slot]=len(physical)
                pos.append(local);atomids.append(ids[neighbors]);ptr.append(ptr[-1]+len(local))
                physical.append(p);tda.append(h);valid.append(lag>=0)
                view_centers.append(int(ids[center]));steps.append(int(arrays['timesteps'][frame]))
        del x,tree
    with np.load(packet_path) as packets:
        pool={int(c):i for i,c in enumerate(packets['atom_ids'])}
        classes=packets['labels'][[pool[c] for c in source['center_ids']],anchor]
    values=dict(positions=np.concatenate(pos).astype(np.float32),atom_ids=np.concatenate(atomids).astype(np.int64),
        offsets=np.array(ptr,np.int64),views=mapping,physical=np.stack(physical),tda=np.stack(tda),tda_valid=np.array(valid),
        center_ids=np.array(view_centers,np.int64),steps=np.array(steps,np.int64),times=times.astype(np.float64),
        future_physical=future_p,future_tda=future_h,current_ptm=classes)
    folder.mkdir(parents=True,exist_ok=True);hashes={}
    for key,value in values.items():
        if not np.isfinite(value).all():raise ValueError(f'Nonfinite causal array {key}')
        path=folder/(key+'.npy');temp=folder/(key+'.building.npy');np.save(temp,value,allow_pickle=False);temp.replace(path);hashes[key]=file_hash(path)
    value=dict(identity=identity,task=task,source=source['id'],lineage=source['lineage'],material='Al',potential='al-lee2003-meam',
        scale=scale,static=False,anchors=len(centers),views=len(physical),atoms=ptr[-1],labelled_views=sum(valid),hashes=hashes,
        temperature_K=source['temperature_K'])
    save_json(receipt,value);return value


def prepare(config,workers):
    root=resolve_path(config['release']);root.mkdir(parents=True,exist_ok=True)
    cohort=json.loads(resolve_path(config['cohort']).read_text())
    structural=json.loads((resolve_path(config['structural_release'])/'manifest.json').read_text())
    packets=json.loads(resolve_path(config['packet_release']).read_text());packet_by_id={r['id']:r for r in packets['sources']}
    assert packets['state']=='complete' and packets['cohort_sha256']==file_hash(resolve_path(config['cohort']))
    sources=[];tasks=[]
    for raw in cohort['sources']:
        split=raw['validation_role'] if raw['split']=='val' else raw['split']
        source=dict(id=raw['id'],lineage=raw['lineage'],kind='dynamic',path=str(dataset_path(raw['dataset'])/raw['relative_trajectory_path']),
            manifest_sha256=raw['manifest_sha256'],center_ids=raw['center_atom_ids'],timestep_fs=raw['timestep_fs'],temperature_K=raw['temperature_K'],split=split)
        sources.append(source)
        if packet_by_id[raw['id']]['manifest_sha256']!=raw['manifest_sha256']:raise ValueError('Packet/source provenance differs')
        for frame in cohort['native_anchors']:
            tasks.append(dict(id=f'{len(tasks):06d}',source=len(sources)-1,frame=frame,count=16,split=split))
    train_roots={s['lineage'] for s in sources if s['split']=='train'}
    heldout={s['lineage'] for s in sources if s['split']!='train'}
    if train_roots & heldout or len(sources)!=150 or len(train_roots)!=90:raise ValueError('Changed native whole-source split')
    if train_roots!={s['lineage'] for s in structural['sources'] if s['id'].startswith('native_') and s['split']=='train'}:
        raise ValueError('Structural and causal training ancestry differs')
    plan=dict(protocol='shared_causal_targets_v1',config=config,sources=sources,tasks=tasks,scale=structural['scales']['Al'],
        structural_identity=structural['identity'],cohort_sha256=file_hash(resolve_path(config['cohort'])),
        packet_release_sha256=file_hash(resolve_path(config['packet_release'])),
        producer_hashes={p:file_hash(p) for p in [__file__,'src/data/structural_pretraining/prepare.py','src/analysis/liquid_structure.py']})
    plan['identity']=digest(plan)
    if (root/'plan.json').exists() and json.loads((root/'plan.json').read_text())!=plan:raise ValueError('Changed immutable causal plan')
    save_json(root/'plan.json',plan)
    args=[(str(root),sources[t['source']],t,plan['scale'],plan['identity'],str(resolve_path(config['packet_cache'])/packet_by_id[sources[t['source']]['id']]['shard'])) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(prepare_task,a) for a in args]
        for i,future in enumerate(as_completed(futures),1):
            future.result();save_json(root/'status.json',dict(state='preparing',completed=i,total=len(tasks)))
            if i%50==0:print(f'Causal target shards {i}/{len(tasks)}',flush=True)
    records=[];current=[];future=[]
    for t in tasks:
        folder=root/'shards'/t['id'];r=json.loads((folder/'complete.json').read_text());records.append(r)
        if t['split']=='train':
            maps=np.load(folder/'views.npy')[:,2]
            current.append(np.concatenate((np.load(folder/'physical.npy')[maps],np.load(folder/'tda.npy')[maps]),-1))
            future.append(np.concatenate((np.load(folder/'future_physical.npy'),np.load(folder/'future_tda.npy')),-1))
    all_targets=np.concatenate((np.concatenate(current)[:,None],np.concatenate(future)),1).astype(np.float64)
    mean=all_targets.mean((0,1));std=np.maximum(all_targets.std((0,1)),1e-4)
    result=dict(state='complete',identity=plan['identity'],shards=records,sources=sources,scales=structural['scales'],
        normalization=structural['normalization'],forecast_normalization=dict(mean=mean.tolist(),std=std.tolist()),
        lags_ps=[.75,3.,9.],counts=dict(Counter({s:sum(r['anchors'] for r in records if r['task']['split']==s) for s in ['train','selection','calibration','test']})),config=config)
    save_json(root/'manifest.json',result);save_json(root/'status.json',dict(state='complete',counts=result['counts']))
    print(json.dumps(result['counts']),flush=True)


class CausalRelease(Release):
    def __init__(self,root):
        super().__init__(root)
        # The structural loader has only train/selection; rebuild explicit
        # four-way roles before any causal sampler can see these records.
        self.groups=defaultdict(list);self.splits={s:[] for s in ['train','selection','calibration','test']}
        for i,(_,_,r) in enumerate(self.rows):
            split=r['task']['split'];self.splits[split].append(i)
            if split=='train':self.groups[('Al','al-lee2003-meam',False)].append(i)
        self.selection=self.splits['selection'];self.group_keys=list(self.groups);self.group_weights=np.ones(1)

    def futures(self,indices):
        return {key:np.stack([self.arrays[self.rows[i][0]][key][self.rows[i][1]] for i in indices]) for key in ('future_physical','future_tda')}

    def targets(self):
        current=[];future=[];sources=[];splits=[];temperatures=[];classes=[];keys=[]
        for name,row,r in self.rows:
            a=self.arrays[name];view=a['views'][row,2]
            current.append(np.concatenate((a['physical'][view],a['tda'][view])))
            future.append(np.concatenate((a['future_physical'][row],a['future_tda'][row]),-1))
            sources.append(r['source']);splits.append(r['task']['split']);temperatures.append(r['temperature_K']);classes.append(a['current_ptm'][row])
            keys.append(f"{r['source']}:{int(a['center_ids'][view])}:{r['task']['frame']}")
        return dict(target=np.concatenate((np.stack(current)[:,None],np.stack(future)),1),source=np.array(sources),
            split=np.array(splits),temperature_K=np.array(temperatures),current_ptm=np.array(classes),row_id=np.array(keys))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--workers',type=int,default=6)
    args=p.parse_args();prepare(json.loads(Path(args.config).read_text()),args.workers)
