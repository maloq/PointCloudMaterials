"""Large, source-separated temporal patches from the repository MD campaigns."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
from numpy.lib.format import open_memmap
from scipy.spatial import cKDTree

from src.data_utils.shooting_binary import ShootingBinaryTrajectory
from src.data_utils.temporal_lammps_binary import TemporalLAMMPSBinaryTrajectory
from src.analysis.liquid_structure import nonaffine_displacement

ROOT=Path(__file__).resolve().parents[2]


def write_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def patches(points,lengths,rows,neighbors):
    points=np.mod(points.astype(np.float64),lengths)
    tree=cKDTree(points,boxsize=lengths,balanced_tree=False)
    distances,ids=tree.query(points[rows],k=neighbors+2,workers=1)
    np.testing.assert_array_equal(ids[:,0],rows)
    # 4 and 6 A graphs require complete 8 and 12 A halos, including jitter.
    if distances[:,193].min()<=8.12 or distances[:,-1].min()<=12.12:
        raise RuntimeError(f'Two-hop halo truncated: 193rd={distances[:,193].min()}, outer={distances[:,-1].min()} A')
    x=points[ids[:,:neighbors+1]]-points[rows,None]
    x-=lengths*np.round(x/lengths)
    return x.astype(np.float32),ids[:,:neighbors+1]


def task_list(cfg):
    benchmark=ROOT/cfg['benchmark'];campaign=Path(cfg['shooting_campaign'])
    manifest=json.loads((campaign/'manifest.json').read_text())
    splits=json.loads((benchmark/'data_summary.json').read_text())['shooting_source_splits']
    tasks=[]
    for parent in manifest['parents']:
        split=next(s for s in splits if parent['source_index'] in splits[s])
        rows=np.load(benchmark/'parents'/f"parent_{parent['parent_index']:03d}.npz")['rows'] if split=='test' else np.sort(np.random.default_rng(cfg['data_seed']+parent['parent_index']).choice(70304,cfg['centers'][split],replace=False))
        for branch in manifest['branches']:
            if branch['parent_index']!=parent['parent_index']:continue
            directory=campaign/branch['branch_dir'];outcome=json.loads((directory/'outcome.json').read_text())
            if outcome['state']!='complete':raise RuntimeError(f'Incomplete source: {directory}')
            tasks.append(dict(name=f"shooting_{branch['branch_index']:04d}",kind='shooting',split=split,material=0,
                temperature=parent['temperature_K'],source=parent['source_index'],parent=parent['parent_index'],shot=branch['shot_index'],
                path=str(directory/outcome['trajectory_artifact']['path']),rows=rows.tolist(),anchors=[0] if split=='test' else [0,20,40,80],lags=[1,4,20,40],cadence_ps=.3))
    old=json.loads((ROOT/cfg['pilot']/'data/manifest.json').read_text())['shards']
    sources={s['snapshot']:s['source'] for s in old if s['material']=='Mg'}
    for i,snapshot in enumerate(sorted(sources,key=lambda s:float(s.removesuffix('ps')))[:5]):
        source=sources[snapshot];trajectory=TemporalLAMMPSBinaryTrajectory.load(source['path'])
        split='train' if i<4 else 'val';n=cfg['ordinary_centers'][split]
        rows=np.sort(np.random.default_rng(cfg['data_seed']+100+i).choice(trajectory.atom_count,n,replace=False))
        tasks.append(dict(name=f'Mg_{snapshot}_{split}',kind='ordinary',split=split,material=1,temperature=0.,source=100+i,parent=-1,shot=-1,
            path=source['path'],rows=rows.tolist(),anchors=[20,40,60,80],lags=[1,4,20,40],cadence_ps=.1))
    ta=next(s['source'] for s in old if s['material']=='Ta');trajectory=TemporalLAMMPSBinaryTrajectory.load(ta['path'])
    ntrain=4*cfg['ordinary_centers']['train'];nval=cfg['ordinary_centers']['val']
    rows=np.random.default_rng(cfg['data_seed']+200).choice(trajectory.atom_count,ntrain+nval,replace=False)
    for split,ids,anchors,lags in [('train',rows[:ntrain],[20,40,60,80],[1,4,20,40]),('val',rows[ntrain:],[150],[1,4,10,20])]:
        tasks.append(dict(name=f'Ta_{split}',kind='ordinary',split=split,material=2,temperature=0.,source=200,parent=-1,shot=-1,
            path=ta['path'],rows=np.sort(ids).tolist(),anchors=anchors,lags=lags,cadence_ps=.1))
    return tasks


def prepare_task(task,cfg):
    started=time.monotonic();directory=Path(cfg['cache'])/task['name'];directory.mkdir(exist_ok=False)
    trajectory=(ShootingBinaryTrajectory if task['kind']=='shooting' else TemporalLAMMPSBinaryTrajectory).load(task['path'])
    if task['kind']=='shooting' and trajectory.positions.dtype!=np.float32:raise TypeError('Shooting training must use float32 source positions')
    expected=np.arange(161)*100 if task['kind']=='shooting' else np.arange(241)*(50 if task['material']==2 else 100)
    np.testing.assert_array_equal(trajectory.timesteps,expected)
    rows=np.array(task['rows']);anchors=np.array(task['anchors']);lags=np.array(task['lags'])
    frames=np.unique(np.r_[anchors,(anchors[:,None]+lags).ravel()]);index={int(f):i for i,f in enumerate(frames)}
    n=len(rows);clouds=open_memmap(directory/'clouds.npy',mode='w+',dtype='float16',shape=(n,len(frames),cfg['neighbors']+1,3))
    identities={};initial={}
    for frame in frames:
        lengths=trajectory.box_high[frame].astype(np.float64)-trajectory.box_low[frame]
        x,ids=patches(trajectory.positions[frame],lengths,rows,cfg['neighbors']);clouds[:,index[int(frame)]]=x
        if frame in anchors:identities[int(frame)]=ids[:,:25];initial[int(frame)]=x[:,:25].astype(np.float64)
    clouds.flush();motion=np.empty((n,len(anchors),len(lags),2),dtype=np.float32)
    for ai,frame in enumerate(anchors):
        for li,lag in enumerate(lags):
            future=int(frame+lag);lengths=trajectory.box_high[future].astype(np.float64)-trajectory.box_low[future]
            y=trajectory.positions[future,identities[int(frame)][:,1:]].astype(np.float64)-trajectory.positions[future,rows,None]
            y-=lengths*np.round(y/lengths)
            d2=nonaffine_displacement(initial[int(frame)][:,1:],y)
            delta=trajectory.positions[future,rows].astype(np.float64)-trajectory.positions[frame,rows]
            delta-=lengths*np.round(delta/lengths)
            motion[:,ai,li,0]=np.log1p(d2);motion[:,ai,li,1]=np.log1p(np.square(delta).sum(1))
    np.save(directory/'motion.npy',motion)
    record={k:v for k,v in task.items() if k!='rows'}
    record.update(directory=str(directory),centers=n,frames=frames.tolist(),anchor_indices=[index[int(f)] for f in anchors],
        future_indices=[[index[int(f+l)] for l in lags] for f in anchors],
        storage_dtype=str(trajectory.positions.dtype),source_manifest_sha256=hashlib.sha256((Path(task['path'])/'manifest.json').read_bytes()).hexdigest(),seconds=time.monotonic()-started)
    record['cache_storage_dtype']='float16'
    np.save(directory/'center_rows.npy',rows);write_json(directory/'manifest.json',record)
    return record


def prepare_evaluation(cfg):
    """Rebuild wider halos for exactly the existing 10,240 benchmark centers."""
    cache=Path(cfg['cache']);base=ROOT/cfg['benchmark'];parents=json.loads((base/'parent_manifest.json').read_text())
    signature=dict(parent_manifest_sha256=hashlib.sha256((base/'parent_manifest.json').read_bytes()).hexdigest(),points=cfg['neighbors']+1,rows=10240)
    if (cache/'benchmark_manifest.json').exists():
        assert json.loads((cache/'benchmark_manifest.json').read_text())==signature
        assert np.load(cache/'benchmark_clouds.npy',mmap_mode='r').shape==(10240,cfg['neighbors']+1,3)
        return
    clouds=open_memmap(cache/'benchmark_clouds.npy',mode='w+',dtype='float16',shape=(10240,cfg['neighbors']+1,3))
    for parent in parents:
        index=parent['parent_index'];old=np.load(base/'parents'/f'parent_{index:03d}.npz')
        path=Path(sorted(parent['input_manifests'])[0]).parent;trajectory=ShootingBinaryTrajectory.load(path)
        lengths=trajectory.box_high[0].astype(np.float64)-trajectory.box_low[0]
        x,_=patches(trajectory.positions[0],lengths,old['rows'],cfg['neighbors'])
        np.testing.assert_allclose(x[:,:193],old['clouds'],rtol=0,atol=1e-5)
        clouds[index*256:(index+1)*256]=x
    clouds.flush()
    write_json(cache/'benchmark_manifest.json',signature)


def prepare(cfg,out):
    cache=Path(cfg['cache']);cache.mkdir(parents=True,exist_ok=True)
    tasks=task_list(cfg);write_json(out/'data_tasks.json',[{k:v for k,v in t.items() if k!='rows'} for t in tasks])
    records=[];pending=[]
    # Only reuse exact preflight tasks, with recorded source and row identities.
    for task in tasks:
        directory=cache/task['name']
        if directory.exists():
            record=json.loads((directory/'manifest.json').read_text())
            np.testing.assert_array_equal(np.load(directory/'center_rows.npy'),task['rows'])
            assert record['path']==task['path'] and record['anchors']==task['anchors'] and record['lags']==task['lags']
            assert record['source_manifest_sha256']==hashlib.sha256((Path(task['path'])/'manifest.json').read_bytes()).hexdigest()
            records.append(record)
        else:pending.append(task)
    with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(prepare_task,t,cfg) for t in pending]
        for future in as_completed(futures,timeout=cfg['preparation_seconds']):
            record=future.result();records.append(record)
            write_json(out/'prepare_status.json',dict(state='running',completed=len(records),total=len(tasks),last=record['name']))
            print('PREPARED',record['name'],len(records),'/',len(tasks),flush=True)
    prepare_evaluation(cfg)
    records.sort(key=lambda r:r['name']);write_json(cache/'manifest.json',dict(shards=records,config=cfg))
    write_json(out/'data_manifest.json',dict(shards=records,config=cfg))
    counts={split:{label:sum(r['centers']*len(r['frames']) for r in records if r['split']==split and r['material']==m)
        for m,label in enumerate(('Al','Mg','Ta'))} for split in ('train','val','test')}
    write_json(out/'data_summary.json',dict(cache=str(cache),clouds_by_split_material=counts,
        shooting_training_parent_centers=22*cfg['centers']['train'],neighborhood_points=cfg['neighbors']+1,
        supervision='Coordinates, temporal identity and log1p(D2min/MSD); no TDA, SOAP or bond-order targets',
        limits='Al/Mg validation sources are held out. Ta validation uses disjoint IDs and a later time block within one source. Mg/Ta coordinates are float16 source data.',
        evaluation='Original six Al test sources are excluded from training/pretraining; TDA/SOAP/BOO used only in frozen-feature evaluation'))
    write_json(out/'prepare_status.json',dict(state='complete',completed=len(records),total=len(tasks)))


class TemporalPairs:
    """Memory-map the explicit cache; sample materials equally without disk copies."""
    def __init__(self,cfg):
        self.cfg=cfg;self.shards=json.loads((Path(cfg['cache'])/'manifest.json').read_text())['shards']
        self.clouds=[np.load(Path(s['directory'])/'clouds.npy',mmap_mode='r') for s in self.shards]
        self.motion=[np.load(Path(s['directory'])/'motion.npy',mmap_mode='r') for s in self.shards]
        self.subsets=[np.random.default_rng(cfg['data_seed']+i+300).permutation(s['centers']) for i,s in enumerate(self.shards)]
        self.pools={split:[[i for i,s in enumerate(self.shards) if s['split']==split and s['material']==m] for m in range(3)] for split in ('train','val','test')}
        means=[];stds=[]
        for pool in self.pools['train']:
            values=np.concatenate([v.reshape(-1,2) for i,v in enumerate(self.motion) if i in pool])
            means.append(values.mean(0));stds.append(np.maximum(values.std(0),1e-3))
        self.mean=np.stack(means);self.std=np.stack(stds)

    def batch(self,split,n,rng,*,fraction=1.,points=193):
        x=np.empty((n,points,3),np.float32);y=np.empty_like(x);motion=np.empty((n,2),np.float32)
        material=np.zeros(n,dtype=np.int64) if split=='test' else np.arange(n,dtype=np.int64)%3
        rng.shuffle(material)
        condition=np.empty((n,5),np.float32)
        for m in (range(1) if split=='test' else range(3)):
            ids=np.flatnonzero(material==m);pool=self.pools[split][m]
            chosen=rng.choice(pool,len(ids))
            for si in np.unique(chosen):
                take=ids[chosen==si];s=self.shards[si];size=len(take)
                centers=self.subsets[si][rng.integers(max(1,int(s['centers']*fraction)),size=size)]
                anchor=rng.integers(len(s['anchors']),size=size);lag=rng.integers(len(s['lags']),size=size)
                a=np.array(s['anchor_indices'])[anchor];b=np.array(s['future_indices'])[anchor,lag]
                x[take]=self.clouds[si][centers,a,:points];y[take]=self.clouds[si][centers,b,:points]
                motion[take]=(self.motion[si][centers,anchor,lag]-self.mean[m])/self.std[m]
                condition[take,0]=s['temperature']/600.;condition[take,1]=np.log1p(np.array(s['lags'])[lag]*s['cadence_ps'])
                condition[take,2:]=np.eye(3,dtype=np.float32)[m]
        return x,y,material,condition,motion
