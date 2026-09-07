"""Build source-disjoint liquid geometry and repeated-future benchmark data."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial import cKDTree
from src.analysis.liquid_structure import bond_order,persistence_image,nonaffine_displacement
from src.data_utils.shooting_binary import ShootingBinaryTrajectory
from experiments.smooth_temporal_encoder_20260905.prepare import write_json


def frame_geometry(points,lengths,rows,coordination_radius):
    points=np.mod(points.astype(np.float64),lengths)
    tree=cKDTree(points,boxsize=lengths,balanced_tree=False)
    distances,ids=tree.query(points[rows],k=194,workers=1)
    assert np.array_equal(ids[:,0],rows),'Central atom identity changed in neighbor query'
    # 2*4 A receptive field plus the maximum bounded input jitter.
    margin=float(distances[:,-1].min()-8.)
    if margin<=np.sqrt(3)*.06:
        raise RuntimeError(f'192-neighbor buffer truncates the complete two-hop receptive field: margin={margin} A')
    cloud=points[ids[:,:193]]-points[rows,None]
    cloud-=lengths*np.round(cloud/lengths)
    core=ids[:,:13]
    _,neighbors=tree.query(points[core],k=13,workers=1)
    vectors=points[neighbors[:,:,1:]]-points[core[:,:,None]]
    vectors-=lengths*np.round(vectors/lengths)
    order,connections=bond_order(vectors,coordination_radius)
    return cloud.astype(np.float32),ids[:,:193],order,connections,margin


def local_geometry(cloud,coordination_radius):
    tree=cKDTree(cloud)
    _,neighbors=tree.query(cloud[:13],k=13)
    vectors=cloud[neighbors[:,1:]]-cloud[:13,None]
    order,connections=bond_order(vectors[None],coordination_radius)
    return order[0],connections[0],persistence_image(cloud[:65])


def prepare_parent(cfg,parent,branches):
    started=time.monotonic()
    root=Path(cfg['shooting_campaign'])
    count=cfg['centers_per_parent']
    rows=np.sort(np.random.default_rng(cfg['seed']+parent['parent_index']).choice(70304,count,replace=False))
    files=[]
    for branch in sorted(branches,key=lambda b:b['shot_index']):
        directory=root/branch['branch_dir']
        outcome=json.loads((directory/'outcome.json').read_text())
        if outcome['state']!='complete':
            raise RuntimeError(f'Incomplete shooting branch: {directory}')
        trajectory=ShootingBinaryTrajectory.load(directory/outcome['trajectory_artifact']['path'])
        assert trajectory.positions.dtype==np.float32
        assert np.array_equal(trajectory.timesteps,np.arange(161)*100)
        files.append(trajectory)
    assert len(files)==8
    first=files[0]
    lengths=first.box_high[0].astype(np.float64)-first.box_low[0]
    cloud,ids,order,connections,margin=frame_geometry(first.positions[0],lengths,rows,3.7)
    tda=np.stack([persistence_image(x[:65]) for x in cloud])
    future_order=np.empty((count,8,3,8),dtype=np.float32)
    future_tda=np.empty((count,8,3,144),dtype=np.float32)
    mobility=np.empty((count,8,3,2),dtype=np.float32)
    acquisition=np.empty((count,8,3,3),dtype=np.uint8)
    for shot,trajectory in enumerate(files):
        np.testing.assert_array_equal(trajectory.positions[0,rows],first.positions[0,rows])
        np.testing.assert_array_equal(trajectory.box_high,first.box_high)
        for hi,frame in enumerate(cfg['horizon_frames']):
            orders,counts=[],[]
            for t in (frame-2,frame-1,frame):
                later,later_ids,later_order,later_conn,later_margin=frame_geometry(trajectory.positions[t],lengths,rows,3.7)
                orders.append(later_order);counts.append(later_conn)
                if t==frame:
                    future_tda[:,shot,hi]=np.stack([persistence_image(x[:65]) for x in later])
                    y=trajectory.positions[t,ids[:,1:25]].astype(np.float64)-trajectory.positions[t,rows,None]
                    y-=lengths*np.round(y/lengths)
                    d2=nonaffine_displacement(cloud[:,1:25],y)
                    displacement=trajectory.positions[t,rows].astype(np.float64)-first.positions[0,rows]
                    displacement-=lengths*np.round(displacement/lengths)
                    mobility[:,shot,hi,0]=np.log1p(d2)
                    mobility[:,shot,hi,1]=np.log1p(np.sum(displacement**2,1))
            future_order[:,shot,hi]=np.mean(orders,0)
            acquisition[:,shot,hi]=np.all(np.stack(counts)>=7,axis=0)
    destination=ROOT/cfg['output']/'parents'/f"parent_{parent['parent_index']:03d}.npz"
    np.savez(destination,clouds=cloud,order=order,connections=connections,tda=tda,rows=rows,
             future_order=future_order,future_tda=future_tda,mobility=mobility,acquisition=acquisition)
    return dict(parent_index=parent['parent_index'],source_index=parent['source_index'],
        source_run_id=parent['source_run_id'],temperature_K=parent['temperature_K'],
        velocity_seed=parent['source_velocity_seed'],source_split=parent['source_split'],
        phase=parent['phase'],count=count,minimum_halo_margin_A=margin,
        input_manifests={str(t.root/'manifest.json'):hashlib.sha256((t.root/'manifest.json').read_bytes()).hexdigest() for t in files},
        seconds=time.monotonic()-started)


def prepare_supplement(cfg,shard,index,split):
    assert shard['radius_A']+shard['minimum_candidate_margin_A']>8.11
    pilot=ROOT/cfg['pilot']
    x=np.load(pilot/'data'/f"{shard['stem']}.clouds.npy",mmap_mode='r')
    frames=np.linspace(0,x.shape[1]-1,4,dtype=int)
    centers=min(len(x),128)
    offsets=np.asarray(x[:centers,frames,0])*shard['radius_A']
    offsets=offsets.reshape(-1,192,3)
    clouds=np.concatenate((np.zeros((len(offsets),1,3),dtype=np.float32),offsets),1)
    order,connections,tda=zip(*(local_geometry(c,cfg['coordination_radius_A'][shard['material']]) for c in clouds))
    np.savez(ROOT/cfg['output']/'supplements'/f'supplement_{index:03d}.npz',clouds=clouds,order=np.stack(order),connections=np.stack(connections),tda=np.stack(tda))
    return dict(index=index,kind='ordinary',split=split,material=shard['material'],source=shard['source'],
                source_group=f"{shard['material']}_{shard['snapshot']}",count=len(clouds),storage_dtype=shard['source']['storage_dtype'])


def prepare_static(cfg,frame,index,split):
    pilot=ROOT/cfg['pilot']
    points=np.load(frame['path'])
    coords=np.load(pilot/'full_static_Al/coords.npy',mmap_mode='r')
    pick=np.random.default_rng(cfg['seed']+index).choice(frame['count'],cfg['static_centers_per_source'],replace=False)+frame['offset']
    _,ids=cKDTree(points,balanced_tree=False).query(coords[pick],k=194,workers=1)
    clouds=(points[ids[:,:193]]-coords[pick,None]).astype(np.float32)
    assert np.linalg.norm(points[ids[:,-1]]-coords[pick],axis=1).min()>8.11
    order,connections,tda=zip(*(local_geometry(c,3.7) for c in clouds))
    np.savez(ROOT/cfg['output']/'supplements'/f'supplement_{index:03d}.npz',clouds=clouds,order=np.stack(order),connections=np.stack(connections),tda=np.stack(tda))
    return dict(index=index,kind='static',split=split,material='Al',source=frame['path'],source_group=f"Al_{frame['source'].replace('.npy','')}",count=len(clouds),storage_dtype='float32')


def run(cfg,out):
    root=Path(cfg['shooting_campaign'])
    manifest=json.loads((root/'manifest.json').read_text())
    assert json.loads((root/'summary.json').read_text())['state']=='complete'
    (out/'parents').mkdir(exist_ok=False)
    (out/'supplements').mkdir(exist_ok=False)
    parents=[]
    with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(prepare_parent,cfg,p,[b for b in manifest['branches'] if b['parent_index']==p['parent_index']]) for p in manifest['parents']]
        for future in as_completed(futures):
            value=future.result();parents.append(value)
            print('Prepared parent',value['parent_index'],f"{len(parents)}/40",value['seconds'],flush=True)
            write_json(out/'parent_manifest.json',parents)
    old=json.loads((ROOT/cfg['pilot']/'data/manifest.json').read_text())
    snapshots={m:sorted({s['snapshot'] for s in old['shards'] if s['material']==m},key=lambda s:float(s.replace('ps',''))) for m in ('Al','Mg')}
    assignment={f'{m}_{s}':('train' if i<4 else 'val' if i==4 else 'test') for m,ss in snapshots.items() for i,s in enumerate(ss)}
    tasks=[]
    for s in old['shards']:
        split=s['split'] if s['material']=='Ta' else assignment[f"{s['material']}_{s['snapshot']}"]
        if s['split']==split:
            tasks.append((s,split))
    supplements=[]
    with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(prepare_supplement,cfg,s,i,split) for i,(s,split) in enumerate(tasks)]
        coverage=json.loads((ROOT/cfg['pilot']/'full_static_Al/coverage.json').read_text())
        futures.extend(pool.submit(prepare_static,cfg,f,len(tasks)+i,assignment[f"Al_{f['source'].replace('.npy','')}"]) for i,f in enumerate(coverage['frames']))
        for future in as_completed(futures):
            value=future.result();supplements.append(value)
            print('Prepared supplement',value['index'],value['kind'],value['material'],flush=True)
    write_json(out/'supplement_manifest.json',supplements)
    parts={k:[] for k in ('clouds','order','connections','tda')}
    metadata={k:[] for k in ('split','material','temperature','source','parent','kind')}
    targets={k:[] for k in ('future_order','future_tda','mobility','acquisition')}
    for p in sorted(parents,key=lambda p:p['parent_index']):
        saved=np.load(out/'parents'/f"parent_{p['parent_index']:03d}.npz")
        split='test' if p['source_split']=='validation' else 'val' if p['velocity_seed']==cfg['selection_source_velocity_seed'] else 'train'
        for k in parts:parts[k].append(saved[k])
        for k in targets:targets[k].append(saved[k])
        for k,v in dict(split=split,material=0,temperature=p['temperature_K'],source=p['source_index'],parent=p['parent_index'],kind='shooting').items():metadata[k].extend([v]*p['count'])
    for s in sorted(supplements,key=lambda s:s['index']):
        saved=np.load(out/'supplements'/f"supplement_{s['index']:03d}.npz")
        for k in parts:parts[k].append(saved[k])
        for k,v in dict(split=s['split'],material={'Al':0,'Mg':1,'Ta':2}[s['material']],temperature=0.,source=100+s['index'],parent=-1,kind=s['kind']).items():metadata[k].extend([v]*s['count'])
    for k,v in parts.items():np.save(out/f'{k}.npy',np.concatenate(v))
    np.savez(out/'metadata.npz',**{k:np.array(v) for k,v in metadata.items()})
    np.savez(out/'dynamic_targets.npz',**{k:np.concatenate(v) for k,v in targets.items()})
    source_splits={s:{int(p['source_index']) for p in parents if ('test' if p['source_split']=='validation' else 'val' if p['velocity_seed']==cfg['selection_source_velocity_seed'] else 'train')==s} for s in ('train','val','test')}
    assert not (source_splits['train']&source_splits['val'] or source_splits['train']&source_splits['test'] or source_splits['val']&source_splits['test'])
    write_json(out/'data_summary.json',dict(rows=len(metadata['split']),shooting_rows=sum(p['count'] for p in parents),
        split_counts={s:metadata['split'].count(s) for s in ('train','val','test')},shooting_source_splits={k:sorted(v) for k,v in source_splits.items()},
        configuration=cfg,shooting_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        primary_position_dtype='float32',topology_library='gudhi 3.11.0'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    out=ROOT/cfg['output'];out.mkdir(exist_ok=False)
    write_json(out/'config.json',cfg)
    status=dict(state='running',pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat())
    write_json(out/'prepare_status.json',status)
    try:
        run(cfg,out)
        status.update(state='complete',finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        status.update(state='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        write_json(out/'prepare_status.json',status)


if __name__=='__main__':
    main()
