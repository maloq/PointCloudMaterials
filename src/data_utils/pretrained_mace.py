"""Source-separated spatial/short-time/forecast quadruplets for MLIP fine-tuning."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from numpy.lib.format import open_memmap
from scipy.spatial import cKDTree
from src.data_utils.shooting_binary import ShootingBinaryTrajectory
from src.data_utils.temporal_lammps_binary import TemporalLAMMPSBinaryTrajectory
from src.data_utils.temporal_campaign import write_json
from src.analysis.liquid_structure import persistence_image


def prepare_shard(record,cfg):
    started=time.monotonic();directory=Path(cfg['cache'])/record['name']
    directory.mkdir(parents=True,exist_ok=False)
    old=Path(record['directory']);wide=np.load(old/'clouds.npy',mmap_mode='r')
    maximum=cfg[('al' if record['material']==0 else 'ordinary')+'_centers_'+record['split']]
    if record['material']==0 and record['kind']=='ordinary':
        maximum=cfg['al_continuations']['centers'][record['split']]
    # Cached atom IDs are sorted. Taking their first entries would select a
    # spatially biased portion of the system. Subsample across the entire pool.
    selection_seed=cfg['seed']+(record['parent'] if record['kind']=='shooting' else record['source'])
    select=np.sort(np.random.default_rng(selection_seed).choice(record['centers'],min(maximum,record['centers']),replace=False))
    rows=np.load(old/'center_rows.npy')[select];n=len(rows);a=len(record['anchors']);p=cfg['points']
    x=open_memmap(directory/'clouds.npy',mode='w+',dtype='float16',shape=(a*n,4,p,3))
    targets=open_memmap(directory/'tda.npy',mode='w+',dtype='float32',shape=(a*n,4,144))
    ids=np.empty((a*n,4),dtype=np.int64);frames=np.empty_like(ids)
    conditions=np.empty((a*n,5),dtype=np.float32)
    trajectory=(ShootingBinaryTrajectory if record['kind']=='shooting' else TemporalLAMMPSBinaryTrajectory).load(record['path'])
    halo_min=float('inf');neighbor_max=0.
    for ai,frame in enumerate(record['anchors']):
        sl=slice(ai*n,(ai+1)*n)
        source_index=record['anchor_indices'][ai];short_index=record['future_indices'][ai][0]
        li=1+(np.arange(n)%3);future_index=np.array(record['future_indices'][ai])[li]
        base=wide[select,source_index].astype(np.float32);short=wide[select,short_index].astype(np.float32);future=wide[select,future_index].astype(np.float32)
        for values in (base,short,future):halo_min=min(halo_min,float(np.linalg.norm(values[:,p],axis=-1).min()))
        x[sl,0]=base[:,:p];x[sl,2]=short[:,:p];x[sl,3]=future[:,:p]
        lengths=trajectory.box_high[frame].astype(np.float64)-trajectory.box_low[frame]
        pos=np.mod(trajectory.positions[frame].astype(np.float64)-trajectory.box_low[frame],lengths)
        tree=cKDTree(pos,boxsize=lengths,balanced_tree=False)
        distances,near=tree.query(pos[rows],k=13,workers=1)
        # Draw one of the six closest actual atoms, fixed by the data seed.
        choice=np.random.default_rng(cfg['seed']+ai+record['source']).integers(1,7,size=n)
        spatial_ids=near[np.arange(n),choice]
        neighbor_max=max(neighbor_max,float(distances[np.arange(n),choice].max()))
        dist,spatial=tree.query(pos[spatial_ids],k=p+1,workers=1)
        halo_min=min(halo_min,float(dist[:,-1].min()))
        cloud=pos[spatial[:,:p]]-pos[spatial_ids,None];cloud-=lengths*np.round(cloud/lengths)
        x[sl,1]=cloud
        ids[sl]=np.stack((rows,spatial_ids,rows,rows),1)
        future_frames=np.array(record['frames'])[future_index]
        frames[sl]=np.stack((np.full(n,frame),np.full(n,frame),np.full(n,frame+record['lags'][0]),future_frames),1)
        conditions[sl,0]=record['temperature']/600.
        conditions[sl,1]=np.log1p((future_frames-frame)*record['cadence_ps'])
        conditions[sl,2:]=np.eye(3,dtype=np.float32)[record['material']]
        # Keep target construction in float32, independently of cache storage.
        target_views=(base[:,:65],cloud[:,:65].astype(np.float32),short[:,:65],future[:,:65])
        for local in range(n):
            for view in range(4):targets[ai*n+local,view]=persistence_image(target_views[view][local])
    if halo_min<=10.02:raise RuntimeError(f"{record['name']}: {p} points truncate the required 10.02 A halo; excluded radius {halo_min}")
    x.flush();targets.flush();np.save(directory/'ids.npy',ids);np.save(directory/'frames.npy',frames);np.save(directory/'condition.npy',conditions)
    result=dict(record,directory=str(directory),anchors_count=a*n,selected_centers=n,minimum_excluded_radius_A=halo_min,maximum_spatial_neighbor_distance_A=neighbor_max,selection_seed=selection_seed,seconds=time.monotonic()-started)
    result['cache_storage_dtype']='float16'
    write_json(directory/'manifest.json',result)
    return result


def prepare(cfg):
    if 'al_continuations' in cfg:return prepare_with_continuations(cfg)
    out=Path(cfg['output']);cache=Path(cfg['cache']);cache.mkdir(parents=True,exist_ok=True)
    if (cache/'manifest.json').exists():
        previous=json.loads((cache/'manifest.json').read_text())['config']
        for key in ('source_cache','seed','points','shooting_shots','al_centers_train','al_centers_val','ordinary_centers_train','ordinary_centers_val'):
            if previous[key]!=cfg[key]:raise ValueError(f'Prepared MACE data differ for {key}: {previous[key]} vs {cfg[key]}; select a new cache')
    source=json.loads((Path(cfg['source_cache'])/'manifest.json').read_text())['shards']
    tasks=[s for s in source if s['split'] in ('train','val') and (s['kind']!='shooting' or s['shot'] in cfg['shooting_shots'])]
    records=[]
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        futures=[]
        for s in tasks:
            manifest=cache/s['name']/'manifest.json'
            if manifest.exists():
                records.append(json.loads(manifest.read_text()))
            else:futures.append(pool.submit(prepare_shard,s,cfg))
        for future in as_completed(futures):
            r=future.result();records.append(r)
            print('PREPARED',r['name'],len(records),'/',len(tasks),r['anchors_count'],flush=True)
            write_json(out/'status.json',dict(state='preparing',completed=len(records),total=len(tasks)))
    records.sort(key=lambda r:r['name'])
    write_json(cache/'manifest.json',dict(shards=records,config=cfg))
    summarize(records,cfg)
    audit_sampling(records,cfg)
    return records


def summarize(records,cfg):
    out=Path(cfg['output'])
    counts={}
    for split in ('train','val'):
        counts[split]={}
        for m,label in enumerate(('Al','Mg','Ta')):
            unique=set();anchors=0
            for r in records:
                if r['split']!=split or r['material']!=m:continue
                directory=Path(r['directory']);ids=np.load(directory/'ids.npy');frames=np.load(directory/'frames.npy');anchors+=len(ids)
                for f,i in zip(frames.ravel(),ids.ravel()):
                    # Replicas share the initial state within a temperature parent.
                    # Different temperature parents have distinct initial coordinates.
                    source_key=f"initial_Al_parent_{r['parent']}" if r['kind']=='shooting' and f==0 else r['path']
                    unique.add((source_key,int(f),int(i)))
            counts[split][label]=dict(anchor_quadruplets=anchors,view_slots=4*anchors,distinct_neighborhood_states=len(unique))
    summary=dict(counts=counts,neighborhood_points=cfg.get('model_points',cfg['points']),cached_neighborhood_points=cfg['points'],physical_cutoff_A=5.,required_halo_A=cfg.get('outer_radius_A',10.),
        pretrained_checkpoint_sha256=hashlib.sha256(Path(cfg['pretrained_checkpoint']).read_bytes()).hexdigest(),
        spatial='One of the six nearest actual atoms, independently centered, full halo.',
        temporal=('Same atom at 0.1 ps for Al/Mg/Ta. Al shooting pairs are excluded from temporal VICReg.' if 'temporal_lag_ps' in cfg else 'Same atom, shortest stored lag: Al 0.3 ps; Mg/Ta 0.1 ps.'),
        forecast='Same atom, Al shooting 1.2/6/12 ps; ordinary continuations 0.4/2/4 ps. Ta later validation 0.4/1/2 ps.',
        tda='144D alpha-complex persistence images H0/H1/H2 of center +64 neighbors for all four views; train-only PCA/scaler.',
        limits='Al/Mg held-out sources. Ta has disjoint IDs and later frames of one source, not independent-source validation. Mg/Ta positions are stored float16. No GeoFrame or EMA teacher. No static-Al analysis frames added to training.')
    if 'al_continuations' in cfg:
        summary['limits']='Al shooting source groups and Al/Mg continuation validation sources are held out within their campaigns. Ta uses disjoint IDs and later times of one source. Al continuations/Mg/Ta coordinates are float16. Static Al 166/170/174/175 ps frames are ancestors of continuation training data; 177 ps is a validation ancestor and 240 ps is excluded from continuation training. Static analysis is descriptive, not a fully independent test.'
    write_json(out/'data_summary.json',summary)


class Quadruplets:
    def __init__(self,cfg):
        self.records=json.loads((Path(cfg['cache'])/'manifest.json').read_text())['shards']
        points=cfg.get('model_points',cfg['points'])
        self.clouds=[np.load(Path(r['directory'])/'clouds.npy',mmap_mode='r')[:,:,:points] for r in self.records]
        self.tda=[np.load(Path(r['directory'])/'tda.npy',mmap_mode='r') for r in self.records]
        self.conditions=[np.load(Path(r['directory'])/'condition.npy') for r in self.records]
        self.pools={split:[np.array([(i,j) for i,r in enumerate(self.records) if r['split']==split and r['material']==m for j in range(r['anchors_count'])],dtype=np.int64) for m in range(3)] for split in ('train','val')}
        self.required_temporal_lag=cfg.get('temporal_lag_ps')
        self.temporal_eligible=np.array([self.required_temporal_lag is None or abs(r['lags'][0]*r['cadence_ps']-self.required_temporal_lag)<1e-9 for r in self.records])
        self.al_subpools={}
        if self.required_temporal_lag is not None:
            for split in ('train','val'):
                al=self.pools[split][0]
                self.al_subpools[split]=[al[self.temporal_eligible[al[:,0]]==enabled] for enabled in (False,True)]
                if any(len(p)==0 for p in self.al_subpools[split]):
                    raise ValueError(f'{split}: mixed Al training requires both shooting and 0.1 ps continuation pools')

    def temporal_mask(self,indices):
        return self.temporal_eligible[indices[:,0]]

    def epoch_steps(self,batch_size):
        if self.required_temporal_lag is not None:
            pools=self.al_subpools['train']+self.pools['train'][1:]
            quotas=[batch_size//6,batch_size//6,batch_size//3,batch_size//3]
            return max(len(p)//q for p,q in zip(pools,quotas))
        return max(len(p) for p in self.pools['train'])//(batch_size//3)

    def validation_indices(self,per_material,rng):
        if self.required_temporal_lag is None:
            return np.stack([p[rng.choice(len(p),per_material,replace=False)] for p in self.pools['val']],1).reshape(-1,2)
        al=np.concatenate([p[rng.choice(len(p),per_material//2,replace=False)] for p in self.al_subpools['val']])
        others=[p[rng.choice(len(p),per_material,replace=False)] for p in self.pools['val'][1:]]
        return np.stack([al]+others,1).reshape(-1,2)


    def get(self,indices):
        x=np.stack([self.clouds[i][j] for i,j in indices]).astype(np.float32);t=np.stack([self.tda[i][j] for i,j in indices]);c=np.stack([self.conditions[i][j] for i,j in indices])
        m=np.array([self.records[i]['material'] for i,j in indices],dtype=np.int64)
        return x,t,c,m

    def epoch(self,split,batch_size,rng):
        if self.required_temporal_lag is not None:
            pools=self.al_subpools[split]+self.pools[split][1:]
            quotas=[batch_size//6,batch_size//6,batch_size//3,batch_size//3]
            steps=max(len(p)//q for p,q in zip(pools,quotas))
            draws=[np.concatenate([rng.permutation(len(p)) for _ in range((steps*q+len(p)-1)//len(p))])[:steps*q] for p,q in zip(pools,quotas)]
            for step in range(steps):
                yield np.concatenate([p[d[step*q:(step+1)*q]] for p,d,q in zip(pools,draws,quotas)])
            return
        pools=self.pools[split];n=max(len(p) for p in pools);per=batch_size//3
        # Exhaust the largest material once, cycling independently shuffled smaller pools.
        draws=[np.concatenate([rng.permutation(len(p)) for _ in range((n+len(p)-1)//len(p))])[:n] for p in pools]
        for start in range(0,n-per+1,per):
            indices=np.concatenate([p[d[start:start+per]] for p,d in zip(pools,draws)])
            yield indices


def audit_sampling(records,cfg):
    """Record spatial coverage and verify the shared initial-state counting rule."""
    out=Path(cfg['output']);coverage={};initial={}
    if 'outer_radius_A' in cfg:
        minima=[]
        for r in records:
            clouds=np.load(Path(r['directory'])/'clouds.npy',mmap_mode='r')
            excluded=float(np.linalg.norm(clouds[:,:,cfg['model_points']].astype(np.float32),axis=-1).min())
            if excluded<=cfg['outer_radius_A']+.02:
                raise ValueError(f"{r['name']}: compact {cfg['model_points']}-point input truncates the smooth context; excluded radius {excluded} A")
            minima.append(dict(shard=r['name'],minimum_excluded_radius_A=excluded))
        write_json(out/'compact_context_coverage.json',dict(input_points=cfg['model_points'],outer_radius_A=cfg['outer_radius_A'],shards=minima))
    for material,name in enumerate(('Al','Mg','Ta')):
        id_counts=np.zeros(4,dtype=np.int64);spatial_counts=np.zeros((3,4),dtype=np.int64)
        for r in records:
            if r['material']!=material:continue
            trajectory=(ShootingBinaryTrajectory if r['kind']=='shooting' else TemporalLAMMPSBinaryTrajectory).load(r['path'])
            if r['kind']=='shooting':
                digest=hashlib.sha256(trajectory.positions[0].tobytes()+trajectory.box_low[0].tobytes()+trajectory.box_high[0].tobytes()).hexdigest()
                if r['parent'] in initial:
                    if initial[r['parent']]['initial_state_sha256']!=digest:
                        raise ValueError(f"Shooting replicas of parent {r['parent']} do not share their initial coordinates/box; distinct-state counting must be revised")
                    initial[r['parent']]['replicas']+=1
                else:initial[r['parent']]=dict(parent=r['parent'],replicas=1,initial_state_sha256=digest)
            if r['split']!='train':continue
            rows=np.unique(np.load(Path(r['directory'])/'ids.npy')[:,0])
            id_counts+=np.histogram(rows/trajectory.atom_count,bins=np.linspace(0,1,5))[0]
            frame=r['anchors'][0];lengths=trajectory.box_high[frame].astype(np.float64)-trajectory.box_low[frame]
            positions=np.mod(trajectory.positions[frame,rows].astype(np.float64)-trajectory.box_low[frame],lengths)/lengths
            for axis in range(3):spatial_counts[axis]+=np.histogram(positions[:,axis],bins=np.linspace(0,1,5))[0]
        coverage[name]=dict(id_quartile_fractions=(id_counts/id_counts.sum()).tolist(),spatial_quartile_fractions_xyz=(spatial_counts/spatial_counts.sum(1,keepdims=True)).tolist())
    write_json(out/'sampling_coverage_audit.json',coverage)
    write_json(out/'initial_state_deduplication_check.json',list(initial.values()))


def prepare_continuation(task,cfg):
    """Reuse the established full-halo producer before making identical four-view records."""
    from src.data_utils.temporal_campaign import prepare_task
    source_cache=Path(cfg['al_continuations']['source_cache'])
    manifest=source_cache/task['name']/'manifest.json'
    if manifest.exists():
        record=json.loads(manifest.read_text())
        np.testing.assert_array_equal(np.load(source_cache/task['name']/'center_rows.npy'),task['rows'])
    else:
        source_cache.mkdir(parents=True,exist_ok=True)
        record=prepare_task(task,dict(cache=str(source_cache),neighbors=768))
    target=Path(cfg['cache'])/task['name']/'manifest.json'
    if target.exists():return json.loads(target.read_text())
    return prepare_shard(record,cfg)


def prepare_with_continuations(cfg):
    """Add source-separated Al 0.1 ps data; preserve shooting data for forecasting."""
    cache=Path(cfg['cache']);cache.mkdir(parents=True,exist_ok=True);out=Path(cfg['output'])
    base_manifest=Path(cfg['reuse_prepared_cache'])/'manifest.json'
    original=json.loads(base_manifest.read_text())
    for key in ('seed','points','shooting_shots','al_centers_train','al_centers_val','ordinary_centers_train','ordinary_centers_val'):
        if original['config'][key]!=cfg[key]:raise ValueError(f'Reused four-view cache differs for {key}; prepare a matching base dataset')
    if (cache/'manifest.json').exists():
        previous=json.loads((cache/'manifest.json').read_text())['config']
        for key in ('al_continuations','reuse_prepared_cache','seed','points'):
            if previous[key]!=cfg[key]:raise ValueError(f'Al continuation cache differs for {key}; choose a new output cache')
    records=list(original['shards']);tasks=[];settings=cfg['al_continuations']
    for split in ('train','val'):
        for index,snapshot in enumerate(settings[split+'_snapshots']):
            directory=Path(settings['root'])/snapshot
            metadata=json.loads((directory/'metadata.json').read_text())
            assert metadata['state']=='complete' and metadata['sample_interval_ps']==.1 and metadata['timestep_fs']==1.
            path=directory/'trajectory_binary_float16';trajectory=TemporalLAMMPSBinaryTrajectory.load(path)
            np.testing.assert_array_equal(trajectory.timesteps,np.arange(241)*100)
            source=300+index+(0 if split=='train' else 50)
            rows=np.sort(np.random.default_rng(cfg['seed']+source).choice(trajectory.atom_count,settings['centers'][split],replace=False))
            tasks.append(dict(name=f'Al_continuation_{snapshot}_{split}',kind='ordinary',split=split,material=0,temperature=0.,source=source,parent=-1,shot=-1,path=str(path.resolve()),rows=rows.tolist(),anchors=settings['anchors'],lags=settings['lags'],cadence_ps=.1))
    with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
        for future in as_completed([pool.submit(prepare_continuation,task,cfg) for task in tasks]):
            record=future.result();records.append(record)
            print('AL_CONTINUATION_PREPARED',record['name'],record['anchors_count'],flush=True)
            write_json(out/'status.json',dict(state='preparing_0.1ps_Al',completed=len(records)-len(original['shards']),total=len(tasks)))
    records.sort(key=lambda r:r['name'])
    write_json(cache/'manifest.json',dict(shards=records,config=cfg,base_manifest_sha256=hashlib.sha256(base_manifest.read_bytes()).hexdigest()))
    summarize(records,cfg);audit_sampling(records,cfg)
    eligibility=[]
    for r in records:
        frames=np.load(Path(r['directory'])/'frames.npy');ids=np.load(Path(r['directory'])/'ids.npy')
        lag=(frames[:,2]-frames[:,0])*r['cadence_ps']
        enabled=r['lags'][0]*r['cadence_ps']==cfg['temporal_lag_ps']
        if enabled:
            np.testing.assert_allclose(lag,.1,rtol=0,atol=1e-9)
            np.testing.assert_array_equal(ids[:,0],ids[:,2])
        eligibility.append(dict(shard=r['name'],split=r['split'],material=r['material'],anchors=len(ids),temporal_vicreg=enabled,pair_lags_ps=np.unique(lag).tolist()))
    write_json(out/'temporal_pair_audit.json',dict(required_lag_ps=.1,shards=eligibility))
    return records
