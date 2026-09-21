"""Full-cell quenches amortized over paired, tracked 80-atom observations."""
import json
import os
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial import cKDTree
import torch
from src.project_runtime.paths import resolve_path,dataset_path,machine
from src.data.structural_pretraining.prepare import save_json,digest,file_hash,geometry_packet
from src.data.structural_pretraining.support import REFERENCE_RADIUS,OUTER_RADIUS,EDGE_CUTOFF,support_weights
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.data.relaxed_targets.worker import AbsolutePositions,publish,verify_archive
from src.data.conversion.relaxation import read_relaxed,convert
from src.simulation.relaxation import relax_frame
from src.analysis.liquid_structure import persistence_image
from src.training_methods.neighborhood_jepa.prepare import distributed_neighbors
from src.training_methods.neighborhood_jepa.regularization.data import order_targets
from src.training_methods.neighborhood_jepa.v2.geometry import moments

ARMS=('instantaneous','hot_to_relaxed','relaxed_to_relaxed')

def arm_for_name(config,name):
    if 'runs' not in config:return name
    return next(r['arm'] for r in config['runs'] if r['name']==name)


def reuse_paired(plan,task,source,root):
    previous=json.loads(resolve_path(plan['config']['paired_parent_plan']).read_text());pc=previous['config']
    for key in ('seed','scale','potential_sha256'):
        if pc[key]!=plan['config'][key]:raise ValueError(f'Paired reuse differs: {key}')
    p=resolve_path(pc['cache'])/'cells'/task['id']/'complete.json'
    if not p.exists():return None
    record=json.loads(p.read_text());old_source=next(s for s in previous['sources'] if s['id']==source['id'])
    if old_source['manifest_sha256']!=source['manifest_sha256'] or record['identity']!=previous['identity']:
        raise ValueError('Paired source or release identity changed')
    path=p.with_name('clouds.npz')
    if file_hash(path)!=record['clouds_sha256']:raise ValueError('Paired clouds changed')
    centers=source['pool_atom_ids'] if source['pilot_fit'] and source.get('validation_role',source['split'])=='train' else source['center_atom_ids']
    columns=7 if source['pilot_fit'] else 1
    with np.load(path) as a:
        ids=a['query_atom_ids'][:,0];lookup={int(v):i for i,v in enumerate(ids)}
        if a['query_atom_ids'].shape[1]<columns or any(int(v) not in lookup for v in centers):return None
        rows=np.array([lookup[int(v)] for v in centers]);values={k:a[k][rows,:columns] for k in a.files}
        np.testing.assert_array_equal(values['query_atom_ids'][:,0],centers)
    if record['relaxation']['fmax_eV_per_A']>.01:raise ValueError('Paired quench not converged')
    np.savez_compressed(root/'clouds.npz',**values)
    result=dict(record,identity=plan['identity'],task=task,clouds_sha256=file_hash(root/'clouds.npz'),
        reused_paired=str(path),reused_paired_sha256=record['clouds_sha256'])
    save_json(root/'complete.json',result);return result

def arm_clouds(hot,cold,arm):
    if arm not in ARMS:raise ValueError(arm)
    inputs=hot.copy() if arm=='instantaneous' else cold.copy()
    if arm=='hot_to_relaxed':inputs[:,0,0]=hot[:,0,0]
    return inputs,hot if arm=='instantaneous' else cold


def target_cloud(points,scale):
    """Same physical observation support for decoder labels and encoder graphs."""
    x=np.asarray(points,np.float32)
    selected=x[np.linalg.norm(x,axis=-1)*REFERENCE_RADIUS/scale<OUTER_RADIUS]
    if len(selected)<13 or np.linalg.norm(selected[0])!=0:
        raise ValueError(f'Insufficient centered observation after support crop: {len(selected)} atoms')
    return selected


@lru_cache(maxsize=1)
def old_cells(plan_path,frames):
    old=json.loads(Path(plan_path).read_text());root=resolve_path(old['config']['output'])/'technical/tasks';result={}
    for task in old['tasks']:
        if task['frame'] not in frames:continue
        source=old['sources'][task['source_index']]
        if source['family']!='independent_train':continue
        path=root/f'{task["id"]}.json'
        if not path.exists():continue
        r=json.loads(path.read_text())
        if r['state']=='complete':result[source['manifest_sha256'],task['frame']]=(r,old['config'])
    return result


def reuse(plan,task,source,root):
    # Saved local clouds retain pre-quantization precision. They can cover center
    # assays, but not all six neighboring query centers needed for JEPA training.
    if source['pilot_fit']:return None
    found=old_cells(str(resolve_path(plan['config']['old_relaxation_plan'])),tuple(plan['config']['frames'])).get((source['manifest_sha256'],task['frame']))
    if found is None:return None
    record,cfg=found;target=Path(record['targets'])
    if file_hash(target)!=record['targets_sha256']:raise ValueError(f'Changed cached relaxation: {target}')
    archive=resolve_path(cfg['archive'])/'frames'/record['target_key'];verify_archive(archive)
    meta=json.loads((archive/'metadata.json').read_text())
    if set(meta['potential_checksums'].values())!=set(plan['config']['potential_sha256']) or meta['fmax_eV_per_A']>.01:
        raise ValueError(f'Cached quench is not the generating-potential reference: {archive}')
    with np.load(target) as a:
        rows=np.searchsorted(a['center_atom_ids'],source['center_atom_ids'])
        np.testing.assert_array_equal(a['center_atom_ids'][rows],source['center_atom_ids'])
        np.savez_compressed(root/'clouds.npz',hot=a['observed_clouds'][rows,None],cold=a['relaxed_clouds'][rows,None],
            query_atom_ids=a['center_atom_ids'][rows,None],neighbor_atom_ids=a['neighbor_atom_ids'][rows,None],
            query_hot=np.zeros((len(rows),1,3),np.float32),query_cold=np.zeros((len(rows),1,3),np.float32))
    result=dict(identity=plan['identity'],task=task,clouds_sha256=file_hash(root/'clouds.npz'),archive=str(archive),relaxation=meta,
        reused_targets=str(target),reused_targets_sha256=record['targets_sha256'])
    save_json(root/'complete.json',result);return result

def freeze(config):
    root=resolve_path(config['output'])/'technical';root.mkdir(parents=True,exist_ok=True)
    path=root/'plan.json'
    if path.exists():
        plan=json.loads(path.read_text())
        if plan['config']!=config:raise ValueError('Relaxed pilot plan changed')
        return plan
    assay=json.loads(resolve_path(config['assay_plan']).read_text());rng=np.random.default_rng(config['seed'])
    selected=[]
    for role,count in [('train',config['train_sources_per_temperature']),('selection',config['selection_sources_per_temperature'])]:
        for temp in (400,450,500,510,520):
            choices=sorted([s for s in assay['sources'] if s.get('validation_role',s['split'])==role and s['temperature_K']==temp],key=lambda s:s['id'])
            selected.extend(choices[i]['id'] for i in rng.choice(len(choices),len(choices) if count is None else count,replace=False))
    sources=[];tasks=[]
    for s in assay['sources']:
        source=dict(s,path=str(dataset_path(s['dataset'])/s['relative_trajectory_path']),pilot_fit=s['id'] in selected)
        sources.append(source)
        frames=set(config['frames'])
        training_frames=config.get('training_frames',config['frames'])
        if not set(training_frames)<=set(config['frames']):raise ValueError('Training origins must be observed assay origins')
        if source['pilot_fit']:frames.update(f+1 for f in training_frames)
        for frame in sorted(frames):
            anchor=frame if frame in config['frames'] else frame-1
            tasks.append(dict(id=f'{s["id"]}-{frame}',source=s['id'],frame=frame,anchor=anchor,priority=0 if source['pilot_fit'] and anchor in training_frames else 1))
    tasks.sort(key=lambda t:(t['priority'],t['source'],t['frame']))
    potential=[resolve_path(p) for p in config['potential_files']]
    for p,h in zip(potential,config['potential_sha256'],strict=True):
        if file_hash(p)!=h:raise ValueError(f'Generating potential mismatch: {p}')
    plan=dict(config=config,sources=sources,tasks=tasks,assay_identity=assay['identity'],assay_sha256=file_hash(resolve_path(config['assay_plan'])),
              identity=digest(dict(config=config,sources=sources)),potential_files=list(map(str,potential)),
              producer={str(p):file_hash(p) for p in (Path(__file__),Path('src/simulation/relaxation.py'))})
    save_json(path,plan);return plan


def settings(plan,ranks,*,minimizer='fire',force_tolerance=.01):
    e=machine()['execution'];os.environ.update(e['mpi_environment']);p=plan['potential_files']
    return dict(mass=26.9815385,minimizer=minimizer,timestep_ps=.001,force_tolerance=force_tolerance,
        max_iterations=10000,max_evaluations=50000,frame_timeout_seconds=2400,
        lammps_command=[v.format(ranks=ranks) for v in e['mpi_launcher']]+[plan['config']['lammps']],
        potential_files=p,pair_commands=['pair_style meam',f'pair_coeff * * {p[0]} Al {p[1]} Al'])


def paired_clouds(hot,cold,box,queries):
    """Select IDs from observations; preserve those exact atoms after quenching."""
    tree=cKDTree(np.mod(hot,box),boxsize=box)
    _,ids=tree.query(np.mod(hot[queries],box),k=80,workers=1)
    if not np.array_equal(ids[...,0],queries):raise ValueError('Center must be the first observed neighbor')
    clouds=[]
    for positions in (hot,cold):
        x=positions[ids]-positions[queries][...,None,:];x-=box*np.rint(x/box);clouds.append(x.astype(np.float32))
    return clouds[0],clouds[1],ids


def query_atoms(raw,source,anchor,scale,seed):
    ids=source['pool_atom_ids'] if source['pilot_fit'] and source.get('validation_role',source['split'])=='train' else source['center_atom_ids']
    center=np.searchsorted(raw.atom_ids,ids);np.testing.assert_array_equal(raw.atom_ids[center],ids)
    if not source['pilot_fit']:return center[:,None]
    box=(raw.box_high[anchor]-raw.box_low[anchor]).astype(np.float64);x=np.mod(raw.positions[anchor].astype(np.float64),box)
    tree=cKDTree(x,boxsize=box);factor=REFERENCE_RADIUS/scale;rng=np.random.default_rng(np.random.SeedSequence([seed,source['id'],anchor]));result=[]
    for c in center:
        candidates=np.array(sorted(tree.query_ball_point(x[c],4.25/factor)));candidates=candidates[candidates!=c]
        v=x[candidates]-x[c];v-=box*np.rint(v/box)
        chosen=distributed_neighbors((v*factor).astype(np.float32),6,rng);result.append(np.r_[c,candidates[chosen]])
    return np.array(result)


def produce(plan,task,ranks,recovery=None):
    config=plan['config'];source=next(s for s in plan['sources'] if s['id']==task['source']);frame=task['frame']
    source=dict(source,pilot_fit=source['pilot_fit'] and task['anchor'] in config.get('training_frames',config['frames']))
    root=resolve_path(config['cache'])/'cells'/task['id'];root.mkdir(parents=True,exist_ok=True)
    receipt=root/'complete.json'
    if receipt.exists():
        result=json.loads(receipt.read_text())
        if result['identity']!=plan['identity'] or file_hash(root/'clouds.npz')!=result['clouds_sha256']:raise ValueError(f'Paired cell changed: {root}')
        return result
    reused=reuse_paired(plan,task,source,root) if 'paired_parent_plan' in config else None
    if reused is None:reused=reuse(plan,task,source,root)
    if reused is not None:return reused
    raw=ShootingBinaryTrajectory.load(resolve_path(source['path']))
    if file_hash(raw.root/'manifest.json')!=source['manifest_sha256']:raise ValueError(f'Raw source changed: {source["id"]}')
    np.testing.assert_allclose(raw.timesteps[[task['anchor'],frame]]*source['timestep_fs']/1000,np.array([task['anchor'],frame])*.75,rtol=0,atol=1e-6)
    hot=raw.positions[frame].astype(np.float64);low=raw.box_low[frame].astype(np.float64);box=raw.box_high[frame].astype(np.float64)-low
    work=resolve_path(config['scratch'])/'cells'/task['id'];archive=resolve_path(config['archive'])/'cells'/task['id']
    if recovery is not None:
        work=resolve_path(config['scratch'])/'retries'/recovery['name']/task['id']
    pending=root/'pending.json'
    if pending.exists():
        result=json.loads(pending.read_text())
        if result['identity']!=plan['identity'] or file_hash(root/'clouds.npz')!=result['clouds_sha256']:raise ValueError(f'Interrupted cell changed: {root}')
        if not (work/'conversion.json').exists():convert(work,delete_source=True,local_cloud_dtype='float32')
        if not archive.exists():publish(work,archive)
        verify_archive(archive);save_json(receipt,result);return result
    # Recover from an interrupted publication without repeating a converged quench.
    if not (work/'metadata.json').exists():
        if archive.exists():raise ValueError(f'Archive without paired-cloud provenance: {archive}')
        absolute=SimpleNamespace(**vars(raw),atom_count=raw.atom_count);absolute.positions=AbsolutePositions(raw)
        execution=settings(plan,ranks)
        if recovery is not None:
            execution.update(recovery['limits'],restart_dump=recovery['restart_dump'],restart_sha256=recovery['restart_sha256'])
        try:relax_frame(absolute,frame,work,execution)
        except Exception:
            failure=resolve_path(config['archive'])/'failures'/task['id']
            if recovery is not None:failure=resolve_path(config['archive'])/'failed-retries'/recovery['name']/task['id']
            if work.exists() and not failure.exists():publish(work,failure)
            raise
    cold,meta=read_relaxed(work);cold-=low
    queries=query_atoms(raw,source,task['anchor'],config['scale'],config['seed'])
    a,b,neighbors=paired_clouds(hot,cold,box,queries)
    query_positions=[]
    for x in (hot,cold):
        q=x[queries]-x[queries[:,0,None]];q-=box*np.rint(q/box);query_positions.append((q*(REFERENCE_RADIUS/config['scale'])).astype(np.float32))
    for clouds in (a,b):
        if not np.isfinite(clouds).all():raise FloatingPointError(f'Nonfinite target clouds: {task}')
    temp=root/'clouds.tmp.npz';np.savez_compressed(temp,hot=a,cold=b,query_atom_ids=raw.atom_ids[queries],neighbor_atom_ids=raw.atom_ids[neighbors],query_hot=query_positions[0],query_cold=query_positions[1]);temp.replace(root/'clouds.npz')
    result=dict(identity=plan['identity'],task=task,clouds_sha256=file_hash(root/'clouds.npz'),archive=str(archive),relaxation=meta)
    save_json(pending,result)
    # Preserve centered float32 clouds before the mandated full-cell conversion.
    convert(work,delete_source=True,local_cloud_dtype='float32')
    if not archive.exists():publish(work,archive)
    save_json(receipt,result);return result


def graph_arrays(clouds,scale):
    x=[target_cloud(p,scale)*(REFERENCE_RADIUS/scale) for p in np.asarray(clouds,np.float32).reshape(-1,80,3)]
    counts=np.array([len(p) for p in x]);ptr=np.r_[0,counts.cumsum()].astype(np.int64);edges=[];eptr=[0]
    for points in x:
        pairs=cKDTree(points).query_pairs(EDGE_CUTOFF,output_type='ndarray');edge=np.concatenate((pairs,pairs[:,::-1]),0).T.astype(np.int32);edges.append(edge);eptr.append(eptr[-1]+edge.shape[1])
    flat=np.concatenate(x);g=np.repeat(np.arange(len(x)),counts)
    fixed=moments(torch.from_numpy(flat),torch.from_numpy(g),len(x)).numpy()
    return dict(positions=flat,weights=support_weights(flat),offsets=ptr,edges=np.concatenate(edges,axis=1),edge_offsets=np.array(eptr,np.int64)),fixed


def build_training(plan):
    """Three native cache releases consumed by the existing unchanged trainer."""
    config=plan['config'];cache=resolve_path(config['cache']);parent=json.loads(resolve_path(config['normalization_manifest']).read_text());order_norm=json.loads(resolve_path(config['order_manifest']).read_text())
    root=resolve_path(config['output'])/'technical';completed=[]
    for source in plan['sources']:
        if not source['pilot_fit']:continue
        for frame in config.get('training_frames',config['frames']):
            for f in (frame,frame+1):
                if not (cache/'cells'/f'{source["id"]}-{f}'/'complete.json').exists():return False
            completed.append((source,frame))
    for arm in ARMS:
        out=cache/arm
        if (out/'manifest.json').exists():continue
        records=[]
        for source,frame in completed:
            sid=f'{source["id"]}-{frame}';folder=out/'shards'/sid;folder.mkdir(parents=True,exist_ok=True)
            with np.load(cache/'cells'/sid/'clouds.npz') as now, np.load(cache/'cells'/f'{source["id"]}-{frame+1}'/'clouds.npz') as future:
                np.testing.assert_array_equal(now['query_atom_ids'],future['query_atom_ids'])
                hot=np.stack([now['hot'],future['hot']],1);cold=np.stack([now['cold'],future['cold']],1)
                inputs,target=arm_clouds(hot,cold,arm)
                center_targets=[target_cloud(p,config['scale']) for p in target[:,:,0].reshape(-1,80,3)]
                physical=np.stack([geometry_packet(p) for p in center_targets]).reshape(len(hot),2,85)
                tda=np.stack([persistence_image(p) for p in center_targets]).reshape(len(hot),2,144)
                orders=np.stack([order_targets(p*(REFERENCE_RADIUS/config['scale']),config['scale']) for p in center_targets]).reshape(len(hot),2,8)
                graphs,_=graph_arrays(inputs,config['scale']);_,fixed=graph_arrays(target,config['scale'])
                views=np.full((len(hot),3,7),-1,np.int32);views[:,1:]=np.arange(len(hot)*14).reshape(len(hot),2,7)
                values=dict(**graphs,views=views,physical=physical,tda=tda,query_atom_ids=now['query_atom_ids'].copy(),
                    query_positions=now['query_cold' if arm=='relaxed_to_relaxed' else 'query_hot'].copy(),times=np.array([-.75,0,.75],np.float32))
            for name,value in values.items():np.save(folder/f'{name}.npy',value,allow_pickle=False)
            (out/'moments').mkdir(exist_ok=True);np.save(out/'moments'/f'{sid}.npy',fixed)
            (out/'order').mkdir(exist_ok=True);np.save(out/'order'/f'{sid}.npy',orders)
            records.append(dict(id=sid,source=f'native_{source["id"]}',lineage=source['lineage'],split=source.get('validation_role',source['split']),material='Al',potential='al-lee2003-meam',group=0,species=1,scale=config['scale'],anchors=len(hot),temperature_K=source['temperature_K'],frame=frame,hashes={n:file_hash(folder/f'{n}.npy') for n in values}))
        identity=digest(dict(plan=plan['identity'],arm=arm,records=records))
        manifest=dict(parent,state='complete',identity=identity,parent=str(out),shards=records,
            train_roots=sorted({r['lineage'] for r in records if r['split']=='train'}),selection_roots=sorted({r['lineage'] for r in records if r['split']=='selection'}),
              paired_relaxation_identity=plan['identity'],arm=arm,support='Observed nearest80 candidate IDs retained across quench; input and target both cropped at normalized radius8 per domain; target moments use the identical graph crop',normalization_source=str(resolve_path(config['normalization_manifest'])))
        current_order=dict(order_norm)
        if config.get('normalize_target_domain',False):
            from src.training_methods.neighborhood_jepa.v2.geometry import blocks
            train=[r for r in records if r['split']=='train'];normalization={}
            for name,width in [('physical',85),('tda',144)]:
                values=np.concatenate([np.load(out/'shards'/r['id']/f'{name}.npy').reshape(-1,width) for r in train]).astype(np.float64)
                normalization[name]=dict(mean=values.mean(0).tolist(),std=np.maximum(values.std(0),1e-4).tolist())
            order=np.concatenate([np.load(out/'order'/f'{r["id"]}.npy').reshape(-1,8) for r in train]).astype(np.float64)
            current_order.update(mean=order.mean(0).tolist(),std=np.maximum(order.std(0),1e-4).tolist())
            eq=torch.from_numpy(np.concatenate([np.load(out/'moments'/f'{r["id"]}.npy').reshape(r['anchors'],14,-1)[:,[0,7]].reshape(-1,120) for r in train]))
            scales=torch.stack([b.square().mean((0,2)).sqrt().clamp_min(.001) for b in blocks(eq)])
            manifest.update(normalization=normalization,geometry_scales=scales.tolist(),normalization_source='This arm target domain; training sources only')
        save_json(out/'order-manifest.json',dict(current_order,base_identity=identity,identity=digest([identity,'order']),shards=records))
        # Original order loader expects its manifest beside order/*.npy.
        order_dir=out/'order-cache';order_dir.mkdir(exist_ok=True)
        if not (order_dir/'order').is_symlink():(order_dir/'order').symlink_to(out/'order',target_is_directory=True)
        save_json(order_dir/'manifest.json',json.loads((out/'order-manifest.json').read_text()))
        save_json(out/'manifest.json',manifest)
    save_json(root/'training-ready.json',dict(state='complete',arms=list(ARMS),rows={a:sum(r['anchors'] for r in json.loads((cache/a/'manifest.json').read_text())['shards']) for a in ARMS}))
    return True
