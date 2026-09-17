"""Outcome-independent full-timeline sampling for the local predictability assay."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from src.data.predictive_memory.prepare import file_hash, write_json
from src.data.predictive_memory.targets import physical_packet, taper
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.project_runtime.paths import dataset_path, load_json, resolve_path
from src.research.forecast_crystallization.local_metrics import first_sustained_onset, risk_windows
from .plan import build_queue


def time_guard(deadline):
    if time.time() >= datetime.fromisoformat(deadline).timestamp():
        raise TimeoutError(f'Preparation deadline reached: {deadline}; incomplete shards are not a release')


def freeze(config):
    plan = json.loads(resolve_path(config['plan']).read_text())
    queue = build_queue(plan)
    inherited = json.loads(resolve_path(plan['source_manifest']).read_text())
    legacy_root = resolve_path(config['legacy_output']) / 'technical'
    legacy = json.loads((legacy_root/'local_observations.json').read_text())['sources']
    old = {r['trajectory_manifest_sha256']: r for r in legacy}
    records = []
    for source in inherited['sources']:
        raw_root = dataset_path(source['dataset'])/source['relative_trajectory_path']
        if file_hash(raw_root/'manifest.json') != source['manifest_sha256']:
            raise RuntimeError(f"Manifest changed for source {source['id']}: {raw_root}")
        raw = ShootingBinaryTrajectory.load(raw_root)
        rng = np.random.default_rng(np.random.SeedSequence([plan['sampling']['selection_seed'], source['id']]))
        previous = old.get(source['manifest_sha256'])
        if previous is not None:
            if previous['split'] != source['split']:
                raise ValueError(f"Legacy/source split mismatch for {source['id']}")
            directory = legacy_root/'labels'/previous['directory']
            for name, field in [('atom_ids.npy', 'atom_ids_sha256'), ('labels.npy', 'labels_sha256')]:
                if file_hash(directory/name) != previous[field]:
                    raise RuntimeError(f'Legacy label artifact changed: {directory/name}')
            pool = np.load(directory/'atom_ids.npy')
            if len(pool) != 64:
                raise ValueError('Legacy center pool must contain 64 atom IDs')
            legacy_record = dict(directory=previous['directory'], source_index=previous['source_index'],
                                 labels_sha256=previous['labels_sha256'])
        else:
            pool = np.sort(rng.choice(raw.atom_ids, 64, replace=False))
            legacy_record = None
        centers = np.sort(rng.choice(pool, plan['sampling']['native_centers_per_source'], replace=False))
        records.append(dict(**source, center_atom_ids=centers.tolist(), pool_atom_ids=pool.tolist(), legacy=legacy_record))
    cohort = dict(protocol=plan['protocol'], seed=plan['sampling']['selection_seed'], sources=records,
        plan_sha256=file_hash(resolve_path(config['plan'])), cadence_ps=.75,
        native_anchors=np.linspace(64,664,16,dtype=int).tolist(),
        dense_anchors=(np.array(queue['primary_common_origins_ps'])/.75).astype(int).tolist(),
        horizon_frames=(np.array(plan['sampling']['horizons_ps'])/.75).astype(int).tolist(),
        confirmation_frames=8, created_at=datetime.now(timezone.utc).isoformat())
    path=resolve_path(config['output'])/'technical/cohort.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        existing=json.loads(path.read_text())
        if {k:v for k,v in existing.items() if k!='created_at'} != {k:v for k,v in cohort.items() if k!='created_at'}:
            raise ValueError('Existing immutable cohort differs')
        return existing
    write_json(path,cohort)
    return cohort


def shell_features(x, u):
    """Two smooth annuli; count and five invariant geometry/motion moments each."""
    radius=np.linalg.norm(x,axis=-1)
    radial=(x*u).sum(-1)/np.maximum(radius,1e-12)
    result=[]
    for inner,outer in [(7.,17.),(17.,25.)]:
        weight=(1-taper(radius,inner,inner+2))*taper(radius,outer-2,outer)
        total=weight.sum()
        if total<=0:
            raise ValueError(f'Empty descriptor shell {inner}--{outer} A')
        w=weight/total; mean_u=w@u
        result.extend([total,w@radius,w@(u*u).sum(-1),w@radial,w@radial**2,mean_u@mean_u])
    return np.asarray(result,dtype=np.float32)


def full_ptm(points,lengths,rows):
    from ovito.data import DataCollection, Particles, SimulationCell
    from ovito.modifiers import PolyhedralTemplateMatchingModifier
    data=DataCollection(); particles=Particles(count=len(points))
    particles.create_property('Position',data=points); data.objects.append(particles)
    cell=SimulationCell(pbc=(True,True,True));cell[...]=np.column_stack((np.diag(lengths),np.zeros(3)))
    data.objects.append(cell)
    data.apply(PolyhedralTemplateMatchingModifier(rmsd_cutoff=.1))
    return np.asarray(data.particles['Structure Type'])[rows]


def prepare_source(source,config,verification_ids):
    os.environ['OVITO_THREAD_COUNT']='1'
    from src.analysis.liquid_structure import bond_order
    from src.research.smooth_temporal_encoder.prepare import ptm_labels
    started=time.monotonic(); time_guard(config['preparation_deadline_utc'])
    sid=source['id']; cache=resolve_path(config['cache']); destination=cache/f'source-{sid:04d}.npz'
    receipt=destination.with_suffix('.json')
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved['source_manifest_sha256']!=source['manifest_sha256'] or saved['atom_ids']!=source['center_atom_ids'] or saved['producer_sha256']!=file_hash(Path(__file__)):
            raise ValueError(f'Existing shard identity differs: {sid}')
        if file_hash(destination)!=saved['shard_sha256']:
            raise ValueError(f'Existing shard content differs: {sid}')
        return saved
    raw=ShootingBinaryTrajectory.load(dataset_path(source['dataset'])/source['relative_trajectory_path'])
    if file_hash(raw.root/'manifest.json')!=source['manifest_sha256']:
        raise RuntimeError(f'Manifest changed after cohort freeze: {sid}')
    raw.verify_checksums()
    times=raw.timesteps.astype(np.float64)*source['timestep_fs']/1000
    np.testing.assert_array_equal(times,np.arange(801)*.75)
    if raw.manifest['velocity_units']!='angstrom_per_ps' or not np.all(raw.atom_types==1):
        raise ValueError(f'Expected single-species Al and A/ps velocity: {sid}')
    ids=np.array(source['center_atom_ids'],dtype=np.int64);rows=np.searchsorted(raw.atom_ids,ids)
    np.testing.assert_array_equal(raw.atom_ids[rows],ids)
    packets=np.empty((len(ids),801,128),np.float32)
    orders=np.empty((len(ids),801,8),np.float32);shells=np.empty((len(ids),801,12),np.float32)
    labels=np.empty((len(ids),801),np.uint8)
    if source['legacy'] is not None:
        directory=resolve_path(config['legacy_output'])/'technical/labels'/source['legacy']['directory']
        oldids=np.load(directory/'atom_ids.npy');index=np.searchsorted(oldids,ids)
        np.testing.assert_array_equal(oldids[index],ids)
        if file_hash(directory/'labels.npy')!=source['legacy']['labels_sha256']:
            raise RuntimeError('Legacy labels changed after freeze')
        labels[:]=np.load(directory/'labels.npy')[index]
    verification=[]
    for frame in range(801):
        if frame%25==0: time_guard(config['preparation_deadline_utc'])
        lengths=raw.box_high[frame].astype(np.float64)-raw.box_low[frame].astype(np.float64)
        # The binary producer stores coordinates relative to box_low. Uniform shifts
        # do not affect the periodic relative geometry used by either assay.
        points=np.mod(raw.positions[frame].astype(np.float64),lengths)
        velocity=raw.velocities[frame].astype(np.float64)
        tree=cKDTree(points,boxsize=lengths)
        _,nearest=tree.query(points[rows],k=80,workers=1)
        np.testing.assert_array_equal(nearest[:,0],rows)
        offsets=points[nearest[:,1:]]-points[rows,None];offsets-=lengths*np.round(offsets/lengths)
        if source['legacy'] is None:
            labels[:,frame]=ptm_labels(offsets/10.,.1)
        if sid in verification_ids and frame in (80,640):
            expected=full_ptm(points,lengths,rows)
            np.testing.assert_array_equal(labels[:,frame],expected,err_msg=f'PTM patch/full mismatch source={sid}, frame={frame}')
            verification.append(dict(frame=frame,centers=len(ids),identical=True))
        first13=nearest[:,:13]
        _,order_neighbors=tree.query(points[first13],k=13,workers=1)
        bonds=points[order_neighbors[:,:,1:]]-points[first13][:,:,None,:]
        bonds-=lengths*np.round(bonds/lengths)
        orders[:,frame],_=bond_order(bonds,3.5)
        neighbor_lists=tree.query_ball_point(points[rows],25,workers=1)
        for center,neighbors in enumerate(neighbor_lists):
            neighbors=np.asarray(neighbors,dtype=int)
            x=points[neighbors]-points[rows[center]];x-=lengths*np.round(x/lengths)
            u=velocity[neighbors]-velocity[rows[center]]
            # Same float32 local chart as frame_observation, then physical_packet.
            packets[center,frame]=physical_packet(x.astype(np.float32),u.astype(np.float32))
            shells[center,frame]=shell_features(x,u)
    for name,array in [('packet',packets),('order',orders),('shell',shells)]:
        if not np.isfinite(array).all(): raise FloatingPointError(f'Nonfinite {name} for source {sid}')
    temp=destination.with_suffix('.building.npz')
    np.savez(temp,packet=packets,order=orders,shell=shells,labels=labels,atom_ids=ids,times_ps=times)
    temp.replace(destination)
    saved=dict(source_id=sid,source_manifest_sha256=source['manifest_sha256'],atom_ids=ids.tolist(),
        shard=destination.name,shard_sha256=file_hash(destination),producer_sha256=file_hash(Path(__file__)),
        checksum_verified=True,legacy_labels_reused=source['legacy'] is not None,
        ptm_verification=verification,elapsed_seconds=time.monotonic()-started)
    write_json(receipt,saved)
    return saved


def finalize(cohort,config,receipts):
    cache=resolve_path(config['cache']);output=resolve_path(config['output']);technical=output/'technical'
    anchors=np.asarray(cohort['native_anchors']);lags=np.asarray(cohort['horizon_frames'])
    train=[];rowarrays={k:[] for k in ['source_id','center_id','anchor','split','event_bin','risk','onset_frame']}
    coverage=[]
    for source in cohort['sources']:
        shard=np.load(cache/f"source-{source['id']:04d}.npz")
        if source['split']=='train':
            frames=anchors[:,None]+np.r_[0,lags][None,:]
            train.append(shard['packet'][:,frames].reshape(-1,128))
        crystal=np.isin(shard['labels'],[1,2,3])
        for persistence in (3,5,9):
            onset=first_sustained_onset(crystal,persistence)
            for gridname,grid in [('native',anchors),('dense',np.asarray(cohort['dense_anchors']))]:
                risk=risk_windows(crystal,onset,grid,3)
                delay=onset[:,None]-grid[None,:]
                positives=(delay[...,None]>0)&(delay[...,None]<=lags)&risk[...,None]
                coverage.append(dict(source_id=source['id'],split=source['split'],
                    validation_role=source.get('validation_role'),temperature_K=source['temperature_K'],
                    persistence_frames=persistence,grid=gridname,eligible=int(risk.sum()),
                    event_centers=int(np.any(positives,axis=(1,2)).sum()),positives=positives.sum((0,1)).tolist()))
                if persistence==3 and gridname=='native':
                    shape=risk.shape
                    values=dict(source_id=np.full(shape,source['id']),
                        center_id=np.broadcast_to(shard['atom_ids'][:,None],shape),
                        anchor=np.broadcast_to(grid,shape),
                        split=np.full(shape,source.get('validation_role',source['split'])),
                        event_bin=np.where(risk & (delay<=lags[-1]),np.searchsorted(lags,delay,side='left'),len(lags)),
                        risk=risk,onset_frame=np.broadcast_to(onset[:,None],shape))
                    for key,value in values.items():rowarrays[key].append(value.ravel())
    values=np.concatenate(train).astype(np.float64)
    normalizer=dict(mean=values.mean(0).tolist(),scale=np.maximum(values.std(0),1e-4).tolist(),
        population='train sources; native anchors; current plus six futures; population standard deviation')
    release=dict(protocol=cohort['protocol'],schema_version=1,state='complete',
        cohort_sha256=file_hash(technical/'cohort.json'),plan_sha256=cohort['plan_sha256'],
        native_anchors=cohort['native_anchors'],dense_anchors=cohort['dense_anchors'],
        horizon_frames=cohort['horizon_frames'],cadence_ps=.75,normalizer=normalizer,
        sources=[dict(**s,**next(r for r in receipts if r['source_id']==s['id'])) for s in cohort['sources']])
    np.savez(technical/'native_rows.npz',**{k:np.concatenate(v) for k,v in rowarrays.items()})
    write_json(technical/'coverage.json',coverage)
    write_json(cache/'release.json',release);write_json(technical/'release.json',release)
    write_json(technical/'prepare_status.json',dict(state='complete',sources=len(receipts),cache=str(cache)))
    return release


def prepare(config):
    cohort=freeze(config);cache=resolve_path(config['cache']);cache.mkdir(parents=True,exist_ok=True)
    output=resolve_path(config['output']);technical=output/'technical'
    verification_ids=[next(s['id'] for s in cohort['sources'] if s['split']=='train' and s['temperature_K']==t)
                      for t in (400,450,500,510,520)]
    receipts=[]
    with ProcessPoolExecutor(max_workers=config['workers'],mp_context=multiprocessing.get_context('spawn')) as executor:
        futures={executor.submit(prepare_source,s,config,verification_ids):s['id'] for s in cohort['sources']}
        for future in as_completed(futures):
            try: result=future.result()
            except Exception as error:
                for pending in futures: pending.cancel()
                raise RuntimeError(f"Failed source {futures[future]}") from error
            receipts.append(result)
            print(f"Prepared {len(receipts)}/150, source {result['source_id']}, {result['elapsed_seconds']:.1f}s",flush=True)
            write_json(technical/'prepare_status.json',dict(state='running',sources=len(receipts),total=150))
    finalize(cohort,config,receipts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stage',choices=['freeze','prepare'],default='prepare')
    args=parser.parse_args();config=load_json(args.config)
    (freeze if args.stage=='freeze' else prepare)(config)


if __name__=='__main__':main()
