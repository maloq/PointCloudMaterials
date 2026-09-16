"""Consecutive, identity-tracked observations for the local-motion experiment."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
import torch
from threadpoolctl import threadpool_limits

from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.experiment_runner.registry import sha256, write_json
from src.project_runtime.paths import load_json, portable_config, resolve_path
from src.research.mace_velocity.data import labels, local_clouds
from src.research.mace_velocity.inference import load_encoder
from src.research.mace_velocity.inventory import read
from src.research.mace_velocity.train import encode


class AllocationEnding(RuntimeError):
    """A resumable unit has finished; there is insufficient allocation time."""


def check_deadline(config):
    if time.time() >= config['runtime']['stop_unix']:
        raise AllocationEnding('Allocation deadline reached; completed units/checkpoints retained')


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix('.building.npz')
    np.savez(temporary, **arrays)
    temporary.replace(path)


def contiguous_window(eligible, length):
    """A middle window of genuinely consecutive recorded frame indices."""
    eligible = np.asarray(eligible, dtype=np.int64)
    runs = np.split(eligible, np.flatnonzero(np.diff(eligible) != 1)+1)
    choices = [r[i:i+length] for r in runs for i in range(len(r)-length+1)]
    if not choices:
        raise ValueError(f'No {length}-frame consecutive window in {len(eligible)} eligible frames')
    middle = np.median(eligible)
    return min(choices, key=lambda r: abs(np.mean(r)-middle))


def preparation_identity(config):
    source = load_json(config['source_config'])
    paths = [Path(source['output'])/'technical/inventory.json',
             Path(source['cache'])/'manifest.json', Path(source['output'])/'technical/teacher.npz',
             Path(config['checkpoint']), Path(__file__)]
    return dict(protocol=config['protocol'], frames=config['sequence_frames'],
        centers=config['centers_per_source'], smoke_source_ids=config['smoke_source_ids'],
        hashes={portable_config(str(p)): sha256(p) for p in paths})


def plan(config, root):
    cache = Path(config['cache']); cache.mkdir(parents=True, exist_ok=True)
    identity = preparation_identity(config)
    path = cache/'plan.json'
    if path.exists():
        saved = read(path)
        if saved['identity'] != identity: raise ValueError(f'Changed sequence preparation: {path}')
        return saved
    source = load_json(config['source_config'])
    inventory = read(Path(source['output'])/'technical/inventory.json')
    paired = read(Path(source['cache'])/'manifest.json')
    records = []; excluded = []
    for record in paired['records']:
        s = record['source']
        if config['smoke_source_ids'] and s['id'] not in config['smoke_source_ids']: continue
        if s['format'] == 'paired_dump_conversion':
            excluded.append(dict(source_id=s['id'], reason='Converted stratified pairs are not consecutive original observations'))
            continue
        frames = contiguous_window(s['eligible_frames'], config['sequence_frames'])
        centers = sorted(set(p['center_atom_id'] for p in record['pairs']))[:config['centers_per_source']]
        if len(centers) != config['centers_per_source']: raise ValueError(f'Insufficient tracked centers: {s["id"]}')
        records.append(dict(source=s, frames=frames.tolist(), center_atom_ids=centers))
    if not records: raise ValueError('Empty sequence selection')
    lineage_splits = {}
    for record in records:
        s = record['source']; lineage_splits.setdefault(s['lineage'], set()).add(s['split'])
    if any(len(v) != 1 for v in lineage_splits.values()): raise ValueError('Preparation lineage crosses splits')
    payload = dict(identity=identity, records=records, excluded=excluded,
        source_inventory_counts=inventory['counts'])
    write_json(path, payload); write_json(root/'technical/sequence-plan.json', payload)
    return payload


def label_one(cloud):
    with threadpool_limits(limits=1):
        return labels(cloud)


def read_sequence(record, config):
    """Trace actual producers; return only selected frames, with explicit units."""
    source = record['source']; frames = np.asarray(record['frames'])
    path = resolve_path(source['path'])
    if source['format'] == 'shooting_binary':
        trajectory = ShootingBinaryTrajectory.load(path)
        if sha256(path/'manifest.json') != source['manifest_sha256']:
            raise ValueError(f'Source manifest changed: {path}')
        positions = np.asarray(trajectory.positions[frames])
        velocities = np.asarray(trajectory.velocities[frames])
        lengths = trajectory.box_high[frames].astype(float)-trajectory.box_low[frames]
        steps = trajectory.timesteps[frames]
        identity = dict(manifest_sha256=source['manifest_sha256'], storage_dtype=trajectory.storage_dtype.name)
    elif source['format'] == 'legacy_npz':
        with np.load(path) as values:
            positions = values['positions_A'][frames]
            velocities = values['velocities_A_per_ps'][frames]
            cells = values['cell_vectors_A'][frames]
            steps = values['step'][frames]
        if positions.dtype != np.float32 or velocities.dtype != np.float32:
            raise ValueError(f'Changed legacy precision: {path}')
        if not np.allclose(cells, cells*np.eye(3), atol=0, rtol=0): raise ValueError(f'Nonorthogonal cell: {path}')
        lengths = np.diagonal(cells, axis1=1, axis2=2).astype(float)
        identity = dict(file_sha256=sha256(path), storage_dtype='float32')
    else:
        raise ValueError(f'Unsupported consecutive producer: {source["format"]}')
    if positions.shape != velocities.shape or positions.shape[1:] != (source['atom_count'],3):
        raise ValueError(f'Unexpected trajectory shape at {path}: {positions.shape}, {velocities.shape}')
    times = steps.astype(float)*source['timestep_fs']/1000
    if np.any(np.diff(times) <= 0): raise ValueError(f'Nonincreasing physical time: {path}')
    identity['selected_arrays_sha256'] = hashlib.sha256(b''.join(np.ascontiguousarray(a).tobytes()
        for a in (positions,velocities,lengths,steps))).hexdigest()
    return positions, velocities, lengths, times, identity


def prepare(config, root, lane):
    torch.set_num_threads(config['cpu_threads'])
    device = config['devices'][lane]
    selection = read(Path(config['cache'])/'plan.json')
    if selection['identity'] != preparation_identity(config): raise ValueError('Changed sequence preparation identity')
    model, inference = load_encoder(config['checkpoint'], device)
    inference = dict(inference, micro_batch_size=config['micro_batch_size'])
    records = selection['records'][lane::len(config['devices'])]
    started = time.monotonic()
    with ProcessPoolExecutor(config['label_workers_per_lane'], mp_context=multiprocessing.get_context('spawn')) as pool:
        for number, record in enumerate(records):
            check_deadline(config)
            sid = record['source']['id']; path = Path(config['cache'])/f'source-{sid:04d}.npz'
            sidecar = path.with_suffix('.json')
            if sidecar.exists():
                saved = read(sidecar)
                if saved['record'] != record or sha256(path) != saved['sha256']:
                    raise ValueError(f'Changed completed sequence unit: {path}')
                continue
            p,v,lengths,times,identity = read_sequence(record, config)
            centers = np.array(record['center_atom_ids'])-1
            clouds = []; neighbor_ids = []; displacements = []
            previous_p = None
            for frame in range(len(times)):
                points = np.mod(p[frame].astype(float), lengths[frame])
                local = local_clouds(points,v[frame],lengths[frame],centers,config['candidate_radius_A'])
                clouds.extend(local)
                nearest = cKDTree(points,boxsize=lengths[frame]).query(points[centers],k=80,workers=1)[1]
                if not np.array_equal(nearest[:,0], centers): raise ValueError(f'Center-ID mismatch: {sid}, {frame}')
                neighbor_ids.append(nearest)
                if previous_p is not None:
                    # Same previous-frame atom IDs, relative to the same tracked center.
                    old = previous_p[neighbor_ids[-2]]-previous_p[centers,None]
                    old -= lengths[frame-1]*np.round(old/lengths[frame-1])
                    new = points[neighbor_ids[-2]]-points[centers,None]
                    new -= lengths[frame]*np.round(new/lengths[frame])
                    delta = new-old; delta -= lengths[frame]*np.round(delta/lengths[frame])
                    displacements.append(np.sqrt(np.mean(np.sum(delta[:,1:]**2,axis=-1),axis=1)))
                previous_p = points
            future = list(pool.map(label_one, clouds))
            target = np.stack(future)
            embedding = encode(inference,model,clouds,device).cpu().numpy()
            n = len(centers); nt = len(times)
            ids = np.stack(neighbor_ids)
            retention = np.array([[len(np.intersect1d(ids[t,c,1:],ids[t+1,c,1:]))/79
                for c in range(n)] for t in range(nt-1)])
            # A counterfactual storage round trip on native float32 source frames.
            precision = dict(native_dtype=identity['storage_dtype'])
            if identity['storage_dtype'] == 'float32' and number < config['precision_sources_per_lane']:
                pp = p[0].astype(np.float16).astype(np.float32)
                vv = v[0].astype(np.float16).astype(np.float32)
                roundtrip = local_clouds(pp,vv,lengths[0],centers,config['candidate_radius_A'])
                zz = encode(inference,model,roundtrip,device).cpu().numpy()
                yy = np.stack(list(pool.map(label_one,roundtrip)))
                precision.update(position_rms_A=float(np.sqrt(np.mean((pp-p[0])**2))),
                    velocity_rms_A_per_ps=float(np.sqrt(np.mean((vv-v[0])**2))),
                    structural_increment_squared=np.sum((zz[:,:256]-embedding[:n,:256])**2,axis=1).tolist(),
                    target_increment_squared=np.mean((yy-target[:n])**2,axis=0).tolist())
            arrays = dict(embedding=embedding.reshape(nt,n,304).transpose(1,0,2),
                raw_target=target.reshape(nt,n,169).transpose(1,0,2),time_ps=np.tile(times,(n,1)),
                center_atom_id=centers+1,neighbor_retention=retention.T.astype(np.float32),
                atom_matched_rms_A=np.stack(displacements).T.astype(np.float32))
            extra = []
            for x,_ in clouds:
                r = np.linalg.norm(x[1:80],axis=1)
                covariance = np.cov(x[1:80].astype(float).T,ddof=0)
                eigenvalues = np.linalg.eigvalsh(covariance)
                extra.append([r.min(),r.mean(),r.std(),(eigenvalues[-1]-eigenvalues[0])/eigenvalues.sum()])
            arrays['extra_observables'] = np.asarray(extra,dtype=np.float32).reshape(nt,n,4).transpose(1,0,2)
            if any(not np.isfinite(a).all() for a in arrays.values()): raise FloatingPointError(f'Nonfinite sequence: {sid}')
            atomic_npz(path, **arrays)
            write_json(sidecar,dict(record=record,source_identity=identity,precision=precision,sha256=sha256(path)))
            status = dict(state='preparing',lane=lane,completed=number+1,total=len(records),
                source_id=sid,elapsed_seconds=time.monotonic()-started)
            write_json(root/f'technical/prepare-lane{lane}.json',status)
            print('MOTION PREPARE',status,flush=True)
    write_json(root/f'technical/prepare-lane{lane}.json',dict(state='complete',sources=len(records)))


def assemble(config, root):
    selection = read(Path(config['cache'])/'plan.json')
    records = []; fields = {}; sources = []; splits = []; groups = []; temperatures = []; lineages = []
    group_ids = {}; lineage_ids = {}; source_weights = {}
    for record in selection['records']:
        s = record['source']; key = s['balance_group']
        if s['split']=='train': source_weights[key] = source_weights.get(key,0)+1
        if key not in group_ids: group_ids[key] = len(group_ids)
        if s['lineage'] not in lineage_ids: lineage_ids[s['lineage']] = len(lineage_ids)
    weights = []
    for record in selection['records']:
        s = record['source']; path = Path(config['cache'])/f'source-{s["id"]:04d}.npz'
        stamp = read(path.with_suffix('.json'))
        if stamp['record'] != record or sha256(path)!=stamp['sha256']: raise ValueError(f'Changed sequence unit: {path}')
        with np.load(path) as a:
            for key in a.files: fields.setdefault(key,[]).append(a[key])
            n = len(a['embedding'])
        sources.extend([s['id']]*n);splits.extend([{'train':0,'val':1,'test':2}[s['split']]]*n)
        groups.extend([group_ids[s['balance_group']]]*n);temperatures.extend([s['temperature_K']]*n)
        lineages.extend([lineage_ids[s['lineage']]]*n)
        weights.extend([1/source_weights[s['balance_group']] if s['split']=='train' else 1.]*n)
        records.append(dict(source_id=s['id'],sha256=stamp['sha256'],source_identity=stamp['source_identity'],precision=stamp['precision']))
    arrays = {k:np.concatenate(v) for k,v in fields.items()}
    arrays.update(source_id=np.array(sources),split=np.array(splits),balance_group=np.array(groups),
        lineage_id=np.array(lineages),temperature_K=np.array(temperatures),weights=np.array(weights,dtype=np.float32))
    source = load_json(config['source_config'])
    norm = dict(np.load(Path(source['output'])/'technical/teacher.npz'))
    arrays['target'] = ((arrays['raw_target']-norm['target_mean'])/norm['target_scale']).astype(np.float32)
    path = Path(config['cache'])/'sequences.npz'
    atomic_npz(path,**arrays)
    report = dict(state='complete',identity=selection['identity'],sha256=sha256(path),
        sequences=len(sources),observations=int(arrays['embedding'].shape[0]*arrays['embedding'].shape[1]),
        by_split={name:int(np.sum(arrays['split']==i)) for i,name in enumerate(('train','validation','development_test'))},
        records=records,excluded=selection['excluded'])
    write_json(Path(config['cache'])/'complete.json',report)
    write_json(root/'technical/prepare-status.json',report)
    print('MOTION ASSEMBLED',report['observations'],flush=True)


def load(config):
    root = Path(config['cache']); report = read(root/'complete.json')
    if sha256(root/'sequences.npz') != report['sha256']: raise ValueError('Changed consecutive feature cache')
    return dict(np.load(root/'sequences.npz'))
