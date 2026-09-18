"""Freeze existing source identities and compute matched local descriptors."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.spatial import cKDTree

from src.data.structural_pretraining.prepare import (
    REFERENCE_RADIUS, chart, offsets, source_arrays, geometry_packet, file_hash, save_json,
)
from src.project_runtime.paths import dataset_path, resolve_path


def freeze(config):
    root = Path(config['output'])
    technical = root/'technical'
    technical.mkdir(parents=True, exist_ok=True)
    path = technical/'plan.json'
    if path.exists():
        plan = json.loads(path.read_text())
        if plan['config'] != config:
            raise ValueError(f'Existing stability plan has different configuration: {path}')
        return plan
    import torch
    cohort = json.loads(Path(config['cohort']).read_text())
    rng = np.random.default_rng(config['seed'])
    sources = []
    for split, count in [('train', config['reference_sources_per_temperature']),
                         ('test', config['test_sources_per_temperature'])]:
        for temperature in config['temperatures_K']:
            choices = sorted((s for s in cohort['sources'] if s['split'] == split
                              and s['temperature_K'] == temperature), key=lambda s: s['id'])
            indices = np.sort(rng.choice(len(choices), count, replace=False))
            for i in indices:
                s = dict(choices[i])
                s['selected_centers'] = sorted(rng.choice(s['center_atom_ids'], config['centers_per_source'], replace=False).tolist())
                s['frames'] = list(range(s['frame_count'])) if split == 'test' else list(range(0, s['frame_count'], config['reference_stride']))
                s['path'] = str(dataset_path(s['dataset'])/s['relative_trajectory_path'])
                sources.append(s)
    checkpoints = {}
    for name, spec in config['checkpoints'].items():
        source = Path(spec['path'])
        if file_hash(source) != spec['sha256']:
            raise ValueError(f'Checkpoint changed: {source}')
        target = technical/f'{name}.pt'
        shutil.copy2(source, target)
        if file_hash(target) != spec['sha256']:
            raise ValueError(f'Checkpoint changed while copying: {source}')
        saved = torch.load(target, map_location='cpu', weights_only=False)
        if saved['input_frames'] != 1 or saved['state_dim'] != 128:
            raise ValueError('This assay requires native 128D snapshot checkpoints')
        # Check inference dependencies; unrelated preparation/training changes do
        # not change a frozen encoder. Input primitives are audited separately.
        for filename, expected in saved['identity']['implementation']['files'].items():
            if filename.startswith('src/models/') or filename.endswith(('/batches.py', '/compilation.py')):
                if file_hash(filename) != expected:
                    raise ValueError(f'Encoder implementation changed: {filename}')
        release = json.loads((resolve_path(saved['identity']['config']['release'])/'manifest.json').read_text())
        fitted = {s['lineage'] for s in release['sources'] if s['split'] in ('train', 'selection') and 'lineage' in s}
        overlap = fitted & {s['lineage'] for s in sources if s['split'] == 'test'}
        if overlap:
            raise ValueError(f'Test ancestry overlaps training/selection: {overlap}')
        checkpoints[name] = dict(path=str(target), sha256=spec['sha256'], step=saved['step'],
            architecture=saved['architecture'], scales=saved['scales'], identity=saved['identity'])
    scales = [c['scales']['Al'] for c in checkpoints.values()]
    if len(set(scales)) != 1:
        raise ValueError(f'Encoders use different Al scales: {scales}')
    plan = dict(protocol='structural_trajectory_stability_v1', config=config, sources=sources,
        checkpoints=checkpoints, scale=scales[0], cadence_ps=cohort['cadence_ps'],
        created_at=datetime.now(timezone.utc).isoformat(), cohort_sha256=file_hash(config['cohort']))
    save_json(path, plan)
    return plan


def prepare_source(args):
    source, plan = args
    from ase import Atoms
    from dscribe.descriptors import SOAP
    from src.analysis.liquid_structure import persistence_image
    sid = source['id']
    root = Path(plan['config']['output'])/'technical'/'sources'/str(sid)
    root.mkdir(parents=True, exist_ok=True)
    if (root/'complete.json').exists():
        return json.loads((root/'complete.json').read_text())
    started = time.monotonic()
    raw = source_arrays(dict(source, kind='dynamic'))
    times = raw['timesteps'].astype(np.float64)*source['timestep_fs']/1000
    np.testing.assert_allclose(np.diff(times), plan['cadence_ps'], rtol=0, atol=1e-12)
    ids = np.array(source['selected_centers'], dtype=np.int64)
    centers = np.searchsorted(raw['atom_ids'], ids)
    np.testing.assert_array_equal(raw['atom_ids'][centers], ids)
    cache = resolve_path(plan['config']['physical_cache'])
    prior_path = cache/f'source-{sid:04d}.npz'
    receipt = json.loads(prior_path.with_suffix('.json').read_text())
    if receipt['source_manifest_sha256'] != source['manifest_sha256'] or file_hash(prior_path) != receipt['shard_sha256']:
        raise ValueError(f'Physical/order/PTM cache provenance mismatch: {sid}')
    prior = np.load(prior_path)
    loc = np.searchsorted(prior['atom_ids'], ids)
    np.testing.assert_array_equal(prior['atom_ids'][loc], ids)
    np.testing.assert_array_equal(prior['times_ps'], times)
    frames = np.array(source['frames'])
    # Flatten frame-major; every method and input follows this one row order.
    order = prior['order'][loc][:, frames].transpose(1, 0, 2).reshape(-1, 8)
    labels = prior['labels'][loc][:, frames].T.reshape(-1)
    soap = SOAP(species=['Al'], periodic=False, r_cut=7., n_max=8, l_max=6,
                sigma=.3, sparse=False, dtype='float64')
    radius = 17*plan['scale']/REFERENCE_RADIUS
    positions, atom_ids, pointers, center_indices = [], [], [0], []
    geometries, topology, soaps, nearest_ids = [], [], [], []
    for fi, frame in enumerate(frames):
        points, tree, box = chart(raw, frame, False)
        if box.min() <= 4*radius:
            raise ValueError(f'Periodic box too small for the encoder support: {sid}, {frame}')
        groups = tree.query_ball_point(points[centers], radius, return_sorted=True)
        for center, neighbors in zip(centers, groups, strict=True):
            neighbors = np.asarray(neighbors, dtype=np.int64)
            local = offsets(points, center, neighbors, box)
            distance2 = np.square(local.astype(np.float64)).sum(-1)
            near = np.lexsort((raw['atom_ids'][neighbors], distance2))[:80]
            if len(near) != 80:
                raise ValueError(f'Incomplete 80-atom descriptor support: {sid}, {frame}, {center}')
            positions.append(local); atom_ids.append(raw['atom_ids'][neighbors])
            pointers.append(pointers[-1]+len(local))
            center_indices.append(int(np.flatnonzero(neighbors == center).item()))
            geometries.append(geometry_packet(local))
            topology.append(persistence_image(local[near]))
            soaps.append(soap.create(Atoms('Al'*len(local), positions=local), centers=[[0., 0., 0.]])[0])
            nearest_ids.append(raw['atom_ids'][neighbors[near]])
        if fi % 40 == 0:
            save_json(root/'progress.json', dict(source=sid, frames_done=fi+1, frames_total=len(frames), seconds=time.monotonic()-started))
    np.save(root/'positions.npy', np.concatenate(positions), allow_pickle=False)
    np.save(root/'atom_ids.npy', np.concatenate(atom_ids), allow_pickle=False)
    np.savez(root/'observations.npz', offsets=np.array(pointers), center_indices=np.array(center_indices),
        centers=ids, frames=frames, times_ps=times[frames], labels=labels,
        geometry=np.stack(geometries), tda=np.stack(topology), soap=np.stack(soaps),
        bond_order=order[:, :6], order=order, nearest_ids=np.stack(nearest_ids))
    complete = dict(source=sid, frames=len(frames), centers=len(ids), observations=len(labels),
        seconds=time.monotonic()-started, source_manifest_sha256=source['manifest_sha256'],
        reused_physical_cache_sha256=receipt['shard_sha256'],
        hashes={p.name: file_hash(p) for p in [root/'positions.npy', root/'atom_ids.npy', root/'observations.npz']})
    save_json(root/'complete.json', complete)
    return complete


def prepare(plan):
    with ProcessPoolExecutor(max_workers=plan['config']['workers']) as pool:
        futures = [pool.submit(prepare_source, (source, plan)) for source in plan['sources']]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)
