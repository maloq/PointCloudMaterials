"""Replay exact cached views and attach smoothly pooled group measurements."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import torch

from src.experiment_runner.artifacts import write_json
from src.experiment_runner.registry import sha256
from src.project_runtime.paths import load_json
from src.research.mace_context.data import load_clouds
from src.research.mace_context.engine import load_model
from src.research.mace_context.recovery_train import dual_encode
from .physics import group_observables, GROUP_NAMES


def progress(root, stage, **fields):
    payload = dict(state='running', stage=stage, updated_unix=time.time(), **fields)
    write_json(root/'technical/status.json', payload)
    print(json.dumps(payload), flush=True)


def inputs(config):
    parent = load_json(config['context_config'])
    diagnostic = Path(parent['diagnostics'])/'technical'
    with np.load(diagnostic/'probes.npz') as data:
        probes = {k: data[k] for k in ['source', 'context', 'split', 'temperature', 'atom_id', 'hot', 'relaxed']}
    with np.load(diagnostic/'temporal.npz') as data:
        temporal = {k: data[k] for k in ['source', 'context', 'atom_id', 'hot']}
    with np.load(config['features']) as data:
        features = {k: data[k] for k in data.files}
    if features['z'].shape != (len(probes['source']), 512):
        raise ValueError(f'Frozen dual-feature producer shape mismatch: {features["z"].shape}')
    splits = {s: set(probes['source'][probes['split'] == s]) for s in ['train', 'val', 'test']}
    if any(splits[a] & splits[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]):
        raise ValueError('Simulation sources overlap between splits')
    if set(temporal['source']) != splits['test']:
        raise ValueError('Tracked temporal cohort must contain only the six held-out sources')
    return parent, probes, temporal, features


def prepare(config, root):
    parent, probes, temporal, features = inputs(config)
    checkpoint_hash = sha256(Path(config['joint_checkpoint']))
    if checkpoint_hash != config['checkpoint_sha256']:
        raise ValueError(f'Wrong frozen checkpoint: {checkpoint_hash}')
    manifest_path = Path(parent['cache'])/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    provenance = dict(config=config, checkpoint_sha256=checkpoint_hash,
        feature_sha256=sha256(Path(config['features'])), context_manifest_sha256=sha256(manifest_path),
        probes_sha256=sha256(Path(parent['diagnostics'])/'technical/probes.npz'),
        temporal_sha256=sha256(Path(parent['diagnostics'])/'technical/temporal.npz'),
        group_names=GROUP_NAMES, source_splits={s: sorted(map(int, np.unique(probes['source'][probes['split'] == s])))
                                              for s in ['train', 'val', 'test']})
    provenance_path = root/'technical/input-provenance.json'
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise ValueError('Resume inputs/config differ from retained local-state preparation')
    write_json(provenance_path, provenance)
    shard_dir = root/'technical/prepared'; shard_dir.mkdir(exist_ok=True)
    missing = [r for r in manifest['records'] if not (shard_dir/r['file']).exists()]
    encoder = dict(parent, device=config['device'], cpu_threads=config['cpu_threads'],
                   micro_batch_size=config['micro_batch_size'])
    model = None
    if missing:
        model, _ = load_model(encoder)
        saved = torch.load(config['joint_checkpoint'], map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model_state'], strict=True)
        model.eval().requires_grad_(False)
    physics = lambda cloud: group_observables(cloud, inner=parent['inner_radius_A'],
        outer=parent['outer_radius_A'], halo=parent['candidate_radius_A'], coordination_radius=config['coordination_radius_A'])
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=config['physics_workers']) as pool:
        for index, record in enumerate(manifest['records']):
            target = shard_dir/record['file']
            if target.exists():
                continue
            path = Path(parent['cache'])/record['file']
            if sha256(path) != record['sha256']:
                raise ValueError(f'Cached geometry hash mismatch: {path}')
            clouds = load_clouds(path); rows = np.asarray(record['rows'])
            if len(clouds) != 3*len(rows):
                raise ValueError(f'Expected row-major anchor/spatial/previous views: {path}')
            replay = dual_encode(encoder, model, clouds[:6:3]).cpu().numpy()
            np.testing.assert_allclose(replay, features['z'][rows[:2]], rtol=3e-5, atol=3e-6)
            selected = [cloud for i, cloud in enumerate(clouds) if i % 3 != 0]
            encoded = dual_encode(encoder, model, selected).cpu().numpy().reshape(len(rows), 2, 512)
            observations = np.stack(list(pool.map(physics, clouds))).reshape(len(rows), 3, 16)
            np.savez(target, rows=rows, spatial=encoded[:, 0], previous=encoded[:, 1], group=observations,
                     source_sha256=np.array(record['sha256']))
            progress(root, 'prepare_pairs', contexts=index+1, total=len(manifest['records']),
                     elapsed_seconds=time.monotonic()-started)
        for index, record in enumerate(manifest['temporal']):
            target = shard_dir/record['file']
            if target.exists():
                continue
            path = Path(parent['cache'])/record['file']
            if sha256(path) != record['sha256']:
                raise ValueError(f'Cached temporal geometry hash mismatch: {path}')
            values = np.stack(list(pool.map(physics, load_clouds(path))))
            np.savez(target, rows=record['rows'], group=values, source_sha256=np.array(record['sha256']))
            progress(root, 'prepare_temporal_physics', contexts=index+1, total=len(manifest['temporal']))
    n = len(features['z'])
    spatial, previous = np.full((n, 512), np.nan), np.full((n, 512), np.nan)
    group = np.full((n, 3, 16), np.nan)
    seen = np.zeros(n, int)
    for record in manifest['records']:
        with np.load(shard_dir/record['file']) as data:
            np.testing.assert_array_equal(data['rows'], record['rows'])
            if str(data['source_sha256']) != record['sha256']:
                raise ValueError('Prepared shard provenance differs from context manifest')
            rows = data['rows']; spatial[rows] = data['spatial']; previous[rows] = data['previous']
            group[rows] = data['group']; seen[rows] += 1
    if not np.all(seen == 1):
        raise ValueError('Pair rows do not form an exact partition')
    shape = features['temporal_z'].shape[:2]
    tg = np.full((np.prod(shape), 16), np.nan); seen = np.zeros(len(tg), int)
    for record in manifest['temporal']:
        with np.load(shard_dir/record['file']) as data:
            np.testing.assert_array_equal(data['rows'], record['rows'])
            if str(data['source_sha256']) != record['sha256']:
                raise ValueError('Prepared temporal provenance differs from manifest')
            tg[data['rows']] = data['group']; seen[data['rows']] += 1
    if not np.all(seen == 1) or not all(np.isfinite(x).all() for x in [spatial, previous, group, tg]):
        raise ValueError('Incomplete prepared views or temporal labels')
    np.savez(root/'technical/local-data.npz', spatial=spatial.astype(np.float32),
             previous=previous.astype(np.float32), group=group, temporal_group=tg.reshape(*shape, 16))
    progress(root, 'prepared', elapsed_seconds=time.monotonic()-started)


def load_prepared(config, root):
    parent, probes, temporal, features = inputs(config)
    with np.load(root/'technical/local-data.npz') as data:
        local = {k: data[k] for k in data.files}
    return probes, temporal, features, local
