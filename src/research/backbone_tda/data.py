"""Paired current-frame TDA targets and immutable frozen-encoder states."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch

from src.analysis.liquid_structure import persistence_image
from src.data.predictive_memory.prepare import file_hash, write_json
from src.project_runtime.paths import resolve_path
from src.research.local_predictability.native_data import NativeWindows


KINDS = ('mace', 'axial_gatr')


def check_deadline(config):
    if time.time() + 60 >= datetime.fromisoformat(config['deadline_utc']).timestamp():
        raise TimeoutError('TDA deadline reached; completed source shards are resumable')


def progress(root, stage, **values):
    record = dict(state='running', stage=stage,
                  updated_at=datetime.now(timezone.utc).isoformat(), **values)
    write_json(Path(root)/'status.json', record)
    print(json.dumps(record), flush=True)


def nearest_cloud(frame, count=80):
    """Keep the center and nearest atoms in the observed periodic local chart.

    Rank the actual float32 encoder positions; resolve exact ties by atom ID.
    No full-cell features, future frames, velocities or PTM classes enter TDA.
    """
    positions, ids = frame['positions'].numpy(), frame['ids'].numpy()
    if positions.shape != (len(ids), 3) or len(ids) < count:
        raise ValueError(f'Need at least {count} observed atoms, got {positions.shape}')
    distance2 = np.square(positions.astype(np.float64)).sum(1)
    selected = np.lexsort((ids, distance2))[:count]
    return positions[selected], ids[selected]


def save_arrays(path, identity, **arrays):
    path = Path(path)
    temporary = path.with_suffix('.building.npz')
    np.savez(temporary, **arrays)
    temporary.replace(path)
    write_json(path.with_suffix('.json'), dict(identity=identity, sha256=file_hash(path)))


def completed(path, identity):
    path = Path(path)
    receipt = path.with_suffix('.json')
    if not receipt.exists():
        return False
    saved = json.loads(receipt.read_text())
    if saved['identity'] != identity or saved['sha256'] != file_hash(path):
        raise ValueError(f'Completed TDA artifact identity/checksum changed: {path}')
    return True


def initialize_worker(cohort, audit):
    global _WINDOWS
    torch.set_num_threads(1)
    _WINDOWS = NativeWindows(cohort, audit, device='cpu', cpu_cache_gib=0, gpu_cache_gib=0)


def target_source(indices, path, identity, deadline):
    targets, neighbors = [], []
    for index in indices:
        check_deadline(dict(deadline_utc=deadline))
        row = _WINDOWS.rows[index]
        frame = _WINDOWS.frame(row['source_id'], row['center_id'], row['anchor'])
        cloud, ids = nearest_cloud(frame)
        targets.append(persistence_image(cloud))
        neighbors.append(ids)
    targets = np.stack(targets)
    if targets.shape != (len(indices), 144) or not np.isfinite(targets).all():
        raise ValueError(f'Invalid persistence images: {path}')
    save_arrays(path, identity, indices=indices, targets=targets, neighbor_ids=np.stack(neighbors))
    # Each job owns a source; release its memory maps before accepting another.
    _WINDOWS.raw.clear()
    return str(path)


def prepare_targets(config, parent, data, root, cache, identity):
    folder = cache/'targets'; folder.mkdir(parents=True, exist_ok=True)
    source_ids = data.labels['source_id']
    tasks = []
    for sid in np.unique(source_ids):
        indices = np.flatnonzero(source_ids == sid)
        path = folder/f'source-{sid:04d}.npz'
        if not completed(path, identity):
            tasks.append((indices, path, identity, config['deadline_utc']))
    progress(root, 'targets', completed_sources=150-len(tasks), sources=150)
    if tasks:
        with ProcessPoolExecutor(max_workers=config['target_workers'],
                mp_context=multiprocessing.get_context('spawn'), initializer=initialize_worker,
                initargs=(resolve_path(parent['data_output'])/'technical/cohort.json',
                          root/'source_audit.json')) as pool:
            futures = [pool.submit(target_source, *task) for task in tasks]
            for done, future in enumerate(as_completed(futures), 1):
                future.result()
                if done % 10 == 0 or done == len(tasks):
                    progress(root, 'targets', completed_sources=150-len(tasks)+done, sources=150)
    targets = np.empty((len(source_ids), 144), np.float32)
    for sid in np.unique(source_ids):
        path = folder/f'source-{sid:04d}.npz'
        with np.load(path) as shard:
            expected = np.flatnonzero(source_ids == sid)
            np.testing.assert_array_equal(shard['indices'], expected)
            targets[expected] = shard['targets']
    return targets


@torch.no_grad()
def extract_states(config, parent, data, root, cache, identity):
    from src.research.local_predictability.backbone_v2 import identity as model_identity, verify_gate
    from src.research.local_predictability.model_factory import build_model
    from src.research.local_predictability.native_runtime import ObservationPrefetcher
    from src.research.local_predictability.native_data import BoundedCache

    models = {}
    for kind in KINDS:
        path = resolve_path(parent['output'])/'technical'/kind/'physical_means/snapshot/best.pt'
        saved = torch.load(path, map_location='cpu', weights_only=False)
        expected = model_identity(parent, data, kind, 'physical_means', 'snapshot')
        expected['gate_receipt_sha256'] = verify_gate(parent, data, kind)
        if saved['identity'] != expected:
            raise ValueError(f'Frozen encoder identity differs: {path}')
        model = build_model(kind, 'physical_means', 'snapshot', parent).cuda().eval()
        model.load_state_dict(saved['model'])
        models[kind] = model.requires_grad_(False)
    # Single extraction pass, with both encoders consuming identical observations.
    # Input residency is operational only and does not alter the parent identity.
    data.windows.frames = BoundedCache(2 * 1024**3)
    data.windows.observations = BoundedCache(0)
    folder = cache/'states'; folder.mkdir(parents=True, exist_ok=True)
    source_ids = data.labels['source_id']
    states = {kind: np.empty((len(source_ids), 128), np.float32) for kind in KINDS}
    with ObservationPrefetcher(data.windows, config['prefetch_workers']) as inputs:
        for number, sid in enumerate(np.unique(source_ids), 1):
            check_deadline(config)
            indices = np.flatnonzero(source_ids == sid)
            path = folder/f'source-{sid:04d}.npz'
            if not completed(path, identity):
                values = {kind: [] for kind in KINDS}
                for batch, observed in inputs.batches(indices, 'snapshot', config['extraction_batch_size']):
                    check_deadline(config)
                    for kind, model in models.items():
                        values[kind].append(model.encoder(observed).cpu().numpy())
                save_arrays(path, identity, indices=indices,
                            **{k: np.concatenate(v) for k, v in values.items()})
            with np.load(path) as shard:
                np.testing.assert_array_equal(shard['indices'], indices)
                for kind in KINDS:
                    value = shard[kind]
                    if value.shape != (len(indices), 128) or not np.isfinite(value).all():
                        raise ValueError(f'Invalid encoder states: {path}, {kind}')
                    states[kind][indices] = value
            if number % 5 == 0 or number == 150:
                progress(root, 'states', completed_sources=number, sources=150)
    # Independently saved parent exports protect model-loading and row alignment.
    for kind in KINDS:
        base = resolve_path(parent['output'])/'technical'/kind/'physical_means/snapshot'
        for split in ('selection', 'calibration', 'test'):
            receipt = json.loads((base/f'{split}_export.json').read_text())
            path = base/f'{split}_predictions.npz'
            if (receipt['checkpoint_sha256'] != identity['checkpoints'][kind]
                    or receipt['sha256'] != file_hash(path)):
                raise ValueError(f'Parent export provenance changed: {path}')
            with np.load(path) as saved:
                for exported, label in (('source', 'source_id'), ('center', 'center_id'), ('anchor', 'anchor')):
                    np.testing.assert_array_equal(saved[exported], data.labels[label][saved['indices']])
                np.testing.assert_allclose(states[kind][saved['indices']], saved['state'],
                    rtol=2e-4, atol=2e-4, err_msg=f'{kind}: cross-device frozen export parity')
    del models
    torch.cuda.empty_cache()
    return states


def current_classes(parent, data):
    """Join observed anchor PTM labels only; never filter the fitting population."""
    release = json.loads((resolve_path(parent['data_output'])/'technical/release.json').read_text())
    result = np.empty(len(data.windows.rows), np.uint8)
    for source in release['sources']:
        indices = np.flatnonzero(data.labels['source_id'] == source['id'])
        with np.load(resolve_path(parent['cache'])/source['shard']) as shard:
            lookup = {int(atom): i for i, atom in enumerate(shard['atom_ids'])}
            centers = [lookup[int(atom)] for atom in data.labels['center_id'][indices]]
            result[indices] = shard['labels'][centers, data.labels['anchor'][indices]]
    return result
