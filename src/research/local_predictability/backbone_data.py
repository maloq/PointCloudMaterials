"""Versioned native populations and immutable full-timeline packet joins."""
from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np
import torch
from src.data.predictive_memory.prepare import file_hash, write_json
from src.project_runtime.paths import resolve_path
from .native_data import NativeWindows, conditions, HORIZON_FRAMES, small_set_indices, selection_indices


def join_packets(rows, source_id, shard):
    """Join by source/center/physical frame, never by label or risk membership."""
    indices = np.array([i for i, row in enumerate(rows) if row['source_id'] == source_id])
    ids = np.asarray(shard['atom_ids'])
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f'Duplicate packet centers in source {source_id}')
    lookup = {int(atom): i for i, atom in enumerate(ids)}
    centers = np.array([lookup[rows[i]['center_id']] for i in indices])
    frames = np.array([rows[i]['anchor'] for i in indices])[:, None] + np.r_[0, HORIZON_FRAMES]
    packet = shard['packet']
    if packet.shape != (len(ids), len(shard['times_ps']), 128):
        raise ValueError(f'Unexpected packet shape for source {source_id}: {packet.shape}')
    np.testing.assert_allclose(shard['times_ps'][frames], frames * .75, rtol=0, atol=1e-6)
    values = packet[centers[:, None], frames]
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError(f'Invalid float32 physical packets in source {source_id}')
    return indices, values


def population_splits(labels, objective):
    if objective == 'physical_means':
        allowed = np.arange(len(labels['split']))
    elif objective == 'onset':
        allowed = np.flatnonzero(labels['risk'])
    else:
        raise ValueError(f'Unknown objective: {objective}')
    return {s: allowed[labels['split'][allowed] == s] for s in ('train', 'selection', 'calibration', 'test')}


@dataclass
class BackboneData:
    windows: NativeWindows
    labels: dict
    cond: torch.Tensor
    events: torch.Tensor
    targets: torch.Tensor
    raw_targets: np.ndarray
    identity: dict

    def splits(self, objective):
        return population_splits(self.labels, objective)

    def selection(self, objective):
        if objective == 'physical_means':
            return selection_indices(self.windows.rows)
        allowed = self.splits(objective)['selection']
        chosen = []
        for sid in np.unique(self.labels['source_id'][allowed]):
            eligible = allowed[self.labels['source_id'][allowed] == sid]
            rng = np.random.default_rng(np.random.SeedSequence([20260919, int(sid), 64]))
            chosen.extend(sorted(rng.choice(eligible, min(64, len(eligible)), replace=False).tolist()))
        return chosen

    def diagnostic(self):
        indices = small_set_indices(self.windows.rows)
        values = self.raw_targets[indices].astype(np.float64)
        mean = values.reshape(-1, 128).mean(0)
        scale = np.maximum(values.reshape(-1, 128).std(0), 1e-4)
        targets = torch.zeros_like(self.targets)
        targets[indices] = torch.as_tensor((values-mean)/scale, dtype=torch.float32, device=targets.device)
        identity = dict(indices=indices, normalizer=dict(mean=mean.tolist(), scale=scale.tolist()),
            population='32 fixed training windows, present and six futures; diagnostic only')
        return replace(self, targets=targets), indices, identity


def prepare_data(config, root, device='cuda'):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    source_root = resolve_path(config['data_output']) / 'technical'
    cache = resolve_path(config['cache'])
    release_path = source_root / 'release.json'
    release = json.loads(release_path.read_text())
    if release['state'] != 'complete' or file_hash(cache/'release.json') != file_hash(release_path):
        raise ValueError('Require the complete, identical local and cache releases')
    if release['cohort_sha256'] != file_hash(source_root/'cohort.json'):
        raise ValueError('Release and native cohort identities differ')
    if release['plan_sha256'] != file_hash(resolve_path(config['plan'])):
        raise ValueError('Target definitions differ from the frozen release plan')
    if release['horizon_frames'] != HORIZON_FRAMES.tolist() or release['cadence_ps'] != .75:
        raise ValueError('Physical targets require .75, 3, 9, 24, 48, 96 ps horizons')
    gates = json.loads((source_root/'assay_gates.json').read_text())
    if not all(gates[k] for k in ('source_integrity', 'assay_integrity', 'coverage')):
        raise ValueError(f'Release audit failed: {gates}')
    sources = []
    for source in release['sources']:
        if not source['checksum_verified']:
            raise ValueError(f"Source arrays were not audited: {source['id']}")
        sources.append({**{k: source[k] for k in ('id', 'lineage', 'split', 'dataset',
            'relative_trajectory_path', 'manifest_sha256')}, 'status': 'passed'})
    audit = root/'source_audit.json'
    write_json(audit, dict(status='passed', counts={'passed': len(sources)}, sources=sources,
        producer_release_sha256=file_hash(release_path)))
    windows = NativeWindows(source_root/'cohort.json', audit, device=device,
        cpu_cache_gib=config['cpu_cache_gib'], gpu_cache_gib=config['gpu_cache_gib'])
    with np.load(source_root/'native_rows.npz') as values:
        labels = dict(values)
    shared = resolve_path(config['shared'])
    with np.load(shared/'native_index.npz') as index:
        for key in ('source_id', 'center_id', 'anchor', 'split'):
            actual = np.array([row[key] for row in windows.rows])
            np.testing.assert_array_equal(actual, labels[key], err_msg=f'Label join: {key}')
            np.testing.assert_array_equal(actual, index[key], err_msg=f'Frozen index join: {key}')
    raw = np.empty((len(windows.rows), 7, 128), np.float32)
    filled = np.zeros(len(raw), dtype=bool)
    for source in release['sources']:
        path = cache/source['shard']
        if file_hash(path) != source['shard_sha256']:
            raise ValueError(f'Changed immutable packet shard: {path}')
        with np.load(path) as shard:
            indices, values = join_packets(windows.rows, source['id'], shard)
        if filled[indices].any():
            raise ValueError(f"Duplicate source packet join: {source['id']}")
        raw[indices] = values; filled[indices] = True
    if not filled.all():
        raise ValueError('Incomplete full native physical population')
    normalizer = release['normalizer']
    mean, scale = np.array(normalizer['mean']), np.array(normalizer['scale'])
    if mean.shape != (128,) or scale.shape != (128,) or not (scale >= 1e-4).all():
        raise ValueError('Invalid release train-only normalizer')
    stats = json.loads((shared/'conditions.json').read_text())
    cond, _ = conditions(windows.rows, stats)
    identity = dict(release_sha256=file_hash(release_path), cohort_sha256=release['cohort_sha256'],
        labels_sha256=file_hash(source_root/'native_rows.npz'),
        conditions_sha256=file_hash(shared/'conditions.json'), normalizer=normalizer,
        horizons_ps=(HORIZON_FRAMES*.75).tolist(), rows=len(raw))
    return BackboneData(windows, labels, torch.tensor(cond, device=device),
        torch.tensor(labels['event_bin'], dtype=torch.long, device=device),
        torch.tensor((raw-mean)/scale, dtype=torch.float32, device=device), raw, identity)
