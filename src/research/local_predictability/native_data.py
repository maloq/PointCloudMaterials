"""Raw native windows with immutable, byte-bounded observation caches.

The encoder path reads only declared observation frames. Physical packets use
the maintained producer and are kept separate from observations and assay labels.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.data.predictive_memory.observations import assemble, frame_observation
from src.data.predictive_memory.targets import physical_packet
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.project_runtime.paths import dataset_path
from .audit_sources import sha256

HORIZON_FRAMES = np.array([1, 4, 12, 32, 64, 128])
TEMPERATURES = [400, 450, 500, 510, 520]


def build_rows(cohort):
    sources = cohort['sources']
    if len(sources) != 150 or len({s['lineage'] for s in sources}) != 150:
        raise ValueError('Native index requires all 150 unique source lineages')
    if cohort['native_anchors'] != list(range(64, 665, 40)):
        raise ValueError('Native origins must be 48..498 ps at 30 ps intervals')
    if cohort['horizon_frames'] != HORIZON_FRAMES.tolist():
        raise ValueError('Unexpected future target horizons')
    rows = []
    for source in sources:
        centers = source['center_atom_ids']
        if len(centers) != 16 or len(set(centers)) != 16 or not set(centers) <= set(source['pool_atom_ids']):
            raise ValueError(f'Invalid frozen center selection: {source["id"]}')
        for center in centers:
            for anchor in cohort['native_anchors']:
                rows.append(dict(source_id=source['id'], center_id=center, anchor=anchor,
                    row_id=f'{source["id"]}:{center}:{anchor}',
                    split=source.get('validation_role', source['split']),
                    temperature_K=source['temperature_K']))
    if len(rows) != 38400 or len({r['row_id'] for r in rows}) != 38400:
        raise ValueError('Native comparison requires exactly 38,400 distinct rows')
    return rows


def selection_indices(rows, *, per_source=64):
    groups = {}
    for index, row in enumerate(rows):
        if row['split'] == 'selection':
            groups.setdefault(row['source_id'], []).append(index)
    chosen = []
    for source, indices in sorted(groups.items()):
        rng = np.random.default_rng(np.random.SeedSequence([20260919, source, 64]))
        chosen.extend(sorted(rng.choice(indices, per_source, replace=False).tolist()))
    if len(groups) != 15:
        raise ValueError('Checkpoint selection requires the frozen 15-source half')
    return chosen


def small_set_indices(rows):
    by_source = {}
    for index, row in enumerate(rows):
        if row['split'] == 'train':
            by_source.setdefault(row['source_id'], []).append(index)
    rng = np.random.default_rng(20260919)
    # Five distinct temperatures plus three other source lineages, never outcomes.
    chosen = []
    for temperature in TEMPERATURES:
        eligible = [s for s, indices in sorted(by_source.items()) if rows[indices[0]]['temperature_K'] == temperature]
        chosen.append(int(rng.choice(eligible)))
    remaining = sorted(set(by_source) - set(chosen))
    chosen.extend(rng.choice(remaining, 3, replace=False).tolist())
    return [int(i) for source in chosen for i in rng.choice(by_source[source], 4, replace=False)]


def byte_count(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (list, tuple)):
        return sum(byte_count(item) for item in value)
    if isinstance(value, dict):
        return sum(byte_count(item) for item in value.values())
    if hasattr(value, '__dataclass_fields__'):
        return byte_count(vars(value))
    return 0


class BoundedCache:
    def __init__(self, maximum_bytes):
        self.maximum_bytes = int(maximum_bytes)
        self.values = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = 0

    def get(self, key):
        if key not in self.values:
            self.misses += 1
            return None
        self.hits += 1
        value, size = self.values.pop(key)
        self.values[key] = value, size
        return value

    def put(self, key, value):
        size = byte_count(value)
        if size > self.maximum_bytes:
            return value
        if key in self.values:
            _, old = self.values.pop(key)
            self.bytes -= old
        while self.values and self.bytes + size > self.maximum_bytes:
            _, (_, old) = self.values.popitem(last=False)
            self.bytes -= old
        self.values[key] = value, size
        self.bytes += size
        return value


class NativeWindows:
    def __init__(self, cohort_path, audit_path, *, device='cuda', cpu_cache_gib=8, gpu_cache_gib=16):
        self.cohort_path = Path(cohort_path)
        self.cohort = json.loads(self.cohort_path.read_text())
        audit = json.loads(Path(audit_path).read_text())
        if audit['status'] != 'passed' or audit['counts'] != {'passed': 150}:
            raise ValueError('Require a complete successful raw-source audit')
        audited = {s['id']: s for s in audit['sources']}
        self.sources = {s['id']: s for s in self.cohort['sources']}
        for sid, source in self.sources.items():
            verified = audited[sid]
            if any(source[k] != verified[k] for k in ('lineage', 'split', 'dataset', 'relative_trajectory_path', 'manifest_sha256')):
                raise ValueError(f'Frozen cohort does not match audited source {sid}')
        self.rows = build_rows(self.cohort)
        self.device = torch.device(device)
        self.raw = {}
        self.frames = BoundedCache(cpu_cache_gib * 1024 ** 3)
        self.observations = BoundedCache(gpu_cache_gib * 1024 ** 3)
        self.accessed_frames = []
        self.trace_access = False

    def trajectory(self, sid):
        if sid not in self.raw:
            source = self.sources[sid]
            root = dataset_path(source['dataset']) / source['relative_trajectory_path']
            if sha256(root / 'manifest.json') != source['manifest_sha256']:
                raise ValueError(f'Manifest changed after audit: {sid}')
            self.raw[sid] = ShootingBinaryTrajectory.load(root)
        return self.raw[sid]

    def frame(self, sid, center, frame):
        key = (sid, center, frame)
        value = self.frames.get(key)
        if value is None:
            raw = self.trajectory(sid)
            if self.trace_access:
                self.accessed_frames.append(key)
            value = frame_observation(raw.positions[frame], raw.velocities[frame],
                raw.box_high[frame].astype(np.float64) - raw.box_low[frame].astype(np.float64),
                raw.atom_ids, center, 17., 5.)
            self.frames.put(key, value)
        return value

    def observation(self, index, variant):
        if variant not in ('snapshot', 'history12', 'repeat12'):
            raise ValueError(variant)
        row = self.rows[index]
        key = (row['row_id'], variant)
        value = self.observations.get(key)
        if value is None:
            anchor = row['anchor']
            frames = [anchor] if variant == 'snapshot' else list(range(anchor - 16, anchor + 1))
            observed = [self.frame(row['source_id'], row['center_id'],
                         anchor if variant == 'repeat12' else frame) for frame in frames]
            value = assemble(observed, np.array(frames) * .75, radius=17., cutoff=5.).to(self.device)
            self.observations.put(key, value)
        return value

    def packet(self, index, lag=0):
        row = self.rows[index]
        # Targets are physically separate from the observation builder. Avoid
        # constructing a 17 A graph solely to make a 7 A physical packet.
        raw = self.trajectory(row['source_id']); frame = row['anchor'] + int(lag)
        position = raw.positions[frame].astype(np.float64)
        center = int(np.searchsorted(raw.atom_ids, row['center_id']))
        if raw.atom_ids[center] != row['center_id']:
            raise ValueError('Tracked center ID missing')
        box = raw.box_high[frame].astype(np.float64) - raw.box_low[frame].astype(np.float64)
        x = position - position[center]; x -= box * np.round(x / box)
        keep = np.linalg.norm(x, axis=-1) < 7.
        velocity = raw.velocities[frame]
        u = velocity[keep].astype(np.float64) - velocity[center].astype(np.float64)
        return physical_packet(x[keep].astype(np.float32), u.astype(np.float32))

    def statistics(self):
        return {name: dict(bytes=cache.bytes, maximum_bytes=cache.maximum_bytes,
                           entries=len(cache.values), hits=cache.hits, misses=cache.misses)
                for name, cache in [('frame_cache', self.frames), ('device_cache', self.observations)]}


def conditions(rows, statistics=None):
    result = np.zeros((len(rows), 7), np.float64)
    for i, row in enumerate(rows):
        result[i, TEMPERATURES.index(int(row['temperature_K']))] = 1
        result[i, 5] = row['anchor'] * .75 / 600.
    result[:, 6] = result[:, 5] ** 2
    if statistics is None:
        train = np.array([r['split'] == 'train' for r in rows])
        if not train.any():
            raise ValueError('Conditions require a training population')
        statistics = dict(mean=result[train].mean(0).tolist(),
                          scale=np.maximum(result[train].std(0), 1e-4).tolist(),
                          definition='temperature one-hot; t/600 and (t/600)^2; all seven standardized on physical training rows')
    result = (result - np.array(statistics['mean'])) / np.array(statistics['scale'])
    return result.astype(np.float32), statistics


class SourceSampler:
    def __init__(self, rows, allowed=None):
        self.rng = np.random.default_rng(20260919)
        self.by_source = {}
        for index in range(len(rows)) if allowed is None else allowed:
            row = rows[index]
            if row['split'] != 'train':
                continue
            self.by_source.setdefault(row['source_id'], []).append(index)
        self.sources = np.array(sorted(self.by_source))
        if len(self.sources) < 8:
            raise ValueError('Effective batch eight requires at least eight training sources')

    def batch(self):
        return [int(self.rng.choice(self.by_source[int(source)])) for source in self.rng.choice(self.sources, 8)]

    def state_dict(self):
        return self.rng.bit_generator.state

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state
