"""The sample contract is independent of models, losses and random training seeds."""
import hashlib
import json
from pathlib import Path

import numpy as np

from src.research.forecast_crystallization.local_metrics import first_sustained_onset, risk_windows

ROLES = ('train', 'selection', 'calibration', 'test')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def audit_sources(sources, counts, centers):
    for field in ('id', 'lineage', 'manifest_sha256'):
        if len({s[field] for s in sources}) != len(sources):
            raise ValueError(f'Duplicate source/ancestor: {field}')
    actual = {role: sum(s['role'] == role for s in sources) for role in ROLES}
    if actual != counts or sum(actual.values()) != len(sources):
        raise ValueError(f'Source roles changed: {actual}; expected {counts}')
    for s in sources:
        ids = s['center_atom_ids']
        if len(ids) != centers or ids != sorted(set(ids)):
            raise ValueError(f'Expected {centers} sorted distinct centers: {s["id"]}')
        if not set(s['legacy_center_atom_ids']).issubset(ids):
            raise ValueError(f'Legacy centers absent from fixed pool: {s["id"]}')


def onset_rows(labels, frames, config):
    """Only fully observed prospective windows; confirmation frames never enter inputs."""
    labels = np.asarray(labels)
    frames = np.asarray(frames, dtype=np.int64)
    persistence = config['onset_persistence_frames']
    steps = np.asarray(config['horizons_ps']) / config['cadence_ps']
    if not np.array_equal(steps, steps.astype(int)):
        raise ValueError('Horizon must be representable on the observed timeline')
    if (np.any(np.diff(frames) <= 0) or
        np.any(frames < config['negative_history_frames'] - 1) or
        np.any(frames + int(steps[-1]) + persistence - 1 >= labels.shape[1])):
        raise ValueError('Origins require preceding observations and full future confirmation')
    crystal = np.isin(labels, [1, 2, 3])
    onset = first_sustained_onset(crystal, persistence)
    risk = risk_windows(crystal, onset, frames, config['negative_history_frames']).T
    frame_index, center_index = np.nonzero(risk)
    delay = (onset[center_index] - frames[frame_index]) * config['cadence_ps']
    return dict(frame_index=frame_index, center_index=center_index,
                event=np.searchsorted(config['horizons_ps'], delay).astype(np.int8),
                delay=delay.astype(np.float32), onset_frame=onset[center_index])


def centered(points, box, rows, neighbors):
    offsets = points[neighbors] - points[rows, None]
    offsets -= box * np.rint(offsets / box)
    if not np.isfinite(offsets).all() or np.any(offsets[:, 0]):
        raise ValueError('Invalid centered periodic patch')
    return offsets.astype(np.float32)


def assert_prediction_rows(expected, actual):
    """No model-specific omissions, duplicates, reordering or replacement populations."""
    expected, actual = np.asarray(expected), np.asarray(actual)
    if not np.array_equal(expected, actual) or len(np.unique(actual)) != len(actual):
        raise ValueError('Predictions must cover the exact ordered fixed evaluation sample IDs')
