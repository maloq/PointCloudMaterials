"""Declared past-only observations for the fixed-endpoint/site-persistence assay."""
import hashlib

import numpy as np


def packet(bank, rows, columns, observation, seed):
    frames = np.asarray(observation['frames'], int)
    if (frames.ndim != 1 or not len(frames) or np.any(np.diff(frames) <= 0)
            or frames[0] < 0 or frames[-1] >= rows['indices'].shape[1]):
        raise ValueError(f'Invalid observed-frame indices: {observation}')
    values = bank[rows['indices'][:, frames]][:, :, columns]
    kind = observation['kind']
    if kind == 'snapshot':
        if len(frames) != 1:
            raise ValueError('Snapshot requires exactly one observed frame')
        result = values[:, 0]
    elif kind == 'mean':
        result = values.mean(1)
    elif kind == 'early_plus_change':
        result = np.concatenate((values[:, 0], values[:, -1] - values[:, 0]), 1)
    elif kind == 'early_repeated':
        result = np.concatenate((values[:, 0], np.zeros_like(values[:, 0])), 1)
    else:
        if kind == 'repeat':
            # Only the endpoint is consumed; eight slots match the history layout.
            values = np.repeat(values[:, -1:], observation['slots'], axis=1)
        elif kind == 'shuffle_past':
            # Keep the current frame fixed. A shared random permutation within each
            # matched set cannot distinguish its case from its four controls.
            values = values.copy()
            for pair in np.unique(rows['pair']):
                key = hashlib.sha256(f'{seed}:{pair}'.encode()).digest()[:8]
                rng = np.random.default_rng(int.from_bytes(key, 'little'))
                order = np.r_[rng.permutation(len(frames) - 1), len(frames) - 1]
                ids = np.flatnonzero(rows['pair'] == pair)
                values[ids] = values[ids][:, order]
        elif kind == 'changes':
            values = values - values[:, :1]
        elif kind != 'history':
            raise ValueError(f'Unknown temporal input {kind}')
        result = np.concatenate((values.reshape(len(values), -1), values.mean(1),
                                 values.std(1), values[:, -1] - values[:, 0]), 1)
    if not np.isfinite(result).all():
        raise FloatingPointError(f'Nonfinite temporal input: {observation}')
    return np.asarray(result, dtype=np.float32)


def context(observation, cadence):
    frames = observation['frames']
    return dict(observation=observation, observed_frames=len(frames),
        observation_offsets_from_appearance_ps=[(i - 8) * cadence for i in frames],
        latest_observed_lead_ps=(8 - frames[-1]) * cadence,
        observed_span_ps=(frames[-1] - frames[0]) * cadence,
        motion='descriptor/embedding changes only; no velocity or atom displacement input',
        conditions=[], labels_as_inputs=False,
        permutation='same deterministic permutation of past slots within matched set; endpoint fixed'
        if observation['kind'] == 'shuffle_past' else None)
