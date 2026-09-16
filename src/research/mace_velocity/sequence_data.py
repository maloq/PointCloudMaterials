"""Verified position/velocity sequence IO for the native encoder."""
import hashlib
from pathlib import Path

import numpy as np

from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.experiment_runner.registry import sha256
from src.project_runtime.paths import resolve_path


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix('.building.npz')
    np.savez(temporary, **arrays)
    temporary.replace(path)


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
