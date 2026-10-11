"""Checksum-bound access to trajectories validated by repository converters."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from src.experiment_runner.artifacts import file_hash
from src.project_runtime.paths import resolve_path


def read(path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=8)
def arrays(path, manifest_sha256):
    path = resolve_path(path)
    manifest = read(path / 'manifest.json')
    if file_hash(path / 'manifest.json') != manifest_sha256:
        raise ValueError(f'Source manifest changed: {path}')
    if manifest['state'] != 'complete' or manifest['format'] not in (
            'pointcloudmaterials.shooting_trajectory', 'pointcloudmaterials.temporal_lammps_trajectory'):
        raise ValueError(f'Unsupported or incomplete trajectory: {path}')
    result = {
        name: np.load(path / manifest['arrays'][name]['file'], mmap_mode='r', allow_pickle=False)
        for name in ('positions', 'atom_ids', 'atom_types', 'timesteps', 'box_low', 'box_high')
    }
    if np.any(result['atom_types'] != 1):
        raise ValueError(f'Pure-element source contains unexpected atom types: {path}')
    return result


def geometry(source, frame):
    part_index, raw_frame = source['frames'][frame]
    part = source['parts'][part_index]
    a = arrays(part['path'], part['manifest_sha256'])
    box = a['box_high'][raw_frame].astype(float) - a['box_low'][raw_frame].astype(float)
    points = np.mod(a['positions'][raw_frame].astype(float), box)
    return points, box


def frame_geometry(raw, frame):
    box = raw.box_high[frame].astype(float) - raw.box_low[frame].astype(float)
    points = np.mod(raw.positions[frame].astype(float), box)
    return points, box
