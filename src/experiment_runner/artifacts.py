"""Readable run outputs and explicit access to the two repository analysis layouts."""

import json
import os
from pathlib import Path


def result_folders(root):
    root = Path(root)
    for name in ('plots', 'tables', 'technical'):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def analysis_artifacts(root):
    """Resolve recorded legacy layouts; new producers own a named scientific bundle."""
    root = Path(root)
    if (root / 'analysis_metrics.json').is_file() or (
        root / 'analysis_inference_cache.npz.meta.json'
    ).is_file():
        return root  # Repository layout before September 12, 2026.
    legacy = root / 'technical'
    if any((legacy / name).exists() for name in (
        'analysis_metrics.json', 'analysis_inference_cache.npz.meta.json', 'snapshots', 'real_md')):
        return legacy
    return root / 'analyses/standard-v1/data'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)
