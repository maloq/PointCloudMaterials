"""Readable run outputs and explicit access to the two repository analysis layouts."""

import json
import hashlib
import os
from pathlib import Path


def read_json_object(path: Path) -> dict:
    """Read required UTF-8 metadata and reject a non-object JSON root."""
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}, got {type(value).__name__}.")
    return value


def file_hash(path):
    """Stream an artifact's SHA-256 without loading it into host memory."""
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def implementation_hashes(*paths):
    """Bind explicit repository dependencies without importing their producers."""
    repository = Path(__file__).resolve().parents[2]
    return {path: file_hash(repository / path) for path in paths}


def json_digest(value, *, allow_nan=False, separators=None):
    """Hash sorted JSON using the producer's declared serialization policy."""
    payload = json.dumps(value, sort_keys=True, allow_nan=allow_nan, separators=separators)
    return hashlib.sha256(payload.encode()).hexdigest()


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


def write_json(path, value, *, allow_nan=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=allow_nan) + '\n')
    temp.replace(path)
