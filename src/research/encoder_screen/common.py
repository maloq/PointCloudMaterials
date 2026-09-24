"""Immutable identities and atomic queue receipts."""
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*2**20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.writing')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def checked(path, expected):
    if sha(path) != expected:
        raise ValueError(f'Input changed: {path}')
    return Path(path)


def load_config(path):
    """Resolve only the declared machine-path fields of this protocol."""
    from src.project_runtime.paths import resolve_path
    config = json.loads(Path(path).read_text())
    if config['protocol'] != 'native_snapshot_physical_screen_v1':
        raise ValueError(f'Unsupported snapshot protocol: {config["protocol"]}')
    for key in ('output', 'reference', 'reuse_geoframe'):
        config[key] = str(resolve_path(config[key]).resolve())
    for task in config['tasks']:
        for key in ('checkpoint', 'producer'):
            task[key] = str(resolve_path(task[key]).resolve())
        task['producer_files'] = {str(resolve_path(k).resolve()): v for k, v in task['producer_files'].items()}
    return config
