"""Native frozen exports through their recorded producer and leased cache."""
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.project_runtime.paths import resolve_path
from src.research.equivariant_context.cache import RetainedCache
from .common import folder, result, read


def binding(c, spec):
    m = read(folder(c) / 'manifest.json')
    return dict(dataset_identity=m['identity'], model=spec, positions_sha256=m['files']['positions.npy'],
                inference_sha256=sha(Path(__file__).parents[1] / 'birth_prediction/infer.py'))


def encode(c, index):
    spec = dict(c['encoders'][index])
    for k in ('checkpoint', 'producer'):
        spec[k] = str(resolve_path(spec[k]))
    meta = binding(c, c['encoders'][index])
    key = digest(meta)
    with RetainedCache(resolve_path(c['feature_cache']), 6).lease(key, deadline=time.time() + 14400, metadata=meta) as root:
        if not (root / 'complete.json').exists():
            request = dict(model=spec, positions=str(folder(c) / 'positions.npy'),
                           destination=str(root / 'states.npy'), batch_size=256)
            path = result(c) / 'technical' / f'inference-{spec["name"]}.json'
            write_json(path, request)
            subprocess.run([sys.executable, str(Path(__file__).parents[1] / 'birth_prediction/infer.py'), str(path)],
                           check=True, cwd=spec['producer'], env=dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'))
            write_json(root / 'complete.json', dict(metadata=meta, sha256=sha(root / 'states.npy')))
        done = read(root / 'complete.json')
        if done['metadata'] != meta or sha(root / 'states.npy') != done['sha256']:
            raise ValueError('Changed leased encoder features')
        write_json(result(c) / 'technical' / f'features-{spec["name"]}.json', dict(key=key, metadata=meta, path=str(root)))


def bank(c, data, manifest, arm):
    if arm['bank'] == 'prior':
        return np.zeros((len(data['parent']), 0), np.float32)
    if arm['bank'] in ('descriptors', 'density'):
        prefixes = arm['prefixes']
        ix = [i for i, name in enumerate(manifest['columns']) if any(name.startswith(p) for p in prefixes)]
        if not ix:
            raise ValueError(f'Empty descriptor bank: {arm}')
        return data['descriptors'][:, ix]
    spec = next(s for s in c['encoders'] if s['name'] == arm['bank'])
    meta = binding(c, spec)
    key = digest(meta)
    with RetainedCache(resolve_path(c['feature_cache']), 6).lease(key, deadline=time.time() + 3600, metadata=meta, shared=True) as root:
        receipt = read(root / 'complete.json')
        if receipt['metadata'] != meta or sha(root / 'states.npy') != receipt['sha256']:
            raise ValueError('Frozen states unavailable or changed; rerun encode')
        z = np.load(root / 'states.npy')
    if arm.get('add_descriptors'):
        ix = [i for i, name in enumerate(manifest['columns']) if name.startswith(('bond_order/', 'tda/'))]
        z = np.concatenate([z, data['descriptors'][:, ix]], 1)
    return z
