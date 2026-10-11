"""Freeze spatial target columns from an existing training-only transform."""
import json

import numpy as np

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.project_runtime.paths import resolve_path


def freeze(packed_cache, destination, families=('bond_order','cna','tda')):
    if tuple(families) not in (('bond_order','tda'),('bond_order','cna','tda')):
        raise ValueError('Select bond_order/tda, optionally including cna')
    root=resolve_path(packed_cache)
    manifest=json.loads((root/'manifest.json').read_text())
    if any(s['task']['role']!='train' for s in manifest['shards']):
        raise ValueError('Target selection must come from training rows only')
    transform=root/'target-standardization.npz'
    transform_sha256=sha(transform)
    if transform_sha256!=manifest['files']['target-standardization.npz']:
        raise ValueError('Changed training target transform')
    with np.load(transform) as values:
        active=values['active']
    columns=[c for c,a in zip(manifest['columns'],active)
             if a and c['family'] in families]
    result=dict(protocol='spatial-training-active-columns-v1',columns=columns,families=list(families),
        reference_packed_identity=manifest['identity'],reference_manifest_sha256=sha(root/'manifest.json'),
        reference_transform_sha256=transform_sha256,reference_training_rows=manifest['binding']['rows'],
        selection='training SD >= 1e-4 within declared families; no held-out errors or labels',
        training_ancestry=sorted({s['task']['lineage'] for s in manifest['shards']}))
    result['identity']=digest(result)
    target=resolve_path(destination)
    if target.exists() and json.loads(target.read_text())!=result:
        raise ValueError('Target selection changed; choose a new selection file')
    write_json(target,result)
    return result


def load(spec):
    path=resolve_path(spec['path'])
    if sha(path)!=spec['sha256']:raise ValueError(f'Changed target selection: {path}')
    value=json.loads(path.read_text());identity=value.pop('identity')
    if digest(value)!=identity or value['protocol']!='spatial-training-active-columns-v1':
        raise ValueError('Invalid target selection contract')
    return [c['name'] for c in value['columns']]
