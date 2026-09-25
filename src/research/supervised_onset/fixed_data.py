"""Graph adapter for the immutable Al64 sample contract; no resampling or targets fitted."""
import json
from pathlib import Path

import numpy as np

from src.data.fixed_cohort.dataset import read_release, verify_release
from src.data.fixed_cohort.protocol import assert_prediction_rows
from src.project_runtime.paths import resolve_path
from src.research.structural_state.data import graph_arrays
from .common import sha, digest, write_json


def prepare(study):
    from .data import audit_population
    c = study.config
    declaration = c['fixed_dataset']
    root, plan = read_release(declaration['root'])
    if declaration['identity'] != plan['identity'] or declaration['track'] != 'all64':
        raise ValueError('Require the pinned complete all64 release')
    if c['encoder']['radius'] != plan['config']['radius_A']:
        raise ValueError('Encoder support must match the fixed dataset radius')
    benchmark = json.loads((root/'benchmark/manifest.json').read_text())
    inputs = dict(release_identity=plan['identity'], track='all64',
        manifest_sha256=sha(root/'benchmark/manifest.json'),
        population_sha256=sha(root/'benchmark/population.npz'),
        producer_sha256=sha(__file__), graph_producer_sha256=sha(Path(graph_arrays.__code__.co_filename)),
        radius_A=c['encoder']['radius'], cutoff_A=c['encoder']['cutoff'])
    manifest = study.cache/'manifest.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved['inputs'] != inputs:
            raise ValueError('Fixed graph cache producer or input changed; use a fresh cache')
        for name, checksum in saved['files'].items():
            if sha(study.cache/name) != checksum:
                raise ValueError(f'Fixed graph cache checksum mismatch: {name}')
        return saved
    verify_release(root)
    with np.load(root/'benchmark/population.npz') as arrays:
        pop = {k: arrays[k] for k in arrays.files}
    pop['event'] = pop['event'].astype(np.int64)  # torch categorical indices
    pop['graph'] = pop['patch_index'].copy()
    sources = [dict(s, split=s['role']) for s in plan['sources']]
    counts = audit_population(pop, sources)
    assert_prediction_rows(pop['sample_id'], pop['sample_id'])
    study.cache.mkdir(parents=True, exist_ok=True)
    stats = {}
    fit = np.flatnonzero(pop['role'] == 'train')
    for domain in ('hot', 'cold'):
        patches = [None] * len(pop['event'])
        for sid in np.unique(pop['source']):
            x = np.load(root/'benchmark/sources'/str(sid)/f'{domain}.npy', mmap_mode='r')
            x = x.reshape(-1, *x.shape[2:])
            for row in np.flatnonzero(pop['source'] == sid):
                positions = x[pop['patch_index'][row]]
                patch = positions[np.linalg.norm(positions, axis=1) < c['encoder']['radius']]
                if len(patch) < 13 or np.any(patch[0]) or not np.isfinite(patch).all():
                    raise ValueError(f'Invalid fixed patch: {domain}/{sid}/{row}')
                patches[row] = patch
        arrays = graph_arrays(patches, c['encoder']['cutoff'])
        for name, value in arrays.items():
            np.save(study.cache/f'{domain}-{name}.npy', value)
        stats[domain] = dict(nodes=len(arrays['positions']), edges=arrays['edges'].shape[1],
            n_ref=float(np.mean([len(patches[i]) for i in fit])),
            d0=float(np.median([np.linalg.norm(patches[i][1:], axis=1).min() for i in fit])))
        del arrays, patches
        print(json.dumps(dict(stage='fixed_graphs_complete', domain=domain, **stats[domain])), flush=True)
    np.savez(study.cache/'population.npz', **pop)
    result = dict(state='complete', inputs=inputs, counts=counts, domains=stats,
        fixed_dataset=declaration, descriptors=[],
        sources=[dict(id=s['id'], role=s['role'], lineage=s['lineage'],
                      manifest_sha256=s['manifest_sha256']) for s in sources],
        coordinate_units='Angstrom', support=plan['inputs']['positions'],
        potential_sha256=plan['potential_sha256'], test_status=benchmark['test_status'],
        files={p.name:sha(p) for p in study.cache.iterdir() if p.suffix in ('.npy','.npz')})
    result['identity'] = digest(result)
    write_json(manifest, result)
    return result
