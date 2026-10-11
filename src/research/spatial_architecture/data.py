"""Freeze spatial-only monitoring panels using the selected release's train moments."""
from collections import Counter

import numpy as np

from src.project_runtime.paths import resolve_path
from src.research.liquid_predictability.rich_multimaterial_data import CachedPatches
from src.research.structural_response.data import cache, digest, read, sha, spatial_data, write_json


def prepare_monitoring(c):
    dest = cache(c)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = resolve_path(c['cache']) / 'manifest.json'
    packed_path = resolve_path(c['loader']['packed_cache']) / 'manifest.json'
    binding = dict(dataset=c['prepared_identity'], manifest_sha256=sha(manifest),
        packed_manifest_sha256=sha(packed_path), monitoring=c['monitoring'],
        role=c['data']['evaluation_role'], seed=c['seed'], descriptors=c['descriptors'])
    identity = digest(binding)
    if (dest/'complete.json').exists():
        done = read(dest/'complete.json')
        if done['identity'] != identity or any(sha(dest/f) != h for f,h in done['files'].items()):
            raise ValueError('Changed spatial monitoring release; use a new derived cache')
        return done
    data, columns, weights = spatial_data(c)
    held = CachedPatches(c, c['data']['evaluation_role'])
    if held.identity != c['prepared_identity'] or held.columns != data.columns:
        raise ValueError('Training and evaluation descriptor releases differ')
    held.use_transform(data)
    packed = read(packed_path)
    train_anc = {s['task']['lineage'] for s in data.shards}
    held_anc = {s['task']['lineage'] for s in held.shards}
    if train_anc & held_anc or any(s['task']['role'] != 'train' for s in data.shards):
        raise ValueError('Held-out ancestry overlaps spatial training')
    n = c['monitoring']['rows']
    selected = np.sort(np.random.default_rng(c['seed']+71).choice(len(held), n, replace=False))
    panels = {'pooled': selected}
    by_material = held.audit_ids(c['monitoring']['material_rows'], c['seed']+72)
    panels.update({m: by_material[m] for m in c['monitoring']['materials']})
    metadata = dict(packed_identity=data.packed_identity, rows=len(data),
        material_rows=packed['material_rows'], sources=packed['source_count'],
        ancestry_groups=len(train_anc), descriptor_columns=[data.columns[i] for i in columns],
        weights=weights.tolist(), active_columns=int((weights > 0).sum()),
        validation_rows=n, evaluation_pool_rows=len(held), evaluation_role=held.role,
        evaluation_ancestry_groups=len(held_anc), ancestry_overlap=0,
        selection_rule='fixed final epoch; pooled and material panels are diagnostics only',
        binding=binding, panels={})
    files = []
    for label, ids in panels.items():
        folder = dest if label == 'pooled' else dest/('monitor-'+label)
        folder.mkdir(parents=True, exist_ok=True)
        batch = held.batch(ids)
        which = np.searchsorted(held.ends, ids, side='right')
        materials = np.asarray([held.shards[i]['task']['material'] for i in which])
        lineages = np.asarray([held.shards[i]['task']['lineage'] for i in which])
        np.savez(folder/'spatial-selection.npz', positions=batch['positions'],
            target=batch['target'][:, columns], rows=ids, materials=materials, lineages=lineages)
        metadata['panels'][label] = dict(rows=len(ids), material_rows=dict(Counter(materials.tolist())),
            ancestry_groups=len(set(lineages)), overlap_with_pooled=int(np.intersect1d(ids, selected).size))
        if label != 'pooled':
            write_json(folder/'spatial.json', dict(weights=weights.tolist(), validation_rows=len(ids)))
            files.append(str((folder/'spatial.json').relative_to(dest)))
        files.append(str((folder/'spatial-selection.npz').relative_to(dest)))
    write_json(dest/'spatial.json', metadata)
    files.append('spatial.json')
    done = dict(identity=identity, binding=binding, files={f:sha(dest/f) for f in files})
    write_json(dest/'complete.json', done)
    data.close()
    held.close()
    return done
