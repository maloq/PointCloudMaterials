"""Reuse frozen observations; derive distances and independent diagnostic scan paths."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

from src.data.fixed_cohort.protocol import sha, write_json, digest
from src.project_runtime.paths import resolve_path
from src.research.crystallization_origin import ancestry
from src.research.crystallization_origin.extract import raw_source, frame_geometry
from src.research.structured_context.geometry import stencil, representatives
from src.research.encoder_context.geometry import physical_targets
from src.research.equivariant_context.cache import RetainedCache
from .common import study


def observations(points, box, tree, queries, solid, labels):
    unique, inverse = np.unique(queries, return_inverse=True)
    neighbors = tree.query(points[unique], k=80, workers=1)[1]
    np.testing.assert_array_equal(neighbors[:, 0], unique)
    xyz = points[neighbors] - points[unique, None]
    xyz -= box * np.rint(xyz / box)
    valid = np.linalg.norm(xyz, axis=-1) < 8.
    visible = (solid[neighbors] & valid).any(1)[inverse].reshape(queries.shape)
    ptm = (np.isin(labels[neighbors], [1, 2, 3]) & valid).any(1)[inverse].reshape(queries.shape)
    with torch.no_grad():
        descriptor = physical_targets(torch.as_tensor(xyz, dtype=torch.float32)).numpy()
    desc = descriptor[inverse].reshape((*queries.shape, -1))[:, 0]
    return dict(descriptor=desc, visible_local=visible[:, 0], visible_context=visible.any(1),
                ptm_local=ptm[:, 0], ptm_context=ptm.any(1)), unique, inverse.reshape(queries.shape), xyz


def source_task(config_path, sid):
    s = study(config_path)
    c, pop = s.config, s.pop
    item = next(p for p in s.plan['sources'] if p['id'] == sid)
    dest = s.technical / 'sources' / str(sid)
    dest.mkdir(parents=True, exist_ok=True)
    receipt = dest / 'complete.json'
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old['identity'] != s.identity or any(sha(dest / n) != h for n, h in old['files'].items()):
            raise ValueError(f'Changed spatial labels: {sid}')
        return old
    started = time.monotonic()
    manifest = json.loads((s.features / 'manifest.json').read_text())
    feature = s.features / f'{sid}.npz'
    if sha(feature) != manifest['shards'][str(sid)]['sha256']:
        raise ValueError(f'Changed encoder features: {sid}')
    with np.load(feature) as a:
        rows, query_ids = a['rows'], a['query_atom_ids']
    np.testing.assert_array_equal(rows, np.flatnonzero(pop['source'] == sid))
    audit = resolve_path(c['audit']) / 'technical/sources' / str(sid)
    graph_receipt = json.loads((audit / 'graph-complete.json').read_text())
    if sha(audit / 'graph.npz') != graph_receipt['sha256']:
        raise ValueError(f'Changed full-cell graph: {sid}')
    with np.load(audit / 'graph.npz') as a:
        graph = {k: a[k] for k in a.files}
    audit_config = json.loads(resolve_path(c['audit_config']).read_text())
    events, roots, _ = ancestry.establish(graph, audit_config['lineage']['thresholds'][0])
    raw = raw_source(item)
    frames = np.unique(pop['frame'][rows])
    scan_frames = set(frames)
    access = ancestry.GeometryAccess(audit_config, item, audit, graph)
    regular, path_records = [], []
    path_index = 0
    rng = np.random.default_rng(c['seed'] + sid)
    no_scan = []
    for frame in frames:
        frame = int(frame)
        points, box, dense = access.frame(frame)
        labels = access.labels[frame-access.chunk_start]
        tree = cKDTree(points, boxsize=box)
        nodes = np.flatnonzero((graph['frame'] == frame) & (graph['size'] >= 64))
        # Confirmation is past-only even though the audit stores backfilled roots.
        known = [n for n in nodes if any(events[r-1]['confirmation_frame'] <= frame for r in roots[n])]
        solid = np.isin(dense, known) if known else np.zeros(len(points), bool)
        crystal_rows = np.flatnonzero(solid)
        crystal_tree = cKDTree(points[solid], boxsize=box) if solid.any() else None
        selected = np.flatnonzero(pop['frame'][rows] == frame)
        queries = np.searchsorted(raw.atom_ids, query_ids[selected])
        np.testing.assert_array_equal(raw.atom_ids[queries], query_ids[selected])
        distance = crystal_tree.query(points[queries[:, 0]], workers=1)[0] if crystal_tree else np.full(len(selected), np.inf)
        observed, _, _, _ = observations(points, box, tree, queries, solid, labels)
        regular.append(dict(rows=rows[selected], distance=distance.astype(np.float32), **observed))
        if item['role'] not in ('calibration', 'test') or frame not in scan_frames:
            continue
        if crystal_tree is None:
            no_scan.append(dict(frame=frame, reason='no confirmed crystal'))
            continue
        center_rows = np.searchsorted(raw.atom_ids, item['center_atom_ids'])
        dd, nn = crystal_tree.query(points[center_rows], workers=1)
        candidates = np.flatnonzero((dd >= c['paths']['minimum_start_A']) &
                                   (dd <= c['paths']['maximum_start_A']) & ~solid[center_rows])
        if not len(candidates):
            no_scan.append(dict(frame=frame, reason='no fixed center in declared start-distance range'))
            continue
        for candidate in rng.permutation(candidates)[:c['paths']['pairs_per_frame']]:
            start = points[center_rows[candidate]]
            delta = points[crystal_rows[nn[candidate]]] - start
            delta -= box * np.rint(delta / box)
            length = float(np.linalg.norm(delta))
            direction = delta / length
            offsets = np.r_[np.arange(0, length, c['paths']['step_A']), length]
            for kind, sign in (('toward', 1.), ('away', -1.)):
                waypoints = np.mod(start + sign*offsets[:, None]*direction, box)
                atoms = tree.query(waypoints, workers=1)[1]
                keep = np.r_[True, atoms[1:] != atoms[:-1]]
                atoms, nominal_travel = atoms[keep], offsets[keep]
                distances = crystal_tree.query(points[atoms], workers=1)[0]
                if kind == 'away' and np.any(distances <= c['paths']['control_clearance_A']):
                    # This is a declared no-approach path control, not a random-path population.
                    no_scan.append(dict(frame=frame, reason='away path approaches another interface'))
                    continue
                query = np.stack([representatives(points, int(a), tree, box, stencil())[0] for a in atoms])
                actual = points[query]-points[atoms, None]
                actual -= box*np.rint(actual/box)
                observed, unique, inverse, xyz = observations(points, box, tree, query, solid, labels)
                name = f'path-{path_index:03d}.npz'
                np.savez_compressed(dest/name, positions=xyz.astype(np.float32), inverse=inverse,
                                    actual=actual.astype(np.float32), atom=raw.atom_ids[atoms],
                                    distance=distances.astype(np.float32), travel_A=nominal_travel.astype(np.float32),
                                    step=np.arange(len(atoms)), **observed)
                path_records.append(dict(source=sid, role=item['role'], frame=frame, path_id=path_index,
                                         kind=kind, file=name, sha256=sha(dest/name), rows=len(atoms)))
                path_index += 1
    values = {k: np.concatenate([x[k] for x in regular]) for k in regular[0]}
    order = np.argsort(values['rows']); values = {k: v[order] for k, v in values.items()}
    np.testing.assert_array_equal(values['rows'], rows)
    np.savez_compressed(dest/'labels.npz', **values)
    write_json(dest/'paths.json', dict(paths=path_records, excluded=no_scan))
    files = {p.name: sha(p) for p in dest.glob('*.npz')}
    files['paths.json'] = sha(dest/'paths.json')
    record = dict(identity=s.identity, source=sid, rows=len(rows), paths=len(path_records),
                  seconds=time.monotonic()-started, files=files)
    write_json(receipt, record)
    return record


def prepare(config_path):
    s = study(config_path)
    policy = s.config['cache_policy']
    cache = RetainedCache(resolve_path(policy['features']), policy['encoders_kept'])
    metadata = {k: s.pointer[k] for k in ('identity', 'domain', 'encoder_sha256')}
    results = []
    with cache.lease(s.pointer['key'], deadline=time.time()+8*3600, metadata=metadata, shared=True):
        with ProcessPoolExecutor(max_workers=s.config['cpu_workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(source_task, str(Path(config_path).resolve()), p['id']) for p in s.plan['sources']]
            for f in as_completed(futures):
                result = f.result(); results.append(result)
                write_json(s.technical/'preparation-state.json', dict(state='running', completed=len(results), total=len(futures)))
                print(json.dumps(result | {'files': len(result['files'])}), flush=True)
    write_json(s.technical/'prepared.json', dict(identity=s.identity, sources=results, rows=len(s.pop['source']),
                                               paths=sum(r['paths'] for r in results)))
    write_json(s.technical/'preparation-state.json', dict(state='complete', completed=len(results), total=len(results)))
