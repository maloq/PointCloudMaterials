"""Present-only sampling, common physical paths and sustained-onset labels."""
from collections import Counter
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.research.crystallization_origin.extract import full_labels, frame_geometry
from src.research.crystallization_origin import ancestry
from src.research.liquid_predictability.descriptors import patch_descriptors
from .common import read, folder, plan, load_raw, exact_frames

STRATA = ('clear_liquid', 'visible_interface', 'crystalline_center')
CAUSES = ('parent_present_arrival', 'local_establishment', 'external_or_interface_formation', 'unresolved_or_in_progress')


def producer_identity():
    import importlib.metadata
    return dict(files={str(Path(f).name): sha(Path(f)) for f in
                      (__file__, patch_descriptors.__code__.co_filename, ancestry.__file__, full_labels.__code__.co_filename)},
                libraries={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'gudhi', 'ovito')})


def patches(points, box, centers):
    _, ids = cKDTree(points, boxsize=box).query(points[centers], k=80, workers=1)
    if not np.array_equal(ids[:, 0], centers):
        raise ValueError('Coincident atoms or wrong tracked center')
    delta = points[ids] - points[centers, None]
    delta -= box * np.rint(delta / box)
    return delta.astype(np.float32)


def descriptors(positions):
    bank, names = [], None
    for xyz in positions:
        values, actual = patch_descriptors(xyz)
        if names is not None and actual != names:
            raise ValueError('Descriptor schema changed between patches')
        names = actual
        bank.append(values)
    return np.stack(bank), names


def parent(c, index):
    p = plan(c)['parents'][index]
    root = folder(c) / 'parents' / f'{index:03d}'
    root.mkdir(parents=True, exist_ok=True)
    producer = producer_identity()
    identity = digest(dict(plan=plan(c)['identity'], parent=index, producer=producer))
    if (root / 'parent.json').exists():
        receipt = read(root / 'parent.json')
        if receipt['identity'] != identity or sha(root / 'parent.npz') != receipt['sha256']:
            raise ValueError(f'Changed parent observations: {index}')
        return receipt
    raw = load_raw(p['shots'][0])
    points, box = frame_geometry(raw, 0)
    labels = full_labels(points, box, c['ptm_rmsd_cutoff'])
    crystal = np.isin(labels, [1, 2, 3])
    distance = (cKDTree(points[crystal], boxsize=box).query(points, workers=1)[0]
                if crystal.any() else np.full(len(points), np.inf))
    stratum = np.where(crystal, 2, np.where(distance <= 8., 1, 0))
    rng = np.random.default_rng(np.random.SeedSequence([c['seed'], index]))
    centers, weights = [], []
    counts = np.bincount(stratum, minlength=3)
    for s in range(3):
        pool = np.flatnonzero(stratum == s)
        n = min(len(pool), c['centers_per_stratum'])
        if n:
            centers.extend(rng.choice(pool, n, replace=False).tolist())
            weights.extend([len(pool) / (len(points) * n)] * n)
    centers = np.asarray(centers, np.int64)
    xyz = patches(points, box, centers)
    bank, names = descriptors(xyz)
    np.savez_compressed(root / 'parent.npz', positions=xyz, descriptors=bank, centers=centers,
        atom_ids=raw.atom_ids[centers], weights=np.asarray(weights), strata=stratum[centers], labels=labels)
    receipt = dict(identity=identity, parent_id=p['parent_id'], source=p['source'], role=p['role'],
        stratum_population=counts.tolist(), columns=names, n=len(centers), producer=producer, sha256=sha(root / 'parent.npz'))
    write_json(root / 'parent.json', receipt)
    return receipt


def event_labels(c, maps, graph, points_at, box_at, centers):
    """Reuse the established repository lineage semantics, at fixed 1.5 ps persistence."""
    persistence = 1 + round(c['persistence_ps'] / c['label_cadence_ps'])
    events, roots, uncertain = ancestry.establish(graph, dict(size=c['establishment_atoms'], persistence_frames=persistence))
    by_id = {e['event_id']: e for e in events}
    for event in events:
        f = event['birth_frame']
        points, box, dense = points_at[f], box_at[f], maps[f]
        member = dense == event['birth_node']
        crystal = points[member]
        displacement = crystal - crystal[0]
        displacement -= box * np.rint(displacement / box)
        center = np.mod(crystal[0] + displacement.mean(0), box)
        ambiguous_extent = bool(np.any(np.ptp(displacement, axis=0) > box / 2))
        offsets = points[centers] - center
        offsets -= box * np.rint(offsets / box)
        nodes = np.flatnonzero(graph['frame'] == f)
        existing = [n for n in nodes if any(r != event['event_id'] and by_id[r]['confirmation_frame'] <= f for r in roots[n])]
        old = np.isin(dense, existing)
        nearest = float(cKDTree(points[old], boxsize=box).query(crystal)[0].min()) if old.any() else None
        kind = ('left_censored_existing' if event['left_censored'] else
                'unresolved_establishment_merge' if event['establishment_merge_ambiguous'] else
                'unresolved_periodic_extent' if ambiguous_extent else
                'interface_associated' if nearest is not None and nearest <= 8 else 'isolated_establishment')
        event.update(kind=kind, center_distances_A=np.linalg.norm(offsets, axis=1).tolist())
    solid = graph['center_solid']
    horizon = c['horizons_ps'][-1]
    outcome = np.full(len(centers), -1, np.int16)  # parent-solid centers are not at risk
    time_ps = np.full(len(centers), np.nan)
    causes = np.full(len(centers), -1, np.int8)
    for i in range(len(centers)):
        if solid[i, 0]:
            continue
        outcome[i] = 0  # observed event-free through 12 ps, with confirmation buffer
        for f in range(1, round(horizon / c['label_cadence_ps']) + 1):
            if not solid[i, f:f + persistence].all():
                continue
            rr, reason = ancestry.onset_roots(graph, roots, uncertain, f, i, persistence)
            label = ancestry.classify(0, f, i, rr, reason, events, 8.)
            # No pre-parent PTM history is asserted. This label means arrival
            # from a cluster already present at t=0 and subsequently confirmed
            # persistent, not proof of pre-parent establishment.
            if reason is None and rr and all(by_id[r]['left_censored'] for r in rr):
                label = 'existing_crystal_arrival'
            cause = (0 if label == 'existing_crystal_arrival' else
                     1 if label == 'local_nucleus_establishment' else
                     2 if label in ('external_new_crystal_arrival', 'interface_associated_new_cluster') else 3)
            t = f * c['label_cadence_ps']
            bin_id = int(np.searchsorted(c['horizons_ps'], t - 1e-9))
            outcome[i] = 1 + cause * len(c['horizons_ps']) + bin_id
            time_ps[i], causes[i] = t, cause
            break
    return outcome, time_ps, causes, events


def branch(c, parent_index, shot_index):
    p = plan(c)['parents'][parent_index]
    root = folder(c) / 'parents' / f'{parent_index:03d}'
    parent_record = parent(c, parent_index)
    path = root / f'shot-{shot_index:02d}.npz'
    receipt_path = path.with_suffix('.json')
    identity = digest(dict(parent=parent_record['identity'], shot=p['shots'][shot_index], producer=sha(Path(__file__))))
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt['identity'] != identity or sha(path) != receipt['sha256']:
            raise ValueError(f'Changed branch derivation: {path}')
        return receipt
    with np.load(root / 'parent.npz') as a:
        centers, parent_xyz, labels0 = a['centers'], a['positions'], a['labels']
    raw = load_raw(p['shots'][shot_index])
    times = np.arange(round(c['followup_ps'] / c['label_cadence_ps']) + 1) * c['label_cadence_ps']
    indices = exact_frames(raw, plan(c)['protocol']['timestep_fs'], times)
    points, box = frame_geometry(raw, 0)
    if not np.array_equal(patches(points, box, centers), parent_xyz):
        raise ValueError(f'Sibling time-zero observations differ: {parent_index}/{shot_index}')
    target_columns = [parent_record['columns'].index(name) for name in c['path_descriptors']]
    future = np.empty((len(centers), len(c['horizons_ps']), len(target_columns)), np.float32)
    fraction = np.empty((len(centers), len(c['horizons_ps'])), np.float32)
    # Full geometry is temporarily retained for event localization. 51 x 70k
    # frames fit on a CPU worker; only compact targets and lineage facts persist.
    maps, all_points, boxes, previous, edges = [], [], [], [], []
    frames, sizes, starts, center_nodes, solid_history = [-1], [0], [], [], []
    settings = c['lineage']
    began = time.monotonic()
    for f, frame_index in enumerate(indices):
        points, box = frame_geometry(raw, int(frame_index))
        labels = labels0 if f == 0 else full_labels(points, box, c['ptm_rmsd_cutoff'])
        local, local_sizes = ancestry.components(points, box, labels, settings)
        offset = len(sizes) - 1
        starts.append(offset)
        immediate = set()
        for lag, old in enumerate(reversed(previous), 1):
            pa, ch, count, strong = ancestry.overlap(old, local, sizes, local_sizes, settings)
            if lag == 1:
                immediate = set(ch[strong].tolist())
            else:
                retain = np.array([int(v) not in immediate for v in ch], bool)
                pa, ch, count, strong = (v[retain] for v in (pa, ch, count, strong))
            edges.extend(zip(pa.tolist(), (ch + offset).tolist(), count.tolist(), strong.astype(int).tolist(), [lag] * len(pa)))
        nodes = np.where(local > 0, local + offset, 0).astype(np.int32)
        maps.append(nodes); all_points.append(points); boxes.append(box)
        center_nodes.append(nodes[centers]); solid_history.append(np.isin(labels[centers], [1, 2, 3]))
        frames.extend([f] * (len(local_sizes) - 1)); sizes.extend(local_sizes[1:].tolist())
        previous.append(nodes); previous = previous[-(settings['maximum_missing_frames'] + 1):]
        match = np.flatnonzero(np.isclose(times[f], c['horizons_ps'], rtol=0, atol=1e-8))
        if len(match):
            h = int(match[0])
            xyz = patches(points, box, centers)
            bank, names = descriptors(xyz)
            if names != parent_record['columns']:
                raise ValueError('Future descriptor schema differs from observation')
            future[:, h] = bank[:, target_columns]
            _, neighbors = cKDTree(points, boxsize=box).query(points[centers], k=80)
            present = np.linalg.norm(xyz, axis=-1) < 8
            fraction[:, h] = (np.isin(labels[neighbors], [1, 2, 3]) * present).sum(1) / present.sum(1)
        if f % 10 == 0:
            write_json(root / f'shot-{shot_index:02d}-progress.json', dict(frame=f, total=len(indices), seconds=time.monotonic() - began))
    graph = dict(frame=np.array(frames), size=np.array(sizes), start=np.array(starts),
                 edges=np.asarray(edges, np.int64).reshape(-1, 5), center_node=np.stack(center_nodes).T,
                 center_solid=np.stack(solid_history).T)
    category, onset_time, cause, events = event_labels(c, maps, graph, all_points, boxes, centers)
    np.savez_compressed(path, future=future, crystalline_fraction=fraction, event=category, onset_ps=onset_time, cause=cause)
    receipt = dict(identity=identity, sha256=sha(path), centers=len(centers), frames=len(indices),
        timeline_ps=times.tolist(), categories=dict(Counter(map(str, category))), events=events,
        producer_sha256=sha(Path(__file__)), seconds=time.monotonic() - began)
    write_json(receipt_path, receipt)
    return receipt


def prepare(c, index, shots=None):
    parent(c, index)
    for j in range(12) if shots is None else shots:
        branch(c, index, j)


def seal(c):
    p = plan(c)
    banks = {k: [] for k in ('positions', 'descriptors', 'weights', 'strata', 'atom_ids', 'parent', 'future', 'event', 'onset_ps', 'cause', 'crystalline_fraction')}
    receipts = []
    for i, item in enumerate(p['parents']):
        root = folder(c) / 'parents' / f'{i:03d}'
        record = read(root / 'parent.json')
        if sha(root / 'parent.npz') != record['sha256']:
            raise ValueError('Parent shard checksum changed')
        with np.load(root / 'parent.npz') as a:
            for k in ('positions', 'descriptors', 'weights', 'strata', 'atom_ids'):
                banks[k].append(a[k])
            banks['parent'].append(np.full(len(a['centers']), i, np.int16))
        futures = {k: [] for k in ('future', 'event', 'onset_ps', 'cause', 'crystalline_fraction')}
        branch_receipts = []
        for j in range(12):
            path = root / f'shot-{j:02d}.npz'
            receipt = read(path.with_suffix('.json'))
            if sha(path) != receipt['sha256']:
                raise ValueError(f'Branch checksum changed: {path}')
            with np.load(path) as a:
                for k in futures:
                    futures[k].append(a[k])
            branch_receipts.append(receipt['sha256'])
        for k in futures:
            banks[k].append(np.stack(futures[k], axis=1))
        receipts.append(dict(parent=record, shots=branch_receipts))
    files = {}
    for k, parts in banks.items():
        path = folder(c) / f'{k}.npy'
        np.save(path, np.concatenate(parts))
        files[path.name] = sha(path)
    manifest = dict(identity=digest(dict(plan=p['identity'], receipts=receipts)), plan_identity=p['identity'],
                    parents=receipts, files=files, columns=receipts[0]['parent']['columns'],
                    path_descriptors=c['path_descriptors'], causes=CAUSES, strata=STRATA)
    write_json(folder(c) / 'manifest.json', manifest)
    return dict(identity=manifest['identity'], observations=sum(len(x) for x in banks['parent']), branches=480)


def load(c):
    root = folder(c)
    m = read(root / 'manifest.json')
    for name, expected in m['files'].items():
        if sha(root / name) != expected:
            raise ValueError(f'Changed sealed future-law data: {name}')
    return {name.removesuffix('.npy'): np.load(root / name) for name in m['files']}, m
