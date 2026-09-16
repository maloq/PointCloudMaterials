"""Recover the original center IDs and three VICReg views from their producers."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.experiment_runner.registry import sha256, write_json
from src.training_methods.embedding_forecast.data import source_records


def read_json(path):
    return json.loads(Path(path).read_text())


def frame_tree(trajectory, frame):
    low = trajectory.box_low[frame].astype(np.float64)
    lengths = trajectory.box_high[frame].astype(np.float64)-low
    points = np.mod(trajectory.positions[frame].astype(np.float64)-low, lengths)
    return points, lengths, cKDTree(points, boxsize=lengths)


def halos(frame, centers, radius):
    points, lengths, tree = frame
    neighbors = tree.query_ball_point(points[centers], radius, workers=1)
    nearest = tree.query(points[centers], k=80, workers=1)[1]
    clouds = []
    for center, ids, original in zip(centers, neighbors, nearest, strict=True):
        # Match the producer's cKDTree k=80 tie ordering exactly. Absolute
        # float16 trajectory positions can have exactly equal squared radii.
        remaining = np.setdiff1d(ids, original)
        ids = np.r_[original, remaining]
        x = points[ids]-points[center]
        x -= lengths*np.round(x/lengths)
        if ids[0] != center:
            raise ValueError(f'Tracked atom {center} is not the first halo node')
        clouds.append(x.astype(np.float32))
    return clouds


def save_clouds(path, clouds):
    np.savez(path, positions=np.concatenate(clouds), pointers=np.cumsum([0]+[len(x) for x in clouds]))


def load_clouds(path):
    values = np.load(path)
    x, ptr = values['positions'], values['pointers']
    return [x[a:b] for a, b in zip(ptr[:-1], ptr[1:], strict=True)]


def prepare(config):
    root = Path(config['cache'])
    root.mkdir(parents=True, exist_ok=True)
    if (root/'manifest.json').exists():
        raise FileExistsError(f'Preserve prepared context data: {root}')
    manifest = read_json(Path(config['training_cache'])/'manifest.json')
    parent = Path(manifest['protocol']['source_manifest'])
    if sha256(parent) != manifest['protocol']['source_sha256']:
        raise ValueError('Original target manifest has changed')
    sources = source_records(dict(sources_config=config['sources_config'], cadence_ps=.75))
    probes = np.load(Path(config['diagnostics'])/'technical/probes.npz')
    temporal = np.load(Path(config['diagnostics'])/'technical/temporal.npz')
    trajectory = None
    counts, records = [], []
    for context, record in enumerate(manifest['shards']):
        source = sources[record['source']]
        if trajectory is None or trajectory.root != Path(source['path']):
            trajectory = ShootingBinaryTrajectory.load(source['path'])
        ids = np.flatnonzero(probes['context'] == context)
        origin = Path(record['source_directory'])
        original_ids = np.load(origin/'neighbor_ids.npy')[:, 0]
        lookup = {int(atom): i for i, atom in enumerate(original_ids)}
        rows = np.array([lookup[int(atom)] for atom in probes['atom_id'][ids]])
        all_centers = np.load(origin/'centers.npy')
        centers = all_centers[rows]
        np.testing.assert_array_equal(trajectory.atom_ids[centers], probes['atom_id'][ids])
        if not np.all(probes['split'][ids] == record['split']):
            raise ValueError(f'Split mismatch at context {context}')
        frame = frame_tree(trajectory, record['frame'])
        # Replay the actual history producer's RNG before subselecting rows.
        rng = np.random.default_rng(np.random.SeedSequence([manifest['protocol']['seed'], context]))
        k = manifest['protocol']['neighbor_k']
        candidates = frame[2].query(frame[0][all_centers], k=k+1, workers=1)[1][:, 1:]
        spatial = candidates[np.arange(len(all_centers)), rng.integers(0, k, len(all_centers))][rows]
        views = [halos(frame, centers, config['candidate_radius_A']),
                 halos(frame, spatial, config['candidate_radius_A']),
                 halos(frame_tree(trajectory, record['frame']-1), centers, config['candidate_radius_A'])]
        legacy_path = Path(config['training_cache'])/record['views']
        if sha256(legacy_path) != manifest['checksums'][record['views']]:
            raise ValueError(f'Original history cache changed: {legacy_path}')
        legacy = np.load(legacy_path, mmap_mode='r')[rows, :, -1]
        # Float32 offsets add no second float16 storage round trip.
        for view in range(3):
            np.testing.assert_array_equal(np.stack([x[:80] for x in views[view]]).astype(np.float16), legacy[:, view])
        clouds = [views[v][row] for row in range(len(rows)) for v in range(3)]
        filename = f'context-{context:03d}.npz'
        save_clouds(root/filename, clouds)
        counts.extend(len(x) for x in clouds)
        records.append(dict(file=filename, rows=ids.tolist(), source=record['source'],
                            split=record['split'], frame=record['frame'], context=context,
                            original_rows=rows.tolist(), sha256=sha256(root/filename)))
        write_json(root/'status.json', dict(state='preparing', contexts=context+1, total=90))
        print(f'CONTEXT {context+1}/90 nodes={np.mean([len(x) for x in clouds]):.1f}', flush=True)
    temporal_records = []
    for source_index in np.unique(temporal['source']):
        source = sources[int(source_index)]
        trajectory = ShootingBinaryTrajectory.load(source['path'])
        for context, anchor in enumerate(source['anchors']):
            ids = np.flatnonzero((temporal['source'] == source_index) & (temporal['context'] == context))
            centers = np.searchsorted(trajectory.atom_ids, temporal['atom_id'][ids])
            np.testing.assert_array_equal(trajectory.atom_ids[centers], temporal['atom_id'][ids])
            clouds, flattened = [], []
            for column, frame_index in enumerate(range(anchor-8, anchor+9)):
                clouds.extend(halos(frame_tree(trajectory, frame_index), centers, config['candidate_radius_A']))
                flattened.extend((ids*17+column).tolist())
            filename = f'temporal-{source_index}-{context}.npz'
            save_clouds(root/filename, clouds)
            temporal_records.append(dict(file=filename, rows=flattened, sha256=sha256(root/filename)))
            print(f'TEMPORAL source={source_index} context={context}', flush=True)
    write_json(root/'manifest.json', dict(state='complete', records=records, temporal=temporal_records,
        candidate_radius_A=config['candidate_radius_A'], config=config,
        provenance={str(path):sha256(path) for path in [parent, Path(config['training_cache'])/'manifest.json',
            Path(config['diagnostics'])/'technical/probes.npz', Path(config['diagnostics'])/'technical/temporal.npz']},
        node_count_quantiles=np.quantile(counts, [0, .5, .95, 1]).tolist()))
    write_json(root/'status.json', dict(state='complete', contexts=len(records), examples=len(probes['z'])))


def training_clouds(config):
    root = Path(config['cache'])
    manifest = read_json(root/'manifest.json')
    values = [None]*sum(len(r['rows']) for r in manifest['records'])
    for record in manifest['records']:
        path = root/record['file']
        if sha256(path) != record['sha256']:
            raise ValueError(f'Context cache changed: {path}')
        clouds = load_clouds(path)
        for i, row in enumerate(record['rows']):
            values[row] = clouds[i*3:i*3+3]
    return values
