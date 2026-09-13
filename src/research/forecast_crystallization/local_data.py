"""Label tracked forecast centers with PTM, independently of learned embeddings."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from src.data_utils.shooting_binary import ShootingBinaryTrajectory
from src.experiment_runner.artifacts import result_folders
from src.experiment_runner.registry import sha256, write_json
from src.research.smooth_temporal_encoder.prepare import ptm_labels
from src.training_methods.embedding_forecast.data import verify_cache


def label_source(config, record, source):
    os.environ['OVITO_THREAD_COUNT'] = '1'
    started = time.monotonic()
    directory = Path(config['output']) / 'technical' / 'labels' / record['directory']
    directory.mkdir(parents=True)
    trajectory = ShootingBinaryTrajectory.load(source['path'])
    cache = Path(config['cache']) / record['directory']
    frames = np.load(cache / 'frames.npy')
    np.testing.assert_array_equal(trajectory.timesteps[frames], np.load(cache / 'timesteps.npy'))
    rng = np.random.default_rng(np.random.SeedSequence([config['seed'], record['preparation_seed']]))
    selected = np.sort(rng.choice(record['centers'], config['centers_per_source'], replace=False))
    atom_ids = np.load(cache / 'atom_ids.npy')[selected]
    rows = np.searchsorted(trajectory.atom_ids, atom_ids)
    np.testing.assert_array_equal(trajectory.atom_ids[rows], atom_ids)
    labels = np.empty((len(rows), len(frames)), dtype=np.uint8)
    for column, frame in enumerate(frames):
        low = trajectory.box_low[frame].astype(np.float64)
        lengths = trajectory.box_high[frame].astype(np.float64) - low
        points = np.mod(trajectory.positions[frame].astype(np.float64) - low, lengths)
        tree = cKDTree(points, boxsize=lengths)
        _, neighbors = tree.query(points[rows], k=80, workers=1)
        np.testing.assert_array_equal(neighbors[:, 0], rows)
        offsets = points[neighbors[:, 1:]] - points[rows, None]
        offsets -= lengths * np.round(offsets / lengths)
        # PTM is scale invariant. Radius 10 A separates clouds in the existing assay.
        labels[:, column] = ptm_labels(offsets / 10., config['ptm_rmsd_cutoff'])
    np.save(directory / 'labels.npy', labels)
    np.save(directory / 'embedding_rows.npy', selected)
    np.save(directory / 'atom_ids.npy', atom_ids)
    metadata = dict(**record, selected_centers=len(rows),
                    trajectory_manifest_sha256=sha256(Path(source['path']) / 'manifest.json'),
                    labels_sha256=sha256(directory / 'labels.npy'),
                    embedding_rows_sha256=sha256(directory / 'embedding_rows.npy'),
                    atom_ids_sha256=sha256(directory / 'atom_ids.npy'),
                    elapsed_seconds=time.monotonic() - started)
    write_json(directory / 'complete.json', metadata)
    return metadata


def prepare(config):
    root = result_folders(config['output']) / 'technical'
    if (root / 'local_observations.json').exists():
        raise FileExistsError(f'Local labels already complete: {root}')
    manifest = verify_cache(Path(config['cache']))
    sources = json.loads(Path(config['sources_config']).read_text())['sources']
    write_json(root / 'config.json', config)
    records = []
    with ProcessPoolExecutor(max_workers=config['label_workers'],
                             mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = []
        for record in manifest['shards']:
            source = sources[record['source_index']]
            if source['name'] != record['name'] or source['preparation_seed'] != record['preparation_seed']:
                raise ValueError(f'Source identity mismatch: {record["name"]}')
            futures.append(executor.submit(label_source, config, record, source))
        for future in as_completed(futures):
            records.append(future.result())
            print(f'Local PTM {len(records)}/{len(futures)} complete: {records[-1]["name"]}; '
                  f'{records[-1]["elapsed_seconds"]:.1f} s', flush=True)
            write_json(root / 'labels_progress.json', dict(completed=len(records), total=len(futures)))
    records.sort(key=lambda row: row['source_index'])
    write_json(root / 'local_observations.json', dict(sources=records,
        cache_manifest_sha256=sha256(Path(config['cache']) / 'manifest.json'),
        cadence_ps=manifest['cadence_ps'], frames=records[0]['frames']))


def verify_patches(config):
    """Compare saved selected-center labels to full periodic PTM on declared frames."""
    os.environ['OVITO_THREAD_COUNT'] = '1'
    from ovito.data import DataCollection, Particles, SimulationCell
    from ovito.modifiers import PolyhedralTemplateMatchingModifier
    root = Path(config['output']) / 'technical'
    observations = json.loads((root / 'local_observations.json').read_text())
    sources = json.loads(Path(config['sources_config']).read_text())['sources']
    records = {s['source_index']: s for s in observations['sources']}
    results = []
    for index in config['verification_sources']:
        record = records[index]
        directory = root / 'labels' / record['directory']
        trajectory = ShootingBinaryTrajectory.load(sources[index]['path'])
        ids = np.load(directory / 'atom_ids.npy')
        rows = np.searchsorted(trajectory.atom_ids, ids)
        labels = np.load(directory / 'labels.npy')
        for frame in config['verification_frames']:
            low = trajectory.box_low[frame].astype(np.float64)
            lengths = trajectory.box_high[frame].astype(np.float64)-low
            data = DataCollection(); particles = Particles(count=len(trajectory.atom_ids))
            particles.create_property('Position', data=np.mod(trajectory.positions[frame].astype(np.float64)-low, lengths))
            data.objects.append(particles)
            cell = SimulationCell(pbc=(True, True, True))
            cell[...] = np.column_stack((np.diag(lengths), np.zeros(3)))
            data.objects.append(cell)
            data.apply(PolyhedralTemplateMatchingModifier(rmsd_cutoff=config['ptm_rmsd_cutoff']))
            actual = np.asarray(data.particles['Structure Type'])[rows]
            np.testing.assert_array_equal(actual, labels[:, frame], err_msg=f'Patch/full PTM mismatch: source {index}, frame {frame}')
            results.append(dict(source_index=index, frame=frame, atom_ids=ids.tolist(), centers=len(ids), identical=True))
    write_json(root / 'selected_center_ptm_verification.json', results)
    print(f'Full-periodic PTM matched {sum(r["centers"] for r in results)} selected-center labels.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--verify-patches', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    (verify_patches if args.verify_patches else prepare)(config)


if __name__ == '__main__':
    main()
