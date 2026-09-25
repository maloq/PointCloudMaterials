"""Freeze and build the Al64 benchmark and its label-free structural corpus on CPU."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.data.trajectories.lammps import TemporalLAMMPSBinaryTrajectory
from src.project_runtime.paths import dataset_path, resolve_path, portable_config
from .protocol import ROLES, audit_sources, centered, digest, onset_rows, sha, write_json


def implementation():
    base = Path(__file__).resolve().parents[3]
    files = list(Path(__file__).parent.glob('*.py')) + [base / p for p in (
        'src/research/smooth_temporal_encoder/prepare.py',
        'src/research/forecast_crystallization/local_metrics.py',
        'src/data/trajectories/shooting.py', 'src/data/trajectories/lammps.py')]
    return {str(p.relative_to(base)): sha(p) for p in files}


def freeze(config, root):
    path = root / 'plan.json'
    if path.exists():
        plan = json.loads(path.read_text())
        if plan['config'] != config or plan['implementation'] != implementation():
            raise ValueError('Frozen config/producer changed; use a new release directory')
        if sha(resolve_path(config['parent_plan'])) != plan['parent_sha256']:
            raise ValueError('Parent source contract changed')
        return plan
    if config['protocol'] != 'fixed_al64_v1' or config['external_inputs'] != []:
        raise ValueError('Require fixed Al64 protocol without condition covariates')
    parent_path = resolve_path(config['parent_plan'])
    parent = json.loads(parent_path.read_text())
    cells_root = resolve_path(parent['config']['cache']) / 'cells'
    legacy_root = resolve_path(config['legacy_labels'])
    label_root = resolve_path(config['label_cache'])
    dataset_roots = {s['dataset']: dataset_path(s['dataset']) for s in parent['sources']}
    archive_roots = {}
    portable_roots = {}
    sources = []
    for old in sorted(parent['sources'], key=lambda s: s['id']):
        role = old.get('validation_role', old['split'])
        source = {k: old[k] for k in ('id', 'dataset', 'relative_trajectory_path',
                  'manifest_sha256', 'lineage', 'frame_count', 'atom_count', 'timestep_fs')}
        source.update(role=role, center_atom_ids=old['pool_atom_ids'],
                      legacy_center_atom_ids=old['center_atom_ids'],
                      legacy_shard_sha256=old['shard_sha256'],
                      audit_metadata={'temperature_K': old['temperature_K']})
        raw = dataset_roots[old['dataset']] / old['relative_trajectory_path']
        if sha(raw / 'manifest.json') != old['manifest_sha256']:
            raise ValueError(f'Raw source manifest changed: {old["id"]}')
        if sha(label_root / old['shard']) != old['shard_sha256']:
            raise ValueError(f'16-center label shard changed: {old["id"]}')
        if old['legacy'] is not None:
            previous = legacy_root / old['legacy']['directory']
            if sha(previous / 'labels.npy') != old['legacy']['labels_sha256']:
                raise ValueError(f'64-center label hash changed: {old["id"]}')
            np.testing.assert_array_equal(np.load(previous / 'atom_ids.npy'), old['pool_atom_ids'])
            source['legacy_labels'] = dict(directory=old['legacy']['directory'],
                labels_sha256=sha(previous / 'labels.npy'), ids_sha256=sha(previous / 'atom_ids.npy'))
        else:
            source['legacy_labels'] = None
        cells = []
        for frame in parent['config']['frames_by_role'][role]:
            receipt = cells_root / f'{old["id"]}-{frame}' / 'complete.json'
            record = json.loads(receipt.read_text())  # Missing cell is an error, never a model-specific skip.
            if record['identity'] != parent['identity']:
                raise ValueError(f'Quench belongs to another protocol: {receipt}')
            saved = Path(record['archive'])
            if str(saved.parent) not in archive_roots:
                archive_roots[str(saved.parent)] = resolve_path(saved.parent)
                portable_roots[str(saved.parent)] = portable_config(str(archive_roots[str(saved.parent)]))
            archive = archive_roots[str(saved.parent)] / saved.name
            meta = json.loads((archive / 'metadata.json').read_text())
            if (meta['source_manifest_sha256'] != old['manifest_sha256'] or
                meta['source_frame'] != frame or meta['fmax_eV_per_A'] > .01 or
                set(meta['potential_checksums'].values()) != set(parent['config']['potential_sha256'])):
                raise ValueError(f'Quench ancestry/potential/convergence changed: {archive}')
            cells.append(dict(frame=frame, archive=portable_roots[str(saved.parent)] + '/' + saved.name,
                metadata_sha256=sha(archive / 'metadata.json'),
                manifest_sha256=sha(archive / 'relaxed_binary_float16/manifest.json')))
        source['cells'] = cells
        sources.append(source)
    audit_sources(sources, config['expected_source_counts'], config['centers_per_source'])
    plan = dict(protocol=config['protocol'], config=config, sources=sources,
        parent_sha256=sha(parent_path), parent_identity=parent['identity'],
        legacy_population_sha256=sha(resolve_path(config['legacy_population'])),
        potential_sha256=parent['config']['potential_sha256'], implementation=implementation(),
        test_status='Historical test sources reused; not a new untouched test',
        inputs=dict(positions='Current center-relative nearest80 in Angstrom, radius8 mask',
                    motion='Optional current center-relative velocities, Angstrom/ps',
                    relaxation='Matched full-cell fixed-box FIRE; observed nearest80 IDs retained',
                    context='Full source and frame references retained; declare context separately per model',
                    history='None in exported patches; raw causal history remains addressable by sample ID',
                    conditions=[], training_only_teachers=[]),
        structural=dict(roles=config['structural_roles'], center_sampling='Inherited outcome-independent 64-atom pool',
            frames=list(range(0, 801, config['structural_stride_frames'])),
            labels='No crystallization labels, no filtering by outcome; geometry only',
            excluded_ancestry=[s['lineage'] for s in sources if s['role'] not in config['structural_roles']]))
    plan['identity'] = digest(plan)
    write_json(path, plan)
    write_json(root / 'splits.json', dict(identity=plan['identity'], sources=[
        {k: s[k] for k in ('id', 'role', 'lineage', 'manifest_sha256', 'center_atom_ids',
                           'legacy_center_atom_ids')} for s in sources]))
    return plan


def verify_files(directory, record, identity):
    if record['identity'] != identity:
        raise ValueError(f'Shard belongs to another fixed release: {directory}')
    for name, checksum in record['files'].items():
        if sha(directory / name) != checksum:
            raise ValueError(f'Shard checksum changed: {directory / name}')


def save_arrays(directory, arrays):
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        np.save(directory / f'{name}.npy', value)
    return {f'{name}.npy': sha(directory / f'{name}.npy') for name in arrays}


def prepare_source(task):
    plan, source, root_value, locations = task
    os.environ['OVITO_THREAD_COUNT'] = '1'
    root = Path(root_value)
    config = plan['config']; sid = source['id']; started = time.monotonic()
    folder = root / 'benchmark' / 'sources' / str(sid)
    receipt = folder / 'complete.json'
    structural_dir = root / 'structural' / 'sources' / str(sid)
    if receipt.exists():
        record = json.loads(receipt.read_text())
        verify_files(folder, record, plan['identity'])
        if source['role'] in config['structural_roles']:
            verify_files(structural_dir, record['structural'], plan['identity'])
        return record
    raw = ShootingBinaryTrajectory.load(Path(locations['datasets'][source['dataset']]) /
                                        source['relative_trajectory_path'])
    if sha(raw.root / 'manifest.json') != source['manifest_sha256']:
        raise ValueError(f'Raw source changed after freeze: {sid}')
    raw.verify_checksums()
    if not np.all(raw.atom_types == 1) or raw.manifest['velocity_units'] != 'angstrom_per_ps':
        raise ValueError(f'Expected Al/type1 and Angstrom/ps: {sid}')
    np.testing.assert_array_equal(raw.timesteps * source['timestep_fs'] / 1000,
                                  np.arange(801) * config['cadence_ps'])
    atoms = np.array(source['center_atom_ids'], dtype=np.int64)
    rows = np.searchsorted(raw.atom_ids, atoms)
    np.testing.assert_array_equal(raw.atom_ids[rows], atoms)
    old_path = Path(locations['labels']) / f'source-{sid:04d}.npz'
    if sha(old_path) != source['legacy_shard_sha256']:
        raise ValueError(f'Legacy labels changed: {sid}')
    with np.load(old_path) as old:
        old_atoms, old_labels = old['atom_ids'], old['labels']
    legacy = source['legacy_labels']
    if legacy is not None:
        directory = Path(locations['legacy']) / legacy['directory']
        if (sha(directory / 'labels.npy') != legacy['labels_sha256'] or
            sha(directory / 'atom_ids.npy') != legacy['ids_sha256']):
            raise ValueError(f'64-center labels changed after freeze: {sid}')
        labels = np.load(directory / 'labels.npy')
    else:
        from src.research.smooth_temporal_encoder.prepare import ptm_labels
        labels = np.empty((len(atoms), 801), dtype=np.uint8)
    if labels.shape != (64, 801):
        raise ValueError(f'Unexpected label shape for {sid}: {labels.shape}')
    cells = {c['frame']: c for c in source['cells']}
    paired_frames = sorted(cells)
    structural_frames = plan['structural']['frames'] if source['role'] in config['structural_roles'] else []
    frames = sorted(set(paired_frames + structural_frames + ([] if legacy else list(range(801)))))
    paired = {k: [] for k in ('hot', 'cold', 'velocity', 'neighbor_ids')}
    structural = {k: [] for k in ('positions', 'neighbor_ids')}
    # Full periodic PTM cross-check once per role, in addition to every legacy center/frame.
    verify = sid == min(s['id'] for s in plan['sources'] if s['role'] == source['role'])
    if verify:
        frames = sorted(set(frames) | {80, 640})
    verification = []
    for frame in frames:
        box = raw.box_high[frame].astype(float) - raw.box_low[frame].astype(float)
        points = np.mod(raw.positions[frame].astype(float), box)
        tree = cKDTree(points, boxsize=box)
        neighbors = tree.query(points[rows], k=config['candidate_atoms'], workers=1)[1]
        np.testing.assert_array_equal(neighbors[:, 0], rows)
        hot = centered(points, box, rows, neighbors)
        if legacy is None:
            # Match the established producer's float64 local chart exactly.
            offsets = points[neighbors[:, 1:]] - points[rows, None]
            offsets -= box * np.round(offsets / box)
            labels[:, frame] = ptm_labels(offsets / 10., .1)
        if verify and frame in (80, 640):
            from src.research.local_predictability.data import full_ptm
            np.testing.assert_array_equal(labels[:, frame], full_ptm(points, box, rows),
                                          err_msg=f'PTM full-cell mismatch {sid}/{frame}')
            verification.append(frame)
        if frame in structural_frames:
            structural['positions'].append(hot)
            structural['neighbor_ids'].append(raw.atom_ids[neighbors].astype(np.int32))
        if frame in cells:
            cell = cells[frame]
            archive = resolve_path(cell['archive'])
            if (sha(archive / 'metadata.json') != cell['metadata_sha256'] or
                sha(archive / 'relaxed_binary_float16/manifest.json') != cell['manifest_sha256']):
                raise ValueError(f'Relaxed cell changed: {sid}/{frame}')
            cold = TemporalLAMMPSBinaryTrajectory.load(archive / 'relaxed_binary_float16')
            cold.verify_checksums()
            np.testing.assert_array_equal(cold.atom_ids, raw.atom_ids)
            np.testing.assert_array_equal(cold.timesteps[0], raw.timesteps[frame])
            np.testing.assert_allclose(cold.box_high[0].astype(float) - cold.box_low[0].astype(float),
                                       box, rtol=0, atol=1e-5)
            relaxed = np.mod(cold.positions[0].astype(float), box)
            velocity = raw.velocities[frame].astype(np.float32)
            paired['hot'].append(hot)
            paired['cold'].append(centered(relaxed, box, rows, neighbors))
            paired['velocity'].append(velocity[neighbors] - velocity[rows, None])
            paired['neighbor_ids'].append(raw.atom_ids[neighbors].astype(np.int32))
    old_indices = np.searchsorted(atoms, old_atoms)
    np.testing.assert_array_equal(atoms[old_indices], old_atoms)
    np.testing.assert_array_equal(labels[old_indices], old_labels,
                                  err_msg=f'Historical 16-center label mismatch for {sid}')
    arrays = {k: np.stack(v) for k, v in paired.items()}
    arrays.update(frames=np.array(paired_frames, dtype=np.int32), atom_ids=atoms, labels=labels)
    for name in ('hot', 'cold', 'velocity'):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f'Nonfinite {name}: source {sid}')
    record = dict(source=sid, identity=plan['identity'], files=save_arrays(folder, arrays),
                  labels_reused=legacy is not None, legacy_labels_exact=True,
                  full_ptm_verification_frames=verification)
    if structural_frames:
        values = {k: np.stack(v) for k, v in structural.items()}
        values.update(frames=np.array(structural_frames, dtype=np.int32), atom_ids=atoms)
        record['structural'] = dict(identity=plan['identity'], source=sid, role=source['role'],
            rows=len(structural_frames) * len(atoms), files=save_arrays(structural_dir, values))
        write_json(structural_dir / 'complete.json', record['structural'])
    record['seconds'] = time.monotonic() - started
    write_json(receipt, record)
    return record


def finalize(plan, root, records):
    config = plan['config']
    original = resolve_path(config['legacy_population'])
    if sha(original) != plan['legacy_population_sha256']:
        raise ValueError('Historical population changed after freeze')
    with np.load(original) as saved:
        old = {k: saved[k] for k in ('source', 'frame', 'atom', 'role', 'event', 'delay')}
    old_rows = {(int(s), int(f), int(a)): i for i, (s, f, a) in enumerate(zip(
        old['source'], old['frame'], old['atom'], strict=True))}
    parts = []; coverage = {}; structural_sources = []
    for source in plan['sources']:
        sid = source['id']; folder = root / 'benchmark/sources' / str(sid)
        frames = np.load(folder / 'frames.npy')
        eligible = np.flatnonzero(frames >= config['minimum_onset_anchor'])
        row = onset_rows(np.load(folder / 'labels.npy'), frames[eligible], config)
        fi, ci = eligible[row.pop('frame_index')], row.pop('center_index')
        atoms = np.asarray(source['center_atom_ids'])[ci]
        n = len(ci)
        keys = [(sid, int(f), int(a)) for f, a in zip(frames[fi], atoms, strict=True)]
        row.update(source=np.full(n, sid, np.int32), frame=frames[fi], atom=atoms,
            role=np.full(n, source['role']), patch_index=fi * config['centers_per_source'] + ci,
            legacy_row=np.array([old_rows.get(k, -1) for k in keys], dtype=np.int64),
            sample_id=np.array([f'{source["manifest_sha256"]}:{f}:{a}' for _, f, a in keys]))
        parts.append(row)
        if source['role'] in config['structural_roles']:
            structural_sources.append(dict(id=sid, role=source['role'], lineage=source['lineage'],
                manifest_sha256=source['manifest_sha256'],
                rows=len(plan['structural']['frames']) * config['centers_per_source'],
                paired_rows=len(frames) * config['centers_per_source'],
                files=records[sid]['structural']['files']))
    population = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    matched = np.flatnonzero(population['legacy_row'] >= 0)
    matched = matched[np.argsort(population['legacy_row'][matched])]
    np.testing.assert_array_equal(population['legacy_row'][matched], np.arange(len(old['source'])))
    for key in old:
        np.testing.assert_array_equal(population[key][matched], old[key],
                                      err_msg=f'Historical comparison population differs: {key}')
    for role in ROLES:
        mask = population['role'] == role
        coverage[role] = dict(sources=sum(s['role'] == role for s in plan['sources']),
            centers=sum(s['role'] == role for s in plan['sources']) * config['centers_per_source'],
            windows=int(mask.sum()), legacy_windows=int(np.sum(mask & (population['legacy_row'] >= 0))),
            positive_windows={str(h): int(np.sum(mask & (population['delay'] <= h))) for h in (3., 6., 12.)},
            structural_rows=sum(s['rows'] for s in structural_sources if s['role'] == role),
            paired_structural_rows=sum(s['paired_rows'] for s in structural_sources if s['role'] == role))
    np.savez(root / 'benchmark/population.npz', **population)
    np.save(root / 'benchmark/legacy_order.npy', matched)
    benchmark = dict(state='complete', identity=plan['identity'], plan_sha256=sha(root / 'plan.json'),
        population_sha256=sha(root / 'benchmark/population.npz'),
        legacy_order_sha256=sha(root / 'benchmark/legacy_order.npy'),
        sources=[dict(id=s['id'], role=s['role'], files=records[s['id']]['files']) for s in plan['sources']],
        counts=coverage, horizons_ps=config['horizons_ps'], test_status=plan['test_status'], inputs=plan['inputs'])
    structural = dict(state='complete', identity=plan['identity'], plan_sha256=sha(root / 'plan.json'),
        sources=structural_sources, counts={r: coverage[r]['structural_rows'] for r in config['structural_roles']},
        inputs=['current relative positions', 'Al species', 'radius mask'], conditions=[],
        labels=[], normalization='Angstrom; no fitted population statistics',
        heldout_exclusion='Only explicit training sources may fit weights or normalization; selection is validation only. No calibration/test source or shooting descendants included.',
        paired_view='All candidate paired patches, without onset filtering; use the same role restriction.')
    write_json(root / 'benchmark/manifest.json', benchmark)
    write_json(root / 'structural/manifest.json', structural)
    write_json(root / 'manifest.json', dict(state='complete', identity=plan['identity'],
        counts=coverage, files={p: sha(root / p) for p in (
            'plan.json', 'splits.json', 'benchmark/manifest.json', 'structural/manifest.json')}))
    print(json.dumps(dict(stage='complete', identity=plan['identity'], counts=coverage)), flush=True)


def run(config_path, workers):
    config = json.loads(Path(config_path).read_text())
    root = resolve_path(config['cache']); root.mkdir(parents=True, exist_ok=True)
    with (root / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = freeze(config, root)
        if (root / 'manifest.json').exists():
            from .dataset import verify_release
            verify_release(root)
            print(json.dumps(dict(stage='already_complete', identity=plan['identity'])), flush=True)
            return
        locations = dict(datasets={name: str(dataset_path(name)) for name in {s['dataset'] for s in plan['sources']}},
                         labels=str(resolve_path(config['label_cache'])), legacy=str(resolve_path(config['legacy_labels'])))
        tasks = [(plan, s, str(root), locations) for s in plan['sources']]
        records = {}
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = {pool.submit(prepare_source, task): task[1]['id'] for task in tasks}
            try:
                for future in as_completed(futures):
                    result = future.result(); records[result['source']] = result
                    print(json.dumps(dict(stage='source_complete', source=result['source'],
                        completed=len(records), total=len(tasks), seconds=result['seconds'])), flush=True)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        finalize(plan, root, records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    run(args.config, args.workers)


if __name__ == '__main__':
    main()
