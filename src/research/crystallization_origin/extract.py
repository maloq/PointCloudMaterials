"""Resumable full-cell PTM, using the fixed benchmark's classifier and raw MD."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import time
import traceback

import numpy as np

from src.data.fixed_cohort.dataset import read_release
from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.experiment_runner.artifacts import result_folders
from src.project_runtime.paths import dataset_path, resolve_path


def raw_source(item):
    raw = ShootingBinaryTrajectory.load(dataset_path(item['dataset']) / item['relative_trajectory_path'])
    if sha(raw.root / 'manifest.json') != item['manifest_sha256']:
        raise ValueError(f'Changed raw trajectory manifest: source {item["id"]}')
    if len(raw.positions) != item['frame_count'] or len(raw.atom_ids) != item['atom_count']:
        raise ValueError(f'Changed trajectory shape: {item["id"]}')
    if np.any(np.diff(raw.atom_ids) <= 0):
        raise ValueError(f'Atom IDs must be sorted and unique: {item["id"]}')
    np.testing.assert_allclose(raw.timesteps * item['timestep_fs'] / 1000,
                               np.arange(item['frame_count']) * .75, rtol=0, atol=1e-6)
    return raw


def frame_geometry(raw, frame):
    box = raw.box_high[frame].astype(float) - raw.box_low[frame].astype(float)
    points = np.mod(raw.positions[frame].astype(float), box)
    if not np.isfinite(points).all() or np.any(box <= 0):
        raise ValueError(f'Invalid periodic geometry: {raw.root}, frame {frame}')
    return points, box


def full_labels(points, box, cutoff):
    from ovito.data import DataCollection, Particles, SimulationCell
    from ovito.modifiers import PolyhedralTemplateMatchingModifier
    data = DataCollection()
    particles = Particles(count=len(points))
    particles.create_property('Position', data=points)
    data.objects.append(particles)
    cell = SimulationCell(pbc=(True, True, True))
    cell[...] = np.column_stack((np.diag(box), np.zeros(3)))
    data.objects.append(cell)
    data.apply(PolyhedralTemplateMatchingModifier(rmsd_cutoff=cutoff))
    return np.asarray(data.particles['Structure Type']).astype(np.uint8)


def extraction_contract(config, plan):
    return dict(release_identity=plan['identity'], ptm=config['ptm'],
                producer_sha256=sha(Path(__file__)), ovito=importlib.metadata.version('ovito'),
                coordinates='original observed float16 MD, float64 periodic geometry')


def source(config, plan, item, expected_identity):
    os.environ['OVITO_THREAD_COUNT'] = '1'
    contract = extraction_contract(config, plan)
    if digest(contract) != expected_identity:
        raise ValueError('PTM producer changed during execution; restart with its frozen version')
    folder = resolve_path(config['output']) / 'technical/sources' / str(item['id'])
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'extraction.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = digest(dict(contract=contract, source=item))
        receipt = folder / 'ptm-complete.json'
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if old['identity'] != identity:
                raise ValueError(f'Changed source extraction identity: {item["id"]}')
            for name, checksum in old['files'].items():
                if sha(folder / name) != checksum:
                    raise ValueError(f'Changed PTM chunk: {folder / name}')
            return old
        started = time.monotonic()
        raw = raw_source(item)
        rows = np.searchsorted(raw.atom_ids, item['center_atom_ids'])
        np.testing.assert_array_equal(raw.atom_ids[rows], item['center_atom_ids'])
        labels_path = resolve_path(config['release']) / 'benchmark/sources' / str(item['id']) / 'labels.npy'
        old_labels = np.load(labels_path)
        label_hash = sha(labels_path)
        files, mismatches = {}, []
        frames = item['frame_count']
        chunk = config['ptm']['chunk_frames']
        for start in range(0, frames, chunk):
            stop = min(frames, start + chunk)
            path = folder / f'ptm-{start:04d}-{stop:04d}.npz'
            record = path.with_suffix('.json')
            if record.exists():
                entry = json.loads(record.read_text())
                if entry['identity'] != identity or entry['sha256'] != sha(path):
                    raise ValueError(f'Changed partial PTM: {path}')
            else:
                values = np.stack([full_labels(*frame_geometry(raw, f), config['ptm']['rmsd_cutoff'])
                                   for f in range(start, stop)])
                mismatch = np.argwhere(values[:, rows].T != old_labels[:, start:stop])
                temporary = path.with_suffix('.building.npz')
                np.savez_compressed(temporary, labels=values)
                temporary.replace(path)
                entry = dict(identity=identity, start=start, stop=stop, sha256=sha(path),
                             center_mismatches=[dict(atom=int(raw.atom_ids[rows[c]]), frame=int(start + f),
                                 fixed=int(old_labels[c, start + f]), full=int(values[f, rows[c]]))
                                 for c, f in mismatch])
                write_json(record, entry)
            files[path.name] = entry['sha256']
            mismatches.extend(entry['center_mismatches'])
            write_json(folder / 'extraction-progress.json', dict(state='running', source=item['id'],
                       frames=stop, total_frames=frames, seconds=time.monotonic() - started))
        result = dict(identity=identity, source=item['id'], role=item['role'], files=files,
                      frames=frames, atom_count=item['atom_count'], fixed_labels_sha256=label_hash,
                      center_mismatches=mismatches, seconds=time.monotonic() - started,
                      source_manifest_sha256=item['manifest_sha256'])
        write_json(receipt, result)
        write_json(folder / 'extraction-progress.json', dict(state='complete', frames=frames, total_frames=frames))
        return result


def run(config):
    _, plan = read_release(config['release'])
    root = result_folders(resolve_path(config['output']))
    contract = extraction_contract(config, plan)
    path = root / 'technical/extraction-contract.json'
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError('Refusing to mix PTM producers in an existing audit')
    write_json(path, contract)
    write_json(root / 'technical/config.json', config)
    order = {sid: i for i, sid in enumerate(config['pilot_sources'])}
    sources = sorted(plan['sources'], key=lambda s: (order.get(s['id'], len(order)), s['id']))
    if any(s['role'] != 'train' for s in sources[:len(order)]):
        raise ValueError('The predefined pilot must contain only training sources')
    # Validate every raw path/manifest before launching a long queue.
    for item in sources:
        raw_source(item)
    completed = []
    try:
        with ProcessPoolExecutor(max_workers=config['workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
            pending = {pool.submit(source, config, plan, item, digest(contract)): item['id'] for item in sources}
            for future in as_completed(pending):
                result = future.result()
                completed.append(result['source'])
                write_json(root / 'technical/extraction-state.json', dict(state='running', completed=completed,
                            total_sources=len(sources), last_source=result['source']))
                print(json.dumps(dict(source=result['source'], seconds=result['seconds'],
                                      completed=len(completed), center_mismatches=len(result['center_mismatches']))), flush=True)
        write_json(root / 'technical/extraction-state.json', dict(state='complete', completed=completed,
                    total_sources=len(sources)))
    except Exception:
        write_json(root / 'technical/extraction-state.json', dict(state='failed', completed=completed,
                   total_sources=len(sources), traceback=traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(json.loads(resolve_path(args.config).read_text()))


if __name__ == '__main__':
    main()
