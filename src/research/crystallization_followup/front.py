"""Observed bond-coherent components inside declared radii, with a bounded halo."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import sph_harm_y
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from src.project_runtime.paths import resolve_path, dataset_path
from src.data.structural_pretraining.prepare import save_json, file_hash, digest
from src.data.trajectories.shooting import ShootingBinaryTrajectory

RADII = (12., 20., 25.)
NAMES = ('ordered_fraction', 'mean_coherent_bonds_fraction', 'has_ordered',
         'nearest_ordered_distance_over_radius', 'largest_component_fraction',
         'nearest_component_fraction', 'q6_mean', 'q6_std', 'center_coherence_mean',
         'center_coherence_ordered', 'ordered_centroid_norm_over_radius',
         'direction_eigen_min', 'direction_eigen_mid', 'direction_eigen_max')
WIDTH = len(RADII)*len(NAMES)


def frame_features(points, box, centers):
    """No PTM or future labels. Full-cell preprocessing has at most a 10 A halo.

    q6 uses 12 nearest bonds, required to lie within 5 A. Coherent bonds have
    normalized q6 dot > .7; an ordered atom has >=7 coherent bonds. Component
    connectivity is restricted to atoms inside each requested observation ball.
    """
    points = np.mod(np.asarray(points, dtype=float), box)
    tree = cKDTree(points, boxsize=box)
    distance, indices = tree.query(points, k=13, workers=1)
    if not np.array_equal(indices[:, 0], np.arange(len(points))) or np.any(distance[:, 1] <= 0):
        raise ValueError('Coincident atoms or invalid self-neighbor in original MD')
    if distance[:, -1].max() > 5.:
        raise ValueError(f'12-neighbor support exceeds declared 5 A halo: {distance[:, -1].max()}')
    neighbors = indices[:, 1:]
    bonds = points[neighbors]-points[:, None]; bonds -= box*np.round(bonds/box)
    theta = np.arccos(np.clip(bonds[..., 2]/distance[:, 1:], -1, 1))
    phi = np.arctan2(bonds[..., 1], bonds[..., 0])
    q = np.stack([sph_harm_y(6, m, theta, phi).mean(1) for m in range(-6, 7)], -1)
    norm = np.linalg.norm(q, axis=-1)
    unit = np.divide(q, norm[:, None], out=np.zeros_like(q), where=norm[:, None] > 1e-14)
    corr = np.einsum('im,ijm->ij', unit.conj(), unit[neighbors]).real
    coherent = corr > .7; counts = coherent.sum(1); ordered = counts >= 7
    i = np.repeat(np.arange(len(points)), 12); j = neighbors.ravel()
    keep = coherent.ravel() & ordered[i] & ordered[j]
    graph = csr_matrix((np.ones(keep.sum(), dtype=np.uint8), (i[keep], j[keep])), shape=(len(points), len(points)))
    q6 = np.sqrt(4*np.pi/13)*norm
    output = np.empty((len(centers), len(RADII), len(NAMES)), np.float32)
    for ci, center in enumerate(centers):
        pool = np.array(tree.query_ball_point(points[center], max(RADII)), dtype=int)
        relative = points[pool]-points[center]; relative -= box*np.round(relative/box)
        radius = np.linalg.norm(relative, axis=-1)
        alignment = (unit[center].conj()*unit[pool]).sum(-1).real
        for ri, cutoff in enumerate(RADII):
            mask = radius < cutoff; ids = pool[mask]; r = radius[mask]; x = relative[mask]
            solid = ordered[ids]; n = len(ids); nsolid = int(solid.sum())
            largest = nearest_size = centroid = 0.; nearest = 1.; eig = np.zeros(3)
            if nsolid:
                subset = ids[solid]
                _, component = connected_components(graph[subset][:, subset], directed=False)
                sizes = np.bincount(component)
                near = int(np.argmin(r[solid]))
                nearest = float(r[solid][near]/cutoff)
                largest = float(sizes.max()/n); nearest_size = float(sizes[component[near]]/n)
                centroid = float(np.linalg.norm(x[solid].mean(0))/cutoff)
                angular = solid & (r > 1e-8)
                if angular.any():
                    directions = x[angular]/r[angular, None]
                    eig = np.linalg.eigvalsh(directions.T@directions/len(directions))
            alignment_local = alignment[mask]
            output[ci, ri] = [nsolid/n, counts[ids].mean()/12, float(nsolid > 0), nearest,
                largest, nearest_size, q6[ids].mean(), q6[ids].std(), alignment_local.mean(),
                alignment_local[solid].mean() if nsolid else 0., centroid, *eig]
    if not np.isfinite(output).all(): raise FloatingPointError('Nonfinite observed-front descriptor')
    return output.reshape(len(centers), -1), dict(max_neighbor_radius_A=float(distance[:, -1].max()),
                                                ordered_fraction=float(ordered.mean()))


def source(plan, item):
    config = plan['followup_config']['front_cache']
    folder = resolve_path(config)/str(item['id']); folder.mkdir(parents=True, exist_ok=True)
    identity = digest(dict(parent=plan['followup_parent_sha256'], producer=file_hash(Path(__file__)),
                           source=item, radii=RADII, lag_frames=4))
    receipt = folder/'complete.json'
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old['identity'] != identity or old['sha256'] != file_hash(folder/'front.npz'):
            raise ValueError(f'Front cache changed: {folder}')
        return old
    raw = ShootingBinaryTrajectory.load(dataset_path(item['dataset'])/item['relative_trajectory_path'])
    if file_hash(raw.root/'manifest.json') != item['manifest_sha256']:
        raise ValueError(f'Original MD manifest changed: {item["id"]}')
    centers = np.searchsorted(raw.atom_ids, item['center_atom_ids'])
    np.testing.assert_array_equal(raw.atom_ids[centers], item['center_atom_ids'])
    np.testing.assert_allclose(raw.timesteps*item['timestep_fs']/1000, np.arange(801)*.75, rtol=0, atol=1e-6)
    origins = sorted(map(int, plan['observed_histories'][str(item['id'])]))
    frames = sorted({f for origin in origins for f in (origin-4, origin)})
    values = []; stats = []; started = time.monotonic()
    for frame in frames:
        partial = folder/f'frame-{frame}.npz'; record = partial.with_suffix('.json')
        if record.exists():
            old = json.loads(record.read_text())
            if old['identity'] != identity or file_hash(partial) != old['sha256']:
                raise ValueError(f'Front partial identity changed: {partial}')
            with np.load(partial) as a: value = a['features'].copy()
            stat = old['statistics']
        else:
            box = (raw.box_high[frame]-raw.box_low[frame]).astype(float)
            value, stat = frame_features(raw.positions[frame], box, centers)
            np.savez_compressed(partial, features=value)
            save_json(record, dict(identity=identity, frame=frame, sha256=file_hash(partial), statistics=stat))
        values.append(value); stats.append(stat)
    np.savez_compressed(folder/'front.building.npz', frames=frames, atom_ids=raw.atom_ids[centers], features=np.stack(values))
    (folder/'front.building.npz').replace(folder/'front.npz')
    result = dict(identity=identity, source=item['id'], manifest_sha256=item['manifest_sha256'],
                  producer_sha256=file_hash(Path(__file__)), frames=frames, radii_A=RADII,
                  atom_ids=item['center_atom_ids'], feature_names=NAMES, sha256=file_hash(folder/'front.npz'),
                  seconds=time.monotonic()-started, max_neighbor_radius_A=max(x['max_neighbor_radius_A'] for x in stats))
    save_json(receipt, result); return result


def prepare(config):
    from .queue import freeze
    plan = freeze(config); root = resolve_path(config['output'])/'technical'; records = []
    with ProcessPoolExecutor(max_workers=config['front_workers']) as pool:
        futures = [pool.submit(source, plan, item) for item in plan['sources']]
        for future in as_completed(futures):
            records.append(future.result())
            save_json(root/'front-preparation.json', dict(state='preparing', completed=len(records), total=len(futures)))
            print(json.dumps(dict(source=records[-1]['source'], seconds=records[-1]['seconds'], completed=len(records))), flush=True)
    save_json(root/'front-preparation.json', dict(state='complete', completed=len(records), sources=records))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    parser.add_argument('--after', help='Completed forecast queue required before GPU preflight and submission')
    args = parser.parse_args(); config = json.loads(resolve_path(args.config).read_text())
    root = resolve_path(config['output'])/'technical'
    try:
        prepare(config)
        if args.after:
            from src.training_methods.shared_pretraining.queue import deadline_for_job
            deadline = deadline_for_job()
            other = resolve_path(args.after)/'technical'
            specs = json.loads((other/'queue.json').read_text())
            while True:
                statuses = [json.loads((other/'runs'/s['name']/'status.json').read_text())
                            if (other/'runs'/s['name']/'status.json').exists() else {'state':'pending'} for s in specs]
                if any(s['state'] == 'failed' for s in statuses):
                    raise RuntimeError('Optimization queue failed; inspect before front continuation')
                if all(s['state'] == 'complete' for s in statuses): break
                if time.time() > deadline-1800:
                    save_json(root/'handoff.json', dict(state='checkpointed', reason='allocation ending')); return
                time.sleep(20)
            if time.time() > deadline-1800:
                save_json(root/'handoff.json', dict(state='checkpointed', reason='allocation ending')); return
            command = [sys.executable, '-u', '-m', 'src.research.crystallization_followup.queue']
            with (root/'verification.log').open('a') as log:
                subprocess.run(['srun', f'--jobid={os.environ["SLURM_JOB_ID"]}', '--overlap', '--exact',
                                '-N1', '-n1', '-c6', '--gres=gpu:1', '--mem=30G', *command,
                                'verify', '--config', args.config], check=True, stdout=log, stderr=subprocess.STDOUT)
            # Submit from the current frozen producer; queue.submit takes a second
            # immutable executable snapshot for the scientific GPU workers.
            from .queue import submit
            submit(args.config)
            save_json(root/'handoff.json', dict(state='submitted', allocation=os.environ['SLURM_JOB_ID']))
    except Exception as error:
        import traceback
        root.mkdir(parents=True, exist_ok=True)
        save_json(root/'handoff.json', dict(state='failed', error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__': main()
