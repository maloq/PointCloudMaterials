"""Separate disordered-state discovery on a spatial training slab of static Al."""

from concurrent.futures import ThreadPoolExecutor
import json

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score

from src.experiment_runner.artifacts import write_json
from src.experiment_runner.registry import sha256
from src.research.mace_context.cluster_diagnosis import project, projector
from .data import progress
from .methods import AffineMap, adjusted_pair_agreement, fit_states, state_membership, uncertainty
from .physics import group_observables, GROUP_NAMES


def discover(config, root):
    cfg = config['static']
    manifest = json.loads((Path(cfg['sample_cache'])/'metadata.json').read_text())
    with np.load(Path(cfg['analysis'])/'technical/analysis_inference_cache.npz') as data:
        z, coords = data['inv_latents'], data['coords']
    metadata = json.loads((root/'technical/maps.json').read_text())
    head = projector(config)
    directory = root/'technical/static'; directory.mkdir(exist_ok=True)
    rng = np.random.default_rng(config['seed'])
    models, slices, confidence_slices = {}, {}, {}
    metrics, physical, resampling = [], [], []
    offset = 0
    for frame_index, shard in enumerate(manifest['shards']):
        name = Path(shard['file']).stem; n = shard['count']
        c, raw = coords[offset:offset+n], z[offset:offset+n]
        old = np.load(Path(cfg['diagnosis'])/f'technical/physical-{name}.npz')
        types = old['ptm']; other = types == 0
        original = np.load(Path(cfg['analysis'])/f'technical/snapshots/{name}/md_space/local_structure_coords_clusters.npz')
        np.testing.assert_array_equal(c, original['coords'])
        if len(types) != n:
            raise ValueError(f'Static PTM row mismatch: {name}')
        near = cKDTree(c).query(c, k=7, workers=config['cpu_threads'])[1][:, 1:]
        selected = rng.choice(old['sampled_rows'], cfg['physical_samples_per_frame'], replace=False)
        physics_path = directory/f'group-{name}.npz'
        points_path = Path(cfg['source'])/shard['file']
        points_hash = sha256(points_path)
        if physics_path.exists():
            with np.load(physics_path) as data:
                np.testing.assert_array_equal(data['selected'], selected)
                if str(data['source_sha256']) != points_hash:
                    raise ValueError(f'Static geometry changed: {name}')
                observations = data['observables']
        else:
            points = np.load(points_path).astype(np.float64)
            tree = cKDTree(points)
            distance, center_ids = tree.query(c[selected])
            np.testing.assert_allclose(distance, 0., atol=1e-7, rtol=0)
            groups = tree.query_ball_point(c[selected], 18.)
            clouds = [points[np.r_[center, np.setdiff1d(ids, [center])]]-points[center]
                      for center, ids in zip(center_ids, groups, strict=True)]
            with ThreadPoolExecutor(max_workers=config['physics_workers']) as pool:
                observations = np.stack(list(pool.map(group_observables, clouds)))
            np.savez(physics_path, selected=selected, observables=observations, source_sha256=np.array(points_hash))
        train = other & (c[:, 0] < 85.)
        test = other & (c[:, 0] > 185.)
        if frame_index == 0:
            if name != cfg['discovery_frame']:
                raise ValueError(f'Expected first discovery frame {cfg["discovery_frame"]}, got {name}')
            fit_rows = rng.choice(np.flatnonzero(train), cfg['fit_samples'], replace=False)
            np.save(directory/'discovery-fit-rows.npy', fit_rows)
        projected = project(head, raw[:, :256])
        for method, spec in metadata.items():
            block = {'inner': raw[:, :256], 'dual': raw, 'projector': projected}[spec['block']]
            values = AffineMap.load(root/f'technical/maps/{method}.npz')(block)
            for size in config['state_minimum_sizes']:
                key = f'{method}-m{size}'
                if frame_index == 0:
                    state = fit_states(values[fit_rows], size, config['state_minimum_samples'])
                    models[key] = state; joblib.dump(state, directory/f'{key}.joblib')
                state = models[key]
                chunks = [state_membership(state, v, soft=False)[:2] for v in np.array_split(values, max(1, n//8192))]
                labels = np.concatenate([a for a, b in chunks]); strength = np.concatenate([b for a, b in chunks])
                membership = state_membership(state, values[selected])[2]
                u = uncertainty(membership)
                np.savez(directory/f'{key}-{name}.npz', labels=labels, strength=strength,
                    selected=selected, membership=membership, **u)
                for region, mask in [('all', np.ones(n, bool)), ('PTM_other', other), ('PTM_other_spatial_test', test)]:
                    edges = mask[:, None] & mask[near]
                    a = np.broadcast_to(labels[:, None], near.shape)[edges]; b = labels[near][edges]
                    sample = mask[selected]
                    metrics.append(dict(model=method, minimum_size=size, frame=name, region=region,
                        discovered_states=len(state.cluster_persistence_), assigned_fraction=float(np.mean(labels[mask] >= 0)),
                        mean_strength=float(strength[mask].mean()),
                        mean_unassigned_mass=float(u['unassigned_mass'][sample].mean()) if sample.any() else np.nan,
                        mean_ambiguity=float(u['ambiguity'][sample].mean()) if sample.any() else np.nan,
                        **adjusted_pair_agreement(a, b)))
                    eligible = sample & (labels[selected] >= 0)
                    assigned = labels[selected][eligible]; obs = observations[eligible]
                    for col, observable in enumerate(GROUP_NAMES):
                        if len(obs) > 1 and len(np.unique(assigned)) > 1:
                            total = np.square(obs[:, col]-obs[:, col].mean()).sum()
                            between = sum(np.sum(assigned == k)*(obs[assigned == k, col].mean()-obs[:, col].mean())**2
                                          for k in np.unique(assigned))
                            eta = between/total if total > 0 else np.nan
                        else:
                            eta = np.nan
                        physical.append(dict(model=method, minimum_size=size, frame=name, region=region,
                            observable=observable, assigned_samples=len(obs), explained_variance=eta))
                if frame_index == 0:
                    # Resample contiguous 20 A tiles, not individual atoms.
                    tile = np.floor(c[fit_rows]/20).astype(int)
                    _, tile_ids = np.unique(tile, axis=0, return_inverse=True)
                    for draw in range(config['static']['spatial_subsamples']):
                        ids = np.unique(tile_ids)
                        keep_tiles = rng.choice(ids, int(np.ceil(.8*len(ids))), replace=False)
                        keep = np.isin(tile_ids, keep_tiles)
                        repeated = fit_states(values[fit_rows[keep]], size, config['state_minimum_samples'])
                        rows = selected[test[selected]]
                        repeated_labels = state_membership(repeated, values[rows], soft=False)[0]
                        original_labels = labels[rows]; both = (original_labels >= 0) & (repeated_labels >= 0)
                        meaningful = both.sum() > 1 and len(np.unique(original_labels[both])) > 1 and len(np.unique(repeated_labels[both])) > 1
                        resampling.append(dict(model=method, minimum_size=size, draw=draw,
                            assigned_overlap=float(both.mean()),
                            assigned_ari=adjusted_rand_score(original_labels[both], repeated_labels[both]) if meaningful else np.nan))
                    if size == config['state_minimum_sizes'][0]:
                        slab = np.abs(c[:, 2]-np.median(c[:, 2])) < cfg['slice_halfwidth_A']
                        slices[method] = (c[slab], labels[slab]); confidence_slices[method] = (c[slab], strength[slab])
            progress(root, 'static_discovery', frame=name, model=method)
        offset += n
        # Publish complete-so-far tables without dropping earlier frames.
        pd.DataFrame(metrics).to_csv(root/'tables/static_states.csv', index=False)
        pd.DataFrame(physical).to_csv(root/'tables/static_physics.csv', index=False)
        pd.DataFrame(resampling).to_csv(root/'tables/static_state_stability.csv', index=False)
    if offset != len(z):
        raise ValueError('Static metadata does not partition frozen embeddings')
    for label, gallery in [('states', slices), ('strength', confidence_slices)]:
        fig, axes = plt.subplots(3, 3, figsize=(14, 13), constrained_layout=True)
        for ax, (method, (c, values)) in zip(axes.flat, gallery.items(), strict=True):
            if label == 'states':
                color = plt.get_cmap('tab20')(np.maximum(values, 0) % 20)
                color[values < 0] = [.72, .72, .72, 1.]
                ax.scatter(c[:, 0], c[:, 1], c=color, s=2, rasterized=True)
            else:
                ax.scatter(c[:, 0], c[:, 1], c=values, cmap='viridis', vmin=0, vmax=1, s=2, rasterized=True)
            ax.axvline(85, color='black', lw=.7); ax.axvline(185, color='black', lw=.7)
            ax.set(title=method, xlabel='x (A)', ylabel='y (A)', aspect='equal')
        fig.suptitle(f'Al {cfg["discovery_frame"]}: {label}; fit x<85 A, test x>185 A; gray = unassigned')
        fig.savefig(root/f'plots/disordered-{label}.png', dpi=180); plt.close(fig)
    write_json(directory/'protocol.json', dict(discovery_frame=cfg['discovery_frame'],
        fit='PTM Other, x < 85 A; fixed density settings; no temporal or test-frame fitting',
        test='PTM Other, x > 185 A', transfer='The same discovery catalog is applied unchanged to the other five frames',
        limitation='Spatially separated description on one trajectory; not independent trajectories for state-catalog validation.'))
