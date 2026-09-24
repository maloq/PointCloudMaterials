"""Evaluate a retained checkpoint on the same independent material assay."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score

from src.experiment_runner.metric_docs import write_metric_table
from src.research.geoframe_continuity.analysis import load_model, groups, gather_groups, signatures
from .reference import CONTEXT_NAMES, write_json
from .metrics import participation, boundary_coherence, classification, fidelity, perturbation


@torch.inference_mode()
def encode(model, clouds, batch=256):
    result = []
    for first in range(0, len(clouds), batch):
        x = torch.as_tensor(clouds[first:first+batch], dtype=torch.float32, device='cuda')
        features = model.encoder.forward_features(x)
        values = torch.stack([features, model.vicreg.project_features(features)], 1)
        if not torch.isfinite(values).all():
            raise FloatingPointError(f'Nonfinite embedding at row {first}')
        result.append(values.cpu().numpy())
    return np.concatenate(result)


def frame_metrics(z, arrays, record):
    n = record['anchor_count']; split = arrays['split'][:n]
    ki = 0 if record['material']=='Al' else 1
    order = arrays['order'][:n, ki]
    primary = arrays['context'][:, 1]
    test_pairs = (arrays['split'][:n]==1) & (arrays['split'][n:]==1)
    stable = (arrays['context'] == primary[:, None]).all(1)
    crystal = np.isin(arrays['ptm'][:n, 1], [1, 2, 3])
    liquid = ~crystal & (arrays['solid_fraction'][:n, 1]<=.1)
    result = dict(collapse=participation(z[:n]), liquid_collapse=participation(z[:n][liquid]),
        reference_coverage=dict(liquid_fraction=float(liquid.mean()),
                                cutoff_stable_fraction=float(stable[:n].mean())),
        context=classification(z[:n], primary[:n], split, range(7)),
        ptm=classification(z[:n], arrays['ptm'][:n, 1], split, range(5)),
        liquid_order=fidelity(z[:n], order[:, :6], order[:, 6], split, liquid))
    for cutoff, j in [('008', 0), ('010', 1), ('012', 2)]:
        result['context_cutoff_'+cutoff] = classification(z[:n], arrays['context'][:n, j], split, range(7))
    if record['material']=='Al':
        result['planar_fault'] = classification(z[:n], arrays['fault'][:n], split, range(5))
    pairs = test_pairs & stable[:n] & stable[n:]
    result['spatial'] = boundary_coherence(z, primary, arrays['pair_distance'], n, pairs)
    nonbulk = pairs & (primary[:n] != 1) & (primary[n:] != 1)
    result['nonbulk_spatial'] = boundary_coherence(z, primary, arrays['pair_distance'], n, nonbulk)
    shuffled = z[np.random.default_rng(20260923).permutation(len(z))]
    result['shuffled_spatial'] = boundary_coherence(shuffled, primary, arrays['pair_distance'], n, nonbulk)
    result['collapsed_spatial'] = boundary_coherence(np.zeros_like(z), primary, arrays['pair_distance'], n, nonbulk)
    ids = arrays['topology_rows']
    result['liquid_topology'] = fidelity(z[ids], arrays['topology'], order[ids, 6], split[ids], liquid[ids])
    fit = split==0; test = split==1
    km = KMeans(n_clusters=7, n_init=5, random_state=20260923).fit(z[:n][fit])
    clusters = km.predict(z[:n])
    mask = test & (primary[:n] != 1)
    result['nonbulk_cluster_ami'] = float(adjusted_mutual_info_score(primary[:n][mask], clusters[mask]))
    result['cluster_context_counts'] = [[int(((clusters==k) & (primary[:n]==j) & test).sum())
                                          for j in range(7)] for k in range(7)]
    return result, clusters


def spatial_plot(root, name, record, arrays, clusters):
    n = record['anchor_count']; xyz = arrays['coords'][:n]
    # A finite slab makes interior planar defects visible without projection overplotting.
    slab = np.abs(xyz[:, 2]-np.median(xyz[:, 2])) < np.ptp(xyz[:, 2])*.12
    fields = [arrays['ptm'][:n, 1], arrays['context'][:n, 1], clusters,
              arrays['order'][:n, 0 if record['material']=='Al' else 1, 4]]
    titles = ['PTM motif', 'Independent context', 'Embedding K=7', 'Averaged bond order']
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for ax, field, title in zip(axes.flat, fields, titles):
        scatter = ax.scatter(xyz[slab, 0], xyz[slab, 1], c=field[slab], s=6,
                             cmap='tab10' if title!='Averaged bond order' else 'viridis')
        ax.set(title=title, xlabel='x (Å)', ylabel='y (Å)', aspect='equal')
        fig.colorbar(scatter, ax=ax, shrink=.7)
    fig.suptitle(f'{name}: {record["material"]} {Path(record["file"]).stem}\nStatic reference; candidate labels do not establish future nucleation')
    fig.savefig(root/'plots'/f'{name}-frame-{record["frame_index"]:02d}-spatial.png', dpi=170)
    plt.close(fig)


def run(checkpoint, output, name, plots=False):
    root = Path(output); assay = root/'technical/reference'
    manifest = json.loads((assay/'manifest.json').read_text())
    for filename, expected in manifest['files'].items():
        if hashlib.sha256((assay/filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Changed reference array: {assay/filename}')
    model = load_model(checkpoint)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    metrics = {}; provenance = dict(checkpoint=str(Path(checkpoint).resolve()),
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256((assay/'manifest.json').read_bytes()).hexdigest(),
        precision='float32', grouping='deterministic FPS', projector_applied=1)
    folder = root/'technical/evaluations'/name; folder.mkdir(parents=True, exist_ok=True)
    for record in manifest['frames']:
        i = record['frame_index']; arrays = np.load(assay/f'frame-{i:02d}.npz')
        clouds = arrays['clouds']; z = encode(model, clouds)
        # Require deterministic repeated evaluation before measuring perturbations.
        repeated = encode(model, clouds[:256])
        np.testing.assert_array_equal(z[:256], repeated)
        selected = clouds[:256]
        noise = np.random.default_rng(20260923+i).normal(size=selected.shape).astype(np.float32)
        changes = {str(a): encode(model, selected+noise*a/record['radius_A']) for a in (1e-4, .01, .1)}
        ng, centers, ci, gi = groups(torch.from_numpy(selected).cuda())
        ng2, _ = gather_groups(torch.from_numpy(selected+noise*1e-4/record['radius_A']).cuda(), ci, gi)
        switches = float((signatures(ng)!=signatures(ng2)).any(-1).float().mean())
        for j, representation in enumerate(('encoder', 'projector')):
            values, clusters = frame_metrics(z[:, j], arrays, record)
            values['continuity'] = {amplitude: perturbation(z[:256, j], value[:, j]) for amplitude, value in changes.items()}
            values['triad_index_switch_fraction_1e4_A'] = switches
            metrics[f'frame_{i:02d}_{record["material"]}_{representation}'] = values
            if plots and representation=='projector':
                spatial_plot(root, name, record, arrays, clusters)
                from umap import UMAP
                xy = UMAP(n_neighbors=30, min_dist=.1, random_state=20260923, n_jobs=1).fit_transform(z[:record['anchor_count'], j])
                fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
                for ax, labels, title in zip(axes, [clusters, arrays['context'][:record['anchor_count'], 1]],
                                              ['Embedding clusters', 'Independent reference context']):
                    sc = ax.scatter(*xy.T, c=labels, s=3, cmap='tab10', vmin=0, vmax=9)
                    ax.set(title=title, xticks=[], yticks=[]); fig.colorbar(sc, ax=ax, ticks=range(7))
                fig.suptitle(f'{name} / {record["material"]} / {Path(record["file"]).stem}\nRefitted UMAP: exploratory view, not a metric of motion')
                fig.savefig(root/'plots'/f'{name}-frame-{i:02d}-umap.png', dpi=170); plt.close(fig)
        np.savez(folder/f'frame-{i:02d}.npz', embeddings=z)
        print(f'{name}: evaluated {record["material"]}/{Path(record["file"]).stem}', flush=True)
    from . import prediction
    corpus, prediction_config, clouds = prediction.prepare()
    future_z = encode(model, clouds)
    np.savez(folder/'future-embeddings.npz', embeddings=future_z)
    for j, representation in enumerate(('encoder', 'projector')):
        metrics['prediction_'+representation] = prediction.evaluate(future_z[:, j], corpus,
                                            prediction_config, folder, representation)
    # Matched dimensionality and optimizer-selection rule, with current physics only.
    metrics['prediction_current_physics'] = prediction.evaluate(np.zeros_like(future_z[:, 0]), corpus,
                                        prediction_config, folder, 'current-physics')
    provenance['future_assay_identity'] = corpus.manifest['identity']
    write_json(folder/'provenance.json', provenance)
    write_metric_table(metrics, root, family='geoframe_evolution', name=name)
    write_json(folder/'metrics.json', metrics)
    return metrics


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', required=True); p.add_argument('--output', required=True)
    p.add_argument('--name', required=True); p.add_argument('--plots', action='store_true')
    a = p.parse_args(); run(a.checkpoint, a.output, a.name, a.plots)


if __name__ == '__main__':
    main()
