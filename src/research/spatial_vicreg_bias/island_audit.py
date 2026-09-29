"""Exploratory audit of the islands in the saved joint-descriptor PaCMAP view.

Regions are hand-selected from the user's screenshot, not physical labels or
held-out hypothesis tests. No descriptor or neural model is fitted here.
"""
import argparse
import json
from pathlib import Path

import faiss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from src.research.structural_state.common import sha, write_json
from .pacmap_views import settings, context


def counts(x):
    keys, n = np.unique(x, return_counts=True)
    return {str(k): int(v) for k, v in zip(keys, n)}


def quantiles(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return np.quantile(x, [.1, .5, .9]).tolist() if len(x) else None


def nearest(x, queries, k):
    index = faiss.IndexFlatL2(x.shape[1])
    index.add(np.ascontiguousarray(x, dtype=np.float32))
    return index.search(np.ascontiguousarray(queries, dtype=np.float32), k)


def run(config, output):
    out = Path(output)
    for sub in ('plots', 'data', 'technical'):
        (out/sub).mkdir(parents=True, exist_ok=True)
    faiss.omp_set_num_threads(4)
    c, corr, parent = settings(config)
    a, ref, labels, models, bindings = context(c, corr, parent)
    path = Path(c['output'])/'data/descriptors-joint-all_test.npz'
    with np.load(path) as saved:
        for key in ('original_row', 'source', 'frame', 'atom'):
            if not np.array_equal(a[key], saved[key]):
                raise ValueError(f'Projection identity mismatch: {key}')
        y = saved['pacmap2']
    model = models['joint']
    columns = model['columns']
    names = np.asarray(a['names'])[columns]
    x = ((a['targets'][:, columns]-model['mean'])/model['sd']/model['balance']).astype(np.float32)
    families = np.array([s.split('/')[0] for s in names])
    # All rectangular selections are fixed before inspecting descriptor signatures.
    boxes = {'A purple upper-left': (-17, -12, 1, 7),
             'B orange upper-right': (13, 17, 7, 11),
             'C thin far-left': (-20, -15, -12, -7),
             'D bottom-left': (-17, -12, -23, -15),
             'E bottom-right': (4, 10, -25, -18),
             'F small central bridge': (-5, -1, .5, 3.5)}
    group = np.full(len(x), -1, np.int32)
    for i, bounds in enumerate(boxes.values()):
        lo, hi, bottom, top = bounds
        take = (y[:, 0]>lo)&(y[:, 0]<hi)&(y[:, 1]>bottom)&(y[:, 1]<top)
        if np.any(group[take]>=0) or not take.any():
            raise ValueError(f'Invalid screenshot region {i}')
        group[take] = i
    background = np.flatnonzero(group<0)
    neighborhoods = {}
    spaces = {'joint': x, 'joint_z_clipped5': np.clip(x*model['balance'], -5, 5)/model['balance']}
    spaces.update({f: x[:, families==f] for f in np.unique(families)})
    for label, values in spaces.items():
        _, nn = nearest(values, values, 22)
        # Exclude self by identity, including exact-distance ties.
        neighborhoods[label] = np.stack([row[row!=i][:20] for i, row in enumerate(nn)])
    report = dict(projection=str(path), projection_sha256=sha(path), rows=len(x),
                  dimensions=x.shape[1], exploratory=True, neural_training=False,
                  feature_bindings=bindings, boxes=boxes, regions={},
                  definitions={'neighbor_purity': 'Fraction of the exact 20 Euclidean nearest other rows in the full displayed feature population belonging to this screenshot region.',
                  'outside_inside_ratio': 'Per-row distance to nearest point outside this region divided by distance to nearest other point inside it; quantiles 10/50/90. Distances use the frozen standardized family-balanced joint metric.',
                  'feature_contribution': 'Sum of squared coordinate differences to each row\'s nearest point in the main population, divided by the total squared difference. Main population excludes all six selected regions.',
                  'per_row_family_contribution': 'Quantiles of family squared distance / total squared distance, avoiding domination by extreme observations.',
                  'sensitivity': 'Exact 20-NN region purity in each separate family and in joint features with each training-standardized coordinate clipped to [-5,5] before family balancing. No fit or layout selection; diagnostic only.',
                  'quantiles': '10th, 50th, 90th percentiles; nonfinite interface distances excluded and counted separately.',
                  'limitations': 'Repeated atom-time observations are dependent. Regions selected visually, no inferential p-values. Comparisons are descriptive, not new state labels.'})
    for i, name in enumerate(boxes):
        ids = np.flatnonzero(group==i)
        outside = np.flatnonzero(group!=i)
        _, other = nearest(x[outside], x[ids], 1)
        dout = np.sum((x[ids].astype(float)-x[outside[other[:, 0]]].astype(float))**2, axis=1)
        inside = cdist(x[ids].astype(float), x[ids].astype(float), metric='sqeuclidean')
        np.fill_diagonal(inside, np.inf)
        din = inside.min(axis=1)
        dm, match = nearest(x[background], x[ids], 1)
        matched = background[match[:, 0]]
        diff = (x[ids].astype(float)-x[matched].astype(float))**2
        contributions = diff.sum(0)/diff.sum()
        top = np.argsort(contributions)[::-1][:12]
        signatures = []
        for j in top:
            signatures.append(dict(feature=str(names[j]), contribution=float(contributions[j]),
                island_raw=quantiles(a['targets'][ids, columns[j]]),
                nearest_main_raw=quantiles(a['targets'][matched, columns[j]]),
                train_sd=float(model['sd'][j])))
        r = dict(n=len(ids), source_counts=counts(a['source'][ids]), frame_counts=counts(a['frame'][ids]),
                 distinct_source_atom_pairs=len(set(zip(a['source'][ids].tolist(), a['atom'][ids].tolist()))),
                 ptm_counts=counts(a['ptm'][ids]), joint_cluster_counts=counts(labels['joint'][ids]),
                 support_crystal_fraction=quantiles(a['support_fraction'][ids]),
                 interface_distance_A=quantiles(ref['distance'][ids]),
                 no_interface=int(np.sum(~np.isfinite(ref['distance'][ids]))),
                 nearest20_region_purity={space: float(np.mean(group[nn[ids]]==i)) for space, nn in neighborhoods.items()},
                 outside_inside_ratio=quantiles(np.sqrt(dout/np.maximum(din, 1e-20))),
                 median_main_distance=float(np.median(np.sqrt(dm[:, 0]))),
                 family_contributions={f: float(contributions[families==f].sum()) for f in np.unique(families)},
                 per_row_family_contributions={f: quantiles(diff[:, families==f].sum(1)/diff.sum(1)) for f in np.unique(families)},
                 extra_signatures={s: quantiles(a['targets'][ids, a['names'].index(s)]) for s in (
                     'cna/fixed36_421', 'cna/fixed36_422', 'cna/fixed32_555',
                     'cna/fixed36_666', 'cna/adaptive12_444',
                     'bond_order/l4_q', 'bond_order/l6_q', 'tda/n32_h2_loglife_max', 'tda/n80_h2_loglife_max')},
                 top_features=signatures)
        report['regions'][name] = r
        print(name, json.dumps({k: v for k, v in r.items() if k!='top_features'}), flush=True)
        print('Top features:', json.dumps(signatures[:5]), flush=True)
    np.savez_compressed(out/'data/selected-rows.npz', region=group, original_row=a['original_row'],
                        source=a['source'], frame=a['frame'], atom=a['atom'])
    report['implementation_sha256'] = sha(Path(__file__))
    write_json(out/'technical/audit.json', report)
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.scatter(*y.T, c=labels['joint'], cmap='tab10', vmin=-.5, vmax=9.5, s=.8, alpha=.75, linewidths=0)
    for i, name in enumerate(boxes):
        center = np.median(y[group==i], axis=0)
        ax.annotate(f'{name[0]} ({np.sum(group==i)})', center, xytext=(15, 12), textcoords='offset points',
                    fontsize=12, weight='bold', arrowprops={'arrowstyle': '->', 'color': '#222'},
                    bbox={'facecolor': 'white', 'edgecolor': '#ddd', 'alpha': .9})
    ax.set(xlabel='PaCMAP 1', ylabel='PaCMAP 2', title='Joint rich descriptors: exploratory island audit\nSaved coordinates and cluster labels unchanged')
    fig.tight_layout(); fig.savefig(out/'plots/annotated-islands.png', dpi=170); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.config, args.output)
