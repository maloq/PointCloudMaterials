"""Train-fitted PaCMAP views of saved shooting encoders; no encoder/readout fit."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import shutil

import numpy as np

from src.experiment_runner.artifacts import file_hash as sha, json_digest, write_json
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_rows
from src.project_runtime.paths import resolve_path

FAMILY = 'predictive_pacmap'
LABELS = {'mace_vicreg': 'Frozen VICReg128', 'mace_epi': 'Frozen Epi128',
          'mm_tda_block_direct_full': 'MM-TDA-BLOCK-DIRECT-FULL (256)'}
STRATA = ['Clear liquid', 'Visible interface', 'Crystalline center']


def read(path):
    return json.loads(Path(path).read_text())


def settings(path):
    c = read(path)
    if c['protocol'] != 'predictive_followup_pacmap_v1':
        raise ValueError('Expected the shooting follow-up PaCMAP protocol')
    followup = read(resolve_path(c['followup_config']))
    return c, followup, resolve_path(followup['output'])/'analyses'/c['analysis']


def saved_embeddings(c, data, manifest):
    from .common import features, jobs, folder
    for source in c['sources']:
        if source != 'joint':
            z = features(c, data, manifest, source)
            if source in ('mace_vicreg', 'mace_epi'):
                origin = dict(feature_sha256=manifest['files'][source+'.npy'],
                    producer=manifest['binding']['frozen_encoders'])
            else:
                origin = read(resolve_path(c['output'])/'technical'/f'features-{source}.json')
            yield source, LABELS[source], z, origin
    for arm in jobs(c, 'joint'):
        root = folder(c, arm)
        done = read(root/'technical/complete.json')
        path = root/'technical/predictions.npz'
        if (done['state'] != 'complete' or done['arm'] != arm or
                done['target_identity'] != manifest['identity'] or
                sha(path) != done['predictions_sha256'] or
                sha(root/'technical/best.pt') != done['checkpoint_sha256']):
            raise ValueError(f'Changed or incomplete joint encoder: {root}')
        with np.load(path) as saved:
            for key in ('parent', 'atom_ids'):
                if not np.array_equal(saved[key], data[key]):
                    raise ValueError(f'Embedding row identity mismatch: {root}/{key}')
            z = saved['z']
        label = f'New MACE128 | {arm["target"]}, {arm["variance"]} | seed {arm["seed"]}'
        yield root.name, label, z, dict(path=str(path), **done)


def project(config):
    import pacmap
    from .common import load
    from src.research.predictive_baseline.data import moments

    c, followup, root = settings(config)
    check_metric_docs(family=FAMILY)
    if version('pacmap') != c['pacmap_version']:
        raise ValueError('PaCMAP version differs from the declared recipe')
    if c['feature_scaling'] != 'source_weighted_train_standardization':
        raise ValueError('Unknown feature geometry')
    data, manifest = load(followup)
    train = np.flatnonzero(data['roles'] == 'train')
    test = np.flatnonzero(data['roles'] == 'test')
    if len(np.unique(np.column_stack((data['parent'], data['atom_ids'])), axis=0)) != len(data['parent']):
        raise ValueError('Observation identities are not unique')
    for subdir in ('data', 'technical/projections', 'technical/models', 'plots'):
        (root/subdir).mkdir(parents=True, exist_ok=True)
    binding = dict(config=c, followup_config_sha256=sha(resolve_path(c['followup_config'])),
        target_identity=manifest['identity'], producer_sha256=sha(__file__),
        packages={k: version(k) for k in ('pacmap', 'numpy', 'scipy', 'numba', 'faiss-cpu', 'scikit-learn')})
    identity = json_digest(binding)
    if (root/'technical/binding.json').exists():
        if read(root/'technical/binding.json')['identity'] != identity:
            raise ValueError('Projection protocol changed; use a new analysis revision')
    write_json(root/'technical/binding.json', dict(identity=identity, **binding))
    shutil.copy2(__file__, root/'technical/pacmap_views.py')
    shutil.copy2(config, root/'technical/config.json')
    metadata = dict(row=test.tolist(), parent=data['parent'][test].tolist(),
        atom_id=data['atom_ids'][test].tolist(), strata=data['strata'][test].tolist(),
        sources=data['sources'][test].tolist(), inclusion_weight=data['weights'][test].tolist())
    source_names = sorted(set(metadata['sources']))
    metadata['source_index'] = [source_names.index(s) for s in metadata['sources']]
    fields = {
        'structure': dict(label='Present structure', values=metadata['strata'], categories=STRATA),
        'source': dict(label='Held-out source', values=metadata['source_index'],
                       categories=[f'S{i+1}' for i in range(len(source_names))])}
    y = np.asarray(data['y'][test], dtype=np.float64)
    for horizon in (3, 6, 12):
        index = manifest['target_columns'].index(f'crystalline_fraction@{horizon}ps')
        fields[f'crystal_{horizon}'] = dict(label=f'Mean local crystalline fraction at {horizon} ps (%)',
                                           values=(100*y[:, :, index].mean(1)).tolist(), unit='%')
    index = manifest['target_columns'].index('crystalline_fraction@6ps')
    fields['spread_6'] = dict(label='Across-shot SD of crystalline fraction at 6 ps (percentage points)',
                              values=(100*y[:, :, index].std(1, ddof=1)).tolist(), unit='pp')
    index = manifest['target_columns'].index('bond_order/l6_qbar@6ps')
    fields['q6_6'] = dict(label='Mean q-bar-6 at 6 ps', values=y[:, :, index].mean(1).tolist(), unit='')
    write_json(root/'data/observations.json', dict(**metadata, source_names=source_names, fields=fields))
    entries, coverage = [], []
    for key, label, z, origin in saved_embeddings(followup, data, manifest):
        if z.ndim != 2 or len(z) != len(data['parent']) or not np.isfinite(z).all():
            raise ValueError(f'Invalid embedding: {key}, {z.shape}')
        zsha = hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest()
        receipt_path = root/'technical/projections'/f'{key}.json'
        coords_path = root/'data'/f'{key}.npz'
        model_path = root/'technical/models'/f'{key}.pkl'
        provenance = dict(analysis_identity=identity, feature_sha256=zsha, origin=origin)
        if receipt_path.exists():
            saved = read(receipt_path)
            if (saved['provenance'] != provenance or sha(coords_path) != saved['coordinate_sha256']
                    or sha(model_path) != saved['model_sha256']):
                raise ValueError(f'Changed projection: {key}')
            print(f'Reuse {key}', flush=True)
        else:
            center, scale = moments(z[train], data['weights'][train], c['normalization_floor'])
            xtrain = ((z[train]-center)/scale).astype(np.float32)
            xtest = ((z[test]-center)/scale).astype(np.float32)
            reducer = pacmap.PaCMAP(**c['pacmap'])
            print(f'Fit {key}: {len(train)} training / {len(test)} held-out / z{z.shape[1]}', flush=True)
            train_xy = reducer.fit_transform(xtrain, init='pca')
            test_xy = reducer.transform(xtest, basis=xtrain, init='pca')
            if test_xy.shape != (len(test), 2) or not np.isfinite(test_xy).all():
                raise ValueError(f'Invalid held-out projection: {key}')
            np.savez_compressed(coords_path, train_xy=train_xy, test_xy=test_xy,
                train_rows=train, test_rows=test, parent=data['parent'][test], atom_ids=data['atom_ids'][test],
                feature_center=center, feature_scale=scale)
            with model_path.open('wb') as stream:
                pickle.dump(reducer, stream, protocol=pickle.HIGHEST_PROTOCOL)
            write_json(receipt_path, dict(provenance=provenance, coordinate_sha256=sha(coords_path),
                                         model_sha256=sha(model_path)))
        entries.append(dict(key=key, label=label, dimension=z.shape[1], coordinate_sha256=sha(coords_path)))
        coverage.append(dict(encoder=key, embedding_dimensions=z.shape[1], projection_dimensions=2,
            fitting_observations=len(train), displayed_observations=len(test),
            fitting_sources=len(np.unique(data['sources'][train])), displayed_sources=len(source_names)))
    write_metric_rows(coverage, root, family=FAMILY, name='projection-coverage')
    write_json(root/'technical/projections-complete.json', dict(state='complete', identity=identity,
        encoders=entries, observations_sha256=sha(root/'data/observations.json')))
    return root


def draw_static(root, entries, observations, primary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    keys = [primary, *LABELS]
    titles = ['New MACE128 (full / free; seed 20261001)', *LABELS.values()]
    strata = np.asarray(observations['strata'])
    for field, population, filename in (
            ('structure', 'all', 'present-structure'), ('crystal_6', 'all', 'future-crystallinity-6ps'),
            ('crystal_6', 'liquid', 'clear-liquid-future-6ps'), ('source', 'all', 'source-audit')):
        mask = np.ones(len(strata), dtype=bool) if population == 'all' else strata == 0
        f = observations['fields'][field]
        values = np.asarray(f['values'])
        if 'categories' in f:
            palette = ['#2878b5', '#e7a132', '#bd4762'] if field == 'structure' else list(plt.get_cmap('tab10').colors[:6])
            cmap = ListedColormap(palette)
            norm = BoundaryNorm(np.arange(len(palette)+1)-.5, cmap.N)
        else:
            cmap = 'viridis'
            norm = plt.Normalize(0, 100 if population == 'all' else float(values[mask].max()))
        fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.3), constrained_layout=True)
        # Fixed shuffled draw order across panels avoids painting an entire stratum last.
        order = np.random.default_rng(20261001).permutation(np.flatnonzero(mask))
        for ax, key, title in zip(axes.flat, keys, titles, strict=True):
            xy = entries[key]
            scatter = ax.scatter(xy[order, 0], xy[order, 1], c=values[order], s=9 if population == 'all' else 17,
                                 cmap=cmap, norm=norm, alpha=.85, linewidths=0, rasterized=True)
            ax.set_title(title, fontsize=11, loc='left')
            # A filtered view keeps the full projection's extents; no hidden refit or zoom.
            span = np.ptp(xy, axis=0)
            ax.set_xlim(xy[:, 0].min()-.05*span[0], xy[:, 0].max()+.05*span[0])
            ax.set_ylim(xy[:, 1].min()-.05*span[1], xy[:, 1].max()+.05*span[1])
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_color('#d2d8df')
        bar = fig.colorbar(scatter, ax=axes, shrink=.77, pad=.025)
        bar.set_label(f['label'])
        if 'categories' in f:
            bar.set_ticks(range(len(f['categories'])), labels=f['categories'])
        group = 'All held-out environments' if population == 'all' else 'Clear-liquid held-out environments'
        fig.suptitle(f'PaCMAP | {group}\n{int(mask.sum()):,} identical observations per panel; six sources', fontsize=17)
        fig.supxlabel('Fit: 4,224 training observations. Independent maps; axes are arbitrary.\n'
                      'Stratified sample: point density is not population frequency. Colors never enter the projection.', fontsize=10)
        for suffix in ('png', 'pdf'):
            fig.savefig(root/'plots'/f'{filename}.{suffix}', dpi=180)
        plt.close(fig)


def render(config):
    from plotly.offline import get_plotlyjs
    c, _, root = settings(config)
    receipt = read(root/'technical/projections-complete.json')
    if sha(root/'data/observations.json') != receipt['observations_sha256']:
        raise ValueError('Changed observation colors or identities')
    observations = read(root/'data/observations.json')
    coords, models = {}, []
    for entry in receipt['encoders']:
        path = root/'data'/f'{entry["key"]}.npz'
        if sha(path) != entry['coordinate_sha256']:
            raise ValueError(f'Changed coordinates: {path}')
        with np.load(path) as a:
            if not np.array_equal(a['test_rows'], observations['row']):
                raise ValueError('Different observations in projection panels')
            coords[entry['key']] = a['test_xy']
        models.append(dict(**entry, xy=np.round(coords[entry['key']], 6).tolist()))
    draw_static(root, coords, observations, c['primary_joint'])
    payload = dict(observations=observations, models=models, primary=c['primary_joint'], frozen=list(LABELS))
    template = Path(__file__).with_name('pacmap_view.html')
    page = template.read_text().replace('__PLOTLY_JS__', get_plotlyjs()).replace(
        '__PAYLOAD__', json.dumps(payload, allow_nan=False).replace('</', '<\\/'))
    (root/'index.html').write_text(page)
    shutil.copy2(template, root/'technical/pacmap_view.html')
    write_json(root/'technical/render-complete.json', dict(state='complete',
        projection_identity=receipt['identity'], implementation_sha256=sha(__file__),
        template_sha256=sha(template), page_sha256=sha(root/'index.html'),
        packages={k:version(k) for k in ('plotly', 'matplotlib')},
        plots={p.name:sha(p) for p in sorted((root/'plots').glob('*'))}))
    print(f'Published {root}/index.html', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('project', 'render', 'run'))
    parser.add_argument('--config', default='configs/analysis/predictive_pacmap_20261001.json')
    args = parser.parse_args()
    if args.stage in ('project', 'run'): project(args.config)
    if args.stage in ('render', 'run'): render(args.config)
