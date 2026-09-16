"""Separate rank-boundary continuity from measured temporal embedding variation."""

import csv
import json
from pathlib import Path
import shutil

import numpy as np

from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.experiment_runner.registry import sha256, write_json


def write_rows(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_smoothness(config):
    original = Path(config['output'])
    root = original.with_name(original.name+'-smoothness')
    local = Path(config['local_output'])
    local = local.with_name(local.name+'-smoothness')
    for sub in ['plots', 'tables', 'technical']:
        (root/sub).mkdir(parents=True, exist_ok=True)
    if (root/'technical/summary.json').exists():
        raise FileExistsError(f'Preserve completed smoothness analysis: {root}')
    diagnostics = Path(config['diagnostics'])/'technical'
    inputs = [diagnostics/'probes.npz', diagnostics/'temporal.npz',
              original/'technical/summary.json', original/'technical/metric-contract.json']
    with np.load(inputs[0]) as probes, np.load(inputs[1]) as temporal:
        train = probes['split'] == 'train'
        contexts = probes['context'][train]
        source = temporal['source']
        sources, counts = np.unique(source, return_counts=True)
        np.testing.assert_array_equal(sources, np.unique(probes['source'][probes['split'] == 'test']))
        np.testing.assert_array_equal(counts, np.full(6, 24))
        assert probes['z'].shape == (5760, 256), probes['z'].shape
        assert temporal['z'].shape == (144, 17, 256), temporal['z'].shape
    old = {r['name']: r for r in json.loads(inputs[2].read_text())['results']}
    names = [f'frozen-{m}' for m in config['modes']]+[f'trained-{m}' for m in config['training_modes']]
    lags = np.asarray(config['temporal_lags_steps'])
    # extract.py and data.py both request source_records(cadence_ps=.75).
    lag_ps = lags*.75
    temporal_rows, boundary_rows, source_rows = [], [], []
    results = {}
    for name in names:
        path = original/'technical'/name/'features.npz'
        inputs.append(path)
        if json.loads((path.parent/'status.json').read_text())['state'] != 'complete':
            raise ValueError(f'Incomplete context feature extraction: {path.parent}')
        with np.load(path) as features:
            z = features['z'].astype(np.float64)
            tz = features['temporal_z'].astype(np.float64)
            crossing = features['crossing_z'].astype(np.float64)
            epsilons = features['epsilons']
        assert z.shape == (5760, 256), (name, z.shape)
        assert tz.shape == (144, 17, 256), (name, tz.shape)
        assert crossing.shape == (4, 72, 2, 256), (name, crossing.shape)
        if not all(np.isfinite(a).all() for a in [z, tz, crossing]):
            raise ValueError(f'Nonfinite encoder output: {path}')
        total_variance = z[train].var(axis=0).mean()
        residual = z[train].copy()
        for context in np.unique(contexts):
            mask = contexts == context
            residual[mask] -= residual[mask].mean(axis=0)
        within_variance = np.mean(residual**2)
        if min(total_variance, within_variance) <= 0:
            raise ValueError(f'Collapsed training embedding: {name}')
        energy = np.stack([np.mean((tz[:, lag:]-tz[:, :-lag])**2, axis=(1, 2)) for lag in lags], axis=1)
        per_source = np.stack([energy[source == s].mean(axis=0) for s in sources])
        raw = per_source.mean(axis=0)
        normalized = {'total': raw/total_variance, 'within_frame': raw/within_variance}
        boundary = np.mean((crossing[:, :, 1]-crossing[:, :, 0])**2, axis=(1, 2))
        # Independently reproduce the existing scientific export before extending it.
        np.testing.assert_allclose(total_variance, old[name]['train_feature_variance'], rtol=1e-12)
        np.testing.assert_allclose(lag_ps, [r['lag_ps'] for r in old[name]['temporal']], rtol=0, atol=0)
        np.testing.assert_allclose(normalized['total'], [r['relative_latent_mse'] for r in old[name]['temporal']], rtol=1e-12)
        np.testing.assert_allclose(boundary/raw[0], [r['fraction_of_075ps_energy'] for r in old[name]['boundary']], rtol=1e-12)
        results[name] = dict(total_variance=float(total_variance), within_frame_variance=float(within_variance),
            lag_ps=lag_ps.tolist(), raw_increment_mse=raw.tolist(),
            relative_increment_mse={k: v.tolist() for k, v in normalized.items()},
            source_increment_mse=per_source.tolist(), epsilon_A=epsilons.tolist(),
            boundary_fraction=(boundary/raw[0]).tolist())
        for j, lag in enumerate(lag_ps):
            temporal_rows.append(dict(method=name, lag_ps=lag, raw_increment_mse=raw[j],
                train_total_variance=total_variance, train_within_frame_variance=within_variance,
                relative_increment_mse_total=normalized['total'][j],
                relative_increment_mse_within_frame=normalized['within_frame'][j]))
            for i, source_id in enumerate(sources):
                source_rows.append(dict(method=name, source=int(source_id), lag_ps=lag,
                                        raw_increment_mse=per_source[i, j]))
        for epsilon, mse in zip(epsilons, boundary, strict=True):
            boundary_rows.append(dict(method=name, epsilon_A=epsilon, raw_crossing_mse=mse,
                                     relative_crossing_mse=mse/total_variance, fraction_of_075ps_energy=mse/raw[0]))
    # All tracks and overlapping time pairs from a source stay together.
    draws = np.random.default_rng(config['seed']).integers(0, len(sources), (config['bootstrap_draws'], len(sources)))
    pairs = [(name, name.split('-')[0]+'-mean80', 'architecture') for name in names if not name.endswith('-mean80')]
    pairs += [(f'trained-{m}', f'frozen-{m}', 'continuation') for m in config['training_modes']]
    comparison_rows = []
    for candidate, reference, comparison in pairs:
        a, b = results[candidate], results[reference]
        for normalization in ['total', 'within_frame']:
            an = np.asarray(a['source_increment_mse'])/a[f'{normalization}_variance']
            bn = np.asarray(b['source_increment_mse'])/b[f'{normalization}_variance']
            ratio = an.mean(axis=0)/bn.mean(axis=0)
            samples = an[draws].mean(axis=1)/bn[draws].mean(axis=1)
            low, high = np.quantile(samples, [.025, .975], axis=0)
            for j, lag in enumerate(lag_ps):
                comparison_rows.append(dict(candidate=candidate, reference=reference, comparison=comparison,
                    normalization=normalization, lag_ps=lag, squared_increment_ratio=ratio[j],
                    ratio_ci95_low=low[j], ratio_ci95_high=high[j], rms_increment_ratio=np.sqrt(ratio[j])))
    snapshot_metric_docs(root, 'mace_context_smoothness')
    for name, rows in [('temporal', temporal_rows), ('boundary', boundary_rows),
                       ('paired-comparisons', comparison_rows), ('source-increments', source_rows)]:
        write_rows(root/'tables'/f'{name}.csv', rows)
    write_json(root/'technical/summary.json', dict(results=results, comparisons=comparison_rows,
        source_ids=sources.tolist(), bootstrap_draws=config['bootstrap_draws'], seed=config['seed']))
    write_json(root/'technical/provenance.json', dict(config=config,
        inputs={str(p): sha256(p) for p in inputs}, original_metrics_reproduced=True))
    plot_smoothness(results, root)
    (root/'README.md').write_text(
        '# MACE embedding smoothness\n\n'
        'Matched 144 tracked series from six held-out simulations, 17 frames at 0.75 ps. '
        'Temporal increments include physical evolution. Normalization uses training embeddings only. '
        'All original temporal and crossing scores were independently reproduced.\n\n'
        '![Smoothness comparison](plots/smoothness.png)\n\n'
        '[Temporal scores](tables/temporal.csv), [paired source intervals](tables/paired-comparisons.csv), '
        '[controlled crossings](tables/boundary.csv), and [definitions](tables/METRICS.md).\n')
    for sub in ['plots', 'tables']:
        shutil.copytree(root/sub, local/sub, dirs_exist_ok=True)
    (local/'technical').mkdir(exist_ok=True)
    for filename in ['summary.json', 'provenance.json', 'metric-contract.json']:
        shutil.copy2(root/'technical'/filename, local/'technical'/filename)
    shutil.copy2(root/'README.md', local/'README.md')
    print(json.dumps([r for r in comparison_rows if r['lag_ps'] in [.75, 12] and r['candidate'].startswith('trained-')], indent=2), flush=True)


def plot_smoothness(results, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    colors = ['#4b5563', '#007f70', '#bc5a20']
    labels = ['Original 80-atom mean', 'Context + smooth inner mean', 'Context + tracked center']
    for mode, color, label in zip(['mean80', 'halo_inner', 'halo_center'], colors, labels, strict=True):
        r = results[f'trained-{mode}']
        for ax, normalization in zip(axes[:2], ['total', 'within_frame'], strict=True):
            ax.plot(r['lag_ps'], r['relative_increment_mse'][normalization], 'o-', color=color, label=label)
        axes[2].plot(r['epsilon_A'], r['boundary_fraction'], 'o-', color=color, label=label)
    axes[0].set(title='Temporal change / total variance', xlabel='Physical lag (ps)',
                ylabel='Normalized squared embedding change', ylim=(0, None))
    axes[1].set(title='Temporal change / within-frame variance', xlabel='Physical lag (ps)',
                ylabel='Normalized squared embedding change', ylim=(0, None))
    axes[2].set(title='Controlled 80th/81st atom crossing', xlabel='Radial perturbation parameter (Å)',
                ylabel='Squared change / natural 0.75 ps change', xscale='log', yscale='log')
    axes[0].legend(fontsize=8, loc='upper left')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle('MACE smoothness after matched 8-epoch continuation', fontsize=14)
    fig.savefig(root/'plots/smoothness.png', dpi=180)
    fig.savefig(root/'plots/smoothness.pdf')
    plt.close(fig)
