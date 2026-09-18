"""Export source-weighted scores, trajectory figures and a self-contained gallery."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from src.data.structural_pretraining.prepare import file_hash, save_json
from src.experiment_runner.metric_docs import snapshot_metric_docs
from .metrics import reference_statistics, trajectory_metrics, stratified_draws, rms_interval

METHODS = ['mace', 'gatr', 'tda', 'soap', 'bond_order', 'radial', 'angular']
LABELS = dict(mace='MACE', gatr='GATr', tda='TDA (80 atoms)', soap='SOAP (7 Å)',
              bond_order='Bond order', radial='Radial (32)', angular='Angular (16)')
COLORS = dict(zip(METHODS, ['#2066a8', '#d55e00', '#009e73', '#8b5da8', '#66574e', '#cb9b12', '#64a5a0']))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def features(folder, a):
    return dict(mace=np.load(folder/'mace.npy'), gatr=np.load(folder/'gatr.npy'), tda=a['tda'],
        soap=a['soap'], bond_order=a['bond_order'], radial=a['geometry'][:, :32], angular=a['geometry'][:, 64:80])


def report(plan):
    root = Path(plan['config']['output'])
    for name in ['tables', 'plots']:
        (root/name).mkdir(exist_ok=True)
    if (root/'technical/summary.json').exists():
        raise FileExistsError(f'Preserve completed stability report: {root}')
    train = {m: [] for m in METHODS}
    test = []
    for source in plan['sources']:
        folder = root/'technical/sources'/str(source['id'])
        receipt = json.loads((folder/'complete.json').read_text())
        for filename, expected in receipt['hashes'].items():
            if file_hash(folder/filename) != expected:
                raise ValueError(f'Changed observation artifact: {folder/filename}')
        for name in ['mace', 'gatr']:
            verification = json.loads((folder/f'{name}-verification.json').read_text())
            if file_hash(folder/f'{name}.npy') != verification['features_sha256']:
                raise ValueError(f'Changed encoder artifact: {folder/name}')
        a = dict(np.load(folder/'observations.npz'))
        f = features(folder, a)
        if source['split'] == 'train':
            for method in METHODS:
                train[method].append(f[method])
        else:
            shape = (len(a['frames']), len(a['centers']))
            np.testing.assert_array_equal(a['frames'], np.arange(source['frame_count']))
            test.append(dict(source=source, data=a, features={m: x.reshape(*shape, -1) for m, x in f.items()}))
    references = {m: reference_statistics(np.concatenate(train[m])) for m in METHODS}
    draws = stratified_draws([t['source']['temperature_K'] for t in test], plan['config']['bootstrap_draws'], plan['config']['seed'])
    lags = plan['config']['lag_frames']
    source_rows, lag_rows, summary_rows, phase_rows, normalization_rows, pair_rows = [], [], [], [], [], []
    all_metrics = {}
    for method in METHODS:
        ref = references[method]
        results = []
        for item in test:
            source, a = item['source'], item['data']
            z = item['features'][method]
            result = trajectory_metrics(z, ref, lags)
            result['source'] = source['id']
            results.append(result)
            source_rows.append(dict(method=method, source=source['id'], temperature_K=source['temperature_K'],
                **{key: value for key, value in result.items() if key not in ('jump2', 'lag2', 'source')}))
            labels = a['labels'].reshape(z.shape[:2])
            for phase, code in [('unclassified', 0), ('FCC', 1), ('HCP', 2), ('BCC', 3)]:
                mask = (labels[:-1] == code) & (labels[1:] == code)
                phase_rows.append(dict(method=method, source=source['id'], phase=phase,
                    pairs=int(mask.sum()), jump2_mean=float(result['jump2'][mask].mean()) if mask.any() else None))
        all_metrics[method] = results
        j, ci = rms_interval([r['jump2_mean'] for r in results], draws)
        standardized, sci = rms_interval([r['standardized_jump2'] for r in results], draws)
        acceleration, aci = rms_interval([r['acceleration2'] for r in results], draws)
        rough = np.array([r['roughness'] for r in results])
        rough_ci = np.quantile(rough[draws].mean(1), [.025, .975])
        lag_values = np.stack([r['lag2'] for r in results])
        for k, lag in enumerate(lags):
            value, interval = rms_interval(lag_values[:, k], draws)
            lag_rows.append(dict(method=method, lag_ps=lag*plan['cadence_ps'], rms_jump=value,
                ci95_low=interval[0], ci95_high=interval[1]))
        summary_rows.append(dict(method=method, dimensions=ref['dimensions'], sources=len(test),
            tracks=sum(len(t['data']['centers']) for t in test), lag_ps=plan['cadence_ps'],
            rms_jump=j, ci95_low=ci[0], ci95_high=ci[1],
            p95_jump=float(np.quantile(np.concatenate([np.sqrt(r['jump2']).ravel() for r in results]), .95)),
            standardized_rms_jump=standardized, standardized_ci95_low=sci[0], standardized_ci95_high=sci[1],
            rms_acceleration=acceleration, acceleration_ci95_low=aci[0], acceleration_ci95_high=aci[1],
            roughness=float(rough.mean()), roughness_ci95_low=rough_ci[0], roughness_ci95_high=rough_ci[1],
            increment_cosine=float(np.mean([r['increment_cosine'] for r in results])),
            reversal_fraction=float(np.mean([r['reversal_fraction'] for r in results])),
            jump_075_to_12_ratio=float(np.sqrt(lag_values[:, 0].mean()/lag_values[:, lags.index(16)].mean())),
            reference_effective_rank=ref['effective_rank'], reference_total_variance=ref['trace']))
        normalization_rows.append(dict(method=method, reference_rows=sum(len(x) for x in train[method]),
            reference_sources=plan['config']['reference_sources_per_temperature']*len(plan['config']['temperatures_K']),
            dimensions=ref['dimensions'], active_standardized_dimensions=int(ref['active'].sum()),
            independent_pair_rms_scale=ref['scale'], total_variance=ref['trace'], effective_rank=ref['effective_rank']))
    for method in METHODS[1:]:
        a = np.array([r['jump2_mean'] for r in all_metrics['mace']])
        b = np.array([r['jump2_mean'] for r in all_metrics[method]])
        ratios = np.sqrt(a[draws].mean(1)/b[draws].mean(1))
        ci = np.quantile(ratios, [.025, .975])
        pair_rows.append(dict(candidate='mace', reference=method, rms_jump_ratio=np.sqrt(a.mean()/b.mean()), ci95_low=ci[0], ci95_high=ci[1]))
    # Membership changes are measured, never identified as purely numerical noise.
    membership_rows = []
    for item in test:
        a = item['data']; ids = a['nearest_ids'].reshape(len(a['frames']), len(a['centers']), 80)
        churn = np.array([[1-len(np.intersect1d(ids[t, c], ids[t+1, c], assume_unique=True))/80
            for c in range(len(a['centers']))] for t in range(len(a['frames'])-1)])
        item['churn'] = churn
        for method in METHODS:
            result = next(r for r in all_metrics[method] if r['source'] == item['source']['id'])
            rho = spearmanr(churn.ravel(), np.sqrt(result['jump2']).ravel()).statistic
            membership_rows.append(dict(source=item['source']['id'], method=method,
                mean_80_atom_replacement_fraction=float(churn.mean()), jump_churn_spearman=float(rho)))
    snapshot_metric_docs(root, 'trajectory_stability')
    for name, rows in [('summary', summary_rows), ('per-source', source_rows), ('lag-curves', lag_rows),
                       ('phase-conditioned', phase_rows), ('normalization', normalization_rows),
                       ('paired-comparisons', pair_rows), ('neighbor-turnover', membership_rows)]:
        write_csv(root/'tables'/f'{name}.csv', rows)
    np.savez(root/'technical/reference-statistics.npz', **{m+'_'+k: v for m, ref in references.items() for k, v in ref.items()})
    make_plots(root, test, references, all_metrics, summary_rows, lag_rows)
    save_json(root/'technical/summary.json', dict(protocol=plan['protocol'], summary=summary_rows,
        paired_comparisons=pair_rows, observations=sum(t['features']['mace'].shape[0]*t['features']['mace'].shape[1] for t in test),
        source_ids=[t['source']['id'] for t in test], reference_rows=sum(map(len, train['mace']))))
    write_report(root, plan, summary_rows, pair_rows)
    from .explore import export
    export(plan)


def make_plots(root, test, references, results, summary, lags):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for i, row in enumerate(summary):
        m = row['method']; color = COLORS[m]
        axes[0, 0].errorbar(i, row['rms_jump'], yerr=[[row['rms_jump']-row['ci95_low']], [row['ci95_high']-row['rms_jump']]], fmt='o', color=color, capsize=4)
        axes[1, 0].errorbar(i, row['roughness'], yerr=[[row['roughness']-row['roughness_ci95_low']], [row['roughness_ci95_high']-row['roughness']]], fmt='o', color=color, capsize=4)
        axes[1, 1].scatter(row['rms_jump'], row['reference_effective_rank'], color=color, s=65)
        offset = {'bond_order': (-38, 10), 'tda': (5, -12), 'soap': (-58, 6), 'radial': (5, 6)}.get(m, (5, 4))
        axes[1, 1].annotate(LABELS[m], (row['rms_jump'], row['reference_effective_rank']), xytext=offset, textcoords='offset points', fontsize=9)
        curve = [r for r in lags if r['method'] == m]
        x = [r['lag_ps'] for r in curve]
        axes[0, 1].plot(x, [r['rms_jump'] for r in curve], 'o-', ms=3, color=color, label=LABELS[m])
        axes[0, 1].fill_between(x, [r['ci95_low'] for r in curve], [r['ci95_high'] for r in curve], color=color, alpha=.08)
    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_xticks(range(len(METHODS)), [LABELS[m] for m in METHODS], rotation=25, ha='right')
    axes[0, 0].set(title='Frame-to-frame movement at 0.75 ps', ylabel='RMS jump / training-reference pair distance', ylim=(0, None))
    axes[0, 1].set(title='Movement over longer time intervals', xlabel='Lag (ps)', ylabel='Normalized RMS jump', xscale='log', ylim=(0, None))
    axes[0, 1].legend(fontsize=8, ncol=2)
    axes[1, 0].axhline(1.5, color='#777777', ls='--', lw=1, label='Independent-frame reference: 1.5')
    axes[1, 0].set(title='Changes in direction and speed', ylabel='Second-difference roughness (0 = linear)', ylim=(0, 2.1))
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].set(title='Movement and reference feature variation', xlabel='Normalized RMS jump (0.75 ps)', ylabel='Training-reference effective rank', xlim=(0, max(r['rms_jump'] for r in summary)*1.38))
    for ax in axes.flat:
        ax.grid(alpha=.15)
    fig.suptitle('MACE, GATr and descriptors along matched Al trajectories\n40 tracked atoms · 10 held-out sources · 400–520 K · 0–600 ps', fontsize=14)
    fig.savefig(root/'plots/comparison.png', dpi=180); fig.savefig(root/'plots/comparison.pdf'); plt.close(fig)
    # Deterministic display: first sampled test source per temperature and its
    # first sampled atom. No track is selected for visually favorable behavior.
    first = [next(t for t in test if t['source']['temperature_K'] == temp) for temp in sorted({t['source']['temperature_K'] for t in test})]
    fig, axes = plt.subplots(3, 5, figsize=(18, 9), sharex='col', constrained_layout=True)
    for column, item in enumerate(first):
        time_ps = item['data']['times_ps']; sid = item['source']['id']; center = item['data']['centers'][0]
        for m in ['mace', 'gatr', 'tda', 'soap']:
            z = item['features'][m][:, 0].astype(float)
            # PC1 is fitted solely to training reference observations.
            ref_parts = []
            for train_source in [s for s in json.loads((root/'technical/plan.json').read_text())['sources'] if s['split'] == 'train']:
                folder = root/'technical/sources'/str(train_source['id'])
                a = np.load(folder/'observations.npz')
                ref_parts.append(features(folder, a)[m])
            ref = np.concatenate(ref_parts).astype(float)
            _, _, vt = np.linalg.svd(ref-references[m]['mean'], full_matrices=False)
            pc = (z-references[m]['mean'])@vt[0]/references[m]['scale']
            axes[0, column].plot(time_ps, pc, color=COLORS[m], lw=.7, alpha=.85, label=LABELS[m])
            jumps = np.sqrt(next(r for r in results[m] if r['source'] == sid)['jump2'][:, 0])
            axes[1, column].plot(time_ps[1:], jumps, color=COLORS[m], lw=.6, alpha=.75)
        order = item['data']['order'].reshape(len(time_ps), -1, 8)
        labels = item['data']['labels'].reshape(len(time_ps), -1)[:, 0]
        axes[2, column].plot(time_ps, order[:, 0, 4], color='#333333', lw=.8, label='q̄6 bond order')
        axes[2, column].fill_between(time_ps, 0, .7, where=np.isin(labels, [1, 2, 3]), color='#009e73', alpha=.12, label='PTM crystalline')
        axes[0, column].set_title(f'{item["source"]["temperature_K"]:g} K · source {sid}\nAtom {center}')
        axes[2, column].set(xlabel='Time (ps)', ylim=(0, .7))
        for ax in axes[:, column]:
            ax.grid(alpha=.15)
    axes[0, 0].set_ylabel('Training PC1 / pair-distance scale\nSeparate projection for each method')
    axes[1, 0].set_ylabel('Normalized adjacent jump')
    axes[2, 0].set_ylabel('Physical structural context')
    axes[0, 0].legend(fontsize=7); axes[2, 0].legend(fontsize=7)
    fig.suptitle('Unsmoothed trajectories: identical atoms, frames and source data', fontsize=14)
    fig.savefig(root/'plots/trajectories.png', dpi=180); fig.savefig(root/'plots/trajectories.pdf'); plt.close(fig)
    # Heatmaps preserve every trajectory and frame; no temporal filtering.
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    combined = [np.concatenate([np.sqrt(r['jump2']).T for r in results[m]], axis=0) for m in ['mace', 'gatr', 'tda', 'soap']]
    limit = np.quantile(np.concatenate([a.ravel() for a in combined]), .99)
    for ax, m, values in zip(axes, ['mace', 'gatr', 'tda', 'soap'], combined, strict=True):
        im = ax.imshow(values, origin='lower', aspect='auto', extent=(.75, 600, 0, values.shape[0]), vmin=0, vmax=limit, cmap='magma')
        ax.set_ylabel(LABELS[m]+'\nTrack index')
        for k in range(4, 40, 4):
            ax.axhline(k, color='white', alpha=.2, lw=.5)
    axes[-1].set_xlabel('Time (ps)')
    fig.colorbar(im, ax=axes, label='Normalized jump (shared scale; saturated above pooled 99th percentile)', shrink=.8)
    fig.suptitle('All 40 tracks: adjacent-frame movement at 0.75 ps', fontsize=14)
    fig.savefig(root/'plots/all-tracks.png', dpi=180); plt.close(fig)


def write_report(root, plan, summary, pairs):
    lines = ['# Trajectory stability of MACE, GATr and descriptors', '',
        'Matched exploratory audit of 10 training/selection-held-out Al MEAM sources at 400, 450, 500, 510 and 520 K. '
        'Four seeded centers per source, all 801 frames from 0 to 600 ps: 32,040 observations and 32,000 adjacent pairs. '
        'Five separate training sources supply 420 normalization observations. No model is fitted or updated.', '',
        f'MACE selected update {plan["checkpoints"]["mace"]["step"]}; GATr selected update {plan["checkpoints"]["gatr"]["step"]}. '
        'Weights and source selection were frozen at the beginning of this audit. Native compiled mixed precision with FP32 geometry; raw encoder z128, no projector or prediction head.', '',
        '| Representation | RMS jump at 0.75 ps (95% source interval) | Roughness | Effective rank |',
        '|---|---:|---:|---:|']
    for row in summary:
        lines.append(f'| {LABELS[row["method"]]} | {row["rms_jump"]:.3f} ({row["ci95_low"]:.3f}–{row["ci95_high"]:.3f}) | {row["roughness"]:.3f} | {row["reference_effective_rank"]:.2f} |')
    lines += ['', '[Normalized RMS jump](../../../docs/research_glossary.md#normalized-rms-jump) divides movement by '
        'the RMS independent-pair distance of the separate training reference. A value of 1 therefore means movement '
        'as large as that reference distance. Roughness measures second differences relative to adjacent increments: '
        '0 for a linear path, about 1.5 for independent frames, and 2 for exact alternation. '
        'Intervals resample complete sources within temperature; they condition on the fixed training reference.', '',
        '![Comparison](plots/comparison.png)', '', '![Trajectories](plots/trajectories.png)', '',
        '![All tracked atoms](plots/all-tracks.png)', '',
        '[Explore all 40 trajectories interactively](explore.html).', '', '## Interpretation limits', '',
        'Finite-lag jumps contain real atomic motion, neighborhood replacement and stored-coordinate quantization. '
        'This measures observed temporal variability, not a pure numerical-noise estimate. No temporal smoothing is applied. '
        'The native saved cadence cannot resolve sub-0.75-ps jitter. The sources use float16 full-box positions; '
        'the audit does not infer a full-precision noise floor.', '',
        'Descriptors have different observation support: the encoders use the trained 16.87 Å neighborhood, '
        'TDA uses the nearest 80 atoms including the center, SOAP uses a 7 Å nominal Gaussian-density cutoff, '
        'radial/angular features use the 5–7 Å smooth support, and bond order uses 12-neighbor local and neighbor-averaged invariants. '
        'This compares the deployed representations, not an isolated architecture effect. '
        'The raw covariance-trace normalization is basis invariant; coordinate-standardized results are also exported as a sensitivity check. '
        'A small jump or low effective rank alone does not establish usefulness or collapse.', '',
        'PTM-unclassified is not synonymous with liquid. Phase-conditioned tables require the same PTM class at both endpoints '
        'and do not establish metastability. These test sources were explored in earlier research; they are held out from '
        'these encoder fits/selection, not a newly untouched confirmatory test. Only one trained seed per encoder is evaluated.', '',
        '[Exact metric definitions](tables/METRICS.md) · [Scores](tables/summary.csv) · '
        '[Paired comparisons](tables/paired-comparisons.csv) · [Per-source values](tables/per-source.csv) · '
        '[Phase conditions](tables/phase-conditioned.csv) · [Neighbor turnover](tables/neighbor-turnover.csv)', '',
        'Reproduce with `conda run -n pointnet-torch214 python -m src.research.trajectory_stability '
        '--config configs/analysis/trajectory_stability.json`. Completed reports are preserved; change the output directory for another audit.', '']
    (root/'README.md').write_text('\n'.join(lines))
    table = ''.join(f'<tr><td>{LABELS[r["method"]]}</td><td>{r["rms_jump"]:.3f}</td><td>{r["ci95_low"]:.3f}–{r["ci95_high"]:.3f}</td><td>{r["roughness"]:.3f}</td></tr>' for r in summary)
    (root/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Al trajectory stability</title><style>body{font:16px system-ui;max-width:1450px;margin:32px auto;padding:0 24px;color:#24313d}img{width:100%;height:auto}td,th{text-align:left;padding:8px 22px 8px 0;border-bottom:1px solid #ddd}a{color:#2066a8}</style>'
        '<h1>Al trajectory stability</h1><p>40 atoms · 10 held-out sources · 400–520 K · 801 frames · 0.75 ps cadence</p>'
        '<p>All methods follow identical atoms and frames. Jumps include physical dynamics and neighborhood changes. '
        'Normalization uses 420 observations from five separate training sources.</p>'
        '<table><tr><th>Representation</th><th>RMS jump</th><th>95% source interval</th><th>Roughness</th></tr>'+table+'</table>'
        '<p><a href="explore.html">Explore all 40 trajectories</a> · <a href="RESULTS.md">Findings</a> · <a href="README.md">Scientific report</a> · <a href="tables/summary.csv">Metric CSV</a> · <a href="tables/METRICS.md">Exact definitions</a></p>'
        ''.join(f'<h2>{label}</h2><img src="plots/{name}.png" alt="{label}">' for name, label in [('comparison','Matched comparison'),('trajectories','Unsmoothened atom trajectories'),('all-tracks','Every tracked atom')])+'</html>')
