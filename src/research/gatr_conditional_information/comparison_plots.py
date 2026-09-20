"""Figures and a self-contained report for the paired latest-checkpoint assay."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots import save

NAMES = dict(plus_gatr='GATr latest', plus_mace='MACE latest', old_gatr='GATr previous',
    old_mace='MACE previous', plus_soap='SOAP', gatr='GATr latest', mace='MACE latest', soap='SOAP', tda='TDA',
    delta_gatr='GATr angular difference', delta_mace='MACE angular difference')
COLORS = dict(plus_gatr='#3660c9', plus_mace='#d77c29', old_gatr='#a6b6df', old_mace='#edc291',
    plus_soap='#9347b3', gatr='#3660c9', mace='#d77c29', soap='#9347b3', tda='#666666',
    delta_gatr='#209b89', delta_mace='#a94848')


def grouped(ax, frame, targets, methods, labels):
    width = .78/len(methods)
    for k, method in enumerate(methods):
        rows = frame[frame.method == method].set_index('target').loc[targets]
        x = np.arange(len(targets))+(k-(len(methods)-1)/2)*width
        values, lo, hi = (rows[key].to_numpy() for key in ('improvement_percent', 'low', 'high'))
        ax.bar(x, values, width, color=COLORS[method], label=NAMES[method])
        ax.errorbar(x, values, yerr=np.stack((values-lo, hi-values)),
            fmt='none', ecolor='#333', capsize=2, lw=.9)
    ax.axhline(0, color='#555', lw=.8)
    ax.set_xticks(range(len(targets)), labels)
    ax.set_ylabel('Reduction in held-out error (%)')


def render(root, tables, config, plan):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False})
    gains = pd.DataFrame(tables['conditional_gains'])
    matched = pd.DataFrame(tables['spatial_matched_gains'])
    targets = ['q4', 'q6', 'qbar6', 'angular_arrangement']
    labels = ['q4', 'q6', 'q̄6', 'Angular\narrangement']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    for ax, family in zip(axes, ('linear', 'nonlinear'), strict=True):
        frame = gains[(gains.task == 'structure') & (gains.family == family) & (gains.baseline == 'radial_control')]
        grouped(ax, frame, targets, ['old_gatr', 'plus_gatr', 'old_mace', 'plus_mace', 'plus_soap'], labels)
        ax.set_title(f'{family.title()} readout · unseen simulation source')
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle('Information beyond radial structure and density\nFinal local checkpoints at update 622; common controls and identical observations', fontsize=14)
    save(fig, root, 'conditional-information')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for ax, family in zip(axes, ('linear', 'nonlinear'), strict=True):
        frame = matched[(matched.family == family) & (matched.caliper_A == .05) & (matched.baseline == 'radial_control')]
        grouped(ax, frame, targets[1:], ['plus_gatr', 'plus_mace', 'plus_soap'], labels[1:])
        ax.set_title(f'{family.title()} readout · predicted pair differences')
    axes[0].legend(fontsize=8)
    fig.suptitle('609 matched pairs across nine sources · before radial-duplication control\nRadial RMS ≤0.05 Å; density gap ≤2%; paired source-bootstrap intervals', fontsize=13)
    save(fig, root, 'spatial-matched-information')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for ax, table, is_matched in ((axes[0], gains, False), (axes[1], matched, True)):
        pieces = []
        for name in ('gatr', 'mace'):
            take = (table.task == 'structure') & (table.family == 'nonlinear') & (table.baseline == 'duplicate_'+name)
            if is_matched:
                take &= table.caliper_A == .05
            pieces.append(table[take])
        grouped(ax, pd.concat(pieces), targets[1:], ['plus_gatr', 'delta_gatr', 'plus_mace', 'delta_mace'], labels[1:])
        ax.set_title('609 strictly matched pairs' if is_matched else '32,040 trajectory observations')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=4, fontsize=8)
    fig.suptitle('Structural information beyond the redundant-radial-input control\nNonlinear source-held-out probes; each state compared with its own radial duplicate', fontsize=14)
    save(fig, root, 'structural-duplication-control')

    horizons = [f'crystallize_{h}ps' for h in config['future_horizons_ps']]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for column, family in enumerate(('linear', 'nonlinear')):
        for row, conditional in enumerate((False, True)):
            pieces = []
            for name in ('gatr', 'mace'):
                baseline = f'current_duplicate_{name}' if conditional else f'duplicate_{name}'
                method = f'current_delta_{name}' if conditional else f'plus_{name}'
                f = gains[(gains.family == family) & (gains.task == 'future') & (gains.baseline == baseline) & (gains.method == method)].copy()
                f['method'] = 'plus_'+name
                pieces.append(f)
            grouped(axes[row, column], pd.concat(pieces), horizons, ['plus_gatr', 'plus_mace'], [f'{h} ps' for h in config['future_horizons_ps']])
            axes[row, column].set_ylabel('Brier-error reduction (%)')
            title = 'Angular difference after current order' if conditional else 'Original state beyond radial duplication'
            axes[row, column].set_title(f'{family.title()} · {title}')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Future sustained local crystallization\n17,674 at-risk observations; complete follow-up; unseen simulation sources', fontsize=14)
    save(fig, root, 'future-crystallization')

    summary = pd.DataFrame(tables['stability_summary']).set_index('method')
    methods = ['old_gatr', 'gatr', 'old_mace', 'mace', 'soap', 'tda']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    rows = summary.loc[methods]
    value = rows.normalized_rms_jump.to_numpy()
    axes[0].bar(range(len(methods)), value, color=[COLORS[m] for m in methods])
    axes[0].errorbar(range(len(methods)), value, yerr=[value-rows.low, rows.high-value], fmt='none', color='#333', capsize=3)
    axes[0].set_xticks(range(len(methods)), [NAMES[m].replace(' ', '\n') for m in methods])
    axes[0].set(title='Consecutive-frame jitter · 0.75 ps', ylabel='RMS jump / independent-reference distance')
    lags = pd.DataFrame(tables['stability_lags'])
    for method in methods:
        f = lags[lags.method == method].groupby('lag_ps').displacement2.mean()
        axes[1].plot(f.index, np.sqrt(f.values), 'o-', ms=3, color=COLORS[method], label=NAMES[method])
    axes[1].set(xscale='log', xlabel='Lag (ps)', ylabel='Normalized RMS displacement', title='Change with increasing time separation')
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle('Trajectory stability on the same forty atom tracks\nEach representation normalized by the same five training-reference sources', fontsize=14)
    save(fig, root, 'trajectory-stability')

    def pick(table, target, method, baseline='radial_control', family='nonlinear', caliper=None, digits=2):
        take = (table.target == target) & (table.method == method) & (table.baseline == baseline) & (table.family == family)
        if caliper is not None:
            take &= table.caliper_A == caliper
        found = table[take]
        if len(found) != 1:
            raise ValueError(f'Ambiguous report cell: {target}, {method}, {baseline}')
        r = found.iloc[0]
        return f'{r.improvement_percent:+.{digits}f}% [{r.low:+.{digits}f}, {r.high:+.{digits}f}]'
    lines = ['# Latest MACE and GATr: information beyond radial structure', '',
        'Completed on the user-approved H100 on nodesumo01. Both final training checkpoints are pinned at '
        '**update 622**, after five epoch equivalents. Their validation-selected exports were at update 576; '
        'this assay intentionally uses the final weights. The newer expanded-data MACE campaign had no checkpoint '
        'when this run was frozen. No encoder was trained or modified.', '',
        '## Fixed population and controls', '',
        'The test retains the previous 32,040 observations from ten Al trajectories, four tracked atom identities '
        'per trajectory, 801 frames at 0.75 ps. The spatial extension retains 22,381 environments from the same '
        '70 snapshots. All test source ancestries are excluded from both encoders’ training and selection. '
        'These previously explored sources make this an exploratory comparison, not a blind confirmation.', '',
        'Each encoder runs with its own hash-verified frozen training implementation, native BF16/FP32 boundaries '
        'and compiled execution. Both now crop to normalized radius 8 with a 6–8 taper, about 7.94 Å outer support '
        'for Al. They were trained on mixed materials with equivariant bond-order supervision. The previous '
        'Al-only checkpoints used larger neighborhoods and different training objectives/budgets; changes do '
        'not isolate the effect of bond supervision.', '',
        'The common radial control contains the previous 152 radial/density features, 33 new local-support radius '
        'quantiles and five local radial moments, temperature/time, and **both** new encoders’ radius-only states. '
        'Those states use identical sorted radii placed on a fixed Fibonacci angular pattern. Each original '
        'state and its difference from its own radial-only state are tested separately. Additional duplication '
        'controls add another copy of that encoder’s radial state without adding information.', '',
        'Linear and nonlinear random-Fourier-feature ridge probes use ten whole-source holdouts and nested '
        'source splits for regularization. All scaling uses training sources only. Spatial probes reuse the '
        'trajectory-selected settings. Brackets below are 95% whole-source bootstrap intervals stratified by '
        'temperature; overlapping frames/pairs are not independent replicates. They do not include refitting uncertainty.', '',
        '## Structural information', '',
        'Percentage reductions in held-out prediction error beyond the common radial control; positive is better. '
        'The table uses nonlinear probes. Old and new exports are read out under the same new controls.', '',
        '| Target | Previous GATr | Latest GATr | Previous MACE | Latest MACE | SOAP |',
        '|---|---:|---:|---:|---:|---:|']
    for target in targets:
        cells = [pick(gains, target, method) for method in ('old_gatr', 'plus_gatr', 'old_mace', 'plus_mace', 'plus_soap')]
        lines.append(f'| {target} | '+' | '.join(cells)+' |')
    lines += ['', '## Directly matched angular differences', '',
        '**609 pairs across nine sources** pass both ≤0.05 Å radial RMS criteria and ≤2% density difference. '
        'Matching preserves the previous first-80 and larger-support radius-quantile criteria exactly; changing '
        'the encoder support did not select new pairs. The looser 0.10 Å sensitivity retains 125,006 pairs. '
        'The sparse four-track population still has only three strict pairs and cannot support an inferential conclusion.', '',
        '| Matched target contrast | Latest GATr | Latest MACE | SOAP |', '|---|---:|---:|---:|']
    for target in targets[1:]:
        lines.append(f'| {target} | '+' | '.join(pick(matched, target, method, caliper=.05)
            for method in ('plus_gatr', 'plus_mace', 'plus_soap'))+' |')
    lines += ['', '## Structural gains after radial duplication', '',
        'The preceding tables alone can overstate angular information: adding a nearly redundant state '
        'changes the ridge prior and nonlinear kernel. The stronger comparisons below replace a second '
        'copy of the corresponding radius-only state with the original state. Both candidates have the '
        'same added dimension; no extra angular information exists in the duplicate.', '',
        '| Population | Target | Original GATr beyond its radial duplicate | Original MACE beyond its radial duplicate |',
        '|---|---|---:|---:|']
    for label, table, caliper in (('All trajectory rows', gains, None), ('609 matched pairs', matched, .05)):
        for target in targets[1:]:
            lines.append(f'| {label} | {target} | '+' | '.join(pick(table, target, 'plus_'+name, 'duplicate_'+name,
                caliper=caliper) for name in ('gatr', 'mace'))+' |')
    lines += ['', 'For example, the apparent matched GATr qbar6 gain above becomes '
        +pick(matched, 'qbar6', 'plus_gatr', 'duplicate_gatr', caliper=.05)
        +' against redundant radial input. This control must accompany the raw R-star gains.', '',
        'The explicitly scaled GATr angular difference yields an angular-moment gain of '
        +pick(gains, 'angular_arrangement', 'delta_gatr', 'duplicate_gatr')
        +' on all trajectory observations, but '
        +pick(matched, 'angular_arrangement', 'delta_gatr', 'duplicate_gatr', caliper=.05)
        +' on strict spatial pairs. At the looser 0.10 Å spatial threshold the gain is '
        +pick(matched, 'angular_arrangement', 'delta_gatr', 'duplicate_gatr', caliper=.1)
        +'. Thus small recoverable angular responses should not be described as literally absent.', '']
    lines += ['', '## Future crystallization', '',
        'There are 17,674 eligible prospective rows before first sustained local FCC/HCP/BCC onset, '
        'with the current and previous two frames noncrystalline. Eight consecutive crystalline frames confirm '
        'onset; all 24/48/96 ps horizons include complete confirmation. This is one realized future, not an '
        'iso-configurational propensity or committor.', '',
        'The stricter nonlinear comparisons below control redundant radial inputs. The final two columns also '
        'condition on the six current bond-order and sixteen angular descriptors, then add the encoder’s '
        'angular difference. Scores are reductions in Brier error.', '',
        '| Horizon | GATr original / duplicate control | MACE original / duplicate control | GATr after current order | MACE after current order |',
        '|---|---:|---:|---:|---:|']
    for horizon, target in zip(config['future_horizons_ps'], horizons, strict=True):
        cells = [pick(gains, target, 'plus_'+name, 'duplicate_'+name, digits=4) for name in ('gatr', 'mace')]
        cells += [pick(gains, target, 'current_delta_'+name, 'current_duplicate_'+name) for name in ('gatr', 'mace')]
        lines.append(f'| {horizon} ps | '+' | '.join(cells)+' |')
    scores = pd.DataFrame(tables['source_scores'])
    lines += ['', 'Absolute Brier errors are needed to judge forecasting usefulness, not only relative increments:', '',
        '| Horizon | Training prevalence only | Radial control | Radial + GATr | Radial + MACE |', '|---|---:|---:|---:|---:|']
    for horizon, target in zip(config['future_horizons_ps'], horizons, strict=True):
        f = scores[(scores.task == 'future') & (scores.family == 'nonlinear') & (scores.target == target)]
        baseline = f[f.method == 'radial_control'].constant_loss.mean()
        losses = [f[f.method == m].loss.mean() for m in ('radial_control', 'plus_gatr', 'plus_mace')]
        lines.append(f'| {horizon} ps | {baseline:.5f} | '+' | '.join(f'{x:.5f}' for x in losses)+' |')
    lines += ['', 'An absent gain with these finite, clipped least-squares readouts does not prove absence of future '
        'information. If absolute errors do not beat prevalence, useful forecasting has not been demonstrated. '
        'There are forty tracks and ten independent source replicates; no causal effect is identified.', '',
        '## Temporal stability', '',
        'Consecutive 0.75 ps RMS change is divided by the independent-pair distance scale from the original five '
        'training-reference sources. Smaller is smoother relative to that representation’s variability. '
        'Smoothness is useful only alongside retained structural information.', '',
        '| Representation | Normalized RMS jump | Reversing successive increments | Effective rank on reference |', '|---|---:|---:|---:|']
    for method in methods:
        r = summary.loc[method]
        lines.append(f'| {NAMES[method]} | {r.normalized_rms_jump:.3f} [{r.low:.3f}, {r.high:.3f}] | {100*r.reversal_fraction:.1f}% | {r.effective_rank:.2f} |')
    lines += ['', '## Artifacts and reproduction', '',
        '- [Structural comparison](plots/conditional-information.png)',
        '- [Strictly matched environments](plots/spatial-matched-information.png)',
        '- [Structural gains after radial duplication](plots/structural-duplication-control.png)',
        '- [Future crystallization controls](plots/future-crystallization.png)',
        '- [Trajectory stability](plots/trajectory-stability.png)',
        '- [Metric definitions](tables/METRICS.md); source scores, linear/nonlinear results and matching sensitivities in `tables/`.',
        '- Frozen checkpoint identities, producer paths, native inference checks and source-held-out predictions in `technical/`.', '',
        '```bash', 'conda run -n pointnet-torch214 python -m src.research.gatr_conditional_information.comparison \\',
        '  --config configs/analysis/conditional_information_local_last.json', '```', '']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    (root/'README.md').write_text('# Latest encoder conditional information\n\n[Results](RESULTS.md) · [Figures](index.html) · [Metrics](tables/METRICS.md)\n')
    figures = ('structural-duplication-control', 'conditional-information', 'spatial-matched-information', 'future-crystallization', 'trajectory-stability')
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Latest MACE/GATr conditional information</title>'
        '<style>body{max-width:1300px;margin:24px auto;font:17px system-ui}img{max-width:100%}</style>'
        '<h1>Latest MACE and GATr: information beyond radial structure</h1><p>Final update 622 · H100 · fixed source holdouts</p>'
        '<p><a href="RESULTS.md">Full findings</a> · <a href="tables/METRICS.md">Metric definitions</a></p>'
        +''.join(f'<img src="plots/{name}.png" alt="{name}">' for name in figures))
