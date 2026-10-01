"""Render saved pilot evidence without replacing its numerical definitions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.fixed_cohort.protocol import sha, write_json
from src.project_runtime.paths import resolve_path
from .common import read, output


def publish(c, destination):
    root = Path(destination); plots = root / 'plots'; plots.mkdir(parents=True, exist_ok=True)
    paths = dict(learning='mechanisms-v1/tables/learning.csv',
                 acquisition='mechanisms-v1/tables/acquisition.csv',
                 bootstrap='shooting-reliability-v1/tables/source-bootstrap.csv',
                 sources='shooting-reliability-v1/tables/source-errors.csv',
                 atomistic='atomistic-v1/tables/horizon-noise-cost.csv')
    data = {key: pd.read_csv(output(c) / 'analyses' / value) for key, value in paths.items()}
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': '#ffffff', 'axes.facecolor': '#ffffff'})
    fig, axs = plt.subplots(2, 2, figsize=(13, 10), layout='constrained')
    colors = ['#64748b', '#007f86']
    ax = axs[0, 0]
    subset = data['sources'].query('total_shots == 12 and population == "clear_liquid"')
    means = subset.groupby(['source', 'arm']).corrected_squared_feature_error.mean().unstack()
    for _, row in means.iterrows():
        ax.plot([0, 1], row[['prior', 'descriptor_ridge']], color='#64748b', alpha=.5, marker='o', markersize=4)
    for i, arm in enumerate(['prior', 'descriptor_ridge']):
        mean = means[arm].mean(); ax.scatter(i, mean, color=colors[i], s=100, zorder=4)
        ax.annotate(f'{mean:.4f}', (i, mean), xytext=(12, 6), textcoords='offset points', weight='bold')
    ax.set(xticks=[0, 1], xticklabels=['No geometry\n(training prior)', 'Current geometry\n(descriptor ridge)'],
           ylabel='Corrected future-feature error (lower is better)', xlim=(-.3, 1.7),
           title='A. Can current geometry predict the future law?')
    ax.text(.02, .97, '6 historical held-out sources; 12 shots\nEach line = one source', transform=ax.transAxes, va='top')
    ax.axhline(0, color='#cbd5e1', lw=1)
    ax.set_ylim(-.006, max(means.max()) * 1.3)
    ax = axs[0, 1]
    learn = data['learning']
    for i, metric in enumerate(['exact_value_mse', 'exact_response_mse']):
        paired = learn.pivot(index='seed', columns='arm', values=metric)
        base = paired['values'].mean()
        for j, arm in enumerate(['values', 'responses']):
            yy = paired[arm] / base
            ax.bar(i * 3 + j, yy.mean(), color=colors[j], width=.7,
                   label=['Values only', 'Values + responses'][j] if i == 0 else None)
            ax.scatter(i * 3 + j + np.linspace(-.12, .12, len(yy)), yy, color='#17202a', s=18, zorder=3)
            ax.text(i * 3 + j, yy.mean() + .06, f'{yy.mean():.2f}', ha='center')
    ax.set(xticks=[.5, 3.5], xticklabels=['Future-feature error', 'Derivative error'],
           ylabel='Error / mean values-only error', ylim=(0, 1.8),
           title='B. Do response labels help the toy model?')
    ax.legend(frameon=False, loc='upper right')
    ax.text(.02, .98, '3 seeds; matched data\nNot equal cost; 250 epochs', transform=ax.transAxes, va='top')
    ax = axs[1, 0]
    cancel = data['acquisition'].query('system == "cancellation"')
    for i, rule in enumerate(['naive', 'corrected']):
        yy = cancel.query('rule == @rule').discovery_score
        ax.bar(i, yy.mean(), color=colors[i], width=.6)
        ax.scatter(i + np.linspace(-.18, .18, len(yy)), yy, color=colors[i], s=8, alpha=.4)
        ax.text(i, yy.mean() + .035, f'Mean {yy.mean():.3f}', ha='center')
    ax.axhline(0, color='#17202a', lw=1, linestyle='--', label='True law response = 0')
    ax.set(xticks=[0, 1], xticklabels=['Naive squared\nbranch response', 'Independent-branch\ncorrection'],
           ylabel='Estimated squared response', title='C. Individual paths move; the distribution may not')
    ax.legend(frameon=False, loc='upper right')
    ax.text(.02, .98, 'Analytic cancellation control; 64 repeats', transform=ax.transAxes, va='top')
    ax.set_ylim(-.2, .85)
    ax = axs[1, 1]
    physical = data['atomistic'].query('horizon_fs == 100').sort_values('parent')
    ax.scatter(physical.parent, physical.relative_mc_noise, color=colors[1], s=48)
    ax.axhline(1, color='#a34b24', linestyle='--', label='Estimated noise = |estimated signal|')
    for parent in [0, 1, 8, 12]:
        r = physical.query('parent == @parent').iloc[0]
        ax.annotate(str(parent), (parent, r.relative_mc_noise), xytext=(5, 6), textcoords='offset points')
    ax.set(xlabel='Development parent (all share an FCC prototype)', ylabel='Variance of mean / |corrected signal| (log)',
           title='D. Are four atomistic shots enough?', yscale='log', xticks=range(0, 16, 3))
    ax.legend(frameon=False, loc='upper left')
    ax.text(.02, .84, '100 fs; 4 shots; 16 states\nUnstable near zero; not a confidence bound', transform=ax.transAxes, va='top')
    fig.suptitle('What the completed response-atlas pilot established', fontsize=17)
    fig.savefig(plots / 'pilot-evidence.png', dpi=160)
    fig.savefig(plots / 'pilot-evidence.svg')
    plt.close(fig)
    # Empirical outcome examples are drawn only from training sources. Deliberate
    # spread quantiles illustrate variability, not typicality or new performance.
    from src.research.shooting_laws.common import folder, plan
    sc = read(resolve_path(c['shooting_config'])); parent_plan = plan(sc)
    cache = folder(sc)
    arrays = {name: np.load(cache / f'{name}.npy') for name in ['parent', 'strata', 'atom_ids', 'crystalline_fraction']}
    train = np.array([parent_plan['parents'][int(p)]['role'] == 'train' for p in arrays['parent']])
    candidates = np.flatnonzero(train & (arrays['strata'] == 0))
    spread = arrays['crystalline_fraction'][candidates, :, -1].var(1)
    ordered = candidates[np.argsort(spread, kind='stable')]
    examples = []
    for label, quantile in [('Low spread', 0), ('Median spread', .5), ('High spread', 1)]:
        i = int(ordered[round(quantile * (len(ordered) - 1))])
        parent = int(arrays['parent'][i])
        examples.append(dict(label=label, observation=i, parent=parent, atom_id=int(arrays['atom_ids'][i]),
                             source=parent_plan['parents'][parent]['source'],
                             fractions=arrays['crystalline_fraction'][i].round(6).tolist()))
    write_json(root / 'technical/shooting-examples.json', dict(horizons_ps=sc['horizons_ps'], examples=examples,
        selection='Training clear-liquid observations at min/median/max empirical 12-ps variance; illustrative, not representative',
        observable='Crystalline fraction of up to 80 nearest atoms within 8 angstrom around the same central atom at each future observation'))
    write_json(root / 'technical/render.json', dict(publication_only=True, producer_sha256=sha(Path(__file__)),
        csv_inputs={str(output(c) / 'analyses' / p): sha(output(c) / 'analyses' / p) for p in paths.values()},
        example_inputs={str(cache / f'{k}.npy'): sha(cache / f'{k}.npy') for k in arrays},
        original_definitions=[str(output(c) / 'analyses' / name / 'tables/METRICS.md') for name in
                              ['mechanisms-v1', 'atomistic-v1', 'shooting-reliability-v1']],
        transformations='Saved rows only: source/repeat means; toy errors divided by mean values-only error; no fitting/replacement metrics',
        examples=examples))
    print(plots / 'pilot-evidence.png')


def followup_figure(c, destination):
    root = Path(destination); plots = root / 'plots'; plots.mkdir(parents=True, exist_ok=True)
    toy_path = output(c) / 'analyses/followup-v1/tables/toy-learning.csv'
    from src.research.shooting_laws.common import result
    shooting_path = result(read(resolve_path(c['shooting_config']))) / 'analyses/comparison-v1/tables/comparison.csv'
    toy = pd.read_csv(toy_path).query('budget_kind == "total_seconds" and budget == 45')
    shooting = pd.read_csv(shooting_path).query('population == "clear_liquid" and role == "test" and metric == "path_nll" and arm != "prior"')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), layout='constrained')
    order = ['density', 'bond_order', 'topology', 'order_topology', 'mace_vicreg', 'mace_epi', 'mace_vicreg_plus', 'mace_epi_plus']
    names = ['Density', 'Bond order', 'Topology', 'Order + topology', 'MACE VICReg', 'MACE Epi', 'VICReg + descriptors', 'Epi + descriptors']
    for i, arm in enumerate(order):
        row = shooting[shooting.arm == arm].iloc[0]
        axes[0].errorbar(row.delta_from_prior, i,
            xerr=[[row.delta_from_prior - row.delta_ci_low], [row.delta_ci_high - row.delta_from_prior]],
            fmt='o', color='#007f86', capsize=3)
    axes[0].axvline(0, color='#64748b', linestyle='--', lw=1)
    axes[0].set(yticks=range(len(order)), yticklabels=names,
        xlabel='Path NLL change from prior (negative is better)',
        title='Recovered shooting comparison\n6 historical sources; source-bootstrap intervals')
    axes[0].invert_yaxis()
    colors = ['#94a3b8', '#64748b', '#007f86']
    for i, metric in enumerate(['exact_value_mse', 'exact_response_mse']):
        base = toy.query('arm == "values32"')[metric].mean()
        for j, arm in enumerate(['values8', 'values32', 'responses8']):
            yy = toy.query('arm == @arm')[metric] / base
            x = 4 * i + j
            axes[1].bar(x, yy.mean(), width=.7, color=colors[j], label=['8 value shots', '32 value shots', '8 value + response shots'][j] if i == 0 else None)
            axes[1].scatter(x + np.linspace(-.16, .16, len(yy)), yy, color='#17202a', s=15, zorder=4)
    axes[1].set(xticks=[1, 5], xticklabels=['Future-feature error', 'Derivative error'],
        ylabel='Error / mean 32-shot value-only error', ylim=(0, 3.5),
        title='New toy comparison at 45 seconds\nAcquisition + training; 5 paired seeds')
    axes[1].legend(frameon=False, loc='upper right')
    fig.savefig(plots / 'completed-followups.png', dpi=160)
    fig.savefig(plots / 'completed-followups.svg'); plt.close(fig)
    write_json(root / 'technical/followup-render.json', dict(publication_only=True,
        producer_sha256=sha(Path(__file__)), csv_inputs={str(p): sha(p) for p in [toy_path, shooting_path]},
        transformations='Saved confidence intervals unchanged; toy errors divided by mean values32 error at same budget',
        limitations='Initialization seeds share data; historical sources reused; no atomistic learning result'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--followup-config')
    args = parser.parse_args(); publish(read(args.config), args.destination)
    if args.followup_config:
        followup_figure(read(args.followup_config), args.destination)
