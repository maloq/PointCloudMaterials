"""Compare separately fitted history lengths on identical future windows."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from src.experiment_runner.artifacts import result_folders
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.experiment_runner.registry import sha256, write_json
from src.training_methods.embedding_forecast.metrics import source_bootstrap
from src.training_methods.embedding_forecast.run import forecast_directory


def source_means(values, sources):
    """Average windows within each source before averaging sources or fitted seeds."""
    return np.stack([values[sources == source].mean(axis=0, dtype=np.float64)
                     for source in np.unique(sources)])


def summarize(groups, seed, repetitions=2000):
    """Each group's measures have axes (fit seed, independent source, optional time)."""
    rows, paired = [], {}
    for group in groups:
        architecture, history = group['architecture'], group['history_ps']
        measures = {k: np.mean(v, axis=0) for k, v in group['measures'].items()}
        anchor = next(g for g in groups if g['architecture'] == architecture and g['history_ps'] == 0)
        gain = source_bootstrap(measures['mse'], np.mean(anchor['measures']['mse'], axis=0),
                                group['source_ids'], seed, repetitions)
        history_gain = source_bootstrap(measures['mse'], measures['history_mean_mse'],
                                        group['source_ids'], seed, repetitions)
        interval = gain['ci95']
        mse_seeds = np.mean(group['measures']['mse'], axis=1)
        bins = measures['bin_mse'].mean(0)
        row = dict(architecture=architecture, history_ps=history, history_frames=group['history_frames'],
                   parameters=group['parameters'], seeds=len(mse_seeds), sources=len(group['source_ids']),
                   windows=group['windows'], mse=float(mse_seeds.mean()),
                   seed_std=float(mse_seeds.std()) if len(mse_seeds) > 1 else None,
                   raw_mse=float(measures['raw_mse'].mean()), bin_0_3_mse=float(bins[0]),
                   bin_3_6_mse=float(bins[1]), bin_6_9_mse=float(bins[2]),
                   mse_0p75ps=float(measures['mse_by_step'][:, 0].mean()),
                   mse_9ps=float(measures['mse_by_step'][:, -1].mean()),
                   increment_mse=float(measures['increment_mse'].mean()),
                   persistence_mse=float(measures['persistence_mse'].mean()),
                   history_mean_mse=float(measures['history_mean_mse'].mean()),
                   reverse_past_mse=float(measures['reverse_past_mse'].mean()),
                   repeat_anchor_mse=float(measures['repeat_anchor_mse'].mean()),
                   gain_vs_anchor=gain['gain'], gain_vs_anchor_ci95_lower=interval[0] if interval else None,
                   gain_vs_anchor_ci95_upper=interval[1] if interval else None,
                   gain_vs_history_mean=history_gain['gain'])
        rows.append(row)
        paired[f'{architecture}_history{history}'] = dict(vs_anchor=gain, vs_history_mean=history_gain)
    return rows, paired


def load_groups(plan):
    groups, provenance = [], []
    reference_ids = reference_checkpoint = None
    reference_protocol = None
    for entry in plan['configs']:
        config = json.loads(Path(entry['path']).read_text())
        protocol = {k: config[k] for k in ('data', 'anchor_history_ps', 'stride_ps', 'horizons_ps', 'training', 'seeds')}
        if reference_protocol is None:
            reference_protocol = protocol
        if protocol != reference_protocol or config['history_ps'] != entry['history_ps']:
            raise ValueError(f'Context comparison changed the matched scientific protocol: {entry["path"]}')
        for variant in config['variants']:
            measures = {}
            for seed in plan['seeds']:
                directory = forecast_directory(Path(config['output']) / f'{variant["name"]}-seed{seed}')
                if json.loads((directory / 'status.json').read_text())['state'] != 'complete':
                    raise ValueError(f'Context fit is incomplete: {directory}')
                if json.loads((directory / 'config.json').read_text()) != dict(config=config, variant=variant, seed=seed):
                    raise ValueError(f'Context fit configuration differs from the plan: {directory}')
                checkpoint = torch.load(directory / 'best.pt', map_location='cpu', weights_only=False)
                if reference_checkpoint is None:
                    reference_checkpoint = {k: checkpoint[k] for k in ('mean', 'scale', 'cache_manifest_sha256', 'implementation_sha256')}
                for key in ('mean', 'scale'):
                    if not torch.equal(checkpoint[key], reference_checkpoint[key]):
                        raise ValueError(f'Context comparisons require the same training normalization: {directory}: {key}')
                for key in ('cache_manifest_sha256', 'implementation_sha256'):
                    if checkpoint[key] != reference_checkpoint[key]:
                        raise ValueError(f'Context comparison identity differs: {directory}: {key}')
                with np.load(directory / 'test_errors.npz') as archive:
                    ids = {k: archive[k] for k in ('source', 'atom_id', 'anchor_frame', 'temperature_K')}
                    if reference_ids is None:
                        reference_ids = ids
                    for key in ids:
                        np.testing.assert_array_equal(ids[key], reference_ids[key], err_msg=f'Unpaired {key}: {directory}')
                    for key in ('mse', 'raw_mse', 'mse_by_step', 'bin_mse', 'increment_mse', 'persistence_mse', 'history_mean_mse'):
                        measures.setdefault(key, []).append(source_means(archive[key], ids['source']))
                source_ids = np.unique(ids['source'])
                interventions = json.loads((directory / 'history_interventions.json').read_text())
                for name in ('reverse_past', 'repeat_anchor'):
                    values = np.array([interventions[name]['per_source'][str(s)]['mse'] for s in source_ids])
                    measures.setdefault(name + '_mse', []).append(values)
                summary = json.loads((directory / 'data_summary.json').read_text())
                provenance.append(dict(directory=str(directory), best_sha256=sha256(directory / 'best.pt'),
                                       selected_epoch=checkpoint['epoch'], seed=seed))
            groups.append(dict(architecture=variant['architecture'], history_ps=config['history_ps'],
                               history_frames=summary['history_steps'], parameters=summary['parameters'],
                               windows=len(ids['source']), source_ids=source_ids,
                               measures={k: np.stack(v) for k, v in measures.items()}))
    for architecture in {g['architecture'] for g in groups}:
        if len({g['parameters'] for g in groups if g['architecture'] == architecture}) != 1:
            raise ValueError(f'History comparison changes parameter count within {architecture}.')
    return groups, provenance


def plot_comparison(groups, rows, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = dict(autoregressive_gru='Autoregressive', mean_residual_gru='Direct')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for architecture, label in labels.items():
        values = sorted((r for r in rows if r['architecture'] == architecture), key=lambda r: r['history_ps'])
        x = [r['history_ps'] for r in values]
        axes[0].errorbar(x, [r['mse'] for r in values], yerr=[r['seed_std'] for r in values], marker='o', label=label)
        axes[1].plot(x, [r['mse_9ps'] for r in values], marker='o', label=label)
        axes[2].plot(x, [100 * r['gain_vs_anchor'] for r in values], marker='o', label=label)
    for axis, ylabel in zip(axes, ('Full-path standardized MSE', '+9 ps standardized MSE', 'Gain over trained anchor / %')):
        axis.set(xlabel='Observed history / ps', ylabel=ylabel)
        axis.grid(alpha=.2)
        axis.legend()
    fig.tight_layout()
    fig.savefig(root / 'plots/context-quality.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for axis, (architecture, label) in zip(axes, labels.items()):
        for group in groups:
            if group['architecture'] == architecture:
                curve = group['measures']['mse_by_step'].mean(axis=(0, 1))
                axis.plot(np.arange(1, 13) * .75, curve, label=f'{group["history_ps"]:g} ps')
        axis.set(title=label, xlabel='Future time / ps', ylabel='Standardized MSE')
        axis.legend(title='History')
    fig.tight_layout()
    fig.savefig(root / 'plots/horizon-errors.png', dpi=160)
    plt.close(fig)


def compare(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    root = result_folders(plan['output'])
    groups, provenance = load_groups(plan)
    rows, paired = summarize(groups, seed=20260913)
    snapshot_metric_docs(root, 'forecast_context')
    with (root / 'tables/context-quality.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    curves = {f'{g["architecture"]}_history{g["history_ps"]}': {
        k: g['measures'][k].mean(axis=(0, 1)).tolist() for k in ('mse_by_step', 'bin_mse')}
        for g in groups}
    write_json(root / 'technical/comparison.json', dict(rows=rows, paired=paired, curves=curves, provenance=provenance))
    plot_comparison(groups, rows, root)
    lines = ['# Observed-history comparison', '',
             'Matched 9 ps forecasts; identical sources, anchors, targets, normalization and per-fit update budgets.', '',
             '| Model | History / ps | Path MSE | +9 ps MSE | Gain vs trained anchor |',
             '| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f'| {row["architecture"]} | {row["history_ps"]} | {row["mse"]:.6f} | '
                     f'{row["mse_9ps"]:.6f} | {100 * row["gain_vs_anchor"]:.2f}% |')
    lines.extend(['', '[Full metrics](tables/context-quality.csv) · [Definitions](tables/METRICS.md) · '
                  '[Context plot](plots/context-quality.png) · [Horizon plot](plots/horizon-errors.png)', '',
                  'Two fitted seeds and 27 previously examined test sources: exploratory evidence. '
                  'Anchor gains include averaging/denoising benefits; compare the history-mean baseline and '
                  'reversal intervention before attributing gains to temporal ordering. Compact-model results '
                  'do not establish the optimal history for the larger production models.', ''])
    (root / 'RESULTS.md').write_text('\n'.join(lines))
    write_json(root / 'technical/comparison_status.json', dict(state='complete', fits=len(provenance)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    compare(args.plan)


if __name__ == '__main__':
    main()
