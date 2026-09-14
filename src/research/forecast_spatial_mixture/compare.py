"""Collect paired completed history/spatial/mixture fits and local transition assays."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.experiment_runner.artifacts import result_folders
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.experiment_runner.registry import sha256, write_json
from src.research.forecast_crystallization.local_analyze import export_rows
from src.research.forecast_spatial_mixture.evaluate import directory
from src.project_runtime.paths import load_json


def paired_f1(candidate, baseline, repetitions, seed):
    """Average confusion counts across seeds, then bootstrap paired simulation sources."""
    candidate, baseline = np.asarray(candidate).mean(0), np.asarray(baseline).mean(0)
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(candidate), size=(repetitions, len(candidate)))
    def f1(values):
        return 2*values[..., 0]/(2*values[..., 0]+values[..., 1]+values[..., 2])
    samples = f1(candidate[indices].sum(1))-f1(baseline[indices].sum(1))
    if not np.isfinite(samples).all():
        raise ValueError('A source bootstrap sample has no positive event or positive prediction.')
    return dict(event_f1_difference=float(f1(candidate.sum(0))-f1(baseline.sum(0))),
                ci95=np.quantile(samples, [.025, .975]).tolist())


def collect(plan):
    root = result_folders(Path(plan['output']))
    embedding, events, states, fixed = [], [], [], []
    statistics, by_name, embedding_sources = {}, {}, {}
    reference_ids = None; reference_mean = reference_scale = reference_cache = None
    for run in plan['runs'] + plan.get('reference_runs', []):
        fit = Path(run['fit'])/'technical'; local = directory(plan, run)
        if json.loads((fit/'status.json').read_text())['state'] != 'complete' or json.loads((local/'status.json').read_text())['state'] != 'complete':
            raise ValueError(f'Paired collection requires completed forecast and local assay: {run}')
        payload = torch.load(fit/'best.pt', map_location='cpu', weights_only=False)
        if reference_mean is None:
            reference_mean, reference_scale = payload['mean'], payload['scale']
            reference_cache = payload['cache_manifest_sha256']
        torch.testing.assert_close(payload['mean'], reference_mean, rtol=0, atol=0)
        torch.testing.assert_close(payload['scale'], reference_scale, rtol=0, atol=0)
        if reference_cache != payload['cache_manifest_sha256']:
            raise ValueError('Paired forecasts must share the same embedding cache.')
        metrics = json.loads((fit/'test_metrics.json').read_text())
        name, seed = run['name'], run['seed']
        embedding.append(dict(method=name, seed=seed, selected_epoch=payload['epoch']+1,
            history_ps=payload['config']['history_ps'],
            spatial_neighbors=payload['variant'].get('spatial_neighbors', 0),
            parameters=json.loads((fit/'data_summary.json').read_text())['parameters'],
            **metrics['source_mean']))
        with np.load(fit/'test_errors.npz') as rows:
            ids = {k: rows[k] for k in ('source', 'atom_id', 'anchor_frame')}
            if reference_ids is None:
                reference_ids = ids
            for k in ids:
                np.testing.assert_array_equal(ids[k], reference_ids[k], err_msg=f'Unpaired test {k}: {run}')
            sources = np.unique(ids['source'])
            mse = rows['mse']
            embedding_sources[(name, seed)] = np.array([mse[ids['source'] == s].mean() for s in sources])
        report = json.loads((local/'results.json').read_text())
        if report['prediction']['checkpoint_sha256'] != sha256(fit/'best.pt'):
            raise ValueError(f'Local assay did not evaluate this selected checkpoint: {fit}')
        for score, values in report['results'].items():
            for output, key in ((events, 'onset'), (states, 'state'), (fixed, 'fixed_lead')):
                output.extend(dict(method=name, seed=seed, **r) for r in values[key])
        with np.load(local/'source-statistics.npz') as arrays:
            statistics[(name, seed)] = {k: arrays[k] for k in arrays.files}
        by_name.setdefault(name, []).append(seed)
    seed_sets = {tuple(sorted(seeds)) for seeds in by_name.values()}
    if len(seed_sets) != 1:
        raise ValueError(f'All ablations require matched completed seeds: {by_name}')
    seeds = sorted(next(iter(seed_sets)))
    pairs = plan.get('comparison_pairs', [('history12_deterministic', 'history6_deterministic'),
             ('history12_spatial', 'history12_deterministic'),
             ('history12_gaussian', 'history12_deterministic'),
             ('history12_mixture4', 'history12_gaussian'),
             ('history12_spatial_mixture4', 'history12_mixture4'),
             ('history12_spatial_mixture4', 'history12_spatial'),
             ('history12_spatial_mixture4', 'history12_deterministic')])
    paired, embedding_pairs = [], []
    for candidate, baseline in pairs:
        a = np.stack([embedding_sources[(candidate, s)] for s in seeds]).mean(0)
        b = np.stack([embedding_sources[(baseline, s)] for s in seeds]).mean(0)
        rng = np.random.default_rng(plan['bootstrap_seed'])
        indices = rng.integers(len(a), size=(plan['bootstrap_repetitions'], len(a)))
        embedding_pairs.append(dict(method=candidate, baseline=baseline, mse_difference=float((a-b).mean()),
            ci95=np.quantile((a-b)[indices].mean(1), [.025, .975]).tolist()))
        a_types = ['mean_margin']
        if 'mixture4' in candidate or 'gaussian' in candidate:
            a_types.append('frame_crystal_probability')
        for score in a_types:
            b_score = score if ('mixture4' in baseline or 'gaussian' in baseline) else 'mean_margin'
            for horizon in plan['horizons_ps']:
                for persistence in plan['persistence_frames']:
                    key_a = f'{score}_{horizon}ps_p{persistence}'
                    key_b = f'{b_score}_{horizon}ps_p{persistence}'
                    result = paired_f1([statistics[(candidate, s)][key_a] for s in seeds],
                        [statistics[(baseline, s)][key_b] for s in seeds], plan['bootstrap_repetitions'], plan['bootstrap_seed'])
                    paired.append(dict(method=candidate, baseline=baseline, score_type=score,
                        baseline_score_type=b_score, horizon_ps=horizon, persistence_frames=persistence, **result))
    report = dict(protocol=plan, embedding=embedding, onset=events, state=states, fixed_lead=fixed,
                  paired_onset=paired, paired_embedding=embedding_pairs)
    write_json(root/'technical/comparison.json', report)
    snapshot_metric_docs(root, 'forecast_spatial_mixture')
    for name, rows in [('embedding', embedding), ('local-onset', events), ('future-state', states),
                       ('fixed-lead', fixed), ('paired-onset', paired), ('paired-embedding', embedding_pairs)]:
        export_rows(root/'tables'/f'{name}.csv', rows)
    plot(root, events, list(by_name))
    (root/'README.md').write_text('# History, spatial context and trajectory mixtures\n\n'
        'Completed paired experiments: [embedding scores](tables/embedding.csv), '
        '[local transitions](tables/local-onset.csv), [fixed-lead recall](tables/fixed-lead.csv), '
        '[paired transition differences](tables/paired-onset.csv), and [definitions](tables/METRICS.md).\n\n'
        'The main physical outcome is when the tracked local center first becomes crystalline '
        'for three consecutive sampled frames. Tables also contain a nine-frame sensitivity. '
        'Previously examined test sources make this an exploratory comparison.\n\n'
        '![Local transition F1 and correctly timed recall](plots/local-transitions.png)\n\n'
        '[Open the full-size plot](plots/local-transitions.png). '
        'Bars show mean scores across fitted seeds; black dots show individual seeds.\n')
    print('Completed paired comparison:', root, flush=True)


def plot(root, events, methods):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in zip(axes, ['f1', 'timed_within_1_5_ps_recall'],
                                ['Local transition F1 at 9 ps', 'All-event recall within 1.5 ps']):
        for score, shift, color in [('mean_margin', -.17, '#326ba8'),
                                    ('frame_crystal_probability', .17, '#df8c32')]:
            for i, name in enumerate(methods):
                values = [r[metric] for r in events if r['method'] == name and r['score_type'] == score
                          and r['horizon_ps'] == 9 and r['persistence_frames'] == 3]
                if values:
                    ax.bar(i+shift, np.mean(values), width=.32, color=color)
                    ax.scatter(np.full(len(values), i+shift), values, color='black', s=12)
        ax.set(title=title, ylim=(0, 1), xticks=range(len(methods)))
        ax.set_xticklabels([m.replace('history', '').replace('_', '\n') for m in methods], fontsize=8)
        ax.grid(axis='y', alpha=.2)
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color='#326ba8', label='Mean-path readout'),
                            Patch(color='#df8c32', label='Distribution readout')], fontsize=8)
    fig.tight_layout(); fig.savefig(root/'plots/local-transitions.png', dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--stage', choices=('collect', 'summarize'), default='collect')
    args = parser.parse_args()
    if args.stage == 'summarize':
        from .summary import summarize
        summarize(load_json(args.plan))
    else:
        collect(load_json(args.plan))


if __name__ == '__main__':
    main()
