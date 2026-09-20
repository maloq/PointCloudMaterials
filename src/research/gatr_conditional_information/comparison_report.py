"""Paired latest-checkpoint scores and temporal stability on fixed populations."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.structural_pretraining.prepare import file_hash, save_json
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.research.trajectory_stability.metrics import reference_statistics, trajectory_metrics, stratified_draws, rms_interval
from .comparison_data import load
from .comparison_probe import feature_sets
from .metrics import matched_pairs, pair_mask, improvement
from .report import load_predictions, source_scores, pair_scores


def contrasts(rows, keys, config):
    frame = pd.DataFrame(rows)
    results = []
    for values, part in frame.groupby(keys, sort=False):
        meta = dict(zip(keys, values if isinstance(values, tuple) else (values,), strict=True))
        pairs = [('radial', 'radial_control')]
        pairs += [('radial_control', method) for method in part.method.unique()
                  if method not in ('radial', 'radial_control') and not method.startswith('current_')]
        for name in ('gatr', 'mace'):
            pairs += [(f'duplicate_{name}', f'plus_{name}'), (f'duplicate_{name}', f'delta_{name}'),
                      (f'old_{name}', f'plus_{name}')]
            for candidate in (f'current_{name}', f'current_delta_{name}'):
                pairs += [('current_order', candidate), (f'current_duplicate_{name}', candidate)]
        for baseline, method in pairs:
            if baseline not in part.method.values or method not in part.method.values:
                continue
            left = part[part.method == baseline].set_index('source')
            right = part[part.method == method].set_index('source')
            common = left.index.intersection(right.index)
            left, right = left.loc[common], right.loc[common]
            draws = stratified_draws(left.temperature_K.to_numpy(), config['bootstrap_draws'], config['seed'])
            result = improvement(left.loss.to_numpy(), right.loss.to_numpy(), draws)
            if (left.temperature_K.value_counts() == 1).all():
                result.update(low=None, high=None)
            results.append(dict(meta, baseline=baseline, method=method, sources=len(common), n=int(right.n.sum()), **result))
    return results


def spatial_predictions(root, a, sources, family, method):
    result = {key: np.full((len(a['source']), 22), np.nan) for key in ('prediction', 'mean', 'scale')}
    seen = np.zeros(len(a['source']), bool)
    for source in sources:
        sid = source['id']
        path = root/'technical/spatial-probes'/family/method/f'{sid}.npz'
        rec = json.loads(path.with_suffix('.json').read_text())
        if file_hash(path) != rec['sha256'] or sid in rec['training_sources']:
            raise ValueError('Invalid source-held-out spatial prediction')
        p = np.load(path)
        ix = p['indices']
        if seen[ix].any() or not np.all(a['source'][ix] == sid):
            raise ValueError('Spatial row correspondence failed')
        seen[ix] = True
        for key, stored in (('prediction', 'prediction'), ('mean', 'target_mean'), ('scale', 'target_scale')):
            result[key][ix] = p[stored]
    if not seen.all() or not all(np.isfinite(x).all() for x in result.values()):
        raise ValueError('Incomplete spatial predictions')
    return result


def balance(a, sources, pairs, config):
    rows = []
    for source in sources:
        belongs = a['source'][pairs['left']] == source['id']
        for caliper in config['match_calipers_A']:
            take = belongs & pair_mask(pairs, caliper, config)
            rows.append(dict(source=source['id'], temperature_K=source['temperature_K'], caliper_A=caliper,
                candidate_pairs=int(belongs.sum()), matched_pairs=int(take.sum()),
                mean_radial_rms_A=float(pairs['radial_rms_A'][take].mean()) if take.any() else None,
                mean_full_radial_rms_A=float(pairs['full_radial_rms_A'][take].mean()) if take.any() else None,
                mean_density_relative=float(pairs['density_relative'][take].mean()) if take.any() else None))
    return rows


def stability(config, a, sources):
    reference, reference_sources, _ = load(config, split='train')
    for key in ('soap', 'tda'):
        reference[key] = np.concatenate([np.load(Path(config['parent_audit'])/'technical/sources'/str(s['id'])/'observations.npz')[key]
                                        for s in reference_sources])
    lags = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    records, lag_rows = [], []
    refs = {}
    for method in ('old_gatr', 'gatr', 'old_mace', 'mace', 'soap', 'tda'):
        refs[method] = ref = reference_statistics(reference[method])
        for source in sources:
            take = a['source'] == source['id']
            z = a[method][take].reshape(801, 4, -1)
            m = trajectory_metrics(z, ref, lags)
            records.append(dict(method=method, source=source['id'], temperature_K=source['temperature_K'],
                **{k: v for k, v in m.items() if k not in ('jump2', 'lag2')}))
            for lag, value in zip(lags, m['lag2'], strict=True):
                lag_rows.append(dict(method=method, source=source['id'], temperature_K=source['temperature_K'],
                    lag_ps=lag*.75, displacement2=value))
    summary = []
    for method, part in pd.DataFrame(records).groupby('method', sort=False):
        draws = stratified_draws(part.temperature_K.to_numpy(), config['bootstrap_draws'], config['seed'])
        estimate, bounds = rms_interval(part.jump2_mean.to_numpy(), draws)
        summary.append(dict(method=method, normalized_rms_jump=estimate, low=bounds[0], high=bounds[1],
            roughness=part.roughness.mean(), reversal_fraction=part.reversal_fraction.mean(),
            increment_cosine=part.increment_cosine.mean(), effective_rank=refs[method]['effective_rank'],
            reference_variance_trace=refs[method]['trace']))
    return dict(stability_source=records, stability_summary=summary, stability_lags=lag_rows)


def report(config):
    root = Path(config['output'])
    a, sources, plan = load(config)
    spatial, _, _ = load(config, 'spatial')
    pairs = matched_pairs(a, config)
    spatial_pairs = matched_pairs(spatial, config)
    # These are exactly the previous matching populations, not a newly chosen subset.
    for caliper, expected in ((.025, 0), (.05, 609), (.1, 125006)):
        if int(pair_mask(spatial_pairs, caliper, config).sum()) != expected:
            raise ValueError('Spatial matching population changed from the previous assay')
    np.savez(root/'technical/matched-pairs.npz', **pairs)
    np.savez(root/'technical/spatial-matched-pairs.npz', **spatial_pairs)
    rows, matched, spatial_rows, spatial_matched = [], [], [], []
    for task in ('structure', 'future'):
        for family in ('linear', 'nonlinear'):
            for method in feature_sets(a, task):
                prediction = load_predictions(root, a, sources, task, family, method)
                rows.extend(source_scores(a, sources, task, family, method, prediction, config))
                matched.extend(pair_scores(a, sources, pairs, task, family, method, prediction, config))
    for family in ('linear', 'nonlinear'):
        for method in feature_sets(spatial, 'structure', spatial=True):
            prediction = spatial_predictions(root, spatial, sources, family, method)
            spatial_rows.extend(source_scores(spatial, sources, 'structure', family, method, prediction, config))
            spatial_matched.extend(pair_scores(spatial, sources, spatial_pairs, 'structure', family, method, prediction, config))
    tables = dict(source_scores=rows, matched_scores=matched, spatial_source_scores=spatial_rows,
        spatial_matched_scores=spatial_matched, matching_balance=balance(a, sources, pairs, config),
        spatial_matching_balance=balance(spatial, sources, spatial_pairs, config))
    tables.update(conditional_gains=contrasts(rows, ['task', 'family', 'target'], config),
        matched_gains=contrasts(matched, ['task', 'family', 'caliper_A', 'target'], config),
        spatial_conditional_gains=contrasts(spatial_rows, ['task', 'family', 'target'], config),
        spatial_matched_gains=contrasts(spatial_matched, ['task', 'family', 'caliper_A', 'target'], config))
    tables.update(stability(config, a, sources))
    cohort = []
    for s in sources:
        take = a['source'] == s['id']
        cohort.append(dict(source=s['id'], temperature_K=s['temperature_K'], observations=int(take.sum()),
            eligible_future_rows=int((take & a['future_eligible']).sum()),
            sustained_crystallizing_atoms=int(((a['onset_frame'] < 801) & take & (a['frame'] == 0)).sum())))
    tables['cohort'] = cohort
    snapshot_metric_docs(root, 'conditional_checkpoint_comparison')
    for name, data in tables.items():
        pd.DataFrame(data).to_csv(root/'tables'/f'{name}.csv', index=False)
    from .comparison_plots import render
    render(root, tables, config, plan)
    save_json(root/'technical/report-summary.json', dict(observations=len(a['source']), spatial_observations=len(spatial['source']),
        eligible_future_rows=int(a['future_eligible'].sum()), strict_spatial_pairs=609, strict_spatial_sources=9,
        checkpoint_steps={k: v['step'] for k, v in plan['checkpoints'].items()},
        radial_feature_dimensions=a['radial'].shape[1], common_control_dimensions=feature_sets(a, 'structure')['radial_control'].shape[1]))
    print(f'Completed paired checkpoint report: {root.resolve()}', flush=True)
