"""Post-hoc analysis of frozen spatial predictions; no fitting or model selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.research.local_predictability.metrics import source_weights
from .evaluate import alarms, csv_rows


DISTANCES = (4, 8, 12, 20, 32)
SEED = 20260926


def read_npz(path):
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def interval(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return [None, None]
    rng = np.random.default_rng(SEED)
    draws = rng.integers(len(values), size=(5000, len(values)))
    return np.quantile(values[draws].mean(1), [.025, .975]).tolist()


def source_means(values, sources):
    return np.array([values[sources == sid].mean() for sid in np.unique(sources)])


def run(root, output):
    output.mkdir(parents=True, exist_ok=True)
    for part in ('tables', 'technical', 'plots'):
        (output / part).mkdir(exist_ok=True)
    models = json.loads((root / 'technical/launch.json').read_text())['models']
    original, scans, indices, receipts = {}, {}, {}, {}
    for model in models:
        folder = root / model / 'technical'
        receipt = json.loads((folder / 'complete.json').read_text())
        for name in ('predictions.npz', 'path-predictions.npz', 'paths.json', 'metrics.json'):
            actual = hashlib.sha256((folder / name).read_bytes()).hexdigest()
            if actual != receipt['files'][name]:
                raise ValueError(f'Changed frozen artifact: {folder / name}')
        receipts[model] = receipt
        original[model] = read_npz(folder / 'predictions.npz')
        scans[model] = read_npz(folder / 'path-predictions.npz')
        indices[model] = json.loads((folder / 'paths.json').read_text())
    reference = models[0]
    for model in models[1:]:
        for key in original[reference]:
            if key != 'logp':
                np.testing.assert_array_equal(original[reference][key], original[model][key])
        for key in scans[reference]:
            if key != 'logp':
                np.testing.assert_array_equal(scans[reference][key], scans[model][key])
        if indices[model] != indices[reference]:
            raise ValueError('Models have different scan records')

    point_rows, subset_rows, paired_rows, alarm_rows, per_path = [], [], [], [], []
    losses, profiles = {}, {}
    for model in models:
        p, a, records = original[model], scans[model], indices[model]
        ids = np.flatnonzero(p['role'] == 'test')
        loss = -p['logp'][ids, p['event'][ids]].astype(float)
        losses[model] = source_means(loss, p['source'][ids])
        low, high = interval(losses[model])
        metric = json.loads((root / model / 'technical/metrics.json').read_text())
        point_rows.append(dict(model=model, nll=losses[model].mean(), nll_ci_low=low, nll_ci_high=high,
            brier_within8A=metric['test']['within_8A']['brier'],
            ap_within8A=metric['test']['within_8A']['average_precision']))
        for name, mask in [('context_visible', p['visible_context']),
                           ('context_clear', ~p['visible_context']),
                           ('all_PTM_crystal_clear_context', ~p['ptm_context'])]:
            select = np.flatnonzero((p['role'] == 'test') & mask)
            subset_loss = -p['logp'][select, p['event'][select]].astype(float)
            subset_rows.append(dict(model=model, subset=name, rows=len(select),
                sources=len(np.unique(p['source'][select])), near8_rows=int((p['distance'][select] <= 8).sum()),
                source_mean_nll=source_means(subset_loss, p['source'][select]).mean()))
        track = 'local' if model in ('geometry_mlp', 'mace_local') else 'context'
        profiles[model] = {}
        for k, radius in enumerate(DISTANCES):
            risk = np.exp(a['logp'][:, :k+1].astype(float)).sum(1)
            calibration = []
            for j, record in enumerate(records):
                lo, hi = a['offsets'][j:j+2]
                if record['role'] == 'calibration' and record['kind'] == 'away':
                    calibration.append(alarms(risk[lo:hi]).max())
            threshold = float(np.quantile(calibration, .95, method='higher'))
            if radius == 8:
                np.testing.assert_allclose(threshold, metric['paths']['threshold'], rtol=0, atol=1e-12)
            rows = []
            for j, record in enumerate(records):
                if record['role'] != 'test':
                    continue
                lo, hi = a['offsets'][j:j+2]
                hits = np.flatnonzero(alarms(risk[lo:hi]) > threshold) + 1
                first = int(hits[0]) if len(hits) else None
                observed = a['visible_' + track][lo:hi]
                visible_ids = np.flatnonzero(observed)
                row = dict(model=model, score_radius_A=radius, source=record['source'],
                    frame=record['frame'], path_id=record['path_id'], kind=record['kind'],
                    alerted=first is not None,
                    warning_A=None if first is None else float(a['distance'][lo+first]),
                    no_reference_crystal=None if first is None else not bool(observed[first-1:first+1].any()),
                    no_PTM_crystal=None if first is None else not bool(a['ptm_'+track][lo+first-1:lo+first+1].any()),
                    first_reference_visible_A=(float(a['distance'][lo+visible_ids[0]]) if len(visible_ids) else None))
                rows.append(row)
            per_path.extend(rows)
            toward = [r for r in rows if r['kind'] == 'toward']
            away = [r for r in rows if r['kind'] == 'away']
            warnings = np.array([np.nan if r['warning_A'] is None else r['warning_A'] for r in toward])
            sources = np.array([r['source'] for r in toward])
            fpr_sources = np.array([r['source'] for r in away])
            fpr_values = np.array([r['alerted'] for r in away])
            source_fpr = source_means(fpr_values, fpr_sources)
            fpr_low, fpr_high = interval(source_fpr)
            summary = dict(model=model, score_radius_A=radius, threshold=threshold,
                calibration_false_alarms=int((np.asarray(calibration) > threshold).sum()),
                calibration_controls=len(calibration), toward_paths=len(toward),
                toward_sources=len(np.unique(sources)), test_controls=len(away),
                test_control_sources=len(np.unique(fpr_sources)),
                false_alarms=int(fpr_values.sum()), false_alarm_rate=fpr_values.mean(),
                source_mean_false_alarm_rate=source_fpr.mean(), source_fpr_ci_low=fpr_low, source_fpr_ci_high=fpr_high,
                misses=int(np.isnan(warnings).sum()), missed_fraction=np.isnan(warnings).mean(),
                conditional_median_warning_A=float(np.nanmedian(warnings)),
                early_no_reference_crystal=sum(r['no_reference_crystal'] is True and r['warning_A'] > 8 for r in toward),
                early_no_PTM_crystal=sum(r['no_PTM_crystal'] is True and r['warning_A'] > 8 for r in toward),
                reference_visible_at_alarm=sum(r['no_reference_crystal'] is False for r in toward),
                median_first_reference_visible_A=float(np.median([r['first_reference_visible_A'] for r in toward])))
            for distance in (8, 12, 20, 32):
                detected = warnings >= distance
                source_recall = source_means(detected, sources)
                lo, hi = interval(source_recall)
                summary.update({f'recall_at{distance}A': detected.mean(), f'source_recall_at{distance}A': source_recall.mean(),
                                f'source_recall_at{distance}A_ci_low': lo, f'source_recall_at{distance}A_ci_high': hi})
            if radius == 8:
                np.testing.assert_allclose(summary['missed_fraction'], metric['paths']['test_no_alarm_fraction'])
                np.testing.assert_allclose(summary['false_alarm_rate'], metric['paths']['test_control_false_alarm_rate'])
            alarm_rows.append(summary)
            profiles[model][radius] = (np.linspace(0, 36, 181), np.array([(warnings >= d).mean() for d in np.linspace(0, 36, 181)]))

    for left, right in [('mace_local', 'geometry_mlp'), ('symmetric_invariant', 'mace_local'),
                        ('vector_messages', 'symmetric_invariant'), ('harmonic_hierarchy', 'symmetric_invariant'),
                        ('vector_messages', 'harmonic_hierarchy')]:
        difference = losses[left] - losses[right]
        lo, hi = interval(difference)
        paired_rows.append(dict(model=left, reference=right, source_mean_nll_delta=difference.mean(),
            ci_low=lo, ci_high=hi, improved_sources=int((difference < 0).sum()), sources=len(difference)))

    # Same test paths, including endpoint atoms absent from the original at-risk population.
    scan_support, scan_calibration = [], []
    for model in models:
        a, records = scans[model], indices[model]
        ids = np.concatenate([np.arange(a['offsets'][i], a['offsets'][i+1]) for i, r in enumerate(records)
                              if r['role'] == 'test' and r['kind'] == 'toward'])
        for label in ('visible_local', 'visible_context', 'ptm_local', 'ptm_context'):
            clear = ~a[label][ids]
            scan_support.append(dict(model=model, visibility=label, points=len(ids), clear_points=int(clear.sum()),
                clear_points_between8and32A=int((clear & (a['distance'][ids] > 8) & (a['distance'][ids] <= 32)).sum())))
        for name, p, take in [('fixed_test', original[model], np.flatnonzero(original[model]['role'] == 'test')),
                              ('toward_scan_test', a, ids)]:
            risk = np.exp(p['logp'][:, :2].astype(float)).sum(1)
            for lower, upper in ((0, 4), (4, 8), (8, 12), (12, 20), (20, 32), (32, np.inf)):
                ii = take[(p['distance'][take] >= lower) & (p['distance'][take] < upper)]
                if not len(ii):
                    continue
                scan_calibration.append(dict(model=model, population=name, lower_A=lower,
                    upper_A=None if np.isinf(upper) else upper, rows=len(ii),
                    mean_probability_within8A=float(risk[ii].mean()),
                    fraction_within8A=float((p['distance'][ii] <= 8).mean())))
    for name, rows in [('pointwise', point_rows), ('paired-nll', paired_rows), ('visibility-subsets', subset_rows),
                       ('alarm-radius', alarm_rows), ('paths', per_path), ('scan-support', scan_support),
                       ('fixed-versus-scan', scan_calibration)]:
        csv_rows(output / 'tables' / f'{name}.csv', rows)
    snapshot_metric_docs(output, 'spatial_approach_review')
    (output / 'technical/inputs.json').write_text(json.dumps(dict(run=str(root), receipts=receipts,
        seed=SEED, source_bootstrap_draws=5000, fitting=False, model_selection=False), indent=2) + '\n')
    plot(output, point_rows, alarm_rows, profiles)
    print(json.dumps(dict(output=str(output), models=len(models), paired_nll=paired_rows), indent=2))


def plot(output, points, alarms_table, profiles):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['geometry_mlp', 'mace_local', 'symmetric_invariant', 'vector_messages', 'harmonic_hierarchy']
    labels = ['Geometry', 'Local MACE', 'Symmetric', 'Vector', 'Harmonic']
    colors = ['#999999', '#6746a3', '#de8e21', '#1b7892', '#3a9454']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    p = {row['model']: row for row in points}
    values = np.array([p[n]['nll'] for n in names])
    axes[0].barh(labels, values, color=colors)
    axes[0].errorbar(values, np.arange(5), xerr=[values-[p[n]['nll_ci_low'] for n in names],
        np.array([p[n]['nll_ci_high'] for n in names])-values], fmt='none', color='black', capsize=3)
    axes[0].invert_yaxis(); axes[0].set(xlabel='Test distance NLL (lower is better)', title='Matched fixed observations')
    for ax, radius in zip(axes[1:], (8, 20)):
        for name, label, color in zip(names, labels, colors):
            x, y = profiles[name][radius]
            row = next(r for r in alarms_table if r['model'] == name and r['score_radius_A'] == radius)
            ax.plot(x, y*100, color=color, label=f'{label} (FA {row["false_alarm_rate"]*100:.1f}%)')
        ax.set(xlabel='Warning distance ≥ D (Å)', ylabel='Detected paths / all approach paths (%)',
               title=f'Alarm: P(distance ≤ {radius} Å)' + (' — exploratory' if radius == 20 else ' — original'),
               ylim=(0, 100), xlim=(0, 36))
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Fixed Al snapshots: spatial warning, one seed; thresholds set on calibration paths', fontsize=12)
    fig.tight_layout(); fig.savefig(output / 'plots/comparison.png', dpi=180)
    fig.savefig(output / 'plots/comparison.svg'); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.run.resolve(), args.output.resolve())
