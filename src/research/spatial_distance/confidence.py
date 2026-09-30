"""Fixed probability alarms, with misses, false alarms and input visibility."""
import argparse
import json
from pathlib import Path

import numpy as np

from src.experiment_runner.artifacts import file_hash as sha, write_json
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.research.local_predictability.metrics import source_weights
from src.research.spatial_approach.evaluate import csv_rows

RADII = (4, 8, 12, 20, 32)
THRESHOLDS = (.5, .75, .95)


def read(path):
    with np.load(path) as a:
        return {k: a[k] for k in a.files}


def first_alarm(probability, threshold, consecutive):
    above = probability > threshold
    windows = np.lib.stride_tricks.sliding_window_view(above, consecutive)
    indices = np.flatnonzero(windows.all(-1))
    return int(indices[0] + consecutive - 1) if len(indices) else None


def path_metrics(model, arrays, records, risk, threshold, consecutive, support):
    rows = []
    for i, record in enumerate(records):
        if record['role'] != 'test':
            continue
        lo, hi = arrays['offsets'][i:i+2]
        first = first_alarm(risk[lo:hi], threshold, consecutive)
        if first is None:
            distance, visible, ptm = None, None, None
        else:
            distance = float(arrays['distance'][lo+first])
            take = slice(lo+first-consecutive+1, lo+first+1)
            visible = bool(arrays['visible_'+support][take].any())
            ptm = bool(arrays['ptm_'+support][take].any())
        rows.append(dict(model=model, source=record['source'], frame=record['frame'],
            path_id=record['path_id'], kind=record['kind'], alerted=first is not None,
            warning_A=distance, reference_visible_at_alarm=visible, ptm_visible_at_alarm=ptm))
    toward = [r for r in rows if r['kind'] == 'toward']
    away = [r for r in rows if r['kind'] == 'away']
    warning = np.array([np.nan if r['warning_A'] is None else r['warning_A'] for r in toward])
    detected = np.isfinite(warning)
    summary = dict(model=model, threshold=threshold, consecutive=consecutive,
        toward_paths=len(toward), detections=int(detected.sum()), misses=int((~detected).sum()),
        conditional_median_warning_A=float(np.median(warning[detected])) if detected.any() else None,
        conditional_p10_warning_A=float(np.quantile(warning[detected], .1)) if detected.any() else None,
        conditional_p90_warning_A=float(np.quantile(warning[detected], .9)) if detected.any() else None,
        away_paths=len(away), false_alarms=sum(r['alerted'] for r in away),
        away_false_alarm_rate=float(np.mean([r['alerted'] for r in away])),
        reference_visible_at_alarm=sum(r['reference_visible_at_alarm'] is True for r in toward),
        ptm_visible_at_alarm=sum(r['ptm_visible_at_alarm'] is True for r in toward),
        early_clear_reference=sum(r['reference_visible_at_alarm'] is False and r['warning_A'] > 8 for r in toward),
        early_clear_ptm=sum(r['ptm_visible_at_alarm'] is False and r['warning_A'] > 8 for r in toward))
    for radius in (8, 12, 20, 32):
        summary[f'recall_at{radius}A'] = float((warning >= radius).mean())
    return summary, rows


def point_metrics(model, population, arrays, cdf, source):
    weights = source_weights(source)
    rows = []
    for k, radius in enumerate(RADII):
        target = arrays['distance'] <= radius
        for threshold in THRESHOLDS:
            for subset, mask in [('all', np.ones(len(target), bool)),
                                 ('reference_context_clear', ~arrays['visible_context']),
                                 ('ptm_context_clear', ~arrays['ptm_context'])]:
                take = mask & (cdf[:, k] > threshold)
                mass = weights[take].sum()
                rows.append(dict(model=model, population=population, radius_A=radius,
                    threshold=threshold, subset=subset, opportunities=int(mask.sum()),
                    threshold_exceedances=int(take.sum()), source_weighted_coverage=float(mass / weights[mask].sum()) if mask.any() else None,
                    source_weighted_precision=float(weights[take] @ target[take] / mass) if mass else None,
                    source_weighted_mean_probability=float(weights[take] @ cdf[take, k] / mass) if mass else None))
    return rows


def tables(model, fixed, scans, records, fixed_cdf, scan_cdf):
    support = 'local' if model in ('geometry_mlp', 'mace_local') else 'context'
    alarms, paths = [], []
    for k, radius in enumerate(RADII):
        for threshold in THRESHOLDS:
            for consecutive in (1, 2):
                summary, rows = path_metrics(model, scans, records, scan_cdf[:, k], threshold, consecutive, support)
                alarms.append(dict(radius_A=radius, **summary))
                paths.extend(dict(radius_A=radius, threshold=threshold, consecutive=consecutive, **r) for r in rows)
    fixed_ids = np.flatnonzero(fixed['role'] == 'test')
    scan_ids, sources = [], []
    for i, r in enumerate(records):
        if r['role'] == 'test':
            ids = np.arange(scans['offsets'][i], scans['offsets'][i+1])
            scan_ids.extend(ids); sources.extend([r['source']]*len(ids))
    points = point_metrics(model, 'fixed_at_risk_test', {k: v[fixed_ids] for k,v in fixed.items()},
                          fixed_cdf[fixed_ids], fixed['source'][fixed_ids])
    points += point_metrics(model, 'controlled_scan_test',
        {k: v[scan_ids] for k,v in scans.items() if k != 'offsets'}, scan_cdf[scan_ids], np.asarray(sources))
    return alarms, paths, points


def run(root, output):
    snapshot_metric_docs(output, 'spatial_confidence')
    models = json.loads((root/'technical/launch.json').read_text())['models']
    alarms, paths, points, receipts, visibility = [], [], [], {}, []
    reference = None
    for model in models:
        folder = root/model/'technical'
        receipt = json.loads((folder/'complete.json').read_text())
        for name in ('predictions.npz', 'path-predictions.npz', 'paths.json'):
            if sha(folder/name) != receipt['files'][name]:
                raise ValueError(f'Changed input: {folder/name}')
        fixed, scans = read(folder/'predictions.npz'), read(folder/'path-predictions.npz')
        records = json.loads((folder/'paths.json').read_text())
        if reference is None:
            reference = fixed, scans, records
        else:
            for a, b in ((fixed, reference[0]), (scans, reference[1])):
                for key in a:
                    if key != 'logp':
                        np.testing.assert_array_equal(a[key], b[key])
            if records != reference[2]:
                raise ValueError('Scan records differ across models')
        pcdf = np.exp(fixed['logp'].astype(float)).cumsum(-1)[:, :5]
        scdf = np.exp(scans['logp'].astype(float)).cumsum(-1)[:, :5]
        a, b, c = tables(model, fixed, scans, records, pcdf, scdf)
        alarms.extend(a); paths.extend(b); points.extend(c); receipts[model] = receipt
    for field in ('visible_local', 'visible_context', 'ptm_local', 'ptm_context'):
        for consecutive in (1, 2):
            row, _ = path_metrics('label_oracle_'+field, scans, records, scans[field].astype(float),
                                  .5, consecutive, 'local' if field.endswith('local') else 'context')
            visibility.append(row)
    for name, rows in [('alarms', alarms), ('paths', paths), ('confidence-reliability', points), ('visibility-only', visibility)]:
        csv_rows(output/'tables'/f'{name}.csv', rows)
    write_json(output/'technical/inputs.json', dict(run=str(root), receipts=receipts,
        probability='P(distance to confirmed reference crystal <= radius_A)', fitting=False))
    print(json.dumps(dict(output=str(output), models=len(models), alarm_rows=len(alarms))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.run, args.output)
