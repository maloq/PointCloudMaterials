"""Distance likelihood, controlled scan alarms and source-level uncertainty."""
import csv
import json

import numpy as np
from sklearn.metrics import average_precision_score

from src.data.fixed_cohort.protocol import write_json
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.research.local_predictability.metrics import source_weights


def csv_rows(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def bootstrap_mean(values, sources, seed, count):
    values, sources = np.asarray(values), np.asarray(sources)
    unique = np.unique(sources)
    per_source = np.array([values[sources == s].mean() for s in unique])
    if len(unique) < 2:
        return None
    rng = np.random.default_rng(seed)
    draws = rng.choice(per_source, (count, len(unique)), replace=True).mean(1)
    return np.quantile(draws, [.025, .975]).tolist()


def alarms(risk):
    # Alarm at the second observation: no future scan position is consulted.
    return np.minimum(risk[:-1], risk[1:])


def evaluate(s, variant):
    root = s.root/variant
    for folder in ('tables', 'plots'):
        (root/folder).mkdir(parents=True, exist_ok=True)
    tech = root/'technical'
    with np.load(tech/'predictions.npz') as a:
        p = {k: a[k] for k in a.files}
    with np.load(tech/'path-predictions.npz') as a:
        paths = {k: a[k] for k in a.files}
    index = json.loads((tech/'paths.json').read_text())
    risk = np.exp(p['logp'][:, :2].astype(float)).sum(1)
    scan_risk = np.exp(paths['logp'][:, :2].astype(float)).sum(1)
    calibration = []
    for i, record in enumerate(index):
        start, end = paths['offsets'][i:i+2]
        if record['role'] == 'calibration' and record['kind'] == 'away':
            calibration.append(float(alarms(scan_risk[start:end]).max()))
    if not calibration:
        raise ValueError('No complete far-control scan for threshold calibration')
    threshold = float(np.quantile(calibration, 1-s.config['false_alarm_rate'], method='higher'))
    track = 'local' if variant in ('geometry_mlp', 'mace_local') else 'context'
    path_rows = []
    for i, record in enumerate(index):
        start, end = paths['offsets'][i:i+2]
        sequence = scan_risk[start:end]
        alarm = np.flatnonzero(alarms(sequence) > threshold)+1
        first = int(alarm[0]) if len(alarm) else None
        warning = None if first is None else float(paths['distance'][start+first])
        visible = paths[f'visible_{track}'][start:end]
        ptm = paths[f'ptm_{track}'][start:end]
        path_rows.append(dict(source=record['source'], role=record['role'], frame=record['frame'],
            path_id=record['path_id'], kind=record['kind'], points=int(end-start),
            alerted=first is not None, first_alarm_step=first, warning_distance_A=warning,
            alert_without_established_crystal_in_inputs=(None if first is None else not bool(visible[first-1:first+1].any())),
            alert_without_any_PTM_crystal_in_inputs=(None if first is None else not bool(ptm[first-1:first+1].any()))))
    controls = [p for p in path_rows if p['role']=='test' and p['kind']=='away']
    toward = [p for p in path_rows if p['role']=='test' and p['kind']=='toward']
    if not toward or not controls:
        raise ValueError('Test scans require both toward paths and far away-path controls')
    warnings = np.array([np.nan if p['warning_distance_A'] is None else p['warning_distance_A'] for p in toward])
    sources = np.array([p['source'] for p in toward])
    ps = dict(threshold=threshold, score='probability nearest confirmed crystal atom is within 8 A',
              rule='strict threshold exceeded at two consecutive observations',
              calibration_controls=len(calibration), test_controls=len(controls), test_toward_paths=len(toward),
              calibration_control_false_alarm_rate=float(np.mean(np.asarray(calibration)>threshold)),
              test_control_false_alarm_rate=float(np.mean([p['alerted'] for p in controls])),
              test_no_alarm_fraction=float(np.isnan(warnings).mean()),
              conditional_median_warning_distance_A=float(np.nanmedian(warnings)) if np.isfinite(warnings).any() else None,
              early_alert_without_visible_crystal_fraction=float(np.mean([
                  p['alert_without_established_crystal_in_inputs'] is True and
                  p['warning_distance_A'] is not None and p['warning_distance_A']>8 for p in toward])))
    for distance in (8, 12, 20, 32):
        recall = np.isfinite(warnings) & (warnings >= distance)
        ps[f'recall_at_{distance}A'] = float(recall.mean())
        ps[f'source_mean_recall_at_{distance}A'] = float(np.mean([recall[sources==sid].mean() for sid in np.unique(sources)]))
        ps[f'source_mean_recall_at_{distance}A_ci95'] = bootstrap_mean(recall, sources, s.config['seed'], s.config['bootstrap'])
    metrics = dict(paths=ps)
    prior = np.asarray(json.loads((s.technical/'training-prior.json').read_text())['probabilities'])
    for role in ('train', 'selection', 'calibration', 'test'):
        ids = np.flatnonzero(p['role']==role); w = source_weights(p['source'][ids])
        nll = -p['logp'][ids, p['event'][ids]].astype(float)
        values = dict(rows=len(ids), sources=len(np.unique(p['source'][ids])), nll=float(w@nll),
                      nll_source_ci95=bootstrap_mean(nll, p['source'][ids], s.config['seed'], s.config['bootstrap']),
                      training_prior_nll=float(w@-np.log(prior[p['event'][ids]].clip(1e-12))))
        for k, distance in enumerate(s.config['distance_bins_A']):
            y = p['distance'][ids] <= distance
            score = np.exp(p['logp'][ids, :k+1].astype(float)).sum(1)
            values[f'within_{distance}A'] = dict(prevalence=float(w@y), brier=float(w@((score-y)**2)),
                average_precision=float(average_precision_score(y, score, sample_weight=w)) if y.any() else None,
                mean_probability=float(w@score))
        metrics[role] = values
    test = np.flatnonzero(p['role']=='test')
    edges = [0, 4, 8, 12, 16, 20, 24, 32, 48, np.inf]
    bins = []
    for left, right in zip(edges[:-1], edges[1:]):
        ids = test[(p['distance'][test]>=left) & (p['distance'][test]<right)]
        if not np.isfinite(right):
            ids = test[p['distance'][test]>=left]  # Includes snapshots with no confirmed crystal.
        visible = p[f'visible_{track}'][ids]
        bins.append(dict(lower_A=left, upper_A=None if np.isinf(right) else right, rows=len(ids),
            mean_risk=float(risk[ids].mean()) if len(ids) else None,
            point_alert_fraction=float((risk[ids]>threshold).mean()) if len(ids) else None,
            crystal_visible_fraction=float(visible.mean()) if len(ids) else None,
            no_visible_crystal_rows=int((~visible).sum()),
            no_visible_crystal_alert_fraction=float((risk[ids[~visible]]>threshold).mean()) if (~visible).any() else None))
    calibration_bins = []
    for lo, hi in zip(np.linspace(0,1,11)[:-1], np.linspace(0,1,11)[1:]):
        ids = test[(risk[test]>=lo) & ((risk[test]<hi) if hi<1 else (risk[test]<=hi))]
        calibration_bins.append(dict(probability_low=float(lo), probability_high=float(hi), rows=len(ids),
            predicted=float(risk[ids].mean()) if len(ids) else None,
            observed=float((p['distance'][ids]<=8).mean()) if len(ids) else None))
    snapshot_metric_docs(root, 'spatial_approach')
    csv_rows(root/'tables/path-warning.csv', path_rows)
    csv_rows(root/'tables/distance-profile.csv', bins)
    csv_rows(root/'tables/calibration.csv', calibration_bins)
    write_json(tech/'metrics.json', metrics)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6,4))
    finite = [b for b in bins if b['upper_A'] is not None and b['rows']]
    ax.plot([(b['lower_A']+b['upper_A'])/2 for b in finite], [b['mean_risk'] for b in finite], 'o-', label='Predicted proximity')
    ax.plot([(b['lower_A']+b['upper_A'])/2 for b in finite], [b['crystal_visible_fraction'] for b in finite], '--', label='Crystal present in inputs')
    ax.axhline(threshold, color='gray', ls=':', label='Path-calibrated alarm threshold')
    ax.set(xlabel='Nearest confirmed crystal atom (Å)', ylabel='Probability / fraction', title=variant)
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(root/'plots/distance-profile.png', dpi=160); plt.close(fig)
    return metrics


def collect(s):
    rows = []
    for variant in s.config['variants']:
        m = json.loads((s.root/variant/'technical/metrics.json').read_text())
        rows.append(dict(model=variant, test_distance_nll=m['test']['nll'],
                         brier_within8A=m['test']['within_8A']['brier'],
                         path_false_alarm_rate=m['paths']['test_control_false_alarm_rate'],
                         recall_at12A=m['paths']['recall_at_12A'], recall_at20A=m['paths']['recall_at_20A'],
                         conditional_median_warning_A=m['paths']['conditional_median_warning_distance_A'],
                         missed_paths=m['paths']['test_no_alarm_fraction']))
    snapshot_metric_docs(s.root, 'spatial_approach')
    csv_rows(s.root/'tables/comparison.csv', rows)
    lines = ['# Spatial approach on fixed Al snapshots', '',
             'All five predictors use the same fixed Al64 rows. The four MACE readouts share one frozen observed encoder; geometry MLP uses local descriptors only. '
             'Models are selected by validation distance likelihood, never AP or warning distance.', '',
             '| Model | Test NLL | Path false alarms | Recall at ≥12 Å | Recall at ≥20 Å | Median warning Å, detected only |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        median = 'undefined' if r['conditional_median_warning_A'] is None else f"{r['conditional_median_warning_A']:.2f}"
        lines.append(f"| {r['model']} | {r['test_distance_nll']:.4f} | {r['path_false_alarm_rate']:.3f} | {r['recall_at12A']:.3f} | {r['recall_at20A']:.3f} | {median} |")
    lines += ['', 'Distances are to atoms in confirmed components, not a fitted continuous interface. '
              'Scan routes are constructed toward known crystals; away controls stay beyond 32 Å. '
              'Their detection results are conditional diagnostics, not performance of an autonomous search policy. '
              'Pointwise predictions retain every original Al64 sample; this is the historical at-risk population, not all liquid volume.', '',
              'Each model exports distance profiles, visibility flags, calibration and per-path outcomes. '
              'Source-bootstrap intervals are in its `technical/metrics.json`. '
              'The frozen encoder was previously supervised for temporal onset; this study fits only new spatial readouts.', '']
    (s.root/'RESULTS.md').write_text('\n'.join(lines))
    write_json(s.technical/'state.json', dict(state='complete', identity=s.identity, models=len(rows)))
