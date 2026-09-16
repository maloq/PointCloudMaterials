"""Validation-calibrated finite-horizon local alarms, including missed events."""
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .objective import cumulative_risk


def population(samples, k):
    event = np.array([s['event_bin'] for s in samples])
    observed = np.array([s['observed_bins'] for s in samples])
    eligible = np.array([s['at_risk'] for s in samples])
    happened = (event >= 0) & (event <= k)
    known = eligible & (happened | (observed > k))
    return happened, known


def source_weights(samples, known):
    ids = np.array([s['source_id'] for s in samples])[known]
    unique, count = np.unique(ids, return_counts=True)
    return np.array([1/count[np.searchsorted(unique, sid)] for sid in ids], dtype=float)


def thresholds_from_validation(logits, samples, maximum_false_alarm_rate):
    if not 0 <= maximum_false_alarm_rate < 1:
        raise ValueError('False-alarm limit must be in [0,1)')
    if any(s['split'] != 'val' for s in samples):
        raise ValueError('Alarm thresholds require validation sources only')
    risk = cumulative_risk(torch.as_tensor(logits)).numpy()
    thresholds = []
    for k in range(risk.shape[1]):
        happened, known = population(samples, k)
        y, score = happened[known], risk[known, k]
        weights = source_weights(samples, known)
        if not y.any() or y.all():
            thresholds.append(None)  # No defensible two-class operating point.
            continue
        above_max = np.nextafter(score.max(), np.array(np.inf, dtype=score.dtype))
        candidates = np.r_[above_max, np.unique(score)[::-1]]
        allowed = [t for t in candidates if weights[(score >= t) & ~y].sum()/weights[~y].sum()
                   <= maximum_false_alarm_rate+1e-12]
        thresholds.append(float(min(allowed)))
    return thresholds


def event_report(logits, samples, bin_edges_ps, thresholds):
    risk = cumulative_risk(torch.as_tensor(logits)).numpy()
    edges = np.asarray(bin_edges_ps)
    rows = []
    for k, edge in enumerate(edges):
        happened, known = population(samples, k)
        y, score = happened[known], risk[known, k]
        w = source_weights(samples, known)
        row = dict(horizon_ps=float(edge), known_examples=int(known.sum()), events=int(y.sum()),
                   threshold=thresholds[k], average_precision=None, auroc=None,
                   brier=None, mean_risk=None, observed_frequency=None,
                   precision=None, recall=None, false_alarm_rate=None,
                   detected_events=None, missed_events=None, interval_timing_mae_ps=None,
                   timed_within_first_bin_recall=None)
        if len(y):
            row.update(brier=float(np.average((score-y)**2, weights=w)),
                       mean_risk=float(np.average(score, weights=w)), observed_frequency=float(np.average(y, weights=w)))
        if y.any():
            row['average_precision'] = float(average_precision_score(y, score, sample_weight=w))
        if y.any() and not y.all():
            row['auroc'] = float(roc_auc_score(y, score, sample_weight=w))
        if thresholds[k] is not None and len(y):
            alarm = score >= thresholds[k]
            hit = alarm & y
            row.update(detected_events=int(hit.sum()), missed_events=int((y & ~alarm).sum()))
            for name, numerator, denominator in [('precision', w[hit].sum(), w[alarm].sum()),
                ('recall', w[hit].sum(), w[y].sum()), ('false_alarm_rate', w[alarm & ~y].sum(), w[~y].sum())]:
                row[name] = float(numerator/denominator) if denominator else None
            # Labels identify a bin, not a continuous event time. Penalize distance
            # from that interval instead of inventing sub-cadence ground truth.
            pred_bin = np.argmax(risk[known, :k+1] >= thresholds[k], axis=1)
            actual_bin = np.array([s['event_bin'] for s in samples])[known]
            prediction = edges[pred_bin]
            right = edges[np.maximum(actual_bin, 0)]
            left = np.r_[0., edges[:-1]][np.maximum(actual_bin, 0)]
            distance = np.maximum(np.maximum(left-prediction, prediction-right), 0.)
            if hit.any():
                row['interval_timing_mae_ps'] = float(np.average(distance[hit], weights=w[hit]))
            if y.any():
                row['timed_within_first_bin_recall'] = float(w[hit & (distance <= edges[0])].sum()/w[y].sum())
        rows.append(row)
    return rows
