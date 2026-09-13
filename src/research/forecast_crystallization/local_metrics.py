"""Local crystal transition metrics, validation thresholds and source bootstrap."""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def first_sustained_onset(crystal, persistence):
    """First frame of a fully observed crystal run; T denotes right censoring."""
    windows = np.lib.stride_tricks.sliding_window_view(crystal, persistence, axis=-1)
    valid = windows.all(axis=-1)
    return np.where(valid.any(axis=-1), valid.argmax(axis=-1), crystal.shape[-1])


def risk_windows(crystal, onset, anchors, negative_history_frames):
    recent = crystal[..., anchors[:, None] - np.arange(negative_history_frames)]
    return (anchors < onset[..., None]) & ~recent.any(axis=-1)


def select_threshold(actual, score):
    """Maximize validation F1; select the highest threshold in an exact tie."""
    if not actual.any() or actual.all():
        raise ValueError('Threshold selection requires both positive and negative validation examples.')
    precision, recall, thresholds = precision_recall_curve(actual, score)
    f1 = 2*precision[:-1]*recall[:-1] / np.maximum(precision[:-1]+recall[:-1], 1e-15)
    index = np.flatnonzero(f1 == f1.max())[-1]
    return float(thresholds[index]), float(f1[index])


def ratio(numerator, denominator):
    return None if denominator == 0 else float(numerator / denominator)


def counts(actual, predicted):
    return np.array([np.sum(actual & predicted), np.sum(~actual & predicted),
                     np.sum(actual & ~predicted), np.sum(~actual & ~predicted)], dtype=np.int64)


def from_counts(values):
    tp, fp, fn, tn = values
    recall, specificity = ratio(tp, tp+fn), ratio(tn, tn+fp)
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        n=int(values.sum()), positive=int(tp+fn), prevalence=ratio(tp+fn, values.sum()),
        accuracy=ratio(tp+tn, values.sum()), precision=ratio(tp, tp+fp), recall=recall,
        specificity=specificity, false_positive_rate=ratio(fp, fp+tn),
        balanced_accuracy=None if recall is None or specificity is None else (recall+specificity)/2,
        f1=ratio(2*tp, 2*tp+fp+fn))


def classification(actual, score, threshold):
    predicted = score >= threshold
    result = from_counts(counts(actual, predicted))
    both_classes = actual.any() and not actual.all()
    result.update(auroc=float(roc_auc_score(actual, score)) if both_classes else None,
                  average_precision=float(average_precision_score(actual, score)) if actual.any() else None)
    return result


def source_bootstrap(per_source, repetitions, seed):
    """Resample independent simulation sources, retaining all their centers/windows."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(per_source), size=(repetitions, len(per_source)))
    summed = np.asarray(per_source)[indices].sum(axis=1)
    tp, fp, fn, tn = summed[:, :4].T
    numerators = dict(accuracy=tp+tn, precision=tp, recall=tp, f1=2*tp, false_positive_rate=fp)
    denominators = dict(accuracy=tp+fp+fn+tn, precision=tp+fp, recall=tp+fn, f1=2*tp+fp+fn,
                        false_positive_rate=fp+tn)
    if summed.shape[1] == 7:
        numerators.update(timing_mae_ps=summed[:, 4], timing_bias_ps=summed[:, 5],
                          timed_within_1_5_ps_recall=summed[:, 6])
        denominators.update(timing_mae_ps=tp, timing_bias_ps=tp, timed_within_1_5_ps_recall=tp+fn)
    result = {}
    for key, numerator in numerators.items():
        valid = denominators[key] > 0
        result[key] = dict(ci95=np.quantile(numerator[valid]/denominators[key][valid], [.025, .975]).tolist()
                          if valid.any() else None, valid_repetitions=int(valid.sum()))
    return result


def onset_metrics(actual, score_path, threshold, actual_delay_ps, cadence_ps, source_ids, source_universe, bootstrap, seed):
    """Measure timing only for detected events and separately count all missed events."""
    score = score_path.max(axis=-1)
    predicted = score >= threshold
    predicted_delay = (np.argmax(score_path >= threshold, axis=-1)+1)*cadence_ps
    true_positive = actual & predicted
    errors = predicted_delay - actual_delay_ps
    detected_errors = errors[true_positive]
    result = classification(actual, score, threshold)
    result.update(timing_mae_ps=float(np.abs(detected_errors).mean()) if len(detected_errors) else None,
        timing_bias_ps=float(detected_errors.mean()) if len(detected_errors) else None,
        timing_median_absolute_error_ps=float(np.median(np.abs(detected_errors))) if len(detected_errors) else None,
        timing_p90_absolute_error_ps=float(np.quantile(np.abs(detected_errors), .9)) if len(detected_errors) else None,
        timing_within_0_75_ps=ratio(np.sum(np.abs(detected_errors) <= .75), len(detected_errors)),
        timing_within_1_5_ps=ratio(np.sum(np.abs(detected_errors) <= 1.5), len(detected_errors)),
        timing_within_3_ps=ratio(np.sum(np.abs(detected_errors) <= 3), len(detected_errors)),
        timed_within_1_5_ps_recall=ratio(np.sum(np.abs(detected_errors) <= 1.5), actual.sum()))
    per_source = []
    for source in source_universe:
        mask = source_ids == source
        hit = mask & true_positive
        per_source.append([*counts(actual[mask], predicted[mask]),
                           np.abs(errors[hit]).sum(), errors[hit].sum(),
                           np.sum(np.abs(errors[hit]) <= 1.5)])
    result['source_bootstrap'] = source_bootstrap(per_source, bootstrap, seed)
    return result, np.asarray(per_source)
