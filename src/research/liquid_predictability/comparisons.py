"""Matched prediction rows and paired, whole-source uncertainty calculations."""
import numpy as np


def prediction_rows(exported_ids, requested_ids):
    """Locate declared rows; duplicate or missing prediction IDs are errors."""
    ordered = np.argsort(exported_ids)
    ids = exported_ids[ordered]
    if np.any(ids[1:] == ids[:-1]) or len(np.unique(requested_ids)) != len(requested_ids):
        raise ValueError('Duplicated prediction or evaluation IDs')
    positions = np.searchsorted(ids, requested_ids)
    if np.any(positions == len(ids)) or not np.array_equal(ids[positions], requested_ids):
        raise ValueError('Unmatched evaluation rows')
    return ordered[positions]


def source_resamples(source, weights, *values, seed, draws):
    """Preserve row weights, source order and the declared seeded draw sequence."""
    _, inverse = np.unique(source, return_inverse=True)
    totals = np.stack(
        [np.bincount(inverse, weights=weights * value) for value in (np.ones(len(weights)), *values)], 1
    )
    bootstrap = np.random.default_rng(seed).integers(0, len(totals), (draws, len(totals)))
    return totals, bootstrap


def paired_scores(source, weights, mse, reference_mse, *, seed, draws,
                  rmse_comparisons, gain=None, nll_comparisons=None):
    values = [mse, reference_mse]
    if gain is not None:
        values.append(gain)
    totals, bootstrap = source_resamples(source, weights, *values, seed=seed, draws=draws)
    sampled = totals[bootstrap].sum(1)
    reduction = 1 - np.sqrt(sampled[:, 1] / sampled[:, 2])
    result = dict(
        sources=len(totals),
        rmse_reduction_fraction=float(1 - np.sqrt((weights @ mse) / (weights @ reference_mse))),
        rmse_reduction_ci95_low=float(np.quantile(reduction, .025)),
        rmse_reduction_ci95_high=float(np.quantile(reduction, .975)),
        rmse_reduction_familywise_upper=float(np.quantile(reduction, 1 - .05 / rmse_comparisons)),
    )
    if gain is not None:
        nll = sampled[:, 3] / sampled[:, 0]
        result.update(
            nll_gain=float(weights @ gain),
            nll_gain_ci95_low=float(np.quantile(nll, .025)),
            nll_gain_ci95_high=float(np.quantile(nll, .975)),
            nll_gain_familywise_low=float(np.quantile(nll, .025 / nll_comparisons)),
            nll_gain_familywise_high=float(np.quantile(nll, 1 - .025 / nll_comparisons)),
        )
    return result
