"""Independent float64 ridge solution and explicitly balanced H0/H1/H2 errors."""

import numpy as np


def ridge_predict(train_x, train_y, evaluation_x, alpha=1.0):
    """Solve centered ridge normal equations; standardization uses training only."""
    x = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    scale[scale == 0] = 1
    x = (x - mean) / scale
    y_mean = y.mean(axis=0)
    gram = x.T @ x
    gram.flat[::len(gram) + 1] += alpha
    weights = np.linalg.solve(gram, x.T @ (y - y_mean))
    return ((np.asarray(evaluation_x, dtype=np.float64) - mean) / scale) @ weights + y_mean


def balanced_errors(prediction, target, block_scales):
    """Raw descriptor error, equal block weighting despite 16/64/64 widths."""
    difference = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    if difference.ndim != 2 or difference.shape[1] != 144:
        raise ValueError(f'Expected [rows,144] descriptors, got {difference.shape}')
    block_errors = np.column_stack([
        np.mean(difference[:, left:right] ** 2, axis=1) / float(scale) ** 2
        for (left, right), scale in zip(((0, 16), (16, 80), (80, 144)), block_scales, strict=True)
    ])
    return block_errors.mean(axis=1), block_errors


def ridge_path(train_x, train_y, evaluation_x, alphas):
    """SVD ridge path for very small penalties, without squared conditioning."""
    x = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.float64)
    mean, scale = x.mean(axis=0), x.std(axis=0, ddof=0)
    scale[scale == 0] = 1
    u, singular, vt = np.linalg.svd((x-mean)/scale, full_matrices=False)
    y_mean = y.mean(axis=0)
    projected_target = u.T @ (y-y_mean)
    evaluation = ((np.asarray(evaluation_x,dtype=np.float64)-mean)/scale) @ vt.T
    return {alpha: (evaluation * (singular/(singular**2+alpha))) @ projected_target + y_mean
            for alpha in alphas}


def paired_interval(reference, candidate, sources, *, seed, draws=4000):
    """Seed-average errors first, then resample whole paired test simulations."""
    reference = np.asarray(reference).mean(axis=0)
    candidate = np.asarray(candidate).mean(axis=0)
    names = np.unique(sources)
    a = np.array([reference[sources == name].mean() for name in names])
    b = np.array([candidate[sources == name].mean() for name in names])
    ids = np.random.default_rng(seed).integers(0, len(names), size=(draws, len(names)))
    changes = 1 - b[ids].mean(axis=1) / a[ids].mean(axis=1)
    return dict(relative_reduction=float(1 - b.mean() / a.mean()),
                ci95=np.quantile(changes, [.025, .975]).tolist(), sources=names.tolist(),
                per_source_reduction=(1 - b/a).tolist())
