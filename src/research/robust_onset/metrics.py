"""Shared onset horizons and dimensionless input perturbation diagnostics."""

import numpy as np

HORIZONS_PS = (0.75, 3.0, 6.0, 9.0, 12.0)


def horizon_index(horizon_ps):
    """Index of the actual first-onset hazard bin, in physical picoseconds."""
    if horizon_ps not in HORIZONS_PS:
        raise ValueError(f'Onset horizon must be one of {HORIZONS_PS} ps, got {horizon_ps}')
    return HORIZONS_PS.index(horizon_ps)


def local_spacing(patch):
    """Mean distance from the fixed center to its twelve nearest other atoms."""
    x = np.asarray(patch, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3 or len(x) < 13 or np.any(x[0]) or not np.isfinite(x).all():
        raise ValueError('Spacing requires a finite centered patch with twelve neighbors')
    distances = np.linalg.norm(x[1:], axis=1)
    if np.any(distances <= 0):
        raise ValueError('Coincident atom in local spacing')
    return float(np.partition(distances, 11)[:12].mean())


def perturb_patch(patch, fraction, rng):
    """Gaussian per-coordinate sigma = fraction * d12 / sqrt(3); center fixed."""
    d = local_spacing(patch)
    y = np.asarray(patch, dtype=np.float32).copy()
    y[1:] += rng.normal(size=y[1:].shape).astype(np.float32) * (fraction * d / np.sqrt(3))
    mse = float(np.square(y[1:].astype(float) - patch[1:]).sum(1).mean())
    return y, dict(spacing_A=d, input_mse_A2=mse, input_relative_mse=mse / d**2)
