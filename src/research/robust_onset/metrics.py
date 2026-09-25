"""Dimensionless input perturbations and a source-weighted Smooth-AP adaptation."""
import numpy as np
import torch

HORIZONS_PS = (.75, 3., 6., 9., 12.)


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
    y[1:] += rng.normal(size=y[1:].shape).astype(np.float32) * (fraction*d/np.sqrt(3))
    mse = float(np.square(y[1:].astype(float)-patch[1:]).sum(1).mean())
    return y, dict(spacing_A=d, input_mse_A2=mse, input_relative_mse=mse/d**2)


def smooth_ap(score, positive, weights, temperature):
    """Weighted empirical AP with sigmoid ranks; exact self mass, no subsampling.

    Applied to the entire fitting at-risk population. This is an adaptation of
    Brown et al. (2020), not the SOAP optimizer and not an AUROC pairwise loss.
    Distinct scores converge to sklearn's weighted AP as temperature -> 0.
    """
    if score.ndim != 1 or score.shape != positive.shape or weights.shape != score.shape:
        raise ValueError('AP scores, labels and weights must be aligned vectors')
    if positive.dtype != torch.bool or not positive.any() or positive.all() or temperature <= 0:
        raise ValueError('Smooth-AP requires both classes and positive temperature')
    if not torch.isfinite(score).all() or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('AP scores and positive weights must be finite')
    ix = positive.nonzero().flatten()
    rank = torch.sigmoid((score[None, :] - score[ix, None])/temperature)
    # An item always counts itself, including its full source weight.
    self_mask = ix[:, None] == torch.arange(len(score), device=score.device)[None, :]
    rank = torch.where(self_mask, torch.ones_like(rank), rank)
    mass = rank * weights[None, :]
    precision = mass[:, positive].sum(1)/mass.sum(1)
    return (precision*weights[ix]).sum()/weights[ix].sum()
