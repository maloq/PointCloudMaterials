from __future__ import annotations

import numpy as np

from src.temporal_vamp.shooting_dynamics import (
    fit_selected_ridge_residual,
    invariant_velocity_token_features,
)


def _rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(matrix) < 0.0:
        matrix[:, 0] *= -1.0
    return matrix


def test_velocity_features_are_invariant_to_joint_global_rotation() -> None:
    rng = np.random.default_rng(7)
    relative = rng.normal(size=(5, 12, 3))
    relative[:, 0] = 0.0
    neighbor_velocity = rng.normal(size=(5, 12, 3))
    center_velocity = rng.normal(size=(5, 3))
    rotation = _rotation(8)
    original = invariant_velocity_token_features(
        relative, neighbor_velocity, center_velocity
    )
    rotated = invariant_velocity_token_features(
        relative @ rotation,
        neighbor_velocity @ rotation,
        center_velocity @ rotation,
    )
    np.testing.assert_allclose(rotated, original, rtol=2.0e-5, atol=2.0e-5)


def test_selected_ridge_residual_recovers_incremental_signal() -> None:
    rng = np.random.default_rng(9)
    features = rng.normal(size=(180, 12))
    base = rng.normal(scale=0.1, size=(180, 4))
    weights = rng.normal(size=(12, 4))
    target = base + features @ weights
    fitted = fit_selected_ridge_residual(
        features,
        base,
        target,
        optimization_rows=np.arange(0, 120),
        selection_rows=np.arange(120, 150),
        validation_rows=np.arange(150, 180),
        dimensions=[4, 8, 12],
        alphas=[1.0e-6, 0.1, 10.0],
    )
    assert fitted.selected_dimension == 12
    assert fitted.validation_r2 > 0.999

