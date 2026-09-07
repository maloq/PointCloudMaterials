from __future__ import annotations

import numpy as np

from src.temporal_vamp.predictability_map import (
    aggregate_by_parent,
    regression_metrics,
    split_shot_noise_ceiling,
)


def test_parent_aggregation_and_regression_metrics() -> None:
    values = np.asarray(
        [
            [[0.0], [1.0]],
            [[2.0], [3.0]],
            [[10.0], [11.0]],
            [[14.0], [15.0]],
        ],
        dtype=np.float32,
    )
    parents = np.asarray([0, 0, 1, 1], dtype=np.int64)
    mean, variance = aggregate_by_parent(values, parents, parent_count=2)
    np.testing.assert_allclose(mean[..., 0], [[1.0, 2.0], [12.0, 13.0]])
    np.testing.assert_allclose(variance[..., 0], [[2.0, 2.0], [8.0, 8.0]])
    metrics = regression_metrics(
        mean.reshape(4, 1), mean.reshape(4, 1), np.arange(4, dtype=np.int64)
    )
    assert metrics["mse"] == 0.0
    assert metrics["r2"] == 1.0


def test_split_shot_ceiling_detects_reproducible_mean_not_variance() -> None:
    rng = np.random.default_rng(7)
    parent_count = 6
    shots = 12
    centers = 32
    dimensions = 4
    parent_signal = rng.normal(size=(parent_count, centers, dimensions))
    projected = np.concatenate(
        [
            parent_signal[parent][None]
            + 0.15 * rng.normal(size=(shots, centers, dimensions))
            for parent in range(parent_count)
        ],
        axis=0,
    ).astype(np.float32)
    branch_parent = np.repeat(np.arange(parent_count), shots)
    rff = projected.copy()
    metrics = split_shot_noise_ceiling(
        projected,
        rff,
        branch_parent,
        np.asarray([4, 5], dtype=np.int64),
        repetitions=20,
        seed=3,
    )
    assert metrics["mean_future"]["estimated_full_shot_reliability_mean"] > 0.98
    assert metrics["future_law"]["estimated_full_shot_reliability_mean"] > 0.98
    assert metrics["log_variance"]["estimated_full_shot_reliability_mean"] < 0.5
