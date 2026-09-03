import numpy as np

from src.data_utils.shooting_binary_dataset import propagate_ballistic_positions
from src.temporal_vamp.shooting_outcomes import classify_endpoint_frames
from src.temporal_vamp.shooting_short_horizon import (
    _aggregate_branch_predictions,
    _prediction_metrics_by_horizon,
)


def test_ballistic_positions_use_metal_units_and_periodic_wrapping() -> None:
    positions = np.asarray([[9.5, 0.5, 5.0]], dtype=np.float32)
    velocities = np.asarray([[2.0, -2.0, 1.0]], dtype=np.float32)
    propagated = propagate_ballistic_positions(
        positions,
        velocities,
        np.asarray([10.0, 10.0, 10.0], dtype=np.float32),
        0.5,
    )
    np.testing.assert_allclose(propagated, [[0.5, 9.5, 5.5]])
    assert np.all(propagated >= 0.0)
    assert np.all(propagated < 10.0)


def test_prediction_metrics_by_horizon_keep_horizons_separate() -> None:
    target = np.asarray(
        [
            [0.0, 0.0, 2.0, 2.0],
            [1.0, 1.0, 3.0, 3.0],
            [2.0, 2.0, 4.0, 4.0],
        ],
        dtype=np.float64,
    )
    prediction = target.copy()
    prediction[:, 2:] = 3.0
    metrics = _prediction_metrics_by_horizon(
        prediction,
        target,
        np.arange(3, dtype=np.int64),
        horizon_count=2,
        signature_dim=2,
    )
    assert metrics[0]["mse"] == 0.0
    assert metrics[0]["r2"] == 1.0
    assert metrics[1]["mse"] > 0.0
    assert metrics[1]["r2"] < 1.0


def test_aggregate_branch_predictions_averages_siblings_only() -> None:
    branch_parent = np.asarray([0, 0, 1, 1], dtype=np.int64)
    prediction = np.asarray(
        [
            [[1.0], [3.0]],
            [[3.0], [5.0]],
            [[10.0], [20.0]],
            [[14.0], [24.0]],
        ],
        dtype=np.float64,
    ).reshape(8, 1)
    aggregated = _aggregate_branch_predictions(
        prediction,
        branch_parent,
        parent_count=2,
        center_count=2,
    )
    np.testing.assert_allclose(
        aggregated,
        np.asarray([[2.0], [4.0], [12.0], [22.0]], dtype=np.float64),
    )


def test_endpoint_outcome_requires_persistent_repository_threshold() -> None:
    crystal = classify_endpoint_frames(
        np.asarray([0.02, 0.02, 0.02]),
        np.asarray([101, 130, 125]),
        crystal_cluster_threshold_atoms=100,
        maximum_liquid_crystalline_fraction=0.01,
    )
    liquid = classify_endpoint_frames(
        np.asarray([0.003, 0.004, 0.002]),
        np.asarray([12, 18, 10]),
        crystal_cluster_threshold_atoms=100,
        maximum_liquid_crystalline_fraction=0.01,
    )
    censored = classify_endpoint_frames(
        np.asarray([0.008, 0.012, 0.009]),
        np.asarray([90, 110, 95]),
        crystal_cluster_threshold_atoms=100,
        maximum_liquid_crystalline_fraction=0.01,
    )
    assert crystal == (True, False, False)
    assert liquid == (False, True, False)
    assert censored == (False, False, True)
