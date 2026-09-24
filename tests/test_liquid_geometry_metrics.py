"""Scientific controls for liquid geometry fits and trajectory identity."""

import numpy as np
import pytest

from src.research.liquid_geometry.metrics import (
    conditional_rank,
    fit_physical_metric,
    fit_transform,
    lag_pairs,
    neighbor_metrics,
    participation,
)


def test_participation_is_shift_scale_rotation_invariant():
    rng = np.random.default_rng(51)
    x = rng.normal(size=(90, 5)) * np.array([1, 2, 3, 4, 5])
    rotation, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    before = participation(x)
    after = participation(7 * x @ rotation + 13)
    assert after["rank"] == pytest.approx(before["rank"])
    assert after["top_fraction"] == pytest.approx(before["top_fraction"])
    assert after["trace"] == pytest.approx(49 * before["trace"])
    covariance = np.cov(x, rowvar=False)
    assert before["rank"] == pytest.approx(np.trace(covariance)**2 / np.trace(covariance @ covariance))
    assert participation(np.ones((8, 4))) == {"rank": 0, "trace": 0, "top_fraction": 0}


@pytest.mark.parametrize("mode", ["raw", "standardized", "whitened"])
def test_transforms_fit_training_only_and_preserve_evaluation_shift(mode):
    rng = np.random.default_rng(72)
    train = rng.normal(size=(80, 4)) * [1, 2, 4, 8]
    evaluation = rng.normal(size=(10, 4))
    train_a, eval_a, meta_a = fit_transform(train, evaluation, mode=mode)
    train_b, eval_b, meta_b = fit_transform(train, evaluation + 100, mode=mode)
    np.testing.assert_array_equal(train_a, train_b)
    assert meta_a == meta_b
    assert np.linalg.norm(eval_a - eval_b) > 10
    np.testing.assert_allclose(train_a.mean(axis=0), 0, atol=1e-15)
    train_c, eval_c, _ = fit_transform(train + 7, evaluation + 7, mode=mode)
    np.testing.assert_allclose(train_a, train_c, atol=1e-13)
    np.testing.assert_allclose(eval_a, eval_c, atol=1e-13)


def test_whitening_matches_regularized_covariance_and_global_rescaling():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(80, 4)) @ rng.normal(size=(4, 4))
    evaluation = rng.normal(size=(10, 4))
    a, b, metadata = fit_transform(train, evaluation, mode="whitened")
    scaled_a, scaled_b, _ = fit_transform(13 * train, 13 * evaluation, mode="whitened")
    np.testing.assert_allclose(a, scaled_a, atol=1e-12)
    np.testing.assert_allclose(b, scaled_b, atol=1e-12)
    covariance = np.cov(train, rowvar=False)
    whitening = np.asarray(metadata["whitening"])
    np.testing.assert_allclose(
        whitening @ (covariance + metadata["ridge_variance"] * np.eye(4)) @ whitening,
        np.eye(4), atol=1e-12,
    )


def test_source_exclusion_precedes_search_and_mse_averages_targets():
    # The closest rows have the same source and intentionally misleading targets.
    ref_z = np.array([[0], [0.1], [1], [3], [6]], dtype=float)
    query_z = np.array([[0], [1]], dtype=float)
    ref_y = np.array([[90, 90], [80, 80], [1, 3], [3, 5], [6, 8]], dtype=float)
    query_y = np.array([[0, 2], [1, 3]], dtype=float)
    metrics = neighbor_metrics(
        ref_z, query_z, ref_y, query_y,
        np.array(["a", "a", "b", "b", "c"]), np.array(["a", "b"]), k=2,
    )
    np.testing.assert_allclose(metrics["neighbor_distance"], [2, 0.95])
    np.testing.assert_array_equal(metrics["neighbor_indices"], [[2, 3], [1, 0]])
    expected_b = np.mean((ref_y[[1, 0]] - query_y[1])**2)
    np.testing.assert_allclose(metrics["neighbor_target_mse"], [5, expected_b])
    with pytest.raises(ValueError, match="different-source references"):
        neighbor_metrics(ref_z, query_z, ref_y, query_y, [0] * 5, [0] * 2, k=1)


def test_physical_metric_recovers_target_geometry_from_nuisance_dominated_input():
    rng = np.random.default_rng(81)
    train_signal = rng.normal(size=(250, 2))
    eval_signal = rng.normal(size=(60, 2))
    train_x = np.column_stack((train_signal, 1000 * rng.normal(size=(250, 8))))
    eval_x = np.column_stack((eval_signal, 1000 * rng.normal(size=(60, 8))))
    target_matrix = np.array([[1, 2], [-2, 1]])
    train_y = train_signal @ target_matrix
    train_pred, eval_pred, metadata = fit_physical_metric(train_x, train_y, eval_x, alpha=1e-6)
    expected = (eval_signal @ target_matrix - train_y.mean(0)) / train_y.std(0)
    np.testing.assert_allclose(eval_pred, expected, atol=1e-7)
    np.testing.assert_allclose(train_pred.mean(0), 0, atol=1e-14)
    shifted_eval = eval_x.copy()
    shifted_eval[:, :2] += 10
    again, changed, unchanged = fit_physical_metric(train_x, train_y, shifted_eval, alpha=1e-6)
    np.testing.assert_array_equal(train_pred, again)
    assert metadata == unchanged
    assert np.linalg.norm(changed - eval_pred) > 1


def test_lag_pairs_preserve_source_atom_and_exact_frame_with_unsorted_rows():
    source = np.array(["a", "a", "b", "a", "b", "a", "a"])
    atom = np.array([4, 4, 4, 5, 4, 4, 5])
    frame = np.array([12, 10, 10, 10, 12, 13, 13])
    past, future = lag_pairs(source, atom, frame, lag_frames=2)
    np.testing.assert_array_equal(past, [1, 2])
    np.testing.assert_array_equal(future, [0, 4])
    empty_a, empty_b = lag_pairs(source, atom, frame, lag_frames=7)
    assert empty_a.dtype == empty_b.dtype == np.int64
    assert len(empty_a) == len(empty_b) == 0
    with pytest.raises(ValueError, match="Duplicate"):
        lag_pairs(["a", "a"], [1, 1], [2, 2], lag_frames=1)
    with pytest.raises(TypeError, match="frame"):
        lag_pairs(["a", "a"], [1, 1], [2.0, 3.0], lag_frames=1)


def test_conditional_rank_removes_between_temperature_variation():
    x = np.array([[0, 0], [1, 0], [0, 20], [0, 21], [5, 5]], dtype=float)
    result = conditional_rank(x, np.array([400, 400, 600, 600, 800]))
    assert participation(x[:4])["rank"] > 1
    assert result[400]["rank"] == result[600]["rank"] == 1
    assert result[800] == {
        "n": 1, "status": "insufficient_rows", "rank": None, "trace": None, "top_fraction": None,
    }


@pytest.mark.parametrize("invalid", [np.array([[1, np.nan], [2, 3]]), np.array([[1, np.inf], [2, 3]])])
def test_nonfinite_values_are_rejected(invalid):
    with pytest.raises(ValueError, match="nonfinite"):
        participation(invalid)
    with pytest.raises(ValueError, match="nonfinite"):
        fit_transform(np.ones((3, 2)), invalid)


def test_malformed_inputs_and_degenerate_whitening_fail_loudly():
    with pytest.raises(ValueError, match="shape"):
        participation(np.ones(5))
    with pytest.raises(ValueError, match="feature counts"):
        fit_transform(np.ones((3, 2)), np.ones((2, 3)))
    with pytest.raises(ValueError, match="nonzero training variance"):
        fit_transform(np.ones((3, 2)), np.ones((2, 2)), mode="whitened")
    with pytest.raises(ValueError, match="train_y rows"):
        fit_physical_metric(np.ones((3, 2)), np.ones((4, 1)), np.ones((2, 2)))
