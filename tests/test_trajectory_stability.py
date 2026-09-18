"""Scientific controls for the matched temporal-stability assay."""
import numpy as np
import pytest

from src.research.trajectory_stability.metrics import (
    reference_statistics, trajectory_metrics, stratified_draws, rms_interval,
)


def test_linear_motion_has_zero_roughness_and_linear_lag_growth():
    reference = reference_statistics(np.arange(40.).reshape(20, 2))
    t = np.arange(50.)[:, None, None]
    # Different atom offsets must never create artificial cross-atom jumps.
    tracks = np.concatenate([2*t, 2*t+1e6], axis=1)
    tracks = np.concatenate([tracks, -tracks], axis=-1)
    metrics = trajectory_metrics(tracks, reference, [1, 2, 4])
    assert metrics['roughness'] == 0
    assert metrics['increment_cosine'] == pytest.approx(1)
    np.testing.assert_allclose(np.sqrt(metrics['lag2']/metrics['lag2'][0]), [1, 2, 4])


def test_alternating_motion_is_maximally_rough_even_with_tiny_amplitude():
    reference = reference_statistics(np.array([[-1.], [1.]]))
    track = (np.arange(100) % 2 * 2-1)[:, None, None]*1e-8
    metrics = trajectory_metrics(track, reference, [1, 2])
    assert metrics['roughness'] == pytest.approx(2)
    assert metrics['increment_cosine'] == pytest.approx(-1)
    assert metrics['jump2_mean'] < 1e-14
    assert metrics['lag2'][1] == 0


def test_independent_frames_reproduce_reference_limits():
    rng = np.random.default_rng(440)
    reference = reference_statistics(rng.normal(size=(50000, 3)))
    metrics = trajectory_metrics(rng.normal(size=(10000, 4, 3)), reference, [1, 2, 4])
    np.testing.assert_allclose(metrics['lag2'], 1., atol=.02)
    assert metrics['roughness'] == pytest.approx(1.5, abs=.015)


def test_native_normalization_is_translation_scale_and_basis_invariant():
    rng = np.random.default_rng(123)
    train = rng.normal(size=(40, 3))*[.1, .2, .8]
    z = rng.normal(size=(20, 2, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    original = trajectory_metrics(z, reference_statistics(train), [1, 4])
    moved = trajectory_metrics((z@q)*17+38, reference_statistics((train@q)*17+38), [1, 4])
    np.testing.assert_allclose(original['lag2'], moved['lag2'], rtol=1e-12)
    assert moved['roughness'] == pytest.approx(original['roughness'])


def test_float64_moments_preserve_small_changes_on_large_mean():
    reference = reference_statistics(np.array([[1e6-1e-3], [1e6+1e-3]]))
    assert reference['trace'] == pytest.approx(1e-6, rel=1e-6)
    assert reference['effective_rank'] == pytest.approx(1)
    with pytest.raises(ValueError, match='Collapsed'):
        reference_statistics(np.ones((10, 3)))


def test_source_bootstrap_preserves_temperature_and_weights_sources_equally():
    temperatures = [400, 400, 500, 500]
    draws = stratified_draws(temperatures, 100, 12)
    np.testing.assert_array_equal(np.sort(np.array(temperatures)[draws], axis=1), np.tile(temperatures, (100, 1)))
    value, interval = rms_interval([1., 1., 9., 9.], draws)
    assert value == pytest.approx(np.sqrt(5))
    np.testing.assert_allclose(interval, value)
