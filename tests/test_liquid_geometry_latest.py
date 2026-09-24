"""Exact cohort bridge controls: conditioning, physical geometry and source pairs."""
import numpy as np

from src.research.liquid_geometry.latest import effective_rank, paired_association, retrieve


def test_retrieval_uses_temperature_and_training_targets():
    # Cross-temperature points are closer, but may not become neighbors.
    ref = np.array([[0.], [2.], [.1], [2.1]])
    query = np.array([[.1], [2.]])
    target = np.array([[0.], [2.], [10.], [12.]])
    truth = np.array([[0.], [12.]])
    result = retrieve(ref, query, target, truth, np.array([1, 2, 3, 4]), np.array([5, 6]),
                      np.array([400, 400, 500, 500]), np.array([400, 500]), k=1)
    np.testing.assert_array_equal(result["neighbor"], [0., 0.])
    np.testing.assert_array_equal(result["reconstruction"], [0., 0.])
    assert np.all(result["random"] > 0)


def test_participation_is_scale_invariant_and_constant_explicit():
    x = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    assert effective_rank(x) == 2.
    assert effective_rank(30 * x + 100) == 2.
    assert effective_rank(np.ones((5, 2))) == 0.
    assert effective_rank(np.ones((1, 2))) is None


def test_source_association_removes_temperature_offsets():
    t = np.repeat([400., 500.], 6)
    x = np.tile(np.arange(6.), 2) + np.repeat([0., 100.], 6)
    y = np.tile(-np.arange(6.), 2) + np.repeat([0., 100.], 6)
    result = paired_association(x, y, t, draws=40, seed=22)
    assert result["spearman"] == -1.
    assert result["upper"] < -.99
    assert result["valid_draws"] == 40


def test_source_association_requires_four_defined_sources():
    result = paired_association([1., 2., 3.], [3., 2., 1.], [400., 400., 400.], draws=20, seed=22)
    assert result == dict(spearman=None, lower=None, upper=None, valid_draws=0)
