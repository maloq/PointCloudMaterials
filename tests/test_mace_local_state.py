"""Scientific checks for local conditioning, geometry and uncertainty semantics."""

import numpy as np

from src.research.mace_local_state.methods import (
    adjusted_pair_agreement, fit_states, neighborhood_metrics, ridge_maps,
    state_membership, temporal_map, uncertainty)
from src.research.mace_local_state.physics import group_observables


def test_temporal_coordinates_remove_context_only_signal():
    rng = np.random.default_rng(14)
    contexts = np.repeat(np.arange(12), 80)
    persistent = rng.normal(size=len(contexts))
    x = np.column_stack([contexts*10., persistent, rng.normal(size=len(contexts))])
    past = np.column_stack([contexts*10., persistent+.03*rng.normal(size=len(contexts)), rng.normal(size=len(contexts))])
    model, _ = temporal_map(x, past, contexts, 1, .001)
    assert abs(model.matrix[0, 0]) < 1e-12
    assert np.corrcoef(model(x).ravel(), persistent)[0, 1]**2 > .995
    np.testing.assert_allclose(model(x+np.array([1000., 0., 0.])), model(x), atol=1e-10)


def test_physical_map_recovers_heldout_group_metric():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(200, 8)); matrix = rng.normal(size=(8, 3)); y = x@matrix
    model = ridge_maps(x[:120], y[:120], [1e-8])[0]
    np.testing.assert_allclose(model(x[120:]), y[120:], atol=1e-8)
    result = neighborhood_metrics(model(x[120:]), y[120:], np.repeat([0, 1], 40), 8)
    assert all(row[2] == 1. for row in result)
    assert np.linalg.eigvalsh(model.matrix@model.matrix.T).min() > -1e-10


def test_uncertainty_retains_unassigned_mass_and_mixed_membership():
    values = uncertainty(np.array([[.4, .4], [.8, .0]]))
    np.testing.assert_allclose(values['unassigned_mass'], [.2, .2])
    np.testing.assert_allclose(values['ambiguity'], [1., 0.])
    np.testing.assert_allclose(values['margin'], [0., .8])
    assert np.isnan(adjusted_pair_agreement(np.zeros(10, int), np.zeros(10, int))['adjusted_agreement'])
    assert adjusted_pair_agreement(np.full(10, -1), np.full(10, -1))['assigned_pair_fraction'] == 0


def test_no_density_states_does_not_force_assignments():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(40, 3))
    model = fit_states(x, 40, 10)
    labels, strength, membership = state_membership(model, x[:3])
    np.testing.assert_array_equal(labels, -1)
    np.testing.assert_array_equal(strength, 0)
    assert membership.shape == (3, 0)
    np.testing.assert_array_equal(uncertainty(membership)['unassigned_mass'], 1)


def test_group_physics_rotation_and_atom_permutation_invariance():
    rng = np.random.default_rng(8)
    grid = np.stack(np.meshgrid(*[np.arange(-6, 7)*2.5]*3), -1).reshape(-1, 3)
    x = grid+rng.normal(scale=.04, size=grid.shape)
    center = np.argmin(np.linalg.norm(x, axis=1))
    x = x-x[center]
    ids = np.flatnonzero(np.linalg.norm(x, axis=1) < 18)
    x = x[np.r_[center, ids[ids != center]]]
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    permutation = np.r_[0, rng.permutation(np.arange(1, len(x)))]
    expected = group_observables(x)
    np.testing.assert_allclose(group_observables(x[permutation]@rotation), expected, rtol=1e-5, atol=1e-7)


def test_pipeline_maps_ignore_test_values_and_export_metrics(tmp_path, monkeypatch):
    """Exercise source splits, output shapes and exact-zero TDA pixels together."""
    import json
    from src.experiment_runner.artifacts import result_folders
    from src.research.mace_local_state import compare
    from src.research.mace_local_state.methods import AffineMap
    rng = np.random.default_rng(28)
    n = 384
    probes = dict(source=np.repeat(np.arange(6), 64), context=np.repeat(np.arange(12), 32),
                  split=np.repeat(['train', 'train', 'train', 'val', 'test', 'test'], 64))
    raw = rng.normal(size=(n, 40))
    group = rng.normal(size=(n, 3, 16)); group[:, 0] += raw[:, :16]
    for key in ['hot', 'relaxed']:
        probes[key] = rng.normal(size=(n, 144)); probes[key][:, 0] = 0.
    temporal = dict(source=np.repeat([4, 5], 4))
    time = rng.normal(size=(8, 9, 40))
    arrays = dict(anchor=raw, previous=raw+.1*rng.normal(size=raw.shape),
                  spatial=raw+.1*rng.normal(size=raw.shape), temporal=time,
                  crossing=np.tile(rng.normal(size=(4, 8, 1, 40)), (1, 1, 2, 1)))
    values = {k: dict(inner=x[..., :20], dual=x, projector=x[..., :16]) for k, x in arrays.items()}
    local = dict(group=group)
    monkeypatch.setattr(compare, 'load_prepared', lambda *args: (probes, temporal, {}, local))
    monkeypatch.setattr(compare, 'blocks', lambda *args: values)
    config = dict(temporal_dimensions=[8, 16], temporal_regularization=[.01], ridge_alphas=[.1, 10.],
                  neighbor_k=8, seed=9, state_minimum_sizes=[20], state_minimum_samples=5,
                  state_source_subsamples=1, evaluation_lags_steps=[1, 2])
    first = result_folders(tmp_path/'first'); compare.fit(config, first)
    test = probes['split'] == 'test'
    raw[test] += 1000.; group[test] += 1000.
    second = result_folders(tmp_path/'second'); compare.fit(config, second)
    metadata = json.loads((first/'technical/maps.json').read_text())
    for name in metadata:
        a = AffineMap.load(first/f'technical/maps/{name}.npz')
        b = AffineMap.load(second/f'technical/maps/{name}.npz')
        np.testing.assert_array_equal(a.matrix, b.matrix)
        np.testing.assert_array_equal(a.mean, b.mean)
    raw[test] -= 1000.; group[test] -= 1000.
    compare.evaluate(config, first)
    assert (first/'tables/comparison.csv').is_file()
    assert (first/'tables/state_stability.csv').is_file()
