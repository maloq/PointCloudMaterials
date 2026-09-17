"""Periodic target identity, split isolation and topology metric weighting."""
import json

import numpy as np
import pytest
import torch

from src.analysis.liquid_structure import persistence_image
from src.data.predictive_memory.observations import frame_observation
from src.research.backbone_tda.data import nearest_cloud, completed, save_arrays
from src.research.backbone_tda.probes import select_ridge, residual_probe, scores
from src.data_utils.topology_targets import BLOCKS


def test_periodic_cloud_includes_center_and_breaks_ties_by_id():
    box = np.array([100., 100., 100.])
    x = np.array([[99., 0., 0.], [0., 0., 0.], [98., 0., 0.], [95., 0., 0.]])
    ids = np.array([9, 8, 3, 7])
    frame = frame_observation(x, np.zeros_like(x), box, ids, 9, 17., 5.)
    cloud, selected = nearest_cloud(frame, 3)
    np.testing.assert_array_equal(selected, [9, 3, 8])
    np.testing.assert_array_equal(cloud, [[0, 0, 0], [-1, 0, 0], [1, 0, 0]])
    permutation = [2, 3, 1, 0]
    shifted = (x[permutation]+[12, 34, 56]) % box
    other = frame_observation(shifted, np.zeros_like(x), box, ids[permutation], 9, 17., 5.)
    for a, b in zip(nearest_cloud(other, 3), (cloud, selected), strict=True):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match='at least 80'):
        nearest_cloud(frame)


def test_persistence_target_is_rigid_motion_and_permutation_invariant():
    x = np.random.default_rng(4).normal(size=(80, 3))*2
    rotation = np.linalg.qr(np.random.default_rng(9).normal(size=(3, 3)))[0]
    original = persistence_image(x)
    moved = persistence_image((x@rotation+[3, -2, 1])[::-1])
    assert original.shape == (144,)
    np.testing.assert_allclose(original, moved, rtol=1e-5, atol=1e-7)


def test_cache_rejects_identity_drift_and_corruption(tmp_path):
    path = tmp_path/'source.npz'; identity = {'source': 7}
    assert not completed(path, identity)
    save_arrays(path, identity, targets=np.ones((2, 144)))
    assert completed(path, identity)
    with pytest.raises(ValueError, match='identity/checksum'):
        completed(path, {'source': 8})
    with path.open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='identity/checksum'):
        completed(path, identity)


def test_source_balancing_and_equal_homology_weight():
    target = np.zeros((3, 144))
    predicted = target.copy(); predicted[-1, :16] = 3.
    sources = np.array([1, 1, 2]); contexts = np.array(['a', 'a', 'b'])
    result = scores(predicted, target, sources, contexts, np.ones(3))
    assert result['balanced_mse'] == pytest.approx(1.5)  # (0 + 9/3)/2 sources
    assert result['blocks']['H0']['mse'] == pytest.approx(4.5)
    assert result['blocks']['H0']['r2'] is None
    assert result['blocks']['H1']['scaled_mse'] == 0
    duplicated = np.array([0, 0, 1, 1, 2])
    again = scores(predicted[duplicated], target[duplicated], sources[duplicated], contexts[duplicated], np.ones(3))
    assert again['balanced_mse'] == result['balanced_mse']


def test_readouts_never_use_test_targets_or_test_feature_scaling():
    torch.set_num_threads(1)
    rng = np.random.default_rng(22)
    x = rng.normal(size=(80, 4)).astype(np.float32)
    y = (x@rng.normal(size=(4, 144))).astype(np.float32)
    train, selected = np.arange(48), np.arange(48, 64)
    test = np.arange(64, 80)
    cfg = dict(seed=1, probe_width=16, probe_updates=12, probe_batch_size=16,
        probe_learning_rate=.002, probe_weight_decay=0., probe_evaluate_every=4,
        probe_patience=3, deadline_utc='2099-01-01T00:00:00+00:00')
    ridge, fit = select_ridge(x, y, train, selected, [.1, 1., 10.], np.ones(3))
    a, saved_a = residual_probe(x, y, ridge, train, selected, cfg, device='cpu')
    changed_x, changed_y = x.copy(), y.copy()
    changed_x[test] += 100; changed_y[test] += 1e5
    other, fit_b = select_ridge(changed_x, changed_y, train, selected, [.1, 1., 10.], np.ones(3))
    b, saved_b = residual_probe(changed_x, changed_y, other, train, selected, cfg, device='cpu')
    assert fit == fit_b
    np.testing.assert_array_equal(a[:64], b[:64])
    np.testing.assert_array_equal(saved_a['feature_mean'], x[train].mean(0, dtype=np.float64))
    assert saved_a['trace'] == saved_b['trace']
    for key in saved_a['model']:
        torch.testing.assert_close(saved_a['model'][key], saved_b['model'][key], atol=0, rtol=0)
    selected_loss = np.mean([np.mean((a[selected, block]-y[selected, block])**2) for block in BLOCKS])
    assert selected_loss <= saved_a['trace'][0]['selection_loss']+1e-6
