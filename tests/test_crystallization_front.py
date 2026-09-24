import itertools
import numpy as np
import torch
from src.research.crystallization_followup.front import frame_features, NAMES
from src.research.crystallization_followup.data import extra_mask


def crystal():
    basis = np.array([[0,0,0], [0,.5,.5], [.5,0,.5], [.5,.5,0]])
    points = (np.array(list(itertools.product(range(8), repeat=3)))[:, None]+basis).reshape(-1, 3)*4.05
    return points, np.full(3, 8*4.05)


def test_fcc_order_and_components():
    points, box = crystal(); values, _ = frame_features(points, box, [0, 56])
    a = values.reshape(2, 3, -1)
    np.testing.assert_allclose(a[:, :, NAMES.index('ordered_fraction')], 1.)
    np.testing.assert_allclose(a[:, :, NAMES.index('largest_component_fraction')], 1.)
    np.testing.assert_allclose(a[:, :, NAMES.index('q6_mean')], .57452426, atol=1e-6)
    np.testing.assert_allclose(a[:, :, NAMES.index('nearest_ordered_distance_over_radius')], 0.)


def test_front_is_invariant_to_periodic_translation_and_cubic_rotation():
    points, box = crystal(); rng = np.random.default_rng(23)
    points = np.mod(points+rng.normal(0, .16, points.shape), box)
    original, _ = frame_features(points, box, [0, 56])
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
    moved = np.mod(points@rotation.T+[8.123, -3.724, 17.412], box)
    result, _ = frame_features(moved, box, [0, 56])
    np.testing.assert_allclose(result, original, atol=2e-6, rtol=2e-6)


def test_front_feature_ablation_masks_do_not_mix_radii():
    base = dict(history_rates=False, dense_mode='off', quench_descriptors=False,
                front_mode='rates', front_radius_A=20., front_features='geometry')
    mask = extra_mask(base, 'cpu')
    assert torch.count_nonzero(mask) == 16
    assert not mask[:584].any() and not mask[584:598].any()
    assert mask[600] == 1 and mask[642] == 1
