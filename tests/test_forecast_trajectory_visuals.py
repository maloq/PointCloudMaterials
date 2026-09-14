"""Diagnostic case semantics and reconstruction of periodic local structure."""

from types import SimpleNamespace

import numpy as np

from src.research.forecast_spatial_mixture.trajectories import example_masks, measured_clouds


def test_diagnostic_selection_retains_misses_and_censored_negatives():
    crystal = np.zeros((4, 30), dtype=bool)
    crystal[:3, 13:] = True
    anchors = np.array([5])
    scores = np.zeros((4, 1, 12))
    scores[0, 0, 5:] = .8  # 4.5 ps forecast versus 6 ps truth: included boundary.
    scores[1, 0, 1:] = .8  # 1.5 ps forecast versus 6 ps truth: early by 4.5 ps.
    scores[3, 0, 7:] = .8  # No observed onset at all: a false alarm, not dropped.
    masks, onsets = example_masks(crystal, anchors, scores, .5, .75, 3, 3)
    for index, category in enumerate(('correct', 'early', 'missed', 'false-alarm')):
        np.testing.assert_array_equal(np.flatnonzero(masks[category]), [index])
    assert onsets[3] == 30
    crystal[0, 4] = True  # Recent transient crystal makes this origin ineligible.
    masks, _ = example_masks(crystal, anchors, scores, .5, .75, 3, 3)
    assert not masks['correct'].any()


def test_measured_cloud_tracks_center_and_wraps_across_periodic_boundary():
    rng = np.random.default_rng(12)
    positions = rng.uniform(2, 8, size=(2, 90, 3))
    positions[:, 0] = [9.9, 5, 5]
    positions[:, 1] = [.1, 5, 5]
    positions[1, 0] = [.2, 5, 5]
    trajectory = SimpleNamespace(positions=positions, atom_ids=np.arange(1, 91),
        box_low=np.zeros((2, 3)), box_high=np.full((2, 3), 10.))
    clouds, ids = measured_clouds(trajectory, [0, 1], 1)
    assert clouds.shape == (2, 80, 3)
    np.testing.assert_array_equal(ids[:, 0], [1, 1])
    np.testing.assert_array_equal(clouds[:, 0], np.zeros((2, 3)))
    np.testing.assert_allclose(clouds[0, np.flatnonzero(ids[0] == 2)[0]], [.2, 0, 0], atol=1e-6)
    np.testing.assert_allclose(clouds[1, np.flatnonzero(ids[1] == 2)[0]], [-.1, 0, 0], atol=1e-6)
    assert np.max(abs(clouds)) <= 5


def test_feature_contributions_exactly_reconstruct_the_change_in_linear_score():
    from src.research.forecast_spatial_mixture.trajectory_projection import channel_contributions, readout_parameters
    import torch
    torch.manual_seed(14)
    raw = torch.randn(29, 7, dtype=torch.float64)
    mean, scale = torch.randn(7, dtype=torch.float64), torch.rand(7, dtype=torch.float64)+.3
    probe = dict(mean=torch.randn(7, dtype=torch.float64), std=torch.rand(7, dtype=torch.float64)+.3,
                 coefficients=torch.randn(8, 2, dtype=torch.float64))
    p = readout_parameters(probe, mean, scale)
    standardized = ((raw-mean)/scale).numpy()
    original = torch.cat(((raw-probe['mean'])/probe['std'], torch.ones(29, 1)), dim=1)@probe['coefficients']
    score = (original[:, 1]-original[:, 0]).numpy()
    np.testing.assert_allclose(standardized@p['weight']+p['bias'], score, atol=1e-12)
    contributions = channel_contributions(standardized, p['weight'])
    np.testing.assert_allclose(contributions.sum(1), score-score[:17].mean(), atol=1e-12)


def test_time_summary_retains_the_last_frame_and_does_not_mix_bins():
    from src.research.forecast_spatial_mixture.trajectory_projection import time_blocks
    time = np.arange(17)*.75
    values = np.column_stack((np.arange(17), -np.arange(17)))
    t, coordinates = time_blocks(time, values)
    np.testing.assert_allclose(t, [2.625, 8.625, 12])
    np.testing.assert_allclose(coordinates, [[3.5, -3.5], [11.5, -11.5], [16, -16]])
