"""Event timing, censoring, risk sets and the physical projection of forecasts."""

import numpy as np
import pytest
import torch

from src.research.forecast_crystallization.local_metrics import (
    first_sustained_onset, risk_windows, select_threshold, onset_metrics,
)
from src.research.forecast_crystallization.local_predict import forecast_scores


def test_onset_is_start_of_confirmed_episode_and_end_is_censored():
    crystal = np.array([[0,0,1,1,1,0,0,0], [0,1,1,0,0,0,1,1], [1,1,1,0,0,0,0,0]], dtype=bool)
    np.testing.assert_array_equal(first_sustained_onset(crystal, 3), [2, 8, 0])
    np.testing.assert_array_equal(first_sustained_onset(crystal, 5), [8, 8, 8])


def test_at_risk_excludes_existing_crystal_and_uses_only_observed_history():
    crystal = np.array([[0,0,0,1,0,0,0,1,1,1]], dtype=bool)
    onset = first_sustained_onset(crystal, 3)
    anchors = np.arange(2, 9)
    np.testing.assert_array_equal(risk_windows(crystal, onset, anchors, 3),
                                  [[True, False, False, False, True, False, False]])


def test_threshold_uses_validation_f1_and_fails_for_undefined_selection():
    threshold, f1 = select_threshold(np.array([0,1,0,1], dtype=bool), np.array([.1,.8,.6,.9]))
    assert threshold == .8 and f1 == 1
    with pytest.raises(ValueError, match='both positive and negative'):
        select_threshold(np.zeros(4, dtype=bool), np.arange(4))


def test_timing_keeps_missed_events_in_end_to_end_recall():
    actual = np.array([1,1,0,0], dtype=bool)
    paths = np.array([[.1,.8,.9], [.1,.2,.3], [.7,.8,.9], [.1,.2,.3]])
    result, per_source = onset_metrics(actual, paths, .5, np.array([2.25,2.25,9,9]), .75,
                                       np.array([0,0,1,1]), np.arange(3), 100, 7)
    assert [result[k] for k in ('tp','fp','fn','tn')] == [1,1,1,1]
    assert result['timing_mae_ps'] == .75
    assert result['timing_bias_ps'] == -.75
    assert result['timing_within_1_5_ps'] == 1
    assert result['timed_within_1_5_ps_recall'] == .5
    assert result['recall'] == .5
    np.testing.assert_array_equal(per_source[2], np.zeros(7))


class RepeatLast(torch.nn.Module):
    history_steps = 3
    output_steps = 2

    def forward(self, history):
        return {'mean': history[:, -1:, :].expand(-1, 2, -1)}


def test_projection_preserves_atom_anchor_order_and_embedding_normalization():
    z = torch.arange(2*8*4).reshape(2,8,4).half()
    anchors = np.array([2,4,6])
    mean = torch.tensor([1.,2.,3.,4.])
    scale = torch.tensor([2.,3.,4.,5.])
    weight = torch.tensor([.2,-.3,.5,.1], dtype=torch.float64)
    bias = torch.tensor(-.7, dtype=torch.float64)
    scores = forecast_scores(z, anchors, RepeatLast(), mean, scale, weight, bias, 4)
    expected = (z[:, anchors].double()@weight+bias).numpy()
    np.testing.assert_allclose(scores, np.repeat(expected[...,None], 2, axis=-1), rtol=1e-6, atol=1e-6)
