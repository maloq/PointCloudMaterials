import copy
import numpy as np
import pytest
import torch

from src.research.crystallization_followup.data import (
    EXTRA_DIM, balanced_moments, dense_observations, history_information)
from src.research.crystallization_followup.metrics import window_scores, paired_source_gain
from src.research.crystallization_followup.model import FollowupForecaster, censored_nll
from src.research.structured_context.model import StructuredForecaster


def spec():
    original = dict(protocol='path_refinement_v2', norm_eps=1e-8, dropout=0.,
                    aggregation='attention', attention='factorized', geometry_bias=True,
                    equivariant=False, history_ps=72, radius_A=25, gaussian_rank=0,
                    state_weight=1., present_weight=0., residual_anchor=False,
                    motion_input=False, teacher_mode='scheduled', teacher_epochs=6.)
    original.update(method='ar_mse', head_width=16, heads=2, depth=1, encoder='mace',
                    context_layout='cuboctahedral_followup_v1', shell_radii_A=[10., 20.],
                    target_encoder='reference_mace', information_context='both',
                    secondary_domain='off', history_rates=False, dense_mode='off',
                    quench_descriptors=False, short_weight=0., absolute_clock=True,
                    front_mode='off', front_radius_A=25., front_features='all')
    return original


def test_irregular_secants_have_correct_units():
    offsets = torch.tensor([[-12., -3., 0.], [-9., -.75, 0.]])
    values = offsets[..., None].expand(-1, -1, 97)*2 + 7
    out = history_information(values, offsets, True)
    torch.testing.assert_close(out[:, :93], torch.full((2, 93), 7.))
    torch.testing.assert_close(out[:, 93:279], torch.full((2, 186), 2.))
    with pytest.raises(ValueError, match='strictly increasing'):
        history_information(values, torch.zeros_like(offsets), True)


def test_dense_join_ignores_future_and_uses_actual_frames():
    x = np.broadcast_to(np.arange(50)[None, :, None], (2, 50, 97)).copy().astype(float)
    a = dense_observations(x, [32], [1])
    np.testing.assert_array_equal(a.reshape(5, 97)[:, 0], [16, 24, 28, 31, 32])
    x[:, 33:] = np.nan
    np.testing.assert_array_equal(a, dense_observations(x, [32], [1]))


def test_moments_use_only_training_and_equal_source_mass():
    x = torch.tensor([[0.], [0.], [6.], [999.]])
    mean, scale = balanced_moments(x, {10: [0, 1], 20: [2]})
    torch.testing.assert_close(mean, torch.tensor([3.]))
    torch.testing.assert_close(scale, torch.tensor([3.]))


def test_short_likelihood_censors_after_horizon_and_has_no_future_gradient():
    logits = torch.zeros(4, 32, 4, requires_grad=True)
    event = torch.tensor([0, 15, 16, 128])
    loss = censored_nll(logits, event)
    torch.testing.assert_close(loss, torch.tensor([1., 16., 16., 16.])*np.log(2))
    loss.sum().backward()
    assert torch.count_nonzero(logits.grad.flatten(1)[:, 16:]) == 0
    assert logits.grad[2, 3, 3] > 0  # censored example requires survival, not an event


def test_perfect_cdf_has_zero_brier_and_restricted_time_error():
    event = np.array([0, 15, 16, 128])
    p = (np.arange(128)[None] >= event[:, None]).astype(float)
    scores = window_scores(p, event)
    np.testing.assert_array_equal(scores['brier12'], np.zeros(4))
    np.testing.assert_array_equal(scores['restricted_time_mae12'], np.zeros(4))
    gain = paired_source_gain([1, 1, 4], [0, 0, 0], [1, 1, 2], 100)
    assert gain['gain'] == 2.5  # source-weighted, not window-weighted


def test_dual_repeat_control_matches_parameter_capacity():
    s = spec(); s['secondary_domain'] = 'observed'
    a = FollowupForecaster(s)
    s = dict(s, secondary_domain='relaxed')
    b = FollowupForecaster(s)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())


def test_followup_zero_auxiliary_matches_original_ar_objective():
    torch.set_num_threads(1); s = spec(); torch.manual_seed(21)
    base = StructuredForecaster(copy.deepcopy(s))
    extended = FollowupForecaster(copy.deepcopy(s))
    missing, unexpected = extended.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith('extra_') for k in missing)
    observed = dict(features=torch.randn(2, 75, 128), geometry=torch.zeros(2, 75, 4),
                    condition=torch.zeros(2, 7), information=torch.randn(2, 504))
    observed['geometry'][:, :, 3] = torch.tensor([-12., -3., 0.]).repeat_interleave(25)
    target = dict(state=torch.randn(2, 32, 265), present=torch.randn(2, 265), event=torch.tensor([8, 128]))
    for model in (base, extended):
        x, w = model.context.inputs(observed['features'], observed['geometry'])
        model.context.normalization.calibrate(x, w)
    base.train(); extended.train()
    expected = base.loss(observed, target, 0.)
    actual = extended.loss(dict(observed, extra=torch.randn(2, EXTRA_DIM)), target, 0.)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
