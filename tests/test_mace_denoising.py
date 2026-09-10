"""Identity tracking, controlled baselines and homology-balanced supervision."""

import numpy as np
import pytest
import torch

from src.models.encoders.mace_denoising import AtomTemporalFusion, ResidualFrameFusion
from src.training_methods.mace_denoising import topology_loss, fit_targets, transform_target, raw_prediction


@pytest.mark.parametrize('atom', [False, True])
def test_zero_initial_correction_and_learning_from_every_frame(atom):
    torch.manual_seed(17)
    fusion = (AtomTemporalFusion if atom else ResidualFrameFusion)([-3., -2.25, -1.5, -.75, 0.])
    pooled = torch.randn(2, 5, 256, requires_grad=True)
    nodes = torch.randn(2, 5, 80, 256, requires_grad=True) if atom else None
    torch.testing.assert_close(fusion(pooled, nodes), pooled[:, -1], rtol=0, atol=0)
    optimizer = torch.optim.Adam(fusion.parameters(), lr=.001)
    fusion(pooled, nodes).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    pooled.grad = None
    if nodes is not None:
        nodes.grad = None
    fusion(pooled, nodes).square().mean().backward()
    gradient = nodes.grad.abs().sum((0, 2, 3)) if atom else pooled.grad.abs().sum((0, 2))
    assert (gradient > 0).all()


def test_atom_fusion_tracks_shared_identities_and_anchor_control_ignores_history():
    torch.manual_seed(29)
    fusion = AtomTemporalFusion([-1., -.5, 0.]).eval()
    torch.nn.init.normal_(fusion.temporal.output.weight, std=.05)
    nodes = torch.randn(2, 3, 80, 256)
    pooled = torch.randn(2, 3, 256)
    permutation = torch.randperm(80)
    expected = fusion(pooled, nodes)
    torch.testing.assert_close(fusion(pooled, nodes[:, :, permutation]), expected, rtol=1e-5, atol=1e-6)
    mismatched = nodes.clone()
    mismatched[:, 0] = nodes[:, 0, permutation]
    assert (fusion(pooled, mismatched)-expected).abs().max() > 1e-6
    fusion.anchor_only = True
    torch.testing.assert_close(fusion(pooled, mismatched), fusion(pooled, nodes), rtol=0, atol=0)


def test_homology_blocks_have_equal_weight_despite_different_grid_sizes():
    target = torch.zeros(4, 144)
    for block in (slice(0, 16), slice(16, 80), slice(80, 144)):
        prediction = target.clone()
        prediction[:, block] = 1
        assert float(topology_loss(prediction, target, 'blocks')) == pytest.approx(1/3)


def test_target_units_roundtrip_and_variance_floor():
    rng = np.random.default_rng(4)
    targets = rng.normal(size=(300, 144)).astype(np.float32)
    targets[:, 80:] *= 1e-6
    scaling = fit_targets(targets, 32, .05)
    assert scaling['block_scale'][2] >= .05*scaling['block_std'].max()
    encoded = transform_target(targets, scaling, 'blocks')
    np.testing.assert_allclose(raw_prediction(encoded, scaling, 'blocks'), targets, rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize('atom', [False, True])
def test_mixed_cadence_is_per_history_and_preserves_batch_assignment(atom):
    torch.manual_seed(47)
    fusion = (AtomTemporalFusion if atom else ResidualFrameFusion)([-3., -2.25, -1.5, -.75, 0.]).eval()
    correction = fusion.temporal if atom else fusion.correction
    torch.nn.init.normal_(correction.output.weight, std=.05)
    pooled = torch.randn(2, 5, 256)
    nodes = torch.randn(2, 5, 80, 256) if atom else None
    times = torch.tensor([[-4., -3., -2., -1., 0.], [-.4/.75, -.3/.75, -.2/.75, -.1/.75, 0.]])
    actual = fusion(pooled, nodes, times)
    expected = torch.cat([fusion(pooled[i:i+1], None if nodes is None else nodes[i:i+1], times[i:i+1]) for i in range(2)])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    swapped_times = fusion(pooled, nodes, times.flip(0))
    assert (actual-swapped_times).abs().max() > 1e-5


def test_potential_audit_uses_exact_paired_source_test():
    from src.analysis.mace_potential_audit import paired_statistics
    reference = np.arange(1., 10.)
    temperatures = np.repeat([400, 450, 510], 3)
    same = paired_statistics(reference, reference, temperatures, 42)
    assert same['exact_two_sided_p'] == 1.
    assert same['difference_bootstrap_95_interval'] == [0., 0.]
    shifted = paired_statistics(reference, reference+1., temperatures, 42)
    assert shifted['source_count'] == 9
    assert shifted['permutation_count'] == 512
    assert shifted['exact_two_sided_p'] == pytest.approx(2/512)
    np.testing.assert_allclose(shifted['difference_bootstrap_95_interval'], [1., 1.])
