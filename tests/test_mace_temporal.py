"""Temporal inference and end-to-end symmetry/gradient contracts with real MACE."""

import copy

import numpy as np
import pytest
import torch
from mace import modules
from e3nn import o3
from omegaconf import OmegaConf

from src.models.encoders import build_encoder
from src.models.encoders.mace_temporal import PretrainedMACETemporalEncoder


@pytest.fixture(scope="module")
def mace_checkpoint(tmp_path_factory):
    """Small real MACE with the repository's two 128-scalar output blocks."""
    torch.manual_seed(81)
    backbone = modules.MACE(
        r_max=5.0,
        num_bessel=4,
        num_polynomial_cutoff=5,
        max_ell=1,
        interaction_cls=modules.RealAgnosticResidualInteractionBlock,
        interaction_cls_first=modules.RealAgnosticInteractionBlock,
        num_interactions=2,
        num_elements=3,
        hidden_irreps=o3.Irreps("128x0e"),
        MLP_irreps=o3.Irreps("16x0e"),
        atomic_energies=np.zeros(3),
        avg_num_neighbors=12.0,
        atomic_numbers=[12, 13, 73],
        correlation=2,
        gate=torch.nn.functional.silu,
        radial_MLP=[16],
    ).float()
    path = tmp_path_factory.mktemp("temporal_mace") / "mace.model"
    torch.save(backbone, path)
    return path


@pytest.fixture
def encoder(mace_checkpoint):
    torch.manual_seed(23)
    return PretrainedMACETemporalEncoder(
        mace_checkpoint,
        frame_offsets_ps=[-0.2, -0.1, 0.0],
        time_scale_ps=0.1,
        width=32,
        num_heads=4,
        num_layers=2,
        feedforward_dim=64,
        frame_batch_size=4,
        accelerated=False,
    ).eval()


@pytest.fixture
def histories():
    torch.manual_seed(17)
    grid = torch.cartesian_prod(torch.arange(4), torch.arange(4), torch.arange(5)).float() * 2.4
    points = grid[None, None] + 0.15 * torch.randn(2, 3, 80, 3)
    points = points - points[:, :, :1]
    return points, torch.tensor([0, 2])


def test_past_frames_receive_gradients_and_change_anchor_embedding(encoder, histories):
    points, material = histories
    points.requires_grad_(True)
    head = torch.nn.Linear(encoder.invariant_dim, 32)
    target = torch.randn(2, 32)
    z = encoder(points, material)
    assert z.shape == (2, 32)
    (head(z) - target).square().mean().backward()
    assert torch.isfinite(points.grad).all()
    assert (points.grad.abs().sum(dim=(0, 2, 3)) > 0).all()
    for module in (encoder.mace.backbone.interactions, encoder.blocks, encoder.time_embedding, head):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0
    with torch.no_grad():
        changed = points.detach().clone()
        changed[:, 0, 1:] *= 0.8
        assert (encoder(changed, material) - z).abs().max() > 1e-7


def test_temporal_order_and_physical_spacing_matter(encoder, histories):
    points, material = histories
    with torch.no_grad():
        reference = encoder(points, material)
        reordered = encoder(points[:, [1, 0, 2]], material)
        assert (reference - reordered).abs().max() > 1e-7
        encoder.frame_offsets_ps.mul_(2)
        spaced = encoder(points, material)
        assert (reference - spaced).abs().max() > 1e-5


def test_rigid_motion_atom_permutation_and_batch_independence(encoder, histories):
    points, material = histories
    # Independent rotations/translations/permutations per observed frame.
    q = torch.linalg.qr(torch.randn(2, 3, 3, 3))[0]
    moved = points @ q + torch.randn(2, 3, 1, 3)
    moved = torch.stack([
        torch.stack([frame[torch.randperm(80)] for frame in history])
        for history in moved
    ])
    with torch.no_grad():
        expected = encoder(points, material)
        torch.testing.assert_close(encoder(moved, material), expected, rtol=1e-4, atol=2e-5)
        separate = torch.cat([encoder(points[i:i + 1], material[i:i + 1]) for i in range(2)])
        torch.testing.assert_close(separate, expected, rtol=1e-4, atol=2e-5)


def test_frame_chunking_preserves_output_and_joint_gradients(encoder, histories):
    points, material = histories
    unchunked = copy.deepcopy(encoder)
    unchunked.frame_batch_size = 100
    expected = unchunked(points, material)
    actual = encoder(points, material)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)
    derivative = torch.randn_like(actual)
    actual.backward(derivative)
    expected.backward(derivative)
    for (name, p), (_, reference) in zip(encoder.named_parameters(), unchunked.named_parameters()):
        if reference.grad is None:
            assert p.grad is None, name
        else:
            torch.testing.assert_close(p.grad, reference.grad, rtol=3e-4, atol=2e-5, msg=name)


def test_config_construction_and_checkpoint_restore(mace_checkpoint, histories, tmp_path):
    cfg = OmegaConf.load("configs/mace_temporal_encoder.yaml")
    cfg.encoder.kwargs.pretrained_checkpoint = str(mace_checkpoint)
    cfg.encoder.kwargs.accelerated = False
    cfg.encoder.kwargs.frame_offsets_ps = [-0.2, -0.1, 0.0]
    model = build_encoder(cfg).eval()
    points, material = histories
    with torch.no_grad():
        expected = model(points, material)
    checkpoint = tmp_path / "encoder.pt"
    torch.save(dict(encoder=model.state_dict(), config=OmegaConf.to_container(cfg)), checkpoint)
    saved = torch.load(checkpoint, weights_only=True)
    restored = build_encoder(OmegaConf.create(saved["config"])).eval()
    restored.load_state_dict(saved["encoder"], strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(points, material), expected, rtol=0, atol=0)


@pytest.mark.parametrize("offsets", [[-0.1, 0.1], [0.0, -0.1, 0.0], [-0.1, -0.1, 0.0]])
def test_rejects_future_or_unordered_histories(mace_checkpoint, offsets):
    with pytest.raises(ValueError, match="frame_offsets_ps"):
        PretrainedMACETemporalEncoder(mace_checkpoint, offsets, time_scale_ps=0.1, accelerated=False)


def test_rejects_single_cloud_or_wrong_window(encoder, histories):
    points, material = histories
    for wrong in (points[:, -1], points[:, :2]):
        with pytest.raises(ValueError, match="Expected history shape"):
            encoder(wrong, material)


def test_unpooled_mace_retains_existing_pooled_features(encoder, histories):
    points, material = histories
    with torch.no_grad():
        nodes = encoder.mace.raw_node_features(points[:, -1], material)
        expected = encoder.mace.raw_features(points[:, -1], material)
    assert nodes.shape == (len(points), 80, 256)
    torch.testing.assert_close(nodes.mean(1), expected, rtol=0, atol=0)


def test_training_replay_preserves_full_batch_covariance_gradients(encoder, histories):
    from src.training_methods.mace_temporal import cached_step, objective
    model = torch.nn.Module()
    model.encoder = encoder.train()
    model.tda = torch.nn.Linear(encoder.invariant_dim, 32)
    points, material = histories
    target = torch.randn(2, 32)
    cfg = dict(microbatch_size=1, loss=dict(tda=1., variance=25., covariance=1.))
    cached_step(model, (points, target, material), cfg)
    expected = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    z = model.encoder(points, material)
    loss, _ = objective(model, z, target, cfg['loss'])
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, expected[name], rtol=3e-4, atol=3e-5, msg=name)
