from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")

from src.training_methods.contrastive_learning.pointcontrast import PointContrastLoss


class _DummyEncoder(torch.nn.Module):
    def __init__(self, num_points: int, latent_dim: int):
        super().__init__()
        self.proj = torch.nn.Linear(num_points * 3, latent_dim)

    def forward(self, x: torch.Tensor):
        if x.dim() != 3 or x.shape[-1] != 3:
            raise ValueError(f"Expected (B,N,3), got {tuple(x.shape)}")
        flat = x.reshape(x.shape[0], -1)
        return self.proj(flat), None


def _make_loss(*, queue_size: int = 0) -> PointContrastLoss:
    return PointContrastLoss(
        enabled=True,
        weight=1.0,
        temperature=0.1,
        embed_dim=32,
        start_epoch=0,
        jitter_std=0.0,
        jitter_mode="absolute",
        jitter_scale=1.0,
        drop_ratio=0.0,
        view_points=16,
        neighbor_view=False,
        neighbor_view_mode="both",
        neighbor_k=8,
        neighbor_max_relative_distance=0.0,
        view_crop_mode="random",
        drop_apply_to_both=True,
        rotation_mode="none",
        rotation_deg=0.0,
        strain_std=0.0,
        strain_volume_preserve=True,
        occlusion_mode="none",
        occlusion_view="second",
        occlusion_slab_frac=0.0,
        occlusion_cone_deg=0.0,
        input_dim=24,
        invariant_mode="norms",
        invariant_max_factor=4.0,
        invariant_groups=0,
        invariant_use_third_order=True,
        invariant_eps=1e-6,
        queue_size=queue_size,
        symmetric=True,
        normalize_embeddings=True,
    )


def test_pointcontrast_forward_produces_finite_loss_and_metrics():
    torch.manual_seed(7)
    loss_mod = _make_loss(queue_size=0)
    loss_mod.train()

    encoder = _DummyEncoder(num_points=16, latent_dim=24)
    pc = torch.randn(6, 16, 3)

    loss, metrics = loss_mod.compute_loss(
        pc=pc,
        encoder=encoder,
        prepare_input=lambda x: x,
        split_output=lambda enc_out: enc_out,
        current_epoch=0,
    )

    assert loss is not None
    assert torch.isfinite(loss).item(), f"loss is non-finite: {loss}"
    assert "pointcontrast_ce_ab" in metrics
    assert "pointcontrast_pos_sim" in metrics
    assert "pointcontrast_neg_sim" in metrics


def test_pointcontrast_queue_updates_when_enabled():
    torch.manual_seed(11)
    loss_mod = _make_loss(queue_size=10)
    loss_mod.train()

    encoder = _DummyEncoder(num_points=16, latent_dim=24)
    pc = torch.randn(4, 16, 3)

    _, metrics = loss_mod.compute_loss(
        pc=pc,
        encoder=encoder,
        prepare_input=lambda x: x,
        split_output=lambda enc_out: enc_out,
        current_epoch=0,
    )

    assert int(loss_mod._queue_filled.item()) > 0
    assert "pointcontrast_queue_fill" in metrics
    qfill = float(metrics["pointcontrast_queue_fill"].item())
    assert 0.0 <= qfill <= 1.0


def test_pointcontrast_from_config_uses_pointcontrast_fields_and_data_defaults():
    cfg = SimpleNamespace(
        pointcontrast_enabled=True,
        pointcontrast_weight=1.0,
        pointcontrast_temperature=0.2,
        pointcontrast_embed_dim=64,
        pointcontrast_start_epoch=0,
        pointcontrast_jitter_std=0.0,
        pointcontrast_jitter_mode="absolute",
        pointcontrast_drop_ratio=0.0,
        pointcontrast_neighbor_view=False,
        pointcontrast_neighbor_view_mode="both",
        pointcontrast_neighbor_k=8,
        pointcontrast_neighbor_max_relative_distance=0.0,
        pointcontrast_view_crop_mode="random",
        pointcontrast_drop_apply_to_both=True,
        pointcontrast_rotation_mode="none",
        pointcontrast_rotation_deg=0.0,
        pointcontrast_strain_std=0.0,
        pointcontrast_strain_volume_preserve=True,
        pointcontrast_occlusion_mode="none",
        pointcontrast_occlusion_view="second",
        pointcontrast_occlusion_slab_frac=0.0,
        pointcontrast_occlusion_cone_deg=0.0,
        pointcontrast_invariant_mode="norms",
        pointcontrast_invariant_max_factor=4.0,
        pointcontrast_invariant_groups=0,
        pointcontrast_invariant_use_third_order=True,
        pointcontrast_invariant_eps=1e-6,
        pointcontrast_queue_size=0,
        pointcontrast_symmetric=True,
        pointcontrast_normalize_embeddings=True,
        data=SimpleNamespace(model_points=12, num_points=16),
    )

    loss_mod = PointContrastLoss.from_config(cfg, input_dim=20)
    assert loss_mod.enabled is True
    assert loss_mod._view_sampler.view_points == 12
    assert loss_mod.embed_dim == 64
