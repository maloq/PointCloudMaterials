from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.training_methods.temporal_ssl.temporal_ssl_module import SwAVLoss


def _make_swav_loss(*, sinkhorn_iterations: int = 5) -> SwAVLoss:
    cfg = OmegaConf.create(
        {
            "swav_enabled": True,
            "swav_weight": 0.5,
            "swav_projection_dim": 8,
            "swav_hidden_dim": 0,
            "swav_num_prototypes": 5,
            "swav_temperature": 0.2,
            "swav_epsilon": 0.05,
            "swav_sinkhorn_iterations": sinkhorn_iterations,
            "swav_start_epoch": 0,
            "swav_freeze_prototypes_steps": 2,
            "swav_view_mode": "center_adjacent",
            "swav_view_points": None,
        }
    )
    return SwAVLoss.from_config(cfg, input_dim=6)


def test_swav_sinkhorn_balances_assignments() -> None:
    torch.manual_seed(0)
    swav = _make_swav_loss(sinkhorn_iterations=50)
    logits = 0.1 * torch.randn(7, swav.num_prototypes)

    assignments = swav._sinkhorn(logits)

    assert assignments.shape == logits.shape
    torch.testing.assert_close(
        assignments.sum(dim=1),
        torch.ones(logits.shape[0]),
        atol=1e-5,
        rtol=1e-5,
    )
    expected_proto_mass = torch.full((swav.num_prototypes,), logits.shape[0] / swav.num_prototypes)
    torch.testing.assert_close(
        assignments.sum(dim=0),
        expected_proto_mass,
        atol=5e-3,
        rtol=5e-3,
    )


def test_swav_loss_is_finite_and_backpropagates() -> None:
    torch.manual_seed(1)
    swav = _make_swav_loss()
    view_a = torch.randn(7, 6, requires_grad=True)
    view_b = torch.randn(7, 6, requires_grad=True)

    loss, metrics = swav.compute_loss(
        view_features=[view_a, view_b],
        current_epoch=0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert "swav_assignment_entropy" in metrics
    assert "swav_assignment_max_prob" in metrics
    assert view_a.grad is not None
    assert view_b.grad is not None
    assert swav.prototypes.weight.grad is not None
