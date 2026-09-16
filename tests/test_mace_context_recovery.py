"""Protect the scientific distinction between the two joint-training variants."""

import torch

from src.research.mace_context import recovery_train
from src.research.mace_context.recovery_data import PhysicalHeads


def test_control_physical_heads_do_not_backpropagate_into_encoder(monkeypatch):
    monkeypatch.setattr(recovery_train, 'loss_from_features', lambda model, z: (z.square().mean(), {}))
    torch.manual_seed(42)
    heads = PhysicalHeads(512, 8)
    target = torch.randn(2, 292)
    mean, scale, weights = torch.zeros(512), torch.ones(512), torch.ones(292)
    config = {'joint': {'physics_weight': 10.}}
    gradients = {}
    for variant in ['dual_ssl', 'dual_physics']:
        z = torch.ones(8, 512, requires_grad=True)
        heads.zero_grad(set_to_none=True)
        loss, _ = recovery_train.joint_loss(config, None, heads, z, target, mean, scale, weights, variant)
        loss.backward()
        gradients[variant] = z.grad.clone()
        assert sum(float(p.grad.square().sum()) for p in heads.parameters()) > 0
    assert torch.count_nonzero(gradients['dual_ssl'][6:]) == 0
    assert torch.count_nonzero(gradients['dual_physics'][6:, :256]) > 0
    assert torch.count_nonzero(gradients['dual_physics'][6:, 256:]) > 0
    torch.testing.assert_close(gradients['dual_ssl'][:6], gradients['dual_physics'][:6])
