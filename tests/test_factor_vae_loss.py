from types import SimpleNamespace

import pytest
import torch

from src.training_methods.contrastive_learning.vicreg_module import FactorVAELoss


def _factor_vae_loss() -> FactorVAELoss:
    return FactorVAELoss(
        enabled=True,
        gamma=7.0,
        input_dim=4,
        hidden_dim=16,
        num_hidden_layers=2,
    )


def test_dimension_permutation_preserves_every_empirical_marginal() -> None:
    torch.manual_seed(3)
    z = torch.arange(32, dtype=torch.float32).reshape(8, 4)

    permuted = FactorVAELoss.permute_dimensions(z)

    assert permuted.shape == z.shape
    for dimension in range(z.shape[1]):
        torch.testing.assert_close(
            permuted[:, dimension].sort().values,
            z[:, dimension].sort().values,
        )


def test_tc_and_discriminator_losses_have_disjoint_gradient_targets() -> None:
    torch.manual_seed(5)
    factor_vae = _factor_vae_loss()
    z_a = torch.randn(12, 4, requires_grad=True)
    z_b = torch.randn(12, 4, requires_grad=True)

    total_correlation, discriminator_loss, metrics = factor_vae.compute_loss(
        z_a=z_a,
        z_b=z_b,
    )
    total_correlation.backward()

    assert z_a.grad is not None and torch.isfinite(z_a.grad).all()
    assert z_b.grad is not None and torch.isfinite(z_b.grad).all()
    assert all(
        parameter.grad is None
        for parameter in factor_vae.discriminator.parameters()
    )

    z_a.grad = None
    z_b.grad = None
    discriminator_loss.backward()

    assert z_a.grad is None
    assert z_b.grad is None
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in factor_vae.discriminator.parameters()
    )
    assert set(metrics) == {
        "factor_vae_tc_weighted",
        "factor_vae_discriminator",
        "factor_vae_discriminator_accuracy",
        "factor_vae_discriminator_joint_logit_gap",
        "factor_vae_discriminator_factorized_logit_gap",
        "factor_vae_discriminator_logit_separation",
    }
    assert 0.0 <= metrics["factor_vae_discriminator_accuracy"].item() <= 1.0


def test_factor_vae_rejects_a_single_sample_batch() -> None:
    factor_vae = _factor_vae_loss()

    with pytest.raises(ValueError, match="at least two samples"):
        factor_vae.compute_loss(
            z_a=torch.randn(1, 4),
            z_b=torch.randn(1, 4),
        )


def test_discriminator_uses_independent_batch_halves() -> None:
    factor_vae = _factor_vae_loss()
    z_a = torch.arange(48, dtype=torch.float32).reshape(12, 4)
    z_b = z_a + 100.0
    discriminator_inputs = []
    original_discriminator_logits = factor_vae._discriminator_logits

    def record_discriminator_input(z, *, train_discriminator):
        discriminator_inputs.append(z.detach().clone())
        return original_discriminator_logits(
            z,
            train_discriminator=train_discriminator,
        )

    factor_vae._discriminator_logits = record_discriminator_input
    factor_vae.permute_dimensions = lambda z: z

    factor_vae.discriminator_loss(z_a=z_a, z_b=z_b)

    split_index = z_a.shape[0] // 2
    torch.testing.assert_close(
        discriminator_inputs[0],
        torch.cat([z_a[:split_index], z_b[:split_index]], dim=0),
    )
    torch.testing.assert_close(
        discriminator_inputs[1],
        torch.cat([z_a[split_index:], z_b[split_index:]], dim=0),
    )


def test_disabled_factor_vae_allocates_no_discriminator_parameters() -> None:
    factor_vae = FactorVAELoss.from_config(
        SimpleNamespace(factor_vae_enabled=False),
        input_dim=4,
    )

    assert factor_vae.discriminator is None
    assert sum(parameter.numel() for parameter in factor_vae.parameters()) == 0


def test_enabled_factor_vae_requires_positive_gamma() -> None:
    with pytest.raises(ValueError, match="factor_vae_gamma must be > 0"):
        FactorVAELoss(
            enabled=True,
            gamma=0.0,
            input_dim=4,
            hidden_dim=16,
            num_hidden_layers=2,
        )


def test_factor_vae_rejects_negative_latent_noise() -> None:
    with pytest.raises(ValueError, match="factor_vae_latent_noise_std must be >= 0"):
        FactorVAELoss(
            enabled=True,
            gamma=1.0,
            input_dim=4,
            hidden_dim=16,
            num_hidden_layers=2,
            latent_noise_std=-0.1,
        )


def test_latent_noise_preserves_encoder_gradient_flow() -> None:
    factor_vae = FactorVAELoss(
        enabled=True,
        gamma=1.0,
        input_dim=4,
        hidden_dim=16,
        num_hidden_layers=2,
        latent_noise_std=1.0,
    )
    z_a = torch.randn(12, 4, requires_grad=True)
    z_b = torch.randn(12, 4, requires_grad=True)

    factor_vae.total_correlation_loss(z_a=z_a, z_b=z_b).backward()

    assert z_a.grad is not None and torch.isfinite(z_a.grad).all()
    assert z_b.grad is not None and torch.isfinite(z_b.grad).all()


def test_factor_vae_delays_and_linearly_warms_up_gamma() -> None:
    factor_vae = FactorVAELoss(
        enabled=True,
        gamma=0.5,
        input_dim=4,
        hidden_dim=16,
        num_hidden_layers=2,
        start_epoch=3,
        gamma_warmup_epochs=4,
    )

    assert not factor_vae.should_run(current_epoch=2)
    assert factor_vae.effective_gamma(current_epoch=2) == 0.0
    assert factor_vae.effective_gamma(current_epoch=3) == pytest.approx(0.125)
    assert factor_vae.effective_gamma(current_epoch=4) == pytest.approx(0.25)
    assert factor_vae.effective_gamma(current_epoch=6) == pytest.approx(0.5)
    assert factor_vae.effective_gamma(current_epoch=20) == pytest.approx(0.5)


def test_factor_vae_calibrates_discriminator_before_gamma_warmup() -> None:
    factor_vae = FactorVAELoss(
        enabled=True,
        gamma=5.0,
        input_dim=4,
        hidden_dim=16,
        num_hidden_layers=2,
        start_epoch=2,
        discriminator_warmup_epochs=2,
        gamma_warmup_epochs=5,
    )

    assert not factor_vae.should_run(current_epoch=1)
    assert factor_vae.should_run(current_epoch=2)
    assert factor_vae.effective_gamma(current_epoch=2) == 0.0
    assert factor_vae.effective_gamma(current_epoch=3) == 0.0
    assert factor_vae.effective_gamma(current_epoch=4) == pytest.approx(1.0)
    assert factor_vae.effective_gamma(current_epoch=8) == pytest.approx(5.0)


def test_random_coordinate_groups_cover_every_coordinate_once() -> None:
    factor_vae = FactorVAELoss(
        enabled=True,
        gamma=1.0,
        input_dim=8,
        hidden_dim=16,
        num_hidden_layers=2,
        discriminator_coordinate_group_size=2,
        discriminator_bottleneck_dim=1,
    )

    coordinate_groups = factor_vae._sample_coordinate_groups(device=torch.device("cpu"))

    assert coordinate_groups.shape == (4, 2)
    torch.testing.assert_close(
        coordinate_groups.flatten().sort().values,
        torch.arange(8),
    )
    linear_layers = [
        layer
        for layer in factor_vae.discriminator
        if isinstance(layer, torch.nn.Linear)
    ]
    assert linear_layers[0].in_features == 2
    assert linear_layers[0].out_features == 1


def test_grouped_discriminator_uses_the_same_partition_for_both_classes() -> None:
    factor_vae = FactorVAELoss(
        enabled=True,
        gamma=1.0,
        input_dim=8,
        hidden_dim=16,
        num_hidden_layers=2,
        discriminator_coordinate_group_size=2,
    )
    z_a = torch.randn(12, 8)
    z_b = torch.randn(12, 8)
    discriminator_inputs = []
    original_discriminator_logits = factor_vae._discriminator_logits

    def record_discriminator_input(z, *, train_discriminator):
        discriminator_inputs.append(z.detach().clone())
        return original_discriminator_logits(
            z,
            train_discriminator=train_discriminator,
        )

    factor_vae._discriminator_logits = record_discriminator_input
    factor_vae.discriminator_loss(z_a=z_a, z_b=z_b)

    assert [value.shape for value in discriminator_inputs] == [
        torch.Size([48, 2]),
        torch.Size([48, 2]),
    ]


def test_spectral_norm_is_applied_without_mutating_state_in_encoder_loss() -> None:
    factor_vae = FactorVAELoss(
        enabled=True,
        gamma=1.0,
        input_dim=8,
        hidden_dim=16,
        num_hidden_layers=2,
        discriminator_spectral_norm=True,
        discriminator_coordinate_group_size=2,
        discriminator_bottleneck_dim=1,
    )
    z_a = torch.randn(12, 8, requires_grad=True)
    z_b = torch.randn(12, 8, requires_grad=True)
    buffers_before = {
        name: value.detach().clone()
        for name, value in factor_vae.discriminator.named_buffers()
    }

    factor_vae.total_correlation_loss(z_a=z_a, z_b=z_b).backward()

    linear_layers = [
        layer
        for layer in factor_vae.discriminator
        if isinstance(layer, torch.nn.Linear)
    ]
    assert linear_layers
    assert all("weight" in layer.parametrizations for layer in linear_layers)
    assert z_a.grad is not None and torch.isfinite(z_a.grad).all()
    assert z_b.grad is not None and torch.isfinite(z_b.grad).all()
    assert all(
        parameter.grad is None
        for parameter in factor_vae.discriminator.parameters()
    )
    for name, value in factor_vae.discriminator.named_buffers():
        torch.testing.assert_close(value, buffers_before[name])


def test_coordinate_group_size_must_divide_embedding_dimension() -> None:
    with pytest.raises(ValueError, match="must divide"):
        FactorVAELoss(
            enabled=True,
            gamma=1.0,
            input_dim=8,
            hidden_dim=16,
            num_hidden_layers=2,
            discriminator_coordinate_group_size=3,
        )
