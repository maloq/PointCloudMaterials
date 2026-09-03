import os
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.training_methods.base_ssl_module import BaseSSLModule
from src.training_methods.contrastive_learning.vicreg import VICRegLoss


def test_midpoint_factor_vae_projector_is_context_invariant_in_eval() -> None:
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("configs")):
        cfg = compose(config_name="vicreg_geo_frame_multiscale_factor_vae_midpoint_no_noise")

    assert cfg.vicreg_projector_bn_eval_batch_stats is False
    vicreg = VICRegLoss.from_config(cfg, input_dim=8).eval()
    batch_norm_layers = [
        layer for layer in vicreg.projector if isinstance(layer, torch.nn.BatchNorm1d)
    ]
    assert len(batch_norm_layers) == 2
    assert all(type(layer) is torch.nn.BatchNorm1d for layer in batch_norm_layers)

    generator = torch.Generator().manual_seed(13)
    anchors = torch.randn(2, 8, generator=generator)
    low_context = torch.randn(14, 8, generator=generator) - 5.0
    high_context = torch.randn(14, 8, generator=generator) + 5.0

    low_context_output = vicreg.projector(torch.cat([anchors, low_context], dim=0))[:2]
    high_context_output = vicreg.projector(torch.cat([anchors, high_context], dim=0))[:2]

    torch.testing.assert_close(
        low_context_output,
        high_context_output,
        rtol=0.0,
        atol=0.0,
    )


def test_eval_export_rejects_batch_dependent_projector() -> None:
    module = SimpleNamespace(
        training=False,
        representation_source="vicreg_projector",
        vicreg=SimpleNamespace(
            projector=torch.nn.Identity(),
            projector_bn_eval_batch_stats=True,
        ),
    )

    with pytest.raises(RuntimeError, match="batch companions"):
        BaseSSLModule._output_representation(module, torch.randn(2, 8))
