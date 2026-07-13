from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from src.training_methods.contrastive_learning.vicreg import VICRegLoss


def _vicreg_with_mirror_probability(probability: float) -> VICRegLoss:
    cfg = OmegaConf.create(
        {
            "vicreg_enabled": True,
            "vicreg_weight": 1.0,
            "vicreg_embed_dim": 3,
            "vicreg_projector_mode": "identity",
            "vicreg_mirror_prob": probability,
            "vicreg_jitter_std": 0.0,
            "vicreg_drop_ratio": 0.0,
            "vicreg_rotation_mode": "none",
            "vicreg_strain_std": 0.0,
            "vicreg_occlusion_mode": "none",
        }
    )
    return VICRegLoss.from_config(cfg, input_dim=3)


def test_vicreg_view_mirror_flips_exactly_one_axis_and_preserves_distances() -> None:
    vicreg = _vicreg_with_mirror_probability(1.0)
    points = torch.arange(1, 1 + 4 * 6 * 3, dtype=torch.float32).reshape(4, 6, 3)

    mirrored = vicreg.apply_view_postprocessing(
        points,
        use_neighbor=False,
        apply_occlusion=False,
        view_points=None,
    )

    changed_dimensions = (mirrored != points).any(dim=1)
    torch.testing.assert_close(
        changed_dimensions.sum(dim=1),
        torch.ones((points.shape[0],), dtype=torch.int64),
    )
    torch.testing.assert_close(mirrored.abs(), points.abs())
    torch.testing.assert_close(torch.cdist(mirrored, mirrored), torch.cdist(points, points))


def test_vicreg_view_mirror_probability_zero_is_identity() -> None:
    vicreg = _vicreg_with_mirror_probability(0.0)
    points = torch.randn(3, 8, 3)

    transformed = vicreg.apply_view_postprocessing(
        points,
        use_neighbor=False,
        apply_occlusion=False,
        view_points=None,
    )

    torch.testing.assert_close(transformed, points)


def test_vicreg_rejects_invalid_mirror_probability() -> None:
    with pytest.raises(ValueError, match="vicreg_mirror_prob must be in \\[0, 1\\]"):
        _vicreg_with_mirror_probability(1.01)


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/vicreg_vn_molecular.yaml",
        "configs/vicreg_vn_molecular_multi.yaml",
        "configs/vicreg_geo_frame_multi.yaml",
        "configs/vicreg_mace_molecular.yaml",
        "configs/vicreg_nequip_molecular.yaml",
        "configs/line_jepa_geo_frame_simple.yaml",
        "configs/line_jepa_static_hybrid.yaml",
    ],
)
def test_active_vicreg_configs_enable_mirror_views(config_path: str) -> None:
    cfg = OmegaConf.load(config_path)
    assert float(cfg.vicreg_mirror_prob) == 0.5
