from __future__ import annotations

import pytest
import torch

from src.training_methods.contrastive_learning.vicreg import VICRegLoss


def _vicreg_with_mirror_probability(probability: float) -> VICRegLoss:
    return VICRegLoss(
        enabled=True,
        weight=1.0,
        sim_coeff=25.0,
        std_coeff=25.0,
        cov_coeff=1.0,
        embed_dim=3,
        start_epoch=0,
        jitter_std=0.0,
        jitter_mode="absolute",
        jitter_scale=1.0,
        drop_ratio=0.0,
        view_points=None,
        neighbor_view=False,
        neighbor_view_mode="none",
        neighbor_k=4,
        neighbor_max_relative_distance=0.0,
        drop_apply_to_both=True,
        rotation_mode="none",
        rotation_deg=0.0,
        mirror_prob=probability,
        strain_std=0.0,
        strain_volume_preserve=True,
        occlusion_mode="none",
        occlusion_view="second",
        occlusion_slab_frac=0.2,
        occlusion_cone_deg=20.0,
        occlusion_prob=1.0,
        std_eps=1e-4,
        std_target=1.0,
        input_dim=3,
        projector_mode="identity",
    )


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
