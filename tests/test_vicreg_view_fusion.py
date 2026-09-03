from __future__ import annotations

import torch

from src.training_methods.contrastive_learning.vicreg_module import VICRegModule
from src.utils.pointcloud_ops import crop_to_num_points


class _StaticViewAugmenter:
    @staticmethod
    def _resolve_neighbor_flags(*, device) -> tuple[bool, bool]:
        del device
        return False, False

    @staticmethod
    def _resolve_pair_occlusion_flags(
        *,
        use_neighbor_a: bool,
        use_neighbor_b: bool,
        device,
    ) -> tuple[bool, bool]:
        del use_neighbor_a, use_neighbor_b, device
        return False, False

    @staticmethod
    def _augment(
        points: torch.Tensor,
        *,
        use_neighbor: bool,
        apply_occlusion: bool,
        view_points: int | None,
    ) -> torch.Tensor:
        assert not use_neighbor
        assert not apply_occlusion
        if view_points is not None:
            points = crop_to_num_points(points, view_points)
        return points + 0.01 * torch.randn_like(points)


def test_static_view_pair_shares_crop_without_changing_augmented_views() -> None:
    points = torch.randn(5, 17, 3)
    augmenter = _StaticViewAugmenter()

    torch.manual_seed(29)
    reference_a = augmenter._augment(
        points,
        use_neighbor=False,
        apply_occlusion=False,
        view_points=9,
    )
    reference_b = augmenter._augment(
        points,
        use_neighbor=False,
        apply_occlusion=False,
        view_points=9,
    )

    module = object.__new__(VICRegModule)
    torch.nn.Module.__init__(module)
    module.vicreg = augmenter
    torch.manual_seed(29)
    views = module._build_contrastive_view_pair(points, view_points=9)

    torch.testing.assert_close(views["y_a"], reference_a, atol=0, rtol=0)
    torch.testing.assert_close(views["y_b"], reference_b, atol=0, rtol=0)
