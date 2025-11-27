from __future__ import annotations

import math
import torch
import torch.nn as nn

from ..base import Decoder
from ..registry import register_decoder
from ..encoders.vn_encoders import VNLinear, VNLinearLeakyReLU, VNBatchNorm


@register_decoder("VN_Equivariant")
class VNEquivariantDecoder(Decoder):
    """
    Vector Neuron Equivariant Decoder.

    Takes equivariant latent representation (B, latent_size, 3) and generates
    a point cloud (B, num_points, 3) while preserving SO(3) equivariance.

    Architecture:
    - Progressively expands VN channels through multiple layers
    - Uses VN layers to maintain equivariance throughout
    - Final layer projects to individual point coordinates
    """

    def __init__(
        self,
        num_points: int,
        latent_size: int,
        hidden_dims: tuple[int, int, int] = (512, 256, 128),
        use_batchnorm: bool = True,
        negative_slope: float = 0.1,
    ):
        super().__init__()
        self._n = num_points

        # latent_size is the total dimension, so we divide by 3 to get VN channels
        # Input eq_z has shape (B, latent_size // 3, 3)
        c_in = latent_size // 3
        h1, h2, h3 = [max(1, dim // 3) for dim in hidden_dims]

        # Expansion layers with VN
        self.vn1 = VNLinearLeakyReLU(
            c_in, h1, dim=3,
            negative_slope=negative_slope,
            use_batchnorm=use_batchnorm
        )
        self.vn2 = VNLinearLeakyReLU(
            h1, h2, dim=3,
            negative_slope=negative_slope,
            use_batchnorm=use_batchnorm
        )
        self.vn3 = VNLinearLeakyReLU(
            h2, h3, dim=3,
            negative_slope=negative_slope,
            use_batchnorm=use_batchnorm
        )

        # Final projection to num_points
        # Each VN channel will produce one 3D point
        self.vn_final = VNLinear(h3, num_points)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Equivariant latent (B, latent_size, 3)

        Returns:
            Point cloud (B, num_points, 3)
        """
        # z shape: (B, latent_size, 3)
        # Progressively expand through VN layers
        x = self.vn1(z)  # (B, h1, 3)
        x = self.vn2(x)  # (B, h2, 3)
        x = self.vn3(x)  # (B, h3, 3)

        # Final projection to num_points
        x = self.vn_final(x)  # (B, num_points, 3)

        return x


__all__ = [
    "VNEquivariantDecoder",
]
