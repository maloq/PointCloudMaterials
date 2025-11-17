from __future__ import annotations

import math
import torch
import torch.nn as nn

from ..base import Decoder
from ..registry import register_decoder
from ..encoders.vn import VNLinear, VNLinearLeakyReLU, VNBatchNorm


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

        # latent_size is already the number of VN channels (not total dimension)
        # Input eq_z has shape (B, latent_size, 3)
        c_in = latent_size
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


@register_decoder("VN_Equivariant_Grid")
class VNEquivariantGridDecoder(Decoder):
    """
    Vector Neuron Equivariant Decoder with Grid-based generation.

    Similar to folding decoder but uses VN layers to maintain equivariance.
    Combines a 2D grid template with equivariant features to generate points.
    """

    def __init__(
        self,
        num_points: int,
        latent_size: int,
        hidden_dim: int = 512,
        use_batchnorm: bool = True,
        negative_slope: float = 0.1,
    ):
        super().__init__()
        self._n = num_points

        # Create 2D grid template (rotation-invariant)
        side = int(math.ceil(math.sqrt(num_points)))
        xs, ys = torch.linspace(-1, 1, side), torch.linspace(-1, 1, side)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        grid = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)[:num_points]
        self.register_buffer("grid", grid)  # (N, 2)

        # latent_size is already the number of VN channels
        c_latent = latent_size
        c_hidden = max(1, hidden_dim // 3)

        # Grid embedding to VN space (per-point)
        self.grid_embed = nn.Linear(2, 3)  # Lift 2D grid to 3D

        # VN layers for processing
        # Input will be per-point: latent + grid_embed
        self.vn1 = VNLinearLeakyReLU(
            c_latent + 1, c_hidden, dim=4,  # dim=4 for per-point features
            negative_slope=negative_slope,
            use_batchnorm=use_batchnorm
        )
        self.vn2 = VNLinearLeakyReLU(
            c_hidden, c_hidden // 2, dim=4,
            negative_slope=negative_slope,
            use_batchnorm=use_batchnorm
        )
        self.vn3 = VNLinearLeakyReLU(
            c_hidden // 2, 1, dim=4,  # Reduce to 1 VN channel per point
            negative_slope=negative_slope,
            use_batchnorm=use_batchnorm
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Equivariant latent (B, latent_size, 3)

        Returns:
            Point cloud (B, num_points, 3)
        """
        B = z.size(0)

        # Expand latent to all points
        z_exp = z.unsqueeze(-1).expand(B, -1, 3, self._n)  # (B, c_latent, 3, N)

        # Embed grid to 3D VN features
        grid = self.grid.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)
        grid_3d = self.grid_embed(grid)  # (B, N, 3)
        grid_vn = grid_3d.transpose(1, 2).unsqueeze(1)  # (B, 1, 3, N)

        # Concatenate latent and grid features
        x = torch.cat([z_exp, grid_vn], dim=1)  # (B, c_latent+1, 3, N)

        # Process through VN layers
        x = self.vn1(x)  # (B, c_hidden, 3, N)
        x = self.vn2(x)  # (B, c_hidden//2, 3, N)
        x = self.vn3(x)  # (B, 1, 3, N)

        # Remove channel dimension and transpose to (B, N, 3)
        x = x.squeeze(1).transpose(1, 2)  # (B, N, 3)

        return x


@register_decoder("VN_Equivariant_Folding")
class VNEquivariantFoldingDecoder(Decoder):
    """
    Vector Neuron Equivariant Folding Decoder.

    Two-stage folding decoder using VN layers to maintain equivariance.
    First stage produces coarse shape, second stage refines it.
    """

    def __init__(
        self,
        num_points: int,
        latent_size: int,
        hidden_dim: int = 512,
        use_batchnorm: bool = True,
        negative_slope: float = 0.1,
    ):
        super().__init__()
        self._n = num_points

        # Create 2D grid template
        side = int(math.ceil(math.sqrt(num_points)))
        xs, ys = torch.linspace(-1, 1, side), torch.linspace(-1, 1, side)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        grid = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)[:num_points]
        self.register_buffer("grid", grid)  # (N, 2)

        # latent_size is already the number of VN channels
        c_latent = latent_size
        c_hidden = max(1, hidden_dim // 3)

        # Grid embedding
        self.grid_embed = nn.Linear(2, 3)

        # Stage 1: grid + latent -> coarse shape
        self.stage1_vn1 = VNLinearLeakyReLU(
            c_latent + 1, c_hidden, dim=4,
            negative_slope=negative_slope,
            use_batchnorm=use_batchnorm
        )
        self.stage1_vn2 = VNLinearLeakyReLU(
            c_hidden, c_hidden // 2, dim=4,
            negative_slope=negative_slope,
            use_batchnorm=use_batchnorm
        )
        self.stage1_out = VNLinear(c_hidden // 2, 1)

        # Stage 2: coarse + latent -> refined shape
        self.stage2_vn1 = VNLinearLeakyReLU(
            c_latent + 1, c_hidden, dim=4,
            negative_slope=negative_slope,
            use_batchnorm=use_batchnorm
        )
        self.stage2_vn2 = VNLinearLeakyReLU(
            c_hidden, c_hidden // 2, dim=4,
            negative_slope=negative_slope,
            use_batchnorm=use_batchnorm
        )
        self.stage2_out = VNLinear(c_hidden // 2, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Equivariant latent (B, latent_size, 3)

        Returns:
            Point cloud (B, num_points, 3)
        """
        B = z.size(0)

        # Expand latent to all points
        z_exp = z.unsqueeze(-1).expand(B, -1, 3, self._n)  # (B, c_latent, 3, N)

        # Embed grid
        grid = self.grid.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)
        grid_3d = self.grid_embed(grid)  # (B, N, 3)
        grid_vn = grid_3d.transpose(1, 2).unsqueeze(1)  # (B, 1, 3, N)

        # Stage 1: Coarse reconstruction
        x1 = torch.cat([z_exp, grid_vn], dim=1)  # (B, c_latent+1, 3, N)
        x1 = self.stage1_vn1(x1)
        x1 = self.stage1_vn2(x1)
        coarse = self.stage1_out(x1)  # (B, 1, 3, N)

        # Stage 2: Refinement using coarse output
        x2 = torch.cat([z_exp, coarse], dim=1)  # (B, c_latent+1, 3, N)
        x2 = self.stage2_vn1(x2)
        x2 = self.stage2_vn2(x2)
        refined = self.stage2_out(x2)  # (B, 1, 3, N)

        # Remove channel dimension and transpose
        output = refined.squeeze(1).transpose(1, 2)  # (B, N, 3)

        return output


__all__ = [
    "VNEquivariantDecoder",
    "VNEquivariantGridDecoder",
    "VNEquivariantFoldingDecoder",
]
