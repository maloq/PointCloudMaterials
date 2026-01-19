from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from ..base import Decoder
from ..registry import register_decoder
from ..encoders.vn_encoders import VNLinear, VNLinearLeakyReLU, VNBatchNorm
from .snowflake_vn import VNSnowflakeDecoder as VNSnowflakeDecoderCore


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

        # Input eq_z has shape (B, latent_size, 3) where latent_size is the number of VN channels
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
            z: Equivariant latent (B, latent_size, 3) where latent_size is the number of VN channels

        Returns:
            Point cloud (B, num_points, 3)
        """
        # z shape: (B, latent_size, 3) - latent_size VN channels, each being a 3D vector
        # Progressively expand through VN layers
        x = self.vn1(z)  # (B, h1, 3)
        x = self.vn2(x)  # (B, h2, 3)
        x = self.vn3(x)  # (B, h3, 3)

        # Final projection to num_points
        x = self.vn_final(x)  # (B, num_points, 3)

        return x


@register_decoder("VN_Snowflake")
class VNSnowflakeDecoderWrapper(Decoder):
    """
    SO(3)-Equivariant Snowflake Point Decoder.

    Adapts the VNSnowflakeDecoder for use with the standard equivariant autoencoder
    interface. Takes an equivariant latent representation (B, latent_size, 3) and 
    generates a point cloud (B, num_points, 3) while preserving SO(3) equivariance.

    Architecture:
    - Learns seed point initialization from the latent
    - Progressively upsamples through VN-equivariant deconvolution blocks
    - Uses attention-based context aggregation at each stage
    """

    def __init__(
        self,
        num_points: int,
        latent_size: int,
        num_seeds: int = 16,
        hidden_channels: int = 64,
        stages: Tuple[Tuple[int, int, int, int], ...] | None = None,
        k: int = 8,
        use_batchnorm: bool = True,
        negative_slope: float = 0.1,
    ):
        """
        Args:
            num_points: Number of output points to generate.
            latent_size: Number of VN channels in the input latent (B, latent_size, 3).
            num_seeds: Number of seed points to initialize (must divide num_points by power of 2).
            hidden_channels: Hidden VN channel dimension for seed features.
            stages: Tuple of (c_ctx, c_child, up_factor, disp_hidden) per stage.
                    If None, auto-computed based on num_seeds and num_points.
            k: Number of neighbors for kNN in attention blocks.
            use_batchnorm: Whether to use batch normalization in initial projection.
            negative_slope: Negative slope for LeakyReLU activations.
        """
        super().__init__()
        self._n = num_points
        self.num_seeds = num_seeds
        self.latent_size = latent_size
        self.hidden_channels = hidden_channels

        # Auto-compute stages if not provided
        if stages is None:
            stages = self._compute_stages(num_seeds, num_points, hidden_channels)
        
        # Validate that stages produce the right number of points
        total_up = 1
        for stage in stages:
            total_up *= stage[2]  # up_factor
        expected_points = num_seeds * total_up
        if expected_points != num_points:
            raise ValueError(
                f"Stages produce {expected_points} points (seeds={num_seeds} * {total_up}), "
                f"but num_points={num_points}. Adjust num_seeds or stages."
            )

        # Project latent to seed positions: (B, latent_size, 3) -> (B, num_seeds, 3)
        # We use a VN linear to maintain equivariance
        self.seed_pos_proj = VNLinear(latent_size, num_seeds)

        # Project latent to initial vector features: (B, latent_size, 3) -> (B, num_seeds, hidden_channels, 3)
        # First expand channels, then reshape
        self.seed_feat_proj = VNLinearLeakyReLU(
            latent_size, num_seeds * hidden_channels, 
            dim=3,
            negative_slope=negative_slope,
            use_batchnorm=use_batchnorm
        )

        # Core snowflake decoder (upsampling stages)
        c_in = hidden_channels
        self.decoder_core = VNSnowflakeDecoderCore(
            c_in=c_in,
            stages=stages,
            k=k,
        )

        # Final projection to single vector channel (output point positions)
        # Output of decoder_core is (B, N, c_out, 3), we want (B, N, 3)
        self.final_proj = VNLinear(self.decoder_core.out_c, 1)

    @staticmethod
    def _compute_stages(
        num_seeds: int, 
        num_points: int, 
        hidden_channels: int
    ) -> Tuple[Tuple[int, int, int, int], ...]:
        """
        Auto-compute stages configuration based on seed count and target points.
        
        Returns tuple of (c_ctx, c_child, up_factor, disp_hidden) per stage.
        """
        import math
        
        ratio = num_points / num_seeds
        if ratio < 1:
            raise ValueError(f"num_points ({num_points}) must be >= num_seeds ({num_seeds})")
        
        # Find number of stages needed (assuming up_factor=2 per stage)
        num_stages = max(1, int(math.ceil(math.log2(ratio))))
        
        # Verify we can achieve exact point count
        total_up = 2 ** num_stages
        if num_seeds * total_up != num_points:
            # Fallback: try different up factors
            # For simplicity, use equal up factors when possible
            pass  # Continue with the approximation
        
        stages = []
        c_in = hidden_channels
        
        for i in range(num_stages):
            c_ctx = min(c_in * 2, 128)  # Context channels
            c_child = c_ctx  # Child channels (keep same)
            up_factor = 2  # Standard upsampling factor
            disp_hidden = c_ctx  # Displacement MLP hidden dim
            
            stages.append((c_ctx, c_child, up_factor, disp_hidden))
            c_in = c_child
        
        return tuple(stages)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Equivariant latent (B, latent_size, 3) where latent_size is the number of VN channels.

        Returns:
            Point cloud (B, num_points, 3)
        """
        B = z.shape[0]
        
        # Generate seed positions (equivariant)
        # z: (B, latent_size, 3) -> (B, num_seeds, 3)
        x_seeds = self.seed_pos_proj(z)  # (B, num_seeds, 3)

        # Generate seed vector features (equivariant)
        # z: (B, latent_size, 3) -> (B, num_seeds * hidden_channels, 3)
        v_flat = self.seed_feat_proj(z)  # (B, num_seeds * hidden_channels, 3)
        # Reshape to (B, num_seeds, hidden_channels, 3)
        v_seeds = v_flat.view(B, self.num_seeds, self.hidden_channels, 3)

        # Run snowflake upsampling
        x_out, v_out = self.decoder_core(x_seeds, v_seeds)  # (B, num_points, c_out, 3)

        # Project vector features to final positions
        # v_out: (B, num_points, c_out, 3) -> need to add displacements
        # Option 1: Use x_out directly
        # Option 2: Add learned displacement from v_out
        
        # Project v_out to single channel and add as displacement
        # v_out: (B, N, c_out, 3) -> transpose to (B, c_out, 3, N) for VNLinear? 
        # Actually VNLinear in vn_encoders expects (B, C, 3) or (B, C, 3, N)
        # But our v_out is (B, N, C, 3)
        
        # Reshape for final projection: (B, N, c_out, 3) -> (B*N, c_out, 3)
        N = x_out.shape[1]
        v_reshaped = v_out.view(B * N, self.decoder_core.out_c, 3)  # (B*N, c_out, 3)
        displacement = self.final_proj(v_reshaped)  # (B*N, 1, 3)
        displacement = displacement.view(B, N, 3)  # (B, N, 3)

        # Final output: seed-derived positions + learned displacement
        output = x_out + displacement

        return output


__all__ = [
    "VNEquivariantDecoder",
    "VNSnowflakeDecoderWrapper",
]
