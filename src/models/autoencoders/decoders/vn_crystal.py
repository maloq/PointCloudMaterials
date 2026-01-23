from __future__ import annotations

import math
import torch
import torch.nn as nn

from ..base import Decoder
from ..registry import register_decoder
from ..encoders.vn_encoders import VNLinear, VNLinearLeakyReLU, VNBatchNorm

@register_decoder("VN_Crystal")
class VNCrystalDecoder(Decoder):
    use_invariant_latent = False
    """
    Vector Neuron Equivariant Crystal Decoder with optional correction for imperfect crystals.
    
    Takes equivariant latent representation (B, latent_channels, 3) and generates
    a crystal lattice structure (B, num_points, 3) while preserving SO(3) equivariance.
    
    Correction Modes:
        - 'none': Perfect crystal (no corrections)
        - 'fixed': Learnable fractional coordinate offsets (same for all samples)
        - 'latent': Latent-conditioned corrections (sample-dependent)
        - 'hybrid': Fixed offsets scaled by latent-derived factor
    
    Args:
        num_points: Number of output points
        latent_channels: Number of VN channels in input (NOT latent_size, but latent_size//3 from encoder)
        hidden_dims: Hidden layer dimensions for VN processing
        use_batchnorm: Whether to use VN batch normalization
        negative_slope: LeakyReLU negative slope
        correction_mode: Type of correction ('none', 'fixed', 'latent', 'hybrid')
        correction_scale: Initial scale for corrections (prevents large initial deviations)
        correction_hidden_dim: Hidden dimension for latent-conditioned correction network
    """
    
    CORRECTION_MODES = ('none', 'fixed', 'latent', 'hybrid')
    
    def __init__(
        self,
        num_points: int,
        latent_size: int,  # This should match encoder's latent_size // 3
        hidden_dims: tuple[int, ...] = (64, 32),
        use_batchnorm: bool = True,
        negative_slope: float = 0.1,
        correction_mode: str = 'none',
        correction_scale: float = 0.1,
        correction_hidden_dim: int = 64,
    ):
        super().__init__()
        
        if correction_mode not in self.CORRECTION_MODES:
            raise ValueError(f"correction_mode must be one of {self.CORRECTION_MODES}, got '{correction_mode}'")
        
        self.num_points = num_points
        self.latent_channels = latent_size//3
        self.correction_mode = correction_mode
        self.correction_scale = correction_scale
        
        # Store config
        self.config = {
            'num_points': num_points,
            'latent_size': latent_size,
            'latent_channels': latent_size//3,
            'hidden_dims': hidden_dims,
            'correction_mode': correction_mode,
            'correction_scale': correction_scale,
        }
        
        # Pre-calculate a fixed integer grid
        grid_dim = math.ceil(num_points ** (1/3)) + 1  # +1 to ensure enough points
        x = torch.arange(grid_dim, dtype=torch.float32)
        y = torch.arange(grid_dim, dtype=torch.float32)
        z = torch.arange(grid_dim, dtype=torch.float32)
        
        grid = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1).reshape(-1, 3)
        grid = grid - grid.mean(dim=0)  # Center around origin
        
        # Sort by distance from origin to get a compact shape (approx sphere)
        # This prevents "flat" shapes when num_points is small compared to grid_dim^3
        dists = torch.norm(grid, dim=-1)
        _, idx = torch.sort(dists)
        grid = grid[idx]
        
        # Register as buffer (fixed, not learnable)
        self.register_buffer('base_grid', grid[:num_points, :])
        
        # VN Layers to process latent code
        layers = []
        c_in = self.latent_channels
        
        for h_dim in hidden_dims:
            h = max(1, h_dim // 3)  # Ensure at least 1 channel
            layers.append(VNLinearLeakyReLU(
                c_in, h, dim=3,
                negative_slope=negative_slope,
                use_batchnorm=use_batchnorm
            ))
            c_in = h
        
        self.vn_layers = nn.Sequential(*layers)
        self.vn_final = VNLinear(c_in, 3)  # Output 3 basis vectors
        
        # Correction mechanisms
        self._init_correction_layers(correction_hidden_dim, self.latent_channels)
    
    def _init_correction_layers(self, hidden_dim: int, latent_channels: int):
        """Initialize correction layers based on mode."""
        
        if self.correction_mode == 'none':
            return
        
        if self.correction_mode in ('fixed', 'hybrid'):
            # Learnable fractional coordinate offsets
            # Initialized to zero (no initial correction)
            self.correction_offsets = nn.Parameter(
                torch.zeros(self.num_points, 3)
            )
        
        if self.correction_mode in ('latent', 'hybrid'):
            # Network to extract invariant features from equivariant latent
            # Input: latent norms (B, latent_channels) + pairwise dots (B, latent_channels)
            invariant_dim = latent_channels * 2
            
            if self.correction_mode == 'latent':
                # Full latent-conditioned corrections
                # Predict per-point fractional offsets from invariants
                self.correction_net = nn.Sequential(
                    nn.Linear(invariant_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.LeakyReLU(0.1),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(0.1),
                    nn.Linear(hidden_dim, self.num_points * 3),
                )
                # Initialize last layer to near-zero
                nn.init.zeros_(self.correction_net[-1].weight)
                nn.init.zeros_(self.correction_net[-1].bias)
            else:
                # Hybrid: predict per-point scales for fixed offsets
                self.scale_net = nn.Sequential(
                    nn.Linear(invariant_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.LeakyReLU(0.1),
                    nn.Linear(hidden_dim, self.num_points),
                    nn.Softplus(),  # Ensure positive scales
                )
                # Initialize to output ~1.0
                nn.init.zeros_(self.scale_net[-2].weight)
                nn.init.zeros_(self.scale_net[-2].bias)
    
    def _extract_invariants(self, z: torch.Tensor) -> torch.Tensor:
        """
        Extract rotation-invariant features from equivariant latent.
        
        Args:
            z: Equivariant latent (B, C, 3)
            
        Returns:
            Invariant features (B, C*2)
        """
        # Norms of each vector (B, C)
        norms = torch.norm(z, dim=-1)
        
        # Mean pairwise dot products (simplified)
        # z @ z.T over the 3D dimension gives (B, C, C)
        # We take the mean over one dimension for (B, C)
        dots = torch.einsum('bci,bci->bc', z, z) / (norms + 1e-8)
        
        return torch.cat([norms, dots], dim=-1)
    
    def _compute_corrections(self, z: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        """
        Compute equivariant correction vectors.
        
        The corrections are computed in fractional coordinates (invariant),
        then transformed to Cartesian by multiplying with basis vectors (equivariant).
        
        Args:
            z: Equivariant latent (B, C, 3)
            basis: Predicted basis vectors (B, 3, 3)
            
        Returns:
            Correction vectors (B, N, 3) - equivariant
        """
        B = z.shape[0]
        
        if self.correction_mode == 'none':
            return torch.zeros(B, self.num_points, 3, device=z.device, dtype=z.dtype)
        
        if self.correction_mode == 'fixed':
            # Fixed fractional offsets, transformed by basis
            # corrections_frac: (N, 3) -> (B, N, 3)
            corrections_frac = self.correction_offsets.unsqueeze(0).expand(B, -1, -1)
            corrections_cart = torch.bmm(corrections_frac, basis) * self.correction_scale
            return corrections_cart
        
        if self.correction_mode == 'latent':
            # Latent-conditioned fractional offsets
            invariants = self._extract_invariants(z)  # (B, C*2)
            corrections_frac = self.correction_net(invariants)  # (B, N*3)
            corrections_frac = corrections_frac.view(B, self.num_points, 3)
            corrections_cart = torch.bmm(corrections_frac, basis) * self.correction_scale
            return corrections_cart
        
        if self.correction_mode == 'hybrid':
            # Fixed offsets with latent-conditioned per-point scales
            invariants = self._extract_invariants(z)  # (B, C*2)
            scales = self.scale_net(invariants)  # (B, N)
            
            # Scale the fixed offsets per-point
            # (N, 3) * (B, N, 1) -> (B, N, 3)
            corrections_frac = self.correction_offsets.unsqueeze(0) * scales.unsqueeze(-1)
            corrections_cart = torch.bmm(corrections_frac, basis) * self.correction_scale
            return corrections_cart
        
        raise ValueError(f"Unknown correction mode: {self.correction_mode}")
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            z: Equivariant latent (B, latent_channels, 3)
            
        Returns:
            Point cloud (B, num_points, 3)
        """
        B = z.shape[0]
        
        # 1. Predict Basis Vectors (Equivariant)
        x = self.vn_layers(z)
        basis = self.vn_final(x)  # (B, 3, 3)
        
        # 2. Generate Ideal Lattice Points
        base = self.base_grid.unsqueeze(0).expand(B, -1, -1)  # (B, N, 3)
        pts_ideal = torch.bmm(base, basis)  # (B, N, 3)
        
        # 3. Apply Corrections for Imperfect Crystals
        corrections = self._compute_corrections(z, basis)  # (B, N, 3)
        pts = pts_ideal + corrections
        
        return pts
    
    def forward_with_components(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass returning intermediate components for analysis.
        
        Returns:
            Dictionary with:
                - 'points': Final point cloud (B, N, 3)
                - 'points_ideal': Ideal lattice without corrections (B, N, 3)
                - 'corrections': Correction vectors (B, N, 3)
                - 'basis': Predicted basis vectors (B, 3, 3)
        """
        B = z.shape[0]
        
        x = self.vn_layers(z)
        basis = self.vn_final(x)
        
        base = self.base_grid.unsqueeze(0).expand(B, -1, -1)
        pts_ideal = torch.bmm(base, basis)
        
        corrections = self._compute_corrections(z, basis)
        pts = pts_ideal + corrections
        
        return {
            'points': pts,
            'points_ideal': pts_ideal,
            'corrections': corrections,
            'basis': basis,
        }
    
    def get_correction_magnitude(self) -> torch.Tensor:
        """Get the magnitude of learned corrections (for monitoring)."""
        if self.correction_mode == 'none':
            return torch.tensor(0.0)
        
        if hasattr(self, 'correction_offsets'):
            return torch.norm(self.correction_offsets, dim=-1).mean() * self.correction_scale
        
        return torch.tensor(0.0)
    
    def get_num_params(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  config={self.config},\n"
            f"  num_params={self.get_num_params():,}\n"
            f")"
        )

__all__ = [
    "VNCrystalDecoder",
]
