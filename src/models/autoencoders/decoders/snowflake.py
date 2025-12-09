from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base     import Decoder
from ..registry import register_decoder

class SnowflakePointDeconv(nn.Module):
    """
    Snowflake Point Deconvolution (SPD) module.
    Splits each parent point into 'k' child points by predicting offsets.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, up_factor: int = 4):
        super().__init__()
        self.up_factor = up_factor
        
        # MLP to predict offsets for the split points
        # Input: [Point_Feats + Global_Z]
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim * up_factor) # Predict offsets for k children
        )

    def forward(self, p_prev: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p_prev: (B, N_prev, 3) Parent point cloud
            z:      (B, L)         Global latent vector
        Returns:
            p_next: (B, N_prev * k, 3) Child point cloud
        """
        B, N_prev, _ = p_prev.shape
        
        # 1. Expand global Z to match number of parent points
        z_expand = z.unsqueeze(1).expand(B, N_prev, -1) # (B, N, L)
        
        # 2. Concatenate parent coordinates with global features as context
        #    (In full SnowflakeNet, this usually includes per-point features, 
        #     but for a pure decoder interface, we use geometry + global z)
        feat = torch.cat([p_prev, z_expand], dim=-1)    # (B, N, 3+L)
        
        # 3. Process with MLP (Flatten B,N for BatchNorm1d)
        feat_flat = feat.reshape(B * N_prev, -1)
        deltas = self.net(feat_flat)                    # (B*N, 3*k)
        
        # 4. Reshape deltas to (B, N, k, 3)
        deltas = deltas.reshape(B, N_prev, self.up_factor, 3)
        
        # 5. Expand parent points to (B, N, k, 3) and add deltas
        p_expanded = p_prev.unsqueeze(2).expand(B, N_prev, self.up_factor, 3)
        p_next = p_expanded + deltas
        
        # 6. Flatten to final point list (B, N*k, 3)
        return p_next.reshape(B, -1, 3)


@register_decoder("Snowflake")
class SnowflakeDecoder(Decoder):
    """
    SnowflakeNet Decoder adapted for the standard Decoder interface.
    Generates points in a coarse-to-fine manner:
    Seed -> (Up 1) -> (Up 2) -> ... -> Target N
    """
    def __init__(
        self, 
        num_points: int, 
        latent_size: int, 
        hidden_dim: int = 512, 
        up_factor: int = 4
    ):
        super().__init__()
        self.num_points = num_points
        self.up_factor = up_factor

        # Determine the number of points at the coarse 'seed' stage
        # We assume num_points is divisible by up_factor.
        # Example: 2048 points, up_factor=4 -> Seed size = 512
        if num_points % up_factor != 0:
            raise ValueError(f"num_points ({num_points}) must be divisible by up_factor ({up_factor})")
        
        self.seed_points = num_points // up_factor
        
        # --- Stage 1: Seed Generator (Z -> Coarse Cloud) ---
        # Projects latent Z into a coarse shape (N_seed, 3)
        self.fc_seed = nn.Linear(latent_size, self.seed_points * 3)
        
        # --- Stage 2: Snowflake Point Deconvolution (Coarse -> Fine) ---
        # Input to SPD is (3 coords + latent_size)
        self.spd = SnowflakePointDeconv(
            in_dim=3 + latent_size,
            hidden_dim=hidden_dim,
            out_dim=3,
            up_factor=up_factor
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        
        # 1. Generate Seed Points
        # (B, L) -> (B, N_seed * 3) -> (B, N_seed, 3)
        coarse = self.fc_seed(z).reshape(B, self.seed_points, 3)
        
        # 2. Snowflake Up-sampling
        # (B, N_seed, 3) -> (B, N_full, 3)
        fine = self.spd(coarse, z)
        
        return fine


@register_decoder("SnowflakeDeep")
class SnowflakeDecoderDeep(Decoder):
    """
    A deeper 3-stage Snowflake Decoder for higher fidelity.
    Ideal for larger point clouds (e.g. 2048 or 4096).
    Flow: Seed(128) -> x4 -> Coarse(512) -> x4 -> Fine(2048)
    """
    def __init__(
        self, 
        num_points: int, 
        latent_size: int, 
        hidden_dim: int = 512
    ):
        super().__init__()
        # Hardcoded 2-step upsampling (x4 then x4 = x16 total)
        # Verify num_points is divisible by 16
        if num_points % 16 != 0:
             raise ValueError(f"SnowflakeDeep requires num_points divisible by 16 (e.g. 2048), got {num_points}")

        self.seed_n = num_points // 16
        
        # Stage 1: Seed
        self.fc_seed = nn.Linear(latent_size, self.seed_n * 3)
        
        # Stage 2: First Split (x4)
        self.spd1 = SnowflakePointDeconv(
            in_dim=3 + latent_size,
            hidden_dim=hidden_dim,
            out_dim=3,
            up_factor=4
        )
        
        # Stage 3: Second Split (x4)
        self.spd2 = SnowflakePointDeconv(
            in_dim=3 + latent_size,
            hidden_dim=hidden_dim,
            out_dim=3,
            up_factor=4
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        
        # 1. Seed
        p0 = self.fc_seed(z).reshape(B, self.seed_n, 3)
        
        # 2. Up 1
        p1 = self.spd1(p0, z)
        
        # 3. Up 2
        p2 = self.spd2(p1, z)
        
        return p2