from __future__ import annotations
import math, torch
import torch.nn as nn
import torch.nn.functional as F

from ..base     import Decoder
from ..registry import register_decoder


class _TransposedBatchNormRelu(nn.Module):
    """BN + ReLU for (B,N,C) by transposing to (B,C,N) for BatchNorm1d."""

    def __init__(self, num_features: int, dropout_rate: float = 0.0):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B,N,C)
        x = x.transpose(1, 2)                                # (B,C,N)
        x = F.relu(self.bn(x))
        x = x.transpose(1, 2)                                # (B,N,C)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

# ---------------------------------------------------------------------------
#  Two‑step Folding decoder (original width=512)
# ---------------------------------------------------------------------------
@register_decoder("Folding")
class FoldingDecoderTwoStep(Decoder):
    def __init__(self, num_points: int, latent_size: int, hidden_dim: int = 512, dropout_rate: float = 0.0):
        super().__init__()
        self._n = num_points

        # Fixed 2‑D grid template
        side = int(math.ceil(math.sqrt(num_points)))
        xs, ys = torch.linspace(-1, 1, side), torch.linspace(-1, 1, side)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        grid = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)[:num_points]
        self.register_buffer("grid", grid)                  # (N,2)

        def make_stage(in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                _TransposedBatchNormRelu(hidden_dim, dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                _TransposedBatchNormRelu(hidden_dim, dropout_rate),
                nn.Linear(hidden_dim, 3),
                _TransposedBatchNormRelu(3),
                nn.Linear(3, 3),
            )

        self.stage1 = make_stage(latent_size + 2)
        self.stage2 = make_stage(latent_size + 3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:     # (B,L)
        B = z.size(0)
        z_exp = z.unsqueeze(1).expand(B, self._n, -1)       # (B,N,L)
        g     = self.grid.unsqueeze(0).expand(B, -1, -1)    # (B,N,2)

        coarse   = self.stage1(torch.cat((z_exp, g), dim=-1))
        refined  = self.stage2(torch.cat((z_exp, coarse), dim=-1))
        return refined


# ---------------------------------------------------------------------------
#  Smaller Folding decoder  (hidden‑dim 256)
# ---------------------------------------------------------------------------
@register_decoder("FoldingSmall")
class FoldingDecoderTwoStepSmall(FoldingDecoderTwoStep):
    def __init__(self, num_points: int, latent_size: int, hidden_dim: int = 256, dropout_rate: float = 0.0):
        super().__init__(num_points, latent_size, hidden_dim, dropout_rate)