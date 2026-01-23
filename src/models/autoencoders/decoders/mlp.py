from __future__ import annotations
import torch
import torch.nn as nn

from ..base import Decoder
from ..registry import register_decoder


@register_decoder("MLP")
class MLPDecoder(Decoder):
    """Latent vector → 3‑layer MLP → (B,N,3)."""
    use_invariant_latent = True

    def __init__(self, num_points: int, latent_size: int):
        super().__init__()
        self._n = num_points
        self.net = nn.Sequential(
            nn.Linear(latent_size, 512), nn.ReLU(inplace=True), nn.BatchNorm1d(512),
            nn.Linear(512, 256),         nn.ReLU(inplace=True), nn.BatchNorm1d(256),
            nn.Linear(256, num_points * 3),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(z.size(0), self._n, 3)


@register_decoder("MLPLarge")
class MLPDecoderLarge(Decoder):
    """
    Larger MLP decoder for better reconstruction - good for baseline testing.
    
    Takes invariant latent vector (B, latent_size) and produces point cloud (B, N, 3).
    """
    use_invariant_latent = True

    def __init__(
        self,
        num_points: int,
        latent_size: int,
        hidden_dims: tuple = (1024, 1024, 512, 256),
        use_batchnorm: bool = True,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self._n = num_points
        
        layers = []
        in_dim = latent_size
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, num_points * 3))
        self.net = nn.Sequential(*layers)
        
        # Initialize final layer with small weights for stability
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Invariant latent (B, latent_size)
            
        Returns:
            Point cloud (B, num_points, 3)
        """
        return self.net(z).view(z.size(0), self._n, 3)