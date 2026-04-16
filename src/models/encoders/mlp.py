from __future__ import annotations
import torch
import torch.nn as nn

from .base import Encoder
from .registry import register_encoder


@register_encoder("MLP")
class MLPEncoder(Encoder):
    """Flatten points → 3‑layer MLP → latent vector."""

    def __init__(self, num_points: int, latent_size: int):
        super().__init__()
        self._n = num_points
        self.invariant_dim = int(latent_size)
        self.net = nn.Sequential(
            nn.Linear(num_points * 3, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256),            nn.ReLU(inplace=True),
            nn.Linear(256, latent_size),
        )

    def forward(self, x: torch.Tensor):
        latent = self.net(x.reshape(x.size(0), -1))
        return latent, None, None
