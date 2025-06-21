from __future__ import annotations
import torch
import torch.nn as nn

from ..base import Decoder
from ..registry import register_decoder


@register_decoder("MLP")
class MLPDecoder(Decoder):
    """Latent vector → 3‑layer MLP → (B,N,3)."""

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