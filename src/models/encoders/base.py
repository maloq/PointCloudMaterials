from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Encoder(ABC, nn.Module):
    expects_channel_first: bool = False
    invariant_dim: int | None = None
    equivariant_dim: int | None = None

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """Return encoder features in the repository's latent convention."""

