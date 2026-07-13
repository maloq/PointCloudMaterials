from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn as nn


class Encoder(ABC, nn.Module):
    input_layout: Literal["bn3", "b3n"] = "bn3"
    output_contract: Literal[
        "invariant",
        "invariant_aux",
        "invariant_equivariant",
    ]
    invariant_dim: int | None = None
    equivariant_dim: int | None = None

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """Return encoder features in the repository's latent convention."""
