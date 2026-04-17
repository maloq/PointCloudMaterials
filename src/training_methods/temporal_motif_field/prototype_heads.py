from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PrototypeOutput:
    logits: torch.Tensor
    probs: torch.Tensor
    recon: torch.Tensor
    hard_assign: torch.Tensor


class SoftPrototypeBank(nn.Module):
    def __init__(
        self,
        *,
        num_prototypes: int,
        dim: int,
        temperature: float,
    ) -> None:
        super().__init__()
        self.num_prototypes = int(num_prototypes)
        self.dim = int(dim)
        self.temperature = float(temperature)

        if self.num_prototypes <= 0:
            raise ValueError(f"num_prototypes must be > 0, got {self.num_prototypes}.")
        if self.dim <= 0:
            raise ValueError(f"dim must be > 0, got {self.dim}.")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}.")

        self.prototypes = nn.Parameter(torch.empty(self.num_prototypes, self.dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.prototypes, mean=0.0, std=self.dim ** -0.5)

    def forward(self, z: torch.Tensor) -> PrototypeOutput:
        if z.ndim != 2 or int(z.shape[1]) != self.dim:
            raise ValueError(
                "SoftPrototypeBank expects z with shape (B, D). "
                f"Expected D={self.dim}, got shape={tuple(z.shape)}."
            )

        distances_sq = self.squared_distances(z)
        logits = -distances_sq / self.temperature
        probs = torch.softmax(logits, dim=-1)
        recon = probs @ self.prototypes
        hard_assign = probs.argmax(dim=-1)
        return PrototypeOutput(
            logits=logits,
            probs=probs,
            recon=recon,
            hard_assign=hard_assign,
        )

    def squared_distances(self, z: torch.Tensor) -> torch.Tensor:
        z_sq = z.square().sum(dim=-1, keepdim=True)
        p_sq = self.prototypes.square().sum(dim=-1).unsqueeze(0)
        cross = z @ self.prototypes.transpose(0, 1)
        return (z_sq + p_sq - 2.0 * cross).clamp_min(0.0)
