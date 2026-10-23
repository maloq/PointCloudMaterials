import torch
import torch.nn as nn


class NormInvariantHead(nn.Module):
    """Reduce (B, C, 3) equivariants to channel norms, or pass through invariants."""

    def __init__(self, channels: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.channels = int(channels)
        self.mode = "norms"
        self.output_dim = self.channels
        self.eps = float(eps)

    def forward(
        self, inv_z: torch.Tensor | None, eq_z: torch.Tensor | None
    ) -> torch.Tensor | None:
        if eq_z is not None:
            return torch.sqrt((eq_z * eq_z).sum(dim=-1) + self.eps)
        return inv_z
