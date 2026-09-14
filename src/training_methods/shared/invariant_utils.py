import warnings

import torch
import torch.nn as nn


class NormInvariantHead(nn.Module):
    """Reduce equivariant latents to per-channel norms, or pass through invariants."""

    def __init__(self, channels: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.channels = max(0, int(channels))
        self.mode = "norms"
        self.output_dim = self.channels
        self.eps = float(eps)

    def _coerce_eq_latent(self, eq_z: torch.Tensor | None) -> torch.Tensor | None:
        if eq_z is None:
            return None
        if eq_z.dim() == 3:
            if eq_z.shape[-1] == 3:
                return eq_z
            if eq_z.shape[1] == 3:
                warnings.warn(
                    f"Transposing eq_z from shape {tuple(eq_z.shape)} to (B, C, 3). "
                    "Ensure encoder output convention is correct.",
                )
                return eq_z.transpose(1, 2).contiguous()
            raise ValueError(
                f"Cannot coerce 3D eq_z with shape {tuple(eq_z.shape)} to (B, C, 3): "
                "neither last dim nor dim-1 equals 3."
            )
        if eq_z.dim() == 4 and eq_z.shape[-1] == 3:
            if eq_z.shape[1] == self.channels:
                return eq_z.mean(dim=2)
            if eq_z.shape[2] == self.channels:
                return eq_z.mean(dim=1)
            raise ValueError(
                f"Cannot coerce 4D eq_z with shape {tuple(eq_z.shape)}: "
                f"no dimension matches channels={self.channels}."
            )
        raise ValueError(
            f"Unsupported eq_z shape {tuple(eq_z.shape)} (dim={eq_z.dim()}). "
            "Expected 3D (B, C, 3) or 4D (B, C, ?, 3)."
        )

    def _fit_output_dim(self, feat: torch.Tensor | None) -> torch.Tensor | None:
        if feat is None:
            return None
        if feat.dim() > 2:
            feat = feat.reshape(feat.shape[0], -1)
        if feat.dim() != 2:
            raise ValueError(
                f"Expected 2D feature tensor after reshape, got dim={feat.dim()} "
                f"shape={tuple(feat.shape)}."
            )
        if feat.shape[1] != self.output_dim:
            raise ValueError(
                f"Feature dimension mismatch: got {feat.shape[1]}, "
                f"expected output_dim={self.output_dim}."
            )
        return feat

    def _norms(self, eq_z: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((eq_z * eq_z).sum(dim=-1) + self.eps)

    def forward(self, inv_z: torch.Tensor | None, eq_z: torch.Tensor | None) -> torch.Tensor | None:
        if eq_z is None and inv_z is not None and inv_z.dim() == 3 and inv_z.shape[-1] == 3:
            eq_z = inv_z
            inv_z = None

        eq = self._coerce_eq_latent(eq_z)
        if eq is not None:
            return self._fit_output_dim(self._norms(eq))

        if inv_z is not None:
            return self._fit_output_dim(inv_z)
        return None
