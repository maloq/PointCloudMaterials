import torch
import torch.nn.functional as F
from typing import Literal, Tuple


def neighbor_pair_latent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    distances: torch.Tensor,
    *,
    weight: Literal["binary", "gaussian"] = "binary",
    sigma: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """Compute a pairwise latent smoothness loss between matched latents.

    Args:
        z1: Tensor of shape (B, D) – latents for anchor samples.
        z2: Tensor of shape (B, D) – latents for neighbor samples.
        distances: Tensor of shape (B,) – Euclidean distances between sample centers.
        weight: 'binary' (all edges weight 1) or 'gaussian' (exp(-d^2/(2*sigma^2))).
        sigma: Gaussian bandwidth when weight='gaussian'. Must be > 0.

    Returns:
        (loss, stats) where loss is a scalar tensor and stats is a dict with diagnostics.
    """
    if z1.ndim != 2 or z2.ndim != 2:
        raise ValueError("z1 and z2 must be 2D tensors (B, D)")
    if z1.shape != z2.shape:
        raise ValueError(f"z1 and z2 must have same shape, got {z1.shape} vs {z2.shape}")

    if distances.ndim != 1 or distances.shape[0] != z1.shape[0]:
        raise ValueError("distances must be 1D tensor of length B")

    # Squared latent distances per pair
    d_lat2 = torch.sum((z1 - z2) ** 2, dim=1)  # (B,)

    if weight == "binary":
        w = torch.ones_like(d_lat2)
    elif weight == "gaussian":
        if sigma <= 0:
            raise ValueError("sigma must be positive for gaussian weighting")
        w = torch.exp(-(distances ** 2) / (2.0 * (sigma ** 2)))
    else:
        raise ValueError(f"Unknown weight mode: {weight}")

    # Avoid division by zero
    eps = 1e-8
    loss = torch.sum(w * d_lat2) / (torch.sum(w) + eps)

    stats = {
        "pair_weight_mean": torch.mean(w).detach(),
        "pair_weight_min": torch.min(w).detach(),
        "pair_weight_max": torch.max(w).detach(),
        "latent_pair_dist_mean": torch.mean(torch.sqrt(d_lat2 + eps)).detach(),
    }
    return loss, stats

