from __future__ import annotations
from typing import Literal, Optional
import torch
import torch.nn.functional as F


def _pairwise_squared_distances(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, N, C) point clouds
    returns: (B, N, N) squared Euclidean distances
    """
    # (x^2) sum per point
    x2 = (x * x).sum(dim=-1, keepdim=True)            # (B, N, 1)
    # D^2 = ||xi||^2 + ||xj||^2 - 2 xi·xj
    D2 = x2 + x2.transpose(1, 2) - 2.0 * torch.bmm(x, x.transpose(1, 2))
    D2 = D2.clamp_min_(0.0)                           # numerical safety
    # zero exact diagonal (stability + correctness)
    B, N, _ = D2.shape
    idx = torch.arange(N, device=D2.device)
    D2[:, idx, idx] = 0.0
    return D2


def _zscore(D: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Per-sample z-score of a (B, N, N) matrix, including off/diagonal.
    """
    mu = D.mean(dim=(1, 2), keepdim=True)
    sd = D.std(dim=(1, 2), keepdim=True).clamp_min(eps)
    return (D - mu) / sd


def _make_mask(N: int, device, dtype, mask_diagonal: bool, upper_only: bool) -> torch.Tensor:
    """
    Build a (1, N, N) mask with 1s for entries to keep, 0s for entries to ignore.
    """
    M = torch.ones((N, N), device=device, dtype=dtype)
    if mask_diagonal:
        i = torch.arange(N, device=device)
        M[i, i] = 0
    if upper_only:
        M = torch.triu(M, diagonal=1)  # keep strictly upper triangle
    return M.unsqueeze(0)  # (1, N, N)


@torch.jit.script
def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean over last two dims with mask. mask is broadcastable to x.
    """
    x = x * mask
    denom = mask.sum(dim=(-1, -2)).clamp_min(1.0)  # (B,)
    return (x.sum(dim=(-1, -2)) / denom).mean()    # scalar


def pairwise_distance_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    squared: bool = False,
    normalize: bool = True,
    mask_diagonal: bool = True,
    upper_only: bool = True,
    loss_type: Literal["l2", "smooth_l1"] = "l2",
    beta: float = 0.02,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """
    Compare (B, N, 3) point clouds by matching their pairwise distance matrices.

    - If `normalize=True`, both distance matrices are z-scored per sample, which
      makes the loss scale-invariant and reduces sensitivity to dataset radius.
    - `upper_only=True` avoids double counting symmetrical pairs and is faster.
    - `mask_diagonal=True` ignores trivial self-distances.
    - `loss_type="smooth_l1"` is a robust alternative to L2; tune `beta` to your scale.

    Returns a scalar by default (`reduction="mean"`). If `reduction="none"`, returns
    a (B,) tensor of per-sample losses.
    """
    if pred.dim() != 3 or target.dim() != 3:
        raise ValueError("pred and target must be (B, N, C)")

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    B, N, C = pred.shape
    device = pred.device
    dtype = pred.dtype

    Dp2 = _pairwise_squared_distances(pred)
    Dt2 = _pairwise_squared_distances(target)

    Dp = Dp2 if squared else Dp2.sqrt()
    Dt = Dt2 if squared else Dt2.sqrt()

    if normalize:
        Dp = _zscore(Dp)
        Dt = _zscore(Dt)

    if loss_type == "l2":
        loss_mat = (Dp - Dt) ** 2
    elif loss_type == "smooth_l1":
        loss_mat = F.smooth_l1_loss(Dp, Dt, reduction="none", beta=beta)
    else:
        raise ValueError("loss_type must be 'l2' or 'smooth_l1'")

    mask = _make_mask(N, device, dtype=loss_mat.dtype, mask_diagonal=mask_diagonal, upper_only=upper_only)

    if reduction == "mean":
        return _masked_mean(loss_mat, mask)
    elif reduction == "sum":
        return (loss_mat * mask).sum()
    elif reduction == "none":
        # Per-sample masked mean
        denom = mask.sum(dim=(-1, -2)).clamp_min(1.0)             # (1,)
        per_sample = (loss_mat * mask).sum(dim=(-1, -2)) / denom  # (B,)
        return per_sample
    else:
        raise ValueError("reduction must be 'mean' | 'sum' | 'none'")
