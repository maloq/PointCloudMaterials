from __future__ import annotations

import math

import geomloss
import torch


def chamfer_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    point_reduction: str = "mean",
) -> tuple[torch.Tensor, None]:
    """Pure-PyTorch symmetric Chamfer distance for batched point clouds."""
    pred = _to_b_n_3(pred)
    target = _to_b_n_3(target)

    pred_sq = (pred**2).sum(-1, keepdim=True)
    target_sq = (target**2).sum(-1).unsqueeze(1)
    dists2 = pred_sq + target_sq - 2.0 * torch.bmm(pred, target.transpose(1, 2))
    dists = torch.sqrt(dists2.clamp_min(0.0) + 1e-8)

    min_pred_to_target = dists.min(dim=2)[0]
    min_target_to_pred = dists.min(dim=1)[0]

    if point_reduction == "sum":
        distance = min_pred_to_target.sum(1) + min_target_to_pred.sum(1)
    elif point_reduction == "mean":
        distance = min_pred_to_target.mean(1) + min_target_to_pred.mean(1)
    else:
        raise ValueError(
            "point_reduction must be one of {'mean', 'sum'}, "
            f"got {point_reduction!r}."
        )
    return distance.mean(), None


def sinkhorn_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    blur: float = 0.02,
    p: int = 2,
    scaling: float = 0.5,
) -> tuple[torch.Tensor, None]:
    """Batched Sinkhorn distance between point clouds using geomloss."""
    pred = _to_b_n_3(pred)
    target = _to_b_n_3(target)

    blur = _coerce_positive_float(blur)
    scaling = _coerce_float(scaling)
    if not (0.0 < scaling < 1.0):
        raise ValueError(f"scaling must be in (0, 1), got {scaling}.")

    loss = geomloss.SamplesLoss(
        loss="sinkhorn",
        p=int(p),
        blur=blur,
        scaling=scaling,
        diameter=_estimate_diameter(pred, target, blur, eps=1e-6),
        backend="tensorized",
    )
    return loss(pred, target).mean(), None


def _to_b_n_3(point_cloud: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(point_cloud):
        raise TypeError(f"point_cloud must be a torch.Tensor, got {type(point_cloud)!r}.")
    if point_cloud.dim() != 3:
        raise ValueError(
            "point_cloud must have shape (B, N, 3) or (B, 3, N), "
            f"got {tuple(point_cloud.shape)}."
        )
    if point_cloud.shape[-1] == 3:
        return point_cloud
    if point_cloud.shape[1] == 3:
        return point_cloud.transpose(1, 2).contiguous()
    raise ValueError(
        "point_cloud must have exactly one coordinate dimension of size 3, "
        f"got {tuple(point_cloud.shape)}."
    )


def _coerce_float(value: float) -> float:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(
                f"Expected scalar tensor for float conversion, got shape {tuple(value.shape)}."
            )
        value = value.item()
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Expected a finite float, got {value!r}.") from exc
    if not math.isfinite(out):
        raise ValueError(f"Expected a finite float, got {out!r}.")
    return out


def _coerce_positive_float(value: float) -> float:
    out = _coerce_float(value)
    if out <= 0.0:
        raise ValueError(f"Expected a positive float, got {out}.")
    return out


def _estimate_diameter(
    pred: torch.Tensor,
    target: torch.Tensor,
    blur: float,
    *,
    eps: float,
) -> float:
    mins = torch.minimum(pred.amin(dim=1), target.amin(dim=1))
    maxs = torch.maximum(pred.amax(dim=1), target.amax(dim=1))
    diag = torch.linalg.norm(maxs - mins, dim=-1)
    diameter = float(diag.max().item())
    if not math.isfinite(diameter) or diameter <= 0.0:
        return max(float(blur), float(eps))
    return max(diameter, float(blur), float(eps))


__all__ = ["chamfer_distance", "sinkhorn_distance"]
