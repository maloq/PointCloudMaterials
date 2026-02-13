import torch
from typing import Optional


def crop_to_num_points(
    points: torch.Tensor,
    num_points: Optional[int],
    *,
    mode: str = "nearest_origin",
) -> torch.Tensor:
    """Select a fixed-size subset of points."""
    if num_points is None:
        return points
    num_points = int(num_points)
    if num_points <= 0:
        return points
    if points.dim() != 3 or points.size(-1) != 3:
        raise ValueError(f"points must have shape (B, N, 3), got {tuple(points.shape)}")
    bsz, n_pts, _ = points.shape
    if num_points == n_pts:
        return points
    if num_points > n_pts:
        raise ValueError(f"num_points ({num_points}) cannot exceed input points ({n_pts})")
    mode_norm = str(mode).strip().lower()
    if mode_norm in {"nearest", "nearest_origin", "center"}:
        dist2 = (points.float() ** 2).sum(dim=-1)
        idx = dist2.topk(k=num_points, dim=1, largest=False).indices
    elif mode_norm in {"random", "uniform"}:
        # Random subset without replacement per sample.
        rand = torch.rand((bsz, n_pts), device=points.device)
        idx = rand.topk(k=num_points, dim=1, largest=False).indices
    else:
        raise ValueError(f"Unsupported crop mode {mode!r}")
    idx = idx.unsqueeze(-1).expand(-1, -1, 3)
    return points.gather(1, idx)


def shift_to_neighbor(
    points: torch.Tensor,
    *,
    neighbor_k: int = 8,
    max_relative_distance: Optional[float] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Shift point cloud so a random near-origin point becomes the new center.

    By default this samples uniformly from the `neighbor_k` closest non-origin
    points. When `max_relative_distance` is set (>0), the candidate pool is
    expanded to also include points whose origin distance is within
    `max_relative_distance * object_size`, where object_size is the per-sample
    bounding-box diagonal.
    """
    if points.dim() != 3 or points.size(-1) != 3:
        raise ValueError(f"points must have shape (B, N, 3), got {tuple(points.shape)}")
    bsz, n_pts, _ = points.shape
    if n_pts <= 1:
        return points
    k = int(neighbor_k) if neighbor_k is not None else 0
    if k <= 0:
        k = max(n_pts - 1, 1)
    k = min(k, n_pts - 1)
    dist2 = (points.float() ** 2).sum(dim=-1)
    dist2 = dist2.masked_fill(dist2 <= eps, float("inf"))

    # Base candidate pool: k nearest non-origin points.
    _, idx_nearest = dist2.topk(k=k, dim=1, largest=False)
    candidate_mask = torch.zeros((bsz, n_pts), dtype=torch.bool, device=points.device)
    candidate_mask.scatter_(1, idx_nearest, True)

    # Optional extension: include points up to a distance fraction of object size.
    frac = float(max_relative_distance) if max_relative_distance is not None else 0.0
    if frac > 0:
        mins = points.float().amin(dim=1)
        maxs = points.float().amax(dim=1)
        diag = (maxs - mins).norm(dim=-1).clamp_min(float(eps))
        max_dist2 = (diag * frac) ** 2
        within_dist = (dist2 > eps) & (dist2 <= max_dist2.unsqueeze(1))
        candidate_mask = candidate_mask | within_dist

    weights = candidate_mask.float()
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    neighbor_idx = torch.multinomial(weights, num_samples=1, replacement=True).squeeze(1)

    batch_idx = torch.arange(bsz, device=points.device)
    offsets = points[batch_idx, neighbor_idx]
    if not torch.isfinite(offsets).all():
        offsets = torch.where(torch.isfinite(offsets), offsets, torch.zeros_like(offsets))
    return points - offsets.unsqueeze(1)


def normalize_unit_range(
    points: torch.Tensor,
    *,
    constant: Optional[float],
    clamp: bool = False,
) -> torch.Tensor:
    """Normalize centered points into [0, 1] using a fixed constant."""
    if constant is None:
        return points
    constant = float(constant)
    if constant <= 0:
        return points
    out = points / (2.0 * constant) + 0.5
    if clamp:
        out = out.clamp(0.0, 1.0)
    return out
