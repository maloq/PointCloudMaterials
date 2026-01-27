import torch
from typing import Optional


def crop_to_num_points(points: torch.Tensor, num_points: Optional[int]) -> torch.Tensor:
    """Crop point cloud to num_points by keeping closest points to the origin."""
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
    dist2 = (points.float() ** 2).sum(dim=-1)
    idx = dist2.topk(k=num_points, dim=1, largest=False).indices
    idx = idx.unsqueeze(-1).expand(-1, -1, 3)
    return points.gather(1, idx)


def shift_to_neighbor(
    points: torch.Tensor,
    *,
    neighbor_k: int = 8,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Shift point cloud so a random neighbor of the origin becomes the new center."""
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
    _, idx = dist2.topk(k=k, dim=1, largest=False)
    choice = torch.randint(0, k, (bsz,), device=points.device)
    batch_idx = torch.arange(bsz, device=points.device)
    neighbor_idx = idx[batch_idx, choice]
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
