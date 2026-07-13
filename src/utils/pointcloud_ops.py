import torch


def _crop_to_num_points_indices(
    points: torch.Tensor,
    num_points: int | None,
) -> torch.Tensor | None:
    if num_points is None:
        return None
    if num_points < 1:
        raise ValueError(f"num_points must be positive when cropping is enabled, got {num_points}.")
    if points.dim() != 3 or points.size(-1) != 3:
        raise ValueError(f"points must have shape (B, N, 3), got {tuple(points.shape)}")

    batch_size, input_points, _ = points.shape
    if num_points > input_points:
        raise ValueError(f"num_points ({num_points}) cannot exceed input points ({input_points})")
    if num_points == input_points:
        return torch.arange(input_points, device=points.device).unsqueeze(0).expand(batch_size, -1)

    distances_squared = points.float().square().sum(dim=-1)
    return distances_squared.topk(k=num_points, dim=1, largest=False).indices


def crop_to_num_points(points: torch.Tensor, num_points: int | None) -> torch.Tensor:
    """Keep the configured number of points nearest the origin."""
    indices = _crop_to_num_points_indices(points, num_points)
    if indices is None:
        return points
    return points.gather(1, indices.unsqueeze(-1).expand(-1, -1, 3))


def crop_to_num_points_with_indices(
    points: torch.Tensor,
    num_points: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep points nearest the origin and return their input indices."""
    if points.dim() != 3 or points.size(-1) != 3:
        raise ValueError(f"points must have shape (B, N, 3), got {tuple(points.shape)}")
    batch_size, input_points, _ = points.shape
    indices = _crop_to_num_points_indices(points, num_points)
    if indices is None:
        indices = torch.arange(input_points, device=points.device).unsqueeze(0).expand(batch_size, -1)
        return points, indices
    return points.gather(1, indices.unsqueeze(-1).expand(-1, -1, 3)), indices


def shift_to_neighbor(
    points: torch.Tensor,
    *,
    neighbor_k: int,
    max_relative_distance: float,
) -> torch.Tensor:
    """Shift each point cloud to a configured near-origin neighbor."""
    shifted, _ = shift_to_neighbor_with_indices(
        points,
        neighbor_k=neighbor_k,
        max_relative_distance=max_relative_distance,
    )
    return shifted


def shift_to_neighbor_with_indices(
    points: torch.Tensor,
    *,
    neighbor_k: int,
    max_relative_distance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift point clouds and return the selected point index per sample."""
    if points.dim() != 3 or points.size(-1) != 3:
        raise ValueError(f"points must have shape (B, N, 3), got {tuple(points.shape)}")
    batch_size, input_points, _ = points.shape
    if input_points < 2:
        raise ValueError(
            "Neighbor-centered views require at least two input points, "
            f"got shape {tuple(points.shape)}."
        )
    if neighbor_k < 1:
        raise ValueError(f"neighbor_k must be positive, got {neighbor_k}.")
    if max_relative_distance < 0.0:
        raise ValueError(
            "max_relative_distance must be non-negative, "
            f"got {max_relative_distance}."
        )

    if neighbor_k >= input_points:
        raise ValueError(
            "neighbor_k must be smaller than the number of input points, "
            f"got neighbor_k={neighbor_k}, input_points={input_points}."
        )
    candidate_count = neighbor_k
    distances_squared = points.float().square().sum(dim=-1)
    distances_squared = distances_squared.masked_fill(distances_squared <= 1e-12, float("inf"))
    nearest_indices = distances_squared.topk(
        k=candidate_count,
        dim=1,
        largest=False,
    ).indices
    candidate_mask = torch.zeros(
        (batch_size, input_points),
        dtype=torch.bool,
        device=points.device,
    )
    candidate_mask.scatter_(1, nearest_indices, True)

    if max_relative_distance > 0.0:
        box_diagonal = (points.float().amax(dim=1) - points.float().amin(dim=1)).norm(dim=-1)
        maximum_distance_squared = (box_diagonal * max_relative_distance).square()
        candidate_mask |= (
            (distances_squared > 1e-12)
            & (distances_squared <= maximum_distance_squared.unsqueeze(1))
        )

    neighbor_index = torch.multinomial(candidate_mask.float(), num_samples=1).squeeze(1)
    offsets = points[torch.arange(batch_size, device=points.device), neighbor_index]
    if not torch.isfinite(offsets).all():
        raise RuntimeError(
            "Neighbor selection found no finite non-origin point; "
            f"points_shape={tuple(points.shape)}, neighbor_k={neighbor_k}, "
            f"max_relative_distance={max_relative_distance}."
        )
    return points - offsets.unsqueeze(1), neighbor_index
