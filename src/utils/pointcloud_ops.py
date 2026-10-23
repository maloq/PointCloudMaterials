import torch


def crop_to_num_points(points: torch.Tensor, num_points: int | None) -> torch.Tensor:
    """Keep the nearest points in a (B, N, 3) batch; full-size views reuse the input."""
    if num_points is None or num_points == points.shape[1]:
        return points
    distances_squared = points.float().square().sum(dim=-1)
    indices = distances_squared.topk(k=num_points, dim=1, largest=False).indices
    return points.gather(1, indices.unsqueeze(-1).expand(-1, -1, 3))


def shift_to_neighbor(
    points: torch.Tensor,
    *,
    neighbor_k: int,
    max_relative_distance: float,
) -> torch.Tensor:
    """Shift each point cloud to a configured near-origin neighbor."""
    batch_size, input_points, _ = points.shape
    coordinates = points.float()
    distances_squared = coordinates.square().sum(dim=-1)
    distances_squared = distances_squared.masked_fill(distances_squared <= 1e-12, float("inf"))
    nearest_indices = distances_squared.topk(
        k=neighbor_k,
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
        box_diagonal = (coordinates.amax(dim=1) - coordinates.amin(dim=1)).norm(dim=-1)
        maximum_distance_squared = (box_diagonal * max_relative_distance).square()
        candidate_mask |= (
            (distances_squared > 1e-12)
            & (distances_squared <= maximum_distance_squared.unsqueeze(1))
        )

    neighbor_index = torch.multinomial(candidate_mask.float(), num_samples=1).squeeze(1)
    offsets = points[torch.arange(batch_size, device=points.device), neighbor_index]
    return points - offsets.unsqueeze(1)
