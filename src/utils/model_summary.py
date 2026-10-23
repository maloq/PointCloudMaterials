from __future__ import annotations

import math

import torch


def make_model_summary_point_cloud(
    *,
    batch_size: int,
    num_points: int,
    sequence_length: int | None = None,
) -> torch.Tensor:
    """Deterministic geometry for the configured encoder's model summary."""
    idx = torch.arange(num_points, dtype=torch.float32)
    z = torch.linspace(-0.75, 0.75, num_points, dtype=torch.float32)
    theta = idx * (math.pi * (3.0 - math.sqrt(5.0)))
    radius = torch.sqrt((1.0 - z.square()).clamp_min(0.05))
    points = torch.stack(
        (
            radius * torch.cos(theta),
            radius * torch.sin(theta),
            z + 0.05 * torch.sin(idx * 0.37),
        ),
        dim=-1,
    )
    points = points - points.mean(dim=0, keepdim=True)
    scale = torch.linalg.vector_norm(points, dim=-1).amax().clamp_min(1.0)
    points = points / scale

    batch_offsets = torch.linspace(0.0, 0.01, batch_size, dtype=torch.float32).view(batch_size, 1, 1)
    batch = points.unsqueeze(0).repeat(batch_size, 1, 1)
    batch = batch + torch.cat(
        (
            batch_offsets,
            torch.zeros_like(batch_offsets),
            -batch_offsets,
        ),
        dim=-1,
    )

    if sequence_length is None:
        return batch.contiguous()

    frames = []
    center = (sequence_length - 1) / 2.0
    for frame_idx in range(sequence_length):
        angle = 0.03 * (float(frame_idx) - center)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rotation = torch.tensor(
            (
                (cos_a, -sin_a, 0.0),
                (sin_a, cos_a, 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=torch.float32,
        )
        frames.append(batch @ rotation.T)
    return torch.stack(frames, dim=1).contiguous()
