from __future__ import annotations

import math

import torch


def resolve_model_summary_batch_size(cfg, *, default: int = 1) -> int:
    raw_value = getattr(cfg, "model_summary_batch_size", default)
    batch_size = int(raw_value)
    if batch_size < 1:
        raise ValueError(
            "model_summary_batch_size must be >= 1 so PyTorch Lightning can estimate FLOPs, "
            f"got {raw_value!r}."
        )
    return batch_size


def make_model_summary_point_cloud(
    *,
    batch_size: int,
    num_points: int,
    sequence_length: int | None = None,
) -> torch.Tensor:
    batch_size = int(batch_size)
    num_points = int(num_points)
    if batch_size < 1:
        raise ValueError(f"Summary point-cloud batch_size must be >= 1, got {batch_size}.")
    if num_points < 1:
        raise ValueError(f"Summary point-cloud num_points must be >= 1, got {num_points}.")
    if sequence_length is not None and int(sequence_length) < 1:
        raise ValueError(
            "Summary temporal sequence_length must be >= 1 when provided, "
            f"got {sequence_length}."
        )

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
    center = (int(sequence_length) - 1) / 2.0
    for frame_idx in range(int(sequence_length)):
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
