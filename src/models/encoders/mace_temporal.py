"""Learn one structural embedding from a causal history of physical MACE clouds."""

import math

import torch
from torch import nn

from .pretrained_mace import PretrainedMACEEncoder
from .registry import register_encoder


@register_encoder("PretrainedMACETemporal")
class PretrainedMACETemporalEncoder(nn.Module):
    """Shared MACE followed by temporal attention, read at the final anchor.

    ``points`` is float32 (B, T, 80, 3), in Å, centered on the same atom
    throughout each history. ``material`` is int64 (B,), using the physical
    MACE convention 0=Al, 1=Mg, 2=Ta. Frames follow ``frame_offsets_ps``;
    the last frame is the anchor and every other frame precedes it.

    This physical, material-aware API is separate from the normalized
    single-cloud EncoderAdapter API. Losses and prediction heads belong to
    the caller. No frame features are detached or temporally averaged.
    """

    input_layout = "btn3"
    output_contract = "invariant"
    equivariant_dim = None

    def __init__(
        self,
        pretrained_checkpoint,
        frame_offsets_ps: list[float],
        time_scale_ps: float,
        width: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 512,
        dropout: float = 0.0,
        frame_batch_size: int = 512,
        accelerated: bool = True,
        performance=None,
    ):
        super().__init__()
        if (
            not frame_offsets_ps
            or not all(math.isfinite(t) for t in frame_offsets_ps)
            or frame_offsets_ps[-1] != 0.0
            or any(a >= b for a, b in zip(frame_offsets_ps, frame_offsets_ps[1:]))
        ):
            raise ValueError(
                "frame_offsets_ps must be finite, strictly increasing and end at "
                f"the anchor (0 ps); got {frame_offsets_ps}. Future frames are not inputs."
            )
        if not math.isfinite(time_scale_ps) or time_scale_ps <= 0:
            raise ValueError(f"time_scale_ps must be finite and positive, got {time_scale_ps}.")
        if width <= 0 or num_heads <= 0 or width % num_heads:
            raise ValueError(f"width={width} must be positive and divisible by num_heads={num_heads}.")
        if num_layers <= 0 or feedforward_dim <= 0 or frame_batch_size <= 0:
            raise ValueError("num_layers, feedforward_dim and frame_batch_size must be positive.")

        self.invariant_dim = width
        self.frame_batch_size = frame_batch_size
        self.register_buffer("frame_offsets_ps", torch.tensor(frame_offsets_ps, dtype=torch.float32))
        self.register_buffer("time_scale_ps", torch.tensor(time_scale_ps, dtype=torch.float32))
        self.mace = PretrainedMACEEncoder(
            pretrained_checkpoint, accelerated=accelerated, performance=performance
        )
        self.frame_projection = nn.Linear(256, width)
        self.time_embedding = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        # Construct each block separately so their initial parameters differ.
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=width,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(width)

    def forward(self, points: torch.Tensor, material: torch.Tensor) -> torch.Tensor:
        """Return (B, width) anchor embeddings with gradients through all frames."""
        frames = len(self.frame_offsets_ps)
        if points.ndim != 4 or points.shape[1:] != (frames, 80, 3):
            raise ValueError(
                f"Expected history shape (B, {frames}, 80, 3) in Å, got {tuple(points.shape)}. "
                "Spatial/hot/relaxed training views are not a temporal history."
            )
        if material.shape != (len(points),) or material.dtype != torch.long:
            raise ValueError(
                f"Expected material int64 shape ({len(points)},), got "
                f"{tuple(material.shape)}, {material.dtype}; 0=Al, 1=Mg, 2=Ta."
            )
        clouds = points.flatten(0, 1)
        channels = material.repeat_interleave(frames)
        # Chunk only the spatial backbone. Every token stays in the autograd graph.
        features = torch.cat([
            self.mace(clouds[start:start + self.frame_batch_size], channels[start:start + self.frame_batch_size])
            for start in range(0, len(clouds), self.frame_batch_size)
        ]).reshape(len(points), frames, 256)
        return self.forward_features(features)

    def forward_features(self, features: torch.Tensor, times=None) -> torch.Tensor:
        """Apply the identical fusion to cached, standardized frozen MACE features."""
        positions = (self.frame_offsets_ps / self.time_scale_ps)[None] if times is None else times
        tokens = self.frame_projection(features) + self.time_embedding(positions[..., None])
        # All tokens are observed at the anchor. No triangular mask is required;
        # earlier contextual tokens are not exported as causal predictions.
        for block in self.blocks:
            tokens = block(tokens)
        return self.output_norm(tokens[:, -1])
