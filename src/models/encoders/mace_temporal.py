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


@register_encoder('PretrainedMACEHistoryGeometry')
class PretrainedMACEHistoryGeometryEncoder(PretrainedMACETemporalEncoder):
    """Normalized coordinate-only histories in the original VICReg encoder API.

    MACE remains trainable. Activation checkpointing changes memory use only;
    the projector and full-batch VICReg statistics stay in VICRegModule.
    """

    def __init__(self, pretrained_checkpoint, reference_radius_A, performance,
                 frame_offsets_ps, fusion, frame_batch_size=128, activation_checkpointing=True):
        super().__init__(pretrained_checkpoint, frame_offsets_ps,
            time_scale_ps=frame_offsets_ps[-1]-frame_offsets_ps[-2],
            frame_batch_size=frame_batch_size, performance=performance)
        self.reference_radius_A = float(reference_radius_A)
        self.activation_checkpointing = activation_checkpointing
        self.fusion_kind = fusion
        if fusion != 'transformer':
            del self.frame_projection, self.time_embedding, self.blocks, self.output_norm
            self.invariant_dim = 256
        if fusion == 'residual':
            from .mace_denoising import ResidualFrameFusion
            self.fusion = ResidualFrameFusion(frame_offsets_ps)
        elif fusion in ('atom_anchor', 'atom_temporal'):
            from .mace_denoising import AtomTemporalFusion
            self.fusion = AtomTemporalFusion(frame_offsets_ps, anchor_only=fusion=='atom_anchor')
        elif fusion not in ('mean', 'transformer'):
            raise ValueError(f'Unknown normalized history fusion: {fusion}')

    def forward(self, points):
        from torch.utils.checkpoint import checkpoint
        frames = len(self.frame_offsets_ps)
        if points.ndim != 4 or points.shape[1:] != (frames, 80, 3):
            raise ValueError(f'Expected normalized (B, {frames}, 80, 3) histories, got {points.shape}')
        observed = points[:, -1:] if self.fusion_kind == 'atom_anchor' else points
        flat = observed.flatten(0, 1)*self.reference_radius_A
        channel = torch.zeros(len(flat), dtype=torch.long, device=flat.device)
        encode = self.mace.raw_node_features if self.fusion_kind.startswith('atom_') else self.mace.raw_features
        pieces = []
        for start in range(0, len(flat), self.frame_batch_size):
            args = (flat[start:start+self.frame_batch_size], channel[start:start+self.frame_batch_size])
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                pieces.append(checkpoint(encode, *args, use_reentrant=False))
            else:
                pieces.append(encode(*args))
        features = torch.cat(pieces)
        nodes = None
        if self.fusion_kind.startswith('atom_'):
            nodes = features.reshape(len(points), observed.shape[1], 80, 256)
            if self.fusion_kind == 'atom_anchor':
                nodes = nodes.expand(-1, frames, -1, -1)
            pooled = nodes.mean(2)
        else:
            pooled = features.reshape(len(points), frames, 256)
        if self.fusion_kind == 'mean':
            return pooled.mean(1)
        if self.fusion_kind == 'transformer':
            return self.forward_features(pooled)
        return self.fusion(pooled, nodes)
