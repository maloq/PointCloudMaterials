"""Residual frame fusion and atom-identity temporal fusion for frozen MACE."""

import torch
from torch import nn

from .pretrained_mace import PretrainedMACEEncoder
from .registry import register_encoder


class TemporalCorrection(nn.Module):
    """A small time-aware correction with an exactly zero initial output."""

    def __init__(self, frame_offsets_ps, width=64):
        super().__init__()
        offsets = torch.tensor(frame_offsets_ps, dtype=torch.float32)
        if offsets[-1] != 0 or not (offsets.diff() > 0).all():
            raise ValueError(f'Expected increasing causal frame offsets ending at zero: {frame_offsets_ps}')
        self.register_buffer('times', offsets / offsets.diff()[-1])
        self.project = nn.Linear(256, width)
        self.time_embedding = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        self.block = nn.TransformerEncoderLayer(width, 4, width*4, dropout=0., activation='gelu',
                                                batch_first=True, norm_first=True)
        self.output = nn.Linear(width, 256)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def tokens(self, features, times=None):
        positions = self.times[None] if times is None else times
        return self.block(self.project(features) + self.time_embedding(positions[..., None]))

    def forward(self, features, times=None):
        return self.output(self.tokens(features, times)[:, -1])


class ResidualFrameFusion(nn.Module):
    invariant_dim = 256

    def __init__(self, frame_offsets_ps):
        super().__init__()
        self.correction = TemporalCorrection(frame_offsets_ps)

    def forward(self, pooled, nodes=None, times=None):
        return pooled[:, -1] + self.correction(pooled, times)


class AtomTemporalFusion(nn.Module):
    """Follow each atom through time, then learn spatial pooling over its history."""
    invariant_dim = 256

    def __init__(self, frame_offsets_ps, anchor_only=False):
        super().__init__()
        self.anchor_only = anchor_only
        self.temporal = TemporalCorrection(frame_offsets_ps)
        self.spatial_score = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
        self.spatial_value = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 64))

    def forward(self, pooled, nodes, times=None):
        b, t, n, c = nodes.shape
        if self.anchor_only:
            nodes = nodes[:, -1:].expand(-1, t, -1, -1)
        histories = nodes.transpose(1, 2).reshape(b*n, t, c)
        atom_times = None if times is None else times.repeat_interleave(n, dim=0)
        atoms = self.temporal.tokens(histories, atom_times)[:, -1].reshape(b, n, -1)
        weights = self.spatial_score(atoms).softmax(dim=1)
        summary = (weights * self.spatial_value(atoms)).sum(dim=1)
        return pooled[:, -1] + self.temporal.output(summary)


@register_encoder('PretrainedMACEDenoising')
class PretrainedMACEDenoisingEncoder(nn.Module):
    """Physical material API; frozen backbone, trainable fusion, no target required."""
    input_layout = 'btn3'
    output_contract = 'invariant'
    invariant_dim = 256
    equivariant_dim = None

    def __init__(self, pretrained_checkpoint, frame_offsets_ps, fusion, performance,
                 frame_batch_size=128, accelerated=True, require_frame_offsets=False):
        super().__init__()
        self.frame_count = len(frame_offsets_ps)
        self.require_frame_offsets = require_frame_offsets
        self.register_buffer('time_scale_ps', torch.tensor(frame_offsets_ps[-1]-frame_offsets_ps[-2]))
        self.frame_batch_size = frame_batch_size
        self.mace = PretrainedMACEEncoder(pretrained_checkpoint, accelerated=accelerated, performance=performance)
        self.mace.requires_grad_(False)
        if fusion == 'residual':
            self.fusion = ResidualFrameFusion(frame_offsets_ps)
        elif fusion in ('atom_temporal', 'atom_anchor'):
            self.fusion = AtomTemporalFusion(frame_offsets_ps, anchor_only=fusion == 'atom_anchor')
        else:
            raise ValueError(f'Unknown denoising fusion: {fusion}')
        for name in ('pooled_mean', 'node_mean'):
            self.register_buffer(name, torch.zeros(256))
        for name in ('pooled_std', 'node_std'):
            self.register_buffer(name, torch.ones(256))

    def forward(self, points, material, frame_offsets_ps=None):
        if points.ndim != 4 or points.shape[1:] != (self.frame_count, 80, 3):
            raise ValueError(f'Expected (B, {self.frame_count}, 80, 3) physical Å histories, got {points.shape}')
        if self.require_frame_offsets and frame_offsets_ps is None:
            raise ValueError('This mixed-cadence encoder requires frame_offsets_ps with shape (B, T).')
        if frame_offsets_ps is not None and frame_offsets_ps.shape != points.shape[:2]:
            raise ValueError(f'Expected per-history times {points.shape[:2]}, got {frame_offsets_ps.shape}')
        flat = points.flatten(0, 1)
        species = material.repeat_interleave(self.frame_count)
        with torch.no_grad():
            nodes = torch.cat([self.mace.raw_node_features(flat[i:i+self.frame_batch_size],
                species[i:i+self.frame_batch_size]) for i in range(0, len(flat), self.frame_batch_size)])
            nodes = nodes.reshape(len(points), self.frame_count, 80, 256)
            pooled = (nodes.mean(2)-self.pooled_mean)/self.pooled_std
            # Match the cached atom representation used in training.
            nodes = (nodes.half().float()-self.node_mean)/self.node_std
        times = None if frame_offsets_ps is None else frame_offsets_ps / self.time_scale_ps
        return self.fusion(pooled, nodes, times)
