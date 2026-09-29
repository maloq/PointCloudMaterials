"""Typed patch encoding and spatial context, without task prediction heads."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from src.research.encoder_context.geometry import graph
from src.research.equivariant_context.features import TypedPatchExport
from src.research.equivariant_context.model import VectorMessages, geometry
from src.research.supervised_onset.model import CapacityEncoder


@dataclass(frozen=True)
class PatchLayout:
    scalars: int
    vectors: int
    atoms: int = 80

    @property
    def width(self):
        return self.scalars + 3 * self.vectors

    def pack(self, scalar, vector):
        return torch.cat((scalar, vector.flatten(-2)), dim=-1)

    def unpack(self, values):
        if values.shape[-1] != self.width:
            raise ValueError(f'Typed patch width: expected {self.width}, got {values.shape}')
        scalar = values[..., : self.scalars]
        vector = values[..., self.scalars :].reshape(*values.shape[:-1], self.vectors, 3)
        return scalar, vector


class TypedPatchTrunk(nn.Module):
    """One geometry-only MACE with scalar and equivariant vector exports.

    Modules retain their original names so retained encoder/trunk tensors can be
    loaded strictly. The layout describes the exported fields, not the raw l=2
    fields also produced internally by TypedPatchExport.
    """

    def __init__(self, encoder_config, config, *, vector_channels):
        super().__init__()
        self.config = config
        self.patch = TypedPatchExport(CapacityEncoder(**encoder_config))
        self.latent_dim = self.encoder.projection.out_features
        self.layout = PatchLayout(self.latent_dim, vector_channels)
        self.vector_export = nn.Linear(2 * self.encoder.channels, vector_channels, bias=False)
        self.register_buffer('scalar_mean', torch.zeros(self.latent_dim))
        self.register_buffer('scalar_scale', torch.ones(self.latent_dim))
        self.register_buffer('vector_scale', torch.ones(vector_channels))

    @property
    def encoder(self):
        return self.patch.encoder

    def encode(self, positions):
        chunk = self.config['patch_chunk']
        outputs = []
        for start in range(0, len(positions), chunk):
            xyz = positions[start : start + chunk]
            count = len(xyz)
            if count < chunk:
                pad = xyz.new_full((chunk - count, self.layout.atoms, 3), 100.0)
                pad[:, 0] = 0
                xyz = torch.cat((xyz, pad))
            g = graph(xyz, self.encoder)
            if (
                self.training
                and torch.is_grad_enabled()
                and self.config['activation_checkpointing']
            ):
                value = checkpoint(self.patch, g, use_reentrant=False)
            else:
                value = self.patch(g)
            outputs.append(value[:count])
        raw = torch.cat(outputs).float()
        channels = self.encoder.channels
        scalar = raw[:, : self.layout.scalars]
        vector = raw[:, self.layout.scalars : self.layout.scalars + 6 * channels]
        vector = vector.reshape(-1, 2 * channels, 3)
        # Bias-free channel mixing preserves Cartesian components.
        vector = F.linear(vector.transpose(-1, -2), self.vector_export.weight.float())
        vector = vector.transpose(-1, -2)
        return (
            (scalar - self.scalar_mean) / self.scalar_scale,
            vector / self.vector_scale[None, :, None],
        )


class SpatialContextTrunk(TypedPatchTrunk):
    """Mix typed patch fields using the observed query-relative geometry."""

    def __init__(self, encoder_config, config):
        predictor = config['predictor']
        super().__init__(encoder_config, config, vector_channels=predictor['field_channels'])
        width = predictor['width']
        self.stem = nn.Sequential(nn.Linear(self.latent_dim, width), nn.LayerNorm(width), nn.SiLU())
        self.geometry = nn.Linear(4, width)
        self.blocks = nn.ModuleList(
            [VectorMessages(width, self.layout.vectors) for _ in range(predictor['depth'])]
        )

    def context(self, scalar, vector, actual):
        g = geometry(actual, actual)
        state = self.stem(scalar) + self.geometry(g['node'])
        fields = {1: vector}
        for block in self.blocks:
            state, fields = block(state, fields, g)
        return state, fields
