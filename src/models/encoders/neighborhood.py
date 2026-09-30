"""Local snapshot encoder shared by label-free training and feature extraction."""

import torch
from torch import nn
from e3nn import o3
from src.models.encoders.structural import StructuralMACE

ELLS = (1, 2, 4, 6)
CHANNELS = 4


def split_tensors(e):
    return [
        x.reshape(*e.shape[:-1], CHANNELS, 2 * l + 1)
        for x, l in zip(e.split([CHANNELS * (2 * l + 1) for l in ELLS], -1), ELLS)
    ]


def invariants(e):
    # Channel Gram matrices retain alignment information; rotation-invariant.
    return torch.cat(
        [
            (x @ x.transpose(-1, -2) / (2 * l + 1)).flatten(-2)
            for x, l in zip(split_tensors(e), ELLS)
        ],
        -1,
    )


class NeighborhoodEncoder(nn.Module):
    """One independently evaluated local snapshot; no teacher or history inside E."""

    def __init__(self, channels=128, export_norm='layernorm', *, backend='cueq'):
        super().__init__()
        self.base = StructuralMACE(
            channels=channels, readout_hidden=104 * channels // 32, backend=backend
        )
        width = channels
        self.angular_weights = nn.Sequential(
            nn.Linear(width + 1, 2 * channels),
            nn.SiLU(),
            nn.Linear(2 * channels, len(ELLS) * CHANNELS),
        )
        self.harmonics = o3.SphericalHarmonics(
            list(ELLS), normalize=True, normalization='component'
        )
        self.compress = nn.Sequential(
            nn.Linear(128 + len(ELLS) * CHANNELS**2, 6 * channels),
            nn.LayerNorm(6 * channels),
            nn.SiLU(),
            nn.Linear(6 * channels, 128),
        )
        if export_norm == 'layernorm':
            self.output_norm = nn.LayerNorm(128, elementwise_affine=False)
        elif export_norm == 'raw':
            self.output_norm = nn.Identity()
        else:
            raise ValueError(f'Unknown snapshot export normalization: {export_norm}')

    def forward(self, batch):
        base = self.base
        atoms = base.atom_features(batch)
        x, g, w = atoms['positions'], atoms['graph'], atoms['weights']
        scalar, count = atoms['scalars'], atoms['count']
        z = base.output_norm(base.pool(atoms['features'], x, g, count))
        # Tensor creation, contractions and export are FP32. Harmonics precede pooling.
        with torch.autocast(x.device.type, enabled=False):
            radius = x.norm(dim=-1)
            weights = w.float() * (radius > 1e-6)
            coeff = self.angular_weights(
                torch.cat((scalar.float(), (radius / 8)[:, None]), -1)
            ).reshape(-1, len(ELLS), CHANNELS)
            harmonics = self.harmonics(x).float().split([2 * l + 1 for l in ELLS], -1)
            denom = x.new_zeros(count).index_add(0, g, weights).clamp_min(1e-8)
            tensors = []
            for k, y in enumerate(harmonics):
                atom = (coeff[:, k, :, None] * y[:, None] * weights[:, None, None]).flatten(1)
                pooled = x.new_zeros(count, atom.shape[-1]).index_add(0, g, atom) / denom[:, None]
                tensors.append(pooled)
            eq = torch.cat(tensors, -1)
            inv = self.output_norm(
                z.float() + self.compress(torch.cat((z.float(), invariants(eq)), -1))
            )
        return torch.cat((inv, eq), -1)

    def export(self, batch):
        encoded = self(batch)
        return {'invariant': encoded[:, :128], 'equivariant': encoded[:, 128:]}
