"""Geometry-only local eSEN, using pinned FAIR eSEN blocks (see esen_vendor).

The original energy/species interface is replaced by the same centered local80
input and center-plus-tapered-mean scalar export as the spatial MACE comparison.
"""
import math
from pathlib import Path

import torch
from torch import nn

# Installed e3nn's packaged Wigner constants contain builtin Python slices.
with torch.serialization.safe_globals([slice]):
    from .esen_vendor.common.rotation import rotation_to_wigner
from .esen_vendor.common.so3 import CoefficientMapping, SO3_Grid
from .esen_vendor.esen_block import eSEN_Block
from .esen_vendor.nn.embedding import EdgeDegreeEmbedding
from .esen_vendor.nn.layer_norm import get_normalization_layer
from .esen_vendor.nn.radial import GaussianSmearing


def edge_rotation(vectors):
    """Deterministic edge frame; avoid upstream random gauges during graph replay.

    The least-aligned Cartesian axis supplies only an SO(2) gauge. The complete
    +/-m pairs and gated spectral nonlinearity remove its physical significance.
    Unlike the upstream direct-force path, input derivatives remain connected.
    """
    x = nn.functional.normalize(vectors, dim=-1)
    axis = nn.functional.one_hot(x.detach().abs().argmin(-1), 3).to(x)
    z = nn.functional.normalize(torch.cross(x, axis, dim=-1), dim=-1)
    y = nn.functional.normalize(torch.cross(x, z, dim=-1), dim=-1)
    return torch.stack((z, x, -y), dim=1)


class SpatialESEN(nn.Module):
    def __init__(self, *, channels=128, code_dim=128, cutoff=5., radius=8.,
                 num_interactions=2, max_ell=3, max_m=3, radial_basis=32,
                 hidden_channels=128, edge_channels=128, norm_type='layer_norm_sh'):
        super().__init__()
        self.channels, self.radius, self.cutoff = channels, radius, cutoff
        self.max_ell, self.max_m = max_ell, max_m
        self.mapping = CoefficientMapping(max_ell, max_m)
        self.grids = nn.ModuleDict({
            'lmax_lmax': SO3_Grid(max_ell, max_ell, rescale=True),
            'lmax_mmax': SO3_Grid(max_ell, max_m, rescale=True)})
        jd = torch.load(Path(__file__).with_name('esen_vendor') / 'Jd.pt', weights_only=True)
        for ell in range(max_ell + 1):
            self.register_buffer(f'Jd_{ell}', jd[ell].float())
        # One learned constant atom channel, plus the same center indicator as
        # MACE. No species, material ID, length-scale feature or conditions.
        self.node_embedding = nn.Linear(1, channels, bias=False)
        self.center_embedding = nn.Linear(1, channels, bias=False)
        self.source_constant = nn.Parameter(torch.empty(1, edge_channels))
        self.target_constant = nn.Parameter(torch.empty(1, edge_channels))
        nn.init.uniform_(self.source_constant, -.001, .001)
        nn.init.uniform_(self.target_constant, -.001, .001)
        self.radial = GaussianSmearing(0., cutoff, radial_basis, 2.)
        edge_widths = [radial_basis + 2 * edge_channels, edge_channels, edge_channels]
        self.degree_embedding = EdgeDegreeEmbedding(
            channels, max_ell, max_m, 1, edge_widths, 5., cutoff, self.mapping,
            self.grids['lmax_lmax'].mapping.coefficient_idx(max_ell, max_m), True)
        self.blocks = nn.ModuleList([
            eSEN_Block(channels, hidden_channels, max_ell, max_m, self.mapping,
                       self.grids, edge_widths, cutoff, norm_type, 'gate', 'spectral', True)
            for _ in range(num_interactions)])
        self.norm = get_normalization_layer(norm_type, lmax=max_ell, num_channels=channels)
        width = 2 * channels
        self.pool_norm = nn.LayerNorm(width)
        self.projection = nn.Linear(width, code_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)
        self.residual = nn.Sequential(nn.Linear(width, code_dim), nn.SiLU(), nn.Linear(code_dim, code_dim))
        nn.init.normal_(self.residual[-1].weight, std=.01)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, q, keep=None):
        with torch.autocast('cuda', enabled=False):
            q = q.float() - q[:, :1].float()
            batch, atoms, _ = q.shape
            if atoms != 80:
                raise ValueError('SpatialESEN requires nearest80 neighborhoods')
            with torch.no_grad():
                valid = q.norm(dim=-1) < self.radius
                if keep is not None:
                    valid &= keep
                allowed = ((q[:, None] - q[:, :, None]).norm(dim=-1) < self.cutoff)
                allowed &= valid[:, :, None] & valid[:, None, :]
                allowed &= ~torch.eye(atoms, dtype=torch.bool, device=q.device)[None]
                b, i, j = allowed.nonzero(as_tuple=True)
            positions = q.flatten(0, 1)
            edge = torch.stack((b * atoms + i, b * atoms + j))
            # Upstream eSEN uses source minus target edge vectors.
            vectors = positions[edge[0]] - positions[edge[1]]
            distance = vectors.norm(dim=-1)
            wigner = rotation_to_wigner(edge_rotation(vectors), 0, self.max_ell,
                [getattr(self, f'Jd_{ell}') for ell in range(self.max_ell + 1)], rot_clip=True)
            inverse = wigner.transpose(1, 2).contiguous()
            radial = self.radial(distance)
            centers = torch.arange(batch, device=q.device) * atoms
            indicator = positions.new_zeros(len(positions), 1)
            indicator[centers] = 1
            node = self.node_embedding(torch.ones_like(indicator)) + self.center_embedding(indicator)
            x = positions.new_zeros(len(positions), (self.max_ell + 1)**2, self.channels)
            x[:, 0] = node
        edge_features = torch.cat((radial, self.source_constant.expand(len(edge[0]), -1),
                                   self.target_constant.expand(len(edge[0]), -1)), dim=1)
        x = self.degree_embedding(x, edge_features, distance, edge, inverse)
        for block in self.blocks:
            x = block(x, edge_features, distance, edge, wigner, inverse)
        x = self.norm(x)
        with torch.autocast('cuda', enabled=False):
            scalar = x[:, 0].float().reshape(batch, atoms, self.channels)
            # Same single 6–8 A outer taper, normalization and export as MACE.
            distance = (q.square().sum(-1) + nn.functional.one_hot(
                torch.zeros(batch, dtype=torch.long, device=q.device), atoms)).sqrt()
            weight = .5 * (1 + torch.cos(math.pi * ((distance - 6.) / 2.).clamp(0, 1))) * valid
            pooled = (scalar * weight[..., None]).sum(1) / weight.sum(1, keepdim=True).clamp_min(1e-6)
            u = self.pool_norm(torch.cat((scalar[:, 0], pooled), dim=-1))
            return self.projection(u) + self.residual(u)
