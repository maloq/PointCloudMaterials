"""Geometry-only local TensorNet with the common 128-dimensional spatial export.

Pinned TorchMD-Net tensor operations are in tensornet_vendor. No dependency on
TorchMD-Net's neighbor kernels, species/charge inputs, or force/energy heads.
"""
import math

import torch
from torch import nn

from .tensornet_vendor import blocks
from .tensornet_vendor.radial import ExpNormalSmearing


def tensor_norm(tensor):
    # Retain FP64 for derivative audits and FP32 reductions under neural AMP.
    if tensor.dtype in (torch.float16, torch.bfloat16):
        tensor = tensor.float()
    return tensor.square().sum((-2, -1))


def tensor_message_passing(edge_index, factor, tensor, natoms):
    # Radial factors retain FP32 under AMP, promoting the product from BF16.
    # Allocate aggregation in that product's dtype, not the projection's dtype.
    message = factor * tensor.index_select(0, edge_index[1])
    return message.new_zeros((natoms, *tensor.shape[1:])).index_add(0, edge_index[0], message)


# This module owns the only users of the vendored definitions. Preserve all
# algebra while making their normalization and output norms safe under AMP.
blocks.tensor_norm = tensor_norm
blocks.tensor_message_passing = tensor_message_passing


class GeometryEmbedding(blocks.TensorEmbedding):
    def __init__(self, channels, radial_basis, cutoff):
        super().__init__(channels, radial_basis, nn.SiLU, 0., cutoff,
                         trainable_rbf=False, max_z=1)
        self.center_embedding = nn.Linear(1, channels, bias=False)

    def _get_atomic_number_message(self, center, edge_index):
        # The single embedding row is a learned constant for every atom. The
        # only varying input here is the center flag; no atom types are accepted.
        node = self.emb.weight.expand(len(center), -1) + self.center_embedding(center)
        pairs = torch.cat((node[edge_index[0]], node[edge_index[1]]), dim=-1)
        return self.emb2(pairs)[..., None, None]


class SpatialTensorNet(nn.Module):
    def __init__(self, *, channels=128, code_dim=128, cutoff=5., radius=8.,
                 num_interactions=2, radial_basis=32):
        super().__init__()
        self.channels, self.radius, self.cutoff = channels, radius, cutoff
        self.radial = ExpNormalSmearing(0., cutoff, radial_basis, trainable=False)
        self.embedding = GeometryEmbedding(channels, radial_basis, cutoff)
        self.blocks = nn.ModuleList([
            blocks.Interaction(radial_basis, channels, nn.SiLU, 0., cutoff, 'O(3)')
            for _ in range(num_interactions)])
        self.out_norm = nn.LayerNorm(3 * channels)
        self.scalar_projection = nn.Linear(3 * channels, channels)
        width = 2 * channels
        self.pool_norm = nn.LayerNorm(width)
        self.projection = nn.Linear(width, code_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)
        self.residual = nn.Sequential(nn.Linear(width, code_dim), nn.SiLU(),
                                      nn.Linear(code_dim, code_dim))
        nn.init.normal_(self.residual[-1].weight, std=.01)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, q, keep=None):
        with torch.autocast(q.device.type, enabled=False):
            # Production data/weights are FP32; retain double for numerical
            # differentiation audits instead of silently truncating the input.
            q = q.to(self.projection.weight.dtype)
            q = q - q[:, :1]
            batch, atoms, _ = q.shape
            if atoms != 80:
                raise ValueError('SpatialTensorNet requires nearest80 neighborhoods')
            with torch.no_grad():
                valid = q.norm(dim=-1) < self.radius
                if keep is not None:
                    valid = valid & keep
                allowed = (q[:, None] - q[:, :, None]).norm(dim=-1) < self.cutoff
                allowed = allowed & valid[:, :, None] & valid[:, None, :]
                # TensorNet includes self edges in its original embedding.
                b, i, j = allowed.nonzero(as_tuple=True)
            positions = q.flatten(0, 1)
            edge = torch.stack((b * atoms + i, b * atoms + j))
            vectors = positions[edge[0]] - positions[edge[1]]
            self_edge = edge[0] == edge[1]
            # Avoid differentiating sqrt(0) on self edges even at higher orders.
            distance = (vectors.square().sum(-1) + self_edge.to(q.dtype)).sqrt()
            distance = distance.masked_fill(self_edge, 0.)
            direction = vectors / distance.masked_fill(self_edge, 1.)[:, None]
            radial = self.radial(distance)
            center = positions.new_zeros(len(positions), 1)
            center[torch.arange(batch, device=q.device) * atoms] = 1
            neutral = positions.new_zeros(len(positions))
        x = self.embedding(center, edge, distance, direction, radial)
        for block in self.blocks:
            # q in the original block is charge: fixed zero, never user input.
            x = block(x, edge, distance, radial, neutral)
        with torch.autocast(q.device.type, enabled=False):
            x = x.to(q.dtype)
            parts = blocks.decompose_tensor(x)
            scalar = self.out_norm(torch.cat([tensor_norm(t) for t in parts], -1))
        scalar = nn.functional.silu(self.scalar_projection(scalar))
        with torch.autocast(q.device.type, enabled=False):
            scalar = scalar.to(q.dtype).reshape(batch, atoms, self.channels)
            distance = (q.square().sum(-1) + nn.functional.one_hot(
                torch.zeros(batch, dtype=torch.long, device=q.device), atoms)).sqrt()
            weight = .5 * (1 + torch.cos(math.pi * ((distance - 6.) / 2.).clamp(0, 1))) * valid
            pooled = (scalar * weight[..., None]).sum(1) / weight.sum(1, keepdim=True).clamp_min(1e-6)
            u = self.pool_norm(torch.cat((scalar[:, 0], pooled), -1))
            return self.projection(u) + self.residual(u)

    def compile_blocks(self):
        import torch._functorch.config
        torch._functorch.config.backward_pass_autocast = 'off'
        options = {'emulate_precision_casts': True}
        self.embedding.compile(fullgraph=True, dynamic=True, options=options)
        for block in self.blocks:
            block.compile(fullgraph=True, dynamic=True, options=options)
