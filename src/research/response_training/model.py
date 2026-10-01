"""Differentiable full-periodic-cell MACE128 and complete-predictor JVPs."""
import torch
from torch import nn

from src.research.supervised_onset.model import CapacityEncoder


class CellEncoder(CapacityEncoder):
    def pooled_graph(self, graph):
        scalar = self.atom_features(graph)[:, :self.channels]
        scalar = scalar.reshape(graph['size'], 256, self.channels)
        return torch.cat((scalar.mean(1), scalar.var(1, unbiased=False)), -1)


class Predictor(nn.Module):
    def __init__(self, c, box):
        super().__init__()
        self.encoder = CellEncoder(d0=2.8, n_ref=256., radius=8., **c['encoder'])
        self.register_buffer('box', torch.as_tensor(box, dtype=torch.float32))
        self.head = nn.Sequential(nn.Linear(c['encoder']['code_dim'], c['head_width']), nn.SiLU(),
                                  nn.Linear(c['head_width'], 256))
        nn.init.normal_(self.head[-1].weight, std=.001)
        nn.init.zeros_(self.head[-1].bias)

    def graph(self, positions):
        batch, atoms, xyz = positions.shape
        if atoms != 256 or xyz != 3:
            raise ValueError(f'Expected complete 256-atom Al cells, got {positions.shape}')
        # Only discrete edge/image selection is detached. Neural radial and angular
        # features below MUST retain input-coordinate gradients for response loss.
        with torch.no_grad():
            delta = positions[:, None, :, :] - positions[:, :, None, :]
            shift = self.box*torch.round(delta/self.box)
            radius = (delta-shift).norm(dim=-1)
            allowed = (radius < self.encoder.cutoff) & ~torch.eye(atoms, dtype=torch.bool, device=positions.device)[None]
            b, i, j = allowed.nonzero(as_tuple=True)
            images = shift[b, i, j]
        x = positions.flatten(0, 1)
        edge = torch.stack((b*atoms+i, b*atoms+j))
        vectors = x[edge[1]] - x[edge[0]] - images
        attrs = x.new_ones(len(x), 1)
        radial, cutoff = self.encoder.radial_embedding(vectors.norm(dim=-1, keepdim=True),
            attrs, edge, self.encoder.atomic_numbers)
        if cutoff is not None:
            raise ValueError('Expected the recorded radial-embedded smooth cutoff')
        return dict(attrs=attrs, center=torch.zeros_like(attrs), weight=x.new_ones(len(x)),
            edge=edge, angular=self.encoder.spherical_harmonics(vectors), radial=radial,
            group=torch.arange(batch, device=x.device).repeat_interleave(atoms),
            centers=torch.arange(batch, device=x.device)*atoms, size=batch)

    def encode(self, positions):
        return self.encoder(self.graph(positions))

    def forward(self, positions):
        return self.head(self.encode(positions))


def responses(model, q, basis, *, create_graph=False):
    if basis.shape != (*q.shape, 2):
        raise ValueError(f'Expected two full-cell directions: {basis.shape} versus {q.shape}')
    return torch.stack([torch.autograd.functional.jvp(model, q, basis[..., j],
        create_graph=create_graph, strict=True)[1] for j in range(2)], -1)


@torch.no_grad()
def initialize(model, q, microbatch):
    pooled = torch.cat([model.encoder.pooled_graph(model.graph(x)) for x in q.split(microbatch)])
    model.encoder.pooled_mean.copy_(pooled.mean(0))
    model.encoder.pooled_scale.copy_(pooled.std(0, unbiased=False).clamp_min(1e-5))
