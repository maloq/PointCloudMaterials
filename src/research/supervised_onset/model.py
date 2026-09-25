"""Native cuEquivariance MACE with one exported supervised state."""
import torch
from torch import nn
from src.models.encoders.spatial_mace import SpatialMACE


class CapacityEncoder(SpatialMACE):
    """Width-independent 128-D export with a trainable projected residual.

    Keep the exact geometry pathway and fitting-only pool normalization. A
    projection replaces concatenation so the atom width is independent of the
    exported state width. Every capacity arm, including the small control, uses
    this same export rule.
    """
    def __init__(self, *, code_dim=128, **config):
        super().__init__(code_dim=code_dim, **config)
        width, output = 2*self.channels, code_dim
        self.register_buffer('pooled_mean', torch.zeros(width))
        self.register_buffer('pooled_scale', torch.ones(width))
        self.projection = nn.Linear(width, output, bias=False)
        nn.init.orthogonal_(self.projection.weight)
        self.readout = nn.Sequential(nn.Linear(width, 128), nn.SiLU(), nn.Linear(128, output))
        nn.init.normal_(self.readout[-1].weight, std=.01)
        nn.init.zeros_(self.readout[-1].bias)

    def export_pooled(self, pooled):
        normalized = (pooled-self.pooled_mean)/self.pooled_scale
        return self.projection(normalized)+self.readout(normalized)


class Model(nn.Module):
    def __init__(self, config, arm):
        super().__init__()
        self.input = arm['input']
        self.encoder = CapacityEncoder(**config)
        d = self.encoder.projection.out_features
        self.fusion = nn.Sequential(nn.Linear(3*d, 192), nn.SiLU(), nn.Linear(192, d)) if self.input == 'paired' else nn.Identity()
        self.hazard = nn.Sequential(nn.Linear(d, 128), nn.SiLU(), nn.Linear(128, 5))
        self.teacher_projection = nn.Linear(d, d) if arm['teacher'] else None

    def forward(self, banks, ids):
        if self.input == 'paired':
            hot = self.encoder(banks['hot'].batch(ids))
            cold = self.encoder(banks['cold'].batch(ids))
            # Shared weights, paired same-anchor views, one final 128-D export.
            return self.fusion(torch.cat((hot, cold, cold-hot), -1))
        return self.encoder(banks[self.input].batch(ids))

    def logits(self, z):
        return self.hazard(z)

    def prepare_pass(self, banks, ids, chunk):
        domains = ('hot', 'cold') if self.input == 'paired' else (self.input,)
        for domain in domains:
            banks[domain].prepare(ids, chunk)
