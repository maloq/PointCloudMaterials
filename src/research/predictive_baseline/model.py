"""Present geometry -> native scalar MACE128 -> 274 future-statistic values."""
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.research.encoder_context.geometry import graph
from src.research.supervised_onset.model import CapacityEncoder


class Predictor(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.encoder = CapacityEncoder(**c['encoder'])
        self.head = nn.Sequential(nn.Linear(c['encoder']['code_dim'], c['head_width']), nn.SiLU(),
                                  nn.Linear(c['head_width'], 18 + c['target']['rff_features']))
        nn.init.normal_(self.head[-1].weight, std=.001)
        nn.init.zeros_(self.head[-1].bias)
        self.activation_checkpointing = c['training']['activation_checkpointing']

    def encode(self, positions):
        g = graph(positions, self.encoder)
        if self.training and torch.is_grad_enabled() and self.activation_checkpointing:
            return checkpoint(self.encoder, g, use_reentrant=False)
        return self.encoder(g)

    def forward(self, positions):
        return self.head(self.encode(positions))


@torch.no_grad()
def initialize(model, data, c, device):
    ids = np.flatnonzero(data['roles'] == 'train')
    w = data['weights'][ids]
    rng = np.random.default_rng(c['target']['seed'])
    ids = rng.choice(ids, c['training']['pool_normalization_samples'], replace=True, p=w/w.sum())
    batch = c['training']['batch_size']
    model.eval()
    pooled = torch.cat([model.encoder.pooled_graph(graph(torch.tensor(data['positions'][ix], device=device), model.encoder))
                        for ix in (ids[start:start+batch] for start in range(0, len(ids), batch))])
    model.encoder.pooled_mean.copy_(pooled.mean(0))
    model.encoder.pooled_scale.copy_(pooled.std(0, unbiased=False).clamp_min(1e-5))
