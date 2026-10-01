"""Identical head initialization; optional nonnegative second-moment mapping."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.research.predictive_baseline.data import cache, moments


class FeatureHead(nn.Module):
    def __init__(self, network, c, arm):
        super().__init__()
        self.network = network
        self.dimensions = 274 if arm['target'] == 'full' else 18
        self.constrained = arm['variance'] == 'nonnegative'
        with np.load(cache(c)/'feature_map.npz') as f:
            center = f['center'].copy()
            scale = np.sqrt(f['metric'])
        self.register_buffer('center', torch.tensor(center, dtype=torch.float64))
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float64))
        self.floor = c['variance_floor']
        prior_variance = center[9:18]-center[:9]**2
        if np.any(prior_variance <= self.floor):
            raise ValueError('Invalid training prior variance for positive-variance initialization')
        # Inverse softplus maps the zero-network prediction back to the training prior.
        self.register_buffer('variance_bias', torch.tensor(np.log(np.expm1(prior_variance-self.floor)), dtype=torch.float64))

    def forward(self, z):
        value = self.network(z).double()
        if self.constrained:
            mean = value[:, :9]/self.scale[:9]+self.center[:9]
            variance = F.softplus(value[:, 9:18]/self.scale[9:18]+self.variance_bias)+self.floor
            second = (mean.square()+variance-self.center[9:18])*self.scale[9:18]
            value = torch.cat((value[:, :9], second, value[:, 18:]), -1)
        return value[:, :self.dimensions]


class Forecast(nn.Module):
    def __init__(self, c, arm, data, x):
        super().__init__()
        self.joint = arm['source'] == 'joint'
        self.c = c
        if self.joint:
            from src.research.predictive_baseline.model import Predictor
            original = Predictor(c)
            self.encoder = original.encoder
            network = original.head
        else:
            tr = data['roles'] == 'train'
            mean, scale = moments(x[tr], data['weights'][tr])
            self.register_buffer('input_mean', torch.tensor(mean, dtype=torch.float32))
            self.register_buffer('input_scale', torch.tensor(scale, dtype=torch.float32))
            network = nn.Sequential(nn.Linear(x.shape[1], c['head_width']), nn.SiLU(), nn.Linear(c['head_width'], 274))
            nn.init.normal_(network[-1].weight, std=.001)
            nn.init.zeros_(network[-1].bias)
        self.head = FeatureHead(network, c, arm)

    def encode(self, x):
        if self.joint:
            from src.research.encoder_context.geometry import graph
            from torch.utils.checkpoint import checkpoint
            g = graph(x, self.encoder)
            if self.training and torch.is_grad_enabled() and self.c['training']['activation_checkpointing']:
                return checkpoint(self.encoder, g, use_reentrant=False)
            return self.encoder(g)
        return (x-self.input_mean)/self.input_scale

    def forward(self, x):
        return self.head(self.encode(x))
