"""Native MACE with resident immutable geometry and code-only physical heads."""
import numpy as np
import torch
from torch import nn

from src.training_methods.bcr.model import Encoder
from src.training_methods.bcr.data import taper
from src.models.encoders.mace_causal import normalize_atom_features
from src.models.encoders.graph_bank import GraphBank
from .data import BLOCKS


class GeometryEncoder(Encoder):
    def __init__(self, **config):
        super().__init__(**config)
        width = 2 * self.channels
        if config['code_dim'] <= width:
            raise ValueError('Export must contain pooled channels and a learned residual')
        self.register_buffer('pooled_mean', torch.zeros(width))
        self.register_buffer('pooled_scale', torch.ones(width))
        self.readout = nn.Sequential(nn.Linear(width, 128), nn.SiLU(),
                                     nn.Linear(128, config['code_dim'] - width))
        # The initial state exposes the pooled signal directly. New channels can
        # learn additional functions without hiding that signal behind an MLP.
        nn.init.normal_(self.readout[-1].weight, std=.01)
        nn.init.zeros_(self.readout[-1].bias)

    def pooled_graph(self, graph):
        attrs, w = graph['attrs'], graph['weight']
        h = self.node_embedding(attrs) + self.center_embedding(graph['center'])
        for k, (interaction, product) in enumerate(zip(self.interactions, self.products, strict=True)):
            message, skip = interaction(node_attrs=attrs, node_feats=h, edge_attrs=graph['angular'],
                edge_feats=graph['radial'], edge_index=graph['edge'], cutoff=graph['cutoff'], first_layer=k == 0)
            h = normalize_atom_features(product(message, sc=skip, node_attrs=attrs)) * w[:, None]
        h = h[:, :self.channels]
        pooled = h.new_zeros(graph['size'], self.channels).index_add(0, graph['group'], h * w[:, None]) / self.n_ref
        return torch.cat((h[graph['centers']], pooled), -1)

    def export_pooled(self, pooled):
        normalized = (pooled - self.pooled_mean) / self.pooled_scale
        return torch.cat((normalized, self.readout(normalized)), -1)

    def forward(self, graph):
        return self.export_pooled(self.pooled_graph(graph))


class StructuralModel(nn.Module):
    def __init__(self, config, dynamics=False):
        super().__init__()
        self.encoder = GeometryEncoder(**config)
        # Every arm has the same initial encoder and heads, including unused heads.
        self.heads = nn.ModuleDict({d: nn.Linear(config['code_dim'], 89) for d in ('observed', 'relaxed')})
        if dynamics:
            self.heads.update({d: nn.Linear(config['code_dim'], 8)
                               for d in ('current_order', 'future_residual')})

    def forward(self, graph):
        return self.encoder(graph)

    @torch.no_grad()
    def bound_heads(self, maximum):
        for head in self.heads.values():
            head.weight.mul_((maximum / head.weight.norm().clamp_min(1e-12)).clamp_max(1))


def block_error(prediction, target):
    return torch.stack([(prediction[:, sl] - target[:, sl]).square().mean(-1)
                        for sl in BLOCKS.values()], -1).mean(-1)


def pair_distances(values):
    if len(values) % 2:
        raise ValueError('Relational endpoints must stay in adjacent pairs')
    return (values[0::2] - values[1::2]).square().mean(-1).clamp_min(1e-20).sqrt()


def teacher_distances(values):
    return block_error(values[0::2], values[1::2]).clamp_min(1e-20).sqrt()


def objective(model, z, targets, arm, relation_scale):
    loss = block_error(model.heads[arm['input']](z), targets[arm['input']]).mean()
    if arm['relaxed_weight']:
        loss = loss + arm['relaxed_weight'] * block_error(model.heads['relaxed'](z), targets['relaxed']).mean()
    if 'current_order' in model.heads:
        loss = loss + arm['current_weight'] * (model.heads['current_order'](z)-targets['current_order']).square().mean()
        if arm['future_weight']:
            loss = loss + arm['future_weight'] * (model.heads['future_residual'](z)-targets['future_residual']).square().mean()
    distances = pair_distances(z)
    if arm['relation_weight']:
        scale_z, scale_g = relation_scale
        if scale_z <= 0 or scale_g <= 0:
            raise ValueError('Positive training-only relational distance scales required')
        relation = nn.functional.huber_loss(distances / scale_z, teacher_distances(targets[arm['input']]) / scale_g)
        loss = loss + arm['relation_weight'] * relation
    return loss, distances.detach().mean()


@torch.no_grad()
def calibrate_heads(model, features, targets, ridge, maximum):
    """Fit-only ridge initialization with an explicit bounded weight norm.

    The intercept is unpenalized. Increase the predetermined penalty until the
    norm cap is satisfied; do not clip an already fitted solution or use tuning
    observations to choose the initialization.
    """
    x = features.double()
    mean = x.mean(0)
    centered = x - mean
    gram = centered.T @ centered
    eye = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    receipt = {}
    for domain, head in model.heads.items():
        y = targets[domain].double()
        ym = y.mean(0)
        cross = centered.T @ (y - ym)
        alpha = float(ridge)
        for _ in range(16):
            weight = torch.linalg.solve(gram + alpha * eye, cross).T
            if float(weight.norm()) <= maximum:
                break
            alpha *= 10
        else:
            raise ValueError(f'Could not calibrate a bounded physical head: {domain}')
        head.weight.copy_(weight)
        head.bias.copy_(ym - mean @ weight.T)
        receipt[domain] = dict(ridge=alpha, weight_norm=float(weight.norm()))
    return receipt
