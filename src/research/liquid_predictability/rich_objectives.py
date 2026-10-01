"""Declared TDA blocks, structured readouts and topology-distance supervision."""
import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .rich_descriptor_head import NormalizedResidualHead


def descriptor_blocks(columns):
    blocks = {}
    for i, column in enumerate(columns):
        name = column['name']
        if column['family'] != 'tda':
            key = column['family']
        else:
            match = re.fullmatch(r'tda/(n(?:32|80)_h[012])_(.+)', name)
            if not match:
                raise ValueError(f'Unrecognized TDA producer column: {name}')
            prefix, suffix = match.groups()
            kind = ('spectrum' if re.fullmatch(r'(image|death)\d+', suffix)
                    else 'betti' if suffix.startswith('betti') else 'statistics')
            key = f'tda_{prefix}_{kind}'
        blocks.setdefault(key, []).append(i)
    if sorted(i for values in blocks.values() for i in values) != list(range(len(columns))):
        raise ValueError('Descriptor blocks must partition the output')
    return blocks


def objective_weights(data, settings):
    """Keep exported predictions/metrics in the original coordinate standardization."""
    blocks = descriptor_blocks(data.columns)
    weights = np.zeros(len(data.mean), np.float32)
    topology = np.zeros_like(weights)
    tda_blocks = [key for key in blocks if key.startswith('tda_')]
    spectra = [key for key in tda_blocks if key.endswith('_spectrum')]
    for key, indices in blocks.items():
        ids = np.asarray(indices)[data.active[indices]]
        if not len(ids):
            continue
        # Per-block error in raw units / summed fitting variances in that block.
        variance = data.scale[ids].astype(np.float64) ** 2
        block_weights = variance / variance.sum()
        if settings['normalization'] == 'coordinate':
            block_weights = np.full(len(ids), 1 / len(ids))
        elif settings['normalization'] != 'block':
            raise ValueError(settings['normalization'])
        weights[ids] = block_weights / (4 * (len(tda_blocks) if key.startswith('tda_') else 1))
        if key in spectra:
            topology[ids] = variance / variance.sum() / len(spectra)
    if settings['weighting'] == 'family':
        weights = data.loss_weight.copy()
    elif settings['weighting'] != 'structured':
        raise ValueError(settings['weighting'])
    if not np.isclose(weights.sum(), 1) or not np.isclose(topology.sum(), 1):
        raise ValueError('A declared descriptor block has no active targets')
    return weights, topology


class StructuredDescriptorHead(NormalizedResidualHead):
    """Shared residual trunk with a nonlinear readout for each semantic block."""

    def __init__(self, inputs, outputs, config):
        super().__init__(inputs, outputs, dict(config, kind='normalized_residual_v1'))
        del self.output
        self.heads = nn.ModuleList()
        self.indices = []
        for ids in config['output_blocks'].values():
            self.indices.append(ids)
            self.heads.append(nn.Sequential(nn.Linear(config['width'], config['block_width']),
                nn.LayerNorm(config['block_width'], elementwise_affine=False), nn.SiLU(),
                nn.Linear(config['block_width'], len(ids))))
        with torch.no_grad():
            for head in self.heads:
                head[-1].weight.mul_(.01)
                head[-1].bias.zero_()

    def forward(self, state):
        with torch.autocast(device_type=state.device.type, enabled=False):
            x = self.input_norm(state.float())
            h = self.stem(x)
            for block in self.blocks:
                h = h + self.residual_scale * block(h)
            h = self.output_norm(h)
            result = self.skip(x)
            for ids, head in zip(self.indices, self.heads):
                result[:, ids] = result[:, ids] + head(h)
            return result


def topology_distance_loss(state, targets, weights, settings):
    """TDL ordered contrastive loss (Luo et al. 2023), on geometry-only spectra.

    Uniformly spaced rows of the globally shuffled batch form the comparison
    panel. All ranks take the same panel after an autograd-aware gather. Ties
    include every equally distant candidate; self-pairs are excluded.
    """
    from src.research.crystal_vector.parallel import world_size
    if world_size() > 1:
        from torch.distributed.nn.functional import all_gather
        state = torch.cat(all_gather(state), 0)
        targets = torch.cat(all_gather(targets), 0)
    n = min(len(state), settings['panel_size'])
    ids = torch.arange(n, device=state.device) * len(state) // n
    z = F.normalize(state[ids].float(), dim=1)
    y = targets[ids].float() * weights.sqrt()
    distances = torch.cdist(y, y).square()
    distances.fill_diagonal_(-torch.inf)
    order = distances.argsort(dim=1, descending=True, stable=True)
    sorted_distance = distances.gather(1, order)[:, :-1]
    logits = z @ z.T / settings['temperature']
    sorted_logits = logits.gather(1, order)[:, :-1]
    # searchsorted on negated distances finds the last tied candidate, inclusively.
    tie_end = torch.searchsorted((-sorted_distance).contiguous(),
                                  (-sorted_distance).contiguous(), right=True) - 1
    normalizer = sorted_logits.logcumsumexp(dim=1).gather(1, tie_end)
    return (normalizer - sorted_logits).mean()
