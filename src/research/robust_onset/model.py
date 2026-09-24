"""Smooth multiscale contractions retain local tensor strength AND alignment."""
import numpy as np
import torch
from torch import nn

from src.training_methods.bcr.data import taper
from src.models.encoders.mace_causal import normalize_atom_features
from src.research.structural_state.model import GeometryEncoder, GraphBank as FixedGraphBank, gather_indices


class GraphBank(FixedGraphBank):
    def __init__(self, arrays, encoder, device):
        super().__init__(arrays, encoder, device)
        self.radius = torch.as_tensor(np.linalg.norm(arrays['positions'], axis=1), device=device)

    def batch(self, indices):
        result = super().batch(indices)
        nodes, _, _ = gather_indices(self.offsets, np.asarray(indices, dtype=np.int64))
        result['radius'] = self.radius[torch.as_tensor(nodes, device=self.device)]
        return result


class TensorEncoder(GeometryEncoder):
    def __init__(self, **config):
        super().__init__(**config)
        self.scales = (4., 6., 8.)
        width = self.channels + len(self.scales)*(5*self.channels+1)
        self.pooled_mean = torch.zeros(width)
        self.pooled_scale = torch.ones(width)
        self.readout = nn.Sequential(nn.Linear(width, 128), nn.SiLU(), nn.Linear(128, config['code_dim']))

    def pooled_graph(self, graph):
        attrs, w = graph['attrs'], graph['weight']
        h = self.node_embedding(attrs) + self.center_embedding(graph['center'])
        for k, (interaction, product) in enumerate(zip(self.interactions, self.products, strict=True)):
            message, skip = interaction(node_attrs=attrs, node_feats=h, edge_attrs=graph['angular'],
                edge_feats=graph['radial'], edge_index=graph['edge'], cutoff=graph['cutoff'], first_layer=k == 0)
            h = normalize_atom_features(product(message, sc=skip, node_attrs=attrs))*w[:, None]
        c, b, g = self.channels, graph['size'], graph['group']
        if h.shape[1] != 9*c:
            raise ValueError(f'Expected native MACE 0e+1o+2e channels; got {h.shape}')
        scalar, vector, tensor = h[:, :c], h[:, c:4*c].reshape(-1,c,3), h[:, 4*c:].reshape(-1,c,5)
        parts = [scalar[graph['centers']]]
        for radius in self.scales:
            weight = taper(graph['radius'], radius)
            # Fixed density scale avoids discontinuous neighbor-count denominators.
            norm = self.n_ref*(radius/self.radius)**3
            def pool(value):
                shape = (b,)+value.shape[1:]
                weighted = value*weight.reshape((-1,)+(1,)*(value.ndim-1))
                return value.new_zeros(shape).index_add(0, g, weighted)/norm
            parts += [pool(scalar), pool(vector.square().sum(-1)), pool(vector).square().sum(-1),
                      pool(tensor.square().sum(-1)), pool(tensor).square().sum(-1), pool(torch.ones_like(weight)[:,None])]
        return torch.cat(parts, dim=1)

    def export_pooled(self, pooled):
        return self.readout((pooled-self.pooled_mean)/self.pooled_scale)


class Model(nn.Module):
    def __init__(self, encoder_config, tensor_pool, temperatures):
        super().__init__()
        self.encoder = (TensorEncoder if tensor_pool else GeometryEncoder)(**encoder_config)
        d = encoder_config['code_dim']
        self.heads = nn.ModuleDict({k: nn.Linear(d, n) for k,n in
            [('observed',89), ('relaxed',89), ('current',8), ('future',24)]})
        self.hazard = nn.Sequential(nn.Linear(d+temperatures,64), nn.SiLU(), nn.Linear(64,5))

    def forward(self, graph):
        return self.encoder(graph)

    def logits(self, z, conditions):
        return self.hazard(torch.cat((z, conditions), dim=1))

    @torch.no_grad()
    def bound_heads(self, maximum=10.):
        for layer in self.modules():
            if isinstance(layer, nn.Linear) and layer not in self.encoder.modules():
                layer.weight.mul_(min(1., maximum/float(layer.weight.norm().clamp_min(1e-12))))
