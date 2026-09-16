"""Local phase-space extension of the smooth-inner MACE encoder.

The structure block is independent of velocities. Activity is even and flow is
odd under velocity reversal. All blocks are O(3), translation, permutation and
uniform-velocity-boost invariant. No time index or trajectory-level input is used.
"""

import torch
from torch import nn

from .mace_context import context_features


def weighted_pool(values, graph):
    total = values.new_zeros(graph.batch_size)
    total.index_add_(0, graph.pool_index, graph.weights)
    result = values.new_zeros(graph.batch_size, values.shape[1])
    result.index_add_(0, graph.pool_index, values * graph.weights[:, None])
    return result / total[:, None]


def velocity_moments(graph, velocities, velocity_scale=4.):
    """Smooth radial velocity messages at each target: 33 even + 16 odd scalars."""
    if graph.input_index is None or graph.input_batch is None:
        raise ValueError('Velocity messages require make_context_graph input alignment')
    v = velocities[graph.input_index] / velocity_scale
    target_input = graph.first_keep[graph.second_keep]
    bulk = weighted_pool(v[target_input], graph)
    u = v - bulk[graph.input_batch]
    edges = graph.first_keep[graph.second_edges]
    sender, receiver = edges
    r = graph.positions[sender] - graph.positions[receiver]
    distance = r.norm(dim=1)
    direction = r / distance[:, None]
    du = u[sender] - u[receiver]
    radial = (direction * du).sum(1)
    local_radial = (direction * u[receiver]).sum(1)
    even = torch.stack((du.square().sum(1), radial.square(),
                        local_radial.square(), (u[sender]*u[receiver]).sum(1)), 1)
    odd = torch.stack((radial, local_radial), 1)
    centers = torch.linspace(1.5, 4.8, 8, device=r.device, dtype=r.dtype)
    t = ((distance-3.5)/1.5).clamp(0, 1)
    cutoff = (1-t).pow(3)*(1+3*t+6*t.square())
    basis = torch.exp(-.5*((distance[:, None]-centers)/.5).square()) * cutoff[:, None] / 12.
    # Receivers are exactly the sorted target indices in the pruned first layer.
    target = torch.searchsorted(graph.second_keep, graph.second_edges[1])
    outputs = []
    for values in (even, odd):
        messages = (basis[:, :, None] * values[:, None, :]).flatten(1)
        result = messages.new_zeros(len(target_input), messages.shape[1])
        result.index_add_(0, target, messages)
        outputs.append(result)
    return torch.cat((u[target_input].square().sum(1, keepdim=True), outputs[0]), 1), outputs[1]


class MACEVelocityEncoder(nn.Module):
    """304 channels: structure[256], activity[32], directed motion[16]."""

    def __init__(self, mace, feature_mean, feature_scale, *, use_velocity=True, velocity_scale=4.):
        super().__init__()
        self.mace = mace
        self.use_velocity = use_velocity
        self.velocity_scale = velocity_scale
        self.register_buffer('feature_mean', torch.as_tensor(feature_mean, dtype=torch.float32))
        self.register_buffer('feature_scale', torch.as_tensor(feature_scale, dtype=torch.float32))
        self.activity = nn.Sequential(nn.Linear(289, 96), nn.SiLU(), nn.Linear(96, 32))
        self.flow = nn.Sequential(nn.Linear(305, 96), nn.SiLU(), nn.Linear(96, 16))

    def forward(self, graph, velocities):
        nodes = context_features(self.mace, graph, return_nodes=True)
        h = (nodes-self.feature_mean)/self.feature_scale
        structure = weighted_pool(h, graph)
        if not self.use_velocity:
            return torch.cat((structure, structure.new_zeros(len(structure), 48)), 1)
        even, odd = velocity_moments(graph, velocities, self.velocity_scale)
        activity = self.activity(torch.cat((h, even), 1)) - self.activity(torch.cat((h, torch.zeros_like(even)), 1))
        flow = .5*(self.flow(torch.cat((h, even, odd), 1))-self.flow(torch.cat((h, even, -odd), 1)))
        return torch.cat((structure, weighted_pool(activity, graph), weighted_pool(flow, graph)), 1)
