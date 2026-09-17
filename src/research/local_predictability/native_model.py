"""Packed, exactly nested native states for local predictability protocol v1.

Reuse the audited per-atom MACE, observation and physical-packet producers.
Windows share spatial kernels but never edges, attention, or pooling support.
"""
from dataclasses import replace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from src.models.encoders.predictive_memory.model import PredictiveMemoryEncoder
from src.models.encoders.mace_causal import normalize_atom_features


def anchor_only(observation):
    """Keep exactly the current support, excluding history-only identities."""
    keep = observation.weights[-1] > 0
    mapping = torch.full((len(keep),), -1, dtype=torch.long, device=keep.device)
    mapping[keep] = torch.arange(int(keep.sum()), device=keep.device)
    edge = observation.edges[-1]
    valid = keep[edge[0]] & keep[edge[1]]
    return replace(observation, positions=observation.positions[-1:, keep],
                   velocities=observation.velocities[-1:, keep], weights=observation.weights[-1:, keep],
                   offsets_ps=observation.offsets_ps[-1:], atom_ids=observation.atom_ids[keep],
                   edges=(mapping[edge[:, valid]],))


class NativeEncoder(PredictiveMemoryEncoder):
    def __init__(self, *, variant='snapshot', max_spatial_edges=250000,
                 activation_checkpoint=True, channels=16, output_dim=128, num_layers=2):
        if variant not in ('snapshot', 'history12', 'repeat12'):
            raise ValueError(f'Unknown native variant: {variant}')
        super().__init__(channels=channels, output_dim=output_dim, num_layers=num_layers,
                         use_velocity=True, use_history=variant != 'snapshot',
                         repeat_anchor=variant == 'repeat12', activation_checkpoint=activation_checkpoint)
        self.variant = variant
        self.history_alpha = nn.Parameter(torch.zeros(num_layers))
        self.max_spatial_edges = max_spatial_edges
        for block in self.temporal:
            block.duration = 12.

    def forward(self, observations):
        if not isinstance(observations, (list, tuple)):
            observations = [observations]
        if not observations:
            raise ValueError('An empty native batch has no states')
        selected = []
        for observation in observations:
            if observation.radius_A != 17. or observation.cutoff_A != 5.:
                raise ValueError('Native protocol requires radius 17 A and cutoff 5 A')
            observation = anchor_only(observation) if self.variant == 'snapshot' else observation
            if self.variant == 'repeat12':
                # Remove identities outside current support before repeating.
                current = anchor_only(observation)
                count = len(observation.offsets_ps)
                observation = replace(current, positions=current.positions.expand(count, -1, -1),
                    velocities=current.velocities.expand(count, -1, -1),
                    weights=current.weights.expand(count, -1), edges=current.edges * count,
                    offsets_ps=observation.offsets_ps)
            times = observation.offsets_ps
            expected = torch.arange(-12., .001, .75, device=times.device, dtype=times.dtype)
            if self.variant == 'snapshot':
                expected = expected[-1:]
            if not torch.equal(times, expected):
                raise ValueError('Native input must contain the entire declared causal cadence')
            selected.append(observation)

        shapes = [o.weights.shape for o in selected]
        counts = [t * n for t, n in shapes]
        x = torch.cat([o.positions.flatten(0, 1) for o in selected])
        u = torch.cat([o.velocities.flatten(0, 1) for o in selected]) / self.velocity_scale
        w = torch.cat([o.weights.flatten() for o in selected])
        # Frame-sized graph units permit bounded packed spatial activations.
        frame_units, offset = [], 0
        for observation, (t, n) in zip(selected, shapes, strict=True):
            for edge in observation.edges:
                frame_units.append((offset, offset + n, edge + offset))
                offset += n
        chunks, pending, edge_count = [], [], 0
        for unit in frame_units:
            if pending and edge_count + unit[2].shape[1] > self.max_spatial_edges:
                chunks.append(pending)
                pending, edge_count = [], 0
            pending.append(unit)
            edge_count += unit[2].shape[1]
        if pending:
            chunks.append(pending)
        packed_edges = [(units[0][0], units[-1][1],
                         torch.cat([unit[2] for unit in units], 1) - units[0][0]) for units in chunks]
        h = self.node_embedding(torch.ones(len(x), 1, device=x.device, dtype=x.dtype))
        h = h + self.initial_motion(torch.stack((u.square().sum(-1), (x * u).sum(-1) / self.cutoff_A), -1))
        use_checkpoint = self.training and torch.is_grad_enabled() and self.activation_checkpoint
        for layer in range(self.num_layers):
            spatial = []
            for start, end, edges in packed_edges:
                args = (h[start:end], x[start:end], u[start:end], w[start:end], edges)
                def run_spatial(*values, current_layer=layer):
                    return self.spatial(*values, current_layer)
                spatial.append(checkpoint(run_spatial, *args, use_reentrant=False)
                               if use_checkpoint else run_spatial(*args))
            h = torch.cat(spatial)
            if self.variant != 'snapshot':
                parts, offset = [], 0
                for observation, (t, n), count in zip(selected, shapes, counts, strict=True):
                    current = h[offset:offset + count].reshape(t, n, -1)
                    temporal = []
                    for start in range(0, n, 256):
                        args = (current[:, start:start + 256], observation.positions[:, start:start + 256],
                                observation.offsets_ps.to(x.dtype), observation.weights[:, start:start + 256])
                        block = self.temporal[layer]
                        temporal.append(checkpoint(block, *args, use_reentrant=False)
                                        if use_checkpoint else block(*args))
                    candidate = normalize_atom_features(torch.cat(temporal, 1)) * observation.weights[..., None]
                    # Crucially gate normalization AND support effects. Alpha=0
                    # preserves every spatial feature and its shared gradient.
                    nested = current + torch.tanh(self.history_alpha[layer]) * (candidate - current)
                    parts.append(nested.flatten(0, 1))
                    offset += count
                h = torch.cat(parts)
        states, offset = [], 0
        for observation, (t, n), count in zip(selected, shapes, counts, strict=True):
            current = h[offset:offset + count].reshape(t, n, -1)[-1]
            states.append(self.pool(current, observation.positions[-1], observation.weights[-1]))
            offset += count
        return torch.cat(states)


class PhysicalMeans(nn.Module):
    """A present decoder and six linear conditional-mean heads, no label inputs."""
    def __init__(self, *, variant='snapshot', **encoder_options):
        super().__init__()
        self.encoder = NativeEncoder(variant=variant, **encoder_options)
        self.present = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128))
        self.future = nn.Linear(128 + 7, 6 * 128)

    def decode(self, state, conditions):
        if state.shape != (len(conditions), 128) or conditions.shape[1] != 7:
            raise ValueError('Require [batch,128] state and [batch,7] declared conditions')
        return self.present(state), self.future(torch.cat((state, conditions), -1)).reshape(-1, 6, 128)

    def forward(self, observations, conditions):
        state = self.encoder(observations)
        present, future = self.decode(state, conditions)
        return dict(state=state, present=present, future=future)
