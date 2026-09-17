"""Causal atom/time attention using pinned upstream GATr blocks.

Observations are already centered and boost invariant. Atom IDs establish the
producer's cross-time correspondence only; they never enter learned features.
"""
from dataclasses import replace

import torch
from torch import nn
from gatr.interface import embed_point, embed_oriented_plane
from gatr.layers.linear import EquiLinear
from gatr.layers.gatr_block import GATrBlock
from gatr.layers.attention.config import SelfAttentionConfig
from gatr.layers.mlp.config import MLPConfig

GATR_REVISION = '6afc26f26b8fcf51136ae8c1d264a36e14b6e497'


def support_bias(weights):
    """Multiplicative smooth support inside attention normalization."""
    return weights.clamp_min(torch.finfo(weights.dtype).tiny).log().masked_fill(weights <= 0, -torch.inf)


def temporal_bias(weights, offsets):
    """[B,N,1,T,T] causal key support and C2 age envelope, in physical ps."""
    lag = offsets[:, :, None] - offsets[:, None, :]
    age = (lag / 12.).clamp(0, 1)
    envelope = (1 - age).pow(3) * (1 + 3 * age + 6 * age.square())
    bias = support_bias(weights.transpose(1, 2))[:, :, None, :] + support_bias(envelope)[:, None]
    bias = bias.masked_fill(lag[:, None] < 0, -torch.inf)
    # Absent queries are discarded. Give them a finite dummy self row to avoid
    # all-masked softmax backward on older SDPA kernels. Observed rows stay exact.
    absent = weights.transpose(1, 2) <= 0
    diagonal = torch.eye(weights.shape[1], dtype=torch.bool, device=weights.device)
    bias = torch.where(absent[..., None] & diagonal, 0., bias)
    return bias.unsqueeze(2)


class AxialGATrEncoder(nn.Module):
    def __init__(self, *, variant='snapshot', mv_channels=8, scalar_channels=128,
                 num_layers=2, activation_checkpoint=False):
        super().__init__()
        if variant not in ('snapshot', 'history12', 'repeat12'):
            raise ValueError(f'Unknown axial GATr variant: {variant}')
        self.variant = variant
        self.mv_channels = mv_channels
        self.scalar_channels = scalar_channels
        self.input = EquiLinear(2, mv_channels, in_s_channels=5, out_s_channels=scalar_channels)
        attention = SelfAttentionConfig(num_heads=4, pos_encoding=False)
        checkpoint = ['attention', 'mlp'] if activation_checkpoint else None
        def block():
            return GATrBlock(mv_channels, scalar_channels, attention, MLPConfig(), checkpoint=checkpoint)
        self.spatial = nn.ModuleList([block() for _ in range(num_layers)])
        self.temporal = nn.ModuleList([block() for _ in range(num_layers)])
        self.time_embedding = nn.ModuleList([nn.Linear(2, scalar_channels) for _ in range(num_layers)])
        self.history_alpha = nn.Parameter(torch.zeros(num_layers))
        # Fixed origin point, never a mean over atoms, frames, or batch peers.
        self.register_buffer('join_reference', embed_point(torch.zeros(3)))
        self.readout = nn.Sequential(nn.Linear(scalar_channels, 128), nn.SiLU(), nn.Linear(128, 128))

    def _select(self, observation):
        if observation.radius_A != 17. or observation.cutoff_A != 5.:
            raise ValueError('Axial GATr requires the same 17 A / 5 A native observations')
        if self.variant in ('snapshot', 'repeat12'):
            keep = observation.weights[-1] > 0
            current = replace(observation, positions=observation.positions[-1:, keep],
                velocities=observation.velocities[-1:, keep], weights=observation.weights[-1:, keep],
                atom_ids=observation.atom_ids[keep])
            if self.variant == 'snapshot':
                return replace(current, offsets_ps=observation.offsets_ps[-1:])
            t = len(observation.offsets_ps)
            return replace(current, positions=current.positions.expand(t, -1, -1),
                velocities=current.velocities.expand(t, -1, -1), weights=current.weights.expand(t, -1),
                offsets_ps=observation.offsets_ps)
        return observation

    def atom_features(self, observations):
        """Return scalar atom states for causal-prefix tests as well as readout.

        Prefix diagnostics retain their original physical offsets. The public
        forward additionally enforces the complete native observation cadence.
        """
        selected = [self._select(o) for o in observations]
        t = selected[0].weights.shape[0]
        if any(o.weights.shape[0] != t for o in selected):
            raise ValueError('A native batch must share its temporal grid')
        n = max(o.weights.shape[1] for o in selected)
        b = len(selected)
        x = selected[0].positions.new_zeros(b, t, n, 3)
        u = torch.zeros_like(x)
        w = x.new_zeros(b, t, n)
        centers = []
        for i, o in enumerate(selected):
            count = o.weights.shape[1]
            observed = o.weights > 0
            # Mask inputs before ANY geometric operation, including absent values
            # that were deliberately perturbed for an invariance diagnostic.
            x[i, :, :count] = torch.where(observed[..., None], o.positions, 0.) / 5.
            u[i, :, :count] = torch.where(observed[..., None], o.velocities, 0.) / 10.
            w[i, :, :count] = o.weights
            center = observed[-1] & (o.positions[-1] == 0).all(-1)
            if int(center.sum()) != 1:
                raise ValueError('Need exactly one observed zero-displacement center per window')
            centers.append(center.to(torch.int64).argmax())
        offsets = torch.stack([o.offsets_ps for o in selected]).to(x.dtype)
        if not bool((offsets[:, 1:] > offsets[:, :-1]).all()):
            raise ValueError('Observation times must increase strictly')
        mask = w > 0
        # The count retains density lost by normalized attention; no future or
        # other window contributes to this frame-local, smooth scalar summary.
        count = w.sum(-1, keepdim=True).expand_as(w) / 1000.
        scalars = torch.stack((w, x.square().sum(-1), u.square().sum(-1), (x*u).sum(-1), count), -1)
        vectors = torch.stack((embed_point(x), embed_oriented_plane(u, torch.zeros_like(u))), -2)
        mv, scalar = self.input(vectors, scalars=scalars)
        spatial_mask = support_bias(w).reshape(b*t, 1, 1, n)
        time_mask = temporal_bias(w, offsets).reshape(b*n, 1, t, t) if t > 1 else None
        time_features = torch.stack((offsets / 12., (offsets / 12.).square()), -1)
        for layer, block in enumerate(self.spatial):
            mv, scalar = block(mv.reshape(b*t, n, self.mv_channels, 16),
                scalar.reshape(b*t, n, self.scalar_channels), reference_mv=self.join_reference,
                attention_mask=spatial_mask)
            mv = mv.reshape(b, t, n, self.mv_channels, 16) * mask[..., None, None]
            scalar = scalar.reshape(b, t, n, self.scalar_channels) * mask[..., None]
            if self.variant != 'snapshot' and t > 1:
                time_scalar = scalar + self.time_embedding[layer](time_features)[:, :, None]
                next_mv, next_scalar = self.temporal[layer](
                    mv.transpose(1, 2).reshape(b*n, t, self.mv_channels, 16),
                    time_scalar.transpose(1, 2).reshape(b*n, t, self.scalar_channels),
                    reference_mv=self.join_reference, attention_mask=time_mask)
                next_mv = next_mv.reshape(b, n, t, self.mv_channels, 16).transpose(1, 2) * mask[..., None, None]
                next_scalar = next_scalar.reshape(b, n, t, self.scalar_channels).transpose(1, 2) * mask[..., None]
                gate = self.history_alpha[layer].tanh()
                mv = mv + gate * (next_mv - mv)
                scalar = scalar + gate * (next_scalar - scalar)
        return scalar, torch.stack(centers)

    def forward(self, observations):
        if not observations:
            raise ValueError('An empty batch has no states')
        for o in observations:
            expected = torch.arange(-12., .001, .75, device=o.offsets_ps.device, dtype=o.offsets_ps.dtype)
            if self.variant == 'snapshot':
                if float(o.offsets_ps[-1]) != 0.:
                    raise ValueError('Current snapshot must be at offset zero')
            elif not torch.equal(o.offsets_ps, expected):
                raise ValueError('History requires all 17 native frames ending at zero')
        features, centers = self.atom_features(observations)
        return self.readout(features[torch.arange(len(observations), device=features.device), -1, centers])
