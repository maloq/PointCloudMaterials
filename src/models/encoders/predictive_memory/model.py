"""Joint native MACE state with strictly radius-limited, masked atom histories."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
# Installed e3nn stores Python slices in its packaged Wigner constants. Keep
# PyTorch's safe loader enabled and allow that exact builtin only for the import.
with torch.serialization.safe_globals([slice]):
    from e3nn import o3
from src.models.encoders.mace_causal import (
    CausalMACEEncoder, CausalAtomAttention, SmoothMultiscalePooling,
    invariants, normalize_atom_features, taper,
)


class MaskedAttention(CausalAtomAttention):
    """Exact full causal attention; linear equivariant values are summed first.

    This avoids the [T,T,N,9C] value tensor, without truncating history. Constant
    age support retains the oldest frame; no slowness/window-envelope objective.
    """
    def forward(self, h, positions, times, support):
        inv = invariants(h, self.channels)
        score = torch.einsum('knc,jnc->kjn', self.query(inv), self.key(inv))/self.channels**.5
        dt = times[:, None]-times[None, :]
        dr = (positions[:, None]-positions[None, :])/self.cutoff
        pair = torch.stack((dt[:, :, None].expand_as(score)/self.duration, dr.square().sum(-1)), -1)
        score = 2*torch.tanh((score+self.time_geometry(pair).squeeze(-1))/2)
        weight = torch.exp(score)*support[None]*(dt >= 0)[:, :, None]
        weight = weight/weight.sum(1, keepdim=True).clamp_min(torch.finfo(h.dtype).tiny)
        displacement = o3.spherical_harmonics([1, 2], dr, normalize=False, normalization='component')
        update = torch.einsum('kjn,jnc->knc', weight, self.value(h))
        update = update+self.displacement(torch.einsum('kjn,kjnd->knd', weight, displacement))
        return (h+self.gate(inv)*update)*support[..., None]


class MaskedPooling(SmoothMultiscalePooling):
    def forward(self, h, positions, observation_weights):
        radius, inv = positions.norm(dim=-1), invariants(h, self.channels)
        summaries = []
        for (inner, outer), score, value in zip(self.scales, self.scores, self.values, strict=True):
            support = taper(radius, inner, outer)*observation_weights
            weight = support*torch.exp(5*torch.tanh(score(inv).squeeze(-1)/5))
            weight = weight/weight.sum().clamp_min(torch.finfo(h.dtype).tiny)
            pooled = (weight[:, None]*h).sum(0)
            summaries.extend(((weight[:, None]*value(inv)).sum(0),
                (weight[:, None]*inv[:, self.channels:]).sum(0),
                invariants(pooled, self.channels)[self.channels:], torch.log1p(support.sum()).reshape(1)))
        return self.compress(torch.cat(summaries)).unsqueeze(0)


class PredictiveMemoryEncoder(CausalMACEEncoder):
    def __init__(self, *, channels=16, output_dim=128, num_layers=2, correlation=2,
                 frame_chunk=4, radius_A=17., cutoff_A=5., use_velocity=True,
                 use_history=True, repeat_anchor=False, activation_checkpoint=True):
        super().__init__(channels=channels, output_dim=output_dim, num_layers=num_layers,
            correlation=correlation, cutoff_A=cutoff_A, history_duration_ps=48.,
            scales_A=((0., 3.), (5., 7.), (radius_A-2, radius_A)),
            use_velocity=use_velocity, use_history=use_history, repeat_anchor=repeat_anchor)
        irreps = o3.Irreps(f'{channels}x0e + {channels}x1o + {channels}x2e')
        self.temporal = nn.ModuleList([MaskedAttention(irreps, channels, 48., cutoff_A) for _ in range(num_layers)])
        self.pool = MaskedPooling(channels, self.pool.scales, output_dim)
        self.radius_A, self.frame_chunk = radius_A, frame_chunk
        self.activation_checkpoint = activation_checkpoint

    def spatial(self, h, x, u, w, edges, layer):
        attrs = torch.ones(len(h), 1, dtype=x.dtype, device=x.device)
        sender, receiver = edges
        vectors = x[receiver]-x[sender]
        angular = self.spherical_harmonics(vectors)
        radial, cutoff = self.radial_embedding(vectors.norm(dim=-1, keepdim=True), attrs, edges, self.atomic_numbers)
        if self.use_velocity:
            du = u[receiver]-u[sender]
            motion = torch.cat((du.square().sum(-1, keepdim=True),
                (vectors*du).sum(-1, keepdim=True)/self.cutoff_A,
                (u[sender]*u[receiver]).sum(-1, keepdim=True)), -1)
            radial = radial*(1+torch.tanh(self.edge_motion[layer](motion)))
        h, sc = self.interactions[layer](node_attrs=attrs, node_feats=h*w[:, None], edge_attrs=angular,
            edge_feats=radial, edge_index=edges, cutoff=cutoff, first_layer=layer == 0)
        h = self.products[layer](node_feats=h, sc=sc, node_attrs=attrs)
        if self.use_velocity:
            h = h+self.motion_vectors[layer](o3.spherical_harmonics([1, 2], u, normalize=False, normalization='component'))
        return normalize_atom_features(h)*w[:, None]

    def forward(self, observation):
        if observation.radius_A != self.radius_A or observation.cutoff_A != self.cutoff_A:
            raise ValueError('Observation radius/cutoff disagree with the encoder')
        if self.repeat_anchor:
            observation = observation.repeated_anchor()
        x, u, w, times = observation.positions, observation.velocities/self.velocity_scale, observation.weights, observation.offsets_ps
        t, n = x.shape[:2]
        if not self.use_history and t != 1:
            raise ValueError('Snapshot encoder requires a snapshot observation')
        if times[-1] != 0 or times[0] < -48 or torch.any(times[1:] <= times[:-1]):
            raise ValueError('Require full causal physical offsets within [-48, 0]')
        h = self.node_embedding(torch.ones(t*n, 1, device=x.device, dtype=x.dtype)).reshape(t, n, -1)
        if self.use_velocity:
            h = h+self.initial_motion(torch.stack((u.square().sum(-1), (x*u).sum(-1)/self.cutoff_A), -1))
        # Chunk spatial work, preserving every node, edge and frame. Checkpointing
        # recomputes activations; it does not detach historical learned features.
        use_checkpoint = self.training and torch.is_grad_enabled() and self.activation_checkpoint
        for layer in range(self.num_layers):
            chunks = []
            for start in range(0, t, self.frame_chunk):
                end = min(t, start+self.frame_chunk)
                edges = torch.cat([observation.edges[k]+(k-start)*n for k in range(start, end)], 1)
                args = (h[start:end].flatten(0, 1), x[start:end].flatten(0, 1),
                        u[start:end].flatten(0, 1), w[start:end].flatten(), edges)
                def run_spatial(*args, current_layer=layer):
                    return self.spatial(*args, current_layer)
                result = checkpoint(run_spatial, *args, use_reentrant=False) if use_checkpoint else run_spatial(*args)
                chunks.append(result.reshape(end-start, n, -1))
            h = torch.cat(chunks)
            if self.use_history:
                pieces = []
                for start in range(0, n, 256):
                    args = (h[:, start:start+256], x[:, start:start+256], times.to(x.dtype), w[:, start:start+256])
                    block = self.temporal[layer]
                    pieces.append(checkpoint(block, *args, use_reentrant=False) if use_checkpoint else block(*args))
                h = normalize_atom_features(torch.cat(pieces, 1))*w[..., None]
        return self.pool(h[-1], x[-1], w[-1])
