"""One trainable MACE state: geometry/motion/history interact before pooling."""
import numpy as np
import torch
from torch import nn
from mace import modules
from e3nn import o3

from src.data_utils.causal_history import AtomicHistory
from .mace_causal_batch import PackedHistory, graph_sum
from .mace_backend import mace_backend_config


def taper(value, inner, outer):
    """C2 compact support, including inside attention normalization."""
    a = ((value-inner)/(outer-inner)).clamp(0, 1)
    return (1-a).pow(3)*(1+3*a+6*a.square())


def invariants(h, channels):
    scalar, vector, tensor = torch.split(h, [channels, 3*channels, 5*channels], dim=-1)
    v2 = vector.reshape(*h.shape[:-1], channels, 3).square().sum(-1)
    q2 = tensor.reshape(*h.shape[:-1], channels, 5).square().sum(-1)
    return torch.cat((scalar, v2, q2), -1)


def normalize_atom_features(h):
    """Bound product amplitudes using one rotation-invariant scale per atom.

    The additive one retains amplitude information; this is not population
    whitening or a variance/collapse objective on the exported state.
    """
    return h/torch.sqrt(1+h.square().mean(-1, keepdim=True))


class CausalAtomAttention(nn.Module):
    def __init__(self, irreps, channels, history_duration_ps, cutoff_A):
        super().__init__()
        self.channels = channels
        self.duration = history_duration_ps
        self.cutoff = cutoff_A
        self.query = nn.Linear(3*channels, channels, bias=False)
        self.key = nn.Linear(3*channels, channels, bias=False)
        self.time_geometry = nn.Sequential(nn.Linear(2, channels), nn.SiLU(), nn.Linear(channels, 1))
        self.value = o3.Linear(irreps, irreps)
        self.displacement = o3.Linear('1x1o + 1x2e', irreps)
        self.gate = nn.Sequential(nn.Linear(3*channels, channels), nn.SiLU(), nn.Linear(channels, 1), nn.Sigmoid())

    def forward(self, h, positions, times):
        inv = invariants(h, self.channels)
        # Axes: query frame, sender frame, persistent atom, channel.
        scores = torch.einsum('knc,jnc->kjn', self.query(inv), self.key(inv))/self.channels**.5
        dt = times[:, None]-times[None, :]
        dr = (positions[:, None]-positions[None, :])/self.cutoff
        pair = torch.stack((dt[:, :, None].expand_as(scores)/self.duration, dr.square().sum(-1)), -1)
        scores = scores+self.time_geometry(pair).squeeze(-1)
        # Bound odds so extreme learned logits cannot defeat a vanishing age
        # envelope at practical precision near the sliding-window boundary.
        scores = 2*torch.tanh(scores/2)
        causal = dt >= 0
        age = taper(-times, 0., self.duration)
        support = causal[:, :, None]*age[None, :, None]
        # Zero-support frames are excluded from the stabilization maximum too.
        allowed = support > 0
        maximum = scores.masked_fill(~allowed, -torch.inf).amax(1, keepdim=True)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        weights = torch.exp((scores-maximum).masked_fill(~allowed, -torch.inf))*support
        weights = weights/weights.sum(1, keepdim=True).clamp_min(torch.finfo(h.dtype).tiny)
        displacement = o3.spherical_harmonics([1, 2], dr, normalize=False, normalization='component')
        values = self.value(h)[None]+self.displacement(displacement)
        update = (weights[..., None]*values).sum(1)
        return h+self.gate(inv)*update


class SmoothMultiscalePooling(nn.Module):
    def __init__(self, channels, scales_A, output_dim):
        super().__init__()
        self.channels = channels
        self.scales = tuple(tuple(s) for s in scales_A)
        if not self.scales or any(len(s) != 2 or not 0 <= s[0] < s[1] for s in self.scales):
            raise ValueError('Pooling scales must contain [inner, outer] radii with 0 <= inner < outer')
        self.scores = nn.ModuleList([nn.Linear(3*channels, 1) for _ in self.scales])
        self.values = nn.ModuleList([nn.Sequential(nn.Linear(3*channels, channels), nn.SiLU())
                                     for _ in self.scales])
        self.compress = nn.Sequential(nn.Linear(len(self.scales)*(5*channels+1), output_dim),
                                      nn.SiLU(), nn.Linear(output_dim, output_dim))

    def forward(self, h, positions, node_graph=None, num_graphs=None):
        radius = positions.norm(dim=-1)
        inv = invariants(h, self.channels)
        summaries = []
        for (inner, outer), score, value in zip(self.scales, self.scores, self.values, strict=True):
            support = taper(radius, inner, outer)
            weight = support*torch.exp(5*torch.tanh(score(inv).squeeze(-1)/5))
            if node_graph is None:
                total = weight.sum()
                reduce = lambda v: v.sum(0)
            else:
                total = graph_sum(weight, node_graph, num_graphs)[node_graph]
                reduce = lambda v: graph_sum(v, node_graph, num_graphs)
            weight = weight/total.clamp_min(torch.finfo(h.dtype).tiny)
            mean_tensor = reduce(weight[:, None]*h)
            # Strength and alignment are distinct contractions of vector/tensor channels.
            strength = reduce(weight[:, None]*inv[:, self.channels:])
            alignment = invariants(mean_tensor, self.channels)[..., self.channels:]
            count = torch.log1p(reduce(support)).reshape(-1, 1) if node_graph is not None else torch.log1p(support.sum()).reshape(1)
            summaries.extend((reduce(weight[:, None]*value(inv)), strength, alignment, count))
        result = self.compress(torch.cat(summaries, dim=-1))
        return result.unsqueeze(0) if node_graph is None else result


class CausalMACEEncoder(nn.Module):
    """Native tensor MACE, initialized from scratch, with one current-state readout.

    This is a distinct architecture, not a loader for the old 128+128 scalar MLIP.
    Inputs/outputs are O(3) equivariant/invariant; times and distances are physical.
    """
    def __init__(self, *, channels=16, output_dim=128, num_layers=2, cutoff_A=5.,
                 atomic_numbers=(13,), correlation=2, num_bessel=6,
                 radial_width=32, avg_num_neighbors=12., history_duration_ps=2.25,
                 velocity_scale_A_per_ps=4., scales_A=((0., 3.), (5., 7.), (7., 9.)),
                 use_velocity=True, use_history=True, repeat_anchor=False, mace_backend='e3nn'):
        super().__init__()
        if (num_layers < 2 or channels < 1 or output_dim < 1 or cutoff_A <= 0
                or history_duration_ps <= 0 or velocity_scale_A_per_ps <= 0):
            raise ValueError('Require at least two spatial layers and positive widths, radii and scales')
        self.channels, self.invariant_dim = channels, output_dim
        self.cutoff_A, self.num_layers = cutoff_A, num_layers
        self.history_duration_ps = history_duration_ps
        self.velocity_scale = velocity_scale_A_per_ps
        self.use_velocity, self.use_history = use_velocity, use_history
        self.repeat_anchor = repeat_anchor
        self.mace_backend = mace_backend
        cueq_config = mace_backend_config(mace_backend)
        irreps = o3.Irreps(f'{channels}x0e + {channels}x1o + {channels}x2e')
        # Keep tensor channels even in the last product, rather than using the
        # scalar-only reshape of PretrainedMACEEncoder.
        self._mace_kwargs = dict(r_max=cutoff_A, num_bessel=num_bessel, num_polynomial_cutoff=5,
            max_ell=2, interaction_cls=modules.RealAgnosticResidualInteractionBlock,
            interaction_cls_first=modules.RealAgnosticInteractionBlock, num_interactions=num_layers,
            num_elements=len(atomic_numbers), hidden_irreps=irreps, MLP_irreps=o3.Irreps(f'{channels}x0e'),
            atomic_energies=np.zeros(len(atomic_numbers)), avg_num_neighbors=avg_num_neighbors,
            atomic_numbers=list(atomic_numbers), correlation=correlation, gate=torch.nn.functional.silu,
            radial_MLP=[radial_width], keep_last_layer_irreps=True)
        backbone = modules.MACE(**self._mace_kwargs, cueq_config=cueq_config).float()
        # Energy heads are not part of the state encoder.
        self.node_embedding = backbone.node_embedding
        self.radial_embedding = backbone.radial_embedding
        self.spherical_harmonics = backbone.spherical_harmonics
        self.interactions, self.products = backbone.interactions, backbone.products
        self.register_buffer('atomic_numbers', backbone.atomic_numbers)
        self.initial_motion = nn.Sequential(nn.Linear(2, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.edge_motion = nn.ModuleList([nn.Sequential(nn.Linear(3, radial_width), nn.SiLU(),
                                         nn.Linear(radial_width, num_bessel)) for _ in range(num_layers)])
        self.motion_vectors = nn.ModuleList([o3.Linear('1x1o + 1x2e', irreps) for _ in range(num_layers)])
        self.temporal = nn.ModuleList([CausalAtomAttention(irreps, channels, history_duration_ps, cutoff_A)
                                       for _ in range(num_layers)])
        self.pool = SmoothMultiscalePooling(channels, scales_A, output_dim)

    def atom_features(self, history: AtomicHistory | PackedHistory, *, validate=True):
        if history.cutoff_A != self.cutoff_A or history.spatial_layers < self.num_layers:
            raise ValueError('History graph does not cover this encoder message-passing radius/depth')
        if history.context_radius_A < max(s[1] for s in self.pool.scales):
            raise ValueError('History graph does not cover all pooling scales')
        if self.repeat_anchor:
            history = history.repeated_anchor()
        if not self.use_history:
            history = history.current()
        x, u = history.positions, history.velocities/self.velocity_scale
        t, n = x.shape[:2]
        offsets = history.offsets_ps
        if validate and (len(offsets) != t or offsets[-1] != 0 or torch.any(offsets > 0)
                or torch.any(offsets[1:] <= offsets[:-1])
                or offsets[0] < -self.history_duration_ps-1e-8):
            raise ValueError('Require causal increasing physical offsets ending at zero within history duration')
        match = history.atomic_numbers[:, None] == self.atomic_numbers[None]
        if validate and not torch.all(match.sum(1) == 1):
            raise ValueError(f'Unsupported species in history: {history.atomic_numbers.unique().tolist()}')
        attrs = match.to(x.dtype).repeat(t, 1)
        edges = history.edges
        sender, receiver = edges
        vectors = x.flatten(0, 1)[receiver]-x.flatten(0, 1)[sender]
        lengths = (history.boxes[sender//n, history.node_graph[sender % n]]
                   if isinstance(history, PackedHistory) else history.boxes[sender//n])
        vectors = vectors-lengths*torch.round(vectors/lengths)
        distances = vectors.norm(dim=-1, keepdim=True)
        if validate and torch.any(distances <= 0):
            raise ValueError('Coincident atoms on a spatial edge')
        angular = self.spherical_harmonics(vectors)
        radial, cutoff = self.radial_embedding(distances, attrs, edges, self.atomic_numbers)
        h = self.node_embedding(attrs)
        if self.use_velocity:
            inv = torch.stack((u.square().sum(-1), (x*u).sum(-1)/self.cutoff_A), -1).flatten(0, 1)
            h = h+self.initial_motion(inv)
            du = u.flatten(0, 1)[receiver]-u.flatten(0, 1)[sender]
            edge_motion = torch.cat((du.square().sum(-1, keepdim=True),
                                    (vectors*du).sum(-1, keepdim=True)/self.cutoff_A,
                                    (u.flatten(0, 1)[sender]*u.flatten(0, 1)[receiver]).sum(-1, keepdim=True)), -1)
            velocity_tensor = o3.spherical_harmonics([1, 2], u.flatten(0, 1),
                                                     normalize=False, normalization='component')
        for k, (interaction, product) in enumerate(zip(self.interactions, self.products, strict=True)):
            edge_features = radial
            if self.use_velocity:
                edge_features = radial*(1+torch.tanh(self.edge_motion[k](edge_motion)))
            h, sc = interaction(node_attrs=attrs, node_feats=h, edge_attrs=angular,
                edge_feats=edge_features, edge_index=edges, cutoff=cutoff, first_layer=k == 0)
            h = product(node_feats=h, sc=sc, node_attrs=attrs)
            if self.use_velocity:
                h = h+self.motion_vectors[k](velocity_tensor)
            h = normalize_atom_features(h).reshape(t, n, -1)
            if self.use_history:
                h = self.temporal[k](h, x, offsets.to(x.dtype))
            h = normalize_atom_features(h).flatten(0, 1)
        return h.reshape(t, n, -1)

    def forward(self, history: AtomicHistory | PackedHistory, *, validate=True):
        h = self.atom_features(history, validate=validate)[-1]
        result = (self.pool(h, history.positions[-1], history.node_graph, history.num_graphs)
                  if isinstance(history, PackedHistory) else self.pool(h, history.positions[-1]))
        if validate and not torch.isfinite(result).all():
            raise FloatingPointError('Nonfinite causal MACE embedding')
        return result
