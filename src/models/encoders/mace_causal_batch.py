"""Disjoint packing of verified same-timeline atomic histories, without padding."""
from dataclasses import dataclass, replace

import torch

from src.data_utils.causal_history import AtomicHistory


@dataclass
class PackedHistory:
    positions: torch.Tensor             # [time, sum atoms, 3]
    velocities: torch.Tensor
    boxes: torch.Tensor                 # [time, graphs, 3]
    offsets_ps: torch.Tensor             # common physical offsets [time]
    atomic_numbers: torch.Tensor
    atom_ids: torch.Tensor               # IDs may repeat across independent graphs
    edges: torch.Tensor                  # flattened time-major disjoint nodes
    node_graph: torch.Tensor             # [sum atoms]
    num_graphs: int
    cutoff_A: float
    context_radius_A: float
    spatial_layers: int

    def to(self, device):
        return replace(self, **{k: v.to(device) for k, v in vars(self).items() if isinstance(v, torch.Tensor)})

    def current(self):
        n = self.positions.shape[1]
        start = (len(self.offsets_ps)-1)*n
        keep = self.edges[0] >= start
        return replace(self, positions=self.positions[-1:], velocities=self.velocities[-1:],
                       boxes=self.boxes[-1:], offsets_ps=self.offsets_ps[-1:], edges=self.edges[:, keep]-start)

    def repeated_anchor(self):
        now = self.current()
        t, n = self.positions.shape[:2]
        return replace(self, positions=now.positions.expand(t, -1, -1),
                       velocities=now.velocities.expand(t, -1, -1), boxes=now.boxes.expand(t, -1, -1),
                       edges=torch.cat([now.edges+k*n for k in range(t)], dim=1))


def pack_histories(histories, *, validated=False):
    """Same-ID temporal messages remain inside each sample's disjoint atom axis.

    Runtime validates all CPU histories once before transfer. Public callers get
    timeline/support checks by default. Graph construction/provenance is unchanged.
    """
    if not histories:
        raise ValueError('Cannot pack an empty atomic-history batch')
    first = histories[0]
    signature = (first.cutoff_A, first.context_radius_A, first.spatial_layers)
    for h in histories:
        if (h.cutoff_A, h.context_radius_A, h.spatial_layers) != signature:
            raise ValueError('Packed histories require identical verified spatial support/depth')
        if h.positions.device != first.positions.device or h.positions.dtype != first.positions.dtype:
            raise ValueError('Packed histories must share device and floating-point dtype')
        if not validated and not torch.equal(h.offsets_ps, first.offsets_ps):
            raise ValueError('Packed histories require the same physical offsets; do not pad or round times')
    sizes = [h.positions.shape[1] for h in histories]
    total = sum(sizes)
    edges, batch, offset = [], [], 0
    for i, (h, n) in enumerate(zip(histories, sizes, strict=True)):
        edges.append((h.edges//n)*total+(h.edges % n)+offset)
        batch.append(torch.full((n,), i, dtype=torch.long, device=h.positions.device))
        offset += n
    return PackedHistory(
        positions=torch.cat([h.positions for h in histories], dim=1),
        velocities=torch.cat([h.velocities for h in histories], dim=1),
        boxes=torch.stack([h.boxes for h in histories], dim=1), offsets_ps=first.offsets_ps,
        atomic_numbers=torch.cat([h.atomic_numbers for h in histories]),
        atom_ids=torch.cat([h.atom_ids for h in histories]), edges=torch.cat(edges, dim=1),
        node_graph=torch.cat(batch), num_graphs=len(histories), cutoff_A=first.cutoff_A,
        context_radius_A=first.context_radius_A, spatial_layers=first.spatial_layers)


def graph_sum(values, node_graph, num_graphs):
    shape = (num_graphs, *values.shape[1:])
    index = node_graph.reshape(-1, *([1]*(values.ndim-1))).expand_as(values)
    return values.new_zeros(shape).scatter_add(0, index, values)


def validate_history(history: AtomicHistory, model):
    """CPU-only one-time validation before enabling the synchronization-free path."""
    h = history
    if h.positions.device.type != 'cpu':
        raise ValueError('Validate producer histories on CPU before GPU residency')
    t, n, xyz = h.positions.shape
    if xyz != 3 or h.velocities.shape != (t, n, 3) or h.boxes.shape != (t, 3):
        raise ValueError('Invalid cached positions/velocities/orthorhombic-cell shapes')
    if h.atomic_numbers.shape != (n,) or h.atom_ids.shape != (n,) or h.edges.shape[0] != 2:
        raise ValueError('Cached atom identities/edges have invalid shapes')
    if len(h.offsets_ps) != t or h.offsets_ps[-1] != 0 or (h.offsets_ps[1:] <= h.offsets_ps[:-1]).any():
        raise ValueError('Cached offsets must increase causally and end at zero')
    if h.offsets_ps[0] < -model.history_duration_ps-1e-8 or (h.offsets_ps > 0).any():
        raise ValueError('Cached history exceeds the observed physical-time support')
    if h.cutoff_A != model.cutoff_A or h.spatial_layers < model.num_layers:
        raise ValueError('Cached graph does not cover the encoder cutoff/depth')
    if h.context_radius_A < max(s[1] for s in model.pool.scales):
        raise ValueError('Cached graph does not cover all pooling scales')
    if not all(torch.isfinite(v).all() for v in (h.positions, h.velocities, h.boxes, h.offsets_ps)) or (h.boxes <= 0).any():
        raise ValueError('Cached history contains nonfinite arrays or invalid cells')
    if h.atom_ids.unique().numel() != n or not torch.isin(h.atomic_numbers, model.atomic_numbers.cpu()).all():
        raise ValueError('Cached history has duplicate IDs or unsupported species')
    if h.edges.numel():
        if h.edges.min() < 0 or h.edges.max() >= t*n or not torch.equal(h.edges[0]//n, h.edges[1]//n):
            raise ValueError('Cached spatial edges cross time frames or exceed the node range')
        sender, receiver = h.edges
        dr = h.positions.flatten(0, 1)[receiver]-h.positions.flatten(0, 1)[sender]
        cell = h.boxes[sender//n]
        dr = dr-cell*torch.round(dr/cell)
        if (dr.square().sum(-1) <= 0).any():
            raise ValueError('Coincident atoms on a cached spatial edge')
