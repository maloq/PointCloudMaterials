"""Immutable GPU geometry and bounded reusable batch-index plans."""
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F
from src.training_methods.bcr.data import taper


def gather_indices(offsets, lengths, indices):
    sizes = lengths[indices]
    starts = np.r_[0, np.cumsum(sizes)]
    local = np.arange(starts[-1]) - np.repeat(starts[:-1], sizes)
    return local + np.repeat(offsets[indices], sizes), starts, sizes


@dataclass(frozen=True)
class BatchPlan:
    owner: object
    nodes: torch.Tensor
    edges: torch.Tensor
    shifts: torch.Tensor
    group: torch.Tensor
    centers: torch.Tensor
    size: int

    @property
    def nbytes(self):
        return sum(t.numel()*t.element_size() for t in
                   (self.nodes, self.edges, self.shifts, self.group, self.centers))


class GraphBank:
    """One geometry identity per bank; learned features are never cached.

    Rebuilt/noisy geometry needs its own bank. Cache entries are only indices
    into immutable tensors owned by this bank. Ordinary sampled batches do not
    populate the LRU, so they cannot evict repeatedly used evaluation plans.
    node_capacity fixes the atom axis, including partial batches, using zero
    attributes/weights and no extra edges. Graph count and physical normalization
    stay unchanged. Batches exceeding this explicit capacity are rejected.
    """
    def __init__(self, arrays, encoder, device, *, plan_cache_bytes=256*2**20,
                 node_capacity=None):
        if list(encoder.radial_embedding.parameters()):
            raise ValueError('Caching requires a parameter-free radial embedding')
        self.device = torch.device(device)
        self.offsets = arrays['offsets']
        self.edge_offsets = arrays['edge_offsets']
        self.node_lengths = np.diff(self.offsets)
        self.edge_lengths = np.diff(self.edge_offsets)
        if node_capacity is not None and (not isinstance(node_capacity, int) or node_capacity < 1):
            raise ValueError('node_capacity must be a positive integer or None')
        self.node_capacity = node_capacity
        self.plan_cache_bytes = int(plan_cache_bytes)
        if self.plan_cache_bytes < 0:
            raise ValueError('Plan cache budget must be nonnegative')
        self._plans = OrderedDict()
        self._plan_identity = object()
        self.cached_plan_bytes = 0
        positions = torch.as_tensor(arrays['positions'], device=device)
        edge = torch.as_tensor(arrays['edges'], device=device, dtype=torch.long)
        self.edge = edge
        self.weight = taper(positions.norm(dim=-1), encoder.radius)
        self.center = torch.zeros((len(positions), 1), device=device)
        self.center[torch.as_tensor(self.offsets[:-1], device=device)] = 1
        self.attrs = torch.ones_like(self.center)
        offset = np.repeat(self.offsets[:-1], np.diff(self.edge_offsets))
        absolute = edge + torch.as_tensor(offset, device=device)[None]
        angular, radial = [], []
        with torch.no_grad():
            for start in range(0, edge.shape[1], 131072):
                e = absolute[:, start:start + 131072]
                v = positions[e[1]] - positions[e[0]]
                angular.append(encoder.spherical_harmonics(v))
                r, c = encoder.radial_embedding(v.norm(dim=-1, keepdim=True), self.attrs, e, encoder.atomic_numbers)
                radial.append(r * (self.weight[e[0]] * self.weight[e[1]])[:, None])
                if c is not None:
                    raise ValueError('Native MACE must apply its cutoff inside radial features')
        self.angular = torch.cat(angular)
        self.radial = torch.cat(radial)


    def plan(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or not len(indices) or indices.min() < 0 or indices.max() >= len(self.node_lengths):
            raise ValueError('Batch IDs must be a nonempty vector of valid graph indices')
        nodes, ptr, lengths = gather_indices(self.offsets, self.node_lengths, indices)
        edges, _, edge_lengths = gather_indices(self.edge_offsets, self.edge_lengths, indices)
        tensor = lambda a: torch.as_tensor(a, device=self.device, dtype=torch.long)
        return BatchPlan(self._plan_identity, tensor(nodes), tensor(edges), tensor(np.repeat(ptr[:-1], edge_lengths)),
            tensor(np.repeat(np.arange(len(indices)), lengths)), tensor(ptr[:-1]), len(indices))

    def prepare(self, indices, chunk):
        """Retain plans for a declared repeated pass, bounded by a byte budget."""
        if chunk < 1:
            raise ValueError('Positive chunk size required')
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or not len(indices) or indices.min() < 0 or indices.max() >= len(self.node_lengths):
            raise ValueError('Batch IDs must be a nonempty vector of valid graph indices')
        required = 8*(2*self.node_lengths[indices].sum()+2*self.edge_lengths[indices].sum()+len(indices))
        if required > self.plan_cache_bytes:
            return  # Avoid building and evicting the entire pass before use.
        for start in range(0, len(indices), chunk):
            rows = np.asarray(indices[start:start+chunk], dtype=np.int64)
            key = rows.tobytes()
            if key in self._plans:
                self._plans.move_to_end(key)
                continue
            plan = self.plan(rows)
            if plan.nbytes > self.plan_cache_bytes:
                continue
            while self.cached_plan_bytes + plan.nbytes > self.plan_cache_bytes:
                _, removed = self._plans.popitem(last=False)
                self.cached_plan_bytes -= removed.nbytes
            self._plans[key] = plan
            self.cached_plan_bytes += plan.nbytes

    def materialize(self, plan):
        if plan.owner is not self._plan_identity:
            raise ValueError('Batch plan belongs to a different geometry bank')
        ni, ei = plan.nodes, plan.edges
        padding = 0 if self.node_capacity is None else self.node_capacity - ni.numel()
        if padding < 0:
            raise ValueError(f'Batch has {ni.numel()} atoms, exceeding node_capacity={self.node_capacity}; '
                             'increase the declared capacity or reduce the microbatch')
        graph = dict(attrs=self.attrs[ni], center=self.center[ni], weight=self.weight[ni],
            edge=self.edge[:, ei] + plan.shifts[None], angular=self.angular[ei],
            radial=self.radial[ei], cutoff=None, group=plan.group,
            centers=plan.centers, size=plan.size)
        if padding:
            # Disconnected nodes contribute zero after every spatial block and
            # in pooling. Real edges, centers, graph count and n_ref are intact.
            # Capacity is fixed even for the last, incomplete batch of a pass.
            for name in ('attrs', 'center'):
                graph[name] = F.pad(graph[name], (0, 0, 0, padding))
            for name in ('weight', 'group'):
                graph[name] = F.pad(graph[name], (0, padding))
        return graph

    def batch(self, indices):
        rows = np.asarray(indices, dtype=np.int64)
        if rows.ndim != 1:
            raise ValueError('Batch IDs must be a vector')
        key = rows.tobytes()
        plan = self._plans.get(key)
        if plan is None:
            plan = self.plan(rows)
        else:
            self._plans.move_to_end(key)
        return self.materialize(plan)
