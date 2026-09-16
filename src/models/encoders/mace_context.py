"""Exact two-hop MACE context with an explicit central or inner-region readout.

Coordinates are physical Angstrom offsets; index zero is the tracked center.
The caller supplies a complete candidate halo, including any augmentation margin.
Only ancestors of readout atoms are evaluated. This is graph pruning, not a
smaller cutoff: both native 5 A message-passing neighborhoods remain complete.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
import torch

from .pretrained_mace import dense_matmul_precision


def inner_weights(distance, inner=5., outer=7.):
    """Compact C2 quintic taper, including zero slope at both shell endpoints."""
    t = np.clip((np.asarray(distance, dtype=np.float64) - inner) / (outer - inner), 0, 1)
    # Factoring avoids cancellation and negative FP32 weights near the cutoff.
    return (1-t)**3 * (1+3*t+6*t**2)


@dataclass
class ContextGraph:
    positions: torch.Tensor
    first_edges: torch.Tensor
    first_keep: torch.Tensor
    second_edges: torch.Tensor
    second_keep: torch.Tensor
    pool_index: torch.Tensor
    weights: torch.Tensor
    batch_size: int
    # Indices into concatenated, unpruned input clouds, for aligned extra fields.
    input_index: torch.Tensor | None = None
    input_batch: torch.Tensor | None = None


def _incoming(tree, x, receivers):
    neighbors = tree.query_ball_point(x[receivers], 5., workers=1, return_sorted=True)
    receiver = np.repeat(receivers, [len(ids) for ids in neighbors])
    sender = np.concatenate(neighbors).astype(np.int64)
    valid = (sender != receiver) & (np.linalg.norm(x[sender]-x[receiver], axis=1) < 5.)
    return np.stack([sender[valid], receiver[valid]])


def make_context_graph(clouds, mode, *, device, inner=5., outer=7.):
    if mode not in ('mean80', 'halo_mean80', 'halo_inner', 'halo_center'):
        raise ValueError(f'Unknown context readout: {mode}')
    if not 0 < inner < outer:
        raise ValueError(f'Invalid pooling radii: {inner}, {outer}')
    arrays = {key: [] for key in ('positions', 'first_edges', 'first_keep',
              'second_edges', 'second_keep', 'pool_index', 'weights', 'input_index', 'input_batch')}
    first_start = second_start = cloud_start = 0
    for batch, cloud in enumerate(clouds):
        x = np.asarray(cloud, dtype=np.float32)
        if mode == 'mean80':
            x = x[:80]
        distance = np.linalg.norm(x-x[0], axis=1)
        if mode in ('mean80', 'halo_mean80'):
            targets = np.arange(80)
        elif mode == 'halo_center':
            targets = np.array([0])
        else:
            targets = np.flatnonzero(distance < outer)
        tree = cKDTree(x)
        second = _incoming(tree, x, targets)
        first_nodes = np.union1d(targets, second[0])
        first = _incoming(tree, x, first_nodes)
        input_nodes = np.union1d(first_nodes, first[0])
        arrays['positions'].append(x[input_nodes])
        arrays['input_index'].append(input_nodes + cloud_start)
        arrays['input_batch'].append(np.full(len(input_nodes), batch, dtype=np.int64))
        arrays['first_edges'].append(np.searchsorted(input_nodes, first)+first_start)
        arrays['first_keep'].append(np.searchsorted(input_nodes, first_nodes)+first_start)
        arrays['second_edges'].append(np.searchsorted(first_nodes, second)+second_start)
        arrays['second_keep'].append(np.searchsorted(first_nodes, targets)+second_start)
        arrays['pool_index'].append(np.full(len(targets), batch, dtype=np.int64))
        arrays['weights'].append(inner_weights(distance[targets], inner, outer)
                                 if mode == 'halo_inner' else np.ones(len(targets)))
        first_start += len(input_nodes)
        second_start += len(first_nodes)
        cloud_start += len(cloud)
    values = {key: torch.as_tensor(np.concatenate(value, axis=1 if key.endswith('edges') else 0),
              dtype=torch.float32 if key in ('positions', 'weights') else torch.long,
              device=device) for key, value in arrays.items()}
    return ContextGraph(**values, batch_size=len(clouds))


def context_features(mace, graph, *, return_center=False, return_nodes=False):
    """256 raw invariant channels, with the original checkpoint parameter names."""
    model = mace.backbone
    if float(model.r_max) != 5. or len(model.interactions) != 2:
        raise ValueError('Context pruning is defined for the repository two-layer 5 A MACE')
    with dense_matmul_precision('highest'):
        attrs = torch.nn.functional.one_hot(mace.element_indices[0].expand(len(graph.positions)),
                    len(model.atomic_numbers)).to(graph.positions.dtype)
        positions = graph.positions
        geometries = []
        for edges, keep in [(graph.first_edges, graph.first_keep),
                            (graph.second_edges, graph.second_keep)]:
            vectors = positions[edges[1]]-positions[edges[0]]
            angular = model.spherical_harmonics(vectors)
            radial, cutoff = model.radial_embedding(vectors.norm(dim=-1, keepdim=True),
                                                    attrs, edges, model.atomic_numbers)
            geometries.append((attrs, edges, angular, radial, cutoff, keep))
            positions, attrs = positions[keep], attrs[keep]
    with dense_matmul_precision(mace.dense_precision):
        h = model.node_embedding(geometries[0][0])
        blocks = []
        for layer, (attrs, edges, angular, radial, cutoff, keep) in enumerate(geometries):
            h, sc = model.interactions[layer](node_attrs=attrs, node_feats=h,
                    edge_attrs=angular, edge_feats=radial, edge_index=edges,
                    cutoff=cutoff, first_layer=layer == 0)
            h = model.products[layer](node_feats=h[keep], sc=None if sc is None else sc[keep],
                                      node_attrs=attrs[keep])
            blocks.append(h[graph.second_keep] if layer == 0 else h)
        features = torch.cat(blocks, dim=1)
        if return_nodes:
            if return_center:
                raise ValueError('Select node features or a center readout, not both')
            return features
        result = features.new_zeros(graph.batch_size, 256)
        result.index_add_(0, graph.pool_index, features*graph.weights[:, None])
        total = features.new_zeros(graph.batch_size)
        total.index_add_(0, graph.pool_index, graph.weights)
        pooled = result/total[:, None]
        if return_center:
            # make_context_graph keeps each cloud's targets in input-index order.
            # The tracked center is input index zero and the first readout node.
            first = torch.cat([torch.ones(1, dtype=torch.bool, device=features.device),
                               graph.pool_index[1:] != graph.pool_index[:-1]])
            return torch.cat([pooled, features[first]], dim=1)
        return pooled
