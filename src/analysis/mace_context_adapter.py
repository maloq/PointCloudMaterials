"""Static full-frame inference for the joint smooth-inner/center MACE encoder.

Compute each required atom feature once, with both complete 5 A message hops,
then reuse it in overlapping readouts. Static snapshots have no box metadata:
only verified interior centers are accepted; periodic lengths are not inferred.
"""

import json
from pathlib import Path
import time

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
import torch
from torch import nn

from src.data.static import PointCloudDataset
from src.experiment_runner.artifacts import write_json
from src.experiment_runner.registry import sha256
from src.models.encoders.mace_context import (
    ContextGraph, _incoming, context_features, inner_weights, make_context_graph,
)
from src.models.encoders.pretrained_mace import PretrainedMACEEncoder


class MACEContextAnalysis(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.protocol != 'mace_context_recovery_static_v1':
            raise ValueError(f'Unsupported static context protocol: {cfg.protocol}')
        self.mace = PretrainedMACEEncoder(
            cfg.pretrained_checkpoint,
            performance=OmegaConf.to_container(cfg.performance, resolve=True),
        )

    def forward(self, points):
        raise ValueError('Context embeddings require full-frame inference, not cropped static points')

    def encode_clouds(self, clouds, *, inner, outer):
        graph = make_context_graph(clouds, 'halo_inner', device=self.mace.element_indices.device,
                                   inner=inner, outer=outer)
        return context_features(self.mace, graph, return_center=True)


def node_graph(points, tree, targets, device):
    """Exact ancestors of arbitrary output atoms, in the requested target order."""
    second = _incoming(tree, points, targets)
    first_nodes = np.union1d(targets, second[0])
    first = _incoming(tree, points, first_nodes)
    input_nodes = np.union1d(first_nodes, first[0])
    # Translate before FP32 conversion to keep geometric subtraction accurate.
    positions = (points[input_nodes] - points[input_nodes].mean(0)).astype(np.float32)
    values = dict(positions=positions,
                  first_edges=np.searchsorted(input_nodes, first),
                  first_keep=np.searchsorted(input_nodes, first_nodes),
                  second_edges=np.searchsorted(first_nodes, second),
                  second_keep=np.searchsorted(first_nodes, targets),
                  pool_index=np.arange(len(targets)), weights=np.ones(len(targets)))
    values = {key: torch.as_tensor(value, device=device,
              dtype=torch.float32 if key in ('positions', 'weights') else torch.long)
              for key, value in values.items()}
    return ContextGraph(**values, batch_size=len(targets))


def _pool(points, centers, tree, features, inner, outer):
    groups = tree.query_ball_point(centers, outer, workers=1, return_sorted=True)
    result = np.empty((len(centers), 512), dtype=np.float32)
    distances, center_ids = tree.query(centers)
    np.testing.assert_allclose(distances, 0, atol=1e-7, rtol=0,
                               err_msg='Static centers must be exact source atoms')
    for row, (center, ids, atom) in enumerate(zip(centers, groups, center_ids, strict=True)):
        distance = np.linalg.norm(points[ids]-center, axis=1)
        weights = inner_weights(distance, inner, outer)
        result[row, :256] = np.sum(features[ids].astype(np.float64)*weights[:, None], axis=0)/weights.sum()
        result[row, 256:] = features[atom]
    if not np.isfinite(result).all():
        raise FloatingPointError('Invalid pooled static context features')
    return result


@torch.inference_mode()
def encode_frame(model, points, centers, settings, *, progress=None):
    scale = float(settings['coordinate_scale'])
    x = points.astype(np.float64)*scale
    c = centers.astype(np.float64)*scale
    inner, outer = float(settings['inner_radius_A']), float(settings['outer_radius_A'])
    required = outer + 10.
    margin = float(np.minimum(c-x.min(0), x.max(0)-c).min())
    if margin <= required:
        raise ValueError(f'Static center margin {margin:.6f} cannot support {required} model A')
    tree = cKDTree(x)
    needed = np.zeros(len(x), dtype=bool)
    for start in range(0, len(c), 2048):
        groups = tree.query_ball_point(c[start:start+2048], outer, workers=1)
        needed[np.concatenate(groups).astype(np.int64)] = True
    targets = np.flatnonzero(needed)
    tiles = np.floor(x[targets]/12.).astype(np.int64)
    order = np.lexsort((targets, tiles[:, 2], tiles[:, 1], tiles[:, 0]))
    targets = targets[order]
    features = np.full((len(x), 256), np.nan, dtype=np.float32)
    batch = int(settings['node_batch_size'])
    started = time.monotonic()
    device = model.mace.element_indices.device
    for start in range(0, len(targets), batch):
        ids = targets[start:start+batch]
        graph = node_graph(x, tree, ids, device)
        z = context_features(model.mace, graph)
        if not torch.isfinite(z).all():
            raise FloatingPointError(f'Nonfinite static node features at {start}/{len(targets)}')
        features[ids] = z.cpu().numpy()
        if progress is not None and (start == 0 or (start//batch+1) % 50 == 0):
            progress(nodes_done=min(start+batch, len(targets)), nodes_total=len(targets),
                     elapsed_seconds=time.monotonic()-started)
    embeddings = np.empty((len(c), 512), dtype=np.float32)
    for start in range(0, len(c), 1024):
        embeddings[start:start+1024] = _pool(x, c[start:start+1024], tree, features, inner, outer)
    # Independently replay complete individual halos on every analyzed frame.
    selected = np.unique(np.linspace(0, len(c)-1, 6, dtype=int))
    direct = []
    for index in selected:
        center = c[index]
        ids = np.asarray(tree.query_ball_point(center, required+1.), dtype=np.int64)
        atom = int(tree.query(center)[1])
        ids = np.r_[atom, ids[ids != atom]]
        cloud = (x[ids]-center).astype(np.float32)
        direct.append(model.encode_clouds([cloud], inner=inner, outer=outer).cpu().numpy()[0])
    direct = np.asarray(direct)
    error = float(np.square(embeddings[selected].astype(np.float64)-direct).sum()/np.square(direct.astype(np.float64)).sum())
    if error > 1e-8:
        raise AssertionError(f'Reused node features disagree with direct halo inference: relative squared error={error}')
    return embeddings, dict(centers=len(c), required_nodes=len(targets), coordinate_scale=scale,
        boundary_margin_model_A=margin, required_radius_model_A=required,
        direct_halo_relative_squared_error=error, elapsed_seconds=time.monotonic()-started)


def collect_context_inference(model, dataloader, cfg, out_dir, *, max_batches, max_samples):
    dataset = dataloader.dataset
    if not isinstance(dataset, PointCloudDataset) or dataset._cache_coord_arrays is None:
        raise TypeError('Static context collection requires the full cached PointCloudDataset with source coordinates')
    settings = OmegaConf.to_container(cfg.data.context_encoder, resolve=True)
    if settings['element_channel'] != 'Al':
        raise ValueError('This checkpoint protocol uses the trained, fixed Al geometry channel')
    metadata = json.loads((Path(cfg.data.sample_cache.cache_dir)/'metadata.json').read_text())
    sources = {s['name']: s for s in metadata['request']['sources']}
    total = len(dataset)
    if max_batches is not None:
        total = min(total, max_batches*int(dataloader.batch_size))
    if max_samples is not None:
        total = min(total, max_samples)
    result = np.empty((total, 512), dtype=np.float32)
    coordinates = np.empty((total, 3), dtype=np.float32)
    records, offset = [], 0
    for shard, coords in zip(metadata['shards'], dataset._cache_coord_arrays, strict=True):
        if offset == total:
            break
        coords = coords[:min(len(coords), total-offset)]
        path = Path(sources[shard['source']]['root'])/shard['file']
        def progress(**fields):
            write_json(Path(out_dir)/'context-inference-status.json', dict(state='running',
                frame=shard['file'], completed_centers=offset, total_centers=total, **fields))
            print(f'[context] {shard["file"]}: {fields}', flush=True)
        progress(nodes_done=0, nodes_total=None)
        z, record = encode_frame(model, np.load(path), coords, settings, progress=progress)
        result[offset:offset+len(coords)] = z
        coordinates[offset:offset+len(coords)] = coords
        records.append(dict(file=str(path), source_sha256=sha256(path), **record))
        offset += len(coords)
        write_json(Path(out_dir)/'context-inference-protocol.json', dict(
            protocol=cfg.protocol, settings=settings, frames=records, completed_centers=offset,
            representation='raw 256 smooth-inner channels followed by raw 256 center channels; no projector or prediction heads',
            boundaries='Nonperiodic full source snapshots; center margin exceeds the complete two-hop support.',
            storage='Source float32 positions, float64 relative geometry, float32 MACE and embedding storage; no float16 round trip.'))
    write_json(Path(out_dir)/'context-inference-status.json', dict(state='complete', total_centers=total))
    return dict(inv_latents=result, coords=coordinates, eq_latents=np.empty(0),
                phases=np.empty(0), instance_ids=np.empty(0), anchor_frame_indices=np.empty(0))
