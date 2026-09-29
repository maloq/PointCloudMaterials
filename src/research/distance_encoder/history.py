"""Causal MD histories of trainable local MACE embeddings, with a matched control."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.data.fixed_cohort.protocol import digest
from src.project_runtime.paths import resolve_path
from src.research.supervised_onset.model import CapacityEncoder
from src.research.spatial_distance.model import parameters
from .data import ResidentDataset


def history_index(config, role):
    """Join identical center-ID slices at exact physical offsets, never by row adjacency."""
    plan = json.loads((resolve_path(config['labels']['root'])/'plan.json').read_text())
    offsets = config['history']['offsets_ps']
    if offsets != [-6.0, -3.0, 0.0]:
        raise ValueError('This release implements exactly three observations at -6/-3/0 ps')
    shards = [s for s in plan['shards'] if s['task']['split'] == role]
    lookup, start = {}, 0
    for shard in shards:
        t = shard['task']
        key = (t['source'], t['frame'], t['start'], t['stop'])
        if key in lookup:
            raise ValueError(f'Duplicate structural center slice: {key}')
        lookup[key] = start
        start += shard['rows']
    indices, sources, materials = [], [], []
    omitted = {}
    for shard in shards:
        t = shard['task']; source = plan['sources'][t['source']]
        frames = sorted(map(int, source['audit_frames']))
        if source['kind'] == 'native':
            times = np.asarray(frames)*.75
        else:
            times = np.asarray([source['audit_source']['times_ps'][source['audit_frames'][str(f)]] for f in frames])
        np.testing.assert_allclose(np.diff(times), 3., rtol=0, atol=1e-6)
        position = frames.index(t['frame'])
        if position < 2:
            omitted[t['source']] = omitted.get(t['source'], 0)+shard['rows']
            continue
        keys = [(t['source'], f, t['start'], t['stop']) for f in frames[position-2:position+1]]
        missing = [k for k in keys if k not in lookup]
        if missing:
            raise ValueError(f'Missing tracked-center history shard: {missing}')
        indices.append(np.arange(shard['rows'])[:, None]+np.asarray([lookup[k] for k in keys])[None])
        sources.extend([t['source']]*shard['rows'])
        materials.extend([shard['material']]*shard['rows'])
    index = np.concatenate(indices)
    sources, materials = np.asarray(sources), np.asarray(materials)
    names, counts = np.unique(materials, return_counts=True)
    weights = np.empty(len(index), np.float32)
    groups = materials if role == 'train' else sources
    group_names, group_counts = np.unique(groups, return_counts=True)
    for name, count in zip(group_names, group_counts):
        weights[groups == name] = len(index)/(len(group_names)*count)
    metadata = dict(rows=len(index), original_rows=start, omitted_initial_rows=start-len(index),
        omitted_by_source=omitted, material_counts=dict(zip(names.tolist(), counts.tolist())),
        source_count=len(np.unique(sources)), offsets_ps=offsets,
        rule='same source, center ID and 3-ps cadence; omit first two retained label frames uniformly')
    return index, weights, metadata


class HistoryDataset(ResidentDataset):
    def __init__(self, config, role, device):
        super().__init__(config, role, device)
        indices, weight, metadata = history_index(config, role)
        if metadata['original_rows'] != self.n:
            raise ValueError('History and coordinate row orders differ')
        self.index = torch.as_tensor(indices, dtype=torch.long, device=device)
        self.weight = torch.as_tensor(weight, device=device)
        self.distance = self.distance[self.index[:, -1]]
        self.sources = self.sources[indices[:, -1]]
        self.material = self.material[indices[:, -1]]
        self.n = len(indices); self.counts = metadata['material_counts']
        self.history_metadata = metadata
        self.identity = digest(dict(parent=self.identity, history=metadata))
        self.mode = config['history']['mode']
        self.target_counts = {}
        for i, name in enumerate(self.material_names):
            target = self.distance[torch.as_tensor(self.material == i, device=device)]
            self.target_counts[name] = dict(rows=len(target), zero=int((target == 0).sum()),
                within20=int((target <= 20).sum()), censored64=int((target >= 64).sum()))

    def batch(self, ids):
        index = self.index[ids]
        if self.mode == 'repeated_current':
            index = index[:, -1:]
        return self.positions[index], self.distance[ids], self.weight[ids]


class HistoryHead(nn.Module):
    """Ordered embeddings plus increments; a learned residual on the current state."""
    def __init__(self, frames):
        super().__init__()
        self.fusion = nn.Sequential(nn.Linear((2*frames-1)*128, 128), nn.SiLU(), nn.Linear(128, 128))
        nn.init.normal_(self.fusion[-1].weight, std=.001)
        nn.init.zeros_(self.fusion[-1].bias)
        self.distance = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 3))

    def state(self, z):
        changes = z[:, 1:]-z[:, :-1]
        return z[:, -1]+self.fusion(torch.cat((z.flatten(1), changes.flatten(1)), -1))

    def forward(self, z):
        state = self.state(z)
        return self.distance(state), state


class HistoryDistanceEncoder(nn.Module):
    def __init__(self, encoder_config, history):
        super().__init__()
        if history['mode'] not in ('real', 'repeated_current'):
            raise ValueError('History mode must be real or repeated_current')
        self.encoder = CapacityEncoder(**encoder_config)
        self.frames = len(history['offsets_ps'])
        self.head = HistoryHead(self.frames)
        self.input_frames = self.frames if history['mode'] == 'real' else 1

    def forward(self, graph, return_embedding=False):
        z = self.encoder(graph).reshape(-1, self.input_frames, 128)
        # Reusing the same differentiable tensor is equivalent to repeated
        # identical deterministic encoder evaluations, without redundant work.
        if self.input_frames == 1:
            z = z.expand(-1, self.frames, -1)
        raw, state = self.head(z)
        raw = raw.float()[:, None]
        parts = parameters(raw, raw[..., 0]*0)
        return (parts, state.float()) if return_embedding else parts
