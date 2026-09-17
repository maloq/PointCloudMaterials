"""Encoder-only input: union of IDs from allowed observation frames only."""
from dataclasses import dataclass, replace
import numpy as np
from scipy.spatial import cKDTree
import torch
from .targets import taper


@dataclass
class Observation:
    positions: torch.Tensor
    velocities: torch.Tensor
    weights: torch.Tensor
    offsets_ps: torch.Tensor
    atom_ids: torch.Tensor
    edges: tuple[torch.Tensor, ...]
    radius_A: float
    cutoff_A: float

    def to(self, device):
        return replace(self, **{name: (value.to(device) if isinstance(value, torch.Tensor)
                        else tuple(v.to(device) for v in value))
                        for name, value in vars(self).items() if name not in ('radius_A', 'cutoff_A')})

    def repeated_anchor(self):
        t = len(self.offsets_ps)
        return replace(self, positions=self.positions[-1:].expand(t, -1, -1),
                       velocities=self.velocities[-1:].expand(t, -1, -1),
                       weights=self.weights[-1:].expand(t, -1), edges=(self.edges[-1],)*t)


def frame_observation(x, u, box, atom_ids, center_id, radius, cutoff):
    """No full-cell neighbor features: crop first, then build the spatial graph."""
    center = np.flatnonzero(atom_ids == center_id)
    if len(center) != 1 or np.min(box) <= 4*radius:
        raise ValueError('This local-chart graph requires one center and box lengths greater than 4*radius')
    relative = np.asarray(x, dtype=np.float64)-x[center[0]].astype(np.float64)
    relative -= box*np.round(relative/box)
    distance = np.linalg.norm(relative, axis=-1)
    selected = np.flatnonzero(distance < radius)
    local = relative[selected]
    pairs = cKDTree(local).query_pairs(cutoff, output_type='ndarray')
    edges = np.concatenate((pairs, pairs[:, ::-1]), axis=0).T.astype(np.int32)
    return dict(ids=torch.from_numpy(atom_ids[selected].astype(np.int64)),
        positions=torch.from_numpy(local.astype(np.float32)),
        velocities=torch.from_numpy((u[selected].astype(np.float64)-u[center[0]].astype(np.float64)).astype(np.float32)),
        weights=torch.from_numpy(taper(distance[selected], radius-2, radius).astype(np.float32)),
        edges=torch.from_numpy(edges))


def assemble(frames, times, *, radius, cutoff):
    times = np.asarray(times, dtype=np.float64)
    if len(times) != len(frames) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Observation frames require strictly increasing finite physical times')
    ids = torch.unique(torch.cat([f['ids'] for f in frames]), sorted=True)
    x = torch.zeros(len(frames), len(ids), 3)
    u = torch.zeros_like(x)
    weights = torch.zeros(len(frames), len(ids))
    edges = []
    for k, frame in enumerate(frames):
        idx = torch.searchsorted(ids, frame['ids'])
        x[k, idx], u[k, idx], weights[k, idx] = frame['positions'], frame['velocities'], frame['weights']
        edges.append(idx[frame['edges'].long()])
    return Observation(x, u, weights, torch.from_numpy(times-times[-1]), ids, tuple(edges), radius, cutoff)
