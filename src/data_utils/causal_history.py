"""Identity-preserving, periodic, observed-only graphs for the causal MACE state."""
from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial import cKDTree
import torch


@dataclass
class AtomicHistory:
    # Atom axis has one persistent ordering across frames; positions are relative
    # to the tracked center and unwrapped backwards from its current image.
    positions: torch.Tensor             # [time, atom, 3], Angstrom
    velocities: torch.Tensor            # [time, atom, 3], center-relative A/ps
    boxes: torch.Tensor                 # [time, 3], orthorhombic lengths
    offsets_ps: torch.Tensor             # [time], strictly increasing, ends at zero
    atomic_numbers: torch.Tensor         # [atom]
    atom_ids: torch.Tensor               # [atom], actual producer IDs
    edges: torch.Tensor                  # [2, edge], flattened time-major nodes
    cutoff_A: float
    context_radius_A: float
    spatial_layers: int

    def to(self, device):
        return replace(self, **{k: v.to(device) for k, v in vars(self).items()
                                if isinstance(v, torch.Tensor)})

    def current(self):
        n = self.positions.shape[1]
        start = (len(self.offsets_ps)-1)*n
        keep = self.edges[0] >= start
        return replace(self, positions=self.positions[-1:], velocities=self.velocities[-1:],
                       boxes=self.boxes[-1:], offsets_ps=self.offsets_ps[-1:],
                       edges=self.edges[:, keep]-start)

    def repeated_anchor(self):
        """Capacity-matched training control; physical offsets remain unchanged."""
        now = self.current()
        t, n = self.positions.shape[:2]
        return replace(self, positions=now.positions.expand(t, -1, -1),
                       velocities=now.velocities.expand(t, -1, -1),
                       boxes=now.boxes.expand(t, -1),
                       edges=torch.cat([now.edges+k*n for k in range(t)], dim=1))


def physical_windows(times, history_offsets_ps, future_lags_ps):
    """Exact physical-time matching; never round lags to a convenient frame."""
    times = np.asarray(times, dtype=np.float64)
    past = np.asarray(history_offsets_ps, dtype=np.float64)
    future = np.asarray(future_lags_ps, dtype=np.float64)
    if (times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or past.ndim != 1 or len(past) == 0 or past[-1] != 0
            or not np.isfinite(past).all() or np.any(np.diff(past) <= 0)
            or np.any(past > 0) or future.ndim != 1 or len(future) == 0
            or not np.isfinite(future).all() or np.any(future <= 0)
            or np.any(np.diff(future) <= 0)):
        raise ValueError('Require increasing times, causal offsets ending at 0, and positive increasing future lags')
    windows = []
    for anchor, time in enumerate(times):
        requested = time+np.r_[past, future]
        difference = np.abs(requested[:, None]-times[None])
        ids = difference.argmin(1)
        if np.all(difference[np.arange(len(ids)), ids] < 1e-8):
            windows.append((anchor, ids[:len(past)], ids[len(past):]))
    if not windows:
        raise ValueError(f'No exact causal windows: source span={times[-1]-times[0]:g} ps, '
                         f'history={past.tolist()}, future={future.tolist()} ps')
    return windows


def build_history(positions, velocities, boxes, times, atom_ids, atomic_numbers,
                  center_atom_id, *, cutoff_A, context_radius_A, spatial_layers, prune=True):
    """Build from *observed frames only*, using exact space-time ancestor closure.

    Temporal edges are same-ID, past-to-present; spatial edges use minimum images.
    Backwards closure includes all message paths to current pooling support, even
    when an atom's historical neighbors lie outside a static candidate halo.
    """
    x = np.asarray(positions, dtype=np.float64)
    v = np.asarray(velocities, dtype=np.float64)
    box = np.asarray(boxes, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    ids = np.asarray(atom_ids)
    species = np.asarray(atomic_numbers)
    if (x.ndim != 3 or x.shape[-1] != 3 or v.shape != x.shape
            or box.shape != (len(x), 3) or times.shape != (len(x),)
            or ids.shape != (x.shape[1],) or species.shape != ids.shape):
        raise ValueError(f'Invalid history arrays: positions={x.shape}, velocities={v.shape}, boxes={box.shape}')
    if not all(np.isfinite(a).all() for a in (x, v, box, times)):
        raise ValueError('Nonfinite atomic history')
    if (len(np.unique(ids)) != len(ids) or ids.dtype.kind not in 'iu'
            or species.dtype.kind not in 'iu' or np.any(species <= 0)):
        raise ValueError('History requires unique integer atom IDs and positive atomic numbers')
    centers = np.flatnonzero(ids == center_atom_id)
    if len(centers) != 1 or np.any(np.diff(times) <= 0):
        raise ValueError('History requires a unique tracked center and increasing physical times')
    if min(cutoff_A, context_radius_A) <= 0 or spatial_layers < 2:
        raise ValueError('Positive radii and at least two spatial layers are required')
    if np.min(box) <= 2*max(cutoff_A, context_radius_A):
        raise ValueError('Spatial cutoff/pooling support overlaps periodic images')
    center = int(centers[0])
    wrapped = np.mod(x, box[:, None])
    trees = [cKDTree(frame, boxsize=lengths) for frame, lengths in zip(wrapped, box, strict=True)]
    required = [set() for _ in x]
    required[-1] = set(trees[-1].query_ball_point(wrapped[-1, center], context_radius_A))
    for _ in range(spatial_layers):
        # Spatial -> temporal is the forward order in every block.
        for k in range(len(x)-2, -1, -1):
            required[k].update(required[k+1])
        expanded = []
        for k, nodes in enumerate(required):
            neighbors = trees[k].query_ball_point(wrapped[k, sorted(nodes)], cutoff_A)
            expanded.append(nodes.union(*(set(a) for a in neighbors)))
        required = expanded
    selected = np.array(sorted(set.union(*required)) if prune else np.arange(x.shape[1]), dtype=int)
    # Keep the center first, but preserve true IDs rather than inferring ID=row+1.
    selected = np.r_[center, selected[selected != center]]
    row = {int(old): new for new, old in enumerate(selected)}
    edges = []
    for k, tree in enumerate(trees):
        neighbors = tree.query_ball_point(wrapped[k, selected], cutoff_A)
        for receiver, senders in enumerate(neighbors):
            edges.extend((row[j]+k*len(selected), receiver+k*len(selected))
                         for j in senders if j in row and j != selected[receiver])
    relative = x[:, selected]-x[:, center:center+1]
    relative -= box[:, None]*np.round(relative/box[:, None])
    # Unwrap relative fractional coordinates, allowing measured box evolution.
    fractional = relative/box[:, None]
    for k in range(len(x)-2, -1, -1):
        fractional[k] += np.round(fractional[k+1]-fractional[k])
    relative = fractional*box[:, None]
    velocity = v[:, selected]-v[:, center:center+1]
    return AtomicHistory(torch.tensor(relative, dtype=torch.float32),
                         torch.tensor(velocity, dtype=torch.float32),
                         torch.tensor(box, dtype=torch.float32),
                         torch.tensor(times-times[-1], dtype=torch.float64),
                         torch.tensor(species[selected], dtype=torch.long),
                         torch.tensor(ids[selected], dtype=torch.long),
                         torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous(),
                         float(cutoff_A), float(context_radius_A), spatial_layers)
