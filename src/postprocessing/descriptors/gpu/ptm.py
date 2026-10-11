"""Opt-in float64 CUDA PTM (FCC/HCP/BCC), native orthorhombic full cells.

``voronoi`` orders 18 nearest neighbors by clipped-face solid angle. ``distance``
is an explicitly approximate ordering; it must not be published as OVITO PTM.
Both retain the pinned upstream convex-hull/graph/RMSD matcher.
"""
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.cpp_extension import load


def implementation_files():
    root = Path(__file__).parent
    return [Path(__file__), root/'ptm_kernels.cu', *sorted((root/'ptm').iterdir())]


@lru_cache(maxsize=1)
def extension():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for p in implementation_files():
        digest.update(p.name.encode()); digest.update(p.read_bytes())
    return load('pcm_ptm_'+digest.hexdigest()[:12], [str(root/'ptm_kernels.cu')],
                extra_cuda_cflags=['-O3', '--fmad=false'], verbose=False)


@lru_cache(maxsize=None)
def initialize(device_index):
    error = torch.zeros(1, dtype=torch.int32, device=f'cuda:{device_index}')
    extension().initialize(error)
    if int(error):
        raise RuntimeError(f'CUDA PTM reference graph initialization failed: {int(error)}')


@torch.no_grad()
def labels(points, box, cutoff, *, neighbor_radius, ordering='voronoi', device='cuda', diagnostics=False,
           device_output=False, joint_cutoff=None):
    if ordering not in ('voronoi', 'distance'):
        raise ValueError(f'Unknown CUDA PTM ordering: {ordering}')
    points = np.asarray(points, np.float64); box = np.asarray(box, np.float64)
    if box.shape != (3,) or not np.isfinite(box).all() or np.any(box <= 0):
        raise ValueError('CUDA PTM requires three positive orthorhombic lengths')
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 19 or not np.isfinite(points).all():
        raise ValueError('CUDA PTM requires finite (N,3) full-cell points with at least 19 atoms')
    if not np.isfinite(neighbor_radius) or neighbor_radius <= 0 or not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError('Positive finite neighbor radius and RMSD cutoff required')
    dims = np.floor(box/neighbor_radius).astype(int).tolist()
    if min(dims) < 3:
        raise ValueError('CUDA PTM requires each box length >= three neighbor radii')
    x = torch.as_tensor(np.mod(points, box), device=device)
    initialize(x.device.index)
    native = extension(); b = torch.as_tensor(box, device=device); n = len(points)
    integer = dict(dtype=torch.int32, device=device)
    double = dict(dtype=torch.float64, device=device)
    error = torch.zeros(1, **integer)
    counts = torch.zeros(math.prod(dims), **integer)
    bins = torch.empty((math.prod(dims), 128), **integer)
    cells = torch.empty((n, 3), **integer)
    vectors = torch.empty((n, 18, 3), **double)
    indices = torch.empty((n, 18), **integer)
    qx = qb = neighbors = degree = None
    if joint_cutoff is not None:
        if not 0 < joint_cutoff < neighbor_radius:
            raise ValueError('Joint bond-order cutoff must be positive and smaller than PTM search radius')
        # Preserve the freud float32 preprocessing separately from PTM's float64
        # positions. The shared bins cover both query radii without changing either.
        qx = torch.as_tensor(np.asarray(points-box/2, np.float32), device=device)
        qb = torch.as_tensor(np.asarray(box, np.float32), device=device)
        neighbors = torch.empty((n, 64), **integer)
        degree = torch.empty(n, **integer)
    native.nearest(x, b, *dims, neighbor_radius, counts, bins, cells, vectors, indices, error,
                   qx, qb, joint_cutoff or 0., neighbors, degree)
    code = int(error)
    if code:
        raise RuntimeError(f'CUDA PTM neighbor search failed ({code}): 1=bin capacity 128 exceeded; 2=fewer than 18 neighbors inside radius {neighbor_radius}; 6=bond neighbors exceed 64; 7=no bond neighbors; no truncation/fallback')
    areas = None
    if ordering == 'voronoi':
        areas = torch.empty((n, 18), **double)
        native.angles(vectors, areas, error)
        code = int(error)
        if code:
            raise RuntimeError(f'CUDA PTM Voronoi clipping failed ({code}): 3=coincident atoms; 4=polygon capacity exceeded; 5=unbounded face')
    types = torch.empty(n, dtype=torch.uint8, device=device)
    rmsd = torch.empty(n, **double)
    native.match(vectors, areas, cutoff, types, rmsd, error)
    result = types if device_output else types.cpu().numpy()
    if joint_cutoff is not None:
        if diagnostics:
            raise ValueError('Use standalone PTM diagnostics or the joint pipeline audit')
        return result, (qx, qb, neighbors, degree)
    if diagnostics:
        return result, dict(rmsd=rmsd.cpu().numpy(), neighbors=indices.cpu().numpy(),
                            solid_angles=None if areas is None else areas.cpu().numpy())
    return result
