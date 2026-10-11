"""Experimental CUDA extraction; opt-in API, never selected by production jobs.

The default resident path uses CUDA PTM and fused descriptor reductions.
``freud`` preserves the reference neighbors/harmonics;
``cuda`` also evaluates those on GPU and requires a numerical protocol audit.
Supports the current orthorhombic, l=6, four-detector protocol only.
"""
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.cpp_extension import load
from ..ptm import full_labels

CAP = 64


@lru_cache(maxsize=1)
def extension():
    path = Path(__file__).with_name('kernels.cu')
    name = 'pcm_structure_' + hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return load(name, [str(path)], extra_cuda_cflags=['-O3', '--fmad=false'], verbose=False)


def reference_neighbors(points, box, cutoff, *, pack=True):
    import freud
    if freud.__version__ != '3.4.0':
        raise RuntimeError('Reference requires the pinned freud 3.4.0')
    freud.parallel.set_num_threads(int(os.environ.get('OVITO_THREAD_COUNT', '1')))
    x = np.asarray(points-box/2, np.float32)
    query = freud.locality.AABBQuery(freud.box.Box(*box), x)
    neighbors = query.query(x, dict(r_max=cutoff, exclude_ii=True)).toNeighborList()
    if not pack:
        return query, neighbors, None
    degree = np.asarray(neighbors.neighbor_counts)
    if degree.max() > CAP:
        raise ValueError(f'GPU prototype capacity {CAP} exceeded: {degree.max()}')
    packed = np.full((len(x), CAP), -1, np.int32)
    row = np.asarray(neighbors.query_point_indices)
    col = np.asarray(neighbors.point_indices)
    slot = np.arange(len(row))-np.repeat(np.asarray(neighbors.segments), degree)
    packed[row, slot] = col
    return query, neighbors, packed


@torch.no_grad()
def prepare(points, box, cutoff, backend='cuda', device='cuda'):
    native = extension()
    if backend == 'freud':
        import freud
        query, neighbors, packed = reference_neighbors(points, box, cutoff)
        order = freud.order.Steinhardt(l=6).compute(query, neighbors=neighbors)
        return (torch.as_tensor(packed, device=device),
                torch.as_tensor(np.asarray(order.particle_harmonics).view(np.float32).reshape(-1, 13, 2).copy(), device=device),
                torch.as_tensor(order.ql.copy(), device=device))
    if backend != 'cuda':
        raise ValueError(f'Unknown GPU descriptor backend: {backend}')
    box32 = np.asarray(box, np.float32)
    if box32.shape != (3,) or not np.isfinite(box32).all() or cutoff <= 0:
        raise ValueError('Three finite orthorhombic lengths and positive cutoff required')
    dims = np.floor(box32/cutoff).astype(int).tolist()
    if min(dims) < 3:
        raise ValueError('Each box length must be at least three cutoffs for unique cell images')
    x = torch.as_tensor(np.asarray(points-box/2, np.float32), device=device)
    if x.ndim != 2 or x.shape[1] != 3 or not bool(torch.isfinite(x).all()):
        raise ValueError('Finite (N,3) coordinates required')
    length = torch.as_tensor(box32, device=device)
    n = len(points)
    integer = dict(device=device, dtype=torch.int32)
    packed = torch.empty((n, CAP), **integer)
    degree = torch.empty(n, **integer)
    counts = torch.zeros(math.prod(dims), **integer)
    bins = torch.empty((math.prod(dims), 128), **integer)
    cells = torch.empty((n, 3), **integer)
    error = torch.zeros(1, **integer)
    native.neighbors(x, length, *dims, cutoff, packed, degree, counts, bins, cells, error)
    code = int(error)
    if code:
        raise RuntimeError(f'GPU neighbor overflow {code}: bin capacity128 / neighbors64; no truncation allowed')
    if bool((degree == 0).any()):
        raise ValueError('Undefined q6: at least one atom has no neighbors')
    h = torch.empty((n, 13, 2), device=device, dtype=torch.float32)
    q = torch.empty(n, device=device, dtype=torch.float32)
    native.harmonics(x, length, packed, degree, h, q)
    return packed, h, q


@torch.no_grad()
def finish(prepared, ptm, config, *, diagnostics=False):
    neighbors, h, q = prepared
    native = extension()
    corr = torch.empty(neighbors.shape, device=neighbors.device, dtype=torch.float32)
    native.correlations(neighbors, h, q, corr)
    if not bool(torch.isfinite(corr).all()):
        raise FloatingPointError('Nonfinite CUDA bond correlations')
    crystal = (ptm >= 1) & (ptm <= 3) if isinstance(ptm, torch.Tensor) else torch.as_tensor(np.isin(ptm, [1, 2, 3]), device=neighbors.device)
    minimum = config['minimum_tracked_atoms']
    result = [native.components(neighbors, crystal, None, 0., minimum)]
    counts = [crystal.sum()]
    masks = []
    for threshold, required in config['bond_order']['thresholds']:
        mask = torch.empty_like(crystal)
        native.solid(neighbors, corr, threshold, required, mask)
        result.append(native.components(neighbors, mask, corr, threshold, minimum))
        counts.append(mask.sum())
        masks.append(mask)
    # Double bin boundaries reproduce numpy's declared float64 linspace bins.
    selected = corr[neighbors >= 0].double()
    bins = torch.as_tensor(np.linspace(-1, 1, 41), device=neighbors.device)
    eligible = selected[(selected >= -1) & (selected <= 1)]
    index = torch.bucketize(eligible, bins, right=True).sub_(1).clamp_max_(39)
    histogram = torch.bincount(index, minlength=40)
    output = (torch.stack(result).cpu().numpy(), torch.stack(counts).cpu().numpy(), histogram.cpu().numpy())
    if diagnostics:
        return output, dict(neighbors=neighbors.cpu().numpy(), q6=q.cpu().numpy(),
                            correlations=corr.cpu().numpy(), solid=torch.stack(masks).cpu().numpy())
    return output


@torch.no_grad()
def finish_joint(prepared, ptm, config, *, diagnostics=False):
    """Resident labels and fused reductions; same float32 bond arithmetic."""
    neighbors, h, q = prepared
    native = extension(); device = neighbors.device; n = len(ptm)
    corr = torch.empty(neighbors.shape, device=device, dtype=torch.float32)
    histogram = torch.zeros(40, device=device, dtype=torch.int64)
    error = torch.zeros(1, device=device, dtype=torch.int32)
    edges = torch.as_tensor(np.linspace(-1, 1, 41), device=device)
    native.correlation_histogram(neighbors, h, q, edges, corr, histogram, error)
    thresholds, required = zip(*config['bond_order']['thresholds'])
    if len(thresholds) != 3:
        raise ValueError('Joint CUDA extraction requires three coherent-bond detectors')
    thresholds = torch.tensor(thresholds, device=device, dtype=torch.float32)
    required = torch.tensor(required, device=device, dtype=torch.int32)
    masks = torch.empty((4, n), device=device, dtype=torch.bool)
    counts = torch.zeros(4, device=device, dtype=torch.int64)
    native.masks(neighbors, corr, ptm, thresholds, required, masks, counts)
    result = [native.components(neighbors, masks[0], None, 0., config['minimum_tracked_atoms'])]
    for i, (threshold, _) in enumerate(config['bond_order']['thresholds'], 1):
        result.append(native.components(neighbors, masks[i], corr, threshold, config['minimum_tracked_atoms']))
    if int(error):
        raise FloatingPointError('Nonfinite joint CUDA bond correlation')
    output = (torch.stack(result).cpu().numpy(), counts.cpu().numpy(), histogram.cpu().numpy())
    if diagnostics:
        return output, dict(neighbors=neighbors.cpu().numpy(), q6=q.cpu().numpy(),
                            correlations=corr.cpu().numpy(), solid=masks[1:].cpu().numpy())
    return output


def structure(points, box, config, *, backend='resident', ptm_backend='voronoi', include_ptm=False, diagnostics=False):
    if config['detectors'] != ['ptm', 'coherent_loose', 'coherent', 'coherent_strict'] or config['bond_order']['l'] != 6:
        raise ValueError('CUDA prototype requires the declared four-detector l=6 protocol')
    if backend in ('joint', 'resident'):
        if ptm_backend not in ('voronoi', 'distance'):
            raise ValueError('Joint CUDA extraction requires CUDA PTM')
        from .ptm import labels
        if backend == 'joint':
            ptm, (x, b, neighbors, degree) = labels(points, box, config['ptm_rmsd_cutoff'],
                neighbor_radius=2*config['neighbor_cutoff_A'], ordering=ptm_backend,
                device_output=True, joint_cutoff=config['neighbor_cutoff_A'])
            h = torch.empty((len(points), 13, 2), device=x.device, dtype=torch.float32)
            q = torch.empty(len(points), device=x.device, dtype=torch.float32)
            extension().harmonics(x, b, neighbors, degree, h, q)
        else:
            ptm = labels(points, box, config['ptm_rmsd_cutoff'],
                neighbor_radius=2*config['neighbor_cutoff_A'], ordering=ptm_backend, device_output=True)
            neighbors, h, q = prepare(points, box, config['neighbor_cutoff_A'], 'cuda')
        result = finish_joint((neighbors, h, q), ptm, config, diagnostics=diagnostics)
        if diagnostics:
            result, details = result
        output = (*result, ptm.cpu().numpy()) if include_ptm else result
        return (output, details) if diagnostics else output
    if ptm_backend == 'ovito':
        ptm = full_labels(points, box, config['ptm_rmsd_cutoff'])
    elif ptm_backend in ('voronoi', 'distance'):
        from .ptm import labels
        ptm = labels(points, box, config['ptm_rmsd_cutoff'],
                     neighbor_radius=2*config['neighbor_cutoff_A'], ordering=ptm_backend)
    else:
        raise ValueError(f'Unknown PTM backend: {ptm_backend}')
    result = finish(prepare(points, box, config['neighbor_cutoff_A'], backend), ptm, config, diagnostics=diagnostics)
    if diagnostics:
        result, details = result
    output = (*result, ptm) if include_ptm else result
    return (output, details) if diagnostics else output
