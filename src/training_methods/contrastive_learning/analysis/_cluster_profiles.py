from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.spatial import ConvexHull, QhullError, cKDTree

_MATERIAL_PROPERTIES: list[tuple[str, str]] = [
    ("nn_mean", "NN mean"),
    ("coord_mean", "Coordination"),
    ("rdf_peak_r", "RDF peak r"),
    ("rdf_peak_g", "RDF peak g"),
    ("bond_angle_mean", "Bond angle"),
    ("density", "Density"),
]

_TOPOLOGY_PROPERTIES: list[tuple[str, str]] = [
    ("radius_gyration", "Rg"),
    ("linearity", "Linearity"),
    ("planarity", "Planarity"),
    ("sphericity", "Sphericity"),
    ("graph_clustering", "Graph clustering"),
]

_ALL_PROFILE_PROPERTIES: list[tuple[str, str]] = (
    _MATERIAL_PROPERTIES + _TOPOLOGY_PROPERTIES
)


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def resolve_point_scale(cfg: Any) -> float:
    data_cfg = getattr(cfg, "data", None)
    if data_cfg is None:
        return 1.0

    normalize = bool(getattr(data_cfg, "normalize", True))
    if not normalize:
        return 1.0

    radius = float(getattr(data_cfg, "radius", 1.0))
    norm_scale = None
    synth_cfg = getattr(data_cfg, "synthetic", None)
    if synth_cfg is not None:
        norm_scale = getattr(synth_cfg, "normalization_scale", None)
    if norm_scale is None:
        norm_scale = getattr(data_cfg, "normalization_scale", None)
    if norm_scale is None:
        norm_scale = 1.0
    norm_scale = float(norm_scale)
    if abs(norm_scale) < 1e-12:
        return radius
    return radius / norm_scale


def _safe_xyz(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        arr = np.reshape(arr, (-1, arr.shape[-1]))
    if arr.shape[1] > 3:
        arr = arr[:, :3]
    if arr.shape[1] < 3:
        raise ValueError(f"Expected points with at least 3 columns, got {arr.shape}")
    finite = np.all(np.isfinite(arr), axis=1)
    arr = arr[finite]
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return arr.astype(np.float32, copy=False)


def _extract_points_from_item(item: Any) -> torch.Tensor:
    if isinstance(item, dict):
        if "points" not in item:
            raise KeyError("Batch dict is missing required key 'points'")
        points = item["points"]
    elif torch.is_tensor(item):
        points = item
    elif isinstance(item, (tuple, list)) and len(item) > 0:
        points = item[0]
    else:
        raise TypeError(f"Unsupported sample type: {type(item)!r}")
    if not torch.is_tensor(points):
        points = torch.as_tensor(points)
    return points


def _load_point_cloud_from_dataset(
    dataset: Any,
    sample_index: int,
    *,
    point_scale: float = 1.0,
) -> np.ndarray | None:
    if dataset is None or not hasattr(dataset, "__getitem__"):
        return None
    try:
        item = dataset[int(sample_index)]
    except (IndexError, KeyError):
        return None
    points = _extract_points_from_item(item)
    points_np = points.detach().cpu().numpy()
    points_np = _safe_xyz(points_np)
    if point_scale != 1.0:
        points_np = points_np * float(point_scale)
    return points_np


def _estimate_nn_distance(points: np.ndarray, k: int = 6) -> float:
    n = len(points)
    if n < 2:
        return float("nan")
    k_actual = min(int(k) + 1, n)
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k_actual)
    if dists.ndim == 1:
        nn = dists
    else:
        nn = dists[:, 1] if dists.shape[1] > 1 else dists[:, 0]
    if nn.size == 0:
        return float("nan")
    return float(np.mean(nn))


def _compute_coordination_numbers(points: np.ndarray, cutoff: float) -> np.ndarray:
    n = len(points)
    if n == 0 or not np.isfinite(cutoff) or cutoff <= 0.0:
        return np.empty((0,), dtype=np.float32)
    tree = cKDTree(points)
    neighbors = tree.query_ball_point(points, r=float(cutoff))
    coords = np.asarray([max(0, len(v) - 1) for v in neighbors], dtype=np.float32)
    return coords


def _compute_bond_angles(points: np.ndarray, cutoff: float) -> np.ndarray:
    n = len(points)
    if n < 3 or not np.isfinite(cutoff) or cutoff <= 0.0:
        return np.empty((0,), dtype=np.float32)
    tree = cKDTree(points)
    angles: list[float] = []
    for i in range(n):
        neighbors = tree.query_ball_point(points[i], r=float(cutoff))
        neighbors = [j for j in neighbors if j != i]
        if len(neighbors) < 2:
            continue
        if len(neighbors) > 6:
            dists = np.linalg.norm(points[neighbors] - points[i], axis=1)
            neighbors = [neighbors[j] for j in np.argsort(dists)[:6]]
        center = points[i]
        for left_idx in range(len(neighbors)):
            for right_idx in range(left_idx + 1, len(neighbors)):
                vec_left = points[neighbors[left_idx]] - center
                vec_right = points[neighbors[right_idx]] - center
                norm_left = np.linalg.norm(vec_left)
                norm_right = np.linalg.norm(vec_right)
                if norm_left < 1e-8 or norm_right < 1e-8:
                    continue
                cos_val = np.clip(
                    np.dot(vec_left, vec_right) / (norm_left * norm_right),
                    -1.0,
                    1.0,
                )
                angles.append(float(np.degrees(np.arccos(cos_val))))
    return np.asarray(angles, dtype=np.float32)


def _compute_convex_volume(points: np.ndarray) -> float:
    n = len(points)
    if n < 4:
        span = np.ptp(points, axis=0) if n > 0 else np.zeros((3,), dtype=np.float32)
        return float(np.prod(np.maximum(span, 1e-6)))
    try:
        hull = ConvexHull(points)
        volume = float(hull.volume)
        if np.isfinite(volume) and volume > 0.0:
            return volume
    except (QhullError, ValueError):
        pass
    span = np.ptp(points, axis=0)
    return float(np.prod(np.maximum(span, 1e-6)))


def _compute_rdf_peak(points: np.ndarray, nn_mean: float) -> tuple[float | None, float | None]:
    n = len(points)
    if n < 2:
        return None, None
    diff = points[:, None, :] - points[None, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)
    tri = np.triu_indices(n, k=1)
    pair_dists = dist_mat[tri]
    if pair_dists.size == 0:
        return None, None
    upper = np.percentile(pair_dists, 95)
    if np.isfinite(nn_mean) and nn_mean > 0.0:
        upper = min(float(upper), float(nn_mean) * 5.0)
    upper = float(max(upper, np.percentile(pair_dists, 60), 1e-6))
    hist, edges = np.histogram(pair_dists, bins=40, range=(0.0, upper))
    radii = 0.5 * (edges[1:] + edges[:-1])
    dr = float(edges[1] - edges[0]) if len(edges) > 1 else 1.0
    volume = _compute_convex_volume(points)
    density = float(n) / max(volume, 1e-8)
    shell_vol = 4.0 * np.pi * (radii**2) * dr
    expected = 0.5 * float(n) * density * shell_vol
    g_r = np.divide(
        hist.astype(np.float32),
        expected.astype(np.float32),
        out=np.zeros_like(radii, dtype=np.float32),
        where=expected > 1e-12,
    )
    if g_r.size < 2:
        return None, None
    peak_idx = int(np.argmax(g_r[1:]) + 1)
    return float(radii[peak_idx]), float(g_r[peak_idx])


def _compute_graph_clustering(points: np.ndarray, knn_k: int = 4) -> float:
    n = len(points)
    if n < 3:
        return float("nan")
    k_eff = max(1, min(int(knn_k), n - 1))
    tree = cKDTree(points)
    _, nn_idx = tree.query(points, k=k_eff + 1)
    if nn_idx.ndim == 1:
        nn_idx = nn_idx[:, None]
    neighbors: list[set[int]] = []
    for row in nn_idx:
        neighbor_set = {int(v) for v in np.atleast_1d(row)[1:] if int(v) >= 0}
        neighbors.append(neighbor_set)
    coeffs: list[float] = []
    for i in range(n):
        nbrs = list(neighbors[i])
        degree = len(nbrs)
        if degree < 2:
            coeffs.append(0.0)
            continue
        links = 0
        for left_idx in range(degree):
            left = nbrs[left_idx]
            for right_idx in range(left_idx + 1, degree):
                right = nbrs[right_idx]
                if right in neighbors[left]:
                    links += 1
        coeffs.append(2.0 * links / (degree * (degree - 1)))
    return float(np.mean(coeffs)) if coeffs else float("nan")


def _compute_shape_descriptors(points: np.ndarray) -> dict[str, float]:
    if len(points) < 3:
        return {
            "radius_gyration": float("nan"),
            "linearity": float("nan"),
            "planarity": float("nan"),
            "sphericity": float("nan"),
        }
    centered = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.maximum(eigvals, 0.0))[::-1]
    if eigvals.size < 3:
        eigvals = np.pad(eigvals, (0, 3 - eigvals.size), mode="constant")
    l1, l2, l3 = float(eigvals[0]), float(eigvals[1]), float(eigvals[2])
    denom = max(l1, 1e-8)
    return {
        "radius_gyration": float(np.sqrt(np.sum(eigvals))),
        "linearity": float((l1 - l2) / denom),
        "planarity": float((l2 - l3) / denom),
        "sphericity": float(l3 / denom),
    }


def _compute_sample_properties(points: np.ndarray) -> dict[str, float]:
    coords = _safe_xyz(points)
    if coords.size == 0:
        return {name: float("nan") for name, _ in _ALL_PROFILE_PROPERTIES}

    center_idx = int(np.argmin(np.linalg.norm(coords, axis=1)))
    centered = coords - coords[center_idx]
    nn_mean = _estimate_nn_distance(centered)
    cutoff = float(nn_mean * 1.35) if np.isfinite(nn_mean) else float("nan")

    coord = _compute_coordination_numbers(centered, cutoff)
    angles = _compute_bond_angles(centered, cutoff)
    rdf_peak_r, rdf_peak_g = _compute_rdf_peak(centered, nn_mean)
    shape = _compute_shape_descriptors(centered)
    volume = _compute_convex_volume(centered)
    density = float(len(centered) / max(volume, 1e-8))
    return {
        "nn_mean": float(nn_mean) if np.isfinite(nn_mean) else float("nan"),
        "coord_mean": float(np.mean(coord)) if coord.size else float("nan"),
        "rdf_peak_r": float(rdf_peak_r) if rdf_peak_r is not None else float("nan"),
        "rdf_peak_g": float(rdf_peak_g) if rdf_peak_g is not None else float("nan"),
        "bond_angle_mean": float(np.mean(angles)) if angles.size else float("nan"),
        "density": float(density),
        "radius_gyration": float(shape["radius_gyration"]),
        "linearity": float(shape["linearity"]),
        "planarity": float(shape["planarity"]),
        "sphericity": float(shape["sphericity"]),
        "graph_clustering": float(_compute_graph_clustering(centered, knn_k=4)),
    }
