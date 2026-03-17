"""3D geometry, edge building, orientation, and representative-index utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .cluster_colors import _darken_rgb


def _build_rotation_view_specs(
    *,
    base_elev: float,
    base_azim: float,
    num_views: int = 4,
) -> list[tuple[str, float, float]]:
    return [
        (
            f"view{idx + 1}",
            float(base_elev),
            float(base_azim + 90.0 * idx),
        )
        for idx in range(int(num_views))
    ]


def _estimate_ball_radius_world(
    points: np.ndarray,
    *,
    sample_limit: int = 1024,
    random_seed: int = 0,
) -> float:
    """Estimate a non-overlapping sphere radius from point spacing.

    The returned radius is chosen so spheres touch at the closest sampled
    nearest-neighbor distance without overlap (uniform global radius).
    """
    pts = np.asarray(points, dtype=np.float32)[:, :3]
    n_pts = pts.shape[0]

    mins = np.min(pts, axis=0).astype(np.float64)
    maxs = np.max(pts, axis=0).astype(np.float64)
    extents = np.maximum(maxs - mins, 1e-8)
    volume = float(extents[0] * extents[1] * extents[2])
    spacing_density = float(np.cbrt(volume / max(1, n_pts)))

    spacing_nn_min = None
    if n_pts >= 3:
        m = min(int(sample_limit), n_pts)
        if m >= 3:
            rng = np.random.default_rng(int(random_seed))
            sample_idx = (
                np.arange(n_pts, dtype=int)
                if m == n_pts
                else rng.choice(n_pts, size=m, replace=False)
            )
            sample_pts = pts[sample_idx].astype(np.float32, copy=False)
            diff = sample_pts[:, None, :] - sample_pts[None, :, :]
            dist2 = np.sum(diff * diff, axis=2, dtype=np.float32)
            np.fill_diagonal(dist2, np.inf)
            nn = np.sqrt(np.min(dist2, axis=1, initial=np.inf))
            nn_valid = nn[np.isfinite(nn) & (nn > 1e-9)]
            if nn_valid.size >= 16:
                spacing_nn_min = float(np.min(nn_valid))

    spacing = spacing_nn_min if spacing_nn_min is not None else spacing_density
    return float(max(0.499 * spacing, 1e-9))


def _set_equal_axes_3d(ax: Any, coords: np.ndarray) -> None:
    mins = np.min(coords, axis=0)
    maxs = np.max(coords, axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    if not np.isfinite(span) or span <= 0.0:
        span = 1.0
    half = 0.5 * span
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 1.0))


def _draw_cube_wireframe(
    ax: Any,
    mins: np.ndarray,
    maxs: np.ndarray,
    *,
    color: str = "black",
    linewidth: float = 1.0,
    alpha: float = 0.95,
) -> None:
    x0, y0, z0 = [float(v) for v in mins]
    x1, y1, z1 = [float(v) for v in maxs]
    corners = np.asarray(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for a, b in edges:
        ax.plot(
            [corners[a, 0], corners[b, 0]],
            [corners[a, 1], corners[b, 1]],
            [corners[a, 2], corners[b, 2]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )


def _sample_indices_stratified(
    labels: np.ndarray,
    max_points: int | None,
    *,
    random_seed: int = 0,
) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    n_samples = int(labels.size)
    if max_points is None or max_points <= 0 or n_samples <= int(max_points):
        return np.arange(n_samples, dtype=int)

    max_points = int(max_points)
    unique, counts = np.unique(labels, return_counts=True)
    weights = counts.astype(np.float64) / max(1, int(np.sum(counts)))
    raw = np.floor(weights * max_points).astype(int)
    raw = np.maximum(raw, 1)

    while int(np.sum(raw)) > max_points:
        idx = int(np.argmax(raw))
        if raw[idx] <= 1:
            break
        raw[idx] -= 1
    while int(np.sum(raw)) < max_points:
        idx = int(np.argmax(counts - raw))
        raw[idx] += 1

    rng = np.random.default_rng(int(random_seed))
    selected_parts: list[np.ndarray] = []
    for cluster_label, quota in zip(unique, raw):
        members = np.flatnonzero(labels == cluster_label)
        if members.size <= int(quota):
            selected_parts.append(members)
            continue
        picked = rng.choice(members, size=int(quota), replace=False)
        selected_parts.append(np.sort(picked))

    if not selected_parts:
        return np.arange(min(max_points, n_samples), dtype=int)
    selected = np.concatenate(selected_parts)
    if selected.size > max_points:
        selected = rng.choice(selected, size=max_points, replace=False)
    return np.sort(selected.astype(int, copy=False))


def _extract_points_from_sample(sample: Any) -> np.ndarray:
    if isinstance(sample, dict):
        if "points" not in sample:
            raise KeyError("Dataset sample dict is missing required key 'points'.")
        points = sample["points"]
    elif torch.is_tensor(sample):
        points = sample
    elif isinstance(sample, (tuple, list)) and len(sample) > 0:
        points = sample[0]
    else:
        raise TypeError(f"Unsupported dataset sample type: {type(sample)!r}.")

    if not torch.is_tensor(points):
        points = torch.as_tensor(points)
    pts = points.detach().cpu().numpy()
    if pts.ndim == 3 and pts.shape[0] == 1:
        pts = pts[0]
    if pts.ndim != 2:
        pts = np.reshape(pts, (-1, pts.shape[-1]))
    if pts.shape[1] < 3:
        raise ValueError(
            "Sample points must have at least 3 columns for xyz coordinates, "
            f"got shape {pts.shape}."
        )
    pts = pts[:, :3]
    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    if pts.size == 0:
        raise ValueError("Sample contains no finite xyz points.")
    return pts.astype(np.float32, copy=False)


def _load_points_from_dataset(
    dataset: Any,
    sample_index: int,
    *,
    point_scale: float = 1.0,
) -> np.ndarray:
    if dataset is None or not hasattr(dataset, "__getitem__"):
        raise TypeError(
            "Dataset does not support indexing; cannot extract representative cluster samples."
        )
    idx = int(sample_index)
    if idx < 0:
        raise ValueError(f"Sample index must be non-negative, got {idx}.")
    try:
        sample = dataset[idx]
    except Exception as exc:
        raise IndexError(
            f"Failed to access dataset sample at index {idx}. Dataset type: {type(dataset)!r}."
        ) from exc
    points = _extract_points_from_sample(sample)
    if point_scale != 1.0:
        points = points * float(point_scale)
    return points


def _resolve_local_coordination_shell(
    points: np.ndarray,
    *,
    min_shell_neighbors: int = 3,
    max_shell_neighbors: int = 6,
    shell_gap_ratio: float = 1.18,
) -> dict[str, Any]:
    pts = np.asarray(points, dtype=np.float32)[:, :3]
    n_points = pts.shape[0]
    if n_points < 2:
        return {
            "directed_neighbors": [[] for _ in range(n_points)],
            "distance_matrix": np.full((n_points, n_points), np.inf, dtype=np.float32),
            "shell_neighbor_counts": np.zeros((n_points,), dtype=np.int32),
            "shell_cutoffs": np.zeros((n_points,), dtype=np.float32),
        }
    min_shell_neighbors = max(1, int(min_shell_neighbors))
    max_shell_neighbors = min(max(min_shell_neighbors, int(max_shell_neighbors)), n_points - 1)

    dmat = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(dmat, np.inf)
    sorted_idx = np.argsort(dmat, axis=1)
    sorted_dist = np.take_along_axis(dmat, sorted_idx, axis=1)

    candidate_count = min(n_points - 1, max_shell_neighbors + 1)
    local_shell_counts = np.zeros(n_points, dtype=np.int32)
    local_cutoffs = np.zeros(n_points, dtype=np.float32)
    directed_neighbors: list[list[int]] = [[] for _ in range(n_points)]
    for point_idx in range(n_points):
        candidate_dist = sorted_dist[point_idx, :candidate_count]
        finite_mask = np.isfinite(candidate_dist)
        candidate_dist = candidate_dist[finite_mask]
        candidate_neighbors = sorted_idx[point_idx, :candidate_dist.size]

        lower_bound = min(min_shell_neighbors, candidate_dist.size)
        upper_bound = min(max_shell_neighbors, candidate_dist.size)

        shell_count = upper_bound
        if candidate_dist.size >= 2:
            gap_start = max(0, lower_bound - 1)
            gap_stop = min(upper_bound, candidate_dist.size - 1)
            for gap_idx in range(gap_start, gap_stop):
                curr = float(candidate_dist[gap_idx])
                nxt = float(candidate_dist[gap_idx + 1])
                if curr <= 0.0:
                    continue
                if (nxt / curr) >= float(shell_gap_ratio):
                    shell_count = gap_idx + 1
                    break
        local_shell_counts[point_idx] = int(shell_count)

        if candidate_dist.size > shell_count:
            cutoff = 0.5 * (
                float(candidate_dist[shell_count - 1]) + float(candidate_dist[shell_count])
            )
        else:
            cutoff = float(candidate_dist[shell_count - 1]) * 1.06
        local_cutoffs[point_idx] = float(cutoff)

        neighbor_ids = [
            int(candidate_neighbors[idx])
            for idx, dist in enumerate(candidate_dist)
            if idx < int(shell_count) or float(dist) <= float(cutoff)
        ]
        if not neighbor_ids:
            neighbor_ids = [int(candidate_neighbors[0])]
        directed_neighbors[point_idx] = sorted(set(int(v) for v in neighbor_ids if int(v) != point_idx))

    return {
        "directed_neighbors": directed_neighbors,
        "distance_matrix": dmat.astype(np.float32, copy=False),
        "shell_neighbor_counts": local_shell_counts,
        "shell_cutoffs": local_cutoffs,
    }


def _build_knn_edges(
    points: np.ndarray,
    *,
    knn_k: int,
    mutual_only: bool,
) -> list[tuple[int, int]]:
    pts = np.asarray(points, dtype=np.float32)[:, :3]
    n_points = int(pts.shape[0])
    if n_points < 2:
        return []
    k_eff = min(max(1, int(knn_k)), n_points - 1)
    dmat = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(dmat, np.inf)
    sorted_idx = np.argsort(dmat, axis=1)
    neighbor_sets = [
        {int(v) for v in sorted_idx[point_idx, :k_eff].tolist()}
        for point_idx in range(n_points)
    ]
    edges: set[tuple[int, int]] = set()
    for point_idx, neighbors in enumerate(neighbor_sets):
        for neighbor_idx in neighbors:
            if mutual_only and point_idx not in neighbor_sets[neighbor_idx]:
                continue
            edge = (min(point_idx, neighbor_idx), max(point_idx, neighbor_idx))
            if edge[0] != edge[1]:
                edges.add(edge)
    return sorted(edges)


def _build_local_coordination_edges(
    points: np.ndarray,
    *,
    min_shell_neighbors: int = 3,
    max_shell_neighbors: int = 6,
    shell_gap_ratio: float = 1.18,
    edge_mode: str = "coordination_shell",
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    shell = _resolve_local_coordination_shell(
        points,
        min_shell_neighbors=min_shell_neighbors,
        max_shell_neighbors=max_shell_neighbors,
        shell_gap_ratio=shell_gap_ratio,
    )
    neighbor_lists = shell["directed_neighbors"]
    local_shell_counts = np.asarray(shell["shell_neighbor_counts"], dtype=np.int32)
    local_cutoffs = np.asarray(shell["shell_cutoffs"], dtype=np.float32)
    n_points = int(local_shell_counts.shape[0])
    if n_points < 2:
        return [], {
            "edge_mode": str(edge_mode),
            "num_edges": 0,
            "shell_neighbor_count_median": 0.0,
            "shell_cutoff_median": 0.0,
        }

    edge_mode_norm = str(edge_mode).strip().lower()
    alias_map = {
        "coordination_shell": "coordination_shell",
        "shell_union": "coordination_shell",
        "union": "coordination_shell",
        "coordination_shell_mutual": "coordination_shell_mutual",
        "shell_mutual": "coordination_shell_mutual",
        "mutual": "coordination_shell_mutual",
    }
    if edge_mode_norm not in alias_map:
        raise ValueError(
            "Unsupported representative edge mode: "
            f"{edge_mode!r}. Expected one of "
            "['coordination_shell', 'coordination_shell_mutual']."
        )
    edge_mode_use = alias_map[edge_mode_norm]

    neighbor_sets = [set(neighbors) for neighbors in neighbor_lists]
    union_edges: set[tuple[int, int]] = set()
    mutual_edges: set[tuple[int, int]] = set()
    for point_idx, neighbors in enumerate(neighbor_lists):
        for neighbor_idx in neighbors:
            edge = (min(point_idx, neighbor_idx), max(point_idx, neighbor_idx))
            if edge[0] == edge[1]:
                continue
            union_edges.add(edge)
            if point_idx in neighbor_sets[neighbor_idx]:
                mutual_edges.add(edge)

    if edge_mode_use == "coordination_shell":
        edges = sorted(union_edges)
    else:
        edges = sorted(mutual_edges if mutual_edges else union_edges)

    edge_info = {
        "edge_mode": str(edge_mode_use),
        "num_edges": int(len(edges)),
        "shell_neighbor_count_median": float(np.median(local_shell_counts)),
        "shell_neighbor_count_min": int(np.min(local_shell_counts)),
        "shell_neighbor_count_max": int(np.max(local_shell_counts)),
        "shell_cutoff_median": float(np.median(local_cutoffs)),
        "candidate_union_edges": int(len(union_edges)),
        "candidate_mutual_edges": int(len(mutual_edges)),
    }

    return edges, edge_info


def _draw_edges(
    ax: Any,
    points: np.ndarray,
    edges: list[tuple[int, int]],
    *,
    edge_color: str = "#5f5f5f",
    point_colors: np.ndarray | None = None,
    edge_alpha: float = 0.72,
    edge_linewidth: float = 1.05,
) -> None:
    pts = np.asarray(points, dtype=np.float32)[:, :3]
    point_rgb: np.ndarray | None = None
    if point_colors is not None:
        point_rgb = np.clip(np.asarray(point_colors, dtype=np.float32)[:, :3], 0.0, 1.0)

    for edge in edges:
        p1, p2 = pts[edge[0]], pts[edge[1]]
        if point_rgb is not None:
            edge_rgb = 0.5 * (point_rgb[int(edge[0])] + point_rgb[int(edge[1])])
            edge_rgb = _darken_rgb(edge_rgb, 0.78)
            color_use: Any = (float(edge_rgb[0]), float(edge_rgb[1]), float(edge_rgb[2]))
        else:
            color_use = edge_color
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color=color_use,
            linewidth=float(edge_linewidth),
            alpha=float(edge_alpha),
        )


def _draw_local_coordination_edges(
    ax: Any,
    points: np.ndarray,
    *,
    min_shell_neighbors: int = 3,
    max_shell_neighbors: int = 6,
    shell_gap_ratio: float = 1.18,
    edge_mode: str = "coordination_shell",
    edge_color: str = "#5f5f5f",
    point_colors: np.ndarray | None = None,
    edge_alpha: float = 0.72,
    edge_linewidth: float = 1.05,
) -> dict[str, Any]:
    pts = np.asarray(points, dtype=np.float32)[:, :3]
    if pts.shape[0] < 2:
        return {"edge_mode": "coordination_shell", "num_edges": 0,
                "shell_neighbor_count_median": 0.0, "shell_cutoff_median": 0.0}
    edges, edge_info = _build_local_coordination_edges(
        pts,
        min_shell_neighbors=min_shell_neighbors,
        max_shell_neighbors=max_shell_neighbors,
        shell_gap_ratio=shell_gap_ratio,
        edge_mode=edge_mode,
    )
    _draw_edges(
        ax,
        pts,
        edges,
        edge_color=edge_color,
        point_colors=point_colors,
        edge_alpha=edge_alpha,
        edge_linewidth=edge_linewidth,
    )
    return edge_info


def _compute_pca_orientation_basis(
    points: np.ndarray,
    *,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    pts = np.asarray(points, dtype=np.float32)[:, :3]

    center_idx = int(np.argmin(np.linalg.norm(pts, axis=1)))
    centered = pts - pts[center_idx]
    cov = centered.T @ centered
    eigvals_raw, eigvecs_raw = np.linalg.eigh(cov)

    order = np.argsort(eigvals_raw)[::-1]
    eigvals = np.asarray(eigvals_raw[order], dtype=np.float64)
    basis = np.asarray(eigvecs_raw[:, order], dtype=np.float64)
    if basis.shape != (3, 3):
        raise ValueError(f"Expected PCA basis shape (3, 3), got {basis.shape}.")

    # Resolve eigenvector sign ambiguity to keep orientation deterministic.
    for axis_idx in range(3):
        proj = centered @ basis[:, axis_idx]
        anchor_idx = int(np.argmax(np.abs(proj)))
        if float(proj[anchor_idx]) < 0.0:
            basis[:, axis_idx] *= -1.0

    det_basis = float(np.linalg.det(basis))
    if not np.isfinite(det_basis):
        raise ValueError(
            "Invalid PCA basis determinant while orienting representative points: "
            f"det={det_basis}."
        )
    if det_basis < 0.0:
        basis[:, 2] *= -1.0
        det_basis = float(np.linalg.det(basis))
    if det_basis <= 0.0:
        raise ValueError(
            "Failed to build a right-handed PCA basis for representative points: "
            f"det={det_basis}."
        )
    return centered, basis, eigvals, int(center_idx), float(det_basis)


def _orient_points_for_crystal_view(
    points: np.ndarray,
    *,
    method: str = "pca",
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, Any]]:
    pts = np.asarray(points, dtype=np.float32)[:, :3]
    method_norm = str(method).strip().lower()
    if method_norm == "none":
        return pts.astype(np.float32, copy=False), {
            "orientation_method": "none",
            "center_index": int(np.argmin(np.linalg.norm(pts, axis=1))),
        }
    if method_norm != "pca":
        raise ValueError(
            "Unsupported orientation method for representative points: "
            f"{method!r}. Expected one of ['pca', 'none']."
        )

    centered, basis, eigvals, center_idx, det_basis = _compute_pca_orientation_basis(
        pts,
        eps=float(eps),
    )
    oriented = centered @ basis
    return oriented.astype(np.float32, copy=False), {
        "orientation_method": "pca",
        "center_index": int(center_idx),
        "pca_eigenvalues": [float(v) for v in eigvals.tolist()],
        "pca_basis_det": float(det_basis),
    }


def _compute_cluster_representative_indices(
    latents: np.ndarray,
    cluster_labels: np.ndarray,
) -> dict[int, int]:
    lat = np.asarray(latents, dtype=np.float32)
    labels = np.asarray(cluster_labels, dtype=int).reshape(-1)
    if lat.ndim != 2:
        lat = np.reshape(lat, (lat.shape[0], -1))
    if labels.size != lat.shape[0]:
        raise ValueError(
            "latents and cluster_labels length mismatch: "
            f"{lat.shape[0]} vs {labels.size}."
        )
    representatives: dict[int, int] = {}
    for cluster_id in sorted(int(v) for v in np.unique(labels) if int(v) >= 0):
        idx = np.flatnonzero(labels == cluster_id)
        if idx.size == 0:
            continue
        center = lat[idx].mean(axis=0, keepdims=True)
        dists = np.linalg.norm(lat[idx] - center, axis=1)
        representatives[cluster_id] = int(idx[int(np.argmin(dists))])
    if not representatives:
        raise ValueError("No non-negative clusters found to select representative samples.")
    return representatives
