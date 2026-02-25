"""Cluster-figure utilities for post-training analysis visualizations."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import textwrap
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.training_methods.contrastive_learning.analysis_utils import _sample_indices

def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def _cluster_palette(n_colors: int) -> list[str]:
    if n_colors <= 0:
        return []
    # Use matplotlib's tab10 categorical palette for cluster labeling.
    base = [mcolors.to_hex(c) for c in plt.get_cmap("tab10")(np.linspace(0, 1, 10))]
    if n_colors <= len(base):
        return base[:n_colors]
    extras = [mcolors.to_hex(c) for c in plt.cm.tab20(np.linspace(0, 1, 20))]
    palette = list(base)
    for color in extras:
        if len(palette) >= n_colors:
            break
        if color not in palette:
            palette.append(color)
    if len(palette) < n_colors:
        raise RuntimeError(
            f"Failed to build enough distinct colors for {n_colors} clusters; got {len(palette)}."
        )
    return palette


def _build_cluster_color_map(cluster_labels: np.ndarray) -> dict[int, str]:
    labels = np.asarray(cluster_labels, dtype=int).reshape(-1)
    valid_ids = sorted(int(v) for v in np.unique(labels) if int(v) >= 0)
    if not valid_ids:
        raise ValueError("Cannot build cluster color map: no non-negative cluster IDs were found.")
    colors = _cluster_palette(len(valid_ids))
    return {cid: colors[i] for i, cid in enumerate(valid_ids)}


def _boost_saturation(colors: np.ndarray, factor: float) -> np.ndarray:
    arr = np.asarray(colors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"colors must have shape (N, 3) for saturation boost, got {arr.shape}."
        )
    if float(factor) <= 0.0:
        raise ValueError(f"saturation factor must be > 0, got {factor}.")
    if abs(float(factor) - 1.0) < 1e-6:
        return arr.copy()
    # Rec.709 luminance weights preserve perceived brightness while boosting chroma.
    lum = (
        0.2126 * arr[:, 0:1]
        + 0.7152 * arr[:, 1:2]
        + 0.0722 * arr[:, 2:3]
    )
    out = lum + (arr - lum) * float(factor)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _srgb_to_linear(colors: np.ndarray) -> np.ndarray:
    arr = np.asarray(colors, dtype=np.float32)
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(
            "sRGB input must be in [0, 1] for linear conversion; "
            f"got min={float(np.min(arr)):.4f}, max={float(np.max(arr)):.4f}."
        )
    return np.power(arr, 2.2).astype(np.float32, copy=False)


def _linear_to_srgb(colors: np.ndarray) -> np.ndarray:
    arr = np.asarray(colors, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    return np.power(arr, 1.0 / 2.2).astype(np.float32, copy=False)


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
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(
            f"points must have shape (N, >=3) for radius estimation, got {pts.shape}."
        )
    pts = pts[:, :3]
    n_pts = int(pts.shape[0])
    if n_pts == 0:
        raise ValueError("Cannot estimate sphere radius from zero points.")
    finite = np.all(np.isfinite(pts), axis=1)
    if not np.any(finite):
        raise ValueError("All points are non-finite; cannot estimate sphere radius.")
    pts = pts[finite]
    n_pts = int(pts.shape[0])
    if n_pts == 0:
        raise ValueError("No finite points remain after filtering; cannot estimate sphere radius.")

    mins = np.min(pts, axis=0).astype(np.float64)
    maxs = np.max(pts, axis=0).astype(np.float64)
    extents = np.maximum(maxs - mins, 1e-8)
    volume = float(extents[0] * extents[1] * extents[2])
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError(
            f"Invalid bounding-box volume for radius estimation: {volume}."
        )
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

    if spacing_nn_min is None:
        spacing = spacing_density
    else:
        spacing = float(spacing_nn_min)

    if not np.isfinite(spacing) or spacing <= 1e-9:
        raise ValueError(
            "Computed invalid point spacing for sphere radius estimation: "
            f"spacing={spacing}, spacing_density={spacing_density}, spacing_nn_min={spacing_nn_min}."
        )

    radius = 0.499 * spacing
    if not np.isfinite(radius) or radius <= 1e-9:
        raise ValueError(
            f"Computed invalid sphere radius from spacing={spacing}: radius={radius}."
        )
    return float(radius)


def _compute_center_to_edge_colors(
    points: np.ndarray,
    base_color: str,
    *,
    center_lighten: float = 0.85,
    edge_darken: float = 0.10,
    radius_percentile: float = 95.0,
    gamma: float = 0.85,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(
            f"points must have shape (N, >=3) for radial coloring, got {pts.shape}."
        )
    pts = pts[:, :3]
    if pts.shape[0] == 0:
        raise ValueError("Cannot compute radial colors for zero points.")

    try:
        base_rgb = np.asarray(mcolors.to_rgb(str(base_color)), dtype=np.float32)
    except ValueError as exc:
        raise ValueError(f"Invalid base_color {base_color!r} for radial coloring.") from exc
    if base_rgb.shape != (3,):
        raise ValueError(
            f"Expected base color to convert to 3 channels, got shape {base_rgb.shape}."
        )

    if pts.shape[0] == 1:
        return base_rgb.reshape(1, 3)

    centroid = np.mean(pts, axis=0, dtype=np.float64)
    dists = np.linalg.norm(pts - centroid[None, :], axis=1)
    if not np.all(np.isfinite(dists)):
        raise ValueError(
            "Encountered non-finite point distances while computing radial colors."
        )

    radius_scale = float(np.percentile(dists, float(radius_percentile)))
    if not np.isfinite(radius_scale) or radius_scale <= 1e-12:
        t = np.zeros_like(dists, dtype=np.float32)
    else:
        t = np.clip(dists / radius_scale, 0.0, 1.0).astype(np.float32, copy=False)
    t = np.power(t, float(gamma)).astype(np.float32, copy=False)

    center_rgb = base_rgb + float(center_lighten) * (1.0 - base_rgb)
    edge_rgb = base_rgb * (1.0 - float(edge_darken))
    colors = center_rgb[None, :] * (1.0 - t[:, None]) + edge_rgb[None, :] * t[:, None]
    return np.clip(colors, 0.0, 1.0).astype(np.float32, copy=False)


def _compute_local_crystal_order_score(
    points: np.ndarray,
    *,
    knn_k: int = 6,
) -> float:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(
            f"Expected points with shape (N, >=3) for crystal order scoring, got {pts.shape}."
        )
    pts = pts[:, :3]
    if pts.shape[0] <= int(knn_k):
        raise ValueError(
            f"Need more than knn_k={knn_k} points to score local order, got {pts.shape[0]}."
        )

    center_idx = int(np.argmin(np.linalg.norm(pts, axis=1)))
    pts = pts - pts[center_idx]
    dmat = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(dmat, np.inf)
    nn = np.sort(dmat, axis=1)[:, : int(knn_k)]
    if not np.all(np.isfinite(nn)):
        raise ValueError("Encountered non-finite nearest-neighbor distances while scoring order.")

    nn_flat = nn.reshape(-1)
    nn_mean = float(np.mean(nn_flat))
    if not np.isfinite(nn_mean) or nn_mean <= 1e-12:
        raise ValueError(
            f"Invalid nearest-neighbor mean distance computed for score: {nn_mean}."
        )
    nn_cv = float(np.std(nn_flat) / nn_mean)

    first_nn = nn[:, 0]
    cutoff = float(np.median(first_nn) * 1.35)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError(f"Invalid coordination cutoff computed from nearest neighbors: {cutoff}.")
    coord = np.sum(dmat <= cutoff, axis=1).astype(np.float32)
    coord_mean = float(np.mean(coord))
    coord_cv = float(np.std(coord) / max(coord_mean, 1e-6))
    shell_cv = float(np.std(first_nn) / max(float(np.mean(first_nn)), 1e-6))

    disorder = nn_cv + 0.7 * coord_cv + 0.6 * shell_cv
    return float(1.0 / max(disorder, 1e-6))


def _resolve_crystal_like_clusters(
    dataset: Any,
    cluster_labels: np.ndarray,
    *,
    point_scale: float,
    min_clusters: int = 2,
    score_quantile: float = 0.6,
    score_samples_per_cluster: int = 64,
    score_knn_k: int = 6,
    random_seed: int = 0,
) -> dict[str, Any]:
    labels = np.asarray(cluster_labels, dtype=int).reshape(-1)
    if labels.size == 0:
        raise ValueError("Cannot resolve crystal-like clusters: cluster_labels is empty.")
    if not hasattr(dataset, "__getitem__"):
        raise TypeError(
            "Cannot resolve crystal-like clusters: dataset is not indexable and sample "
            "point clouds cannot be evaluated."
        )

    cluster_ids = sorted(int(v) for v in np.unique(labels) if int(v) >= 0)
    if not cluster_ids:
        raise ValueError("Cannot resolve crystal-like clusters: no non-negative cluster IDs were found.")
    min_clusters_eff = max(1, min(int(min_clusters), len(cluster_ids)))
    q = float(score_quantile)
    if not (0.0 <= q <= 1.0):
        raise ValueError(
            f"score_quantile must be in [0, 1], got {q}."
        )
    samples_per_cluster = max(1, int(score_samples_per_cluster))
    knn_k = max(2, int(score_knn_k))

    rng = np.random.default_rng(int(random_seed))
    cluster_stats: dict[int, dict[str, Any]] = {}
    score_by_cluster: dict[int, float] = {}

    for cluster_id in cluster_ids:
        idx = np.flatnonzero(labels == int(cluster_id))
        if idx.size == 0:
            continue
        if idx.size > samples_per_cluster:
            eval_idx = np.sort(rng.choice(idx, size=samples_per_cluster, replace=False))
        else:
            eval_idx = idx
        sample_scores: list[float] = []
        for sample_idx in eval_idx:
            points = _load_points_from_dataset(
                dataset,
                int(sample_idx),
                point_scale=float(point_scale),
            )
            sample_scores.append(
                _compute_local_crystal_order_score(points, knn_k=knn_k)
            )
        if not sample_scores:
            raise ValueError(
                f"Cluster {cluster_id} produced no valid samples for crystal-order scoring."
            )
        arr = np.asarray(sample_scores, dtype=np.float64)
        cluster_score = float(np.median(arr))
        score_by_cluster[int(cluster_id)] = cluster_score
        cluster_stats[int(cluster_id)] = {
            "cluster_size": int(idx.size),
            "score_samples": int(arr.size),
            "order_score_mean": float(np.mean(arr)),
            "order_score_median": cluster_score,
            "order_score_std": float(np.std(arr)),
            "order_score_min": float(np.min(arr)),
            "order_score_max": float(np.max(arr)),
        }

    if len(score_by_cluster) < min_clusters_eff:
        raise ValueError(
            "Not enough clusters were scored for crystal-like selection: "
            f"required at least {min_clusters_eff}, got {len(score_by_cluster)}."
        )

    ranked = sorted(score_by_cluster.items(), key=lambda kv: kv[1], reverse=True)
    threshold = float(np.quantile(np.asarray(list(score_by_cluster.values()), dtype=np.float64), q))
    selected = [cid for cid, score in ranked if float(score) >= threshold]
    if len(selected) < min_clusters_eff:
        selected = [cid for cid, _ in ranked[:min_clusters_eff]]

    return {
        "cluster_ids": [int(v) for v in sorted(selected)],
        "min_clusters_requested": int(min_clusters),
        "min_clusters_effective": int(min_clusters_eff),
        "score_quantile": float(q),
        "score_threshold": float(threshold),
        "score_samples_per_cluster": int(samples_per_cluster),
        "score_knn_k": int(knn_k),
        "order_score_by_cluster": {int(k): float(v) for k, v in score_by_cluster.items()},
        "cluster_stats": cluster_stats,
        "selection_method": "unsupervised_local_order_score",
    }


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


def _save_md_cluster_snapshot(
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    color_map: dict[int, str],
    out_file: Path,
    *,
    title: str,
    visible_cluster_ids: list[int] | None = None,
    max_points: int | None = None,
    point_size: float = 5.6,
    alpha: float = 0.62,
    halo_scale: float = 1.0,
    halo_alpha: float = 0.0,
    view_elev: float = 24.0,
    view_azim: float = 35.0,
) -> dict[str, Any]:
    coords_arr = np.asarray(coords, dtype=np.float32)
    labels = np.asarray(cluster_labels, dtype=int).reshape(-1)
    if coords_arr.ndim != 2 or coords_arr.shape[1] < 3:
        raise ValueError(
            f"coords must have shape (N, >=3), got {coords_arr.shape}."
        )
    coords_arr = coords_arr[:, :3]
    if labels.size != coords_arr.shape[0]:
        raise ValueError(
            "coords and cluster_labels length mismatch: "
            f"{coords_arr.shape[0]} vs {labels.size}."
        )

    mask = labels >= 0
    if visible_cluster_ids is not None:
        visible = np.asarray(sorted(set(int(v) for v in visible_cluster_ids)), dtype=int)
        if visible.size == 0:
            raise ValueError("visible_cluster_ids was provided but empty after normalization.")
        mask &= np.isin(labels, visible)
    if not np.any(mask):
        raise ValueError("No points remained after applying cluster visibility filters.")

    coords_use = coords_arr[mask]
    labels_use = labels[mask]

    sample_idx = _sample_indices_stratified(labels_use, max_points, random_seed=0)
    coords_plot = coords_use[sample_idx]
    labels_plot = labels_use[sample_idx]
    unique_labels = sorted(int(v) for v in np.unique(labels_plot) if int(v) >= 0)
    if not unique_labels:
        raise ValueError("No non-negative cluster labels available for plotting.")

    fig = plt.figure(figsize=(7.8, 7.8), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    if float(point_size) <= 0.0:
        raise ValueError(f"point_size must be > 0, got {point_size}.")
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
    if not np.isfinite(float(view_elev)) or not np.isfinite(float(view_azim)):
        raise ValueError(
            f"view_elev/view_azim must be finite, got elev={view_elev}, azim={view_azim}."
        )

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for cluster_id in unique_labels:
        cluster_mask = labels_plot == cluster_id
        if not np.any(cluster_mask):
            continue
        base_color = color_map.get(cluster_id, "#777777")
        cluster_points = coords_plot[cluster_mask]
        point_colors = _compute_center_to_edge_colors(
            cluster_points,
            base_color,
        )
        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            cluster_points[:, 2],
            c=point_colors,
            s=float(point_size),
            alpha=alpha,
            linewidths=0.0,
            depthshade=False,
        )

    _set_equal_axes_3d(ax, coords_arr)
    ax.view_init(elev=float(view_elev), azim=float(view_azim))
    _draw_cube_wireframe(
        ax,
        np.min(coords_arr, axis=0),
        np.max(coords_arr, axis=0),
        linewidth=1.2,
    )
    ax.set_title(title, fontsize=13, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        if hasattr(axis, "pane"):
            axis.pane.fill = False
            axis.pane.set_edgecolor("white")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.95)
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

    return {
        "out_file": str(out_file),
        "num_points_total": int(coords_arr.shape[0]),
        "num_points_visible": int(coords_use.shape[0]),
        "num_points_rendered": int(coords_plot.shape[0]),
        "clusters_rendered": unique_labels,
        "view_elev": float(view_elev),
        "view_azim": float(view_azim),
    }


def _enumerate_cluster_combinations(
    cluster_ids: list[int],
    order_scores: dict[int, float],
    *,
    min_size: int,
    max_size: int,
    max_outputs: int,
) -> list[tuple[int, ...]]:
    if not cluster_ids:
        return []
    min_size = max(1, int(min_size))
    max_size = max(min_size, int(max_size))
    max_size = min(max_size, len(cluster_ids))
    max_outputs = max(1, int(max_outputs))

    combos: list[tuple[int, ...]] = []
    for size in range(min_size, max_size + 1):
        combos.extend(tuple(int(v) for v in c) for c in combinations(cluster_ids, size))
    if not combos:
        return []

    def _combo_score(combo: tuple[int, ...]) -> tuple[float, int]:
        vals = [float(order_scores.get(int(cid), 0.0)) for cid in combo]
        return (float(np.mean(vals)), len(combo))

    combos_sorted = sorted(
        combos,
        key=lambda c: (_combo_score(c)[0], _combo_score(c)[1]),
        reverse=True,
    )
    return combos_sorted[:max_outputs]


def _save_cluster_combination_library(
    *,
    out_dir: Path,
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    color_map: dict[int, str],
    k_value: int,
    order_scores: dict[int, float],
    min_size: int,
    max_size: int,
    max_outputs: int,
    max_points: int | None,
    point_size: float,
    alpha: float,
    halo_scale: float,
    halo_alpha: float,
    view_elev: float,
    view_azim: float,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cluster_ids = sorted(int(v) for v in np.unique(np.asarray(cluster_labels, dtype=int)) if int(v) >= 0)
    combos = _enumerate_cluster_combinations(
        cluster_ids,
        order_scores,
        min_size=int(min_size),
        max_size=int(max_size),
        max_outputs=int(max_outputs),
    )
    if not combos:
        raise ValueError("No cluster combinations were generated for visualization.")

    records: list[dict[str, Any]] = []
    for idx, combo in enumerate(combos, start=1):
        combo_tag = "-".join(str(int(v)) for v in combo)
        out_path = out_dir / f"combo_{idx:03d}_clusters_{combo_tag}.png"
        title = f"MD clusters combo {idx:03d} (k={k_value}, ids={combo_tag})"
        panel = _save_md_cluster_snapshot(
            coords,
            cluster_labels,
            color_map,
            out_path,
            title=title,
            visible_cluster_ids=[int(v) for v in combo],
            max_points=max_points,
            point_size=point_size,
            alpha=alpha,
            halo_scale=halo_scale,
            halo_alpha=halo_alpha,
            view_elev=view_elev,
            view_azim=view_azim,
        )
        panel["combo_index"] = int(idx)
        panel["cluster_ids"] = [int(v) for v in combo]
        panel["combo_score"] = float(
            np.mean([float(order_scores.get(int(cid), 0.0)) for cid in combo])
        )
        records.append(panel)

    return {
        "out_dir": str(out_dir),
        "num_outputs": int(len(records)),
        "min_size": int(min_size),
        "max_size": int(max_size),
        "max_outputs_requested": int(max_outputs),
        "records": records,
    }


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


def _draw_local_knn_edges(
    ax: Any,
    points: np.ndarray,
    *,
    knn_k: int = 4,
    edge_color: str = "#6f6f6f",
    point_colors: np.ndarray | None = None,
    edge_alpha: float = 0.6,
    edge_linewidth: float = 0.8,
) -> None:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"points must have shape (N, >=3), got {pts.shape}.")
    pts = pts[:, :3]
    n_points = int(pts.shape[0])
    if n_points < 2:
        return
    if not (0.0 <= float(edge_alpha) <= 1.0):
        raise ValueError(f"edge_alpha must be in [0, 1], got {edge_alpha}.")
    if float(edge_linewidth) <= 0.0:
        raise ValueError(f"edge_linewidth must be > 0, got {edge_linewidth}.")

    point_rgb: np.ndarray | None = None
    if point_colors is not None:
        point_colors_arr = np.asarray(point_colors, dtype=np.float32)
        if point_colors_arr.ndim != 2 or point_colors_arr.shape[0] != n_points:
            raise ValueError(
                "point_colors must have shape (N, 3) or (N, 4) matching points, "
                f"got {point_colors_arr.shape} for {n_points} points."
            )
        if point_colors_arr.shape[1] not in (3, 4):
            raise ValueError(
                f"point_colors second dimension must be 3 or 4, got {point_colors_arr.shape[1]}."
            )
        point_rgb = np.clip(point_colors_arr[:, :3], 0.0, 1.0)

    k = max(1, min(int(knn_k), n_points - 1))
    dmat = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    nn_idx = np.argsort(dmat, axis=1)[:, 1 : k + 1]
    drawn: set[tuple[int, int]] = set()
    for i in range(n_points):
        for j in nn_idx[i]:
            j = int(j)
            edge = (min(i, j), max(i, j))
            if edge in drawn:
                continue
            drawn.add(edge)
            p1, p2 = pts[edge[0]], pts[edge[1]]
            if point_rgb is not None:
                edge_rgb = 0.5 * (point_rgb[edge[0]] + point_rgb[edge[1]])
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


def _orient_points_for_crystal_view(
    points: np.ndarray,
    *,
    method: str = "pca",
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, Any]]:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(
            f"points must have shape (N, >=3) for orientation, got {pts.shape}."
        )
    pts = pts[:, :3]
    if pts.shape[0] < 4:
        raise ValueError(
            "Need at least 4 points to compute a stable crystal orientation, "
            f"got {pts.shape[0]}."
        )
    if not np.all(np.isfinite(pts)):
        raise ValueError("Encountered non-finite coordinates while orienting points.")

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

    center_idx = int(np.argmin(np.linalg.norm(pts, axis=1)))
    centered = pts - pts[center_idx]
    cov = centered.T @ centered
    cov_trace = float(np.trace(cov))
    if not np.isfinite(cov_trace) or cov_trace <= float(eps):
        raise ValueError(
            "Degenerate representative geometry for PCA orientation: "
            f"covariance trace={cov_trace:.6e}, points_shape={pts.shape}."
        )
    try:
        eigvals_raw, eigvecs_raw = np.linalg.eigh(cov)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            "Failed to compute PCA orientation for representative points: "
            f"points_shape={pts.shape}, center_index={center_idx}."
        ) from exc

    order = np.argsort(eigvals_raw)[::-1]
    eigvals = np.asarray(eigvals_raw[order], dtype=np.float64)
    basis = np.asarray(eigvecs_raw[:, order], dtype=np.float64)
    if basis.shape != (3, 3):
        raise ValueError(
            f"Expected PCA basis shape (3, 3), got {basis.shape}."
        )

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

    oriented = centered @ basis
    if not np.all(np.isfinite(oriented)):
        raise ValueError(
            "Non-finite coordinates produced by PCA orientation for representative points."
        )
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


def _save_cluster_representatives_figure(
    dataset: Any,
    latents: np.ndarray,
    cluster_labels: np.ndarray,
    color_map: dict[int, str],
    out_file: Path,
    *,
    k_value: int,
    point_scale: float,
    target_points: int = 64,
    knn_k: int = 4,
    orientation_method: str = "pca",
    view_elev: float = 22.0,
    view_azim: float = 38.0,
    projection: str = "ortho",
) -> dict[str, Any]:
    if int(target_points) < 4:
        raise ValueError(
            f"target_points must be >= 4 for orientation stability, got {target_points}."
        )
    if int(knn_k) < 1:
        raise ValueError(f"knn_k must be >= 1, got {knn_k}.")
    if not np.isfinite(float(view_elev)) or not np.isfinite(float(view_azim)):
        raise ValueError(
            f"view_elev/view_azim must be finite, got elev={view_elev}, azim={view_azim}."
        )
    proj_norm = str(projection).strip().lower()
    if proj_norm not in {"ortho", "persp"}:
        raise ValueError(
            "projection must be either 'ortho' or 'persp', "
            f"got {projection!r}."
        )
    method_norm = str(orientation_method).strip().lower()
    if method_norm not in {"pca", "none"}:
        raise ValueError(
            "orientation_method must be one of ['pca', 'none'], "
            f"got {orientation_method!r}."
        )

    reps = _compute_cluster_representative_indices(latents, cluster_labels)
    cluster_ids = sorted(reps.keys())
    n_clusters = len(cluster_ids)
    n_cols = min(3, n_clusters)
    n_rows = int(np.ceil(n_clusters / max(1, n_cols)))
    fig = plt.figure(figsize=(4.1 * n_cols, 4.2 * n_rows), dpi=220)

    records: list[dict[str, Any]] = []
    for pos, cluster_id in enumerate(cluster_ids):
        ax = fig.add_subplot(n_rows, n_cols, pos + 1, projection="3d")
        if hasattr(ax, "set_proj_type"):
            ax.set_proj_type(proj_norm)
        ax.view_init(elev=float(view_elev), azim=float(view_azim))
        sample_idx = int(reps[cluster_id])
        points = _load_points_from_dataset(
            dataset,
            sample_idx,
            point_scale=point_scale,
        )
        center_idx = int(np.argmin(np.linalg.norm(points, axis=1)))
        centered = points - points[center_idx]
        d = np.linalg.norm(centered, axis=1)
        keep = np.argsort(d)[: min(int(target_points), len(centered))]
        local = centered[keep]
        if local.size == 0:
            raise ValueError(
                f"Representative sample at index {sample_idx} for cluster {cluster_id} has no points after filtering."
            )
        local_oriented, orientation_info = _orient_points_for_crystal_view(
            local,
            method=method_norm,
        )
        base_color = color_map.get(cluster_id, "#777777")
        point_colors = _compute_center_to_edge_colors(
            local_oriented,
            base_color,
        )
        _draw_local_knn_edges(
            ax,
            local_oriented,
            knn_k=knn_k,
            point_colors=point_colors,
            edge_alpha=0.62,
            edge_linewidth=0.82,
        )
        ax.scatter(
            local_oriented[:, 0],
            local_oriented[:, 1],
            local_oriented[:, 2],
            c=point_colors,
            s=56,
            alpha=0.94,
            edgecolors="black",
            linewidths=0.25,
            depthshade=False,
        )
        _set_equal_axes_3d(ax, local_oriented)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        panel_label = f"C{pos + 1}"
        ax.set_title(
            panel_label,
            fontsize=11,
            color=base_color,
            pad=3,
            fontweight="bold",
        )
        records.append(
            {
                "panel_label": panel_label,
                "cluster_id": int(cluster_id),
                "sample_index": sample_idx,
                "num_points_plotted": int(local_oriented.shape[0]),
                "orientation": orientation_info,
            }
        )

    fig.suptitle(
        f"Cluster representatives nearest latent centroids (k={k_value})",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return {
        "out_file": str(out_file),
        "orientation_method": method_norm,
        "view_elev": float(view_elev),
        "view_azim": float(view_azim),
        "projection": proj_norm,
        "representatives": records,
    }


def _build_ortho_camera(
    elev_deg: float,
    azim_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build orthonormal camera basis vectors matching matplotlib's 3D view.

    Returns (right, up, forward) unit vectors where *forward* points from the
    scene toward the camera (larger dot product = closer to camera).
    """
    elev = np.radians(float(elev_deg))
    azim = np.radians(float(azim_deg))

    cam_x = np.cos(elev) * np.cos(azim)
    cam_y = np.cos(elev) * np.sin(azim)
    cam_z = np.sin(elev)
    forward = np.array([cam_x, cam_y, cam_z], dtype=np.float64)

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(world_up, forward)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        right = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    else:
        right = right / right_norm

    up = np.cross(forward, right)
    up = up / np.linalg.norm(up)
    return right, up, forward


def _project_to_screen(
    coords_3d: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
    img_w: int,
    img_h: int,
    margin_frac: float = 0.06,
    projection: str = "orthographic",
    perspective_fov_deg: float = 34.0,
    perspective_distance_factor: float = 1.4,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Project 3D points to screen space.

    Returns ``(screen_xy (N, 2), depth (N,), meta)`` with screen-y flipped so
    that world-up corresponds to image-up.
    """
    proj_mode = str(projection).strip().lower()
    if proj_mode not in {"orthographic", "ortho", "perspective", "persp"}:
        raise ValueError(
            "projection must be one of ['orthographic', 'ortho', 'perspective', 'persp'], "
            f"got {projection!r}."
        )

    pts = np.asarray(coords_3d, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"Expected (N, >=3) coordinate array, got shape {pts.shape}.")

    pts3 = pts[:, :3]
    margin_px = float(margin_frac) * min(img_w, img_h)
    usable = min(img_w, img_h) - 2.0 * margin_px
    if usable <= 1.0:
        raise ValueError(
            f"Projection usable pixel span must be > 1, got {usable} "
            f"(img_w={img_w}, img_h={img_h}, margin_frac={margin_frac})."
        )

    if proj_mode in {"orthographic", "ortho"}:
        sx = pts3 @ right
        sy = pts3 @ up
        depth = pts3 @ forward

        x_min, x_max = float(sx.min()), float(sx.max())
        y_min, y_max = float(sy.min()), float(sy.max())
        max_range = max(x_max - x_min, y_max - y_min, 1e-12)
        scale = usable / max_range

        cx = (sx - 0.5 * (x_min + x_max)) * scale + img_w * 0.5
        cy = -(sy - 0.5 * (y_min + y_max)) * scale + img_h * 0.5
        meta = {
            "projection": "orthographic",
            "screen_scale": float(scale),
            "camera_distance": 0.0,
        }
        return np.stack([cx, cy], axis=1), depth, meta

    if not (5.0 <= float(perspective_fov_deg) <= 130.0):
        raise ValueError(
            f"perspective_fov_deg must be in [5, 130], got {perspective_fov_deg}."
        )
    if float(perspective_distance_factor) < 1.05:
        raise ValueError(
            "perspective_distance_factor must be >= 1.05 to avoid clipping, "
            f"got {perspective_distance_factor}."
        )

    center = np.mean(pts3, axis=0, dtype=np.float64)
    rel = pts3 - center[None, :]
    x_rel = rel @ right
    y_rel = rel @ up
    z_rel = rel @ (-forward)

    fov_rad = np.deg2rad(float(perspective_fov_deg))
    xy_extent = max(
        float(np.max(np.abs(x_rel))),
        float(np.max(np.abs(y_rel))),
        1e-8,
    )
    z_extent = max(float(np.max(np.abs(z_rel))), 1e-8)
    dist_from_fov = xy_extent / np.tan(0.5 * fov_rad)
    camera_distance = z_extent + max(
        dist_from_fov,
        float(perspective_distance_factor) * xy_extent,
    )
    z_cam = z_rel + camera_distance
    min_z = float(np.min(z_cam))
    if min_z <= 1e-6:
        raise ValueError(
            "Perspective camera produced non-positive depth. "
            f"min_z={min_z:.6f}, camera_distance={camera_distance:.6f}, "
            f"z_extent={z_extent:.6f}."
        )

    x_proj = x_rel / z_cam
    y_proj = y_rel / z_cam
    x_min, x_max = float(x_proj.min()), float(x_proj.max())
    y_min, y_max = float(y_proj.min()), float(y_proj.max())
    max_range = max(x_max - x_min, y_max - y_min, 1e-12)
    scale = usable / max_range
    cx = (x_proj - 0.5 * (x_min + x_max)) * scale + img_w * 0.5
    cy = -(y_proj - 0.5 * (y_min + y_max)) * scale + img_h * 0.5
    depth = -z_cam
    meta = {
        "projection": "perspective",
        "screen_scale": float(scale),
        "camera_distance": float(camera_distance),
    }
    return np.stack([cx, cy], axis=1), depth, meta


def _render_sphere_shading(
    radius_px: int,
    light_dir: tuple[float, float, float] = (-0.35, -0.45, 0.82),
    fill_light_dir: tuple[float, float, float] = (0.55, 0.22, 0.46),
    ambient: float = 0.22,
    diffuse_coeff: float = 0.68,
    fill_diffuse_coeff: float = 0.22,
    specular_coeff: float = 0.15,
    shininess: float = 42.0,
    sss_strength: float = 0.22,
    sss_wrap: float = 0.55,
    sss_falloff: float = 1.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-compute Blinn-Phong shading for a sphere sprite.

    Returns ``(diffuse_shade, specular_shade, alpha)`` float32 arrays of shape
    ``(D, D)`` where ``D = 2 * radius_px + 1``.  The caller should combine them
    as ``pixel = color * diffuse_shade + white * specular_shade``.
    """
    r = int(radius_px)
    if r < 1:
        raise ValueError(f"Sphere radius must be >= 1 pixel, got {r}.")

    yy, xx = np.mgrid[-r : r + 1, -r : r + 1].astype(np.float64)
    dist2 = xx * xx + yy * yy
    r2 = float(r * r)
    inside = dist2 <= r2

    zz = np.sqrt(np.clip(r2 - dist2, 0.0, None))
    inv_r = 1.0 / float(r)
    nx = xx * inv_r
    ny = -yy * inv_r
    nz = zz * inv_r

    L = np.asarray(light_dir, dtype=np.float64)
    L_len = np.linalg.norm(L)
    if L_len < 1e-12:
        raise ValueError("light_dir must be non-zero.")
    L = L / L_len
    L_fill = np.asarray(fill_light_dir, dtype=np.float64)
    L_fill_len = np.linalg.norm(L_fill)
    if L_fill_len < 1e-12:
        raise ValueError("fill_light_dir must be non-zero.")
    L_fill = L_fill / L_fill_len

    NdotL = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0.0, 1.0)
    NdotL_fill = np.clip(nx * L_fill[0] + ny * L_fill[1] + nz * L_fill[2], 0.0, 1.0)

    Hx, Hy, Hz = L[0], L[1], L[2] + 1.0
    H_len = np.sqrt(Hx**2 + Hy**2 + Hz**2)
    Hx, Hy, Hz = Hx / H_len, Hy / H_len, Hz / H_len
    NdotH = np.clip(nx * Hx + ny * Hy + nz * Hz, 0.0, 1.0)

    Hfx, Hfy, Hfz = L_fill[0], L_fill[1], L_fill[2] + 1.0
    Hf_len = np.sqrt(Hfx**2 + Hfy**2 + Hfz**2)
    Hfx, Hfy, Hfz = Hfx / Hf_len, Hfy / Hf_len, Hfz / Hf_len
    NdotH_fill = np.clip(nx * Hfx + ny * Hfy + nz * Hfz, 0.0, 1.0)

    sss_wrap_val = float(sss_wrap)
    if not (0.0 <= sss_wrap_val <= 1.0):
        raise ValueError(f"sss_wrap must be in [0, 1], got {sss_wrap}.")
    sss_falloff_val = float(sss_falloff)
    if sss_falloff_val <= 0.0:
        raise ValueError(f"sss_falloff must be > 0, got {sss_falloff}.")
    # Fast approximate subsurface scattering using wrapped back-lighting.
    back_light = np.clip((-NdotL + sss_wrap_val) / max(1.0 + sss_wrap_val, 1e-6), 0.0, 1.0)
    sss = float(sss_strength) * np.power(back_light, sss_falloff_val)

    diff = np.clip(
        float(ambient)
        + float(diffuse_coeff) * NdotL
        + float(fill_diffuse_coeff) * NdotL_fill
        + sss,
        0.0,
        1.0,
    )
    diff[~inside] = 0.0
    spec = (
        float(specular_coeff) * (NdotH ** float(shininess))
        + 0.35 * float(specular_coeff) * (NdotH_fill ** max(4.0, 0.55 * float(shininess)))
    )
    spec[~inside] = 0.0

    dist = np.sqrt(dist2)
    alpha = np.clip(float(r) + 0.5 - dist, 0.0, 1.0)

    return diff.astype(np.float32), spec.astype(np.float32), alpha.astype(np.float32)


def _apply_ssao(
    img: np.ndarray,
    depth_buf: np.ndarray,
    *,
    kernel_radius: int = 10,
    n_samples: int = 32,
    strength: float = 0.45,
    depth_bias: float = 0.002,
) -> np.ndarray:
    """Screen-space ambient occlusion: darken crevices between nearby spheres."""
    h, w = depth_buf.shape
    has_geometry = np.isfinite(depth_buf)
    if not np.any(has_geometry):
        return img

    d_vals = depth_buf[has_geometry]
    d_min, d_max = float(d_vals.min()), float(d_vals.max())
    d_range = max(d_max - d_min, 1e-12)
    depth_norm = np.full_like(depth_buf, np.nan)
    depth_norm[has_geometry] = (depth_buf[has_geometry] - d_min) / d_range

    rng = np.random.default_rng(42)
    angles = rng.uniform(0, 2 * np.pi, n_samples)
    radii_frac = np.sqrt(rng.uniform(0, 1, n_samples))
    offsets_x = np.round(radii_frac * kernel_radius * np.cos(angles)).astype(int)
    offsets_y = np.round(radii_frac * kernel_radius * np.sin(angles)).astype(int)

    pad = int(kernel_radius)
    padded = np.pad(depth_norm, pad, mode="constant", constant_values=np.nan)
    padded_geom = np.pad(has_geometry, pad, mode="constant", constant_values=False)

    occlusion = np.zeros((h, w), dtype=np.float32)
    for i in range(n_samples):
        oy, ox = int(offsets_y[i]), int(offsets_x[i])
        shifted = padded[pad + oy : pad + oy + h, pad + ox : pad + ox + w]
        shifted_valid = padded_geom[pad + oy : pad + oy + h, pad + ox : pad + ox + w]
        diff = shifted - depth_norm
        occluded = shifted_valid & has_geometry & (diff > depth_bias)
        occlusion += occluded.astype(np.float32)

    occlusion /= max(1, n_samples)

    factor = np.clip(1.0 - strength * occlusion, 0.0, 1.0)
    factor[~has_geometry] = 1.0

    result = img.copy()
    result[:, :, 0] *= factor
    result[:, :, 1] *= factor
    result[:, :, 2] *= factor
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _save_md_cluster_snapshot_pretty(
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    color_map: dict[int, str],
    out_file: Path,
    *,
    title: str,
    visible_cluster_ids: list[int] | None = None,
    max_points: int | None = None,
    view_elev: float = 24.0,
    view_azim: float = 35.0,
    image_width: int = 2200,
    image_height: int = 2200,
    sphere_radius_px: int = 12,
    projection: str = "perspective",
    perspective_fov_deg: float = 34.0,
    perspective_distance_factor: float = 1.4,
    perspective_near_radius_scale: float = 1.35,
    perspective_far_radius_scale: float = 0.78,
    color_mode: str = "matplotlib_match",
    saturation_boost: float = 1.06,
    light_dir: tuple[float, float, float] = (-0.35, -0.45, 0.82),
    fill_light_dir: tuple[float, float, float] = (0.55, 0.22, 0.46),
    bg_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ambient: float = 0.22,
    diffuse: float = 0.68,
    fill_diffuse: float = 0.22,
    specular: float = 0.15,
    shininess: float = 42.0,
    sss_strength: float = 0.22,
    sss_wrap: float = 0.55,
    sss_falloff: float = 1.8,
    wireframe_color: tuple[int, int, int] = (30, 30, 30),
    wireframe_width: int = 2,
    ssao_enabled: bool = True,
    ssao_strength: float = 0.52,
    ssao_samples: int = 32,
    linear_color_pipeline: bool = True,
) -> dict[str, Any]:
    """Render a publication-quality 3D sphere visualization.

    Each point is rendered as a shaded sprite with perspective projection,
    back-to-front compositing, and SSAO contact shadows.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for pretty sphere rendering.  Install it with "
            "'pip install Pillow'."
        ) from exc

    coords_arr = np.asarray(coords, dtype=np.float32)
    labels = np.asarray(cluster_labels, dtype=int).reshape(-1)
    if coords_arr.ndim != 2 or coords_arr.shape[1] < 3:
        raise ValueError(f"coords must have shape (N, >=3), got {coords_arr.shape}.")
    coords_arr = coords_arr[:, :3]
    if labels.size != coords_arr.shape[0]:
        raise ValueError(
            f"coords and cluster_labels length mismatch: {coords_arr.shape[0]} vs {labels.size}."
        )
    image_width = max(128, int(image_width))
    image_height = max(128, int(image_height))
    sphere_radius_px = max(1, int(sphere_radius_px))
    wireframe_width = max(0, int(wireframe_width))
    ssao_samples = max(1, int(ssao_samples))
    ssao_strength = float(np.clip(float(ssao_strength), 0.0, 1.0))
    perspective_far_radius_scale = max(1e-6, float(perspective_far_radius_scale))
    perspective_near_radius_scale = max(
        perspective_far_radius_scale,
        float(perspective_near_radius_scale),
    )

    mask = labels >= 0
    if visible_cluster_ids is not None:
        visible = np.asarray(sorted(set(int(v) for v in visible_cluster_ids)), dtype=int)
        mask &= np.isin(labels, visible)
    if not np.any(mask):
        raise ValueError("No points remained after applying cluster visibility filters.")

    coords_use = coords_arr[mask]
    labels_use = labels[mask]

    sample_idx = _sample_indices_stratified(labels_use, max_points, random_seed=0)
    coords_plot = coords_use[sample_idx]
    labels_plot = labels_use[sample_idx]
    unique_labels = sorted(int(v) for v in np.unique(labels_plot) if int(v) >= 0)
    n_pts = coords_plot.shape[0]
    color_mode_norm = str(color_mode).strip().lower()
    if color_mode_norm != "matplotlib_match":
        color_mode_norm = "flat"

    point_colors = np.zeros((n_pts, 3), dtype=np.float32)
    for cluster_id in unique_labels:
        cmask = labels_plot == cluster_id
        if not np.any(cmask):
            continue
        base_color = color_map.get(cluster_id, "#777777")
        if color_mode_norm == "matplotlib_match":
            point_colors[cmask] = _compute_center_to_edge_colors(
                coords_plot[cmask],
                base_color,
            )
        else:
            point_colors[cmask] = np.asarray(
                mcolors.to_rgb(str(base_color)),
                dtype=np.float32,
            )

    if abs(float(saturation_boost) - 1.0) > 1e-6:
        point_colors = _boost_saturation(point_colors, float(saturation_boost))
    point_colors = np.clip(point_colors, 0.0, 1.0)

    right, up, forward = _build_ortho_camera(view_elev, view_azim)

    box_mins = np.min(coords_arr, axis=0)
    box_maxs = np.max(coords_arr, axis=0)
    box_corners = np.array(
        [
            [box_mins[0], box_mins[1], box_mins[2]],
            [box_maxs[0], box_mins[1], box_mins[2]],
            [box_maxs[0], box_maxs[1], box_mins[2]],
            [box_mins[0], box_maxs[1], box_mins[2]],
            [box_mins[0], box_mins[1], box_maxs[2]],
            [box_maxs[0], box_mins[1], box_maxs[2]],
            [box_maxs[0], box_maxs[1], box_maxs[2]],
            [box_mins[0], box_maxs[1], box_maxs[2]],
        ],
        dtype=np.float32,
    )

    title_margin = 60
    render_h = max(65, image_height - title_margin)
    all_coords = np.concatenate([coords_plot, box_corners], axis=0)
    all_screen, all_depth, proj_meta = _project_to_screen(
        all_coords,
        right,
        up,
        forward,
        image_width,
        render_h,
        margin_frac=0.08,
        projection=projection,
        perspective_fov_deg=perspective_fov_deg,
        perspective_distance_factor=perspective_distance_factor,
    )
    all_screen[:, 1] += title_margin

    pt_screen = all_screen[:n_pts]
    pt_depth = all_depth[:n_pts]
    box_screen = all_screen[n_pts:]
    box_depth = all_depth[n_pts:]

    # Estimate radius from the full labeled cloud so sampling/subset renders
    # keep physically consistent ball size.
    radius_ref_points = coords_arr[labels >= 0]
    auto_radius_world = _estimate_ball_radius_world(
        radius_ref_points,
        sample_limit=1024,
        random_seed=0,
    )
    # Backward-compatible user control: sphere_radius_px acts as a size multiplier
    # around the new physically derived default.
    user_radius_scale = max(1e-6, float(sphere_radius_px) / 12.0)
    sphere_radius_world = max(1e-9, float(auto_radius_world * user_radius_scale))
    max_radius_cap = max(2, int(0.08 * min(image_width, render_h)))

    proj_mode = str(proj_meta["projection"]).strip().lower()
    if proj_mode == "perspective":
        z_cam = np.clip(-pt_depth, 1e-9, None)
        radius_raw = (
            sphere_radius_world * float(proj_meta["screen_scale"]) / z_cam
        )
        radius_med = max(1e-9, float(np.median(radius_raw)))
        radius_raw = np.clip(
            radius_raw,
            float(perspective_far_radius_scale) * radius_med,
            float(perspective_near_radius_scale) * radius_med,
        )
        point_radii = np.maximum(
            1,
            np.round(radius_raw).astype(np.int32),
        )
        point_radii = np.clip(point_radii, 1, int(max_radius_cap))
        point_depth_per_px = (z_cam / max(float(proj_meta["screen_scale"]), 1e-9)).astype(np.float32)
    else:
        radius_px_const = float(sphere_radius_world * float(proj_meta["screen_scale"]))
        radius_px_const = float(np.clip(radius_px_const, 1.0, float(max_radius_cap)))
        point_radii = np.full(n_pts, int(round(radius_px_const)), dtype=np.int32)
        point_depth_per_px = np.full(
            n_pts,
            float(1.0 / max(float(proj_meta["screen_scale"]), 1e-9)),
            dtype=np.float32,
        )

    sprite_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def _get_sprite(radius: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r = int(radius)
        if r in sprite_cache:
            return sprite_cache[r]
        diff_t, spec_t, alpha_t = _render_sphere_shading(
            r,
            light_dir=light_dir,
            fill_light_dir=fill_light_dir,
            ambient=ambient,
            diffuse_coeff=diffuse,
            fill_diffuse_coeff=fill_diffuse,
            specular_coeff=specular,
            shininess=shininess,
            sss_strength=sss_strength,
            sss_wrap=sss_wrap,
            sss_falloff=sss_falloff,
        )
        yy_s, xx_s = np.mgrid[-r : r + 1, -r : r + 1].astype(np.float64)
        z_px = np.sqrt(np.clip(float(r * r) - (xx_s ** 2 + yy_s ** 2), 0.0, None))
        z_px = z_px.astype(np.float32)
        sprite_cache[r] = (diff_t, spec_t, alpha_t, z_px)
        return sprite_cache[r]

    # -- Split box edges into back (behind spheres) and front (on top) ----------
    all_box_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    back_edges: list[tuple[int, int]] = []
    front_edges: list[tuple[int, int]] = []
    if wireframe_width > 0:
        depth_mid = float(np.median(box_depth))
        for a_idx, b_idx in all_box_edges:
            edge_depth = 0.5 * (float(box_depth[a_idx]) + float(box_depth[b_idx]))
            if edge_depth < depth_mid:
                back_edges.append((a_idx, b_idx))
            else:
                front_edges.append((a_idx, b_idx))

    bg_pil = Image.new("RGB", (image_width, image_height),
                        tuple(int(c * 255) for c in bg_color))
    if back_edges:
        bg_draw = ImageDraw.Draw(bg_pil)
        for a_idx, b_idx in back_edges:
            xa, ya = float(box_screen[a_idx, 0]), float(box_screen[a_idx, 1])
            xb, yb = float(box_screen[b_idx, 0]), float(box_screen[b_idx, 1])
            bg_draw.line([(xa, ya), (xb, yb)], fill=wireframe_color,
                         width=wireframe_width)

    img = np.asarray(bg_pil, dtype=np.float32) / 255.0
    if bool(linear_color_pipeline):
        img = _srgb_to_linear(img)
        point_colors = _srgb_to_linear(point_colors)

    # -- Render spheres back-to-front with depth buffer -----------------------
    depth_buf = np.full((image_height, image_width), -np.inf, dtype=np.float32)

    order = np.argsort(pt_depth)

    for idx in order:
        cx_f, cy_f = float(pt_screen[idx, 0]), float(pt_screen[idx, 1])
        ix, iy = int(round(cx_f)), int(round(cy_f))
        radius_i = int(point_radii[idx])
        diff_shade, spec_shade, alpha_tmpl, z_px = _get_sprite(radius_i)
        d = 2 * radius_i + 1

        x0, y0 = ix - radius_i, iy - radius_i
        sx0 = max(0, -x0)
        sy0 = max(0, -y0)
        sx1 = min(d, image_width - x0)
        sy1 = min(d, image_height - y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        dx1 = dx0 + (sx1 - sx0)
        dy1 = dy0 + (sy1 - sy0)

        if dx0 >= dx1 or dy0 >= dy1:
            continue

        a = alpha_tmpl[sy0:sy1, sx0:sx1]
        ds = diff_shade[sy0:sy1, sx0:sx1]
        ss = spec_shade[sy0:sy1, sx0:sx1]

        color = point_colors[idx]

        sphere_rgb = np.clip(
            color[None, None, :] * ds[:, :, None] + ss[:, :, None],
            0.0,
            1.0,
        )

        a3 = a[:, :, None]
        img[dy0:dy1, dx0:dx1, :] = (
            sphere_rgb * a3 + img[dy0:dy1, dx0:dx1, :] * (1.0 - a3)
        )

        depth_mask = a > 0.1
        local_depth = float(pt_depth[idx]) + z_px[sy0:sy1, sx0:sx1] * float(point_depth_per_px[idx])
        patch = depth_buf[dy0:dy1, dx0:dx1]
        patch[depth_mask] = local_depth[depth_mask]

    # -- SSAO contact shadows -------------------------------------------------
    has_geom = np.isfinite(depth_buf)
    depth_buf[~has_geom] = np.nan
    if ssao_enabled and np.any(has_geom):
        img = _apply_ssao(
            img,
            depth_buf,
            kernel_radius=max(3, int(np.median(point_radii) * 1.6)),
            n_samples=ssao_samples,
            strength=ssao_strength,
        )

    # -- Front wireframe edges (on top of spheres) + Title ---------------------
    if bool(linear_color_pipeline):
        img = _linear_to_srgb(img)
    img_uint8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode="RGB")
    draw = ImageDraw.Draw(pil_img)

    if front_edges:
        for a_idx, b_idx in front_edges:
            xa, ya = float(box_screen[a_idx, 0]), float(box_screen[a_idx, 1])
            xb, yb = float(box_screen[b_idx, 0]), float(box_screen[b_idx, 1])
            draw.line([(xa, ya), (xb, yb)], fill=wireframe_color,
                      width=wireframe_width)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), title, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((image_width - tw) / 2, 12), title, fill=(0, 0, 0), font=font)

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(str(out_file), dpi=(300, 300))

    return {
        "out_file": str(out_file),
        "num_points_total": int(coords_arr.shape[0]),
        "num_points_visible": int(coords_use.shape[0]),
        "num_points_rendered": int(n_pts),
        "clusters_rendered": unique_labels,
        "view_elev": float(view_elev),
        "view_azim": float(view_azim),
        "image_size": (int(image_width), int(image_height)),
        "sphere_radius_px": int(sphere_radius_px),
        "sphere_radius_reference_points": int(radius_ref_points.shape[0]),
        "sphere_radius_world_auto": float(auto_radius_world),
        "sphere_radius_world_used": float(sphere_radius_world),
        "sphere_radius_user_scale": float(user_radius_scale),
        "projection": str(proj_mode),
        "color_mode": str(color_mode_norm),
        "saturation_boost": float(saturation_boost),
        "contrast_boost": 1.0,
        "point_radius_min": int(np.min(point_radii)),
        "point_radius_max": int(np.max(point_radii)),
        "point_radius_median": float(np.median(point_radii)),
        "render_mode": "pretty",
    }


def _resolve_blender_executable(blender_executable: str) -> str:
    exe = str(blender_executable).strip()
    if exe == "":
        raise ValueError("blender_executable must be a non-empty string.")
    if "/" in exe or "\\" in exe:
        path = Path(exe).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Blender executable does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Blender executable path is not a file: {path}")
        return str(path)
    resolved = shutil.which(exe)
    if resolved is None:
        raise FileNotFoundError(
            "Blender executable was not found in PATH. "
            f"Tried '{exe}'. Install Blender or provide an absolute path."
        )
    return str(resolved)


def _save_md_cluster_snapshot_raytrace_blender(
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    color_map: dict[int, str],
    out_file: Path,
    *,
    title: str,
    visible_cluster_ids: list[int] | None = None,
    max_points: int | None = None,
    view_elev: float = 24.0,
    view_azim: float = 35.0,
    image_width: int = 1600,
    image_height: int = 1600,
    projection: str = "perspective",
    perspective_fov_deg: float = 34.0,
    camera_distance_factor: float = 2.8,
    sphere_radius_fraction: float = 0.0105,
    blender_executable: str = "blender",
    cycles_samples: int = 64,
    use_denoise: bool = True,
    use_gpu: bool = False,
    timeout_seconds: int = 1200,
    wireframe_enabled: bool = True,
    wireframe_width_fraction: float = 0.0017,
) -> dict[str, Any]:
    """Render a raytraced MD snapshot with Blender Cycles.

    This is an additive renderer used alongside the existing matplotlib and
    software-shaded outputs. It requires a Blender executable.
    """
    coords_arr = np.asarray(coords, dtype=np.float32)
    labels = np.asarray(cluster_labels, dtype=int).reshape(-1)
    if coords_arr.ndim != 2 or coords_arr.shape[1] < 3:
        raise ValueError(f"coords must have shape (N, >=3), got {coords_arr.shape}.")
    coords_arr = coords_arr[:, :3]
    if labels.size != coords_arr.shape[0]:
        raise ValueError(
            f"coords and cluster_labels length mismatch: {coords_arr.shape[0]} vs {labels.size}."
        )
    image_width = max(256, int(image_width))
    image_height = max(256, int(image_height))
    cycles_samples = max(1, int(cycles_samples))
    timeout_seconds = max(30, int(timeout_seconds))
    projection_norm = str(projection).strip().lower()
    if projection_norm not in {"perspective", "persp", "orthographic", "ortho"}:
        projection_norm = "perspective"
    perspective_fov_deg = float(np.clip(float(perspective_fov_deg), 5.0, 130.0))
    camera_distance_factor = max(1.01, float(camera_distance_factor))
    sphere_radius_fraction = max(1e-6, float(sphere_radius_fraction))
    wireframe_width_fraction = max(0.0, float(wireframe_width_fraction))
    if max_points is not None:
        raise ValueError(
            "Raytrace renderer requires all points and does not support downsampling. "
            f"Expected max_points=None, got {max_points}."
        )

    mask = labels >= 0
    if visible_cluster_ids is not None:
        visible = np.asarray(sorted(set(int(v) for v in visible_cluster_ids)), dtype=int)
        mask &= np.isin(labels, visible)
    if not np.any(mask):
        raise ValueError("No points remained after applying cluster visibility filters.")

    coords_use = coords_arr[mask]
    labels_use = labels[mask]
    coords_plot = coords_use
    labels_plot = labels_use
    unique_labels = sorted(int(v) for v in np.unique(labels_plot) if int(v) >= 0)

    bbox_min = np.min(coords_arr, axis=0)
    bbox_max = np.max(coords_arr, axis=0)
    bbox_diag = max(1e-8, float(np.linalg.norm(bbox_max - bbox_min)))
    # Keep physical sizing anchored to the full labeled cloud so downsampling
    # and subset views do not inflate sphere diameter.
    radius_ref_points = coords_arr[labels >= 0]
    auto_radius_world = _estimate_ball_radius_world(
        radius_ref_points,
        sample_limit=1024,
        random_seed=0,
    )
    # Keep backward compatibility with existing config value while making size
    # physically data-driven by default.
    user_radius_scale = max(1e-6, float(sphere_radius_fraction) / 0.0105)
    sphere_radius_world = max(1e-9, float(auto_radius_world * user_radius_scale))
    wireframe_width_world = float(wireframe_width_fraction) * bbox_diag

    clusters_payload: list[dict[str, Any]] = []
    for cluster_id in unique_labels:
        cmask = labels_plot == cluster_id
        pts = coords_plot[cmask]
        if pts.shape[0] == 0:
            continue
        color_rgb = np.asarray(
            mcolors.to_rgb(str(color_map.get(cluster_id, "#777777"))),
            dtype=np.float32,
        )
        clusters_payload.append(
            {
                "cluster_id": int(cluster_id),
                "color": [float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2]), 1.0],
                "points": np.round(pts.astype(np.float64), 6).tolist(),
            }
        )
    blender_exec = _resolve_blender_executable(blender_executable)
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "title": str(title),
        "bbox_min": [float(v) for v in bbox_min],
        "bbox_max": [float(v) for v in bbox_max],
        "clusters": clusters_payload,
        "render": {
            "image_width": int(image_width),
            "image_height": int(image_height),
            "projection": str(projection_norm),
            "view_elev": float(view_elev),
            "view_azim": float(view_azim),
            "perspective_fov_deg": float(perspective_fov_deg),
            "camera_distance_factor": float(camera_distance_factor),
            "sphere_radius_world": float(sphere_radius_world),
            "cycles_samples": int(cycles_samples),
            "use_denoise": bool(use_denoise),
            "use_gpu": bool(use_gpu),
            "wireframe_enabled": bool(wireframe_enabled),
            "wireframe_width_world": float(wireframe_width_world),
            "wireframe_color": [0.12, 0.12, 0.12, 1.0],
            "background_color": [1.0, 1.0, 1.0, 1.0],
            "background_strength": 1.0,
        },
        "out_file": str(out_file),
    }

    blender_script = textwrap.dedent(
        """
        import argparse
        import json
        import math
        import shutil
        import sys
        from pathlib import Path

        import bpy
        from mathutils import Vector


        def _parse_args():
            argv = sys.argv
            if "--" not in argv:
                raise RuntimeError("Blender script expected '--' argument separator.")
            argv = argv[argv.index("--") + 1 :]
            p = argparse.ArgumentParser(description="Raytrace MD clusters via Blender.")
            p.add_argument("--payload_json", type=str, required=True)
            return p.parse_args(argv)


        def _principled_input(node, names):
            for name in names:
                if name in node.inputs:
                    return node.inputs[name]
            return None

        def _build_material(name, rgba):
            mat = bpy.data.materials.new(name=name)
            mat.use_nodes = True
            nt = mat.node_tree
            nt.nodes.clear()
            out = nt.nodes.new("ShaderNodeOutputMaterial")
            out.location = (300, 0)
            bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
            bsdf.location = (0, 0)
            nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

            base_col = (float(rgba[0]), float(rgba[1]), float(rgba[2]), 1.0)
            _principled_input(bsdf, ["Base Color"]).default_value = base_col

            sss_inp = _principled_input(bsdf, ["Subsurface", "Subsurface Weight"])
            if sss_inp is not None:
                sss_inp.default_value = 0.05
            sss_col = _principled_input(bsdf, ["Subsurface Color"])
            if sss_col is not None:
                sss_col.default_value = base_col
            sss_rad = _principled_input(bsdf, ["Subsurface Radius"])
            if sss_rad is not None:
                sss_rad.default_value = (1.0, 0.35, 0.22)

            rough = _principled_input(bsdf, ["Roughness"])
            if rough is not None:
                rough.default_value = 0.34
            spec = _principled_input(bsdf, ["Specular", "Specular IOR Level"])
            if spec is not None:
                spec.default_value = 0.32
            return mat


        def _add_area_light(name, center, location, energy, size):
            light_data = bpy.data.lights.new(name=name, type="AREA")
            light_data.energy = float(energy)
            light_data.size = float(size)
            obj = bpy.data.objects.new(name, light_data)
            bpy.context.scene.collection.objects.link(obj)
            obj.location = Vector(location)
            direction = center - obj.location
            obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
            return obj


        def _add_wireframe_box(bbox_min, bbox_max, width_world, color_rgba):
            x0, y0, z0 = bbox_min
            x1, y1, z1 = bbox_max
            corners = [
                (x0, y0, z0),
                (x1, y0, z0),
                (x1, y1, z0),
                (x0, y1, z0),
                (x0, y0, z1),
                (x1, y0, z1),
                (x1, y1, z1),
                (x0, y1, z1),
            ]
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ]
            curve = bpy.data.curves.new("MDBoxEdges", "CURVE")
            curve.dimensions = "3D"
            curve.bevel_depth = max(float(width_world), 1e-6)
            curve.bevel_resolution = 1
            for a_idx, b_idx in edges:
                spline = curve.splines.new("POLY")
                spline.points.add(1)
                xa, ya, za = corners[a_idx]
                xb, yb, zb = corners[b_idx]
                spline.points[0].co = (xa, ya, za, 1.0)
                spline.points[1].co = (xb, yb, zb, 1.0)
            obj = bpy.data.objects.new("MDBoxWire", curve)
            bpy.context.scene.collection.objects.link(obj)
            mat = _build_material("MDBoxWireMat", color_rgba)
            obj.data.materials.append(mat)
            return obj


        def main():
            args = _parse_args()
            payload_path = Path(args.payload_json)
            if not payload_path.exists():
                raise FileNotFoundError(f"Payload JSON is missing: {payload_path}")
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            cfg = payload["render"]

            bpy.ops.wm.read_factory_settings(use_empty=True)
            scene = bpy.context.scene
            scene.render.engine = "CYCLES"
            scene.render.resolution_x = int(cfg["image_width"])
            scene.render.resolution_y = int(cfg["image_height"])
            scene.render.resolution_percentage = 100
            scene.render.image_settings.file_format = "PNG"
            scene.render.film_transparent = False
            out_path = Path(str(payload["out_file"])).expanduser()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            scene.render.filepath = str(out_path)
            scene.cycles.samples = int(cfg["cycles_samples"])
            if hasattr(scene.cycles, "use_adaptive_sampling"):
                scene.cycles.use_adaptive_sampling = True
            if hasattr(scene.cycles, "use_denoising"):
                scene.cycles.use_denoising = bool(cfg["use_denoise"])
            scene.view_settings.view_transform = "Standard"
            scene.view_settings.look = "None"
            scene.view_settings.exposure = 0.0
            scene.view_settings.gamma = 1.0

            if bool(cfg.get("use_gpu", False)):
                prefs = bpy.context.preferences
                addon = prefs.addons.get("cycles", None)
                if addon is None:
                    raise RuntimeError("Cycles addon is unavailable; cannot enable GPU raytracing.")
                cprefs = addon.preferences
                cprefs.get_devices()
                gpu_count = 0
                for dev in cprefs.devices:
                    dev.use = True
                    if dev.type != "CPU":
                        gpu_count += 1
                if gpu_count <= 0:
                    raise RuntimeError(
                        "use_gpu=True was requested but no GPU Cycles device is available."
                    )
                scene.cycles.device = "GPU"
            else:
                scene.cycles.device = "CPU"

            world = bpy.data.worlds.new("MDWorld")
            scene.world = world
            world.use_nodes = True
            bg = world.node_tree.nodes.get("Background", None)
            if bg is None:
                raise RuntimeError("Blender world background node is missing.")
            bg.inputs[0].default_value = tuple(float(v) for v in cfg["background_color"])
            bg.inputs[1].default_value = float(cfg.get("background_strength", 1.0))

            bbox_min = Vector(tuple(float(v) for v in payload["bbox_min"]))
            bbox_max = Vector(tuple(float(v) for v in payload["bbox_max"]))
            center = 0.5 * (bbox_min + bbox_max)
            extent = bbox_max - bbox_min
            span = max(float(extent.x), float(extent.y), float(extent.z))
            if not math.isfinite(span) or span <= 1e-8:
                span = 1.0

            camera_data = bpy.data.cameras.new("MDCamera")
            camera_obj = bpy.data.objects.new("MDCamera", camera_data)
            scene.collection.objects.link(camera_obj)
            scene.camera = camera_obj

            elev = math.radians(float(cfg["view_elev"]))
            azim = math.radians(float(cfg["view_azim"]))
            cam_dir = Vector(
                (
                    math.cos(elev) * math.cos(azim),
                    math.cos(elev) * math.sin(azim),
                    math.sin(elev),
                )
            )
            cam_dist = float(cfg["camera_distance_factor"]) * span
            camera_obj.location = center + cam_dir * cam_dist
            camera_obj.rotation_euler = (center - camera_obj.location).to_track_quat("-Z", "Y").to_euler()
            proj = str(cfg["projection"]).lower()
            if proj in {"perspective", "persp"}:
                camera_data.type = "PERSP"
                camera_data.lens_unit = "FOV"
                camera_data.angle = math.radians(float(cfg["perspective_fov_deg"]))
            elif proj in {"orthographic", "ortho"}:
                camera_data.type = "ORTHO"
                camera_data.ortho_scale = 2.25 * span
            else:
                raise ValueError(f"Unsupported projection mode for Blender render: {proj!r}.")

            light_size = 0.65 * span
            _add_area_light(
                "KeyLight",
                center,
                center + Vector((1.9 * span, -1.7 * span, 2.1 * span)),
                energy=600.0,
                size=light_size,
            )
            _add_area_light(
                "FillLight",
                center,
                center + Vector((-2.2 * span, 1.6 * span, 0.9 * span)),
                energy=170.0,
                size=0.9 * light_size,
            )
            _add_area_light(
                "RimLight",
                center,
                center + Vector((-0.4 * span, -2.0 * span, 1.8 * span)),
                energy=280.0,
                size=0.8 * light_size,
            )

            pointcloud_add_supported = False
            if hasattr(bpy.data, "pointclouds"):
                probe_pc = bpy.data.pointclouds.new("MDPointCloudProbe")
                pointcloud_add_supported = hasattr(probe_pc.points, "add")
                bpy.data.pointclouds.remove(probe_pc)

            if pointcloud_add_supported:
                for cluster in payload["clusters"]:
                    pts = cluster["points"]
                    if len(pts) == 0:
                        continue
                    cid = int(cluster["cluster_id"])
                    pc_data = bpy.data.pointclouds.new(f"Cluster_{cid:02d}_Points")
                    pc_data.points.add(len(pts))
                    co_flat = [float(v) for p in pts for v in p]
                    rad_flat = [float(cfg["sphere_radius_world"])] * len(pts)
                    pc_data.points.foreach_set("co", co_flat)
                    pc_data.points.foreach_set("radius", rad_flat)

                    obj = bpy.data.objects.new(f"Cluster_{cid:02d}", pc_data)
                    scene.collection.objects.link(obj)
                    mat = _build_material(f"Cluster_{cid:02d}_Mat", cluster["color"])
                    obj.data.materials.append(mat)
            else:
                # Blender 5.x removed PointCloud.points.add(). Use fast
                # vertex-instancing fallback that works in Cycles.
                print(
                    "INFO: PointCloud points.add API unavailable; "
                    "using mesh-vertex instancing fallback."
                )
                inst_offset = Vector((max(50.0 * span, 10.0), 0.0, 0.0))
                sphere_radius = float(cfg["sphere_radius_world"])
                if not math.isfinite(sphere_radius) or sphere_radius <= 0.0:
                    raise RuntimeError(
                        f"Invalid sphere radius for instancing fallback: {sphere_radius}."
                    )
                for cluster in payload["clusters"]:
                    pts = cluster["points"]
                    if len(pts) == 0:
                        continue
                    cid = int(cluster["cluster_id"])
                    mesh = bpy.data.meshes.new(f"Cluster_{cid:02d}_Verts")
                    shifted_pts = [
                        (
                            float(p[0]) - float(inst_offset.x),
                            float(p[1]) - float(inst_offset.y),
                            float(p[2]) - float(inst_offset.z),
                        )
                        for p in pts
                    ]
                    mesh.from_pydata(shifted_pts, [], [])
                    mesh.update()

                    instancer = bpy.data.objects.new(f"Cluster_{cid:02d}_Instancer", mesh)
                    scene.collection.objects.link(instancer)
                    instancer.instance_type = "VERTS"
                    instancer.show_instancer_for_render = False
                    instancer.show_instancer_for_viewport = False

                    bpy.ops.mesh.primitive_ico_sphere_add(
                        subdivisions=2,
                        radius=sphere_radius,
                        location=(float(inst_offset.x), float(inst_offset.y), float(inst_offset.z)),
                    )
                    sphere_obj = bpy.context.active_object
                    if sphere_obj is None:
                        raise RuntimeError(
                            f"Failed to create template sphere for cluster {cid}."
                        )
                    sphere_obj.name = f"Cluster_{cid:02d}_TemplateSphere"
                    sphere_obj.parent = instancer
                    sphere_obj.matrix_parent_inverse = instancer.matrix_world.inverted()
                    bpy.ops.object.shade_smooth()

                    mat = _build_material(f"Cluster_{cid:02d}_Mat", cluster["color"])
                    sphere_obj.data.materials.append(mat)

            if bool(cfg.get("wireframe_enabled", True)) and float(cfg.get("wireframe_width_world", 0.0)) > 0.0:
                _add_wireframe_box(
                    bbox_min,
                    bbox_max,
                    float(cfg["wireframe_width_world"]),
                    cfg["wireframe_color"],
                )

            result = bpy.ops.render.render(write_still=True)
            if "FINISHED" not in set(result):
                raise RuntimeError(
                    "Blender render operator did not finish successfully: "
                    f"result={result}."
                )

            # Blender may resolve output path with frame tokens depending on
            # internal render settings/version. Ensure requested output exists.
            expected = out_path
            if not expected.exists():
                resolved = Path(
                    bpy.path.abspath(scene.render.frame_path(frame=scene.frame_current))
                )
                if resolved.exists():
                    shutil.copy2(resolved, expected)
                else:
                    candidates = sorted(
                        expected.parent.glob(f"{expected.stem}*{expected.suffix}")
                    )
                    if len(candidates) == 1 and candidates[0].exists():
                        shutil.copy2(candidates[0], expected)
                    else:
                        raise FileNotFoundError(
                            "Blender render finished but output is missing. "
                            f"expected={expected}, resolved={resolved}, "
                            f"candidates={[str(p) for p in candidates]}."
                        )


        if __name__ == "__main__":
            main()
        """
    )

    with tempfile.TemporaryDirectory(prefix="md_raytrace_blender_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        payload_path = tmp_root / "payload.json"
        script_path = tmp_root / "render_md_clusters.py"
        payload_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        script_path.write_text(blender_script, encoding="utf-8")

        cmd = [
            blender_exec,
            "-b",
            "--factory-startup",
            "-P",
            str(script_path),
            "--",
            "--payload_json",
            str(payload_path),
        ]
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=int(timeout_seconds),
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                "Blender raytrace render timed out after "
                f"{timeout_seconds}s for output {out_file}. "
                "Increase timeout_seconds or reduce raytrace sample count/point count."
            ) from exc
        if proc.returncode != 0:
            raise RuntimeError(
                "Blender raytrace render failed with non-zero exit code "
                f"{proc.returncode} for output {out_file}. "
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{proc.stdout[-4000:]}\n"
                f"STDERR:\n{proc.stderr[-4000:]}"
            )
        if "Traceback (most recent call last):" in proc.stderr:
            raise RuntimeError(
                "Blender raytrace script raised a Python exception "
                "even though Blender returned exit code 0. "
                f"Output target: {out_file}\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{proc.stdout[-4000:]}\n"
                f"STDERR:\n{proc.stderr[-4000:]}"
            )

    if not out_file.exists():
        candidates = sorted(out_file.parent.glob(f"{out_file.stem}*{out_file.suffix}"))
        raise FileNotFoundError(
            "Blender reported success but output image is missing. "
            f"expected={out_file}, candidates={[str(p) for p in candidates]}.\n"
            f"STDOUT (tail):\n{proc.stdout[-4000:]}\n"
            f"STDERR (tail):\n{proc.stderr[-4000:]}"
        )

    return {
        "out_file": str(out_file),
        "num_points_total": int(coords_arr.shape[0]),
        "num_points_visible": int(coords_use.shape[0]),
        "num_points_rendered": int(coords_plot.shape[0]),
        "clusters_rendered": unique_labels,
        "view_elev": float(view_elev),
        "view_azim": float(view_azim),
        "projection": str(projection_norm),
        "image_size": (int(image_width), int(image_height)),
        "cycles_samples": int(cycles_samples),
        "sphere_radius_reference_points": int(radius_ref_points.shape[0]),
        "sphere_radius_world_auto": float(auto_radius_world),
        "sphere_radius_user_scale": float(user_radius_scale),
        "sphere_radius_world": float(sphere_radius_world),
        "color_saturation_boost": 1.0,
        "color_contrast_boost": 1.0,
        "blender_executable": str(blender_exec),
        "render_mode": "raytrace_blender",
    }


def _prepare_icl_features(
    latents: np.ndarray,
    *,
    l2_normalize: bool,
    standardize: bool,
    pca_variance: float | None,
    pca_max_components: int,
    random_state: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(latents, dtype=np.float32)
    if x.ndim != 2:
        x = np.reshape(x, (x.shape[0], -1))
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.shape[0] < 3:
        raise ValueError(
            f"Need at least 3 samples to compute ICL curve, got {x.shape[0]}."
        )

    info: dict[str, Any] = {
        "input_dim": int(x.shape[1]),
        "l2_normalize": bool(l2_normalize),
        "standardize": bool(standardize),
    }

    if l2_normalize:
        x = _l2_normalize_rows(x)
    if standardize and x.shape[0] > 1:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    if (
        pca_variance is not None
        and float(pca_variance) > 0.0
        and x.shape[1] > 2
        and x.shape[0] > 3
    ):
        n_max = min(int(pca_max_components), x.shape[1], x.shape[0] - 1)
        if n_max >= 2:
            pca = PCA(n_components=n_max, random_state=random_state)
            proj = pca.fit_transform(x)
            if float(pca_variance) >= 1.0:
                keep = n_max
            else:
                csum = np.cumsum(pca.explained_variance_ratio_)
                keep = int(np.searchsorted(csum, float(pca_variance)) + 1)
                keep = max(2, min(keep, n_max))
            x = proj[:, :keep]
            info["pca_components"] = int(keep)
            info["pca_explained_variance"] = float(
                np.sum(pca.explained_variance_ratio_[:keep])
            )
        else:
            info["pca_components"] = int(x.shape[1])
            info["pca_explained_variance"] = 1.0
    else:
        info["pca_components"] = int(x.shape[1])
        info["pca_explained_variance"] = 1.0

    info["output_dim"] = int(x.shape[1])
    return x.astype(np.float32, copy=False), info


def _compute_icl_curve(
    features: np.ndarray,
    k_values: list[int],
    *,
    covariance_type: str = "diag",
    random_state: int = 42,
) -> dict[int, dict[str, float]]:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {x.shape}.")
    if x.shape[0] < 3:
        raise ValueError(f"Need at least 3 samples for ICL curve, got {x.shape[0]}.")

    # Keep covariance_type for backward-compatible function signature.
    _ = covariance_type

    curve: dict[int, dict[str, float]] = {}
    for k in k_values:
        k_eff = int(k)
        if k_eff < 2:
            raise ValueError(f"Invalid k value {k_eff}; expected >= 2.")
        if k_eff >= x.shape[0]:
            raise ValueError(
                f"Invalid k value {k_eff}: must be < number of samples ({x.shape[0]})."
            )
        model = KMeans(n_clusters=k_eff, random_state=random_state, n_init=10)
        try:
            labels = model.fit_predict(x)
        except Exception as exc:
            raise RuntimeError(
                "Failed to fit KMeans for ICL curve "
                f"at k={k_eff}."
            ) from exc

        # KMeans surrogate for model-selection curve:
        # lower inertia is better; entropy term penalizes highly fragmented assignments.
        inertia = float(model.inertia_)
        counts = np.bincount(labels, minlength=k_eff).astype(np.float64)
        probs = counts / np.clip(counts.sum(), 1.0, None)
        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
        icl = float(inertia + entropy)
        curve[k_eff] = {
            "bic": inertia,
            "entropy": entropy,
            "icl": icl,
        }
    return curve


def _save_icl_curve_figure(
    icl_curve: dict[int, dict[str, float]],
    *,
    selected_k: int,
    out_file: Path,
    y_label: str = "ICL",
) -> dict[str, Any]:
    if not icl_curve:
        raise ValueError("ICL curve is empty; nothing to plot.")
    k_values = sorted(int(k) for k in icl_curve.keys())
    scores = [float(icl_curve[k]["icl"]) for k in k_values]
    if selected_k not in icl_curve:
        raise ValueError(
            f"selected_k={selected_k} is missing from ICL curve keys {k_values}."
        )

    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=220)
    ax.plot(
        k_values,
        scores,
        color="black",
        linestyle="--",
        linewidth=1.6,
        marker="o",
        markersize=6.5,
        markerfacecolor="black",
        markeredgewidth=0.0,
    )
    ax.axvline(
        x=float(selected_k),
        color="black",
        linewidth=1.2,
        linestyle=(0, (5, 5)),
    )
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel(y_label)
    ax.set_xticks(k_values)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

    argmin_idx = int(np.argmin(np.asarray(scores, dtype=np.float64)))
    return {
        "out_file": str(out_file),
        "k_values": [int(v) for v in k_values],
        "icl_values": [float(v) for v in scores],
        "icl_best_k": int(k_values[argmin_idx]),
        "icl_value_selected_k": float(icl_curve[int(selected_k)]["icl"]),
    }


def _save_fixed_k_cluster_figure_set(
    *,
    out_dir: Path,
    dataset: Any,
    latents: np.ndarray,
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    k_value: int,
    point_scale: float,
    l2_normalize: bool,
    standardize: bool,
    pca_variance: float | None,
    pca_max_components: int,
    md_max_points: int | None,
    icl_k_min: int,
    icl_k_max: int,
    icl_max_samples: int | None,
    icl_covariance_type: str,
    representative_points: int,
    md_point_size: float,
    md_point_alpha: float,
    md_halo_scale: float,
    md_halo_alpha: float,
    md_view_elev: float,
    md_view_azim: float,
    representative_orientation_method: str,
    representative_view_elev: float,
    representative_view_azim: float,
    representative_projection: str,
    visible_cluster_sets: list[list[int]] | None = None,
    pretty_render_resolution: int = 2200,
    pretty_render_sphere_radius: int = 7,
    pretty_render_projection: str = "perspective",
    pretty_render_perspective_fov_deg: float = 34.0,
    pretty_render_perspective_distance_factor: float = 1.4,
    pretty_render_color_mode: str = "matplotlib_match",
    pretty_render_saturation_boost: float = 1.06,
    pretty_render_wireframe_width: int = 1,
    raytrace_render_enabled: bool = False,
    raytrace_blender_executable: str = "blender",
    raytrace_render_resolution: int = 1600,
    raytrace_render_max_points: int | None = None,
    raytrace_render_samples: int = 64,
    raytrace_render_projection: str = "perspective",
    raytrace_render_fov_deg: float = 34.0,
    raytrace_render_camera_distance_factor: float = 2.8,
    raytrace_render_sphere_radius_fraction: float = 0.0105,
    raytrace_render_timeout_sec: int = 1200,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = np.asarray(cluster_labels, dtype=int).reshape(-1)
    coords_arr = np.asarray(coords, dtype=np.float32)
    lat_arr = np.asarray(latents, dtype=np.float32)
    if lat_arr.ndim != 2:
        lat_arr = np.reshape(lat_arr, (lat_arr.shape[0], -1))
    if labels.size != lat_arr.shape[0]:
        raise ValueError(
            "cluster_labels and latents size mismatch: "
            f"{labels.size} vs {lat_arr.shape[0]}."
        )
    if coords_arr.ndim != 2 or coords_arr.shape[1] < 3:
        raise ValueError(
            f"coords must have shape (N, >=3), got {coords_arr.shape}."
        )
    if coords_arr.shape[0] != labels.size:
        raise ValueError(
            "coords and cluster_labels size mismatch: "
            f"{coords_arr.shape[0]} vs {labels.size}."
        )

    cluster_ids = sorted(int(v) for v in np.unique(labels) if int(v) >= 0)
    if len(cluster_ids) != int(k_value):
        raise ValueError(
            f"Expected exactly k={k_value} non-negative clusters, but found {len(cluster_ids)}: {cluster_ids}."
        )
    color_map = _build_cluster_color_map(labels)
    projection_norm = str(pretty_render_projection).strip().lower()
    if projection_norm not in {"orthographic", "ortho", "perspective", "persp"}:
        projection_norm = "perspective"
    pretty_render_resolution = max(128, int(pretty_render_resolution))
    pretty_render_sphere_radius = max(1, int(pretty_render_sphere_radius))
    color_mode_norm = str(pretty_render_color_mode).strip().lower()
    if color_mode_norm != "matplotlib_match":
        color_mode_norm = "flat"
    pretty_render_saturation_boost = max(1e-6, float(pretty_render_saturation_boost))
    pretty_render_wireframe_width = max(0, int(pretty_render_wireframe_width))
    raytrace_projection_norm = str(raytrace_render_projection).strip().lower()
    if raytrace_projection_norm not in {"orthographic", "ortho", "perspective", "persp"}:
        raytrace_projection_norm = "perspective"
    raytrace_render_resolution = max(256, int(raytrace_render_resolution))
    if raytrace_render_max_points is not None:
        raise ValueError(
            "raytrace_render_max_points is unsupported: raytrace renderer always uses "
            "all points. Set it to null/0 or remove the override."
        )
    raytrace_render_samples = max(1, int(raytrace_render_samples))
    raytrace_render_fov_deg = float(np.clip(float(raytrace_render_fov_deg), 5.0, 130.0))
    raytrace_render_camera_distance_factor = max(
        1.01,
        float(raytrace_render_camera_distance_factor),
    )
    raytrace_render_sphere_radius_fraction = max(
        1e-6,
        float(raytrace_render_sphere_radius_fraction),
    )
    raytrace_render_timeout_sec = max(30, int(raytrace_render_timeout_sec))

    md_snapshot_kwargs = {
        "max_points": md_max_points,
        "point_size": float(md_point_size),
        "alpha": float(md_point_alpha),
        "halo_scale": float(md_halo_scale),
        "halo_alpha": float(md_halo_alpha),
    }
    pretty_render_kwargs = {
        "max_points": md_max_points,
        "image_width": pretty_render_resolution,
        "image_height": pretty_render_resolution,
        "sphere_radius_px": pretty_render_sphere_radius,
        "projection": projection_norm,
        "perspective_fov_deg": float(pretty_render_perspective_fov_deg),
        "perspective_distance_factor": float(pretty_render_perspective_distance_factor),
        "color_mode": color_mode_norm,
        "saturation_boost": pretty_render_saturation_boost,
        "wireframe_width": pretty_render_wireframe_width,
    }
    raytrace_render_kwargs = {
        "max_points": raytrace_render_max_points,
        "image_width": raytrace_render_resolution,
        "image_height": raytrace_render_resolution,
        "projection": raytrace_projection_norm,
        "perspective_fov_deg": raytrace_render_fov_deg,
        "camera_distance_factor": raytrace_render_camera_distance_factor,
        "sphere_radius_fraction": raytrace_render_sphere_radius_fraction,
        "blender_executable": str(raytrace_blender_executable),
        "cycles_samples": raytrace_render_samples,
        "timeout_seconds": raytrace_render_timeout_sec,
        "wireframe_enabled": bool(pretty_render_wireframe_width > 0),
    }

    # -- 01  All-clusters views (4 rotations) with pretty + optional raytrace --
    all_view_specs = [
        ("view1", float(md_view_elev), float(md_view_azim)),
        ("view2", float(md_view_elev), float(md_view_azim + 90.0)),
        ("view3", float(md_view_elev), float(md_view_azim + 180.0)),
        ("view4", float(md_view_elev), float(md_view_azim + 270.0)),
    ]
    panel_all_views: list[dict[str, Any]] = []
    for view_name, elev, azim in all_view_specs:
        out_name = (
            f"01_md_clusters_all_k{k_value}.png"
            if view_name == "view1"
            else f"01_md_clusters_all_k{k_value}_{view_name}.png"
        )
        snapshot_path = out_dir / out_name
        snapshot_title = f"MD space clusters (k={k_value}, all clusters, {view_name})"
        panel_view = _save_md_cluster_snapshot(
            coords_arr,
            labels,
            color_map,
            snapshot_path,
            title=snapshot_title,
            view_elev=float(elev),
            view_azim=float(azim),
            **md_snapshot_kwargs,
        )
        pretty_path = snapshot_path.with_stem(snapshot_path.stem + "_pretty")
        pretty_info = _save_md_cluster_snapshot_pretty(
            coords_arr,
            labels,
            color_map,
            pretty_path,
            title=snapshot_title,
            view_elev=float(elev),
            view_azim=float(azim),
            **pretty_render_kwargs,
        )
        panel_view["pretty_render"] = pretty_info
        if bool(raytrace_render_enabled):
            raytrace_path = snapshot_path.with_stem(snapshot_path.stem + "_raytrace")
            raytrace_info = _save_md_cluster_snapshot_raytrace_blender(
                coords_arr,
                labels,
                color_map,
                raytrace_path,
                title=snapshot_title,
                view_elev=float(elev),
                view_azim=float(azim),
                **raytrace_render_kwargs,
            )
            panel_view["raytrace_render"] = raytrace_info
        panel_view["view_name"] = str(view_name)
        panel_all_views.append(panel_view)
    panel_all = panel_all_views[0]

    # -- 02  Selected cluster-subset views ------------------------------------

    panel_selected_sets: list[dict[str, Any]] = []
    if visible_cluster_sets:
        for set_idx, id_set in enumerate(visible_cluster_sets):
            ids = sorted(int(v) for v in id_set)
            unknown = [c for c in ids if c not in cluster_ids]
            if unknown:
                raise ValueError(
                    f"visible_cluster_sets[{set_idx}] references cluster IDs "
                    f"{unknown} which do not exist.  Available: {cluster_ids}."
                )
            tag = "-".join(str(c) for c in ids)
            set_path = out_dir / f"02_md_clusters_set_{tag}_k{k_value}.png"
            set_title = f"MD space clusters (k={k_value}, clusters {tag})"
            panel_set = _save_md_cluster_snapshot(
                coords_arr,
                labels,
                color_map,
                set_path,
                title=set_title,
                visible_cluster_ids=ids,
                view_elev=float(md_view_elev),
                view_azim=float(md_view_azim),
                **md_snapshot_kwargs,
            )
            pretty_set_path = set_path.with_stem(set_path.stem + "_pretty")
            pretty_set = _save_md_cluster_snapshot_pretty(
                coords_arr,
                labels,
                color_map,
                pretty_set_path,
                title=set_title,
                visible_cluster_ids=ids,
                view_elev=float(md_view_elev),
                view_azim=float(md_view_azim),
                **pretty_render_kwargs,
            )
            panel_set["pretty_render"] = pretty_set
            if bool(raytrace_render_enabled):
                raytrace_set_path = set_path.with_stem(set_path.stem + "_raytrace")
                raytrace_set = _save_md_cluster_snapshot_raytrace_blender(
                    coords_arr,
                    labels,
                    color_map,
                    raytrace_set_path,
                    title=set_title,
                    visible_cluster_ids=ids,
                    view_elev=float(md_view_elev),
                    view_azim=float(md_view_azim),
                    **raytrace_render_kwargs,
                )
                panel_set["raytrace_render"] = raytrace_set
            panel_set["cluster_ids_shown"] = ids
            panel_selected_sets.append(panel_set)

    # -- 03  ICL curve --------------------------------------------------------

    lat_for_icl = lat_arr
    if icl_max_samples is not None and lat_arr.shape[0] > int(icl_max_samples):
        idx = _sample_indices(lat_arr.shape[0], int(icl_max_samples))
        lat_for_icl = lat_arr[idx]
    icl_features, icl_prep = _prepare_icl_features(
        lat_for_icl,
        l2_normalize=l2_normalize,
        standardize=standardize,
        pca_variance=pca_variance,
        pca_max_components=pca_max_components,
        random_state=42,
    )
    if int(icl_k_min) < 2:
        raise ValueError(f"icl_k_min must be >= 2, got {icl_k_min}.")
    if int(icl_k_max) < int(icl_k_min):
        raise ValueError(
            f"icl_k_max must be >= icl_k_min, got min={icl_k_min}, max={icl_k_max}."
        )
    k_values_icl = [int(k) for k in range(int(icl_k_min), int(icl_k_max) + 1)]
    if int(k_value) not in k_values_icl:
        k_values_icl.append(int(k_value))
    k_values_icl = sorted(set(k_values_icl))
    max_valid_k = int(icl_features.shape[0] - 1)
    k_values_icl = [k for k in k_values_icl if k <= max_valid_k]
    if int(k_value) not in k_values_icl:
        raise ValueError(
            f"Cannot evaluate ICL at k={k_value}; available max k is {max_valid_k} "
            f"for {icl_features.shape[0]} ICL samples."
        )
    icl_curve = _compute_icl_curve(
        icl_features,
        k_values_icl,
        covariance_type=icl_covariance_type,
        random_state=42,
    )
    panel_icl = _save_icl_curve_figure(
        icl_curve,
        selected_k=int(k_value),
        out_file=out_dir / f"03_cluster_count_icl_k{k_value}.png",
    )

    # -- 04  Cluster representatives ------------------------------------------

    panel_reps = _save_cluster_representatives_figure(
        dataset,
        lat_arr,
        labels,
        color_map,
        out_dir / f"04_cluster_representatives_k{k_value}.png",
        k_value=int(k_value),
        point_scale=float(point_scale),
        target_points=int(representative_points),
        knn_k=3,
        orientation_method=str(representative_orientation_method),
        view_elev=float(representative_view_elev),
        view_azim=float(representative_view_azim),
        projection=str(representative_projection),
    )

    return {
        "k_value": int(k_value),
        "output_dir": str(out_dir),
        "cluster_ids": cluster_ids,
        "cluster_color_map": {int(k): str(v) for k, v in color_map.items()},
        "panel_all_clusters": panel_all,
        "panel_all_clusters_views": panel_all_views,
        "panel_selected_sets": panel_selected_sets,
        "panel_icl": panel_icl,
        "panel_representatives": panel_reps,
        "pretty_render_settings": {
            "resolution": int(pretty_render_resolution),
            "sphere_radius_px": int(pretty_render_sphere_radius),
            "projection": str(projection_norm),
            "perspective_fov_deg": float(pretty_render_perspective_fov_deg),
            "perspective_distance_factor": float(pretty_render_perspective_distance_factor),
            "color_mode": str(color_mode_norm),
            "saturation_boost": float(pretty_render_saturation_boost),
            "wireframe_width": int(pretty_render_wireframe_width),
        },
        "raytrace_render_settings": {
            "enabled": bool(raytrace_render_enabled),
            "blender_executable": str(raytrace_blender_executable),
            "resolution": int(raytrace_render_resolution),
            "max_points": (
                None if raytrace_render_max_points is None else int(raytrace_render_max_points)
            ),
            "samples": int(raytrace_render_samples),
            "projection": str(raytrace_projection_norm),
            "fov_deg": float(raytrace_render_fov_deg),
            "camera_distance_factor": float(raytrace_render_camera_distance_factor),
            "sphere_radius_fraction": float(raytrace_render_sphere_radius_fraction),
            "timeout_sec": int(raytrace_render_timeout_sec),
        },
        "visible_cluster_sets": [
            sorted(int(v) for v in s) for s in (visible_cluster_sets or [])
        ],
        "icl_curve_raw": {
            int(k): {key: float(val) for key, val in metrics.items()}
            for k, metrics in icl_curve.items()
        },
        "icl_feature_prep": icl_prep,
        "icl_num_samples": int(icl_features.shape[0]),
        "icl_covariance_type": str(icl_covariance_type),
    }
