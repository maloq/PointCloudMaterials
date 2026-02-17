import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Tuple
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.path.append(os.getcwd())

from src.data_utils.data_module import RealPointCloudDataModule, SyntheticPointCloudDataModule
from src.training_methods.contrastive_learning.cluster_profile_analysis import (
    generate_cluster_profile_reports,
    resolve_point_scale,
)
from src.training_methods.contrastive_learning.analysis_utils import (
    _sample_indices,
    build_real_coords_dataloader,
    evaluate_latent_equivariance,
    gather_inference_batches,
)
from src.training_methods.contrastive_learning.contrastive_module import BarlowTwinsModule
from src.utils.spd_metrics import (
    compute_cluster_metrics,
    compute_embedding_quality_metrics,
)
from src.utils.spd_utils import apply_rotation
from src.utils.model_utils import load_model_from_checkpoint, resolve_config_path
from src.vis_tools.md_cluster_plot import (
    save_interactive_md_plot,
)
from src.vis_tools.latent_analysis_vis import (
    compute_hdbscan_labels,
    compute_kmeans_labels,
    save_clustering_analysis,
    save_equivariance_plot,
    save_latent_statistics,
    save_latent_tsne,
    save_local_structure_assignments,
    save_md_space_clusters_plot,
    save_pca_visualization,
    save_tsne_plot_with_coords,
)
from src.vis_tools.tsne_vis import compute_tsne


def _as_list_of_str(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(v) for v in list(value)]


def _as_list_of_int(value: Any, *, field_name: str = "value") -> list[int] | None:
    if value is None:
        return None
    values = list(value)
    if not values:
        raise ValueError(f"Invalid {field_name}: expected a non-empty list")
    return [int(v) for v in values]


def _positive_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


def _unwrap_dataset(dataset: Any) -> Any:
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def _extract_class_names(dataset: Any) -> Dict[int, str] | None:
    dataset = _unwrap_dataset(dataset)
    class_names_raw = getattr(dataset, "class_names", None)
    if class_names_raw is None:
        return None
    class_names = dict(class_names_raw)
    return {int(k): str(v) for k, v in class_names.items()}


def _fmt_metric(value: Any, digits: int = 4) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _as_int_mapping(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if isinstance(value, DictConfig):
        value = dict(value)
    if not hasattr(value, "items"):
        return {}

    out: dict[str, int] = {}
    for key, raw in value.items():
        try:
            runs = int(raw)
        except (TypeError, ValueError):
            continue
        if runs > 0:
            out[str(key)] = runs
    return out


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def _cluster_palette(n_colors: int) -> list[str]:
    if n_colors <= 0:
        return []
    base = [
        "#b30000",  # red
        "#e6a6ef",  # pink
        "#5a189a",  # purple
        "#0ea323",  # green
        "#1f36d6",  # blue
        "#f2b30b",  # orange
        "#1f7a8c",
        "#888888",
    ]
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
    point_size: float = 4.0,
    alpha: float = 0.96,
    halo_scale: float = 5.0,
    halo_alpha: float = 0.26,
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
    if float(halo_scale) <= 0.0:
        raise ValueError(f"halo_scale must be > 0, got {halo_scale}.")
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
    if not (0.0 <= float(halo_alpha) <= 1.0):
        raise ValueError(f"halo_alpha must be in [0, 1], got {halo_alpha}.")
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
        color = color_map.get(cluster_id, "#777777")
        if float(halo_alpha) > 0.0 and float(halo_scale) > 1.0:
            ax.scatter(
                coords_plot[cluster_mask, 0],
                coords_plot[cluster_mask, 1],
                coords_plot[cluster_mask, 2],
                c=color,
                s=float(point_size) * float(halo_scale),
                alpha=float(halo_alpha) * float(alpha),
                linewidths=0.0,
                depthshade=False,
            )
        ax.scatter(
            coords_plot[cluster_mask, 0],
            coords_plot[cluster_mask, 1],
            coords_plot[cluster_mask, 2],
            c=color,
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
) -> None:
    if len(points) < 2:
        return
    k = max(1, min(int(knn_k), len(points) - 1))
    dmat = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    nn_idx = np.argsort(dmat, axis=1)[:, 1 : k + 1]
    drawn: set[tuple[int, int]] = set()
    for i in range(len(points)):
        for j in nn_idx[i]:
            j = int(j)
            edge = (min(i, j), max(i, j))
            if edge in drawn:
                continue
            drawn.add(edge)
            p1, p2 = points[edge[0]], points[edge[1]]
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                color=edge_color,
                linewidth=0.8,
                alpha=0.6,
            )


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
) -> dict[str, Any]:
    reps = _compute_cluster_representative_indices(latents, cluster_labels)
    cluster_ids = sorted(reps.keys())
    n_clusters = len(cluster_ids)
    n_cols = min(3, n_clusters)
    n_rows = int(np.ceil(n_clusters / max(1, n_cols)))
    fig = plt.figure(figsize=(4.1 * n_cols, 4.2 * n_rows), dpi=220)

    records: list[dict[str, Any]] = []
    for pos, cluster_id in enumerate(cluster_ids):
        ax = fig.add_subplot(n_rows, n_cols, pos + 1, projection="3d")
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
        _draw_local_knn_edges(ax, local, knn_k=knn_k)
        color = color_map.get(cluster_id, "#777777")
        ax.scatter(
            local[:, 0],
            local[:, 1],
            local[:, 2],
            c=color,
            s=56,
            alpha=0.94,
            edgecolors="black",
            linewidths=0.25,
            depthshade=False,
        )
        _set_equal_axes_3d(ax, local)
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
            color=color,
            pad=3,
            fontweight="bold",
        )
        records.append(
            {
                "panel_label": panel_label,
                "cluster_id": int(cluster_id),
                "sample_index": sample_idx,
                "num_points_plotted": int(local.shape[0]),
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
        "representatives": records,
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
    if covariance_type not in {"full", "diag", "spherical", "tied"}:
        raise ValueError(
            "Invalid covariance_type for ICL curve. "
            f"Expected one of full/diag/spherical/tied, got {covariance_type!r}."
        )
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {x.shape}.")
    if x.shape[0] < 3:
        raise ValueError(f"Need at least 3 samples for ICL curve, got {x.shape[0]}.")

    curve: dict[int, dict[str, float]] = {}
    for k in k_values:
        k_eff = int(k)
        if k_eff < 2:
            raise ValueError(f"Invalid k value {k_eff}; expected >= 2.")
        if k_eff >= x.shape[0]:
            raise ValueError(
                f"Invalid k value {k_eff}: must be < number of samples ({x.shape[0]})."
            )
        model = GaussianMixture(
            n_components=k_eff,
            covariance_type=covariance_type,
            random_state=random_state,
            n_init=3,
            reg_covar=1e-5,
            max_iter=300,
        )
        try:
            model.fit(x)
        except Exception as exc:
            raise RuntimeError(
                "Failed to fit GaussianMixture for ICL curve "
                f"at k={k_eff}, covariance_type={covariance_type}."
            ) from exc
        bic = float(model.bic(x))
        probs = model.predict_proba(x)
        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
        # ICL approximation: BIC penalized by soft-assignment entropy.
        icl = float(bic - 2.0 * entropy)
        curve[k_eff] = {
            "bic": bic,
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
    min_crystal_clusters: int,
    crystal_score_quantile: float,
    crystal_score_samples_per_cluster: int,
    crystal_score_knn_k: int,
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
    combo_min_size: int,
    combo_max_size: int,
    combo_max_outputs: int,
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

    all_view_specs = [
        ("view1", float(md_view_elev), float(md_view_azim)),
        ("view2", float(md_view_elev), float(md_view_azim + 90.0)),
        ("view3", float(md_view_elev), float(md_view_azim + 180.0)),
        ("view4", float(md_view_elev), float(md_view_azim + 270.0)),
    ]
    panel_all_views: list[dict[str, Any]] = []
    for view_name, elev, azim in all_view_specs:
        out_name = (
            f"02_md_clusters_all_k{k_value}.png"
            if view_name == "view1"
            else f"02_md_clusters_all_k{k_value}_{view_name}.png"
        )
        panel_view = _save_md_cluster_snapshot(
            coords_arr,
            labels,
            color_map,
            out_dir / out_name,
            title=f"MD space clusters (k={k_value}, all clusters, {view_name})",
            max_points=md_max_points,
            point_size=float(md_point_size),
            alpha=float(md_point_alpha),
            halo_scale=float(md_halo_scale),
            halo_alpha=float(md_halo_alpha),
            view_elev=float(elev),
            view_azim=float(azim),
        )
        panel_view["view_name"] = str(view_name)
        panel_all_views.append(panel_view)
    panel_all = panel_all_views[0]

    crystal_info = _resolve_crystal_like_clusters(
        dataset,
        labels,
        point_scale=float(point_scale),
        min_clusters=int(min_crystal_clusters),
        score_quantile=float(crystal_score_quantile),
        score_samples_per_cluster=int(crystal_score_samples_per_cluster),
        score_knn_k=int(crystal_score_knn_k),
        random_seed=0,
    )
    order_scores = {
        int(cid): float(score)
        for cid, score in crystal_info.get("order_score_by_cluster", {}).items()
    }
    if not order_scores:
        raise ValueError(
            "Crystal-like selection did not produce order_score_by_cluster."
        )

    combination_library = _save_cluster_combination_library(
        out_dir=out_dir / f"01b_cluster_combinations_k{k_value}",
        coords=coords_arr,
        cluster_labels=labels,
        color_map=color_map,
        k_value=int(k_value),
        order_scores=order_scores,
        min_size=int(combo_min_size),
        max_size=int(combo_max_size),
        max_outputs=int(combo_max_outputs),
        max_points=md_max_points,
        point_size=float(md_point_size),
        alpha=float(md_point_alpha),
        halo_scale=float(md_halo_scale),
        halo_alpha=float(md_halo_alpha),
        view_elev=float(md_view_elev),
        view_azim=float(md_view_azim),
    )
    panel_crystal_path = out_dir / f"01_md_clusters_crystal_only_k{k_value}.png"
    crystal_ids = [int(v) for v in crystal_info.get("cluster_ids", [])]
    if len(crystal_ids) < max(1, int(min_crystal_clusters)):
        raise ValueError(
            "Crystal-only panel requires at least "
            f"{int(min_crystal_clusters)} crystal-like clusters, but only "
            f"{len(crystal_ids)} were selected: {crystal_ids}."
        )
    panel_crystal = _save_md_cluster_snapshot(
        coords_arr,
        labels,
        color_map,
        panel_crystal_path,
        title=(
            f"MD space clusters (k={k_value}, crystal-like only, "
            f"top quantile={float(crystal_info['score_quantile']):.2f})"
        ),
        visible_cluster_ids=crystal_ids,
        max_points=md_max_points,
        point_size=float(md_point_size),
        alpha=float(md_point_alpha),
        halo_scale=float(md_halo_scale),
        halo_alpha=float(md_halo_alpha),
        view_elev=float(md_view_elev),
        view_azim=float(md_view_azim),
    )

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

    panel_reps = _save_cluster_representatives_figure(
        dataset,
        lat_arr,
        labels,
        color_map,
        out_dir / f"04_cluster_representatives_k{k_value}.png",
        k_value=int(k_value),
        point_scale=float(point_scale),
        target_points=int(representative_points),
        knn_k=4,
    )

    return {
        "k_value": int(k_value),
        "output_dir": str(out_dir),
        "cluster_ids": cluster_ids,
        "cluster_color_map": {int(k): str(v) for k, v in color_map.items()},
        "panel_01_crystal_only": panel_crystal,
        "panel_01b_cluster_combinations": combination_library,
        "panel_02_all_clusters": panel_all,
        "panel_02_all_clusters_views": panel_all_views,
        "panel_03_cluster_count_icl": panel_icl,
        "panel_04_cluster_representatives": panel_reps,
        "crystal_like_selection": crystal_info,
        "icl_curve_raw": {
            int(k): {key: float(val) for key, val in metrics.items()}
            for k, metrics in icl_curve.items()
        },
        "icl_feature_prep": icl_prep,
        "icl_num_samples": int(icl_features.shape[0]),
        "icl_covariance_type": str(icl_covariance_type),
    }


def _extract_pc_and_class_id(batch: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(batch, dict):
        if "points" not in batch:
            raise KeyError("Batch dict is missing required key 'points'")
        return batch["points"], batch.get("class_id")

    if torch.is_tensor(batch):
        return batch, None

    if isinstance(batch, (tuple, list)) and len(batch) > 0:
        points = batch[0]
        class_id = None
        if len(batch) > 1:
            candidate = batch[1]
            if torch.is_tensor(candidate):
                if candidate.dtype in (
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                    torch.uint8,
                    torch.long,
                ) and candidate.dim() <= 1:
                    class_id = candidate
                elif candidate.dim() <= 1:
                    warnings.warn(
                        f"Batch element [1] has dtype={candidate.dtype} and dim={candidate.dim()}. "
                        "Expected integer dtype for class_id. Skipping -- provide integer labels "
                        "or use a dict batch to avoid silent label loss.",
                    )
            elif isinstance(candidate, (int, np.integer)):
                class_id = torch.as_tensor(candidate, dtype=torch.long)
        return points, class_id

    raise TypeError(f"Unsupported batch type for extraction: {type(batch)!r}")


def _random_rotation_matrices(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    seed: int | None = None,
) -> torch.Tensor:
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
    rand = torch.randn(
        (batch_size, 3, 3),
        dtype=torch.float32,
        device="cpu",
        generator=generator,
    )
    q, r = torch.linalg.qr(rand)
    d = torch.diagonal(r, dim1=-2, dim2=-1).sign()
    d = torch.where(d == 0, torch.ones_like(d), d)
    q = q * d.unsqueeze(-2)
    det = torch.det(q)
    neg_mask = det < 0
    if torch.any(neg_mask):
        q[neg_mask, :, 0] *= -1.0
    return q.to(device=device, dtype=dtype)


@contextmanager
def _temporary_disable_dataset_aug(dataloader: torch.utils.data.DataLoader):
    changes: list[tuple[Any, str, float]] = []
    ds = getattr(dataloader, "dataset", None)
    while ds is not None:
        for attr in ("rotation_scale", "noise_scale", "jitter_scale", "scaling_range"):
            if hasattr(ds, attr):
                prev = float(getattr(ds, attr))
                changes.append((ds, attr, prev))
                if prev != 0.0:
                    setattr(ds, attr, 0.0)
        ds = getattr(ds, "dataset", None)
    try:
        yield
    finally:
        for target, attr, prev in changes:
            setattr(target, attr, float(prev))


def _collect_test_latents_labels(
    model: BarlowTwinsModule,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    *,
    max_samples_total: int | None = None,
    apply_random_rotations: bool = False,
    rotation_seed_base: int | None = None,
    progress_every_batches: int = 25,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, bool]:
    latents: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    labels_available = True
    collected = 0
    max_samples = None if max_samples_total is None else max(1, int(max_samples_total))
    every = max(1, int(progress_every_batches))

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            pc, class_id = _extract_pc_and_class_id(batch)
            if not torch.is_tensor(pc):
                pc = torch.as_tensor(pc)
            pc = pc.to(device=device, dtype=model.dtype, non_blocking=True)
            if apply_random_rotations:
                seed = None
                if rotation_seed_base is not None:
                    seed = int(rotation_seed_base) + int(batch_idx)
                rots = _random_rotation_matrices(
                    int(pc.shape[0]),
                    pc.device,
                    pc.dtype,
                    seed=seed,
                )
                pc = apply_rotation(pc, rots)
            if hasattr(model, "_prepare_model_input"):
                pc = model._prepare_model_input(pc)

            z, _, _ = model(pc)
            z_cpu = z.detach().to(dtype=torch.float32).cpu()
            batch_size = int(z_cpu.shape[0])
            if class_id is None:
                labels_available = False
                class_id_cpu = torch.zeros((batch_size,), dtype=torch.long)
            else:
                if not torch.is_tensor(class_id):
                    class_id = torch.as_tensor(class_id)
                class_id = class_id.detach().view(-1)
                if class_id.numel() == 1 and batch_size > 1:
                    class_id = class_id.expand(batch_size)
                class_id_cpu = class_id.to(dtype=torch.long).cpu()

            take = min(int(z_cpu.shape[0]), int(class_id_cpu.shape[0]))
            if max_samples is not None:
                take = min(take, max_samples - collected)
            if take <= 0:
                break

            latents.append(z_cpu[:take])
            labels.append(class_id_cpu[:take])

            collected += int(take)
            if verbose and (batch_idx + 1) % every == 0:
                print(
                    "[analysis][test] "
                    f"batch={batch_idx + 1} samples={collected}"
                    + (f"/{max_samples}" if max_samples is not None else "")
                )
            if max_samples is not None and collected >= max_samples:
                break

    lat_arr = torch.cat(latents, dim=0).numpy() if latents else np.empty((0,), dtype=np.float32)
    if labels_available:
        label_arr = torch.cat(labels, dim=0).numpy() if labels else np.empty((0,), dtype=np.int64)
    else:
        label_arr = np.zeros((int(lat_arr.shape[0]),), dtype=np.int64)
    return lat_arr, label_arr, labels_available


def _resolve_test_metric_settings(cfg: DictConfig) -> Dict[str, Any]:
    test_cluster_acc_methods = _as_list_of_str(
        getattr(cfg, "test_cluster_acc_methods", ["kmeans", "kmeans++", "gmm_full"])
    ) or ["kmeans", "kmeans++", "gmm_full"]
    test_cluster_acc_runs = int(getattr(cfg, "test_cluster_acc_runs", 1))
    test_cluster_acc_runs_by_method = _as_int_mapping(getattr(cfg, "test_cluster_acc_runs_by_method", {}))
    cluster_acc_seed = int(getattr(cfg, "cluster_acc_seed", 0))

    test_max_samples = _positive_int_or_none(getattr(cfg, "analysis_test_max_samples", None))
    if test_max_samples is None:
        test_max_samples = _positive_int_or_none(getattr(cfg, "max_test_samples", None))

    rotation_runs = int(getattr(cfg, "analysis_test_rotation_runs", 5))
    rotation_seed = int(getattr(cfg, "analysis_test_rotation_seed", 12345))

    return {
        "acc_eval_methods": test_cluster_acc_methods,
        "acc_eval_runs": test_cluster_acc_runs,
        "acc_eval_runs_by_method": test_cluster_acc_runs_by_method,
        "acc_random_seed": cluster_acc_seed,
        "max_samples": test_max_samples,
        "rotation_runs": rotation_runs,
        "rotation_seed": rotation_seed,
    }


def _to_finite_float(value: Any) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if np.isfinite(val) else None


def _primary_accuracy_key(cluster_metrics: dict[str, Any], eval_k: int | None) -> str | None:
    preferred = f"ACC_KMEANS_HUNGARIAN_K{eval_k}"
    if preferred in cluster_metrics:
        return preferred
    for key in sorted(cluster_metrics.keys()):
        if (
            key.startswith("ACC_")
            and not key.endswith("_MEAN")
            and not key.endswith("_STD")
            and not key.endswith("_BEST")
            and not key.endswith("_RUNS")
        ):
            return key
    return None


def _aggregate_metric(values: list[float | None]) -> dict[str, Any]:
    valid = [float(v) for v in values if _to_finite_float(v) is not None]
    if not valid:
        return {"mean": None, "std": None, "min": None, "max": None, "n_valid": 0}
    arr = np.asarray(valid, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_valid": int(arr.size),
    }


def _infer_label_cluster_count(labels: np.ndarray) -> int | None:
    labels_arr = np.asarray(labels).reshape(-1)
    if labels_arr.size == 0:
        return None
    k = int(np.unique(labels_arr).size)
    return k if k > 1 else None


def _resolve_hungarian_eval_k_from_labels(
    labels: np.ndarray,
) -> int | None:
    inferred_k = _infer_label_cluster_count(labels)
    return inferred_k


def _evaluate_test_run(
    model: BarlowTwinsModule,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    settings: dict[str, Any],
    *,
    progress_every_batches: int,
    apply_random_rotations: bool,
    rotation_seed_base: int | None = None,
    include_embedding_metrics: bool = False,
) -> dict[str, Any]:
    latents, labels, labels_available = _collect_test_latents_labels(
        model,
        dataloader,
        device,
        max_samples_total=settings["max_samples"],
        apply_random_rotations=apply_random_rotations,
        rotation_seed_base=rotation_seed_base,
        progress_every_batches=progress_every_batches,
        verbose=True,
    )
    cluster_metrics = {}
    resolved_eval_k = None
    if labels_available:
        resolved_eval_k = _resolve_hungarian_eval_k_from_labels(labels)
        cluster_metrics = compute_cluster_metrics(
            latents,
            labels,
            stage="test",
            hungarian_eval_k=resolved_eval_k,
            acc_eval_methods=settings["acc_eval_methods"],
            acc_eval_runs=settings["acc_eval_runs"],
            acc_eval_runs_by_method=settings["acc_eval_runs_by_method"],
            acc_random_seed=settings["acc_random_seed"],
        )
        if cluster_metrics is None:
            cluster_metrics = {}
    embedding_metrics = {}
    if include_embedding_metrics:
        embedding_labels = labels if labels_available else np.zeros((int(latents.shape[0]),), dtype=np.int64)
        embedding_metrics = compute_embedding_quality_metrics(
            latents,
            embedding_labels,
            include_expensive=True,
        )
        if embedding_metrics is None:
            embedding_metrics = {}

    acc_key = _primary_accuracy_key(cluster_metrics, resolved_eval_k)
    result = {
        "num_samples": int(latents.shape[0]),
        "labels_available": bool(labels_available),
        "accuracy": _to_finite_float(cluster_metrics.get(acc_key)),
        "nmi": _to_finite_float(cluster_metrics.get("NMI")),
        "ari": _to_finite_float(cluster_metrics.get("ARI")),
        "cluster_metrics": cluster_metrics,
        "hungarian_eval_k": resolved_eval_k,
    }
    if not labels_available:
        result["label_metrics_skipped_reason"] = "Batch does not provide class_id labels."
    result["accuracy_key"] = acc_key
    if embedding_metrics:
        result["embedding_metrics"] = embedding_metrics
    return result


def _run_test_phase_metrics(
    model: BarlowTwinsModule,
    dm: Any,
    cfg: DictConfig,
    device: str,
    *,
    progress_every_batches: int = 25,
) -> Dict[str, Any]:
    settings = _resolve_test_metric_settings(cfg)
    test_dl = dm.test_dataloader()
    with _temporary_disable_dataset_aug(test_dl):
        canonical_result = _evaluate_test_run(
            model,
            test_dl,
            device,
            settings,
            progress_every_batches=progress_every_batches,
            apply_random_rotations=False,
            include_embedding_metrics=True,
        )
        multi_rotation_runs = []
        for run_idx in range(settings["rotation_runs"]):
            run_result = _evaluate_test_run(
                model,
                test_dl,
                device,
                settings,
                progress_every_batches=progress_every_batches,
                apply_random_rotations=True,
                rotation_seed_base=settings["rotation_seed"] + run_idx * 10000,
            )
            run_result["rotation_run"] = int(run_idx + 1)
            multi_rotation_runs.append(run_result)

    return {
        "settings": settings,
        "canonical": canonical_result,
        "multi_rotation": {
            "num_rotations": int(settings["rotation_runs"]),
            "accuracy": _aggregate_metric([run.get("accuracy") for run in multi_rotation_runs]),
            "nmi": _aggregate_metric([run.get("nmi") for run in multi_rotation_runs]),
            "ari": _aggregate_metric([run.get("ari") for run in multi_rotation_runs]),
            "runs": multi_rotation_runs,
        },
    }


def _build_inference_cache_spec(
    *,
    checkpoint_path: str,
    cfg: DictConfig,
    max_batches_latent: int | None,
    max_samples_total: int | None,
    seed_base: int,
) -> dict[str, Any]:
    return {
        "version": 1,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "data_kind": str(getattr(cfg.data, "kind", "unknown")),
        "data_path": str(getattr(cfg.data, "data_path", "")),
        "data_files": _as_list_of_str(getattr(cfg.data, "data_files", None)) or [],
        "batch_size": int(getattr(cfg, "batch_size", 0)),
        "max_batches_latent": None if max_batches_latent is None else int(max_batches_latent),
        "max_samples_total": None if max_samples_total is None else int(max_samples_total),
        "seed_base": int(seed_base),
        "collect_coords": True,
    }


def _inference_cache_spec_hash(spec: dict[str, Any]) -> str:
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _inference_cache_paths(out_dir: Path, cache_filename: str) -> tuple[Path, Path]:
    npz_path = Path(out_dir) / cache_filename
    meta_path = npz_path.with_suffix(npz_path.suffix + ".meta.json")
    return npz_path, meta_path


def _validate_inference_cache_arrays(cache: dict[str, np.ndarray]) -> None:
    required = ("inv_latents", "eq_latents", "phases", "coords", "instance_ids")
    missing = [key for key in required if key not in cache]
    if missing:
        raise ValueError(
            "Inference cache is missing required arrays: "
            f"{missing}. Required keys: {list(required)}."
        )
    for key in required:
        if not isinstance(cache[key], np.ndarray):
            raise TypeError(
                f"Inference cache field '{key}' must be np.ndarray, got {type(cache[key])!r}."
            )

    inv = cache["inv_latents"]
    n = int(inv.shape[0]) if inv.ndim >= 1 else 0
    eq = cache["eq_latents"]
    if eq.size > 0 and (eq.ndim < 1 or int(eq.shape[0]) != n):
        raise ValueError(
            "eq_latents has incompatible first dimension: "
            f"expected {n}, got shape {eq.shape}."
        )
    phases = cache["phases"].reshape(-1)
    if phases.size not in (0, n):
        raise ValueError(
            f"phases must have size 0 or {n}, got {phases.size}."
        )
    coords = cache["coords"]
    if coords.size > 0:
        if coords.ndim != 2 or coords.shape[1] < 3:
            raise ValueError(
                f"coords must have shape (N, >=3) or be empty, got {coords.shape}."
            )
        if int(coords.shape[0]) != n:
            raise ValueError(
                f"coords first dimension mismatch: expected {n}, got {coords.shape[0]}."
            )
    instance_ids = cache["instance_ids"].reshape(-1)
    if instance_ids.size not in (0, n):
        raise ValueError(
            f"instance_ids must have size 0 or {n}, got {instance_ids.size}."
        )


def _load_inference_cache(
    *,
    out_dir: Path,
    cache_filename: str,
    expected_spec: dict[str, Any],
) -> tuple[dict[str, np.ndarray] | None, str]:
    npz_path, meta_path = _inference_cache_paths(out_dir, cache_filename)
    if not npz_path.exists():
        return None, f"cache file does not exist: {npz_path}"
    if not meta_path.exists():
        return None, f"cache metadata does not exist: {meta_path}"

    with meta_path.open("r") as handle:
        meta = json.load(handle)
    if not isinstance(meta, dict):
        raise ValueError(f"Cache metadata at {meta_path} must be a JSON object, got {type(meta)!r}.")

    expected_hash = _inference_cache_spec_hash(expected_spec)
    cached_hash = str(meta.get("spec_sha256", ""))
    if cached_hash != expected_hash:
        return None, (
            "cache spec mismatch: "
            f"expected sha256={expected_hash}, found sha256={cached_hash}"
        )

    with np.load(npz_path) as data:
        cache = {key: np.asarray(data[key]) for key in data.files}
    _validate_inference_cache_arrays(cache)
    return cache, f"loaded cache from {npz_path}"


def _save_inference_cache(
    *,
    out_dir: Path,
    cache_filename: str,
    cache: dict[str, np.ndarray],
    spec: dict[str, Any],
) -> None:
    _validate_inference_cache_arrays(cache)
    npz_path, meta_path = _inference_cache_paths(out_dir, cache_filename)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        npz_path,
        inv_latents=cache["inv_latents"],
        eq_latents=cache["eq_latents"],
        phases=cache["phases"],
        coords=cache["coords"],
        instance_ids=cache["instance_ids"],
    )
    meta = {
        "spec": spec,
        "spec_sha256": _inference_cache_spec_hash(spec),
        "num_samples": int(cache["inv_latents"].shape[0]) if cache["inv_latents"].ndim >= 1 else 0,
    }
    with meta_path.open("w") as handle:
        json.dump(meta, handle, indent=2)


def load_barlow_model(
    checkpoint_path: str, cuda_device: int = 0, cfg: DictConfig | None = None
) -> Tuple[BarlowTwinsModule, DictConfig, str]:
    """Restore the Barlow Twins module together with its Hydra cfg and device string."""
    if cfg is None:
        config_dir, config_name = resolve_config_path(checkpoint_path)
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parents[2]
        absolute_config_dir = project_root / config_dir
        relative_config_dir = os.path.relpath(absolute_config_dir, current_dir)
        with initialize(version_base=None, config_path=relative_config_dir):
            cfg = compose(config_name=config_name)

    device = f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu"
    model: BarlowTwinsModule = load_model_from_checkpoint(
        checkpoint_path, cfg, device=device, module=BarlowTwinsModule
    )
    model.to(device).eval()
    return model, cfg, device


def build_datamodule(cfg: DictConfig):
    """Instantiate and setup the matching datamodule."""
    if getattr(cfg, "data", None) is None:
        raise ValueError("Config missing data section")
    if getattr(cfg.data, "kind", None) == "synthetic":
        dm = SyntheticPointCloudDataModule(cfg)
    else:
        dm = RealPointCloudDataModule(cfg)
    dm.setup(stage="test")
    return dm


def run_post_training_analysis(
    checkpoint_path: str,
    output_dir: str,
    cuda_device: int = 0,
    cfg: DictConfig | None = None,
    max_batches_latent: int | None = None,
    max_samples_visualization: int | None = None,
    data_files_override: list[str] | None = None,
    test_rotation_runs: int | None = None,
    test_max_samples: int | None = None,
) -> Dict[str, Any]:
    """Generate qualitative and quantitative diagnostics for Barlow Twins."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    step_idx = [0]

    def _step(msg: str) -> None:
        step_idx[0] += 1
        elapsed = time.perf_counter() - t0
        print(f"[analysis][step {step_idx[0]}][{elapsed:7.1f}s] {msg}")

    _step("Loading model")
    model, cfg, device = load_barlow_model(checkpoint_path, cuda_device=cuda_device, cfg=cfg)

    if test_rotation_runs is not None:
        cfg.analysis_test_rotation_runs = int(test_rotation_runs)
    if test_max_samples is not None:
        cfg.analysis_test_max_samples = int(test_max_samples)

    def _resolve_analysis_files() -> list[str] | None:
        if cfg.data.kind != "real":
            return None
        if data_files_override:
            return data_files_override
        files = _as_list_of_str(cfg.data.analysis_data_files)
        if files:
            return files
        if cfg.data.analysis_data_file:
            return [cfg.data.analysis_data_file]
        data_files = _as_list_of_str(cfg.data.data_files)
        analysis_single = bool(cfg.data.analysis_single_timestep)
        if not analysis_single:
            return data_files
        mid_idx = len(data_files) // 2
        return [data_files[mid_idx]]

    analysis_files = _resolve_analysis_files()
    if analysis_files is not None:
        cfg.data.data_files = analysis_files
        print(f"Analysis data_files: {analysis_files}")

    # Analysis uses a fixed worker count to keep behavior consistent across runs.
    analysis_num_workers = 4
    cfg.num_workers = analysis_num_workers
    print(f"Analysis dataloader workers: {analysis_num_workers}")

    tsne_max_samples = int(getattr(cfg, "analysis_tsne_max_samples", 8000))
    if max_samples_visualization is not None:
        tsne_max_samples = min(tsne_max_samples, max_samples_visualization)
    clustering_max_samples = int(getattr(cfg, "analysis_clustering_max_samples", 20000))
    cluster_method = str(getattr(cfg, "analysis_cluster_method", "auto")).lower()
    cluster_l2_normalize = bool(getattr(cfg, "analysis_cluster_l2_normalize", True))
    cluster_standardize = bool(getattr(cfg, "analysis_cluster_standardize", True))
    cluster_pca_var = float(getattr(cfg, "analysis_cluster_pca_var", 0.98))
    cluster_pca_max_components = int(
        getattr(cfg, "analysis_cluster_pca_max_components", 32)
    )
    cluster_k_values_cfg = _as_list_of_int(
        getattr(cfg, "analysis_cluster_k_values", None),
        field_name="analysis_cluster_k_values",
    )
    cluster_k_values = cluster_k_values_cfg or [3, 4, 5, 6]
    cluster_k_values = [int(k) for k in cluster_k_values if int(k) >= 2]
    cluster_k_values = list(dict.fromkeys(cluster_k_values))
    if not cluster_k_values:
        cluster_k_values = [3, 4, 5, 6]
    md_use_all_points = bool(getattr(cfg, "analysis_md_use_all_points", True))
    hdbscan_enabled = bool(getattr(cfg, "analysis_hdbscan_enabled", True))
    hdbscan_fit_fraction = float(getattr(cfg, "analysis_hdbscan_fit_fraction", 0.25))
    hdbscan_max_fit_samples = int(getattr(cfg, "analysis_hdbscan_max_fit_samples", 50000))
    hdbscan_target_k_min = int(getattr(cfg, "analysis_hdbscan_target_k_min", 5))
    hdbscan_target_k_max = int(getattr(cfg, "analysis_hdbscan_target_k_max", 6))
    hdbscan_min_samples = getattr(cfg, "analysis_hdbscan_min_samples", None)
    if hdbscan_min_samples is not None:
        hdbscan_min_samples = int(hdbscan_min_samples)
    hdbscan_cluster_selection_epsilon = float(
        getattr(cfg, "analysis_hdbscan_cluster_selection_epsilon", 0.0)
    )
    hdbscan_cluster_selection_method = str(
        getattr(cfg, "analysis_hdbscan_cluster_selection_method", "leaf")
    ).lower()
    _hdbscan_mcs_raw = getattr(cfg, "analysis_hdbscan_min_cluster_size_candidates", None)
    hdbscan_min_cluster_size_candidates = (
        [int(v) for v in _hdbscan_mcs_raw]
        if _hdbscan_mcs_raw is not None
        else None
    )
    progress_every_batches = int(getattr(cfg, "analysis_progress_every_batches", 25))
    cluster_profile_enabled = bool(getattr(cfg, "analysis_cluster_profile_enabled", True))
    cluster_profile_samples_per_cluster = max(
        1,
        int(getattr(cfg, "analysis_cluster_profile_samples_per_cluster", 8)),
    )
    profile_points_default = int(
        getattr(
            cfg.data,
            "model_points",
            getattr(cfg.data, "num_points", 48),
        )
    )
    cluster_profile_target_points = max(8, profile_points_default)
    cluster_profile_knn_k = max(
        1,
        int(getattr(cfg, "analysis_cluster_profile_knn_k", 3)),
    )
    cluster_profile_max_property_samples = max(
        1,
        int(getattr(cfg, "analysis_cluster_profile_max_property_samples", 256)),
    )
    cluster_figure_set_enabled = bool(
        getattr(cfg, "analysis_cluster_figure_set_enabled", True)
    )
    cluster_figure_k = max(2, int(getattr(cfg, "analysis_cluster_figure_set_k", 6)))
    cluster_figure_min_crystal_clusters = max(
        1,
        int(getattr(cfg, "analysis_cluster_figure_min_crystal_clusters", 2)),
    )
    cluster_figure_crystal_score_quantile = float(
        getattr(
            cfg,
            "analysis_cluster_figure_crystal_score_quantile",
            getattr(cfg, "analysis_cluster_figure_crystal_min_fraction", 0.6),
        )
    )
    cluster_figure_crystal_score_samples = max(
        1,
        int(getattr(cfg, "analysis_cluster_figure_crystal_score_samples", 64)),
    )
    cluster_figure_crystal_score_knn = max(
        2,
        int(getattr(cfg, "analysis_cluster_figure_crystal_score_knn", 6)),
    )
    cluster_figure_md_max_points = _positive_int_or_none(
        getattr(cfg, "analysis_cluster_figure_md_max_points", None)
    )
    cluster_figure_md_point_size = float(
        getattr(cfg, "analysis_cluster_figure_md_point_size", 4.0)
    )
    cluster_figure_md_alpha = float(
        getattr(cfg, "analysis_cluster_figure_md_alpha", 0.96)
    )
    cluster_figure_md_halo_scale = float(
        getattr(cfg, "analysis_cluster_figure_md_halo_scale", 5.0)
    )
    cluster_figure_md_halo_alpha = float(
        getattr(cfg, "analysis_cluster_figure_md_halo_alpha", 0.26)
    )
    cluster_figure_md_view_elev = float(
        getattr(cfg, "analysis_cluster_figure_md_view_elev", 24.0)
    )
    cluster_figure_md_view_azim = float(
        getattr(cfg, "analysis_cluster_figure_md_view_azim", 35.0)
    )
    cluster_figure_combo_min_size = max(
        1,
        int(getattr(cfg, "analysis_cluster_figure_combo_min_size", 1)),
    )
    cluster_figure_combo_max_size = max(
        cluster_figure_combo_min_size,
        int(getattr(cfg, "analysis_cluster_figure_combo_max_size", 3)),
    )
    cluster_figure_combo_max_outputs = max(
        1,
        int(getattr(cfg, "analysis_cluster_figure_combo_max_outputs", 40)),
    )
    if cluster_figure_md_point_size <= 0.0:
        raise ValueError(
            f"analysis_cluster_figure_md_point_size must be > 0, got {cluster_figure_md_point_size}."
        )
    if not (0.0 <= cluster_figure_md_alpha <= 1.0):
        raise ValueError(
            f"analysis_cluster_figure_md_alpha must be in [0, 1], got {cluster_figure_md_alpha}."
        )
    if cluster_figure_md_halo_scale <= 0.0:
        raise ValueError(
            f"analysis_cluster_figure_md_halo_scale must be > 0, got {cluster_figure_md_halo_scale}."
        )
    if not (0.0 <= cluster_figure_md_halo_alpha <= 1.0):
        raise ValueError(
            "analysis_cluster_figure_md_halo_alpha must be in [0, 1], "
            f"got {cluster_figure_md_halo_alpha}."
        )
    if not np.isfinite(cluster_figure_md_view_elev) or not np.isfinite(cluster_figure_md_view_azim):
        raise ValueError(
            "analysis_cluster_figure_md_view_elev and analysis_cluster_figure_md_view_azim "
            f"must be finite, got elev={cluster_figure_md_view_elev}, "
            f"azim={cluster_figure_md_view_azim}."
        )
    cluster_figure_icl_k_min = int(
        getattr(cfg, "analysis_cluster_figure_icl_k_min", 2)
    )
    cluster_figure_icl_k_max = int(
        getattr(cfg, "analysis_cluster_figure_icl_k_max", 20)
    )
    cluster_figure_icl_max_samples = _positive_int_or_none(
        getattr(cfg, "analysis_cluster_figure_icl_max_samples", 20000)
    )
    cluster_figure_icl_covariance = str(
        getattr(cfg, "analysis_cluster_figure_icl_covariance", "diag")
    ).lower()
    cluster_figure_representative_points = max(
        16,
        int(
            getattr(
                cfg,
                "analysis_cluster_figure_representative_points",
                profile_points_default,
            )
        ),
    )
    inference_cache_enabled = bool(
        getattr(cfg, "analysis_inference_cache_enabled", True)
    )
    inference_cache_force_recompute = bool(
        getattr(cfg, "analysis_inference_cache_force_recompute", False)
    )
    inference_cache_file = str(
        getattr(cfg, "analysis_inference_cache_file", "analysis_inference_cache.npz")
    ).strip()
    if inference_cache_file == "":
        raise ValueError("analysis_inference_cache_file must be a non-empty file name.")
    print(f"t-SNE sample cap: {tsne_max_samples}")
    print(f"Clustering metrics cap: {clustering_max_samples}")
    print(f"Clustering k values (configured): {cluster_k_values}")
    print(
        "Fixed-k figure set: "
        f"enabled={cluster_figure_set_enabled}, "
        f"k={cluster_figure_k}, "
        f"min_crystal_clusters={cluster_figure_min_crystal_clusters}, "
        f"score_quantile={cluster_figure_crystal_score_quantile:.2f}, "
        f"combo_size=[{cluster_figure_combo_min_size},{cluster_figure_combo_max_size}], "
        f"combo_max_outputs={cluster_figure_combo_max_outputs}"
    )
    _step("Building datamodule")
    dm = build_datamodule(cfg)
    is_synthetic = getattr(cfg.data, "kind", None) == "synthetic"

    dm.setup(stage="fit")
    all_metrics: Dict[str, Any] = {}

    if is_synthetic:
        _step("Running test-phase metrics")
        test_phase_metrics = _run_test_phase_metrics(
            model,
            dm,
            cfg,
            device,
            progress_every_batches=progress_every_batches,
        )
        all_metrics["test_phase"] = test_phase_metrics
    else:
        print(
            "Skipping test-phase metrics for real data: dataset batches do not include class_id labels."
        )

    print("Using ALL dataset splits (train + test) for latent analysis")

    if is_synthetic:
        train_dataset = getattr(dm, "train_dataset", None)
        test_dataset = getattr(dm, "test_dataset", None)
        if train_dataset is None or test_dataset is None:
            raise ValueError("Synthetic datamodule is missing train/test datasets.")
        combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        dl = torch.utils.data.DataLoader(
            combined_dataset,
            batch_size=cfg.batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True,
        )
    else:
        dl = build_real_coords_dataloader(cfg, dm, use_train_data=True, use_full_dataset=True)
        print(
            "Real data detected: using full dataset for local-structure clustering visualization"
        )

    class_names = _extract_class_names(dm.train_dataset)
    print(f"Loaded class names: {class_names}")

    if max_batches_latent is None:
        max_batches_latent = _positive_int_or_none(getattr(cfg, "analysis_max_batches_latent", None))
    max_samples_total = getattr(cfg, "analysis_max_samples_total", None)
    if max_samples_total is None and not is_synthetic:
        max_samples_total = 20000
    if not is_synthetic and md_use_all_points:
        max_samples_total = None
    max_samples_total = _positive_int_or_none(max_samples_total)

    seed_base = getattr(cfg, "analysis_seed_base", 123)
    cache_spec = _build_inference_cache_spec(
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        max_batches_latent=max_batches_latent,
        max_samples_total=max_samples_total,
        seed_base=int(seed_base),
    )

    _step("Collecting inference batches")
    cache: dict[str, np.ndarray] | None = None
    cache_loaded = False
    if inference_cache_enabled and not inference_cache_force_recompute:
        cache, cache_msg = _load_inference_cache(
            out_dir=out_dir,
            cache_filename=inference_cache_file,
            expected_spec=cache_spec,
        )
        cache_loaded = cache is not None
        print(f"[analysis][cache] {cache_msg}")
    elif inference_cache_enabled and inference_cache_force_recompute:
        print("[analysis][cache] Forced recompute requested; skipping cache load.")

    if cache is None:
        if max_batches_latent is None:
            print("Gathering inference batches (ALL batches)...")
        else:
            print(f"Gathering inference batches (up to {max_batches_latent} batches)...")
        if max_samples_total is not None:
            print(f"Collecting up to {max_samples_total} samples for analysis")
        cache = gather_inference_batches(
            model,
            dl,
            device,
            max_batches=max_batches_latent,
            max_samples_total=max_samples_total,
            collect_coords=True,
            seed_base=seed_base,
            progress_every_batches=progress_every_batches,
            verbose=True,
        )
        if inference_cache_enabled:
            _save_inference_cache(
                out_dir=out_dir,
                cache_filename=inference_cache_file,
                cache=cache,
                spec=cache_spec,
            )
            cache_npz, _ = _inference_cache_paths(out_dir, inference_cache_file)
            print(f"[analysis][cache] Saved inference cache: {cache_npz}")

    n_samples = len(cache["inv_latents"])
    print(f"Collected {n_samples} samples for analysis")
    has_phases = cache["phases"].size == n_samples
    all_metrics["inference_cache"] = {
        "enabled": bool(inference_cache_enabled),
        "file": str((out_dir / inference_cache_file)),
        "loaded_from_cache": bool(cache_loaded),
        "force_recompute": bool(inference_cache_force_recompute),
        "spec_sha256": _inference_cache_spec_hash(cache_spec),
    }

    if is_synthetic:
        _step("Computing t-SNE visualization")
        save_latent_tsne(
            cache["inv_latents"],
            cache["phases"],
            out_dir,
            max_samples=tsne_max_samples,
            class_names=class_names,
        )

    _step("Computing PCA analysis")
    pca_stats = save_pca_visualization(
        cache["inv_latents"],
        cache["phases"],
        out_dir,
        max_samples=None,
        class_names=class_names,
    )
    all_metrics["pca"] = pca_stats

    _step("Computing latent statistics")
    latent_stats = save_latent_statistics(
        cache["inv_latents"],
        cache["eq_latents"],
        cache["phases"],
        out_dir,
        class_names=class_names,
    )
    all_metrics["latent_stats"] = latent_stats

    _step("Computing clustering analysis")
    clustering_metrics = save_clustering_analysis(
        cache["inv_latents"],
        cache["phases"],
        out_dir,
        max_samples=clustering_max_samples,
        class_names=class_names,
        cluster_method=cluster_method,
        l2_normalize=cluster_l2_normalize,
        standardize=cluster_standardize,
        pca_variance=cluster_pca_var,
        pca_max_components=cluster_pca_max_components,
        k_values=cluster_k_values,
    )
    all_metrics["clustering"] = clustering_metrics

    coords = cache["coords"]

    md_metrics_key = "synthetic_md" if is_synthetic else "real_md"
    num_latents = len(cache["inv_latents"])
    configured_k_values = _as_list_of_int(
        clustering_metrics.get("k_values_evaluated", None),
        field_name="clustering.k_values_evaluated",
    ) or []
    if not configured_k_values:
        configured_k_values = [int(k) for k in cluster_k_values]
    configured_k_values = [max(2, min(int(k), num_latents)) for k in configured_k_values]
    configured_k_values = list(dict.fromkeys(configured_k_values))
    if not configured_k_values:
        configured_k_values = [max(2, min(num_latents, 3))]

    selected_method_by_k_cfg: Dict[int, str] = {
        int(key): str(value).lower()
        for key, value in clustering_metrics.get("selected_method_by_k", {}).items()
    }

    def _compute_labels_for_k(k_value: int, *, method_name: str):
        return compute_kmeans_labels(
            cache["inv_latents"],
            k_value,
            method=method_name,
            l2_normalize=cluster_l2_normalize,
            standardize=cluster_standardize,
            pca_variance=cluster_pca_var,
            pca_max_components=cluster_pca_max_components,
            return_info=True,
        )

    cluster_labels_by_k: Dict[int, np.ndarray] = {}
    cluster_methods_by_k: Dict[int, str] = {}
    for k_val in configured_k_values:
        method_for_k = selected_method_by_k_cfg.get(int(k_val), cluster_method)
        labels_k, fit_info_k = _compute_labels_for_k(
            int(k_val),
            method_name=method_for_k,
        )
        cluster_labels_by_k[int(k_val)] = labels_k
        cluster_methods_by_k[int(k_val)] = str(fit_info_k.get("method", method_for_k))

    if cluster_figure_set_enabled:
        if cluster_figure_k > num_latents:
            raise ValueError(
                "analysis_cluster_figure_set_k is larger than the number of collected samples: "
                f"k={cluster_figure_k}, samples={num_latents}. "
                "Reduce analysis_cluster_figure_set_k or collect more samples."
            )
        if cluster_figure_k not in cluster_labels_by_k:
            method_for_fig = selected_method_by_k_cfg.get(int(cluster_figure_k), cluster_method)
            fig_labels, fig_fit_info = _compute_labels_for_k(
                int(cluster_figure_k),
                method_name=method_for_fig,
            )
            cluster_labels_by_k[int(cluster_figure_k)] = fig_labels
            cluster_methods_by_k[int(cluster_figure_k)] = str(
                fig_fit_info.get("method", method_for_fig)
            )

    primary_k = int(configured_k_values[0])
    cluster_labels = cluster_labels_by_k[primary_k]
    cluster_label_method = cluster_methods_by_k.get(primary_k, cluster_method)
    all_metrics["clustering"]["labels_k_method"] = cluster_label_method
    all_metrics["clustering"]["labels_method_by_k"] = {
        int(k): cluster_methods_by_k[int(k)] for k in configured_k_values
    }
    all_metrics["clustering"]["primary_k"] = int(primary_k)
    all_metrics["clustering"]["k_values_used"] = [int(k) for k in configured_k_values]
    point_scale = resolve_point_scale(cfg)

    if cluster_figure_set_enabled:
        _step("Generating fixed-k cluster figure set")
        figure_set_dir = out_dir / f"cluster_figure_set_k{cluster_figure_k}"
        with _temporary_disable_dataset_aug(dl):
            figure_set_metrics = _save_fixed_k_cluster_figure_set(
                out_dir=figure_set_dir,
                dataset=getattr(dl, "dataset", None),
                latents=cache["inv_latents"],
                coords=coords,
                cluster_labels=cluster_labels_by_k[int(cluster_figure_k)],
                k_value=int(cluster_figure_k),
                point_scale=point_scale,
                l2_normalize=cluster_l2_normalize,
                standardize=cluster_standardize,
                pca_variance=cluster_pca_var,
                pca_max_components=cluster_pca_max_components,
                min_crystal_clusters=cluster_figure_min_crystal_clusters,
                crystal_score_quantile=cluster_figure_crystal_score_quantile,
                crystal_score_samples_per_cluster=cluster_figure_crystal_score_samples,
                crystal_score_knn_k=cluster_figure_crystal_score_knn,
                md_max_points=cluster_figure_md_max_points,
                icl_k_min=cluster_figure_icl_k_min,
                icl_k_max=cluster_figure_icl_k_max,
                icl_max_samples=cluster_figure_icl_max_samples,
                icl_covariance_type=cluster_figure_icl_covariance,
                representative_points=cluster_figure_representative_points,
                md_point_size=cluster_figure_md_point_size,
                md_point_alpha=cluster_figure_md_alpha,
                md_halo_scale=cluster_figure_md_halo_scale,
                md_halo_alpha=cluster_figure_md_halo_alpha,
                md_view_elev=cluster_figure_md_view_elev,
                md_view_azim=cluster_figure_md_view_azim,
                combo_min_size=cluster_figure_combo_min_size,
                combo_max_size=cluster_figure_combo_max_size,
                combo_max_outputs=cluster_figure_combo_max_outputs,
            )
        all_metrics["cluster_figure_set"] = figure_set_metrics

    _step("Computing t-SNE visualization (clusters)")
    tsne_n_iter = int(getattr(cfg, "analysis_tsne_n_iter", 1000))
    tsne_idx = _sample_indices(len(cache["inv_latents"]), tsne_max_samples)
    tsne_latents = cache["inv_latents"][tsne_idx]
    tsne_perplexity = min(50, max(5, len(tsne_latents) // 100))
    tsne_coords = compute_tsne(tsne_latents, perplexity=tsne_perplexity, n_iter=tsne_n_iter)

    for idx_k, k_val in enumerate(configured_k_values):
        labels_k = cluster_labels_by_k[int(k_val)]
        method_k = cluster_methods_by_k.get(int(k_val), cluster_method)
        out_name = "latent_tsne_clusters.png" if idx_k == 0 else f"latent_tsne_clusters_k{k_val}.png"
        save_tsne_plot_with_coords(
            tsne_coords,
            labels_k[tsne_idx],
            out_dir,
            out_name=out_name,
            title=f"Latent space t-SNE ({method_k}, k={k_val})",
        )

        if idx_k == 0:
            _step("Saving coordinate-space clustering outputs")
            interactive_max_points = _positive_int_or_none(
                getattr(cfg, "analysis_interactive_max_points", None)
            )
            if md_use_all_points:
                interactive_max_points = None
            coord_files = save_local_structure_assignments(
                coords,
                cluster_labels,
                out_dir,
            )
            if coord_files:
                save_md_space_clusters_plot(
                    coords,
                    cluster_labels,
                    out_dir / "md_space_clusters.png",
                    max_points=interactive_max_points,
                )
                interactive_path = None
                interactive_paths: Dict[int, str] = {}
                try:
                    for inner_idx, inner_k in enumerate(configured_k_values):
                        labels_k = cluster_labels_by_k[int(inner_k)]
                        out_path = (
                            out_dir / "md_space_clusters.html"
                            if inner_idx == 0
                            else out_dir / f"md_space_clusters_k{inner_k}.html"
                        )
                        save_interactive_md_plot(
                            coords,
                            labels_k,
                            out_path,
                            palette="Set3",
                            max_points=interactive_max_points,
                            marker_size=3.0,
                            marker_line_width=0.0,
                            aspect_mode="cube",
                        )
                        if inner_idx == 0:
                            interactive_path = out_path
                        interactive_paths[int(inner_k)] = str(out_path)
                except ImportError:
                    interactive_path = None
                    print("Plotly not installed; skipping interactive MD plot.")

                hdbscan_info: Dict[str, Any] | None = None
                hdbscan_path = None
                hdbscan_coord_files: Dict[str, str] = {}
                if hdbscan_enabled:
                    _step("Running HDBSCAN clustering (sampled fit)")
                    try:
                        hdbscan_labels, hdbscan_info = compute_hdbscan_labels(
                            cache["inv_latents"],
                            sample_fraction=hdbscan_fit_fraction,
                            max_fit_samples=hdbscan_max_fit_samples,
                            random_state=42,
                            l2_normalize=cluster_l2_normalize,
                            standardize=cluster_standardize,
                            pca_variance=cluster_pca_var,
                            pca_max_components=cluster_pca_max_components,
                            target_clusters_min=hdbscan_target_k_min,
                            target_clusters_max=hdbscan_target_k_max,
                            min_cluster_size_candidates=hdbscan_min_cluster_size_candidates,
                            min_samples=hdbscan_min_samples,
                            cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
                            cluster_selection_method=hdbscan_cluster_selection_method,
                            return_info=True,
                        )
                        if hdbscan_labels.size == len(coords):
                            hdbscan_coord_files = save_local_structure_assignments(
                                coords,
                                hdbscan_labels,
                                out_dir,
                                prefix="local_structure_hdbscan",
                            )
                            try:
                                hdbscan_path = out_dir / "md_space_clusters_hdbscan.html"
                                n_hdb_clusters = int(
                                    len(np.unique(hdbscan_labels[hdbscan_labels >= 0]))
                                )
                                save_interactive_md_plot(
                                    coords,
                                    hdbscan_labels,
                                    hdbscan_path,
                                    palette="Set3",
                                    max_points=interactive_max_points,
                                    marker_size=3.0,
                                    marker_line_width=0.0,
                                    title=(
                                        "MD local-structure clusters "
                                        f"(HDBSCAN, n={len(coords)}, k={n_hdb_clusters})"
                                    ),
                                    label_prefix="HDBSCAN",
                                    aspect_mode="cube",
                                )
                            except ImportError:
                                hdbscan_path = None
                                print("Plotly not installed; skipping HDBSCAN interactive MD plot.")
                        else:
                            print(
                                "Warning: HDBSCAN labels do not match coordinate count; "
                                "skipping HDBSCAN MD outputs."
                            )
                    except ImportError:
                        print("HDBSCAN package not installed; skipping HDBSCAN analysis.")

                unique_labels, counts = np.unique(cluster_labels, return_counts=True)
                all_metrics[md_metrics_key] = {
                    "primary_k": int(primary_k),
                    "k_values_used": [int(k) for k in configured_k_values],
                    "n_clusters": int(len(unique_labels)),
                    "cluster_counts": {int(k): int(v) for k, v in zip(unique_labels, counts)},
                    "coords_files": coord_files,
                }
                if interactive_path is not None:
                    all_metrics[md_metrics_key]["interactive_html"] = str(interactive_path)
                if interactive_paths:
                    all_metrics[md_metrics_key]["interactive_htmls"] = interactive_paths
                if hdbscan_info is not None:
                    all_metrics[md_metrics_key]["hdbscan"] = hdbscan_info
                    if hdbscan_coord_files:
                        all_metrics[md_metrics_key]["hdbscan_coords_files"] = hdbscan_coord_files
                    if hdbscan_path is not None:
                        all_metrics[md_metrics_key]["hdbscan_interactive_html"] = str(hdbscan_path)

    if cluster_profile_enabled:
        _step("Generating per-cluster MD sample profiles")
        point_scale = resolve_point_scale(cfg)
        profile_root = out_dir / "cluster_profiles_by_k"
        with _temporary_disable_dataset_aug(dl):
            cluster_profile_summary = generate_cluster_profile_reports(
                out_root=profile_root,
                dataset=getattr(dl, "dataset", None),
                latents=cache["inv_latents"],
                coords=coords,
                cluster_labels_by_k=cluster_labels_by_k,
                cluster_methods_by_k=cluster_methods_by_k,
                samples_per_cluster=cluster_profile_samples_per_cluster,
                target_points=cluster_profile_target_points,
                knn_k=cluster_profile_knn_k,
                max_cluster_property_samples=cluster_profile_max_property_samples,
                point_scale=point_scale,
                random_seed=int(seed_base),
            )
        if md_metrics_key not in all_metrics:
            all_metrics[md_metrics_key] = {}
        all_metrics[md_metrics_key]["cluster_profiles"] = cluster_profile_summary

    _step("Evaluating equivariance")
    eq_metrics, eq_err = evaluate_latent_equivariance(model, dl, device, max_batches=2)
    save_equivariance_plot(eq_err, out_dir / "equivariance.png")
    all_metrics["equivariance"] = eq_metrics

    _step("Writing metrics")
    metrics_path = out_dir / "analysis_metrics.json"
    with metrics_path.open("w") as handle:
        json.dump(all_metrics, handle, indent=2)

    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total samples analyzed: {n_samples}")

    if "test_phase" in all_metrics:
        test_phase = all_metrics["test_phase"]
        canonical = test_phase.get("canonical", {})
        multi_rotation = test_phase.get("multi_rotation", {})
        print(
            "Test canonical: "
            f"ACC={_fmt_metric(canonical.get('accuracy', 'N/A'))}, "
            f"NMI={_fmt_metric(canonical.get('nmi', 'N/A'))}, "
            f"ARI={_fmt_metric(canonical.get('ari', 'N/A'))}"
        )
        print(
            "Test multi-rotation (mean): "
            f"ACC={_fmt_metric(multi_rotation.get('accuracy', {}).get('mean', 'N/A'))}, "
            f"NMI={_fmt_metric(multi_rotation.get('nmi', {}).get('mean', 'N/A'))}, "
            f"ARI={_fmt_metric(multi_rotation.get('ari', {}).get('mean', 'N/A'))}"
        )
        print(f"Test multi-rotation runs: {multi_rotation.get('num_rotations', 'N/A')}")

    if "pca" in all_metrics and all_metrics["pca"]:
        print(f"PCA: {all_metrics['pca'].get('n_components_95_var', 'N/A')} components for 95% variance")

    if "clustering" in all_metrics and all_metrics["clustering"]:
        k_values_summary = all_metrics["clustering"].get(
            "k_values_used",
            all_metrics["clustering"].get("k_values_evaluated", "N/A"),
        )
        print(f"Cluster k values: {k_values_summary}")
        print(f"Primary k: {all_metrics['clustering'].get('primary_k', 'N/A')}")
        if "labels_k_method" in all_metrics["clustering"]:
            print(f"Primary k method: {all_metrics['clustering'].get('labels_k_method')}")
        if "ari_with_gt" in all_metrics["clustering"]:
            print(f"ARI with ground truth: {_fmt_metric(all_metrics['clustering']['ari_with_gt'])}")
            print(f"NMI with ground truth: {_fmt_metric(all_metrics['clustering']['nmi_with_gt'])}")

    if "equivariance" in all_metrics:
        eq = all_metrics["equivariance"]
        print(
            "Equivariant latent error (seeded mean): "
            f"{_fmt_metric(eq.get('eq_latent_rel_error_mean', 'N/A'))}"
        )
        print(
            "Equivariant latent error (seeded median): "
            f"{_fmt_metric(eq.get('eq_latent_rel_error_median', 'N/A'))}"
        )
        if "eq_latent_rel_error_unseeded" in eq:
            print(
                "Equivariant latent error (unseeded mean): "
                f"{_fmt_metric(eq.get('eq_latent_rel_error_unseeded', 'N/A'))}"
            )
        if "eq_latent_rel_error_unseeded_median" in eq:
            print(
                "Equivariant latent error (unseeded median): "
                f"{_fmt_metric(eq.get('eq_latent_rel_error_unseeded_median', 'N/A'))}"
            )
        if "eq_latent_nondeterminism_contribution" in eq:
            print(
                "Non-determinism contribution (unseeded - seeded): "
                f"{_fmt_metric(eq.get('eq_latent_nondeterminism_contribution', 'N/A'))}"
            )

    print("=" * 60)
    print(f"\nSaved all analyses to {out_dir}")
    print(f"Total runtime: {time.perf_counter() - t0:.1f}s")
    print("Generated files:")
    if has_phases:
        print("  - latent_tsne_ground_truth.png: t-SNE with ground truth labels")
    print("  - latent_tsne_clusters.png: t-SNE with clustering labels")
    print("  - latent_pca_analysis.png: PCA projection and variance")
    print("  - latent_pca_3d.png: 3D PCA projection")
    print("  - latent_statistics.png: Comprehensive latent statistics")
    print("  - clustering_analysis.png: Clustering quality metrics")
    print("  - equivariance.png: Equivariant latent error distribution")
    print("  - analysis_metrics.json: All numerical metrics")
    if "cluster_figure_set" in all_metrics:
        k_fig = all_metrics["cluster_figure_set"].get("k_value", "N/A")
        print(
            f"  - cluster_figure_set_k{k_fig}/01_md_clusters_crystal_only_k{k_fig}.png: "
            "MD space (crystal-like clusters only)"
        )
        print(
            f"  - cluster_figure_set_k{k_fig}/02_md_clusters_all_k{k_fig}.png: "
            "MD space (all clusters)"
        )
        print(
            f"  - cluster_figure_set_k{k_fig}/02_md_clusters_all_k{k_fig}_view2.png: "
            "MD space (all clusters, side view 2)"
        )
        print(
            f"  - cluster_figure_set_k{k_fig}/02_md_clusters_all_k{k_fig}_view3.png: "
            "MD space (all clusters, side view 3)"
        )
        print(
            f"  - cluster_figure_set_k{k_fig}/02_md_clusters_all_k{k_fig}_view4.png: "
            "MD space (all clusters, side view 4)"
        )
        print(
            f"  - cluster_figure_set_k{k_fig}/01b_cluster_combinations_k{k_fig}/combo_*.png: "
            "cluster-subset combinations for manual selection"
        )
        print(
            f"  - cluster_figure_set_k{k_fig}/03_cluster_count_icl_k{k_fig}.png: "
            "ICL vs number of clusters"
        )
        print(
            f"  - cluster_figure_set_k{k_fig}/04_cluster_representatives_k{k_fig}.png: "
            "nearest-center representative per cluster"
        )
    if md_metrics_key in all_metrics:
        print("  - local_structure_coords_clusters.csv: local-structure centers with cluster IDs")
        print("  - local_structure_coords_clusters.npz: local-structure centers + cluster IDs")
        print("  - md_space_clusters.png: 3D MD space clusters")
        print("  - md_space_clusters.html: interactive 3D MD space clusters")
        if len(configured_k_values) > 1:
            print("  - md_space_clusters_k*.html: interactive 3D MD plots for configured k values")
            print("  - latent_tsne_clusters_k*.png: t-SNE plots for configured k values")
        if hdbscan_enabled:
            print("  - local_structure_hdbscan_coords_clusters.csv: MD centers with HDBSCAN labels")
            print("  - local_structure_hdbscan_coords_clusters.npz: MD centers + HDBSCAN labels")
            print("  - md_space_clusters_hdbscan.html: interactive 3D MD HDBSCAN clusters")
        if "cluster_profiles" in all_metrics[md_metrics_key]:
            print("  - cluster_profiles_by_k/k_*/cluster_XX_structures.png: 8 structures per cluster")
            print("  - cluster_profiles_by_k/k_*/cluster_XX_properties.png: per-sample topology/material metrics")
            print("  - cluster_profiles_by_k/k_*/cluster_comparison_property_heatmap.png: cross-cluster properties")
            print("  - cluster_profiles_by_k/k_*/cluster_comparison_distance_size.png: cross-cluster size/distance")

    return all_metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run post-training analysis for contrastive (Barlow Twins) checkpoints.",
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to a trained checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write analysis outputs (default: <ckpt_dir>/analysis).",
    )
    parser.add_argument(
        "--cuda_device",
        type=int,
        default=0,
        help="CUDA device index (default: 0).",
    )
    parser.add_argument(
        "--max_batches_latent",
        type=int,
        default=None,
        help="Max batches to use for latent analysis (default: all).",
    )
    parser.add_argument(
        "--max_samples_visualization",
        type=int,
        default=None,
        help="Max samples for t-SNE (default: analysis_tsne_max_samples or 8000).",
    )
    parser.add_argument(
        "--data_file",
        action="append",
        default=None,
        help="Override real data files (repeat for multiple). Example: --data_file 175ps.off",
    )
    parser.add_argument(
        "--test_rotation_runs",
        type=int,
        default=None,
        help="Override number of random-rotation test runs (default: analysis_test_rotation_runs or 5).",
    )
    parser.add_argument(
        "--test_max_samples",
        type=int,
        default=None,
        help="Override max samples for test-phase metrics (default: analysis_test_max_samples or max_test_samples).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint_path = os.path.expanduser(args.checkpoint_path)
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(os.getcwd(), checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(checkpoint_path), "analysis")
    else:
        output_dir = os.path.expanduser(output_dir)
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)

    run_post_training_analysis(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        cuda_device=int(args.cuda_device),
        cfg=None,
        max_batches_latent=args.max_batches_latent,
        max_samples_visualization=args.max_samples_visualization,
        data_files_override=args.data_file,
        test_rotation_runs=args.test_rotation_runs,
        test_max_samples=args.test_max_samples,
    )


if __name__ == "__main__":
    main()
