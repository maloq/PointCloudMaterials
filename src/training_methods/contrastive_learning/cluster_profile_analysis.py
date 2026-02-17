from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
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
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 3 and points.shape[0] == 1:
        points = points[0]
    if points.ndim != 2:
        points = np.reshape(points, (-1, points.shape[-1]))
    if points.shape[1] > 3:
        points = points[:, :3]
    if points.shape[1] < 3:
        raise ValueError(f"Expected points with at least 3 columns, got {points.shape}")
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return points.astype(np.float32, copy=False)


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
    except Exception:
        return None
    points = _extract_points_from_item(item)
    points_np = points.detach().cpu().numpy()
    points_np = _safe_xyz(points_np)
    if point_scale != 1.0:
        points_np = points_np * float(point_scale)
    return points_np


def _compute_cluster_centers_and_distances(
    latents: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    if latents.size == 0 or labels.size != len(latents):
        return {}, np.empty((0,), dtype=np.float32)
    lat = np.asarray(latents, dtype=np.float32)
    if lat.ndim != 2:
        lat = np.reshape(lat, (lat.shape[0], -1))
    labels = np.asarray(labels, dtype=int)
    centers: dict[int, np.ndarray] = {}
    distances = np.full((lat.shape[0],), np.nan, dtype=np.float32)
    for cluster_id in np.unique(labels):
        if int(cluster_id) < 0:
            continue
        idx = np.flatnonzero(labels == int(cluster_id))
        if idx.size == 0:
            continue
        center = lat[idx].mean(axis=0)
        centers[int(cluster_id)] = center.astype(np.float32, copy=False)
        d = np.linalg.norm(lat[idx] - center[None, :], axis=1)
        distances[idx] = d.astype(np.float32, copy=False)
    return centers, distances


def _select_progressive_cluster_samples(
    cluster_indices: np.ndarray,
    distances: np.ndarray,
    *,
    num_samples: int,
) -> np.ndarray:
    cluster_indices = np.asarray(cluster_indices, dtype=int)
    if cluster_indices.size == 0:
        return np.empty((0,), dtype=int)

    d = np.asarray(distances, dtype=np.float32)[cluster_indices]
    finite = np.isfinite(d)
    if not finite.any():
        order = np.arange(cluster_indices.size, dtype=int)
    else:
        inf_fill = np.nanmax(d[finite]) + 1.0
        order = np.argsort(np.where(finite, d, inf_fill))
    sorted_idx = cluster_indices[order]
    if sorted_idx.size <= int(num_samples):
        return sorted_idx

    target = np.linspace(0, sorted_idx.size - 1, int(num_samples))
    selected_positions: list[int] = []
    used: set[int] = set()
    for pos in target:
        cand = int(np.round(pos))
        if cand in used:
            offset = 1
            while True:
                left = cand - offset
                right = cand + offset
                found = None
                if left >= 0 and left not in used:
                    found = left
                elif right < sorted_idx.size and right not in used:
                    found = right
                if found is not None:
                    cand = found
                    break
                if left < 0 and right >= sorted_idx.size:
                    break
                offset += 1
        if cand not in used:
            used.add(cand)
            selected_positions.append(cand)

    if len(selected_positions) < int(num_samples):
        for cand in range(sorted_idx.size):
            if cand in used:
                continue
            selected_positions.append(cand)
            used.add(cand)
            if len(selected_positions) >= int(num_samples):
                break
    selected_positions = selected_positions[: int(num_samples)]
    selected_positions = sorted(selected_positions)
    return sorted_idx[np.asarray(selected_positions, dtype=int)]


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
        nb = tree.query_ball_point(points[i], r=float(cutoff))
        nb = [j for j in nb if j != i]
        if len(nb) < 2:
            continue
        if len(nb) > 6:
            d = np.linalg.norm(points[nb] - points[i], axis=1)
            nb = [nb[j] for j in np.argsort(d)[:6]]
        center = points[i]
        for a in range(len(nb)):
            for b in range(a + 1, len(nb)):
                v1 = points[nb[a]] - center
                v2 = points[nb[b]] - center
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 < 1e-8 or n2 < 1e-8:
                    continue
                cos_val = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                angles.append(float(np.degrees(np.arccos(cos_val))))
    return np.asarray(angles, dtype=np.float32)


def _compute_convex_volume(points: np.ndarray) -> float:
    n = len(points)
    if n < 4:
        span = np.ptp(points, axis=0) if n > 0 else np.zeros((3,), dtype=np.float32)
        return float(np.prod(np.maximum(span, 1e-6)))
    try:
        hull = ConvexHull(points)
        vol = float(hull.volume)
        if np.isfinite(vol) and vol > 0.0:
            return vol
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
    bins = 40
    hist, edges = np.histogram(pair_dists, bins=bins, range=(0.0, upper))
    r = 0.5 * (edges[1:] + edges[:-1])
    dr = float(edges[1] - edges[0]) if len(edges) > 1 else 1.0
    vol = _compute_convex_volume(points)
    rho = float(n) / max(vol, 1e-8)
    shell_vol = 4.0 * np.pi * (r**2) * dr
    expected = 0.5 * float(n) * rho * shell_vol
    g_r = np.divide(
        hist.astype(np.float32),
        expected.astype(np.float32),
        out=np.zeros_like(r, dtype=np.float32),
        where=expected > 1e-12,
    )
    if g_r.size < 2:
        return None, None
    peak_idx = int(np.argmax(g_r[1:]) + 1)
    return float(r[peak_idx]), float(g_r[peak_idx])


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
        nset = {int(v) for v in np.atleast_1d(row)[1:] if int(v) >= 0}
        neighbors.append(nset)
    coeffs: list[float] = []
    for i in range(n):
        nbrs = list(neighbors[i])
        deg = len(nbrs)
        if deg < 2:
            coeffs.append(0.0)
            continue
        links = 0
        for a in range(deg):
            na = nbrs[a]
            for b in range(a + 1, deg):
                nb = nbrs[b]
                if nb in neighbors[na]:
                    links += 1
        coeffs.append(2.0 * links / (deg * (deg - 1)))
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
    points = _safe_xyz(points)
    if points.size == 0:
        return {name: float("nan") for name, _ in _ALL_PROFILE_PROPERTIES}

    center_idx = int(np.argmin(np.linalg.norm(points, axis=1)))
    centered = points - points[center_idx]
    nn_mean = _estimate_nn_distance(centered)
    cutoff = float(nn_mean * 1.35) if np.isfinite(nn_mean) else float("nan")

    coord = _compute_coordination_numbers(centered, cutoff)
    angles = _compute_bond_angles(centered, cutoff)
    rdf_peak_r, rdf_peak_g = _compute_rdf_peak(centered, nn_mean)
    shape = _compute_shape_descriptors(centered)
    vol = _compute_convex_volume(centered)
    density = float(len(centered) / max(vol, 1e-8))
    out = {
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
    return out


def _draw_knn_edges(
    ax: Any,
    points: np.ndarray,
    *,
    knn_k: int = 4,
    edge_color: str = "#884422",
) -> None:
    if len(points) < 2:
        return
    tree = cKDTree(points)
    k = min(int(knn_k) + 1, len(points))
    _, indices = tree.query(points, k=k)
    if indices.ndim == 1:
        indices = indices[:, None]
    drawn: set[tuple[int, int]] = set()
    for i in range(len(points)):
        for j in np.atleast_1d(indices[i])[1:]:
            j = int(j)
            if j == i or j < 0 or j >= len(points):
                continue
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
                linewidth=0.7,
                alpha=0.55,
            )


def _plot_local_structure_sample(
    ax: Any,
    points: np.ndarray,
    *,
    target_count: int = 48,
    knn_k: int = 4,
) -> None:
    points = _safe_xyz(points)
    if points.size == 0:
        ax.axis("off")
        return
    center_idx = int(np.argmin(np.linalg.norm(points, axis=1)))
    centered = points - points[center_idx]
    d = np.linalg.norm(centered, axis=1)
    keep = np.argsort(d)[: min(int(target_count), len(centered))]
    coords = centered[keep]
    if len(coords) == 0:
        ax.axis("off")
        return

    _draw_knn_edges(ax, coords, knn_k=knn_k)
    dist = np.linalg.norm(coords, axis=1)
    vmax = float(np.max(dist)) if dist.size else 1.0
    colors = plt.cm.viridis(dist / max(vmax, 1e-6))
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=colors,
        s=42,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.35,
    )
    lim = float(max(np.max(np.abs(coords)) * 1.1, 1e-2))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def _make_row_zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(arr.shape[0]):
        row = arr[i]
        finite = np.isfinite(row)
        if not finite.any():
            continue
        mu = float(np.mean(row[finite]))
        sigma = float(np.std(row[finite]))
        if sigma < 1e-8:
            out[i, finite] = 0.0
        else:
            out[i, finite] = (row[finite] - mu) / sigma
    return out


def _save_cluster_structures_figure(
    output_path: Path,
    cluster_id: int,
    k_value: int,
    sample_records: list[dict[str, Any]],
    *,
    samples_per_cluster: int,
    target_points: int,
    knn_k: int,
) -> None:
    n_slots = max(1, int(samples_per_cluster))
    n_cols = 4 if n_slots >= 4 else n_slots
    n_rows = int(np.ceil(n_slots / n_cols))
    fig = plt.figure(figsize=(4.1 * n_cols, 4.3 * n_rows))

    for slot in range(n_slots):
        ax = fig.add_subplot(n_rows, n_cols, slot + 1, projection="3d")
        if slot >= len(sample_records):
            ax.axis("off")
            continue
        rec = sample_records[slot]
        points = rec["points"]
        _plot_local_structure_sample(
            ax,
            points,
            target_count=target_points,
            knn_k=knn_k,
        )
        dist = rec.get("distance_to_center", float("nan"))
        sample_id = rec.get("sample_index", -1)
        title = f"S{slot + 1} idx={sample_id}\nd_center={dist:.4f}"
        center = rec.get("md_center")
        if center is not None:
            cx, cy, cz = center
            title += f"\n({cx:.1f}, {cy:.1f}, {cz:.1f})"
        ax.set_title(title, fontsize=8)

    fig.suptitle(
        f"Cluster {cluster_id} local structures (k={k_value})\n"
        f"{samples_per_cluster} samples ordered from center-near to center-far",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _save_cluster_properties_figure(
    output_path: Path,
    cluster_id: int,
    k_value: int,
    sample_records: list[dict[str, Any]],
) -> None:
    if not sample_records:
        return

    sample_labels = [f"S{i + 1}" for i in range(len(sample_records))]
    distances = np.asarray(
        [float(rec.get("distance_to_center", np.nan)) for rec in sample_records],
        dtype=np.float32,
    )

    def _build_matrix(prop_defs: list[tuple[str, str]]) -> tuple[np.ndarray, list[str]]:
        mat = np.full((len(prop_defs), len(sample_records)), np.nan, dtype=np.float32)
        names: list[str] = []
        for i, (key, title) in enumerate(prop_defs):
            names.append(title)
            for j, rec in enumerate(sample_records):
                val = rec.get("properties", {}).get(key, np.nan)
                mat[i, j] = np.nan if val is None else float(val)
        return mat, names

    mat_props, mat_labels = _build_matrix(_MATERIAL_PROPERTIES)
    topo_props, topo_labels = _build_matrix(_TOPOLOGY_PROPERTIES)
    mat_plot = _make_row_zscore(mat_props)
    topo_plot = _make_row_zscore(topo_props)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 11),
        gridspec_kw={"height_ratios": [1.2, 1.1, 0.8]},
    )

    im0 = axes[0].imshow(mat_plot, cmap="coolwarm", aspect="auto")
    axes[0].set_title("Material-science metrics (row-wise z-score)")
    axes[0].set_yticks(np.arange(len(mat_labels)))
    axes[0].set_yticklabels(mat_labels)
    axes[0].set_xticks(np.arange(len(sample_labels)))
    axes[0].set_xticklabels(sample_labels)
    for i in range(mat_props.shape[0]):
        for j in range(mat_props.shape[1]):
            val = mat_props[i, j]
            if np.isfinite(val):
                axes[0].text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im0, ax=axes[0], fraction=0.02, pad=0.02)

    im1 = axes[1].imshow(topo_plot, cmap="coolwarm", aspect="auto")
    axes[1].set_title("Topological descriptors (row-wise z-score)")
    axes[1].set_yticks(np.arange(len(topo_labels)))
    axes[1].set_yticklabels(topo_labels)
    axes[1].set_xticks(np.arange(len(sample_labels)))
    axes[1].set_xticklabels(sample_labels)
    for i in range(topo_props.shape[0]):
        for j in range(topo_props.shape[1]):
            val = topo_props[i, j]
            if np.isfinite(val):
                axes[1].text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im1, ax=axes[1], fraction=0.02, pad=0.02)

    axes[2].bar(np.arange(len(sample_labels)), distances, color="#4477AA")
    axes[2].set_xticks(np.arange(len(sample_labels)))
    axes[2].set_xticklabels(sample_labels)
    axes[2].set_ylabel("Distance in latent space")
    axes[2].set_title("Distance from cluster center")
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Cluster {cluster_id} sample properties (k={k_value})",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _save_cluster_comparison_heatmap(
    output_path: Path,
    k_value: int,
    cluster_summaries: list[dict[str, Any]],
) -> None:
    if not cluster_summaries:
        return
    prop_defs = _ALL_PROFILE_PROPERTIES
    mat = np.full((len(cluster_summaries), len(prop_defs)), np.nan, dtype=np.float32)
    cluster_labels: list[str] = []
    for i, summary in enumerate(cluster_summaries):
        cluster_labels.append(f"C{int(summary['cluster_id'])}")
        means = summary.get("properties_mean", {})
        for j, (key, _) in enumerate(prop_defs):
            val = means.get(key, np.nan)
            mat[i, j] = np.nan if val is None else float(val)

    z = np.zeros_like(mat, dtype=np.float32)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        finite = np.isfinite(col)
        if not finite.any():
            continue
        mu = float(np.mean(col[finite]))
        sigma = float(np.std(col[finite]))
        if sigma < 1e-8:
            z[finite, j] = 0.0
        else:
            z[finite, j] = (col[finite] - mu) / sigma

    fig, ax = plt.subplots(figsize=(1.2 * len(prop_defs) + 4.5, 0.75 * len(cluster_summaries) + 3.0))
    im = ax.imshow(z, cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(prop_defs)))
    ax.set_xticklabels([name for _, name in prop_defs], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(cluster_labels)))
    ax.set_yticklabels(cluster_labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(f"Cluster comparison for k={k_value} (raw values, z-score colors)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _save_cluster_distance_comparison(
    output_path: Path,
    k_value: int,
    labels: np.ndarray,
    distances: np.ndarray,
) -> None:
    cluster_ids = [int(v) for v in np.unique(labels) if int(v) >= 0]
    if not cluster_ids:
        return
    counts = [int(np.sum(labels == cid)) for cid in cluster_ids]
    dist_data = [distances[labels == cid] for cid in cluster_ids]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [1.0, 1.2]},
    )
    axes[0].bar(np.arange(len(cluster_ids)), counts, color="#66AA66")
    axes[0].set_xticks(np.arange(len(cluster_ids)))
    axes[0].set_xticklabels([f"C{cid}" for cid in cluster_ids])
    axes[0].set_ylabel("Samples")
    axes[0].set_title("Cluster sizes")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].boxplot(dist_data, labels=[f"C{cid}" for cid in cluster_ids], showfliers=False)
    axes[1].set_ylabel("Distance to cluster center")
    axes[1].set_title("Distance distribution by cluster")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle(f"Cluster distance overview (k={k_value})", fontsize=12, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def generate_cluster_profile_reports(
    *,
    out_root: Path,
    dataset: Any,
    latents: np.ndarray,
    coords: np.ndarray,
    cluster_labels_by_k: Dict[int, np.ndarray],
    cluster_methods_by_k: Dict[int, str],
    samples_per_cluster: int,
    target_points: int,
    knn_k: int,
    max_cluster_property_samples: int,
    point_scale: float,
    random_seed: int = 123,
) -> dict[str, Any]:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    latents = np.asarray(latents, dtype=np.float32)
    if latents.ndim != 2:
        latents = np.reshape(latents, (latents.shape[0], -1))
    num_samples = int(latents.shape[0])
    if num_samples == 0:
        return {}
    has_coords = coords.size and coords.shape[0] == num_samples and coords.shape[1] >= 3
    property_cache: dict[int, dict[str, float]] = {}
    rng = np.random.default_rng(random_seed)

    def _get_properties(sample_idx: int) -> dict[str, float] | None:
        sample_idx = int(sample_idx)
        if sample_idx in property_cache:
            return property_cache[sample_idx]
        points = _load_point_cloud_from_dataset(
            dataset,
            sample_idx,
            point_scale=point_scale,
        )
        if points is None or points.size == 0:
            return None
        props = _compute_sample_properties(points)
        property_cache[sample_idx] = props
        return props

    summary: dict[str, Any] = {
        "root_dir": str(out_root),
        "k_values": [],
    }

    for k_value in sorted(cluster_labels_by_k.keys()):
        labels = np.asarray(cluster_labels_by_k[int(k_value)], dtype=int)
        if labels.shape[0] != num_samples:
            continue
        centers, distances = _compute_cluster_centers_and_distances(latents, labels)
        cluster_ids = [int(v) for v in np.unique(labels) if int(v) >= 0]
        if not cluster_ids:
            continue

        k_dir = out_root / f"k_{int(k_value)}"
        k_dir.mkdir(parents=True, exist_ok=True)

        cluster_summaries: list[dict[str, Any]] = []
        for cluster_id in cluster_ids:
            idx = np.flatnonzero(labels == int(cluster_id))
            if idx.size == 0:
                continue
            selected = _select_progressive_cluster_samples(
                idx,
                distances,
                num_samples=samples_per_cluster,
            )
            sample_records: list[dict[str, Any]] = []
            for s_idx in selected:
                points = _load_point_cloud_from_dataset(
                    dataset,
                    int(s_idx),
                    point_scale=point_scale,
                )
                if points is None or points.size == 0:
                    continue
                props = _get_properties(int(s_idx))
                if props is None:
                    continue
                md_center = None
                if has_coords:
                    md_center = [float(v) for v in coords[int(s_idx), :3]]
                sample_records.append(
                    {
                        "sample_index": int(s_idx),
                        "distance_to_center": float(distances[int(s_idx)]),
                        "md_center": md_center,
                        "points": points,
                        "properties": props,
                    }
                )

            eval_idx = idx
            if int(max_cluster_property_samples) > 0 and idx.size > int(max_cluster_property_samples):
                eval_idx = rng.choice(
                    idx,
                    size=int(max_cluster_property_samples),
                    replace=False,
                )
            eval_props: list[dict[str, float]] = []
            for e_idx in eval_idx:
                props = _get_properties(int(e_idx))
                if props is not None:
                    eval_props.append(props)

            prop_means: dict[str, float | None] = {}
            prop_stds: dict[str, float | None] = {}
            for key, _ in _ALL_PROFILE_PROPERTIES:
                values = [
                    float(p.get(key))
                    for p in eval_props
                    if p.get(key) is not None and np.isfinite(float(p.get(key)))
                ]
                if values:
                    arr = np.asarray(values, dtype=np.float32)
                    prop_means[key] = float(np.mean(arr))
                    prop_stds[key] = float(np.std(arr))
                else:
                    prop_means[key] = None
                    prop_stds[key] = None

            structures_path = k_dir / f"cluster_{cluster_id:02d}_structures.png"
            properties_path = k_dir / f"cluster_{cluster_id:02d}_properties.png"
            _save_cluster_structures_figure(
                structures_path,
                cluster_id,
                int(k_value),
                sample_records,
                samples_per_cluster=samples_per_cluster,
                target_points=target_points,
                knn_k=knn_k,
            )
            _save_cluster_properties_figure(
                properties_path,
                cluster_id,
                int(k_value),
                sample_records,
            )

            sample_records_json = []
            for rec in sample_records:
                item = {k: v for k, v in rec.items() if k != "points"}
                sample_records_json.append(item)

            cluster_summary = {
                "cluster_id": int(cluster_id),
                "cluster_size": int(idx.size),
                "cluster_center_norm": float(np.linalg.norm(centers.get(int(cluster_id), np.zeros((1,))))),
                "distance_min": float(np.nanmin(distances[idx])) if idx.size else None,
                "distance_mean": float(np.nanmean(distances[idx])) if idx.size else None,
                "distance_max": float(np.nanmax(distances[idx])) if idx.size else None,
                "distance_std": float(np.nanstd(distances[idx])) if idx.size else None,
                "samples_selected": sample_records_json,
                "properties_mean": prop_means,
                "properties_std": prop_stds,
                "structures_figure": str(structures_path),
                "properties_figure": str(properties_path),
            }
            cluster_summaries.append(cluster_summary)

            with (k_dir / f"cluster_{cluster_id:02d}_summary.json").open("w") as handle:
                json.dump(cluster_summary, handle, indent=2, default=_json_default)

        _save_cluster_comparison_heatmap(
            k_dir / "cluster_comparison_property_heatmap.png",
            int(k_value),
            cluster_summaries,
        )
        _save_cluster_distance_comparison(
            k_dir / "cluster_comparison_distance_size.png",
            int(k_value),
            labels,
            distances,
        )

        k_summary = {
            "k": int(k_value),
            "method": str(cluster_methods_by_k.get(int(k_value), "unknown")),
            "num_clusters": int(len(cluster_ids)),
            "clusters": cluster_summaries,
            "comparison_property_heatmap": str(k_dir / "cluster_comparison_property_heatmap.png"),
            "comparison_distance_size": str(k_dir / "cluster_comparison_distance_size.png"),
        }
        with (k_dir / "k_summary.json").open("w") as handle:
            json.dump(k_summary, handle, indent=2, default=_json_default)
        summary["k_values"].append(k_summary)

    with (out_root / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, default=_json_default)
    return summary
