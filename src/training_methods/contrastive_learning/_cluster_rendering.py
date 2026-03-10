"""Matplotlib figure rendering for MD snapshots and cluster representatives."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from src.training_methods.contrastive_learning._cluster_colors import (
    _boost_saturation,
    _cluster_label_color,
    _compute_center_to_edge_colors,
)
from src.training_methods.contrastive_learning._cluster_geometry import (
    _build_local_coordination_edges,
    _compute_cluster_representative_indices,
    _draw_edges,
    _load_points_from_dataset,
    _orient_points_for_crystal_view,
    _sample_indices_stratified,
    _set_equal_axes_3d,
    _draw_cube_wireframe,
)


def _log_saved_figure(path: Path | str) -> None:
    print(f"[analysis][savefig] {Path(path).resolve()}")


def _ensure_connected_edges(
    points: np.ndarray,
    edges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}.")
    n_points = int(pts.shape[0])
    if n_points <= 1:
        return []

    parent = list(range(n_points))
    rank = [0] * n_points

    def _find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def _union(a: int, b: int) -> bool:
        root_a = _find(a)
        root_b = _find(b)
        if root_a == root_b:
            return False
        if rank[root_a] < rank[root_b]:
            parent[root_a] = root_b
        elif rank[root_a] > rank[root_b]:
            parent[root_b] = root_a
        else:
            parent[root_b] = root_a
            rank[root_a] += 1
        return True

    edge_set: set[tuple[int, int]] = set()
    for raw_edge in edges:
        i = int(raw_edge[0])
        j = int(raw_edge[1])
        if i == j:
            continue
        edge = (min(i, j), max(i, j))
        edge_set.add(edge)
        _union(edge[0], edge[1])

    if len({_find(i) for i in range(n_points)}) == 1:
        return sorted(edge_set)

    dmat = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    candidate_edges: list[tuple[float, tuple[int, int]]] = []
    for i in range(n_points):
        for j in range(i + 1, n_points):
            candidate_edges.append((float(dmat[i, j]), (i, j)))
    candidate_edges.sort(key=lambda item: item[0])

    for _, edge in candidate_edges:
        if _union(edge[0], edge[1]):
            edge_set.add(edge)
        if len({_find(i) for i in range(n_points)}) == 1:
            break

    if len({_find(i) for i in range(n_points)}) != 1:
        raise RuntimeError(
            "Failed to construct a connected edge graph for cluster representative rendering."
        )
    return sorted(edge_set)


def _compute_structure_half_span(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}.")
    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    span = float(np.max(maxs - mins))
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError(f"Computed invalid structure span {span} for points shape {pts.shape}.")
    return 0.5 * span


def _set_equal_axes_3d_with_half_span(
    ax: Any,
    coords: np.ndarray,
    *,
    half_span: float,
) -> None:
    pts = np.asarray(coords, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"coords must have shape (N, >=3), got {pts.shape}.")
    pts = pts[:, :3]
    half = float(half_span)
    if not np.isfinite(half) or half <= 0.0:
        raise ValueError(f"half_span must be positive and finite, got {half_span}.")

    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    center = 0.5 * (mins + maxs)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 1.0))


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
    saturation_boost: float = 1.0,
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
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for cluster_id in unique_labels:
        cluster_mask = labels_plot == cluster_id
        if not np.any(cluster_mask):
            continue
        base_color = color_map.get(cluster_id, "#777777")
        cluster_points = coords_plot[cluster_mask]
        point_colors = np.repeat(
            np.asarray(mcolors.to_rgb(str(base_color)), dtype=np.float32)[None, :],
            cluster_points.shape[0],
            axis=0,
        )
        if abs(float(saturation_boost) - 1.0) > 1e-6:
            point_colors = _boost_saturation(
                point_colors,
                float(saturation_boost),
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
    _log_saved_figure(out_file)

    return {
        "out_file": str(out_file),
        "num_points_total": int(coords_arr.shape[0]),
        "num_points_visible": int(coords_use.shape[0]),
        "num_points_rendered": int(coords_plot.shape[0]),
        "clusters_rendered": unique_labels,
        "saturation_boost": float(saturation_boost),
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


def _prepare_cluster_representative_structures(
    dataset: Any,
    latents: np.ndarray,
    cluster_labels: np.ndarray,
    color_map: dict[int, str],
    *,
    point_scale: float,
    target_points: int,
) -> list[dict[str, Any]]:
    reps = _compute_cluster_representative_indices(latents, cluster_labels)
    prepared: list[dict[str, Any]] = []
    for cluster_id in sorted(reps.keys()):
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
        prepared.append(
            {
                "cluster_id": int(cluster_id),
                "sample_index": int(sample_idx),
                "base_color": str(color_map.get(cluster_id, "#777777")),
                "centered_points": np.asarray(centered, dtype=np.float32),
                "local_points": np.asarray(local, dtype=np.float32),
            }
        )
    if not prepared:
        raise ValueError("No representative structures were prepared for rendering.")
    return prepared


def _build_cluster_representative_variant_specs(
    preferred_orientation: str,
) -> list[dict[str, Any]]:
    preferred = str(preferred_orientation).strip().lower()
    if preferred not in {"pca", "none"}:
        raise ValueError(
            "Unsupported preferred representative orientation: "
            f"{preferred_orientation!r}. Expected one of ['pca', 'none']."
        )

    def _orientation_slug(method: str) -> str:
        if method in {"pca", "none"}:
            return method
        raise ValueError(
            "Unsupported representative orientation while building filename suffix: "
            f"{method!r}."
        )

    preferred_slug = _orientation_slug(preferred)
    return [
        {
            "variant_name": f"{preferred_slug}_reciprocal",
            "file_suffix": f"_{preferred_slug}_reciprocal",
            "edge_method": "coordination_shell_mutual",
            "orientation_method": preferred,
        },
        {
            "variant_name": f"{preferred_slug}_degree_capped",
            "file_suffix": f"_{preferred_slug}_degree_capped",
            "edge_method": "coordination_shell_degree_capped",
            "orientation_method": preferred,
        },
    ]


def _render_cluster_representatives_variant(
    prepared_records: list[dict[str, Any]],
    out_file: Path,
    *,
    knn_k: int,
    orientation_method: str,
    edge_method: str,
    view_elev: float,
    view_azim: float,
    projection: str,
    variant_name: str,
) -> dict[str, Any]:
    proj_norm = str(projection).strip().lower()
    n_clusters = len(prepared_records)
    n_cols = min(3, n_clusters)
    n_rows = int(np.ceil(n_clusters / max(1, n_cols)))
    fig = plt.figure(figsize=(3.45 * n_cols, 3.5 * n_rows), dpi=220, facecolor="white")

    summary_records: list[dict[str, Any]] = []
    for pos, prepared in enumerate(prepared_records):
        ax = fig.add_subplot(n_rows, n_cols, pos + 1, projection="3d")
        ax.set_facecolor("white")
        if hasattr(ax, "set_proj_type"):
            ax.set_proj_type(proj_norm)
        ax.view_init(elev=float(view_elev), azim=float(view_azim))

        local_oriented, orientation_info = _orient_points_for_crystal_view(
            prepared["local_points"],
            method=str(orientation_method),
        )
        base_color = str(prepared["base_color"])
        # Keep representative coloring consistent with the paper single-column
        # variant (08_cluster_representatives_spatial_neighbors_paper_k*).
        point_colors = _compute_center_to_edge_colors(local_oriented, base_color)
        edges, edge_info = _build_local_coordination_edges(
            local_oriented,
            min_shell_neighbors=max(2, int(knn_k) - 1),
            max_shell_neighbors=max(5, int(knn_k) + 2),
            shell_gap_ratio=1.22,
            edge_mode=str(edge_method),
        )
        _draw_edges(
            ax,
            local_oriented,
            edges,
            point_colors=point_colors,
            edge_alpha=0.60,
            edge_linewidth=0.94,
        )
        ax.scatter(
            local_oriented[:, 0],
            local_oriented[:, 1],
            local_oriented[:, 2],
            c=point_colors,
            s=58,
            alpha=0.97,
            edgecolors="#222222",
            linewidths=0.36,
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
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            if hasattr(axis, "pane"):
                axis.pane.fill = False
                axis.pane.set_edgecolor((1.0, 1.0, 1.0, 1.0))
            if hasattr(axis, "line"):
                axis.line.set_color((1.0, 1.0, 1.0, 1.0))
        panel_label = f"C{pos + 1}"
        title_color = _cluster_label_color(base_color, darken_factor=0.58)
        ax.set_title(
            panel_label,
            fontsize=12,
            color=title_color,
            pad=2,
            fontweight="bold",
        )
        summary_records.append(
            {
                "panel_label": panel_label,
                "cluster_id": int(prepared["cluster_id"]),
                "sample_index": int(prepared["sample_index"]),
                "num_points_plotted": int(local_oriented.shape[0]),
                "edge_info": edge_info,
                "orientation": orientation_info,
            }
        )

    fig.suptitle("Cluster representatives", fontsize=12, fontweight="bold", y=0.965)
    fig.subplots_adjust(left=0.02, right=0.988, bottom=0.035, top=0.91, wspace=0.03, hspace=0.06)
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    _log_saved_figure(out_file)
    return {
        "variant_name": str(variant_name),
        "out_file": str(out_file),
        "orientation_method": str(orientation_method),
        "edge_method": str(edge_method),
        "view_elev": float(view_elev),
        "view_azim": float(view_azim),
        "projection": str(proj_norm),
        "representatives": summary_records,
    }


def _build_squidpy_generic_spatial_neighbor_edges(
    points: np.ndarray,
    *,
    n_neighs: int,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"points must have shape (N, >=3), got {pts.shape}.")
    pts = pts[:, :3]
    n_points = int(pts.shape[0])
    if n_points < 2:
        return [], {
            "graph_method": "squidpy_spatial_neighbors_generic_knn",
            "coord_type": "generic",
            "n_neighs": 0,
            "num_edges": 0,
            "degree_median": 0.0,
            "degree_min": 0,
            "degree_max": 0,
            "edge_distance_mean": 0.0,
            "edge_distance_median": 0.0,
        }

    k_eff = min(max(1, int(n_neighs)), n_points - 1)
    dmat = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(dmat, np.inf)
    knn_indices = np.argsort(dmat, axis=1)[:, :k_eff]

    edges: set[tuple[int, int]] = set()
    edge_distances: list[float] = []
    for point_idx in range(n_points):
        for neighbor_idx in knn_indices[point_idx]:
            neighbor_idx_int = int(neighbor_idx)
            edge = (min(int(point_idx), neighbor_idx_int), max(int(point_idx), neighbor_idx_int))
            if edge[0] == edge[1]:
                continue
            if edge not in edges:
                edges.add(edge)
                edge_distances.append(float(np.linalg.norm(pts[edge[0]] - pts[edge[1]])))

    degree_counts = np.zeros((n_points,), dtype=np.int32)
    for src_idx, dst_idx in edges:
        degree_counts[int(src_idx)] += 1
        degree_counts[int(dst_idx)] += 1

    if edge_distances:
        edge_distance_mean = float(np.mean(edge_distances))
        edge_distance_median = float(np.median(edge_distances))
    else:
        edge_distance_mean = 0.0
        edge_distance_median = 0.0

    return sorted(edges), {
        "graph_method": "squidpy_spatial_neighbors_generic_knn",
        "coord_type": "generic",
        "n_neighs": int(k_eff),
        "num_edges": int(len(edges)),
        "degree_median": float(np.median(degree_counts)),
        "degree_min": int(np.min(degree_counts)),
        "degree_max": int(np.max(degree_counts)),
        "edge_distance_mean": float(edge_distance_mean),
        "edge_distance_median": float(edge_distance_median),
    }


def _render_two_shell_cluster_representatives_variant(
    prepared_records: list[dict[str, Any]],
    out_file: Path,
    *,
    knn_k: int,
    view_elev: float,
    view_azim: float,
    projection: str,
    variant_name: str,
    connection_mode: str = "spatial_neighbors_generic",
) -> dict[str, Any]:
    mode = str(connection_mode).strip().lower()
    if mode != "spatial_neighbors_generic":
        raise ValueError(
            "Unsupported connection_mode for two-shell representative rendering: "
            f"{connection_mode!r}. Expected 'spatial_neighbors_generic'."
        )

    proj_norm = str(projection).strip().lower()
    n_clusters = len(prepared_records)
    n_cols = min(3, n_clusters)
    n_rows = int(np.ceil(n_clusters / max(1, n_cols)))
    # Match paper representative styling so perceived colors are consistent with
    # 08_cluster_representatives_spatial_neighbors_paper_k*.
    point_size = 48.0
    point_linewidth = 0.26
    # Spatial-neighbor graphs are denser than the paper variant edges and can
    # make colors look desaturated/darker by overpainting. Keep points identical
    # to paper colors but lower edge prominence for perceptual color parity.
    edge_alpha = 0.42
    edge_linewidth = 0.86
    fig = plt.figure(figsize=(3.45 * n_cols, 3.5 * n_rows), dpi=300, facecolor="white")
    summary_records: list[dict[str, Any]] = []

    for pos, prepared in enumerate(prepared_records):
        ax = fig.add_subplot(n_rows, n_cols, pos + 1, projection="3d")
        ax.set_facecolor("white")
        if hasattr(ax, "set_proj_type"):
            ax.set_proj_type(proj_norm)
        ax.view_init(elev=float(view_elev), azim=float(view_azim))

        base_color = str(prepared["base_color"])
        oriented_points, orientation_info = _orient_points_for_crystal_view(
            prepared["local_points"],
            method="pca",
        )
        # Keep representative coloring consistent with the paper single-column
        # variant (08_cluster_representatives_spatial_neighbors_paper_k*).
        point_colors = _compute_center_to_edge_colors(oriented_points, base_color)
        spatial_n_neighs = max(1, int(knn_k))
        edges, spatial_neighbor_info = _build_squidpy_generic_spatial_neighbor_edges(
            oriented_points,
            n_neighs=int(spatial_n_neighs),
        )
        edge_info = {
            "connection_mode": str(mode),
            "num_edges": int(len(edges)),
            "input_point_count": int(oriented_points.shape[0]),
            "spatial_neighbors": spatial_neighbor_info,
        }

        _draw_edges(
            ax,
            oriented_points,
            edges,
            point_colors=point_colors,
            edge_alpha=float(edge_alpha),
            edge_linewidth=float(edge_linewidth),
        )
        ax.scatter(
            oriented_points[:, 0],
            oriented_points[:, 1],
            oriented_points[:, 2],
            c=point_colors,
            s=float(point_size),
            alpha=0.97,
            edgecolors="#222222",
            linewidths=float(point_linewidth),
            depthshade=False,
        )
        _set_equal_axes_3d(ax, oriented_points)
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
                axis.pane.set_edgecolor((1.0, 1.0, 1.0, 1.0))
            if hasattr(axis, "line"):
                axis.line.set_color((1.0, 1.0, 1.0, 1.0))
        panel_label = f"C{pos + 1}"
        title_color = _cluster_label_color(base_color, darken_factor=0.58)
        ax.set_title(
            panel_label,
            fontsize=12,
            color=title_color,
            pad=2,
            fontweight="bold",
        )
        summary_records.append(
            {
                "panel_label": panel_label,
                "cluster_id": int(prepared["cluster_id"]),
                "sample_index": int(prepared["sample_index"]),
                "num_points_plotted": int(oriented_points.shape[0]),
                "orientation": orientation_info,
                "edge_info": edge_info,
            }
        )

    fig.suptitle("Cluster representatives", fontsize=12, fontweight="bold", y=0.965)
    fig.subplots_adjust(left=0.02, right=0.988, bottom=0.035, top=0.91, wspace=0.03, hspace=0.06)
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    _log_saved_figure(out_file)
    return {
        "variant_name": str(variant_name),
        "out_file": str(out_file),
        "orientation_method": "pca",
        "connection_mode": str(mode),
        "view_elev": float(view_elev),
        "view_azim": float(view_azim),
        "projection": str(proj_norm),
        "representatives": summary_records,
    }


def _render_spatial_neighbors_paper_figure(
    prepared_records: list[dict[str, Any]],
    out_file: Path,
    *,
    knn_k: int,
    view_elev: float = 22.0,
    view_azim: float = 38.0,
    projection: str = "ortho",
) -> dict[str, Any]:
    """Paper-quality single-column figure of spatial-neighbor cluster representatives.

    Style mirrors ``figure_local_base_paper.png`` from the synthetic generator:
    DPI 300, point_size 48, edge_linewidth 0.94, shared display half-span,
    coordination-shell mutual bonds, row labels colored by cluster, no axis chrome.
    """
    proj_norm = str(projection).strip().lower()
    n_clusters = len(prepared_records)
    min_shell_neighbors = max(2, int(knn_k) - 2)
    max_shell_neighbors = max(5, int(knn_k) + 1)
    # Match synthetic local_base_paper panel styling.
    point_size = 48.0
    point_linewidth = 0.26
    edge_alpha = 0.60
    edge_linewidth = 0.94

    # -- compute shared display half-span (like _compute_paper_display_half_span) --
    all_half_spans: list[float] = []
    oriented_cache: list[
        tuple[np.ndarray, np.ndarray, list[tuple[int, int]], dict[str, Any], dict[str, Any]]
    ] = []
    for prepared in prepared_records:
        oriented_points, orientation_info = _orient_points_for_crystal_view(
            prepared["local_points"],
            method="pca",
        )
        base_color = str(prepared["base_color"])
        point_colors = _compute_center_to_edge_colors(
            oriented_points,
            base_color,
        )
        edges, edge_info = _build_local_coordination_edges(
            oriented_points,
            min_shell_neighbors=int(min_shell_neighbors),
            max_shell_neighbors=int(max_shell_neighbors),
            shell_gap_ratio=1.22,
            edge_mode="coordination_shell_mutual",
        )
        edges = _ensure_connected_edges(oriented_points, edges)
        edge_info = dict(edge_info)
        edge_info["edge_mode"] = "coordination_shell_mutual_connected"
        edge_info["requested_knn_k"] = int(knn_k)
        edge_info["min_shell_neighbors"] = int(min_shell_neighbors)
        edge_info["max_shell_neighbors"] = int(max_shell_neighbors)
        half_span = _compute_structure_half_span(oriented_points)
        all_half_spans.append(half_span)
        oriented_cache.append((oriented_points, point_colors, edges, edge_info, orientation_info))

    display_half_span = float(max(all_half_spans) * 0.94) if all_half_spans else 1.0

    # -- build figure: one column, one row per cluster --
    panel_height = 7.85 / 4.0
    fig_width = 6.35 / 2.0
    fig_height = panel_height * n_clusters
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=300, facecolor="white")

    # Manual axes placement following local_base_paper proportions.
    top = 0.922
    bottom = 0.040
    left_ax = 0.160
    ax_width = 0.790
    row_gap = -0.014
    total_h = top - bottom - row_gap * max(0, n_clusters - 1)
    ax_height = total_h / max(1, n_clusters)

    summary_records: list[dict[str, Any]] = []
    for pos, prepared in enumerate(prepared_records):
        oriented_points, point_colors, edges, edge_info, orientation_info = oriented_cache[pos]
        base_color = str(prepared["base_color"])

        y0 = top - (pos + 1) * ax_height - pos * row_gap
        ax = fig.add_axes([left_ax, y0, ax_width, ax_height], projection="3d")
        ax.set_facecolor("white")
        if hasattr(ax, "set_proj_type"):
            ax.set_proj_type(proj_norm)
        ax.view_init(elev=float(view_elev), azim=float(view_azim))

        # Draw with paper style parameters
        _draw_edges(
            ax,
            oriented_points,
            edges,
            point_colors=point_colors,
            edge_alpha=float(edge_alpha),
            edge_linewidth=float(edge_linewidth),
        )
        ax.scatter(
            oriented_points[:, 0],
            oriented_points[:, 1],
            oriented_points[:, 2],
            c=point_colors,
            s=float(point_size),
            alpha=0.97,
            edgecolors="#222222",
            linewidths=float(point_linewidth),
            depthshade=False,
        )

        # Shared display bounds
        _set_equal_axes_3d_with_half_span(
            ax,
            oriented_points,
            half_span=float(display_half_span),
        )

        # Clean axis chrome
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
                axis.pane.set_edgecolor((1.0, 1.0, 1.0, 1.0))
            if hasattr(axis, "line"):
                axis.line.set_color((1.0, 1.0, 1.0, 1.0))

        # Row label (like paper variant)
        label_color = _cluster_label_color(base_color, darken_factor=0.68)
        bbox = ax.get_position()
        y_center = 0.5 * (bbox.y0 + bbox.y1)
        fig.text(
            left_ax - 0.028,
            y_center,
            f"C{pos + 1}",
            ha="right",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=label_color,
        )

        summary_records.append(
            {
                "panel_label": f"C{pos + 1}",
                "cluster_id": int(prepared["cluster_id"]),
                "sample_index": int(prepared["sample_index"]),
                "num_points_plotted": int(oriented_points.shape[0]),
                "orientation": orientation_info,
                "edge_info": edge_info,
            }
        )

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    _log_saved_figure(out_file)
    return {
        "variant_name": "spatial_neighbors_paper",
        "out_file": str(out_file),
        "orientation_method": "pca",
        "connection_mode": "coordination_shell_mutual_connected",
        "view_elev": float(view_elev),
        "view_azim": float(view_azim),
        "projection": str(proj_norm),
        "display_half_span": float(display_half_span),
        "representatives": summary_records,
    }


def _save_cluster_representatives_figure(
    dataset: Any,
    latents: np.ndarray,
    cluster_labels: np.ndarray,
    color_map: dict[int, str],
    out_file: Path,
    *,
    point_scale: float,
    target_points: int = 64,
    knn_k: int = 4,
    orientation_method: str = "pca",
    view_elev: float = 22.0,
    view_azim: float = 38.0,
    projection: str = "ortho",
) -> dict[str, Any]:
    proj_norm = str(projection).strip().lower()
    method_norm = str(orientation_method).strip().lower()
    prepared_records = _prepare_cluster_representative_structures(
        dataset,
        latents,
        cluster_labels,
        color_map,
        point_scale=float(point_scale),
        target_points=int(target_points),
    )
    out_file = Path(out_file)
    variant_specs = _build_cluster_representative_variant_specs(method_norm)
    variant_summaries: list[dict[str, Any]] = []
    for spec in variant_specs:
        file_suffix = str(spec["file_suffix"]).strip()
        if file_suffix == "":
            raise ValueError(
                "Representative variant spec is missing a filename suffix; "
                f"variant={spec!r}."
            )
        variant_path = out_file.with_name(f"{out_file.stem}{file_suffix}{out_file.suffix}")
        variant_summary = _render_cluster_representatives_variant(
            prepared_records,
            variant_path,
            knn_k=int(knn_k),
            orientation_method=str(spec["orientation_method"]),
            edge_method=str(spec["edge_method"]),
            view_elev=float(view_elev),
            view_azim=float(view_azim),
            projection=str(proj_norm),
            variant_name=str(spec["variant_name"]),
        )
        variant_summaries.append(variant_summary)

    stem_parts = out_file.stem.rsplit("_k", 1)

    k_token = stem_parts[1].strip()
    two_shell_spatial_neighbors_summary = _render_two_shell_cluster_representatives_variant(
        prepared_records,
        out_file.with_name(
            f"07_cluster_representatives_two_shells_pca_spatial_neighbors_k{k_token}.png"
        ),
        knn_k=int(knn_k),
        view_elev=float(view_elev),
        view_azim=float(view_azim),
        projection=str(proj_norm),
        variant_name="pca_two_shells_spatial_neighbors",
        connection_mode="spatial_neighbors_generic",
    )
    spatial_neighbors_paper_summary = _render_spatial_neighbors_paper_figure(
        prepared_records,
        out_file.with_name(
            f"08_cluster_representatives_spatial_neighbors_paper_k{k_token}.png"
        ),
        knn_k=int(knn_k),
        view_elev=float(view_elev),
        view_azim=float(view_azim),
        projection=str(proj_norm),
    )
    primary_summary = dict(variant_summaries[0])
    primary_summary["variants"] = variant_summaries
    primary_summary["primary_variant_name"] = str(variant_summaries[0]["variant_name"])
    primary_summary["projection"] = str(proj_norm)
    primary_summary["pca_two_shell_figures"] = {
        "spatial_neighbors": two_shell_spatial_neighbors_summary,
        "spatial_neighbors_paper": spatial_neighbors_paper_summary,
    }
    return primary_summary
