"""Matplotlib figure rendering for MD snapshots and cluster representatives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from .cluster_colors import (
    _boost_saturation,
    _cluster_label_color,
    _compute_center_to_edge_colors,
)
from .cluster_geometry import (
    _build_local_coordination_edges,
    _compute_cluster_representative_indices,
    _draw_edges,
    _load_points_from_dataset,
    _orient_points_for_crystal_view,
    _sample_indices_stratified,
    _set_equal_axes_3d,
    _draw_cube_wireframe,
)
from .representative_structures import (
    _build_cluster_representative_analysis_summary,
    analyze_cluster_representatives,
    materialize_cluster_representative_analysis_summary,
)
from .output_layout import log_saved_figure as _log_saved_figure


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
    labels = np.asarray(cluster_labels, dtype=int)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError(f"coords must have shape (N, 3), got {coords_arr.shape}.")
    if labels.ndim != 1 or labels.shape[0] != coords_arr.shape[0]:
        raise ValueError(
            "cluster_labels must have shape (N,) matching coords. "
            f"labels={labels.shape}, coords={coords_arr.shape}."
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
    missing_colors = [cluster_id for cluster_id in unique_labels if cluster_id not in color_map]
    if missing_colors:
        raise KeyError(f"MD cluster snapshot is missing colors for clusters {missing_colors}.")

    fig = plt.figure(figsize=(7.8, 7.8), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for cluster_id in unique_labels:
        cluster_mask = labels_plot == cluster_id
        if not np.any(cluster_mask):
            continue
        base_color = color_map[cluster_id]
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


def _prepare_cluster_representative_structures(
    dataset: Any,
    latents: np.ndarray,
    cluster_labels: np.ndarray,
    color_map: dict[int, str],
    *,
    point_scale: float,
    target_points: int,
    selection_features: np.ndarray | None = None,
    selection_info: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    reps = _compute_cluster_representative_indices(
        latents,
        cluster_labels,
        selection_features=selection_features,
    )
    prepared: list[dict[str, Any]] = []
    for cluster_id in sorted(reps.keys()):
        if cluster_id not in color_map:
            raise KeyError(
                f"Representative rendering is missing a color for cluster {cluster_id}."
            )
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
                "base_color": str(color_map[cluster_id]),
                "centered_points": np.asarray(centered, dtype=np.float32),
                "local_points": np.asarray(local, dtype=np.float32),
                "selection_info": {} if selection_info is None else dict(selection_info),
            }
        )
    if not prepared:
        raise ValueError("No representative structures were prepared for rendering.")
    return prepared


def _build_cluster_representative_render_cache(
    dataset: Any,
    latents: np.ndarray,
    cluster_labels: np.ndarray,
    color_map: dict[int, str],
    *,
    point_scale: float,
    target_points: int,
    representative_ptm_enabled: bool,
    representative_cna_enabled: bool,
    representative_cna_max_signatures: int,
    representative_center_atom_tolerance: float,
    representative_shell_min_neighbors: int,
    representative_shell_max_neighbors: int,
    selection_features: np.ndarray | None = None,
    selection_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prepared_records = _prepare_cluster_representative_structures(
        dataset,
        latents,
        cluster_labels,
        color_map,
        point_scale=float(point_scale),
        target_points=int(target_points),
        selection_features=selection_features,
        selection_info=selection_info,
    )
    structure_analysis_summary = _build_cluster_representative_analysis_summary(
        prepared_records,
        ptm_enabled=bool(representative_ptm_enabled),
        cna_enabled=bool(representative_cna_enabled),
        cna_max_signatures=int(representative_cna_max_signatures),
        center_atom_tolerance=float(representative_center_atom_tolerance),
        shell_min_neighbors=int(representative_shell_min_neighbors),
        shell_max_neighbors=int(representative_shell_max_neighbors),
    )
    return {
        "prepared_records": prepared_records,
        "structure_analysis_summary": structure_analysis_summary,
        "selection_info": {} if selection_info is None else dict(selection_info),
    }


def _attach_structure_analysis_to_summary(
    summary: dict[str, Any],
    analysis_by_cluster_id: dict[int, dict[str, Any]],
) -> None:
    representatives = summary.get("representatives")
    if not isinstance(representatives, list):
        raise ValueError(
            "Representative render summary must contain a 'representatives' list."
        )
    for record in representatives:
        cluster_id = int(record["cluster_id"])
        if cluster_id in analysis_by_cluster_id:
            record["structure_analysis"] = dict(analysis_by_cluster_id[cluster_id])


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
    dpi: int = 220,
) -> dict[str, Any]:
    proj_norm = str(projection).strip().lower()
    if proj_norm not in {"ortho", "persp"}:
        raise ValueError(
            "Representative projection must be 'ortho' or 'persp', "
            f"got {projection!r}."
        )
    n_clusters = len(prepared_records)
    n_cols = min(3, n_clusters)
    n_rows = int(np.ceil(n_clusters / max(1, n_cols)))
    fig = plt.figure(figsize=(3.45 * n_cols, 3.5 * n_rows), dpi=dpi, facecolor="white")

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
        # Keep representative coloring consistent across representative variants.
        point_colors = _compute_center_to_edge_colors(local_oriented, base_color)
        if edge_method == "knn_connected":
            edges, edge_info = _build_knn_representative_edges(
                local_oriented, knn_k=max(2, int(knn_k))
            )
        else:
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


def _build_knn_representative_edges(
    points: np.ndarray,
    *,
    knn_k: int,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    pts = np.asarray(points, dtype=np.float32)
    n_points = int(pts.shape[0])
    if n_points < 2:
        return [], {
            "graph_method": "representative_knn",
            "knn_k": 0,
            "num_edges": 0,
            "edge_distance_mean": 0.0,
            "edge_distance_median": 0.0,
        }

    k_eff = min(max(1, int(knn_k)), n_points - 1)
    tree = cKDTree(pts)
    _, indices = tree.query(pts, k=k_eff + 1)
    indices = np.asarray(indices, dtype=np.int64).reshape(n_points, -1)

    edges: set[tuple[int, int]] = set()
    for point_idx in range(n_points):
        for neighbor_idx in indices[point_idx, 1:]:
            neighbor_idx_int = int(neighbor_idx)
            edge = (min(point_idx, neighbor_idx_int), max(point_idx, neighbor_idx_int))
            if edge[0] == edge[1] or edge in edges:
                continue
            edges.add(edge)

    connected_edges = _ensure_connected_edges(pts, sorted(edges))
    distances = np.asarray([
        np.linalg.norm(pts[src] - pts[dst]) for src, dst in connected_edges
    ])

    return connected_edges, {
        "graph_method": "representative_knn",
        "knn_k": int(k_eff),
        "num_edges": int(len(connected_edges)),
        "edge_distance_mean": float(distances.mean()) if distances.size else 0.0,
        "edge_distance_median": float(np.median(distances)) if distances.size else 0.0,
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
    representative_ptm_enabled: bool = False,
    representative_cna_enabled: bool = False,
    representative_cna_max_signatures: int = 5,
    representative_center_atom_tolerance: float = 1e-6,
    representative_shell_min_neighbors: int = 8,
    representative_shell_max_neighbors: int = 24,
    representative_render_cache: dict[str, Any] | None = None,
    selection_features: np.ndarray | None = None,
    selection_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    proj_norm = str(projection).strip().lower()
    if proj_norm not in {"ortho", "persp"}:
        raise ValueError(
            "Representative projection must be 'ortho' or 'persp', "
            f"got {projection!r}."
        )
    method_norm = str(orientation_method).strip().lower()
    if method_norm not in {"pca", "none"}:
        raise ValueError(f"Unsupported representative orientation {method_norm!r}.")
    out_file = Path(out_file)
    variant_specs = [{
        "variant_name": f"{method_norm}_reciprocal",
        "file_suffix": f"_{method_norm}_reciprocal",
        "edge_method": "coordination_shell_mutual",
        "orientation_method": method_norm,
    }]
    variant_summaries: list[dict[str, Any]] = []
    stem_parts = out_file.stem.rsplit("_k", 1)
    if len(stem_parts) != 2 or not stem_parts[1].strip():
        raise ValueError(
            "Representative output filename must contain a '_k<value>' suffix, "
            f"got {out_file.name!r}."
        )
    k_token = stem_parts[1].strip()
    if representative_render_cache is None:
        prepared_records = _prepare_cluster_representative_structures(
            dataset,
            latents,
            cluster_labels,
            color_map,
            point_scale=float(point_scale),
            target_points=int(target_points),
            selection_features=selection_features,
            selection_info=selection_info,
        )
        structure_analysis_summary = analyze_cluster_representatives(
            prepared_records,
            out_file.parent,
            k_token=str(k_token),
            ptm_enabled=bool(representative_ptm_enabled),
            cna_enabled=bool(representative_cna_enabled),
            cna_max_signatures=int(representative_cna_max_signatures),
            center_atom_tolerance=float(representative_center_atom_tolerance),
            shell_min_neighbors=int(representative_shell_min_neighbors),
            shell_max_neighbors=int(representative_shell_max_neighbors),
        )
        representative_selection_summary = {} if selection_info is None else dict(selection_info)
    else:
        cached_prepared_records = representative_render_cache["prepared_records"]
        if not isinstance(cached_prepared_records, list) or not cached_prepared_records:
            raise ValueError(
                "representative_render_cache must contain a non-empty 'prepared_records' list."
            )
        expected_cluster_ids = sorted(int(v) for v in np.unique(np.asarray(cluster_labels, dtype=int)) if int(v) >= 0)
        cached_cluster_ids = sorted(
            int(record["cluster_id"])
            for record in cached_prepared_records
        )
        if cached_cluster_ids != expected_cluster_ids:
            raise ValueError(
                "representative_render_cache cluster ids do not match the current labels. "
                f"cached_cluster_ids={cached_cluster_ids}, expected_cluster_ids={expected_cluster_ids}."
            )
        prepared_records = [
            {
                **dict(record),
                "base_color": str(color_map[int(record["cluster_id"])]),
            }
            for record in cached_prepared_records
        ]
        cached_structure_summary = representative_render_cache["structure_analysis_summary"]
        if bool(representative_ptm_enabled) or bool(representative_cna_enabled):
            if not isinstance(cached_structure_summary, dict):
                raise ValueError(
                    "representative_render_cache must contain a dict 'structure_analysis_summary' "
                    "when PTM or CNA analysis is enabled."
                )
            structure_analysis_summary = materialize_cluster_representative_analysis_summary(
                cached_structure_summary,
                out_file.parent,
                k_token=str(k_token),
            )
        else:
            structure_analysis_summary = None
        cached_selection_info = representative_render_cache["selection_info"]
        representative_selection_summary = (
            {} if cached_selection_info is None else dict(cached_selection_info)
        )
    analysis_by_cluster_id: dict[int, dict[str, Any]] = {}
    if structure_analysis_summary is not None:
        analysis_by_cluster_id = {
            int(record["cluster_id"]): dict(record)
            for record in structure_analysis_summary["representatives"]
        }
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
        _attach_structure_analysis_to_summary(variant_summary, analysis_by_cluster_id)
        variant_summaries.append(variant_summary)
    stale_spatial_neighbors_paper_path = out_file.with_name(
        f"08_cluster_representatives_spatial_neighbors_paper_k{k_token}.png"
    )
    if stale_spatial_neighbors_paper_path.exists():
        stale_spatial_neighbors_paper_path.unlink()
    knn_edges_summary = _render_cluster_representatives_variant(
        prepared_records,
        out_file.with_name(
            f"09_cluster_representatives_knn_edges_k{k_token}.png"
        ),
        knn_k=int(knn_k),
        orientation_method="pca",
        edge_method="knn_connected",
        view_elev=float(view_elev),
        view_azim=float(view_azim),
        projection=str(proj_norm),
        variant_name="knn_edges",
        dpi=300,
    )
    _attach_structure_analysis_to_summary(knn_edges_summary, analysis_by_cluster_id)
    primary_summary = dict(variant_summaries[0])
    primary_summary["variants"] = variant_summaries
    primary_summary["primary_variant_name"] = str(variant_summaries[0]["variant_name"])
    primary_summary["projection"] = str(proj_norm)
    primary_summary["structure_analysis"] = structure_analysis_summary
    primary_summary["representative_selection"] = representative_selection_summary
    primary_summary["pca_two_shell_figures"] = {
        "knn_edges": knn_edges_summary,
    }
    return primary_summary
