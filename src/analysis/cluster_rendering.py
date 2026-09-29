"""Matplotlib figure rendering for MD snapshots and cluster representatives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from .cluster_colors import (
    _boost_saturation,
)
from .cluster_geometry import (
    _compute_cluster_representative_indices,
    _load_points_from_dataset,
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
    for record in representatives:
        cluster_id = int(record["cluster_id"])
        if cluster_id in analysis_by_cluster_id:
            record["structure_analysis"] = dict(analysis_by_cluster_id[cluster_id])


def _save_cluster_representatives_figure(
    dataset: Any,
    latents: np.ndarray,
    cluster_labels: np.ndarray,
    color_map: dict[int, str],
    out_file: Path,
    *,
    point_scale: float,
    target_points: int = 64,
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
    from .representative_style import render_representatives
    render_records=[]
    for prepared in prepared_records:
        cid=int(prepared['cluster_id'])
        # Keep the existing PTM/CNA support and diagnostics; display at most 64 atoms.
        points=np.asarray(prepared['local_points'])[:64]
        cna=analysis_by_cluster_id.get(cid,{}).get('cna')
        if cna is not None:
            cutoff=float(cna['cutoff']);cutoff_source='saved focal CNA shell cutoff'
        else:
            radii=np.sort(np.linalg.norm(points,axis=1))
            cutoff=float(1.2*np.median(radii[1:13]));cutoff_source='1.2 × median first-12 distance (display only)'
        render_records.append(dict(cluster_id=cid,sample_index=prepared['sample_index'],
            base_color=prepared['base_color'],points=points,cutoff=cutoff,cutoff_source=cutoff_source,
            units='analysis coordinate units'))
    primary_summary=render_representatives(render_records,out_file,orientation=method_norm,
        view_elev=float(view_elev),view_azim=float(view_azim),projection=proj_norm)
    _attach_structure_analysis_to_summary(primary_summary,analysis_by_cluster_id)
    primary_summary['structure_analysis']=structure_analysis_summary
    primary_summary['representative_selection']=representative_selection_summary
    _log_saved_figure(out_file)
    _log_saved_figure(out_file.with_suffix('.html'))
    return primary_summary
