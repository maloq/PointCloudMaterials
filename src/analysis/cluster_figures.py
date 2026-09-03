"""Cluster-figure utilities for post-training analysis visualizations.

Public API is re-exported from focused submodules. This module also contains
the top-level orchestrator ``_save_fixed_k_cluster_figure_set``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .utils import _sample_indices
from .cluster_colors import (
    _build_cluster_color_map,
    _cluster_palette,
)
from .cluster_geometry import (
    _build_rotation_view_specs,
)
from .cluster_rendering import (
    _save_cluster_representatives_figure,
    _save_md_cluster_snapshot,
)
from .cluster_blender import (
    _save_md_cluster_snapshots_raytrace_blender,
)
from .output_layout import write_json
from .representative_structures import detect_crystal_like_cluster_ids

# Re-export public symbols used by predict_and_visualize.py
__all__ = [
    "_build_cluster_color_map",
    "_save_fixed_k_cluster_figure_set",
]


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
    icl_enabled: bool,
    icl_k_min: int,
    icl_k_max: int,
    icl_max_samples: int | None,
    representative_points: int,
    md_point_size: float,
    md_point_alpha: float,
    md_halo_scale: float,
    md_halo_alpha: float,
    md_saturation_boost: float,
    md_view_elev: float,
    md_view_azim: float,
    md_num_views: int,
    representative_orientation_method: str,
    representative_view_elev: float,
    representative_view_azim: float,
    representative_projection: str,
    representative_ptm_enabled: bool,
    representative_cna_enabled: bool,
    representative_cna_max_signatures: int,
    representative_center_atom_tolerance: float,
    representative_shell_min_neighbors: int,
    representative_shell_max_neighbors: int,
    cluster_color_assignment: dict[int, Any] | None = None,
    random_state: int = 42,
    raytrace_render_enabled: bool = False,
    raytrace_blender_executable: str = "blender",
    raytrace_render_resolution: int = 1200,
    raytrace_render_max_points: int | None = None,
    raytrace_render_samples: int = 32,
    raytrace_render_denoise: bool = True,
    raytrace_render_high_quality: bool = False,
    raytrace_render_projection: str = "perspective",
    raytrace_render_fov_deg: float = 34.0,
    raytrace_render_camera_distance_factor: float = 2.8,
    raytrace_render_sphere_radius_fraction: float = 0.0105,
    raytrace_render_timeout_sec: int = 1200,
    raytrace_render_use_gpu: bool = False,
    representative_render_cache: dict[str, Any] | None = None,
    representative_selection_features: np.ndarray | None = None,
    representative_selection_info: dict[str, Any] | None = None,
    include_all_cluster_panels: bool = True,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stale_patterns = (
        "*_pretty.png",
        "01_md_clusters_all_k*.png",
        "01_md_clusters_all_k*_raytrace.png",
        "01_md_clusters_all_k*_raytrace_gallery.png",
        "02_md_clusters_*_k*.png",
        "03_cluster_count_icl_k*.png",
        "04_cluster_representatives_k*.png",
        "05_cluster_representatives_fcc_shell_k*.png",
        "06_cluster_representatives_two_shells_pca_center_first_k*.png",
        "07_cluster_representatives_two_shells_pca_intrashell_k*.png",
        "07_cluster_representatives_two_shells_pca_spatial_neighbors_k*.png",
        "08_cluster_representatives_spatial_neighbors_paper_k*.png",
        "09_cluster_representatives_knn_edges_k*.png",
        "10_cluster_representatives_structure_analysis_k*.json",
        "10_cluster_representatives_structure_analysis_k*.csv",
        "cluster_color_assignment_k*.json",
    )
    stale_paths: set[Path] = set()
    for pattern in stale_patterns:
        stale_paths.update(out_dir.glob(pattern))
    for stale_path in stale_paths:
        stale_path.unlink()

    labels = np.asarray(cluster_labels, dtype=int)
    coords_arr = np.asarray(coords, dtype=np.float32)
    lat_arr = np.asarray(latents, dtype=np.float32)
    if labels.ndim != 1:
        raise ValueError(f"Cluster figure labels must have shape (N,), got {labels.shape}.")
    if coords_arr.ndim != 2 or coords_arr.shape != (labels.shape[0], 3):
        raise ValueError(
            "Cluster figure coordinates must have shape (N, 3) matching labels, "
            f"got coords={coords_arr.shape}, labels={labels.shape}."
        )
    if lat_arr.ndim != 2:
        raise ValueError(f"Cluster figure latents must have shape (N, D), got {lat_arr.shape}.")
    if lat_arr.shape[0] != labels.shape[0]:
        raise ValueError(
            "Cluster figure latents and labels must have the same row count, "
            f"got latents={lat_arr.shape}, labels={labels.shape}."
        )

    cluster_ids = sorted(int(v) for v in np.unique(labels) if int(v) >= 0)
    palette = _cluster_palette(len(cluster_ids))
    color_map = _build_cluster_color_map(
        labels,
        cluster_color_assignment=cluster_color_assignment,
    )
    color_assignment_path = out_dir / f"cluster_color_assignment_k{k_value}.json"
    color_assignment_payload = {
        "note": (
            "Edit assignment values and rerun with "
            "--cluster_color_assignment_file <this file>. "
            "Each assignment value may be either a palette index or an explicit color string."
        ),
        "cluster_ids": [int(cluster_id) for cluster_id in cluster_ids],
        "palette_by_index": {
            str(idx): str(color) for idx, color in enumerate(palette)
        },
        "assignment": {
            str(cluster_id): str(color_map[int(cluster_id)])
            for cluster_id in cluster_ids
        },
    }
    write_json(color_assignment_path, color_assignment_payload)
    raytrace_projection_norm = str(raytrace_render_projection).strip().lower()
    if raytrace_projection_norm not in {"orthographic", "ortho", "perspective", "persp"}:
        raise ValueError(
            "Raytrace projection must be one of "
            "['orthographic', 'ortho', 'perspective', 'persp'], "
            f"got {raytrace_render_projection!r}."
        )

    md_snapshot_kwargs = {
        "max_points": md_max_points,
        "point_size": float(md_point_size),
        "alpha": float(md_point_alpha),
        "halo_scale": float(md_halo_scale),
        "halo_alpha": float(md_halo_alpha),
        "saturation_boost": float(md_saturation_boost),
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
        "use_denoise": bool(raytrace_render_denoise),
        "use_gpu": bool(raytrace_render_use_gpu),
        "timeout_seconds": raytrace_render_timeout_sec,
        "wireframe_enabled": True,
    }

    # -- 04  Cluster representatives ------------------------------------------

    panel_reps = _save_cluster_representatives_figure(
        dataset,
        lat_arr,
        labels,
        color_map,
        out_dir / f"04_cluster_representatives_k{k_value}.png",
        point_scale=float(point_scale),
        target_points=int(representative_points),
        knn_k=4,
        orientation_method=str(representative_orientation_method),
        view_elev=float(representative_view_elev),
        view_azim=float(representative_view_azim),
        projection=str(representative_projection),
        representative_ptm_enabled=bool(representative_ptm_enabled),
        representative_cna_enabled=bool(representative_cna_enabled),
        representative_cna_max_signatures=int(representative_cna_max_signatures),
        representative_center_atom_tolerance=float(representative_center_atom_tolerance),
        representative_shell_min_neighbors=int(representative_shell_min_neighbors),
        representative_shell_max_neighbors=int(representative_shell_max_neighbors),
        representative_render_cache=representative_render_cache,
        selection_features=representative_selection_features,
        selection_info=representative_selection_info,
    )
    structure_analysis = panel_reps["structure_analysis"]
    if structure_analysis is None:
        raise RuntimeError(
            "Automatic crystal-like cluster rendering requires representative PTM "
            "analysis. Set figure_set.representatives.ptm_enabled=true."
        )
    crystal_like_cluster_ids = detect_crystal_like_cluster_ids(structure_analysis)
    print(
        "[analysis] PTM crystal-like clusters "
        f"(FCC/HCP/BCC) for k={k_value}: {crystal_like_cluster_ids}"
    )

    raytrace_pending_jobs: list[dict[str, Any]] = []

    def _render_cluster_view(
        *,
        snapshot_path: Path,
        snapshot_title: str,
        view_name: str,
        elev: float,
        azim: float,
        visible_cluster_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        panel_view = _save_md_cluster_snapshot(
            coords_arr,
            labels,
            color_map,
            snapshot_path,
            title=snapshot_title,
            visible_cluster_ids=visible_cluster_ids,
            view_elev=float(elev),
            view_azim=float(azim),
            **md_snapshot_kwargs,
        )
        if bool(raytrace_render_enabled):
            raytrace_pending_jobs.append(
                {
                    "out_file": Path(snapshot_path).with_stem(
                        Path(snapshot_path).stem + "_raytrace"
                    ),
                    "title": str(snapshot_title),
                    "visible_cluster_ids": (
                        None
                        if visible_cluster_ids is None
                        else [int(v) for v in visible_cluster_ids]
                    ),
                    "view_elev": float(elev),
                    "view_azim": float(azim),
                    "panel_view": panel_view,
                }
            )
        panel_view["view_name"] = str(view_name)
        return panel_view

    def _run_pending_raytrace_batch() -> None:
        if not raytrace_pending_jobs:
            return
        if not bool(raytrace_render_enabled):
            raise RuntimeError(
                "Internal error: pending raytrace jobs exist while raytrace rendering is disabled."
            )
        blender_jobs = [
            {key: value for key, value in job.items() if key != "panel_view"}
            for job in raytrace_pending_jobs
        ]
        raytrace_results = _save_md_cluster_snapshots_raytrace_blender(
            coords_arr,
            labels,
            color_map,
            blender_jobs,
            **raytrace_render_kwargs,
        )
        if len(raytrace_results) != len(raytrace_pending_jobs):
            raise RuntimeError(
                "Blender raytrace batch returned an unexpected result count: "
                f"expected={len(raytrace_pending_jobs)}, got={len(raytrace_results)}."
            )
        for job, raytrace_info in zip(raytrace_pending_jobs, raytrace_results, strict=True):
            job["panel_view"]["raytrace_render"] = raytrace_info

    # -- 01  All-clusters rotation views with optional raytrace ---------------
    all_view_specs = _build_rotation_view_specs(
        base_elev=float(md_view_elev),
        base_azim=float(md_view_azim),
        num_views=max(1, int(md_num_views)),
    )
    panel_all_views: list[dict[str, Any]] = []
    panel_all: dict[str, Any] | None = None
    if bool(include_all_cluster_panels):
        for view_name, elev, azim in all_view_specs:
            out_name = (
                f"01_md_clusters_all_k{k_value}.png"
                if view_name == "view1"
                else f"01_md_clusters_all_k{k_value}_{view_name}.png"
            )
            snapshot_path = out_dir / out_name
            snapshot_title = f"MD space clusters (k={k_value}, all clusters, {view_name})"
            panel_view = _render_cluster_view(
                snapshot_path=snapshot_path,
                snapshot_title=snapshot_title,
                view_name=str(view_name),
                elev=float(elev),
                azim=float(azim),
            )
            panel_all_views.append(panel_view)
        panel_all = panel_all_views[0]

    # -- 02  Automatically detected crystal-like cluster views ----------------

    panel_crystal_like: dict[str, Any] | None = None
    panel_crystal_like_views: list[dict[str, Any]] = []
    if crystal_like_cluster_ids:
        for view_name, elev, azim in all_view_specs:
            out_name = (
                f"02_md_clusters_crystal_like_k{k_value}.png"
                if view_name == "view1"
                else f"02_md_clusters_crystal_like_k{k_value}_{view_name}.png"
            )
            panel_view = _render_cluster_view(
                snapshot_path=out_dir / out_name,
                snapshot_title=(
                    f"MD space PTM crystal-like clusters "
                    f"(k={k_value}, {view_name})"
                ),
                view_name=str(view_name),
                elev=float(elev),
                azim=float(azim),
                visible_cluster_ids=crystal_like_cluster_ids,
            )
            panel_crystal_like_views.append(panel_view)
        panel_crystal_like = dict(panel_crystal_like_views[0])
        panel_crystal_like["views"] = panel_crystal_like_views
        panel_crystal_like["cluster_ids_shown"] = crystal_like_cluster_ids
    else:
        print(
            "[analysis] No PTM crystal-like cluster representatives were detected for "
            f"k={k_value}; no 02_md_clusters_crystal_like image will be written."
        )

    if bool(raytrace_render_enabled):
        _run_pending_raytrace_batch()
        if panel_crystal_like is not None:
            panel_crystal_like["raytrace_render"] = panel_crystal_like_views[0][
                "raytrace_render"
            ]

    # -- 03  ICL curve --------------------------------------------------------

    panel_icl: dict[str, Any] | None = None
    icl_curve: dict[int, dict[str, float]] = {}
    icl_prep: dict[str, Any] | None = None
    icl_num_samples = 0
    if bool(icl_enabled):
        from .cluster_icl import (
            _compute_icl_curve,
            _prepare_icl_features,
            _save_icl_curve_figure,
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
            random_state=int(random_state),
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
        invalid_k_values = [k for k in k_values_icl if k > max_valid_k]
        if invalid_k_values:
            raise ValueError(
                "ICL k values must be smaller than the number of ICL samples. "
                f"invalid={invalid_k_values}, max_valid_k={max_valid_k}, "
                f"num_samples={icl_features.shape[0]}."
            )
        icl_curve = _compute_icl_curve(
            icl_features,
            k_values_icl,
            random_state=int(random_state),
        )
        panel_icl = _save_icl_curve_figure(
            icl_curve,
            selected_k=int(k_value),
            out_file=out_dir / f"03_cluster_count_icl_k{k_value}.png",
        )
        icl_num_samples = int(icl_features.shape[0])

    return {
        "k_value": int(k_value),
        "output_dir": str(out_dir),
        "cluster_ids": cluster_ids,
        "cluster_color_palette": {
            int(idx): str(color) for idx, color in enumerate(palette)
        },
        "cluster_color_map": {int(k): str(v) for k, v in color_map.items()},
        "cluster_color_assignment_file": str(color_assignment_path),
        "random_state": int(random_state),
        "panel_all_clusters": panel_all,
        "panel_all_clusters_views": panel_all_views,
        "crystal_like_cluster_ids": crystal_like_cluster_ids,
        "panel_crystal_like": panel_crystal_like,
        "panel_crystal_like_views": panel_crystal_like_views,
        "icl_enabled": bool(icl_enabled),
        "panel_icl": panel_icl,
        "panel_representatives": panel_reps,
        "panel_representatives_two_shells_pca": panel_reps["pca_two_shell_figures"],
        "panel_representatives_structure_analysis": panel_reps.get("structure_analysis"),
        "md_render_settings": {
            "max_points": None if md_max_points is None else int(md_max_points),
            "point_size": float(md_point_size),
            "alpha": float(md_point_alpha),
            "halo_scale": float(md_halo_scale),
            "halo_alpha": float(md_halo_alpha),
            "saturation_boost": float(md_saturation_boost),
            "view_elev": float(md_view_elev),
            "view_azim": float(md_view_azim),
            "num_views": int(md_num_views),
        },
        "raytrace_render_settings": {
            "enabled": bool(raytrace_render_enabled),
            "blender_executable": str(raytrace_blender_executable),
            "resolution": int(raytrace_render_resolution),
            "max_points": (
                None if raytrace_render_max_points is None else int(raytrace_render_max_points)
            ),
            "samples": int(raytrace_render_samples),
            "denoise": bool(raytrace_render_denoise),
            "high_quality": bool(raytrace_render_high_quality),
            "projection": str(raytrace_projection_norm),
            "fov_deg": float(raytrace_render_fov_deg),
            "camera_distance_factor": float(raytrace_render_camera_distance_factor),
            "sphere_radius_fraction": float(raytrace_render_sphere_radius_fraction),
            "timeout_sec": int(raytrace_render_timeout_sec),
            "use_gpu": bool(raytrace_render_use_gpu),
            "blender_processes_per_snapshot": int(
                bool(raytrace_render_enabled) and bool(raytrace_pending_jobs)
            ),
            "renders_per_blender_process": int(len(raytrace_pending_jobs)),
        },
        "icl_curve_raw": {
            int(k): {key: float(val) for key, val in metrics.items()}
            for k, metrics in icl_curve.items()
        },
        "icl_feature_prep": icl_prep,
        "icl_num_samples": int(icl_num_samples),
    }
