import argparse
from dataclasses import dataclass
import hashlib
import json
import os
import re
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Tuple
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

sys.path.append(os.getcwd())

from src.data_utils.data_module import RealPointCloudDataModule, SyntheticPointCloudDataModule
from src.training_methods.contrastive_learning.cluster_figure_utils import (
    _build_cluster_color_map,
    _save_horizontal_image_gallery,
    _save_fixed_k_cluster_figure_set,
)
from src.training_methods.contrastive_learning._cluster_rendering import (
    _build_cluster_representative_render_cache,
)
from src.training_methods.contrastive_learning.real_md_qualitative_analysis import (
    run_real_md_qualitative_analysis,
)
from src.training_methods.contrastive_learning.analysis_utils import (
    _sample_indices,
    build_real_coords_dataloader,
    evaluate_latent_equivariance,
    gather_inference_batches,
)
from src.training_methods.contrastive_learning.contrastive_module import BarlowTwinsModule
from src.utils.model_utils import load_model_from_checkpoint, resolve_config_path
from src.vis_tools.md_cluster_plot import (
    save_interactive_md_plot,
)
from src.vis_tools.latent_analysis_vis import (
    _prepare_clustering_features,
    compute_hdbscan_labels,
    compute_kmeans_labels,
    save_equivariance_plot,
    save_latent_statistics,
    save_local_structure_assignments,
    save_md_space_clusters_plot,
    save_pca_visualization,
    save_tsne_plot_with_coords,
)
from src.vis_tools.tsne_vis import compute_tsne, save_tsne_plot


@dataclass(frozen=True)
class HDBSCANSettings:
    enabled: bool
    fit_fraction: float
    max_fit_samples: int
    target_k_min: int
    target_k_max: int
    min_samples: int | None
    min_samples_candidates: list[int] | None
    cluster_selection_epsilon: float
    cluster_selection_method: str
    min_cluster_size_candidates: list[int] | None
    refit_full_data: bool


@dataclass(frozen=True)
class AnalysisSettings:
    tsne_max_samples: int
    cluster_method: str
    cluster_l2_normalize: bool
    cluster_standardize: bool
    cluster_pca_var: float
    cluster_pca_max_components: int
    cluster_k_values: list[int]
    data_overlap_fraction: float
    md_overlap_fraction: float
    md_use_all_points: bool
    progress_every_batches: int
    inference_cache_enabled: bool
    inference_cache_force_recompute: bool
    inference_cache_file: str
    seed_base: int
    hdbscan: HDBSCANSettings


@dataclass(frozen=True)
class FigureSetSettings:
    enabled: bool
    figure_only: bool
    k: int
    md_max_points: int | None
    md_point_size: float
    md_alpha: float
    md_halo_scale: float
    md_halo_alpha: float
    md_saturation_boost: float
    md_view_elev: float
    md_view_azim: float
    visible_cluster_sets: list[list[int]] | None
    cluster_color_assignment: dict[int, int | str] | None
    profile_point_scale_enabled: bool
    icl_enabled: bool
    icl_k_min: int
    icl_k_max: int
    icl_max_samples: int | None
    icl_covariance: str
    representative_points: int
    representative_orientation: str
    representative_view_elev: float
    representative_view_azim: float
    representative_projection: str
    representative_ptm_enabled: bool
    representative_cna_enabled: bool
    representative_cna_max_signatures: int
    representative_center_atom_tolerance: float
    representative_shell_min_neighbors: int
    representative_shell_max_neighbors: int
    real_md_profile_target_points: int
    raytrace_enabled: bool
    raytrace_kwargs: dict[str, Any]

    def build_run_kwargs(
        self,
        *,
        dataset: Any,
        latents: np.ndarray,
        coords: np.ndarray,
        point_scale: float,
        random_state: int,
        l2_normalize: bool,
        standardize: bool,
        pca_variance: float | None,
        pca_max_components: int,
    ) -> dict[str, Any]:
        return {
            "dataset": dataset,
            "latents": latents,
            "coords": coords,
            "k_value": int(self.k),
            "point_scale": float(point_scale),
            "l2_normalize": bool(l2_normalize),
            "standardize": bool(standardize),
            "pca_variance": pca_variance,
            "pca_max_components": int(pca_max_components),
            "md_max_points": self.md_max_points,
            "icl_enabled": bool(self.icl_enabled),
            "icl_k_min": int(self.icl_k_min),
            "icl_k_max": int(self.icl_k_max),
            "icl_max_samples": self.icl_max_samples,
            "icl_covariance_type": str(self.icl_covariance),
            "representative_points": int(self.representative_points),
            "md_point_size": float(self.md_point_size),
            "md_point_alpha": float(self.md_alpha),
            "md_halo_scale": float(self.md_halo_scale),
            "md_halo_alpha": float(self.md_halo_alpha),
            "md_saturation_boost": float(self.md_saturation_boost),
            "md_view_elev": float(self.md_view_elev),
            "md_view_azim": float(self.md_view_azim),
            "representative_orientation_method": str(self.representative_orientation),
            "representative_view_elev": float(self.representative_view_elev),
            "representative_view_azim": float(self.representative_view_azim),
            "representative_projection": str(self.representative_projection),
            "representative_ptm_enabled": bool(self.representative_ptm_enabled),
            "representative_cna_enabled": bool(self.representative_cna_enabled),
            "representative_cna_max_signatures": int(self.representative_cna_max_signatures),
            "representative_center_atom_tolerance": float(
                self.representative_center_atom_tolerance
            ),
            "representative_shell_min_neighbors": int(
                self.representative_shell_min_neighbors
            ),
            "representative_shell_max_neighbors": int(
                self.representative_shell_max_neighbors
            ),
            "visible_cluster_sets": self.visible_cluster_sets,
            "cluster_color_assignment": self.cluster_color_assignment,
            "random_state": int(random_state),
            "raytrace_render_enabled": bool(self.raytrace_enabled),
            **self.raytrace_kwargs,
        }


@dataclass(frozen=True)
class HDBSCANResult:
    labels: np.ndarray | None
    info: dict[str, Any] | None
    color_map: dict[int, str] | None


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


def _validate_overlap_fraction(value: Any) -> float:
    overlap = float(value)
    if overlap < 0.0 or overlap >= 1.0:
        raise ValueError(f"overlap_fraction must be in [0, 1), got {overlap}.")
    return overlap


def _resolve_point_scale(cfg: Any) -> float:
    data_cfg = getattr(cfg, "data", None)
    if data_cfg is None:
        return 1.0
    if not bool(getattr(data_cfg, "normalize", True)):
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


def _resolve_cluster_k_values(k_values: list[int], *, n_samples: int) -> list[int]:
    resolved = [
        max(2, min(int(k), int(n_samples)))
        for k in k_values
        if int(k) >= 2
    ]
    resolved = list(dict.fromkeys(resolved))
    if resolved:
        return resolved
    return [max(2, min(int(n_samples), 3))]


def _resolve_optional_cluster_k(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    resolved = int(value)
    if resolved < 2:
        raise ValueError(f"{field_name} must be >= 2, got {resolved}.")
    return resolved


def _apply_unified_cluster_k_override(
    cfg: DictConfig,
    *,
    cluster_k_override: int | None,
) -> int | None:
    resolved_k = _resolve_optional_cluster_k(
        cluster_k_override,
        field_name="cluster_k",
    )
    if resolved_k is None:
        resolved_k = _resolve_optional_cluster_k(
            getattr(cfg, "analysis_cluster_k", None),
            field_name="analysis_cluster_k",
        )
    if resolved_k is None:
        return None

    existing_k_values = _as_list_of_int(
        getattr(cfg, "analysis_cluster_k_values", None),
        field_name="analysis_cluster_k_values",
    )
    ordered_k_values = (
        [int(resolved_k)]
        if existing_k_values is None
        else [int(resolved_k)] + [int(k) for k in existing_k_values if int(k) != int(resolved_k)]
    )
    with open_dict(cfg):
        cfg.analysis_cluster_k = int(resolved_k)
        cfg.analysis_cluster_k_values = ordered_k_values
        cfg.analysis_cluster_figure_set_k = int(resolved_k)
        cfg.analysis_real_md_k = int(resolved_k)
    return int(resolved_k)


def _resolve_analysis_files(
    cfg: DictConfig,
    *,
    data_files_override: list[str] | None,
) -> list[str] | None:
    if cfg.data.kind != "real":
        return None
    if data_files_override:
        return data_files_override

    canonical_files = _as_list_of_str(
        OmegaConf.select(cfg, "data.analysis_data_files", default=None)
    )
    if canonical_files:
        return canonical_files

    data_files = _as_list_of_str(OmegaConf.select(cfg, "data.data_files", default=None))
    if not data_files:
        raise ValueError(
            "Cannot resolve analysis data files: data.data_files is missing or empty, "
            "and no analysis-specific file override was provided."
        )
    return data_files


def _resolve_analysis_settings(
    cfg: DictConfig,
    *,
    max_samples_visualization: int | None,
) -> AnalysisSettings:
    tsne_max_samples = int(getattr(cfg, "analysis_tsne_max_samples", 8000))
    if max_samples_visualization is not None:
        tsne_max_samples = min(tsne_max_samples, max_samples_visualization)

    cluster_k_values_cfg = _as_list_of_int(
        getattr(cfg, "analysis_cluster_k_values", None),
        field_name="analysis_cluster_k_values",
    )
    cluster_k_values = cluster_k_values_cfg or [3, 4, 5, 6]
    cluster_k_values = [int(k) for k in cluster_k_values if int(k) >= 2]
    cluster_k_values = list(dict.fromkeys(cluster_k_values))
    if not cluster_k_values:
        cluster_k_values = [3, 4, 5, 6]

    data_overlap_fraction = _validate_overlap_fraction(
        getattr(cfg.data, "overlap_fraction", 0.0)
    )
    analysis_md_overlap_boost = float(getattr(cfg, "analysis_md_overlap_boost", 0.25))
    analysis_md_overlap_fraction_raw = getattr(cfg, "analysis_md_overlap_fraction", None)
    if analysis_md_overlap_fraction_raw is None:
        md_overlap_fraction = min(0.95, data_overlap_fraction + analysis_md_overlap_boost)
    else:
        md_overlap_fraction = _validate_overlap_fraction(analysis_md_overlap_fraction_raw)
    cfg.data.overlap_fraction = float(md_overlap_fraction)

    hdbscan_min_samples = getattr(cfg, "analysis_hdbscan_min_samples", None)
    if hdbscan_min_samples is not None:
        hdbscan_min_samples = int(hdbscan_min_samples)
    hdbscan_min_samples_candidates_raw = getattr(
        cfg,
        "analysis_hdbscan_min_samples_candidates",
        None,
    )
    hdbscan_min_samples_candidates = (
        [int(v) for v in hdbscan_min_samples_candidates_raw]
        if hdbscan_min_samples_candidates_raw is not None
        else None
    )
    if hdbscan_min_samples_candidates is not None and len(hdbscan_min_samples_candidates) == 0:
        hdbscan_min_samples_candidates = None
    hdbscan_min_cluster_size_candidates_raw = getattr(
        cfg,
        "analysis_hdbscan_min_cluster_size_candidates",
        None,
    )
    hdbscan_min_cluster_size_candidates = (
        [int(v) for v in hdbscan_min_cluster_size_candidates_raw]
        if hdbscan_min_cluster_size_candidates_raw is not None
        else None
    )
    hdbscan = HDBSCANSettings(
        enabled=bool(getattr(cfg, "analysis_hdbscan_enabled", True)),
        fit_fraction=float(getattr(cfg, "analysis_hdbscan_fit_fraction", 0.75)),
        max_fit_samples=int(getattr(cfg, "analysis_hdbscan_max_fit_samples", 50000)),
        target_k_min=int(getattr(cfg, "analysis_hdbscan_target_k_min", 5)),
        target_k_max=int(getattr(cfg, "analysis_hdbscan_target_k_max", 6)),
        min_samples=hdbscan_min_samples,
        min_samples_candidates=hdbscan_min_samples_candidates,
        cluster_selection_epsilon=float(
            getattr(cfg, "analysis_hdbscan_cluster_selection_epsilon", 0.0)
        ),
        cluster_selection_method=str(
            getattr(cfg, "analysis_hdbscan_cluster_selection_method", "auto")
        ).lower(),
        min_cluster_size_candidates=hdbscan_min_cluster_size_candidates,
        refit_full_data=bool(getattr(cfg, "analysis_hdbscan_refit_full_data", True)),
    )

    inference_cache_file = str(
        getattr(cfg, "analysis_inference_cache_file", "analysis_inference_cache.npz")
    ).strip()
    if inference_cache_file == "":
        raise ValueError("analysis_inference_cache_file must be a non-empty file name.")

    return AnalysisSettings(
        tsne_max_samples=tsne_max_samples,
        cluster_method=str(getattr(cfg, "analysis_cluster_method", "auto")).lower(),
        cluster_l2_normalize=bool(getattr(cfg, "analysis_cluster_l2_normalize", True)),
        cluster_standardize=bool(getattr(cfg, "analysis_cluster_standardize", True)),
        cluster_pca_var=float(getattr(cfg, "analysis_cluster_pca_var", 0.98)),
        cluster_pca_max_components=int(
            getattr(cfg, "analysis_cluster_pca_max_components", 32)
        ),
        cluster_k_values=cluster_k_values,
        data_overlap_fraction=data_overlap_fraction,
        md_overlap_fraction=float(md_overlap_fraction),
        md_use_all_points=bool(getattr(cfg, "analysis_md_use_all_points", True)),
        progress_every_batches=int(getattr(cfg, "analysis_progress_every_batches", 25)),
        inference_cache_enabled=bool(
            getattr(cfg, "analysis_inference_cache_enabled", True)
        ),
        inference_cache_force_recompute=bool(
            getattr(cfg, "analysis_inference_cache_force_recompute", False)
        ),
        inference_cache_file=inference_cache_file,
        seed_base=int(getattr(cfg, "analysis_seed_base", 123)),
        hdbscan=hdbscan,
    )


def _resolve_figure_set_settings(
    cfg: DictConfig,
    *,
    out_dir: Path,
    visible_cluster_sets: list[list[int]] | None,
    cluster_color_assignment: dict[int, int | str] | None,
    cluster_color_assignment_file: str | None,
    cluster_figure_only: bool,
    md_render_saturation_boost: float | None,
    raytrace_render_enabled: bool | None,
) -> FigureSetSettings:
    figure_only = bool(
        cluster_figure_only or getattr(cfg, "analysis_cluster_figure_only", False)
    )
    enabled = bool(getattr(cfg, "analysis_cluster_figure_set_enabled", True)) or figure_only
    cluster_k = max(2, int(getattr(cfg, "analysis_cluster_figure_set_k", 6)))
    md_saturation_cfg = float(
        getattr(cfg, "analysis_cluster_figure_md_saturation_boost", 1.18)
    )
    md_saturation_boost = float(
        md_saturation_cfg
        if md_render_saturation_boost is None
        else md_render_saturation_boost
    )
    raytrace_enabled_use = bool(
        getattr(cfg, "analysis_cluster_figure_raytrace_enabled", False)
        if raytrace_render_enabled is None
        else raytrace_render_enabled
    )
    raytrace_kwargs = {
        "raytrace_blender_executable": str(
            getattr(cfg, "analysis_cluster_figure_raytrace_blender_executable")
        ).strip(),
        "raytrace_render_resolution": int(
            getattr(cfg, "analysis_cluster_figure_raytrace_resolution")
        ),
        "raytrace_render_max_points": _positive_int_or_none(
            getattr(cfg, "analysis_cluster_figure_raytrace_max_points", None)
        ),
        "raytrace_render_samples": int(
            getattr(cfg, "analysis_cluster_figure_raytrace_samples")
        ),
        "raytrace_render_projection": str(
            getattr(cfg, "analysis_cluster_figure_raytrace_projection")
        ).strip().lower(),
        "raytrace_render_fov_deg": float(
            getattr(cfg, "analysis_cluster_figure_raytrace_fov_deg")
        ),
        "raytrace_render_camera_distance_factor": float(
            getattr(cfg, "analysis_cluster_figure_raytrace_camera_distance_factor")
        ),
        "raytrace_render_sphere_radius_fraction": float(
            getattr(cfg, "analysis_cluster_figure_raytrace_sphere_radius_fraction")
        ),
        "raytrace_render_timeout_sec": int(
            getattr(cfg, "analysis_cluster_figure_raytrace_timeout_sec")
        ),
        "raytrace_render_use_gpu": bool(
            getattr(cfg, "analysis_cluster_figure_raytrace_use_gpu", False)
        ),
        "raytrace_parallel_views": bool(
            getattr(cfg, "analysis_cluster_figure_raytrace_parallel_views", False)
        ),
        "raytrace_parallel_max_workers": _positive_int_or_none(
            getattr(cfg, "analysis_cluster_figure_raytrace_parallel_max_workers", None)
        ),
    }

    visible_cluster_sets_use = visible_cluster_sets
    cfg_visible_sets_raw = getattr(cfg, "analysis_cluster_figure_visible_sets", None)
    if cfg_visible_sets_raw is not None and visible_cluster_sets_use is None:
        visible_cluster_sets_use = [
            [int(v) for v in str(cluster_set).split(",")]
            for cluster_set in cfg_visible_sets_raw
        ]

    cluster_color_assignment_cfg = _normalize_cluster_color_assignment(
        OmegaConf.select(cfg, "analysis_cluster_figure_color_assignment", default=None),
        field_name="analysis_cluster_figure_color_assignment",
    )
    cluster_color_assignment_file_cfg = OmegaConf.select(
        cfg,
        "analysis_cluster_figure_color_assignment_file",
        default=None,
    )
    cluster_color_assignment_cfg_file = (
        _load_cluster_color_assignment_file(
            str(cluster_color_assignment_file_cfg),
            base_dir=out_dir,
        )
        if cluster_color_assignment_file_cfg is not None
        else None
    )
    cluster_color_assignment_cli = _normalize_cluster_color_assignment(
        cluster_color_assignment,
        field_name="cluster_color_assignment",
    )
    cluster_color_assignment_cli_file = (
        _load_cluster_color_assignment_file(
            str(cluster_color_assignment_file),
            base_dir=out_dir,
        )
        if cluster_color_assignment_file is not None
        else None
    )
    cluster_color_assignment_use = _merge_cluster_color_assignments(
        cluster_color_assignment_cfg_file,
        cluster_color_assignment_cfg,
        cluster_color_assignment_cli_file,
        cluster_color_assignment_cli,
    )

    representative_points_default = int(
        getattr(
            cfg.data,
            "model_points",
            getattr(cfg.data, "num_points", 48),
        )
    )
    representative_points = max(
        16,
        int(
            getattr(
                cfg,
                "analysis_cluster_figure_representative_points",
                representative_points_default,
            )
        ),
    )
    representative_orientation = str(
        getattr(cfg, "analysis_cluster_figure_representative_orientation", "pca")
    ).strip().lower()
    if representative_orientation not in {"pca", "none"}:
        raise ValueError(
            "analysis_cluster_figure_representative_orientation must be one of "
            "['pca', 'none'], got "
            f"{representative_orientation!r}."
        )
    representative_cna_max_signatures = int(
        getattr(cfg, "analysis_cluster_figure_representative_cna_max_signatures", 5)
    )
    representative_shell_min_neighbors = int(
        getattr(
            cfg,
            "analysis_cluster_figure_representative_shell_min_neighbors",
            8,
        )
    )
    representative_shell_max_neighbors = int(
        getattr(
            cfg,
            "analysis_cluster_figure_representative_shell_max_neighbors",
            24,
        )
    )
    if representative_cna_max_signatures <= 0:
        raise ValueError(
            "analysis_cluster_figure_representative_cna_max_signatures must be > 0, "
            f"got {representative_cna_max_signatures}."
        )
    if representative_shell_min_neighbors < 2:
        raise ValueError(
            "analysis_cluster_figure_representative_shell_min_neighbors must be >= 2, "
            f"got {representative_shell_min_neighbors}."
        )
    if representative_shell_max_neighbors <= representative_shell_min_neighbors:
        raise ValueError(
            "analysis_cluster_figure_representative_shell_max_neighbors must exceed "
            "analysis_cluster_figure_representative_shell_min_neighbors, got "
            f"{representative_shell_max_neighbors} <= {representative_shell_min_neighbors}."
        )

    return FigureSetSettings(
        enabled=enabled,
        figure_only=figure_only,
        k=cluster_k,
        md_max_points=_positive_int_or_none(
            getattr(cfg, "analysis_cluster_figure_md_max_points", None)
        ),
        md_point_size=float(
            getattr(cfg, "analysis_cluster_figure_md_point_size", 5.6)
        ),
        md_alpha=float(getattr(cfg, "analysis_cluster_figure_md_alpha", 0.62)),
        md_halo_scale=float(
            getattr(cfg, "analysis_cluster_figure_md_halo_scale", 1.0)
        ),
        md_halo_alpha=float(
            getattr(cfg, "analysis_cluster_figure_md_halo_alpha", 0.0)
        ),
        md_saturation_boost=md_saturation_boost,
        md_view_elev=float(getattr(cfg, "analysis_cluster_figure_md_view_elev", 24.0)),
        md_view_azim=float(getattr(cfg, "analysis_cluster_figure_md_view_azim", 35.0)),
        visible_cluster_sets=visible_cluster_sets_use,
        cluster_color_assignment=cluster_color_assignment_use,
        profile_point_scale_enabled=bool(
            getattr(cfg, "analysis_cluster_profile_point_scale_enabled", False)
        ),
        icl_enabled=bool(getattr(cfg, "analysis_cluster_figure_icl_enabled", False)),
        icl_k_min=int(getattr(cfg, "analysis_cluster_figure_icl_k_min", 2)),
        icl_k_max=int(getattr(cfg, "analysis_cluster_figure_icl_k_max", 20)),
        icl_max_samples=_positive_int_or_none(
            getattr(cfg, "analysis_cluster_figure_icl_max_samples", 20000)
        ),
        icl_covariance=str(
            getattr(cfg, "analysis_cluster_figure_icl_covariance", "diag")
        ).lower(),
        representative_points=representative_points,
        representative_orientation=representative_orientation,
        representative_view_elev=float(
            getattr(cfg, "analysis_cluster_figure_representative_view_elev", 22.0)
        ),
        representative_view_azim=float(
            getattr(cfg, "analysis_cluster_figure_representative_view_azim", 38.0)
        ),
        representative_projection=str(
            getattr(cfg, "analysis_cluster_figure_representative_projection", "ortho")
        ).strip().lower(),
        representative_ptm_enabled=bool(
            getattr(cfg, "analysis_cluster_figure_representative_ptm_enabled", False)
        ),
        representative_cna_enabled=bool(
            getattr(cfg, "analysis_cluster_figure_representative_cna_enabled", False)
        ),
        representative_cna_max_signatures=representative_cna_max_signatures,
        representative_center_atom_tolerance=float(
            getattr(
                cfg,
                "analysis_cluster_figure_representative_center_atom_tolerance",
                1e-6,
            )
        ),
        representative_shell_min_neighbors=representative_shell_min_neighbors,
        representative_shell_max_neighbors=representative_shell_max_neighbors,
        real_md_profile_target_points=int(
            getattr(
                cfg,
                "analysis_cluster_profile_target_points",
                max(
                    32,
                    int(
                        getattr(
                            cfg.data,
                            "model_points",
                            getattr(cfg.data, "num_points", 64),
                        )
                    ),
                ),
            )
        ),
        raytrace_enabled=raytrace_enabled_use,
        raytrace_kwargs=raytrace_kwargs,
    )


def _print_resolved_analysis_settings(
    *,
    unified_cluster_k: int | None,
    analysis_settings: AnalysisSettings,
    figure_settings: FigureSetSettings,
) -> None:
    if unified_cluster_k is not None:
        print(
            "Unified analysis cluster count: "
            f"k={unified_cluster_k} "
            "(applied to analysis_cluster_k_values, "
            "analysis_cluster_figure_set_k, analysis_real_md_k)"
        )
    print(f"t-SNE sample cap: {analysis_settings.tsne_max_samples}")
    print(f"Clustering k values (configured): {analysis_settings.cluster_k_values}")
    print(
        "MD overlap fraction (analysis): "
        f"{analysis_settings.data_overlap_fraction:.3f} -> "
        f"{analysis_settings.md_overlap_fraction:.3f}"
    )
    print(
        "HDBSCAN settings: "
        f"fit_fraction={analysis_settings.hdbscan.fit_fraction:.3f}, "
        f"max_fit_samples={analysis_settings.hdbscan.max_fit_samples}, "
        f"target_k=[{analysis_settings.hdbscan.target_k_min}, "
        f"{analysis_settings.hdbscan.target_k_max}], "
        f"selection_method={analysis_settings.hdbscan.cluster_selection_method}, "
        f"refit_full_data={analysis_settings.hdbscan.refit_full_data}"
    )
    print(
        "Fixed-k figure set: "
        f"enabled={figure_settings.enabled}, "
        f"figure_only={figure_settings.figure_only}, "
        f"k={figure_settings.k}, "
        f"visible_sets={figure_settings.visible_cluster_sets or []}, "
        f"md_saturation={figure_settings.md_saturation_boost:.2f}, "
        f"raytrace_enabled={figure_settings.raytrace_enabled}, "
        "raytrace_projection="
        f"{figure_settings.raytrace_kwargs['raytrace_render_projection']}, "
        f"raytrace_samples={figure_settings.raytrace_kwargs['raytrace_render_samples']}, "
        f"raytrace_res={figure_settings.raytrace_kwargs['raytrace_render_resolution']}, "
        "raytrace_max_points="
        f"{figure_settings.raytrace_kwargs['raytrace_render_max_points']}, "
        f"raytrace_gpu={figure_settings.raytrace_kwargs['raytrace_render_use_gpu']}, "
        "raytrace_parallel_views="
        f"{figure_settings.raytrace_kwargs['raytrace_parallel_views']}, "
        "raytrace_parallel_max_workers="
        f"{figure_settings.raytrace_kwargs['raytrace_parallel_max_workers']}, "
        f"icl_enabled={figure_settings.icl_enabled}, "
        "profile_point_scale_enabled="
        f"{figure_settings.profile_point_scale_enabled}, "
        f"rep_orientation={figure_settings.representative_orientation}, "
        "rep_view=("
        f"{figure_settings.representative_view_elev:.1f},"
        f"{figure_settings.representative_view_azim:.1f}"
        "), "
        f"rep_projection={figure_settings.representative_projection}, "
        f"rep_ptm={figure_settings.representative_ptm_enabled}, "
        f"rep_cna={figure_settings.representative_cna_enabled}, "
        "rep_cna_shell=("
        f"{figure_settings.representative_shell_min_neighbors},"
        f"{figure_settings.representative_shell_max_neighbors}"
        "), "
        "cluster_color_overrides="
        f"{sorted(figure_settings.cluster_color_assignment) if figure_settings.cluster_color_assignment else []}"
    )


def _build_analysis_dataloader(
    cfg: DictConfig,
    dm: Any,
    *,
    is_synthetic: bool,
) -> torch.utils.data.DataLoader:
    print("Using ALL dataset splits (train + test) for latent analysis")
    if is_synthetic:
        train_dataset = getattr(dm, "train_dataset", None)
        test_dataset = getattr(dm, "test_dataset", None)
        if train_dataset is None or test_dataset is None:
            raise ValueError("Synthetic datamodule is missing train/test datasets.")
        combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        return torch.utils.data.DataLoader(
            combined_dataset,
            batch_size=cfg.batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True,
        )

    dl = build_real_coords_dataloader(
        cfg,
        dm,
        use_train_data=True,
        use_full_dataset=True,
        prefer_existing_full_dataset=True,
    )
    print(
        "Real data detected: using full dataset for local-structure clustering visualization"
    )
    return dl


def _resolve_analysis_max_samples_total(
    cfg: DictConfig,
    *,
    is_synthetic: bool,
    md_use_all_points: bool,
) -> int | None:
    max_samples_total = getattr(cfg, "analysis_max_samples_total", None)
    if max_samples_total is None and not is_synthetic:
        max_samples_total = 20000
    if not is_synthetic and md_use_all_points:
        max_samples_total = None
    return _positive_int_or_none(max_samples_total)


def _build_clustering_state(
    latents: np.ndarray,
    phases: np.ndarray,
    *,
    requested_k_values: list[int],
    cluster_method: str,
    random_state: int,
    l2_normalize: bool,
    standardize: bool,
    pca_variance: float | None,
    pca_max_components: int,
    prepared_features: np.ndarray | None = None,
    prep_info: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[int], Dict[int, np.ndarray], Dict[int, str]]:
    configured_k_values = _resolve_cluster_k_values(requested_k_values, n_samples=len(latents))
    cluster_labels_by_k: Dict[int, np.ndarray] = {}
    cluster_methods_by_k: Dict[int, str] = {}
    feature_prep: dict[str, Any] | None = (
        dict(prep_info)
        if prep_info is not None
        else None
    )

    for k_value in configured_k_values:
        labels_k, info_k = compute_kmeans_labels(
            latents,
            int(k_value),
            random_state=int(random_state),
            method=cluster_method,
            l2_normalize=l2_normalize,
            standardize=standardize,
            pca_variance=pca_variance,
            pca_max_components=pca_max_components,
            prepared_features=prepared_features,
            prep_info=feature_prep,
            return_info=True,
        )
        cluster_labels_by_k[int(k_value)] = labels_k
        cluster_methods_by_k[int(k_value)] = str(info_k.get("method", "kmeans"))
        if feature_prep is None:
            feature_prep = {
                key: info_k[key]
                for key in (
                    "input_dim",
                    "output_dim",
                    "l2_normalize",
                    "standardize",
                    "pca_components",
                    "pca_explained_variance",
                )
                if key in info_k
            }

    primary_k = int(configured_k_values[0])
    metrics: dict[str, Any] = {
        "cluster_method_requested": str(cluster_method).lower(),
        "random_state": int(random_state),
        "k_values_requested": [int(k) for k in requested_k_values],
        "k_values_used": [int(k) for k in configured_k_values],
        "primary_k": int(primary_k),
        "labels_k_method": str(cluster_methods_by_k[int(primary_k)]),
        "labels_method_by_k": {
            int(k): str(cluster_methods_by_k[int(k)])
            for k in configured_k_values
        },
    }
    if feature_prep:
        metrics["cluster_feature_prep"] = feature_prep

    if phases.size == len(latents):
        unique_phases = np.unique(phases)
        if unique_phases.size > 1:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

            gt_k = max(2, min(int(unique_phases.size), len(latents)))
            gt_labels = cluster_labels_by_k.get(int(gt_k))
            if gt_labels is None:
                gt_labels = compute_kmeans_labels(
                    latents,
                    int(gt_k),
                    random_state=int(random_state),
                    method=cluster_method,
                    l2_normalize=l2_normalize,
                    standardize=standardize,
                    pca_variance=pca_variance,
                    pca_max_components=pca_max_components,
                    prepared_features=prepared_features,
                    prep_info=feature_prep,
                )
            metrics["ari_with_gt"] = float(adjusted_rand_score(phases, gt_labels))
            metrics["nmi_with_gt"] = float(normalized_mutual_info_score(phases, gt_labels))

    return metrics, configured_k_values, cluster_labels_by_k, cluster_methods_by_k


def _normalize_data_sources_for_cache(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, list):
        raise ValueError(
            "cfg.data.data_sources must be a list when present, "
            f"got {type(value)!r}."
        )

    normalized: list[dict[str, Any]] = []
    for idx, src in enumerate(value):
        if not isinstance(src, dict):
            raise ValueError(
                "Each cfg.data.data_sources entry must be a dict, "
                f"got {type(src)!r} at index {idx}."
            )
        files = _as_list_of_str(src.get("data_files"))
        normalized.append(
            {
                "name": None if src.get("name") is None else str(src.get("name")),
                "data_path": str(src.get("data_path", "")),
                "data_files": files or [],
                "radius": None if src.get("radius") is None else float(src.get("radius")),
            }
        )
    return normalized


def _configure_real_analysis_inputs(
    cfg: DictConfig,
    analysis_files: list[str],
) -> list[str]:
    if getattr(cfg.data, "kind", None) != "real":
        raise ValueError(
            "_configure_real_analysis_inputs can only be used for real datasets, "
            f"got kind={getattr(cfg.data, 'kind', None)!r}."
        )
    normalized_files = [str(v) for v in analysis_files]
    if not normalized_files:
        raise ValueError("analysis_files must be a non-empty list.")

    with open_dict(cfg.data):
        cfg.data.data_files = normalized_files
        if len(normalized_files) == 1:
            cfg.data.data_sources = None
            return [normalized_files[0]]

    data_path = getattr(cfg.data, "data_path", None)
    if not data_path:
        raise ValueError(
            "cfg.data.data_path is required to split analysis outputs per snapshot, "
            f"but got data_path={data_path!r} for analysis_files={normalized_files}."
        )

    source_names: list[str] = []
    seen_names: set[str] = set()
    data_sources: list[dict[str, Any]] = []
    for file_idx, file_name in enumerate(normalized_files):
        source_name = str(file_name)
        if source_name in seen_names:
            source_name = f"{file_idx:02d}_{source_name}"
        seen_names.add(source_name)
        source_names.append(source_name)
        data_sources.append(
            {
                "name": source_name,
                "data_path": str(data_path),
                "data_files": [str(file_name)],
            }
        )
    with open_dict(cfg.data):
        cfg.data.data_sources = data_sources
    return source_names


def _unwrap_dataset_with_subset_indices(
    dataset: Any,
) -> tuple[Any, list[int] | None]:
    indices: list[int] | None = None
    while isinstance(dataset, torch.utils.data.Subset):
        current_indices = [int(v) for v in list(dataset.indices)]
        if indices is None:
            indices = current_indices
        else:
            indices = [indices[i] for i in current_indices]
        dataset = dataset.dataset
    while hasattr(dataset, "dataset") and not isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    return dataset, indices


def _resolve_sample_source_groups(
    dataset: Any,
    *,
    n_samples: int,
) -> list[tuple[str, np.ndarray]]:
    if n_samples < 0:
        raise ValueError(f"n_samples must be >= 0, got {n_samples}.")
    if n_samples == 0:
        return []

    base_dataset, subset_indices = _unwrap_dataset_with_subset_indices(dataset)
    sample_source_names_raw = getattr(base_dataset, "sample_source_names", None)
    if sample_source_names_raw is None:
        return []

    sample_source_names = [str(v) for v in list(sample_source_names_raw)]
    if subset_indices is not None:
        if any(int(i) < 0 or int(i) >= len(sample_source_names) for i in subset_indices):
            raise IndexError(
                "Subset indices reference sample_source_names out of bounds: "
                f"len(sample_source_names)={len(sample_source_names)}, "
                f"max_index={max(subset_indices) if subset_indices else 'N/A'}."
            )
        sample_source_names = [sample_source_names[int(i)] for i in subset_indices]

    if len(sample_source_names) < int(n_samples):
        raise ValueError(
            "Not enough sample_source_names to map collected analysis samples: "
            f"have {len(sample_source_names)}, need {n_samples}."
        )
    sample_source_names = sample_source_names[: int(n_samples)]

    grouped_indices: dict[str, list[int]] = {}
    for sample_idx, source_name in enumerate(sample_source_names):
        grouped_indices.setdefault(str(source_name), []).append(int(sample_idx))

    return [
        (source_name, np.asarray(indices, dtype=int))
        for source_name, indices in grouped_indices.items()
    ]


def _sanitize_snapshot_output_name(name: str) -> str:
    stem = Path(str(name)).stem or Path(str(name)).name or str(name)
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    if sanitized:
        return sanitized
    return "snapshot"


def _build_unique_snapshot_output_names(source_names: list[str]) -> dict[str, str]:
    used: set[str] = set()
    output_names: dict[str, str] = {}
    for source_name in source_names:
        base = _sanitize_snapshot_output_name(source_name)
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        output_names[str(source_name)] = candidate
    return output_names


def _resolve_visible_cluster_sets_for_labels(
    labels: np.ndarray,
    visible_cluster_sets: list[list[int]] | None,
    *,
    context: str,
) -> list[list[int]] | None:
    if not visible_cluster_sets:
        return None

    available = {
        int(v)
        for v in np.unique(np.asarray(labels, dtype=int).reshape(-1))
        if int(v) >= 0
    }
    resolved: list[list[int]] = []
    for set_idx, cluster_set in enumerate(visible_cluster_sets):
        normalized = [int(v) for v in cluster_set]
        present = [cluster_id for cluster_id in normalized if cluster_id in available]
        missing = [cluster_id for cluster_id in normalized if cluster_id not in available]
        if missing and present:
            print(
                f"[analysis] {context}: visible_cluster_sets[{set_idx}] "
                f"drops missing cluster IDs {missing}; using {present}."
            )
        elif missing:
            print(
                f"[analysis] {context}: skipping visible_cluster_sets[{set_idx}]={normalized} "
                "because none of those clusters are present in this snapshot."
            )
        if present:
            resolved.append(present)
    return resolved or None


def _save_snapshot_raytrace_galleries_by_view(
    snapshot_figure_sets: dict[str, Any],
    *,
    requested_visible_cluster_sets: list[list[int]] | None,
) -> dict[str, Any]:
    snapshots = list(snapshot_figure_sets.get("snapshots") or [])
    if not snapshots:
        raise ValueError("snapshot_figure_sets['snapshots'] must be non-empty.")
    k_value = int(snapshot_figure_sets.get("k_value", -1))
    if k_value < 2:
        raise ValueError(
            f"Invalid snapshot_figure_sets k_value={k_value}; expected an integer >= 2."
        )
    root_dir_raw = snapshot_figure_sets.get("root_dir")
    if not root_dir_raw:
        raise ValueError("snapshot_figure_sets is missing 'root_dir'.")
    gallery_root = Path(str(root_dir_raw)) / "_galleries_by_view" / f"cluster_figure_set_k{k_value}"
    gallery_root.mkdir(parents=True, exist_ok=True)
    stale_paths: set[Path] = set()
    for pattern in (
        "01_md_clusters_all_k*_view*_raytrace_gallery.png",
        "02_md_clusters_set_*_k*_view*_raytrace_gallery.png",
    ):
        stale_paths.update(gallery_root.glob(pattern))
    for stale_path in stale_paths:
        stale_path.unlink()

    def _build_view_lookup(panel_views: Any, *, context: str) -> dict[str, dict[str, Any]]:
        if not isinstance(panel_views, list) or not panel_views:
            raise RuntimeError(f"{context}: expected a non-empty list of panel views.")
        lookup: dict[str, dict[str, Any]] = {}
        for panel_idx, panel in enumerate(panel_views):
            if not isinstance(panel, dict):
                raise RuntimeError(
                    f"{context}: panel view #{panel_idx} must be a dict, got {type(panel)!r}."
                )
            view_name = str(panel.get("view_name", "")).strip()
            if not view_name:
                raise RuntimeError(
                    f"{context}: panel view #{panel_idx} is missing a non-empty 'view_name'."
                )
            if view_name in lookup:
                raise RuntimeError(f"{context}: duplicate view_name={view_name!r}.")
            lookup[view_name] = panel
        return lookup

    def _extract_raytrace_path(panel_view: dict[str, Any], *, context: str) -> Path:
        raytrace_info = panel_view.get("raytrace_render")
        if not isinstance(raytrace_info, dict):
            raise RuntimeError(
                f"{context}: missing raytrace_render metadata. "
                "Re-run with --raytrace_render_enabled."
            )
        out_file = raytrace_info.get("out_file")
        if not out_file:
            raise RuntimeError(f"{context}: raytrace_render metadata is missing 'out_file'.")
        path = Path(str(out_file))
        if not path.exists():
            raise FileNotFoundError(f"{context}: expected raytraced image at {path}, but it is missing.")
        return path

    def _snapshot_identity(snapshot_entry: dict[str, Any]) -> dict[str, str]:
        source_name = str(snapshot_entry.get("source_name", "")).strip()
        output_name = str(snapshot_entry.get("output_name", "")).strip()
        if not source_name or not output_name:
            raise RuntimeError(
                "Each snapshot entry must contain non-empty 'source_name' and 'output_name' fields."
            )
        return {
            "source_name": source_name,
            "output_name": output_name,
        }

    first_identity = _snapshot_identity(snapshots[0])
    first_figure_set = snapshots[0].get("figure_set")
    if not isinstance(first_figure_set, dict):
        raise RuntimeError(
            f"Snapshot {first_identity['source_name']} is missing figure_set metadata."
        )
    all_view_lookup = _build_view_lookup(
        first_figure_set.get("panel_all_clusters_views"),
        context=f"snapshot={first_identity['source_name']} panel_all_clusters_views",
    )
    ordered_view_names = list(all_view_lookup.keys())

    summary: dict[str, Any] = {
        "root_dir": str(gallery_root),
        "k_value": int(k_value),
        "all_clusters": [],
        "selected_sets": [],
    }

    for view_name in ordered_view_names:
        image_paths: list[Path] = []
        snapshot_records: list[dict[str, str]] = []
        for snapshot_entry in snapshots:
            identity = _snapshot_identity(snapshot_entry)
            figure_set = snapshot_entry.get("figure_set")
            if not isinstance(figure_set, dict):
                raise RuntimeError(
                    f"Snapshot {identity['source_name']} is missing figure_set metadata."
                )
            view_lookup = _build_view_lookup(
                figure_set.get("panel_all_clusters_views"),
                context=f"snapshot={identity['source_name']} panel_all_clusters_views",
            )
            if view_name not in view_lookup:
                raise RuntimeError(
                    f"Snapshot {identity['source_name']} is missing all-clusters view {view_name!r}. "
                    f"Available views: {sorted(view_lookup.keys())}."
                )
            raytrace_path = _extract_raytrace_path(
                view_lookup[view_name],
                context=f"snapshot={identity['source_name']} all-clusters {view_name}",
            )
            image_paths.append(raytrace_path)
            snapshot_records.append(
                {
                    **identity,
                    "raytrace_out_file": str(raytrace_path),
                }
            )
        gallery_info = _save_horizontal_image_gallery(
            image_paths,
            gallery_root / f"01_md_clusters_all_k{k_value}_{view_name}_raytrace_gallery.png",
        )
        gallery_info["view_name"] = str(view_name)
        gallery_info["snapshots"] = snapshot_records
        summary["all_clusters"].append(gallery_info)

    requested_sets_norm = [
        tuple(sorted(int(v) for v in cluster_set))
        for cluster_set in (requested_visible_cluster_sets or [])
    ]
    for cluster_ids in requested_sets_norm:
        tag = "-".join(str(cluster_id) for cluster_id in cluster_ids)
        for view_name in ordered_view_names:
            image_paths = []
            included_snapshots: list[dict[str, str]] = []
            missing_snapshots: list[dict[str, str]] = []
            for snapshot_entry in snapshots:
                identity = _snapshot_identity(snapshot_entry)
                figure_set = snapshot_entry.get("figure_set")
                if not isinstance(figure_set, dict):
                    raise RuntimeError(
                        f"Snapshot {identity['source_name']} is missing figure_set metadata."
                    )
                selected_sets = figure_set.get("panel_selected_sets")
                if not isinstance(selected_sets, list):
                    raise RuntimeError(
                        f"Snapshot {identity['source_name']} is missing panel_selected_sets metadata."
                    )
                matched_set: dict[str, Any] | None = None
                for panel_set in selected_sets:
                    if not isinstance(panel_set, dict):
                        raise RuntimeError(
                            f"Snapshot {identity['source_name']} has a non-dict selected-set panel."
                        )
                    shown_ids = tuple(
                        sorted(int(v) for v in panel_set.get("cluster_ids_shown", []))
                    )
                    if shown_ids == cluster_ids:
                        matched_set = panel_set
                        break
                if matched_set is None:
                    missing_snapshots.append(identity)
                    continue
                set_view_lookup = _build_view_lookup(
                    matched_set.get("views"),
                    context=(
                        f"snapshot={identity['source_name']} "
                        f"panel_selected_sets[{tag}] views"
                    ),
                )
                if view_name not in set_view_lookup:
                    raise RuntimeError(
                        f"Snapshot {identity['source_name']} is missing selected-set view {view_name!r} "
                        f"for cluster set {cluster_ids}. Available views: {sorted(set_view_lookup.keys())}."
                    )
                raytrace_path = _extract_raytrace_path(
                    set_view_lookup[view_name],
                    context=(
                        f"snapshot={identity['source_name']} "
                        f"cluster_set={cluster_ids} {view_name}"
                    ),
                )
                image_paths.append(raytrace_path)
                included_snapshots.append(
                    {
                        **identity,
                        "raytrace_out_file": str(raytrace_path),
                    }
                )
            if not image_paths:
                print(
                    "[analysis] Skipping cross-snapshot raytrace gallery for "
                    f"cluster set {list(cluster_ids)} ({view_name}) because no snapshot "
                    "produced that exact selected-set panel."
                )
                summary["selected_sets"].append(
                    {
                        "cluster_ids": [int(v) for v in cluster_ids],
                        "view_name": str(view_name),
                        "out_file": None,
                        "snapshots": [],
                        "snapshots_missing": missing_snapshots,
                    }
                )
                continue
            if missing_snapshots:
                missing_names = [entry["source_name"] for entry in missing_snapshots]
                print(
                    "[analysis] Cross-snapshot raytrace gallery for "
                    f"cluster set {list(cluster_ids)} ({view_name}) excludes snapshots "
                    f"without that exact selected-set panel: {missing_names}"
                )
            gallery_info = _save_horizontal_image_gallery(
                image_paths,
                gallery_root
                / f"02_md_clusters_set_{tag}_k{k_value}_{view_name}_raytrace_gallery.png",
            )
            gallery_info["cluster_ids"] = [int(v) for v in cluster_ids]
            gallery_info["view_name"] = str(view_name)
            gallery_info["snapshots"] = included_snapshots
            gallery_info["snapshots_missing"] = missing_snapshots
            summary["selected_sets"].append(gallery_info)
    return summary


def _ordered_md_k_values(
    k_values_order: list[int],
    *,
    primary_k: int,
) -> list[int]:
    ordered = [int(primary_k)] + [int(k) for k in k_values_order if int(k) != int(primary_k)]
    return list(dict.fromkeys(ordered))


def _build_md_plot_title(
    labels_for_k: np.ndarray,
    *,
    k_value: int,
    sample_count: int,
    source_name: str | None = None,
) -> str:
    labels_arr = np.asarray(labels_for_k, dtype=int).reshape(-1)
    visible_clusters = int(len(np.unique(labels_arr)))
    parts = [f"k={int(k_value)}", f"n={int(sample_count)}"]
    if visible_clusters != int(k_value):
        parts.append(f"visible={visible_clusters}")
    if source_name is not None:
        parts.append(f"snapshot={source_name}")
    return f"MD local-structure clusters ({', '.join(parts)})"


def _save_md_cluster_outputs(
    coords: np.ndarray,
    labels_by_k: dict[int, np.ndarray],
    out_dir: Path,
    *,
    k_values_order: list[int],
    primary_k: int,
    shared_cluster_color_maps_by_k: dict[int, dict[int, str]],
    max_points: int | None,
    source_name: str | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered_k_values = _ordered_md_k_values(k_values_order, primary_k=int(primary_k))
    if not ordered_k_values:
        raise ValueError("Cannot save MD cluster outputs: no k values were provided.")

    coords_arr = np.asarray(coords, dtype=np.float32)
    if coords_arr.ndim != 2 or coords_arr.shape[0] < 2 or coords_arr.shape[1] < 3:
        raise ValueError(
            "Cannot save MD cluster outputs: expected coords with shape (n, d), "
            f"n >= 2, d >= 3, got shape={coords_arr.shape}."
        )

    stale_patterns = (
        "md_space_clusters.png",
        "md_space_clusters.html",
        "md_space_clusters_k*.png",
        "md_space_clusters_k*.html",
    )
    stale_paths: set[Path] = set()
    for pattern in stale_patterns:
        stale_paths.update(out_dir.glob(pattern))
    for stale_path in stale_paths:
        stale_path.unlink()

    static_pngs: dict[int, str] = {}
    interactive_htmls: dict[int, str] = {}
    for k_value in ordered_k_values:
        if int(k_value) not in labels_by_k:
            raise KeyError(
                "Cannot save MD cluster outputs because labels are missing for "
                f"k={int(k_value)}. Available k values: {sorted(int(k) for k in labels_by_k)}."
            )
        if int(k_value) not in shared_cluster_color_maps_by_k:
            raise KeyError(
                "Cannot save MD cluster outputs because the shared color map is missing for "
                f"k={int(k_value)}. Available k values: "
                f"{sorted(int(k) for k in shared_cluster_color_maps_by_k)}."
            )
        labels_arr = np.asarray(labels_by_k[int(k_value)], dtype=int).reshape(-1)
        if labels_arr.shape[0] != coords_arr.shape[0]:
            raise ValueError(
                "Cannot save MD cluster outputs because coords/labels lengths do not match. "
                f"k={int(k_value)}, coords_shape={coords_arr.shape}, labels_shape={labels_arr.shape}."
            )
        title = _build_md_plot_title(
            labels_arr,
            k_value=int(k_value),
            sample_count=int(coords_arr.shape[0]),
            source_name=source_name,
        )
        static_path = out_dir / f"md_space_clusters_k{int(k_value)}.png"
        save_md_space_clusters_plot(
            coords_arr,
            labels_arr,
            static_path,
            cluster_color_map=shared_cluster_color_maps_by_k[int(k_value)],
            max_points=max_points,
            title=title,
        )
        static_pngs[int(k_value)] = str(static_path)

    primary_static_path = out_dir / f"md_space_clusters_k{int(primary_k)}.png"
    primary_static_alias = out_dir / "md_space_clusters.png"
    shutil.copyfile(primary_static_path, primary_static_alias)

    results: dict[str, Any] = {
        "static_png": str(primary_static_alias),
        "static_pngs": static_pngs,
    }

    try:
        for k_value in ordered_k_values:
            labels_arr = np.asarray(labels_by_k[int(k_value)], dtype=int).reshape(-1)
            title = _build_md_plot_title(
                labels_arr,
                k_value=int(k_value),
                sample_count=int(coords_arr.shape[0]),
                source_name=source_name,
            )
            interactive_path = out_dir / f"md_space_clusters_k{int(k_value)}.html"
            save_interactive_md_plot(
                coords_arr,
                labels_arr,
                interactive_path,
                palette="tab10",
                cluster_color_map=shared_cluster_color_maps_by_k[int(k_value)],
                max_points=max_points,
                marker_size=3.0,
                marker_line_width=0.0,
                title=title,
                aspect_mode="cube",
            )
            interactive_htmls[int(k_value)] = str(interactive_path)
        primary_interactive_path = out_dir / f"md_space_clusters_k{int(primary_k)}.html"
        primary_interactive_alias = out_dir / "md_space_clusters.html"
        shutil.copyfile(primary_interactive_path, primary_interactive_alias)
        results["interactive_html"] = str(primary_interactive_alias)
        results["interactive_htmls"] = interactive_htmls
    except ImportError:
        context = "full dataset" if source_name is None else f"snapshot {source_name}"
        print(f"Plotly not installed; skipping interactive MD plot for {context}.")

    return results


def _run_optional_hdbscan_analysis(
    latents: np.ndarray,
    *,
    coords_count: int,
    settings: HDBSCANSettings,
    random_state: int,
    l2_normalize: bool,
    standardize: bool,
    pca_variance: float | None,
    pca_max_components: int,
    prepared_features: np.ndarray | None,
    prep_info: dict[str, Any] | None,
    cluster_color_assignment: dict[int, int | str] | None,
    step: Callable[[str], None],
) -> HDBSCANResult:
    if not settings.enabled:
        return HDBSCANResult(labels=None, info=None, color_map=None)

    step("Running HDBSCAN clustering (sampled fit)")
    try:
        hdbscan_labels, hdbscan_info = compute_hdbscan_labels(
            latents,
            sample_fraction=settings.fit_fraction,
            max_fit_samples=settings.max_fit_samples,
            random_state=random_state,
            l2_normalize=l2_normalize,
            standardize=standardize,
            pca_variance=pca_variance,
            pca_max_components=pca_max_components,
            target_clusters_min=settings.target_k_min,
            target_clusters_max=settings.target_k_max,
            min_cluster_size_candidates=settings.min_cluster_size_candidates,
            min_samples=settings.min_samples,
            min_samples_candidates=settings.min_samples_candidates,
            cluster_selection_epsilon=settings.cluster_selection_epsilon,
            cluster_selection_method=settings.cluster_selection_method,
            refit_full_data=settings.refit_full_data,
            prepared_features=prepared_features,
            prep_info=prep_info,
            return_info=True,
        )
        n_hdb_clusters_full = int(hdbscan_info.get("n_clusters_full", -1))
        if (
            settings.cluster_selection_method != "auto"
            and n_hdb_clusters_full >= 0
            and n_hdb_clusters_full < settings.target_k_min
        ):
            print(
                "[analysis][hdbscan] cluster count below target "
                f"({n_hdb_clusters_full} < {settings.target_k_min}); "
                "retrying with cluster_selection_method='auto'."
            )
            hdbscan_labels_retry, hdbscan_info_retry = compute_hdbscan_labels(
                latents,
                sample_fraction=settings.fit_fraction,
                max_fit_samples=settings.max_fit_samples,
                random_state=random_state,
                l2_normalize=l2_normalize,
                standardize=standardize,
                pca_variance=pca_variance,
                pca_max_components=pca_max_components,
                target_clusters_min=settings.target_k_min,
                target_clusters_max=settings.target_k_max,
                min_cluster_size_candidates=settings.min_cluster_size_candidates,
                min_samples=settings.min_samples,
                min_samples_candidates=settings.min_samples_candidates,
                cluster_selection_epsilon=settings.cluster_selection_epsilon,
                cluster_selection_method="auto",
                refit_full_data=settings.refit_full_data,
                prepared_features=prepared_features,
                prep_info=prep_info,
                return_info=True,
            )
            retry_clusters = int(hdbscan_info_retry.get("n_clusters_full", -1))
            retry_noise = float(hdbscan_info_retry.get("noise_fraction_full", 1.0))
            base_noise = float(hdbscan_info.get("noise_fraction_full", 1.0))
            if (
                retry_clusters > n_hdb_clusters_full
                or (retry_clusters == n_hdb_clusters_full and retry_noise < base_noise)
            ):
                hdbscan_labels = hdbscan_labels_retry
                hdbscan_info = hdbscan_info_retry
                print(
                    "[analysis][hdbscan] using retry result: "
                    f"clusters={retry_clusters}, noise={retry_noise:.4f}."
                )
        if hdbscan_labels.size != int(coords_count):
            print(
                "Warning: HDBSCAN labels do not match coordinate count; "
                "skipping HDBSCAN MD outputs."
            )
            return HDBSCANResult(labels=None, info=hdbscan_info, color_map=None)

        valid_hdbscan = hdbscan_labels[hdbscan_labels >= 0]
        if valid_hdbscan.size > 0:
            hdbscan_color_map = {
                int(cluster_id): str(color)
                for cluster_id, color in _build_cluster_color_map(
                    hdbscan_labels,
                    cluster_color_assignment=cluster_color_assignment,
                ).items()
            }
        else:
            hdbscan_color_map = {}
        if np.any(hdbscan_labels < 0):
            hdbscan_color_map[-1] = "lightgray"
        return HDBSCANResult(
            labels=np.asarray(hdbscan_labels, dtype=int),
            info=hdbscan_info,
            color_map=hdbscan_color_map,
        )
    except ImportError:
        print("HDBSCAN package not installed; skipping HDBSCAN analysis.")
        return HDBSCANResult(labels=None, info=None, color_map=None)


def _save_hdbscan_md_outputs(
    coords: np.ndarray,
    hdbscan_labels: np.ndarray | None,
    out_dir: Path,
    *,
    hdbscan_color_map: dict[int, str] | None,
    max_points: int | None,
    title: str,
    context_label: str,
) -> dict[str, Any]:
    if hdbscan_labels is None:
        return {}

    outputs: dict[str, Any] = {}
    hdbscan_coord_files = save_local_structure_assignments(
        coords,
        hdbscan_labels,
        out_dir,
        prefix="local_structure_hdbscan",
    )
    if hdbscan_coord_files:
        outputs["hdbscan_coords_files"] = hdbscan_coord_files
    try:
        hdbscan_path = Path(out_dir) / "md_space_clusters_hdbscan.html"
        n_hdb_clusters = int(len(np.unique(hdbscan_labels[hdbscan_labels >= 0])))
        save_interactive_md_plot(
            coords,
            hdbscan_labels,
            hdbscan_path,
            palette="tab10",
            cluster_color_map=hdbscan_color_map,
            max_points=max_points,
            marker_size=3.0,
            marker_line_width=0.0,
            title=f"{title}, k={n_hdb_clusters})",
            label_prefix="HDBSCAN",
            aspect_mode="cube",
        )
        outputs["hdbscan_interactive_html"] = str(hdbscan_path)
    except ImportError:
        print(
            "Plotly not installed; skipping HDBSCAN interactive MD plot "
            f"for {context_label}."
        )
    return outputs


def _save_snapshot_md_outputs(
    *,
    out_dir: Path,
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_labels_by_k: dict[int, np.ndarray],
    configured_k_values: list[int],
    primary_k: int,
    snapshot_source_groups: list[tuple[str, np.ndarray]],
    snapshot_output_names: dict[str, str],
    shared_cluster_color_maps_by_k: dict[int, dict[int, str]],
    interactive_max_points: int | None,
    hdbscan_result: HDBSCANResult,
) -> dict[str, Any]:
    snapshot_root = Path(out_dir) / "md_clusters_by_snapshot"
    summary: dict[str, Any] = {
        "root_dir": str(snapshot_root),
        "primary_k": int(primary_k),
        "k_values_used": [int(k) for k in configured_k_values],
        "snapshots": [],
    }
    for source_name, indices in snapshot_source_groups:
        snapshot_dirname = snapshot_output_names[str(source_name)]
        snapshot_dir = snapshot_root / snapshot_dirname
        coords_subset = coords[indices]
        labels_subset = cluster_labels[indices]
        coord_files = save_local_structure_assignments(
            coords_subset,
            labels_subset,
            snapshot_dir,
        )
        if not coord_files:
            raise RuntimeError(
                "Failed to save per-snapshot MD cluster assignments. "
                f"snapshot={source_name}, coords_shape={coords_subset.shape}, "
                f"labels_shape={labels_subset.shape}, out_dir={snapshot_dir}."
            )
        snapshot_record: dict[str, Any] = {
            "source_name": str(source_name),
            "output_name": str(snapshot_dirname),
            "sample_count": int(indices.size),
            "coords_files": coord_files,
        }
        snapshot_record.update(
            _save_md_cluster_outputs(
                coords_subset,
                {
                    int(k_value): cluster_labels_by_k[int(k_value)][indices]
                    for k_value in configured_k_values
                },
                snapshot_dir,
                k_values_order=[int(k_value) for k_value in configured_k_values],
                primary_k=int(primary_k),
                shared_cluster_color_maps_by_k=shared_cluster_color_maps_by_k,
                max_points=interactive_max_points,
                source_name=str(source_name),
            )
        )
        if hdbscan_result.info is not None:
            snapshot_record["hdbscan"] = hdbscan_result.info
        if hdbscan_result.labels is not None:
            snapshot_record.update(
                _save_hdbscan_md_outputs(
                    coords_subset,
                    hdbscan_result.labels[indices],
                    snapshot_dir,
                    hdbscan_color_map=hdbscan_result.color_map,
                    max_points=interactive_max_points,
                    title=(
                        "MD local-structure clusters "
                        f"(HDBSCAN, snapshot={source_name}, n={len(coords_subset)}"
                    ),
                    context_label=f"snapshot {source_name}",
                )
            )
        summary["snapshots"].append(snapshot_record)
    return summary


def _build_md_metrics(
    *,
    out_dir: Path,
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_labels_by_k: dict[int, np.ndarray],
    configured_k_values: list[int],
    primary_k: int,
    shared_cluster_color_maps_by_k: dict[int, dict[int, str]],
    interactive_max_points: int | None,
    multi_snapshot_real: bool,
    snapshot_source_groups: list[tuple[str, np.ndarray]],
    snapshot_output_names: dict[str, str],
    hdbscan_result: HDBSCANResult,
) -> dict[str, Any]:
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    metrics: dict[str, Any] = {
        "primary_k": int(primary_k),
        "k_values_used": [int(k) for k in configured_k_values],
        "n_clusters": int(len(unique_labels)),
        "cluster_counts": {int(k): int(v) for k, v in zip(unique_labels, counts)},
    }
    if hdbscan_result.info is not None:
        metrics["hdbscan"] = hdbscan_result.info

    if multi_snapshot_real:
        metrics["by_snapshot"] = _save_snapshot_md_outputs(
            out_dir=out_dir,
            coords=coords,
            cluster_labels=cluster_labels,
            cluster_labels_by_k=cluster_labels_by_k,
            configured_k_values=configured_k_values,
            primary_k=primary_k,
            snapshot_source_groups=snapshot_source_groups,
            snapshot_output_names=snapshot_output_names,
            shared_cluster_color_maps_by_k=shared_cluster_color_maps_by_k,
            interactive_max_points=interactive_max_points,
            hdbscan_result=hdbscan_result,
        )
        return metrics

    coord_files = save_local_structure_assignments(
        coords,
        cluster_labels,
        out_dir,
    )
    if not coord_files:
        raise RuntimeError(
            "Failed to save MD cluster assignments. "
            f"coords_shape={coords.shape}, labels_shape={cluster_labels.shape}, "
            f"out_dir={out_dir}."
        )
    metrics["coords_files"] = coord_files
    metrics.update(
        _save_md_cluster_outputs(
            coords,
            {
                int(k_value): cluster_labels_by_k[int(k_value)]
                for k_value in configured_k_values
            },
            out_dir,
            k_values_order=[int(k_value) for k_value in configured_k_values],
            primary_k=int(primary_k),
            shared_cluster_color_maps_by_k=shared_cluster_color_maps_by_k,
            max_points=interactive_max_points,
        )
    )
    if hdbscan_result.labels is not None:
        metrics.update(
            _save_hdbscan_md_outputs(
                coords,
                hdbscan_result.labels,
                out_dir,
                hdbscan_color_map=hdbscan_result.color_map,
                max_points=interactive_max_points,
                title=f"MD local-structure clusters (HDBSCAN, n={len(coords)}",
                context_label="full dataset",
            )
        )
    return metrics


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
        "data_sources": _normalize_data_sources_for_cache(getattr(cfg.data, "data_sources", None)),
        "data_radius": float(getattr(cfg.data, "radius", 0.0)),
        "data_sample_type": str(getattr(cfg.data, "sample_type", "")),
        "data_overlap_fraction": float(getattr(cfg.data, "overlap_fraction", 0.0)),
        "data_n_samples": int(getattr(cfg.data, "n_samples", 0)),
        "data_num_points": int(getattr(cfg.data, "num_points", 0)),
        "data_drop_edge_samples": bool(getattr(cfg.data, "drop_edge_samples", True)),
        "data_edge_drop_layers": getattr(cfg.data, "edge_drop_layers", None),
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
        raise ValueError(f"Inference cache missing arrays: {missing}")
    inv_latents = np.asarray(cache["inv_latents"])
    if inv_latents.ndim < 2:
        raise ValueError(
            "Inference cache 'inv_latents' must have shape [num_samples, latent_dim], "
            f"got shape={tuple(inv_latents.shape)}."
        )
    num_samples = int(inv_latents.shape[0])

    coords = np.asarray(cache["coords"])
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            "Inference cache 'coords' must have shape [num_samples, 3], "
            f"got shape={tuple(coords.shape)}."
        )
    if coords.shape[0] != num_samples:
        raise ValueError(
            "Inference cache sample mismatch between 'inv_latents' and 'coords': "
            f"{num_samples} vs {coords.shape[0]}. "
            f"All cache shapes: {{'inv_latents': {tuple(inv_latents.shape)}, "
            f"'coords': {tuple(coords.shape)}, "
            f"'eq_latents': {tuple(np.asarray(cache['eq_latents']).shape)}, "
            f"'phases': {tuple(np.asarray(cache['phases']).shape)}, "
            f"'instance_ids': {tuple(np.asarray(cache['instance_ids']).shape)}}}."
        )

    for key in ("eq_latents", "phases", "instance_ids"):
        arr = np.asarray(cache[key])
        if arr.size == 0:
            continue
        if arr.shape[0] != num_samples:
            raise ValueError(
                "Inference cache sample mismatch: "
                f"'inv_latents' has {num_samples} rows but '{key}' has shape={tuple(arr.shape)}."
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
    try:
        _validate_inference_cache_arrays(cache)
    except ValueError as exc:
        return None, f"cache validation failed for {npz_path}: {exc}"
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


def _load_analysis_defaults_config(
    *,
    project_root: Path,
) -> DictConfig:
    defaults_paths = [
        project_root / "configs" / "analysis" / "cluster_figure_raytrace.yaml",
        project_root / "configs" / "analysis" / "real_md_qualitative.yaml",
    ]
    merged_cfg: DictConfig | None = None
    for defaults_path in defaults_paths:
        if not defaults_path.exists():
            raise FileNotFoundError(
                "Missing analysis defaults config file: "
                f"{defaults_path}."
            )
        defaults_cfg = OmegaConf.load(defaults_path)
        if not isinstance(defaults_cfg, DictConfig):
            raise TypeError(
                "Analysis defaults config must be a DictConfig, "
                f"got {type(defaults_cfg)!r} from {defaults_path}."
            )
        merged_cfg = defaults_cfg if merged_cfg is None else OmegaConf.merge(merged_cfg, defaults_cfg)
    if merged_cfg is None:
        raise RuntimeError("Failed to load analysis defaults configs.")
    return merged_cfg


def load_analysis_config(
    checkpoint_path: str,
    cfg: DictConfig | None = None,
) -> DictConfig:
    """Resolve the Hydra config associated with a checkpoint."""
    project_root = Path(__file__).resolve().parents[3]
    analysis_defaults_cfg = _load_analysis_defaults_config(project_root=project_root)
    config_dir, config_name = resolve_config_path(checkpoint_path)
    config_path = Path(config_dir) / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            "Resolved checkpoint config file does not exist: "
            f"{config_path} for checkpoint {checkpoint_path}."
        )
    checkpoint_cfg = OmegaConf.load(config_path)
    if not isinstance(checkpoint_cfg, DictConfig):
        raise TypeError(
            "Checkpoint config must load to a DictConfig, "
            f"got {type(checkpoint_cfg)!r} from {config_path}."
        )
    if cfg is not None and not isinstance(cfg, DictConfig):
        raise TypeError(
            "Provided cfg must be a DictConfig, "
            f"got {type(cfg)!r}."
        )

    merged_cfg = (
        OmegaConf.merge(analysis_defaults_cfg, checkpoint_cfg, cfg)
        if cfg is not None
        else OmegaConf.merge(analysis_defaults_cfg, checkpoint_cfg)
    )
    if not isinstance(merged_cfg, DictConfig):
        raise TypeError(
            "Merged analysis config must be a DictConfig, "
            f"got {type(merged_cfg)!r}."
        )
    return merged_cfg


def _load_override_config_file(
    path: str,
    *,
    field_name: str,
) -> DictConfig:
    resolved = _resolve_input_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{field_name} does not exist: {resolved}")
    loaded_cfg = OmegaConf.load(resolved)
    if not isinstance(loaded_cfg, DictConfig):
        raise TypeError(
            f"{field_name} must load to a DictConfig, got {type(loaded_cfg)!r} from {resolved}."
        )
    return loaded_cfg


def _build_runtime_override_cfg(
    *,
    data_config_path: str | None,
    analysis_config_override_paths: list[str] | None,
) -> DictConfig | None:
    merged_cfg: DictConfig | None = None
    if data_config_path is not None:
        data_cfg = _load_override_config_file(
            data_config_path,
            field_name="--data_config",
        )
        if "data" in data_cfg:
            merged_piece = data_cfg
        else:
            merged_piece = OmegaConf.create({"data": data_cfg})
        explicit_analysis_files = _as_list_of_str(
            OmegaConf.select(merged_piece, "data.analysis_data_files", default=None)
        )
        data_files = _as_list_of_str(OmegaConf.select(merged_piece, "data.data_files", default=None))
        if explicit_analysis_files is None and data_files:
            merged_piece = OmegaConf.merge(
                merged_piece,
                OmegaConf.create(
                    {
                        "data": {
                            "analysis_data_files": list(data_files),
                        },
                    }
                ),
            )
        merged_cfg = merged_piece if merged_cfg is None else OmegaConf.merge(merged_cfg, merged_piece)
    for override_path in analysis_config_override_paths or []:
        override_cfg = _load_override_config_file(
            override_path,
            field_name="--analysis_config_override",
        )
        merged_cfg = override_cfg if merged_cfg is None else OmegaConf.merge(merged_cfg, override_cfg)
    return merged_cfg


def _parse_color_value(value: Any) -> int | str:
    """Parse a single color assignment: palette index (int) or color string."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value).strip()
    if text.lstrip("+-").isdigit():
        return int(text)
    return text


def _normalize_cluster_color_assignment(
    value: Any,
    *,
    field_name: str,
) -> dict[int, int | str] | None:
    if value is None:
        return None
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)

    if isinstance(value, dict):
        return {int(k): _parse_color_value(v) for k, v in value.items()} or None

    entries = [value] if isinstance(value, str) else list(value)
    result: dict[int, int | str] = {}
    for entry in entries:
        entry = str(entry).strip()
        if "=" not in entry:
            raise ValueError(f"{field_name}: expected CLUSTER=VALUE syntax, got {entry!r}")
        cluster_part, value_part = entry.split("=", 1)
        result[int(cluster_part.strip())] = _parse_color_value(value_part.strip())
    return result or None


def _resolve_input_path(
    path: str,
    *,
    base_dir: Path | None = None,
) -> Path:
    expanded = Path(os.path.expanduser(path))
    if expanded.is_absolute():
        return expanded
    candidates: list[Path] = []
    if base_dir is not None:
        candidates.append(base_dir / expanded)
    candidates.append(Path(os.getcwd()) / expanded)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_cluster_color_assignment_file(
    path: str,
    *,
    base_dir: Path | None = None,
) -> dict[int, int | str]:
    resolved = _resolve_input_path(path, base_dir=base_dir)
    with resolved.open("r") as handle:
        payload = json.load(handle)
    raw_assignment = payload.get("assignment", payload) if isinstance(payload, dict) else payload
    assignment = _normalize_cluster_color_assignment(
        raw_assignment, field_name=f"cluster_color_assignment_file({resolved})",
    )
    if assignment is None:
        raise ValueError(f"No assignments found in {resolved}")
    return assignment


def _merge_cluster_color_assignments(
    *assignments: dict[int, int | str] | None,
) -> dict[int, int | str] | None:
    merged: dict[int, int | str] = {}
    for assignment in assignments:
        if assignment:
            merged.update(assignment)
    return merged or None


def _load_existing_metrics(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r") as handle:
        return json.load(handle)


def load_barlow_model(
    checkpoint_path: str, cuda_device: int = 0, cfg: DictConfig | None = None
) -> Tuple[BarlowTwinsModule, DictConfig, str]:
    """Restore the contrastive module together with its Hydra cfg and device string."""
    if cfg is None:
        cfg = load_analysis_config(checkpoint_path, cfg=None)
    if not isinstance(cfg, DictConfig):
        raise TypeError(
            "load_barlow_model expects cfg to be a DictConfig when provided, "
            f"got {type(cfg)!r}."
        )
    device = f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu"
    model: BarlowTwinsModule = load_model_from_checkpoint(
        checkpoint_path, cfg, device=device, module=BarlowTwinsModule
    )
    model.to(device).eval()
    return model, cfg, device


def build_datamodule(
    cfg: DictConfig,
    *,
    require_coords_for_real: bool = False,
):
    """Instantiate the matching datamodule."""
    if getattr(cfg, "data", None) is None:
        raise ValueError("Config missing data section")
    if getattr(cfg.data, "kind", None) == "synthetic":
        dm = SyntheticPointCloudDataModule(cfg)
    else:
        dm = RealPointCloudDataModule(
            cfg,
            return_coords=bool(require_coords_for_real),
        )
    return dm


def _print_figure_set_summary(
    all_metrics: Dict[str, Any],
    n_samples: int,
    out_dir: Path,
    elapsed: float,
) -> None:
    """Print summary of generated cluster figure set files."""
    snapshot_sets = all_metrics.get("cluster_figure_sets_by_snapshot", {})
    has_snapshot_sets = isinstance(snapshot_sets, dict) and bool(snapshot_sets.get("snapshots"))
    if "cluster_figure_set" not in all_metrics and not has_snapshot_sets:
        return
    print(f"\nTotal samples analyzed: {n_samples}")
    print(f"Saved outputs to {out_dir}, runtime: {elapsed:.1f}s")
    if "cluster_figure_set" in all_metrics:
        fs = all_metrics["cluster_figure_set"]
        k_fig = fs.get("k_value", "N/A")
        raytrace_on = bool(fs.get("raytrace_render_settings", {}).get("enabled", False))
        print(f"  - cluster_figure_set_k{k_fig}/cluster_color_assignment_k{k_fig}.json")
        print(f"  - cluster_figure_set_k{k_fig}/01_md_clusters_all_k{k_fig}[_view*].png")
        if raytrace_on:
            print(f"  - cluster_figure_set_k{k_fig}/01_md_clusters_all_k{k_fig}[_view*]_raytrace.png")
            print(f"  - cluster_figure_set_k{k_fig}/01_md_clusters_all_k{k_fig}_raytrace_gallery.png")
        for s in fs.get("visible_cluster_sets", []):
            tag = "-".join(str(c) for c in s)
            print(f"  - cluster_figure_set_k{k_fig}/02_md_clusters_set_{tag}_k{k_fig}[_view*].png")
            if raytrace_on:
                print(f"  - ..._{tag}_k{k_fig}[_view*]_raytrace.png")
                print(f"  - ..._{tag}_k{k_fig}_raytrace_gallery.png")
        print(f"  - cluster_figure_set_k{k_fig}/04_cluster_representatives_k{k_fig}*.png")
        print(
            "  - cluster_figure_set_k"
            f"{k_fig}/08_cluster_representatives_spatial_neighbors_paper_k{k_fig}.png"
        )
        rep_analysis = fs.get("panel_representatives_structure_analysis")
        if isinstance(rep_analysis, dict):
            print(
                "  - cluster_figure_set_k"
                f"{k_fig}/10_cluster_representatives_structure_analysis_k{k_fig}.json"
            )
            print(
                "  - cluster_figure_set_k"
                f"{k_fig}/10_cluster_representatives_structure_analysis_k{k_fig}.csv"
            )
    if has_snapshot_sets:
        print("  - cluster_figure_sets_by_snapshot/<snapshot>/cluster_figure_set_k*/...")
        snapshot_gallery_sets = snapshot_sets.get("raytrace_galleries_by_view", {})
        if isinstance(snapshot_gallery_sets, dict) and snapshot_gallery_sets.get("all_clusters"):
            k_fig = snapshot_gallery_sets.get("k_value", "N/A")
            print(
                "  - cluster_figure_sets_by_snapshot/_galleries_by_view/"
                f"cluster_figure_set_k{k_fig}/01_md_clusters_all_k{k_fig}_view*_raytrace_gallery.png"
            )
            for s in snapshot_sets.get("requested_visible_cluster_sets", []):
                tag = "-".join(str(c) for c in s)
                print(
                    "  - cluster_figure_sets_by_snapshot/_galleries_by_view/"
                    f"cluster_figure_set_k{k_fig}/02_md_clusters_set_{tag}_k{k_fig}_view*_raytrace_gallery.png"
                )


def run_post_training_analysis(
    checkpoint_path: str,
    output_dir: str,
    cuda_device: int = 0,
    cfg: DictConfig | None = None,
    max_batches_latent: int | None = None,
    max_samples_visualization: int | None = None,
    data_files_override: list[str] | None = None,
    visible_cluster_sets: list[list[int]] | None = None,
    cluster_color_assignment: dict[int, int | str] | None = None,
    cluster_color_assignment_file: str | None = None,
    cluster_figure_only: bool = False,
    md_render_saturation_boost: float | None = None,
    raytrace_render_enabled: bool | None = None,
    cluster_k: int | None = None,
) -> Dict[str, Any]:
    """Generate qualitative and quantitative diagnostics for contrastive checkpoints."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    step_idx = [0]

    def _step(msg: str) -> None:
        step_idx[0] += 1
        elapsed = time.perf_counter() - t0
        print(f"[analysis][step {step_idx[0]}][{elapsed:7.1f}s] {msg}")

    _step("Loading analysis config")
    cfg = load_analysis_config(checkpoint_path, cfg=cfg)
    unified_cluster_k = _apply_unified_cluster_k_override(
        cfg,
        cluster_k_override=cluster_k,
    )
    model: BarlowTwinsModule | None = None
    device = f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu"
    analysis_source_names: list[str] | None = None
    analysis_files = _resolve_analysis_files(
        cfg,
        data_files_override=data_files_override,
    )
    if analysis_files is not None:
        analysis_source_names = _configure_real_analysis_inputs(cfg, analysis_files)
        print(f"Analysis data_files: {analysis_files}")
        if analysis_source_names and len(analysis_source_names) > 1:
            print(f"Per-snapshot analysis sources: {analysis_source_names}")

    # Analysis uses a fixed worker count to keep behavior consistent across runs.
    analysis_num_workers = 4
    cfg.num_workers = analysis_num_workers
    print(f"Analysis dataloader workers: {analysis_num_workers}")
    analysis_settings = _resolve_analysis_settings(
        cfg,
        max_samples_visualization=max_samples_visualization,
    )
    hdbscan_settings = analysis_settings.hdbscan
    figure_settings = _resolve_figure_set_settings(
        cfg,
        out_dir=out_dir,
        visible_cluster_sets=visible_cluster_sets,
        cluster_color_assignment=cluster_color_assignment,
        cluster_color_assignment_file=cluster_color_assignment_file,
        cluster_figure_only=cluster_figure_only,
        md_render_saturation_boost=md_render_saturation_boost,
        raytrace_render_enabled=raytrace_render_enabled,
    )
    _print_resolved_analysis_settings(
        unified_cluster_k=unified_cluster_k,
        analysis_settings=analysis_settings,
        figure_settings=figure_settings,
    )
    is_synthetic = getattr(cfg.data, "kind", None) == "synthetic"
    _step("Building datamodule")
    dm = build_datamodule(
        cfg,
        require_coords_for_real=not is_synthetic,
    )
    dm.setup(stage="fit")
    all_metrics: Dict[str, Any] = {}
    dl = _build_analysis_dataloader(
        cfg,
        dm,
        is_synthetic=is_synthetic,
    )

    class_names = _extract_class_names(dm.train_dataset)
    print(f"Loaded class names: {class_names}")

    if max_batches_latent is None:
        max_batches_latent = _positive_int_or_none(getattr(cfg, "analysis_max_batches_latent", None))
    max_samples_total = _resolve_analysis_max_samples_total(
        cfg,
        is_synthetic=is_synthetic,
        md_use_all_points=analysis_settings.md_use_all_points,
    )

    seed_base = int(analysis_settings.seed_base)
    clustering_random_state = int(analysis_settings.seed_base)
    cache_spec = _build_inference_cache_spec(
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        max_batches_latent=max_batches_latent,
        max_samples_total=max_samples_total,
        seed_base=int(seed_base),
    )

    cache: dict[str, np.ndarray] | None = None
    cache_loaded = False
    if figure_settings.figure_only:
        _step("Loading cached inference batches")
        cache, cache_msg = _load_inference_cache(
            out_dir=out_dir,
            cache_filename=analysis_settings.inference_cache_file,
            expected_spec=cache_spec,
        )
        cache_loaded = cache is not None
        print(f"[analysis][cache] {cache_msg}")
        if cache is None:
            raise RuntimeError(
                "--cluster_figure_only requires a valid inference cache because it does not "
                "run model inference. "
                f"Cache load failed: {cache_msg}. "
                "Run the full analysis once without --cluster_figure_only to populate "
                f"{out_dir / analysis_settings.inference_cache_file}."
            )
    else:
        _step("Loading model")
        model, cfg, device = load_barlow_model(
            checkpoint_path,
            cuda_device=cuda_device,
            cfg=cfg,
        )
        _step("Collecting inference batches")
        if (
            analysis_settings.inference_cache_enabled
            and not analysis_settings.inference_cache_force_recompute
        ):
            cache, cache_msg = _load_inference_cache(
                out_dir=out_dir,
                cache_filename=analysis_settings.inference_cache_file,
                expected_spec=cache_spec,
            )
            cache_loaded = cache is not None
            print(f"[analysis][cache] {cache_msg}")
        elif (
            analysis_settings.inference_cache_enabled
            and analysis_settings.inference_cache_force_recompute
        ):
            print("[analysis][cache] Forced recompute requested; skipping cache load.")

    if cache is None and not figure_settings.figure_only:
        if not analysis_settings.inference_cache_enabled:
            print("[analysis][cache] Inference cache disabled; running fresh inference.")
        if max_batches_latent is None:
            print("Gathering inference batches (ALL batches)...")
        else:
            print(f"Gathering inference batches (up to {max_batches_latent} batches)...")
        if max_samples_total is not None:
            print(f"Collecting up to {max_samples_total} samples for analysis")
        if model is None:
            raise RuntimeError(
                "Internal error: model must be loaded before gathering inference batches."
            )
        cache = gather_inference_batches(
            model,
            dl,
            device,
            max_batches=max_batches_latent,
            max_samples_total=max_samples_total,
            collect_coords=True,
            seed_base=seed_base,
            progress_every_batches=analysis_settings.progress_every_batches,
            verbose=True,
        )
        _validate_inference_cache_arrays(cache)
        if analysis_settings.inference_cache_enabled:
            _save_inference_cache(
                out_dir=out_dir,
                cache_filename=analysis_settings.inference_cache_file,
                cache=cache,
                spec=cache_spec,
            )
            cache_npz, _ = _inference_cache_paths(
                out_dir,
                analysis_settings.inference_cache_file,
            )
            print(f"[analysis][cache] Saved inference cache: {cache_npz}")

    _validate_inference_cache_arrays(cache)
    n_samples = len(cache["inv_latents"])
    print(f"Collected {n_samples} samples for analysis")
    all_metrics["inference_cache"] = {
        "enabled": bool(analysis_settings.inference_cache_enabled),
        "file": str((out_dir / analysis_settings.inference_cache_file)),
        "loaded_from_cache": bool(cache_loaded),
        "force_recompute": bool(analysis_settings.inference_cache_force_recompute),
        "spec_sha256": _inference_cache_spec_hash(cache_spec),
    }
    coords = cache["coords"]
    md_metrics_key = "synthetic_md" if is_synthetic else "real_md"
    point_scale = (
        _resolve_point_scale(cfg)
        if figure_settings.profile_point_scale_enabled
        else 1.0
    )
    print(
        "Representative point scaling: "
        f"enabled={figure_settings.profile_point_scale_enabled}, point_scale={point_scale:.6g}"
    )
    _step("Preparing clustering features")
    clustering_features, clustering_feature_prep = _prepare_clustering_features(
        cache["inv_latents"],
        random_state=int(clustering_random_state),
        l2_normalize=analysis_settings.cluster_l2_normalize,
        standardize=analysis_settings.cluster_standardize,
        pca_variance=analysis_settings.cluster_pca_var,
        pca_max_components=analysis_settings.cluster_pca_max_components,
    )
    dataset_obj = getattr(dl, "dataset", None)
    snapshot_source_groups: list[tuple[str, np.ndarray]] = []
    snapshot_output_names: dict[str, str] = {}
    multi_snapshot_real = False
    if not is_synthetic and dataset_obj is not None:
        snapshot_source_groups = _resolve_sample_source_groups(dataset_obj, n_samples=n_samples)
        encountered_source_names = [str(name) for name, _ in snapshot_source_groups]
        if analysis_source_names is not None and len(analysis_source_names) > 1:
            missing_sources = [
                str(name) for name in analysis_source_names if str(name) not in encountered_source_names
            ]
            if missing_sources:
                raise RuntimeError(
                    "Per-snapshot plotting requires collected samples from every requested "
                    "analysis snapshot, but some snapshots are missing from the collected "
                    "prefix. "
                    f"missing={missing_sources}, encountered={encountered_source_names}, "
                    f"n_samples_collected={n_samples}. Increase analysis_max_samples_total / "
                    "max_batches_latent, or disable sampling limits for this analysis run."
                )
        multi_snapshot_real = len(snapshot_source_groups) > 1
        if multi_snapshot_real:
            snapshot_output_names = _build_unique_snapshot_output_names(encountered_source_names)
            print(
                "Per-snapshot plotting enabled for sources: "
                f"{encountered_source_names}"
            )

    figure_set_kwargs = figure_settings.build_run_kwargs(
        dataset=dataset_obj,
        latents=cache["inv_latents"],
        coords=coords,
        point_scale=point_scale,
        random_state=clustering_random_state,
        l2_normalize=analysis_settings.cluster_l2_normalize,
        standardize=analysis_settings.cluster_standardize,
        pca_variance=analysis_settings.cluster_pca_var,
        pca_max_components=analysis_settings.cluster_pca_max_components,
    )

    def _run_figure_set(
        labels_for_k: np.ndarray,
        *,
        figure_out_dir: Path | None = None,
        dataset_override: Any | None = None,
        latents_override: np.ndarray | None = None,
        coords_override: np.ndarray | None = None,
        cluster_color_assignment_override: dict[int, int | str] | None = None,
        visible_cluster_sets_override: list[list[int]] | None = None,
        representative_render_cache_override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _step("Generating fixed-k cluster figure set")
        figure_set_dir = (
            out_dir / f"cluster_figure_set_k{figure_settings.k}"
            if figure_out_dir is None
            else Path(figure_out_dir)
        )
        run_kwargs = dict(figure_set_kwargs)
        if dataset_override is not None:
            run_kwargs["dataset"] = dataset_override
        if latents_override is not None:
            run_kwargs["latents"] = latents_override
        if coords_override is not None:
            run_kwargs["coords"] = coords_override
        if cluster_color_assignment_override is not None:
            run_kwargs["cluster_color_assignment"] = cluster_color_assignment_override
        if visible_cluster_sets_override is not None:
            run_kwargs["visible_cluster_sets"] = visible_cluster_sets_override
        if representative_render_cache_override is not None:
            run_kwargs["representative_render_cache"] = representative_render_cache_override
        with _temporary_disable_dataset_aug(dl):
            return _save_fixed_k_cluster_figure_set(
                out_dir=figure_set_dir,
                cluster_labels=labels_for_k,
                **run_kwargs,
            )

    def _build_shared_cluster_color_map(
        labels_for_k: np.ndarray,
    ) -> dict[int, str]:
        color_map = _build_cluster_color_map(
            labels_for_k,
            cluster_color_assignment=figure_settings.cluster_color_assignment,
        )
        return {int(cluster_id): str(color) for cluster_id, color in color_map.items()}

    def _run_snapshot_figure_sets(
        labels_for_k: np.ndarray,
        *,
        global_color_map: dict[int, int | str] | None = None,
    ) -> dict[str, Any]:
        if not multi_snapshot_real:
            return {}
        if dataset_obj is None:
            raise RuntimeError(
                "Cannot generate per-snapshot cluster figure sets: dataloader dataset is missing."
            )

        min_required_samples = int(figure_settings.k) + 1
        too_small = [
            (str(source_name), int(indices.size))
            for source_name, indices in snapshot_source_groups
            if int(indices.size) < min_required_samples
        ]
        if too_small:
            details = ", ".join(f"{name}: {count}" for name, count in too_small)
            raise RuntimeError(
                "Cannot generate per-snapshot cluster figure sets because at least one "
                "snapshot has too few collected samples for the requested fixed-k analysis. "
                f"Need at least {min_required_samples} samples per snapshot for "
                f"k={figure_settings.k}, got {details}. "
                "Increase analysis_max_samples_total / max_batches_latent, or lower "
                "analysis_cluster_figure_set_k."
            )

        snapshot_root = out_dir / "cluster_figure_sets_by_snapshot"
        summary: dict[str, Any] = {
            "root_dir": str(snapshot_root),
            "k_value": int(figure_settings.k),
            "requested_visible_cluster_sets": [
                sorted(int(v) for v in cluster_set)
                for cluster_set in (figure_settings.visible_cluster_sets or [])
            ],
            "snapshots": [],
        }
        ordered_snapshot_groups = list(snapshot_source_groups)
        if analysis_source_names is not None and len(analysis_source_names) > 1:
            groups_by_name = {
                str(source_name): np.asarray(indices, dtype=int)
                for source_name, indices in snapshot_source_groups
            }
            ordered_snapshot_groups = [
                (str(source_name), groups_by_name[str(source_name)])
                for source_name in analysis_source_names
                if str(source_name) in groups_by_name
            ]
            remaining_groups = [
                (str(source_name), np.asarray(indices, dtype=int))
                for source_name, indices in snapshot_source_groups
                if str(source_name) not in {name for name, _ in ordered_snapshot_groups}
            ]
            ordered_snapshot_groups.extend(remaining_groups)

        for source_name, indices in ordered_snapshot_groups:
            snapshot_dirname = snapshot_output_names[str(source_name)]
            snapshot_dir = (
                snapshot_root / snapshot_dirname / f"cluster_figure_set_k{figure_settings.k}"
            )
            subset_dataset = torch.utils.data.Subset(
                dataset_obj,
                [int(v) for v in indices.tolist()],
            )
            snapshot_visible_sets = _resolve_visible_cluster_sets_for_labels(
                labels_for_k[indices],
                figure_settings.visible_cluster_sets,
                context=f"snapshot={source_name}",
            )
            figure_info = _run_figure_set(
                labels_for_k[indices],
                figure_out_dir=snapshot_dir,
                dataset_override=subset_dataset,
                latents_override=cache["inv_latents"][indices],
                coords_override=coords[indices],
                cluster_color_assignment_override=global_color_map,
                visible_cluster_sets_override=snapshot_visible_sets,
            )
            summary["snapshots"].append(
                {
                    "source_name": str(source_name),
                    "output_name": str(snapshot_dirname),
                    "sample_count": int(indices.size),
                    "figure_set": figure_info,
                }
            )
        if bool(figure_settings.raytrace_enabled):
            summary["raytrace_galleries_by_view"] = _save_snapshot_raytrace_galleries_by_view(
                summary,
                requested_visible_cluster_sets=figure_settings.visible_cluster_sets,
            )
        return summary

    if figure_settings.figure_only:
        clustering_metrics, _, cluster_labels_by_k, _ = _build_clustering_state(
            cache["inv_latents"],
            cache["phases"],
            requested_k_values=[int(figure_settings.k)],
            cluster_method=analysis_settings.cluster_method,
            random_state=clustering_random_state,
            l2_normalize=analysis_settings.cluster_l2_normalize,
            standardize=analysis_settings.cluster_standardize,
            pca_variance=analysis_settings.cluster_pca_var,
            pca_max_components=analysis_settings.cluster_pca_max_components,
            prepared_features=clustering_features,
            prep_info=clustering_feature_prep,
        )
        all_metrics["clustering"] = clustering_metrics
        if figure_settings.enabled:
            if multi_snapshot_real:
                snapshot_figure_sets = _run_snapshot_figure_sets(
                    cluster_labels_by_k[int(figure_settings.k)],
                    global_color_map=_build_shared_cluster_color_map(
                        cluster_labels_by_k[int(figure_settings.k)]
                    ),
                )
                if snapshot_figure_sets:
                    all_metrics["cluster_figure_sets_by_snapshot"] = snapshot_figure_sets
            else:
                all_metrics["cluster_figure_set"] = _run_figure_set(
                    cluster_labels_by_k[int(figure_settings.k)]
                )

        _step("Writing metrics")
        metrics_path = out_dir / "analysis_metrics.json"
        merged_metrics = _load_existing_metrics(metrics_path)
        existing_clustering = merged_metrics.get("clustering", {})
        if isinstance(existing_clustering, dict):
            existing_clustering.update(all_metrics["clustering"])
            merged_metrics["clustering"] = existing_clustering
        else:
            merged_metrics["clustering"] = all_metrics["clustering"]
        merged_metrics["inference_cache"] = all_metrics["inference_cache"]
        if "cluster_figure_set" in all_metrics:
            merged_metrics["cluster_figure_set"] = all_metrics["cluster_figure_set"]
        elif multi_snapshot_real:
            merged_metrics.pop("cluster_figure_set", None)
        if "cluster_figure_sets_by_snapshot" in all_metrics:
            merged_metrics["cluster_figure_sets_by_snapshot"] = all_metrics[
                "cluster_figure_sets_by_snapshot"
            ]
        with metrics_path.open("w") as handle:
            json.dump(merged_metrics, handle, indent=2)

        _print_figure_set_summary(all_metrics, n_samples, out_dir, time.perf_counter() - t0)
        return merged_metrics

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

    _step("Computing clustering labels")
    clustering_requested_k_values = list(analysis_settings.cluster_k_values)
    if figure_settings.enabled and int(figure_settings.k) not in clustering_requested_k_values:
        clustering_requested_k_values.append(int(figure_settings.k))
    clustering_metrics, configured_k_values, cluster_labels_by_k, cluster_methods_by_k = _build_clustering_state(
        cache["inv_latents"],
        cache["phases"],
        requested_k_values=clustering_requested_k_values,
        cluster_method=analysis_settings.cluster_method,
        random_state=clustering_random_state,
        l2_normalize=analysis_settings.cluster_l2_normalize,
        standardize=analysis_settings.cluster_standardize,
        pca_variance=analysis_settings.cluster_pca_var,
        pca_max_components=analysis_settings.cluster_pca_max_components,
        prepared_features=clustering_features,
        prep_info=clustering_feature_prep,
    )
    all_metrics["clustering"] = clustering_metrics

    primary_k = int(configured_k_values[0])
    if not is_synthetic:
        requested_primary_k_raw = getattr(cfg, "analysis_real_md_k", None)
        if requested_primary_k_raw is not None:
            requested_primary_k = int(requested_primary_k_raw)
            if requested_primary_k not in configured_k_values:
                raise KeyError(
                    "Requested analysis_real_md_k is not available in configured clustering results. "
                    f"Requested k={requested_primary_k}, available={configured_k_values}."
                )
            primary_k = int(requested_primary_k)
    cluster_labels = cluster_labels_by_k[primary_k]
    selected_real_md_k = int(
        primary_k
        if getattr(cfg, "analysis_real_md_k", None) is None
        else int(getattr(cfg, "analysis_real_md_k"))
    )
    shared_representative_render_cache: dict[str, Any] | None = None
    if (
        not is_synthetic
        and figure_settings.enabled
        and not multi_snapshot_real
        and bool(getattr(cfg, "analysis_real_md_enabled", True))
        and bool(getattr(cfg, "analysis_cluster_profile_enabled", True))
        and dataset_obj is not None
        and int(selected_real_md_k) == int(figure_settings.k)
        and int(figure_settings.real_md_profile_target_points)
        == int(figure_settings.representative_points)
    ):
        _step("Preparing shared representative structures")
        shared_representative_render_cache = _build_cluster_representative_render_cache(
            dataset_obj,
            np.asarray(cache["inv_latents"], dtype=np.float32),
            np.asarray(cluster_labels_by_k[int(figure_settings.k)], dtype=int),
            _build_shared_cluster_color_map(cluster_labels_by_k[int(figure_settings.k)]),
            point_scale=float(point_scale),
            target_points=int(figure_settings.representative_points),
            representative_ptm_enabled=bool(figure_settings.representative_ptm_enabled),
            representative_cna_enabled=bool(figure_settings.representative_cna_enabled),
            representative_cna_max_signatures=int(
                figure_settings.representative_cna_max_signatures
            ),
            representative_center_atom_tolerance=float(
                figure_settings.representative_center_atom_tolerance
            ),
            representative_shell_min_neighbors=int(
                figure_settings.representative_shell_min_neighbors
            ),
            representative_shell_max_neighbors=int(
                figure_settings.representative_shell_max_neighbors
            ),
        )

    if figure_settings.enabled:
        if multi_snapshot_real:
            snapshot_figure_sets = _run_snapshot_figure_sets(
                cluster_labels_by_k[int(figure_settings.k)],
                global_color_map=_build_shared_cluster_color_map(
                    cluster_labels_by_k[int(figure_settings.k)]
                ),
            )
            if snapshot_figure_sets:
                all_metrics["cluster_figure_sets_by_snapshot"] = snapshot_figure_sets
        else:
            all_metrics["cluster_figure_set"] = _run_figure_set(
                cluster_labels_by_k[int(figure_settings.k)],
                representative_render_cache_override=shared_representative_render_cache,
            )

    _step("Building shared cluster color maps")
    shared_cluster_color_maps_by_k = {
        int(k_val_inner): _build_shared_cluster_color_map(
            cluster_labels_by_k[int(k_val_inner)]
        )
        for k_val_inner in configured_k_values
    }
    shared_cluster_color_map = shared_cluster_color_maps_by_k[int(primary_k)]

    _step("Computing t-SNE visualization (clusters)")
    tsne_n_iter = int(getattr(cfg, "analysis_tsne_n_iter", 1000))
    tsne_idx = _sample_indices(
        len(cache["inv_latents"]),
        analysis_settings.tsne_max_samples,
    )
    tsne_latents = cache["inv_latents"][tsne_idx]
    tsne_perplexity = min(50, max(5, len(tsne_latents) // 100))
    tsne_coords = compute_tsne(
        tsne_latents,
        random_state=clustering_random_state,
        perplexity=tsne_perplexity,
        n_iter=tsne_n_iter,
    )
    if is_synthetic and cache["phases"].size == len(cache["inv_latents"]):
        save_tsne_plot(
            tsne_coords,
            cache["phases"][tsne_idx],
            out_file=str(out_dir / "latent_tsne_ground_truth.png"),
            title=f"Latent space t-SNE (n={len(tsne_latents)}, ground truth phases)",
            legend_title="phase",
            class_names=class_names,
        )

    for idx_k, k_val in enumerate(configured_k_values):
        labels_k = cluster_labels_by_k[int(k_val)]
        method_k = cluster_methods_by_k.get(int(k_val), analysis_settings.cluster_method)
        out_name = "latent_tsne_clusters.png" if idx_k == 0 else f"latent_tsne_clusters_k{k_val}.png"
        save_tsne_plot_with_coords(
            tsne_coords,
            labels_k[tsne_idx],
            out_dir,
            out_name=out_name,
            title=f"Latent space t-SNE ({method_k}, k={k_val})",
            cluster_color_map=shared_cluster_color_maps_by_k.get(int(k_val)),
            paper_out_name=(
                f"latent_tsne_clusters_paper_k{k_val}.svg"
                if (
                    not is_synthetic
                    and bool(getattr(cfg, "analysis_tsne_paper_enabled", True))
                    and int(k_val) == int(primary_k)
                )
                else None
            ),
            paper_title=None,
            paper_label_prefix="C",
        )

        if idx_k == 0:
            _step("Saving coordinate-space clustering outputs")
            interactive_max_points = _positive_int_or_none(
                getattr(cfg, "analysis_interactive_max_points", None)
            )
            if analysis_settings.md_use_all_points:
                interactive_max_points = None
            hdbscan_result = _run_optional_hdbscan_analysis(
                cache["inv_latents"],
                coords_count=len(coords),
                settings=hdbscan_settings,
                random_state=clustering_random_state,
                l2_normalize=analysis_settings.cluster_l2_normalize,
                standardize=analysis_settings.cluster_standardize,
                pca_variance=analysis_settings.cluster_pca_var,
                pca_max_components=analysis_settings.cluster_pca_max_components,
                prepared_features=clustering_features,
                prep_info=clustering_feature_prep,
                cluster_color_assignment=figure_settings.cluster_color_assignment,
                step=_step,
            )
            all_metrics[md_metrics_key] = _build_md_metrics(
                out_dir=out_dir,
                coords=coords,
                cluster_labels=cluster_labels,
                cluster_labels_by_k=cluster_labels_by_k,
                configured_k_values=configured_k_values,
                primary_k=primary_k,
                shared_cluster_color_maps_by_k=shared_cluster_color_maps_by_k,
                interactive_max_points=interactive_max_points,
                multi_snapshot_real=multi_snapshot_real,
                snapshot_source_groups=snapshot_source_groups,
                snapshot_output_names=snapshot_output_names,
                hdbscan_result=hdbscan_result,
            )

    if not is_synthetic and bool(getattr(cfg, "analysis_real_md_enabled", True)):
        _step("Running real-MD qualitative analysis")
        shared_real_md_color_map = shared_cluster_color_maps_by_k.get(
            int(selected_real_md_k),
            shared_cluster_color_map,
        )
        all_metrics["real_md_qualitative"] = run_real_md_qualitative_analysis(
            out_dir=out_dir,
            cfg=cfg,
            dataset=dataset_obj,
            latents=cache["inv_latents"],
            coords=coords,
            cluster_labels_by_k=cluster_labels_by_k,
            cluster_methods_by_k=cluster_methods_by_k,
            cluster_color_map=shared_real_md_color_map,
            frame_groups=snapshot_source_groups,
            frame_output_names=snapshot_output_names,
            requested_frame_order=analysis_source_names,
            point_scale=float(point_scale),
            random_state=int(clustering_random_state),
            representative_render_cache=shared_representative_render_cache,
        )

    _step("Evaluating equivariance")
    eq_metrics, eq_err = evaluate_latent_equivariance(model, dl, device, max_batches=2)
    save_equivariance_plot(eq_err, out_dir / "equivariance.png")
    all_metrics["equivariance"] = eq_metrics

    _step("Writing metrics")
    metrics_path = out_dir / "analysis_metrics.json"
    with metrics_path.open("w") as handle:
        json.dump(all_metrics, handle, indent=2)

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 60}\nANALYSIS SUMMARY\n{'=' * 60}")
    print(f"Total samples: {n_samples}, runtime: {elapsed:.1f}s, output: {out_dir}")
    if "pca" in all_metrics and all_metrics["pca"]:
        print(f"PCA: {all_metrics['pca'].get('n_components_95_var', 'N/A')} components for 95% variance")
    if "clustering" in all_metrics and all_metrics["clustering"]:
        cl = all_metrics["clustering"]
        print(f"Clustering: k_values={cl.get('k_values_used')}, primary_k={cl.get('primary_k')}")
        if "ari_with_gt" in cl:
            print(f"ARI={_fmt_metric(cl['ari_with_gt'])}, NMI={_fmt_metric(cl['nmi_with_gt'])}")
    if "equivariance" in all_metrics:
        eq = all_metrics["equivariance"]
        print(f"Equivariance: mean={_fmt_metric(eq.get('eq_latent_rel_error_mean', 'N/A'))}, "
              f"median={_fmt_metric(eq.get('eq_latent_rel_error_median', 'N/A'))}")
    if "real_md_qualitative" in all_metrics:
        real_md_summary = all_metrics["real_md_qualitative"]
        print(
            "Real-MD qualitative analysis: "
            f"{real_md_summary.get('summary_markdown', real_md_summary.get('root_dir', 'N/A'))}"
        )
    _print_figure_set_summary(all_metrics, n_samples, out_dir, elapsed)

    return all_metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run post-training analysis for contrastive checkpoints.",
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
        help="Override real data files (repeat for multiple). Example: --data_file 175ps.npy",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default=None,
        help=(
            "Path to a data YAML override. Plain data configs like "
            "configs/data/data_ae_Al_80.yaml are wrapped under cfg.data automatically."
        ),
    )
    parser.add_argument(
        "--analysis_config_override",
        action="append",
        default=None,
        help=(
            "Additional YAML config override(s) merged on top of the checkpoint config. "
            "Repeat the flag to apply multiple override files."
        ),
    )
    parser.add_argument(
        "--visible_cluster_sets",
        nargs="+",
        default=None,
        metavar="IDS",
        help=(
            "Cluster ID sets for separate subset views.  Each argument is a "
            "comma-separated list of cluster IDs.  "
            "Example: --visible_cluster_sets '0,1,2' '3,4,5'"
        ),
    )
    parser.add_argument(
        "--cluster_color_assignment",
        action="append",
        default=None,
        metavar="CLUSTER=VALUE",
        help=(
            "Override a cluster-to-color assignment for the fixed-k figure set. "
            "VALUE can be a palette slot index or an explicit color string. "
            "Repeat the flag for multiple overrides, for example "
            "--cluster_color_assignment 0=2 --cluster_color_assignment '1=#ff6d00'."
        ),
    )
    parser.add_argument(
        "--cluster_color_assignment_file",
        type=str,
        default=None,
        help=(
            "JSON file with cluster color overrides. The generated "
            "cluster_color_assignment_k*.json files are valid inputs."
        ),
    )
    parser.add_argument(
        "--cluster_figure_only",
        action="store_true",
        help=(
            "Rerender only the fixed-k cluster figure set from the saved inference cache. "
            "This skips the rest of the analysis pipeline for faster reruns."
        ),
    )
    parser.add_argument(
        "--cluster_figure_md_saturation_boost",
        type=float,
        default=None,
        help=(
            "Saturation multiplier for the static MD cluster renders. If omitted, "
            "uses analysis_cluster_figure_md_saturation_boost from config."
        ),
    )
    parser.add_argument(
        "--cluster_k",
        type=int,
        default=None,
        help=(
            "Primary analysis cluster count. When set, it pins the fixed-k figure "
            "set, the real-MD qualitative analysis, and the primary clustering "
            "outputs to this k. Equivalent config key: analysis_cluster_k."
        ),
    )
    parser.add_argument(
        "--raytrace_render_enabled",
        action="store_true",
        default=None,
        help=(
            "Generate additional Blender Cycles raytraced renders "
            "(*_raytrace.png) alongside existing outputs. All raytrace quality/"
            "camera parameters are read from configs/analysis/"
            "cluster_figure_raytrace.yaml (overridable in the experiment config)."
        ),
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

    vis_sets: list[list[int]] | None = None
    if args.visible_cluster_sets is not None:
        vis_sets = [
            [int(v) for v in s.split(",")]
            for s in args.visible_cluster_sets
        ]
    cluster_color_assignment = _normalize_cluster_color_assignment(
        args.cluster_color_assignment,
        field_name="--cluster_color_assignment",
    )
    runtime_override_cfg = _build_runtime_override_cfg(
        data_config_path=args.data_config,
        analysis_config_override_paths=args.analysis_config_override,
    )

    run_post_training_analysis(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        cuda_device=int(args.cuda_device),
        cfg=runtime_override_cfg,
        max_batches_latent=args.max_batches_latent,
        max_samples_visualization=args.max_samples_visualization,
        data_files_override=args.data_file,
        visible_cluster_sets=vis_sets,
        cluster_color_assignment=cluster_color_assignment,
        cluster_color_assignment_file=args.cluster_color_assignment_file,
        cluster_figure_only=bool(args.cluster_figure_only),
        md_render_saturation_boost=args.cluster_figure_md_saturation_boost,
        raytrace_render_enabled=args.raytrace_render_enabled,
        cluster_k=args.cluster_k,
    )


if __name__ == "__main__":
    main()
