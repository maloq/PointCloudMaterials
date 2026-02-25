import argparse
import hashlib
import json
import os
import sys
import time
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.append(os.getcwd())

from src.data_utils.data_module import RealPointCloudDataModule, SyntheticPointCloudDataModule
from src.training_methods.contrastive_learning.cluster_figure_utils import (
    _save_fixed_k_cluster_figure_set,
)
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
    visible_cluster_sets: list[list[int]] | None = None,
    pretty_render_resolution: int = 2200,
    pretty_render_sphere_radius: int = 12,
    pretty_render_projection: str | None = None,
    pretty_render_perspective_fov_deg: float | None = None,
    pretty_render_perspective_distance_factor: float | None = None,
    pretty_render_color_mode: str | None = None,
    pretty_render_saturation_boost: float | None = None,
    pretty_render_wireframe_width: int | None = None,
    raytrace_render_enabled: bool | None = None,
    raytrace_blender_executable: str | None = None,
    raytrace_render_resolution: int | None = None,
    raytrace_render_max_points: int | None = None,
    raytrace_render_samples: int | None = None,
    raytrace_render_projection: str | None = None,
    raytrace_render_fov_deg: float | None = None,
    raytrace_render_camera_distance_factor: float | None = None,
    raytrace_render_sphere_radius_fraction: float | None = None,
    raytrace_render_timeout_sec: int | None = None,
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
    cluster_figure_md_max_points = _positive_int_or_none(
        getattr(cfg, "analysis_cluster_figure_md_max_points", None)
    )
    cluster_figure_md_point_size = float(
        getattr(cfg, "analysis_cluster_figure_md_point_size", 5.6)
    )
    cluster_figure_md_alpha = float(
        getattr(cfg, "analysis_cluster_figure_md_alpha", 0.62)
    )
    cluster_figure_md_halo_scale = float(
        getattr(cfg, "analysis_cluster_figure_md_halo_scale", 1.0)
    )
    cluster_figure_md_halo_alpha = float(
        getattr(cfg, "analysis_cluster_figure_md_halo_alpha", 0.0)
    )
    cluster_figure_md_view_elev = float(
        getattr(cfg, "analysis_cluster_figure_md_view_elev", 24.0)
    )
    cluster_figure_md_view_azim = float(
        getattr(cfg, "analysis_cluster_figure_md_view_azim", 35.0)
    )
    pretty_render_projection_cfg = str(
        getattr(cfg, "analysis_cluster_figure_pretty_projection", "perspective")
    ).strip().lower()
    pretty_render_projection_use = (
        pretty_render_projection_cfg
        if pretty_render_projection is None
        else str(pretty_render_projection).strip().lower()
    )
    pretty_render_fov_cfg = float(
        getattr(cfg, "analysis_cluster_figure_pretty_fov_deg", 34.0)
    )
    pretty_render_fov_use = float(
        pretty_render_fov_cfg
        if pretty_render_perspective_fov_deg is None
        else pretty_render_perspective_fov_deg
    )
    pretty_render_distance_cfg = float(
        getattr(cfg, "analysis_cluster_figure_pretty_distance_factor", 1.4)
    )
    pretty_render_distance_use = float(
        pretty_render_distance_cfg
        if pretty_render_perspective_distance_factor is None
        else pretty_render_perspective_distance_factor
    )
    pretty_render_color_mode_cfg = str(
        getattr(cfg, "analysis_cluster_figure_pretty_color_mode", "matplotlib_match")
    ).strip().lower()
    pretty_render_color_mode_use = (
        pretty_render_color_mode_cfg
        if pretty_render_color_mode is None
        else str(pretty_render_color_mode).strip().lower()
    )
    pretty_render_saturation_cfg = float(
        getattr(cfg, "analysis_cluster_figure_pretty_saturation_boost", 1.06)
    )
    pretty_render_saturation_use = float(
        pretty_render_saturation_cfg
        if pretty_render_saturation_boost is None
        else pretty_render_saturation_boost
    )
    pretty_render_wireframe_cfg = int(
        getattr(cfg, "analysis_cluster_figure_pretty_wireframe_width", 1)
    )
    pretty_render_wireframe_use = int(
        pretty_render_wireframe_cfg
        if pretty_render_wireframe_width is None
        else pretty_render_wireframe_width
    )
    raytrace_render_enabled_cfg = bool(
        getattr(cfg, "analysis_cluster_figure_raytrace_enabled", False)
    )
    raytrace_render_enabled_use = (
        raytrace_render_enabled_cfg
        if raytrace_render_enabled is None
        else bool(raytrace_render_enabled)
    )
    raytrace_blender_executable_cfg = str(
        getattr(cfg, "analysis_cluster_figure_raytrace_blender_executable", "blender")
    ).strip()
    raytrace_blender_executable_use = (
        raytrace_blender_executable_cfg
        if raytrace_blender_executable is None
        else str(raytrace_blender_executable).strip()
    )
    raytrace_render_resolution_cfg = int(
        getattr(cfg, "analysis_cluster_figure_raytrace_resolution", 1600)
    )
    raytrace_render_resolution_use = int(
        raytrace_render_resolution_cfg
        if raytrace_render_resolution is None
        else raytrace_render_resolution
    )
    raytrace_render_max_points_cfg = _positive_int_or_none(
        getattr(cfg, "analysis_cluster_figure_raytrace_max_points", None)
    )
    raytrace_render_max_points_cli = _positive_int_or_none(raytrace_render_max_points)
    if raytrace_render_max_points_cli is not None:
        raise ValueError(
            "raytrace_render_max_points is no longer supported because raytrace "
            "rendering now always uses all points. Remove the flag or set it <= 0."
        )
    if raytrace_render_max_points_cfg is not None:
        print(
            "[analysis] ignoring analysis_cluster_figure_raytrace_max_points="
            f"{raytrace_render_max_points_cfg}; raytrace now uses all points."
        )
    raytrace_render_max_points_use = None
    raytrace_render_samples_cfg = int(
        getattr(cfg, "analysis_cluster_figure_raytrace_samples", 64)
    )
    raytrace_render_samples_use = int(
        raytrace_render_samples_cfg
        if raytrace_render_samples is None
        else raytrace_render_samples
    )
    raytrace_render_projection_cfg = str(
        getattr(cfg, "analysis_cluster_figure_raytrace_projection", "perspective")
    ).strip().lower()
    raytrace_render_projection_use = (
        raytrace_render_projection_cfg
        if raytrace_render_projection is None
        else str(raytrace_render_projection).strip().lower()
    )
    raytrace_render_fov_cfg = float(
        getattr(cfg, "analysis_cluster_figure_raytrace_fov_deg", 34.0)
    )
    raytrace_render_fov_use = float(
        raytrace_render_fov_cfg
        if raytrace_render_fov_deg is None
        else raytrace_render_fov_deg
    )
    raytrace_render_camera_distance_cfg = float(
        getattr(cfg, "analysis_cluster_figure_raytrace_camera_distance_factor", 2.8)
    )
    raytrace_render_camera_distance_use = float(
        raytrace_render_camera_distance_cfg
        if raytrace_render_camera_distance_factor is None
        else raytrace_render_camera_distance_factor
    )
    raytrace_render_sphere_radius_fraction_cfg = float(
        getattr(cfg, "analysis_cluster_figure_raytrace_sphere_radius_fraction", 0.0105)
    )
    raytrace_render_sphere_radius_fraction_use = float(
        raytrace_render_sphere_radius_fraction_cfg
        if raytrace_render_sphere_radius_fraction is None
        else raytrace_render_sphere_radius_fraction
    )
    raytrace_render_timeout_sec_cfg = int(
        getattr(cfg, "analysis_cluster_figure_raytrace_timeout_sec", 1200)
    )
    raytrace_render_timeout_sec_use = int(
        raytrace_render_timeout_sec_cfg
        if raytrace_render_timeout_sec is None
        else raytrace_render_timeout_sec
    )
    _cfg_visible_sets_raw = getattr(cfg, "analysis_cluster_figure_visible_sets", None)
    if _cfg_visible_sets_raw is not None and visible_cluster_sets is None:
        visible_cluster_sets = [
            [int(v) for v in str(s).split(",")]
            for s in _cfg_visible_sets_raw
        ]
    if cluster_figure_md_point_size <= 0.0:
        raise ValueError(
            f"analysis_cluster_figure_md_point_size must be > 0, got {cluster_figure_md_point_size}."
        )
    if not (0.0 <= cluster_figure_md_alpha <= 1.0):
        raise ValueError(
            f"analysis_cluster_figure_md_alpha must be in [0, 1], got {cluster_figure_md_alpha}."
        )
    if not np.isfinite(cluster_figure_md_view_elev) or not np.isfinite(cluster_figure_md_view_azim):
        raise ValueError(
            "analysis_cluster_figure_md_view_elev and analysis_cluster_figure_md_view_azim "
            f"must be finite, got elev={cluster_figure_md_view_elev}, "
            f"azim={cluster_figure_md_view_azim}."
        )
    if int(pretty_render_resolution) < 128:
        raise ValueError(
            f"pretty_render_resolution must be >= 128, got {pretty_render_resolution}."
        )
    if int(pretty_render_sphere_radius) < 1:
        raise ValueError(
            f"pretty_render_sphere_radius must be >= 1, got {pretty_render_sphere_radius}."
        )
    if pretty_render_projection_use not in {"orthographic", "ortho", "perspective", "persp"}:
        raise ValueError(
            "pretty_render_projection must be one of "
            "['orthographic', 'ortho', 'perspective', 'persp'], got "
            f"{pretty_render_projection_use!r}."
        )
    if not np.isfinite(pretty_render_fov_use) or not (5.0 <= pretty_render_fov_use <= 130.0):
        raise ValueError(
            "pretty_render_perspective_fov_deg must be finite and in [5, 130], got "
            f"{pretty_render_fov_use}."
        )
    if (
        not np.isfinite(pretty_render_distance_use)
        or pretty_render_distance_use < 1.05
    ):
        raise ValueError(
            "pretty_render_perspective_distance_factor must be finite and >= 1.05, got "
            f"{pretty_render_distance_use}."
        )
    if pretty_render_color_mode_use not in {"matplotlib_match", "flat"}:
        raise ValueError(
            "pretty_render_color_mode must be one of ['matplotlib_match', 'flat'], got "
            f"{pretty_render_color_mode_use!r}."
        )
    if (
        not np.isfinite(pretty_render_saturation_use)
        or pretty_render_saturation_use <= 0.0
    ):
        raise ValueError(
            "pretty_render_saturation_boost must be finite and > 0, got "
            f"{pretty_render_saturation_use}."
        )
    if int(pretty_render_wireframe_use) < 0:
        raise ValueError(
            "pretty_render_wireframe_width must be >= 0, got "
            f"{pretty_render_wireframe_use}."
        )
    if raytrace_blender_executable_use == "":
        raise ValueError("raytrace_blender_executable must be a non-empty string.")
    if int(raytrace_render_resolution_use) < 256:
        raise ValueError(
            "raytrace_render_resolution must be >= 256, got "
            f"{raytrace_render_resolution_use}."
        )
    if int(raytrace_render_samples_use) < 1:
        raise ValueError(
            f"raytrace_render_samples must be >= 1, got {raytrace_render_samples_use}."
        )
    if raytrace_render_projection_use not in {"orthographic", "ortho", "perspective", "persp"}:
        raise ValueError(
            "raytrace_render_projection must be one of "
            "['orthographic', 'ortho', 'perspective', 'persp'], got "
            f"{raytrace_render_projection_use!r}."
        )
    if (
        not np.isfinite(raytrace_render_fov_use)
        or not (5.0 <= raytrace_render_fov_use <= 130.0)
    ):
        raise ValueError(
            "raytrace_render_fov_deg must be finite and in [5, 130], got "
            f"{raytrace_render_fov_use}."
        )
    if (
        not np.isfinite(raytrace_render_camera_distance_use)
        or raytrace_render_camera_distance_use <= 1.0
    ):
        raise ValueError(
            "raytrace_render_camera_distance_factor must be finite and > 1.0, got "
            f"{raytrace_render_camera_distance_use}."
        )
    if (
        not np.isfinite(raytrace_render_sphere_radius_fraction_use)
        or raytrace_render_sphere_radius_fraction_use <= 0.0
    ):
        raise ValueError(
            "raytrace_render_sphere_radius_fraction must be finite and > 0, got "
            f"{raytrace_render_sphere_radius_fraction_use}."
        )
    if int(raytrace_render_timeout_sec_use) < 30:
        raise ValueError(
            "raytrace_render_timeout_sec must be >= 30, got "
            f"{raytrace_render_timeout_sec_use}."
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
    cluster_figure_representative_orientation = str(
        getattr(cfg, "analysis_cluster_figure_representative_orientation", "pca")
    ).strip().lower()
    cluster_figure_representative_view_elev = float(
        getattr(cfg, "analysis_cluster_figure_representative_view_elev", 22.0)
    )
    cluster_figure_representative_view_azim = float(
        getattr(cfg, "analysis_cluster_figure_representative_view_azim", 38.0)
    )
    cluster_figure_representative_projection = str(
        getattr(cfg, "analysis_cluster_figure_representative_projection", "ortho")
    ).strip().lower()
    if cluster_figure_representative_orientation not in {"pca", "none"}:
        raise ValueError(
            "analysis_cluster_figure_representative_orientation must be one of "
            f"['pca', 'none'], got {cluster_figure_representative_orientation!r}."
        )
    if (
        not np.isfinite(cluster_figure_representative_view_elev)
        or not np.isfinite(cluster_figure_representative_view_azim)
    ):
        raise ValueError(
            "analysis_cluster_figure_representative_view_elev and "
            "analysis_cluster_figure_representative_view_azim must be finite, got "
            f"elev={cluster_figure_representative_view_elev}, "
            f"azim={cluster_figure_representative_view_azim}."
        )
    if cluster_figure_representative_projection not in {"ortho", "persp"}:
        raise ValueError(
            "analysis_cluster_figure_representative_projection must be one of "
            f"['ortho', 'persp'], got {cluster_figure_representative_projection!r}."
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
        f"visible_sets={visible_cluster_sets or []}, "
        f"pretty_projection={pretty_render_projection_use}, "
        f"pretty_fov={pretty_render_fov_use:.1f}, "
        f"pretty_dist={pretty_render_distance_use:.2f}, "
        f"pretty_color_mode={pretty_render_color_mode_use}, "
        f"pretty_saturation={pretty_render_saturation_use:.2f}, "
        f"pretty_wireframe={pretty_render_wireframe_use}, "
        f"raytrace_enabled={raytrace_render_enabled_use}, "
        f"raytrace_projection={raytrace_render_projection_use}, "
        f"raytrace_samples={raytrace_render_samples_use}, "
        f"raytrace_res={raytrace_render_resolution_use}, "
        f"raytrace_max_points={raytrace_render_max_points_use}, "
        f"rep_orientation={cluster_figure_representative_orientation}, "
        "rep_view=("
        f"{cluster_figure_representative_view_elev:.1f},"
        f"{cluster_figure_representative_view_azim:.1f}"
        "), "
        f"rep_projection={cluster_figure_representative_projection}"
    )
    _step("Building datamodule")
    dm = build_datamodule(cfg)
    is_synthetic = getattr(cfg.data, "kind", None) == "synthetic"

    dm.setup(stage="fit")
    all_metrics: Dict[str, Any] = {}

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
                representative_orientation_method=cluster_figure_representative_orientation,
                representative_view_elev=cluster_figure_representative_view_elev,
                representative_view_azim=cluster_figure_representative_view_azim,
                representative_projection=cluster_figure_representative_projection,
                visible_cluster_sets=visible_cluster_sets,
                pretty_render_resolution=pretty_render_resolution,
                pretty_render_sphere_radius=pretty_render_sphere_radius,
                pretty_render_projection=pretty_render_projection_use,
                pretty_render_perspective_fov_deg=pretty_render_fov_use,
                pretty_render_perspective_distance_factor=pretty_render_distance_use,
                pretty_render_color_mode=pretty_render_color_mode_use,
                pretty_render_saturation_boost=pretty_render_saturation_use,
                pretty_render_wireframe_width=pretty_render_wireframe_use,
                raytrace_render_enabled=raytrace_render_enabled_use,
                raytrace_blender_executable=raytrace_blender_executable_use,
                raytrace_render_resolution=raytrace_render_resolution_use,
                raytrace_render_max_points=raytrace_render_max_points_use,
                raytrace_render_samples=raytrace_render_samples_use,
                raytrace_render_projection=raytrace_render_projection_use,
                raytrace_render_fov_deg=raytrace_render_fov_use,
                raytrace_render_camera_distance_factor=raytrace_render_camera_distance_use,
                raytrace_render_sphere_radius_fraction=raytrace_render_sphere_radius_fraction_use,
                raytrace_render_timeout_sec=raytrace_render_timeout_sec_use,
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
                            palette="tab10",
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
                                    palette="tab10",
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
            f"  - cluster_figure_set_k{k_fig}/01_md_clusters_all_k{k_fig}[_view*].png: "
            "MD space (all clusters, 4 views)"
        )
        print(
            f"  - cluster_figure_set_k{k_fig}/01_md_clusters_all_k{k_fig}[_view*]_pretty.png: "
            "sphere renders (all views)"
        )
        raytrace_settings = all_metrics["cluster_figure_set"].get("raytrace_render_settings", {})
        if bool(raytrace_settings.get("enabled", False)):
            print(
                f"  - cluster_figure_set_k{k_fig}/01_md_clusters_all_k{k_fig}[_view*]_raytrace.png: "
                "Blender Cycles raytraced renders (all views)"
            )
        vis_sets = all_metrics["cluster_figure_set"].get("visible_cluster_sets", [])
        if vis_sets:
            for s in vis_sets:
                tag = "-".join(str(c) for c in s)
                print(
                    f"  - cluster_figure_set_k{k_fig}/02_md_clusters_set_{tag}_k{k_fig}[_pretty].png: "
                    f"clusters {tag}"
                )
                if bool(raytrace_settings.get("enabled", False)):
                    print(
                        f"  - cluster_figure_set_k{k_fig}/02_md_clusters_set_{tag}_k{k_fig}_raytrace.png: "
                        f"Blender Cycles raytraced clusters {tag}"
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
            print("  - cluster_profiles_by_k/k_*/cluster_XX_structures_points_only.png: same structures without connecting lines")
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
        "--pretty_render_resolution",
        type=int,
        default=2200,
        help="Image width/height in pixels for pretty renders (default: 2200).",
    )
    parser.add_argument(
        "--pretty_render_sphere_radius",
        type=int,
        default=12,
        help=(
            "Sphere size scale for pretty renders (default: 12). "
            "Actual ball size is auto-estimated from MD point density and multiplied by this value/12."
        ),
    )
    parser.add_argument(
        "--pretty_render_projection",
        type=str,
        default=None,
        choices=["orthographic", "ortho", "perspective", "persp"],
        help=(
            "Projection mode for pretty renders. If omitted, uses "
            "analysis_cluster_figure_pretty_projection from config."
        ),
    )
    parser.add_argument(
        "--pretty_render_perspective_fov_deg",
        type=float,
        default=None,
        help=(
            "Perspective field-of-view in degrees for pretty renders. If omitted, "
            "uses analysis_cluster_figure_pretty_fov_deg from config."
        ),
    )
    parser.add_argument(
        "--pretty_render_perspective_distance_factor",
        type=float,
        default=None,
        help=(
            "Camera distance factor for perspective renders. If omitted, uses "
            "analysis_cluster_figure_pretty_distance_factor from config."
        ),
    )
    parser.add_argument(
        "--pretty_render_color_mode",
        type=str,
        default=None,
        choices=["matplotlib_match", "flat"],
        help=(
            "Color mode for pretty renders. 'matplotlib_match' reuses the same "
            "per-point color logic as static matplotlib snapshots."
        ),
    )
    parser.add_argument(
        "--pretty_render_saturation_boost",
        type=float,
        default=None,
        help=(
            "Global color saturation multiplier for pretty renders. If omitted, "
            "uses analysis_cluster_figure_pretty_saturation_boost from config."
        ),
    )
    parser.add_argument(
        "--pretty_render_wireframe_width",
        type=int,
        default=None,
        help=(
            "Cube wireframe line width for pretty renders. Set 0 to disable. "
            "If omitted, uses analysis_cluster_figure_pretty_wireframe_width."
        ),
    )
    parser.add_argument(
        "--raytrace_render_enabled",
        action="store_true",
        default=None,
        help=(
            "Generate additional Blender Cycles raytraced renders "
            "(*_raytrace.png) alongside existing outputs."
        ),
    )
    parser.add_argument(
        "--raytrace_blender_executable",
        type=str,
        default=None,
        help=(
            "Blender executable path/name for raytrace rendering. If omitted, uses "
            "analysis_cluster_figure_raytrace_blender_executable."
        ),
    )
    parser.add_argument(
        "--raytrace_render_resolution",
        type=int,
        default=None,
        help=(
            "Image width/height for raytraced renders. If omitted, uses "
            "analysis_cluster_figure_raytrace_resolution."
        ),
    )
    parser.add_argument(
        "--raytrace_render_max_points",
        type=int,
        default=None,
        help=(
            "Deprecated. Raytraced rendering now always uses all points. "
            "Set <=0 or omit this argument."
        ),
    )
    parser.add_argument(
        "--raytrace_render_samples",
        type=int,
        default=None,
        help=(
            "Cycles samples for raytraced render. If omitted, uses "
            "analysis_cluster_figure_raytrace_samples."
        ),
    )
    parser.add_argument(
        "--raytrace_render_projection",
        type=str,
        default=None,
        choices=["orthographic", "ortho", "perspective", "persp"],
        help=(
            "Projection mode for raytraced render. If omitted, uses "
            "analysis_cluster_figure_raytrace_projection."
        ),
    )
    parser.add_argument(
        "--raytrace_render_fov_deg",
        type=float,
        default=None,
        help=(
            "Perspective FOV for raytraced render. If omitted, uses "
            "analysis_cluster_figure_raytrace_fov_deg."
        ),
    )
    parser.add_argument(
        "--raytrace_render_camera_distance_factor",
        type=float,
        default=None,
        help=(
            "Camera distance factor for raytraced render. If omitted, uses "
            "analysis_cluster_figure_raytrace_camera_distance_factor."
        ),
    )
    parser.add_argument(
        "--raytrace_render_sphere_radius_fraction",
        type=float,
        default=None,
        help=(
            "Raytrace sphere size scale (default config value 0.0105 corresponds to 1.0x "
            "auto-estimated physical size). If omitted, uses "
            "analysis_cluster_figure_raytrace_sphere_radius_fraction."
        ),
    )
    parser.add_argument(
        "--raytrace_render_timeout_sec",
        type=int,
        default=None,
        help=(
            "Timeout in seconds for each Blender raytrace render. If omitted, uses "
            "analysis_cluster_figure_raytrace_timeout_sec."
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

    run_post_training_analysis(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        cuda_device=int(args.cuda_device),
        cfg=None,
        max_batches_latent=args.max_batches_latent,
        max_samples_visualization=args.max_samples_visualization,
        data_files_override=args.data_file,
        visible_cluster_sets=vis_sets,
        pretty_render_resolution=int(args.pretty_render_resolution),
        pretty_render_sphere_radius=int(args.pretty_render_sphere_radius),
        pretty_render_projection=args.pretty_render_projection,
        pretty_render_perspective_fov_deg=args.pretty_render_perspective_fov_deg,
        pretty_render_perspective_distance_factor=args.pretty_render_perspective_distance_factor,
        pretty_render_color_mode=args.pretty_render_color_mode,
        pretty_render_saturation_boost=args.pretty_render_saturation_boost,
        pretty_render_wireframe_width=args.pretty_render_wireframe_width,
        raytrace_render_enabled=args.raytrace_render_enabled,
        raytrace_blender_executable=args.raytrace_blender_executable,
        raytrace_render_resolution=args.raytrace_render_resolution,
        raytrace_render_max_points=args.raytrace_render_max_points,
        raytrace_render_samples=args.raytrace_render_samples,
        raytrace_render_projection=args.raytrace_render_projection,
        raytrace_render_fov_deg=args.raytrace_render_fov_deg,
        raytrace_render_camera_distance_factor=args.raytrace_render_camera_distance_factor,
        raytrace_render_sphere_radius_fraction=args.raytrace_render_sphere_radius_fraction,
        raytrace_render_timeout_sec=args.raytrace_render_timeout_sec,
    )


if __name__ == "__main__":
    main()
