import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

sys.path.append(os.getcwd())

from src.data_utils.data_module import RealPointCloudDataModule, SyntheticPointCloudDataModule
from src.training_methods.contrastive_learning.contrastive_module import BarlowTwinsModule
from src.utils.model_utils import load_model_from_checkpoint

from .cluster_profiles import resolve_point_scale
from .cluster_rendering import _build_cluster_representative_render_cache
from .clustering import _build_clustering_state, _run_optional_hdbscan_analysis, prepare_clustering_features
from .config import (
    DEFAULT_ANALYSIS_CONFIG_PATH, _positive_int_or_none, _print_resolved_analysis_settings,
    _resolve_analysis_files, _resolve_analysis_settings, _resolve_figure_set_settings,
    _resolve_input_settings, _resolve_optional_cluster_k, _resolve_run_settings,
    build_runtime_model_config, load_checkpoint_analysis_config, load_checkpoint_training_config,
)
from .figure_sets import (
    build_shared_cluster_color_map, print_figure_set_summary, render_cluster_figure_outputs,
    resolve_snapshot_figure_layout, write_figure_only_metrics,
)
from .inference_cache import (
    _build_inference_cache_spec, _inference_cache_paths, _inference_cache_spec_hash,
    _load_inference_cache, _save_inference_cache, _validate_inference_cache_arrays,
)
from .latent_vis import print_analysis_summary, run_equivariance_evaluation, run_pca_and_latent_stats, run_tsne_visualizations
from .md_outputs import build_md_metrics
from .real_md_qualitative import run_real_md_qualitative_analysis
from .utils import _unwrap_dataset, build_real_coords_dataloader, gather_inference_batches


# ---------------------------------------------------------------------------
# Small helpers that stay in the orchestrator
# ---------------------------------------------------------------------------

def _extract_class_names(dataset: Any) -> Dict[int, str] | None:
    dataset = _unwrap_dataset(dataset)
    class_names_raw = getattr(dataset, "class_names", None)
    if class_names_raw is None:
        return None
    class_names = dict(class_names_raw)
    return {int(k): str(v) for k, v in class_names.items()}


def _build_analysis_dataloader(
    cfg: DictConfig,
    dm: Any,
    *,
    is_synthetic: bool,
    dataloader_num_workers: int,
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
            num_workers=int(dataloader_num_workers),
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            persistent_workers=bool(int(dataloader_num_workers) > 0),
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
    input_settings: Any,
    *,
    is_synthetic: bool,
    md_use_all_points: bool,
) -> int | None:
    max_samples_total = input_settings.max_samples_total
    if max_samples_total is None and not is_synthetic:
        max_samples_total = 20000
    if not is_synthetic and md_use_all_points:
        max_samples_total = None
    return _positive_int_or_none(max_samples_total)


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


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_barlow_model(
    checkpoint_path: str, cuda_device: int = 0, cfg: DictConfig | None = None
) -> Tuple[BarlowTwinsModule, DictConfig, str]:
    """Restore the contrastive module together with its Hydra cfg and device string."""
    if cfg is None:
        cfg = load_checkpoint_training_config(checkpoint_path)
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


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def run_post_training_analysis(
    checkpoint_path: str | None = None,
    output_dir: str | None = None,
    cuda_device: int | None = None,
    *,
    analysis_config_path: str | None = None,
    analysis_cfg: DictConfig | None = None,
) -> Dict[str, Any]:
    """Generate qualitative and quantitative diagnostics for contrastive checkpoints."""
    t0 = time.perf_counter()
    step_idx = [0]

    def _step(msg: str) -> None:
        step_idx[0] += 1
        elapsed = time.perf_counter() - t0
        print(f"[analysis][step {step_idx[0]}][{elapsed:7.1f}s] {msg}")

    # ── Config resolution ──────────────────────────────────────────────
    _step("Loading analysis config")
    if analysis_cfg is None:
        analysis_cfg = load_checkpoint_analysis_config(analysis_config_path)
    if not isinstance(analysis_cfg, DictConfig):
        raise TypeError(
            "analysis_cfg must be a DictConfig when provided, "
            f"got {type(analysis_cfg)!r}."
        )
    run_settings = _resolve_run_settings(
        analysis_cfg,
        checkpoint_path_override=checkpoint_path,
        output_dir_override=output_dir,
        cuda_device_override=cuda_device,
    )
    out_dir = Path(run_settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _step("Loading checkpoint training config")
    cfg = build_runtime_model_config(run_settings.checkpoint_path, analysis_cfg)
    input_settings = _resolve_input_settings(analysis_cfg)
    model: BarlowTwinsModule | None = None
    device = f"cuda:{run_settings.cuda_device}" if torch.cuda.is_available() else "cpu"
    analysis_source_names: list[str] | None = None
    analysis_files = _resolve_analysis_files(cfg, input_settings)
    if analysis_files is not None:
        analysis_source_names = _configure_real_analysis_inputs(cfg, analysis_files)
        print(f"Analysis data_files: {analysis_files}")
        if analysis_source_names and len(analysis_source_names) > 1:
            print(f"Per-snapshot analysis sources: {analysis_source_names}")

    with open_dict(cfg):
        cfg.num_workers = int(input_settings.dataloader_num_workers)
    print(f"Analysis dataloader workers: {input_settings.dataloader_num_workers}")
    analysis_settings = _resolve_analysis_settings(analysis_cfg, cfg)
    hdbscan_settings = analysis_settings.hdbscan
    real_md_selected_k = _resolve_optional_cluster_k(
        OmegaConf.select(analysis_cfg, "real_md.selected_k", default=None),
        field_name="real_md.selected_k",
    )
    if real_md_selected_k is None:
        real_md_selected_k = int(analysis_settings.primary_k)
    figure_settings = _resolve_figure_set_settings(
        analysis_cfg,
        cfg,
        out_dir=out_dir,
        primary_k=int(analysis_settings.primary_k),
    )
    _print_resolved_analysis_settings(
        analysis_settings=analysis_settings,
        figure_settings=figure_settings,
        real_md_selected_k=real_md_selected_k,
    )
    is_synthetic = getattr(cfg.data, "kind", None) == "synthetic"

    # ── Data loading ───────────────────────────────────────────────────
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
        dataloader_num_workers=int(input_settings.dataloader_num_workers),
    )

    class_names = _extract_class_names(dm.train_dataset)
    print(f"Loaded class names: {class_names}")

    max_batches_latent = input_settings.max_batches_latent
    max_samples_total = _resolve_analysis_max_samples_total(
        input_settings,
        is_synthetic=is_synthetic,
        md_use_all_points=analysis_settings.md_use_all_points,
    )

    # ── Inference / cache ──────────────────────────────────────────────
    seed_base = int(analysis_settings.seed_base)
    clustering_random_state = int(analysis_settings.seed_base)
    cache_spec = _build_inference_cache_spec(
        checkpoint_path=run_settings.checkpoint_path,
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
                "figure_set.figure_only requires a valid inference cache because it does not "
                "run model inference. "
                f"Cache load failed: {cache_msg}. "
                "Run the full analysis once with figure_set.figure_only=false to populate "
                f"{out_dir / analysis_settings.inference_cache_file}."
            )
    else:
        _step("Loading model")
        model, cfg, device = load_barlow_model(
            run_settings.checkpoint_path,
            cuda_device=run_settings.cuda_device,
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
        resolve_point_scale(cfg)
        if figure_settings.profile_point_scale_enabled
        else 1.0
    )
    print(
        "Representative point scaling: "
        f"enabled={figure_settings.profile_point_scale_enabled}, point_scale={point_scale:.6g}"
    )

    # ── Clustering feature preparation ─────────────────────────────────
    _step("Preparing clustering features")
    clustering_features, clustering_feature_prep = prepare_clustering_features(
        cache["inv_latents"],
        random_state=int(clustering_random_state),
        l2_normalize=analysis_settings.cluster_l2_normalize,
        standardize=analysis_settings.cluster_standardize,
        pca_variance=analysis_settings.cluster_pca_var,
        pca_max_components=analysis_settings.cluster_pca_max_components,
    )
    dataset_obj = getattr(dl, "dataset", None)
    snapshot_layout = resolve_snapshot_figure_layout(
        dataset_obj,
        is_synthetic=is_synthetic,
        n_samples=n_samples,
        analysis_source_names=analysis_source_names,
    )
    snapshot_source_groups = snapshot_layout.source_groups
    snapshot_output_names = snapshot_layout.output_names
    multi_snapshot_real = snapshot_layout.multi_snapshot_real

    figure_set_run_kwargs = figure_settings.build_run_kwargs(
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

    # ── Figure-only early return ───────────────────────────────────────
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
        cluster_figure_set, snapshot_figure_sets = render_cluster_figure_outputs(
            out_dir=out_dir,
            dataloader=dl,
            figure_settings=figure_settings,
            figure_set_run_kwargs=figure_set_run_kwargs,
            labels_for_k=cluster_labels_by_k[int(figure_settings.k)],
            latents=cache["inv_latents"],
            coords=coords,
            dataset_obj=dataset_obj,
            snapshot_layout=snapshot_layout,
            analysis_source_names=analysis_source_names,
            step=_step,
        )
        if cluster_figure_set is not None:
            all_metrics["cluster_figure_set"] = cluster_figure_set
        if snapshot_figure_sets is not None:
            all_metrics["cluster_figure_sets_by_snapshot"] = snapshot_figure_sets

        _step("Writing metrics")
        metrics_path = out_dir / "analysis_metrics.json"
        merged_metrics = write_figure_only_metrics(
            metrics_path=metrics_path,
            all_metrics=all_metrics,
            multi_snapshot_real=multi_snapshot_real,
        )
        print_figure_set_summary(
            all_metrics,
            n_samples=n_samples,
            out_dir=out_dir,
            elapsed=time.perf_counter() - t0,
        )
        return merged_metrics

    # ── PCA + latent statistics ────────────────────────────────────────
    all_metrics.update(
        run_pca_and_latent_stats(
            cache,
            out_dir,
            class_names=class_names,
            step=_step,
        )
    )

    # ── Clustering ─────────────────────────────────────────────────────
    _step("Computing clustering labels")
    clustering_requested_k_values = list(analysis_settings.cluster_k_values)
    if figure_settings.enabled and int(figure_settings.k) not in clustering_requested_k_values:
        clustering_requested_k_values.append(int(figure_settings.k))
    if (
        not is_synthetic
        and int(real_md_selected_k) not in clustering_requested_k_values
    ):
        clustering_requested_k_values.append(int(real_md_selected_k))
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

    primary_k = int(analysis_settings.primary_k)
    if primary_k not in configured_k_values:
        raise KeyError(
            "Requested clustering.primary_k is not available in configured clustering results. "
            f"Requested k={primary_k}, available={configured_k_values}."
        )
    if int(real_md_selected_k) not in configured_k_values:
        raise KeyError(
            "Requested real_md.selected_k is not available in configured clustering results. "
            f"Requested k={int(real_md_selected_k)}, available={configured_k_values}."
        )
    cluster_labels = cluster_labels_by_k[primary_k]

    # ── Shared representative render cache ─────────────────────────────
    real_md_enabled = bool(OmegaConf.select(analysis_cfg, "real_md.enabled", default=True))
    real_md_profiles_enabled = bool(
        OmegaConf.select(analysis_cfg, "real_md.profiles.enabled", default=True)
    )
    shared_representative_render_cache: dict[str, Any] | None = None
    if (
        not is_synthetic
        and figure_settings.enabled
        and not multi_snapshot_real
        and real_md_enabled
        and real_md_profiles_enabled
        and dataset_obj is not None
        and int(real_md_selected_k) == int(figure_settings.k)
        and int(figure_settings.real_md_profile_target_points)
        == int(figure_settings.representative_points)
    ):
        _step("Preparing shared representative structures")
        shared_representative_render_cache = _build_cluster_representative_render_cache(
            dataset_obj,
            np.asarray(cache["inv_latents"], dtype=np.float32),
            np.asarray(cluster_labels_by_k[int(figure_settings.k)], dtype=int),
            build_shared_cluster_color_map(
                cluster_labels_by_k[int(figure_settings.k)],
                cluster_color_assignment=figure_settings.cluster_color_assignment,
            ),
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

    # ── Figure set rendering ───────────────────────────────────────────
    cluster_figure_set, snapshot_figure_sets = render_cluster_figure_outputs(
        out_dir=out_dir,
        dataloader=dl,
        figure_settings=figure_settings,
        figure_set_run_kwargs=figure_set_run_kwargs,
        labels_for_k=cluster_labels_by_k[int(figure_settings.k)],
        latents=cache["inv_latents"],
        coords=coords,
        dataset_obj=dataset_obj,
        snapshot_layout=snapshot_layout,
        analysis_source_names=analysis_source_names,
        step=_step,
        representative_render_cache=shared_representative_render_cache,
    )
    if cluster_figure_set is not None:
        all_metrics["cluster_figure_set"] = cluster_figure_set
    if snapshot_figure_sets is not None:
        all_metrics["cluster_figure_sets_by_snapshot"] = snapshot_figure_sets

    # ── Shared cluster color maps ──────────────────────────────────────
    _step("Building shared cluster color maps")
    shared_cluster_color_maps_by_k = {
        int(k_val_inner): build_shared_cluster_color_map(
            cluster_labels_by_k[int(k_val_inner)],
            cluster_color_assignment=figure_settings.cluster_color_assignment,
        )
        for k_val_inner in configured_k_values
    }
    shared_cluster_color_map = shared_cluster_color_maps_by_k[int(primary_k)]

    # ── t-SNE ──────────────────────────────────────────────────────────
    run_tsne_visualizations(
        cache,
        out_dir,
        analysis_cfg=analysis_cfg,
        cluster_labels_by_k=cluster_labels_by_k,
        cluster_methods_by_k=cluster_methods_by_k,
        configured_k_values=configured_k_values,
        primary_k=primary_k,
        shared_cluster_color_maps_by_k=shared_cluster_color_maps_by_k,
        class_names=class_names,
        is_synthetic=is_synthetic,
        clustering_random_state=clustering_random_state,
        tsne_max_samples=analysis_settings.tsne_max_samples,
        tsne_n_iter=analysis_settings.tsne_n_iter,
        cluster_method=analysis_settings.cluster_method,
        step=_step,
    )

    # ── MD-space cluster outputs ───────────────────────────────────────
    _step("Saving coordinate-space clustering outputs")
    interactive_max_points = analysis_settings.interactive_max_points
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
    all_metrics[md_metrics_key] = build_md_metrics(
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

    # ── Real-MD qualitative analysis ───────────────────────────────────
    if not is_synthetic and real_md_enabled:
        _step("Running real-MD qualitative analysis")
        shared_real_md_color_map = shared_cluster_color_maps_by_k.get(
            int(real_md_selected_k),
            shared_cluster_color_map,
        )
        all_metrics["real_md_qualitative"] = run_real_md_qualitative_analysis(
            out_dir=out_dir,
            model_cfg=cfg,
            analysis_cfg=analysis_cfg,
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

    # ── Equivariance ───────────────────────────────────────────────────
    all_metrics.update(
        run_equivariance_evaluation(
            model,
            dl,
            device,
            out_dir,
            analysis_cfg=analysis_cfg,
            step=_step,
        )
    )

    # ── Write metrics & summary ────────────────────────────────────────
    _step("Writing metrics")
    metrics_path = out_dir / "analysis_metrics.json"
    with metrics_path.open("w") as handle:
        json.dump(all_metrics, handle, indent=2)

    elapsed = time.perf_counter() - t0
    print_analysis_summary(
        all_metrics,
        n_samples=n_samples,
        out_dir=out_dir,
        elapsed=elapsed,
    )
    print_figure_set_summary(
        all_metrics,
        n_samples=n_samples,
        out_dir=out_dir,
        elapsed=elapsed,
    )

    return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-training analysis pipeline")
    parser.add_argument(
        "config",
        nargs="?",
        default=str(DEFAULT_ANALYSIS_CONFIG_PATH),
        help=f"Path to the analysis config YAML (default: {DEFAULT_ANALYSIS_CONFIG_PATH})",
    )
    args = parser.parse_args()
    run_post_training_analysis(analysis_config_path=args.config)


if __name__ == "__main__":
    # Support `python pipeline.py` in addition to `python -m ...analysis.pipeline`
    if __package__ is None or __package__ == "":
        import importlib, pathlib

        _this = pathlib.Path(__file__).resolve()
        _pkg = "src.analysis.pipeline"
        sys.modules.pop(__name__, None)
        mod = importlib.import_module(_pkg)
        mod.main()
    else:
        main()
