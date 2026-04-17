import argparse
from dataclasses import replace
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

if __package__ is None or __package__ == "":
    # Allow `python src/analysis/pipeline.py ...` from the repo root by making
    # the project importable before any relative imports execute.
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    __package__ = "src.analysis"

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from src.data_utils.data_module import (
    RealPointCloudDataModule,
    SyntheticPointCloudDataModule,
    TemporalLAMMPSDataModule,
    _resolve_temporal_window_start_frames,
)
from src.data_utils.temporal_lammps_dataset import TemporalLAMMPSDumpDataset
from src.training_methods.contrastive_learning.vicreg_module import VICRegModule
from src.utils.model_utils import load_model_from_checkpoint

from .cluster_profiles import resolve_point_scale
from .cluster_rendering import _build_cluster_representative_render_cache
from .clustering import (
    _build_clustering_state,
    _run_optional_hdbscan_analysis,
    build_clustering_method_comparison,
    prepare_clustering_features,
)
from .config import (
    DEFAULT_ANALYSIS_CONFIG_PATH, _positive_int_or_none, _print_resolved_analysis_settings,
    _resolve_analysis_files, _resolve_analysis_settings, _resolve_figure_set_settings,
    _resolve_input_settings, _resolve_optional_cluster_k, _resolve_run_settings,
    _validate_overlap_fraction,
    build_runtime_model_config, load_checkpoint_analysis_config, load_checkpoint_training_config,
)
from .dynamic_motif import run_dynamic_motif_analysis
from .dynamic_motif_cache import collect_tmf_inference_cache
from .figure_sets import (
    build_shared_cluster_color_map, filter_snapshot_figure_layout, print_figure_set_summary,
    render_cluster_figure_outputs, resolve_snapshot_figure_layout, write_figure_only_metrics,
)
from .inference_cache import (
    _build_inference_cache_spec, _inference_cache_paths, _inference_cache_spec_hash,
    _load_inference_cache, _save_inference_cache, _validate_inference_cache_arrays,
)
from .latent_vis import print_analysis_summary, run_equivariance_evaluation, run_pca_and_latent_stats, run_tsne_visualizations
from .md_outputs import build_md_metrics
from .output_layout import real_md_outputs_root, real_md_outputs_root_for_k
from .real_md_qualitative import append_dynamic_motif_summary, run_real_md_qualitative_analysis
from .temporal_real import (
    build_temporal_real_analysis_bundle,
    build_temporal_real_single_snapshot_bundle,
    resolve_temporal_real_inference_spec,
    resolve_temporal_real_snapshot_subset,
    temporal_real_analysis_enabled,
)
from .utils import _sample_indices, _unwrap_dataset, build_real_coords_dataloader, gather_inference_batches


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
    inference_batch_size: int,
    dataloader_num_workers: int,
) -> torch.utils.data.DataLoader:
    if str(getattr(cfg.data, "kind", "")).strip().lower() == "temporal_lammps":
        if not isinstance(dm, TemporalLAMMPSDataModule):
            raise TypeError(
                "Temporal analysis requires a TemporalLAMMPSDataModule, "
                f"got {type(dm)!r}."
            )
        data_cfg = cfg.data
        dump_file = getattr(data_cfg, "dump_file", None)
        if dump_file is None or str(dump_file).strip() == "":
            raise ValueError(
                "Temporal analysis requires cfg.data.dump_file to build a full-sequence analysis dataloader."
            )
        cache_dir = getattr(data_cfg, "cache_dir", None)
        scan = TemporalLAMMPSDumpDataset.scan_dump_file(dump_file, cache_dir=cache_dir)
        radius = dm._resolve_radius(
            dump_file=dump_file,
            data_cfg=data_cfg,
            frame_start=int(getattr(data_cfg, "frame_start", 0)),
            num_points=int(getattr(data_cfg, "num_points", 0)),
        )
        anchor_frames = _resolve_temporal_window_start_frames(
            frame_count=int(scan.frame_count),
            sequence_length=int(getattr(data_cfg, "sequence_length", 0)),
            frame_stride=int(getattr(data_cfg, "frame_stride", 1)),
            frame_start=int(getattr(data_cfg, "frame_start", 0)),
            frame_stop=getattr(data_cfg, "frame_stop", None),
            window_stride=int(getattr(data_cfg, "window_stride", 1)),
        )
        if not anchor_frames:
            raise ValueError(
                "Temporal analysis did not find any valid anchor frames for the configured sequence. "
                f"dump_file={dump_file}, sequence_length={int(getattr(data_cfg, 'sequence_length', 0))}, "
                f"frame_stride={int(getattr(data_cfg, 'frame_stride', 1))}, "
                f"frame_start={int(getattr(data_cfg, 'frame_start', 0))}, "
                f"frame_stop={getattr(data_cfg, 'frame_stop', None)}, "
                f"window_stride={int(getattr(data_cfg, 'window_stride', 1))}."
            )
        full_dataset = TemporalLAMMPSDumpDataset(
            dump_file=dump_file,
            sequence_length=int(getattr(data_cfg, "sequence_length", 0)),
            num_points=int(getattr(data_cfg, "num_points", 0)),
            radius=float(radius),
            frame_stride=int(getattr(data_cfg, "frame_stride", 1)),
            window_stride=int(getattr(data_cfg, "window_stride", 1)),
            frame_start=int(getattr(data_cfg, "frame_start", 0)),
            frame_stop=getattr(data_cfg, "frame_stop", None),
            anchor_frame_indices=anchor_frames,
            center_selection_mode=getattr(data_cfg, "center_selection_mode", None),
            center_atom_ids=getattr(data_cfg, "center_atom_ids", None),
            center_atom_stride=getattr(data_cfg, "center_atom_stride", None),
            max_center_atoms=getattr(data_cfg, "max_center_atoms", None),
            center_selection_seed=int(getattr(data_cfg, "center_selection_seed", 0)),
            center_grid_overlap=getattr(data_cfg, "center_grid_overlap", None),
            center_grid_reference_frame_index=getattr(data_cfg, "center_grid_reference_frame_index", None),
            normalize=bool(getattr(data_cfg, "normalize", True)),
            center_neighborhoods=bool(getattr(data_cfg, "center_neighborhoods", True)),
            selection_method=str(getattr(data_cfg, "selection_method", "closest")),
            cache_dir=cache_dir,
            rebuild_cache=False,
            tree_cache_size=int(getattr(data_cfg, "tree_cache_size", 4)),
            precompute_neighbor_indices=bool(
                getattr(data_cfg, "precompute_neighbor_indices", False)
            ),
            build_lock_timeout_sec=float(
                getattr(data_cfg, "build_lock_timeout_sec", 7200.0)
            ),
            build_lock_stale_sec=float(
                getattr(data_cfg, "build_lock_stale_sec", 86400.0)
            ),
        )
        dm.batch_size = int(inference_batch_size)
        dm.num_workers = int(dataloader_num_workers)
        print(
            "Temporal data detected: using a full anchor-frame dataset for sequence-aware analysis "
            f"({len(anchor_frames)} windows, batch_size={int(inference_batch_size)})."
        )
        return dm._temporal_loader(
            full_dataset,
            shuffle_windows=False,
            shuffle_centers=False,
            drop_last=False,
            mixed_windows_per_batch=None,
        )

    print("Using ALL dataset splits (train + test) for latent analysis")
    if is_synthetic:
        train_dataset = getattr(dm, "train_dataset", None)
        test_dataset = getattr(dm, "test_dataset", None)
        if train_dataset is None or test_dataset is None:
            raise ValueError("Synthetic datamodule is missing train/test datasets.")
        combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        return torch.utils.data.DataLoader(
            combined_dataset,
            batch_size=int(inference_batch_size),
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
        batch_size=int(inference_batch_size),
    )
    print(
        "Real data detected: using full dataset for local-structure clustering visualization"
    )
    return dl


def _resolve_analysis_inference_batch_size(
    cfg: DictConfig,
    input_settings: Any,
) -> int:
    batch_size = input_settings.inference_batch_size
    if batch_size is None:
        batch_size = int(cfg.batch_size)
    resolved = int(batch_size)
    if resolved < 1:
        raise ValueError(f"Analysis inference batch size must be >= 1, got {resolved}.")
    return resolved


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


def _collect_clustering_fit_cache(
    *,
    analysis_cfg: DictConfig,
    fit_settings: Any,
    checkpoint_path: str,
    out_dir: Path,
    model_cfg_for_module: DictConfig,
    model: Any | None,
    cuda_device: int,
    seed_base: int,
    figure_only: bool,
    progress_every_batches: int,
) -> tuple[dict[str, np.ndarray], DictConfig, list[str] | None, Any, bool]:
    fit_cfg = build_runtime_model_config(
        checkpoint_path,
        analysis_cfg,
        data_config_path_override=fit_settings.data_config_path,
    )
    fit_kind = str(getattr(fit_cfg.data, "kind", "")).strip().lower()
    if fit_kind not in {"real", "synthetic"}:
        raise ValueError(
            "clustering.fit_inputs currently supports only static real/synthetic datasets. "
            f"Resolved fit data.kind={fit_kind!r}; provide clustering.fit_inputs.data_config "
            "for a static dataset such as configs/data/data_ae_Al_80.yaml."
        )

    fit_source_names: list[str] | None = None
    fit_analysis_files = None
    if fit_kind == "real":
        fit_analysis_files = _resolve_analysis_files(
            fit_cfg,
            fit_settings.input_settings,
        )
        fit_source_names = _configure_real_analysis_inputs(fit_cfg, fit_analysis_files)
        print(f"Clustering-fit data_files: {fit_analysis_files}")

    with open_dict(fit_cfg):
        fit_cfg.num_workers = int(fit_settings.input_settings.dataloader_num_workers)
    fit_inference_batch_size = _resolve_analysis_inference_batch_size(
        fit_cfg,
        fit_settings.input_settings,
    )
    fit_is_synthetic = fit_kind == "synthetic"
    fit_dm = build_datamodule(
        fit_cfg,
        require_coords_for_real=not fit_is_synthetic,
    )
    fit_dm.setup(stage="fit")
    fit_dl = _build_analysis_dataloader(
        fit_cfg,
        fit_dm,
        is_synthetic=fit_is_synthetic,
        inference_batch_size=int(fit_inference_batch_size),
        dataloader_num_workers=int(fit_settings.input_settings.dataloader_num_workers),
    )
    fit_max_samples_total = _positive_int_or_none(
        fit_settings.input_settings.max_samples_total
    )
    fit_cache_spec = _build_inference_cache_spec(
        checkpoint_path=checkpoint_path,
        cfg=fit_cfg,
        inference_batch_size=int(fit_inference_batch_size),
        max_batches_latent=fit_settings.input_settings.max_batches_latent,
        max_samples_total=fit_max_samples_total,
        seed_base=int(seed_base),
        temporal_real_selection=None,
        temporal_sequence_inference=None,
        collector_mode="generic",
    )

    fit_cache: dict[str, np.ndarray] | None = None
    fit_cache_loaded = False
    if fit_settings.cache_enabled and not fit_settings.cache_force_recompute:
        fit_cache, fit_cache_msg = _load_inference_cache(
            out_dir=out_dir,
            cache_filename=fit_settings.cache_file,
            expected_spec=fit_cache_spec,
        )
        fit_cache_loaded = fit_cache is not None
        print(f"[analysis][clustering-fit cache] {fit_cache_msg}")
    elif fit_settings.cache_enabled and fit_settings.cache_force_recompute:
        print("[analysis][clustering-fit cache] Forced recompute requested; skipping cache load.")

    if fit_cache is None:
        if figure_only:
            raise RuntimeError(
                "figure_set.figure_only requires a valid clustering-fit cache when "
                "clustering.fit_inputs.enabled=true. "
                f"Missing cache: {out_dir / fit_settings.cache_file}. "
                "Run the full analysis once with figure_set.figure_only=false to populate it."
            )
        if model is None:
            model, _, _ = load_vicreg_model(
                checkpoint_path,
                cuda_device=int(cuda_device),
                cfg=model_cfg_for_module,
            )
        fit_cache = gather_inference_batches(
            model,
            fit_dl,
            f"cuda:{int(cuda_device)}" if torch.cuda.is_available() else "cpu",
            max_batches=fit_settings.input_settings.max_batches_latent,
            max_samples_total=fit_max_samples_total,
            collect_coords=True,
            seed_base=int(seed_base),
            progress_every_batches=int(progress_every_batches),
            verbose=True,
            temporal_sequence_mode="static_anchor",
            temporal_static_frame_index=0,
        )
        _validate_inference_cache_arrays(fit_cache)
        if fit_settings.cache_enabled:
            _save_inference_cache(
                out_dir=out_dir,
                cache_filename=fit_settings.cache_file,
                cache=fit_cache,
                spec=fit_cache_spec,
            )
            fit_cache_npz, _ = _inference_cache_paths(out_dir, fit_settings.cache_file)
            print(f"[analysis][clustering-fit cache] Saved inference cache: {fit_cache_npz}")

    _validate_inference_cache_arrays(fit_cache)
    return fit_cache, fit_cfg, fit_source_names, model, bool(fit_cache_loaded)


def _concatenate_inference_caches(
    cache_parts: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if not cache_parts:
        raise ValueError("cache_parts must be non-empty.")
    merged: dict[str, np.ndarray] = {}
    for key in ("inv_latents", "eq_latents", "phases", "coords", "instance_ids"):
        arrays = [np.asarray(part[key]) for part in cache_parts]
        if not arrays:
            raise RuntimeError(f"No arrays were collected for cache key {key!r}.")
        merged[key] = (
            arrays[0].copy()
            if len(arrays) == 1
            else np.concatenate(arrays, axis=0)
        )
    _validate_inference_cache_arrays(merged)
    return merged


def _resolve_temporal_snapshot_visualization_regular_grid_overlap(
    analysis_cfg: DictConfig,
) -> tuple[float, float]:
    snapshot_cfg = OmegaConf.select(
        analysis_cfg,
        "inputs.temporal_real.snapshot_visualization",
        default=None,
    )
    regular_grid_overlap_raw = OmegaConf.select(
        snapshot_cfg,
        "regular_grid_overlap",
        default=None,
    )
    if regular_grid_overlap_raw is not None:
        regular_grid_overlap = float(regular_grid_overlap_raw)
        if not np.isfinite(regular_grid_overlap) or regular_grid_overlap >= 2.0:
            raise ValueError(
                "inputs.temporal_real.snapshot_visualization.regular_grid_overlap "
                f"must be finite and < 2.0, got {regular_grid_overlap_raw!r}."
            )
        return float(regular_grid_overlap), float(regular_grid_overlap - 1.0)

    static_overlap_fraction = _validate_overlap_fraction(
        OmegaConf.select(
            snapshot_cfg,
            "static_overlap_fraction",
            default=0.5,
        )
    )
    return float(1.0 + static_overlap_fraction), float(static_overlap_fraction)


def _collect_temporal_dense_snapshot_cache(
    *,
    analysis_cfg: DictConfig,
    temporal_selection: Any,
    temporal_inference_spec: Any,
    checkpoint_path: str,
    out_dir: Path,
    model_cfg_for_module: DictConfig,
    model: Any | None,
    cuda_device: int,
    seed_base: int,
    figure_only: bool,
    inference_batch_size: int,
    dataloader_num_workers: int,
    progress_every_batches: int,
    frame_indices: list[int],
    source_names: list[str],
    cache_enabled: bool,
    cache_force_recompute: bool,
    cache_file: str,
    collector_mode: str,
    summary_label: str,
    cache_config_path: str,
) -> tuple[dict[str, np.ndarray], Any, Any, dict[str, Any], Any]:
    if cache_file == "":
        raise ValueError(
            f"{cache_config_path}.file must be a non-empty file name."
        )

    regular_grid_overlap, static_overlap_fraction = (
        _resolve_temporal_snapshot_visualization_regular_grid_overlap(analysis_cfg)
    )
    resolved_frame_indices = [int(v) for v in list(frame_indices)]
    resolved_source_names = [str(v) for v in list(source_names)]
    if not resolved_frame_indices:
        raise ValueError(f"{summary_label} requires at least one selected frame.")
    if len(resolved_frame_indices) != len(resolved_source_names):
        raise RuntimeError(
            f"{summary_label} metadata is inconsistent: "
            f"frame_count={len(resolved_frame_indices)}, "
            f"source_name_count={len(resolved_source_names)}."
        )

    snapshot_bundles = [
        build_temporal_real_single_snapshot_bundle(
            analysis_cfg=analysis_cfg,
            selection=temporal_selection,
            batch_size=int(inference_batch_size),
            dataloader_num_workers=int(dataloader_num_workers),
            frame_index=int(frame_index),
            source_name=str(source_name),
            center_grid_overlap=float(regular_grid_overlap),
        )
        for frame_index, source_name in zip(
            resolved_frame_indices,
            resolved_source_names,
            strict=True,
        )
    ]
    snapshot_dataset = torch.utils.data.ConcatDataset(
        [bundle.dataset for bundle in snapshot_bundles]
    )
    snapshot_sample_source_names: list[str] = []
    snapshot_sample_counts: dict[str, int] = {}
    for bundle in snapshot_bundles:
        source_name = str(bundle.selection.analysis_source_names[0])
        sample_count = int(len(bundle.dataset))
        snapshot_sample_source_names.extend([source_name] * sample_count)
        snapshot_sample_counts[source_name] = sample_count
    setattr(snapshot_dataset, "sample_source_names", snapshot_sample_source_names)

    snapshot_selection_spec = {
        **temporal_selection.to_cache_spec(),
        "analysis_snapshot_count": int(len(resolved_frame_indices)),
        "inference_snapshot_fraction": 1.0,
        "inference_frame_indices": [int(v) for v in resolved_frame_indices],
        "analysis_frame_indices": [int(v) for v in resolved_frame_indices],
        "inference_source_names": [str(v) for v in resolved_source_names],
        "analysis_source_names": [str(v) for v in resolved_source_names],
        "center_selection": {
            "mode": "regular_grid",
            "overlap": float(regular_grid_overlap),
            "semantics": "sphere_overlap_depth_in_radius_units",
            "reference_frame": "per_snapshot_anchor",
            "static_overlap_fraction_equivalent": float(static_overlap_fraction),
        },
    }
    snapshot_cache_spec = _build_inference_cache_spec(
        checkpoint_path=checkpoint_path,
        cfg=model_cfg_for_module,
        inference_batch_size=int(inference_batch_size),
        max_batches_latent=None,
        max_samples_total=None,
        seed_base=int(seed_base),
        temporal_real_selection=snapshot_selection_spec,
        temporal_sequence_inference=temporal_inference_spec.to_cache_spec(),
        collector_mode=str(collector_mode),
    )

    snapshot_cache: dict[str, np.ndarray] | None = None
    snapshot_cache_loaded = False
    cache_log_prefix = f"[analysis][{collector_mode} cache]"
    if cache_enabled and not cache_force_recompute:
        snapshot_cache, snapshot_cache_msg = _load_inference_cache(
            out_dir=out_dir,
            cache_filename=cache_file,
            expected_spec=snapshot_cache_spec,
        )
        snapshot_cache_loaded = snapshot_cache is not None
        print(f"{cache_log_prefix} {snapshot_cache_msg}")
    elif cache_enabled and cache_force_recompute:
        print(f"{cache_log_prefix} Forced recompute requested; skipping cache load.")

    if snapshot_cache is None:
        if figure_only:
            raise RuntimeError(
                "figure_set.figure_only requires a valid cache for "
                f"{summary_label.lower()}. Missing cache: {out_dir / cache_file}. "
                "Run the full analysis once with figure_set.figure_only=false to populate it."
            )
        if model is None:
            model, _, _ = load_vicreg_model(
                checkpoint_path,
                cuda_device=int(cuda_device),
                cfg=model_cfg_for_module,
            )
        snapshot_cache_parts: list[dict[str, np.ndarray]] = []
        for snapshot_idx, bundle in enumerate(snapshot_bundles):
            source_name = str(bundle.selection.analysis_source_names[0])
            print(
                f"{summary_label} inference: "
                f"snapshot={source_name}, samples={len(bundle.dataset)}, "
                f"regular_grid_overlap={regular_grid_overlap:.3f} "
                f"(static_overlap_fraction={static_overlap_fraction:.3f})."
            )
            part_cache = gather_inference_batches(
                model,
                bundle.dataloader,
                f"cuda:{int(cuda_device)}" if torch.cuda.is_available() else "cpu",
                max_batches=None,
                max_samples_total=None,
                collect_coords=True,
                seed_base=int(seed_base) + int(snapshot_idx) * 1_000_000,
                progress_every_batches=int(progress_every_batches),
                verbose=True,
                temporal_sequence_mode=str(temporal_inference_spec.mode),
                temporal_static_frame_index=temporal_inference_spec.static_frame_index,
            )
            _validate_inference_cache_arrays(part_cache)
            part_sample_count = int(len(part_cache["inv_latents"]))
            expected_sample_count = int(len(bundle.dataset))
            if part_sample_count != expected_sample_count:
                raise RuntimeError(
                    f"{summary_label} inference collected an unexpected number of samples. "
                    f"snapshot={source_name}, collected={part_sample_count}, "
                    f"expected={expected_sample_count}."
                )
            snapshot_cache_parts.append(part_cache)
        snapshot_cache = _concatenate_inference_caches(snapshot_cache_parts)
        if cache_enabled:
            _save_inference_cache(
                out_dir=out_dir,
                cache_filename=cache_file,
                cache=snapshot_cache,
                spec=snapshot_cache_spec,
            )
            snapshot_cache_npz, _ = _inference_cache_paths(out_dir, cache_file)
            print(f"{cache_log_prefix} Saved inference cache: {snapshot_cache_npz}")

    _validate_inference_cache_arrays(snapshot_cache)
    expected_total_samples = int(sum(int(len(bundle.dataset)) for bundle in snapshot_bundles))
    actual_total_samples = int(len(snapshot_cache["inv_latents"]))
    if actual_total_samples != expected_total_samples:
        raise RuntimeError(
            f"{summary_label} cache sample count does not match the resolved dense snapshot datasets. "
            f"actual={actual_total_samples}, expected={expected_total_samples}, "
            f"cache_file={out_dir / cache_file}."
        )

    snapshot_layout = resolve_snapshot_figure_layout(
        snapshot_dataset,
        is_synthetic=False,
        n_samples=actual_total_samples,
        analysis_source_names=resolved_source_names,
    )
    summary = {
        "enabled": True,
        "cache_enabled": bool(cache_enabled),
        "cache_file": str(out_dir / cache_file),
        "cache_loaded_from_disk": bool(snapshot_cache_loaded),
        "cache_force_recompute": bool(cache_force_recompute),
        "sample_count": int(actual_total_samples),
        "regular_grid_overlap": float(regular_grid_overlap),
        "static_overlap_fraction_equivalent": float(static_overlap_fraction),
        "frame_indices": [int(v) for v in resolved_frame_indices],
        "source_names": [str(v) for v in resolved_source_names],
        "sample_count_by_snapshot": {
            str(source_name): int(count)
            for source_name, count in snapshot_sample_counts.items()
        },
    }
    return snapshot_cache, snapshot_dataset, snapshot_layout, summary, model


def _collect_temporal_snapshot_visualization_cache(
    *,
    analysis_cfg: DictConfig,
    temporal_selection: Any,
    temporal_inference_spec: Any,
    checkpoint_path: str,
    out_dir: Path,
    model_cfg_for_module: DictConfig,
    model: Any | None,
    cuda_device: int,
    seed_base: int,
    figure_only: bool,
    inference_batch_size: int,
    dataloader_num_workers: int,
    progress_every_batches: int,
) -> tuple[dict[str, np.ndarray], Any, Any, dict[str, Any], Any]:
    snapshot_cfg = OmegaConf.select(
        analysis_cfg,
        "inputs.temporal_real.snapshot_visualization",
        default=None,
    )
    cache_enabled = bool(OmegaConf.select(snapshot_cfg, "cache.enabled", default=True))
    cache_force_recompute = bool(
        OmegaConf.select(snapshot_cfg, "cache.force_recompute", default=False)
    )
    cache_file = str(
        OmegaConf.select(
            snapshot_cfg,
            "cache.file",
            default="temporal_snapshot_dense_inference_cache.npz",
        )
    ).strip()
    analysis_frame_indices = [
        int(v) for v in np.asarray(temporal_selection.analysis_frame_indices, dtype=np.int64)
    ]
    analysis_source_names = [str(v) for v in temporal_selection.analysis_source_names]
    return _collect_temporal_dense_snapshot_cache(
        analysis_cfg=analysis_cfg,
        temporal_selection=temporal_selection,
        temporal_inference_spec=temporal_inference_spec,
        checkpoint_path=checkpoint_path,
        out_dir=out_dir,
        model_cfg_for_module=model_cfg_for_module,
        model=model,
        cuda_device=int(cuda_device),
        seed_base=int(seed_base),
        figure_only=bool(figure_only),
        inference_batch_size=int(inference_batch_size),
        dataloader_num_workers=int(dataloader_num_workers),
        progress_every_batches=int(progress_every_batches),
        frame_indices=analysis_frame_indices,
        source_names=analysis_source_names,
        cache_enabled=bool(cache_enabled),
        cache_force_recompute=bool(cache_force_recompute),
        cache_file=cache_file,
        collector_mode="temporal_snapshot_visualization",
        summary_label="Temporal snapshot visualization",
        cache_config_path="inputs.temporal_real.snapshot_visualization.cache",
    )


def _collect_temporal_md_space_animation_cache(
    *,
    analysis_cfg: DictConfig,
    temporal_selection: Any,
    temporal_inference_spec: Any,
    checkpoint_path: str,
    out_dir: Path,
    model_cfg_for_module: DictConfig,
    model: Any | None,
    cuda_device: int,
    seed_base: int,
    figure_only: bool,
    inference_batch_size: int,
    dataloader_num_workers: int,
    progress_every_batches: int,
) -> tuple[dict[str, np.ndarray], Any, Any, dict[str, Any], Any] | None:
    dense_snapshot_count_raw = OmegaConf.select(
        analysis_cfg,
        "real_md.temporal.md_space.dense_snapshot_count",
        default=None,
    )
    if dense_snapshot_count_raw in {None, ""}:
        return None
    dense_snapshot_count = int(dense_snapshot_count_raw)
    if dense_snapshot_count <= 0:
        return None

    frame_indices, source_names = resolve_temporal_real_snapshot_subset(
        analysis_cfg=analysis_cfg,
        selection=temporal_selection,
        snapshot_count=int(dense_snapshot_count),
    )
    cache_enabled = bool(
        OmegaConf.select(
            analysis_cfg,
            "real_md.temporal.md_space.dense_snapshot_cache.enabled",
            default=True,
        )
    )
    cache_force_recompute = bool(
        OmegaConf.select(
            analysis_cfg,
            "real_md.temporal.md_space.dense_snapshot_cache.force_recompute",
            default=False,
        )
    )
    cache_file = str(
        OmegaConf.select(
            analysis_cfg,
            "real_md.temporal.md_space.dense_snapshot_cache.file",
            default="temporal_md_space_dense_inference_cache.npz",
        )
    ).strip()
    return _collect_temporal_dense_snapshot_cache(
        analysis_cfg=analysis_cfg,
        temporal_selection=temporal_selection,
        temporal_inference_spec=temporal_inference_spec,
        checkpoint_path=checkpoint_path,
        out_dir=out_dir,
        model_cfg_for_module=model_cfg_for_module,
        model=model,
        cuda_device=int(cuda_device),
        seed_base=int(seed_base),
        figure_only=bool(figure_only),
        inference_batch_size=int(inference_batch_size),
        dataloader_num_workers=int(dataloader_num_workers),
        progress_every_batches=int(progress_every_batches),
        frame_indices=[int(v) for v in frame_indices.tolist()],
        source_names=[str(v) for v in source_names],
        cache_enabled=bool(cache_enabled),
        cache_force_recompute=bool(cache_force_recompute),
        cache_file=cache_file,
        collector_mode="temporal_md_space_animation",
        summary_label="Temporal MD-space animation",
        cache_config_path="real_md.temporal.md_space.dense_snapshot_cache",
    )


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

def load_vicreg_model(
    checkpoint_path: str, cuda_device: int = 0, cfg: DictConfig | None = None
) -> Tuple[Any, DictConfig, str]:
    """Restore the contrastive module together with its Hydra cfg and device string."""
    if cfg is None:
        cfg = load_checkpoint_training_config(checkpoint_path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(
            "load_vicreg_model expects cfg to be a DictConfig when provided, "
            f"got {type(cfg)!r}."
        )
    device = f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu"
    model = load_model_from_checkpoint(
        checkpoint_path,
        cfg,
        device=device,
        module=_resolve_analysis_module_class(cfg),
    )
    model.to(device).eval()
    return model, cfg, device


def _resolve_analysis_module_class(cfg: DictConfig) -> type:
    model_type = str(getattr(cfg, "model_type", "vicreg")).strip().lower()
    if model_type == "vicreg":
        return VICRegModule
    if model_type in {"temporal_vicreg", "temporal_lejepa"}:
        from src.training_methods.temporal_ssl.temporal_ssl_module import TemporalSSLModule

        return TemporalSSLModule
    if model_type == "temporal_motif_field":
        from src.training_methods.temporal_motif_field.temporal_motif_field_module import (
            TemporalMotifFieldModule,
        )

        return TemporalMotifFieldModule
    raise ValueError(
        "Unsupported checkpoint model_type for analysis. "
        f"Expected one of ['vicreg', 'temporal_vicreg', 'temporal_lejepa', "
        f"'temporal_motif_field'], got {model_type!r}."
    )


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
    elif getattr(cfg.data, "kind", None) == "temporal_lammps":
        dm = TemporalLAMMPSDataModule(cfg)
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
    model: Any | None = None
    device = f"cuda:{run_settings.cuda_device}" if torch.cuda.is_available() else "cpu"
    analysis_source_names: list[str] | None = None
    temporal_bundle = None
    temporal_inference_spec = None
    temporal_real_mode = temporal_real_analysis_enabled(analysis_cfg)
    analysis_files = None if temporal_real_mode else _resolve_analysis_files(cfg, input_settings)
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
    if analysis_settings.cluster_fit is not None and hdbscan_settings.enabled:
        raise ValueError(
            "clustering.fit_inputs is not compatible with clustering.hdbscan.enabled yet. "
            "Disable HDBSCAN or disable clustering.fit_inputs."
        )
    if (
        temporal_real_analysis_enabled(analysis_cfg)
        and bool(
            OmegaConf.select(
                analysis_cfg,
                "inputs.temporal_real.snapshot_visualization.enabled",
                default=True,
            )
        )
        and hdbscan_settings.enabled
    ):
        raise ValueError(
            "inputs.temporal_real.snapshot_visualization is not compatible with "
            "clustering.hdbscan.enabled yet. Disable HDBSCAN or disable dense "
            "temporal snapshot visualization."
        )
    if (
        temporal_real_analysis_enabled(analysis_cfg)
        and OmegaConf.select(
            analysis_cfg,
            "real_md.temporal.md_space.dense_snapshot_count",
            default=None,
        )
        not in {None, "", 0}
        and hdbscan_settings.enabled
    ):
        raise ValueError(
            "real_md.temporal.md_space.dense_snapshot_count is not compatible with "
            "clustering.hdbscan.enabled yet. Disable HDBSCAN or disable dense "
            "MD-space animation sampling."
        )
    real_md_selected_k_override = _resolve_optional_cluster_k(
        OmegaConf.select(analysis_cfg, "real_md.selected_k", default=None),
        field_name="real_md.selected_k",
    )
    if (
        real_md_selected_k_override is not None
        and int(real_md_selected_k_override) != int(analysis_settings.primary_k)
    ):
        raise ValueError(
            "real_md.selected_k conflicts with clustering.primary_k. "
            f"Got real_md.selected_k={int(real_md_selected_k_override)} and "
            f"clustering.primary_k={int(analysis_settings.primary_k)}. "
            "Use clustering.primary_k as the single selected clustering k and "
            "leave real_md.selected_k unset."
        )
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
    )
    is_synthetic = getattr(cfg.data, "kind", None) == "synthetic"
    analysis_inference_batch_size = _resolve_analysis_inference_batch_size(cfg, input_settings)
    print(
        "Analysis inference batch size: "
        f"{analysis_inference_batch_size} "
        f"(checkpoint batch_size={int(cfg.batch_size)})"
    )

    # ── Data loading ───────────────────────────────────────────────────
    all_metrics: Dict[str, Any] = {}
    dm = None
    if temporal_real_mode:
        _step("Building temporal-real analysis dataset")
        default_temporal_static_frame_index = 0
        if str(getattr(cfg.data, "kind", "")).strip().lower() == "temporal_lammps":
            default_temporal_static_frame_index = int(getattr(cfg.data, "sequence_length", 1)) // 2
        configured_static_frame_index = OmegaConf.select(
            analysis_cfg,
            "inputs.temporal_real.static_frame_index",
            default=None,
        )
        temporal_inference_spec = resolve_temporal_real_inference_spec(
            analysis_cfg,
            default_static_frame_index=default_temporal_static_frame_index,
        )
        if (
            temporal_inference_spec.mode == "static_anchor"
            and configured_static_frame_index is None
            and default_temporal_static_frame_index != 0
        ):
            print(
                "Temporal real-data analysis: inputs.temporal_real.static_frame_index was not set; "
                f"defaulting to the temporal center frame index {default_temporal_static_frame_index} "
                "to match temporal_lammps checkpoint training."
            )
        temporal_bundle = build_temporal_real_analysis_bundle(
            analysis_cfg=analysis_cfg,
            model_cfg=cfg,
            batch_size=int(analysis_inference_batch_size),
            dataloader_num_workers=int(input_settings.dataloader_num_workers),
        )
        dl = temporal_bundle.dataloader
        analysis_source_names = list(temporal_bundle.selection.analysis_source_names)
        all_metrics["temporal_real_inputs"] = {
            **temporal_bundle.selection.dump_summary,
            **temporal_bundle.selection.to_cache_spec(),
            "inference": temporal_inference_spec.to_cache_spec(),
        }
        class_names = None
        print(
            "Temporal real-data analysis enabled: "
            f"inference_snapshots={len(temporal_bundle.selection.inference_source_names)}, "
            f"analysis_snapshots={len(temporal_bundle.selection.analysis_source_names)}, "
            f"center_count={len(getattr(temporal_bundle.dataset, '_center_atom_indices', []))}, "
            f"inference_mode={temporal_inference_spec.mode}."
        )
    else:
        _step("Building datamodule")
        dm = build_datamodule(
            cfg,
            require_coords_for_real=not is_synthetic,
        )
        dm.setup(stage="fit")
        dl = _build_analysis_dataloader(
            cfg,
            dm,
            is_synthetic=is_synthetic,
            inference_batch_size=int(analysis_inference_batch_size),
            dataloader_num_workers=int(input_settings.dataloader_num_workers),
        )
        class_names = _extract_class_names(dm.train_dataset)
    print(f"Loaded class names: {class_names}")

    max_batches_latent = input_settings.max_batches_latent
    if temporal_real_mode:
        max_samples_total = None
        if input_settings.max_samples_total is not None:
            print(
                "[analysis] inputs.max_samples_total is ignored for temporal real-data analysis; "
                "the temporal inference snapshot selection already defines the full inference set."
            )
    else:
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
        inference_batch_size=int(analysis_inference_batch_size),
        max_batches_latent=max_batches_latent,
        max_samples_total=max_samples_total,
        seed_base=int(seed_base),
        temporal_real_selection=(
            None if temporal_bundle is None else temporal_bundle.selection.to_cache_spec()
        ),
        temporal_sequence_inference=(
            None if temporal_inference_spec is None else temporal_inference_spec.to_cache_spec()
        ),
        collector_mode=(
            "tmf_sequence"
            if str(getattr(cfg, "model_type", "")).strip().lower() == "temporal_motif_field"
            else "generic"
        ),
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
        model, cfg, device = load_vicreg_model(
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
        if str(getattr(cfg, "model_type", "")).strip().lower() == "temporal_motif_field":
            cache = collect_tmf_inference_cache(
                model,
                dl,
                device,
                max_batches=max_batches_latent,
                max_samples_total=max_samples_total,
                seed_base=seed_base,
                progress_every_batches=analysis_settings.progress_every_batches,
                verbose=True,
            )
        else:
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
                temporal_sequence_mode=(
                    "static_anchor" if temporal_inference_spec is None else temporal_inference_spec.mode
                ),
                temporal_static_frame_index=(
                    0 if temporal_inference_spec is None else temporal_inference_spec.static_frame_index
                ),
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
    fit_cache: dict[str, np.ndarray] | None = None
    fit_cache_loaded = False
    fit_cfg: DictConfig | None = None
    fit_source_names: list[str] | None = None
    if analysis_settings.cluster_fit is not None:
        _step("Loading clustering fit-reference data")
        (
            fit_cache,
            fit_cfg,
            fit_source_names,
            model,
            fit_cache_loaded,
        ) = _collect_clustering_fit_cache(
            analysis_cfg=analysis_cfg,
            fit_settings=analysis_settings.cluster_fit,
            checkpoint_path=run_settings.checkpoint_path,
            out_dir=out_dir,
            model_cfg_for_module=cfg,
            model=model,
            cuda_device=run_settings.cuda_device,
            seed_base=seed_base,
            figure_only=bool(figure_settings.figure_only),
            progress_every_batches=analysis_settings.progress_every_batches,
        )
        all_metrics["clustering_fit_inputs"] = {
            "enabled": True,
            "data_config": analysis_settings.cluster_fit.data_config_path,
            "data_kind": str(getattr(fit_cfg.data, "kind", "")),
            "real_data_files_requested": analysis_settings.cluster_fit.input_settings.real_data_files,
            "real_data_files_resolved": [
                str(v)
                for v in list(getattr(fit_cfg.data, "data_files", []) or [])
            ],
            "source_names": fit_source_names,
            "sample_count": int(len(fit_cache["inv_latents"])),
            "cache_enabled": bool(analysis_settings.cluster_fit.cache_enabled),
            "cache_file": str(out_dir / analysis_settings.cluster_fit.cache_file),
            "cache_loaded_from_disk": bool(fit_cache_loaded),
            "cache_force_recompute": bool(
                analysis_settings.cluster_fit.cache_force_recompute
            ),
        }
    fit_latents_for_clustering = (
        None
        if fit_cache is None
        else np.asarray(fit_cache["inv_latents"], dtype=np.float32)
    )
    temporal_snapshot_visualization_cache: dict[str, np.ndarray] | None = None
    temporal_snapshot_visualization_dataset = None
    temporal_snapshot_visualization_layout = None
    temporal_snapshot_visualization_enabled = bool(
        temporal_bundle is not None
        and OmegaConf.select(
            analysis_cfg,
            "inputs.temporal_real.snapshot_visualization.enabled",
            default=True,
        )
    )
    if temporal_snapshot_visualization_enabled:
        _step("Loading dense temporal snapshot visualization data")
        (
            temporal_snapshot_visualization_cache,
            temporal_snapshot_visualization_dataset,
            temporal_snapshot_visualization_layout,
            temporal_snapshot_visualization_summary,
            model,
        ) = _collect_temporal_snapshot_visualization_cache(
            analysis_cfg=analysis_cfg,
            temporal_selection=temporal_bundle.selection,
            temporal_inference_spec=temporal_inference_spec,
            checkpoint_path=run_settings.checkpoint_path,
            out_dir=out_dir,
            model_cfg_for_module=cfg,
            model=model,
            cuda_device=run_settings.cuda_device,
            seed_base=seed_base,
            figure_only=bool(figure_settings.figure_only),
            inference_batch_size=int(analysis_inference_batch_size),
            dataloader_num_workers=int(input_settings.dataloader_num_workers),
            progress_every_batches=analysis_settings.progress_every_batches,
        )
        all_metrics["temporal_snapshot_visualization"] = dict(
            temporal_snapshot_visualization_summary
        )
    temporal_md_space_animation_cache: dict[str, np.ndarray] | None = None
    temporal_md_space_animation_dataset = None
    temporal_md_space_animation_layout = None
    temporal_md_space_animation_source_names: list[str] | None = None
    temporal_md_space_animation_enabled = bool(
        temporal_bundle is not None
        and OmegaConf.select(analysis_cfg, "real_md.enabled", default=True)
        and OmegaConf.select(analysis_cfg, "real_md.temporal.enabled", default=True)
        and OmegaConf.select(
            analysis_cfg,
            "real_md.temporal.md_space.enabled",
            default=True,
        )
        and OmegaConf.select(
            analysis_cfg,
            "real_md.temporal.md_space.dense_snapshot_count",
            default=None,
        )
        not in {None, "", 0}
    )
    if temporal_md_space_animation_enabled:
        _step("Loading dense temporal MD-space animation data")
        md_animation_cache_result = _collect_temporal_md_space_animation_cache(
            analysis_cfg=analysis_cfg,
            temporal_selection=temporal_bundle.selection,
            temporal_inference_spec=temporal_inference_spec,
            checkpoint_path=run_settings.checkpoint_path,
            out_dir=out_dir,
            model_cfg_for_module=cfg,
            model=model,
            cuda_device=run_settings.cuda_device,
            seed_base=seed_base,
            figure_only=bool(figure_settings.figure_only),
            inference_batch_size=int(analysis_inference_batch_size),
            dataloader_num_workers=int(input_settings.dataloader_num_workers),
            progress_every_batches=analysis_settings.progress_every_batches,
        )
        if md_animation_cache_result is not None:
            (
                temporal_md_space_animation_cache,
                temporal_md_space_animation_dataset,
                temporal_md_space_animation_layout,
                temporal_md_space_animation_summary,
                model,
            ) = md_animation_cache_result
            temporal_md_space_animation_source_names = list(
                temporal_md_space_animation_summary["source_names"]
            )
            all_metrics["temporal_md_space_animation_sampling"] = dict(
                temporal_md_space_animation_summary
            )
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
    clustering_features_for_fixed_k = (
        None if fit_latents_for_clustering is not None else clustering_features
    )
    clustering_feature_prep_for_fixed_k = (
        None if fit_latents_for_clustering is not None else clustering_feature_prep
    )
    dataset_obj = getattr(dl, "dataset", None)
    snapshot_layout_inference = resolve_snapshot_figure_layout(
        dataset_obj,
        is_synthetic=is_synthetic,
        n_samples=n_samples,
        analysis_source_names=(
            None
            if temporal_bundle is None
            else temporal_bundle.selection.inference_source_names
        ),
    )
    snapshot_layout = (
        filter_snapshot_figure_layout(
            snapshot_layout_inference,
            allowed_source_names=temporal_bundle.selection.analysis_source_names,
        )
        if temporal_bundle is not None
        else snapshot_layout_inference
    )
    snapshot_source_groups = snapshot_layout.source_groups
    snapshot_output_names = snapshot_layout.output_names
    multi_snapshot_real = snapshot_layout.multi_snapshot_real
    snapshot_layout_for_outputs = (
        snapshot_layout
        if temporal_snapshot_visualization_layout is None
        else temporal_snapshot_visualization_layout
    )
    snapshot_dataset_obj_for_outputs = (
        dataset_obj
        if temporal_snapshot_visualization_dataset is None
        else temporal_snapshot_visualization_dataset
    )
    snapshot_latents_for_outputs = (
        np.asarray(cache["inv_latents"], dtype=np.float32)
        if temporal_snapshot_visualization_cache is None
        else np.asarray(
            temporal_snapshot_visualization_cache["inv_latents"],
            dtype=np.float32,
        )
    )
    snapshot_coords_for_outputs = (
        np.asarray(coords, dtype=np.float32)
        if temporal_snapshot_visualization_cache is None
        else np.asarray(
            temporal_snapshot_visualization_cache["coords"],
            dtype=np.float32,
        )
    )
    snapshot_analysis_source_names_for_outputs = (
        analysis_source_names
        if temporal_snapshot_visualization_cache is None
        else list(temporal_bundle.selection.analysis_source_names)
    )
    snapshot_cluster_labels_by_k_for_outputs: dict[int, np.ndarray] | None = None

    def _build_figure_set_run_kwargs(figure_settings_for_k: Any) -> dict[str, Any]:
        return figure_settings_for_k.build_run_kwargs(
            dataset=snapshot_dataset_obj_for_outputs,
            latents=snapshot_latents_for_outputs,
            coords=snapshot_coords_for_outputs,
            point_scale=point_scale,
            random_state=clustering_random_state,
            l2_normalize=analysis_settings.cluster_l2_normalize,
            standardize=analysis_settings.cluster_standardize,
            pca_variance=analysis_settings.cluster_pca_var,
            pca_max_components=analysis_settings.cluster_pca_max_components,
        )

    clustering_requested_k_values = list(analysis_settings.cluster_k_values)
    if temporal_snapshot_visualization_cache is not None:
        _step("Projecting cluster labels onto dense temporal snapshots")
        snapshot_fit_latents = (
            np.asarray(cache["inv_latents"], dtype=np.float32)
            if fit_latents_for_clustering is None
            else np.asarray(fit_latents_for_clustering, dtype=np.float32)
        )
        (
            snapshot_clustering_metrics,
            _,
            snapshot_cluster_labels_by_k_for_outputs,
            _,
        ) = _build_clustering_state(
            snapshot_latents_for_outputs,
            np.asarray(temporal_snapshot_visualization_cache["phases"], dtype=int),
            fit_latents=snapshot_fit_latents,
            requested_k_values=clustering_requested_k_values,
            cluster_method=analysis_settings.cluster_method,
            random_state=clustering_random_state,
            l2_normalize=analysis_settings.cluster_l2_normalize,
            standardize=analysis_settings.cluster_standardize,
            pca_variance=analysis_settings.cluster_pca_var,
            pca_max_components=analysis_settings.cluster_pca_max_components,
            prepared_features=None,
            prep_info=None,
        )
        all_metrics.setdefault("temporal_snapshot_visualization", {}).update(
            {
                "clustering_projection": snapshot_clustering_metrics,
            }
        )
    temporal_md_animation_layout_for_outputs = temporal_snapshot_visualization_layout
    temporal_md_animation_order_for_outputs = (
        None
        if temporal_snapshot_visualization_cache is None
        else list(snapshot_analysis_source_names_for_outputs)
    )
    temporal_md_animation_coords_for_outputs = (
        None
        if temporal_snapshot_visualization_cache is None
        else np.asarray(
            temporal_snapshot_visualization_cache["coords"],
            dtype=np.float32,
        )
    )
    temporal_md_animation_cluster_labels_by_k_for_outputs = (
        snapshot_cluster_labels_by_k_for_outputs
    )
    temporal_md_animation_frame_source = (
        None
        if temporal_snapshot_visualization_cache is None
        else "dense_selected_frames"
    )
    if temporal_md_space_animation_cache is not None:
        _step("Projecting cluster labels onto dense MD-space animation frames")
        md_animation_fit_latents = (
            np.asarray(cache["inv_latents"], dtype=np.float32)
            if fit_latents_for_clustering is None
            else np.asarray(fit_latents_for_clustering, dtype=np.float32)
        )
        (
            md_animation_clustering_metrics,
            _,
            temporal_md_animation_cluster_labels_by_k_for_outputs,
            _,
        ) = _build_clustering_state(
            np.asarray(temporal_md_space_animation_cache["inv_latents"], dtype=np.float32),
            np.asarray(temporal_md_space_animation_cache["phases"], dtype=int),
            fit_latents=md_animation_fit_latents,
            requested_k_values=clustering_requested_k_values,
            cluster_method=analysis_settings.cluster_method,
            random_state=clustering_random_state,
            l2_normalize=analysis_settings.cluster_l2_normalize,
            standardize=analysis_settings.cluster_standardize,
            pca_variance=analysis_settings.cluster_pca_var,
            pca_max_components=analysis_settings.cluster_pca_max_components,
            prepared_features=None,
            prep_info=None,
        )
        all_metrics.setdefault("temporal_md_space_animation_sampling", {}).update(
            {
                "clustering_projection": md_animation_clustering_metrics,
            }
        )
        temporal_md_animation_layout_for_outputs = temporal_md_space_animation_layout
        temporal_md_animation_order_for_outputs = (
            None
            if temporal_md_space_animation_source_names is None
            else list(temporal_md_space_animation_source_names)
        )
        temporal_md_animation_coords_for_outputs = np.asarray(
            temporal_md_space_animation_cache["coords"],
            dtype=np.float32,
        )
        temporal_md_animation_frame_source = "dense_md_space_frames"

    def _resolve_visible_cluster_sets_for_k(
        labels_for_k: np.ndarray,
        requested_visible_cluster_sets: list[list[int]] | None,
        *,
        context: str,
    ) -> list[list[int]] | None:
        if not requested_visible_cluster_sets:
            return None
        available_cluster_ids = {
            int(v)
            for v in np.unique(np.asarray(labels_for_k, dtype=int).reshape(-1))
            if int(v) >= 0
        }
        resolved_visible_sets: list[list[int]] = []
        for set_idx, cluster_set in enumerate(requested_visible_cluster_sets):
            normalized_cluster_ids = [int(v) for v in cluster_set]
            present_cluster_ids = [
                cluster_id
                for cluster_id in normalized_cluster_ids
                if cluster_id in available_cluster_ids
            ]
            missing_cluster_ids = [
                cluster_id
                for cluster_id in normalized_cluster_ids
                if cluster_id not in available_cluster_ids
            ]
            if missing_cluster_ids and present_cluster_ids:
                print(
                    f"[analysis] {context}: visible_cluster_sets[{set_idx}] drops missing "
                    f"cluster IDs {missing_cluster_ids}; using {present_cluster_ids}."
                )
            elif missing_cluster_ids:
                print(
                    f"[analysis] {context}: skipping visible_cluster_sets[{set_idx}]="
                    f"{normalized_cluster_ids} because none of those cluster IDs are "
                    "present for this k."
                )
            if present_cluster_ids:
                resolved_visible_sets.append(present_cluster_ids)
        return resolved_visible_sets or None

    # ── Figure-only early return ───────────────────────────────────────
    if figure_settings.figure_only:
        clustering_metrics, configured_k_values, cluster_labels_by_k, _ = _build_clustering_state(
            cache["inv_latents"],
            cache["phases"],
            fit_latents=fit_latents_for_clustering,
            requested_k_values=list(analysis_settings.cluster_k_values),
            cluster_method=analysis_settings.cluster_method,
            random_state=clustering_random_state,
            l2_normalize=analysis_settings.cluster_l2_normalize,
            standardize=analysis_settings.cluster_standardize,
            pca_variance=analysis_settings.cluster_pca_var,
            pca_max_components=analysis_settings.cluster_pca_max_components,
            prepared_features=clustering_features_for_fixed_k,
            prep_info=clustering_feature_prep_for_fixed_k,
        )
        all_metrics["clustering"] = clustering_metrics
        figure_output_labels_by_k = (
            cluster_labels_by_k
            if snapshot_cluster_labels_by_k_for_outputs is None
            else snapshot_cluster_labels_by_k_for_outputs
        )
        cluster_figure_sets_by_k: dict[str, Any] = {}
        primary_cluster_figure_set = None
        primary_snapshot_figure_sets = None
        for k_value in configured_k_values:
            figure_settings_for_k = replace(
                figure_settings,
                k=int(k_value),
                visible_cluster_sets=_resolve_visible_cluster_sets_for_k(
                    figure_output_labels_by_k[int(k_value)],
                    figure_settings.visible_cluster_sets,
                    context=f"figure_only k={int(k_value)}",
                ),
            )
            cluster_figure_set, snapshot_figure_sets = render_cluster_figure_outputs(
                out_dir=out_dir,
                dataloader=dl,
                figure_settings=figure_settings_for_k,
                figure_set_run_kwargs=_build_figure_set_run_kwargs(figure_settings_for_k),
                labels_for_k=figure_output_labels_by_k[int(k_value)],
                latents=snapshot_latents_for_outputs,
                coords=snapshot_coords_for_outputs,
                dataset_obj=snapshot_dataset_obj_for_outputs,
                snapshot_layout=snapshot_layout_for_outputs,
                analysis_source_names=snapshot_analysis_source_names_for_outputs,
                step=_step,
            )
            cluster_figure_sets_by_k[str(int(k_value))] = {
                "cluster_figure_set": cluster_figure_set,
                "cluster_figure_sets_by_snapshot": snapshot_figure_sets,
            }
            if int(k_value) == int(analysis_settings.primary_k):
                primary_cluster_figure_set = cluster_figure_set
                primary_snapshot_figure_sets = snapshot_figure_sets
        if primary_cluster_figure_set is not None:
            all_metrics["cluster_figure_set"] = primary_cluster_figure_set
        if primary_snapshot_figure_sets is not None:
            all_metrics["cluster_figure_sets_by_snapshot"] = primary_snapshot_figure_sets
        if len(configured_k_values) > 1:
            all_metrics["cluster_figure_sets_by_k"] = cluster_figure_sets_by_k

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
    clustering_metrics, configured_k_values, cluster_labels_by_k, cluster_methods_by_k = _build_clustering_state(
        cache["inv_latents"],
        cache["phases"],
        fit_latents=fit_latents_for_clustering,
        requested_k_values=clustering_requested_k_values,
        cluster_method=analysis_settings.cluster_method,
        random_state=clustering_random_state,
        l2_normalize=analysis_settings.cluster_l2_normalize,
        standardize=analysis_settings.cluster_standardize,
        pca_variance=analysis_settings.cluster_pca_var,
        pca_max_components=analysis_settings.cluster_pca_max_components,
        prepared_features=clustering_features_for_fixed_k,
        prep_info=clustering_feature_prep_for_fixed_k,
    )
    all_metrics["clustering"] = clustering_metrics
    clustering_comparison, comparison_labels_by_method = build_clustering_method_comparison(
        cache["inv_latents"],
        cache["phases"],
        fit_latents=fit_latents_for_clustering,
        requested_k_values=clustering_requested_k_values,
        primary_method=analysis_settings.cluster_method,
        compare_methods=analysis_settings.cluster_compare_methods,
        random_state=clustering_random_state,
        l2_normalize=analysis_settings.cluster_l2_normalize,
        standardize=analysis_settings.cluster_standardize,
        pca_variance=analysis_settings.cluster_pca_var,
        pca_max_components=analysis_settings.cluster_pca_max_components,
        prepared_features=clustering_features_for_fixed_k,
        prep_info=clustering_feature_prep_for_fixed_k,
    )
    if clustering_comparison is not None:
        all_metrics["clustering_comparison"] = clustering_comparison

    primary_k = int(analysis_settings.primary_k)
    if primary_k not in configured_k_values:
        raise KeyError(
            "Requested clustering.primary_k is not available in configured clustering results. "
            f"Requested k={primary_k}, available={configured_k_values}."
        )
    cluster_labels = cluster_labels_by_k[primary_k]
    figure_output_labels_by_k = (
        cluster_labels_by_k
        if snapshot_cluster_labels_by_k_for_outputs is None
        else snapshot_cluster_labels_by_k_for_outputs
    )
    figure_output_cluster_labels = figure_output_labels_by_k[int(primary_k)]

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
    cluster_figure_sets_by_k: dict[str, Any] = {}
    primary_cluster_figure_set = None
    primary_snapshot_figure_sets = None
    if figure_settings.enabled:
        for k_value in configured_k_values:
            figure_settings_for_k = replace(
                figure_settings,
                k=int(k_value),
                visible_cluster_sets=_resolve_visible_cluster_sets_for_k(
                    figure_output_labels_by_k[int(k_value)],
                    figure_settings.visible_cluster_sets,
                    context=f"k={int(k_value)}",
                ),
            )
            representative_render_cache_for_k = (
                shared_representative_render_cache
                if int(k_value) == int(primary_k)
                else None
            )
            cluster_figure_set, snapshot_figure_sets = render_cluster_figure_outputs(
                out_dir=out_dir,
                dataloader=dl,
                figure_settings=figure_settings_for_k,
                figure_set_run_kwargs=_build_figure_set_run_kwargs(figure_settings_for_k),
                labels_for_k=figure_output_labels_by_k[int(k_value)],
                latents=snapshot_latents_for_outputs,
                coords=snapshot_coords_for_outputs,
                dataset_obj=snapshot_dataset_obj_for_outputs,
                snapshot_layout=snapshot_layout_for_outputs,
                analysis_source_names=snapshot_analysis_source_names_for_outputs,
                step=_step,
                representative_render_cache=representative_render_cache_for_k,
            )
            cluster_figure_sets_by_k[str(int(k_value))] = {
                "cluster_figure_set": cluster_figure_set,
                "cluster_figure_sets_by_snapshot": snapshot_figure_sets,
            }
            if int(k_value) == int(primary_k):
                primary_cluster_figure_set = cluster_figure_set
                primary_snapshot_figure_sets = snapshot_figure_sets
    if primary_cluster_figure_set is not None:
        all_metrics["cluster_figure_set"] = primary_cluster_figure_set
    if primary_snapshot_figure_sets is not None:
        all_metrics["cluster_figure_sets_by_snapshot"] = primary_snapshot_figure_sets
    if len(cluster_figure_sets_by_k) > 1:
        all_metrics["cluster_figure_sets_by_k"] = cluster_figure_sets_by_k

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
    tsne_metrics = run_tsne_visualizations(
        cache,
        out_dir,
        analysis_cfg=analysis_cfg,
        cluster_labels_by_k=cluster_labels_by_k,
        cluster_methods_by_k=cluster_methods_by_k,
        comparison_labels_by_method=comparison_labels_by_method,
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
    if tsne_metrics:
        all_metrics["latent_projection_visualizations"] = tsne_metrics

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
        coords=snapshot_coords_for_outputs,
        cluster_labels=figure_output_cluster_labels,
        cluster_labels_by_k=figure_output_labels_by_k,
        configured_k_values=configured_k_values,
        primary_k=primary_k,
        shared_cluster_color_maps_by_k=shared_cluster_color_maps_by_k,
        interactive_max_points=interactive_max_points,
        multi_snapshot_real=snapshot_layout_for_outputs.multi_snapshot_real,
        snapshot_source_groups=snapshot_layout_for_outputs.source_groups,
        snapshot_output_names=snapshot_layout_for_outputs.output_names,
        hdbscan_result=hdbscan_result,
    )

    # ── Real-MD qualitative analysis ───────────────────────────────────
    primary_real_md_summary = None
    if not is_synthetic and real_md_enabled:
        _step("Running real-MD qualitative analysis")
        temporal_projection_fit_indices = (
            None
            if temporal_bundle is None
            else np.asarray(
                _sample_indices(
                    n_samples,
                    analysis_settings.tsne_max_samples,
                ),
                dtype=int,
            )
        )
        real_md_summaries_by_k: dict[str, Any] = {}
        primary_real_md_summary = None
        for k_value in configured_k_values:
            shared_real_md_color_map = shared_cluster_color_maps_by_k.get(
                int(k_value),
                shared_cluster_color_map,
            )
            real_md_output_root = (
                real_md_outputs_root(out_dir)
                if int(k_value) == int(primary_k)
                else real_md_outputs_root_for_k(out_dir, k_value=int(k_value))
            )
            representative_render_cache_for_k = (
                shared_representative_render_cache
                if int(k_value) == int(primary_k)
                else None
            )
            real_md_summary = run_real_md_qualitative_analysis(
                out_dir=out_dir,
                model_cfg=cfg,
                analysis_cfg=analysis_cfg,
                dataset=dataset_obj,
                latents=cache["inv_latents"],
                coords=coords,
                instance_ids=np.asarray(cache["instance_ids"]),
                cluster_labels_by_k=cluster_labels_by_k,
                cluster_methods_by_k=cluster_methods_by_k,
                cluster_color_map=shared_real_md_color_map,
                frame_groups=snapshot_source_groups,
                frame_output_names=snapshot_output_names,
                requested_frame_order=analysis_source_names,
                temporal_all_frame_groups=(
                    None if temporal_bundle is None else snapshot_layout_inference.source_groups
                ),
                temporal_all_frame_output_names=(
                    None if temporal_bundle is None else snapshot_layout_inference.output_names
                ),
                temporal_all_frame_order=(
                    None if temporal_bundle is None else temporal_bundle.selection.inference_source_names
                ),
                temporal_md_animation_frame_groups=(
                    None
                    if temporal_md_animation_layout_for_outputs is None
                    else temporal_md_animation_layout_for_outputs.source_groups
                ),
                temporal_md_animation_frame_output_names=(
                    None
                    if temporal_md_animation_layout_for_outputs is None
                    else temporal_md_animation_layout_for_outputs.output_names
                ),
                temporal_md_animation_order=temporal_md_animation_order_for_outputs,
                temporal_md_animation_coords=temporal_md_animation_coords_for_outputs,
                temporal_md_animation_cluster_labels_by_k=(
                    temporal_md_animation_cluster_labels_by_k_for_outputs
                ),
                temporal_md_animation_frame_source=temporal_md_animation_frame_source,
                temporal_projection_fit_indices=temporal_projection_fit_indices,
                point_scale=float(point_scale),
                random_state=int(clustering_random_state),
                representative_render_cache=representative_render_cache_for_k,
                selected_k_override=int(k_value),
                output_root_dir=real_md_output_root,
            )
            real_md_summaries_by_k[str(int(k_value))] = real_md_summary
            if int(k_value) == int(primary_k):
                primary_real_md_summary = real_md_summary
        if primary_real_md_summary is not None:
            all_metrics["real_md_qualitative"] = primary_real_md_summary
        if len(real_md_summaries_by_k) > 1:
            all_metrics["real_md_qualitative_by_k"] = real_md_summaries_by_k

    # ── Dynamic motif analysis ─────────────────────────────────────────
    dynamic_metrics = run_dynamic_motif_analysis(
        cache=cache,
        out_dir=out_dir,
        model_cfg=cfg,
        analysis_cfg=analysis_settings,
        cluster_labels_primary=cluster_labels,
        step=_step,
    )
    if dynamic_metrics:
        all_metrics["dynamic_motif"] = dynamic_metrics
        dynamic_summary_rel = dynamic_metrics.get("artifacts", {}).get("summary_markdown")
        if (
            primary_real_md_summary is not None
            and isinstance(primary_real_md_summary, dict)
            and dynamic_summary_rel is not None
            and "summary_markdown" in primary_real_md_summary
        ):
            append_dynamic_motif_summary(
                Path(primary_real_md_summary["summary_markdown"]),
                dynamic_summary_path=out_dir / dynamic_summary_rel,
                dynamic_metrics=dynamic_metrics,
                out_dir=out_dir,
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
            temporal_sequence_mode=(
                "static_anchor" if temporal_inference_spec is None else temporal_inference_spec.mode
            ),
            temporal_static_frame_index=(
                0 if temporal_inference_spec is None else temporal_inference_spec.static_frame_index
            ),
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
    main()
