import argparse
from dataclasses import asdict, replace
import sys
import time
from pathlib import Path
from typing import Any, Dict

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

from src.data_utils.data_kinds import normalize_data_kind

from .cluster_profiles import resolve_point_scale
from .cluster_rendering import _build_cluster_representative_render_cache
from .clustering import (
    _run_optional_hdbscan_analysis,
    build_clustering_method_comparison,
    compute_clustering_assignment_margins,
    fit_reusable_clustering_models,
    predict_clustering_state_from_models,
    representative_features_from_clustering_model,
)
from .config import (
    DEFAULT_ANALYSIS_CONFIG_PATH, _positive_int_or_none, _print_resolved_analysis_settings,
    _resolve_analysis_files, _resolve_analysis_settings, _resolve_figure_set_settings,
    _resolve_input_settings, _resolve_optional_cluster_k, _resolve_run_settings,
    build_runtime_model_config, load_checkpoint_analysis_config,
)
from .connected_regimes import (
    resolve_connected_regime_settings,
    run_connected_regime_analysis,
)
from .dynamic_motif import run_dynamic_motif_analysis
from .directional_line_jepa import (
    apply_directional_runtime_limits,
    disable_directional_for_non_line_jepa,
    resolve_directional_line_jepa_settings,
    run_directional_line_jepa_analysis,
)
from .figure_sets import (
    build_shared_cluster_color_map, filter_snapshot_figure_layout, print_figure_set_summary,
    render_cluster_figure_outputs, resolve_snapshot_figure_layout, write_figure_only_metrics,
)
from .cluster_gallery import _save_horizontal_image_gallery
from .inference_cache import (
    _build_inference_cache_spec, _inference_cache_spec_hash, _load_inference_cache,
)
from .lazy_static_dataset import build_lazy_static_analysis_dataloader
from .latent_vis import print_analysis_summary, run_equivariance_evaluation, run_pca_and_latent_stats, run_tsne_visualizations
from .md_outputs import build_md_metrics
from .output_layout import real_md_outputs_root, real_md_outputs_root_for_k, write_json
from .pipeline_runtime import (
    _build_analysis_dataloader,
    _collect_clustering_fit_cache,
    _collect_main_inference_cache,
    _configure_static_analysis_inputs,
    _extract_class_names,
    _resolve_analysis_inference_batch_size,
    _resolve_analysis_max_samples_total,
    build_datamodule,
    load_vicreg_model,
)
from .real_md_qualitative import append_dynamic_motif_summary, run_real_md_qualitative_analysis
from .runtime_profile import (
    resolve_analysis_runtime_profile,
    select_evenly_spaced_names,
    subsample_clustering_reference,
)
from .swav_eval import run_swav_prototype_evaluation
from .temporal_dense import (
    _collect_temporal_dense_outputs,
)
from .temporal_real import (
    build_temporal_real_analysis_bundle,
    resolve_temporal_real_inference_spec,
    resolve_temporal_real_snapshot_subset,
    temporal_real_analysis_enabled,
)
from .utils import _sample_indices


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def _validate_temporal_cache_anchor_order(
    cache: dict[str, np.ndarray],
    dataset: Any,
) -> None:
    if not hasattr(dataset, "sample_anchor_frame_indices"):
        raise TypeError(
            "Temporal cache validation requires dataset.sample_anchor_frame_indices, "
            f"got dataset type {type(dataset)!r}."
        )
    cached = np.asarray(cache.get("anchor_frame_indices", np.empty((0,), dtype=np.int64)))
    n_samples = int(np.asarray(cache["inv_latents"]).shape[0])
    if cached.size == 0:
        raise RuntimeError(
            "Temporal inference cache is missing per-sample anchor_frame_indices. "
            "This cache cannot be trusted for time-series evaluation; rerun full "
            "analysis with figure_set.figure_only=false so the cache is regenerated."
        )
    if cached.shape[0] != n_samples:
        raise ValueError(
            "Temporal inference cache anchor_frame_indices length mismatch: "
            f"anchor_frame_indices={cached.shape[0]}, inv_latents={n_samples}."
        )
    expected_all = np.asarray(dataset.sample_anchor_frame_indices, dtype=np.int64)
    if expected_all.shape[0] < n_samples:
        raise ValueError(
            "Temporal dataset has fewer anchor-frame entries than cached samples: "
            f"dataset={expected_all.shape[0]}, cache={n_samples}."
        )
    expected = expected_all[:n_samples]
    mismatch = np.flatnonzero(cached.astype(np.int64, copy=False) != expected)
    if mismatch.size > 0:
        first = int(mismatch[0])
        raise RuntimeError(
            "Temporal inference cache row order does not match dataset anchor-frame order. "
            "Using this cache would mix frames and produce flat/random cluster time series. "
            f"first_mismatch_sample={first}, cached_anchor={int(cached[first])}, "
            f"expected_anchor={int(expected[first])}, mismatch_count={int(mismatch.size)}, "
            f"num_samples={n_samples}. Delete or recompute the inference cache."
        )


def _temporal_stratified_fit_indices(
    cache: dict[str, np.ndarray],
    *,
    max_samples: int,
    random_state: int,
) -> np.ndarray:
    n_samples = int(np.asarray(cache["inv_latents"]).shape[0])
    budget = min(int(max_samples), n_samples)
    if budget <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}.")
    if budget >= n_samples:
        return np.arange(n_samples, dtype=np.int64)

    anchor_frames = np.asarray(cache.get("anchor_frame_indices", np.empty((0,), dtype=np.int64)))
    if anchor_frames.shape[0] != n_samples:
        raise RuntimeError(
            "Temporal-stratified clustering fit requires cache['anchor_frame_indices'] "
            f"with one value per sample, got shape={tuple(anchor_frames.shape)}, "
            f"n_samples={n_samples}."
        )

    rng = np.random.default_rng(int(random_state))
    frame_values = np.unique(anchor_frames.astype(np.int64, copy=False))
    if frame_values.size == 0:
        raise RuntimeError("Cannot sample temporal clustering fit indices from zero frames.")

    base_quota = budget // int(frame_values.size)
    remainder = budget % int(frame_values.size)
    sampled_parts: list[np.ndarray] = []
    for frame_pos, frame_value in enumerate(frame_values.tolist()):
        frame_indices = np.flatnonzero(anchor_frames == int(frame_value)).astype(np.int64)
        quota = int(base_quota + (1 if frame_pos < remainder else 0))
        if quota <= 0:
            continue
        if quota >= int(frame_indices.size):
            sampled_parts.append(frame_indices)
        else:
            sampled_parts.append(
                rng.choice(frame_indices, size=int(quota), replace=False).astype(np.int64)
            )

    if not sampled_parts:
        raise RuntimeError(
            "Temporal-stratified clustering sampler selected zero samples. "
            f"budget={budget}, frame_count={int(frame_values.size)}."
        )
    return np.sort(np.concatenate(sampled_parts).astype(np.int64, copy=False))


def run_post_training_analysis(
    checkpoint_path: str | None = None,
    output_dir: str | None = None,
    cuda_device: int | None = None,
    *,
    analysis_config_path: str | None = None,
    analysis_cfg: DictConfig | None = None,
) -> Dict[str, Any]:
    """Generate qualitative and quantitative diagnostics for contrastive checkpoints."""
    torch.set_float32_matmul_precision("high")
    t0 = time.perf_counter()
    step_idx = [0]
    previous_step_time = [t0]

    def _step(msg: str) -> None:
        step_idx[0] += 1
        now = time.perf_counter()
        elapsed = now - t0
        previous_duration = now - previous_step_time[0]
        previous_step_time[0] = now
        print(
            f"[analysis][step {step_idx[0]}][total={elapsed:7.1f}s]"
            f"[previous={previous_duration:6.1f}s] {msg}"
        )

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
        analysis_source_names = _configure_static_analysis_inputs(cfg, analysis_files)
        print(f"Analysis data_files: {analysis_files}")
        if analysis_source_names and len(analysis_source_names) > 1:
            print(f"Per-snapshot analysis sources: {analysis_source_names}")

    with open_dict(cfg):
        cfg.num_workers = int(input_settings.dataloader_num_workers)
    print(f"Analysis dataloader workers: {input_settings.dataloader_num_workers}")
    analysis_settings = _resolve_analysis_settings(analysis_cfg, cfg)
    hdbscan_settings = analysis_settings.hdbscan
    if temporal_real_mode and analysis_settings.cluster_fit is not None:
        raise ValueError(
            "Temporal dump analysis does not use clustering.fit_inputs. "
            "Clustering is fit from inputs.temporal_real.dump_file through the "
            "main temporal inference cache. Delete clustering.fit_inputs and control the "
            "temporal fit subset with clustering.temporal_fit_max_samples."
        )
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
    runtime_profile = resolve_analysis_runtime_profile(analysis_cfg)
    if runtime_profile.real_md_projection_method is not None:
        with open_dict(analysis_cfg):
            OmegaConf.update(
                analysis_cfg,
                "real_md.projection.method",
                runtime_profile.real_md_projection_method,
                merge=False,
                force_add=True,
            )
    effective_tsne_max_samples = int(analysis_settings.tsne_max_samples)
    if runtime_profile.tsne_max_samples is not None:
        effective_tsne_max_samples = min(
            effective_tsne_max_samples,
            int(runtime_profile.tsne_max_samples),
        )
    figure_settings = _resolve_figure_set_settings(
        analysis_cfg,
        cfg,
        out_dir=out_dir,
        primary_k=int(analysis_settings.primary_k),
    )
    figure_settings = replace(
        figure_settings,
        raytrace_enabled=figure_settings.raytrace_enabled and runtime_profile.raytrace_enabled,
        md_num_views=(
            figure_settings.md_num_views
            if runtime_profile.md_num_views is None
            else min(figure_settings.md_num_views, runtime_profile.md_num_views)
        ),
    )
    directional_line_jepa_settings = resolve_directional_line_jepa_settings(analysis_cfg)
    directional_line_jepa_settings, directional_skip_reason = (
        disable_directional_for_non_line_jepa(
            directional_line_jepa_settings,
            model_type=getattr(cfg, "model_type", None),
        )
    )
    if directional_skip_reason is not None:
        print(f"[analysis][directional-line-jepa] Skipping: {directional_skip_reason}.")
    connected_regime_settings = resolve_connected_regime_settings(
        analysis_cfg,
        default_random_state=int(analysis_settings.seed_base),
    )
    directional_eligible = (
        str(getattr(cfg, "model_type", "")).strip().lower() == "line_jepa"
        and normalize_data_kind(getattr(cfg.data, "kind", None)) in {"static", "line_static"}
    )
    if directional_eligible:
        directional_line_jepa_settings = apply_directional_runtime_limits(
            directional_line_jepa_settings,
            enabled=(
                None
                if figure_settings.figure_only
                else runtime_profile.directional_line_jepa_enabled
            ),
            max_directions=runtime_profile.directional_max_directions,
            max_atoms=runtime_profile.directional_max_atoms_total,
        )
    if directional_line_jepa_settings.enabled and figure_settings.figure_only:
        raise ValueError(
            "directional_line_jepa.enabled=true is incompatible with "
            "figure_set.figure_only=true because directional profiles require model inference. "
            "Run the full analysis with figure_set.figure_only=false."
        )
    if connected_regime_settings.enabled and figure_settings.figure_only:
        raise ValueError(
            "clustering.connected_regimes.enabled=true is incompatible with "
            "figure_set.figure_only=true. Run the full analysis so connected-regime "
            "metrics use the original inference latents and primary clustering labels."
        )
    _print_resolved_analysis_settings(
        analysis_settings=analysis_settings,
        figure_settings=figure_settings,
    )
    is_synthetic = normalize_data_kind(getattr(cfg.data, "kind", None)) == "synthetic"
    analysis_inference_batch_size = _resolve_analysis_inference_batch_size(cfg, input_settings)
    print(
        "Analysis inference batch size: "
        f"{analysis_inference_batch_size} "
        f"(checkpoint batch_size={int(cfg.batch_size)})"
    )
    print(f"Analysis runtime profile: {runtime_profile}")

    max_batches_latent = input_settings.max_batches_latent
    max_samples_total = (
        None
        if temporal_real_mode
        else _resolve_analysis_max_samples_total(
            input_settings,
            is_synthetic=is_synthetic,
            md_use_all_points=analysis_settings.md_use_all_points,
        )
    )
    seed_base = int(analysis_settings.seed_base)
    clustering_random_state = int(analysis_settings.seed_base)
    preloaded_cache: dict[str, np.ndarray] | None = None
    preloaded_cache_message: str | None = None
    normalized_data_kind = normalize_data_kind(getattr(cfg.data, "kind", None))
    if (
        not temporal_real_mode
        and normalized_data_kind in {"static", "line_static"}
        and analysis_settings.inference_cache_enabled
        and not analysis_settings.inference_cache_force_recompute
    ):
        static_cache_spec = _build_inference_cache_spec(
            checkpoint_path=run_settings.checkpoint_path,
            cfg=cfg,
            inference_batch_size=int(analysis_inference_batch_size),
            max_batches_latent=max_batches_latent,
            max_samples_total=max_samples_total,
            seed_base=seed_base,
            collector_mode="generic",
        )
        preloaded_cache, preloaded_cache_message = _load_inference_cache(
            out_dir=out_dir,
            cache_filename=analysis_settings.inference_cache_file,
            expected_spec=static_cache_spec,
        )
        print(f"[analysis][cache preflight] {preloaded_cache_message}")
        if figure_settings.figure_only and preloaded_cache is None:
            raise RuntimeError(
                "figure_set.figure_only requires a valid static inference cache. "
                "Cache preflight failed before dataset construction: "
                f"{preloaded_cache_message}. Run once with figure_set.figure_only=false."
            )

    # ── Data loading ───────────────────────────────────────────────────
    runtime_metrics = asdict(runtime_profile)
    runtime_metrics.update(md_num_views=figure_settings.md_num_views,
                           raytrace_enabled=figure_settings.raytrace_enabled)
    runtime_metrics["directional_line_jepa"] = {
        "enabled": directional_line_jepa_settings.enabled,
        "num_directions": directional_line_jepa_settings.num_directions,
        "max_atoms_total": directional_line_jepa_settings.max_atoms_total,
        "atom_chunk_size": directional_line_jepa_settings.atom_chunk_size,
        "skip_reason": directional_skip_reason,
    }
    all_metrics: Dict[str, Any] = {"runtime_profile": runtime_metrics}
    dm = None
    if temporal_real_mode:
        _step("Building temporal dump analysis dataset")
        default_temporal_static_frame_index = 0
        if normalize_data_kind(getattr(cfg.data, "kind", None)) == "temporal_lammps":
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
                "Temporal dump analysis: inputs.temporal_real.static_frame_index was not set; "
                f"defaulting to the temporal center frame index {default_temporal_static_frame_index} "
                "to match temporal_lammps checkpoint training."
            )
        temporal_bundle = build_temporal_real_analysis_bundle(
            analysis_cfg=analysis_cfg,
            model_cfg=cfg,
            batch_size=int(analysis_inference_batch_size),
            dataloader_num_workers=int(input_settings.dataloader_num_workers),
            temporal_inference_spec=temporal_inference_spec,
        )
        dl = temporal_bundle.inference_dataloader
        analysis_source_names = list(temporal_bundle.selection.analysis_source_names)
        all_metrics["temporal_real_inputs"] = {
            **temporal_bundle.selection.dump_summary,
            **temporal_bundle.selection.to_cache_spec(),
            "inference": temporal_inference_spec.to_cache_spec(),
        }
        class_names = None
        print(
            "Temporal dump analysis enabled: "
            f"inference_snapshots={len(temporal_bundle.selection.inference_source_names)}, "
            f"analysis_snapshots={len(temporal_bundle.selection.analysis_source_names)}, "
            f"center_count={len(getattr(temporal_bundle.dataset, '_center_atom_indices', []))}, "
            f"inference_mode={temporal_inference_spec.mode}."
        )
    else:
        use_lazy_static_dataset = bool(
            preloaded_cache is not None
            and runtime_profile.lazy_static_dataset_on_cache_hit
            and normalized_data_kind in {"static", "line_static"}
        )
        if use_lazy_static_dataset:
            _step("Building lazy cache-backed static dataset")
            dl = build_lazy_static_analysis_dataloader(
                cfg,
                expected_coords=np.asarray(preloaded_cache["coords"], dtype=np.float32),
                batch_size=int(analysis_inference_batch_size),
                dataloader_num_workers=int(input_settings.dataloader_num_workers),
            )
            class_names = None
        else:
            _step("Building datamodule")
            dm = build_datamodule(
                cfg,
                require_coords_for_static=not is_synthetic,
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

    if temporal_real_mode:
        if input_settings.max_samples_total is not None:
            print(
                "[analysis] inputs.max_samples_total is ignored for temporal dump analysis; "
                "the temporal inference snapshot selection already defines the full inference set."
            )

    # ── Inference / cache ──────────────────────────────────────────────
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

    cache, model, cfg, device, cache_loaded = _collect_main_inference_cache(
        out_dir=out_dir,
        cfg=cfg,
        checkpoint_path=run_settings.checkpoint_path,
        cuda_device=run_settings.cuda_device,
        dataloader=dl,
        model=model,
        device=device,
        analysis_settings=analysis_settings,
        figure_only=bool(figure_settings.figure_only),
        cache_spec=cache_spec,
        max_batches_latent=max_batches_latent,
        max_samples_total=max_samples_total,
        seed_base=seed_base,
        temporal_bundle=temporal_bundle,
        step=_step,
        preloaded_cache=preloaded_cache,
        preloaded_cache_message=preloaded_cache_message,
    )
    n_samples = len(cache["inv_latents"])
    if temporal_bundle is not None:
        _validate_temporal_cache_anchor_order(cache, temporal_bundle.dataset)
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
            "data_kind": normalize_data_kind(getattr(fit_cfg.data, "kind", None)),
            "static_data_files_requested": analysis_settings.cluster_fit.input_settings.static_data_files,
            "static_data_files_resolved": [
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
    fit_phases_for_clustering = (
        None
        if fit_cache is None
        or np.asarray(fit_cache["phases"]).shape[0] != int(len(fit_cache["inv_latents"]))
        else np.asarray(fit_cache["phases"], dtype=int)
    )
    if temporal_bundle is not None:
        fit_reference_source = "main_temporal_inference_cache"
    elif fit_latents_for_clustering is None:
        fit_reference_source = "main_inference_cache"
    else:
        fit_reference_source = "clustering_fit_reference_cache"

    def _use_temporal_stratified_main_fit() -> None:
        nonlocal fit_latents_for_clustering, fit_phases_for_clustering, fit_reference_source
        if temporal_bundle is None or fit_latents_for_clustering is not None:
            return
        max_fit_samples = int(
            OmegaConf.select(
                analysis_cfg,
                "clustering.temporal_fit_max_samples",
                default=300000,
            )
        )
        anchor_frames_all = np.asarray(cache["anchor_frame_indices"], dtype=np.int64)

        def _record_temporal_fit_reference(anchor_frames: np.ndarray, sample_count: int) -> None:
            all_metrics["clustering_fit_reference"] = {
                "source": str(fit_reference_source),
                "data_source": "inputs.temporal_real.dump_file",
                "dump_file": str(temporal_bundle.selection.dump_file),
                "sample_count": int(sample_count),
                "total_sample_count": int(n_samples),
                "max_samples": None if max_fit_samples <= 0 else int(max_fit_samples),
                "unique_anchor_frames": int(np.unique(anchor_frames).size),
                "fit_on_all_main_temporal_samples": bool(int(sample_count) >= int(n_samples)),
                "stratified_by_anchor_frame": bool(int(sample_count) < int(n_samples)),
            }
            print(
                "[analysis][clustering] Temporal clustering fit source: "
                f"{fit_reference_source} with {int(sample_count)}/{int(n_samples)} samples "
                f"from {temporal_bundle.selection.dump_file}."
            )

        if max_fit_samples <= 0:
            _record_temporal_fit_reference(anchor_frames_all, int(n_samples))
            return
        fit_indices = _temporal_stratified_fit_indices(
            cache,
            max_samples=int(max_fit_samples),
            random_state=int(clustering_random_state),
        )
        if int(fit_indices.shape[0]) >= int(n_samples):
            _record_temporal_fit_reference(anchor_frames_all, int(n_samples))
            return
        fit_latents_for_clustering = np.asarray(
            cache["inv_latents"],
            dtype=np.float32,
        )[fit_indices]
        main_phases = np.asarray(cache["phases"], dtype=int)
        fit_phases_for_clustering = (
            main_phases[fit_indices]
            if main_phases.shape[0] == int(n_samples)
            else np.empty((0,), dtype=int)
        )
        fit_reference_source = "temporal_stratified_main_inference_cache"
        anchor_frames = np.asarray(cache["anchor_frame_indices"], dtype=np.int64)[fit_indices]
        _record_temporal_fit_reference(anchor_frames, int(fit_indices.shape[0]))

    _use_temporal_stratified_main_fit()
    runtime_fit_source_latents = (
        np.asarray(cache["inv_latents"], dtype=np.float32)
        if fit_latents_for_clustering is None
        else np.asarray(fit_latents_for_clustering, dtype=np.float32)
    )
    runtime_fit_source_phases = (
        np.asarray(cache["phases"], dtype=int)
        if fit_phases_for_clustering is None
        else np.asarray(fit_phases_for_clustering, dtype=int)
    )
    sampled_fit_latents, sampled_fit_phases, runtime_fit_indices = (
        subsample_clustering_reference(
            runtime_fit_source_latents,
            runtime_fit_source_phases,
            max_samples=runtime_profile.clustering_fit_max_samples,
            random_state=clustering_random_state,
        )
    )
    if runtime_fit_indices is not None:
        fit_latents_for_clustering = sampled_fit_latents
        fit_phases_for_clustering = sampled_fit_phases
        fit_reference_source = f"{fit_reference_source}_runtime_subsample"
        print(
            "[analysis][fast-path] Clustering fit subsample: "
            f"{runtime_fit_indices.shape[0]}/{runtime_fit_source_latents.shape[0]} rows."
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

    # ── Reusable clustering model fit ──────────────────────────────────
    clustering_requested_k_values = list(analysis_settings.cluster_k_values)
    clustering_fit_reference_latents = (
        np.asarray(cache["inv_latents"], dtype=np.float32)
        if fit_latents_for_clustering is None
        else np.asarray(fit_latents_for_clustering, dtype=np.float32)
    )
    clustering_fit_reference_phases = (
        np.asarray(cache["phases"], dtype=int)
        if fit_phases_for_clustering is None
        else np.asarray(fit_phases_for_clustering, dtype=int)
    )

    def _fit_cluster_reference(
        reference_latents: np.ndarray,
        reference_phases: np.ndarray,
    ):
        return fit_reusable_clustering_models(
            reference_latents,
            reference_phases,
            requested_k_values=clustering_requested_k_values,
            cluster_method=analysis_settings.cluster_method,
            random_state=clustering_random_state,
            l2_normalize=analysis_settings.cluster_l2_normalize,
            standardize=analysis_settings.cluster_standardize,
            pca_variance=analysis_settings.cluster_pca_var,
            pca_max_components=analysis_settings.cluster_pca_max_components,
        )

    _step("Fitting reusable clustering models")
    (
        clustering_fit_metrics,
        clustering_fit_configured_k_values,
        clustering_fit_labels_by_k,
        clustering_fit_methods_by_k,
        clustering_models_by_k,
    ) = _fit_cluster_reference(
        clustering_fit_reference_latents,
        clustering_fit_reference_phases,
    )
    clustering_fit_metrics["reusable_models_fitted"] = True
    clustering_fit_metrics["fit_reference_source"] = str(fit_reference_source)
    all_metrics["clustering_model_fit"] = clustering_fit_metrics
    temporal_dense_outputs, model = _collect_temporal_dense_outputs(
        analysis_cfg=analysis_cfg,
        temporal_bundle=temporal_bundle,
        temporal_inference_spec=temporal_inference_spec,
        checkpoint_path=run_settings.checkpoint_path,
        out_dir=out_dir,
        model_cfg_for_module=cfg,
        model=model,
        model_loader=load_vicreg_model,
        cuda_device=run_settings.cuda_device,
        seed_base=seed_base,
        figure_only=bool(figure_settings.figure_only),
        inference_batch_size=int(analysis_inference_batch_size),
        dataloader_num_workers=int(input_settings.dataloader_num_workers),
        progress_every_batches=analysis_settings.progress_every_batches,
        step=_step,
    )
    all_metrics.update(temporal_dense_outputs.metrics)
    temporal_snapshot_visualization_cache = temporal_dense_outputs.snapshot_cache
    temporal_snapshot_visualization_dataset = temporal_dense_outputs.snapshot_dataset
    temporal_snapshot_visualization_layout = temporal_dense_outputs.snapshot_layout
    temporal_md_space_animation_cache = temporal_dense_outputs.md_space_cache
    temporal_md_space_animation_layout = temporal_dense_outputs.md_space_layout
    temporal_md_space_animation_source_names = temporal_dense_outputs.md_space_source_names
    temporal_md_space_animation_spatial_bounds = temporal_dense_outputs.md_space_spatial_bounds
    temporal_md_space_animation_reuse_main_cache = (
        temporal_dense_outputs.md_space_reuse_main_cache
    )
    temporal_md_space_animation_enabled = temporal_dense_outputs.md_space_enabled
    clustering_features = None
    clustering_feature_prep = None
    clustering_features_for_fixed_k = None
    clustering_feature_prep_for_fixed_k = None

    dataset_obj = (
        temporal_bundle.dataset
        if temporal_bundle is not None
        else getattr(dl, "dataset", None)
    )
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
    figure_snapshot_layout_for_outputs = snapshot_layout_for_outputs
    figure_analysis_source_names_for_outputs = snapshot_analysis_source_names_for_outputs
    snapshot_figure_limit = runtime_profile.snapshot_figure_limit
    if (
        snapshot_figure_limit is not None
        and len(snapshot_layout_for_outputs.source_groups) > int(snapshot_figure_limit)
    ):
        group_count = len(snapshot_layout_for_outputs.source_groups)
        selected_figure_sources = select_evenly_spaced_names(
            [str(name) for name, _ in snapshot_layout_for_outputs.source_groups],
            snapshot_figure_limit,
        )
        figure_snapshot_layout_for_outputs = filter_snapshot_figure_layout(
            snapshot_layout_for_outputs,
            allowed_source_names=selected_figure_sources,
        )
        figure_analysis_source_names_for_outputs = selected_figure_sources
        print(
            "[analysis][fast-path] Snapshot figures limited to "
            f"{selected_figure_sources} ({len(selected_figure_sources)}/{group_count})."
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

    if temporal_snapshot_visualization_cache is not None:
        _step("Projecting cluster labels onto dense temporal snapshots")
        (
            snapshot_clustering_metrics,
            _,
            snapshot_cluster_labels_by_k_for_outputs,
            _,
        ) = predict_clustering_state_from_models(
            snapshot_latents_for_outputs,
            np.asarray(temporal_snapshot_visualization_cache["phases"], dtype=int),
            fitted_models_by_k=clustering_models_by_k,
            requested_k_values=clustering_requested_k_values,
            cluster_method=analysis_settings.cluster_method,
            random_state=clustering_random_state,
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
    temporal_md_animation_spatial_bounds_for_outputs = None
    if temporal_md_space_animation_cache is not None:
        _step("Projecting cluster labels onto dense MD-space animation frames")
        (
            md_animation_clustering_metrics,
            _,
            temporal_md_animation_cluster_labels_by_k_for_outputs,
            _,
        ) = predict_clustering_state_from_models(
            np.asarray(temporal_md_space_animation_cache["inv_latents"], dtype=np.float32),
            np.asarray(temporal_md_space_animation_cache["phases"], dtype=int),
            fitted_models_by_k=clustering_models_by_k,
            requested_k_values=clustering_requested_k_values,
            cluster_method=analysis_settings.cluster_method,
            random_state=clustering_random_state,
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
        temporal_md_animation_spatial_bounds_for_outputs = (
            temporal_md_space_animation_spatial_bounds
        )
        for cache_key in ("inv_latents", "eq_latents", "phases", "instance_ids"):
            temporal_md_space_animation_cache.pop(cache_key, None)

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

    representative_selection_cache: dict[tuple[str, int], tuple[np.ndarray, dict[str, Any]]] = {}

    def _representative_selection_for(
        latents_for_selection: np.ndarray,
        labels_by_k: dict[int, np.ndarray],
        k_value: int,
        *,
        source_name: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        cache_key = (str(source_name), int(k_value))
        cached = representative_selection_cache.get(cache_key)
        if cached is not None:
            return cached
        features, info = representative_features_from_clustering_model(
            latents_for_selection,
            fitted_model=clustering_models_by_k[int(k_value)],
            expected_labels=labels_by_k[int(k_value)],
        )
        info = {
            **dict(info),
            "source": str(source_name),
            "k": int(k_value),
        }
        representative_selection_cache[cache_key] = (features, info)
        return features, info

    # ── Figure-only early return ───────────────────────────────────────
    if figure_settings.figure_only:
        if fit_latents_for_clustering is None:
            clustering_metrics = dict(clustering_fit_metrics)
            configured_k_values = list(clustering_fit_configured_k_values)
            cluster_labels_by_k = dict(clustering_fit_labels_by_k)
        else:
            (
                clustering_metrics,
                configured_k_values,
                cluster_labels_by_k,
                _,
            ) = predict_clustering_state_from_models(
                cache["inv_latents"],
                cache["phases"],
                fitted_models_by_k=clustering_models_by_k,
                requested_k_values=list(analysis_settings.cluster_k_values),
                cluster_method=analysis_settings.cluster_method,
                random_state=clustering_random_state,
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
            representative_selection_features_for_k, representative_selection_info_for_k = (
                _representative_selection_for(
                    snapshot_latents_for_outputs,
                    figure_output_labels_by_k,
                    int(k_value),
                    source_name="figure_output_cache",
                )
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
                snapshot_layout=figure_snapshot_layout_for_outputs,
                analysis_source_names=figure_analysis_source_names_for_outputs,
                step=_step,
                representative_selection_features=representative_selection_features_for_k,
                representative_selection_info=representative_selection_info_for_k,
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
            pca_max_samples=_positive_int_or_none(
                OmegaConf.select(
                    analysis_cfg,
                    "pca.max_samples",
                    default=effective_tsne_max_samples,
                )
            ),
            latent_stats_max_samples=_positive_int_or_none(
                OmegaConf.select(
                    analysis_cfg,
                    "latent_stats.max_samples",
                    default=effective_tsne_max_samples,
                )
            ),
            latent_stats_correlation_max_samples=_positive_int_or_none(
                OmegaConf.select(
                    analysis_cfg,
                    "latent_stats.correlation_max_samples",
                    default=50000,
                )
            ),
        )
    )

    # ── Clustering ─────────────────────────────────────────────────────
    _step("Computing clustering labels")
    if fit_latents_for_clustering is None:
        clustering_metrics = dict(clustering_fit_metrics)
        configured_k_values = list(clustering_fit_configured_k_values)
        cluster_labels_by_k = dict(clustering_fit_labels_by_k)
        cluster_methods_by_k = dict(clustering_fit_methods_by_k)
    else:
        (
            clustering_metrics,
            configured_k_values,
            cluster_labels_by_k,
            cluster_methods_by_k,
        ) = predict_clustering_state_from_models(
            cache["inv_latents"],
            cache["phases"],
            fitted_models_by_k=clustering_models_by_k,
            requested_k_values=clustering_requested_k_values,
            cluster_method=analysis_settings.cluster_method,
            random_state=clustering_random_state,
        )
    all_metrics["clustering"] = clustering_metrics
    if temporal_md_space_animation_enabled and temporal_md_space_animation_reuse_main_cache:
        dense_snapshot_count = int(
            OmegaConf.select(
                analysis_cfg,
                "real_md.temporal.md_space.dense_snapshot_count",
                default=0,
            )
        )
        frame_indices, source_names = resolve_temporal_real_snapshot_subset(
            analysis_cfg=analysis_cfg,
            selection=temporal_bundle.selection,
            snapshot_count=int(dense_snapshot_count),
        )
        temporal_md_animation_layout_for_outputs = filter_snapshot_figure_layout(
            snapshot_layout_inference,
            allowed_source_names=[str(v) for v in source_names],
        )
        temporal_md_animation_order_for_outputs = [str(v) for v in source_names]
        temporal_md_animation_coords_for_outputs = np.asarray(coords, dtype=np.float32)
        temporal_md_animation_cluster_labels_by_k_for_outputs = cluster_labels_by_k
        temporal_md_animation_frame_source = "main_temporal_inference_subset"
        all_metrics["temporal_md_space_animation_sampling"] = {
            "enabled": True,
            "reused_main_inference_cache": True,
            "dense_snapshot_count": int(dense_snapshot_count),
            "frame_indices": [int(v) for v in frame_indices.tolist()],
            "source_names": [str(v) for v in source_names],
            "sample_count": int(
                sum(
                    int(np.asarray(indices, dtype=int).size)
                    for _source_name, indices in temporal_md_animation_layout_for_outputs.source_groups
                )
            ),
            "sample_count_by_snapshot": {
                str(source_name): int(np.asarray(indices, dtype=int).size)
                for source_name, indices in temporal_md_animation_layout_for_outputs.source_groups
            },
            "clustering_projection": "reused_main_clustering",
        }
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

    # ── Directional Line-JEPA uncertainty / novelty ───────────────────
    directional_line_jepa_metrics = run_directional_line_jepa_analysis(
        model=model,
        model_cfg=cfg,
        analysis_cfg=analysis_cfg,
        cache=cache,
        source_groups=snapshot_layout_inference.source_groups,
        fitted_clustering_model=clustering_models_by_k.get(primary_k),
        primary_k=primary_k,
        out_dir=out_dir,
        step=_step,
        settings=directional_line_jepa_settings,
    )
    if directional_line_jepa_metrics:
        all_metrics["directional_line_jepa"] = directional_line_jepa_metrics

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

    connected_representative_selection_features = None
    if connected_regime_settings.interactive_3d:
        connected_representative_selection_features, _ = (
            _representative_selection_for(
                np.asarray(cache["inv_latents"], dtype=np.float32),
                cluster_labels_by_k,
                int(primary_k),
                source_name="connected_regime_cluster_gallery",
            )
        )
    connected_regime_metrics = run_connected_regime_analysis(
        latents=np.asarray(cache["inv_latents"], dtype=np.float32),
        labels=np.asarray(cluster_labels, dtype=int),
        out_dir=out_dir,
        settings=connected_regime_settings,
        cluster_color_map=shared_cluster_color_map,
        frame_groups=snapshot_layout_inference.source_groups,
        dataset=dataset_obj,
        representatives_out_dir=Path(out_dir) / "real_md" / "representatives",
        representative_point_scale=float(point_scale),
        representative_target_points=int(figure_settings.representative_points),
        representative_selection_features=connected_representative_selection_features,
        step=_step,
    )
    if connected_regime_metrics:
        all_metrics["connected_regimes"] = connected_regime_metrics

    swav_prototype_metrics = run_swav_prototype_evaluation(
        model=model,
        cache=cache,
        out_dir=out_dir,
        analysis_cfg=analysis_cfg,
        cluster_labels_by_k=cluster_labels_by_k,
        cluster_color_maps_by_k=shared_cluster_color_maps_by_k,
        primary_k=int(primary_k),
        frame_groups=snapshot_source_groups,
        proportion_frame_groups=(
            None if temporal_bundle is None else snapshot_layout_inference.source_groups
        ),
        figure_settings=figure_settings,
        figure_set_run_kwargs=_build_figure_set_run_kwargs(figure_settings),
        figure_dataloader=dl,
        figure_snapshot_layout=figure_snapshot_layout_for_outputs,
        figure_analysis_source_names=figure_analysis_source_names_for_outputs,
        step=_step,
    )
    if swav_prototype_metrics:
        all_metrics["swav_prototypes"] = swav_prototype_metrics

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
        representative_selection_features, representative_selection_info = (
            _representative_selection_for(
                cache["inv_latents"],
                cluster_labels_by_k,
                int(figure_settings.k),
                source_name="main_inference_cache",
            )
        )
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
            selection_features=representative_selection_features,
            selection_info=representative_selection_info,
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
            representative_selection_features_for_k = None
            representative_selection_info_for_k = None
            if representative_render_cache_for_k is None:
                (
                    representative_selection_features_for_k,
                    representative_selection_info_for_k,
                ) = _representative_selection_for(
                    snapshot_latents_for_outputs,
                    figure_output_labels_by_k,
                    int(k_value),
                    source_name="figure_output_cache",
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
                snapshot_layout=figure_snapshot_layout_for_outputs,
                analysis_source_names=figure_analysis_source_names_for_outputs,
                step=_step,
                representative_render_cache=representative_render_cache_for_k,
                representative_selection_features=representative_selection_features_for_k,
                representative_selection_info=representative_selection_info_for_k,
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
        tsne_max_samples=effective_tsne_max_samples,
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

    cluster_assignment_margin_cache: dict[int, dict[str, Any]] = {}

    def _cluster_assignment_margins_for_k(k_value: int) -> dict[str, Any]:
        k_int = int(k_value)
        cached = cluster_assignment_margin_cache.get(k_int)
        if cached is not None:
            return cached
        margin_chunk_size = int(
            OmegaConf.select(
                analysis_cfg,
                "real_md.temporal.flicker.margin_chunk_size",
                default=200_000,
            )
        )
        _step(f"Computing cluster assignment margins for k={k_int}")
        margins = compute_clustering_assignment_margins(
            np.asarray(cache["inv_latents"], dtype=np.float32),
            fitted_model=clustering_models_by_k[k_int],
            expected_labels=np.asarray(cluster_labels_by_k[k_int], dtype=int),
            chunk_size=int(margin_chunk_size),
        )
        cluster_assignment_margin_cache[k_int] = margins
        return margins

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
                    effective_tsne_max_samples,
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
            real_md_representative_selection_features = None
            real_md_representative_selection_info = None
            if real_md_profiles_enabled and representative_render_cache_for_k is None:
                (
                    real_md_representative_selection_features,
                    real_md_representative_selection_info,
                ) = _representative_selection_for(
                    cache["inv_latents"],
                    cluster_labels_by_k,
                    int(k_value),
                    source_name="main_inference_cache",
                )
            flicker_metrics_enabled = bool(
                OmegaConf.select(
                    analysis_cfg,
                    "real_md.temporal.flicker.enabled",
                    default=True,
                )
            )
            real_md_assignment_margins_by_k = None
            instance_ids_for_real_md = np.asarray(cache["instance_ids"])
            if (
                flicker_metrics_enabled
                and instance_ids_for_real_md.reshape(-1).shape[0]
                == np.asarray(cache["inv_latents"]).shape[0]
            ):
                real_md_assignment_margins_by_k = {
                    int(k_value): _cluster_assignment_margins_for_k(int(k_value))
                }
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
                temporal_md_animation_spatial_bounds=(
                    temporal_md_animation_spatial_bounds_for_outputs
                ),
                temporal_projection_fit_indices=temporal_projection_fit_indices,
                point_scale=float(point_scale),
                random_state=int(clustering_random_state),
                representative_render_cache=representative_render_cache_for_k,
                representative_selection_features=real_md_representative_selection_features,
                representative_selection_info=real_md_representative_selection_info,
                cluster_assignment_margins_by_k=real_md_assignment_margins_by_k,
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
    if runtime_profile.equivariance_enabled:
        all_metrics.update(
            run_equivariance_evaluation(
                model,
                dl,
                device,
                out_dir,
                analysis_cfg=analysis_cfg,
                step=_step,
                temporal_sequence_mode=(
                    "static_anchor"
                    if temporal_bundle is None
                    else temporal_bundle.collection_inference_spec.mode
                ),
                temporal_static_frame_index=(
                    0
                    if temporal_bundle is None
                    else temporal_bundle.collection_inference_spec.static_frame_index
                ),
            )
        )
    else:
        all_metrics["runtime_profile"]["equivariance_skipped"] = True

    # ── Write metrics & summary ────────────────────────────────────────
    _step("Writing metrics")
    metrics_path = out_dir / "analysis_metrics.json"
    write_json(metrics_path, all_metrics)

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
