import argparse
from dataclasses import asdict, dataclass, replace
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict


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
    _resolve_input_settings, _resolve_run_settings,
    build_runtime_model_config, load_checkpoint_analysis_config,
    default_analysis_config_for_checkpoint,
)
from .connected_regimes import (
    resolve_connected_regime_settings,
    run_connected_regime_analysis,
)
from .gateway_phase import (
    resolve_gateway_phase_settings,
    run_gateway_phase_analysis,
)
from .dynamic_motif import run_dynamic_motif_analysis
from .figure_sets import (
    build_shared_cluster_color_map, filter_snapshot_figure_layout, print_figure_set_summary,
    render_cluster_figure_outputs, resolve_snapshot_figure_layout,
)
from .inference_cache import (
    _build_inference_cache_spec, _inference_cache_spec_hash, _load_inference_cache,
)
from .lazy_static_dataset import build_lazy_static_analysis_dataloader
from .latent_vis import print_analysis_summary, run_equivariance_evaluation, run_pca_and_latent_stats, run_tsne_visualizations
from .md_outputs import build_md_metrics
from .output_layout import real_md_outputs_root, real_md_outputs_root_for_k, write_json
from .report import report_directory, publish_report
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
from .real_md_qualitative import RealMDAnalysis, append_dynamic_motif_summary
from .runtime_profile import (
    resolve_analysis_runtime_profile,
    select_evenly_spaced_names,
    subsample_clustering_reference,
)
from .temporal_dense import (
    _collect_temporal_dense_outputs,
)
from .temporal_real import (
    build_temporal_real_analysis_bundle,
    resolve_temporal_real_inference_spec,
    resolve_temporal_real_snapshot_subset,
    temporal_real_analysis_enabled,
)
from .swav_eval import run_swav_prototype_evaluation
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
            "analysis in a new output directory to regenerate the cache."
        )
    if cached.shape[0] != n_samples:
        raise ValueError(
            "Temporal inference cache anchor_frame_indices length mismatch: "
            f"anchor_frame_indices={cached.shape[0]}, inv_latents={n_samples}."
        )
    expected_all = np.asarray(dataset.sample_anchor_frame_indices, dtype=np.int64)
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
    return _AnalysisRun(checkpoint_path, output_dir, cuda_device, analysis_config_path, analysis_cfg).run()


@dataclass
class _AnalysisRun:
    checkpoint_path: str | None
    output_dir: str | None
    cuda_device: int | None
    analysis_config_path: str | None
    analysis_cfg: DictConfig | None

    def __post_init__(self):
        torch.set_float32_matmul_precision("high")
        self.started = time.perf_counter()
        self.previous_step_time = self.started
        self.step_index = 0
        self.representative_selection_cache = {}

    def step(self, message):
        self.step_index += 1
        now = time.perf_counter()
        previous_duration = now - self.previous_step_time
        self.previous_step_time = now
        print(f"[analysis][step {self.step_index}][total={now - self.started:7.1f}s]"
              f"[previous={previous_duration:6.1f}s] {message}")

    def run(self):
        self.configure()
        self.load_data()
        self.collect_embeddings()
        self.fit_clustering_and_prepare_snapshots()
        self.compute_diagnostics_and_clustering()
        self.build_shared_colors()
        self.render_cluster_figures()
        self.write_md_outputs()
        self.analyze_real_md()
        self.analyze_dynamic_motifs_and_equivariance()
        return self.publish()

    def configure(self):
        self.step("Loading analysis config")
        if self.analysis_cfg is None:
            if self.analysis_config_path is None and self.checkpoint_path is not None:
                self.analysis_config_path = str(default_analysis_config_for_checkpoint(self.checkpoint_path))
            self.analysis_cfg = load_checkpoint_analysis_config(self.analysis_config_path)
        self.run_settings = _resolve_run_settings(
            self.analysis_cfg,
            checkpoint_path_override=self.checkpoint_path,
            output_dir_override=self.output_dir,
            cuda_device_override=self.cuda_device,
        )
        from src.experiment_runner.artifacts import analysis_artifacts
        self.result_root = Path(self.run_settings.output_dir)
        self.result_root.mkdir(parents=True, exist_ok=True)
        self.out_dir = analysis_artifacts(self.result_root)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if (self.out_dir / 'analysis_metrics.json').exists():
            raise FileExistsError(f'Completed numerical analysis already exists: {self.out_dir}. '
                                  'Use a new output directory for recomputation, or publish existing evidence.')
        from src.experiment_runner.metric_docs import check_metric_docs
        check_metric_docs(family='analysis')

        self.step("Loading checkpoint training config")
        self.cfg = build_runtime_model_config(self.run_settings.checkpoint_path, self.analysis_cfg)
        self.input_settings = _resolve_input_settings(self.analysis_cfg)
        self.model: Any | None = None
        self.device = f"cuda:{self.run_settings.cuda_device}" if torch.cuda.is_available() else "cpu"
        self.analysis_source_names: list[str] | None = None
        self.temporal_bundle = None
        self.temporal_inference_spec = None
        self.temporal_real_mode = temporal_real_analysis_enabled(self.analysis_cfg)
        analysis_files = None if self.temporal_real_mode else _resolve_analysis_files(self.cfg, self.input_settings)
        if analysis_files is not None:
            self.analysis_source_names = _configure_static_analysis_inputs(self.cfg, analysis_files)
            print(f"Analysis data_files: {analysis_files}")
            if self.analysis_source_names and len(self.analysis_source_names) > 1:
                print(f"Per-snapshot analysis sources: {self.analysis_source_names}")

        with open_dict(self.cfg):
            self.cfg.num_workers = int(self.input_settings.dataloader_num_workers)
        print(f"Analysis dataloader workers: {self.input_settings.dataloader_num_workers}")
        self.analysis_settings = _resolve_analysis_settings(self.analysis_cfg, self.cfg)
        if self.temporal_real_mode and self.analysis_settings.cluster_fit is not None:
            raise ValueError(
                "Temporal dump analysis does not use clustering.fit_inputs. "
                "Clustering is fit from inputs.temporal_real.dump_file through the "
                "main temporal inference cache. Delete clustering.fit_inputs and control the "
                "temporal fit subset with clustering.temporal_fit_max_samples."
            )
        self.real_md_selected_k = int(self.analysis_settings.primary_k)
        self.runtime_profile = resolve_analysis_runtime_profile(self.analysis_cfg)
        if self.runtime_profile.real_md_projection_method is not None:
            with open_dict(self.analysis_cfg):
                OmegaConf.update(
                    self.analysis_cfg,
                    "real_md.projection.method",
                    self.runtime_profile.real_md_projection_method,
                    merge=False,
                    force_add=True,
                )
        self.effective_tsne_max_samples = int(self.analysis_settings.tsne_max_samples)
        if self.runtime_profile.tsne_max_samples is not None:
            self.effective_tsne_max_samples = min(
                self.effective_tsne_max_samples,
                int(self.runtime_profile.tsne_max_samples),
            )
        self.figure_settings = _resolve_figure_set_settings(
            self.analysis_cfg,
            self.cfg,
            out_dir=self.out_dir,
            primary_k=int(self.analysis_settings.primary_k),
        )
        self.figure_settings = replace(
            self.figure_settings,
            raytrace_enabled=self.figure_settings.raytrace_enabled and self.runtime_profile.raytrace_enabled,
            md_num_views=(
                self.figure_settings.md_num_views
                if self.runtime_profile.md_num_views is None
                else min(self.figure_settings.md_num_views, self.runtime_profile.md_num_views)
            ),
        )
        self.connected_regime_settings = resolve_connected_regime_settings(
            self.analysis_cfg,
            default_random_state=int(self.analysis_settings.seed_base),
        )
        self.gateway_phase_settings = resolve_gateway_phase_settings(self.analysis_cfg)
        self.hdbscan_settings = self.analysis_settings.hdbscan
        if self.analysis_settings.cluster_fit is not None and self.hdbscan_settings.enabled:
            raise ValueError(
                "clustering.fit_inputs is not compatible with clustering.hdbscan.enabled yet. "
                "Disable HDBSCAN or disable clustering.fit_inputs."
            )
        if (
            temporal_real_analysis_enabled(self.analysis_cfg)
            and bool(
                OmegaConf.select(
                    self.analysis_cfg,
                    "inputs.temporal_real.snapshot_visualization.enabled",
                    default=True,
                )
            )
            and self.hdbscan_settings.enabled
        ):
            raise ValueError(
                "inputs.temporal_real.snapshot_visualization is not compatible with "
                "clustering.hdbscan.enabled yet. Disable HDBSCAN or disable dense "
                "temporal snapshot visualization."
            )
        if (
            temporal_real_analysis_enabled(self.analysis_cfg)
            and OmegaConf.select(
                self.analysis_cfg,
                "real_md.temporal.md_space.dense_snapshot_count",
                default=None,
            )
            not in {None, "", 0}
            and self.hdbscan_settings.enabled
        ):
            raise ValueError(
                "real_md.temporal.md_space.dense_snapshot_count is not compatible with "
                "clustering.hdbscan.enabled yet. Disable HDBSCAN or disable dense "
                "MD-space animation sampling."
            )
        _print_resolved_analysis_settings(
            analysis_settings=self.analysis_settings,
            figure_settings=self.figure_settings,
        )
        self.is_synthetic = str(self.cfg.data.kind).strip().lower() == "synthetic"
        self.analysis_inference_batch_size = _resolve_analysis_inference_batch_size(self.cfg, self.input_settings)
        print(
            "Analysis inference batch size: "
            f"{self.analysis_inference_batch_size} "
            f"(checkpoint batch_size={int(self.cfg.batch_size)})"
        )
        print(f"Analysis runtime profile: {self.runtime_profile}")

        self.max_batches_latent = self.input_settings.max_batches_latent
        self.max_samples_total = (
            None
            if self.temporal_real_mode
            else _resolve_analysis_max_samples_total(
                self.input_settings,
                is_synthetic=self.is_synthetic,
                md_use_all_points=self.analysis_settings.md_use_all_points,
            )
        )
        self.seed_base = int(self.analysis_settings.seed_base)
        self.clustering_random_state = int(self.analysis_settings.seed_base)
        self.preloaded_cache: dict[str, np.ndarray] | None = None
        self.preloaded_cache_message: str | None = None
        self.static_cache_spec = None
        self.normalized_data_kind = str(self.cfg.data.kind).strip().lower()
        if (
            not self.temporal_real_mode
            and self.normalized_data_kind == "static"
            and self.analysis_settings.inference_cache_enabled
            and not self.analysis_settings.inference_cache_force_recompute
        ):
            self.static_cache_spec = _build_inference_cache_spec(
                checkpoint_path=self.run_settings.checkpoint_path,
                cfg=self.cfg,
                inference_batch_size=int(self.analysis_inference_batch_size),
                max_batches_latent=self.max_batches_latent,
                max_samples_total=self.max_samples_total,
                seed_base=self.seed_base,
                collector_mode=("tmf_sequence" if self.cfg.model_type == "temporal_motif_field" else "generic"),
            )
            self.preloaded_cache, self.preloaded_cache_message = _load_inference_cache(
                out_dir=self.out_dir,
                cache_filename=self.analysis_settings.inference_cache_file,
                expected_spec=self.static_cache_spec,
            )
            print(f"[analysis][cache preflight] {self.preloaded_cache_message}")

    def load_data(self):
        runtime_metrics = asdict(self.runtime_profile)
        runtime_metrics.update(md_num_views=self.figure_settings.md_num_views,
                               raytrace_enabled=self.figure_settings.raytrace_enabled)
        from src.experiment_runner.registry import sha256
        self.all_metrics: Dict[str, Any] = {"runtime_profile": runtime_metrics,
                                      "checkpoint_sha256": sha256(Path(self.run_settings.checkpoint_path))}
        dm = None
        if self.temporal_real_mode:
            self.step("Building temporal dump analysis dataset")
            default_temporal_static_frame_index = 0
            if str(self.cfg.data.kind).strip().lower() == "temporal_lammps":
                default_temporal_static_frame_index = int(getattr(self.cfg.data, "sequence_length", 1)) // 2
            configured_static_frame_index = OmegaConf.select(
                self.analysis_cfg,
                "inputs.temporal_real.static_frame_index",
                default=None,
            )
            self.temporal_inference_spec = resolve_temporal_real_inference_spec(
                self.analysis_cfg,
                default_static_frame_index=default_temporal_static_frame_index,
            )
            if (
                self.temporal_inference_spec.mode == "static_anchor"
                and configured_static_frame_index is None
                and default_temporal_static_frame_index != 0
            ):
                print(
                    "Temporal dump analysis: inputs.temporal_real.static_frame_index was not set; "
                    f"defaulting to the temporal center frame index {default_temporal_static_frame_index} "
                    "to match temporal_lammps checkpoint training."
                )
            self.temporal_bundle = build_temporal_real_analysis_bundle(
                analysis_cfg=self.analysis_cfg,
                model_cfg=self.cfg,
                batch_size=int(self.analysis_inference_batch_size),
                dataloader_num_workers=int(self.input_settings.dataloader_num_workers),
                temporal_inference_spec=self.temporal_inference_spec,
            )
            self.dl = self.temporal_bundle.inference_dataloader
            self.analysis_source_names = list(self.temporal_bundle.selection.analysis_source_names)
            self.all_metrics["temporal_real_inputs"] = {
                **self.temporal_bundle.selection.dump_summary,
                **self.temporal_bundle.selection.to_cache_spec(),
                "inference": self.temporal_inference_spec.to_cache_spec(),
            }
            self.class_names = None
            print(
                "Temporal dump analysis enabled: "
                f"inference_snapshots={len(self.temporal_bundle.selection.inference_source_names)}, "
                f"analysis_snapshots={len(self.temporal_bundle.selection.analysis_source_names)}, "
                f"center_count={len(getattr(self.temporal_bundle.dataset, '_center_atom_indices', []))}, "
                f"inference_mode={self.temporal_inference_spec.mode}."
            )
        else:
            use_lazy_static_dataset = bool(
                self.preloaded_cache is not None
                and self.runtime_profile.lazy_static_dataset_on_cache_hit
                and self.normalized_data_kind == "static"
            )
            if use_lazy_static_dataset:
                self.step("Building lazy cache-backed static dataset")
                self.dl = build_lazy_static_analysis_dataloader(
                    self.cfg,
                    expected_coords=np.asarray(self.preloaded_cache["coords"], dtype=np.float32),
                    batch_size=int(self.analysis_inference_batch_size),
                    dataloader_num_workers=int(self.input_settings.dataloader_num_workers),
                )
                self.class_names = None
            else:
                self.step("Building datamodule")
                dm = build_datamodule(
                    self.cfg,
                    require_coords_for_static=not self.is_synthetic,
                )
                dm.setup(stage="fit")
                self.dl = _build_analysis_dataloader(
                    self.cfg,
                    dm,
                    is_synthetic=self.is_synthetic,
                    inference_batch_size=int(self.analysis_inference_batch_size),
                    dataloader_num_workers=int(self.input_settings.dataloader_num_workers),
                )
                self.class_names = _extract_class_names(dm.train_dataset)
        print(f"Loaded class names: {self.class_names}")

        if self.temporal_real_mode:
            if self.input_settings.max_samples_total is not None:
                print(
                    "[analysis] inputs.max_samples_total is ignored for temporal dump analysis; "
                    "the temporal inference snapshot selection already defines the full inference set."
                )

    def collect_embeddings(self):
        cache_spec = self.static_cache_spec if self.preloaded_cache is not None else None
        if cache_spec is None:
            cache_spec = _build_inference_cache_spec(
                checkpoint_path=self.run_settings.checkpoint_path,
                cfg=self.cfg,
                inference_batch_size=int(self.analysis_inference_batch_size),
                max_batches_latent=self.max_batches_latent,
                max_samples_total=self.max_samples_total,
                seed_base=int(self.seed_base),
                temporal_real_selection=(
                    None if self.temporal_bundle is None else self.temporal_bundle.selection.to_cache_spec()
                ),
                temporal_sequence_inference=(
                    None if self.temporal_inference_spec is None else self.temporal_inference_spec.to_cache_spec()
                ),
                collector_mode=(
                    "tmf_sequence"
                    if str(getattr(self.cfg, "model_type", "")).strip().lower() == "temporal_motif_field"
                    else "generic"
                ),
            )

        self.cache, self.model, self.cfg, self.device, cache_loaded = _collect_main_inference_cache(
            out_dir=self.out_dir,
            cfg=self.cfg,
            checkpoint_path=self.run_settings.checkpoint_path,
            cuda_device=self.run_settings.cuda_device,
            dataloader=self.dl,
            model=self.model,
            device=self.device,
            analysis_settings=self.analysis_settings,
            cache_spec=cache_spec,
            max_batches_latent=self.max_batches_latent,
            max_samples_total=self.max_samples_total,
            seed_base=self.seed_base,
            temporal_bundle=self.temporal_bundle,
            step=self.step,
            preloaded_cache=self.preloaded_cache,
            preloaded_cache_message=self.preloaded_cache_message,
        )
        self.n_samples = len(self.cache["inv_latents"])
        if self.temporal_bundle is not None:
            _validate_temporal_cache_anchor_order(self.cache, self.temporal_bundle.dataset)
        print(f"Collected {self.n_samples} samples for analysis")
        self.all_metrics["inference_cache"] = {
            "enabled": bool(self.analysis_settings.inference_cache_enabled),
            "file": str((self.out_dir / self.analysis_settings.inference_cache_file)),
            "loaded_from_cache": bool(cache_loaded),
            "force_recompute": bool(self.analysis_settings.inference_cache_force_recompute),
            "spec_sha256": _inference_cache_spec_hash(cache_spec),
        }
        if bool(OmegaConf.select(self.analysis_cfg, 'topology.enabled', default=False)):
            from .topology import run_topology_analysis
            self.all_metrics['topology'] = run_topology_analysis(
                model=self.model, cfg=self.cfg, analysis_cfg=self.analysis_cfg,
                checkpoint_path=self.run_settings.checkpoint_path, out_dir=self.out_dir,
                main_cache=self.cache, step=self.step)
        fit_cache: dict[str, np.ndarray] | None = None
        fit_cache_loaded = False
        fit_cfg: DictConfig | None = None
        fit_source_names: list[str] | None = None
        if self.analysis_settings.cluster_fit is not None:
            self.step("Loading clustering fit-reference data")
            (
                fit_cache,
                fit_cfg,
                fit_source_names,
                self.model,
                fit_cache_loaded,
            ) = _collect_clustering_fit_cache(
                analysis_cfg=self.analysis_cfg,
                fit_settings=self.analysis_settings.cluster_fit,
                checkpoint_path=self.run_settings.checkpoint_path,
                out_dir=self.out_dir,
                model_cfg_for_module=self.cfg,
                model=self.model,
                cuda_device=self.run_settings.cuda_device,
                seed_base=self.seed_base,
                    progress_every_batches=self.analysis_settings.progress_every_batches,
            )
            self.all_metrics["clustering_fit_inputs"] = {
                "enabled": True,
                "data_config": self.analysis_settings.cluster_fit.data_config_path,
                "data_kind": str(fit_cfg.data.kind).strip().lower(),
                "static_data_files_requested": self.analysis_settings.cluster_fit.input_settings.static_data_files,
                "static_data_files_resolved": [
                    str(v)
                    for v in list(getattr(fit_cfg.data, "data_files", []) or [])
                ],
                "source_names": fit_source_names,
                "sample_count": int(len(fit_cache["inv_latents"])),
                "cache_enabled": bool(self.analysis_settings.cluster_fit.cache_enabled),
                "cache_file": str(self.out_dir / self.analysis_settings.cluster_fit.cache_file),
                "cache_loaded_from_disk": bool(fit_cache_loaded),
                "cache_force_recompute": bool(
                    self.analysis_settings.cluster_fit.cache_force_recompute
                ),
            }
        self.fit_latents_for_clustering = (
            None
            if fit_cache is None
            else np.asarray(fit_cache["inv_latents"], dtype=np.float32)
        )
        self.fit_phases_for_clustering = (
            None
            if fit_cache is None
            or np.asarray(fit_cache["phases"]).shape[0] != int(len(fit_cache["inv_latents"]))
            else np.asarray(fit_cache["phases"], dtype=int)
        )
        if self.temporal_bundle is not None:
            self.fit_reference_source = "main_temporal_inference_cache"
        elif self.fit_latents_for_clustering is None:
            self.fit_reference_source = "main_inference_cache"
        else:
            self.fit_reference_source = "clustering_fit_reference_cache"

        def _use_temporal_stratified_main_fit() -> None:
            if self.temporal_bundle is None or self.fit_latents_for_clustering is not None:
                return
            max_fit_samples = int(
                OmegaConf.select(
                    self.analysis_cfg,
                    "clustering.temporal_fit_max_samples",
                    default=300000,
                )
            )
            anchor_frames_all = np.asarray(self.cache["anchor_frame_indices"], dtype=np.int64)

            def _record_temporal_fit_reference(anchor_frames: np.ndarray, sample_count: int) -> None:
                self.all_metrics["clustering_fit_reference"] = {
                    "source": str(self.fit_reference_source),
                    "data_source": "inputs.temporal_real.dump_file",
                    "dump_file": str(self.temporal_bundle.selection.dump_file),
                    "sample_count": int(sample_count),
                    "total_sample_count": int(self.n_samples),
                    "max_samples": None if max_fit_samples <= 0 else int(max_fit_samples),
                    "unique_anchor_frames": int(np.unique(anchor_frames).size),
                    "fit_on_all_main_temporal_samples": bool(int(sample_count) >= int(self.n_samples)),
                    "stratified_by_anchor_frame": bool(int(sample_count) < int(self.n_samples)),
                }
                print(
                    "[analysis][clustering] Temporal clustering fit source: "
                    f"{self.fit_reference_source} with {int(sample_count)}/{int(self.n_samples)} samples "
                    f"from {self.temporal_bundle.selection.dump_file}."
                )

            if max_fit_samples <= 0:
                _record_temporal_fit_reference(anchor_frames_all, int(self.n_samples))
                return
            fit_indices = _temporal_stratified_fit_indices(
                self.cache,
                max_samples=int(max_fit_samples),
                random_state=int(self.clustering_random_state),
            )
            if int(fit_indices.shape[0]) >= int(self.n_samples):
                _record_temporal_fit_reference(anchor_frames_all, int(self.n_samples))
                return
            self.fit_latents_for_clustering = np.asarray(
                self.cache["inv_latents"],
                dtype=np.float32,
            )[fit_indices]
            main_phases = np.asarray(self.cache["phases"], dtype=int)
            self.fit_phases_for_clustering = (
                main_phases[fit_indices]
                if main_phases.shape[0] == int(self.n_samples)
                else np.empty((0,), dtype=int)
            )
            self.fit_reference_source = "temporal_stratified_main_inference_cache"
            anchor_frames = np.asarray(self.cache["anchor_frame_indices"], dtype=np.int64)[fit_indices]
            _record_temporal_fit_reference(anchor_frames, int(fit_indices.shape[0]))

        _use_temporal_stratified_main_fit()
        runtime_fit_source_latents = (
            np.asarray(self.cache["inv_latents"], dtype=np.float32)
            if self.fit_latents_for_clustering is None
            else np.asarray(self.fit_latents_for_clustering, dtype=np.float32)
        )
        runtime_fit_source_phases = (
            np.asarray(self.cache["phases"], dtype=int)
            if self.fit_phases_for_clustering is None
            else np.asarray(self.fit_phases_for_clustering, dtype=int)
        )
        sampled_fit_latents, sampled_fit_phases, runtime_fit_indices = (
            subsample_clustering_reference(
                runtime_fit_source_latents,
                runtime_fit_source_phases,
                max_samples=self.runtime_profile.clustering_fit_max_samples,
                random_state=self.clustering_random_state,
            )
        )
        if runtime_fit_indices is not None:
            self.fit_latents_for_clustering = sampled_fit_latents
            self.fit_phases_for_clustering = sampled_fit_phases
            self.fit_reference_source = f"{self.fit_reference_source}_runtime_subsample"
            print(
                "[analysis][fast-path] Clustering fit subsample: "
                f"{runtime_fit_indices.shape[0]}/{runtime_fit_source_latents.shape[0]} rows."
            )
        self.coords = self.cache["coords"]
        self.md_metrics_key = "synthetic_md" if self.is_synthetic else "real_md"
        self.point_scale = (
            resolve_point_scale(self.cfg)
            if self.figure_settings.profile_point_scale_enabled
            else 1.0
        )
        print(
            "Representative point scaling: "
            f"enabled={self.figure_settings.profile_point_scale_enabled}, point_scale={self.point_scale:.6g}"
        )

    def fit_clustering_and_prepare_snapshots(self):
        self.clustering_requested_k_values = list(self.analysis_settings.cluster_k_values)
        clustering_fit_reference_latents = (
            np.asarray(self.cache["inv_latents"], dtype=np.float32)
            if self.fit_latents_for_clustering is None
            else np.asarray(self.fit_latents_for_clustering, dtype=np.float32)
        )
        clustering_fit_reference_phases = (
            np.asarray(self.cache["phases"], dtype=int)
            if self.fit_phases_for_clustering is None
            else np.asarray(self.fit_phases_for_clustering, dtype=int)
        )


        self.step("Fitting reusable clustering models")
        (
            self.clustering_fit_metrics,
            self.clustering_fit_configured_k_values,
            self.clustering_fit_labels_by_k,
            self.clustering_fit_methods_by_k,
            self.clustering_models_by_k,
        ) = fit_reusable_clustering_models(
            clustering_fit_reference_latents,
            clustering_fit_reference_phases,
            requested_k_values=self.clustering_requested_k_values,
            cluster_method=self.analysis_settings.cluster_method,
            random_state=self.clustering_random_state,
            l2_normalize=self.analysis_settings.cluster_l2_normalize,
            standardize=self.analysis_settings.cluster_standardize,
            pca_variance=self.analysis_settings.cluster_pca_var,
            pca_max_components=self.analysis_settings.cluster_pca_max_components,
        )
        self.clustering_fit_metrics["reusable_models_fitted"] = True
        self.clustering_fit_metrics["fit_reference_source"] = str(self.fit_reference_source)
        self.all_metrics["clustering_model_fit"] = self.clustering_fit_metrics
        temporal_dense_outputs, self.model = _collect_temporal_dense_outputs(
            analysis_cfg=self.analysis_cfg,
            temporal_bundle=self.temporal_bundle,
            temporal_inference_spec=self.temporal_inference_spec,
            checkpoint_path=self.run_settings.checkpoint_path,
            out_dir=self.out_dir,
            model_cfg_for_module=self.cfg,
            model=self.model,
            model_loader=load_vicreg_model,
            cuda_device=self.run_settings.cuda_device,
            seed_base=self.seed_base,
            inference_batch_size=int(self.analysis_inference_batch_size),
            dataloader_num_workers=int(self.input_settings.dataloader_num_workers),
            progress_every_batches=self.analysis_settings.progress_every_batches,
            step=self.step,
        )
        self.all_metrics.update(temporal_dense_outputs.metrics)
        temporal_snapshot_visualization_cache = temporal_dense_outputs.snapshot_cache
        temporal_snapshot_visualization_dataset = temporal_dense_outputs.snapshot_dataset
        temporal_snapshot_visualization_layout = temporal_dense_outputs.snapshot_layout
        temporal_md_space_animation_cache = temporal_dense_outputs.md_space_cache
        temporal_md_space_animation_layout = temporal_dense_outputs.md_space_layout
        temporal_md_space_animation_source_names = temporal_dense_outputs.md_space_source_names
        temporal_md_space_animation_spatial_bounds = temporal_dense_outputs.md_space_spatial_bounds
        self.temporal_md_space_animation_reuse_main_cache = (
            temporal_dense_outputs.md_space_reuse_main_cache
        )
        self.temporal_md_space_animation_enabled = temporal_dense_outputs.md_space_enabled
        self.clustering_features = None
        self.clustering_feature_prep = None
        self.clustering_features_for_fixed_k = None
        self.clustering_feature_prep_for_fixed_k = None

        self.dataset_obj = (
            self.temporal_bundle.dataset
            if self.temporal_bundle is not None
            else getattr(self.dl, "dataset", None)
        )
        self.snapshot_layout_inference = resolve_snapshot_figure_layout(
            self.dataset_obj,
            is_synthetic=self.is_synthetic,
            n_samples=self.n_samples,
            analysis_source_names=(
                None
                if self.temporal_bundle is None
                else self.temporal_bundle.selection.inference_source_names
            ),
        )
        snapshot_layout = (
            filter_snapshot_figure_layout(
                self.snapshot_layout_inference,
                allowed_source_names=self.temporal_bundle.selection.analysis_source_names,
            )
            if self.temporal_bundle is not None
            else self.snapshot_layout_inference
        )
        self.snapshot_source_groups = snapshot_layout.source_groups
        self.snapshot_output_names = snapshot_layout.output_names
        self.multi_snapshot_real = snapshot_layout.multi_snapshot_real
        self.snapshot_layout_for_outputs = (
            snapshot_layout
            if temporal_snapshot_visualization_layout is None
            else temporal_snapshot_visualization_layout
        )
        self.snapshot_dataset_obj_for_outputs = (
            self.dataset_obj
            if temporal_snapshot_visualization_dataset is None
            else temporal_snapshot_visualization_dataset
        )
        self.snapshot_latents_for_outputs = (
            np.asarray(self.cache["inv_latents"], dtype=np.float32)
            if temporal_snapshot_visualization_cache is None
            else np.asarray(
                temporal_snapshot_visualization_cache["inv_latents"],
                dtype=np.float32,
            )
        )
        self.snapshot_coords_for_outputs = (
            np.asarray(self.coords, dtype=np.float32)
            if temporal_snapshot_visualization_cache is None
            else np.asarray(
                temporal_snapshot_visualization_cache["coords"],
                dtype=np.float32,
            )
        )
        snapshot_analysis_source_names_for_outputs = (
            self.analysis_source_names
            if temporal_snapshot_visualization_cache is None
            else list(self.temporal_bundle.selection.analysis_source_names)
        )
        self.figure_snapshot_layout_for_outputs = self.snapshot_layout_for_outputs
        self.figure_analysis_source_names_for_outputs = snapshot_analysis_source_names_for_outputs
        snapshot_figure_limit = self.runtime_profile.snapshot_figure_limit
        if (
            snapshot_figure_limit is not None
            and len(self.snapshot_layout_for_outputs.source_groups) > int(snapshot_figure_limit)
        ):
            group_count = len(self.snapshot_layout_for_outputs.source_groups)
            selected_figure_sources = select_evenly_spaced_names(
                [str(name) for name, _ in self.snapshot_layout_for_outputs.source_groups],
                snapshot_figure_limit,
            )
            self.figure_snapshot_layout_for_outputs = filter_snapshot_figure_layout(
                self.snapshot_layout_for_outputs,
                allowed_source_names=selected_figure_sources,
            )
            self.figure_analysis_source_names_for_outputs = selected_figure_sources
            print(
                "[analysis][fast-path] Snapshot figures limited to "
                f"{selected_figure_sources} ({len(selected_figure_sources)}/{group_count})."
            )
        self.snapshot_cluster_labels_by_k_for_outputs: dict[int, np.ndarray] | None = None


        if temporal_snapshot_visualization_cache is not None:
            self.step("Projecting cluster labels onto dense temporal snapshots")
            (
                snapshot_clustering_metrics,
                _,
                self.snapshot_cluster_labels_by_k_for_outputs,
                _,
            ) = predict_clustering_state_from_models(
                self.snapshot_latents_for_outputs,
                np.asarray(temporal_snapshot_visualization_cache["phases"], dtype=int),
                fitted_models_by_k=self.clustering_models_by_k,
                requested_k_values=self.clustering_requested_k_values,
                cluster_method=self.analysis_settings.cluster_method,
                random_state=self.clustering_random_state,
            )
            self.all_metrics.setdefault("temporal_snapshot_visualization", {}).update(
                {
                    "clustering_projection": snapshot_clustering_metrics,
                }
            )
        self.temporal_md_animation_layout_for_outputs = temporal_snapshot_visualization_layout
        self.temporal_md_animation_order_for_outputs = (
            None
            if temporal_snapshot_visualization_cache is None
            else list(snapshot_analysis_source_names_for_outputs)
        )
        self.temporal_md_animation_coords_for_outputs = (
            None
            if temporal_snapshot_visualization_cache is None
            else np.asarray(
                temporal_snapshot_visualization_cache["coords"],
                dtype=np.float32,
            )
        )
        self.temporal_md_animation_cluster_labels_by_k_for_outputs = (
            self.snapshot_cluster_labels_by_k_for_outputs
        )
        self.temporal_md_animation_frame_source = (
            None
            if temporal_snapshot_visualization_cache is None
            else "dense_selected_frames"
        )
        self.temporal_md_animation_spatial_bounds_for_outputs = None
        if temporal_md_space_animation_cache is not None:
            self.step("Projecting cluster labels onto dense MD-space animation frames")
            (
                md_animation_clustering_metrics,
                _,
                self.temporal_md_animation_cluster_labels_by_k_for_outputs,
                _,
            ) = predict_clustering_state_from_models(
                np.asarray(temporal_md_space_animation_cache["inv_latents"], dtype=np.float32),
                np.asarray(temporal_md_space_animation_cache["phases"], dtype=int),
                fitted_models_by_k=self.clustering_models_by_k,
                requested_k_values=self.clustering_requested_k_values,
                cluster_method=self.analysis_settings.cluster_method,
                random_state=self.clustering_random_state,
            )
            self.all_metrics.setdefault("temporal_md_space_animation_sampling", {}).update(
                {
                    "clustering_projection": md_animation_clustering_metrics,
                }
            )
            self.temporal_md_animation_layout_for_outputs = temporal_md_space_animation_layout
            self.temporal_md_animation_order_for_outputs = (
                None
                if temporal_md_space_animation_source_names is None
                else list(temporal_md_space_animation_source_names)
            )
            self.temporal_md_animation_coords_for_outputs = np.asarray(
                temporal_md_space_animation_cache["coords"],
                dtype=np.float32,
            )
            self.temporal_md_animation_frame_source = "dense_md_space_frames"
            self.temporal_md_animation_spatial_bounds_for_outputs = (
                temporal_md_space_animation_spatial_bounds
            )
            for cache_key in ("inv_latents", "eq_latents", "phases", "instance_ids"):
                temporal_md_space_animation_cache.pop(cache_key, None)

    def compute_diagnostics_and_clustering(self):
        self.all_metrics.update(
            run_pca_and_latent_stats(
                self.cache,
                self.out_dir,
                class_names=self.class_names,
                step=self.step,
                pca_max_samples=_positive_int_or_none(
                    OmegaConf.select(
                        self.analysis_cfg,
                        "pca.max_samples",
                        default=self.effective_tsne_max_samples,
                    )
                ),
                latent_stats_max_samples=_positive_int_or_none(
                    OmegaConf.select(
                        self.analysis_cfg,
                        "latent_stats.max_samples",
                        default=self.effective_tsne_max_samples,
                    )
                ),
                latent_stats_correlation_max_samples=_positive_int_or_none(
                    OmegaConf.select(
                        self.analysis_cfg,
                        "latent_stats.correlation_max_samples",
                        default=50000,
                    )
                ),
            )
        )

        self.step("Computing clustering labels")
        if self.fit_latents_for_clustering is None:
            clustering_metrics = dict(self.clustering_fit_metrics)
            self.configured_k_values = list(self.clustering_fit_configured_k_values)
            self.cluster_labels_by_k = dict(self.clustering_fit_labels_by_k)
            self.cluster_methods_by_k = dict(self.clustering_fit_methods_by_k)
        else:
            (
                clustering_metrics,
                self.configured_k_values,
                self.cluster_labels_by_k,
                self.cluster_methods_by_k,
            ) = predict_clustering_state_from_models(
                self.cache["inv_latents"],
                self.cache["phases"],
                fitted_models_by_k=self.clustering_models_by_k,
                requested_k_values=self.clustering_requested_k_values,
                cluster_method=self.analysis_settings.cluster_method,
                random_state=self.clustering_random_state,
            )
        self.all_metrics["clustering"] = clustering_metrics
        if self.temporal_md_space_animation_enabled and self.temporal_md_space_animation_reuse_main_cache:
            dense_snapshot_count = int(
                OmegaConf.select(
                    self.analysis_cfg,
                    "real_md.temporal.md_space.dense_snapshot_count",
                    default=0,
                )
            )
            frame_indices, source_names = resolve_temporal_real_snapshot_subset(
                analysis_cfg=self.analysis_cfg,
                selection=self.temporal_bundle.selection,
                snapshot_count=int(dense_snapshot_count),
            )
            self.temporal_md_animation_layout_for_outputs = filter_snapshot_figure_layout(
                self.snapshot_layout_inference,
                allowed_source_names=[str(v) for v in source_names],
            )
            self.temporal_md_animation_order_for_outputs = [str(v) for v in source_names]
            self.temporal_md_animation_coords_for_outputs = np.asarray(self.coords, dtype=np.float32)
            self.temporal_md_animation_cluster_labels_by_k_for_outputs = self.cluster_labels_by_k
            self.temporal_md_animation_frame_source = "main_temporal_inference_subset"
            self.all_metrics["temporal_md_space_animation_sampling"] = {
                "enabled": True,
                "reused_main_inference_cache": True,
                "dense_snapshot_count": int(dense_snapshot_count),
                "frame_indices": [int(v) for v in frame_indices.tolist()],
                "source_names": [str(v) for v in source_names],
                "sample_count": int(
                    sum(
                        int(np.asarray(indices, dtype=int).size)
                        for _source_name, indices in self.temporal_md_animation_layout_for_outputs.source_groups
                    )
                ),
                "sample_count_by_snapshot": {
                    str(source_name): int(np.asarray(indices, dtype=int).size)
                    for source_name, indices in self.temporal_md_animation_layout_for_outputs.source_groups
                },
                "clustering_projection": "reused_main_clustering",
            }
        clustering_comparison, self.comparison_labels_by_method = build_clustering_method_comparison(
            self.cache["inv_latents"],
            self.cache["phases"],
            fit_latents=self.fit_latents_for_clustering,
            requested_k_values=self.clustering_requested_k_values,
            primary_method=self.analysis_settings.cluster_method,
            compare_methods=self.analysis_settings.cluster_compare_methods,
            random_state=self.clustering_random_state,
            l2_normalize=self.analysis_settings.cluster_l2_normalize,
            standardize=self.analysis_settings.cluster_standardize,
            pca_variance=self.analysis_settings.cluster_pca_var,
            pca_max_components=self.analysis_settings.cluster_pca_max_components,
            prepared_features=self.clustering_features_for_fixed_k,
            prep_info=self.clustering_feature_prep_for_fixed_k,
        )
        if clustering_comparison is not None:
            self.all_metrics["clustering_comparison"] = clustering_comparison

        self.primary_k = int(self.analysis_settings.primary_k)
        if self.primary_k not in self.configured_k_values:
            raise KeyError(
                "Requested clustering.primary_k is not available in configured clustering results. "
                f"Requested k={self.primary_k}, available={self.configured_k_values}."
            )
        self.cluster_labels = self.cluster_labels_by_k[self.primary_k]
        self.figure_output_labels_by_k = (
            self.cluster_labels_by_k
            if self.snapshot_cluster_labels_by_k_for_outputs is None
            else self.snapshot_cluster_labels_by_k_for_outputs
        )
        self.figure_output_cluster_labels = self.figure_output_labels_by_k[int(self.primary_k)]

    def build_shared_colors(self):
        self.step("Building shared cluster color maps")
        self.shared_cluster_color_maps_by_k = {
            int(k_val_inner): build_shared_cluster_color_map(
                self.cluster_labels_by_k[int(k_val_inner)],
                cluster_color_assignment=self.figure_settings.cluster_color_assignment,
            )
            for k_val_inner in self.configured_k_values
        }
        self.shared_cluster_color_map = self.shared_cluster_color_maps_by_k[int(self.primary_k)]

        if self.gateway_phase_settings.enabled and self.temporal_bundle is None:
            raise ValueError(
                "gateway_phase.enabled=true requires inputs.temporal_real.enabled=true so "
                "stable LAMMPS atom identities and ordered anchor frames are available."
            )
        gateway_phase_metrics = run_gateway_phase_analysis(
            labels=np.asarray(self.cluster_labels, dtype=int),
            instance_ids=np.asarray(self.cache["instance_ids"]),
            coords=np.asarray(self.cache["coords"], dtype=np.float32),
            anchor_frame_indices=np.asarray(
                self.cache.get("anchor_frame_indices", np.empty((0,), dtype=np.int64))
            ),
            out_dir=self.out_dir,
            settings=self.gateway_phase_settings,
            inference_dataset=(
                None if self.temporal_bundle is None else self.temporal_bundle.inference_dataset
            ),
            temporal_dataset=(None if self.temporal_bundle is None else self.temporal_bundle.dataset),
            step=self.step,
        )
        if gateway_phase_metrics:
            self.all_metrics["gateway_phase"] = gateway_phase_metrics

        connected_representative_selection_features = None
        if self.connected_regime_settings.interactive_3d:
            connected_representative_selection_features, _ = (
                self._representative_selection_for(
                    np.asarray(self.cache["inv_latents"], dtype=np.float32),
                    self.cluster_labels_by_k,
                    int(self.primary_k),
                    source_name="connected_regime_cluster_gallery",
                )
            )
        connected_regime_metrics = run_connected_regime_analysis(
            latents=np.asarray(self.cache["inv_latents"], dtype=np.float32),
            labels=np.asarray(self.cluster_labels, dtype=int),
            out_dir=self.out_dir,
            settings=self.connected_regime_settings,
            cluster_color_map=self.shared_cluster_color_map,
            frame_groups=self.snapshot_layout_inference.source_groups,
            dataset=self.dataset_obj,
            representatives_out_dir=Path(self.out_dir) / "real_md" / "representatives",
            representative_point_scale=float(self.point_scale),
            representative_target_points=int(self.figure_settings.representative_points),
            representative_selection_features=connected_representative_selection_features,
            step=self.step,
        )
        if connected_regime_metrics:
            self.all_metrics["connected_regimes"] = connected_regime_metrics


        swav_prototype_metrics = run_swav_prototype_evaluation(
            model=self.model,
            cache=self.cache,
            out_dir=self.out_dir,
            analysis_cfg=self.analysis_cfg,
            cluster_labels_by_k=self.cluster_labels_by_k,
            cluster_color_maps_by_k=self.shared_cluster_color_maps_by_k,
            primary_k=int(self.primary_k),
            frame_groups=self.snapshot_source_groups,
            proportion_frame_groups=(
                None if self.temporal_bundle is None else self.snapshot_layout_inference.source_groups
            ),
            figure_settings=self.figure_settings,
            figure_set_run_kwargs=self._build_figure_set_run_kwargs(self.figure_settings),
            figure_dataloader=self.dl,
            figure_snapshot_layout=self.figure_snapshot_layout_for_outputs,
            figure_analysis_source_names=self.figure_analysis_source_names_for_outputs,
            step=self.step,
        )
        if swav_prototype_metrics:
            self.all_metrics["swav_prototypes"] = swav_prototype_metrics

    def render_cluster_figures(self):
        self.real_md_enabled = bool(OmegaConf.select(self.analysis_cfg, "real_md.enabled", default=True))
        self.real_md_profiles_enabled = bool(
            OmegaConf.select(self.analysis_cfg, "real_md.profiles.enabled", default=True)
        )
        self.shared_representative_render_cache: dict[str, Any] | None = None
        if (
            not self.is_synthetic
            and self.figure_settings.enabled
            and not self.multi_snapshot_real
            and self.real_md_enabled
            and self.real_md_profiles_enabled
            and self.dataset_obj is not None
            and int(self.real_md_selected_k) == int(self.figure_settings.k)
            and int(self.figure_settings.real_md_profile_target_points)
            == int(self.figure_settings.representative_points)
        ):
            self.step("Preparing shared representative structures")
            representative_selection_features, representative_selection_info = (
                self._representative_selection_for(
                    self.cache["inv_latents"],
                    self.cluster_labels_by_k,
                    int(self.figure_settings.k),
                    source_name="main_inference_cache",
                )
            )
            self.shared_representative_render_cache = _build_cluster_representative_render_cache(
                self.dataset_obj,
                np.asarray(self.cache["inv_latents"], dtype=np.float32),
                np.asarray(self.cluster_labels_by_k[int(self.figure_settings.k)], dtype=int),
                build_shared_cluster_color_map(
                    self.cluster_labels_by_k[int(self.figure_settings.k)],
                    cluster_color_assignment=self.figure_settings.cluster_color_assignment,
                ),
                point_scale=float(self.point_scale),
                target_points=int(self.figure_settings.representative_points),
                representative_ptm_enabled=bool(self.figure_settings.representative_ptm_enabled),
                representative_cna_enabled=bool(self.figure_settings.representative_cna_enabled),
                representative_cna_max_signatures=int(
                    self.figure_settings.representative_cna_max_signatures
                ),
                representative_center_atom_tolerance=float(
                    self.figure_settings.representative_center_atom_tolerance
                ),
                representative_shell_min_neighbors=int(
                    self.figure_settings.representative_shell_min_neighbors
                ),
                representative_shell_max_neighbors=int(
                    self.figure_settings.representative_shell_max_neighbors
                ),
                selection_features=representative_selection_features,
                selection_info=representative_selection_info,
            )

        cluster_figure_sets_by_k: dict[str, Any] = {}
        primary_cluster_figure_set = None
        primary_snapshot_figure_sets = None
        if self.figure_settings.enabled:
            for k_value in self.configured_k_values:
                figure_settings_for_k = replace(
                    self.figure_settings,
                    k=int(k_value),
                )
                representative_render_cache_for_k = (
                    self.shared_representative_render_cache
                    if int(k_value) == int(self.primary_k)
                    else None
                )
                representative_selection_features_for_k = None
                representative_selection_info_for_k = None
                if representative_render_cache_for_k is None:
                    (
                        representative_selection_features_for_k,
                        representative_selection_info_for_k,
                    ) = self._representative_selection_for(
                        self.snapshot_latents_for_outputs,
                        self.figure_output_labels_by_k,
                        int(k_value),
                        source_name="figure_output_cache",
                    )
                cluster_figure_set, snapshot_figure_sets = render_cluster_figure_outputs(
                    out_dir=self.out_dir,
                    dataloader=self.dl,
                    figure_settings=figure_settings_for_k,
                    figure_set_run_kwargs=self._build_figure_set_run_kwargs(figure_settings_for_k),
                    labels_for_k=self.figure_output_labels_by_k[int(k_value)],
                    latents=self.snapshot_latents_for_outputs,
                    coords=self.snapshot_coords_for_outputs,
                    dataset_obj=self.snapshot_dataset_obj_for_outputs,
                    snapshot_layout=self.figure_snapshot_layout_for_outputs,
                    analysis_source_names=self.figure_analysis_source_names_for_outputs,
                    step=self.step,
                    representative_render_cache=representative_render_cache_for_k,
                    representative_selection_features=representative_selection_features_for_k,
                    representative_selection_info=representative_selection_info_for_k,
                )
                cluster_figure_sets_by_k[str(int(k_value))] = {
                    "cluster_figure_set": cluster_figure_set,
                    "cluster_figure_sets_by_snapshot": snapshot_figure_sets,
                }
                if int(k_value) == int(self.primary_k):
                    primary_cluster_figure_set = cluster_figure_set
                    primary_snapshot_figure_sets = snapshot_figure_sets
        if primary_cluster_figure_set is not None:
            self.all_metrics["cluster_figure_set"] = primary_cluster_figure_set
        if primary_snapshot_figure_sets is not None:
            self.all_metrics["cluster_figure_sets_by_snapshot"] = primary_snapshot_figure_sets
        if len(cluster_figure_sets_by_k) > 1:
            self.all_metrics["cluster_figure_sets_by_k"] = cluster_figure_sets_by_k

        tsne_metrics = run_tsne_visualizations(
            self.cache,
            self.out_dir,
            analysis_cfg=self.analysis_cfg,
            cluster_labels_by_k=self.cluster_labels_by_k,
            cluster_methods_by_k=self.cluster_methods_by_k,
            comparison_labels_by_method=self.comparison_labels_by_method,
            configured_k_values=self.configured_k_values,
            primary_k=self.primary_k,
            shared_cluster_color_maps_by_k=self.shared_cluster_color_maps_by_k,
            class_names=self.class_names,
            is_synthetic=self.is_synthetic,
            clustering_random_state=self.clustering_random_state,
            tsne_max_samples=self.effective_tsne_max_samples,
            tsne_n_iter=self.analysis_settings.tsne_n_iter,
            cluster_method=self.analysis_settings.cluster_method,
            step=self.step,
        )
        if tsne_metrics:
            self.all_metrics["latent_projection_visualizations"] = tsne_metrics

    def write_md_outputs(self):
        self.step("Saving coordinate-space clustering outputs")
        interactive_max_points = self.analysis_settings.interactive_max_points
        if self.analysis_settings.md_use_all_points:
            interactive_max_points = None
        hdbscan_result = _run_optional_hdbscan_analysis(
            self.cache["inv_latents"],
            coords_count=len(self.coords),
            settings=self.hdbscan_settings,
            random_state=self.clustering_random_state,
            l2_normalize=self.analysis_settings.cluster_l2_normalize,
            standardize=self.analysis_settings.cluster_standardize,
            pca_variance=self.analysis_settings.cluster_pca_var,
            pca_max_components=self.analysis_settings.cluster_pca_max_components,
            prepared_features=self.clustering_features,
            prep_info=self.clustering_feature_prep,
            cluster_color_assignment=self.figure_settings.cluster_color_assignment,
            step=self.step,
        )
        self.all_metrics[self.md_metrics_key] = build_md_metrics(
            hdbscan_result=hdbscan_result,
            out_dir=self.out_dir,
            coords=self.snapshot_coords_for_outputs,
            cluster_labels=self.figure_output_cluster_labels,
            cluster_labels_by_k=self.figure_output_labels_by_k,
            configured_k_values=self.configured_k_values,
            primary_k=self.primary_k,
            shared_cluster_color_maps_by_k=self.shared_cluster_color_maps_by_k,
            interactive_max_points=interactive_max_points,
            multi_snapshot_real=self.snapshot_layout_for_outputs.multi_snapshot_real,
            snapshot_source_groups=self.snapshot_layout_for_outputs.source_groups,
            snapshot_output_names=self.snapshot_layout_for_outputs.output_names,
        )

    def analyze_real_md(self):
        self.primary_real_md_summary = None
        if not self.is_synthetic and self.real_md_enabled:
            self.step("Running real-MD qualitative analysis")
            temporal_projection_fit_indices = (
                None
                if self.temporal_bundle is None
                else np.asarray(
                    _sample_indices(
                        self.n_samples,
                        self.effective_tsne_max_samples,
                    ),
                    dtype=int,
                )
            )
            real_md_analysis = RealMDAnalysis(
                out_dir=self.out_dir,
                model_cfg=self.cfg,
                analysis_cfg=self.analysis_cfg,
                dataset=self.dataset_obj,
                latents=self.cache['inv_latents'],
                coords=self.coords,
                instance_ids=np.asarray(self.cache['instance_ids']),
                cluster_labels_by_k=self.cluster_labels_by_k,
                cluster_methods_by_k=self.cluster_methods_by_k,
                frame_groups=self.snapshot_source_groups,
                frame_output_names=self.snapshot_output_names,
                requested_frame_order=self.analysis_source_names,
                temporal_all_frame_groups=None if self.temporal_bundle is None else self.snapshot_layout_inference.source_groups,
                temporal_all_frame_output_names=None if self.temporal_bundle is None else self.snapshot_layout_inference.output_names,
                temporal_all_frame_order=None if self.temporal_bundle is None else self.temporal_bundle.selection.inference_source_names,
                temporal_md_animation_frame_groups=None if self.temporal_md_animation_layout_for_outputs is None else self.temporal_md_animation_layout_for_outputs.source_groups,
                temporal_md_animation_frame_output_names=None if self.temporal_md_animation_layout_for_outputs is None else self.temporal_md_animation_layout_for_outputs.output_names,
                temporal_md_animation_order=self.temporal_md_animation_order_for_outputs,
                temporal_md_animation_coords=self.temporal_md_animation_coords_for_outputs,
                temporal_md_animation_cluster_labels_by_k=self.temporal_md_animation_cluster_labels_by_k_for_outputs,
                temporal_md_animation_frame_source=self.temporal_md_animation_frame_source,
                temporal_md_animation_spatial_bounds=self.temporal_md_animation_spatial_bounds_for_outputs,
                temporal_projection_fit_indices=temporal_projection_fit_indices,
                point_scale=float(self.point_scale),
                random_state=int(self.clustering_random_state),
            )
            real_md_summaries_by_k: dict[str, Any] = {}
            self.primary_real_md_summary = None
            for k_value in self.configured_k_values:
                shared_real_md_color_map = self.shared_cluster_color_maps_by_k.get(
                    int(k_value),
                    self.shared_cluster_color_map,
                )
                real_md_output_root = (
                    real_md_outputs_root(self.out_dir)
                    if int(k_value) == int(self.primary_k)
                    else real_md_outputs_root_for_k(self.out_dir, k_value=int(k_value))
                )
                representative_render_cache_for_k = (
                    self.shared_representative_render_cache
                    if int(k_value) == int(self.primary_k)
                    else None
                )
                real_md_representative_selection_features = None
                real_md_representative_selection_info = None
                if self.real_md_profiles_enabled and representative_render_cache_for_k is None:
                    (
                        real_md_representative_selection_features,
                        real_md_representative_selection_info,
                    ) = self._representative_selection_for(
                        self.cache["inv_latents"],
                        self.cluster_labels_by_k,
                        int(k_value),
                        source_name="main_inference_cache",
                    )
                flicker_metrics_enabled = bool(
                    OmegaConf.select(
                        self.analysis_cfg,
                        "real_md.temporal.flicker.enabled",
                        default=True,
                    )
                )
                real_md_assignment_margins = None
                instance_ids_for_real_md = np.asarray(self.cache["instance_ids"])
                if (
                    flicker_metrics_enabled
                    and instance_ids_for_real_md.reshape(-1).shape[0]
                    == np.asarray(self.cache["inv_latents"]).shape[0]
                ):
                    real_md_assignment_margins = self._cluster_assignment_margins_for_k(int(k_value))
                real_md_summary = real_md_analysis.run(
                    cluster_color_map=shared_real_md_color_map,
                    representative_render_cache=representative_render_cache_for_k,
                    representative_selection_features=real_md_representative_selection_features,
                    representative_selection_info=real_md_representative_selection_info,
                    selected_k=int(k_value),
                    output_root_dir=real_md_output_root,
                    assignment_margins=real_md_assignment_margins,
                )
                real_md_summaries_by_k[str(int(k_value))] = real_md_summary
                if int(k_value) == int(self.primary_k):
                    self.primary_real_md_summary = real_md_summary
            if self.primary_real_md_summary is not None:
                self.all_metrics["real_md_qualitative"] = self.primary_real_md_summary
            if len(real_md_summaries_by_k) > 1:
                self.all_metrics["real_md_qualitative_by_k"] = real_md_summaries_by_k

    def analyze_dynamic_motifs_and_equivariance(self):
        dynamic_metrics = run_dynamic_motif_analysis(
            cache=self.cache,
            out_dir=self.out_dir,
            analysis_cfg=self.analysis_settings,
            cluster_labels_primary=self.cluster_labels,
            step=self.step,
        )
        if dynamic_metrics:
            self.all_metrics["dynamic_motif"] = dynamic_metrics
            dynamic_summary_rel = dynamic_metrics.get("artifacts", {}).get("summary_markdown")
            if (
                self.primary_real_md_summary is not None
                and isinstance(self.primary_real_md_summary, dict)
                and dynamic_summary_rel is not None
                and "summary_markdown" in self.primary_real_md_summary
            ):
                append_dynamic_motif_summary(
                    Path(self.primary_real_md_summary["summary_markdown"]),
                    dynamic_summary_path=self.out_dir / dynamic_summary_rel,
                    dynamic_metrics=dynamic_metrics,
                    out_dir=self.out_dir,
                )

        if self.runtime_profile.equivariance_enabled:
            self.all_metrics.update(
                run_equivariance_evaluation(
                    self.model,
                    self.dl,
                    self.device,
                    self.out_dir,
                    analysis_cfg=self.analysis_cfg,
                    step=self.step,
                    temporal_sequence_mode=(
                        "static_anchor"
                        if self.temporal_bundle is None
                        else self.temporal_bundle.collection_inference_spec.mode
                    ),
                    temporal_static_frame_index=(
                        0
                        if self.temporal_bundle is None
                        else self.temporal_bundle.collection_inference_spec.static_frame_index
                    ),
                )
            )
        else:
            self.all_metrics["runtime_profile"]["equivariance_skipped"] = True

    def publish(self):
        self.step("Writing metrics")
        metrics_path = self.out_dir / "analysis_metrics.json"
        retain_cache = bool(OmegaConf.select(self.analysis_cfg, 'cache.retain_after_analysis', default=True))
        self.all_metrics['inference_cache']['retained_after_analysis'] = retain_cache
        write_json(metrics_path, self.all_metrics)
        if 'topology' in self.all_metrics:
            write_json(Path(self.run_settings.checkpoint_path).parent/'final_metrics.json', self.all_metrics['topology']['flat_metrics'])
        report_dir = report_directory(self.cfg, self.analysis_cfg)
        publish_report(self.out_dir, self.result_root, update_index=report_dir is None, computed=True, analysis_cfg=self.analysis_cfg)
        if report_dir is not None:
            publish_report(self.out_dir, report_dir)
        if not retain_cache:
            from .inference_cache import discard_inference_cache
            discard_inference_cache(self.out_dir, self.analysis_settings.inference_cache_file)
            for path in (self.out_dir/'topology').glob('*_inference.npz'):
                discard_inference_cache(path.parent, path.name)

        elapsed = time.perf_counter() - self.started
        print_analysis_summary(
            self.all_metrics,
            n_samples=self.n_samples,
            out_dir=self.out_dir,
            elapsed=elapsed,
        )
        print_figure_set_summary(
            self.all_metrics,
            n_samples=self.n_samples,
            out_dir=self.out_dir,
            elapsed=elapsed,
        )

        return self.all_metrics

    def _representative_selection_for(
        self,
        latents_for_selection: np.ndarray,
        labels_by_k: dict[int, np.ndarray],
        k_value: int,
        *,
        source_name: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        cache_key = (str(source_name), int(k_value))
        cached = self.representative_selection_cache.get(cache_key)
        if cached is not None:
            return cached
        features, info = representative_features_from_clustering_model(
            latents_for_selection,
            fitted_model=self.clustering_models_by_k[int(k_value)],
            expected_labels=labels_by_k[int(k_value)],
        )
        info = {
            **dict(info),
            "source": str(source_name),
            "k": int(k_value),
        }
        self.representative_selection_cache[cache_key] = (features, info)
        return features, info

    def _build_figure_set_run_kwargs(self, figure_settings_for_k: Any) -> dict[str, Any]:
        return figure_settings_for_k.build_run_kwargs(
            dataset=self.snapshot_dataset_obj_for_outputs,
            latents=self.snapshot_latents_for_outputs,
            coords=self.snapshot_coords_for_outputs,
            point_scale=self.point_scale,
            random_state=self.clustering_random_state,
            l2_normalize=self.analysis_settings.cluster_l2_normalize,
            standardize=self.analysis_settings.cluster_standardize,
            pca_variance=self.analysis_settings.cluster_pca_var,
            pca_max_components=self.analysis_settings.cluster_pca_max_components,
        )

    def _cluster_assignment_margins_for_k(self, k_value: int) -> dict[str, Any]:
        k_int = int(k_value)
        margin_chunk_size = int(
            OmegaConf.select(
                self.analysis_cfg,
                "real_md.temporal.flicker.margin_chunk_size",
                default=200_000,
            )
        )
        self.step(f"Computing cluster assignment margins for k={k_int}")
        margins = compute_clustering_assignment_margins(
            np.asarray(self.cache["inv_latents"], dtype=np.float32),
            fitted_model=self.clustering_models_by_k[k_int],
            expected_labels=np.asarray(self.cluster_labels_by_k[k_int], dtype=int),
            chunk_size=int(margin_chunk_size),
        )
        return margins



def main() -> None:
    parser = argparse.ArgumentParser(description="Post-training analysis pipeline")
    parser.add_argument(
        "config",
        nargs="?",
        default=str(DEFAULT_ANALYSIS_CONFIG_PATH),
        help=f"Path to the analysis config YAML (default: {DEFAULT_ANALYSIS_CONFIG_PATH})",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--batch', type=Path, help='YAML with explicit checkpoint/config/output entries.')
    modes.add_argument('--collect-root', type=Path, help='Collect standard pipeline topology results across seeds.')
    parser.add_argument('--specification', type=Path, help='Declared variants, seeds and paired comparisons for collection.')
    parser.add_argument('--checkpoint', help='Override checkpoint.path for a single analysis.')
    parser.add_argument('--output-dir', help='Override checkpoint.output_dir for a single analysis.')
    parser.add_argument('--publish-only', action='store_true', help='Publish existing completed batch results to flat galleries.')
    args = parser.parse_args()
    if args.publish_only and args.batch is None:
        parser.error('--publish-only requires --batch')
    if args.batch is not None:
        batch = OmegaConf.load(args.batch)
        for item in batch.runs:
            from src.experiment_runner.artifacts import analysis_artifacts
            output = analysis_artifacts(item.output_dir)
            completed = output/'analysis_metrics.json'
            if args.publish_only:
                settings = load_checkpoint_analysis_config(item.analysis_config)
                cfg = build_runtime_model_config(item.checkpoint, settings)
                report_dir = report_directory(cfg, settings)
                if report_dir is None:
                    raise ValueError(f'--publish-only requires report.root in {item.analysis_config}')
                publish_report(output, report_dir)
                continue
            if completed.exists():
                import json
                from src.experiment_runner.registry import sha256
                saved = json.loads(completed.read_text())
                if saved['topology']['checkpoint_sha256'] != sha256(Path(item.checkpoint)):
                    raise ValueError(f'Completed analysis belongs to a different checkpoint: {completed}')
                settings = load_checkpoint_analysis_config(item.analysis_config)
                if not bool(OmegaConf.select(settings, 'cache.retain_after_analysis', default=True)):
                    from .inference_cache import discard_inference_cache
                    discard_inference_cache(output, settings.cache.file)
                    for path in (output/'topology').glob('*_inference.npz'):
                        discard_inference_cache(path.parent, path.name)
                    saved['inference_cache']['retained_after_analysis'] = False
                    write_json(completed, saved)
                cfg = build_runtime_model_config(item.checkpoint, settings)
                report_dir = report_directory(cfg, settings)
                if report_dir is not None:
                    publish_report(output, report_dir)
                print(f'[analysis][batch] Verified completed result: {completed}', flush=True)
                continue
            run_post_training_analysis(checkpoint_path=item.checkpoint, output_dir=item.output_dir,
                analysis_config_path=item.analysis_config, cuda_device=batch.cuda_device)
        return
    if args.collect_root is not None:
        if args.specification is None:
            parser.error('--collect-root requires --specification')
        from .topology import collect
        collect(args.collect_root, args.specification)
        return
    run_post_training_analysis(analysis_config_path=args.config,
                              checkpoint_path=args.checkpoint, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
