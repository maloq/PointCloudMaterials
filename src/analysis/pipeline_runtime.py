from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import torch
from omegaconf import DictConfig, open_dict

from src.data_utils.data_module import (
    StaticPointCloudDataModule,
    SyntheticPointCloudDataModule,
    TemporalLAMMPSDataModule,
    _resolve_temporal_window_start_frames,
)
from src.data_utils.data_kinds import is_static_data_kind, normalize_data_kind
from src.data_utils.temporal_lammps_dataset import TemporalLAMMPSDumpDataset
from src.training_methods.contrastive_learning.vicreg_module import VICRegModule
from src.utils.model_utils import load_model_from_checkpoint

from .analysis_dataloaders import _analysis_dataloader_kwargs
from .config import (
    _positive_int_or_none,
    _resolve_analysis_files,
    build_runtime_model_config,
    load_checkpoint_training_config,
)
from .dynamic_motif_cache import collect_tmf_inference_cache
from .inference_cache import (
    _build_inference_cache_spec,
    _inference_cache_paths,
    _load_inference_cache,
    _save_inference_cache,
    _validate_inference_cache_arrays,
)
from .utils import _unwrap_dataset, build_static_coords_dataloader, gather_inference_batches


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
    if normalize_data_kind(getattr(cfg.data, "kind", None)) == "temporal_lammps":
        data_cfg = cfg.data
        dump_file = getattr(data_cfg, "dump_file", None)
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
        combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        return torch.utils.data.DataLoader(
            combined_dataset,
            **_analysis_dataloader_kwargs(
                batch_size=int(inference_batch_size),
                dataloader_num_workers=int(dataloader_num_workers),
            ),
        )

    dl = build_static_coords_dataloader(
        cfg,
        dm,
        use_train_data=True,
        use_full_dataset=True,
        prefer_existing_full_dataset=True,
        batch_size=int(inference_batch_size),
    )
    print(
        "Static data detected: using full dataset for local-structure clustering visualization"
    )
    return dl


def _resolve_analysis_inference_batch_size(
    cfg: DictConfig,
    input_settings: Any,
) -> int:
    batch_size = input_settings.inference_batch_size
    if batch_size is None:
        batch_size = int(cfg.batch_size)
    return int(batch_size)


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
    fit_kind = normalize_data_kind(getattr(fit_cfg.data, "kind", None))

    fit_source_names: list[str] | None = None
    fit_analysis_files = None
    if fit_kind == "static":
        fit_analysis_files = _resolve_analysis_files(
            fit_cfg,
            fit_settings.input_settings,
        )
        fit_source_names = _configure_static_analysis_inputs(fit_cfg, fit_analysis_files)
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
        require_coords_for_static=not fit_is_synthetic,
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


def _collect_main_inference_cache(
    *,
    out_dir: Path,
    cfg: DictConfig,
    checkpoint_path: str,
    cuda_device: int,
    dataloader: Any,
    model: Any | None,
    device: str,
    analysis_settings: Any,
    figure_only: bool,
    cache_spec: dict[str, Any],
    max_batches_latent: int | None,
    max_samples_total: int | None,
    seed_base: int,
    temporal_bundle: Any | None,
    step: Callable[[str], None],
) -> tuple[dict[str, np.ndarray], Any | None, DictConfig, str, bool]:
    cache: dict[str, np.ndarray] | None = None
    cache_loaded = False
    if figure_only:
        step("Loading cached inference batches")
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
        step("Loading model")
        model, cfg, device = load_vicreg_model(
            checkpoint_path,
            cuda_device=int(cuda_device),
            cfg=cfg,
        )
        step("Collecting inference batches")
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

    if cache is None and not figure_only:
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
                dataloader,
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
                dataloader,
                device,
                max_batches=max_batches_latent,
                max_samples_total=max_samples_total,
                collect_coords=True,
                seed_base=seed_base,
                progress_every_batches=analysis_settings.progress_every_batches,
                verbose=True,
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
        print("[analysis][cache] Validating freshly collected inference cache...")
        _validate_inference_cache_arrays(cache)
        if analysis_settings.inference_cache_enabled:
            print("[analysis][cache] Writing inference cache...")
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
    return cache, model, cfg, device, bool(cache_loaded)


def _configure_static_analysis_inputs(
    cfg: DictConfig,
    analysis_files: list[str],
) -> list[str]:
    if not is_static_data_kind(getattr(cfg.data, "kind", None)):
        raise ValueError(
            "_configure_static_analysis_inputs can only be used for static datasets, "
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


def load_vicreg_model(
    checkpoint_path: str, cuda_device: int = 0, cfg: DictConfig | None = None
) -> tuple[Any, DictConfig, str]:
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
    if model_type in {"vicreg", "visreg"}:
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
        f"Expected one of ['vicreg', 'visreg', 'temporal_vicreg', 'temporal_lejepa', 'temporal_motif_field'], "
        f"got {model_type!r}."
    )


def build_datamodule(
    cfg: DictConfig,
    *,
    require_coords_for_static: bool = False,
    require_coords_for_real: bool | None = None,
):
    """Instantiate the matching datamodule."""
    if getattr(cfg, "data", None) is None:
        raise ValueError("Config missing data section")
    if require_coords_for_real is not None:
        require_coords_for_static = bool(require_coords_for_real)
    data_kind = normalize_data_kind(getattr(cfg.data, "kind", None))
    if data_kind == "synthetic":
        dm = SyntheticPointCloudDataModule(cfg)
    elif data_kind == "temporal_lammps":
        dm = TemporalLAMMPSDataModule(cfg)
    elif data_kind == "static":
        dm = StaticPointCloudDataModule(
            cfg,
            return_coords=bool(require_coords_for_static),
        )
    else:
        raise ValueError(
            "Unsupported data.kind. Expected one of "
            "['static', 'synthetic', 'temporal_lammps'] "
            f"('real' is accepted as a legacy alias for 'static'), got {getattr(cfg.data, 'kind', None)!r}."
        )
    return dm
