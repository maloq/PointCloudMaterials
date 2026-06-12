from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.data_utils.temporal_lammps_dataset import TemporalLAMMPSDumpDataset

from .analysis_dataloaders import (
    _BatchedConcatDataset,
    _analysis_dataloader_kwargs,
    _analysis_prefetch_factor,
)
from .config import _positive_int_or_none, _validate_overlap_fraction
from .figure_sets import resolve_snapshot_figure_layout
from .inference_cache import (
    _build_inference_cache_spec,
    _inference_cache_paths,
    _load_inference_cache,
    _save_inference_cache,
    _validate_inference_cache_arrays,
)
from .temporal_real import (
    _temporal_identity_or_default_collate,
    build_temporal_real_single_snapshot_bundle,
    resolve_temporal_real_snapshot_subset,
)
from .utils import _unwrap_dataset, gather_inference_batches


_DIAGONAL_CUT_DIRECTION = np.asarray([1.0, 1.0, 1.0], dtype=np.float32) / np.sqrt(3.0)


def _select_dense_snapshot_sample_indices(
    *,
    sample_count: int,
    max_samples: int | None,
    seed: int,
) -> np.ndarray:
    resolved_sample_count = int(sample_count)
    if max_samples is None or int(max_samples) >= resolved_sample_count:
        return np.arange(resolved_sample_count, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    selected = rng.choice(
        resolved_sample_count,
        size=int(max_samples),
        replace=False,
    )
    return np.sort(selected.astype(np.int64, copy=False))


def _normalize_dense_snapshot_sample_selection_mode(raw_value: Any) -> str:
    mode = str(raw_value).strip().lower().replace("-", "_")
    if mode in {"random", "uniform_random", "uniform_random_without_replacement"}:
        return "uniform_random_without_replacement"
    return "diagonal_cut_surface"


def _resolve_equalized_bounds_from_min_max(
    mins: np.ndarray,
    maxs: np.ndarray,
) -> np.ndarray:
    mins_arr = np.asarray(mins, dtype=np.float32).reshape(3)
    maxs_arr = np.asarray(maxs, dtype=np.float32).reshape(3)
    lower = np.minimum(mins_arr, maxs_arr)
    upper = np.maximum(mins_arr, maxs_arr)
    center = 0.5 * (lower + upper)
    span = float(np.max(upper - lower))
    span = max(span, 1e-6)
    half = 0.5 * span
    return np.stack([center - half, center + half], axis=0).astype(np.float32, copy=False)


def _resolve_temporal_sampling_frame_slot(
    *,
    dataset: TemporalLAMMPSDumpDataset,
    temporal_inference_spec: Any,
) -> int:
    sequence_length = int(dataset.sequence_length)
    mode = str(temporal_inference_spec.mode).strip().lower()
    frame_slot = 0
    if mode == "static_anchor" and temporal_inference_spec.static_frame_index is not None:
        frame_slot = int(temporal_inference_spec.static_frame_index)
    if frame_slot < 0:
        frame_slot += int(sequence_length)
    return int(frame_slot)


def _temporal_dense_snapshot_center_coords_for_sampling(
    dataset: Any,
    *,
    temporal_inference_spec: Any,
    source_name: str,
) -> np.ndarray:
    base_dataset = _unwrap_dataset(dataset)
    window_start_frames = np.asarray(base_dataset.window_start_frames, dtype=np.int64)
    frame_slot = _resolve_temporal_sampling_frame_slot(
        dataset=base_dataset,
        temporal_inference_spec=temporal_inference_spec,
    )
    frame_index = int(window_start_frames[0]) + frame_slot * int(base_dataset.frame_stride)
    center_atom_indices = np.asarray(base_dataset.center_atom_indices, dtype=np.int64)
    frame_points = np.asarray(base_dataset.positions[frame_index], dtype=np.float32)
    box_low = np.asarray(base_dataset.box_low[frame_index], dtype=np.float32).reshape(3)
    coords = frame_points[center_atom_indices, :3] + box_low[None, :]
    coords = np.asarray(coords, dtype=np.float32)
    return coords


def _resolve_dense_snapshot_cut_surface_bounds(
    *,
    snapshot_bundles: list[Any],
    temporal_inference_spec: Any,
) -> np.ndarray:
    mins: np.ndarray | None = None
    maxs: np.ndarray | None = None
    for bundle in snapshot_bundles:
        source_name = str(bundle.selection.analysis_source_names[0])
        coords = _temporal_dense_snapshot_center_coords_for_sampling(
            bundle.dataset,
            temporal_inference_spec=temporal_inference_spec,
            source_name=source_name,
        )
        local_mins = np.min(coords[:, :3], axis=0)
        local_maxs = np.max(coords[:, :3], axis=0)
        mins = local_mins if mins is None else np.minimum(mins, local_mins)
        maxs = local_maxs if maxs is None else np.maximum(maxs, local_maxs)
    return _resolve_equalized_bounds_from_min_max(mins, maxs)


def _select_dense_snapshot_cut_surface_indices(
    *,
    coords: np.ndarray,
    max_samples: int | None,
    bounds: np.ndarray,
    visible_depth_fraction: float,
) -> np.ndarray:
    coords_arr = np.asarray(coords, dtype=np.float32)
    if coords_arr.ndim != 2 or coords_arr.shape[1] < 3:
        raise ValueError(
            "Cut-surface sample selection expects coords with shape (N, >=3), "
            f"got {coords_arr.shape}."
        )
    sample_count = int(coords_arr.shape[0])
    if sample_count <= 0:
        raise ValueError("Cut-surface sample selection received zero coordinates.")
    if max_samples is None or int(max_samples) >= sample_count:
        return np.arange(sample_count, dtype=np.int64)
    max_samples_int = int(max_samples)
    if max_samples_int <= 0:
        raise ValueError(f"max_samples must be > 0 when provided, got {max_samples}.")
    depth = float(visible_depth_fraction)
    if not np.isfinite(depth) or depth <= 0.0:
        raise ValueError(
            "Cut-surface visible depth fraction must be finite and positive, "
            f"got {visible_depth_fraction!r}."
        )
    bounds_arr = np.asarray(bounds, dtype=np.float32)
    if bounds_arr.shape != (2, 3):
        raise ValueError(
            f"Cut-surface bounds must have shape (2, 3), got {bounds_arr.shape}."
        )
    center = 0.5 * (bounds_arr[0] + bounds_arr[1])
    span = np.maximum(bounds_arr[1] - bounds_arr[0], 1e-6)
    normalized = (coords_arr[:, :3] - center[None, :]) / span[None, :]
    signed = normalized @ _DIAGONAL_CUT_DIRECTION
    distance = np.abs(signed)
    visible_side = signed <= 0.0
    near_cut = visible_side & (signed >= -depth)

    selected_parts: list[np.ndarray] = []
    selected_mask = np.zeros(sample_count, dtype=bool)

    def _append_closest(mask: np.ndarray) -> None:
        remaining = max_samples_int - int(np.count_nonzero(selected_mask))
        if remaining <= 0:
            return
        candidates = np.flatnonzero(mask & ~selected_mask)
        if candidates.size == 0:
            return
        if candidates.size > remaining:
            order = np.lexsort((candidates, distance[candidates]))
            candidates = candidates[order[:remaining]]
        selected_mask[candidates] = True
        selected_parts.append(candidates.astype(np.int64, copy=False))

    _append_closest(near_cut)
    _append_closest(visible_side)
    _append_closest(np.ones(sample_count, dtype=bool))
    if not selected_parts:
        raise RuntimeError("Cut-surface sample selection did not select any points.")
    selected = np.concatenate(selected_parts)
    return np.sort(selected.astype(np.int64, copy=False))


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
        return float(regular_grid_overlap), float(regular_grid_overlap - 1.0)

    static_overlap_fraction = _validate_overlap_fraction(
        OmegaConf.select(
            snapshot_cfg,
            "static_overlap_fraction",
            default=0.5,
        )
    )
    return float(1.0 + static_overlap_fraction), float(static_overlap_fraction)


def _build_temporal_dense_snapshot_plan(
    *,
    analysis_cfg: DictConfig,
    temporal_selection: Any,
    temporal_inference_spec: Any,
    checkpoint_path: str,
    out_dir: Path,
    model_cfg_for_module: DictConfig,
    seed_base: int,
    inference_batch_size: int,
    dataloader_num_workers: int,
    frame_indices: list[int],
    source_names: list[str],
    cache_enabled: bool,
    cache_force_recompute: bool,
    cache_file: str,
    collector_mode: str,
    summary_label: str,
    cache_config_path: str,
    max_samples_per_snapshot: int | None = None,
    sample_selection_mode: str = "uniform_random_without_replacement",
    cut_surface_visible_depth_fraction: float | None = None,
) -> dict[str, Any]:
    if cache_file == "":
        raise ValueError(
            f"{cache_config_path}.file must be a non-empty file name."
        )
    resolved_sample_selection_mode = _normalize_dense_snapshot_sample_selection_mode(
        sample_selection_mode
    )
    resolved_cut_surface_visible_depth_fraction = (
        None
        if cut_surface_visible_depth_fraction is None
        else float(cut_surface_visible_depth_fraction)
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
            model_cfg=model_cfg_for_module,
            selection=temporal_selection,
            batch_size=int(inference_batch_size),
            dataloader_num_workers=int(dataloader_num_workers),
            frame_index=int(frame_index),
            source_name=str(source_name),
            center_grid_overlap=float(regular_grid_overlap),
            temporal_inference_spec=temporal_inference_spec,
        )
        for frame_index, source_name in zip(
            resolved_frame_indices,
            resolved_source_names,
            strict=True,
        )
    ]

    cut_surface_bounds: np.ndarray | None = None
    if (
        max_samples_per_snapshot is not None
        and resolved_sample_selection_mode == "diagonal_cut_surface"
    ):
        if resolved_cut_surface_visible_depth_fraction is None:
            resolved_cut_surface_visible_depth_fraction = 0.10
        if (
            not np.isfinite(float(resolved_cut_surface_visible_depth_fraction))
            or float(resolved_cut_surface_visible_depth_fraction) <= 0.0
        ):
            raise ValueError(
                "Cut-surface sampling requires a positive finite "
                f"visible_depth_fraction, got {cut_surface_visible_depth_fraction!r}."
            )
        cut_surface_bounds = _resolve_dense_snapshot_cut_surface_bounds(
            snapshot_bundles=snapshot_bundles,
            temporal_inference_spec=temporal_inference_spec,
        )

    selected_indices_by_snapshot: list[np.ndarray] = []
    full_sample_counts: dict[str, int] = {}
    selected_sample_counts: dict[str, int] = {}
    for snapshot_idx, bundle in enumerate(snapshot_bundles):
        source_name = str(bundle.selection.analysis_source_names[0])
        full_sample_count = int(len(bundle.inference_dataset))
        if resolved_sample_selection_mode == "diagonal_cut_surface":
            if cut_surface_bounds is None:
                selected_indices = _select_dense_snapshot_sample_indices(
                    sample_count=full_sample_count,
                    max_samples=max_samples_per_snapshot,
                    seed=int(seed_base) + int(snapshot_idx) * 1_000_003,
                )
            else:
                sampling_coords = _temporal_dense_snapshot_center_coords_for_sampling(
                    bundle.dataset,
                    temporal_inference_spec=temporal_inference_spec,
                    source_name=source_name,
                )
                selected_indices = _select_dense_snapshot_cut_surface_indices(
                    coords=sampling_coords,
                    max_samples=max_samples_per_snapshot,
                    bounds=cut_surface_bounds,
                    visible_depth_fraction=float(
                        resolved_cut_surface_visible_depth_fraction
                    ),
                )
        else:
            selected_indices = _select_dense_snapshot_sample_indices(
                sample_count=full_sample_count,
                max_samples=max_samples_per_snapshot,
                seed=int(seed_base) + int(snapshot_idx) * 1_000_003,
            )
        selected_indices_by_snapshot.append(selected_indices)
        full_sample_counts[source_name] = full_sample_count
        selected_sample_counts[source_name] = int(selected_indices.size)

    snapshot_datasets: list[Any] = []
    snapshot_inference_datasets: list[Any] = []
    snapshot_sample_source_names: list[str] = []
    for bundle, selected_indices in zip(
        snapshot_bundles,
        selected_indices_by_snapshot,
        strict=True,
    ):
        source_name = str(bundle.selection.analysis_source_names[0])
        full_sample_count = int(len(bundle.inference_dataset))
        if full_sample_count != int(len(bundle.dataset)):
            raise RuntimeError(
                "Temporal dense snapshot inference dataset and visualization dataset must "
                "have identical sample counts. "
                f"source_name={source_name}, inference_samples={full_sample_count}, "
                f"visualization_samples={int(len(bundle.dataset))}."
            )
        if int(selected_indices.size) == full_sample_count:
            snapshot_dataset_part = bundle.dataset
            snapshot_inference_dataset_part = bundle.inference_dataset
        else:
            selected_list = [int(v) for v in selected_indices.tolist()]
            snapshot_dataset_part = torch.utils.data.Subset(
                bundle.dataset,
                selected_list,
            )
            snapshot_inference_dataset_part = torch.utils.data.Subset(
                bundle.inference_dataset,
                selected_list,
            )
        snapshot_datasets.append(snapshot_dataset_part)
        snapshot_inference_datasets.append(snapshot_inference_dataset_part)
        snapshot_sample_source_names.extend(
            [source_name] * int(selected_indices.size)
        )

    snapshot_dataset = torch.utils.data.ConcatDataset(snapshot_datasets)
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
    if max_samples_per_snapshot is not None:
        snapshot_selection_spec["sample_limit"] = {
            "max_samples_per_snapshot": int(max_samples_per_snapshot),
            "selection": str(resolved_sample_selection_mode),
        }
        if resolved_sample_selection_mode == "uniform_random_without_replacement":
            snapshot_selection_spec["sample_limit"]["selection_seed_base"] = int(seed_base)
        if resolved_sample_selection_mode == "diagonal_cut_surface":
            snapshot_selection_spec["sample_limit"].update(
                {
                    "diagonal_visible_depth_fraction": float(
                        resolved_cut_surface_visible_depth_fraction
                    ),
                    "bounds": (
                        None
                        if cut_surface_bounds is None
                        else cut_surface_bounds.astype(float).tolist()
                    ),
                    "ranking": "visible_side_nearest_to_diagonal_cut_plane",
                }
            )
    snapshot_selection_spec["inference_loader"] = {
        "batching": "batched_concat_v1",
        "prefetch_factor": _analysis_prefetch_factor(int(dataloader_num_workers)),
        "seed_sequence": "global_batch_index",
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

    return {
        "out_dir": out_dir,
        "cache_enabled": bool(cache_enabled),
        "cache_force_recompute": bool(cache_force_recompute),
        "cache_file": str(cache_file),
        "collector_mode": str(collector_mode),
        "summary_label": str(summary_label),
        "cache_config_path": str(cache_config_path),
        "regular_grid_overlap": float(regular_grid_overlap),
        "static_overlap_fraction": float(static_overlap_fraction),
        "resolved_frame_indices": resolved_frame_indices,
        "resolved_source_names": resolved_source_names,
        "snapshot_bundles": snapshot_bundles,
        "snapshot_datasets": snapshot_datasets,
        "snapshot_inference_datasets": snapshot_inference_datasets,
        "snapshot_dataset": snapshot_dataset,
        "selected_indices_by_snapshot": selected_indices_by_snapshot,
        "full_sample_counts": full_sample_counts,
        "selected_sample_counts": selected_sample_counts,
        "snapshot_cache_spec": snapshot_cache_spec,
        "max_samples_per_snapshot": max_samples_per_snapshot,
        "resolved_sample_selection_mode": str(resolved_sample_selection_mode),
        "resolved_cut_surface_visible_depth_fraction": (
            None
            if resolved_cut_surface_visible_depth_fraction is None
            else float(resolved_cut_surface_visible_depth_fraction)
        ),
        "cut_surface_bounds": cut_surface_bounds,
        "inference_batch_size": int(inference_batch_size),
        "dataloader_num_workers": int(dataloader_num_workers),
        "seed_base": int(seed_base),
    }


def _load_temporal_dense_snapshot_plan_cache(
    plan: dict[str, Any],
) -> tuple[dict[str, np.ndarray] | None, bool]:
    cache_log_prefix = f"[analysis][{plan['collector_mode']} cache]"
    if plan["cache_enabled"] and not plan["cache_force_recompute"]:
        snapshot_cache, snapshot_cache_msg = _load_inference_cache(
            out_dir=plan["out_dir"],
            cache_filename=plan["cache_file"],
            expected_spec=plan["snapshot_cache_spec"],
        )
        print(f"{cache_log_prefix} {snapshot_cache_msg}")
        return snapshot_cache, snapshot_cache is not None
    if plan["cache_enabled"] and plan["cache_force_recompute"]:
        print(f"{cache_log_prefix} Forced recompute requested; skipping cache load.")
    return None, False


def _save_temporal_dense_snapshot_plan_cache(
    *,
    plan: dict[str, Any],
    cache: dict[str, np.ndarray],
) -> None:
    if not plan["cache_enabled"]:
        return
    _save_inference_cache(
        out_dir=plan["out_dir"],
        cache_filename=plan["cache_file"],
        cache=cache,
        spec=plan["snapshot_cache_spec"],
    )
    snapshot_cache_npz, _ = _inference_cache_paths(
        plan["out_dir"],
        plan["cache_file"],
    )
    print(
        f"[analysis][{plan['collector_mode']} cache] "
        f"Saved inference cache: {snapshot_cache_npz}"
    )


def _finalize_temporal_dense_snapshot_plan(
    *,
    plan: dict[str, Any],
    snapshot_cache: dict[str, np.ndarray],
    snapshot_cache_loaded: bool,
) -> tuple[dict[str, np.ndarray], Any, Any, dict[str, Any]]:
    _validate_inference_cache_arrays(snapshot_cache)
    expected_total_samples = int(sum(int(len(ds)) for ds in plan["snapshot_datasets"]))
    actual_total_samples = int(len(snapshot_cache["inv_latents"]))
    if actual_total_samples != expected_total_samples:
        raise RuntimeError(
            f"{plan['summary_label']} cache sample count does not match the resolved "
            "dense snapshot datasets. "
            f"actual={actual_total_samples}, expected={expected_total_samples}, "
            f"cache_file={plan['out_dir'] / plan['cache_file']}."
        )

    snapshot_layout = resolve_snapshot_figure_layout(
        plan["snapshot_dataset"],
        is_synthetic=False,
        n_samples=actual_total_samples,
        analysis_source_names=plan["resolved_source_names"],
    )
    cut_surface_bounds = plan["cut_surface_bounds"]
    summary = {
        "enabled": True,
        "cache_enabled": bool(plan["cache_enabled"]),
        "cache_file": str(plan["out_dir"] / plan["cache_file"]),
        "cache_loaded_from_disk": bool(snapshot_cache_loaded),
        "cache_force_recompute": bool(plan["cache_force_recompute"]),
        "sample_count": int(actual_total_samples),
        "regular_grid_overlap": float(plan["regular_grid_overlap"]),
        "static_overlap_fraction_equivalent": float(plan["static_overlap_fraction"]),
        "frame_indices": [int(v) for v in plan["resolved_frame_indices"]],
        "source_names": [str(v) for v in plan["resolved_source_names"]],
        "sample_count_by_snapshot": {
            str(source_name): int(count)
            for source_name, count in plan["selected_sample_counts"].items()
        },
        "full_sample_count_by_snapshot": {
            str(source_name): int(count)
            for source_name, count in plan["full_sample_counts"].items()
        },
        "max_samples_per_snapshot": (
            None
            if plan["max_samples_per_snapshot"] is None
            else int(plan["max_samples_per_snapshot"])
        ),
        "inference_sample_selection": str(plan["resolved_sample_selection_mode"]),
        "cut_surface_visible_depth_fraction": (
            None
            if plan["resolved_cut_surface_visible_depth_fraction"] is None
            else float(plan["resolved_cut_surface_visible_depth_fraction"])
        ),
        "cut_surface_bounds": (
            None
            if cut_surface_bounds is None
            else cut_surface_bounds.astype(float).tolist()
        ),
    }
    return snapshot_cache, plan["snapshot_dataset"], snapshot_layout, summary


def _run_temporal_dense_snapshot_plan_inference(
    *,
    plan: dict[str, Any],
    model: Any,
    cuda_device: int,
    progress_every_batches: int,
) -> dict[str, np.ndarray]:
    expected_total_samples = int(
        sum(int(v) for v in plan["selected_sample_counts"].values())
    )
    for bundle in plan["snapshot_bundles"]:
        source_name = str(bundle.selection.analysis_source_names[0])
        selected_sample_count = int(plan["selected_sample_counts"][source_name])
        full_sample_count = int(plan["full_sample_counts"][source_name])
        cap_suffix = (
            ""
            if selected_sample_count == full_sample_count
            else (
                f", selected_samples={selected_sample_count}/{full_sample_count}, "
                f"selection={plan['resolved_sample_selection_mode']}"
            )
        )
        print(
            f"{plan['summary_label']} inference: "
            f"snapshot={source_name}, samples={selected_sample_count}{cap_suffix}, "
            f"regular_grid_overlap={plan['regular_grid_overlap']:.3f} "
            f"(static_overlap_fraction={plan['static_overlap_fraction']:.3f})."
        )
    combined_inference_dataset = _BatchedConcatDataset(
        plan["snapshot_inference_datasets"]
    )
    if int(len(combined_inference_dataset)) != expected_total_samples:
        raise RuntimeError(
            f"{plan['summary_label']} combined inference dataset sample count mismatch: "
            f"dataset={int(len(combined_inference_dataset))}, "
            f"expected={expected_total_samples}."
        )
    inference_dataloader = torch.utils.data.DataLoader(
        combined_inference_dataset,
        **_analysis_dataloader_kwargs(
            batch_size=int(plan["inference_batch_size"]),
            dataloader_num_workers=int(plan["dataloader_num_workers"]),
            collate_fn=_temporal_identity_or_default_collate,
        ),
    )
    snapshot_bundles = plan["snapshot_bundles"]
    snapshot_cache = gather_inference_batches(
        model,
        inference_dataloader,
        f"cuda:{int(cuda_device)}" if torch.cuda.is_available() else "cpu",
        max_batches=None,
        max_samples_total=None,
        collect_coords=True,
        seed_base=int(plan["seed_base"]),
        progress_every_batches=int(progress_every_batches),
        verbose=True,
        temporal_sequence_mode=str(snapshot_bundles[0].collection_inference_spec.mode),
        temporal_static_frame_index=(
            snapshot_bundles[0].collection_inference_spec.static_frame_index
        ),
    )
    del inference_dataloader
    _validate_inference_cache_arrays(snapshot_cache)
    if int(len(snapshot_cache["inv_latents"])) != expected_total_samples:
        raise RuntimeError(
            f"{plan['summary_label']} merged cache sample count mismatch: "
            f"merged={int(len(snapshot_cache['inv_latents']))}, "
            f"expected={expected_total_samples}."
        )
    return snapshot_cache


def _slice_inference_cache_rows(
    cache: dict[str, np.ndarray],
    row_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    rows = np.asarray(row_indices, dtype=np.int64)
    sliced: dict[str, np.ndarray] = {}
    for key, value in cache.items():
        arr = np.asarray(value)
        if arr.size == 0:
            sliced[key] = arr.copy()
            continue
        sliced[key] = arr[rows]
    _validate_inference_cache_arrays(sliced)
    return sliced


def _run_temporal_dense_snapshot_union_inference(
    *,
    plans: list[dict[str, Any]],
    model: Any,
    cuda_device: int,
    progress_every_batches: int,
) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    if not plans:
        raise ValueError("Cannot run union dense snapshot inference for zero plans.")

    entry_order: list[tuple[int, str]] = []
    entries: dict[tuple[int, str], dict[str, Any]] = {}
    for plan_idx, plan in enumerate(plans):
        for snapshot_idx, bundle in enumerate(plan["snapshot_bundles"]):
            frame_index = int(plan["resolved_frame_indices"][snapshot_idx])
            source_name = str(plan["resolved_source_names"][snapshot_idx])
            key = (frame_index, source_name)
            if key not in entries:
                entry_order.append(key)
                entries[key] = {
                    "bundle": bundle,
                    "selected_parts": [],
                    "plan_slots": [],
                }
            existing_count = int(len(entries[key]["bundle"].inference_dataset))
            current_count = int(len(bundle.inference_dataset))
            if current_count != existing_count:
                raise RuntimeError(
                    "Cannot fuse dense temporal inference for the same frame with "
                    "different sample counts. "
                    f"frame_index={frame_index}, source_name={source_name}, "
                    f"first_count={existing_count}, current_count={current_count}, "
                    f"plan_idx={plan_idx}."
                )
            selected_indices = np.asarray(
                plan["selected_indices_by_snapshot"][snapshot_idx],
                dtype=np.int64,
            )
            entries[key]["selected_parts"].append(selected_indices)
            entries[key]["plan_slots"].append((plan_idx, snapshot_idx))

    union_inference_datasets: list[Any] = []
    union_indices_by_key: dict[tuple[int, str], np.ndarray] = {}
    union_offsets_by_key: dict[tuple[int, str], int] = {}
    union_total_samples = 0
    requested_total_samples = 0
    for key in entry_order:
        entry = entries[key]
        bundle = entry["bundle"]
        parts = [np.asarray(part, dtype=np.int64) for part in entry["selected_parts"]]
        requested_total_samples += int(sum(int(part.size) for part in parts))
        union_indices = np.unique(np.concatenate(parts)).astype(np.int64, copy=False)
        full_sample_count = int(len(bundle.inference_dataset))
        if int(union_indices.size) == full_sample_count:
            dataset_part = bundle.inference_dataset
        else:
            dataset_part = torch.utils.data.Subset(
                bundle.inference_dataset,
                [int(v) for v in union_indices.tolist()],
            )
        union_offsets_by_key[key] = int(union_total_samples)
        union_indices_by_key[key] = union_indices
        union_total_samples += int(union_indices.size)
        union_inference_datasets.append(dataset_part)

    if union_total_samples <= 0:
        raise RuntimeError("Combined dense temporal inference selected zero samples.")

    print(
        "[analysis] Combined dense temporal inference: "
        f"collectors={', '.join(str(plan['collector_mode']) for plan in plans)}, "
        f"frames={len(entry_order)}, requested_samples={requested_total_samples}, "
        f"union_samples={union_total_samples}."
    )

    first_spec = plans[0]["snapshot_bundles"][0].collection_inference_spec
    for plan in plans:
        for bundle in plan["snapshot_bundles"]:
            spec = bundle.collection_inference_spec
            if (
                str(spec.mode) != str(first_spec.mode)
                or spec.static_frame_index != first_spec.static_frame_index
            ):
                raise RuntimeError(
                    "Cannot fuse dense temporal inference with mixed temporal "
                    "collection specs. "
                    f"first={first_spec.to_cache_spec()}, current={spec.to_cache_spec()}."
                )

    combined_inference_dataset = _BatchedConcatDataset(union_inference_datasets)
    if int(len(combined_inference_dataset)) != int(union_total_samples):
        raise RuntimeError(
            "Combined dense temporal union dataset sample count mismatch: "
            f"dataset={int(len(combined_inference_dataset))}, "
            f"expected={union_total_samples}."
        )
    inference_dataloader = torch.utils.data.DataLoader(
        combined_inference_dataset,
        **_analysis_dataloader_kwargs(
            batch_size=int(plans[0]["inference_batch_size"]),
            dataloader_num_workers=int(plans[0]["dataloader_num_workers"]),
            collate_fn=_temporal_identity_or_default_collate,
        ),
    )
    union_cache = gather_inference_batches(
        model,
        inference_dataloader,
        f"cuda:{int(cuda_device)}" if torch.cuda.is_available() else "cpu",
        max_batches=None,
        max_samples_total=None,
        collect_coords=True,
        seed_base=int(plans[0]["seed_base"]),
        progress_every_batches=int(progress_every_batches),
        verbose=True,
        temporal_sequence_mode=str(first_spec.mode),
        temporal_static_frame_index=first_spec.static_frame_index,
    )
    del inference_dataloader
    _validate_inference_cache_arrays(union_cache)
    if int(len(union_cache["inv_latents"])) != int(union_total_samples):
        raise RuntimeError(
            "Combined dense temporal union cache sample count mismatch: "
            f"cache={int(len(union_cache['inv_latents']))}, "
            f"expected={union_total_samples}."
        )

    plan_row_indices: list[np.ndarray] = []
    for plan in plans:
        plan_rows: list[np.ndarray] = []
        for snapshot_idx, selected_indices in enumerate(
            plan["selected_indices_by_snapshot"]
        ):
            frame_index = int(plan["resolved_frame_indices"][snapshot_idx])
            source_name = str(plan["resolved_source_names"][snapshot_idx])
            key = (frame_index, source_name)
            union_indices = union_indices_by_key[key]
            local_positions = np.searchsorted(
                union_indices,
                np.asarray(selected_indices, dtype=np.int64),
            )
            if not np.array_equal(
                union_indices[local_positions],
                np.asarray(selected_indices, dtype=np.int64),
            ):
                raise RuntimeError(
                    "Combined dense temporal inference failed to map selected "
                    "indices back into the union cache. "
                    f"frame_index={frame_index}, source_name={source_name}."
                )
            plan_rows.append(
                union_offsets_by_key[key] + local_positions.astype(np.int64, copy=False)
            )
        plan_row_indices.append(np.concatenate(plan_rows).astype(np.int64, copy=False))

    return union_cache, plan_row_indices


def _collect_temporal_dense_snapshot_cache(
    *,
    analysis_cfg: DictConfig,
    temporal_selection: Any,
    temporal_inference_spec: Any,
    checkpoint_path: str,
    out_dir: Path,
    model_cfg_for_module: DictConfig,
    model: Any | None,
    model_loader: Callable[..., tuple[Any, Any, Any]],
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
    max_samples_per_snapshot: int | None = None,
    sample_selection_mode: str = "uniform_random_without_replacement",
    cut_surface_visible_depth_fraction: float | None = None,
) -> tuple[dict[str, np.ndarray], Any, Any, dict[str, Any], Any]:
    plan = _build_temporal_dense_snapshot_plan(
        analysis_cfg=analysis_cfg,
        temporal_selection=temporal_selection,
        temporal_inference_spec=temporal_inference_spec,
        checkpoint_path=checkpoint_path,
        out_dir=out_dir,
        model_cfg_for_module=model_cfg_for_module,
        seed_base=int(seed_base),
        inference_batch_size=int(inference_batch_size),
        dataloader_num_workers=int(dataloader_num_workers),
        frame_indices=[int(v) for v in frame_indices],
        source_names=[str(v) for v in source_names],
        cache_enabled=bool(cache_enabled),
        cache_force_recompute=bool(cache_force_recompute),
        cache_file=str(cache_file),
        collector_mode=str(collector_mode),
        summary_label=str(summary_label),
        cache_config_path=str(cache_config_path),
        max_samples_per_snapshot=max_samples_per_snapshot,
        sample_selection_mode=sample_selection_mode,
        cut_surface_visible_depth_fraction=cut_surface_visible_depth_fraction,
    )
    snapshot_cache, snapshot_cache_loaded = _load_temporal_dense_snapshot_plan_cache(
        plan
    )
    if snapshot_cache is None:
        if figure_only:
            raise RuntimeError(
                "figure_set.figure_only requires a valid cache for "
                f"{summary_label.lower()}. Missing cache: {out_dir / cache_file}. "
                "Run the full analysis once with figure_set.figure_only=false to populate it."
            )
        if model is None:
            model, _, _ = model_loader(
                checkpoint_path,
                cuda_device=int(cuda_device),
                cfg=model_cfg_for_module,
            )
        snapshot_cache = _run_temporal_dense_snapshot_plan_inference(
            plan=plan,
            model=model,
            cuda_device=int(cuda_device),
            progress_every_batches=int(progress_every_batches),
        )
        _save_temporal_dense_snapshot_plan_cache(
            plan=plan,
            cache=snapshot_cache,
        )
    snapshot_cache, snapshot_dataset, snapshot_layout, summary = (
        _finalize_temporal_dense_snapshot_plan(
            plan=plan,
            snapshot_cache=snapshot_cache,
            snapshot_cache_loaded=snapshot_cache_loaded,
        )
    )
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
    model_loader: Callable[..., tuple[Any, Any, Any]],
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
        model_loader=model_loader,
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


def _resolve_temporal_md_space_inference_max_samples(
    analysis_cfg: DictConfig,
) -> int | None:
    raw_value = OmegaConf.select(
        analysis_cfg,
        "real_md.temporal.md_space.inference_max_points_per_snapshot",
        default=None,
    )
    if raw_value is None:
        raw_value = OmegaConf.select(
            analysis_cfg,
            "real_md.temporal.md_space.animation_max_points",
            default=OmegaConf.select(
                analysis_cfg,
                "real_md.temporal.animation_max_points",
                default=None,
            ),
        )
    return _positive_int_or_none(raw_value)


def _resolve_temporal_md_space_inference_sample_selection(
    analysis_cfg: DictConfig,
) -> str:
    return _normalize_dense_snapshot_sample_selection_mode(
        OmegaConf.select(
            analysis_cfg,
            "real_md.temporal.md_space.inference_sample_selection",
            default="cut_surface",
        )
    )


def _resolve_temporal_md_space_diagonal_visible_depth_fraction(
    analysis_cfg: DictConfig,
) -> float:
    value = float(
        OmegaConf.select(
            analysis_cfg,
            "real_md.temporal.md_space.diagonal_visible_depth_fraction",
            default=0.10,
        )
    )
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            "real_md.temporal.md_space.diagonal_visible_depth_fraction must be "
            f"positive and finite, got {value!r}."
        )
    return float(value)


def _collect_temporal_md_space_animation_cache(
    *,
    analysis_cfg: DictConfig,
    temporal_selection: Any,
    temporal_inference_spec: Any,
    checkpoint_path: str,
    out_dir: Path,
    model_cfg_for_module: DictConfig,
    model: Any | None,
    model_loader: Callable[..., tuple[Any, Any, Any]],
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
    max_samples_per_snapshot = _resolve_temporal_md_space_inference_max_samples(
        analysis_cfg
    )
    sample_selection_mode = _resolve_temporal_md_space_inference_sample_selection(
        analysis_cfg
    )
    cut_surface_visible_depth_fraction = (
        _resolve_temporal_md_space_diagonal_visible_depth_fraction(analysis_cfg)
        if sample_selection_mode == "diagonal_cut_surface"
        else None
    )
    return _collect_temporal_dense_snapshot_cache(
        analysis_cfg=analysis_cfg,
        temporal_selection=temporal_selection,
        temporal_inference_spec=temporal_inference_spec,
        checkpoint_path=checkpoint_path,
        out_dir=out_dir,
        model_cfg_for_module=model_cfg_for_module,
        model=model,
        model_loader=model_loader,
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
        max_samples_per_snapshot=max_samples_per_snapshot,
        sample_selection_mode=sample_selection_mode,
        cut_surface_visible_depth_fraction=cut_surface_visible_depth_fraction,
    )


def _collect_combined_temporal_dense_snapshot_caches(
    *,
    analysis_cfg: DictConfig,
    temporal_selection: Any,
    temporal_inference_spec: Any,
    checkpoint_path: str,
    out_dir: Path,
    model_cfg_for_module: DictConfig,
    model: Any | None,
    model_loader: Callable[..., tuple[Any, Any, Any]],
    cuda_device: int,
    seed_base: int,
    inference_batch_size: int,
    dataloader_num_workers: int,
    progress_every_batches: int,
) -> tuple[
    tuple[dict[str, np.ndarray], Any, Any, dict[str, Any]],
    tuple[dict[str, np.ndarray], Any, Any, dict[str, Any]],
    Any,
] | None:
    snapshot_cfg = OmegaConf.select(
        analysis_cfg,
        "inputs.temporal_real.snapshot_visualization",
        default=None,
    )
    snapshot_cache_file = str(
        OmegaConf.select(
            snapshot_cfg,
            "cache.file",
            default="temporal_snapshot_dense_inference_cache.npz",
        )
    ).strip()
    snapshot_plan = _build_temporal_dense_snapshot_plan(
        analysis_cfg=analysis_cfg,
        temporal_selection=temporal_selection,
        temporal_inference_spec=temporal_inference_spec,
        checkpoint_path=checkpoint_path,
        out_dir=out_dir,
        model_cfg_for_module=model_cfg_for_module,
        seed_base=int(seed_base),
        inference_batch_size=int(inference_batch_size),
        dataloader_num_workers=int(dataloader_num_workers),
        frame_indices=[
            int(v)
            for v in np.asarray(temporal_selection.analysis_frame_indices, dtype=np.int64)
        ],
        source_names=[str(v) for v in temporal_selection.analysis_source_names],
        cache_enabled=bool(OmegaConf.select(snapshot_cfg, "cache.enabled", default=True)),
        cache_force_recompute=bool(
            OmegaConf.select(snapshot_cfg, "cache.force_recompute", default=False)
        ),
        cache_file=snapshot_cache_file,
        collector_mode="temporal_snapshot_visualization",
        summary_label="Temporal snapshot visualization",
        cache_config_path="inputs.temporal_real.snapshot_visualization.cache",
    )

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
    md_frame_indices, md_source_names = resolve_temporal_real_snapshot_subset(
        analysis_cfg=analysis_cfg,
        selection=temporal_selection,
        snapshot_count=int(dense_snapshot_count),
    )
    md_cache_file = str(
        OmegaConf.select(
            analysis_cfg,
            "real_md.temporal.md_space.dense_snapshot_cache.file",
            default="temporal_md_space_dense_inference_cache.npz",
        )
    ).strip()
    md_sample_selection_mode = _resolve_temporal_md_space_inference_sample_selection(
        analysis_cfg
    )
    md_cut_surface_visible_depth_fraction = (
        _resolve_temporal_md_space_diagonal_visible_depth_fraction(analysis_cfg)
        if md_sample_selection_mode == "diagonal_cut_surface"
        else None
    )
    md_plan = _build_temporal_dense_snapshot_plan(
        analysis_cfg=analysis_cfg,
        temporal_selection=temporal_selection,
        temporal_inference_spec=temporal_inference_spec,
        checkpoint_path=checkpoint_path,
        out_dir=out_dir,
        model_cfg_for_module=model_cfg_for_module,
        seed_base=int(seed_base),
        inference_batch_size=int(inference_batch_size),
        dataloader_num_workers=int(dataloader_num_workers),
        frame_indices=[int(v) for v in md_frame_indices.tolist()],
        source_names=[str(v) for v in md_source_names],
        cache_enabled=bool(
            OmegaConf.select(
                analysis_cfg,
                "real_md.temporal.md_space.dense_snapshot_cache.enabled",
                default=True,
            )
        ),
        cache_force_recompute=bool(
            OmegaConf.select(
                analysis_cfg,
                "real_md.temporal.md_space.dense_snapshot_cache.force_recompute",
                default=False,
            )
        ),
        cache_file=md_cache_file,
        collector_mode="temporal_md_space_animation",
        summary_label="Temporal MD-space animation",
        cache_config_path="real_md.temporal.md_space.dense_snapshot_cache",
        max_samples_per_snapshot=_resolve_temporal_md_space_inference_max_samples(
            analysis_cfg
        ),
        sample_selection_mode=md_sample_selection_mode,
        cut_surface_visible_depth_fraction=md_cut_surface_visible_depth_fraction,
    )

    snapshot_cache, snapshot_cache_loaded = _load_temporal_dense_snapshot_plan_cache(
        snapshot_plan
    )
    md_cache, md_cache_loaded = _load_temporal_dense_snapshot_plan_cache(md_plan)
    if snapshot_cache is not None and md_cache is not None:
        return (
            _finalize_temporal_dense_snapshot_plan(
                plan=snapshot_plan,
                snapshot_cache=snapshot_cache,
                snapshot_cache_loaded=snapshot_cache_loaded,
            ),
            _finalize_temporal_dense_snapshot_plan(
                plan=md_plan,
                snapshot_cache=md_cache,
                snapshot_cache_loaded=md_cache_loaded,
            ),
            model,
        )
    if snapshot_cache is not None or md_cache is not None:
        print(
            "[analysis] Combined dense temporal inference found one valid dense cache; "
            "collecting only the missing dense cache."
        )
        if model is None:
            model, _, _ = model_loader(
                checkpoint_path,
                cuda_device=int(cuda_device),
                cfg=model_cfg_for_module,
            )
        if snapshot_cache is None:
            snapshot_cache = _run_temporal_dense_snapshot_plan_inference(
                plan=snapshot_plan,
                model=model,
                cuda_device=int(cuda_device),
                progress_every_batches=int(progress_every_batches),
            )
            _save_temporal_dense_snapshot_plan_cache(
                plan=snapshot_plan,
                cache=snapshot_cache,
            )
            snapshot_cache_loaded = False
        if md_cache is None:
            md_cache = _run_temporal_dense_snapshot_plan_inference(
                plan=md_plan,
                model=model,
                cuda_device=int(cuda_device),
                progress_every_batches=int(progress_every_batches),
            )
            _save_temporal_dense_snapshot_plan_cache(
                plan=md_plan,
                cache=md_cache,
            )
            md_cache_loaded = False
        return (
            _finalize_temporal_dense_snapshot_plan(
                plan=snapshot_plan,
                snapshot_cache=snapshot_cache,
                snapshot_cache_loaded=snapshot_cache_loaded,
            ),
            _finalize_temporal_dense_snapshot_plan(
                plan=md_plan,
                snapshot_cache=md_cache,
                snapshot_cache_loaded=md_cache_loaded,
            ),
            model,
        )

    if model is None:
        model, _, _ = model_loader(
            checkpoint_path,
            cuda_device=int(cuda_device),
            cfg=model_cfg_for_module,
        )

    union_cache, plan_row_indices = _run_temporal_dense_snapshot_union_inference(
        plans=[snapshot_plan, md_plan],
        model=model,
        cuda_device=int(cuda_device),
        progress_every_batches=int(progress_every_batches),
    )
    snapshot_cache = _slice_inference_cache_rows(union_cache, plan_row_indices[0])
    md_cache = _slice_inference_cache_rows(union_cache, plan_row_indices[1])

    _save_temporal_dense_snapshot_plan_cache(
        plan=snapshot_plan,
        cache=snapshot_cache,
    )
    _save_temporal_dense_snapshot_plan_cache(
        plan=md_plan,
        cache=md_cache,
    )

    return (
        _finalize_temporal_dense_snapshot_plan(
            plan=snapshot_plan,
            snapshot_cache=snapshot_cache,
            snapshot_cache_loaded=False,
        ),
        _finalize_temporal_dense_snapshot_plan(
            plan=md_plan,
            snapshot_cache=md_cache,
            snapshot_cache_loaded=False,
        ),
        model,
    )


def _resolve_temporal_md_space_animation_summary_outputs(
    summary: dict[str, Any],
) -> tuple[list[str], np.ndarray | None]:
    cut_surface_bounds_raw = summary.get("cut_surface_bounds")
    spatial_bounds = (
        None
        if cut_surface_bounds_raw is None
        else np.asarray(cut_surface_bounds_raw, dtype=np.float32)
    )
    return list(summary["source_names"]), spatial_bounds


@dataclass
class TemporalDenseOutputs:
    snapshot_cache: dict[str, np.ndarray] | None = None
    snapshot_dataset: Any | None = None
    snapshot_layout: Any | None = None
    snapshot_summary: dict[str, Any] | None = None
    snapshot_enabled: bool = False
    md_space_cache: dict[str, np.ndarray] | None = None
    md_space_dataset: Any | None = None
    md_space_layout: Any | None = None
    md_space_source_names: list[str] | None = None
    md_space_spatial_bounds: np.ndarray | None = None
    md_space_summary: dict[str, Any] | None = None
    md_space_reuse_main_cache: bool = False
    md_space_enabled: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)


def _collect_temporal_dense_outputs(
    *,
    analysis_cfg: DictConfig,
    temporal_bundle: Any | None,
    temporal_inference_spec: Any | None,
    checkpoint_path: str,
    out_dir: Path,
    model_cfg_for_module: DictConfig,
    model: Any | None,
    model_loader: Callable[..., tuple[Any, Any, Any]],
    cuda_device: int,
    seed_base: int,
    figure_only: bool,
    inference_batch_size: int,
    dataloader_num_workers: int,
    progress_every_batches: int,
    step: Callable[[str], None],
) -> tuple[TemporalDenseOutputs, Any | None]:
    outputs = TemporalDenseOutputs()
    if temporal_bundle is None:
        return outputs, model

    outputs.snapshot_enabled = bool(
        OmegaConf.select(
            analysis_cfg,
            "inputs.temporal_real.snapshot_visualization.enabled",
            default=True,
        )
    )
    outputs.md_space_reuse_main_cache = bool(
        OmegaConf.select(
            analysis_cfg,
            "real_md.temporal.md_space.reuse_main_inference_cache",
            default=False,
        )
    )
    outputs.md_space_enabled = bool(
        OmegaConf.select(analysis_cfg, "real_md.enabled", default=True)
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

    if (
        outputs.snapshot_enabled
        and outputs.md_space_enabled
        and not outputs.md_space_reuse_main_cache
        and not figure_only
    ):
        step("Loading combined dense temporal snapshot and MD-space data")
        combined_dense_result = _collect_combined_temporal_dense_snapshot_caches(
            analysis_cfg=analysis_cfg,
            temporal_selection=temporal_bundle.selection,
            temporal_inference_spec=temporal_inference_spec,
            checkpoint_path=checkpoint_path,
            out_dir=out_dir,
            model_cfg_for_module=model_cfg_for_module,
            model=model,
            model_loader=model_loader,
            cuda_device=int(cuda_device),
            seed_base=int(seed_base),
            inference_batch_size=int(inference_batch_size),
            dataloader_num_workers=int(dataloader_num_workers),
            progress_every_batches=int(progress_every_batches),
        )
        if combined_dense_result is not None:
            snapshot_dense_result, md_animation_dense_result, model = combined_dense_result
            (
                outputs.snapshot_cache,
                outputs.snapshot_dataset,
                outputs.snapshot_layout,
                outputs.snapshot_summary,
            ) = snapshot_dense_result
            (
                outputs.md_space_cache,
                outputs.md_space_dataset,
                outputs.md_space_layout,
                outputs.md_space_summary,
            ) = md_animation_dense_result
            outputs.metrics["temporal_snapshot_visualization"] = dict(
                outputs.snapshot_summary
            )
            (
                outputs.md_space_source_names,
                outputs.md_space_spatial_bounds,
            ) = _resolve_temporal_md_space_animation_summary_outputs(
                outputs.md_space_summary
            )
            outputs.metrics["temporal_md_space_animation_sampling"] = dict(
                outputs.md_space_summary
            )

    if outputs.snapshot_enabled and outputs.snapshot_cache is None:
        step("Loading dense temporal snapshot visualization data")
        (
            outputs.snapshot_cache,
            outputs.snapshot_dataset,
            outputs.snapshot_layout,
            outputs.snapshot_summary,
            model,
        ) = _collect_temporal_snapshot_visualization_cache(
            analysis_cfg=analysis_cfg,
            temporal_selection=temporal_bundle.selection,
            temporal_inference_spec=temporal_inference_spec,
            checkpoint_path=checkpoint_path,
            out_dir=out_dir,
            model_cfg_for_module=model_cfg_for_module,
            model=model,
            model_loader=model_loader,
            cuda_device=int(cuda_device),
            seed_base=int(seed_base),
            figure_only=bool(figure_only),
            inference_batch_size=int(inference_batch_size),
            dataloader_num_workers=int(dataloader_num_workers),
            progress_every_batches=int(progress_every_batches),
        )
        outputs.metrics["temporal_snapshot_visualization"] = dict(
            outputs.snapshot_summary
        )

    if outputs.md_space_enabled and outputs.md_space_reuse_main_cache:
        print(
            "[analysis] real_md.temporal.md_space.reuse_main_inference_cache=true: "
            "reusing the main temporal inference cache for MD-space animation frames."
        )
    if (
        outputs.md_space_enabled
        and not outputs.md_space_reuse_main_cache
        and outputs.md_space_cache is None
    ):
        step("Loading dense temporal MD-space animation data")
        md_animation_cache_result = _collect_temporal_md_space_animation_cache(
            analysis_cfg=analysis_cfg,
            temporal_selection=temporal_bundle.selection,
            temporal_inference_spec=temporal_inference_spec,
            checkpoint_path=checkpoint_path,
            out_dir=out_dir,
            model_cfg_for_module=model_cfg_for_module,
            model=model,
            model_loader=model_loader,
            cuda_device=int(cuda_device),
            seed_base=int(seed_base),
            figure_only=bool(figure_only),
            inference_batch_size=int(inference_batch_size),
            dataloader_num_workers=int(dataloader_num_workers),
            progress_every_batches=int(progress_every_batches),
        )
        if md_animation_cache_result is not None:
            (
                outputs.md_space_cache,
                outputs.md_space_dataset,
                outputs.md_space_layout,
                outputs.md_space_summary,
                model,
            ) = md_animation_cache_result
            (
                outputs.md_space_source_names,
                outputs.md_space_spatial_bounds,
            ) = _resolve_temporal_md_space_animation_summary_outputs(
                outputs.md_space_summary
            )
            outputs.metrics["temporal_md_space_animation_sampling"] = dict(
                outputs.md_space_summary
            )

    return outputs, model
