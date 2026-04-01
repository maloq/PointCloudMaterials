from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from src.data_utils.data_load import PointCloudDataset
from src.data_utils.temporal_lammps_dataset import (
    TemporalLAMMPSDumpDataset,
    estimate_lammps_dump_cutoff_radius,
)


@dataclass(frozen=True)
class TemporalRealAnalysisSelection:
    dump_file: Path
    dump_summary: dict[str, Any]
    sequence_length: int
    frame_stride: int
    analysis_snapshot_count: int
    inference_snapshot_fraction: float
    inference_frame_indices: np.ndarray
    analysis_frame_indices: np.ndarray
    inference_source_names: list[str]
    analysis_source_names: list[str]
    radius: float
    num_points: int
    radius_source: str
    radius_estimation: dict[str, Any] | None
    center_selection: dict[str, Any]
    cache_dir: Path | None

    def to_cache_spec(self) -> dict[str, Any]:
        return {
            "dump_file": str(self.dump_file),
            "sequence_length": int(self.sequence_length),
            "frame_stride": int(self.frame_stride),
            "analysis_snapshot_count": int(self.analysis_snapshot_count),
            "inference_snapshot_fraction": float(self.inference_snapshot_fraction),
            "inference_frame_indices": self.inference_frame_indices.astype(int).tolist(),
            "analysis_frame_indices": self.analysis_frame_indices.astype(int).tolist(),
            "inference_source_names": [str(v) for v in self.inference_source_names],
            "analysis_source_names": [str(v) for v in self.analysis_source_names],
            "radius": float(self.radius),
            "num_points": int(self.num_points),
            "radius_source": str(self.radius_source),
            "radius_estimation": (
                None if self.radius_estimation is None else dict(self.radius_estimation)
            ),
            "center_selection": dict(self.center_selection),
            "cache_dir": None if self.cache_dir is None else str(self.cache_dir),
        }


@dataclass(frozen=True)
class TemporalRealAnalysisBundle:
    dataset: TemporalLAMMPSDumpDataset
    dataloader: torch.utils.data.DataLoader
    selection: TemporalRealAnalysisSelection


@dataclass(frozen=True)
class TemporalRealInferenceSpec:
    mode: str
    static_frame_index: int | None

    def to_cache_spec(self) -> dict[str, Any]:
        return {
            "mode": str(self.mode),
            "static_frame_index": (
                None if self.static_frame_index is None else int(self.static_frame_index)
            ),
        }


def _temporal_identity_or_default_collate(batch: Any) -> Any:
    # TemporalLAMMPSDumpDataset implements batched __getitems__ and may hand the
    # DataLoader an already-collated dict. In that case we must not run the
    # default collate again.
    if isinstance(batch, dict):
        return batch
    return default_collate(batch)


def temporal_real_analysis_enabled(analysis_cfg: Any) -> bool:
    return bool(OmegaConf.select(analysis_cfg, "inputs.temporal_real.enabled", default=False))


def resolve_temporal_real_inference_spec(analysis_cfg: Any) -> TemporalRealInferenceSpec:
    mode_raw = OmegaConf.select(
        analysis_cfg,
        "inputs.temporal_real.inference_mode",
        default="static",
    )
    mode = str(mode_raw).strip().lower()
    if mode in {"static", "static_anchor"}:
        return TemporalRealInferenceSpec(
            mode="static_anchor",
            static_frame_index=0,
        )
    if mode == "temporal":
        return TemporalRealInferenceSpec(
            mode="temporal",
            static_frame_index=None,
        )
    raise ValueError(
        "inputs.temporal_real.inference_mode must be one of "
        "['static', 'static_anchor', 'temporal'], "
        f"got {mode_raw!r}."
    )


def build_temporal_real_analysis_bundle(
    *,
    analysis_cfg: Any,
    model_cfg: Any,
    batch_size: int,
    dataloader_num_workers: int,
) -> TemporalRealAnalysisBundle:
    if not temporal_real_analysis_enabled(analysis_cfg):
        raise ValueError("inputs.temporal_real.enabled must be true to build a temporal analysis bundle.")

    temporal_cfg = OmegaConf.select(analysis_cfg, "inputs.temporal_real", default=None)
    dump_file_raw = OmegaConf.select(temporal_cfg, "dump_file", default=None)
    if dump_file_raw is None or str(dump_file_raw).strip() == "":
        raise ValueError("inputs.temporal_real.dump_file is required when temporal real analysis is enabled.")
    dump_file = Path(str(dump_file_raw)).expanduser().resolve()

    cache_dir_raw = OmegaConf.select(temporal_cfg, "cache_dir", default=None)
    cache_dir = None if cache_dir_raw in {None, ""} else Path(str(cache_dir_raw)).expanduser().resolve()

    sequence_length = int(OmegaConf.select(temporal_cfg, "sequence_length", default=4))
    frame_stride = int(OmegaConf.select(temporal_cfg, "frame_stride", default=1))
    analysis_snapshot_count = int(
        OmegaConf.select(temporal_cfg, "analysis_snapshot_count", default=6)
    )
    inference_snapshot_fraction = float(
        OmegaConf.select(temporal_cfg, "inference_snapshot_fraction", default=0.25)
    )
    time_scale_raw = OmegaConf.select(temporal_cfg, "time_scale", default=None)
    time_unit_raw = OmegaConf.select(temporal_cfg, "time_unit", default=None)
    time_scale = None if time_scale_raw is None else float(time_scale_raw)
    time_unit = None if time_unit_raw is None else str(time_unit_raw).strip()

    radius_raw = OmegaConf.select(temporal_cfg, "radius", default=None)
    num_points_raw = OmegaConf.select(temporal_cfg, "num_points", default=None)
    num_points = (
        int(getattr(model_cfg.data, "num_points"))
        if num_points_raw is None
        else int(num_points_raw)
    )
    scan = TemporalLAMMPSDumpDataset.scan_dump_file(dump_file)
    timestep_deltas = np.diff(scan.timesteps).astype(np.int64, copy=False)
    dump_summary = {
        "source_path": str(dump_file),
        "frame_count": int(scan.frame_count),
        "num_atoms": int(scan.num_atoms),
        "atom_columns": list(scan.atom_columns),
        "first_timestep": int(scan.timesteps[0]),
        "last_timestep": int(scan.timesteps[-1]),
        "unique_timestep_deltas": np.unique(timestep_deltas).astype(np.int64, copy=False).tolist(),
        "box_low_first_frame": scan.box_low[0].astype(np.float64).tolist(),
        "box_high_first_frame": scan.box_high[0].astype(np.float64).tolist(),
        "estimated_positions_cache_bytes": int(scan.frame_count) * int(scan.num_atoms) * 3 * np.dtype(np.float32).itemsize,
        "estimated_positions_cache_gib": (
            int(scan.frame_count) * int(scan.num_atoms) * 3 * np.dtype(np.float32).itemsize
        ) / float(1024 ** 3),
    }
    total_frames = int(scan.frame_count)
    eligible_snapshot_count = total_frames - (sequence_length - 1) * frame_stride
    if eligible_snapshot_count <= 0:
        raise ValueError(
            "Temporal analysis sequence configuration leaves no valid anchor frames. "
            f"frame_count={total_frames}, sequence_length={sequence_length}, frame_stride={frame_stride}."
        )
    if analysis_snapshot_count <= 0:
        raise ValueError(
            f"inputs.temporal_real.analysis_snapshot_count must be > 0, got {analysis_snapshot_count}."
        )
    if analysis_snapshot_count > eligible_snapshot_count:
        raise ValueError(
            "inputs.temporal_real.analysis_snapshot_count exceeds the number of valid anchor frames. "
            f"analysis_snapshot_count={analysis_snapshot_count}, eligible_snapshot_count={eligible_snapshot_count}."
        )
    if inference_snapshot_fraction <= 0.0 or inference_snapshot_fraction > 1.0:
        raise ValueError(
            "inputs.temporal_real.inference_snapshot_fraction must be in (0, 1], "
            f"got {inference_snapshot_fraction}."
        )

    inference_snapshot_count = max(
        analysis_snapshot_count,
        int(np.ceil(float(eligible_snapshot_count) * float(inference_snapshot_fraction))),
    )
    inference_snapshot_count = min(inference_snapshot_count, eligible_snapshot_count)

    eligible_frame_indices = np.arange(eligible_snapshot_count, dtype=np.int64)
    inference_local_idx = _equidistant_selection_indices(
        num_items=eligible_snapshot_count,
        num_selected=inference_snapshot_count,
    )
    inference_frame_indices = eligible_frame_indices[inference_local_idx]
    analysis_local_idx = _equidistant_selection_indices(
        num_items=int(inference_frame_indices.size),
        num_selected=analysis_snapshot_count,
    )
    analysis_frame_indices = inference_frame_indices[analysis_local_idx]

    inference_source_names = [
        _format_temporal_snapshot_name(
            frame_index=int(frame_idx),
            timestep=int(scan.timesteps[int(frame_idx)]),
            time_scale=time_scale,
            time_unit=time_unit,
        )
        for frame_idx in inference_frame_indices.tolist()
    ]
    analysis_frame_set = {int(v) for v in analysis_frame_indices.tolist()}
    analysis_source_names = [
        str(source_name)
        for source_name, frame_idx in zip(inference_source_names, inference_frame_indices.tolist(), strict=True)
        if int(frame_idx) in analysis_frame_set
    ]

    if radius_raw is not None:
        radius = float(radius_raw)
        if radius <= 0.0:
            raise ValueError(f"inputs.temporal_real.radius must be > 0, got {radius}.")
        radius_source = "analysis_override"
        radius_estimation = None
    else:
        model_radius_raw = getattr(model_cfg.data, "radius", None)
        if model_radius_raw is not None and float(model_radius_raw) <= 0.0:
            raise ValueError(f"model_cfg.data.radius must be > 0, got {model_radius_raw}.")

        auto_cutoff_cfg_raw = OmegaConf.select(model_cfg, "data.auto_cutoff", default=None)
        auto_cutoff_cfg = PointCloudDataset._resolve_auto_cutoff_config(
            OmegaConf.to_container(auto_cutoff_cfg_raw, resolve=True) if auto_cutoff_cfg_raw is not None else None,
            default_target_points=int(num_points),
            default_radius=float(model_radius_raw) if model_radius_raw is not None else 0.0,
        )
        if auto_cutoff_cfg is not None:
            reference_frame_index = int(
                auto_cutoff_cfg.get("reference_frame_index", int(inference_frame_indices[0]))
            )
            radius_estimation = estimate_lammps_dump_cutoff_radius(
                dump_file,
                reference_frame_index=reference_frame_index,
                target_points=max(
                    int(num_points),
                    int(auto_cutoff_cfg.get("target_points", num_points)),
                ),
                quantile=float(auto_cutoff_cfg.get("quantile", 1.0)),
                estimation_samples=int(auto_cutoff_cfg.get("estimation_samples_per_file", 4096)),
                seed=int(auto_cutoff_cfg.get("seed", 0)),
                safety_factor=float(auto_cutoff_cfg.get("safety_factor", 1.0)),
                boundary_margin=auto_cutoff_cfg.get("boundary_margin", None),
                periodic=False,
            )
            radius = float(radius_estimation["estimated_radius"])
            radius_source = "auto_cutoff_static_style"
        elif model_radius_raw is not None:
            radius = float(model_radius_raw)
            radius_source = "model_data_radius"
            radius_estimation = None
        else:
            raise ValueError(
                "Temporal real analysis requires an explicit normalization radius. "
                "Set inputs.temporal_real.radius, model_cfg.data.radius, or enable model_cfg.data.auto_cutoff."
            )

    center_selection_cfg = OmegaConf.select(temporal_cfg, "center_selection", default=None)
    dataset_center_kwargs, center_selection_spec = _resolve_temporal_center_selection(
        temporal_cfg=temporal_cfg,
        center_selection_cfg=center_selection_cfg,
    )

    dataset = TemporalLAMMPSDumpDataset(
        dump_file=dump_file,
        cache_dir=cache_dir,
        sequence_length=sequence_length,
        num_points=num_points,
        radius=radius,
        frame_stride=frame_stride,
        anchor_frame_indices=inference_frame_indices.tolist(),
        anchor_source_names=inference_source_names,
        center_selection_seed=int(
            OmegaConf.select(temporal_cfg, "center_selection_seed", default=0)
        ),
        normalize=bool(OmegaConf.select(temporal_cfg, "normalize", default=True)),
        center_neighborhoods=bool(
            OmegaConf.select(temporal_cfg, "center_neighborhoods", default=True)
        ),
        selection_method=str(
            OmegaConf.select(temporal_cfg, "selection_method", default="closest")
        ),
        rebuild_cache=bool(OmegaConf.select(temporal_cfg, "rebuild_cache", default=False)),
        tree_cache_size=int(OmegaConf.select(temporal_cfg, "tree_cache_size", default=4)),
        **dataset_center_kwargs,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        num_workers=int(dataloader_num_workers),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        persistent_workers=bool(int(dataloader_num_workers) > 0),
        collate_fn=_temporal_identity_or_default_collate,
    )
    selection = TemporalRealAnalysisSelection(
        dump_file=dump_file,
        dump_summary=dump_summary,
        sequence_length=sequence_length,
        frame_stride=frame_stride,
        analysis_snapshot_count=analysis_snapshot_count,
        inference_snapshot_fraction=inference_snapshot_fraction,
        inference_frame_indices=inference_frame_indices.astype(np.int64, copy=False),
        analysis_frame_indices=analysis_frame_indices.astype(np.int64, copy=False),
        inference_source_names=[str(v) for v in inference_source_names],
        analysis_source_names=[str(v) for v in analysis_source_names],
        radius=float(radius),
        num_points=int(num_points),
        radius_source=str(radius_source),
        radius_estimation=None if radius_estimation is None else dict(radius_estimation),
        center_selection=dict(center_selection_spec),
        cache_dir=cache_dir,
    )
    return TemporalRealAnalysisBundle(
        dataset=dataset,
        dataloader=dataloader,
        selection=selection,
    )


def _equidistant_selection_indices(*, num_items: int, num_selected: int) -> np.ndarray:
    if num_items <= 0:
        raise ValueError(f"num_items must be > 0, got {num_items}.")
    if num_selected <= 0:
        raise ValueError(f"num_selected must be > 0, got {num_selected}.")
    if num_selected > num_items:
        raise ValueError(
            f"num_selected ({num_selected}) cannot exceed num_items ({num_items})."
        )
    if num_selected == 1:
        return np.asarray([0], dtype=np.int64)
    raw = np.linspace(0, num_items - 1, num=num_selected, dtype=np.float64)
    rounded = np.rint(raw).astype(np.int64)
    unique = np.unique(rounded)
    if unique.size != num_selected:
        raise RuntimeError(
            "Equidistant snapshot selection produced duplicate indices. "
            f"num_items={num_items}, num_selected={num_selected}, raw={raw.tolist()}, "
            f"rounded={rounded.tolist()}."
        )
    return unique.astype(np.int64, copy=False)


def _format_temporal_snapshot_name(
    *,
    frame_index: int,
    timestep: int,
    time_scale: float | None,
    time_unit: str | None,
) -> str:
    if time_scale is None or time_unit is None or time_unit == "":
        return f"frame_{int(frame_index):03d}_t{int(timestep)}"
    value = float(timestep) / float(time_scale)
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}{time_unit}"
    return f"{value:g}{time_unit}"


def _as_list_of_int(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [int(value)]
    return [int(v) for v in list(value)]


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _resolve_temporal_center_selection(
    *,
    temporal_cfg: Any,
    center_selection_cfg: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if center_selection_cfg is None:
        center_atom_ids = _as_list_of_int(
            OmegaConf.select(temporal_cfg, "center_atom_ids", default=None)
        )
        center_atom_stride = _optional_int(
            OmegaConf.select(temporal_cfg, "center_atom_stride", default=None)
        )
        max_center_atoms = _optional_int(
            OmegaConf.select(temporal_cfg, "max_center_atoms", default=None)
        )
        return (
            {
                "center_atom_ids": center_atom_ids,
                "center_atom_stride": center_atom_stride,
                "max_center_atoms": max_center_atoms,
            },
            {
                "mode": (
                    "atom_ids"
                    if center_atom_ids is not None
                    else "atom_stride"
                    if center_atom_stride is not None
                    else "random_subset"
                    if max_center_atoms is not None
                    else "unspecified"
                ),
                "center_atom_ids_count": None if center_atom_ids is None else int(len(center_atom_ids)),
                "center_atom_stride": None if center_atom_stride is None else int(center_atom_stride),
                "max_center_atoms": None if max_center_atoms is None else int(max_center_atoms),
            },
        )

    mode = str(OmegaConf.select(center_selection_cfg, "mode", default="regular_grid")).strip().lower()
    if mode == "regular_grid":
        grid_cfg = OmegaConf.select(center_selection_cfg, "regular_grid", default=None)
        overlap_raw = OmegaConf.select(grid_cfg, "overlap", default=1.0)
        overlap = float(overlap_raw)
        if overlap >= 2.0:
            raise ValueError(
                "inputs.temporal_real.center_selection.regular_grid.overlap must be < 2.0 "
                "so the derived center spacing stays positive. "
                f"Got {overlap}."
            )
        if OmegaConf.select(grid_cfg, "max_centers", default=None) is not None:
            raise ValueError(
                "inputs.temporal_real.center_selection.regular_grid.max_centers is no longer supported. "
                "Use regular_grid.overlap only to control temporal grid density."
            )
        reference_frame_raw = OmegaConf.select(grid_cfg, "reference_frame", default=None)
        if reference_frame_raw in {None, "", "first_anchor"}:
            reference_frame_index = None
            reference_frame_spec = "first_anchor"
        else:
            reference_frame_index = int(reference_frame_raw)
            reference_frame_spec = int(reference_frame_index)
        return (
            {
                "center_selection_mode": "regular_grid",
                "center_grid_overlap": float(overlap),
                "center_grid_reference_frame_index": reference_frame_index,
            },
            {
                "mode": "regular_grid",
                "overlap": float(overlap),
                "semantics": "sphere_overlap_depth_in_radius_units",
                "reference_frame": reference_frame_spec,
            },
        )

    if mode == "atom_ids":
        center_atom_ids = _as_list_of_int(
            OmegaConf.select(center_selection_cfg, "atom_ids", default=None)
        )
        return (
            {
                "center_selection_mode": "atom_ids",
                "center_atom_ids": center_atom_ids,
            },
            {
                "mode": "atom_ids",
                "center_atom_ids_count": None if center_atom_ids is None else int(len(center_atom_ids)),
            },
        )

    if mode == "atom_stride":
        center_atom_stride = _optional_int(
            OmegaConf.select(center_selection_cfg, "atom_stride", default=None)
        )
        return (
            {
                "center_selection_mode": "atom_stride",
                "center_atom_stride": center_atom_stride,
            },
            {
                "mode": "atom_stride",
                "center_atom_stride": None if center_atom_stride is None else int(center_atom_stride),
            },
        )

    if mode == "random_subset":
        max_center_atoms = _optional_int(
            OmegaConf.select(center_selection_cfg, "max_centers", default=None)
        )
        return (
            {
                "center_selection_mode": "random_subset",
                "max_center_atoms": max_center_atoms,
            },
            {
                "mode": "random_subset",
                "max_center_atoms": None if max_center_atoms is None else int(max_center_atoms),
            },
        )

    raise ValueError(
        "inputs.temporal_real.center_selection.mode must be one of "
        "['regular_grid', 'atom_ids', 'atom_stride', 'random_subset'], "
        f"got {mode!r}."
    )
