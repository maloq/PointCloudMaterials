from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.cluster_colors import _cluster_label_color, _cluster_palette, _compute_center_to_edge_colors
from src.analysis.cluster_geometry import _compute_pca_orientation_basis, _draw_edges
from src.analysis.cluster_rendering import (
    _build_knn_representative_edges,
    _compute_structure_half_span,
)
from src.analysis.config import build_runtime_model_config, load_checkpoint_analysis_config, _resolve_figure_set_settings
from src.analysis.temporal_real import _resolve_temporal_center_selection
from src.data_utils.data_load import PointCloudDataset
from src.data_utils.temporal_lammps_dataset import TemporalLAMMPSDumpDataset, estimate_lammps_dump_cutoff_radius

_RENDER_KNN_K = 4
_SPACE_MARGIN_FACTOR = 1.55


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render representative-style contact sheets showing how several tracked "
            "local structures evolve across temporal windows."
        )
    )
    parser.add_argument(
        "--analysis-config",
        default="configs/analysis/checkpoint_analysis_temporal_Al.yaml",
        help="Analysis config with an inputs.temporal_real block.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output directory override. Default: "
            "<analysis checkpoint.output_dir>/temporal_local_evolution."
        ),
    )
    parser.add_argument(
        "--horizons",
        default="3,6,24,120",
        help="Comma-separated sequence lengths to render.",
    )
    parser.add_argument(
        "--num-structures",
        type=int,
        default=4,
        help="Number of local structures to render when center atom ids are not supplied.",
    )
    parser.add_argument(
        "--center-atom-id",
        type=int,
        action="append",
        default=None,
        help="Explicit tracked atom id to render. Pass multiple times to control the exact set.",
    )
    parser.add_argument(
        "--anchor-frame-index",
        type=int,
        default=None,
        help=(
            "Window start frame to render. Default: midpoint of the valid range for the "
            "largest requested horizon."
        ),
    )
    parser.add_argument(
        "--max-display-frames",
        type=int,
        default=6,
        help="Maximum number of frames shown in each PNG contact sheet.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=260,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--gif-duration-ms",
        type=int,
        default=900,
        help="Per-frame duration for GIF animations in milliseconds.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force a rebuild of the temporal dump cache before rendering.",
    )
    return parser


def _parse_horizons(raw_value: str) -> list[int]:
    parts = [part.strip() for part in str(raw_value).split(",") if part.strip()]
    if not parts:
        raise ValueError("At least one horizon must be provided.")
    horizons = sorted({int(part) for part in parts})
    if any(value <= 0 for value in horizons):
        raise ValueError(f"All horizons must be > 0, got {horizons}.")
    return horizons


def _resolve_primary_k(analysis_cfg: Any) -> int:
    primary_k = OmegaConf.select(analysis_cfg, "clustering.primary_k", default=None)
    if primary_k is not None:
        return int(primary_k)
    k_values = OmegaConf.select(analysis_cfg, "clustering.k_values", default=None)
    if k_values is None or len(k_values) == 0:
        raise ValueError("Analysis config must define clustering.primary_k or clustering.k_values.")
    return int(list(k_values)[0])


def _resolve_output_dir(*, args: argparse.Namespace, analysis_cfg: Any) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir).expanduser().resolve()
    checkpoint_out = OmegaConf.select(analysis_cfg, "checkpoint.output_dir", default=None)
    if checkpoint_out is None or str(checkpoint_out).strip() == "":
        raise ValueError("checkpoint.output_dir must be set when --output-dir is not provided.")
    return Path(str(checkpoint_out)).expanduser().resolve() / "temporal_local_evolution"


def _resolve_temporal_render_setup(
    *,
    analysis_cfg: Any,
    model_cfg: Any,
    anchor_frame_index: int,
) -> dict[str, Any]:
    temporal_cfg = OmegaConf.select(analysis_cfg, "inputs.temporal_real", default=None)
    if temporal_cfg is None:
        raise ValueError("Analysis config is missing inputs.temporal_real.")

    dump_file_raw = OmegaConf.select(temporal_cfg, "dump_file", default=None)
    if dump_file_raw is None or str(dump_file_raw).strip() == "":
        raise ValueError("inputs.temporal_real.dump_file is required.")
    dump_file = Path(str(dump_file_raw)).expanduser().resolve()

    cache_dir_raw = OmegaConf.select(temporal_cfg, "cache_dir", default=None)
    cache_dir = None if cache_dir_raw in {None, ""} else Path(str(cache_dir_raw)).expanduser().resolve()

    frame_stride = int(OmegaConf.select(temporal_cfg, "frame_stride", default=1))
    if frame_stride <= 0:
        raise ValueError(f"inputs.temporal_real.frame_stride must be > 0, got {frame_stride}.")

    time_scale_raw = OmegaConf.select(temporal_cfg, "time_scale", default=None)
    time_unit_raw = OmegaConf.select(temporal_cfg, "time_unit", default=None)
    time_scale = None if time_scale_raw is None else float(time_scale_raw)
    time_unit = None if time_unit_raw is None else str(time_unit_raw).strip()

    num_points_raw = OmegaConf.select(temporal_cfg, "num_points", default=None)
    if num_points_raw is None:
        model_points = getattr(model_cfg.data, "num_points", None)
        if model_points is None:
            raise ValueError(
                "Temporal evolution rendering could not resolve num_points. "
                "Set inputs.temporal_real.num_points or model_cfg.data.num_points."
            )
        num_points = int(model_points)
    else:
        num_points = int(num_points_raw)
    if num_points <= 0:
        raise ValueError(f"Resolved num_points must be > 0, got {num_points}.")

    radius_raw = OmegaConf.select(temporal_cfg, "radius", default=None)
    radius_estimation = None
    if radius_raw is not None:
        radius = float(radius_raw)
        if radius <= 0.0:
            raise ValueError(f"inputs.temporal_real.radius must be > 0, got {radius}.")
        radius_source = "analysis_override"
    else:
        model_radius_raw = getattr(model_cfg.data, "radius", None)
        if model_radius_raw is not None and float(model_radius_raw) <= 0.0:
            raise ValueError(f"model_cfg.data.radius must be > 0, got {model_radius_raw}.")
        auto_cutoff_cfg_raw = OmegaConf.select(model_cfg, "data.auto_cutoff", default=None)
        auto_cutoff_cfg = PointCloudDataset._resolve_auto_cutoff_config(
            OmegaConf.to_container(auto_cutoff_cfg_raw, resolve=True)
            if auto_cutoff_cfg_raw is not None
            else None,
            default_target_points=int(num_points),
            default_radius=float(model_radius_raw) if model_radius_raw is not None else 0.0,
        )
        if auto_cutoff_cfg is not None:
            reference_frame_index = int(auto_cutoff_cfg.get("reference_frame_index", anchor_frame_index))
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
        else:
            raise ValueError(
                "Temporal evolution rendering requires an explicit normalization radius. "
                "Set inputs.temporal_real.radius, model_cfg.data.radius, or enable model_cfg.data.auto_cutoff."
            )

    center_selection_cfg = OmegaConf.select(temporal_cfg, "center_selection", default=None)
    dataset_center_kwargs, center_selection_spec = _resolve_temporal_center_selection(
        temporal_cfg=temporal_cfg,
        center_selection_cfg=center_selection_cfg,
    )

    return {
        "dump_file": dump_file,
        "cache_dir": cache_dir,
        "frame_stride": frame_stride,
        "time_scale": time_scale,
        "time_unit": time_unit,
        "num_points": int(num_points),
        "radius": float(radius),
        "radius_source": str(radius_source),
        "radius_estimation": None if radius_estimation is None else dict(radius_estimation),
        "dataset_center_kwargs": dataset_center_kwargs,
        "center_selection_spec": center_selection_spec,
        "center_selection_seed": int(OmegaConf.select(temporal_cfg, "center_selection_seed", default=0)),
        "normalize": bool(OmegaConf.select(temporal_cfg, "normalize", default=True)),
        "center_neighborhoods": bool(OmegaConf.select(temporal_cfg, "center_neighborhoods", default=True)),
        "selection_method": str(
            OmegaConf.select(temporal_cfg, "selection_method", default="closest")
        ),
        "tree_cache_size": int(OmegaConf.select(temporal_cfg, "tree_cache_size", default=4)),
    }


def _resolve_anchor_frame_index(
    *,
    scan: Any,
    max_horizon: int,
    frame_stride: int,
    requested: int | None,
) -> int:
    max_valid_anchor = int(scan.frame_count) - (int(max_horizon) - 1) * int(frame_stride) - 1
    if max_valid_anchor < 0:
        raise ValueError(
            "Requested horizons do not fit inside the dump file. "
            f"frame_count={int(scan.frame_count)}, max_horizon={int(max_horizon)}, frame_stride={int(frame_stride)}."
        )
    if requested is None:
        return int(max_valid_anchor // 2)
    anchor = int(requested)
    if anchor < 0 or anchor > max_valid_anchor:
        raise ValueError(
            "anchor_frame_index is out of range for the largest requested horizon. "
            f"anchor_frame_index={anchor}, valid_range=[0, {max_valid_anchor}], "
            f"frame_count={int(scan.frame_count)}, max_horizon={int(max_horizon)}, frame_stride={int(frame_stride)}."
        )
    return anchor


def _build_selection_pool_dataset(
    *,
    setup: dict[str, Any],
    sequence_length: int,
    anchor_frame_index: int,
    rebuild_cache: bool,
) -> TemporalLAMMPSDumpDataset:
    return TemporalLAMMPSDumpDataset(
        dump_file=setup["dump_file"],
        cache_dir=setup["cache_dir"],
        sequence_length=int(sequence_length),
        num_points=int(setup["num_points"]),
        radius=float(setup["radius"]),
        frame_stride=int(setup["frame_stride"]),
        anchor_frame_indices=[int(anchor_frame_index)],
        center_selection_seed=int(setup["center_selection_seed"]),
        normalize=bool(setup["normalize"]),
        center_neighborhoods=bool(setup["center_neighborhoods"]),
        selection_method=str(setup["selection_method"]),
        rebuild_cache=bool(rebuild_cache),
        tree_cache_size=int(setup["tree_cache_size"]),
        **dict(setup["dataset_center_kwargs"]),
    )


def _select_evenly_spaced_values(values: np.ndarray, count: int) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        raise ValueError("Cannot select from an empty array of values.")
    if count <= 0:
        raise ValueError(f"count must be > 0, got {count}.")
    if count >= arr.size:
        return arr.copy()
    raw = np.linspace(0, arr.size - 1, num=count, dtype=np.float64)
    idx = np.rint(raw).astype(np.int64)
    idx = np.unique(idx)
    if idx.size != count:
        idx = np.linspace(0, arr.size - 1, num=count, dtype=np.int64)
        idx = np.unique(idx)
    if idx.size != count:
        raise RuntimeError(
            "Failed to select a unique evenly spaced subset. "
            f"array_size={arr.size}, requested_count={count}, selected={idx.tolist()}."
        )
    return arr[idx]


def _resolve_selected_center_atom_ids(
    *,
    args: argparse.Namespace,
    selection_dataset: TemporalLAMMPSDumpDataset | None,
) -> tuple[list[int], str]:
    if args.center_atom_id is not None:
        selected = [int(value) for value in args.center_atom_id]
        if not selected:
            raise ValueError("Explicit center atom id list was empty.")
        return selected, "explicit_atom_ids"

    if selection_dataset is None:
        raise ValueError("selection_dataset is required when center atom ids are not supplied.")
    available_ids = np.asarray(
        selection_dataset.atom_ids[selection_dataset._center_atom_indices],
        dtype=np.int64,
    )
    if available_ids.size == 0:
        raise ValueError("Configured temporal center selection produced zero candidate atom ids.")
    picked = _select_evenly_spaced_values(available_ids, int(args.num_structures))
    return [int(value) for value in picked.tolist()], "configured_center_selection"


def _load_sequence_batch(dataset: TemporalLAMMPSDumpDataset) -> dict[str, np.ndarray]:
    indices = list(range(len(dataset)))
    if not indices:
        raise ValueError("Temporal sequence dataset produced zero samples for rendering.")
    batch = dataset.__getitems__(indices)
    points = np.asarray(batch["points"].detach().cpu().numpy(), dtype=np.float32)
    center_positions = np.asarray(batch["center_positions"].detach().cpu().numpy(), dtype=np.float32)
    frame_indices = np.asarray(batch["frame_indices"].detach().cpu().numpy(), dtype=np.int64)
    timesteps = np.asarray(batch["timesteps"].detach().cpu().numpy(), dtype=np.int64)
    center_atom_ids = np.asarray(batch["center_atom_id"].detach().cpu().numpy(), dtype=np.int64)
    return {
        "points": points,
        "center_positions": center_positions,
        "frame_indices": frame_indices,
        "timesteps": timesteps,
        "center_atom_id": center_atom_ids,
    }


def _select_display_frame_indices(sequence_length: int, max_display_frames: int) -> np.ndarray:
    length = int(sequence_length)
    limit = int(max_display_frames)
    if length <= 0:
        raise ValueError(f"sequence_length must be > 0, got {length}.")
    if limit <= 0:
        raise ValueError(f"max_display_frames must be > 0, got {limit}.")
    if length <= limit:
        return np.arange(length, dtype=np.int64)
    raw = np.linspace(0, length - 1, num=limit, dtype=np.float64)
    idx = np.rint(raw).astype(np.int64)
    idx = np.unique(idx)
    if idx.size != limit:
        raise RuntimeError(
            "Display frame subsampling produced duplicate frame indices. "
            f"sequence_length={length}, max_display_frames={limit}, indices={idx.tolist()}."
        )
    return idx.astype(np.int64, copy=False)


def _format_delta_time(
    *,
    anchor_timestep: int,
    current_timestep: int,
    time_scale: float | None,
    time_unit: str | None,
) -> str:
    if time_scale is None or time_unit is None or time_unit == "":
        return f"t={int(current_timestep)}"
    delta_value = (float(current_timestep) - float(anchor_timestep)) / float(time_scale)
    rounded = round(delta_value)
    if abs(delta_value - float(rounded)) < 1e-9:
        return f"+{int(rounded)} {time_unit}"
    return f"+{delta_value:g} {time_unit}"


def _prepare_local_points(points: np.ndarray, *, target_points: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected local points with shape (N, 3), got {pts.shape}.")
    keep_count = min(int(target_points), int(pts.shape[0]))
    distances = np.linalg.norm(pts, axis=1)
    keep = np.argsort(distances)[:keep_count]
    local = np.asarray(pts[keep], dtype=np.float32)
    if local.size == 0:
        raise ValueError("No local points remained after representative-style point selection.")
    return local


def _build_locked_orientation(
    *,
    anchor_points: np.ndarray,
    method: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    method_norm = str(method).strip().lower()
    if method_norm == "none":
        center_index = int(np.argmin(np.linalg.norm(anchor_points, axis=1)))
        return None, {"orientation_method": "none", "center_index": center_index}
    if method_norm != "pca":
        raise ValueError(
            "Unsupported representative orientation method. "
            f"Expected one of ['pca', 'none'], got {method!r}."
        )
    centered, basis, eigvals, center_index, det_basis = _compute_pca_orientation_basis(anchor_points)
    return np.asarray(basis, dtype=np.float32), {
        "orientation_method": "pca_locked_to_anchor",
        "anchor_center_index": int(center_index),
        "anchor_pca_eigenvalues": [float(value) for value in eigvals.tolist()],
        "anchor_pca_basis_det": float(det_basis),
        "anchor_half_span": float(_compute_structure_half_span(centered @ basis)),
    }


def _apply_locked_orientation(
    *,
    points: np.ndarray,
    basis: np.ndarray | None,
    method: str,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    center_index = int(np.argmin(np.linalg.norm(pts, axis=1)))
    centered = pts - pts[center_index]
    method_norm = str(method).strip().lower()
    if method_norm == "none":
        return centered.astype(np.float32, copy=False)
    if basis is None:
        raise ValueError("PCA orientation requested but no anchor basis was provided.")
    return (centered @ np.asarray(basis, dtype=np.float32)).astype(np.float32, copy=False)


def _prepare_structure_render_records(
    *,
    batch: dict[str, np.ndarray],
    target_points: int,
    orientation_cache: dict[int, dict[str, Any]] | None = None,
    space_margin_factor: float = _SPACE_MARGIN_FACTOR,
) -> list[dict[str, Any]]:
    points = np.asarray(batch["points"], dtype=np.float32)
    center_positions = np.asarray(batch["center_positions"], dtype=np.float32)
    frame_indices = np.asarray(batch["frame_indices"], dtype=np.int64)
    timesteps = np.asarray(batch["timesteps"], dtype=np.int64)
    center_atom_ids = np.asarray(batch["center_atom_id"], dtype=np.int64)

    num_structures = int(points.shape[0])
    base_colors = _cluster_palette(num_structures)
    records: list[dict[str, Any]] = []
    for structure_idx in range(num_structures):
        per_frame_points: list[np.ndarray] = []
        oriented_sequence: list[np.ndarray] = []
        half_spans: list[float] = []
        for frame_local_idx in range(int(points.shape[1])):
            local = _prepare_local_points(
                points[structure_idx, frame_local_idx],
                target_points=int(target_points),
            )
            per_frame_points.append(local)

        atom_id = int(center_atom_ids[structure_idx])
        if orientation_cache is None:
            anchor_points = per_frame_points[0]
            basis, orientation_info = _build_locked_orientation(
                anchor_points=anchor_points,
                method="none",
            )
        else:
            cached = orientation_cache.get(atom_id)
            if cached is None:
                raise KeyError(
                    "Missing shared orientation cache entry for selected center atom. "
                    f"center_atom_id={atom_id}."
                )
            basis = cached["basis"]
            orientation_info = dict(cached["orientation"])
        for frame_local_idx in range(int(points.shape[1])):
            oriented = _apply_locked_orientation(
                points=per_frame_points[frame_local_idx],
                basis=basis,
                method="none",
            )
            oriented_sequence.append(oriented)
            half_spans.append(_compute_structure_half_span(oriented))

        frame_records = [
            {
                "local_frame_index": int(frame_local_idx),
                "frame_index": int(frame_indices[structure_idx, frame_local_idx]),
                "timestep": int(timesteps[structure_idx, frame_local_idx]),
                "center_position": [
                    float(value)
                    for value in center_positions[structure_idx, frame_local_idx].tolist()
                ],
            }
            for frame_local_idx in range(int(points.shape[1]))
        ]
        records.append(
            {
                "center_atom_id": atom_id,
                "base_color": str(base_colors[structure_idx]),
                "orientation": dict(orientation_info),
                "oriented_sequence": oriented_sequence,
                "display_half_span": float(max(half_spans) * float(space_margin_factor)),
                "frames": frame_records,
            }
        )
    return records


def _build_shared_orientation_cache(
    *,
    setup: dict[str, Any],
    target_points: int,
    anchor_frame_index: int,
    selected_center_atom_ids: list[int],
    rebuild_cache: bool,
) -> dict[int, dict[str, Any]]:
    orientation_dataset = TemporalLAMMPSDumpDataset(
        dump_file=setup["dump_file"],
        cache_dir=setup["cache_dir"],
        sequence_length=1,
        num_points=int(setup["num_points"]),
        radius=float(setup["radius"]),
        frame_stride=int(setup["frame_stride"]),
        anchor_frame_indices=[int(anchor_frame_index)],
        center_selection_mode="atom_ids",
        center_atom_ids=[int(value) for value in selected_center_atom_ids],
        center_selection_seed=int(setup["center_selection_seed"]),
        normalize=bool(setup["normalize"]),
        center_neighborhoods=bool(setup["center_neighborhoods"]),
        selection_method=str(setup["selection_method"]),
        rebuild_cache=bool(rebuild_cache),
        tree_cache_size=int(setup["tree_cache_size"]),
    )
    batch = _load_sequence_batch(orientation_dataset)
    points = np.asarray(batch["points"], dtype=np.float32)
    center_atom_ids = np.asarray(batch["center_atom_id"], dtype=np.int64)

    orientation_cache: dict[int, dict[str, Any]] = {}
    for structure_idx, atom_id_value in enumerate(center_atom_ids.tolist()):
        anchor_points = _prepare_local_points(
            points[structure_idx, 0],
            target_points=int(target_points),
        )
        basis, orientation_info = _build_locked_orientation(
            anchor_points=anchor_points,
            method="none",
        )
        orientation_cache[int(atom_id_value)] = {
            "basis": None if basis is None else np.asarray(basis, dtype=np.float32),
            "orientation": dict(orientation_info),
        }

    expected_atom_ids = {int(value) for value in selected_center_atom_ids}
    cached_atom_ids = set(orientation_cache.keys())
    if cached_atom_ids != expected_atom_ids:
        raise RuntimeError(
            "Shared orientation cache does not match the requested center atom ids. "
            f"expected={sorted(expected_atom_ids)}, cached={sorted(cached_atom_ids)}."
        )
    return orientation_cache


def _set_origin_centered_axes_3d_with_half_span(
    ax: Any,
    *,
    half_span: float,
) -> None:
    half = float(half_span)
    if not np.isfinite(half) or half <= 0.0:
        raise ValueError(f"half_span must be positive and finite, got {half_span}.")
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(-half, half)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 1.0))


def _render_structure_axis(
    ax: Any,
    *,
    oriented_points: np.ndarray,
    base_color: str,
    half_span: float,
    view_elev: float,
    view_azim: float,
    projection: str,
    title_text: str | None = None,
) -> dict[str, Any]:
    ax.set_facecolor("white")
    if hasattr(ax, "set_proj_type"):
        ax.set_proj_type(str(projection))
    ax.view_init(elev=float(view_elev), azim=float(view_azim))

    point_colors = _compute_center_to_edge_colors(oriented_points, base_color)
    edges, edge_info = _build_knn_representative_edges(
        oriented_points,
        knn_k=int(_RENDER_KNN_K),
    )
    _draw_edges(
        ax,
        oriented_points,
        edges,
        point_colors=point_colors,
        edge_alpha=0.54,
        edge_linewidth=0.82,
    )
    ax.scatter(
        oriented_points[:, 0],
        oriented_points[:, 1],
        oriented_points[:, 2],
        c=point_colors,
        s=52.0,
        alpha=0.97,
        edgecolors="#222222",
        linewidths=0.32,
        depthshade=False,
    )
    _set_origin_centered_axes_3d_with_half_span(
        ax,
        half_span=float(half_span),
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        if hasattr(axis, "pane"):
            axis.pane.fill = False
            axis.pane.set_edgecolor((1.0, 1.0, 1.0, 1.0))
        if hasattr(axis, "line"):
            axis.line.set_color((1.0, 1.0, 1.0, 1.0))
    if title_text is not None:
        ax.set_title(title_text, fontsize=8, pad=2.0)
    return dict(edge_info)


def _figure_to_pil_image(fig: Any) -> Image.Image:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as image:
        converted = image.convert("RGB").copy()
    buffer.close()
    return converted


def _render_structure_png(
    *,
    horizon: int,
    structure_record: dict[str, Any],
    figure_settings: Any,
    time_scale: float | None,
    time_unit: str | None,
    max_display_frames: int,
    out_file: Path,
    dpi: int,
) -> dict[str, Any]:
    display_indices = _select_display_frame_indices(int(horizon), int(max_display_frames))
    displayed_count = int(display_indices.size)
    display_frames = [structure_record["frames"][int(idx)] for idx in display_indices.tolist()]
    base_color = str(structure_record["base_color"])
    label_color = _cluster_label_color(base_color, darken_factor=0.68)

    fig_width = 2.25 * displayed_count + 0.8
    fig_height = 3.0
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=int(dpi), facecolor="white")
    edge_summaries: list[dict[str, Any]] = []
    for panel_idx, frame_idx in enumerate(display_indices.tolist()):
        ax = fig.add_subplot(1, displayed_count, panel_idx + 1, projection="3d")
        time_label = _format_delta_time(
            anchor_timestep=int(structure_record["frames"][0]["timestep"]),
            current_timestep=int(structure_record["frames"][int(frame_idx)]["timestep"]),
            time_scale=time_scale,
            time_unit=time_unit,
        )
        edge_info = _render_structure_axis(
            ax,
            oriented_points=np.asarray(structure_record["oriented_sequence"][int(frame_idx)], dtype=np.float32),
            base_color=base_color,
            half_span=float(structure_record["display_half_span"]),
            view_elev=float(figure_settings.representative_view_elev),
            view_azim=float(figure_settings.representative_view_azim),
            projection=str(figure_settings.representative_projection),
            title_text=f"f+{int(frame_idx)}\n{time_label}",
        )
        edge_summaries.append(
            {
                "local_frame_index": int(frame_idx),
                "edge_info": dict(edge_info),
            }
        )

    shown_note = (
        f"showing all {int(horizon)} frames"
        if displayed_count == int(horizon)
        else f"showing {displayed_count} evenly spaced frames out of {int(horizon)}"
    )
    fig.suptitle(
        f"atom {int(structure_record['center_atom_id'])} | horizon={int(horizon)}",
        fontsize=11,
        fontweight="bold",
        color=label_color,
        y=0.985,
    )
    fig.text(0.5, 0.025, shown_note, ha="center", va="bottom", fontsize=8, color="#444444")
    fig.subplots_adjust(left=0.02, right=0.995, bottom=0.11, top=0.86, wspace=0.02)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file)
    plt.close(fig)
    print(f"[temporal-local-evolution][png] {out_file.resolve()}")

    return {
        "png_file": str(out_file),
        "display_local_frame_indices": [int(value) for value in display_indices.tolist()],
        "display_frames": [
            {
                **dict(frame_record),
                "edge_info": dict(edge_summary["edge_info"]),
            }
            for frame_record, edge_summary in zip(display_frames, edge_summaries, strict=True)
        ],
    }


def _render_structure_gif(
    *,
    horizon: int,
    structure_record: dict[str, Any],
    figure_settings: Any,
    time_scale: float | None,
    time_unit: str | None,
    gif_duration_ms: int,
    out_file: Path,
    dpi: int,
) -> dict[str, Any]:
    if gif_duration_ms <= 0:
        raise ValueError(f"gif_duration_ms must be > 0, got {gif_duration_ms}.")

    base_color = str(structure_record["base_color"])
    label_color = _cluster_label_color(base_color, darken_factor=0.68)
    pil_frames: list[Image.Image] = []
    frame_summaries: list[dict[str, Any]] = []
    for frame_record, oriented_points in zip(
        structure_record["frames"],
        structure_record["oriented_sequence"],
        strict=True,
    ):
        fig = plt.figure(figsize=(3.45, 3.5), dpi=int(dpi), facecolor="white")
        ax = fig.add_subplot(111, projection="3d")
        time_label = _format_delta_time(
            anchor_timestep=int(structure_record["frames"][0]["timestep"]),
            current_timestep=int(frame_record["timestep"]),
            time_scale=time_scale,
            time_unit=time_unit,
        )
        edge_info = _render_structure_axis(
            ax,
            oriented_points=np.asarray(oriented_points, dtype=np.float32),
            base_color=base_color,
            half_span=float(structure_record["display_half_span"]),
            view_elev=float(figure_settings.representative_view_elev),
            view_azim=float(figure_settings.representative_view_azim),
            projection=str(figure_settings.representative_projection),
        )
        fig.suptitle(
            f"atom {int(structure_record['center_atom_id'])}",
            fontsize=11,
            fontweight="bold",
            color=label_color,
            y=0.965,
        )
        fig.text(
            0.5,
            0.05,
            f"horizon={int(horizon)} | f+{int(frame_record['local_frame_index'])} | {time_label}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#444444",
        )
        fig.subplots_adjust(left=0.02, right=0.985, bottom=0.09, top=0.88)
        pil_frames.append(_figure_to_pil_image(fig))
        frame_summaries.append(
            {
                **dict(frame_record),
                "edge_info": dict(edge_info),
            }
        )

    if not pil_frames:
        raise RuntimeError(
            f"No GIF frames were rendered for center_atom_id={structure_record['center_atom_id']}."
        )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        out_file,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(gif_duration_ms),
        loop=0,
        disposal=2,
    )
    for image in pil_frames:
        image.close()
    print(f"[temporal-local-evolution][gif] {out_file.resolve()}")

    return {
        "gif_file": str(out_file),
        "gif_duration_ms": int(gif_duration_ms),
        "frames": frame_summaries,
    }


def main() -> None:
    args = _build_parser().parse_args()
    horizons = _parse_horizons(args.horizons)
    if args.num_structures <= 0:
        raise ValueError(f"--num-structures must be > 0, got {args.num_structures}.")

    analysis_cfg = load_checkpoint_analysis_config(args.analysis_config)
    checkpoint_path = OmegaConf.select(analysis_cfg, "checkpoint.path", default=None)
    if checkpoint_path is None or str(checkpoint_path).strip() == "":
        raise ValueError("Analysis config must define checkpoint.path.")
    model_cfg = build_runtime_model_config(str(checkpoint_path), analysis_cfg)
    primary_k = _resolve_primary_k(analysis_cfg)
    output_dir = _resolve_output_dir(args=args, analysis_cfg=analysis_cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    temporal_cfg = OmegaConf.select(analysis_cfg, "inputs.temporal_real", default=None)
    if temporal_cfg is None:
        raise ValueError("Analysis config must define inputs.temporal_real.")

    dump_file_raw = OmegaConf.select(temporal_cfg, "dump_file", default=None)
    if dump_file_raw is None or str(dump_file_raw).strip() == "":
        raise ValueError("inputs.temporal_real.dump_file is required.")
    dump_file = Path(str(dump_file_raw)).expanduser().resolve()
    scan = TemporalLAMMPSDumpDataset.scan_dump_file(dump_file)

    frame_stride = int(OmegaConf.select(temporal_cfg, "frame_stride", default=1))
    max_horizon = max(int(value) for value in horizons)
    anchor_frame_index = _resolve_anchor_frame_index(
        scan=scan,
        max_horizon=max_horizon,
        frame_stride=frame_stride,
        requested=args.anchor_frame_index,
    )
    setup = _resolve_temporal_render_setup(
        analysis_cfg=analysis_cfg,
        model_cfg=model_cfg,
        anchor_frame_index=anchor_frame_index,
    )
    figure_settings = _resolve_figure_set_settings(
        analysis_cfg=analysis_cfg,
        model_cfg=model_cfg,
        primary_k=int(primary_k),
        out_dir=output_dir,
    )

    selection_dataset = None
    if args.center_atom_id is None:
        selection_dataset = _build_selection_pool_dataset(
            setup=setup,
            sequence_length=max_horizon,
            anchor_frame_index=anchor_frame_index,
            rebuild_cache=bool(args.rebuild_cache),
        )
    selected_center_atom_ids, center_selection_source = _resolve_selected_center_atom_ids(
        args=args,
        selection_dataset=selection_dataset,
    )
    shared_orientation_cache = _build_shared_orientation_cache(
        setup=setup,
        target_points=int(figure_settings.representative_points),
        anchor_frame_index=anchor_frame_index,
        selected_center_atom_ids=selected_center_atom_ids,
        rebuild_cache=bool(args.rebuild_cache) and selection_dataset is None,
    )

    manifest: dict[str, Any] = {
        "analysis_config": str(Path(args.analysis_config).expanduser().resolve()),
        "checkpoint_path": str(Path(str(checkpoint_path)).expanduser().resolve()),
        "dump_file": str(setup["dump_file"]),
        "output_dir": str(output_dir),
        "horizons": [int(value) for value in horizons],
        "anchor_frame_index": int(anchor_frame_index),
        "frame_stride": int(setup["frame_stride"]),
        "num_points": int(setup["num_points"]),
        "radius": float(setup["radius"]),
        "radius_source": str(setup["radius_source"]),
        "radius_estimation": None
        if setup["radius_estimation"] is None
        else dict(setup["radius_estimation"]),
        "center_selection_source": str(center_selection_source),
        "configured_center_selection": dict(setup["center_selection_spec"]),
        "selected_center_atom_ids": [int(value) for value in selected_center_atom_ids],
        "representative_style": {
            "representative_points": int(figure_settings.representative_points),
            "orientation": "original_coordinates",
            "orientation_reference": "no_rotation_applied",
            "view_elev": float(figure_settings.representative_view_elev),
            "view_azim": float(figure_settings.representative_view_azim),
            "projection": str(figure_settings.representative_projection),
            "knn_k": int(_RENDER_KNN_K),
            "style_reference": "09_cluster_representatives_knn_edges_k8.png_like_with_k4",
            "space_margin_factor": float(_SPACE_MARGIN_FACTOR),
        },
        "outputs": [],
    }

    for horizon in horizons:
        horizon_dataset = TemporalLAMMPSDumpDataset(
            dump_file=setup["dump_file"],
            cache_dir=setup["cache_dir"],
            sequence_length=int(horizon),
            num_points=int(setup["num_points"]),
            radius=float(setup["radius"]),
            frame_stride=int(setup["frame_stride"]),
            anchor_frame_indices=[int(anchor_frame_index)],
            center_selection_mode="atom_ids",
            center_atom_ids=[int(value) for value in selected_center_atom_ids],
            center_selection_seed=int(setup["center_selection_seed"]),
            normalize=bool(setup["normalize"]),
            center_neighborhoods=bool(setup["center_neighborhoods"]),
            selection_method=str(setup["selection_method"]),
            rebuild_cache=False,
            tree_cache_size=int(setup["tree_cache_size"]),
        )
        batch = _load_sequence_batch(horizon_dataset)
        structure_records = _prepare_structure_render_records(
            batch=batch,
            target_points=int(figure_settings.representative_points),
            orientation_cache=shared_orientation_cache,
        )
        horizon_dir = output_dir / f"horizon_{int(horizon):03d}"
        horizon_summary = {
            "horizon": int(horizon),
            "num_structures": int(len(structure_records)),
            "structures": [],
        }
        for structure_record in structure_records:
            atom_id = int(structure_record["center_atom_id"])
            stem = f"temporal_local_evolution_atom_{atom_id:09d}"
            png_summary = _render_structure_png(
                horizon=int(horizon),
                structure_record=structure_record,
                figure_settings=figure_settings,
                time_scale=setup["time_scale"],
                time_unit=setup["time_unit"],
                max_display_frames=int(args.max_display_frames),
                out_file=horizon_dir / f"{stem}.png",
                dpi=int(args.dpi),
            )
            gif_summary = _render_structure_gif(
                horizon=int(horizon),
                structure_record=structure_record,
                figure_settings=figure_settings,
                time_scale=setup["time_scale"],
                time_unit=setup["time_unit"],
                gif_duration_ms=int(args.gif_duration_ms),
                out_file=horizon_dir / f"{stem}.gif",
                dpi=int(args.dpi),
            )
            horizon_summary["structures"].append(
                {
                    "center_atom_id": int(atom_id),
                    "base_color": str(structure_record["base_color"]),
                    "orientation": dict(structure_record["orientation"]),
                    "display_half_span": float(structure_record["display_half_span"]),
                    "frame_indices_full": [
                        int(frame_record["frame_index"])
                        for frame_record in structure_record["frames"]
                    ],
                    "timesteps_full": [
                        int(frame_record["timestep"])
                        for frame_record in structure_record["frames"]
                    ],
                    "png": png_summary,
                    "gif": gif_summary,
                }
            )
        manifest["outputs"].append(horizon_summary)

    manifest_path = output_dir / "temporal_local_evolution_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[temporal-local-evolution][manifest] {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
