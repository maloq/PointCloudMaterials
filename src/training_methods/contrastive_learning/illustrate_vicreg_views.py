"""Render a paper-style illustration of VICReg view creation for atomistic samples.

The figure shows a deterministic, paper-friendly approximation of the path

1. raw radius crop around one atom,
2. fixed-size subset downsampled from that crop,
3. the same subset translated so a nearby atom becomes the new origin and then
   cropped to the VICReg view size.

The rendering reuses the same neighborhood styling helpers used by
`figure_local_base_paper_colormap.png` in the synthetic visualization module.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
import yaml

sys.path.append(os.getcwd())

from src.data_utils.synthetic.visualization import (
    _build_local_coordination_edges,
    _compute_radial_colormap_colors,
    _draw_edges,
    _ensure_connected_edges,
    _save_structure_xyz,
)


_INTERPOLATION_PATTERN = re.compile(r"\$\{([^}]+)\}")
_DROPPED_POINT_EDGE_COLOR = "#5A5A5A"


def _resolve_config_path(config_arg: str) -> Path:
    candidate = Path(str(config_arg))
    if candidate.exists():
        return candidate.resolve()

    configs_root = (Path.cwd() / "configs").resolve()
    candidate = configs_root / str(config_arg)
    if candidate.exists():
        return candidate.resolve()

    if candidate.suffix == "":
        yaml_candidate = candidate.with_suffix(".yaml")
        if yaml_candidate.exists():
            return yaml_candidate.resolve()

    raise FileNotFoundError(
        "Could not resolve config path. "
        f"Tried {config_arg!r} relative to the current working directory and to {configs_root}."
    )


def _cfg_get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = cfg
    for part in str(path).split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _deep_merge_dicts(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_interpolations(cfg: dict[str, Any]) -> dict[str, Any]:
    root = cfg

    def _resolve_path(path: str, stack: tuple[str, ...]) -> Any:
        if path in stack:
            raise ValueError(
                "Config interpolation cycle detected: "
                f"{' -> '.join(stack + (path,))}"
            )
        current: Any = root
        for part in str(path).split("."):
            if not isinstance(current, dict) or part not in current:
                raise KeyError(f"Failed to resolve interpolation path {path!r}.")
            current = current[part]
        return _resolve_node(current, stack + (path,))

    def _resolve_string(value: str, stack: tuple[str, ...]) -> Any:
        matches = list(_INTERPOLATION_PATTERN.finditer(value))
        if not matches:
            return value
        if len(matches) == 1 and matches[0].span() == (0, len(value)):
            return _resolve_path(matches[0].group(1), stack)

        resolved_text = value
        for match in matches:
            resolved = _resolve_path(match.group(1), stack)
            resolved_text = resolved_text.replace(match.group(0), str(resolved))
        return resolved_text

    def _resolve_node(node: Any, stack: tuple[str, ...]) -> Any:
        if isinstance(node, dict):
            return {key: _resolve_node(val, stack) for key, val in node.items()}
        if isinstance(node, list):
            return [_resolve_node(val, stack) for val in node]
        if isinstance(node, str):
            return _resolve_string(node, stack)
        return node

    return _resolve_node(root, tuple())


def _load_hydra_config(config_path: Path) -> dict[str, Any]:
    configs_root = (Path.cwd() / "configs").resolve()
    config_path = config_path.resolve()
    if configs_root not in config_path.parents and config_path.parent != configs_root:
        raise ValueError(
            "Config must live inside the repository configs/ tree so Hydra defaults can be composed. "
            f"Got {config_path} outside {configs_root}."
        )

    def _resolve_default_path(group: str, name: str) -> Path:
        group_norm = str(group).strip().lstrip("/")
        name_norm = str(name).strip()
        if name_norm == "":
            raise ValueError(f"Config default for group {group!r} must be non-empty.")
        base = configs_root / group_norm
        if name_norm.endswith(".yaml") or name_norm.endswith(".yml"):
            return (base / name_norm).resolve()
        return (base / f"{name_norm}.yaml").resolve()

    def _compose_file(path: Path) -> dict[str, Any]:
        with path.open("r") as handle:
            cfg = yaml.safe_load(handle) or {}
        if not isinstance(cfg, dict):
            raise ValueError(f"Config at {path} must decode to a mapping, got {type(cfg)!r}.")
        defaults = cfg.get("defaults", [])
        self_cfg = dict(cfg)
        self_cfg.pop("defaults", None)

        if defaults is None:
            defaults_list: list[Any] = []
        else:
            defaults_list = list(defaults)
        if "_self_" not in defaults_list:
            defaults_list.append("_self_")

        merged: dict[str, Any] = {}
        for entry in defaults_list:
            if entry == "_self_":
                merged = _deep_merge_dicts(merged, self_cfg)
                continue
            if isinstance(entry, str):
                raise ValueError(
                    f"Unsupported string default entry {entry!r} in {path}. "
                    "Only '_self_' and single-key mapping defaults are supported here."
                )
            if not isinstance(entry, dict) or len(entry) != 1:
                raise ValueError(
                    f"Unsupported defaults entry {entry!r} in {path}. "
                    "Expected a single-key mapping such as {'data': 'data_ae_multi_material'}."
                )
            group, name = next(iter(entry.items()))
            if name is None:
                continue
            default_path = _resolve_default_path(str(group), str(name))
            if not default_path.exists():
                raise FileNotFoundError(
                    f"Config default {entry!r} referenced by {path} does not exist at {default_path}."
                )
            default_cfg = _compose_file(default_path)
            merged = _deep_merge_dicts(merged, {str(group).lstrip("/"): default_cfg})
        return merged

    unresolved = _compose_file(config_path)
    return _resolve_interpolations(unresolved)


def _load_point_cloud(file_path: Path) -> np.ndarray:
    if not file_path.exists():
        raise FileNotFoundError(f"Point-cloud file does not exist: {file_path}")
    raw = np.load(file_path, allow_pickle=False)
    if isinstance(raw, np.ndarray) and raw.dtype.names:
        if "position" not in raw.dtype.names:
            raise ValueError(
                f"Structured array at {file_path} is missing a 'position' field: fields={raw.dtype.names}"
            )
        points = np.asarray(raw["position"], dtype=np.float32)
    else:
        points = np.asarray(raw, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected point cloud with shape (N, 3) at {file_path}, got {points.shape}.")
    if points.shape[0] < 2:
        raise ValueError(f"Point cloud at {file_path} must contain at least 2 atoms, got {points.shape[0]}.")
    if not np.isfinite(points).all():
        raise ValueError(f"Point cloud at {file_path} contains non-finite coordinates.")
    return points


def _resolve_auto_cutoff_config(
    auto_cutoff_config: dict[str, Any] | None,
    *,
    default_target_points: int,
    default_radius: float,
) -> dict[str, Any] | None:
    if not auto_cutoff_config or not auto_cutoff_config.get("enabled", False):
        return None
    return {
        "target_points": int(auto_cutoff_config.get("target_points", default_target_points)),
        "quantile": float(auto_cutoff_config.get("quantile", 1.0)),
        "estimation_samples_per_file": int(auto_cutoff_config.get("estimation_samples_per_file", 4096)),
        "seed": int(auto_cutoff_config.get("seed", 0)),
        "safety_factor": float(auto_cutoff_config.get("safety_factor", 1.0)),
        "boundary_margin": auto_cutoff_config.get("boundary_margin", default_radius),
    }


def _estimate_source_cutoff_radius(
    *,
    source_root: Path,
    source_files: Sequence[str],
    target_points: int,
    quantile: float,
    estimation_samples_per_file: int,
    seed: int,
    safety_factor: float,
    boundary_margin: float | None,
) -> tuple[float, float]:
    rng = np.random.default_rng(int(seed))
    kth_distances_all: list[np.ndarray] = []

    for file_name in source_files:
        file_path = source_root / str(file_name)
        points = _load_point_cloud(file_path)
        num_atoms = int(points.shape[0])
        candidate_indices = np.arange(num_atoms, dtype=np.int64)
        if boundary_margin is not None and float(boundary_margin) > 0.0:
            margin = float(boundary_margin)
            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)
            interior_mask = np.all(
                (points >= (min_coords + margin)) & (points <= (max_coords - margin)),
                axis=1,
            )
            interior_indices = np.flatnonzero(interior_mask)
            if interior_indices.size > 0:
                candidate_indices = interior_indices.astype(np.int64, copy=False)

        centers_to_sample = min(int(estimation_samples_per_file), int(candidate_indices.size))
        if centers_to_sample <= 0:
            raise ValueError(
                "Auto-cutoff estimation found no candidate center atoms after boundary filtering for "
                f"{file_path} with boundary_margin={boundary_margin}."
            )
        center_indices = rng.choice(candidate_indices, size=centers_to_sample, replace=False)
        tree = cKDTree(points)
        k = min(int(target_points), num_atoms)
        dists, _ = tree.query(points[center_indices], k=k)
        dists = np.asarray(dists, dtype=np.float64)
        kth_dist = dists.reshape(-1) if k == 1 else dists[:, k - 1]
        kth_distances_all.append(kth_dist)

    if not kth_distances_all:
        raise RuntimeError("Auto-cutoff estimation did not collect any kth-neighbor distances.")
    kth_all = np.concatenate(kth_distances_all).astype(np.float64, copy=False)
    estimated_radius = float(np.quantile(kth_all, float(quantile))) * float(safety_factor)
    coverage = float(np.mean(kth_all <= estimated_radius))
    return estimated_radius, coverage


def _resolve_real_source_radius(
    data_cfg: dict[str, Any],
    *,
    source_index: int,
    source_dict: dict[str, Any],
    target_points: int,
) -> tuple[float, dict[str, Any] | None]:
    radius_override = source_dict.get("radius", None)
    if radius_override is not None:
        return float(radius_override), {
            "mode": "override",
            "radius_override": float(radius_override),
        }

    default_radius = float(data_cfg.get("radius", 0.0))
    auto_cfg = _resolve_auto_cutoff_config(
        data_cfg.get("auto_cutoff", None),
        default_target_points=int(target_points),
        default_radius=default_radius,
    )
    if auto_cfg is None:
        return default_radius, None

    data_files = source_dict.get("data_files", [])
    if isinstance(data_files, str):
        data_files = [data_files]
    estimated_radius, coverage = _estimate_source_cutoff_radius(
        source_root=Path(str(source_dict["data_path"])),
        source_files=list(data_files),
        target_points=max(int(auto_cfg["target_points"]), int(target_points)),
        quantile=float(auto_cfg["quantile"]),
        estimation_samples_per_file=int(auto_cfg["estimation_samples_per_file"]),
        seed=int(auto_cfg["seed"]) + int(source_index),
        safety_factor=float(auto_cfg["safety_factor"]),
        boundary_margin=auto_cfg["boundary_margin"],
    )
    return estimated_radius, {
        "mode": "auto_cutoff",
        "quantile": float(auto_cfg["quantile"]),
        "coverage_estimate": float(coverage),
        "estimated_radius": float(estimated_radius),
        "default_radius": float(default_radius),
        "target_points": int(max(int(auto_cfg["target_points"]), int(target_points))),
    }


def _resolve_source_spec(
    cfg: dict[str, Any],
    *,
    source_index: int,
    file_index: int,
    target_points: int,
) -> dict[str, Any]:
    data_cfg = cfg.get("data", None)
    if data_cfg is None:
        raise ValueError("Config is missing a top-level data section.")
    if not isinstance(data_cfg, dict):
        raise ValueError(f"Config data section must be a dict, got {type(data_cfg)!r}.")

    kind = str(data_cfg.get("kind", "")).strip().lower()
    if kind == "synthetic":
        env_dir = Path(str(data_cfg.get("data_path", "")))
        if not env_dir.exists():
            raise FileNotFoundError(f"Synthetic data_path does not exist: {env_dir}")
        atoms_path = env_dir / "atoms.npy"
        atoms_full_path = env_dir / "atoms_full.npy"
        if atoms_path.exists():
            file_path = atoms_path
        elif atoms_full_path.exists():
            file_path = atoms_full_path
        else:
            raise FileNotFoundError(
                "Synthetic dataset directory must contain atoms.npy or atoms_full.npy, "
                f"but neither exists in {env_dir}."
            )
        return {
            "kind": kind,
            "file_path": file_path,
            "radius": float(data_cfg.get("radius", 0.0)),
            "source_name": env_dir.name or "synthetic",
            "radius_details": None,
        }

    if kind == "real":
        data_sources = data_cfg.get("data_sources", None)
        if not data_sources:
            raise ValueError("Real-data config must define data.data_sources.")
        if source_index < 0 or source_index >= len(data_sources):
            raise IndexError(
                f"source_index={source_index} is out of range for {len(data_sources)} data sources."
            )
        source = dict(data_sources[source_index])
        data_files = source.get("data_files", [])
        if isinstance(data_files, str):
            data_files = [data_files]
        if not data_files:
            raise ValueError(f"Selected source has no data_files entries: {source}")
        if file_index < 0 or file_index >= len(data_files):
            raise IndexError(
                f"file_index={file_index} is out of range for {len(data_files)} files in source {source_index}."
            )
        file_name = str(data_files[file_index])
        root = Path(str(source["data_path"]))
        file_path = root / file_name
        radius, radius_details = _resolve_real_source_radius(
            data_cfg,
            source_index=source_index,
            source_dict=source,
            target_points=target_points,
        )
        source_name = str(source.get("name", root.name or f"source_{source_index}"))
        return {
            "kind": kind,
            "file_path": file_path,
            "radius": float(radius),
            "source_name": source_name,
            "radius_details": radius_details,
        }

    raise ValueError(
        "Unsupported data kind for VICReg illustration. "
        f"Expected 'synthetic' or 'real', got {kind!r}."
    )


def _interior_candidate_indices(points: np.ndarray, radius: float) -> np.ndarray:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    interior_mask = np.all(
        (points >= (mins + float(radius))) & (points <= (maxs - float(radius))),
        axis=1,
    )
    interior = np.flatnonzero(interior_mask)
    if interior.size > 0:
        return interior.astype(np.int64, copy=False)
    return np.arange(points.shape[0], dtype=np.int64)


def _query_centered_neighborhood(
    points: np.ndarray,
    tree: cKDTree,
    *,
    center_index: int,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(points[int(center_index)], dtype=np.float32)
    raw_indices = np.asarray(tree.query_ball_point(center, float(radius)), dtype=np.int64)
    if raw_indices.size == 0:
        raise RuntimeError(
            f"Radius query returned no atoms for center_index={center_index} and radius={radius:.6f}."
        )
    raw_points = points[raw_indices] - center[None, :]
    d2 = np.sum(raw_points.astype(np.float64) ** 2, axis=1)
    order = np.argsort(d2, kind="stable")
    raw_indices = raw_indices[order]
    raw_points = raw_points[order].astype(np.float32, copy=False)
    origin_idx = int(np.argmin(np.sum(raw_points.astype(np.float64) ** 2, axis=1)))
    if not float(np.linalg.norm(raw_points[origin_idx])) <= 1e-7:
        raise RuntimeError(
            "Expected centered neighborhood to contain the center atom at the origin, "
            f"but the closest point has norm {float(np.linalg.norm(raw_points[origin_idx])):.6e}."
        )
    return raw_indices, raw_points


def _pick_representative_center(
    points: np.ndarray,
    *,
    radius: float,
    target_points: int,
    candidate_limit: int,
    seed: int,
    forced_center_index: int | None,
) -> dict[str, Any]:
    tree = cKDTree(points)
    if forced_center_index is not None:
        if forced_center_index < 0 or forced_center_index >= points.shape[0]:
            raise IndexError(
                f"forced_center_index={forced_center_index} is out of range for {points.shape[0]} atoms."
            )
        raw_indices, raw_points = _query_centered_neighborhood(
            points,
            tree,
            center_index=int(forced_center_index),
            radius=radius,
        )
        if raw_points.shape[0] < target_points:
            raise ValueError(
                "Forced center does not have enough atoms inside the cutoff sphere for the requested illustration: "
                f"center_index={forced_center_index}, raw_count={raw_points.shape[0]}, target_points={target_points}."
            )
        return {
            "center_index": int(forced_center_index),
            "raw_indices": raw_indices,
            "raw_points": raw_points,
            "candidate_pool_size": 1,
            "candidate_mode": "forced",
        }

    candidate_indices = _interior_candidate_indices(points, radius)
    if candidate_indices.size == 0:
        raise RuntimeError("No candidate center atoms are available.")
    rng = np.random.default_rng(int(seed))
    if candidate_indices.size > int(candidate_limit):
        candidate_indices = rng.choice(candidate_indices, size=int(candidate_limit), replace=False)

    tree = cKDTree(points)
    neighbor_lists = tree.query_ball_point(points[candidate_indices], r=float(radius))
    counts = np.asarray([len(neigh) for neigh in neighbor_lists], dtype=np.int32)
    valid_mask = counts >= int(target_points)
    if not np.any(valid_mask):
        raise ValueError(
            "Could not find any candidate atom with enough neighbors inside the cutoff sphere. "
            f"radius={radius:.6f}, target_points={target_points}, "
            f"best_raw_count={int(np.max(counts)) if counts.size else 0}."
        )

    valid_indices = candidate_indices[valid_mask]
    valid_counts = counts[valid_mask].astype(np.float64, copy=False)
    desired_raw_count = max(int(target_points) + 16, int(round(float(target_points) * 1.25)))
    scores = np.abs(valid_counts - float(desired_raw_count))
    tie_break = np.abs(valid_counts - float(np.median(valid_counts))) * 1e-3
    chosen_pos = int(np.argmin(scores + tie_break))
    center_index = int(valid_indices[chosen_pos])
    raw_indices, raw_points = _query_centered_neighborhood(
        points,
        tree,
        center_index=center_index,
        radius=radius,
    )
    return {
        "center_index": center_index,
        "raw_indices": raw_indices,
        "raw_points": raw_points,
        "candidate_pool_size": int(candidate_indices.size),
        "candidate_mode": "automatic",
    }


def _subset_nearest_origin(
    raw_indices: np.ndarray,
    raw_points: np.ndarray,
    *,
    target_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if raw_points.shape[0] < target_points:
        raise ValueError(
            f"Cannot subset to {target_points} points because the raw neighborhood only has {raw_points.shape[0]} atoms."
        )
    return (
        np.asarray(raw_indices[:target_points], dtype=np.int64),
        np.asarray(raw_points[:target_points], dtype=np.float32),
    )


def _split_nearest_origin_keep_drop(
    source_indices: np.ndarray,
    points: np.ndarray,
    *,
    target_points: int,
) -> dict[str, np.ndarray]:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}.")
    if source_indices.shape != (points.shape[0],):
        raise ValueError(
            "source_indices must match points along axis 0. "
            f"Got source_indices shape {source_indices.shape} and points shape {points.shape}."
        )
    if target_points <= 0:
        raise ValueError(f"target_points must be > 0, got {target_points}.")
    if points.shape[0] < target_points:
        raise ValueError(
            f"Cannot keep {target_points} points because the input only has {points.shape[0]} points."
        )

    dist2 = np.sum(points.astype(np.float64) ** 2, axis=1)
    order = np.argsort(dist2, kind="stable")
    ordered_source_indices = np.asarray(source_indices[order], dtype=np.int64)
    ordered_points = np.asarray(points[order], dtype=np.float32)

    keep_count = int(target_points)
    keep_source_indices = np.asarray(ordered_source_indices[:keep_count], dtype=np.int64)
    keep_points = np.asarray(ordered_points[:keep_count], dtype=np.float32)
    drop_source_indices = np.asarray(ordered_source_indices[keep_count:], dtype=np.int64)
    drop_points = np.asarray(ordered_points[keep_count:], dtype=np.float32)
    return {
        "keep_source_indices": keep_source_indices,
        "keep_points": keep_points,
        "drop_source_indices": drop_source_indices,
        "drop_points": drop_points,
    }


def _choose_neighbor_local_index(
    subset_points: np.ndarray,
    *,
    neighbor_k: int,
    neighbor_rank: int,
) -> int:
    if subset_points.ndim != 2 or subset_points.shape[1] != 3:
        raise ValueError(f"subset_points must have shape (N, 3), got {subset_points.shape}.")
    if subset_points.shape[0] < 2:
        raise ValueError("Need at least 2 points to choose a neighbor-centered view.")
    dist2 = np.sum(subset_points.astype(np.float64) ** 2, axis=1)
    candidate_order = np.argsort(np.where(dist2 <= 1e-14, np.inf, dist2), kind="stable")
    usable = candidate_order[np.isfinite(np.where(dist2[candidate_order] <= 1e-14, np.inf, dist2[candidate_order]))]
    if usable.size == 0:
        raise RuntimeError("Failed to identify any non-origin atom for the neighbor-centered view.")
    usable = usable[: min(int(max(neighbor_k, 1)), usable.size)]
    rank = int(np.clip(neighbor_rank, 0, usable.size - 1))
    return int(usable[rank])


def _build_stage_geometry(points: np.ndarray) -> dict[str, Any]:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Stage point cloud must have shape (N, 3), got {points.shape}.")
    edges, edge_info = _build_local_coordination_edges(
        points,
        min_shell_neighbors=2,
        max_shell_neighbors=5,
        shell_gap_ratio=1.22,
        edge_mode="coordination_shell_mutual",
    )
    return {
        "points": np.asarray(points, dtype=np.float32),
        "edges": _ensure_connected_edges(points, edges),
        "colors": _compute_radial_colormap_colors(points, cmap_name="viridis"),
        "edge_info": edge_info,
    }


def _origin_centered_half_span(stage_points: Sequence[np.ndarray]) -> float:
    extrema: list[float] = []
    for pts in stage_points:
        arr = np.asarray(pts, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Stage points must have shape (N, 3), got {arr.shape}.")
        extrema.append(float(np.max(np.abs(arr))))
    half_span = max(extrema) * 1.08
    if not np.isfinite(half_span) or half_span <= 0.0:
        raise ValueError(f"Computed invalid shared half-span {half_span}.")
    return float(half_span)


def _stage_all_points(stage: dict[str, Any]) -> np.ndarray:
    points = stage.get("all_points", stage["points"])
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Stage points must have shape (N, 3), got {arr.shape}.")
    return arr


def _lookup_stage_position(
    stage: dict[str, Any],
    *,
    target_source_index: int,
) -> np.ndarray | None:
    stage_source_indices = np.asarray(
        stage.get("all_source_indices", stage["source_indices"]),
        dtype=np.int64,
    )
    stage_points = _stage_all_points(stage)
    matches = np.flatnonzero(stage_source_indices == int(target_source_index))
    if matches.size == 0:
        return None
    return np.asarray(stage_points[int(matches[0])], dtype=np.float32)


def _style_stage_axes(ax: Any, *, half_span: float, view_elev: float, view_azim: float) -> None:
    ax.set_facecolor("white")
    if hasattr(ax, "set_proj_type"):
        ax.set_proj_type("ortho")
    ax.view_init(elev=float(view_elev), azim=float(view_azim))
    ax.set_xlim(-half_span, half_span)
    ax.set_ylim(-half_span, half_span)
    ax.set_zlim(-half_span, half_span)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 1.0))
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


def _draw_dropped_points(
    ax: Any,
    dropped_points: np.ndarray,
    *,
    style: str,
) -> None:
    pts = np.asarray(dropped_points, dtype=np.float32)
    if pts.size == 0:
        return
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"dropped_points must have shape (N, 3), got {pts.shape}.")

    style_norm = str(style).strip().lower()
    if style_norm == "white":
        facecolors: Any = "#FFFFFF"
        edgecolors: Any = _DROPPED_POINT_EDGE_COLOR
        alpha = 0.95
    elif style_norm == "transparent":
        facecolors = "none"
        edgecolors = _DROPPED_POINT_EDGE_COLOR
        alpha = 0.72
    else:
        raise ValueError(
            f"Unsupported dropped-point style {style!r}. Expected 'white' or 'transparent'."
        )

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        s=32.0,
        facecolors=facecolors,
        edgecolors=edgecolors,
        linewidths=0.72,
        depthshade=False,
        alpha=float(alpha),
    )


def _draw_stage_panel(
    ax: Any,
    stage: dict[str, Any],
    *,
    dropped_style: str,
    half_span: float,
    view_elev: float,
    view_azim: float,
) -> None:
    points = np.asarray(stage["points"], dtype=np.float32)
    colors = np.asarray(stage["colors"], dtype=np.float32)
    edges = [(int(e0), int(e1)) for e0, e1 in stage["edges"]]
    _style_stage_axes(ax, half_span=half_span, view_elev=view_elev, view_azim=view_azim)
    _draw_edges(
        ax,
        points,
        edges,
        point_colors=colors,
        edge_alpha=0.56,
        edge_linewidth=0.88,
    )
    dropped_points = np.asarray(stage.get("dropped_points", np.zeros((0, 3), dtype=np.float32)))
    if dropped_points.size > 0:
        _draw_dropped_points(ax, dropped_points, style=dropped_style)

    point_size = 40.0 if points.shape[0] <= 128 else 28.0
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        s=float(point_size),
        alpha=0.97,
        edgecolors="#222222",
        linewidths=0.20,
        depthshade=False,
    )
    ax.scatter(
        [0.0],
        [0.0],
        [0.0],
        c="#111111",
        s=15.0,
        marker="+",
        linewidths=1.0,
        depthshade=False,
        alpha=0.95,
    )


def _render_illustration(
    stages: Sequence[dict[str, Any]],
    *,
    output_path: Path,
    source_label: str,
    dropped_style: str,
    radius: float,
    view_elev: float,
    view_azim: float,
) -> None:
    if len(stages) != 3:
        raise ValueError(f"Expected exactly 3 stages to render, got {len(stages)}.")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.6, 3.8),
        dpi=300,
        facecolor="white",
        subplot_kw={"projection": "3d"},
    )
    panel_positions = [
        [0.020, 0.135, 0.298, 0.665],
        [0.351, 0.135, 0.298, 0.665],
        [0.682, 0.135, 0.298, 0.665],
    ]
    for ax, position in zip(axes, panel_positions):
        ax.set_position(position)
    half_span = _origin_centered_half_span([_stage_all_points(stage) for stage in stages])

    for ax, stage in zip(axes, stages):
        _draw_stage_panel(
            ax,
            stage,
            dropped_style=dropped_style,
            half_span=half_span,
            view_elev=view_elev,
            view_azim=view_azim,
        )
        ax.set_title(
            f"{stage['title']}\n{stage['subtitle']}",
            fontsize=9.5,
            pad=4.0,
            color="#202020",
        )

    fig.suptitle(
        f"VICReg view illustration | {source_label} | cutoff={radius:.3f} | dropped={str(dropped_style).lower()}",
        fontsize=10.5,
        y=0.965,
        color="#1a1a1a",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _render_stage_svg(
    stage: dict[str, Any],
    *,
    output_path: Path,
    dropped_style: str,
    half_span: float,
    view_elev: float,
    view_azim: float,
) -> None:
    fig = plt.figure(figsize=(3.15, 3.15), dpi=300, facecolor="white")
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_position([0.01, 0.01, 0.98, 0.98])
    _draw_stage_panel(
        ax,
        stage,
        dropped_style=dropped_style,
        half_span=half_span,
        view_elev=view_elev,
        view_azim=view_azim,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def _save_stage_xyz_and_npz(
    output_path: Path,
    *,
    raw_points: np.ndarray,
    subset_points: np.ndarray,
    neighbor_all_points: np.ndarray,
    neighbor_keep_points: np.ndarray,
    neighbor_drop_points: np.ndarray,
    raw_source_indices: np.ndarray,
    subset_source_indices: np.ndarray,
    neighbor_keep_source_indices: np.ndarray,
    neighbor_drop_source_indices: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    stem = output_path.with_suffix("")
    _save_structure_xyz(
        stem.parent,
        f"{stem.name}_01_raw.xyz",
        raw_points,
        comment="Stage 1 | raw cutoff crop centered on the original atom",
        element="X",
    )
    _save_structure_xyz(
        stem.parent,
        f"{stem.name}_02_subset.xyz",
        subset_points,
        comment="Stage 2 | deterministic subset centered on the original atom",
        element="X",
    )
    _save_structure_xyz(
        stem.parent,
        f"{stem.name}_03_neighbor_all.xyz",
        neighbor_all_points,
        comment="Stage 3a | all translated subset atoms before dropping to VICReg view size",
        element="X",
    )
    _save_structure_xyz(
        stem.parent,
        f"{stem.name}_04_neighbor_kept.xyz",
        neighbor_keep_points,
        comment="Stage 3b | kept translated atoms after dropping to VICReg view size",
        element="X",
    )
    if neighbor_drop_points.shape[0] > 0:
        _save_structure_xyz(
            stem.parent,
            f"{stem.name}_05_neighbor_dropped.xyz",
            neighbor_drop_points,
            comment="Stage 3c | dropped translated atoms after VICReg view crop",
            element="X",
        )

    np.savez_compressed(
        stem.parent / f"{stem.name}_data.npz",
        raw_points_centered=raw_points.astype(np.float32),
        subset_points_centered=subset_points.astype(np.float32),
        neighbor_centered_all_points=neighbor_all_points.astype(np.float32),
        neighbor_centered_kept_points=neighbor_keep_points.astype(np.float32),
        neighbor_centered_dropped_points=neighbor_drop_points.astype(np.float32),
        raw_source_indices=raw_source_indices.astype(np.int64),
        subset_source_indices=subset_source_indices.astype(np.int64),
        neighbor_keep_source_indices=neighbor_keep_source_indices.astype(np.int64),
        neighbor_drop_source_indices=neighbor_drop_source_indices.astype(np.int64),
    )
    with (stem.parent / f"{stem.name}_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)


def build_illustration(
    *,
    cfg: dict[str, Any],
    config_label: str,
    output_path: Path,
    source_index: int,
    file_index: int,
    target_points: int,
    translated_points: int,
    neighbor_k: int,
    neighbor_rank: int,
    candidate_limit: int,
    seed: int,
    forced_center_index: int | None,
    view_elev: float,
    view_azim: float,
) -> dict[str, Any]:
    if translated_points <= 0:
        raise ValueError(f"translated_points must be > 0, got {translated_points}.")
    if translated_points > target_points:
        raise ValueError(
            "translated_points cannot exceed target_points because stage 3 is cropped from stage 2. "
            f"Got translated_points={translated_points}, target_points={target_points}."
        )

    source_spec = _resolve_source_spec(
        cfg,
        source_index=source_index,
        file_index=file_index,
        target_points=target_points,
    )
    points = _load_point_cloud(Path(source_spec["file_path"]))
    radius = float(source_spec["radius"])
    if radius <= 0.0:
        raise ValueError(f"Cutoff radius must be positive, got {radius}.")

    picked = _pick_representative_center(
        points,
        radius=radius,
        target_points=target_points,
        candidate_limit=candidate_limit,
        seed=seed,
        forced_center_index=forced_center_index,
    )
    raw_indices = np.asarray(picked["raw_indices"], dtype=np.int64)
    raw_points = np.asarray(picked["raw_points"], dtype=np.float32)
    center_index = int(picked["center_index"])

    subset_source_indices, subset_points = _subset_nearest_origin(
        raw_indices,
        raw_points,
        target_points=target_points,
    )
    neighbor_local_index = _choose_neighbor_local_index(
        subset_points,
        neighbor_k=neighbor_k,
        neighbor_rank=neighbor_rank,
    )
    neighbor_source_index = int(subset_source_indices[neighbor_local_index])
    neighbor_all_points = subset_points - subset_points[neighbor_local_index][None, :]
    neighbor_split = _split_nearest_origin_keep_drop(
        subset_source_indices,
        neighbor_all_points,
        target_points=translated_points,
    )
    neighbor_keep_source_indices = np.asarray(neighbor_split["keep_source_indices"], dtype=np.int64)
    neighbor_keep_points = np.asarray(neighbor_split["keep_points"], dtype=np.float32)
    neighbor_drop_source_indices = np.asarray(neighbor_split["drop_source_indices"], dtype=np.int64)
    neighbor_drop_points = np.asarray(neighbor_split["drop_points"], dtype=np.float32)

    raw_stage = _build_stage_geometry(raw_points)
    subset_stage = _build_stage_geometry(subset_points)
    neighbor_stage = _build_stage_geometry(neighbor_keep_points)

    stages = [
        {
            **raw_stage,
            "source_indices": raw_indices,
            "all_source_indices": raw_indices,
            "all_points": raw_points,
            "title": "1. Raw cutoff subcloud",
            "subtitle": f"n={raw_points.shape[0]} | center atom at origin",
        },
        {
            **subset_stage,
            "source_indices": subset_source_indices,
            "all_source_indices": subset_source_indices,
            "all_points": subset_points,
            "title": "2. Fixed-size subset",
            "subtitle": f"n={subset_points.shape[0]} | nearest-to-center keep",
        },
        {
            **neighbor_stage,
            "source_indices": neighbor_keep_source_indices,
            "all_source_indices": subset_source_indices,
            "all_points": neighbor_all_points,
            "dropped_points": neighbor_drop_points,
            "dropped_source_indices": neighbor_drop_source_indices,
            "title": "3. Neighbor-centered view",
            "subtitle": (
                f"keep n={neighbor_keep_points.shape[0]} | "
                f"drop n={neighbor_drop_points.shape[0]} after translation"
            ),
        },
    ]

    source_label = f"{source_spec['source_name']}:{Path(source_spec['file_path']).name}"
    output_transparent_path = output_path.with_name(f"{output_path.stem}_transparent{output_path.suffix}")
    stage_half_span = _origin_centered_half_span([_stage_all_points(stage) for stage in stages])
    stage_svg_paths = {
        "raw": output_path.with_name(f"{output_path.stem}_step01_raw.svg"),
        "subset": output_path.with_name(f"{output_path.stem}_step02_subset.svg"),
        "neighbor_white": output_path.with_name(f"{output_path.stem}_step03_neighbor_white.svg"),
        "neighbor_transparent": output_path.with_name(f"{output_path.stem}_step03_neighbor_transparent.svg"),
    }
    _render_illustration(
        stages,
        output_path=output_path,
        source_label=source_label,
        dropped_style="white",
        radius=radius,
        view_elev=view_elev,
        view_azim=view_azim,
    )
    _render_illustration(
        stages,
        output_path=output_transparent_path,
        source_label=source_label,
        dropped_style="transparent",
        radius=radius,
        view_elev=view_elev,
        view_azim=view_azim,
    )
    _render_stage_svg(
        stages[0],
        output_path=stage_svg_paths["raw"],
        dropped_style="white",
        half_span=stage_half_span,
        view_elev=view_elev,
        view_azim=view_azim,
    )
    _render_stage_svg(
        stages[1],
        output_path=stage_svg_paths["subset"],
        dropped_style="white",
        half_span=stage_half_span,
        view_elev=view_elev,
        view_azim=view_azim,
    )
    _render_stage_svg(
        stages[2],
        output_path=stage_svg_paths["neighbor_white"],
        dropped_style="white",
        half_span=stage_half_span,
        view_elev=view_elev,
        view_azim=view_azim,
    )
    _render_stage_svg(
        stages[2],
        output_path=stage_svg_paths["neighbor_transparent"],
        dropped_style="transparent",
        half_span=stage_half_span,
        view_elev=view_elev,
        view_azim=view_azim,
    )

    metadata = {
        "config_name": str(config_label),
        "source_kind": str(source_spec["kind"]),
        "source_name": str(source_spec["source_name"]),
        "source_file": str(source_spec["file_path"]),
        "cutoff_radius": float(radius),
        "radius_details": source_spec["radius_details"],
        "center_atom_index": int(center_index),
        "neighbor_atom_index": int(neighbor_source_index),
        "neighbor_local_index_in_subset": int(neighbor_local_index),
        "neighbor_k": int(neighbor_k),
        "neighbor_rank": int(neighbor_rank),
        "target_points": int(target_points),
        "translated_points": int(translated_points),
        "raw_point_count": int(raw_points.shape[0]),
        "subset_point_count": int(subset_points.shape[0]),
        "neighbor_all_point_count": int(neighbor_all_points.shape[0]),
        "neighbor_keep_point_count": int(neighbor_keep_points.shape[0]),
        "neighbor_drop_point_count": int(neighbor_drop_points.shape[0]),
        "candidate_pool_size": int(picked["candidate_pool_size"]),
        "candidate_mode": str(picked["candidate_mode"]),
        "selection_mode": "deterministic_nearest_origin_subsample",
        "view_note": (
            "This is a paper illustration of the VICReg view path. "
            "The subset steps are intentionally deterministic even if training uses randomization."
        ),
        "output_white_dropped": str(output_path),
        "output_transparent_dropped": str(output_transparent_path),
        "step_svg_outputs": {key: str(path) for key, path in stage_svg_paths.items()},
        "view_elev": float(view_elev),
        "view_azim": float(view_azim),
        "stage_edge_info": {
            "raw": raw_stage["edge_info"],
            "subset": subset_stage["edge_info"],
            "neighbor": neighbor_stage["edge_info"],
        },
    }
    _save_stage_xyz_and_npz(
        output_path,
        raw_points=raw_points,
        subset_points=subset_points,
        neighbor_all_points=neighbor_all_points,
        neighbor_keep_points=neighbor_keep_points,
        neighbor_drop_points=neighbor_drop_points,
        raw_source_indices=raw_indices,
        subset_source_indices=subset_source_indices,
        neighbor_keep_source_indices=neighbor_keep_source_indices,
        neighbor_drop_source_indices=neighbor_drop_source_indices,
        metadata=metadata,
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a paper-style illustration of VICReg view construction."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/vicreg_vn_molecular.yaml",
        help="Hydra config path or config name inside configs/.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path. Defaults to output/vicreg_view_illustrations/<config>_<source>.png",
    )
    parser.add_argument("--source-index", type=int, default=0, help="Real-data source index.")
    parser.add_argument("--file-index", type=int, default=0, help="File index within the selected source.")
    parser.add_argument(
        "--target-points",
        type=int,
        default=None,
        help="Subset size for stage 2. Defaults to data.num_points from the config.",
    )
    parser.add_argument(
        "--translated-points",
        type=int,
        default=None,
        help=(
            "Point count kept in stage 3 after neighbor translation. "
            "Defaults to vicreg_view_points, then data.model_points, then target_points."
        ),
    )
    parser.add_argument(
        "--neighbor-k",
        type=int,
        default=None,
        help="Choose the recentering atom from the k nearest non-origin atoms. Defaults to vicreg_neighbor_k.",
    )
    parser.add_argument(
        "--neighbor-rank",
        type=int,
        default=0,
        help="Which atom to choose inside the k-nearest candidate list after sorting by radius.",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=4096,
        help="Maximum number of candidate center atoms to evaluate when selecting a representative example.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for representative-center selection.")
    parser.add_argument(
        "--center-index",
        type=int,
        default=None,
        help="Optional explicit source-atom index to use as the original center.",
    )
    parser.add_argument("--view-elev", type=float, default=22.0, help="3D elevation angle.")
    parser.add_argument("--view-azim", type=float, default=38.0, help="3D azimuth angle.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve_config_path(args.config)
    cfg = _load_hydra_config(config_path)
    config_label = str(config_path.relative_to(Path.cwd())) if config_path.is_relative_to(Path.cwd()) else str(config_path)

    target_points = int(args.target_points) if args.target_points is not None else int(_cfg_get(cfg, "data.num_points", 0))
    if target_points <= 0:
        raise ValueError(f"target_points must be > 0, got {target_points}.")

    translated_points = (
        int(args.translated_points)
        if args.translated_points is not None
        else int(
            _cfg_get(
                cfg,
                "vicreg_view_points",
                _cfg_get(cfg, "data.model_points", target_points),
            )
        )
    )
    if translated_points <= 0:
        raise ValueError(f"translated_points must be > 0, got {translated_points}.")

    neighbor_k = int(args.neighbor_k) if args.neighbor_k is not None else int(_cfg_get(cfg, "vicreg_neighbor_k", 6))
    if neighbor_k <= 0:
        raise ValueError(f"neighbor_k must be > 0, got {neighbor_k}.")

    if args.output is None:
        out_dir = Path("output") / "vicreg_view_illustrations"
        out_name = (
            f"{config_path.stem}_"
            f"source{int(args.source_index):02d}_"
            f"file{int(args.file_index):02d}.png"
        )
        output_path = out_dir / out_name
    else:
        output_path = Path(args.output)

    metadata = build_illustration(
        cfg=cfg,
        config_label=config_label,
        output_path=output_path,
        source_index=int(args.source_index),
        file_index=int(args.file_index),
        target_points=target_points,
        translated_points=translated_points,
        neighbor_k=neighbor_k,
        neighbor_rank=int(args.neighbor_rank),
        candidate_limit=int(args.candidate_limit),
        seed=int(args.seed),
        forced_center_index=args.center_index,
        view_elev=float(args.view_elev),
        view_azim=float(args.view_azim),
    )
    print(f"Saved VICReg view illustration to {output_path}")
    print(f"Saved transparent dropped-point variant to {metadata['output_transparent_dropped']}")
    for stage_name, stage_path in metadata["step_svg_outputs"].items():
        print(f"  svg_{stage_name}: {stage_path}")
    print(f"  source_file: {metadata['source_file']}")
    print(f"  cutoff_radius: {metadata['cutoff_radius']:.6f}")
    print(f"  center_atom_index: {metadata['center_atom_index']}")
    print(f"  neighbor_atom_index: {metadata['neighbor_atom_index']}")
    print(f"  raw_point_count: {metadata['raw_point_count']}")
    print(f"  subset_point_count: {metadata['subset_point_count']}")
    print(f"  translated_keep_count: {metadata['neighbor_keep_point_count']}")
    print(f"  translated_drop_count: {metadata['neighbor_drop_point_count']}")


if __name__ == "__main__":
    main()
