from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
import os
import re
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.data_utils.data_kinds import normalize_data_kind
from .cluster_profiles import (
    _ALL_PROFILE_PROPERTIES,
    _compute_sample_properties,
    _load_point_cloud_from_dataset,
)
from .cluster_geometry import _sample_indices_stratified
from .cluster_rendering import (
    _save_cluster_representatives_figure,
)
from .cluster_figures import _build_cluster_color_map
from .output_layout import real_md_outputs_root, write_json
from src.vis_tools.real_md_analysis_vis import (
    save_cna_signature_time_series,
    save_cluster_proportion_plots,
    save_descriptor_violin_grid,
    save_embedding_discrete_plot,
    save_temporal_embedding_cluster_animation,
    save_temporal_embedding_trajectory_animation,
    save_temporal_spatial_cluster_animation,
    save_temporal_transition_flow_animation,
    save_transition_flow_plot,
)
from src.vis_tools.tsne_vis import compute_tsne


_TIME_RE = re.compile(r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?P<unit>[A-Za-z]+)")
_BUILTIN_SCALAR_COLUMNS = [str(key) for key, _ in _ALL_PROFILE_PROPERTIES]
_BUILTIN_SCALAR_LABELS = {str(key): str(label) for key, label in _ALL_PROFILE_PROPERTIES}
_FRAME_NAME_SUFFIXES = (
    ".npy",
    ".npz",
    ".xyz",
    ".extxyz",
    ".lammpstrj",
    ".dump",
    ".pos",
)


def _frame_name_text(name: str) -> str:
    text = Path(str(name)).name or str(name)
    lower_text = text.lower()
    for suffix in _FRAME_NAME_SUFFIXES:
        if lower_text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _format_time_label(value: float, unit: str) -> str:
    numeric_value = float(value)
    if abs(numeric_value - round(numeric_value)) < 1e-9:
        numeric_text = f"{int(round(numeric_value))}"
    else:
        numeric_text = f"{numeric_value:g}"
    return f"{numeric_text} {str(unit)}"


@dataclass(frozen=True)
class FrameSlice:
    source_name: str
    output_name: str
    indices: np.ndarray
    order_index: int
    time_value: float | None
    time_unit: str | None

    @property
    def label(self) -> str:
        stem = _frame_name_text(str(self.source_name))
        if self.time_value is None or self.time_unit is None:
            return stem
        return _format_time_label(float(self.time_value), str(self.time_unit))


@dataclass(frozen=True)
class ProjectionFeaturePrep:
    l2_normalize: bool
    scaler: StandardScaler | None
    pca: PCA | None
    pca_keep_components: int


def _to_plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _parse_time_from_name(name: str) -> tuple[float | None, str | None]:
    stem = _frame_name_text(str(name))
    match = _TIME_RE.search(stem)
    if match is None:
        return None, None
    return float(match.group("value")), str(match.group("unit"))


def _build_frame_slices(
    frame_groups: list[tuple[str, np.ndarray]],
    frame_output_names: dict[str, str],
    *,
    requested_order: list[str] | None,
) -> list[FrameSlice]:
    groups_by_name = {
        str(source_name): np.asarray(indices, dtype=int)
        for source_name, indices in frame_groups
    }
    ordered_names: list[str] = []
    if requested_order:
        for source_name in requested_order:
            source_text = str(source_name)
            if source_text in groups_by_name and source_text not in ordered_names:
                ordered_names.append(source_text)
    for source_name, _ in frame_groups:
        source_text = str(source_name)
        if source_text not in ordered_names:
            ordered_names.append(source_text)

    frames: list[FrameSlice] = []
    for order_index, source_name in enumerate(ordered_names):
        time_value, time_unit = _parse_time_from_name(source_name)
        frames.append(
            FrameSlice(
                source_name=str(source_name),
                output_name=str(frame_output_names.get(str(source_name), Path(str(source_name)).stem)),
                indices=np.asarray(groups_by_name[str(source_name)], dtype=int),
                order_index=int(order_index),
                time_value=time_value,
                time_unit=time_unit,
            )
        )
    return frames


def _select_evenly_spaced_frames(
    frames: list[FrameSlice],
    *,
    max_frame_count: int | None,
    field_name: str,
) -> list[FrameSlice]:
    if not frames:
        raise ValueError(f"{field_name} cannot select from an empty frame list.")
    if max_frame_count is None:
        return list(frames)
    resolved_count = int(max_frame_count)
    if resolved_count <= 0:
        raise ValueError(f"{field_name} must be > 0 when provided, got {max_frame_count}.")
    if resolved_count >= len(frames):
        return list(frames)
    if resolved_count == 1:
        return [frames[0]]
    positions = np.rint(
        np.linspace(0, len(frames) - 1, num=resolved_count)
    ).astype(int)
    positions = np.unique(np.clip(positions, 0, len(frames) - 1))
    if positions.size != resolved_count:
        raise RuntimeError(
            "Even frame subsampling produced duplicate positions. "
            f"field={field_name}, frame_count={len(frames)}, "
            f"requested={resolved_count}, positions={positions.tolist()}."
        )
    return [frames[int(pos)] for pos in positions.tolist()]


def _resolve_umap_animation_frame_count(temporal_umap_cfg: Any) -> int | None:
    raw_value = getattr(temporal_umap_cfg, "animation_frame_count", None)
    if raw_value is None:
        raw_value = getattr(temporal_umap_cfg, "frame_count", None)
    if raw_value is None:
        return None
    value = int(raw_value)
    if value == -1:
        return None
    if value <= 0:
        raise ValueError(
            "real_md.temporal.umap.animation_frame_count must be -1 to use all "
            f"frames, or a positive frame cap; got {raw_value!r}."
        )
    return value


def _normalize_cluster_id_list(value: Any, *, field_name: str) -> list[int]:
    raw = _to_plain(value)
    if raw is None:
        raise ValueError(f"{field_name} is required.")
    if isinstance(raw, str):
        tokens = [tok.strip() for tok in raw.split(",") if tok.strip()]
    else:
        tokens = list(raw)
    if not tokens:
        raise ValueError(f"{field_name} must contain at least one cluster ID.")
    return [int(v) for v in tokens]


def _cfg_bool(cfg: Any, field_name: str, default: bool) -> bool:
    return bool(getattr(cfg, field_name, default))


def _cfg_int(cfg: Any, field_name: str, default: int) -> int:
    return int(getattr(cfg, field_name, default))


def _cfg_float(cfg: Any, field_name: str, default: float) -> float:
    return float(getattr(cfg, field_name, default))


def _resolve_temporal_animation_parallel_workers(
    temporal_cfg: Any,
    *,
    task_count: int,
) -> int:
    if task_count < 0:
        raise ValueError(f"task_count must be >= 0, got {task_count}.")
    if task_count <= 1:
        return int(max(task_count, 0))
    requested = int(getattr(temporal_cfg, "animation_parallel_workers", 0))
    if requested < 0:
        raise ValueError(
            "real_md.temporal.animation_parallel_workers must be >= 0, "
            f"got {requested}."
        )
    if requested == 0:
        cpu_count = os.cpu_count() or 1
        return max(1, min(task_count, cpu_count, 2))
    return max(1, min(task_count, requested))


def _temporal_animation_process_context() -> mp.context.BaseContext:
    if os.name == "posix" and "fork" in mp.get_all_start_methods():
        # These renderer workers only touch NumPy/Matplotlib output state, so
        # `fork` avoids the large serialization overhead of `spawn`.
        return mp.get_context("fork")
    return mp.get_context("spawn")


def _run_temporal_animation_task(
    task_kind: str,
    task_kwargs: dict[str, Any],
) -> dict[str, Any]:
    if task_kind == "spatial_clusters":
        return save_temporal_spatial_cluster_animation(**task_kwargs)
    if task_kind == "embedding_clusters":
        return save_temporal_embedding_cluster_animation(**task_kwargs)
    if task_kind == "embedding_trajectories":
        return save_temporal_embedding_trajectory_animation(**task_kwargs)
    if task_kind == "transition_flow":
        return save_temporal_transition_flow_animation(**task_kwargs)
    raise ValueError(
        "Unsupported temporal animation task kind. "
        f"Expected one of ['spatial_clusters', 'embedding_clusters', "
        f"'embedding_trajectories', 'transition_flow'], got {task_kind!r}."
    )


def _execute_temporal_animation_jobs(
    animation_jobs: list[tuple[str, str, dict[str, Any]]],
    *,
    temporal_cfg: Any,
) -> dict[str, Any]:
    if not animation_jobs:
        return {}
    max_workers = _resolve_temporal_animation_parallel_workers(
        temporal_cfg,
        task_count=len(animation_jobs),
    )
    if max_workers <= 1:
        return {
            str(summary_key): _run_temporal_animation_task(task_kind, task_kwargs)
            for summary_key, task_kind, task_kwargs in animation_jobs
        }

    results: dict[str, Any] = {}
    with ProcessPoolExecutor(
        max_workers=int(max_workers),
        mp_context=_temporal_animation_process_context(),
    ) as executor:
        submitted = [
            (
                str(summary_key),
                executor.submit(_run_temporal_animation_task, task_kind, task_kwargs),
            )
            for summary_key, task_kind, task_kwargs in animation_jobs
        ]
        for summary_key, future in submitted:
            results[str(summary_key)] = future.result()
    return results


def _remove_existing_transition_pair_flow_artifacts(out_dir: Path) -> None:
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return
    for pattern in ("transition_pair_*_flow.png", "transition_pair_*_flow.svg"):
        for artifact_path in out_dir.glob(pattern):
            artifact_path.unlink()


def _remove_existing_transition_aggregate_flow_artifacts(out_dir: Path) -> None:
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return
    for name in ("transition_aggregate_flow.png", "transition_aggregate_flow.svg"):
        artifact_path = out_dir / name
        if artifact_path.exists():
            artifact_path.unlink()


def _relative_path_for_markdown(target_path: Path, *, base_dir: Path) -> str:
    target_resolved = Path(target_path).resolve()
    base_resolved = Path(base_dir).resolve()
    return os.path.relpath(target_resolved, start=base_resolved)


def _resolve_transition_pair_plot_indices(
    transition_cfg: Any,
    *,
    pair_count: int,
) -> list[int]:
    if pair_count <= 0:
        return []
    enabled = _cfg_bool(transition_cfg, "pair_plots_enabled", pair_count <= 12)
    if not enabled:
        return []
    max_count = _cfg_int(transition_cfg, "pair_plots_max_count", pair_count)
    if max_count <= 0:
        return []
    if max_count >= pair_count:
        return list(range(pair_count))
    selected = np.linspace(0, pair_count - 1, num=max_count, dtype=np.float64)
    return sorted({int(round(value)) for value in selected.tolist()})


def _resolve_descriptor_sampling_indices(
    frames: list[FrameSlice],
    labels: np.ndarray,
    *,
    max_samples: int | None,
    random_state: int,
) -> np.ndarray:
    labels_arr = np.asarray(labels, dtype=int).reshape(-1)
    total_samples = int(labels_arr.shape[0])
    if max_samples is None or int(max_samples) <= 0 or int(max_samples) >= total_samples:
        return np.arange(total_samples, dtype=int)

    grouped_indices: list[np.ndarray] = []
    for frame in frames:
        frame_labels = labels_arr[frame.indices]
        for cluster_id in sorted(int(v) for v in np.unique(frame_labels) if int(v) >= 0):
            idx = frame.indices[frame_labels == int(cluster_id)]
            if idx.size > 0:
                grouped_indices.append(np.asarray(idx, dtype=int))
    if not grouped_indices:
        raise ValueError("No frame-cluster groups were found while sampling descriptor indices.")

    group_sizes = np.asarray([int(idx.size) for idx in grouped_indices], dtype=np.int64)
    budget = min(int(max_samples), int(np.sum(group_sizes)))
    base = np.zeros_like(group_sizes)
    if budget >= int(group_sizes.size):
        base += 1
        budget -= int(group_sizes.size)
    weights = group_sizes.astype(np.float64) / float(np.sum(group_sizes))
    fractional = weights * float(budget)
    quotas = base + np.floor(fractional).astype(np.int64)
    remainders = fractional - np.floor(fractional)
    leftover = int(min(int(max_samples), int(np.sum(group_sizes))) - int(np.sum(quotas)))
    if leftover > 0:
        order = np.argsort(remainders)[::-1]
        for pos in order:
            if leftover <= 0:
                break
            if quotas[pos] >= group_sizes[pos]:
                continue
            quotas[pos] += 1
            leftover -= 1
    if int(np.sum(quotas)) <= 0:
        raise RuntimeError(
            "Descriptor sampling resolved to zero samples despite a positive max_samples budget. "
            f"max_samples={max_samples}, num_groups={len(grouped_indices)}."
        )

    rng = np.random.default_rng(int(random_state))
    sampled: list[np.ndarray] = []
    for idx_group, quota in zip(grouped_indices, quotas, strict=True):
        if int(quota) <= 0:
            continue
        if int(quota) >= int(idx_group.size):
            sampled.append(np.asarray(idx_group, dtype=int))
            continue
        sampled.append(rng.choice(idx_group, size=int(quota), replace=False))

    sampled_all = np.unique(np.concatenate(sampled).astype(int, copy=False))
    return np.asarray(sampled_all, dtype=int)


def _load_point_cloud_batch(
    dataset: Any,
    sample_indices: np.ndarray,
    *,
    point_scale: float,
) -> np.ndarray:
    point_clouds: list[np.ndarray] = []
    for sample_idx in np.asarray(sample_indices, dtype=int):
        points = _load_point_cloud_from_dataset(
            dataset,
            int(sample_idx),
            point_scale=float(point_scale),
        )
        if points is None:
            raise RuntimeError(
                "Failed to load point cloud for descriptor analysis: "
                f"sample_index={int(sample_idx)}."
            )
        point_clouds.append(np.asarray(points, dtype=np.float32))
    if not point_clouds:
        raise ValueError("No point clouds were loaded for real-MD qualitative analysis.")
    return np.stack(point_clouds, axis=0)


def _build_frame_lookup_for_samples(frames: list[FrameSlice], *, num_samples: int) -> tuple[np.ndarray, list[str], np.ndarray]:
    frame_index = np.full((int(num_samples),), -1, dtype=int)
    frame_names = [""] * int(num_samples)
    frame_time_values = np.full((int(num_samples),), np.nan, dtype=np.float32)
    for frame in frames:
        frame_index[frame.indices] = int(frame.order_index)
        for idx in frame.indices:
            frame_names[int(idx)] = str(frame.source_name)
            if frame.time_value is not None:
                frame_time_values[int(idx)] = float(frame.time_value)
    return frame_index, frame_names, frame_time_values


def _build_builtin_descriptor_table(
    *,
    point_clouds: np.ndarray,
    sample_indices: np.ndarray,
    labels: np.ndarray,
    coords: np.ndarray,
    frame_index_lookup: np.ndarray,
    frame_name_lookup: list[str],
    frame_time_lookup: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row_idx, sample_idx in enumerate(np.asarray(sample_indices, dtype=int)):
        props = _compute_sample_properties(point_clouds[row_idx])
        record: dict[str, Any] = {
            "sample_index": int(sample_idx),
            "cluster_id": int(labels[int(sample_idx)]),
            "frame_index": int(frame_index_lookup[int(sample_idx)]),
            "frame_name": str(frame_name_lookup[int(sample_idx)]),
            "time_value": float(frame_time_lookup[int(sample_idx)]) if np.isfinite(frame_time_lookup[int(sample_idx)]) else np.nan,
            "coord_x": float(coords[int(sample_idx), 0]),
            "coord_y": float(coords[int(sample_idx), 1]),
            "coord_z": float(coords[int(sample_idx), 2]),
        }
        for key in _BUILTIN_SCALAR_COLUMNS:
            record[key] = float(props.get(key, np.nan))
        records.append(record)
    return pd.DataFrame.from_records(records)


def _evaluate_optional_descriptor(
    descriptor_name: str,
    *,
    point_clouds: np.ndarray,
    point_scale: float,
    descriptor_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from src.baselines.descriptor_baselines import (
        CNADescriptorBaseline,
        SOAPDescriptorBaseline,
        SteinhardtDescriptorBaseline,
    )

    name = str(descriptor_name).strip().lower()
    if name == "steinhardt":
        baseline = SteinhardtDescriptorBaseline(
            l_values=descriptor_cfg.get("l_values", [4, 6]),
            center_atom_tolerance=float(descriptor_cfg.get("center_atom_tolerance", 1e-6)),
            shell_min_neighbors=int(descriptor_cfg.get("shell_min_neighbors", 8)),
            shell_max_neighbors=int(descriptor_cfg.get("shell_max_neighbors", 24)),
            append_shell_size=bool(descriptor_cfg.get("append_shell_size", True)),
        )
        features = baseline.transform(point_clouds)
        feature_names = [f"q_{int(v)}" for v in descriptor_cfg.get("l_values", [4, 6])]
        if bool(descriptor_cfg.get("append_shell_size", True)):
            feature_names.append("steinhardt_shell_size")
    elif name == "cna":
        baseline = CNADescriptorBaseline(
            center_atom_tolerance=float(descriptor_cfg.get("center_atom_tolerance", 1e-6)),
            shell_min_neighbors=int(descriptor_cfg.get("shell_min_neighbors", 8)),
            shell_max_neighbors=int(descriptor_cfg.get("shell_max_neighbors", 24)),
            max_signatures=int(descriptor_cfg.get("max_signatures", 12)),
            append_shell_size=bool(descriptor_cfg.get("append_shell_size", True)),
            fit_max_pointclouds=int(descriptor_cfg.get("fit_max_pointclouds", 4000)),
        )
        baseline.fit(point_clouds)
        features = baseline.transform(point_clouds)
        metadata = baseline.metadata()
        feature_names = [f"cna_{sig}" for sig in metadata["signature_vocab"]]
        feature_names.append("cna_other")
        if bool(descriptor_cfg.get("append_shell_size", True)):
            feature_names.append("cna_shell_size")
    elif name == "soap":
        soap_point_scale = descriptor_cfg.get("point_scale", None)
        baseline = SOAPDescriptorBaseline(
            species=str(descriptor_cfg.get("species", "Al")),
            point_scale=float(point_scale if soap_point_scale is None else soap_point_scale),
            center_atom_tolerance=float(descriptor_cfg.get("center_atom_tolerance", 1e-6)),
            shell_min_neighbors=int(descriptor_cfg.get("shell_min_neighbors", 8)),
            shell_max_neighbors=int(descriptor_cfg.get("shell_max_neighbors", 24)),
            r_cut=descriptor_cfg.get("r_cut", None),
            r_cut_multiplier=float(descriptor_cfg.get("r_cut_multiplier", 1.25)),
            r_cut_min=float(descriptor_cfg.get("r_cut_min", 1.01)),
            n_max=int(descriptor_cfg.get("n_max", 8)),
            l_max=int(descriptor_cfg.get("l_max", 6)),
            sigma=float(descriptor_cfg.get("sigma", 0.3)),
            pca_components=descriptor_cfg.get("pca_components", 16),
            fit_max_pointclouds=int(descriptor_cfg.get("fit_max_pointclouds", 4000)),
            n_jobs=int(descriptor_cfg.get("n_jobs", 1)),
        )
        baseline.fit(point_clouds)
        features = baseline.transform(point_clouds)
        feature_names = [f"soap_pc_{idx + 1:02d}" for idx in range(features.shape[1])]
    else:
        raise ValueError(
            f"Unsupported optional descriptor {descriptor_name!r}. "
            "Expected one of ['steinhardt', 'cna', 'soap']."
        )

    if features.shape[1] != len(feature_names):
        raise RuntimeError(
            "Descriptor feature-name mismatch: "
            f"descriptor={name}, features_shape={features.shape}, feature_names={feature_names}."
        )
    df = pd.DataFrame(features, columns=feature_names)
    return df, {"name": str(name), "feature_names": list(feature_names)}


def _prepare_projection_features(
    latents: np.ndarray,
    *,
    random_state: int,
    l2_normalize: bool,
    standardize: bool,
    pca_variance: float | None,
    pca_max_components: int,
) -> tuple[np.ndarray, ProjectionFeaturePrep, dict[str, Any]]:
    x = np.asarray(latents, dtype=np.float32)
    if x.ndim != 2:
        x = np.reshape(x, (x.shape[0], -1))
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    info: dict[str, Any] = {
        "input_dim": int(x.shape[1]),
        "l2_normalize": bool(l2_normalize),
        "standardize": bool(standardize),
    }

    if bool(l2_normalize):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.clip(norms, 1e-8, None)

    scaler: StandardScaler | None = None
    if bool(standardize) and x.shape[0] > 1:
        scaler = StandardScaler()
        x = scaler.fit_transform(x).astype(np.float32, copy=False)

    pca_model: PCA | None = None
    pca_keep = int(x.shape[1])
    use_pca = (
        pca_variance is not None
        and float(pca_variance) > 0.0
        and x.shape[1] > 2
        and x.shape[0] > 3
    )
    if use_pca:
        n_max = min(int(pca_max_components), x.shape[1], x.shape[0] - 1)
        if n_max >= 2:
            pca_model = PCA(n_components=n_max, random_state=int(random_state))
            x_proj = pca_model.fit_transform(x)
            if float(pca_variance) >= 1.0:
                pca_keep = int(n_max)
            else:
                csum = np.cumsum(pca_model.explained_variance_ratio_)
                pca_keep = int(np.searchsorted(csum, float(pca_variance)) + 1)
                pca_keep = max(2, min(pca_keep, int(n_max)))
            x = np.asarray(x_proj[:, :pca_keep], dtype=np.float32)
            info["pca_components"] = int(pca_keep)
            info["pca_explained_variance"] = float(
                np.sum(pca_model.explained_variance_ratio_[:pca_keep])
            )
        else:
            info["pca_components"] = int(x.shape[1])
            info["pca_explained_variance"] = 1.0
    else:
        info["pca_components"] = int(x.shape[1])
        info["pca_explained_variance"] = 1.0

    info["output_dim"] = int(x.shape[1])
    return x.astype(np.float32, copy=False), ProjectionFeaturePrep(
        l2_normalize=bool(l2_normalize),
        scaler=scaler,
        pca=pca_model,
        pca_keep_components=int(pca_keep),
    ), info


def _fit_projection_embedding(
    features: np.ndarray,
    *,
    method: str,
    random_state: int,
    umap_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    umap_backend: str,
    tsne_n_iter: int,
) -> tuple[np.ndarray, dict[str, Any], Any | None]:
    method_norm = str(method).strip().lower()
    if method_norm == "umap":
        backend_info = _resolve_umap_backend(str(umap_backend))
        reducer = _build_umap_reducer(
            backend_info,
            random_state=int(random_state),
            umap_neighbors=int(umap_neighbors),
            umap_min_dist=float(umap_min_dist),
            umap_metric=str(umap_metric),
        )
        try:
            embedding = reducer.fit_transform(features)
        except Exception as exc:
            raise RuntimeError(
                "UMAP fit_transform failed. "
                f"backend={backend_info['backend']}, device={backend_info['device']}, "
                f"requested_backend={backend_info['requested_backend']}, "
                f"features_shape={features.shape}, features_dtype={features.dtype}, "
                f"n_neighbors={int(umap_neighbors)}, min_dist={float(umap_min_dist)}, "
                f"metric={str(umap_metric)!r}."
            ) from exc
        projection_info = {
            "method": "umap",
            "n_neighbors": int(umap_neighbors),
            "min_dist": float(umap_min_dist),
            "metric": str(umap_metric),
            "backend": str(backend_info["backend"]),
            "device": str(backend_info["device"]),
            "backend_preference": str(backend_info["requested_backend"]),
            "gpu_available": bool(backend_info["gpu_available"]),
        }
        backend_reason = backend_info.get("reason")
        if backend_reason is not None:
            projection_info["backend_reason"] = str(backend_reason)
        return np.asarray(embedding, dtype=np.float32), projection_info, reducer
    if method_norm == "tsne":
        perplexity = min(50, max(5, len(features) // 100))
        embedding = compute_tsne(
            features,
            random_state=int(random_state),
            perplexity=int(perplexity),
            n_iter=int(tsne_n_iter),
        )
        return np.asarray(embedding, dtype=np.float32), {
            "method": "tsne",
            "perplexity": int(perplexity),
            "n_iter": int(tsne_n_iter),
        }, None
    if method_norm == "pca":
        pca = PCA(n_components=2)
        embedding = pca.fit_transform(features)
        return np.asarray(embedding, dtype=np.float32), {
            "method": "pca",
            "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_.tolist()],
        }, pca
    raise ValueError(
        "real_md.projection.method must be one of ['umap', 'tsne', 'pca'], "
        f"got {method!r}."
    )


def _transform_projection_features(
    feature_prep: ProjectionFeaturePrep,
    latents: np.ndarray,
    *,
    method: str,
    fitted_projection: Any,
) -> np.ndarray:
    x = np.asarray(latents, dtype=np.float32)
    if x.ndim != 2:
        x = np.reshape(x, (x.shape[0], -1))
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if feature_prep.l2_normalize:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.clip(norms, 1e-8, None)
    if feature_prep.scaler is not None:
        x = feature_prep.scaler.transform(x).astype(np.float32, copy=False)
    if feature_prep.pca is not None:
        x = feature_prep.pca.transform(x)[:, : int(feature_prep.pca_keep_components)]

    method_norm = str(method).strip().lower()
    if method_norm not in {"umap", "pca"}:
        raise ValueError(
            "Projection transforms are only available for methods ['umap', 'pca'], "
            f"got {method!r}."
        )
    if fitted_projection is None or not hasattr(fitted_projection, "transform"):
        raise RuntimeError(
            "Projection model does not provide a transform(...) method. "
            f"method={method!r}, projection_type={type(fitted_projection)!r}."
        )
    try:
        transformed = fitted_projection.transform(x)
    except Exception as exc:
        raise RuntimeError(
            "Projection transform failed. "
            f"method={method!r}, projection_type={type(fitted_projection)!r}, "
            f"input_shape={x.shape}, input_dtype={x.dtype}."
        ) from exc
    return np.asarray(transformed, dtype=np.float32)


def _compute_projection(
    latents: np.ndarray,
    *,
    method: str,
    random_state: int,
    l2_normalize: bool,
    standardize: bool,
    pca_variance: float | None,
    pca_max_components: int,
    umap_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    umap_backend: str,
    tsne_n_iter: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    features, _, prep_info = _prepare_projection_features(
        latents,
        random_state=int(random_state),
        l2_normalize=bool(l2_normalize),
        standardize=bool(standardize),
        pca_variance=pca_variance,
        pca_max_components=int(pca_max_components),
    )
    embedding, projection_info, _ = _fit_projection_embedding(
        features,
        method=method,
        random_state=int(random_state),
        umap_neighbors=int(umap_neighbors),
        umap_min_dist=float(umap_min_dist),
        umap_metric=str(umap_metric),
        umap_backend=str(umap_backend),
        tsne_n_iter=int(tsne_n_iter),
    )
    return embedding, {
        **prep_info,
        **projection_info,
    }


def _resolve_umap_backend(requested_backend: str) -> dict[str, Any]:
    backend_norm = str(requested_backend).strip().lower() or "auto"
    if backend_norm not in {"auto", "cpu", "gpu"}:
        raise ValueError(
            "real_md.projection.umap_backend must be one of ['auto', 'cpu', 'gpu'], "
            f"got {requested_backend!r}."
        )

    gpu_available = False
    gpu_probe_error: str | None = None
    try:
        import torch
    except ImportError as exc:
        gpu_probe_error = f"PyTorch import failed while probing CUDA availability: {exc}"
    else:
        gpu_available = bool(torch.cuda.is_available())

    if backend_norm == "cpu":
        return {
            "backend": "umap-learn",
            "device": "cpu",
            "requested_backend": backend_norm,
            "gpu_available": bool(gpu_available),
            "reason": "CPU backend forced by real_md.projection.umap_backend='cpu'.",
        }

    if gpu_available:
        try:
            from cuml.manifold import UMAP as CuMLUMAP
        except ImportError as exc:
            if backend_norm == "gpu":
                raise ImportError(
                    "real_md.projection.umap_backend='gpu' requires RAPIDS cuML "
                    "(cuml.manifold.UMAP), but it is not installed."
                ) from exc
            return {
                "backend": "umap-learn",
                "device": "cpu",
                "requested_backend": backend_norm,
                "gpu_available": True,
                "reason": (
                    "CUDA is available, but RAPIDS cuML is not installed. "
                    "Falling back to CPU umap-learn."
                ),
            }
        return {
            "backend": "cuml",
            "device": "gpu",
            "requested_backend": backend_norm,
            "gpu_available": True,
            "reason": None,
            "umap_class": CuMLUMAP,
        }

    if backend_norm == "gpu":
        raise RuntimeError(
            "real_md.projection.umap_backend='gpu' requires a CUDA-capable runtime, "
            f"but torch.cuda.is_available() returned False. Probe details: {gpu_probe_error or 'ok'}."
        )

    return {
        "backend": "umap-learn",
        "device": "cpu",
        "requested_backend": backend_norm,
        "gpu_available": False,
        "reason": (
            "CUDA is not available for UMAP acceleration."
            if gpu_probe_error is None
            else gpu_probe_error
        ),
    }


def _build_umap_reducer(
    backend_info: dict[str, Any],
    *,
    random_state: int,
    umap_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
) -> Any:
    reducer_kwargs = {
        "n_components": 2,
        "n_neighbors": int(umap_neighbors),
        "min_dist": float(umap_min_dist),
        "metric": str(umap_metric),
    }
    if str(backend_info["backend"]) == "cuml":
        import inspect

        reducer_cls = backend_info.get("umap_class")
        if reducer_cls is None:
            raise RuntimeError("Missing cuml UMAP class in backend_info.")
        try:
            reducer_signature = inspect.signature(reducer_cls.__init__)
        except (TypeError, ValueError):
            reducer_signature = None
        if reducer_signature is not None and "output_type" in reducer_signature.parameters:
            reducer_kwargs["output_type"] = "numpy"
        if reducer_signature is not None and "random_state" in reducer_signature.parameters:
            reducer_kwargs["random_state"] = None
        return reducer_cls(**reducer_kwargs)

    try:
        import umap
    except ImportError as exc:
        raise ImportError("UMAP projection requested but umap-learn is not installed.") from exc
    import inspect

    try:
        reducer_signature = inspect.signature(umap.UMAP.__init__)
    except (TypeError, ValueError):
        reducer_signature = None
    if reducer_signature is not None:
        if "n_jobs" in reducer_signature.parameters:
            reducer_kwargs["n_jobs"] = -1
        if "random_state" in reducer_signature.parameters:
            reducer_kwargs["random_state"] = None
        if "transform_seed" in reducer_signature.parameters:
            reducer_kwargs["transform_seed"] = None
    return umap.UMAP(**reducer_kwargs)




def _build_cluster_groups(
    real_md_cfg: Any,
    *,
    frames: list[FrameSlice],
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    groups_raw = _to_plain(getattr(real_md_cfg, "cluster_groups", None))
    order_raw = _to_plain(getattr(real_md_cfg, "cluster_group_order", None))
    groups: list[dict[str, Any]] = []
    if isinstance(groups_raw, dict) and groups_raw:
        requested_order = [str(v) for v in list(order_raw)] if order_raw is not None else list(groups_raw.keys())
        for name in requested_order:
            if name not in groups_raw:
                raise KeyError(
                    "real_md.cluster_group_order references a missing group: "
                    f"{name!r}. Available groups: {sorted(groups_raw.keys())}."
                )
            cluster_ids = sorted(_normalize_cluster_id_list(groups_raw[name], field_name=f"real_md.cluster_groups.{name}"))
            groups.append(
                {
                    "name": str(name),
                    "cluster_ids": cluster_ids,
                    "source": "config",
                }
            )
        return groups

    if not frames:
        return []

    auto_top_clusters = max(0, _cfg_int(real_md_cfg, "auto_top_clusters", 2))
    if auto_top_clusters == 0:
        return []
    latest_frame = frames[-1]
    latest_labels = labels[latest_frame.indices]
    cluster_ids, counts = np.unique(latest_labels[latest_labels >= 0], return_counts=True)
    if cluster_ids.size == 0:
        return []
    order = np.argsort(counts)[::-1]
    for pos in order[:auto_top_clusters]:
        cluster_id = int(cluster_ids[int(pos)])
        groups.append(
            {
                "name": f"cluster_{cluster_id}",
                "cluster_ids": [cluster_id],
                "source": "auto_latest_frame",
            }
        )
    return groups


def _cluster_counts_by_frame(
    frames: list[FrameSlice],
    labels: np.ndarray,
) -> tuple[pd.DataFrame, list[int], np.ndarray]:
    labels_arr = np.asarray(labels, dtype=int).reshape(-1)
    cluster_ids = sorted(int(v) for v in np.unique(labels_arr) if int(v) >= 0)
    rows: list[dict[str, Any]] = []
    counts_matrix = np.zeros((len(frames), len(cluster_ids)), dtype=np.int64)
    for frame_pos, frame in enumerate(frames):
        frame_labels = labels_arr[frame.indices]
        count_lookup = {int(k): int(v) for k, v in zip(*np.unique(frame_labels[frame_labels >= 0], return_counts=True))}
        total = int(frame.indices.size)
        for cluster_pos, cluster_id in enumerate(cluster_ids):
            counts_matrix[frame_pos, cluster_pos] = int(count_lookup.get(int(cluster_id), 0))
        row: dict[str, Any] = {
            "frame_index": int(frame.order_index),
            "frame_name": str(frame.source_name),
            "frame_label": str(frame.label),
            "output_name": str(frame.output_name),
            "time_value": np.nan if frame.time_value is None else float(frame.time_value),
            "time_unit": "" if frame.time_unit is None else str(frame.time_unit),
            "num_samples": total,
        }
        for cluster_pos, cluster_id in enumerate(cluster_ids):
            count_val = int(counts_matrix[frame_pos, cluster_pos])
            row[f"cluster_{cluster_id}_count"] = count_val
            row[f"cluster_{cluster_id}_fraction"] = float(count_val / total) if total > 0 else np.nan
        rows.append(row)
    return pd.DataFrame.from_records(rows), cluster_ids, counts_matrix


def _select_temporal_animation_sample_indices(
    frames: list[FrameSlice],
    labels: np.ndarray,
    *,
    max_points_per_frame: int | None,
    random_state: int,
) -> dict[str, np.ndarray]:
    selected: dict[str, np.ndarray] = {}
    labels_arr = np.asarray(labels, dtype=int).reshape(-1)
    for frame in frames:
        frame_indices = np.asarray(frame.indices, dtype=int)
        frame_labels = labels_arr[frame_indices]
        local_indices = _sample_indices_stratified(
            frame_labels,
            max_points_per_frame,
            random_seed=int(random_state) + 1009 + int(frame.order_index),
        )
        selected[str(frame.source_name)] = frame_indices[local_indices].astype(int, copy=False)
    return selected


def _select_temporal_trajectory_sample_indices(
    frames: list[FrameSlice],
    labels: np.ndarray,
    instance_ids: np.ndarray | None,
    *,
    max_points: int | None,
    random_state: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if instance_ids is None:
        raise ValueError(
            "Temporal trajectory animation requires instance_ids so local structures can be tracked across frames."
        )
    labels_arr = np.asarray(labels, dtype=int).reshape(-1)
    instance_ids_arr = np.asarray(instance_ids, dtype=np.int64).reshape(-1)
    if labels_arr.shape[0] != instance_ids_arr.shape[0]:
        raise ValueError(
            "labels and instance_ids length mismatch for temporal trajectory sampling: "
            f"labels={labels_arr.shape[0]}, instance_ids={instance_ids_arr.shape[0]}."
        )

    sorted_ids_by_frame: dict[str, np.ndarray] = {}
    sorted_indices_by_frame: dict[str, np.ndarray] = {}
    common_ids: np.ndarray | None = None
    for frame in frames:
        frame_indices = np.asarray(frame.indices, dtype=int)
        frame_instance_ids = instance_ids_arr[frame_indices]
        order = np.argsort(frame_instance_ids, kind="mergesort")
        sorted_ids = frame_instance_ids[order].astype(np.int64, copy=False)
        if sorted_ids.size == 0:
            raise ValueError(
                f"Temporal trajectory animation encountered an empty frame: {frame.source_name!r}."
            )
        if np.any(np.diff(sorted_ids) == 0):
            raise ValueError(
                "Temporal trajectory animation requires unique instance_ids within each frame, "
                f"but duplicates were found in frame {frame.source_name!r}."
            )
        sorted_indices = frame_indices[order].astype(int, copy=False)
        sorted_ids_by_frame[str(frame.source_name)] = sorted_ids
        sorted_indices_by_frame[str(frame.source_name)] = sorted_indices
        common_ids = (
            sorted_ids.copy()
            if common_ids is None
            else np.intersect1d(common_ids, sorted_ids, assume_unique=True)
        )

    assert common_ids is not None
    if common_ids.size == 0:
        raise RuntimeError("Temporal trajectory animation found no instance_ids shared across frames.")

    reference_frame = frames[0]
    reference_name = str(reference_frame.source_name)
    reference_ids = sorted_ids_by_frame[reference_name]
    reference_indices = sorted_indices_by_frame[reference_name]
    reference_pos = np.searchsorted(reference_ids, common_ids)
    if np.any(reference_pos < 0) or np.any(reference_pos >= reference_ids.size):
        raise RuntimeError(
            "Failed to resolve shared instance_ids in the reference frame for temporal trajectory animation."
        )
    if not np.array_equal(reference_ids[reference_pos], common_ids):
        raise RuntimeError(
            "Reference-frame instance_id ordering mismatch while preparing temporal trajectory animation."
        )
    reference_global_indices = reference_indices[reference_pos]
    reference_labels = labels_arr[reference_global_indices]
    selected_local = _sample_indices_stratified(
        reference_labels,
        max_points,
        random_seed=int(random_state) + 1703,
    )
    selected_ids = common_ids[selected_local].astype(np.int64, copy=False)
    if selected_ids.size == 0:
        raise RuntimeError("Temporal trajectory animation selected zero tracked local structures.")

    selected_indices_by_frame: dict[str, np.ndarray] = {}
    for frame in frames:
        frame_name = str(frame.source_name)
        sorted_ids = sorted_ids_by_frame[frame_name]
        sorted_indices = sorted_indices_by_frame[frame_name]
        selected_pos = np.searchsorted(sorted_ids, selected_ids)
        if np.any(selected_pos < 0) or np.any(selected_pos >= sorted_ids.size):
            raise RuntimeError(
                "Temporal trajectory animation found out-of-range positions while resolving selected instance_ids. "
                f"frame={frame_name!r}."
            )
        if not np.array_equal(sorted_ids[selected_pos], selected_ids):
            raise RuntimeError(
                "Temporal trajectory animation lost one or more selected instance_ids in a frame. "
                f"frame={frame_name!r}."
            )
        selected_indices_by_frame[frame_name] = sorted_indices[selected_pos].astype(int, copy=False)
    return selected_indices_by_frame, selected_ids


def _resolve_transition_tolerance(
    *,
    model_cfg: Any,
    dataset: Any,
    transition_cfg: Any,
) -> float:
    radius_raw = getattr(model_cfg.data, "radius", None)
    if radius_raw is None:
        radius_raw = getattr(dataset, "radius", None)
    radius = 8.0 if radius_raw is None else float(radius_raw)

    center_grid_overlap_raw = getattr(dataset, "center_grid_overlap", None)
    if center_grid_overlap_raw is not None and getattr(dataset, "radius", None) is not None:
        stride = (2.0 - float(center_grid_overlap_raw)) * float(radius)
    else:
        overlap_fraction_raw = getattr(model_cfg.data, "overlap_fraction", None)
        overlap_fraction = 0.0 if overlap_fraction_raw is None else float(overlap_fraction_raw)
        stride = float(radius) * (1.0 - float(overlap_fraction))
    default_transition_tol = max(1e-3, 0.5 * stride)
    transition_tol_cfg = getattr(transition_cfg, "max_distance", None)
    return float(default_transition_tol if transition_tol_cfg is None else transition_tol_cfg)


def _build_cna_signature_summary_tables(
    descriptor_table: pd.DataFrame,
    *,
    cluster_ids: list[int],
    frames: list[FrameSlice],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    cna_columns = [
        str(column)
        for column in descriptor_table.columns
        if str(column).startswith("cna_") and str(column) != "cna_shell_size"
    ]
    if not cna_columns:
        raise ValueError("descriptor_table does not contain any CNA signature columns.")

    cluster_summary = (
        descriptor_table.groupby("cluster_id", sort=True)[cna_columns]
        .mean(numeric_only=True)
        .reindex(cluster_ids)
        .reset_index()
    )
    if "frame_index" not in descriptor_table.columns:
        raise KeyError("descriptor_table must contain 'frame_index' for CNA time-series plots.")
    frame_summary = (
        descriptor_table.groupby("frame_index", sort=True)[cna_columns]
        .mean(numeric_only=True)
        .reindex([int(frame.order_index) for frame in frames])
        .reset_index()
    )
    frame_summary["frame_label"] = [str(frame.label) for frame in frames]
    return cluster_summary, frame_summary, cna_columns


def _compute_transitions(
    frames: list[FrameSlice],
    *,
    coords: np.ndarray,
    labels: np.ndarray,
    instance_ids: np.ndarray | None,
    cluster_ids: list[int],
    max_distance: float,
    require_mutual: bool,
) -> dict[str, Any]:
    cluster_to_pos = {int(cluster_id): int(pos) for pos, cluster_id in enumerate(cluster_ids)}
    aggregate_counts = np.zeros((len(cluster_ids), len(cluster_ids)), dtype=np.int64)
    pair_summaries: list[dict[str, Any]] = []
    match_mode = "nearest_neighbor"
    instance_ids_arr = None if instance_ids is None else np.asarray(instance_ids, dtype=np.int64).reshape(-1)
    for frame_a, frame_b in zip(frames[:-1], frames[1:], strict=True):
        coords_a = np.asarray(coords[frame_a.indices, :3], dtype=np.float32)
        coords_b = np.asarray(coords[frame_b.indices, :3], dtype=np.float32)
        labels_a = np.asarray(labels[frame_a.indices], dtype=int)
        labels_b = np.asarray(labels[frame_b.indices], dtype=int)
        ids_a = None if instance_ids_arr is None else instance_ids_arr[frame_a.indices]
        ids_b = None if instance_ids_arr is None else instance_ids_arr[frame_b.indices]
        if coords_a.size == 0 or coords_b.size == 0:
            continue

        if ids_a is not None and ids_b is not None:
            match_mode = "instance_id"
            order_b = np.argsort(ids_b, kind="mergesort")
            ids_b_sorted = np.asarray(ids_b[order_b], dtype=np.int64)
            pos_b = np.searchsorted(ids_b_sorted, ids_a)
            match_mask = pos_b < ids_b_sorted.size
            match_mask &= ids_b_sorted[np.clip(pos_b, 0, max(ids_b_sorted.size - 1, 0))] == ids_a
            matched_a = np.flatnonzero(match_mask)
            matched_b = order_b[pos_b[match_mask]]
        else:
            tree_b = cKDTree(coords_b)
            dist_ab, idx_ab = tree_b.query(coords_a, k=1)
            dist_ab = np.asarray(dist_ab, dtype=np.float32)
            idx_ab = np.asarray(idx_ab, dtype=np.int64)

            if require_mutual:
                tree_a = cKDTree(coords_a)
                dist_ba, idx_ba = tree_a.query(coords_b, k=1)
                dist_ba = np.asarray(dist_ba, dtype=np.float32)
                idx_ba = np.asarray(idx_ba, dtype=np.int64)
                match_mask = (dist_ab <= float(max_distance)) & (idx_ba[idx_ab] == np.arange(len(coords_a)))
                match_mask &= dist_ba[idx_ab] <= float(max_distance)
            else:
                match_mask = dist_ab <= float(max_distance)
            matched_a = np.flatnonzero(match_mask)
            matched_b = idx_ab[matched_a]

        pair_counts = np.zeros((len(cluster_ids), len(cluster_ids)), dtype=np.int64)
        for idx_local_a, idx_local_b in zip(matched_a, matched_b, strict=True):
            cluster_a = int(labels_a[int(idx_local_a)])
            cluster_b = int(labels_b[int(idx_local_b)])
            if cluster_a < 0 or cluster_b < 0:
                continue
            pair_counts[cluster_to_pos[cluster_a], cluster_to_pos[cluster_b]] += 1
        aggregate_counts += pair_counts
        pair_summaries.append(
            {
                "frame_from": str(frame_a.source_name),
                "frame_to": str(frame_b.source_name),
                "matched_samples": int(np.sum(pair_counts)),
                "coverage_fraction": float(np.sum(pair_counts) / max(1, len(coords_a))),
                "counts": pair_counts,
            }
        )

    return {
        "cluster_ids": [int(v) for v in cluster_ids],
        "aggregate_counts": aggregate_counts,
        "pairs": pair_summaries,
        "max_distance": float(max_distance),
        "require_mutual": bool(require_mutual),
        "match_mode": str(match_mode),
    }


def _finite_metric_summary(values: np.ndarray | list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def _safe_fraction(numerator: int | float, denominator: int | float) -> float:
    denom = float(denominator)
    return float(numerator) / denom if denom > 0.0 else np.nan


def _safe_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size > 0 else np.nan


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, float(percentile))) if arr.size > 0 else np.nan


def _frame_time_delta(frame_a: FrameSlice, frame_b: FrameSlice) -> float:
    if (
        frame_a.time_value is None
        or frame_b.time_value is None
        or frame_a.time_unit is None
        or frame_b.time_unit is None
        or str(frame_a.time_unit) != str(frame_b.time_unit)
    ):
        return np.nan
    return float(frame_b.time_value) - float(frame_a.time_value)


def _median_frame_spacing(frames: list[FrameSlice]) -> tuple[float | None, str | None]:
    if len(frames) < 2:
        return None, None
    units = {str(frame.time_unit) for frame in frames if frame.time_unit is not None}
    if len(units) != 1:
        return None, None
    times = np.asarray(
        [
            np.nan if frame.time_value is None else float(frame.time_value)
            for frame in frames
        ],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(times)):
        return None, None
    deltas = np.diff(times)
    if deltas.size == 0 or np.any(~np.isfinite(deltas)):
        return None, None
    return float(np.median(deltas)), str(next(iter(units)))


def _build_label_to_cluster_pos(cluster_ids: list[int]) -> np.ndarray:
    if not cluster_ids:
        raise ValueError("Flicker metrics require at least one non-negative cluster ID.")
    min_cluster_id = min(int(v) for v in cluster_ids)
    if min_cluster_id < 0:
        raise ValueError(
            "Flicker metrics expect non-negative cluster IDs, "
            f"got minimum cluster ID {min_cluster_id}."
        )
    label_to_pos = np.full(max(int(v) for v in cluster_ids) + 1, -1, dtype=np.int64)
    for pos, cluster_id in enumerate(cluster_ids):
        label_to_pos[int(cluster_id)] = int(pos)
    return label_to_pos


def _labels_to_cluster_positions(labels: np.ndarray, label_to_pos: np.ndarray) -> np.ndarray:
    labels_arr = np.asarray(labels, dtype=int).reshape(-1)
    positions = np.full(labels_arr.shape, -1, dtype=np.int64)
    in_range = (labels_arr >= 0) & (labels_arr < int(label_to_pos.shape[0]))
    positions[in_range] = label_to_pos[labels_arr[in_range]]
    return positions


def _build_sorted_temporal_metric_records(
    frames: list[FrameSlice],
    *,
    coords: np.ndarray,
    labels: np.ndarray,
    instance_ids: np.ndarray,
) -> list[dict[str, Any]]:
    coords_arr = np.asarray(coords, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=int).reshape(-1)
    instance_ids_arr = np.asarray(instance_ids, dtype=np.int64).reshape(-1)
    if coords_arr.ndim != 2 or coords_arr.shape[1] < 3:
        raise ValueError(
            "Flicker metrics require coords with shape (N, >=3), "
            f"got shape={tuple(coords_arr.shape)}."
        )
    if labels_arr.shape[0] != coords_arr.shape[0]:
        raise ValueError(
            "Flicker metrics require matching coords/labels lengths, "
            f"coords={coords_arr.shape[0]}, labels={labels_arr.shape[0]}."
        )
    if instance_ids_arr.shape[0] != labels_arr.shape[0]:
        raise ValueError(
            "Flicker metrics require one instance_id per label, "
            f"instance_ids={instance_ids_arr.shape[0]}, labels={labels_arr.shape[0]}."
        )

    records: list[dict[str, Any]] = []
    for frame in frames:
        frame_indices = np.asarray(frame.indices, dtype=int).reshape(-1)
        frame_ids = instance_ids_arr[frame_indices]
        order = np.argsort(frame_ids, kind="mergesort")
        sorted_ids = np.asarray(frame_ids[order], dtype=np.int64)
        if sorted_ids.size == 0:
            raise ValueError(
                f"Flicker metrics encountered an empty frame: {frame.source_name!r}."
            )
        duplicate_mask = np.diff(sorted_ids) == 0
        if np.any(duplicate_mask):
            duplicate_id = int(sorted_ids[int(np.flatnonzero(duplicate_mask)[0])])
            raise ValueError(
                "Flicker metrics require unique instance_ids within each frame. "
                f"frame={frame.source_name!r}, duplicate_instance_id={duplicate_id}."
            )
        sorted_indices = frame_indices[order].astype(int, copy=False)
        records.append(
            {
                "frame": frame,
                "ids": sorted_ids,
                "indices": sorted_indices,
                "labels": labels_arr[sorted_indices],
                "coords": coords_arr[sorted_indices, :3],
            }
        )
    return records


def _match_sorted_temporal_records(
    record_a: dict[str, Any],
    record_b: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    ids_a = np.asarray(record_a["ids"], dtype=np.int64)
    ids_b = np.asarray(record_b["ids"], dtype=np.int64)
    pos_b = np.searchsorted(ids_b, ids_a)
    in_range = pos_b < ids_b.size
    match_mask = in_range.copy()
    if ids_b.size > 0:
        clipped = np.clip(pos_b, 0, ids_b.size - 1)
        match_mask &= ids_b[clipped] == ids_a
    matched_a = np.flatnonzero(match_mask)
    matched_b = pos_b[match_mask]
    return matched_a.astype(np.int64, copy=False), matched_b.astype(np.int64, copy=False)


def _transition_count_matrix(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    *,
    label_to_pos: np.ndarray,
    cluster_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    pos_a = _labels_to_cluster_positions(labels_a, label_to_pos)
    pos_b = _labels_to_cluster_positions(labels_b, label_to_pos)
    valid = (pos_a >= 0) & (pos_b >= 0)
    counts = np.zeros((int(cluster_count), int(cluster_count)), dtype=np.int64)
    np.add.at(counts, (pos_a[valid], pos_b[valid]), 1)
    return counts, valid


def _spatial_coherence_metrics_for_pair(
    *,
    coords_a: np.ndarray,
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    neighbor_k: int,
    isolated_neighbor_change_fraction: float,
    coherent_same_transition_fraction: float,
) -> dict[str, Any]:
    changed = np.asarray(labels_a, dtype=int).reshape(-1) != np.asarray(labels_b, dtype=int).reshape(-1)
    changed_count = int(np.count_nonzero(changed))
    matched_count = int(np.asarray(labels_a).reshape(-1).shape[0])
    result: dict[str, Any] = {
        "spatial_neighbor_k": int(min(max(int(neighbor_k), 0), max(matched_count - 1, 0))),
        "spatial_changed_samples": changed_count,
        "spatial_neighbor_changed_fraction_mean": np.nan,
        "spatial_same_transition_fraction_mean": np.nan,
        "spatial_source_purity_before_mean": np.nan,
        "spatial_target_purity_after_mean": np.nan,
        "spatial_isolated_change_rate": np.nan,
        "spatial_coherent_change_rate": np.nan,
    }
    if changed_count == 0 or matched_count <= 1 or int(neighbor_k) <= 0:
        return result

    k_eff = min(int(neighbor_k), matched_count - 1)
    tree = cKDTree(np.asarray(coords_a, dtype=np.float32))
    _, neighbor_idx = tree.query(
        np.asarray(coords_a, dtype=np.float32)[changed],
        k=int(k_eff) + 1,
    )
    neighbor_idx = np.asarray(neighbor_idx, dtype=np.int64)
    if neighbor_idx.ndim != 2 or neighbor_idx.shape[1] != int(k_eff) + 1:
        raise RuntimeError(
            "Unexpected nearest-neighbor index shape while computing spatial coherence: "
            f"shape={tuple(neighbor_idx.shape)}, expected_second_dim={int(k_eff) + 1}."
        )
    neighbor_idx = neighbor_idx[:, 1:]
    labels_a_arr = np.asarray(labels_a, dtype=int).reshape(-1)
    labels_b_arr = np.asarray(labels_b, dtype=int).reshape(-1)
    source = labels_a_arr[changed][:, None]
    target = labels_b_arr[changed][:, None]
    neighbor_source = labels_a_arr[neighbor_idx]
    neighbor_target = labels_b_arr[neighbor_idx]
    neighbor_changed = neighbor_source != neighbor_target
    same_transition = (neighbor_source == source) & (neighbor_target == target)
    source_purity = neighbor_source == source
    target_purity = neighbor_target == target
    neighbor_changed_fraction = np.mean(neighbor_changed, axis=1)
    same_transition_fraction = np.mean(same_transition, axis=1)
    result.update(
        {
            "spatial_neighbor_k": int(k_eff),
            "spatial_neighbor_changed_fraction_mean": float(np.mean(neighbor_changed_fraction)),
            "spatial_same_transition_fraction_mean": float(np.mean(same_transition_fraction)),
            "spatial_source_purity_before_mean": float(np.mean(source_purity)),
            "spatial_target_purity_after_mean": float(np.mean(target_purity)),
            "spatial_isolated_change_rate": float(
                np.mean(neighbor_changed_fraction <= float(isolated_neighbor_change_fraction))
            ),
            "spatial_coherent_change_rate": float(
                np.mean(same_transition_fraction >= float(coherent_same_transition_fraction))
            ),
        }
    )
    return result


def _margin_stats_for_mask(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    prefix: str,
) -> dict[str, Any]:
    selected = np.asarray(values, dtype=np.float32).reshape(-1)[np.asarray(mask, dtype=bool).reshape(-1)]
    return {
        f"{prefix}_count": int(selected.size),
        f"{prefix}_mean": _safe_mean(selected),
        f"{prefix}_p10": _safe_percentile(selected, 10),
        f"{prefix}_median": _safe_percentile(selected, 50),
    }


def _resolve_assignment_margin_arrays(
    assignment_margins: dict[str, Any] | None,
    *,
    expected_length: int,
) -> dict[str, Any] | None:
    if assignment_margins is None:
        return None
    required = ("margin", "assigned_score", "runner_up_score", "runner_up_cluster")
    resolved: dict[str, Any] = {
        "score_name": str(assignment_margins.get("score_name", "assignment_score")),
        "higher_is_better": bool(assignment_margins.get("higher_is_better", True)),
    }
    for key in required:
        arr = np.asarray(assignment_margins[key])
        if arr.shape[0] != int(expected_length):
            raise ValueError(
                "Assignment margin array length mismatch for flicker metrics: "
                f"key={key!r}, length={arr.shape[0]}, expected={expected_length}."
            )
        resolved[key] = arr.reshape(-1)
    return resolved


def _common_label_matrix(
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    common_ids = np.asarray(records[0]["ids"], dtype=np.int64)
    for record in records[1:]:
        common_ids = np.intersect1d(
            common_ids,
            np.asarray(record["ids"], dtype=np.int64),
            assume_unique=True,
        )
    if common_ids.size == 0:
        raise RuntimeError("Flicker trajectory metrics found no instance_ids shared by all frames.")

    label_matrix = np.empty((len(records), int(common_ids.size)), dtype=np.int32)
    for frame_pos, record in enumerate(records):
        ids = np.asarray(record["ids"], dtype=np.int64)
        pos = np.searchsorted(ids, common_ids)
        if np.any(pos < 0) or np.any(pos >= ids.size):
            raise RuntimeError(
                "Flicker trajectory metrics found out-of-range positions while aligning common IDs."
            )
        if not np.array_equal(ids[pos], common_ids):
            raise RuntimeError(
                "Flicker trajectory metrics failed to align common instance_ids across frames."
            )
        label_matrix[frame_pos] = np.asarray(record["labels"], dtype=int)[pos]
    return common_ids, label_matrix


def _write_dwell_metrics(
    *,
    label_matrix: np.ndarray,
    cluster_ids: list[int],
    out_dir: Path,
    frame_spacing: float | None,
    time_unit: str | None,
) -> tuple[Path, Path, dict[str, Any]]:
    dwell_lengths_by_cluster: dict[int, list[int]] = {int(cluster_id): [] for cluster_id in cluster_ids}
    labels_by_track = np.asarray(label_matrix, dtype=np.int32).T
    for sequence in labels_by_track:
        run_label = int(sequence[0])
        run_length = 1
        for label in sequence[1:]:
            label_int = int(label)
            if label_int == run_label:
                run_length += 1
                continue
            if run_label in dwell_lengths_by_cluster:
                dwell_lengths_by_cluster[run_label].append(int(run_length))
            run_label = label_int
            run_length = 1
        if run_label in dwell_lengths_by_cluster:
            dwell_lengths_by_cluster[run_label].append(int(run_length))

    hist_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    all_lengths: list[int] = []
    for cluster_id in cluster_ids:
        lengths = np.asarray(dwell_lengths_by_cluster[int(cluster_id)], dtype=np.int64)
        all_lengths.extend(int(v) for v in lengths.tolist())
        if lengths.size > 0:
            unique_lengths, counts = np.unique(lengths, return_counts=True)
            for dwell_frames, count in zip(unique_lengths, counts, strict=True):
                row = {
                    "cluster_id": int(cluster_id),
                    "dwell_frames": int(dwell_frames),
                    "run_count": int(count),
                }
                if frame_spacing is not None:
                    row["dwell_time"] = float(dwell_frames) * float(frame_spacing)
                    row["time_unit"] = str(time_unit)
                hist_rows.append(row)
        summary_row = {
            "cluster_id": int(cluster_id),
            "run_count": int(lengths.size),
            "mean_dwell_frames": _safe_mean(lengths),
            "median_dwell_frames": _safe_percentile(lengths, 50),
            "p90_dwell_frames": _safe_percentile(lengths, 90),
            "max_dwell_frames": int(np.max(lengths)) if lengths.size > 0 else 0,
            "fraction_runs_le_1_frame": _safe_fraction(int(np.count_nonzero(lengths <= 1)), int(lengths.size)),
            "fraction_runs_le_2_frames": _safe_fraction(int(np.count_nonzero(lengths <= 2)), int(lengths.size)),
        }
        if frame_spacing is not None:
            summary_row["mean_dwell_time"] = (
                float(summary_row["mean_dwell_frames"]) * float(frame_spacing)
                if np.isfinite(float(summary_row["mean_dwell_frames"]))
                else np.nan
            )
            summary_row["median_dwell_time"] = (
                float(summary_row["median_dwell_frames"]) * float(frame_spacing)
                if np.isfinite(float(summary_row["median_dwell_frames"]))
                else np.nan
            )
            summary_row["time_unit"] = str(time_unit)
        summary_rows.append(summary_row)

    hist_csv = out_dir / "dwell_time_histogram.csv"
    summary_csv = out_dir / "dwell_time_summary.csv"
    pd.DataFrame.from_records(hist_rows).to_csv(hist_csv, index=False)
    pd.DataFrame.from_records(summary_rows).to_csv(summary_csv, index=False)
    all_lengths_arr = np.asarray(all_lengths, dtype=np.float64)
    return hist_csv, summary_csv, {
        "run_count": int(all_lengths_arr.size),
        "dwell_frames": _finite_metric_summary(all_lengths_arr),
        "fraction_runs_le_1_frame": _safe_fraction(
            int(np.count_nonzero(all_lengths_arr <= 1)),
            int(all_lengths_arr.size),
        ),
        "fraction_runs_le_2_frames": _safe_fraction(
            int(np.count_nonzero(all_lengths_arr <= 2)),
            int(all_lengths_arr.size),
        ),
        "time_unit": None if time_unit is None else str(time_unit),
        "frame_spacing": None if frame_spacing is None else float(frame_spacing),
    }


def _write_recrossing_metrics(
    *,
    label_matrix: np.ndarray,
    cluster_ids: list[int],
    out_dir: Path,
    lags: list[int],
    frame_spacing: float | None,
    time_unit: str | None,
) -> tuple[Path, Path, dict[str, Any]]:
    labels_arr = np.asarray(label_matrix, dtype=np.int32)
    frame_count, track_count = labels_arr.shape
    aggregate_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for lag in lags:
        lag_int = int(lag)
        if lag_int <= 0:
            raise ValueError(f"Recrossing lags must be positive, got {lag}.")
        max_start = int(frame_count) - int(lag_int) - 1
        if max_start <= 0:
            continue
        source = labels_arr[:max_start]
        target = labels_arr[1 : max_start + 1]
        future = labels_arr[1 + lag_int : 1 + lag_int + max_start]
        changed = source != target
        transition_count = int(np.count_nonzero(changed))
        persistent = changed & (future == target)
        recrossed = changed & (future == source)
        other = changed & ~(persistent | recrossed)
        row = {
            "lag_frames": int(lag_int),
            "lag_time": (
                np.nan
                if frame_spacing is None
                else float(lag_int) * float(frame_spacing)
            ),
            "time_unit": "" if time_unit is None else str(time_unit),
            "eligible_transitions": transition_count,
            "persistent_fraction": _safe_fraction(int(np.count_nonzero(persistent)), transition_count),
            "recross_fraction": _safe_fraction(int(np.count_nonzero(recrossed)), transition_count),
            "other_fraction": _safe_fraction(int(np.count_nonzero(other)), transition_count),
        }
        aggregate_rows.append(row)
        for cluster_id in cluster_ids:
            source_mask = changed & (source == int(cluster_id))
            source_count = int(np.count_nonzero(source_mask))
            source_rows.append(
                {
                    "source_cluster_id": int(cluster_id),
                    "lag_frames": int(lag_int),
                    "lag_time": row["lag_time"],
                    "time_unit": row["time_unit"],
                    "eligible_transitions": source_count,
                    "persistent_fraction": _safe_fraction(
                        int(np.count_nonzero(source_mask & (future == target))),
                        source_count,
                    ),
                    "recross_fraction": _safe_fraction(
                        int(np.count_nonzero(source_mask & (future == source))),
                        source_count,
                    ),
                    "other_fraction": _safe_fraction(
                        int(np.count_nonzero(source_mask & ~(future == target) & ~(future == source))),
                        source_count,
                    ),
                }
            )

    bounce_mask = (
        (labels_arr[:-2] == labels_arr[2:])
        & (labels_arr[1:-1] != labels_arr[:-2])
    ) if frame_count >= 3 else np.zeros((0, track_count), dtype=bool)
    departure_mask = (labels_arr[1:-1] != labels_arr[:-2]) if frame_count >= 3 else np.zeros((0, track_count), dtype=bool)
    summary = {
        "tracked_instance_count": int(track_count),
        "frame_count": int(frame_count),
        "single_frame_bounce_count": int(np.count_nonzero(bounce_mask)),
        "single_frame_bounce_rate_per_observation": _safe_fraction(
            int(np.count_nonzero(bounce_mask)),
            int(bounce_mask.size),
        ),
        "single_frame_bounce_rate_per_departure": _safe_fraction(
            int(np.count_nonzero(bounce_mask)),
            int(np.count_nonzero(departure_mask)),
        ),
    }
    aggregate_csv = out_dir / "recrossing_metrics.csv"
    source_csv = out_dir / "recrossing_by_source_cluster.csv"
    pd.DataFrame.from_records(aggregate_rows).to_csv(aggregate_csv, index=False)
    pd.DataFrame.from_records(source_rows).to_csv(source_csv, index=False)
    return aggregate_csv, source_csv, summary


def _save_flicker_figure(fig: Any, out_file: Path) -> str:
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
    finally:
        plt.close(fig)
    print(f"[analysis][savefig] {out_path.resolve()}")
    return str(out_path)


def _flicker_table_x_axis(
    table: pd.DataFrame,
    *,
    time_column: str,
    fallback_column: str,
    fallback_label: str,
) -> tuple[np.ndarray, str]:
    if time_column in table.columns:
        x = pd.to_numeric(table[time_column], errors="coerce").to_numpy(dtype=np.float64)
        if x.size == len(table) and np.all(np.isfinite(x)):
            unit = ""
            if "time_unit" in table.columns:
                units = [
                    str(value).strip()
                    for value in table["time_unit"].dropna().unique().tolist()
                    if str(value).strip()
                ]
                if len(set(units)) == 1:
                    unit = str(units[0])
            label = "time" if not unit else f"time ({unit})"
            return x, label
    if fallback_column in table.columns:
        x = pd.to_numeric(table[fallback_column], errors="raise").to_numpy(dtype=np.float64)
        return x, fallback_label
    return np.arange(len(table), dtype=np.float64), fallback_label


def _style_flicker_axes(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.20, linewidth=0.7)
    ax.tick_params(axis="both", labelsize=9)


def _cluster_display_labels_for_flicker(
    cluster_ids: list[int],
    cluster_display_map: dict[int, str] | None,
) -> list[str]:
    labels: list[str] = []
    for cluster_id in cluster_ids:
        if cluster_display_map is not None and int(cluster_id) in cluster_display_map:
            labels.append(str(cluster_display_map[int(cluster_id)]))
        else:
            labels.append(f"C{int(cluster_id)}")
    return labels


def _cluster_colors_for_flicker(
    cluster_ids: list[int],
    cluster_color_map: dict[int, str] | None,
) -> dict[int, str]:
    fallback_map = _build_cluster_color_map(np.asarray(cluster_ids, dtype=int))
    resolved: dict[int, str] = {}
    for cluster_id in cluster_ids:
        cluster_int = int(cluster_id)
        if cluster_color_map is None:
            color = str(fallback_map[cluster_int])
        elif cluster_int in cluster_color_map:
            color = str(cluster_color_map[cluster_int])
        elif str(cluster_int) in cluster_color_map:
            color = str(cluster_color_map[str(cluster_int)])
        else:
            raise ValueError(
                "Missing cluster color for flicker visualization: "
                f"cluster_id={cluster_int}, "
                f"available={sorted(str(key) for key in cluster_color_map.keys())}."
            )
        mcolors.to_rgba(color)
        resolved[cluster_int] = color
    return resolved


def _rgba_with_alpha(color: str, alpha: float) -> tuple[float, float, float, float]:
    red, green, blue, _ = mcolors.to_rgba(color)
    return (float(red), float(green), float(blue), float(alpha))


def _weighted_metric_mean(
    table: pd.DataFrame,
    *,
    value_column: str,
    weight_column: str,
) -> float:
    values = pd.to_numeric(table[value_column], errors="coerce").to_numpy(dtype=np.float64)
    weights = pd.to_numeric(table[weight_column], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(mask):
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def _save_flicker_churn_timeseries_plot(
    pair_table: pd.DataFrame,
    out_file: Path,
) -> str:
    x, x_label = _flicker_table_x_axis(
        pair_table,
        time_column="time_to",
        fallback_column="pair_index",
        fallback_label="frame pair",
    )
    fig, ax = plt.subplots(figsize=(11.0, 4.8), dpi=220)
    specs = [
        ("label_churn_rate", "label churn", "#111827", "-", 2.25),
        ("excess_churn_rate", "excess churn", "#4b5563", "--", 2.0),
        ("reciprocal_pair_flicker_rate", "reciprocal pair flicker", "#0f766e", "-.", 1.9),
        ("net_population_tv", "net population TV", "#9ca3af", ":", 2.2),
    ]
    for column, label, color, linestyle, linewidth in specs:
        if column not in pair_table.columns:
            raise KeyError(f"Missing required flicker column for churn plot: {column!r}.")
        y = pd.to_numeric(pair_table[column], errors="raise").to_numpy(dtype=np.float64)
        ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=linewidth)
    ax.set_title("Frame-to-frame flicker rates")
    ax.set_xlabel(x_label)
    ax.set_ylabel("fraction of matched structures")
    ax.set_ylim(0.0, 1.0)
    _style_flicker_axes(ax)
    ax.legend(frameon=False, ncol=2)
    return _save_flicker_figure(fig, out_file)


def _save_flicker_spatial_coherence_plot(
    pair_table: pd.DataFrame,
    out_file: Path,
) -> str:
    x, x_label = _flicker_table_x_axis(
        pair_table,
        time_column="time_to",
        fallback_column="pair_index",
        fallback_label="frame pair",
    )
    fig, ax = plt.subplots(figsize=(11.0, 4.8), dpi=220)
    specs = [
        ("spatial_neighbor_changed_fraction_mean", "neighbor changed fraction", "#111827", "-"),
        ("spatial_same_transition_fraction_mean", "same-transition neighbor fraction", "#4b5563", "--"),
        ("spatial_isolated_change_rate", "isolated changed structures", "#991b1b", ":"),
        ("spatial_coherent_change_rate", "coherent changed structures", "#0f766e", "-."),
    ]
    for column, label, color, linestyle in specs:
        if column not in pair_table.columns:
            raise KeyError(f"Missing required flicker column for spatial plot: {column!r}.")
        y = pd.to_numeric(pair_table[column], errors="raise").to_numpy(dtype=np.float64)
        ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=2.0)
    ax.set_title("Spatial coherence of label changes")
    ax.set_xlabel(x_label)
    ax.set_ylabel("fraction")
    ax.set_ylim(0.0, 1.0)
    _style_flicker_axes(ax)
    ax.legend(frameon=False, ncol=2)
    return _save_flicker_figure(fig, out_file)


def _save_flicker_reciprocal_heatmap_plot(
    reciprocal_table: pd.DataFrame,
    *,
    cluster_ids: list[int],
    cluster_colors: dict[int, str],
    cluster_display_labels: list[str],
    out_file: Path,
) -> str:
    matrix = np.full((len(cluster_ids), len(cluster_ids)), np.nan, dtype=np.float64)
    cluster_to_pos = {int(cluster_id): pos for pos, cluster_id in enumerate(cluster_ids)}
    grouped = reciprocal_table.groupby(["cluster_a", "cluster_b"], dropna=False)["reciprocal_rate"].mean()
    for (cluster_a, cluster_b), value in grouped.items():
        left = int(cluster_a)
        right = int(cluster_b)
        if left not in cluster_to_pos or right not in cluster_to_pos:
            raise ValueError(
                "Reciprocal flicker table contains a cluster not present in cluster_ids: "
                f"cluster_a={left}, cluster_b={right}, cluster_ids={cluster_ids}."
            )
        i = cluster_to_pos[left]
        j = cluster_to_pos[right]
        matrix[i, j] = float(value)
        matrix[j, i] = float(value)

    finite_values = matrix[np.isfinite(matrix)]
    vmax = float(np.max(finite_values)) if finite_values.size > 0 and np.max(finite_values) > 0.0 else 1.0
    cmap = plt.cm.magma.copy()
    cmap.set_bad("#f3f4f6")
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=220)
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_title("Mean reciprocal flicker by cluster pair")
    ax.set_xticks(np.arange(len(cluster_ids)))
    ax.set_yticks(np.arange(len(cluster_ids)))
    ax.set_xticklabels(cluster_display_labels, rotation=45, ha="right")
    ax.set_yticklabels(cluster_display_labels)
    for pos, cluster_id in enumerate(cluster_ids):
        ax.get_xticklabels()[pos].set_color(cluster_colors[int(cluster_id)])
        ax.get_yticklabels()[pos].set_color(cluster_colors[int(cluster_id)])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if i == j or not np.isfinite(value) or value <= 0.0:
                continue
            text_color = "#f8fafc" if value < 0.55 * vmax else "#111827"
            ax.text(j, i, f"{100.0 * value:.1f}%", ha="center", va="center", fontsize=8, color=text_color)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("mean reciprocal rate")
    ax.set_xlabel("cluster")
    ax.set_ylabel("cluster")
    return _save_flicker_figure(fig, out_file)


def _save_flicker_dwell_summary_plot(
    dwell_summary_table: pd.DataFrame,
    *,
    cluster_ids: list[int],
    cluster_colors: dict[int, str],
    cluster_display_labels: list[str],
    out_file: Path,
) -> str:
    x = np.arange(len(cluster_ids), dtype=np.float64)
    rows_by_cluster = {
        int(row["cluster_id"]): row
        for _, row in dwell_summary_table.iterrows()
    }
    colors = [cluster_colors[int(cluster_id)] for cluster_id in cluster_ids]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), dpi=220, sharex=True)
    metrics = [
        ("median_dwell_frames", "median dwell", "frames"),
        ("mean_dwell_frames", "mean dwell", "frames"),
        ("p90_dwell_frames", "p90 dwell", "frames"),
        ("fraction_runs_le_2_frames", "runs <= 2 frames", "fraction"),
    ]
    for ax, (column, title, y_label) in zip(axes.ravel(), metrics, strict=True):
        if column not in dwell_summary_table.columns:
            raise KeyError(f"Missing required dwell summary column for plot: {column!r}.")
        values = np.asarray(
            [
                float(rows_by_cluster[int(cluster_id)][column])
                if int(cluster_id) in rows_by_cluster and pd.notna(rows_by_cluster[int(cluster_id)][column])
                else np.nan
                for cluster_id in cluster_ids
            ],
            dtype=np.float64,
        )
        ax.bar(x, values, color=colors, alpha=0.88, edgecolor="#111827", linewidth=0.35)
        ax.set_title(title)
        ax.set_ylabel(y_label)
        if y_label == "fraction":
            ax.set_ylim(0.0, 1.0)
        _style_flicker_axes(ax)
    for ax in axes[-1, :]:
        ax.set_xticks(x)
        ax.set_xticklabels(cluster_display_labels)
        for tick, cluster_id in zip(ax.get_xticklabels(), cluster_ids, strict=True):
            tick.set_color(cluster_colors[int(cluster_id)])
    fig.suptitle("Dwell-time summary by source cluster", y=1.02)
    return _save_flicker_figure(fig, out_file)


def _save_flicker_dwell_distribution_plot(
    dwell_hist_table: pd.DataFrame,
    *,
    cluster_ids: list[int],
    cluster_colors: dict[int, str],
    cluster_display_labels: list[str],
    out_file: Path,
) -> str:
    fig, ax = plt.subplots(figsize=(9.8, 5.2), dpi=220)
    for cluster_id, display_label in zip(cluster_ids, cluster_display_labels, strict=True):
        sub = dwell_hist_table[dwell_hist_table["cluster_id"].astype(int) == int(cluster_id)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("dwell_frames")
        lengths = pd.to_numeric(sub["dwell_frames"], errors="raise").to_numpy(dtype=np.float64)
        counts = pd.to_numeric(sub["run_count"], errors="raise").to_numpy(dtype=np.float64)
        if np.sum(counts) <= 0.0:
            continue
        survival = np.cumsum(counts[::-1])[::-1] / float(np.sum(counts))
        ax.step(
            lengths,
            survival,
            where="post",
            color=cluster_colors[int(cluster_id)],
            linewidth=2.0,
            label=display_label,
        )
    ax.set_title("Dwell-time survival by cluster")
    ax.set_xlabel("dwell length (frames)")
    ax.set_ylabel("P(run length >= dwell length)")
    ax.set_yscale("log")
    ax.set_ylim(1.0e-4, 1.05)
    _style_flicker_axes(ax)
    ax.legend(title="cluster", frameon=False, ncol=min(4, max(1, len(cluster_ids))))
    return _save_flicker_figure(fig, out_file)


def _save_flicker_recrossing_by_lag_plot(
    recrossing_table: pd.DataFrame,
    out_file: Path,
) -> str:
    x, x_label = _flicker_table_x_axis(
        recrossing_table,
        time_column="lag_time",
        fallback_column="lag_frames",
        fallback_label="lag (frames)",
    )
    order = np.argsort(x)
    x = x[order]
    persistent = pd.to_numeric(recrossing_table["persistent_fraction"], errors="raise").to_numpy(dtype=np.float64)[order]
    recross = pd.to_numeric(recrossing_table["recross_fraction"], errors="raise").to_numpy(dtype=np.float64)[order]
    other = pd.to_numeric(recrossing_table["other_fraction"], errors="raise").to_numpy(dtype=np.float64)[order]
    diffs = np.diff(np.unique(x))
    width = 0.70 * float(np.min(diffs)) if diffs.size > 0 and np.min(diffs) > 0.0 else 0.55
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=220)
    ax.bar(x, persistent, width=width, color="#4b5563", label="persistent")
    ax.bar(x, recross, width=width, bottom=persistent, color="#0f766e", label="recrossed")
    ax.bar(x, other, width=width, bottom=persistent + recross, color="#cbd5e1", label="other")
    ax.set_title("Post-transition outcomes by lag")
    ax.set_xlabel(x_label if x_label != "time" else "lag")
    ax.set_ylabel("fraction of changed transitions")
    ax.set_ylim(0.0, 1.0)
    _style_flicker_axes(ax)
    ax.legend(frameon=False, ncol=3)
    return _save_flicker_figure(fig, out_file)


def _save_flicker_recrossing_by_source_plot(
    recrossing_by_source_table: pd.DataFrame,
    *,
    cluster_ids: list[int],
    cluster_colors: dict[int, str],
    cluster_display_labels: list[str],
    out_file: Path,
) -> str:
    fig, ax = plt.subplots(figsize=(9.8, 5.0), dpi=220)
    for cluster_id, display_label in zip(cluster_ids, cluster_display_labels, strict=True):
        sub = recrossing_by_source_table[
            recrossing_by_source_table["source_cluster_id"].astype(int) == int(cluster_id)
        ].copy()
        if sub.empty:
            continue
        x, x_label = _flicker_table_x_axis(
            sub,
            time_column="lag_time",
            fallback_column="lag_frames",
            fallback_label="lag (frames)",
        )
        order = np.argsort(x)
        y = pd.to_numeric(sub["recross_fraction"], errors="raise").to_numpy(dtype=np.float64)[order]
        ax.plot(
            x[order],
            y,
            marker="o",
            markersize=4.0,
            linewidth=1.8,
            color=cluster_colors[int(cluster_id)],
            label=display_label,
        )
        ax.set_xlabel(x_label if x_label != "time" else "lag")
    ax.set_title("Recrossing fraction by source cluster")
    ax.set_ylabel("fraction recrossed")
    ax.set_ylim(0.0, 1.0)
    _style_flicker_axes(ax)
    ax.legend(title="source cluster", frameon=False, ncol=min(4, max(1, len(cluster_ids))))
    return _save_flicker_figure(fig, out_file)


def _save_flicker_cluster_margin_plot(
    margin_table: pd.DataFrame,
    *,
    cluster_ids: list[int],
    cluster_colors: dict[int, str],
    cluster_display_labels: list[str],
    out_file: Path,
) -> str:
    x = np.arange(len(cluster_ids), dtype=np.float64)
    width = 0.36
    changed_margin: list[float] = []
    stable_margin: list[float] = []
    changed_fraction: list[float] = []
    runner_up_target: list[float] = []
    for cluster_id in cluster_ids:
        sub = margin_table[margin_table["source_cluster_id"].astype(int) == int(cluster_id)]
        changed_margin.append(
            _weighted_metric_mean(
                sub,
                value_column="changed_margin_before_mean",
                weight_column="changed_margin_before_count",
            )
        )
        stable_margin.append(
            _weighted_metric_mean(
                sub,
                value_column="stable_margin_before_mean",
                weight_column="stable_margin_before_count",
            )
        )
        source_count = pd.to_numeric(sub["source_count"], errors="coerce").to_numpy(dtype=np.float64)
        changed_count = pd.to_numeric(sub["changed_count"], errors="coerce").to_numpy(dtype=np.float64)
        finite_counts = np.isfinite(source_count) & np.isfinite(changed_count) & (source_count >= 0.0) & (changed_count >= 0.0)
        changed_fraction.append(
            float(np.sum(changed_count[finite_counts]) / np.sum(source_count[finite_counts]))
            if np.any(finite_counts) and np.sum(source_count[finite_counts]) > 0.0
            else np.nan
        )
        runner_up_target.append(
            _weighted_metric_mean(
                sub,
                value_column="runner_up_is_target_before_rate",
                weight_column="changed_count",
            )
        )

    colors = [cluster_colors[int(cluster_id)] for cluster_id in cluster_ids]
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.0), dpi=220, sharex=True)
    axes[0].bar(x - width / 2.0, changed_margin, width=width, color=colors, alpha=0.88, label="changed before")
    axes[0].bar(
        x + width / 2.0,
        stable_margin,
        width=width,
        color=[_rgba_with_alpha(color, 0.32) for color in colors],
        edgecolor=colors,
        linewidth=1.0,
        label="stable before",
    )
    axes[0].set_title("Assignment margin before frame-pair transition")
    axes[0].set_ylabel("weighted mean margin")
    _style_flicker_axes(axes[0])
    axes[0].legend(frameon=False, ncol=2)

    axes[1].bar(x - width / 2.0, changed_fraction, width=width, color=colors, alpha=0.88, label="changed fraction")
    axes[1].bar(
        x + width / 2.0,
        runner_up_target,
        width=width,
        color=[_rgba_with_alpha(color, 0.34) for color in colors],
        edgecolor=colors,
        linewidth=1.0,
        hatch="//",
        label="runner-up is target",
    )
    axes[1].set_ylabel("fraction")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cluster_display_labels)
    for tick, cluster_id in zip(axes[1].get_xticklabels(), cluster_ids, strict=True):
        tick.set_color(cluster_colors[int(cluster_id)])
    _style_flicker_axes(axes[1])
    axes[1].legend(frameon=False, ncol=2)
    axes[1].set_xlabel("source cluster")
    return _save_flicker_figure(fig, out_file)


def _write_temporal_flicker_metric_plots(
    *,
    flicker_dir: Path,
    cluster_ids: list[int],
    cluster_color_map: dict[int, str] | None,
    cluster_display_map: dict[int, str] | None,
    pair_table: pd.DataFrame,
    reciprocal_table: pd.DataFrame,
    dwell_hist_table: pd.DataFrame,
    dwell_summary_table: pd.DataFrame,
    recrossing_table: pd.DataFrame,
    recrossing_by_source_table: pd.DataFrame,
    margin_table: pd.DataFrame | None,
) -> dict[str, str]:
    if pair_table.empty:
        raise ValueError("Cannot write flicker metric plots because frame_pair_flicker_metrics is empty.")
    if reciprocal_table.empty:
        raise ValueError("Cannot write reciprocal flicker heatmap because reciprocal_pair_flicker is empty.")
    if dwell_summary_table.empty:
        raise ValueError("Cannot write dwell flicker plots because dwell_time_summary is empty.")
    if dwell_hist_table.empty:
        raise ValueError("Cannot write dwell distribution plot because dwell_time_histogram is empty.")
    if recrossing_table.empty:
        raise ValueError("Cannot write recrossing plot because recrossing_metrics is empty.")
    if recrossing_by_source_table.empty:
        raise ValueError("Cannot write source-cluster recrossing plot because recrossing_by_source_cluster is empty.")

    ordered_cluster_ids = [int(cluster_id) for cluster_id in cluster_ids]
    cluster_colors = _cluster_colors_for_flicker(ordered_cluster_ids, cluster_color_map)
    cluster_display_labels = _cluster_display_labels_for_flicker(
        ordered_cluster_ids,
        cluster_display_map,
    )
    flicker_path = Path(flicker_dir)
    artifacts = {
        "churn_timeseries_png": _save_flicker_churn_timeseries_plot(
            pair_table,
            flicker_path / "flicker_churn_timeseries.png",
        ),
        "spatial_coherence_timeseries_png": _save_flicker_spatial_coherence_plot(
            pair_table,
            flicker_path / "flicker_spatial_coherence_timeseries.png",
        ),
        "reciprocal_pair_heatmap_png": _save_flicker_reciprocal_heatmap_plot(
            reciprocal_table,
            cluster_ids=ordered_cluster_ids,
            cluster_colors=cluster_colors,
            cluster_display_labels=cluster_display_labels,
            out_file=flicker_path / "flicker_reciprocal_pair_heatmap.png",
        ),
        "dwell_time_summary_png": _save_flicker_dwell_summary_plot(
            dwell_summary_table,
            cluster_ids=ordered_cluster_ids,
            cluster_colors=cluster_colors,
            cluster_display_labels=cluster_display_labels,
            out_file=flicker_path / "flicker_dwell_time_summary.png",
        ),
        "dwell_time_distribution_png": _save_flicker_dwell_distribution_plot(
            dwell_hist_table,
            cluster_ids=ordered_cluster_ids,
            cluster_colors=cluster_colors,
            cluster_display_labels=cluster_display_labels,
            out_file=flicker_path / "flicker_dwell_time_distribution.png",
        ),
        "recrossing_by_lag_png": _save_flicker_recrossing_by_lag_plot(
            recrossing_table,
            flicker_path / "flicker_recrossing_by_lag.png",
        ),
        "recrossing_by_source_cluster_png": _save_flicker_recrossing_by_source_plot(
            recrossing_by_source_table,
            cluster_ids=ordered_cluster_ids,
            cluster_colors=cluster_colors,
            cluster_display_labels=cluster_display_labels,
            out_file=flicker_path / "flicker_recrossing_by_source_cluster.png",
        ),
    }
    if margin_table is not None:
        if margin_table.empty:
            raise ValueError("Cannot write cluster margin flicker plot because cluster_margin_flicker is empty.")
        artifacts["cluster_margin_flicker_png"] = _save_flicker_cluster_margin_plot(
            margin_table,
            cluster_ids=ordered_cluster_ids,
            cluster_colors=cluster_colors,
            cluster_display_labels=cluster_display_labels,
            out_file=flicker_path / "flicker_cluster_margin_summary.png",
        )
    return artifacts


def _compute_and_write_temporal_flicker_metrics(
    *,
    frames: list[FrameSlice],
    frame_source: str,
    coords: np.ndarray,
    labels: np.ndarray,
    instance_ids: np.ndarray | None,
    cluster_ids: list[int],
    assignment_margins: dict[str, Any] | None,
    out_dir: Path,
    cluster_color_map: dict[int, str] | None,
    cluster_display_map: dict[int, str] | None,
    neighbor_k: int,
    isolated_neighbor_change_fraction: float,
    coherent_same_transition_fraction: float,
    recrossing_lags: list[int],
) -> dict[str, Any]:
    flicker_dir = Path(out_dir) / "flicker"
    flicker_dir.mkdir(parents=True, exist_ok=True)
    if len(frames) < 2:
        return {
            "enabled": False,
            "reason": "requires at least two ordered frames",
            "root_dir": str(flicker_dir),
        }
    if instance_ids is None:
        return {
            "enabled": False,
            "reason": "requires instance_ids to track local structures across frames",
            "root_dir": str(flicker_dir),
        }

    labels_arr = np.asarray(labels, dtype=int).reshape(-1)
    margin_arrays = _resolve_assignment_margin_arrays(
        assignment_margins,
        expected_length=int(labels_arr.shape[0]),
    )
    records = _build_sorted_temporal_metric_records(
        frames,
        coords=coords,
        labels=labels_arr,
        instance_ids=np.asarray(instance_ids, dtype=np.int64),
    )
    label_to_pos = _build_label_to_cluster_pos(cluster_ids)
    cluster_count = len(cluster_ids)

    pair_rows: list[dict[str, Any]] = []
    reciprocal_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    aggregate_counts = np.zeros((cluster_count, cluster_count), dtype=np.int64)

    for pair_index, (record_a, record_b) in enumerate(zip(records[:-1], records[1:], strict=True)):
        frame_a = record_a["frame"]
        frame_b = record_b["frame"]
        matched_a, matched_b = _match_sorted_temporal_records(record_a, record_b)
        labels_a_all = np.asarray(record_a["labels"], dtype=int)[matched_a]
        labels_b_all = np.asarray(record_b["labels"], dtype=int)[matched_b]
        counts, valid = _transition_count_matrix(
            labels_a_all,
            labels_b_all,
            label_to_pos=label_to_pos,
            cluster_count=cluster_count,
        )
        aggregate_counts += counts
        matched_count = int(matched_a.size)
        valid_count = int(np.count_nonzero(valid))
        invalid_count = matched_count - valid_count
        labels_a = labels_a_all[valid]
        labels_b = labels_b_all[valid]
        global_a = np.asarray(record_a["indices"], dtype=int)[matched_a][valid]
        global_b = np.asarray(record_b["indices"], dtype=int)[matched_b][valid]
        coords_a = np.asarray(record_a["coords"], dtype=np.float32)[matched_a][valid]
        total = int(np.sum(counts))
        diagonal = int(np.trace(counts))
        changed_count = int(total - diagonal)
        churn_rate = _safe_fraction(changed_count, total)
        row_counts = counts.sum(axis=1)
        col_counts = counts.sum(axis=0)
        net_population_tv = (
            0.5 * float(np.sum(np.abs(col_counts - row_counts))) / float(total)
            if total > 0
            else np.nan
        )
        excess_churn_rate = (
            float(churn_rate) - float(net_population_tv)
            if np.isfinite(churn_rate) and np.isfinite(net_population_tv)
            else np.nan
        )
        reciprocal_total = 0
        top_reciprocal_count = -1
        top_reciprocal_pair: tuple[int | None, int | None] = (None, None)
        for left_pos in range(cluster_count):
            for right_pos in range(left_pos + 1, cluster_count):
                reciprocal_count = int(
                    2 * min(counts[left_pos, right_pos], counts[right_pos, left_pos])
                )
                reciprocal_total += reciprocal_count
                reciprocal_rate = _safe_fraction(reciprocal_count, total)
                reciprocal_rows.append(
                    {
                        "pair_index": int(pair_index),
                        "frame_from": str(frame_a.source_name),
                        "frame_to": str(frame_b.source_name),
                        "time_from": np.nan if frame_a.time_value is None else float(frame_a.time_value),
                        "time_to": np.nan if frame_b.time_value is None else float(frame_b.time_value),
                        "cluster_a": int(cluster_ids[left_pos]),
                        "cluster_b": int(cluster_ids[right_pos]),
                        "a_to_b_count": int(counts[left_pos, right_pos]),
                        "b_to_a_count": int(counts[right_pos, left_pos]),
                        "reciprocal_count": reciprocal_count,
                        "reciprocal_rate": reciprocal_rate,
                    }
                )
                if reciprocal_count > top_reciprocal_count:
                    top_reciprocal_count = reciprocal_count
                    top_reciprocal_pair = (int(cluster_ids[left_pos]), int(cluster_ids[right_pos]))

        changed_mask = labels_a != labels_b
        pair_row: dict[str, Any] = {
            "pair_index": int(pair_index),
            "frame_from": str(frame_a.source_name),
            "frame_to": str(frame_b.source_name),
            "frame_from_label": str(frame_a.label),
            "frame_to_label": str(frame_b.label),
            "time_from": np.nan if frame_a.time_value is None else float(frame_a.time_value),
            "time_to": np.nan if frame_b.time_value is None else float(frame_b.time_value),
            "delta_time": _frame_time_delta(frame_a, frame_b),
            "time_unit": "" if frame_a.time_unit is None else str(frame_a.time_unit),
            "matched_samples": matched_count,
            "valid_matched_samples": valid_count,
            "invalid_label_matched_samples": invalid_count,
            "coverage_fraction": _safe_fraction(matched_count, int(np.asarray(record_a["ids"]).size)),
            "label_churn_count": changed_count,
            "label_churn_rate": churn_rate,
            "net_population_tv": net_population_tv,
            "excess_churn_rate": excess_churn_rate,
            "excess_churn_fraction_of_churn": _safe_fraction(excess_churn_rate, churn_rate),
            "reciprocal_pair_flicker_count": int(reciprocal_total),
            "reciprocal_pair_flicker_rate": _safe_fraction(reciprocal_total, total),
            "top_reciprocal_cluster_a": np.nan if top_reciprocal_pair[0] is None else int(top_reciprocal_pair[0]),
            "top_reciprocal_cluster_b": np.nan if top_reciprocal_pair[1] is None else int(top_reciprocal_pair[1]),
            "top_reciprocal_count": int(max(top_reciprocal_count, 0)),
            "top_reciprocal_rate": _safe_fraction(max(top_reciprocal_count, 0), total),
        }

        spatial_metrics = _spatial_coherence_metrics_for_pair(
            coords_a=coords_a,
            labels_a=labels_a,
            labels_b=labels_b,
            neighbor_k=int(neighbor_k),
            isolated_neighbor_change_fraction=float(isolated_neighbor_change_fraction),
            coherent_same_transition_fraction=float(coherent_same_transition_fraction),
        )
        pair_row.update(spatial_metrics)
        spatial_rows.append(
            {
                key: pair_row[key]
                for key in (
                    "pair_index",
                    "frame_from",
                    "frame_to",
                    "time_from",
                    "time_to",
                    "delta_time",
                    "time_unit",
                    "valid_matched_samples",
                    "label_churn_count",
                    "label_churn_rate",
                    "spatial_neighbor_k",
                    "spatial_changed_samples",
                    "spatial_neighbor_changed_fraction_mean",
                    "spatial_same_transition_fraction_mean",
                    "spatial_source_purity_before_mean",
                    "spatial_target_purity_after_mean",
                    "spatial_isolated_change_rate",
                    "spatial_coherent_change_rate",
                )
            }
        )

        if margin_arrays is not None:
            margins = np.asarray(margin_arrays["margin"], dtype=np.float32)
            runner_up = np.asarray(margin_arrays["runner_up_cluster"], dtype=np.int64)
            pair_row.update(
                _margin_stats_for_mask(
                    margins[global_a],
                    changed_mask,
                    prefix="changed_margin_before",
                )
            )
            pair_row.update(
                _margin_stats_for_mask(
                    margins[global_b],
                    changed_mask,
                    prefix="changed_margin_after",
                )
            )
            pair_row.update(
                _margin_stats_for_mask(
                    margins[global_a],
                    ~changed_mask,
                    prefix="stable_margin_before",
                )
            )
            pair_row.update(
                _margin_stats_for_mask(
                    margins[global_b],
                    ~changed_mask,
                    prefix="stable_margin_after",
                )
            )
            pair_row["changed_runner_up_is_target_before_rate"] = _safe_fraction(
                int(np.count_nonzero(runner_up[global_a][changed_mask] == labels_b[changed_mask])),
                int(np.count_nonzero(changed_mask)),
            )
            pair_row["changed_runner_up_is_source_after_rate"] = _safe_fraction(
                int(np.count_nonzero(runner_up[global_b][changed_mask] == labels_a[changed_mask])),
                int(np.count_nonzero(changed_mask)),
            )
            for cluster_id in cluster_ids:
                source_mask = labels_a == int(cluster_id)
                source_changed = source_mask & changed_mask
                source_stable = source_mask & ~changed_mask
                margin_row: dict[str, Any] = {
                    "pair_index": int(pair_index),
                    "frame_from": str(frame_a.source_name),
                    "frame_to": str(frame_b.source_name),
                    "time_from": pair_row["time_from"],
                    "time_to": pair_row["time_to"],
                    "source_cluster_id": int(cluster_id),
                    "source_count": int(np.count_nonzero(source_mask)),
                    "changed_count": int(np.count_nonzero(source_changed)),
                    "stable_count": int(np.count_nonzero(source_stable)),
                    "changed_fraction": _safe_fraction(
                        int(np.count_nonzero(source_changed)),
                        int(np.count_nonzero(source_mask)),
                    ),
                    "runner_up_is_target_before_rate": _safe_fraction(
                        int(np.count_nonzero(runner_up[global_a][source_changed] == labels_b[source_changed])),
                        int(np.count_nonzero(source_changed)),
                    ),
                }
                margin_row.update(
                    _margin_stats_for_mask(
                        margins[global_a],
                        source_changed,
                        prefix="changed_margin_before",
                    )
                )
                margin_row.update(
                    _margin_stats_for_mask(
                        margins[global_a],
                        source_stable,
                        prefix="stable_margin_before",
                    )
                )
                margin_rows.append(margin_row)

        pair_rows.append(pair_row)

    frame_pair_csv = flicker_dir / "frame_pair_flicker_metrics.csv"
    reciprocal_csv = flicker_dir / "reciprocal_pair_flicker.csv"
    spatial_csv = flicker_dir / "spatial_coherence.csv"
    margin_csv = flicker_dir / "cluster_margin_flicker.csv"
    pair_table = pd.DataFrame.from_records(pair_rows)
    reciprocal_table = pd.DataFrame.from_records(reciprocal_rows)
    spatial_table = pd.DataFrame.from_records(spatial_rows)
    pair_table.to_csv(frame_pair_csv, index=False)
    reciprocal_table.to_csv(reciprocal_csv, index=False)
    spatial_table.to_csv(spatial_csv, index=False)
    margin_table = None
    if margin_arrays is not None:
        margin_table = pd.DataFrame.from_records(margin_rows)
        margin_table.to_csv(margin_csv, index=False)

    common_ids, label_matrix = _common_label_matrix(records)
    frame_spacing, time_unit = _median_frame_spacing(frames)
    dwell_hist_csv, dwell_summary_csv, dwell_summary = _write_dwell_metrics(
        label_matrix=label_matrix,
        cluster_ids=cluster_ids,
        out_dir=flicker_dir,
        frame_spacing=frame_spacing,
        time_unit=time_unit,
    )
    recrossing_csv, recrossing_by_source_csv, recrossing_summary = _write_recrossing_metrics(
        label_matrix=label_matrix,
        cluster_ids=cluster_ids,
        out_dir=flicker_dir,
        lags=recrossing_lags,
        frame_spacing=frame_spacing,
        time_unit=time_unit,
    )

    dwell_hist_table = pd.read_csv(dwell_hist_csv)
    dwell_summary_table = pd.read_csv(dwell_summary_csv)
    recrossing_table = pd.read_csv(recrossing_csv)
    recrossing_by_source_table = pd.read_csv(recrossing_by_source_csv)
    plot_artifacts = _write_temporal_flicker_metric_plots(
        flicker_dir=flicker_dir,
        cluster_ids=cluster_ids,
        cluster_color_map=cluster_color_map,
        cluster_display_map=cluster_display_map,
        pair_table=pair_table,
        reciprocal_table=reciprocal_table,
        dwell_hist_table=dwell_hist_table,
        dwell_summary_table=dwell_summary_table,
        recrossing_table=recrossing_table,
        recrossing_by_source_table=recrossing_by_source_table,
        margin_table=margin_table,
    )

    summary: dict[str, Any] = {
        "enabled": True,
        "root_dir": str(flicker_dir),
        "frame_count": int(len(frames)),
        "frame_source": str(frame_source),
        "tracked_instance_count_all_frames": int(common_ids.size),
        "cluster_ids": [int(v) for v in cluster_ids],
        "neighbor_k": int(neighbor_k),
        "spatial_isolated_neighbor_change_fraction": float(isolated_neighbor_change_fraction),
        "spatial_coherent_same_transition_fraction": float(coherent_same_transition_fraction),
        "recrossing_lags": [int(v) for v in recrossing_lags],
        "artifacts": {
            "frame_pair_metrics_csv": str(frame_pair_csv),
            "reciprocal_pair_flicker_csv": str(reciprocal_csv),
            "spatial_coherence_csv": str(spatial_csv),
            "dwell_time_histogram_csv": str(dwell_hist_csv),
            "dwell_time_summary_csv": str(dwell_summary_csv),
            "recrossing_metrics_csv": str(recrossing_csv),
            "recrossing_by_source_cluster_csv": str(recrossing_by_source_csv),
            **plot_artifacts,
        },
        "label_churn_rate": _finite_metric_summary(pair_table["label_churn_rate"].to_numpy(float)),
        "net_population_tv": _finite_metric_summary(pair_table["net_population_tv"].to_numpy(float)),
        "excess_churn_rate": _finite_metric_summary(pair_table["excess_churn_rate"].to_numpy(float)),
        "reciprocal_pair_flicker_rate": _finite_metric_summary(pair_table["reciprocal_pair_flicker_rate"].to_numpy(float)),
        "spatial_isolated_change_rate": _finite_metric_summary(pair_table["spatial_isolated_change_rate"].to_numpy(float)),
        "spatial_coherent_change_rate": _finite_metric_summary(pair_table["spatial_coherent_change_rate"].to_numpy(float)),
        "dwell_time_distribution": dwell_summary,
        "bounce_recrossing": recrossing_summary,
        "cluster_margin_flicker": {
            "available": margin_arrays is not None,
        },
    }
    if margin_arrays is not None:
        summary["artifacts"]["cluster_margin_flicker_csv"] = str(margin_csv)
        summary["cluster_margin_flicker"].update(
            {
                "score_name": str(margin_arrays["score_name"]),
                "higher_is_better": bool(margin_arrays["higher_is_better"]),
                "changed_margin_before": _finite_metric_summary(
                    pair_table["changed_margin_before_mean"].to_numpy(float)
                ),
                "stable_margin_before": _finite_metric_summary(
                    pair_table["stable_margin_before_mean"].to_numpy(float)
                ),
                "changed_runner_up_is_target_before_rate": _finite_metric_summary(
                    pair_table["changed_runner_up_is_target_before_rate"].to_numpy(float)
                ),
            }
        )

    summary_json = flicker_dir / "flicker_summary.json"
    write_json(summary_json, summary)
    summary["summary_json"] = str(summary_json)
    return summary


def _write_summary_markdown(
    out_file: Path,
    *,
    selected_k: int,
    frames: list[FrameSlice],
    cluster_groups: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Real-MD qualitative analysis",
        "",
        f"- Selected clustering k: `{int(selected_k)}`",
        f"- Frames analysed: `{len(frames)}`",
        f"- Frame order: {', '.join(frame.label for frame in frames)}",
    ]
    if cluster_groups:
        cluster_group_text = ", ".join(
            f"{group['name']}={group['cluster_ids']}" for group in cluster_groups
        )
        lines.append(
            f"- Cluster groups: {cluster_group_text}"
        )
    if "cluster_proportions" in summary:
        paper_plots = summary["cluster_proportions"]["plots"].get("paper", {})
        lines.extend(
            [
                "",
                "## Cluster proportions",
                f"- CSV: `{Path(summary['cluster_proportions']['table_csv']).name}`",
                f"- Stacked area: `{Path(summary['cluster_proportions']['plots']['stacked_area']).name}`",
            ]
        )
        if paper_plots:
            lines.append(
                f"- Paper SVGs: {', '.join(f'`{Path(path).name}`' for path in paper_plots.values())}"
            )
    if "representatives" in summary:
        lines.extend(
            [
                "",
                "## Representatives",
                f"- Root: `{Path(summary['representatives']['root_dir']).name}`",
                f"- Shared-style figure: `{Path(summary['representatives']['primary_figure']).name}`",
                f"- Edge-connected figure: `{Path(summary['representatives']['edge_connected_figure']).name}`",
            ]
        )
        structure_analysis = summary["representatives"]["shared_style"].get("structure_analysis")
        if isinstance(structure_analysis, dict):
            lines.append(
                f"- Structure analysis JSON: `{Path(structure_analysis['json_file']).name}`"
            )
            lines.append(
                f"- Structure analysis CSV: `{Path(structure_analysis['csv_file']).name}`"
            )
    if "spatial" in summary:
        lines.extend(
            [
                "",
                "## Spatial views",
                f"- Filtered cluster views: `{sum(len(frame.get('filtered_views', [])) for frame in summary['spatial'].get('frames', []))}`",
                f"- Zoom views: `{len(summary['spatial'].get('zooms', []))}`",
            ]
        )
    if "latent_projection" in summary:
        projection_info = summary["latent_projection"]["projection_info"]
        lines.extend(
            [
                "",
                "## Latent projection",
                f"- Method: `{projection_info['method']}`",
                f"- CSV: `{Path(summary['latent_projection']['projection_csv']).name}`",
            ]
        )
        if "backend" in projection_info:
            lines.append(
                f"- Backend: `{projection_info['backend']}` on `{projection_info['device']}`"
            )
    if "descriptor_analysis" in summary:
        lines.extend(
            [
                "",
                "## Descriptor analysis",
                f"- Sample table: `{Path(summary['descriptor_analysis']['sample_table_csv']).name}`",
                f"- Scalar violin plot: `{Path(summary['descriptor_analysis']['scalar_violin_plot']).name}`",
            ]
        )
        cna_vis = summary["descriptor_analysis"].get("cna_visualization")
        if isinstance(cna_vis, dict):
            lines.append(
                f"- CNA time series: `{Path(cna_vis['time_series_plot']).name}`"
            )
    if "transitions" in summary:
        lines.extend(
            [
                "",
                "## Transitions",
                f"- Match mode: `{summary['transitions'].get('match_mode', 'nearest_neighbor')}`",
                f"- Match tolerance: `{summary['transitions']['match_tolerance']:.4f}`",
                f"- Aggregate flow diagram: `{Path(summary['transitions']['aggregate_flow']).name}`",
            ]
        )
    if "flicker_metrics" in summary:
        flicker_summary = summary["flicker_metrics"]
        lines.extend(
            [
                "",
                "## Flicker metrics",
                f"- Enabled: `{bool(flicker_summary.get('enabled', False))}`",
            ]
        )
        if flicker_summary.get("enabled", False):
            artifacts = flicker_summary.get("artifacts", {})
            frame_pair_csv = artifacts.get("frame_pair_metrics_csv")
            dwell_csv = artifacts.get("dwell_time_summary_csv")
            recrossing_csv = artifacts.get("recrossing_metrics_csv")
            spatial_csv = artifacts.get("spatial_coherence_csv")
            if frame_pair_csv is not None:
                lines.append(f"- Frame-pair metrics: `{Path(frame_pair_csv).name}`")
            if dwell_csv is not None:
                lines.append(f"- Dwell summary: `{Path(dwell_csv).name}`")
            if recrossing_csv is not None:
                lines.append(f"- Recrossing metrics: `{Path(recrossing_csv).name}`")
            if spatial_csv is not None:
                lines.append(f"- Spatial coherence: `{Path(spatial_csv).name}`")
            margin_csv = artifacts.get("cluster_margin_flicker_csv")
            if margin_csv is not None:
                lines.append(f"- Cluster-margin flicker: `{Path(margin_csv).name}`")
            plot_keys = (
                "churn_timeseries_png",
                "spatial_coherence_timeseries_png",
                "reciprocal_pair_heatmap_png",
                "dwell_time_summary_png",
                "dwell_time_distribution_png",
                "recrossing_by_lag_png",
                "recrossing_by_source_cluster_png",
                "cluster_margin_flicker_png",
            )
            plot_names = [
                f"`{Path(path).name}`"
                for key in plot_keys
                if (path := artifacts.get(key)) is not None
            ]
            if plot_names:
                lines.append(f"- Plots: {', '.join(plot_names)}")
        elif "reason" in flicker_summary:
            lines.append(f"- Reason: {flicker_summary['reason']}")
    if "temporal" in summary:
        lines.extend(
            [
                "",
                "## Temporal animations",
                f"- Frames rendered: `{summary['temporal']['frame_count']}`",
            ]
        )
        if "md_space_animation" in summary["temporal"]:
            lines.append(
                f"- MD-space diagonal-cut animation: `{Path(summary['temporal']['md_space_animation']['out_file']).name}`"
            )
        if "transition_pair_flow_animation" in summary["temporal"]:
            lines.append(
                f"- Transition-flow animation: `{Path(summary['temporal']['transition_pair_flow_animation']['out_file']).name}`"
            )
        if "umap_animation" in summary["temporal"]:
            lines.append(
                f"- UMAP cluster animation: `{Path(summary['temporal']['umap_animation']['out_file']).name}`"
            )
        if "umap_trajectory_animation" in summary["temporal"]:
            lines.append(
                f"- UMAP trajectory animation: `{Path(summary['temporal']['umap_trajectory_animation']['out_file']).name}`"
    )
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_dynamic_motif_summary(
    summary_markdown_path: Path,
    *,
    dynamic_summary_path: Path,
    dynamic_metrics: dict[str, Any],
    out_dir: Path,
) -> None:
    if not summary_markdown_path.exists():
        raise FileNotFoundError(
            "Cannot append dynamic motif summary because the real-MD summary markdown does not exist: "
            f"{summary_markdown_path}."
        )
    if not dynamic_summary_path.exists():
        raise FileNotFoundError(
            "Cannot append dynamic motif summary because the dynamic summary markdown does not exist: "
            f"{dynamic_summary_path}."
        )
    marker = "\n## Dynamic motifs\n"
    original = summary_markdown_path.read_text(encoding="utf-8")
    if marker in original:
        original = original.split(marker, 1)[0].rstrip() + "\n"
    summary_rel = _relative_path_for_markdown(
        dynamic_summary_path,
        base_dir=summary_markdown_path.parent,
    )
    lines = [
        original.rstrip(),
        "",
        "## Dynamic motifs",
        f"- Summary: `{summary_rel}`",
    ]
    artifacts = dynamic_metrics.get("artifacts", {})
    for key in (
        "frame_motif_proportions_csv",
        "transition_matrix_png",
        "dwell_histograms_png",
        "recurrence_heatmap_png",
        "motif_umap_png",
        "neighbor_influence_heatmap_png",
    ):
        artifact_rel = artifacts.get(key)
        if artifact_rel is None:
            continue
        artifact_path = (out_dir / artifact_rel).resolve()
        display_rel = _relative_path_for_markdown(
            artifact_path,
            base_dir=summary_markdown_path.parent,
        )
        lines.append(f"- {key}: `{display_rel}`")
    warnings_list = dynamic_metrics.get("warnings", [])
    if warnings_list:
        lines.append("- Warnings:")
        lines.extend([f"  - {warning}" for warning in warnings_list])
    summary_markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_real_md_qualitative_analysis(
    *,
    out_dir: Path,
    model_cfg: Any,
    analysis_cfg: Any,
    dataset: Any,
    latents: np.ndarray,
    coords: np.ndarray,
    instance_ids: np.ndarray | None,
    cluster_labels_by_k: dict[int, np.ndarray],
    cluster_methods_by_k: dict[int, str],
    cluster_color_map: dict[int, str] | None,
    frame_groups: list[tuple[str, np.ndarray]],
    frame_output_names: dict[str, str],
    requested_frame_order: list[str] | None,
    temporal_all_frame_groups: list[tuple[str, np.ndarray]] | None = None,
    temporal_all_frame_output_names: dict[str, str] | None = None,
    temporal_all_frame_order: list[str] | None = None,
    temporal_md_animation_frame_groups: list[tuple[str, np.ndarray]] | None = None,
    temporal_md_animation_frame_output_names: dict[str, str] | None = None,
    temporal_md_animation_order: list[str] | None = None,
    temporal_md_animation_coords: np.ndarray | None = None,
    temporal_md_animation_cluster_labels_by_k: dict[int, np.ndarray] | None = None,
    temporal_md_animation_frame_source: str | None = None,
    temporal_md_animation_spatial_bounds: np.ndarray | None = None,
    temporal_projection_fit_indices: np.ndarray | None = None,
    point_scale: float,
    random_state: int,
    representative_render_cache: dict[str, Any] | None = None,
    representative_selection_features: np.ndarray | None = None,
    representative_selection_info: dict[str, Any] | None = None,
    cluster_assignment_margins_by_k: dict[int, dict[str, Any]] | None = None,
    selected_k_override: int | None = None,
    output_root_dir: Path | None = None,
) -> dict[str, Any]:
    data_kind = normalize_data_kind(getattr(model_cfg.data, "kind", None))

    clustering_cfg = getattr(analysis_cfg, "clustering", None)
    tsne_cfg = getattr(analysis_cfg, "tsne", None)
    figure_set_cfg = getattr(analysis_cfg, "figure_set", None)
    figure_representatives_cfg = getattr(figure_set_cfg, "representatives", None)
    figure_md_cfg = getattr(figure_set_cfg, "md", None)
    real_md_cfg = getattr(analysis_cfg, "real_md", None)
    time_series_cfg = getattr(real_md_cfg, "time_series", None)
    profile_cfg = getattr(real_md_cfg, "profiles", None)
    descriptor_cfg = getattr(real_md_cfg, "descriptors", None)
    projection_cfg = getattr(real_md_cfg, "projection", None)
    temporal_cfg = getattr(real_md_cfg, "temporal", None)
    transition_cfg = getattr(real_md_cfg, "transitions", None)

    selected_k_raw = selected_k_override
    if selected_k_raw is None:
        selected_k_raw = getattr(real_md_cfg, "selected_k", None)
    if selected_k_raw is None:
        if not cluster_labels_by_k:
            raise ValueError("cluster_labels_by_k is empty.")
        selected_k = min(int(k) for k in cluster_labels_by_k.keys())
    else:
        selected_k = int(selected_k_raw)
    if int(selected_k) not in cluster_labels_by_k:
        raise KeyError(
            "Requested real_md.selected_k is not available. "
            f"Requested k={selected_k}, available={sorted(cluster_labels_by_k.keys())}."
        )
    labels = np.asarray(cluster_labels_by_k[int(selected_k)], dtype=int)
    if instance_ids is not None:
        instance_ids_arr = np.asarray(instance_ids).reshape(-1)
        if instance_ids_arr.size == 0:
            print(
                "[analysis][real_md] instance_ids are unavailable in the inference cache; "
                "falling back to coordinate-based matching where supported. "
                "Identity-tracked temporal outputs remain unavailable until inference is "
                "re-collected with non-empty instance_ids."
            )
            instance_ids_arr = None
        elif instance_ids_arr.shape[0] != len(latents):
            raise ValueError(
                "instance_ids and latents length mismatch: "
                f"instance_ids={instance_ids_arr.shape[0]}, latents={len(latents)}."
            )
    else:
        instance_ids_arr = None
    out_root = real_md_outputs_root(out_dir) if output_root_dir is None else Path(output_root_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    frames = _build_frame_slices(
        frame_groups,
        frame_output_names,
        requested_order=requested_frame_order,
    )
    temporal_frames = (
        _build_frame_slices(
            temporal_all_frame_groups,
            temporal_all_frame_output_names,
            requested_order=temporal_all_frame_order,
        )
        if temporal_all_frame_groups is not None and temporal_all_frame_output_names is not None
        else None
    )
    temporal_md_animation_frames = (
        _build_frame_slices(
            temporal_md_animation_frame_groups,
            temporal_md_animation_frame_output_names,
            requested_order=temporal_md_animation_order,
        )
        if temporal_md_animation_frame_groups is not None
        and temporal_md_animation_frame_output_names is not None
        else None
    )
    labels_color_map = cluster_color_map or _build_cluster_color_map(labels)
    frame_index_lookup, frame_name_lookup, frame_time_lookup = _build_frame_lookup_for_samples(
        frames,
        num_samples=len(latents),
    )
    missing_count = int(np.sum(frame_index_lookup < 0))
    if missing_count > 0 and temporal_frames is None:
        raise RuntimeError(
            "Frame grouping did not cover all collected samples for real-MD qualitative analysis. "
            f"Missing assignments={missing_count}, total_samples={len(latents)}."
        )
    cluster_groups = _build_cluster_groups(real_md_cfg, frames=frames, labels=labels)

    summary: dict[str, Any] = {
        "root_dir": str(out_root),
        "selected_k": int(selected_k),
        "cluster_method": str(cluster_methods_by_k.get(int(selected_k), "unknown")),
        "frames": [
            {
                "frame_index": int(frame.order_index),
                "source_name": str(frame.source_name),
                "output_name": str(frame.output_name),
                "label": str(frame.label),
                "num_samples": int(frame.indices.size),
                "time_value": None if frame.time_value is None else float(frame.time_value),
                "time_unit": None if frame.time_unit is None else str(frame.time_unit),
            }
            for frame in frames
        ],
        "analysis_frame_selection": {
            "assigned_sample_count": int(len(latents) - missing_count),
            "unassigned_sample_count": int(missing_count),
            "uses_subset_of_collected_samples": bool(missing_count > 0),
        },
        "cluster_groups": cluster_groups,
    }

    proportions_dir = out_root / "time_series"
    proportion_frames = temporal_frames if temporal_frames is not None else frames
    proportions_table, cluster_ids, counts_matrix = _cluster_counts_by_frame(
        proportion_frames,
        labels,
    )
    cluster_display_map = {
        int(cluster_id): f"C{pos + 1}"
        for pos, cluster_id in enumerate(cluster_ids)
    }
    cluster_display_labels = [cluster_display_map[int(cluster_id)] for cluster_id in cluster_ids]
    proportions_csv = proportions_dir / "frame_cluster_proportions.csv"
    proportions_dir.mkdir(parents=True, exist_ok=True)
    proportions_table.to_csv(proportions_csv, index=False)
    summary["cluster_proportions"] = {
        "frame_source": (
            "temporal_inference_frames" if temporal_frames is not None else "analysis_frames"
        ),
        "num_frames": int(len(proportion_frames)),
        "table_csv": str(proportions_csv),
        "plots": save_cluster_proportion_plots(
            [frame.label for frame in proportion_frames],
            counts_matrix,
            cluster_ids,
            proportions_dir,
            cluster_color_map=labels_color_map,
            cluster_display_labels=cluster_display_labels,
            save_paper_svg=_cfg_bool(time_series_cfg, "paper_enabled", True),
            stack_alpha=_cfg_float(time_series_cfg, "alpha", 0.78),
            bar_alpha=_cfg_float(time_series_cfg, "bar_alpha", 0.82),
            paper_alpha=_cfg_float(time_series_cfg, "paper_alpha", 0.72),
        ),
    }

    flicker_cfg = getattr(temporal_cfg, "flicker", None)
    flicker_enabled_default = bool(len(proportion_frames) >= 2)
    if _cfg_bool(flicker_cfg, "enabled", flicker_enabled_default):
        recrossing_lags_raw = _to_plain(
            getattr(flicker_cfg, "recrossing_lags", [1, 2, 5, 10])
        )
        if isinstance(recrossing_lags_raw, str):
            recrossing_lags = [
                int(token.strip())
                for token in recrossing_lags_raw.split(",")
                if token.strip()
            ]
        elif isinstance(recrossing_lags_raw, (int, float)):
            recrossing_lags = [int(recrossing_lags_raw)]
        else:
            recrossing_lags = [int(v) for v in list(recrossing_lags_raw)]
        if not recrossing_lags:
            raise ValueError("real_md.temporal.flicker.recrossing_lags must not be empty.")
        selected_assignment_margins = None
        if cluster_assignment_margins_by_k is not None:
            selected_assignment_margins = cluster_assignment_margins_by_k.get(int(selected_k))
            if selected_assignment_margins is None:
                selected_assignment_margins = cluster_assignment_margins_by_k.get(str(int(selected_k)))
        summary["flicker_metrics"] = _compute_and_write_temporal_flicker_metrics(
            frames=proportion_frames,
            frame_source=(
                "temporal_inference_frames" if temporal_frames is not None else "analysis_frames"
            ),
            coords=np.asarray(coords, dtype=np.float32),
            labels=labels,
            instance_ids=instance_ids_arr,
            cluster_ids=cluster_ids,
            assignment_margins=selected_assignment_margins,
            out_dir=out_root,
            cluster_color_map=labels_color_map,
            cluster_display_map=cluster_display_map,
            neighbor_k=_cfg_int(flicker_cfg, "neighbor_k", 12),
            isolated_neighbor_change_fraction=_cfg_float(
                flicker_cfg,
                "isolated_neighbor_change_fraction",
                0.25,
            ),
            coherent_same_transition_fraction=_cfg_float(
                flicker_cfg,
                "coherent_same_transition_fraction",
                0.50,
            ),
            recrossing_lags=recrossing_lags,
        )

    if _cfg_bool(profile_cfg, "enabled", True):
        representatives_dir = out_root / "representatives"
        representatives_dir.mkdir(parents=True, exist_ok=True)
        representative_base_path = representatives_dir / f"04_cluster_representatives_k{int(selected_k)}.png"
        shared_style_summary = _save_cluster_representatives_figure(
            dataset,
            np.asarray(latents, dtype=np.float32),
            labels,
            labels_color_map,
            representative_base_path,
            point_scale=float(point_scale),
            target_points=_cfg_int(
                profile_cfg,
                "target_points",
                max(
                    32,
                    int(getattr(model_cfg.data, "model_points", getattr(model_cfg.data, "num_points", 64))),
                ),
            ),
            knn_k=_cfg_int(profile_cfg, "knn_k", 4),
            orientation_method=str(getattr(figure_representatives_cfg, "orientation", "pca")),
            view_elev=float(getattr(figure_representatives_cfg, "view_elev", 22.0)),
            view_azim=float(getattr(figure_representatives_cfg, "view_azim", 38.0)),
            projection=str(getattr(figure_representatives_cfg, "projection", "ortho")),
            representative_ptm_enabled=bool(
                getattr(figure_representatives_cfg, "ptm_enabled", False)
            ),
            representative_cna_enabled=bool(
                getattr(figure_representatives_cfg, "cna_enabled", False)
            ),
            representative_cna_max_signatures=int(
                getattr(figure_representatives_cfg, "cna_max_signatures", 5)
            ),
            representative_center_atom_tolerance=float(
                getattr(figure_representatives_cfg, "center_atom_tolerance", 1e-6)
            ),
            representative_shell_min_neighbors=int(
                getattr(figure_representatives_cfg, "shell_min_neighbors", 8)
            ),
            representative_shell_max_neighbors=int(
                getattr(figure_representatives_cfg, "shell_max_neighbors", 24)
            ),
            representative_render_cache=representative_render_cache,
            selection_features=representative_selection_features,
            selection_info=representative_selection_info,
        )
        summary["representatives"] = {
            "root_dir": str(representatives_dir),
            "shared_style": shared_style_summary,
            "primary_figure": str(shared_style_summary["out_file"]),
            "edge_connected_figure": str(
                shared_style_summary["pca_two_shell_figures"]["knn_edges"]["out_file"]
            ),
        }

    if _cfg_bool(descriptor_cfg, "enabled", False):
        descriptor_dir = out_root / "descriptors"
        descriptor_max_samples = getattr(descriptor_cfg, "max_samples", 6000)
        descriptor_indices = _resolve_descriptor_sampling_indices(
            frames,
            labels,
            max_samples=None if descriptor_max_samples is None else int(descriptor_max_samples),
            random_state=int(random_state),
        )
        descriptor_points = _load_point_cloud_batch(
            dataset,
            descriptor_indices,
            point_scale=float(point_scale),
        )
        descriptor_table = _build_builtin_descriptor_table(
            point_clouds=descriptor_points,
            sample_indices=descriptor_indices,
            labels=labels,
            coords=np.asarray(coords, dtype=np.float32),
            frame_index_lookup=frame_index_lookup,
            frame_name_lookup=frame_name_lookup,
            frame_time_lookup=frame_time_lookup,
        )

        optional_descriptors_cfg = _to_plain(getattr(descriptor_cfg, "optional", None))
        optional_descriptor_summaries: list[dict[str, Any]] = []
        scalar_columns = list(_BUILTIN_SCALAR_COLUMNS)
        if isinstance(optional_descriptors_cfg, dict):
            for descriptor_name, descriptor_cfg_raw in optional_descriptors_cfg.items():
                optional_descriptor_cfg = (
                    descriptor_cfg_raw if isinstance(descriptor_cfg_raw, dict) else {}
                )
                if not bool(optional_descriptor_cfg.get("enabled", False)):
                    continue
                descriptor_df, descriptor_info = _evaluate_optional_descriptor(
                    str(descriptor_name),
                    point_clouds=descriptor_points,
                    point_scale=float(point_scale),
                    descriptor_cfg=optional_descriptor_cfg,
                )
                if descriptor_df.shape[0] != descriptor_table.shape[0]:
                    raise RuntimeError(
                        "Optional descriptor result row count mismatch: "
                        f"descriptor={descriptor_name}, optional_rows={descriptor_df.shape[0]}, "
                        f"base_rows={descriptor_table.shape[0]}."
                    )
                descriptor_table = pd.concat([descriptor_table.reset_index(drop=True), descriptor_df], axis=1)
                if str(descriptor_name).strip().lower() == "steinhardt":
                    scalar_columns.extend(descriptor_info["feature_names"])
                optional_descriptor_summaries.append(descriptor_info)

        requested_scalar_columns_raw = _to_plain(getattr(descriptor_cfg, "scalar_columns", None))
        if requested_scalar_columns_raw:
            scalar_columns = [str(v) for v in list(requested_scalar_columns_raw)]
        scalar_columns = [column for column in scalar_columns if column in descriptor_table.columns]
        if not scalar_columns:
            raise RuntimeError(
                "No scalar descriptor columns were available for violin plotting. "
                f"Available columns={descriptor_table.columns.tolist()}."
            )

        descriptor_dir.mkdir(parents=True, exist_ok=True)
        sample_table_csv = descriptor_dir / "descriptor_samples.csv"
        descriptor_table.to_csv(sample_table_csv, index=False)

        scalar_violin_path = descriptor_dir / "descriptor_violin_grid.png"
        summary["descriptor_analysis"] = {
            "sample_table_csv": str(sample_table_csv),
            "scalar_violin_plot": str(scalar_violin_path),
            "scalar_columns": scalar_columns,
            "optional_descriptors": optional_descriptor_summaries,
            "num_samples": int(descriptor_table.shape[0]),
        }
        save_descriptor_violin_grid(
            descriptor_table,
            cluster_column="cluster_id",
            scalar_columns=scalar_columns,
            out_file=scalar_violin_path,
            cluster_color_map=labels_color_map,
            cluster_label_map=cluster_display_map,
        )

        cna_descriptor_enabled = any(
            str(info.get("name", "")).strip().lower() == "cna"
            for info in optional_descriptor_summaries
        )
        if cna_descriptor_enabled:
            cna_cluster_summary, cna_frame_summary, cna_signature_columns = _build_cna_signature_summary_tables(
                descriptor_table,
                cluster_ids=cluster_ids,
                frames=frames,
            )
            cna_cluster_csv = descriptor_dir / "cna_signature_by_cluster.csv"
            cna_frame_csv = descriptor_dir / "cna_signature_by_frame.csv"
            cna_cluster_summary.to_csv(cna_cluster_csv, index=False)
            cna_frame_summary.to_csv(cna_frame_csv, index=False)

            cna_time_plot = descriptor_dir / "cna_signature_time_stacked_area.png"
            time_plot_summary = save_cna_signature_time_series(
                [str(frame.label) for frame in frames],
                cna_frame_summary[cna_signature_columns].to_numpy(dtype=np.float64),
                signature_labels=[
                    "other" if column == "cna_other" else column.replace("cna_", "")
                    for column in cna_signature_columns
                ],
                out_file=cna_time_plot,
                save_svg=True,
            )
            summary["descriptor_analysis"]["cna_visualization"] = {
                "cluster_csv": str(cna_cluster_csv),
                "frame_csv": str(cna_frame_csv),
                "time_series_plot": str(time_plot_summary["out_file"]),
                "time_series_plot_svg": time_plot_summary.get("svg"),
                "signature_columns": list(cna_signature_columns),
            }
    else:
        descriptor_table = pd.DataFrame()

    projection_dir = out_root / "latent"
    projection_method = str(getattr(projection_cfg, "method", "umap")).strip().lower()
    projection_max_samples = getattr(projection_cfg, "max_samples", 15000)
    projection_indices = _resolve_descriptor_sampling_indices(
        frames,
        labels,
        max_samples=None if projection_max_samples is None else int(projection_max_samples),
        random_state=int(random_state) + 11,
    )
    projection_l2_normalize = bool(getattr(clustering_cfg, "l2_normalize", True))
    projection_standardize = bool(getattr(clustering_cfg, "standardize", True))
    projection_pca_variance = float(getattr(clustering_cfg, "pca_variance", 0.98))
    projection_pca_max_components = int(getattr(clustering_cfg, "pca_max_components", 32))
    projection_umap_neighbors = _cfg_int(projection_cfg, "umap_neighbors", 30)
    projection_umap_min_dist = _cfg_float(projection_cfg, "umap_min_dist", 0.15)
    projection_umap_metric = str(getattr(projection_cfg, "umap_metric", "euclidean"))
    projection_umap_backend = str(getattr(projection_cfg, "umap_backend", "auto"))
    projection_tsne_n_iter = _cfg_int(tsne_cfg, "n_iter", 1000)

    fitted_projection = None
    projection_feature_prep: ProjectionFeaturePrep | None = None
    temporal_fit_indices = None
    if temporal_projection_fit_indices is not None and projection_method in {"umap", "pca"}:
        temporal_fit_indices = np.asarray(temporal_projection_fit_indices, dtype=int).reshape(-1)
        temporal_fit_indices = np.unique(temporal_fit_indices.astype(int, copy=False))
        fit_features, projection_feature_prep, projection_prep_info = _prepare_projection_features(
            np.asarray(latents, dtype=np.float32)[temporal_fit_indices],
            random_state=int(random_state),
            l2_normalize=projection_l2_normalize,
            standardize=projection_standardize,
            pca_variance=projection_pca_variance,
            pca_max_components=projection_pca_max_components,
        )
        _, projection_model_info, fitted_projection = _fit_projection_embedding(
            fit_features,
            method=projection_method,
            random_state=int(random_state),
            umap_neighbors=projection_umap_neighbors,
            umap_min_dist=projection_umap_min_dist,
            umap_metric=projection_umap_metric,
            umap_backend=projection_umap_backend,
            tsne_n_iter=projection_tsne_n_iter,
        )
        projection_embedding = _transform_projection_features(
            projection_feature_prep,
            np.asarray(latents, dtype=np.float32)[projection_indices],
            method=projection_method,
            fitted_projection=fitted_projection,
        )
        projection_info = {
            **projection_prep_info,
            **projection_model_info,
            "fit_sample_count": int(temporal_fit_indices.size),
            "fit_sample_source": "temporal_inference_fraction",
        }
    else:
        projection_embedding, projection_info = _compute_projection(
            np.asarray(latents, dtype=np.float32)[projection_indices],
            method=projection_method,
            random_state=int(random_state),
            l2_normalize=projection_l2_normalize,
            standardize=projection_standardize,
            pca_variance=projection_pca_variance,
            pca_max_components=projection_pca_max_components,
            umap_neighbors=projection_umap_neighbors,
            umap_min_dist=projection_umap_min_dist,
            umap_metric=projection_umap_metric,
            umap_backend=projection_umap_backend,
            tsne_n_iter=projection_tsne_n_iter,
        )
    projection_dir.mkdir(parents=True, exist_ok=True)
    projection_table = pd.DataFrame(
        {
            "sample_index": projection_indices.astype(int),
            "x": projection_embedding[:, 0].astype(float),
            "y": projection_embedding[:, 1].astype(float),
            "cluster_id": labels[projection_indices].astype(int),
            "frame_index": frame_index_lookup[projection_indices].astype(int),
            "frame_name": [frame_name_lookup[int(idx)] for idx in projection_indices],
            "time_value": [
                float(frame_time_lookup[int(idx)]) if np.isfinite(frame_time_lookup[int(idx)]) else np.nan
                for idx in projection_indices
            ],
        }
    )
    projection_csv = projection_dir / "latent_projection.csv"
    projection_table.to_csv(projection_csv, index=False)
    cluster_plot = projection_dir / f"latent_projection_{projection_method}_clusters.png"
    summary["latent_projection"] = {
        "projection_csv": str(projection_csv),
        "projection_info": projection_info,
        "plots": {
            "cluster": str(cluster_plot),
        },
        "num_samples": int(projection_table.shape[0]),
    }
    save_embedding_discrete_plot(
        projection_embedding,
        labels[projection_indices],
        cluster_plot,
        title=f"Latent projection ({projection_info['method']}) colored by cluster",
        cluster_color_map=labels_color_map,
        display_label_map=cluster_display_map,
    )

    transition_tol = None
    if _cfg_bool(transition_cfg, "enabled", True):
        transition_tol = _resolve_transition_tolerance(
            model_cfg=model_cfg,
            dataset=dataset,
            transition_cfg=transition_cfg,
        )

    if _cfg_bool(temporal_cfg, "enabled", False):
        if temporal_frames is None or not temporal_frames:
            raise ValueError(
                "real_md.temporal.enabled=true requires temporal_all_frame_groups to be provided."
            )
        temporal_dir = out_root / "temporal"
        temporal_dir.mkdir(parents=True, exist_ok=True)
        frame_duration_ms = _cfg_int(temporal_cfg, "frame_duration_ms", 450)
        total_duration_seconds_raw = getattr(temporal_cfg, "total_duration_seconds", None)
        total_duration_seconds = (
            None if total_duration_seconds_raw is None else float(total_duration_seconds_raw)
        )
        animation_max_points_raw = getattr(temporal_cfg, "animation_max_points", None)
        animation_max_points = (
            None
            if animation_max_points_raw is None or int(animation_max_points_raw) <= 0
            else int(animation_max_points_raw)
        )
        temporal_animation_indices = _select_temporal_animation_sample_indices(
            temporal_frames,
            labels,
            max_points_per_frame=animation_max_points,
            random_state=int(random_state),
        )
        temporal_summary: dict[str, Any] = {
            "root_dir": str(temporal_dir),
            "frame_count": int(len(temporal_frames)),
            "render_max_points_per_frame": (
                None if animation_max_points is None else int(animation_max_points)
            ),
            "frames": [
                {
                    "frame_name": str(frame.source_name),
                    "frame_label": str(frame.label),
                    "sample_count": int(frame.indices.size),
                    "render_count": int(temporal_animation_indices[str(frame.source_name)].size),
                }
                for frame in temporal_frames
            ],
        }
        temporal_frame_label_by_name = {
            str(frame.source_name): str(frame.label)
            for frame in temporal_frames
        }
        temporal_animation_jobs: list[tuple[str, str, dict[str, Any]]] = []

        temporal_md_cfg = getattr(temporal_cfg, "md_space", None)
        if _cfg_bool(temporal_md_cfg, "enabled", True):
            md_animation_frames = (
                temporal_md_animation_frames
                if temporal_md_animation_frames is not None
                else temporal_frames
            )
            if md_animation_frames is None or not md_animation_frames:
                raise ValueError(
                    "MD-space temporal animation requires temporal frames, but none were provided."
                )
            md_animation_coords = (
                np.asarray(temporal_md_animation_coords, dtype=np.float32)
                if temporal_md_animation_coords is not None
                else np.asarray(coords, dtype=np.float32)
            )
            md_animation_labels = (
                np.asarray(
                    temporal_md_animation_cluster_labels_by_k[int(selected_k)],
                    dtype=int,
                )
                if temporal_md_animation_cluster_labels_by_k is not None
                else np.asarray(labels, dtype=int)
            )
            if md_animation_coords.shape[0] != md_animation_labels.shape[0]:
                raise ValueError(
                    "Temporal MD-space animation coords/labels length mismatch: "
                    f"coords={md_animation_coords.shape[0]}, "
                    f"labels={md_animation_labels.shape[0]}."
                )
            md_animation_max_points_raw = getattr(
                temporal_md_cfg,
                "animation_max_points",
                animation_max_points,
            )
            md_animation_max_points = (
                None
                if md_animation_max_points_raw is None or int(md_animation_max_points_raw) <= 0
                else int(md_animation_max_points_raw)
            )
            md_temporal_animation_indices = _select_temporal_animation_sample_indices(
                md_animation_frames,
                md_animation_labels,
                max_points_per_frame=md_animation_max_points,
                random_state=int(random_state),
            )
            temporal_md_records = [
                {
                    "frame_name": str(frame.source_name),
                    "frame_label": str(frame.label),
                    "coords": np.asarray(
                        md_animation_coords[
                            md_temporal_animation_indices[str(frame.source_name)]
                        ],
                        dtype=np.float32,
                    ),
                    "labels": np.asarray(
                        md_animation_labels[
                            md_temporal_animation_indices[str(frame.source_name)]
                        ],
                        dtype=int,
                    ),
                }
                for frame in md_animation_frames
            ]
            md_animation_path = temporal_dir / f"md_space_clusters_diagonal_cut_k{int(selected_k)}.gif"
            temporal_summary["md_space_frame_source"] = (
                str(temporal_md_animation_frame_source)
                if temporal_md_animation_frame_source is not None
                else "dense_selected_frames"
                if temporal_md_animation_frames is not None
                else "temporal_inference_frames"
            )
            temporal_summary["md_space_frame_count"] = int(len(md_animation_frames))
            temporal_summary["md_space_render_max_points_per_frame"] = (
                None
                if md_animation_max_points is None
                else int(md_animation_max_points)
            )
            temporal_summary["md_space_frames"] = [
                {
                    "frame_name": str(frame.source_name),
                    "frame_label": str(frame.label),
                    "sample_count": int(frame.indices.size),
                    "render_count": int(
                        md_temporal_animation_indices[str(frame.source_name)].size
                    ),
                }
                for frame in md_animation_frames
            ]
            temporal_animation_jobs.append(
                (
                    "md_space_animation",
                    "spatial_clusters",
                    {
                        "frame_records": temporal_md_records,
                        "out_file": md_animation_path,
                        "cluster_color_map": labels_color_map,
                        "cluster_display_map": cluster_display_map,
                        "point_size": _cfg_float(
                            temporal_md_cfg,
                            "point_size",
                            _cfg_float(figure_md_cfg, "point_size", 5.6),
                        ),
                        "alpha": _cfg_float(
                            temporal_md_cfg,
                            "alpha",
                            _cfg_float(figure_md_cfg, "alpha", 0.62),
                        ),
                        "saturation_boost": _cfg_float(
                            temporal_md_cfg,
                            "saturation_boost",
                            _cfg_float(figure_md_cfg, "saturation_boost", 1.18),
                        ),
                        "view_elev": _cfg_float(
                            temporal_md_cfg,
                            "view_elev",
                            _cfg_float(figure_md_cfg, "view_elev", 24.0),
                        ),
                        "view_azim": _cfg_float(
                            temporal_md_cfg,
                            "view_azim",
                            _cfg_float(figure_md_cfg, "view_azim", 35.0),
                        ),
                        "diagonal_visible_depth_fraction": _cfg_float(
                            temporal_md_cfg,
                            "diagonal_visible_depth_fraction",
                            0.10,
                        ),
                        "spatial_bounds": temporal_md_animation_spatial_bounds,
                        "frame_duration_ms": int(frame_duration_ms),
                        "total_duration_seconds": total_duration_seconds,
                    },
                )
            )

        if _cfg_bool(transition_cfg, "enabled", True):
            if transition_tol is None:
                raise RuntimeError("transition_tol was not resolved for temporal transition animation.")
            temporal_transition_dir = temporal_dir / "transition_pairs"
            _remove_existing_transition_pair_flow_artifacts(temporal_transition_dir)
            temporal_transition_data = _compute_transitions(
                temporal_frames,
                coords=np.asarray(coords, dtype=np.float32),
                labels=labels,
                instance_ids=instance_ids_arr,
                cluster_ids=cluster_ids,
                max_distance=float(transition_tol),
                require_mutual=_cfg_bool(transition_cfg, "require_mutual", True),
            )
            temporal_pair_records: list[dict[str, Any]] = []
            temporal_pair_plots: list[dict[str, Any]] = []
            for pair_summary in temporal_transition_data["pairs"]:
                frame_from = str(pair_summary["frame_from"])
                frame_to = str(pair_summary["frame_to"])
                frame_from_label = temporal_frame_label_by_name.get(frame_from, frame_from)
                frame_to_label = temporal_frame_label_by_name.get(frame_to, frame_to)
                temporal_pair_records.append(
                    {
                        "frame_from": frame_from,
                        "frame_to": frame_to,
                        "frame_from_label": frame_from_label,
                        "frame_to_label": frame_to_label,
                        "counts": np.asarray(pair_summary["counts"], dtype=np.float64),
                        "title": f"Cluster transition flow | {frame_from_label} -> {frame_to_label}",
                    }
                )
                temporal_pair_plot_record = {
                    "frame_from": frame_from,
                    "frame_to": frame_to,
                    "frame_from_label": frame_from_label,
                    "frame_to_label": frame_to_label,
                    "matched_samples": int(pair_summary["matched_samples"]),
                    "coverage_fraction": float(pair_summary["coverage_fraction"]),
                }
                temporal_pair_plots.append(temporal_pair_plot_record)
            if temporal_pair_records:
                transition_animation_path = temporal_dir / "transition_pair_flows.gif"
                temporal_animation_jobs.append(
                    (
                        "transition_pair_flow_animation",
                        "transition_flow",
                        {
                            "pair_records": temporal_pair_records,
                            "out_file": transition_animation_path,
                            "row_labels": cluster_display_labels,
                            "cluster_color_map": labels_color_map,
                            "cluster_ids_for_palette": cluster_ids,
                            "mute_diagonal": _cfg_bool(transition_cfg, "flow_mute_diagonal", True),
                            "min_draw_fraction": _cfg_float(transition_cfg, "flow_min_fraction", 0.001),
                            "frame_duration_ms": int(frame_duration_ms),
                            "total_duration_seconds": total_duration_seconds,
                            "title": "Cluster transition flow",
                        },
                    )
                )
            temporal_summary["transition_pairs"] = {
                "pair_count": int(len(temporal_pair_plots)),
                "match_mode": str(temporal_transition_data["match_mode"]),
                "match_tolerance": float(temporal_transition_data["max_distance"]),
                "require_mutual": bool(temporal_transition_data["require_mutual"]),
                "pairs": temporal_pair_plots,
            }

        temporal_umap_cfg = getattr(temporal_cfg, "umap", None)
        if _cfg_bool(temporal_umap_cfg, "enabled", True):
            if projection_method != "umap":
                raise ValueError(
                    "real_md.temporal.umap.enabled=true requires real_md.projection.method='umap'. "
                    f"Got projection.method={projection_method!r}."
                )
            if projection_feature_prep is None or fitted_projection is None:
                raise RuntimeError(
                    "Temporal UMAP animation requires a fitted transformable projection, "
                    "but no fitted projection state was available."
                )
            umap_animation_frame_limit = _resolve_umap_animation_frame_count(
                temporal_umap_cfg
            )
            umap_animation_frames = _select_evenly_spaced_frames(
                temporal_frames,
                max_frame_count=umap_animation_frame_limit,
                field_name="real_md.temporal.umap.animation_frame_count",
            )
            if len(umap_animation_frames) < len(temporal_frames):
                print(
                    "[analysis][real_md] Temporal UMAP animation frame cap: "
                    f"{len(umap_animation_frames)}/{len(temporal_frames)} frames."
                )
            temporal_summary["umap_animation_frame_count"] = int(
                len(umap_animation_frames)
            )
            temporal_summary["umap_animation_full_frame_count"] = int(
                len(temporal_frames)
            )
            temporal_summary["umap_animation_frame_limit"] = (
                None
                if umap_animation_frame_limit is None
                else int(umap_animation_frame_limit)
            )
            temporal_summary["umap_frames"] = [
                {
                    "frame_name": str(frame.source_name),
                    "frame_label": str(frame.label),
                    "sample_count": int(frame.indices.size),
                    "render_count": int(
                        temporal_animation_indices[str(frame.source_name)].size
                    ),
                }
                for frame in umap_animation_frames
            ]
            all_animation_indices = np.concatenate(
                [
                    temporal_animation_indices[str(frame.source_name)]
                    for frame in umap_animation_frames
                ]
            ).astype(int, copy=False)
            if all_animation_indices.size == 0:
                raise RuntimeError(
                    "Temporal UMAP animation resolved to zero sampled points across all frames."
                )
            all_animation_embedding = _transform_projection_features(
                projection_feature_prep,
                np.asarray(latents, dtype=np.float32)[all_animation_indices],
                method=projection_method,
                fitted_projection=fitted_projection,
            )
            temporal_embedding_records: list[dict[str, Any]] = []
            cursor = 0
            for frame in umap_animation_frames:
                frame_indices = temporal_animation_indices[str(frame.source_name)]
                next_cursor = cursor + int(frame_indices.size)
                temporal_embedding_records.append(
                    {
                        "frame_name": str(frame.source_name),
                        "frame_label": str(frame.label),
                        "embedding": np.asarray(
                            all_animation_embedding[cursor:next_cursor],
                            dtype=np.float32,
                        ),
                        "labels": np.asarray(labels[frame_indices], dtype=int),
                    }
                )
                cursor = next_cursor
            umap_animation_path = temporal_dir / f"latent_projection_{projection_method}_clusters.gif"
            temporal_animation_jobs.append(
                (
                    "umap_animation",
                    "embedding_clusters",
                    {
                        "frame_records": temporal_embedding_records,
                        "out_file": umap_animation_path,
                        "cluster_color_map": labels_color_map,
                        "cluster_display_map": cluster_display_map,
                        "point_size": _cfg_float(temporal_umap_cfg, "point_size", 8.0),
                        "alpha": _cfg_float(temporal_umap_cfg, "alpha", 0.74),
                        "frame_duration_ms": int(frame_duration_ms),
                        "total_duration_seconds": total_duration_seconds,
                        "title": f"{projection_method.upper()} cluster evolution",
                    },
                )
            )
            if _cfg_bool(temporal_umap_cfg, "trajectory_enabled", True):
                trajectory_max_points_raw = getattr(temporal_umap_cfg, "trajectory_max_points", animation_max_points)
                trajectory_max_points = (
                    None
                    if trajectory_max_points_raw is None or int(trajectory_max_points_raw) <= 0
                    else int(trajectory_max_points_raw)
                )
                trajectory_indices_by_frame, trajectory_instance_ids = _select_temporal_trajectory_sample_indices(
                    umap_animation_frames,
                    labels,
                    instance_ids_arr,
                    max_points=trajectory_max_points,
                    random_state=int(random_state) + 31,
                )
                trajectory_concat = np.concatenate(
                    [
                        trajectory_indices_by_frame[str(frame.source_name)]
                        for frame in umap_animation_frames
                    ]
                ).astype(int, copy=False)
                trajectory_embedding = _transform_projection_features(
                    projection_feature_prep,
                    np.asarray(latents, dtype=np.float32)[trajectory_concat],
                    method=projection_method,
                    fitted_projection=fitted_projection,
                )
                temporal_trajectory_records: list[dict[str, Any]] = []
                cursor = 0
                for frame in umap_animation_frames:
                    frame_indices = trajectory_indices_by_frame[str(frame.source_name)]
                    next_cursor = cursor + int(frame_indices.size)
                    temporal_trajectory_records.append(
                        {
                            "frame_name": str(frame.source_name),
                            "frame_label": str(frame.label),
                            "embedding": np.asarray(
                                trajectory_embedding[cursor:next_cursor],
                                dtype=np.float32,
                            ),
                            "labels": np.asarray(labels[frame_indices], dtype=int),
                            "instance_ids": np.asarray(instance_ids_arr[frame_indices], dtype=np.int64),
                        }
                    )
                    cursor = next_cursor
                trajectory_animation_path = (
                    temporal_dir / f"latent_projection_{projection_method}_trajectories.gif"
                )
                temporal_animation_jobs.append(
                    (
                        "umap_trajectory_animation",
                        "embedding_trajectories",
                        {
                            "frame_records": temporal_trajectory_records,
                            "out_file": trajectory_animation_path,
                            "cluster_color_map": labels_color_map,
                            "cluster_display_map": cluster_display_map,
                            "line_width": _cfg_float(temporal_umap_cfg, "trajectory_line_width", 0.8),
                            "line_alpha": _cfg_float(temporal_umap_cfg, "trajectory_line_alpha", 0.22),
                            "history_steps": _cfg_int(
                                temporal_umap_cfg,
                                "trajectory_history_steps",
                                8,
                            ),
                            "fade_min_alpha_fraction": _cfg_float(
                                temporal_umap_cfg,
                                "trajectory_fade_min_alpha_fraction",
                                0.18,
                            ),
                            "fade_power": _cfg_float(
                                temporal_umap_cfg,
                                "trajectory_fade_power",
                                1.0,
                            ),
                            "directional_subsegments": _cfg_int(
                                temporal_umap_cfg,
                                "trajectory_directional_subsegments",
                                6,
                            ),
                            "directional_start_alpha_fraction": _cfg_float(
                                temporal_umap_cfg,
                                "trajectory_directional_start_alpha_fraction",
                                0.32,
                            ),
                            "directional_start_width_fraction": _cfg_float(
                                temporal_umap_cfg,
                                "trajectory_directional_start_width_fraction",
                                0.60,
                            ),
                            "directional_end_width_fraction": _cfg_float(
                                temporal_umap_cfg,
                                "trajectory_directional_end_width_fraction",
                                1.35,
                            ),
                            "endpoint_point_size": _cfg_float(
                                temporal_umap_cfg,
                                "trajectory_endpoint_point_size",
                                3.0,
                            ),
                            "endpoint_point_alpha": _cfg_float(
                                temporal_umap_cfg,
                                "trajectory_endpoint_point_alpha",
                                0.95,
                            ),
                            "frame_duration_ms": int(frame_duration_ms),
                            "total_duration_seconds": total_duration_seconds,
                            "title": f"{projection_method.upper()} cluster trajectories",
                        },
                    )
                )
                temporal_summary["trajectory_sample_count"] = int(trajectory_instance_ids.size)
        temporal_parallel_workers = _resolve_temporal_animation_parallel_workers(
            temporal_cfg,
            task_count=len(temporal_animation_jobs),
        )
        temporal_summary["animation_parallel_workers"] = int(temporal_parallel_workers)
        if temporal_parallel_workers > 1:
            print(
                "[analysis] Rendering temporal animations in parallel with "
                f"{temporal_parallel_workers} worker processes."
            )
        temporal_animation_results = _execute_temporal_animation_jobs(
            temporal_animation_jobs,
            temporal_cfg=temporal_cfg,
        )
        for summary_key, _task_kind, _task_kwargs in temporal_animation_jobs:
            temporal_summary[str(summary_key)] = temporal_animation_results[str(summary_key)]
        temporal_summary["projection_fit_sample_count"] = (
            None if temporal_fit_indices is None else int(temporal_fit_indices.size)
        )
        summary["temporal"] = temporal_summary

    if _cfg_bool(transition_cfg, "enabled", True) and len(frames) >= 2:
        transition_dir = out_root / "transitions"
        transition_dir.mkdir(parents=True, exist_ok=True)
        _remove_existing_transition_pair_flow_artifacts(transition_dir)
        _remove_existing_transition_aggregate_flow_artifacts(transition_dir)
        if transition_tol is None:
            raise RuntimeError("transition_tol was not resolved for transition analysis.")
        transition_data = _compute_transitions(
            frames,
            coords=np.asarray(coords, dtype=np.float32),
            labels=labels,
            instance_ids=instance_ids_arr,
            cluster_ids=cluster_ids,
            max_distance=float(transition_tol),
            require_mutual=_cfg_bool(transition_cfg, "require_mutual", True),
        )
        save_transition_svg = _cfg_bool(transition_cfg, "save_svg", False)
        pair_plot_indices = set(
            _resolve_transition_pair_plot_indices(
                transition_cfg,
                pair_count=len(transition_data["pairs"]),
            )
        )
        pair_records: list[dict[str, Any]] = []
        for pair_list_index, pair_summary in enumerate(transition_data["pairs"]):
            counts = np.asarray(pair_summary["counts"], dtype=np.float64)
            pair_idx = int(len(pair_records) + 1)
            pair_record = {
                "frame_from": str(pair_summary["frame_from"]),
                "frame_to": str(pair_summary["frame_to"]),
                "matched_samples": int(pair_summary["matched_samples"]),
                "coverage_fraction": float(pair_summary["coverage_fraction"]),
            }
            if pair_list_index in pair_plot_indices:
                flow_path = transition_dir / f"transition_pair_{pair_idx:02d}_flow.png"
                save_transition_flow_plot(
                    counts,
                    flow_path,
                    title=None,
                    row_labels=cluster_display_labels,
                    cluster_color_map=labels_color_map,
                    cluster_ids_for_palette=cluster_ids,
                    mute_diagonal=_cfg_bool(transition_cfg, "flow_mute_diagonal", True),
                    min_draw_fraction=_cfg_float(transition_cfg, "flow_min_fraction", 0.001),
                )
                pair_record["flow_plot"] = str(flow_path)
                if save_transition_svg:
                    flow_svg_path = transition_dir / f"transition_pair_{pair_idx:02d}_flow.svg"
                    save_transition_flow_plot(
                        counts,
                        flow_svg_path,
                        title=None,
                        row_labels=cluster_display_labels,
                        cluster_color_map=labels_color_map,
                        cluster_ids_for_palette=cluster_ids,
                        mute_diagonal=_cfg_bool(transition_cfg, "flow_mute_diagonal", True),
                        min_draw_fraction=_cfg_float(transition_cfg, "flow_min_fraction", 0.001),
                    )
                    pair_record["flow_svg"] = str(flow_svg_path)
            pair_records.append(pair_record)

        aggregate_flow_path = transition_dir / "transition_aggregate_flow.png"
        save_transition_flow_plot(
            transition_data["aggregate_counts"],
            aggregate_flow_path,
            title=None,
            row_labels=cluster_display_labels,
            cluster_color_map=labels_color_map,
            cluster_ids_for_palette=cluster_ids,
            mute_diagonal=_cfg_bool(transition_cfg, "flow_mute_diagonal", True),
            min_draw_fraction=_cfg_float(transition_cfg, "flow_min_fraction", 0.001),
        )
        aggregate_flow_svg_path = None
        if save_transition_svg:
            aggregate_flow_svg_path = transition_dir / "transition_aggregate_flow.svg"
            save_transition_flow_plot(
                transition_data["aggregate_counts"],
                aggregate_flow_svg_path,
                title=None,
                row_labels=cluster_display_labels,
                cluster_color_map=labels_color_map,
                cluster_ids_for_palette=cluster_ids,
                mute_diagonal=_cfg_bool(transition_cfg, "flow_mute_diagonal", True),
                min_draw_fraction=_cfg_float(transition_cfg, "flow_min_fraction", 0.001),
            )
        summary["transitions"] = {
            "cluster_ids": [int(v) for v in cluster_ids],
            "cluster_display_labels": cluster_display_labels,
            "match_tolerance": float(transition_data["max_distance"]),
            "require_mutual": bool(transition_data["require_mutual"]),
            "match_mode": str(transition_data["match_mode"]),
            "pairs": pair_records,
            "aggregate_flow": str(aggregate_flow_path),
            "aggregate_flow_svg": (
                None if aggregate_flow_svg_path is None else str(aggregate_flow_svg_path)
            ),
        }

    summary_json = out_root / "summary.json"
    write_json(summary_json, summary)
    summary["summary_json"] = str(summary_json)

    summary_markdown = out_root / "README.md"
    _write_summary_markdown(
        summary_markdown,
        selected_k=int(selected_k),
        frames=frames,
        cluster_groups=cluster_groups,
        summary=summary,
    )
    summary["summary_markdown"] = str(summary_markdown)
    write_json(summary_json, summary)
    return summary


__all__ = ["append_dynamic_motif_summary", "run_real_md_qualitative_analysis"]
