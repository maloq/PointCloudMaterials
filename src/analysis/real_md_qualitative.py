from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

from .cluster_profiles import (
    _ALL_PROFILE_PROPERTIES,
    _compute_sample_properties,
    _json_default,
    _load_point_cloud_from_dataset,
)
from .cluster_rendering import (
    _save_cluster_representatives_figure,
)
from .cluster_figures import _build_cluster_color_map
from src.vis_tools.latent_analysis_vis import _prepare_clustering_features
from src.vis_tools.real_md_analysis_vis import (
    save_cna_cluster_signature_stacked_bar,
    save_cna_signature_time_series,
    save_cluster_proportion_plots,
    save_descriptor_violin_grid,
    save_embedding_continuous_plot,
    save_embedding_discrete_plot,
    save_spatial_cluster_view,
    save_transition_flow_plot,
)
from src.vis_tools.tsne_vis import compute_tsne


_TIME_RE = re.compile(r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?P<unit>[A-Za-z]+)")
_BUILTIN_SCALAR_COLUMNS = [str(key) for key, _ in _ALL_PROFILE_PROPERTIES]
_BUILTIN_SCALAR_LABELS = {str(key): str(label) for key, label in _ALL_PROFILE_PROPERTIES}


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
        stem = Path(str(self.source_name)).stem or str(self.source_name)
        if self.time_value is None or self.time_unit is None:
            return stem
        value = float(self.time_value)
        if abs(value - round(value)) < 1e-9:
            return f"{int(round(value))}{self.time_unit}"
        return f"{value:g}{self.time_unit}"


def _to_plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _parse_time_from_name(name: str) -> tuple[float | None, str | None]:
    stem = Path(str(name)).stem or str(name)
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


def _normalize_bounds(bounds: Any) -> np.ndarray:
    arr = np.asarray(bounds, dtype=np.float32)
    if arr.shape == (6,):
        arr = arr.reshape(2, 3)
    if arr.shape != (2, 3):
        raise ValueError(
            "Spatial zoom bounds must have shape (6,) or (2, 3), "
            f"got {arr.shape}."
        )
    mins = np.minimum(arr[0], arr[1])
    maxs = np.maximum(arr[0], arr[1])
    return np.stack([mins, maxs], axis=0)


def _resolve_half_extent(
    value: Any,
    *,
    default_half_extent: float,
) -> np.ndarray:
    if value is None:
        return np.full((3,), float(default_half_extent), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape == ():
        return np.full((3,), float(arr), dtype=np.float32)
    if arr.shape != (3,):
        raise ValueError(
            "half_extent must be a scalar or length-3 vector, "
            f"got shape {arr.shape}."
        )
    return arr.astype(np.float32, copy=False)


def _compute_hotspot_bounds(
    coords: np.ndarray,
    *,
    default_half_extent: float,
    half_extent_override: Any = None,
    search_radius: float | None = None,
) -> np.ndarray:
    points = np.asarray(coords, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"coords must have shape (N, >=3), got {points.shape}.")
    points = points[:, :3]
    if points.shape[0] == 0:
        raise ValueError("Cannot compute hotspot bounds for an empty point set.")
    if points.shape[0] == 1:
        center = points[0]
    else:
        radius = float(search_radius) if search_radius is not None else float(default_half_extent) * 0.8
        radius = max(radius, 1e-3)
        tree = cKDTree(points)
        neighbor_lists = tree.query_ball_point(points, r=radius)
        counts = np.asarray([len(neigh) for neigh in neighbor_lists], dtype=np.int32)
        center = points[int(np.argmax(counts))]
    half_extent = _resolve_half_extent(
        half_extent_override,
        default_half_extent=float(default_half_extent),
    )
    return np.stack([center - half_extent, center + half_extent], axis=0)


def _cfg_bool(cfg: Any, field_name: str, default: bool) -> bool:
    return bool(getattr(cfg, field_name, default))


def _cfg_int(cfg: Any, field_name: str, default: int) -> int:
    return int(getattr(cfg, field_name, default))


def _cfg_float(cfg: Any, field_name: str, default: float) -> float:
    return float(getattr(cfg, field_name, default))


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
            r_cut_min=float(descriptor_cfg.get("r_cut_min", 0.35)),
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
    tsne_n_iter: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    method_norm = str(method).strip().lower()
    features, prep_info = _prepare_clustering_features(
        latents,
        random_state=int(random_state),
        l2_normalize=bool(l2_normalize),
        standardize=bool(standardize),
        pca_variance=pca_variance,
        pca_max_components=int(pca_max_components),
    )
    if method_norm == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError("UMAP projection requested but umap-learn is not installed.") from exc
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=int(umap_neighbors),
            min_dist=float(umap_min_dist),
            metric=str(umap_metric),
            random_state=int(random_state),
        )
        embedding = reducer.fit_transform(features)
        return np.asarray(embedding, dtype=np.float32), {
            **prep_info,
            "method": "umap",
            "n_neighbors": int(umap_neighbors),
            "min_dist": float(umap_min_dist),
            "metric": str(umap_metric),
        }
    if method_norm == "tsne":
        perplexity = min(50, max(5, len(features) // 100))
        embedding = compute_tsne(
            features,
            random_state=int(random_state),
            perplexity=int(perplexity),
            n_iter=int(tsne_n_iter),
        )
        return np.asarray(embedding, dtype=np.float32), {
            **prep_info,
            "method": "tsne",
            "perplexity": int(perplexity),
            "n_iter": int(tsne_n_iter),
        }
    if method_norm == "pca":
        pca = PCA(n_components=2, random_state=int(random_state))
        embedding = pca.fit_transform(features)
        return np.asarray(embedding, dtype=np.float32), {
            **prep_info,
            "method": "pca",
            "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_.tolist()],
        }
    raise ValueError(
        "real_md.projection.method must be one of ['umap', 'tsne', 'pca'], "
        f"got {method!r}."
    )


def _compute_scalar_values_for_indices(
    *,
    scalar_name: str,
    dataset: Any,
    sample_indices: np.ndarray,
    labels: np.ndarray,
    coords: np.ndarray,
    frame_index_lookup: np.ndarray,
    frame_name_lookup: list[str],
    frame_time_lookup: np.ndarray,
    point_scale: float,
    descriptor_table: pd.DataFrame,
    descriptors_cfg_root: Any,
) -> np.ndarray | None:
    requested_scalar = str(scalar_name)
    sample_indices_arr = np.asarray(sample_indices, dtype=int)

    if requested_scalar in _BUILTIN_SCALAR_COLUMNS:
        projection_points = _load_point_cloud_batch(
            dataset,
            sample_indices_arr,
            point_scale=float(point_scale),
        )
        projection_descriptor_table = _build_builtin_descriptor_table(
            point_clouds=projection_points,
            sample_indices=sample_indices_arr,
            labels=labels,
            coords=np.asarray(coords, dtype=np.float32),
            frame_index_lookup=frame_index_lookup,
            frame_name_lookup=frame_name_lookup,
            frame_time_lookup=frame_time_lookup,
        )
        return projection_descriptor_table[requested_scalar].to_numpy(dtype=np.float32)

    if not descriptor_table.empty and requested_scalar in descriptor_table.columns:
        scalar_lookup = descriptor_table.set_index("sample_index")[requested_scalar]
        aligned = scalar_lookup.reindex(sample_indices_arr)
        if aligned.notna().all():
            return aligned.to_numpy(dtype=np.float32)

    optional_descriptors_cfg = _to_plain(descriptors_cfg_root)
    if not isinstance(optional_descriptors_cfg, dict):
        return None

    projection_points: np.ndarray | None = None
    for descriptor_name, descriptor_cfg_raw in optional_descriptors_cfg.items():
        descriptor_cfg = descriptor_cfg_raw if isinstance(descriptor_cfg_raw, dict) else {}
        if not bool(descriptor_cfg.get("enabled", False)):
            continue
        if projection_points is None:
            projection_points = _load_point_cloud_batch(
                dataset,
                sample_indices_arr,
                point_scale=float(point_scale),
            )
        descriptor_df, _ = _evaluate_optional_descriptor(
            str(descriptor_name),
            point_clouds=projection_points,
            point_scale=float(point_scale),
            descriptor_cfg=descriptor_cfg,
        )
        if requested_scalar in descriptor_df.columns:
            return descriptor_df[requested_scalar].to_numpy(dtype=np.float32)
    return None


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

    spatial_cfg = getattr(real_md_cfg, "spatial", None)
    auto_top_clusters = max(0, _cfg_int(spatial_cfg, "auto_top_clusters", 2))
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


def _build_zoom_specs(
    spatial_cfg: Any,
    data_cfg: Any,
    *,
    frames: list[FrameSlice],
    coords: np.ndarray,
    labels: np.ndarray,
    cluster_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    zoom_specs_raw = _to_plain(getattr(spatial_cfg, "zoom_specs", None))
    default_half_extent = _cfg_float(
        spatial_cfg,
        "auto_zoom_half_extent",
        float(getattr(data_cfg, "radius", 8.0)) * 2.5,
    )
    search_radius = _cfg_float(
        spatial_cfg,
        "auto_zoom_search_radius",
        default_half_extent * 0.8,
    )
    specs: list[dict[str, Any]] = []
    if isinstance(zoom_specs_raw, list):
        for spec_idx, raw in enumerate(zoom_specs_raw):
            if not isinstance(raw, dict):
                raise TypeError(
                    "Each real_md.spatial.zoom_specs entry must be a dict, "
                    f"got {type(raw)!r} at index {spec_idx}."
                )
            name = str(raw.get("name", f"zoom_{spec_idx + 1}"))
            frame_selector = str(raw.get("frame", "latest"))
            target_frame = frames[-1] if frame_selector == "latest" else next(
                (frame for frame in frames if str(frame.source_name) == frame_selector or str(frame.output_name) == frame_selector),
                None,
            )
            if target_frame is None:
                raise ValueError(
                    "Zoom spec references an unknown frame. "
                    f"spec={raw}, available_frames={[frame.source_name for frame in frames]}."
                )
            cluster_ids = None
            if raw.get("cluster_ids") is not None:
                cluster_ids = sorted(_normalize_cluster_id_list(raw.get("cluster_ids"), field_name=f"real_md.spatial.zoom_specs[{spec_idx}].cluster_ids"))
            if raw.get("bounds") is not None:
                bounds = _normalize_bounds(raw["bounds"])
            else:
                frame_coords = coords[target_frame.indices]
                frame_labels = labels[target_frame.indices]
                if cluster_ids is not None:
                    mask = np.isin(frame_labels, np.asarray(cluster_ids, dtype=int))
                    frame_coords = frame_coords[mask]
                if frame_coords.size == 0:
                    raise ValueError(
                        "Zoom spec resolved to zero points after cluster filtering: "
                        f"spec={raw}."
                    )
                if raw.get("center") is not None:
                    center = np.asarray(raw.get("center"), dtype=np.float32)
                    if center.shape != (3,):
                        raise ValueError(
                            "Zoom spec center must have shape (3,), "
                            f"got {center.shape}."
                        )
                    half_extent = _resolve_half_extent(
                        raw.get("half_extent"),
                        default_half_extent=float(default_half_extent),
                    )
                    bounds = np.stack([center - half_extent, center + half_extent], axis=0)
                else:
                    bounds = _compute_hotspot_bounds(
                        frame_coords,
                        default_half_extent=float(default_half_extent),
                        half_extent_override=raw.get("half_extent"),
                        search_radius=float(search_radius),
                    )
            specs.append(
                {
                    "name": str(name),
                    "frame_name": str(target_frame.source_name),
                    "frame_output_name": str(target_frame.output_name),
                    "cluster_ids": None if cluster_ids is None else [int(v) for v in cluster_ids],
                    "bounds": bounds.astype(float).tolist(),
                    "source": "config",
                }
            )

    auto_zoom_enabled = _cfg_bool(spatial_cfg, "auto_zoom_enabled", True)
    if auto_zoom_enabled and cluster_groups and frames:
        latest_frame = frames[-1]
        frame_coords = coords[latest_frame.indices]
        frame_labels = labels[latest_frame.indices]
        auto_limit = max(0, _cfg_int(spatial_cfg, "auto_zoom_count", len(cluster_groups)))
        for group in cluster_groups[:auto_limit]:
            cluster_ids = [int(v) for v in group["cluster_ids"]]
            mask = np.isin(frame_labels, np.asarray(cluster_ids, dtype=int))
            if not np.any(mask):
                continue
            bounds = _compute_hotspot_bounds(
                frame_coords[mask],
                default_half_extent=float(default_half_extent),
                search_radius=float(search_radius),
            )
            specs.append(
                {
                    "name": f"auto_{group['name']}",
                    "frame_name": str(latest_frame.source_name),
                    "frame_output_name": str(latest_frame.output_name),
                    "cluster_ids": cluster_ids,
                    "bounds": bounds.astype(float).tolist(),
                    "source": "auto_hotspot_latest_frame",
                }
            )
    return specs


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
    cluster_ids: list[int],
    max_distance: float,
    require_mutual: bool,
) -> dict[str, Any]:
    cluster_to_pos = {int(cluster_id): int(pos) for pos, cluster_id in enumerate(cluster_ids)}
    aggregate_counts = np.zeros((len(cluster_ids), len(cluster_ids)), dtype=np.int64)
    pair_summaries: list[dict[str, Any]] = []
    for frame_a, frame_b in zip(frames[:-1], frames[1:], strict=True):
        coords_a = np.asarray(coords[frame_a.indices, :3], dtype=np.float32)
        coords_b = np.asarray(coords[frame_b.indices, :3], dtype=np.float32)
        labels_a = np.asarray(labels[frame_a.indices], dtype=int)
        labels_b = np.asarray(labels[frame_b.indices], dtype=int)
        if coords_a.size == 0 or coords_b.size == 0:
            continue
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
        if matched_a.size == 0:
            pair_counts = np.zeros((len(cluster_ids), len(cluster_ids)), dtype=np.int64)
        else:
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
    }


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
                f"- Stacked counts: `{Path(summary['cluster_proportions']['plots']['stacked_bar_count']).name}`",
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
        lines.extend(
            [
                "",
                "## Latent projection",
                f"- Method: `{summary['latent_projection']['projection_info']['method']}`",
                f"- CSV: `{Path(summary['latent_projection']['projection_csv']).name}`",
            ]
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
                f"- CNA cluster composition: `{Path(cna_vis['cluster_plot']).name}`"
            )
            lines.append(
                f"- CNA time series: `{Path(cna_vis['time_series_plot']).name}`"
            )
    if "transitions" in summary:
        lines.extend(
            [
                "",
                "## Transitions",
                f"- Match tolerance: `{summary['transitions']['match_tolerance']:.4f}`",
                f"- Aggregate flow diagram: `{Path(summary['transitions']['aggregate_flow']).name}`",
            ]
        )
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_real_md_qualitative_analysis(
    *,
    out_dir: Path,
    model_cfg: Any,
    analysis_cfg: Any,
    dataset: Any,
    latents: np.ndarray,
    coords: np.ndarray,
    cluster_labels_by_k: dict[int, np.ndarray],
    cluster_methods_by_k: dict[int, str],
    cluster_color_map: dict[int, str] | None,
    frame_groups: list[tuple[str, np.ndarray]],
    frame_output_names: dict[str, str],
    requested_frame_order: list[str] | None,
    point_scale: float,
    random_state: int,
    representative_render_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if getattr(model_cfg.data, "kind", None) != "real":
        raise ValueError(
            "run_real_md_qualitative_analysis only supports model_cfg.data.kind='real', "
            f"got {getattr(model_cfg.data, 'kind', None)!r}."
        )
    if not frame_groups:
        raise ValueError("Real-MD qualitative analysis requires at least one frame group.")

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
    spatial_cfg = getattr(real_md_cfg, "spatial", None)
    transition_cfg = getattr(real_md_cfg, "transitions", None)

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
    if labels.shape[0] != len(latents):
        raise ValueError(
            "cluster labels and latents length mismatch: "
            f"labels={labels.shape[0]}, latents={len(latents)}."
        )
    if coords.shape[0] != len(latents):
        raise ValueError(
            "coords and latents length mismatch: "
            f"coords={coords.shape[0]}, latents={len(latents)}."
        )
    if dataset is None:
        raise ValueError("A dataset object is required to render representative local environments.")

    out_root = Path(out_dir) / "real_md_qualitative"
    out_root.mkdir(parents=True, exist_ok=True)
    frames = _build_frame_slices(
        frame_groups,
        frame_output_names,
        requested_order=requested_frame_order,
    )
    labels_color_map = cluster_color_map or _build_cluster_color_map(labels)
    frame_index_lookup, frame_name_lookup, frame_time_lookup = _build_frame_lookup_for_samples(
        frames,
        num_samples=len(latents),
    )
    if np.any(frame_index_lookup < 0):
        missing_count = int(np.sum(frame_index_lookup < 0))
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
        "cluster_groups": cluster_groups,
    }

    proportions_dir = out_root / "time_series"
    proportions_table, cluster_ids, counts_matrix = _cluster_counts_by_frame(frames, labels)
    cluster_display_map = {
        int(cluster_id): f"C{pos + 1}"
        for pos, cluster_id in enumerate(cluster_ids)
    }
    cluster_display_labels = [cluster_display_map[int(cluster_id)] for cluster_id in cluster_ids]
    proportions_csv = proportions_dir / "frame_cluster_proportions.csv"
    proportions_dir.mkdir(parents=True, exist_ok=True)
    proportions_table.to_csv(proportions_csv, index=False)
    summary["cluster_proportions"] = {
        "table_csv": str(proportions_csv),
        "plots": save_cluster_proportion_plots(
            [frame.label for frame in frames],
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
        )
        summary["representatives"] = {
            "root_dir": str(representatives_dir),
            "shared_style": shared_style_summary,
            "primary_figure": str(
                shared_style_summary["pca_two_shell_figures"]["spatial_neighbors_paper"]["out_file"]
            ),
            "edge_connected_figure": str(
                shared_style_summary["pca_two_shell_figures"]["knn_edges"]["out_file"]
            ),
        }

    if _cfg_bool(descriptor_cfg, "enabled", True):
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

            cna_cluster_plot = descriptor_dir / "cna_signature_cluster_stacked_bar.png"
            cna_time_plot = descriptor_dir / "cna_signature_time_stacked_area.png"
            cluster_plot_summary = save_cna_cluster_signature_stacked_bar(
                cna_cluster_summary,
                out_file=cna_cluster_plot,
                cluster_color_map=labels_color_map,
                cluster_label_map=cluster_display_map,
                save_svg=True,
            )
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
                "cluster_plot": str(cluster_plot_summary["out_file"]),
                "cluster_plot_svg": cluster_plot_summary.get("svg"),
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
    projection_embedding, projection_info = _compute_projection(
        np.asarray(latents, dtype=np.float32)[projection_indices],
        method=projection_method,
        random_state=int(random_state),
        l2_normalize=bool(getattr(clustering_cfg, "l2_normalize", True)),
        standardize=bool(getattr(clustering_cfg, "standardize", True)),
        pca_variance=float(getattr(clustering_cfg, "pca_variance", 0.98)),
        pca_max_components=int(getattr(clustering_cfg, "pca_max_components", 32)),
        umap_neighbors=_cfg_int(projection_cfg, "umap_neighbors", 30),
        umap_min_dist=_cfg_float(projection_cfg, "umap_min_dist", 0.15),
        umap_metric=str(getattr(projection_cfg, "umap_metric", "euclidean")),
        tsne_n_iter=_cfg_int(tsne_cfg, "n_iter", 1000),
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
    frame_plot_labels = np.asarray(
        [frame_index_lookup[int(idx)] for idx in projection_indices],
        dtype=int,
    )
    projection_csv = projection_dir / "latent_projection.csv"
    projection_table.to_csv(projection_csv, index=False)
    cluster_plot = projection_dir / f"latent_projection_{projection_method}_clusters.png"
    frame_plot = projection_dir / f"latent_projection_{projection_method}_frames.png"
    summary["latent_projection"] = {
        "projection_csv": str(projection_csv),
        "projection_info": projection_info,
        "plots": {
            "cluster": str(cluster_plot),
            "frame": str(frame_plot),
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
    save_embedding_discrete_plot(
        projection_embedding,
        frame_plot_labels,
        frame_plot,
        title=f"Latent projection ({projection_info['method']}) colored by frame",
        label_prefix="F",
    )

    scalar_name = str(getattr(descriptor_cfg, "physical_scalar", "coord_mean"))
    scalar_label = _BUILTIN_SCALAR_LABELS.get(scalar_name, scalar_name)
    projection_scalar = _compute_scalar_values_for_indices(
        scalar_name=scalar_name,
        dataset=dataset,
        sample_indices=projection_indices,
        labels=labels,
        coords=np.asarray(coords, dtype=np.float32),
        frame_index_lookup=frame_index_lookup,
        frame_name_lookup=frame_name_lookup,
        frame_time_lookup=frame_time_lookup,
        point_scale=float(point_scale),
        descriptor_table=descriptor_table,
        descriptors_cfg_root=getattr(descriptor_cfg, "optional", None),
    )
    if projection_scalar is not None:
        scalar_plot = projection_dir / f"latent_projection_{projection_method}_{scalar_name}.png"
        scalar_plot_summary = save_embedding_continuous_plot(
            projection_embedding,
            projection_scalar,
            scalar_plot,
            title=f"Latent projection ({projection_info['method']}) colored by {scalar_label}",
            colorbar_label=scalar_label,
            alpha=_cfg_float(projection_cfg, "scalar_alpha", 0.86),
            background_alpha=_cfg_float(projection_cfg, "scalar_background_alpha", 0.06),
        )
        summary["latent_projection"]["plots"]["physical_scalar"] = str(scalar_plot)
        summary["latent_projection"]["physical_scalar"] = {
            "name": str(scalar_name),
            "label": str(scalar_label),
            "finite_fraction": float(scalar_plot_summary["finite_fraction"]),
        }

    if _cfg_bool(spatial_cfg, "enabled", True):
        spatial_dir = out_root / "spatial"
        spatial_dir.mkdir(parents=True, exist_ok=True)
        spatial_summary: dict[str, Any] = {
            "root_dir": str(spatial_dir),
            "frames": [],
            "zooms": [],
            "cluster_groups": cluster_groups,
        }
        spatial_max_points = getattr(spatial_cfg, "max_points", None)
        spatial_point_size = _cfg_float(
            spatial_cfg,
            "point_size",
            _cfg_float(figure_md_cfg, "point_size", 5.6),
        )
        spatial_alpha = _cfg_float(
            spatial_cfg,
            "alpha",
            _cfg_float(figure_md_cfg, "alpha", 0.62),
        )
        spatial_saturation = _cfg_float(
            spatial_cfg,
            "saturation_boost",
            _cfg_float(figure_md_cfg, "saturation_boost", 1.18),
        )
        spatial_elev = _cfg_float(
            spatial_cfg,
            "view_elev",
            _cfg_float(figure_md_cfg, "view_elev", 24.0),
        )
        spatial_azim = _cfg_float(
            spatial_cfg,
            "view_azim",
            _cfg_float(figure_md_cfg, "view_azim", 35.0),
        )
        for frame in frames:
            frame_dir = spatial_dir / frame.output_name
            frame_record: dict[str, Any] = {
                "frame_name": str(frame.source_name),
                "frame_label": str(frame.label),
                "filtered_views": [],
            }
            for group in cluster_groups:
                visible_cluster_ids = [int(v) for v in group["cluster_ids"]]
                if not np.any(np.isin(labels[frame.indices], np.asarray(visible_cluster_ids, dtype=int))):
                    continue
                filtered_view = save_spatial_cluster_view(
                    np.asarray(coords[frame.indices], dtype=np.float32),
                    labels[frame.indices],
                    frame_dir / f"filtered_{group['name']}.png",
                    title=f"MD clusters ({frame.label}, {group['name']})",
                    cluster_color_map=labels_color_map,
                    visible_cluster_ids=visible_cluster_ids,
                    max_points=None if spatial_max_points is None else int(spatial_max_points),
                    point_size=float(spatial_point_size),
                    alpha=float(spatial_alpha),
                    saturation_boost=float(spatial_saturation),
                    view_elev=float(spatial_elev),
                    view_azim=float(spatial_azim),
                )
                filtered_view["group_name"] = str(group["name"])
                frame_record["filtered_views"].append(filtered_view)
            spatial_summary["frames"].append(frame_record)

        zoom_specs = _build_zoom_specs(
            spatial_cfg,
            model_cfg.data,
            frames=frames,
            coords=np.asarray(coords, dtype=np.float32),
            labels=labels,
            cluster_groups=cluster_groups,
        )
        frame_lookup = {str(frame.source_name): frame for frame in frames}
        for zoom_spec in zoom_specs:
            target_frame = frame_lookup[str(zoom_spec["frame_name"])]
            zoom_out = spatial_dir / target_frame.output_name / f"zoom_{zoom_spec['name']}.png"
            zoom_view = save_spatial_cluster_view(
                np.asarray(coords[target_frame.indices], dtype=np.float32),
                labels[target_frame.indices],
                zoom_out,
                title=f"MD clusters ({target_frame.label}, zoom={zoom_spec['name']})",
                cluster_color_map=labels_color_map,
                visible_cluster_ids=zoom_spec["cluster_ids"],
                bounds=np.asarray(zoom_spec["bounds"], dtype=np.float32),
                max_points=None if spatial_max_points is None else int(spatial_max_points),
                point_size=float(spatial_point_size),
                alpha=float(spatial_alpha),
                saturation_boost=float(spatial_saturation),
                view_elev=float(spatial_elev),
                view_azim=float(spatial_azim),
            )
            zoom_view["name"] = str(zoom_spec["name"])
            zoom_view["source"] = str(zoom_spec["source"])
            spatial_summary["zooms"].append(zoom_view)
        summary["spatial"] = spatial_summary

    if _cfg_bool(transition_cfg, "enabled", True) and len(frames) >= 2:
        transition_dir = out_root / "transitions"
        transition_dir.mkdir(parents=True, exist_ok=True)
        stride = float(getattr(model_cfg.data, "radius", 8.0)) * (
            1.0 - float(getattr(model_cfg.data, "overlap_fraction", 0.0))
        )
        default_transition_tol = max(1e-3, 0.5 * stride)
        transition_tol_cfg = getattr(transition_cfg, "max_distance", None)
        transition_tol = float(default_transition_tol if transition_tol_cfg is None else transition_tol_cfg)
        transition_data = _compute_transitions(
            frames,
            coords=np.asarray(coords, dtype=np.float32),
            labels=labels,
            cluster_ids=cluster_ids,
            max_distance=float(transition_tol),
            require_mutual=_cfg_bool(transition_cfg, "require_mutual", True),
        )
        pair_records: list[dict[str, Any]] = []
        for pair_idx, pair_summary in enumerate(transition_data["pairs"], start=1):
            flow_path = transition_dir / f"transition_pair_{pair_idx:02d}_flow.png"
            flow_svg_path = transition_dir / f"transition_pair_{pair_idx:02d}_flow.svg"
            counts = np.asarray(pair_summary["counts"], dtype=np.float64)
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
            pair_records.append(
                {
                    "frame_from": str(pair_summary["frame_from"]),
                    "frame_to": str(pair_summary["frame_to"]),
                    "matched_samples": int(pair_summary["matched_samples"]),
                    "coverage_fraction": float(pair_summary["coverage_fraction"]),
                    "flow_plot": str(flow_path),
                    "flow_svg": str(flow_svg_path),
                }
            )

        aggregate_flow_path = transition_dir / "transition_aggregate_flow.png"
        aggregate_flow_svg_path = transition_dir / "transition_aggregate_flow.svg"
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
            "pairs": pair_records,
            "aggregate_flow": str(aggregate_flow_path),
            "aggregate_flow_svg": str(aggregate_flow_svg_path),
        }

    summary_json = out_root / "summary.json"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=_json_default)
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
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=_json_default)
    return summary


__all__ = ["run_real_md_qualitative_analysis"]
