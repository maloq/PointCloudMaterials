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
from sklearn.preprocessing import StandardScaler

from .cluster_profiles import (
    _ALL_PROFILE_PROPERTIES,
    _compute_sample_properties,
    _json_default,
    _load_point_cloud_from_dataset,
)
from .cluster_geometry import _sample_indices_stratified
from .cluster_rendering import (
    _save_cluster_representatives_figure,
)
from .cluster_figures import _build_cluster_color_map
from .output_layout import real_md_outputs_root
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
    temporal_projection_fit_indices: np.ndarray | None = None,
    point_scale: float,
    random_state: int,
    representative_render_cache: dict[str, Any] | None = None,
    selected_k_override: int | None = None,
    output_root_dir: Path | None = None,
) -> dict[str, Any]:
    data_kind = getattr(model_cfg.data, "kind", None)
    if data_kind not in {"real", "temporal_lammps"}:
        raise ValueError(
            "run_real_md_qualitative_analysis only supports model_cfg.data.kind in "
            "['real', 'temporal_lammps'], "
            f"got {data_kind!r}."
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
    if dataset is None:
        raise ValueError("A dataset object is required to render representative local environments.")

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
        if temporal_fit_indices.size == 0:
            raise ValueError(
                "temporal_projection_fit_indices resolved to an empty array, "
                "but temporal projection fitting was requested."
            )
        if np.any(temporal_fit_indices < 0) or np.any(temporal_fit_indices >= len(latents)):
            raise IndexError(
                "temporal_projection_fit_indices contains out-of-range sample indices. "
                f"min={int(np.min(temporal_fit_indices))}, max={int(np.max(temporal_fit_indices))}, "
                f"num_samples={len(latents)}."
            )
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

        temporal_md_cfg = getattr(temporal_cfg, "md_space", None)
        if _cfg_bool(temporal_md_cfg, "enabled", True):
            temporal_md_records = [
                {
                    "frame_name": str(frame.source_name),
                    "frame_label": str(frame.label),
                    "coords": np.asarray(
                        coords[temporal_animation_indices[str(frame.source_name)]],
                        dtype=np.float32,
                    ),
                    "labels": np.asarray(
                        labels[temporal_animation_indices[str(frame.source_name)]],
                        dtype=int,
                    ),
                }
                for frame in temporal_frames
            ]
            md_animation_path = temporal_dir / f"md_space_clusters_diagonal_cut_k{int(selected_k)}.gif"
            temporal_summary["md_space_animation"] = save_temporal_spatial_cluster_animation(
                temporal_md_records,
                md_animation_path,
                cluster_color_map=labels_color_map,
                cluster_display_map=cluster_display_map,
                point_size=_cfg_float(
                    temporal_md_cfg,
                    "point_size",
                    _cfg_float(figure_md_cfg, "point_size", 5.6),
                ),
                alpha=_cfg_float(
                    temporal_md_cfg,
                    "alpha",
                    _cfg_float(figure_md_cfg, "alpha", 0.62),
                ),
                saturation_boost=_cfg_float(
                    temporal_md_cfg,
                    "saturation_boost",
                    _cfg_float(figure_md_cfg, "saturation_boost", 1.18),
                ),
                view_elev=_cfg_float(
                    temporal_md_cfg,
                    "view_elev",
                    _cfg_float(figure_md_cfg, "view_elev", 24.0),
                ),
                view_azim=_cfg_float(
                    temporal_md_cfg,
                    "view_azim",
                    _cfg_float(figure_md_cfg, "view_azim", 35.0),
                ),
                diagonal_visible_depth_fraction=_cfg_float(
                    temporal_md_cfg,
                    "diagonal_visible_depth_fraction",
                    0.10,
                ),
                frame_duration_ms=_cfg_int(temporal_cfg, "frame_duration_ms", 450),
            )

        if _cfg_bool(transition_cfg, "enabled", True):
            if transition_tol is None:
                raise RuntimeError("transition_tol was not resolved for temporal transition animation.")
            temporal_transition_dir = temporal_dir / "transition_pairs"
            temporal_transition_dir.mkdir(parents=True, exist_ok=True)
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
            for pair_idx, pair_summary in enumerate(temporal_transition_data["pairs"], start=1):
                frame_from = str(pair_summary["frame_from"])
                frame_to = str(pair_summary["frame_to"])
                frame_from_label = temporal_frame_label_by_name.get(frame_from, frame_from)
                frame_to_label = temporal_frame_label_by_name.get(frame_to, frame_to)
                flow_path = temporal_transition_dir / f"transition_pair_{pair_idx:02d}_flow.png"
                save_transition_flow_plot(
                    np.asarray(pair_summary["counts"], dtype=np.float64),
                    flow_path,
                    title=f"Cluster transition flow | {frame_from_label} -> {frame_to_label}",
                    row_labels=cluster_display_labels,
                    cluster_color_map=labels_color_map,
                    cluster_ids_for_palette=cluster_ids,
                    mute_diagonal=_cfg_bool(transition_cfg, "flow_mute_diagonal", True),
                    min_draw_fraction=_cfg_float(transition_cfg, "flow_min_fraction", 0.001),
                )
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
                temporal_pair_plots.append(
                    {
                        "frame_from": frame_from,
                        "frame_to": frame_to,
                        "frame_from_label": frame_from_label,
                        "frame_to_label": frame_to_label,
                        "matched_samples": int(pair_summary["matched_samples"]),
                        "coverage_fraction": float(pair_summary["coverage_fraction"]),
                        "flow_plot": str(flow_path),
                    }
                )
            if temporal_pair_records:
                transition_animation_path = temporal_dir / "transition_pair_flows.gif"
                temporal_summary["transition_pair_flow_animation"] = save_temporal_transition_flow_animation(
                    temporal_pair_records,
                    transition_animation_path,
                    row_labels=cluster_display_labels,
                    cluster_color_map=labels_color_map,
                    cluster_ids_for_palette=cluster_ids,
                    mute_diagonal=_cfg_bool(transition_cfg, "flow_mute_diagonal", True),
                    min_draw_fraction=_cfg_float(transition_cfg, "flow_min_fraction", 0.001),
                    frame_duration_ms=_cfg_int(temporal_cfg, "frame_duration_ms", 450),
                    title="Cluster transition flow",
                )
            temporal_summary["transition_pairs"] = {
                "root_dir": str(temporal_transition_dir),
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
            all_animation_indices = np.concatenate(
                [
                    temporal_animation_indices[str(frame.source_name)]
                    for frame in temporal_frames
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
            for frame in temporal_frames:
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
            temporal_summary["umap_animation"] = save_temporal_embedding_cluster_animation(
                temporal_embedding_records,
                umap_animation_path,
                cluster_color_map=labels_color_map,
                cluster_display_map=cluster_display_map,
                point_size=_cfg_float(temporal_umap_cfg, "point_size", 8.0),
                alpha=_cfg_float(temporal_umap_cfg, "alpha", 0.74),
                frame_duration_ms=_cfg_int(temporal_cfg, "frame_duration_ms", 450),
                title=f"{projection_method.upper()} cluster evolution",
            )
            if _cfg_bool(temporal_umap_cfg, "trajectory_enabled", True):
                trajectory_max_points_raw = getattr(temporal_umap_cfg, "trajectory_max_points", animation_max_points)
                trajectory_max_points = (
                    None
                    if trajectory_max_points_raw is None or int(trajectory_max_points_raw) <= 0
                    else int(trajectory_max_points_raw)
                )
                trajectory_indices_by_frame, trajectory_instance_ids = _select_temporal_trajectory_sample_indices(
                    temporal_frames,
                    labels,
                    instance_ids_arr,
                    max_points=trajectory_max_points,
                    random_state=int(random_state) + 31,
                )
                trajectory_concat = np.concatenate(
                    [
                        trajectory_indices_by_frame[str(frame.source_name)]
                        for frame in temporal_frames
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
                for frame in temporal_frames:
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
                temporal_summary["umap_trajectory_animation"] = save_temporal_embedding_trajectory_animation(
                    temporal_trajectory_records,
                    trajectory_animation_path,
                    cluster_color_map=labels_color_map,
                    cluster_display_map=cluster_display_map,
                    line_width=_cfg_float(temporal_umap_cfg, "trajectory_line_width", 0.8),
                    line_alpha=_cfg_float(temporal_umap_cfg, "trajectory_line_alpha", 0.22),
                    history_steps=_cfg_int(
                        temporal_umap_cfg,
                        "trajectory_history_steps",
                        8,
                    ),
                    fade_min_alpha_fraction=_cfg_float(
                        temporal_umap_cfg,
                        "trajectory_fade_min_alpha_fraction",
                        0.18,
                    ),
                    fade_power=_cfg_float(
                        temporal_umap_cfg,
                        "trajectory_fade_power",
                        1.0,
                    ),
                    directional_subsegments=_cfg_int(
                        temporal_umap_cfg,
                        "trajectory_directional_subsegments",
                        6,
                    ),
                    directional_start_alpha_fraction=_cfg_float(
                        temporal_umap_cfg,
                        "trajectory_directional_start_alpha_fraction",
                        0.32,
                    ),
                    directional_start_width_fraction=_cfg_float(
                        temporal_umap_cfg,
                        "trajectory_directional_start_width_fraction",
                        0.60,
                    ),
                    directional_end_width_fraction=_cfg_float(
                        temporal_umap_cfg,
                        "trajectory_directional_end_width_fraction",
                        1.35,
                    ),
                    endpoint_point_size=_cfg_float(
                        temporal_umap_cfg,
                        "trajectory_endpoint_point_size",
                        3.0,
                    ),
                    endpoint_point_alpha=_cfg_float(
                        temporal_umap_cfg,
                        "trajectory_endpoint_point_alpha",
                        0.95,
                    ),
                    frame_duration_ms=_cfg_int(temporal_cfg, "frame_duration_ms", 450),
                    title=f"{projection_method.upper()} cluster trajectories",
                )
                temporal_summary["trajectory_sample_count"] = int(trajectory_instance_ids.size)
        temporal_summary["projection_fit_sample_count"] = (
            None if temporal_fit_indices is None else int(temporal_fit_indices.size)
        )
        summary["temporal"] = temporal_summary

    if _cfg_bool(transition_cfg, "enabled", True) and len(frames) >= 2:
        transition_dir = out_root / "transitions"
        transition_dir.mkdir(parents=True, exist_ok=True)
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
            "match_mode": str(transition_data["match_mode"]),
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
