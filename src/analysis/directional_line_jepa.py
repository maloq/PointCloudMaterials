from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.data_utils.line_static_dataset import LineStaticPointCloudDataset
from src.vis_tools.latent_analysis_vis import FittedClusteringModel, predict_clustering_model

from .output_layout import write_json

@dataclass(frozen=True)
class DirectionalLineJEPASettings:
    enabled: bool
    num_directions: int
    max_atoms_total: int
    task_batch_size: int
    selection_seed: int
    target_position: str
    relative_error_clip: float | None
    joint_weights: dict[str, float]
    atom_chunk_size: int = 128
    visualize_3d: bool = True


_DEFAULT_JOINT_WEIGHTS = {
    "mean_error": 1.0,
    "max_error": 1.0,
    "relative_residual_error": 1.0,
    "residual_norm": 0.5,
    "anisotropy": 0.5,
    "cluster_surprise": 0.5,
    "phase_conditioned_z": 0.5,
}


def resolve_directional_line_jepa_settings(
    analysis_cfg: DictConfig,
) -> DirectionalLineJEPASettings:
    raw = OmegaConf.select(analysis_cfg, "directional_line_jepa", default={})
    cfg = OmegaConf.to_container(raw, resolve=True) if OmegaConf.is_config(raw) else raw
    if not isinstance(cfg, dict):
        raise TypeError("directional_line_jepa must be a mapping.")

    raw_weights = cfg.get("joint_weights", {})
    if not isinstance(raw_weights, dict):
        raise TypeError("directional_line_jepa.joint_weights must be a mapping.")
    unknown = set(raw_weights) - set(_DEFAULT_JOINT_WEIGHTS)
    if unknown:
        raise ValueError(f"Unsupported joint hardness weights: {sorted(unknown)}.")
    weights = _DEFAULT_JOINT_WEIGHTS | {
        name: float(weight) for name, weight in raw_weights.items()
    }
    if min(weights.values()) < 0.0 or sum(weights.values()) == 0.0:
        raise ValueError(f"Joint hardness weights must be non-negative and non-zero: {weights}.")

    relative_clip = cfg.get("relative_error_clip", 100.0)
    relative_clip = None if relative_clip is None else float(relative_clip)
    target_position = str(cfg.get("target_position", "checkpoint")).strip().lower()
    if target_position not in {"checkpoint", "center", "endpoint"}:
        raise ValueError(
            f"target_position must be checkpoint, center, or endpoint; got {target_position!r}."
        )
    settings = DirectionalLineJEPASettings(
        enabled=bool(cfg.get("enabled", False)),
        num_directions=int(cfg.get("num_directions", 64)),
        max_atoms_total=int(cfg.get("max_atoms_total", 1024)),
        atom_chunk_size=int(cfg.get("atom_chunk_size", 128)),
        task_batch_size=int(cfg.get("task_batch_size", 256)),
        selection_seed=int(cfg.get("selection_seed", 123)),
        target_position=target_position,
        relative_error_clip=relative_clip,
        joint_weights=weights,
        visualize_3d=bool(cfg.get("visualize_3d", True)),
    )
    if (
        settings.num_directions <= 0
        or settings.atom_chunk_size <= 0
        or settings.task_batch_size <= 0
    ):
        raise ValueError(
            "num_directions, atom_chunk_size, and task_batch_size must be positive."
        )
    if settings.max_atoms_total < 0:
        raise ValueError("max_atoms_total must be non-negative (zero means all atoms).")
    if relative_clip is not None and relative_clip <= 0.0:
        raise ValueError("relative_error_clip must be positive or null.")
    return settings


def disable_directional_for_non_line_jepa(
    settings: DirectionalLineJEPASettings,
    *,
    model_type: Any,
) -> tuple[DirectionalLineJEPASettings, str | None]:
    normalized_model_type = str(model_type).strip().lower()
    if not settings.enabled or normalized_model_type == "line_jepa":
        return settings, None
    reason = (
        "directional_line_jepa requires a LineJEPAModule checkpoint; "
        f"resolved model_type={model_type!r}"
    )
    return replace(settings, enabled=False), reason


def apply_directional_runtime_limits(
    settings: DirectionalLineJEPASettings,
    *,
    enabled: bool | None,
    max_directions: int | None,
    max_atoms: int | None,
) -> DirectionalLineJEPASettings:
    """Apply runtime-profile overrides without increasing configured work."""
    directions = settings.num_directions
    atoms = settings.max_atoms_total
    if max_directions is not None:
        directions = min(directions, max_directions)
    if max_atoms is not None:
        atoms = max_atoms if atoms == 0 else min(atoms, max_atoms)
    return replace(
        settings,
        enabled=settings.enabled if enabled is None else enabled,
        num_directions=directions,
        max_atoms_total=atoms,
    )


def fibonacci_sphere_directions(count: int) -> np.ndarray:
    """Return deterministic equal-area directions as exact antipodal pairs."""
    count = int(count)
    if count <= 0 or count % 2 != 0:
        raise ValueError(
            "Directional Line-JEPA requires a positive even direction count so every ray has "
            f"an exact antipode, got {count}."
        )
    half_count = count // 2
    indices = np.arange(half_count, dtype=np.float64)
    z = (indices + 0.5) / float(half_count)
    radial = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    azimuth = golden_angle * indices
    hemisphere = np.stack(
        (radial * np.cos(azimuth), radial * np.sin(azimuth), z),
        axis=1,
    ).astype(np.float32)
    directions = np.concatenate((hemisphere, -hemisphere), axis=0)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions


def compute_directional_error_summaries(
    errors: np.ndarray,
    directions: np.ndarray,
) -> dict[str, np.ndarray]:
    """Reduce an (atom, direction) error field to scalar/vector/tensor summaries."""
    errors = np.asarray(errors, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    if errors.ndim != 2 or directions.shape != (errors.shape[1], 3):
        raise ValueError(
            f"Expected errors (N,K) and directions (K,3), got {errors.shape}, {directions.shape}."
        )
    if not np.isfinite(errors).all() or np.any(errors < 0.0):
        raise ValueError("Directional errors must be finite and non-negative.")
    norms = np.linalg.norm(directions, axis=1)
    if not np.isfinite(directions).all() or np.any(norms <= 0.0):
        raise ValueError("Directions must be finite and non-zero.")
    directions /= norms[:, None]

    max_indices = np.argmax(errors, axis=1)
    anisotropy_tensor = np.einsum(
        "nk,ka,kb->nab", errors, directions, directions, optimize=True,
    ) / float(errors.shape[1])
    eigenvalues = np.linalg.eigvalsh(anisotropy_tensor)
    numerator = 1.5 * np.sum((eigenvalues - eigenvalues.mean(1, keepdims=True)) ** 2, axis=1)
    denominator = np.sum(eigenvalues**2, axis=1)
    anisotropy = np.sqrt(
        np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    )
    summaries = {
        "mean_error": errors.mean(axis=1).astype(np.float32),
        "max_error": errors.max(axis=1).astype(np.float32),
        "std_error": errors.std(axis=1).astype(np.float32),
        "max_error_direction_index": max_indices.astype(np.int64),
        "max_error_direction": directions[max_indices].astype(np.float32),
        "anisotropy_tensor": anisotropy_tensor.astype(np.float32),
        "anisotropy_scalar": np.clip(anisotropy, 0, 1).astype(np.float32),
    }
    summaries.update(compute_directional_harmonic_summaries(errors, directions))
    summaries["novelty_percentile"] = _percentile_ranks(summaries["mean_error"])
    summaries["anisotropy_percentile"] = _percentile_ranks(
        summaries["relative_directional_variation"]
    )
    return summaries


def compute_directional_error_summaries_chunked(
    errors: np.ndarray,
    directions: np.ndarray,
    *,
    atom_chunk_size: int,
) -> dict[str, np.ndarray]:
    """Compute the same summaries without materializing full-size float64 work arrays."""
    errors = np.asarray(errors)
    if errors.ndim != 2:
        raise ValueError(f"Expected directional errors (N,K), got {errors.shape}.")
    if int(atom_chunk_size) <= 0:
        raise ValueError(f"atom_chunk_size must be positive, got {atom_chunk_size}.")
    assembled: dict[str, np.ndarray] = {}
    percentile_fields = {"novelty_percentile", "anisotropy_percentile"}
    for start in range(0, len(errors), int(atom_chunk_size)):
        stop = min(start + int(atom_chunk_size), len(errors))
        chunk = compute_directional_error_summaries(errors[start:stop], directions)
        for name, values in chunk.items():
            if name in percentile_fields:
                continue
            if name not in assembled:
                assembled[name] = np.empty(
                    (len(errors), *values.shape[1:]), dtype=values.dtype
                )
            assembled[name][start:stop] = values
    if not assembled:
        raise ValueError("Directional summary calculation received zero atoms.")
    assembled["novelty_percentile"] = _percentile_ranks(assembled["mean_error"])
    assembled["anisotropy_percentile"] = _percentile_ranks(
        assembled["relative_directional_variation"]
    )
    return assembled


def compute_directional_harmonic_summaries(
    errors: np.ndarray,
    directions: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit l=0, l=1 (polar), and l=2 (axial) components of each error field."""
    errors = np.asarray(errors, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    if errors.ndim != 2 or directions.shape != (errors.shape[1], 3):
        raise ValueError(
            "Directional harmonic fitting expects errors (N,K) and directions (K,3), "
            f"got {errors.shape} and {directions.shape}."
        )
    direction_norms = np.linalg.norm(directions, axis=1)
    if not np.isfinite(errors).all() or not np.isfinite(directions).all():
        raise ValueError("Directional harmonic fitting received non-finite inputs.")
    if np.any(direction_norms <= 0.0):
        raise ValueError("Directional harmonic fitting received a zero-length direction.")
    directions = directions / direction_norms[:, None]
    x, y, z = directions.T
    quadratic_design = np.column_stack(
        (x * x - z * z, y * y - z * z, 2.0 * x * y, 2.0 * x * z, 2.0 * y * z)
    )
    design = np.column_stack((np.ones(len(directions)), directions, quadratic_design))
    coefficients = errors @ np.linalg.pinv(design).T

    polar_vector = coefficients[:, 1:4]
    quadratic_coefficients = coefficients[:, 4:9]
    axial_tensor = np.zeros((len(errors), 3, 3), dtype=np.float64)
    axial_tensor[:, 0, 0] = quadratic_coefficients[:, 0]
    axial_tensor[:, 1, 1] = quadratic_coefficients[:, 1]
    axial_tensor[:, 2, 2] = -quadratic_coefficients[:, 0] - quadratic_coefficients[:, 1]
    axial_tensor[:, 0, 1] = axial_tensor[:, 1, 0] = quadratic_coefficients[:, 2]
    axial_tensor[:, 0, 2] = axial_tensor[:, 2, 0] = quadratic_coefficients[:, 3]
    axial_tensor[:, 1, 2] = axial_tensor[:, 2, 1] = quadratic_coefficients[:, 4]

    polar_response = polar_vector @ directions.T
    axial_response = quadratic_coefficients @ quadratic_design.T
    fitted = coefficients @ design.T
    higher_order_response = errors - fitted
    polar_strength = np.sqrt(np.mean(polar_response**2, axis=1))
    axial_strength = np.sqrt(np.mean(axial_response**2, axis=1))
    complex_strength = np.sqrt(np.mean(higher_order_response**2, axis=1))
    mean_error = errors.mean(axis=1)
    scale = np.maximum(mean_error, 1.0e-12)
    centered_power = np.mean((errors - mean_error[:, None]) ** 2, axis=1)
    residual_power = np.mean(higher_order_response**2, axis=1)
    explained = np.ones_like(centered_power)
    nonconstant = centered_power > 1.0e-15
    explained[nonconstant] = 1.0 - residual_power[nonconstant] / centered_power[nonconstant]

    polar_norm = np.linalg.norm(polar_vector, axis=1)
    polar_direction = np.divide(
        polar_vector,
        polar_norm[:, None],
        out=np.zeros_like(polar_vector),
        where=polar_norm[:, None] > 1.0e-12,
    )
    axial_eigenvalues, axial_eigenvectors = np.linalg.eigh(axial_tensor)
    axial_indices = np.argmax(np.abs(axial_eigenvalues), axis=1)
    axial_direction = np.take_along_axis(
        axial_eigenvectors, axial_indices[:, None, None], axis=2
    ).squeeze(2)
    largest_component = np.argmax(np.abs(axial_direction), axis=1)
    orientation = np.sign(axial_direction[np.arange(len(errors)), largest_component])
    orientation[orientation == 0.0] = 1.0
    axial_direction *= orientation[:, None]

    polar_power = polar_strength**2
    axial_power = axial_strength**2
    low_order_power = polar_power + axial_power
    polar_share = np.divide(
        polar_power,
        low_order_power,
        out=np.full_like(polar_power, 0.5),
        where=low_order_power > 1.0e-15,
    )
    return {
        "harmonic_isotropic_error": coefficients[:, 0].astype(np.float32),
        "polar_vector": polar_vector.astype(np.float32),
        "polar_direction": polar_direction.astype(np.float32),
        "polar_anisotropy": (polar_strength / scale).astype(np.float32),
        "axial_tensor": axial_tensor.astype(np.float32),
        "axial_direction": axial_direction.astype(np.float32),
        "axial_anisotropy": (axial_strength / scale).astype(np.float32),
        "complex_anisotropy": (complex_strength / scale).astype(np.float32),
        "relative_directional_variation": (
            np.sqrt(centered_power) / scale
        ).astype(np.float32),
        "harmonic_explained_fraction": np.clip(explained, 0.0, 1.0).astype(np.float32),
        "polar_share": np.clip(polar_share, 0.0, 1.0).astype(np.float32),
    }


def phase_conditioned_z_scores(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if values.ndim != 1 or labels.ndim != 1:
        raise ValueError(
            "Phase-conditioned z-scores require one-dimensional inputs. "
            f"Got values={values.shape}, labels={labels.shape}."
        )
    if values.shape != labels.shape:
        raise ValueError(
            "Phase-conditioned z-scores require one label per value. "
            f"Got values={tuple(values.shape)}, labels={tuple(labels.shape)}."
        )
    z_scores = np.zeros(values.shape, dtype=np.float64)
    for label in np.unique(labels):
        mask = labels == int(label)
        group_values = values[mask]
        group_std = float(group_values.std())
        if group_std > 0.0:
            z_scores[mask] = (group_values - float(group_values.mean())) / group_std
    return z_scores.astype(np.float32)


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError(
            f"Percentile ranks require a one-dimensional array, got {values.shape}."
        )
    if values.size <= 1:
        return np.zeros(values.shape, dtype=np.float32)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * float(start + stop - 1)
        start = stop
    return (ranks / float(values.size - 1)).astype(np.float32)


def compute_joint_hardness_score(
    components: dict[str, np.ndarray],
    *,
    weights: dict[str, float],
) -> np.ndarray:
    active = [(name, float(weight)) for name, weight in weights.items()
              if weight > 0 and name in components]
    if not active:
        raise ValueError("Joint hardness has no active components.")
    ranks = np.stack([_percentile_ranks(components[name]) for name, _ in active])
    return np.average(ranks, axis=0, weights=[weight for _, weight in active]).astype(np.float32)


def _selected_analysis_indices(
    *,
    source_groups: list[tuple[str, np.ndarray]],
    sample_count: int,
    max_atoms_total: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not source_groups:
        raise ValueError("Directional Line-JEPA requires per-sample source groups.")
    names = np.full((sample_count,), "", dtype=object)
    covered = np.zeros((sample_count,), dtype=bool)
    groups = []
    for source_name, indices in source_groups:
        if indices.ndim != 1:
            raise ValueError(
                f"Source {source_name!r} sample indices must be one-dimensional, "
                f"got {indices.shape}."
            )
        if np.any(indices < 0) or np.any(indices >= sample_count):
            raise IndexError(f"Source {source_name!r} contains out-of-range sample indices.")
        names[indices] = str(source_name)
        covered[indices] = True
        groups.append(indices)
    if not np.all(covered):
        raise RuntimeError(
            f"Source groups miss cache row {int(np.flatnonzero(~covered)[0])}."
        )

    budget = sample_count if max_atoms_total == 0 else min(int(max_atoms_total), sample_count)
    if budget == sample_count:
        selected = np.arange(sample_count, dtype=np.int64)
        return selected, names.astype(str)

    rng = np.random.default_rng(int(seed))
    selected_parts: list[np.ndarray] = []
    remaining = budget
    for group_index, indices in enumerate(groups):
        quota = min(len(indices), int(np.ceil(remaining / (len(groups) - group_index))))
        selected_parts.append(rng.choice(indices, quota, replace=False))
        remaining -= quota
    if remaining:
        already = np.concatenate(selected_parts)
        pool = np.setdiff1d(np.arange(sample_count), already, assume_unique=False)
        selected_parts.append(rng.choice(pool, remaining, replace=False))
    return np.sort(np.concatenate(selected_parts)).astype(np.int64), names.astype(str)


def _to_plain_container(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _build_static_line_dataset(model_cfg: DictConfig) -> LineStaticPointCloudDataset:
    data_cfg = model_cfg.data
    data_sources = _to_plain_container(getattr(data_cfg, "data_sources", None))
    data_files = _to_plain_container(getattr(data_cfg, "data_files", None))
    source_kwargs: dict[str, Any]
    if data_sources:
        source_kwargs = {"data_sources": data_sources}
    elif data_files:
        if isinstance(data_files, str):
            data_files = [data_files]
        source_kwargs = {
            "root": str(getattr(data_cfg, "data_path", "")),
            "data_files": [str(value) for value in data_files],
        }
    else:
        raise ValueError(
            "Directional static Line-JEPA analysis requires cfg.data.data_sources or "
            "cfg.data.data_files + cfg.data.data_path."
        )
    return LineStaticPointCloudDataset(
        **source_kwargs,
        radius=float(getattr(data_cfg, "radius")),
        num_points=int(getattr(data_cfg, "num_points")),
        line_atoms=int(getattr(data_cfg, "line_atoms")),
        line_candidate_atoms=int(getattr(data_cfg, "line_candidate_atoms")),
        line_samples_per_file=1,
        normalize=bool(getattr(data_cfg, "normalize", True)),
        center_neighborhoods=bool(getattr(data_cfg, "center_neighborhoods", True)),
        drop_edge_samples=bool(getattr(data_cfg, "drop_edge_samples", True)),
        edge_drop_layers=getattr(data_cfg, "edge_drop_layers", None),
        line_selection_method=str(getattr(data_cfg, "line_selection_method", "closest")),
        line_min_separation_radius_factor=float(
            getattr(data_cfg, "line_min_separation_radius_factor", 0.0)
        ),
        line_slot_spacing_radius_factor=getattr(
            data_cfg, "line_slot_spacing_radius_factor", None
        ),
        line_fixed_slot_max_deviation_radius_factor=getattr(
            data_cfg, "line_fixed_slot_max_deviation_radius_factor", None
        ),
        line_seed=int(getattr(data_cfg, "line_seed", 0)),
        deterministic_lines=True,
        auto_cutoff_config=_to_plain_container(getattr(data_cfg, "auto_cutoff", None)),
    )


def _valid_directional_line_anchor_mask(
    line_dataset: LineStaticPointCloudDataset,
    *,
    source_slots: np.ndarray,
    atom_ids: np.ndarray,
) -> np.ndarray:
    """Keep anchors with the boundary clearance used by Line-JEPA training."""
    if source_slots.ndim != 1 or atom_ids.ndim != 1 or source_slots.shape != atom_ids.shape:
        raise ValueError(
            "Directional Line-JEPA anchor sources and atom IDs must be parallel "
            "one-dimensional arrays. "
            f"Got source_slots={source_slots.shape}, atom_ids={atom_ids.shape}."
        )

    valid = np.zeros(source_slots.shape, dtype=bool)
    for source_slot_raw in np.unique(source_slots):
        source_slot = int(source_slot_raw)
        if not 0 <= source_slot < len(line_dataset.sources):
            raise IndexError(
                "Directional Line-JEPA anchor has an invalid static source slot. "
                f"source_slot={source_slot}, source_count={len(line_dataset.sources)}."
            )
        rows = np.flatnonzero(source_slots == source_slot)
        allowed_atom_ids = line_dataset.sources[source_slot].center_indices
        insertion_indices = np.searchsorted(allowed_atom_ids, atom_ids[rows])
        within_bounds = insertion_indices < len(allowed_atom_ids)
        matched = np.zeros(rows.shape, dtype=bool)
        matched[within_bounds] = (
            allowed_atom_ids[insertion_indices[within_bounds]] == atom_ids[rows][within_bounds]
        )
        valid[rows] = matched
    return valid


def _predict_cluster_labels(
    latents: np.ndarray,
    fitted_model: FittedClusteringModel,
) -> np.ndarray:
    labels = predict_clustering_model(latents, fitted_model)
    if labels.ndim != 1:
        raise RuntimeError(
            "Clustering prediction returned an unexpected label shape. "
            f"latents={latents.shape}, labels={labels.shape}."
        )
    return labels


def _encode_environment_cache(
    model: Any,
    dataset: LineStaticPointCloudDataset,
    *,
    source_slots: np.ndarray,
    atom_ids: np.ndarray,
    batch_size: int,
    target_encoder: bool,
) -> np.ndarray:
    """Encode each unique environment once and keep the cache in host memory."""
    feature_cache: np.ndarray | None = None
    for start in range(0, len(atom_ids), batch_size):
        stop = min(start + batch_size, len(atom_ids))
        points = dataset.build_atom_environment_batch(
            source_slots=source_slots[start:stop], atom_ids=atom_ids[start:stop]
        )
        features = model.encode_directional_environment_batch(
            points, target_encoder=target_encoder
        ).cpu().numpy()
        if feature_cache is None:
            feature_cache = np.empty((len(atom_ids), features.shape[1]), dtype=np.float32)
        feature_cache[start:stop] = features
    if feature_cache is None:
        raise RuntimeError("Directional environment cache received no atoms.")
    return feature_cache


def _quantiles(values: np.ndarray) -> dict[str, float]:
    result = {
        name: float(np.quantile(values, quantile))
        for name, quantile in (("min", 0), ("p50", .5), ("p90", .9),
                               ("p95", .95), ("p99", .99), ("max", 1))
    }
    return result | {"mean": float(values.mean()), "std": float(values.std())}


def _write_atom_table(path: Path, arrays: dict[str, np.ndarray]) -> None:
    source_slots = arrays["source_slots"]
    source_names = arrays["source_names_by_slot"][source_slots]
    source_paths = arrays["source_paths_by_slot"][source_slots]
    columns = {
        "analysis_sample_index": arrays["analysis_sample_indices"],
        "source_name": source_names,
        "source_path": source_paths,
        "atom_id": arrays["atom_ids"],
        "x": arrays["coords"][:, 0], "y": arrays["coords"][:, 1],
        "z": arrays["coords"][:, 2],
        "center_snap_distance": arrays["center_snap_distance"],
        "phase_or_cluster_label": arrays["phase_or_cluster_label"],
        "mean_prediction_cosine_error": arrays["mean_prediction_cosine_error"],
        "max_prediction_cosine_error": arrays["max_prediction_cosine_error"],
        "std_prediction_cosine_error": arrays["std_prediction_cosine_error"],
        "max_error_direction_x": arrays["max_error_direction"][:, 0],
        "max_error_direction_y": arrays["max_error_direction"][:, 1],
        "max_error_direction_z": arrays["max_error_direction"][:, 2],
        "anisotropy_scalar": arrays["anisotropy_scalar"],
        "novelty_percentile": arrays["novelty_percentile"],
        "relative_directional_variation": arrays["relative_directional_variation"],
        "anisotropy_percentile": arrays["anisotropy_percentile"],
        "polar_anisotropy": arrays["polar_anisotropy"],
        "polar_share": arrays["polar_share"],
        "polar_direction_x": arrays["polar_direction"][:, 0],
        "polar_direction_y": arrays["polar_direction"][:, 1],
        "polar_direction_z": arrays["polar_direction"][:, 2],
        "axial_anisotropy": arrays["axial_anisotropy"],
        "axial_direction_x": arrays["axial_direction"][:, 0],
        "axial_direction_y": arrays["axial_direction"][:, 1],
        "axial_direction_z": arrays["axial_direction"][:, 2],
        "complex_anisotropy": arrays["complex_anisotropy"],
        "harmonic_explained_fraction": arrays["harmonic_explained_fraction"],
        "phase_conditioned_z_score": arrays["phase_conditioned_z_score"],
        "mean_reconstruction_cosine_error": arrays["mean_reconstruction_cosine_error"],
        "mean_residual_norm": arrays["mean_residual_norm"],
        "mean_relative_residual_error": arrays["mean_relative_residual_error"],
        "mean_cluster_surprise": arrays["mean_cluster_surprise"],
        "mean_context_cluster_disagreement": arrays["mean_context_cluster_disagreement"],
        "joint_hardness_score": arrays["joint_hardness_score"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        column_values = tuple(columns.values())
        for row_index in range(len(source_slots)):
            writer.writerow(
                [
                    value.item() if isinstance(value, np.generic) else value
                    for value in (column[row_index] for column in column_values)
                ]
            )


def run_directional_line_jepa_analysis(
    *,
    model: Any,
    model_cfg: DictConfig,
    analysis_cfg: DictConfig,
    cache: dict[str, np.ndarray],
    source_groups: list[tuple[str, np.ndarray]],
    fitted_clustering_model: FittedClusteringModel | None,
    primary_k: int | None,
    out_dir: Path,
    step: Callable[[str], None],
    settings: DirectionalLineJEPASettings | None = None,
) -> dict[str, Any]:
    settings = settings or resolve_directional_line_jepa_settings(analysis_cfg)
    if not settings.enabled:
        return {}
    from src.training_methods.line_jepa.line_jepa_module import LineJEPAModule

    if not isinstance(model, LineJEPAModule):
        raise TypeError(
            "directional_line_jepa.enabled=true requires a LineJEPAModule checkpoint, "
            f"got model={type(model)!r}, model_type={getattr(model_cfg, 'model_type', None)!r}."
        )
    data_kind = str(model_cfg.data.kind).strip().lower()
    if data_kind != "line_static":
        raise NotImplementedError(
            "Directional Line-JEPA requires the current line_static dataset contract. "
            f"Got normalized data kind {data_kind!r}."
        )
    if fitted_clustering_model is None or primary_k is None:
        raise ValueError(
            "Directional Line-JEPA requires the fitted primary clustering model produced by "
            "the analysis pipeline. "
            f"fitted_clustering_model={fitted_clustering_model is not None}, "
            f"primary_k={primary_k!r}."
        )

    sample_count = int(cache["inv_latents"].shape[0])
    coords_all = cache["coords"]
    selected_indices, source_name_by_sample = _selected_analysis_indices(
        source_groups=source_groups,
        sample_count=sample_count,
        max_atoms_total=settings.max_atoms_total,
        seed=settings.selection_seed,
    )
    selected_source_names = source_name_by_sample[selected_indices]
    selected_coords = coords_all[selected_indices]

    step(
        "Building explicit directional Line-JEPA contexts "
        f"({len(selected_indices)} atoms x {settings.num_directions} directions)"
    )
    line_dataset = _build_static_line_dataset(model_cfg)
    source_slots, atom_ids, snap_distances = line_dataset.resolve_explicit_centers(
        source_names=selected_source_names.tolist(),
        center_coords=selected_coords,
    )
    requested_atom_count = int(len(selected_indices))
    context_analysis_sample_indices = selected_indices.copy()
    context_source_slots = source_slots.copy()
    context_atom_ids = atom_ids.copy()
    context_coords = np.asarray(
        [
            line_dataset.sources[int(slot)].points[int(atom_id)]
            for slot, atom_id in zip(source_slots, atom_ids, strict=True)
        ],
        dtype=np.float32,
    )
    source_names_by_slot = np.asarray(
        [source.group_name for source in line_dataset.sources], dtype=str
    )
    source_paths_by_slot = np.asarray(
        [source.path for source in line_dataset.sources], dtype=str
    )
    valid_anchor_mask = _valid_directional_line_anchor_mask(
        line_dataset,
        source_slots=source_slots,
        atom_ids=atom_ids,
    )
    excluded_anchor_count = int((~valid_anchor_mask).sum())
    if excluded_anchor_count:
        excluded_by_source = {
            str(source_name): int(
                np.sum((selected_source_names == source_name) & ~valid_anchor_mask)
            )
            for source_name in np.unique(selected_source_names)
        }
        print(
            "[analysis][directional-line-jepa] Excluding "
            f"{excluded_anchor_count}/{requested_atom_count} cache centers outside the "
            "checkpoint's static line-anchor domain. This preserves the boundary clearance "
            "required for full directional rays. "
            f"excluded_by_source={excluded_by_source}",
            flush=True,
        )
        selected_indices = selected_indices[valid_anchor_mask]
        selected_source_names = selected_source_names[valid_anchor_mask]
        selected_coords = selected_coords[valid_anchor_mask]
        source_slots = source_slots[valid_anchor_mask]
        atom_ids = atom_ids[valid_anchor_mask]
        snap_distances = snap_distances[valid_anchor_mask]
    if not len(selected_indices):
        raise RuntimeError(
            "Directional Line-JEPA found no cache centers inside the checkpoint's static "
            "line-anchor domain. Increase the static analysis edge clearance or use input "
            "frames with an interior region large enough for the configured line geometry."
        )
    atom_coords = context_coords[valid_anchor_mask]
    directions = fibonacci_sphere_directions(settings.num_directions)
    if settings.target_position == "checkpoint":
        if model.prediction_positions == "endpoints":
            resolved_target_position = "endpoint"
        elif model.prediction_positions in {"center", "all", "cycle"}:
            resolved_target_position = "center"
        else:
            raise RuntimeError(
                "Cannot resolve directional target position from checkpoint setting "
                f"line_jepa_prediction_positions={model.prediction_positions!r}."
            )
    else:
        resolved_target_position = settings.target_position
    target_index = 0 if resolved_target_position == "endpoint" else model.target_line_index
    atom_count = int(len(selected_indices))
    direction_count = int(directions.shape[0])
    task_count = atom_count * direction_count
    prediction_cosine_error = np.empty(
        (atom_count, direction_count), dtype=np.float32
    )
    auxiliary_mean_fields = {
        "mean_residual_prediction_error": np.empty((atom_count,), dtype=np.float32),
        "mean_residual_norm": np.empty((atom_count,), dtype=np.float32),
        "mean_relative_residual_error": np.empty((atom_count,), dtype=np.float32),
        "mean_cluster_surprise": np.empty((atom_count,), dtype=np.float32),
        "mean_context_cluster_disagreement": np.empty((atom_count,), dtype=np.float32),
    }
    target_cluster_by_atom = np.empty((atom_count,), dtype=np.int64)
    source_stride = max(len(source.points) for source in line_dataset.sources)
    unique_environment_count = 0
    target_encoding_count = 0

    step(
        "Evaluating directional Line-JEPA predictions in bounded atom chunks "
        f"(atom_chunk_size={settings.atom_chunk_size})"
    )
    model.eval()
    with torch.inference_mode():
        context_mask = np.arange(model.line_atoms) != target_index
        environment_batch_size = settings.task_batch_size * model.line_atoms
        candidate_row_bytes = line_dataset.line_candidate_atoms * np.dtype(np.int64).itemsize
        geometry_batch_size = max(1, (64 * 1024 * 1024) // candidate_row_bytes)

        for atom_start in range(0, atom_count, settings.atom_chunk_size):
            atom_stop = min(atom_start + settings.atom_chunk_size, atom_count)
            chunk_size = atom_stop - atom_start
            chunk_task_count = chunk_size * direction_count
            chunk_atom_rows = np.repeat(
                np.arange(chunk_size, dtype=np.int64), direction_count
            )
            chunk_task_indices = np.arange(chunk_task_count, dtype=np.int64)
            chunk_source_slots = source_slots[atom_start:atom_stop]
            chunk_atom_ids = atom_ids[atom_start:atom_stop]
            candidate_indices = line_dataset.query_explicit_line_candidates(
                source_slots=chunk_source_slots,
                center_atom_ids=chunk_atom_ids,
            )
            line_atom_ids = np.empty(
                (chunk_task_count, model.line_atoms), dtype=np.int64
            )
            line_t = np.empty((chunk_task_count, model.line_atoms), dtype=np.float32)
            line_perp = np.empty((chunk_task_count, model.line_atoms), dtype=np.float32)

            for geometry_start in range(0, chunk_task_count, geometry_batch_size):
                geometry_stop = min(
                    geometry_start + geometry_batch_size, chunk_task_count
                )
                geometry_atoms = chunk_atom_rows[geometry_start:geometry_stop]
                batch = line_dataset.build_explicit_direction_batch(
                    source_slots=chunk_source_slots[geometry_atoms],
                    center_atom_ids=chunk_atom_ids[geometry_atoms],
                    directions=directions[
                        chunk_task_indices[geometry_start:geometry_stop] % direction_count
                    ],
                    target_index=target_index,
                    candidate_indices=candidate_indices[geometry_atoms],
                    materialize_points=False,
                )
                line_atom_ids[geometry_start:geometry_stop] = batch[
                    "line_atom_ids"
                ].numpy()
                line_t[geometry_start:geometry_stop] = batch["line_t"].numpy()
                line_perp[geometry_start:geometry_stop] = batch["line_perp"].numpy()

            flat_source_slots = np.repeat(
                chunk_source_slots[chunk_atom_rows], model.line_atoms
            )
            environment_keys = (
                flat_source_slots * source_stride + line_atom_ids.reshape(-1)
            )
            unique_keys, inverse = np.unique(environment_keys, return_inverse=True)
            line_feature_indices = inverse.reshape(
                chunk_task_count, model.line_atoms
            ).astype(np.int32, copy=False)
            unique_source_slots = (unique_keys // source_stride).astype(np.int64)
            unique_atom_ids = (unique_keys % source_stride).astype(np.int64)
            unique_environment_count += int(len(unique_atom_ids))

            online_feature_cache = _encode_environment_cache(
                model,
                line_dataset,
                source_slots=unique_source_slots,
                atom_ids=unique_atom_ids,
                batch_size=environment_batch_size,
                target_encoder=False,
            )
            target_cache_indices = line_feature_indices[::direction_count, target_index]
            unique_target_indices, target_inverse = np.unique(
                target_cache_indices, return_inverse=True
            )
            if model.target_encoder_mode in {"ema", "frozen"}:
                unique_target_features = _encode_environment_cache(
                    model,
                    line_dataset,
                    source_slots=unique_source_slots[unique_target_indices],
                    atom_ids=unique_atom_ids[unique_target_indices],
                    batch_size=environment_batch_size,
                    target_encoder=True,
                )
                target_feature_cache = unique_target_features[target_inverse]
                target_encoding_count += int(len(unique_target_indices))
            else:
                target_feature_cache = online_feature_cache[target_cache_indices]

            unique_cluster_labels = _predict_cluster_labels(
                online_feature_cache, fitted_clustering_model
            )
            target_embedding_labels = _predict_cluster_labels(
                target_feature_cache, fitted_clustering_model
            )

            chunk_fields = {
                "residual_prediction_error": np.empty(
                    (chunk_task_count,), dtype=np.float32
                ),
                "reconstruction_cosine_error": np.empty(
                    (chunk_task_count,), dtype=np.float32
                ),
                "residual_norm": np.empty((chunk_task_count,), dtype=np.float32),
                "relative_residual_error": np.empty(
                    (chunk_task_count,), dtype=np.float32
                ),
                "cluster_surprise": np.empty((chunk_task_count,), dtype=np.float32),
                "context_cluster_disagreement": np.empty(
                    (chunk_task_count,), dtype=np.float32
                ),
            }
            chunk_target_labels = np.empty((chunk_task_count,), dtype=np.int64)
            for task_start in range(0, chunk_task_count, settings.task_batch_size):
                task_stop = min(task_start + settings.task_batch_size, chunk_task_count)
                local_atom_rows = chunk_atom_rows[task_start:task_stop]
                feature_indices = line_feature_indices[task_start:task_stop]
                line_points_batch = None
                line_direction_batch = None
                if model.directional_feature_mode in {"moments", "encoder"}:
                    line_points_batch = line_dataset.build_atom_environment_batch(
                        source_slots=np.repeat(
                            chunk_source_slots[local_atom_rows], model.line_atoms
                        ),
                        atom_ids=line_atom_ids[task_start:task_stop].reshape(-1),
                    ).reshape(task_stop - task_start, model.line_atoms, -1, 3)
                    line_direction_batch = torch.from_numpy(
                        directions[
                            chunk_task_indices[task_start:task_stop] % direction_count
                        ]
                    )
                outputs = model.evaluate_directional_feature_batch(
                    line_features=torch.from_numpy(online_feature_cache[feature_indices]),
                    line_t=torch.from_numpy(line_t[task_start:task_stop]),
                    line_perp=torch.from_numpy(line_perp[task_start:task_stop]),
                    target_features=torch.from_numpy(target_feature_cache[local_atom_rows]),
                    target_index=target_index,
                    line_points=line_points_batch,
                    line_direction=line_direction_batch,
                )
                for name in (
                    "residual_prediction_error",
                    "reconstruction_cosine_error",
                    "residual_norm",
                    "relative_residual_error",
                ):
                    chunk_fields[name][task_start:task_stop] = (
                        outputs[name].detach().to("cpu", dtype=torch.float32).numpy()
                    )

                line_labels = unique_cluster_labels[feature_indices]
                target_labels = line_labels[:, target_index]
                reconstruction_labels = _predict_cluster_labels(
                    outputs["reconstruction"].detach().cpu().float().numpy(),
                    fitted_clustering_model,
                )
                chunk_target_labels[task_start:task_stop] = target_labels
                chunk_fields["cluster_surprise"][task_start:task_stop] = (
                    reconstruction_labels
                    != target_embedding_labels[local_atom_rows]
                )
                chunk_fields["context_cluster_disagreement"][task_start:task_stop] = (
                    np.mean(
                        line_labels[:, context_mask] != target_labels[:, None], axis=1
                    )
                )

            cosine_chunk = np.clip(
                chunk_fields["reconstruction_cosine_error"].reshape(
                    chunk_size, direction_count
                ),
                0.0,
                2.0,
            ).astype(np.float32, copy=False)
            prediction_cosine_error[atom_start:atom_stop] = cosine_chunk
            if settings.relative_error_clip is not None:
                chunk_fields["relative_residual_error"] = np.clip(
                    chunk_fields["relative_residual_error"],
                    0.0,
                    settings.relative_error_clip,
                ).astype(np.float32, copy=False)
            for source_name, target_name in (
                ("residual_prediction_error", "mean_residual_prediction_error"),
                ("residual_norm", "mean_residual_norm"),
                ("relative_residual_error", "mean_relative_residual_error"),
                ("cluster_surprise", "mean_cluster_surprise"),
                ("context_cluster_disagreement", "mean_context_cluster_disagreement"),
            ):
                auxiliary_mean_fields[target_name][atom_start:atom_stop] = chunk_fields[
                    source_name
                ].reshape(chunk_size, direction_count).mean(axis=1)
            target_cluster_by_atom[atom_start:atom_stop] = chunk_target_labels.reshape(
                chunk_size, direction_count
            )[:, 0]

            completed_tasks = atom_stop * direction_count
            if atom_stop == atom_count or completed_tasks % max(
                settings.task_batch_size * 20, 1
            ) == 0:
                print(
                    "[analysis][directional-line-jepa] "
                    f"atoms={atom_stop}/{atom_count}, tasks={completed_tasks}/{task_count}",
                    flush=True,
                )
            del (
                candidate_indices,
                line_atom_ids,
                line_t,
                line_perp,
                online_feature_cache,
                target_feature_cache,
                chunk_fields,
            )

    step("Computing directional summaries in bounded chunks")
    summaries = compute_directional_error_summaries_chunked(
        prediction_cosine_error,
        directions,
        atom_chunk_size=max(settings.atom_chunk_size, 8_192),
    )
    phase_or_cluster_label = target_cluster_by_atom
    conditioning_source = f"predicted_cluster_k{primary_k}"
    phase_z = phase_conditioned_z_scores(summaries["mean_error"], phase_or_cluster_label)

    mean_fields = {
        **auxiliary_mean_fields,
        "mean_reconstruction_cosine_error": summaries["mean_error"],
    }
    joint_components = {
        "mean_error": summaries["mean_error"],
        "max_error": summaries["max_error"],
        "relative_residual_error": mean_fields["mean_relative_residual_error"],
        "residual_norm": mean_fields["mean_residual_norm"],
        "anisotropy": summaries["anisotropy_scalar"],
        "phase_conditioned_z": phase_z,
    }
    joint_components["cluster_surprise"] = mean_fields["mean_cluster_surprise"]
    joint_hardness = compute_joint_hardness_score(
        joint_components,
        weights=settings.joint_weights,
    )

    arrays: dict[str, np.ndarray] = {
        "directions": directions,
        "prediction_cosine_error": prediction_cosine_error,
        "context_analysis_sample_indices": context_analysis_sample_indices,
        "context_source_slots": context_source_slots,
        "context_atom_ids": context_atom_ids,
        "context_coords": context_coords,
        "context_evaluated_mask": valid_anchor_mask,
        "analysis_sample_indices": selected_indices.astype(np.int64, copy=False),
        "source_names_by_slot": source_names_by_slot,
        "source_paths_by_slot": source_paths_by_slot,
        "source_slots": source_slots,
        "atom_ids": atom_ids,
        "coords": atom_coords,
        "requested_center_coords": selected_coords,
        "center_snap_distance": snap_distances,
        "phase_or_cluster_label": phase_or_cluster_label,
        **summaries,
        **mean_fields,
        "mean_prediction_cosine_error": summaries["mean_error"],
        "max_prediction_cosine_error": summaries["max_error"],
        "std_prediction_cosine_error": summaries["std_error"],
        "target_cluster_label": target_cluster_by_atom,
        "phase_conditioned_z_score": phase_z,
        "joint_hardness_score": joint_hardness,
    }
    output_path = Path(out_dir) / "directional_line_jepa.npz"
    atom_table_path = Path(out_dir) / "directional_line_jepa_atoms.csv"
    summary_path = Path(out_dir) / "directional_line_jepa_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp.npz")
    np.savez(tmp_output_path, **arrays)
    tmp_output_path.replace(output_path)
    _write_atom_table(atom_table_path, arrays)
    visualization_artifacts: dict[str, str] = {}
    if settings.visualize_3d:
        from .directional_line_jepa_vis import render_directional_line_jepa_visualizations

        step("Rendering per-snapshot directional Line-JEPA report")
        rendered = render_directional_line_jepa_visualizations(
            arrays, Path(out_dir) / "directional_line_jepa_report"
        )
        visualization_artifacts = {
            name: str(Path(path).relative_to(out_dir)) for name, path in rendered.items()
        }

    per_snapshot_counts = {}
    for source_slot_raw in np.unique(context_source_slots):
        source_slot = int(source_slot_raw)
        requested_count = int(np.sum(context_source_slots == source_slot))
        evaluated_count = int(np.sum(source_slots == source_slot))
        per_snapshot_counts[str(source_names_by_slot[source_slot])] = {
            "source_path": str(source_paths_by_slot[source_slot]),
            "requested_count": requested_count,
            "evaluated_count": evaluated_count,
            "excluded_boundary_count": requested_count - evaluated_count,
        }

    summary = {
        "enabled": True,
        "atom_count": atom_count,
        "requested_atom_count": requested_atom_count,
        "excluded_outside_line_anchor_domain": excluded_anchor_count,
        "direction_count": direction_count,
        "task_count": task_count,
        "atom_chunk_size": int(settings.atom_chunk_size),
        "encoded_environment_count_chunked": int(unique_environment_count),
        "naive_environment_encoding_count": int(task_count * model.line_atoms),
        "environment_encoding_reuse_factor": float(
            task_count * model.line_atoms / unique_environment_count
        ),
        "target_environment_encoding_count": (
            int(target_encoding_count)
        ),
        "direction_sampling": "fibonacci_hemisphere_exact_antipodal_pairs_v2",
        "primary_prediction_metric": (
            "reconstruction cosine error; lower is better; spatial prediction map uses "
            "the mean over sampled directions"
        ),
        "primary_directional_metric": (
            "standard deviation of reconstruction cosine error across directions divided "
            "by its directional mean"
        ),
        "target_position_requested": settings.target_position,
        "target_position": resolved_target_position,
        "target_line_index": int(target_index),
        "checkpoint_prediction_target": str(model.prediction_target),
        "checkpoint_target_encoder": str(model.target_encoder_mode),
        "directional_feature_mode": str(model.directional_feature_mode),
        "cluster_surprise_available": True,
        "epistemic_uncertainty_available": False,
        "directional_decomposition": {
            "novelty": "percentile rank of mean prediction error over directions",
            "polar_anisotropy": "l=1 component; +d and -d have different prediction error",
            "axial_anisotropy": "traceless l=2 component; an unoriented axis is preferred",
            "complex_anisotropy": "relative RMS directional residual above l=2",
            "harmonic_explained_fraction": "fraction of directional variance explained by l<=2",
        },
        "phase_conditioning_source": conditioning_source,
        "primary_k": int(primary_k),
        "relative_error_clip": settings.relative_error_clip,
        "joint_weights": settings.joint_weights,
        "per_snapshot_counts": per_snapshot_counts,
        "compact_archive": {
            "per_direction_field": "prediction_cosine_error",
            "omitted_per_task_fields": ["line_atom_ids", "line_t", "line_perp"],
            "source_metadata": "source_slots plus source_names_by_slot/source_paths_by_slot",
        },
        "quantiles": {
            "mean_prediction_cosine_error": _quantiles(summaries["mean_error"]),
            "max_prediction_cosine_error": _quantiles(summaries["max_error"]),
            "anisotropy_scalar": _quantiles(summaries["anisotropy_scalar"]),
            "relative_directional_variation": _quantiles(
                summaries["relative_directional_variation"]
            ),
            "polar_anisotropy": _quantiles(summaries["polar_anisotropy"]),
            "axial_anisotropy": _quantiles(summaries["axial_anisotropy"]),
            "complex_anisotropy": _quantiles(summaries["complex_anisotropy"]),
            "harmonic_explained_fraction": _quantiles(
                summaries["harmonic_explained_fraction"]
            ),
            "phase_conditioned_z_score": _quantiles(phase_z),
            "joint_hardness_score": _quantiles(joint_hardness),
        },
        "artifacts": {
            "directional_profiles_npz": str(output_path.relative_to(out_dir)),
            "per_atom_table_csv": str(atom_table_path.relative_to(out_dir)),
            "summary_json": str(summary_path.relative_to(out_dir)),
            **visualization_artifacts,
        },
    }
    write_json(summary_path, summary)
    print(
        "[analysis][directional-line-jepa] Saved "
        f"{atom_count} atoms x {direction_count} directions to {output_path}."
    )
    return summary


__all__ = [
    "DirectionalLineJEPASettings",
    "apply_directional_runtime_limits",
    "compute_directional_error_summaries",
    "compute_directional_error_summaries_chunked",
    "compute_directional_harmonic_summaries",
    "compute_joint_hardness_score",
    "disable_directional_for_non_line_jepa",
    "fibonacci_sphere_directions",
    "phase_conditioned_z_scores",
    "resolve_directional_line_jepa_settings",
    "run_directional_line_jepa_analysis",
]
