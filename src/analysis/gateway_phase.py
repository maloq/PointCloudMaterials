"""Trajectory evidence for a required intermediate and independent CNA structure."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial import cKDTree, distance
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

from src.baselines.descriptor_baselines import CNADescriptorBaseline

from .cluster_profiles import _load_point_cloud_from_dataset
from .output_layout import log_saved_figure, write_json


@dataclass(frozen=True)
class GatewayPhaseSettings:
    enabled: bool
    state_a: tuple[int, ...]
    intermediate: tuple[int, ...]
    state_b: tuple[int, ...]
    bootstrap_samples: int
    random_state: int
    gateway_fraction_threshold: float
    minimum_reactive_paths: int
    lag_strides: tuple[int, ...]
    reactive_path_csv_limit: int
    spatial_enabled: bool
    spatial_neighbor_k: int
    spatial_max_frames: int
    precursor_lags: tuple[int, ...]
    precursor_radius_edges: tuple[float, ...]
    precursor_max_events: int
    cna_enabled: bool
    cna_max_samples: int
    cna_max_signatures: int
    cna_fit_max_pointclouds: int
    shell_min_neighbors: int
    shell_max_neighbors: int


def _int_tuple(value: Any, *, field: str) -> tuple[int, ...]:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    values = tuple(int(item) for item in list(value or []))
    if not values:
        raise ValueError(f"{field} must contain at least one cluster ID.")
    if any(item < 0 for item in values):
        raise ValueError(f"{field} requires non-negative cluster IDs, got {values}.")
    if len(set(values)) != len(values):
        raise ValueError(f"{field} contains duplicate cluster IDs: {values}.")
    return values


def resolve_gateway_phase_settings(analysis_cfg: Any) -> GatewayPhaseSettings:
    prefix = "gateway_phase"
    enabled = bool(OmegaConf.select(analysis_cfg, f"{prefix}.enabled", default=False))
    state_a = _int_tuple(
        OmegaConf.select(analysis_cfg, f"{prefix}.states.a", default=[0, 1]),
        field=f"{prefix}.states.a",
    )
    intermediate = _int_tuple(
        OmegaConf.select(analysis_cfg, f"{prefix}.states.intermediate", default=[2]),
        field=f"{prefix}.states.intermediate",
    )
    state_b = _int_tuple(
        OmegaConf.select(analysis_cfg, f"{prefix}.states.b", default=[3]),
        field=f"{prefix}.states.b",
    )
    groups = [set(state_a), set(intermediate), set(state_b)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError(
            "gateway_phase state groups must be pairwise disjoint: "
            f"A={state_a}, intermediate={intermediate}, B={state_b}."
        )
    settings = GatewayPhaseSettings(
        enabled=enabled,
        state_a=state_a,
        intermediate=intermediate,
        state_b=state_b,
        bootstrap_samples=int(
            OmegaConf.select(analysis_cfg, f"{prefix}.bootstrap_samples", default=1000)
        ),
        random_state=int(
            OmegaConf.select(analysis_cfg, f"{prefix}.random_state", default=123)
        ),
        gateway_fraction_threshold=float(
            OmegaConf.select(
                analysis_cfg, f"{prefix}.gateway_fraction_threshold", default=0.95
            )
        ),
        minimum_reactive_paths=int(
            OmegaConf.select(analysis_cfg, f"{prefix}.minimum_reactive_paths", default=30)
        ),
        lag_strides=tuple(
            int(v)
            for v in OmegaConf.select(
                analysis_cfg, f"{prefix}.lag_strides", default=[1, 2, 4, 8]
            )
        ),
        reactive_path_csv_limit=int(
            OmegaConf.select(
                analysis_cfg, f"{prefix}.reactive_path_csv_limit", default=100000
            )
        ),
        spatial_enabled=bool(
            OmegaConf.select(analysis_cfg, f"{prefix}.spatial.enabled", default=True)
        ),
        spatial_neighbor_k=int(
            OmegaConf.select(analysis_cfg, f"{prefix}.spatial.neighbor_k", default=12)
        ),
        spatial_max_frames=int(
            OmegaConf.select(analysis_cfg, f"{prefix}.spatial.max_frames", default=60)
        ),
        precursor_lags=tuple(
            int(v)
            for v in OmegaConf.select(
                analysis_cfg, f"{prefix}.spatial.precursor_lags", default=[1, 2, 4, 8]
            )
        ),
        precursor_radius_edges=tuple(
            float(v)
            for v in OmegaConf.select(
                analysis_cfg,
                f"{prefix}.spatial.precursor_radius_edges",
                default=[0.0, 4.0, 8.0, 12.0, 16.0],
            )
        ),
        precursor_max_events=int(
            OmegaConf.select(
                analysis_cfg, f"{prefix}.spatial.precursor_max_events", default=2000
            )
        ),
        cna_enabled=bool(
            OmegaConf.select(analysis_cfg, f"{prefix}.cna.enabled", default=True)
        ),
        cna_max_samples=int(
            OmegaConf.select(analysis_cfg, f"{prefix}.cna.max_samples", default=1200)
        ),
        cna_max_signatures=int(
            OmegaConf.select(analysis_cfg, f"{prefix}.cna.max_signatures", default=12)
        ),
        cna_fit_max_pointclouds=int(
            OmegaConf.select(
                analysis_cfg, f"{prefix}.cna.fit_max_pointclouds", default=1200
            )
        ),
        shell_min_neighbors=int(
            OmegaConf.select(
                analysis_cfg, f"{prefix}.cna.shell_min_neighbors", default=8
            )
        ),
        shell_max_neighbors=int(
            OmegaConf.select(
                analysis_cfg, f"{prefix}.cna.shell_max_neighbors", default=24
            )
        ),
    )
    if not enabled:
        return settings
    if settings.bootstrap_samples < 1:
        raise ValueError("gateway_phase.bootstrap_samples must be >= 1.")
    if not 0.0 < settings.gateway_fraction_threshold <= 1.0:
        raise ValueError("gateway_phase.gateway_fraction_threshold must be in (0, 1].")
    if settings.minimum_reactive_paths < 1:
        raise ValueError("gateway_phase.minimum_reactive_paths must be >= 1.")
    if not settings.lag_strides or any(v < 1 for v in settings.lag_strides):
        raise ValueError("gateway_phase.lag_strides must contain positive integers.")
    if settings.spatial_neighbor_k < 1:
        raise ValueError("gateway_phase.spatial.neighbor_k must be >= 1.")
    edges = settings.precursor_radius_edges
    if len(edges) < 2 or any(b <= a for a, b in zip(edges[:-1], edges[1:], strict=True)):
        raise ValueError(
            "gateway_phase.spatial.precursor_radius_edges must be strictly increasing."
        )
    if settings.cna_enabled and settings.cna_max_samples < 3:
        raise ValueError("gateway_phase.cna.max_samples must be >= 3.")
    return settings


def _macrostate_matrix(
    labels: np.ndarray,
    settings: GatewayPhaseSettings,
) -> np.ndarray:
    labels_arr = np.asarray(labels, dtype=np.int32)
    states = np.full(labels_arr.shape, 3, dtype=np.int8)
    states[np.isin(labels_arr, settings.state_a)] = 0
    states[np.isin(labels_arr, settings.intermediate)] = 1
    states[np.isin(labels_arr, settings.state_b)] = 2
    return states


def _align_trajectories(
    *,
    labels: np.ndarray,
    instance_ids: np.ndarray,
    coords: np.ndarray,
    anchor_frame_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    labels_arr = np.asarray(labels, dtype=np.int32).reshape(-1)
    ids_arr = np.asarray(instance_ids, dtype=np.int64).reshape(-1)
    coords_arr = np.asarray(coords, dtype=np.float32)
    frames_arr = np.asarray(anchor_frame_indices, dtype=np.int64).reshape(-1)
    row_count = labels_arr.size
    if ids_arr.size != row_count or frames_arr.size != row_count or coords_arr.shape != (row_count, 3):
        raise ValueError(
            "Gateway analysis requires aligned labels, instance_ids, coords, and anchor frames. "
            f"labels={labels_arr.shape}, ids={ids_arr.shape}, coords={coords_arr.shape}, "
            f"frames={frames_arr.shape}."
        )
    frame_values = np.unique(frames_arr)
    if frame_values.size < 3:
        raise ValueError(
            f"Gateway analysis requires at least three trajectory frames, got {frame_values.tolist()}."
        )
    records: list[tuple[np.ndarray, np.ndarray]] = []
    common_ids: np.ndarray | None = None
    for frame in frame_values:
        rows = np.flatnonzero(frames_arr == frame)
        order = np.argsort(ids_arr[rows], kind="mergesort")
        sorted_rows = rows[order]
        sorted_ids = ids_arr[sorted_rows]
        if np.unique(sorted_ids).size != sorted_ids.size:
            raise ValueError(f"Duplicate instance IDs found in anchor frame {int(frame)}.")
        records.append((sorted_ids, sorted_rows))
        common_ids = sorted_ids if common_ids is None else np.intersect1d(common_ids, sorted_ids)
    if common_ids is None or common_ids.size == 0:
        raise RuntimeError("No atom identities are shared by every gateway-analysis frame.")

    shape = (frame_values.size, common_ids.size)
    label_matrix = np.empty(shape, dtype=np.int32)
    coord_matrix = np.empty((*shape, 3), dtype=np.float32)
    sample_matrix = np.empty(shape, dtype=np.int64)
    for frame_pos, (sorted_ids, sorted_rows) in enumerate(records):
        pos = np.searchsorted(sorted_ids, common_ids)
        if not np.array_equal(sorted_ids[pos], common_ids):
            raise RuntimeError(f"Failed to align common atom identities in frame {frame_pos}.")
        rows = sorted_rows[pos]
        label_matrix[frame_pos] = labels_arr[rows]
        coord_matrix[frame_pos] = coords_arr[rows]
        sample_matrix[frame_pos] = rows
    return {
        "frame_indices": frame_values,
        "instance_ids": common_ids,
        "labels": label_matrix,
        "coords": coord_matrix,
        "sample_indices": sample_matrix,
        "tracking_coverage": np.asarray([common_ids.size / len(record[0]) for record in records]),
    }


def _extract_reactive_episodes(
    states: np.ndarray,
    instance_ids: np.ndarray,
    *,
    source_state: int,
    target_state: int,
    intermediate_state: int = 1,
) -> list[dict[str, Any]]:
    state_matrix = np.asarray(states, dtype=np.int8)
    atom_ids = np.asarray(instance_ids, dtype=np.int64)
    episodes: list[dict[str, Any]] = []
    for atom_pos, sequence in enumerate(state_matrix.T):
        active = False
        start = -1
        first_intermediate = -1
        intermediate_frames = 0
        for frame_pos, state_raw in enumerate(sequence):
            state = int(state_raw)
            if state == int(source_state):
                active = True
                start = int(frame_pos)
                first_intermediate = -1
                intermediate_frames = 0
                continue
            if not active:
                continue
            if state == int(intermediate_state):
                if first_intermediate < 0:
                    first_intermediate = int(frame_pos)
                intermediate_frames += 1
                continue
            if state == int(target_state):
                episodes.append(
                    {
                        "atom_position": int(atom_pos),
                        "atom_id": int(atom_ids[atom_pos]),
                        "start_frame_position": int(start),
                        "end_frame_position": int(frame_pos),
                        "duration_frames": int(frame_pos - start),
                        "visited_intermediate": bool(first_intermediate >= 0),
                        "first_intermediate_frame_position": int(first_intermediate),
                        "intermediate_residence_frames": int(intermediate_frames),
                    }
                )
                active = False
                start = -1
                first_intermediate = -1
                intermediate_frames = 0
    return episodes


def _episode_summary(
    episodes: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
    random_state: int,
    minimum_paths: int,
    threshold: float,
) -> dict[str, Any]:
    count = len(episodes)
    via = int(sum(bool(item["visited_intermediate"]) for item in episodes))
    fraction = float(via / count) if count else float("nan")
    ci = [float("nan"), float("nan")]
    if count:
        atom_ids = np.asarray([int(item["atom_id"]) for item in episodes], dtype=np.int64)
        unique_ids = np.unique(atom_ids)
        totals = np.asarray([np.sum(atom_ids == atom_id) for atom_id in unique_ids], dtype=np.int64)
        successes = np.asarray(
            [
                sum(
                    bool(item["visited_intermediate"])
                    for item in episodes
                    if int(item["atom_id"]) == int(atom_id)
                )
                for atom_id in unique_ids
            ],
            dtype=np.int64,
        )
        rng = np.random.default_rng(int(random_state))
        estimates = np.empty(int(bootstrap_samples), dtype=np.float64)
        for bootstrap_idx in range(int(bootstrap_samples)):
            sampled = rng.integers(0, unique_ids.size, size=unique_ids.size)
            estimates[bootstrap_idx] = float(successes[sampled].sum() / totals[sampled].sum())
        ci = [float(v) for v in np.quantile(estimates, [0.025, 0.975])]
    if count < int(minimum_paths):
        verdict = "inconclusive_too_few_paths"
    elif float(ci[0]) >= float(threshold):
        verdict = "supports_required_intermediate"
    elif np.isfinite(fraction) and fraction < float(threshold):
        verdict = "observed_bypass_paths"
    else:
        verdict = "inconclusive_uncertainty_crosses_threshold"
    return {
        "reactive_path_count": int(count),
        "paths_via_intermediate": int(via),
        "bypass_path_count": int(count - via),
        "fraction_via_intermediate": fraction,
        "fraction_via_intermediate_ci95": ci,
        "unique_reactive_atoms": int(len({int(item["atom_id"]) for item in episodes})),
        "median_duration_frames": (
            float(np.median([int(item["duration_frames"]) for item in episodes]))
            if count
            else float("nan")
        ),
        "median_intermediate_residence_frames": (
            float(
                np.median(
                    [
                        int(item["intermediate_residence_frames"])
                        for item in episodes
                        if bool(item["visited_intermediate"])
                    ]
                )
            )
            if via
            else float("nan")
        ),
        "verdict": verdict,
    }


def _empirical_intermediate_committor(states: np.ndarray) -> dict[str, Any]:
    state_matrix = np.asarray(states, dtype=np.int8)
    outcomes: list[int] = []
    for sequence in state_matrix.T:
        entries = np.flatnonzero((sequence == 1) & np.concatenate(([True], sequence[:-1] != 1)))
        for entry in entries:
            later = sequence[int(entry) + 1 :]
            hits = np.flatnonzero(np.isin(later, [0, 2]))
            if hits.size:
                outcomes.append(int(later[int(hits[0])] == 2))
    return {
        "definition": "probability that an intermediate entry next hits B before A",
        "entry_count_with_resolved_next_hit": int(len(outcomes)),
        "p_hit_b_before_a": float(np.mean(outcomes)) if outcomes else float("nan"),
        "important_limitation": (
            "This is an empirical next-hit probability from observed trajectories, not a "
            "shooting committor estimated from repeated trajectories at one configuration."
        ),
    }


def _lag_sensitivity(
    states: np.ndarray,
    instance_ids: np.ndarray,
    strides: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stride in sorted(set(int(v) for v in strides)):
        episodes = _extract_reactive_episodes(
            np.asarray(states)[::stride],
            instance_ids,
            source_state=0,
            target_state=2,
        )
        count = len(episodes)
        via = sum(bool(item["visited_intermediate"]) for item in episodes)
        rows.append(
            {
                "frame_stride_multiplier": int(stride),
                "reactive_path_count": int(count),
                "fraction_via_intermediate": float(via / count) if count else float("nan"),
            }
        )
    return rows


def _write_episode_csv(path: Path, episodes: Sequence[dict[str, Any]], *, limit: int) -> None:
    fieldnames = list(episodes[0].keys()) if episodes else [
        "atom_position", "atom_id", "start_frame_position", "end_frame_position",
        "duration_frames", "visited_intermediate", "first_intermediate_frame_position",
        "intermediate_residence_frames",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(list(episodes)[: int(limit)])


def _frame_box_lengths(dataset: Any, frame_indices: np.ndarray) -> np.ndarray | None:
    if dataset is None or not hasattr(dataset, "box_lengths"):
        return None
    box_lengths = np.asarray(dataset.box_lengths, dtype=np.float32)
    frames = np.asarray(frame_indices, dtype=np.int64)
    if frames.min() < 0 or frames.max() >= box_lengths.shape[0]:
        raise IndexError(
            "Gateway anchor frames exceed temporal dataset box metadata: "
            f"frames=[{int(frames.min())}, {int(frames.max())}], boxes={box_lengths.shape[0]}."
        )
    return box_lengths[frames]


def _wrapped_positions(coords: np.ndarray, box_length: np.ndarray | None) -> np.ndarray:
    points = np.asarray(coords, dtype=np.float32)
    if box_length is None:
        return points
    return np.mod(points, np.asarray(box_length, dtype=np.float32)[None, :])


def _spatial_coherence(
    states: np.ndarray,
    coords: np.ndarray,
    frame_indices: np.ndarray,
    box_lengths: np.ndarray | None,
    *,
    neighbor_k: int,
    max_frames: int,
) -> list[dict[str, Any]]:
    frame_count, atom_count = np.asarray(states).shape
    selected = np.unique(
        np.linspace(0, frame_count - 1, min(frame_count, int(max_frames))).round().astype(int)
    )
    k = min(int(neighbor_k), atom_count - 1)
    rows: list[dict[str, Any]] = []
    for frame_pos in selected:
        box = None if box_lengths is None else box_lengths[frame_pos]
        points = _wrapped_positions(coords[frame_pos], box)
        tree = cKDTree(points, boxsize=box)
        _, neighbors = tree.query(points, k=k + 1)
        neighbors = np.asarray(neighbors, dtype=np.int64)[:, 1:]
        frame_states = np.asarray(states[frame_pos], dtype=np.int8)
        valid = frame_states < 3
        same = frame_states[neighbors] == frame_states[:, None]
        observed = float(np.mean(same[valid])) if np.any(valid) else float("nan")
        proportions = np.asarray([np.mean(frame_states[valid] == value) for value in range(3)])
        shuffled = float(np.sum(proportions * proportions))
        rows.append(
            {
                "frame_position": int(frame_pos),
                "anchor_frame_index": int(frame_indices[frame_pos]),
                "same_state_neighbor_fraction": observed,
                "shuffled_expectation": shuffled,
                "same_state_excess": observed - shuffled,
            }
        )
    return rows


def _precursor_enrichment(
    episodes: Sequence[dict[str, Any]],
    states: np.ndarray,
    coords: np.ndarray,
    box_lengths: np.ndarray | None,
    *,
    lags: Sequence[int],
    radius_edges: Sequence[float],
    max_events: int,
    random_state: int,
) -> list[dict[str, Any]]:
    if not episodes:
        return []
    rng = np.random.default_rng(int(random_state))
    event_indices = np.arange(len(episodes), dtype=np.int64)
    if event_indices.size > int(max_events):
        event_indices = np.sort(
            rng.choice(event_indices, size=int(max_events), replace=False)
        )
    edges = np.asarray(radius_edges, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for lag in sorted(set(int(v) for v in lags)):
        fractions: list[list[float]] = [[] for _ in range(edges.size - 1)]
        baselines: list[float] = []
        used_events = 0
        tree_cache: dict[int, tuple[cKDTree, np.ndarray, np.ndarray | None]] = {}
        for event_idx in event_indices:
            event = episodes[int(event_idx)]
            end = int(event["end_frame_position"])
            earlier = end - int(lag)
            if earlier < 0:
                continue
            if earlier not in tree_cache:
                box = None if box_lengths is None else box_lengths[earlier]
                points = _wrapped_positions(coords[earlier], box)
                tree_cache[earlier] = (cKDTree(points, boxsize=box), points, box)
            tree, points, box = tree_cache[earlier]
            center = _wrapped_positions(
                coords[end, int(event["atom_position"])][None, :], box
            )[0]
            neighbors = np.asarray(tree.query_ball_point(center, r=float(edges[-1])), dtype=np.int64)
            if neighbors.size == 0:
                continue
            delta = np.abs(points[neighbors] - center[None, :])
            if box is not None:
                delta = np.minimum(delta, np.asarray(box)[None, :] - delta)
            radial = np.linalg.norm(delta, axis=1)
            earlier_states = np.asarray(states[earlier], dtype=np.int8)
            baselines.append(float(np.mean(earlier_states == 1)))
            for bin_idx, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
                shell = neighbors[(radial >= lower) & (radial < upper)]
                if shell.size:
                    fractions[bin_idx].append(float(np.mean(earlier_states[shell] == 1)))
            used_events += 1
        baseline = float(np.mean(baselines)) if baselines else float("nan")
        for bin_idx, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            local = float(np.mean(fractions[bin_idx])) if fractions[bin_idx] else float("nan")
            rows.append(
                {
                    "lag_frames": int(lag),
                    "radius_lower": float(lower),
                    "radius_upper": float(upper),
                    "event_count": int(used_events),
                    "intermediate_fraction": local,
                    "background_intermediate_fraction": baseline,
                    "intermediate_enrichment": local - baseline,
                }
            )
    return rows


def _balanced_cna_sample_indices(
    states_flat: np.ndarray,
    *,
    max_samples: int,
    random_state: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(random_state))
    parts: list[np.ndarray] = []
    quota = max(1, int(max_samples) // 3)
    for state in range(3):
        candidates = np.flatnonzero(np.asarray(states_flat) == state)
        if candidates.size == 0:
            raise ValueError(f"CNA gateway analysis found no samples for macrostate {state}.")
        take = min(quota, candidates.size)
        parts.append(np.sort(rng.choice(candidates, size=take, replace=False)))
    return np.sort(np.concatenate(parts))


def _cna_analysis(
    *,
    dataset: Any,
    states_flat: np.ndarray,
    labels_flat: np.ndarray,
    frame_flat: np.ndarray,
    settings: GatewayPhaseSettings,
    out_dir: Path,
) -> dict[str, Any]:
    if dataset is None:
        raise TypeError("gateway_phase.cna.enabled=true requires the inference dataset.")
    if len(dataset) < len(states_flat):
        raise ValueError(
            "Gateway CNA dataset is shorter than the inference arrays: "
            f"dataset={len(dataset)}, samples={len(states_flat)}."
        )
    sample_indices = _balanced_cna_sample_indices(
        states_flat,
        max_samples=settings.cna_max_samples,
        random_state=settings.random_state,
    )
    point_clouds: list[np.ndarray] = []
    for sample_index in sample_indices:
        points = _load_point_cloud_from_dataset(dataset, int(sample_index), point_scale=1.0)
        point_clouds.append(np.asarray(points, dtype=np.float32))
    clouds = np.stack(point_clouds)
    baseline = CNADescriptorBaseline(
        center_atom_tolerance=1.0e-6,
        shell_min_neighbors=settings.shell_min_neighbors,
        shell_max_neighbors=settings.shell_max_neighbors,
        max_signatures=settings.cna_max_signatures,
        append_shell_size=True,
        fit_max_pointclouds=settings.cna_fit_max_pointclouds,
    )
    features = baseline.fit_transform(clouds)
    signatures = list(baseline.metadata()["signature_vocab"]) + ["other"]
    state_values = np.asarray(states_flat, dtype=np.int8)[sample_indices]
    cluster_values = np.asarray(labels_flat, dtype=np.int32)[sample_indices]
    frame_values = np.asarray(frame_flat, dtype=np.int64)[sample_indices]
    signature_features = features[:, : len(signatures)]
    state_names = ["A", "intermediate", "B"]
    means = np.vstack(
        [np.mean(signature_features[state_values == state], axis=0) for state in range(3)]
    )
    jsd: dict[str, float] = {}
    for first, second in ((0, 1), (1, 2), (0, 2)):
        jsd[f"{state_names[first]}_vs_{state_names[second]}"] = float(
            distance.jensenshannon(means[first], means[second], base=2.0) ** 2
        )

    fold_scores: list[float] = []
    unique_frames = np.unique(frame_values)
    if unique_frames.size >= 2:
        splitter = GroupKFold(n_splits=min(5, int(unique_frames.size)))
        for train, test in splitter.split(signature_features, state_values, groups=frame_values):
            if np.unique(state_values[train]).size < 3 or np.unique(state_values[test]).size < 3:
                continue
            classifier = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=settings.random_state,
            )
            classifier.fit(signature_features[train], state_values[train])
            fold_scores.append(
                float(
                    balanced_accuracy_score(
                        state_values[test], classifier.predict(signature_features[test])
                    )
                )
            )

    table_path = out_dir / "cna_samples.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["sample_index", "frame_index", "cluster_id", "macrostate"] + [
            f"cna_{signature}" for signature in signatures
        ] + ["cna_shell_size"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx, sample_index in enumerate(sample_indices):
            row: dict[str, Any] = {
                "sample_index": int(sample_index),
                "frame_index": int(frame_values[row_idx]),
                "cluster_id": int(cluster_values[row_idx]),
                "macrostate": state_names[int(state_values[row_idx])],
                "cna_shell_size": float(features[row_idx, -1]),
            }
            row.update(
                {
                    f"cna_{signature}": float(signature_features[row_idx, sig_idx])
                    for sig_idx, signature in enumerate(signatures)
                }
            )
            writer.writerow(row)

    figure_path = out_dir / "cna_phase_evidence.png"
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=True)
    image = axes[0].imshow(means, aspect="auto", cmap="magma", vmin=0.0)
    axes[0].set_yticks(range(3), state_names)
    axes[0].set_xticks(range(len(signatures)), signatures, rotation=55, ha="right")
    axes[0].set_title("Mean CNA signature fractions")
    fig.colorbar(image, ax=axes[0], label="fraction")
    projection = PCA(n_components=2, random_state=settings.random_state).fit_transform(
        signature_features
    )
    colors = ["#2a7fde", "#f26a00", "#3f9142"]
    for state, name in enumerate(state_names):
        mask = state_values == state
        axes[1].scatter(
            projection[mask, 0], projection[mask, 1], s=10, alpha=0.45,
            color=colors[state], label=name,
        )
    axes[1].set_title("CNA-only physical structure landscape")
    axes[1].set_xlabel("CNA PCA 1")
    axes[1].set_ylabel("CNA PCA 2")
    axes[1].legend(frameon=False)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    log_saved_figure(figure_path)
    return {
        "enabled": True,
        "sample_count": int(sample_indices.size),
        "signature_names": signatures,
        "mean_signature_fraction_by_state": {
            name: means[idx].tolist() for idx, name in enumerate(state_names)
        },
        "jensen_shannon_divergence": jsd,
        "frame_grouped_cna_classifier_balanced_accuracy": {
            "fold_scores": fold_scores,
            "mean": float(np.mean(fold_scores)) if fold_scores else float("nan"),
            "chance_level": 1.0 / 3.0,
            "interpretation": (
                "Predictability from CNA supports physically distinct local motifs, but does "
                "not by itself establish distinct thermodynamic phases."
            ),
        },
        "artifacts": {
            "samples_csv": str(table_path),
            "figure": str(figure_path),
        },
    }


def _write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_gateway_figure(
    path: Path,
    forward: dict[str, Any],
    reverse: dict[str, Any],
    lag_rows: Sequence[dict[str, Any]],
    spatial_rows: Sequence[dict[str, Any]],
    precursor_rows: Sequence[dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    values = [forward["fraction_via_intermediate"], reverse["fraction_via_intermediate"]]
    axes[0, 0].bar(["A→B", "B→A"], values, color=["#2a7fde", "#3f9142"])
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].set_ylabel("fraction visiting intermediate")
    axes[0, 0].set_title("Reactive paths through intermediate")
    axes[0, 1].plot(
        [row["frame_stride_multiplier"] for row in lag_rows],
        [row["fraction_via_intermediate"] for row in lag_rows],
        marker="o", color="#b61f31",
    )
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].set_xlabel("frame subsampling stride")
    axes[0, 1].set_ylabel("A→B fraction via intermediate")
    axes[0, 1].set_title("Cadence sensitivity")
    if spatial_rows:
        axes[1, 0].plot(
            [row["anchor_frame_index"] for row in spatial_rows],
            [row["same_state_excess"] for row in spatial_rows],
            color="#5a2f8a",
        )
    axes[1, 0].axhline(0.0, color="#555555", linewidth=0.8)
    axes[1, 0].set_xlabel("trajectory frame")
    axes[1, 0].set_ylabel("same-state neighbor excess")
    axes[1, 0].set_title("Spatial coherence above shuffled labels")
    if precursor_rows:
        lags = sorted({int(row["lag_frames"]) for row in precursor_rows})
        bins = sorted({(float(row["radius_lower"]), float(row["radius_upper"])) for row in precursor_rows})
        grid = np.full((len(lags), len(bins)), np.nan)
        for row in precursor_rows:
            grid[lags.index(int(row["lag_frames"])), bins.index((float(row["radius_lower"]), float(row["radius_upper"])))] = float(row["intermediate_enrichment"])
        vmax = (
            max(float(np.nanmax(np.abs(grid))), 1.0e-12)
            if np.any(np.isfinite(grid))
            else 1.0
        )
        image = axes[1, 1].imshow(grid, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        axes[1, 1].set_yticks(range(len(lags)), lags)
        axes[1, 1].set_xticks(range(len(bins)), [f"{a:g}–{b:g}" for a, b in bins], rotation=35, ha="right")
        fig.colorbar(image, ax=axes[1, 1], label="C2 enrichment")
    axes[1, 1].set_xlabel("radius shell")
    axes[1, 1].set_ylabel("frames before B entry")
    axes[1, 1].set_title("Intermediate enrichment before B formation")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    log_saved_figure(path)


def _save_reactive_path_raster(
    path: Path,
    states: np.ndarray,
    episodes: Sequence[dict[str, Any]],
    *,
    max_paths: int = 250,
) -> None:
    selected = sorted(
        episodes,
        key=lambda item: (
            not bool(item["visited_intermediate"]),
            int(item["duration_frames"]),
            int(item["atom_id"]),
        ),
    )[: int(max_paths)]
    if selected:
        width = max(int(item["duration_frames"]) + 1 for item in selected)
        raster = np.full((len(selected), width), -1, dtype=np.int8)
        for row, item in enumerate(selected):
            atom_pos = int(item["atom_position"])
            start = int(item["start_frame_position"])
            end = int(item["end_frame_position"])
            segment = np.asarray(states[start : end + 1, atom_pos], dtype=np.int8)
            raster[row, : segment.size] = segment
    else:
        raster = np.full((1, 2), -1, dtype=np.int8)
    masked = np.ma.masked_where(raster < 0, raster)
    cmap = ListedColormap(["#2a7fde", "#f26a00", "#3f9142", "#bdbdbd"])
    cmap.set_bad("white")
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, ax = plt.subplots(figsize=(11.0, max(3.2, min(9.0, 0.035 * raster.shape[0] + 2.2))))
    ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_xlabel("frames since last A observation")
    ax.set_ylabel("reactive path (via-intermediate paths first)")
    ax.set_title(f"Tracked-atom A→B reactive paths (showing {len(selected)}/{len(episodes)})")
    ax.legend(
        handles=[
            Patch(color="#2a7fde", label="A"),
            Patch(color="#f26a00", label="intermediate"),
            Patch(color="#3f9142", label="B"),
            Patch(color="#bdbdbd", label="other cluster"),
        ],
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log_saved_figure(path)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    forward = summary["forward_gateway"]
    cna = summary["cna_phase_evidence"]
    artifacts = summary["artifacts"]
    state_text = (
        f"A={summary['states']['a']}, intermediate={summary['states']['intermediate']}, "
        f"B={summary['states']['b']}"
    )
    cna_text = (
        "CNA was disabled."
        if not cna.get("enabled", False)
        else (
            "Frame-held-out CNA balanced accuracy: "
            f"{float(cna['frame_grouped_cna_classifier_balanced_accuracy']['mean']):.3f} "
            "(chance 0.333)."
        )
    )
    path.write_text(
        f"""<!doctype html><meta charset="utf-8"><title>Gateway and phase validation</title>
<style>body{{font:16px system-ui,sans-serif;max-width:1050px;margin:2rem auto;line-height:1.5;color:#263238}}
img{{max-width:100%;height:auto}} code{{background:#eef2f5;padding:.12rem .3rem}} .verdict{{font-size:1.15rem;padding:1rem;background:#eef5ff}}</style>
<h1>Gateway-intermediate and CNA phase validation</h1>
<p><b>State definition:</b> {escape(state_text)}</p>
<p class="verdict"><b>Forward result:</b> {escape(str(forward['verdict']))};
{int(forward['paths_via_intermediate'])}/{int(forward['reactive_path_count'])} observed A→B paths visited the intermediate
({float(forward['fraction_via_intermediate']):.3f}, 95% atom-block bootstrap CI
{escape(str(forward['fraction_via_intermediate_ci95']))}).</p>
<p>This is evidence about observed tracked-atom paths. It is not causal proof and cannot exclude visits shorter than the saved-frame interval.</p>
<h2>Gateway evidence</h2><img src="{escape(Path(artifacts['gateway_figure']).name)}" alt="Gateway evidence dashboard">
<p>The cadence panel is essential: apparent bypasses that grow after frame subsampling indicate missed short-lived intermediate visits.</p>
<h2>Individual tracked-atom paths</h2><img src="{escape(Path(artifacts['reactive_path_raster']).name)}" alt="Reactive path sequence raster">
<h2>Independent physical structure</h2><p>{escape(cna_text)}</p>
{('<img src="' + escape(Path(cna['artifacts']['figure']).name) + '" alt="CNA phase evidence">') if cna.get('enabled', False) else ''}
<p>CNA separation establishes different local coordination motifs, not a thermodynamic phase by itself. Spatial coherence, residence time, and fixed-condition thermodynamic sampling are still required for a phase claim.</p>
<h2>Data files</h2><ul>
<li><a href="{escape(Path(artifacts['forward_paths_csv']).name)}">Forward reactive paths</a></li>
<li><a href="{escape(Path(artifacts['lag_sensitivity_csv']).name)}">Cadence sensitivity</a></li>
<li><a href="{escape(Path(artifacts['spatial_coherence_csv']).name)}">Spatial coherence</a></li>
<li><a href="{escape(Path(artifacts['precursor_enrichment_csv']).name)}">Spatial precursor enrichment</a></li>
<li><a href="{escape(Path(artifacts['metrics_json']).name)}">Complete metrics JSON</a></li>
</ul>""",
        encoding="utf-8",
    )


def run_gateway_phase_analysis(
    *,
    labels: np.ndarray,
    instance_ids: np.ndarray,
    coords: np.ndarray,
    anchor_frame_indices: np.ndarray,
    out_dir: Path,
    settings: GatewayPhaseSettings,
    inference_dataset: Any | None = None,
    temporal_dataset: Any | None = None,
    step: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not settings.enabled:
        return {}
    available = {int(v) for v in np.unique(np.asarray(labels, dtype=int)) if int(v) >= 0}
    requested = set(settings.state_a) | set(settings.intermediate) | set(settings.state_b)
    missing = sorted(requested - available)
    if missing:
        raise ValueError(
            "gateway_phase state definition references missing primary cluster IDs: "
            f"missing={missing}, available={sorted(available)}."
        )
    if step is not None:
        step("Analyzing gateway intermediate and CNA phase evidence")
    root = Path(out_dir) / "gateway_phase_validation"
    root.mkdir(parents=True, exist_ok=True)
    aligned = _align_trajectories(
        labels=labels,
        instance_ids=instance_ids,
        coords=coords,
        anchor_frame_indices=anchor_frame_indices,
    )
    states = _macrostate_matrix(aligned["labels"], settings)
    forward_episodes = _extract_reactive_episodes(
        states, aligned["instance_ids"], source_state=0, target_state=2
    )
    reverse_episodes = _extract_reactive_episodes(
        states, aligned["instance_ids"], source_state=2, target_state=0
    )
    forward = _episode_summary(
        forward_episodes,
        bootstrap_samples=settings.bootstrap_samples,
        random_state=settings.random_state,
        minimum_paths=settings.minimum_reactive_paths,
        threshold=settings.gateway_fraction_threshold,
    )
    reverse = _episode_summary(
        reverse_episodes,
        bootstrap_samples=settings.bootstrap_samples,
        random_state=settings.random_state + 1,
        minimum_paths=settings.minimum_reactive_paths,
        threshold=settings.gateway_fraction_threshold,
    )
    lag_rows = _lag_sensitivity(states, aligned["instance_ids"], settings.lag_strides)
    boxes = _frame_box_lengths(temporal_dataset, aligned["frame_indices"])
    spatial_rows = (
        _spatial_coherence(
            states,
            aligned["coords"],
            aligned["frame_indices"],
            boxes,
            neighbor_k=settings.spatial_neighbor_k,
            max_frames=settings.spatial_max_frames,
        )
        if settings.spatial_enabled
        else []
    )
    precursor_rows = (
        _precursor_enrichment(
            forward_episodes,
            states,
            aligned["coords"],
            boxes,
            lags=settings.precursor_lags,
            radius_edges=settings.precursor_radius_edges,
            max_events=settings.precursor_max_events,
            random_state=settings.random_state,
        )
        if settings.spatial_enabled
        else []
    )
    forward_csv = root / "forward_reactive_paths.csv"
    reverse_csv = root / "reverse_reactive_paths.csv"
    lag_csv = root / "lag_sensitivity.csv"
    spatial_csv = root / "spatial_coherence.csv"
    precursor_csv = root / "precursor_enrichment.csv"
    _write_episode_csv(forward_csv, forward_episodes, limit=settings.reactive_path_csv_limit)
    _write_episode_csv(reverse_csv, reverse_episodes, limit=settings.reactive_path_csv_limit)
    _write_rows(lag_csv, lag_rows)
    _write_rows(spatial_csv, spatial_rows)
    _write_rows(precursor_csv, precursor_rows)

    labels_flat = np.asarray(labels, dtype=np.int32).reshape(-1)
    frames_flat = np.asarray(anchor_frame_indices, dtype=np.int64).reshape(-1)
    states_flat = _macrostate_matrix(labels_flat, settings)
    cna_summary = (
        _cna_analysis(
            dataset=inference_dataset,
            states_flat=states_flat,
            labels_flat=labels_flat,
            frame_flat=frames_flat,
            settings=settings,
            out_dir=root,
        )
        if settings.cna_enabled
        else {"enabled": False}
    )
    gateway_figure = root / "gateway_evidence.png"
    path_raster = root / "reactive_path_raster.png"
    _save_gateway_figure(
        gateway_figure, forward, reverse, lag_rows, spatial_rows, precursor_rows
    )
    _save_reactive_path_raster(path_raster, states, forward_episodes)
    metrics_path = root / "gateway_phase_metrics.json"
    report_path = root / "index.html"
    summary = {
        "enabled": True,
        "states": {
            "a": list(settings.state_a),
            "intermediate": list(settings.intermediate),
            "b": list(settings.state_b),
            "other_clusters_are_retained_as_other_state": True,
        },
        "tracked_frame_count": int(aligned["labels"].shape[0]),
        "tracked_atom_count": int(aligned["labels"].shape[1]),
        "tracking_coverage_min": float(np.min(aligned["tracking_coverage"])),
        "forward_gateway": forward,
        "reverse_gateway": reverse,
        "empirical_intermediate_committor": _empirical_intermediate_committor(states),
        "lag_sensitivity": lag_rows,
        "spatial_coherence": spatial_rows,
        "spatial_precursor_enrichment": precursor_rows,
        "cna_phase_evidence": cna_summary,
        "artifacts": {
            "report": str(report_path),
            "metrics_json": str(metrics_path),
            "gateway_figure": str(gateway_figure),
            "reactive_path_raster": str(path_raster),
            "forward_paths_csv": str(forward_csv),
            "reverse_paths_csv": str(reverse_csv),
            "lag_sensitivity_csv": str(lag_csv),
            "spatial_coherence_csv": str(spatial_csv),
            "precursor_enrichment_csv": str(precursor_csv),
        },
    }
    write_json(metrics_path, summary)
    _write_report(report_path, summary)
    return summary


__all__ = [
    "GatewayPhaseSettings",
    "resolve_gateway_phase_settings",
    "run_gateway_phase_analysis",
]
