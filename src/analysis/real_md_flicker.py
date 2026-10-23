"""Temporal cluster diagnostics and CSV export, independent of rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .output_layout import write_json

if TYPE_CHECKING:
    from .real_md_qualitative import FrameSlice


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

def _cluster_count_vector_from_labels(
    labels: np.ndarray,
    *,
    label_to_pos: np.ndarray,
    cluster_count: int,
) -> tuple[np.ndarray, int]:
    positions = _labels_to_cluster_positions(labels, label_to_pos)
    valid = positions >= 0
    counts = np.bincount(
        positions[valid],
        minlength=int(cluster_count),
    ).astype(np.int64, copy=False)
    return counts, int(np.count_nonzero(valid))

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
) -> tuple[tuple[Path, pd.DataFrame], tuple[Path, pd.DataFrame], dict[str, Any]]:
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
    histogram = pd.DataFrame.from_records(hist_rows)
    histogram.to_csv(hist_csv, index=False)
    summary_table = pd.DataFrame.from_records(summary_rows)
    summary_table.to_csv(summary_csv, index=False)
    all_lengths_arr = np.asarray(all_lengths, dtype=np.float64)
    return (hist_csv, histogram), (summary_csv, summary_table), {
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
) -> tuple[tuple[Path, pd.DataFrame], tuple[Path, pd.DataFrame], dict[str, Any]]:
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
    aggregate = pd.DataFrame.from_records(aggregate_rows)
    aggregate.to_csv(aggregate_csv, index=False)
    by_source = pd.DataFrame.from_records(source_rows)
    by_source.to_csv(source_csv, index=False)
    return (aggregate_csv, aggregate), (source_csv, by_source), summary

def _write_cluster_transition_popularity(
    *,
    aggregate_counts: np.ndarray,
    cluster_ids: list[int],
    out_dir: Path,
) -> tuple[tuple[Path, pd.DataFrame], dict[str, Any]]:
    counts_arr = np.asarray(aggregate_counts, dtype=np.int64)
    total_valid = int(np.sum(counts_arr))
    total_changed = int(total_valid - np.trace(counts_arr))
    rows: list[dict[str, Any]] = []
    for source_pos, source_cluster_id in enumerate(cluster_ids):
        for target_pos, target_cluster_id in enumerate(cluster_ids):
            if int(source_cluster_id) == int(target_cluster_id):
                continue
            count = int(counts_arr[source_pos, target_pos])
            rows.append(
                {
                    "source_cluster_id": int(source_cluster_id),
                    "target_cluster_id": int(target_cluster_id),
                    "transition_count": count,
                    "fraction_of_changed_transitions": _safe_fraction(count, total_changed),
                    "fraction_of_all_matched_observations": _safe_fraction(count, total_valid),
                }
            )
    rows.sort(
        key=lambda row: (
            -int(row["transition_count"]),
            int(row["source_cluster_id"]),
            int(row["target_cluster_id"]),
        )
    )
    cumulative = 0
    for rank, row in enumerate(rows, start=1):
        cumulative += int(row["transition_count"])
        row["rank"] = int(rank)
        row["cumulative_fraction_of_changed_transitions"] = _safe_fraction(cumulative, total_changed)

    csv_path = Path(out_dir) / "cluster_transition_popularity.csv"
    table = pd.DataFrame.from_records(rows)
    table.to_csv(csv_path, index=False)
    top_row = rows[0] if rows else None
    return (csv_path, table), {
        "transition_count": int(total_changed),
        "matched_observation_count": int(total_valid),
        "top_transition": (
            None
            if top_row is None
            else {
                "source_cluster_id": int(top_row["source_cluster_id"]),
                "target_cluster_id": int(top_row["target_cluster_id"]),
                "count": int(top_row["transition_count"]),
                "fraction_of_changed_transitions": float(top_row["fraction_of_changed_transitions"]),
                "fraction_of_all_matched_observations": float(top_row["fraction_of_all_matched_observations"]),
            }
        ),
    }


def write_temporal_flicker_metrics(
    *,
    frames: list[FrameSlice],
    frame_source: str,
    coords: np.ndarray,
    labels: np.ndarray,
    instance_ids: np.ndarray | None,
    cluster_ids: list[int],
    assignment_margins: dict[str, Any] | None,
    out_dir: Path,
    neighbor_k: int,
    isolated_neighbor_change_fraction: float,
    coherent_same_transition_fraction: float,
    recrossing_lags: list[int],
    transition_popularity_window_frames: int,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame | None]]:
    flicker_dir = Path(out_dir) / "flicker"
    flicker_dir.mkdir(parents=True, exist_ok=True)
    if len(frames) < 2:
        return {
            "enabled": False,
            "reason": "requires at least two ordered frames",
            "root_dir": str(flicker_dir),
        }, {}
    if instance_ids is None:
        return {
            "enabled": False,
            "reason": "requires instance_ids to track local structures across frames",
            "root_dir": str(flicker_dir),
        }, {}

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
    first_frame_counts, first_frame_valid_count = _cluster_count_vector_from_labels(
        records[0]["labels"],
        label_to_pos=label_to_pos,
        cluster_count=cluster_count,
    )

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
        current_counts, current_valid_count = _cluster_count_vector_from_labels(
            record_b["labels"],
            label_to_pos=label_to_pos,
            cluster_count=cluster_count,
        )
        population_tv_from_first = (
            0.5
            * float(
                np.sum(
                    np.abs(
                        current_counts.astype(np.float64) / float(current_valid_count)
                        - first_frame_counts.astype(np.float64) / float(first_frame_valid_count)
                    )
                )
            )
            if first_frame_valid_count > 0 and current_valid_count > 0
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
            "population_tv_from_first": population_tv_from_first,
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
    (transition_popularity_csv, transition_table), transition_popularity_summary = _write_cluster_transition_popularity(
        aggregate_counts=aggregate_counts,
        cluster_ids=cluster_ids,
        out_dir=flicker_dir,
    )
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
    (dwell_hist_csv, dwell_hist_table), (dwell_summary_csv, dwell_summary_table), dwell_summary = _write_dwell_metrics(
        label_matrix=label_matrix,
        cluster_ids=cluster_ids,
        out_dir=flicker_dir,
        frame_spacing=frame_spacing,
        time_unit=time_unit,
    )
    (recrossing_csv, recrossing_table), (recrossing_by_source_csv, recrossing_by_source_table), recrossing_summary = _write_recrossing_metrics(
        label_matrix=label_matrix,
        cluster_ids=cluster_ids,
        out_dir=flicker_dir,
        lags=recrossing_lags,
        frame_spacing=frame_spacing,
        time_unit=time_unit,
    )

    tables = dict(
        pair_table=pair_table,
        reciprocal_table=reciprocal_table,
        transition_table=transition_table,
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
        "transition_popularity_window_frames": int(transition_popularity_window_frames),
        "artifacts": {
            "frame_pair_metrics_csv": str(frame_pair_csv),
            "reciprocal_pair_flicker_csv": str(reciprocal_csv),
            "cluster_transition_popularity_csv": str(transition_popularity_csv),
            "spatial_coherence_csv": str(spatial_csv),
            "dwell_time_histogram_csv": str(dwell_hist_csv),
            "dwell_time_summary_csv": str(dwell_summary_csv),
            "recrossing_metrics_csv": str(recrossing_csv),
            "recrossing_by_source_cluster_csv": str(recrossing_by_source_csv),
        },
        "label_churn_rate": _finite_metric_summary(pair_table["label_churn_rate"].to_numpy(float)),
        "net_population_tv": _finite_metric_summary(pair_table["net_population_tv"].to_numpy(float)),
        "population_tv_from_first": _finite_metric_summary(
            pair_table["population_tv_from_first"].to_numpy(float)
        ),
        "excess_churn_rate": _finite_metric_summary(pair_table["excess_churn_rate"].to_numpy(float)),
        "reciprocal_pair_flicker_rate": _finite_metric_summary(pair_table["reciprocal_pair_flicker_rate"].to_numpy(float)),
        "cluster_transition_popularity": transition_popularity_summary,
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
    return summary, tables
