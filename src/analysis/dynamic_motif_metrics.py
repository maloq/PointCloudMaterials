from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _safe_entropy(probabilities: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    probs = probs[probs > 0.0]
    if probs.size == 0:
        return float("nan")
    return float(-(probs * np.log(probs)).sum())


def _sorted_state_id_list(df: pd.DataFrame) -> list[tuple[str, int]]:
    if df.empty:
        return []
    unique_pairs = (
        df.loc[:, ["resolved_family", "resolved_motif_id"]]
        .drop_duplicates()
        .sort_values(["resolved_family", "resolved_motif_id"])
    )
    return [
        (str(row.resolved_family), int(row.resolved_motif_id))
        for row in unique_pairs.itertuples(index=False)
    ]


def _state_to_label(family: str, motif_id: int) -> str:
    prefix = "S" if str(family) == "stable" else "B"
    return f"{prefix}{int(motif_id)}"


def _binary_roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    y_score = np.asarray(scores, dtype=np.float64).reshape(-1)
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    positive = y_true == 1
    negative = y_true == 0
    n_pos = int(positive.sum())
    n_neg = int(negative.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(y_score, method="average")
    sum_pos_ranks = float(ranks[positive].sum())
    auc = (sum_pos_ranks - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)


def build_usage_table(
    motif_ids: np.ndarray,
    *,
    confidences: np.ndarray | None,
    motif_family: str,
    total_motif_count: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ids = np.asarray(motif_ids, dtype=np.int64).reshape(-1)
    valid_mask = ids >= 0
    valid_ids = ids[valid_mask]
    confidence_arr = None if confidences is None else np.asarray(confidences, dtype=np.float64).reshape(-1)[valid_mask]
    total_valid = int(valid_ids.size)

    rows: list[dict[str, Any]] = []
    if total_valid > 0:
        unique_ids, counts = np.unique(valid_ids, return_counts=True)
        for idx, count in zip(unique_ids.tolist(), counts.tolist(), strict=True):
            motif_mask = valid_ids == int(idx)
            rows.append(
                {
                    "motif_family": str(motif_family),
                    "motif_id": int(idx),
                    "support_count": int(count),
                    "fraction": float(count) / float(total_valid),
                    "mean_confidence": (
                        float(confidence_arr[motif_mask].mean()) if confidence_arr is not None else float("nan")
                    ),
                }
            )
    table = pd.DataFrame(rows)
    active_count = int(table.shape[0])
    dead_fraction = float("nan")
    if total_motif_count is not None and int(total_motif_count) > 0:
        dead_fraction = float(max(int(total_motif_count) - active_count, 0)) / float(int(total_motif_count))
    usage_entropy = (
        _safe_entropy(table["fraction"].to_numpy(dtype=np.float64))
        if not table.empty
        else float("nan")
    )
    metrics = {
        f"{motif_family}_active_count": int(active_count),
        f"{motif_family}_usage_entropy": float(usage_entropy),
        f"{motif_family}_dead_fraction": float(dead_fraction),
    }
    return table, metrics


def compute_frame_motif_proportions(sample_df: pd.DataFrame) -> pd.DataFrame:
    if sample_df.empty:
        return pd.DataFrame(
            columns=[
                "source_path",
                "frame_index",
                "timestep",
                "motif_family",
                "motif_id",
                "count",
                "fraction",
                "frame_total_samples",
            ]
        )

    required = {"source_path", "frame_index", "timestep", "stable_id", "bridge_active", "bridge_id"}
    missing = sorted(required.difference(sample_df.columns))
    if missing:
        raise KeyError(
            "compute_frame_motif_proportions requires sample_df columns "
            f"{sorted(required)}, missing {missing}."
        )

    rows: list[dict[str, Any]] = []
    grouped = sample_df.groupby(["source_path", "frame_index", "timestep"], sort=True, dropna=False)
    for (source_path, frame_index, timestep), frame in grouped:
        total = int(frame.shape[0])
        stable_counts = frame["stable_id"].value_counts(dropna=False).sort_index()
        for motif_id, count in stable_counts.items():
            rows.append(
                {
                    "source_path": str(source_path),
                    "frame_index": int(frame_index),
                    "timestep": int(timestep),
                    "motif_family": "stable",
                    "motif_id": int(motif_id),
                    "count": int(count),
                    "fraction": float(count) / float(total),
                    "frame_total_samples": int(total),
                }
            )
        active_bridge = frame.loc[frame["bridge_active"].astype(bool)]
        if not active_bridge.empty:
            bridge_counts = active_bridge["bridge_id"].value_counts().sort_index()
            for motif_id, count in bridge_counts.items():
                rows.append(
                    {
                        "source_path": str(source_path),
                        "frame_index": int(frame_index),
                        "timestep": int(timestep),
                        "motif_family": "bridge",
                        "motif_id": int(motif_id),
                        "count": int(count),
                        "fraction": float(count) / float(total),
                        "frame_total_samples": int(total),
                    }
                )
        unknown_fraction = float((frame["stable_id"].to_numpy(dtype=np.int64) < 0).mean())
        if unknown_fraction > 0.0:
            rows.append(
                {
                    "source_path": str(source_path),
                    "frame_index": int(frame_index),
                    "timestep": int(timestep),
                    "motif_family": "unknown",
                    "motif_id": -1,
                    "count": int((frame["stable_id"].to_numpy(dtype=np.int64) < 0).sum()),
                    "fraction": float(unknown_fraction),
                    "frame_total_samples": int(total),
                }
            )
    return pd.DataFrame(rows)


def compute_transition_tables(
    sample_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = {
        "source_path",
        "center_atom_id",
        "frame_index",
        "timestep",
        "resolved_family",
        "resolved_motif_id",
        "resolved_state_label",
    }
    missing = sorted(required.difference(sample_df.columns))
    if missing:
        raise KeyError(
            "compute_transition_tables requires sample_df columns "
            f"{sorted(required)}, missing {missing}."
        )
    if sample_df.empty:
        empty = pd.DataFrame()
        metrics = {
            "transition_entropy": float("nan"),
            "self_transition_fraction": float("nan"),
        }
        return empty, empty, empty, metrics

    ordered = sample_df.sort_values(
        ["source_path", "center_atom_id", "frame_index", "timestep", "sample_index"]
    ).reset_index(drop=True)
    transition_rows: list[dict[str, Any]] = []
    triplet_rows: list[dict[str, Any]] = []
    for (_, _), track in ordered.groupby(["source_path", "center_atom_id"], sort=False):
        track = track.reset_index(drop=True)
        if track.shape[0] < 2:
            continue
        for idx in range(track.shape[0] - 1):
            current = track.iloc[idx]
            nxt = track.iloc[idx + 1]
            transition_rows.append(
                {
                    "source_path": str(current["source_path"]),
                    "center_atom_id": int(current["center_atom_id"]),
                    "frame_index_from": int(current["frame_index"]),
                    "frame_index_to": int(nxt["frame_index"]),
                    "timestep_from": int(current["timestep"]),
                    "timestep_to": int(nxt["timestep"]),
                    "frame_gap": int(nxt["frame_index"]) - int(current["frame_index"]),
                    "timestep_gap": int(nxt["timestep"]) - int(current["timestep"]),
                    "from_family": str(current["resolved_family"]),
                    "from_id": int(current["resolved_motif_id"]),
                    "to_family": str(nxt["resolved_family"]),
                    "to_id": int(nxt["resolved_motif_id"]),
                    "from_label": str(current["resolved_state_label"]),
                    "to_label": str(nxt["resolved_state_label"]),
                }
            )
        if track.shape[0] < 3:
            continue
        for idx in range(track.shape[0] - 2):
            a = track.iloc[idx]
            b = track.iloc[idx + 1]
            c = track.iloc[idx + 2]
            if (
                str(a["resolved_family"]) == "stable"
                and str(b["resolved_family"]) == "bridge"
                and str(c["resolved_family"]) == "stable"
            ):
                triplet_rows.append(
                    {
                        "source_path": str(a["source_path"]),
                        "center_atom_id": int(a["center_atom_id"]),
                        "stable_from_id": int(a["resolved_motif_id"]),
                        "bridge_id": int(b["resolved_motif_id"]),
                        "stable_to_id": int(c["resolved_motif_id"]),
                        "triplet_label": (
                            f"S{int(a['resolved_motif_id'])}->"
                            f"B{int(b['resolved_motif_id'])}->"
                            f"S{int(c['resolved_motif_id'])}"
                        ),
                    }
                )

    transition_events = pd.DataFrame(transition_rows)
    bridge_triplets = pd.DataFrame(triplet_rows)
    if transition_events.empty:
        empty = pd.DataFrame()
        metrics = {
            "transition_entropy": float("nan"),
            "self_transition_fraction": float("nan"),
        }
        return empty, empty, bridge_triplets, metrics

    counts = (
        transition_events.groupby(
            ["from_family", "from_id", "to_family", "to_id", "from_label", "to_label"],
            sort=True,
        )
        .size()
        .reset_index(name="count")
        .sort_values(["count", "from_label", "to_label"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    counts["total_transitions"] = int(counts["count"].sum())
    counts["fraction"] = counts["count"].astype(np.float64) / float(counts["count"].sum())

    probs = counts.copy()
    probs["from_total"] = probs.groupby(["from_family", "from_id"], sort=False)["count"].transform("sum")
    probs["transition_probability"] = probs["count"].astype(np.float64) / probs["from_total"].astype(np.float64)

    self_mask = (
        (counts["from_family"] == counts["to_family"])
        & (counts["from_id"] == counts["to_id"])
    )
    metrics = {
        "transition_entropy": _safe_entropy(counts["fraction"].to_numpy(dtype=np.float64)),
        "self_transition_fraction": float(counts.loc[self_mask, "count"].sum()) / float(counts["count"].sum()),
    }
    return counts, probs, bridge_triplets, metrics


def compute_dwell_tables(
    sample_df: pd.DataFrame,
    *,
    dwell_min_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = {
        "source_path",
        "center_atom_id",
        "frame_index",
        "timestep",
        "resolved_family",
        "resolved_motif_id",
    }
    missing = sorted(required.difference(sample_df.columns))
    if missing:
        raise KeyError(
            "compute_dwell_tables requires sample_df columns "
            f"{sorted(required)}, missing {missing}."
        )
    ordered = sample_df.sort_values(
        ["source_path", "center_atom_id", "frame_index", "timestep", "sample_index"]
    ).reset_index(drop=True)
    dwell_rows: list[dict[str, Any]] = []
    for (source_path, center_atom_id), track in ordered.groupby(["source_path", "center_atom_id"], sort=False):
        track = track.reset_index(drop=True)
        if track.empty:
            continue
        run_start = 0
        for idx in range(1, track.shape[0] + 1):
            is_boundary = idx == track.shape[0]
            if not is_boundary:
                prev = track.iloc[idx - 1]
                current = track.iloc[idx]
                is_boundary = not (
                    str(prev["resolved_family"]) == str(current["resolved_family"])
                    and int(prev["resolved_motif_id"]) == int(current["resolved_motif_id"])
                )
            if not is_boundary:
                continue
            run = track.iloc[run_start:idx]
            run_length = int(run.shape[0])
            if run_length >= int(dwell_min_length):
                dwell_rows.append(
                    {
                        "source_path": str(source_path),
                        "center_atom_id": int(center_atom_id),
                        "motif_family": str(run.iloc[0]["resolved_family"]),
                        "motif_id": int(run.iloc[0]["resolved_motif_id"]),
                        "start_frame_index": int(run.iloc[0]["frame_index"]),
                        "end_frame_index": int(run.iloc[-1]["frame_index"]),
                        "start_timestep": int(run.iloc[0]["timestep"]),
                        "end_timestep": int(run.iloc[-1]["timestep"]),
                        "dwell_length": int(run_length),
                    }
                )
            run_start = idx

    dwell_df = pd.DataFrame(dwell_rows)
    if dwell_df.empty:
        empty = pd.DataFrame()
        metrics = {
            "mean_dwell_overall": float("nan"),
            "median_dwell_overall": float("nan"),
        }
        return dwell_df, empty, empty, metrics

    survival_rows: list[dict[str, Any]] = []
    hazard_rows: list[dict[str, Any]] = []
    for (motif_family, motif_id), motif_runs in dwell_df.groupby(["motif_family", "motif_id"], sort=True):
        lengths = motif_runs["dwell_length"].to_numpy(dtype=np.int64)
        max_len = int(lengths.max())
        for elapsed in range(1, max_len + 1):
            at_risk = int((lengths >= elapsed).sum())
            exits = int((lengths == elapsed).sum())
            if at_risk <= 0:
                continue
            survival_rows.append(
                {
                    "motif_family": str(motif_family),
                    "motif_id": int(motif_id),
                    "elapsed_dwell": int(elapsed),
                    "at_risk_count": int(at_risk),
                    "survival_probability": float(at_risk) / float(lengths.size),
                }
            )
            hazard_rows.append(
                {
                    "motif_family": str(motif_family),
                    "motif_id": int(motif_id),
                    "elapsed_dwell": int(elapsed),
                    "at_risk_count": int(at_risk),
                    "exit_count": int(exits),
                    "hazard": float(exits) / float(at_risk),
                }
            )

    metrics = {
        "mean_dwell_overall": float(dwell_df["dwell_length"].mean()),
        "median_dwell_overall": float(dwell_df["dwell_length"].median()),
    }
    return dwell_df, pd.DataFrame(survival_rows), pd.DataFrame(hazard_rows), metrics


def compute_bridge_tables(
    sample_df: pd.DataFrame,
    *,
    dwell_df: pd.DataFrame,
    transition_counts: pd.DataFrame,
    bridge_triplets: pd.DataFrame,
    bridge_min_support: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if "bridge_id" not in sample_df.columns or "bridge_active" not in sample_df.columns:
        return pd.DataFrame(), pd.DataFrame(), {
            "bridge_fraction": float("nan"),
            "num_bridge_motifs": 0,
        }

    active_bridge = sample_df.loc[sample_df["bridge_active"].astype(bool)].copy()
    if active_bridge.empty:
        return pd.DataFrame(), bridge_triplets, {
            "bridge_fraction": 0.0,
            "num_bridge_motifs": 0,
        }

    bridge_conf = (
        active_bridge["bridge_confidence"].to_numpy(dtype=np.float64)
        if "bridge_confidence" in active_bridge.columns
        else None
    )
    rows: list[dict[str, Any]] = []
    dwell_bridge = dwell_df.loc[dwell_df["motif_family"] == "bridge"].copy()
    incoming = (
        transition_counts.loc[
            (transition_counts["to_family"] == "bridge")
            & (transition_counts["from_family"] == "stable")
        ]
        if not transition_counts.empty
        else pd.DataFrame()
    )
    outgoing = (
        transition_counts.loc[
            (transition_counts["from_family"] == "bridge")
            & (transition_counts["to_family"] == "stable")
        ]
        if not transition_counts.empty
        else pd.DataFrame()
    )

    for bridge_id, motif_df in active_bridge.groupby("bridge_id", sort=True):
        motif_df = motif_df.reset_index(drop=True)
        motif_dwell = dwell_bridge.loc[dwell_bridge["motif_id"] == int(bridge_id)]
        incoming_counter = Counter()
        outgoing_counter = Counter()
        if not incoming.empty:
            incoming_rows = incoming.loc[incoming["to_id"] == int(bridge_id)]
            incoming_counter.update(
                {
                    int(row.from_id): int(row.count)
                    for row in incoming_rows.itertuples(index=False)
                }
            )
        if not outgoing.empty:
            outgoing_rows = outgoing.loc[outgoing["from_id"] == int(bridge_id)]
            outgoing_counter.update(
                {
                    int(row.to_id): int(row.count)
                    for row in outgoing_rows.itertuples(index=False)
                }
            )
        rows.append(
            {
                "motif_family": "bridge",
                "motif_id": int(bridge_id),
                "support_count": int(motif_df.shape[0]),
                "fraction_active": float(motif_df.shape[0]) / float(active_bridge.shape[0]),
                "mean_confidence": (
                    float(motif_df["bridge_confidence"].mean())
                    if "bridge_confidence" in motif_df.columns
                    else float("nan")
                ),
                "median_lifetime": (
                    float(motif_dwell["dwell_length"].median())
                    if not motif_dwell.empty
                    else float("nan")
                ),
                "distinct_track_count": int(
                    motif_df.loc[:, ["source_path", "center_atom_id"]].drop_duplicates().shape[0]
                ),
                "top_incoming_stable_id": (
                    int(max(incoming_counter, key=incoming_counter.get))
                    if incoming_counter
                    else -1
                ),
                "top_outgoing_stable_id": (
                    int(max(outgoing_counter, key=outgoing_counter.get))
                    if outgoing_counter
                    else -1
                ),
                "meets_min_support": bool(int(motif_df.shape[0]) >= int(bridge_min_support)),
            }
        )

    usage_df = pd.DataFrame(rows).sort_values(["support_count", "motif_id"], ascending=[False, True])
    metrics = {
        "bridge_fraction": float(active_bridge.shape[0]) / float(sample_df.shape[0]),
        "num_bridge_motifs": int(usage_df.shape[0]),
    }
    return usage_df, bridge_triplets, metrics


def compute_recurrence_tables(
    dwell_df: pd.DataFrame,
    *,
    recurrence_max_gap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if dwell_df.empty:
        empty = pd.DataFrame()
        return empty, empty, {"revisit_rate": float("nan")}

    revisit_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []

    for (motif_family, motif_id), motif_runs in dwell_df.groupby(["motif_family", "motif_id"], sort=True):
        motif_runs = motif_runs.sort_values(
            ["source_path", "center_atom_id", "start_frame_index", "end_frame_index"]
        )
        total_runs = int(motif_runs.shape[0])
        revisit_events = 0
        unique_tracks = 0
        for (_, _), track_runs in motif_runs.groupby(["source_path", "center_atom_id"], sort=False):
            track_runs = track_runs.reset_index(drop=True)
            local_revisit = 0
            if track_runs.shape[0] >= 2:
                for idx in range(track_runs.shape[0] - 1):
                    current = track_runs.iloc[idx]
                    nxt = track_runs.iloc[idx + 1]
                    gap = int(nxt["start_frame_index"]) - int(current["end_frame_index"]) - 1
                    revisit_rows.append(
                        {
                            "motif_family": str(motif_family),
                            "motif_id": int(motif_id),
                            "source_path": str(current["source_path"]),
                            "center_atom_id": int(current["center_atom_id"]),
                            "gap_frames": int(gap),
                            "returned_within_limit": bool(gap <= int(recurrence_max_gap)),
                        }
                    )
                    if gap <= int(recurrence_max_gap):
                        revisit_events += 1
                        local_revisit += 1
            if local_revisit > 0:
                unique_tracks += 1
        score_rows.append(
            {
                "motif_family": str(motif_family),
                "motif_id": int(motif_id),
                "total_runs": int(total_runs),
                "revisit_events": int(revisit_events),
                "unique_tracks_with_revisit": int(unique_tracks),
                "recurrence_score": float(revisit_events) / float(total_runs) if total_runs > 0 else float("nan"),
            }
        )

    score_df = pd.DataFrame(score_rows)
    revisit_df = pd.DataFrame(revisit_rows)
    revisit_rate = (
        float(score_df["revisit_events"].sum()) / float(score_df["total_runs"].sum())
        if not score_df.empty and float(score_df["total_runs"].sum()) > 0.0
        else float("nan")
    )
    return score_df, revisit_df, {"revisit_rate": revisit_rate}


def compute_prediction_metrics(sample_df: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "future_top1_acc_by_lag": {},
        "future_nll_by_lag": {},
    }
    for column in sorted(sample_df.columns):
        prefix = "future_pred_ids_lag_"
        if not column.startswith(prefix):
            continue
        lag = column[len(prefix) :]
        target_col = f"future_target_ids_lag_{lag}"
        nll_col = f"future_nll_lag_{lag}"
        if target_col not in sample_df.columns:
            continue
        pred = sample_df[column].to_numpy(dtype=np.int64)
        target = sample_df[target_col].to_numpy(dtype=np.int64)
        metrics["future_top1_acc_by_lag"][str(lag)] = float((pred == target).mean())
        if nll_col in sample_df.columns:
            metrics["future_nll_by_lag"][str(lag)] = float(
                sample_df[nll_col].to_numpy(dtype=np.float64).mean()
            )
    return metrics


def compute_hazard_calibration(sample_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {
        "hazard_brier_by_lag": {},
        "hazard_auroc_by_lag": {},
    }
    for column in sorted(sample_df.columns):
        prefix = "hazard_probs_lag_"
        if not column.startswith(prefix):
            continue
        lag = column[len(prefix) :]
        target_col = f"change_target_lag_{lag}"
        if target_col not in sample_df.columns:
            continue
        probs = sample_df[column].to_numpy(dtype=np.float64)
        targets = sample_df[target_col].to_numpy(dtype=np.float64)
        metrics["hazard_brier_by_lag"][str(lag)] = float(np.mean((probs - targets) ** 2))
        metrics["hazard_auroc_by_lag"][str(lag)] = _binary_roc_auc(probs, targets.astype(np.int64))
        bin_edges = np.linspace(0.0, 1.0, 11)
        bin_indices = np.clip(np.digitize(probs, bin_edges, right=True) - 1, 0, 9)
        for bin_idx in range(10):
            mask = bin_indices == bin_idx
            if not np.any(mask):
                continue
            rows.append(
                {
                    "lag": int(lag),
                    "bin_index": int(bin_idx),
                    "bin_left": float(bin_edges[bin_idx]),
                    "bin_right": float(bin_edges[bin_idx + 1]),
                    "sample_count": int(mask.sum()),
                    "predicted_mean": float(probs[mask].mean()),
                    "empirical_rate": float(targets[mask].mean()),
                    "brier_score": float(np.mean((probs[mask] - targets[mask]) ** 2)),
                }
            )
    return pd.DataFrame(rows), metrics


def compute_field_gain(sample_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {"neighbor_gain_by_lag": {}}
    for column in sorted(sample_df.columns):
        prefix = "field_pred_ids_lag_"
        if not column.startswith(prefix):
            continue
        lag = column[len(prefix) :]
        baseline_col = f"future_pred_ids_lag_{lag}"
        target_col = f"future_target_ids_lag_{lag}"
        if baseline_col not in sample_df.columns or target_col not in sample_df.columns:
            continue
        field_acc = float(
            (
                sample_df[column].to_numpy(dtype=np.int64)
                == sample_df[target_col].to_numpy(dtype=np.int64)
            ).mean()
        )
        baseline_acc = float(
            (
                sample_df[baseline_col].to_numpy(dtype=np.int64)
                == sample_df[target_col].to_numpy(dtype=np.int64)
            ).mean()
        )
        rows.append(
            {
                "lag": int(lag),
                "target_motif_id": -1,
                "baseline_acc": float(baseline_acc),
                "field_acc": float(field_acc),
                "gain": float(field_acc - baseline_acc),
            }
        )
        for target_id, motif_df in sample_df.groupby(target_col, sort=True):
            field_motif_acc = float(
                (
                    motif_df[column].to_numpy(dtype=np.int64)
                    == motif_df[target_col].to_numpy(dtype=np.int64)
                ).mean()
            )
            baseline_motif_acc = float(
                (
                    motif_df[baseline_col].to_numpy(dtype=np.int64)
                    == motif_df[target_col].to_numpy(dtype=np.int64)
                ).mean()
            )
            rows.append(
                {
                    "lag": int(lag),
                    "target_motif_id": int(target_id),
                    "baseline_acc": float(baseline_motif_acc),
                    "field_acc": float(field_motif_acc),
                    "gain": float(field_motif_acc - baseline_motif_acc),
                }
            )
        metrics["neighbor_gain_by_lag"][str(lag)] = float(field_acc - baseline_acc)
    return pd.DataFrame(rows), metrics


def build_transition_matrix(
    transition_counts: pd.DataFrame,
    *,
    state_ids: list[tuple[str, int]],
) -> tuple[np.ndarray, list[str]]:
    labels = [_state_to_label(family, motif_id) for family, motif_id in state_ids]
    label_to_index = {state: idx for idx, state in enumerate(state_ids)}
    matrix = np.zeros((len(state_ids), len(state_ids)), dtype=np.float64)
    for row in transition_counts.itertuples(index=False):
        key_from = (str(row.from_family), int(row.from_id))
        key_to = (str(row.to_family), int(row.to_id))
        if key_from not in label_to_index or key_to not in label_to_index:
            continue
        matrix[label_to_index[key_from], label_to_index[key_to]] = float(row.count)
    return matrix, labels


def build_bridge_event_windows(
    sample_df: pd.DataFrame,
    *,
    window_radius: int,
    max_events: int,
) -> list[dict[str, Any]]:
    if "bridge_active" not in sample_df.columns:
        return []
    ordered = sample_df.sort_values(
        ["source_path", "center_atom_id", "frame_index", "timestep", "sample_index"]
    ).reset_index(drop=True)
    bridge_rows = ordered.loc[ordered["bridge_active"].astype(bool)].copy()
    if bridge_rows.empty:
        return []
    if "bridge_confidence" in bridge_rows.columns:
        bridge_rows = bridge_rows.sort_values(["bridge_confidence", "sample_index"], ascending=[False, True])
    else:
        bridge_rows = bridge_rows.sort_values(["sample_index"], ascending=[True])

    events: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for row in bridge_rows.itertuples(index=False):
        key = (str(row.source_path), int(row.center_atom_id), int(row.frame_index))
        if key in seen:
            continue
        seen.add(key)
        track = ordered.loc[
            (ordered["source_path"] == row.source_path)
            & (ordered["center_atom_id"] == row.center_atom_id)
        ].sort_values(["frame_index", "timestep", "sample_index"])
        event_track = track.loc[
            (track["frame_index"] >= int(row.frame_index) - int(window_radius))
            & (track["frame_index"] <= int(row.frame_index) + int(window_radius))
        ]
        events.append(
            {
                "source_path": str(row.source_path),
                "center_atom_id": int(row.center_atom_id),
                "anchor_frame_index": int(row.frame_index),
                "anchor_bridge_id": int(row.bridge_id),
                "anchor_bridge_confidence": (
                    float(row.bridge_confidence)
                    if hasattr(row, "bridge_confidence")
                    else float("nan")
                ),
                "timeline": [
                    {
                        "frame_index": int(sample.frame_index),
                        "resolved_state_label": str(sample.resolved_state_label),
                        "bridge_active": bool(sample.bridge_active),
                    }
                    for sample in event_track.itertuples(index=False)
                ],
            }
        )
        if len(events) >= int(max_events):
            break
    return events


__all__ = [
    "build_bridge_event_windows",
    "build_transition_matrix",
    "build_usage_table",
    "compute_bridge_tables",
    "compute_dwell_tables",
    "compute_field_gain",
    "compute_frame_motif_proportions",
    "compute_hazard_calibration",
    "compute_prediction_metrics",
    "compute_recurrence_tables",
    "compute_transition_tables",
]
