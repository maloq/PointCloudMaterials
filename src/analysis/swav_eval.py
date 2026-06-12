from __future__ import annotations

import csv
from dataclasses import replace
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.vis_tools.latent_analysis_vis import save_md_space_clusters_plot
from src.vis_tools.md_cluster_plot import save_interactive_md_plot

from .cluster_colors import _build_cluster_color_map
from .cluster_geometry import _compute_cluster_representative_indices
from .config import _cfg_bool, _cfg_int, _cfg_select
from .figure_sets import render_cluster_figure_outputs
from .output_layout import write_json


def _swav_is_available(model: Any) -> bool:
    swav = getattr(model, "swav", None)
    return bool(
        swav is not None
        and getattr(swav, "projector", None) is not None
        and getattr(swav, "prototypes", None) is not None
    )


def _resolve_swav(model: Any):
    swav = getattr(model, "swav", None)
    if swav is None:
        raise RuntimeError(
            "SwAV prototype evaluation was requested, but the loaded model has no 'swav' module."
        )
    if getattr(swav, "projector", None) is None or getattr(swav, "prototypes", None) is None:
        raise RuntimeError(
            "SwAV prototype evaluation was requested, but SwAV projector/prototypes are not initialized. "
            "Check that the checkpoint was trained with swav_enabled=true and swav_weight > 0."
        )
    return swav


def _prototype_device(swav: Any) -> torch.device:
    return next(swav.parameters()).device


def compute_swav_assignments(
    model: Any,
    latents: np.ndarray,
    *,
    batch_size: int = 8192,
    assignment_method: str = "sinkhorn",
    sinkhorn_iterations: int | None = None,
    assignment_device: str = "auto",
) -> dict[str, np.ndarray]:
    """Run the learned SwAV projector/prototype head on cached invariant latents."""
    swav = _resolve_swav(model)
    latents_arr = np.asarray(latents, dtype=np.float32)
    assignment_method = str(assignment_method).strip().lower()
    assignment_device_norm = str(assignment_device).strip().lower()
    if assignment_device_norm not in {"auto", "cpu", "gpu", "cuda"}:
        raise ValueError(
            "swav.assignment_device must be one of ['auto', 'cpu', 'gpu'], "
            f"got {assignment_device!r}."
        )

    if hasattr(swav, "normalize_prototypes"):
        swav.normalize_prototypes()

    temperature = float(getattr(swav, "temperature", 1.0))

    device = _prototype_device(swav)
    use_device_sinkhorn = bool(
        assignment_method == "sinkhorn"
        and device.type == "cuda"
        and assignment_device_norm in {"auto", "gpu", "cuda"}
    )
    if assignment_method == "sinkhorn" and assignment_device_norm in {"gpu", "cuda"} and device.type != "cuda":
        raise RuntimeError(
            "swav.assignment_device='gpu' was requested, but SwAV prototypes are on "
            f"{device}. Move the model to CUDA or set swav.assignment_device='cpu'."
        )
    logits_parts: list[torch.Tensor] = []
    softmax_probs_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, int(latents_arr.shape[0]), int(batch_size)):
            end = min(start + int(batch_size), int(latents_arr.shape[0]))
            batch = torch.from_numpy(latents_arr[start:end]).to(device=device, non_blocking=True)
            logits = swav._prototype_logits(batch)
            logits_parts.append(
                logits.detach()
                if use_device_sinkhorn
                else logits.detach().cpu()
            )
            if assignment_method == "softmax":
                softmax_probs = F.softmax(logits / temperature, dim=1)
                softmax_probs_parts.append(softmax_probs.detach().cpu())

    logits_tensor = torch.cat(logits_parts, dim=0)
    if assignment_method == "sinkhorn":
        eval_sinkhorn_iterations = (
            max(20, int(getattr(swav, "sinkhorn_iterations", 0)))
            if sinkhorn_iterations is None
            else int(sinkhorn_iterations)
        )
        with torch.inference_mode():
            probs_tensor = swav._sinkhorn(logits_tensor, iterations=eval_sinkhorn_iterations)
    else:
        eval_sinkhorn_iterations = None
        softmax_probs_tensor = torch.cat(softmax_probs_parts, dim=0)
        probs_tensor = softmax_probs_tensor

    logits_np = logits_tensor.detach().cpu().numpy().astype(np.float32)
    probs_np = probs_tensor.detach().cpu().numpy().astype(np.float32)
    labels = np.argmax(probs_np, axis=1).astype(np.int64)
    confidence = np.max(probs_np, axis=1).astype(np.float32)
    sorted_probs = np.sort(probs_np, axis=1)
    margin = (
        sorted_probs[:, -1] - sorted_probs[:, -2]
        if probs_np.shape[1] >= 2
        else np.ones((probs_np.shape[0],), dtype=np.float32)
    ).astype(np.float32)
    entropy = -(probs_np * np.log(np.clip(probs_np, 1e-12, None))).sum(axis=1).astype(np.float32)
    return {
        "logits": logits_np,
        "probs": probs_np,
        "labels": labels,
        "confidence": confidence,
        "margin": margin,
        "entropy": entropy,
        "assignment_method": np.asarray(assignment_method),
        "sinkhorn_iterations": np.asarray(
            -1 if eval_sinkhorn_iterations is None else eval_sinkhorn_iterations
        ),
    }


def _entropy_from_probs(probs: np.ndarray) -> float:
    p = np.asarray(probs, dtype=np.float64)
    positive = p[p > 0.0]
    if positive.size == 0:
        return 0.0
    return float(-(positive * np.log(positive)).sum())


def _usage_summary(assignments: dict[str, np.ndarray]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    probs = np.asarray(assignments["probs"], dtype=np.float64)
    labels = np.asarray(assignments["labels"], dtype=int)
    confidence = np.asarray(assignments["confidence"], dtype=np.float64)
    margin = np.asarray(assignments["margin"], dtype=np.float64)
    entropy = np.asarray(assignments["entropy"], dtype=np.float64)
    num_samples = int(labels.shape[0])
    num_prototypes = int(probs.shape[1])
    counts = np.bincount(labels, minlength=num_prototypes).astype(np.int64)
    fractions = counts.astype(np.float64) / float(max(num_samples, 1))
    mean_probs = probs.mean(axis=0)
    hard_entropy = _entropy_from_probs(fractions)
    soft_entropy = _entropy_from_probs(mean_probs)
    rows: list[dict[str, Any]] = []
    for proto_id in range(num_prototypes):
        mask = labels == proto_id
        rows.append(
            {
                "prototype_id": int(proto_id),
                "hard_count": int(counts[proto_id]),
                "hard_fraction": float(fractions[proto_id]),
                "mean_probability": float(mean_probs[proto_id]),
                "mean_confidence_for_assigned": (
                    float(confidence[mask].mean()) if np.any(mask) else float("nan")
                ),
                "mean_entropy_for_assigned": (
                    float(entropy[mask].mean()) if np.any(mask) else float("nan")
                ),
            }
        )
    metrics = {
        "num_samples": int(num_samples),
        "num_prototypes": int(num_prototypes),
        "assignment_method": str(np.asarray(assignments.get("assignment_method", "unknown")).item()),
        "sinkhorn_iterations": int(np.asarray(assignments.get("sinkhorn_iterations", -1)).item()),
        "dead_hard_prototypes": [int(v) for v in np.where(counts == 0)[0].tolist()],
        "hard_usage_entropy": float(hard_entropy),
        "soft_usage_entropy": float(soft_entropy),
        "hard_effective_prototypes": float(math.exp(hard_entropy)),
        "soft_effective_prototypes": float(math.exp(soft_entropy)),
        "mean_assignment_confidence": float(confidence.mean()),
        "median_assignment_confidence": float(np.median(confidence)),
        "mean_assignment_margin": float(margin.mean()),
        "median_assignment_margin": float(np.median(margin)),
        "mean_assignment_entropy": float(entropy.mean()),
        "median_assignment_entropy": float(np.median(entropy)),
    }
    return metrics, rows


def _write_dict_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot write {path}: no rows and no fieldnames were provided.")
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _optional_per_sample_cache_values(
    cache: dict[str, np.ndarray],
    key: str,
    *,
    num_samples: int,
    dtype: Any,
    context: str,
) -> np.ndarray | None:
    values = np.asarray(cache[key], dtype=dtype).reshape(-1)
    if values.size == 0:
        return None
    if values.shape[0] != int(num_samples):
        raise ValueError(
            f"{context} expected cache[{key!r}] to be empty or have one value per sample "
            f"({int(num_samples)}), got shape={tuple(np.asarray(cache[key]).shape)}."
        )
    return values


def _build_contingency(
    *,
    prototype_labels: np.ndarray,
    cluster_labels: np.ndarray,
    num_prototypes: int | None = None,
) -> tuple[list[int], list[int], np.ndarray, np.ndarray]:
    proto_labels = np.asarray(prototype_labels, dtype=int).reshape(-1)
    cluster_labels_arr = np.asarray(cluster_labels, dtype=int).reshape(-1)
    if num_prototypes is None:
        prototype_ids = sorted(int(v) for v in np.unique(proto_labels) if int(v) >= 0)
    else:
        prototype_ids = list(range(int(num_prototypes)))
    cluster_ids = sorted(int(v) for v in np.unique(cluster_labels_arr) if int(v) >= 0)
    prototype_counts = np.asarray(
        [int(np.count_nonzero(proto_labels == int(proto_id))) for proto_id in prototype_ids],
        dtype=np.int64,
    )
    contingency = np.zeros((len(prototype_ids), len(cluster_ids)), dtype=np.int64)
    if cluster_ids:
        proto_to_row = {int(value): int(pos) for pos, value in enumerate(prototype_ids)}
        cluster_to_col = {int(value): int(pos) for pos, value in enumerate(cluster_ids)}
        valid_mask = (proto_labels >= 0) & (cluster_labels_arr >= 0)
        for proto_id, cluster_id in zip(proto_labels[valid_mask], cluster_labels_arr[valid_mask]):
            contingency[proto_to_row[int(proto_id)], cluster_to_col[int(cluster_id)]] += 1
    return prototype_ids, cluster_ids, prototype_counts, contingency


def _hungarian_correspondence_pairs(
    contingency: np.ndarray,
) -> list[tuple[int, int, int]]:
    matrix = np.asarray(contingency, dtype=np.int64)
    max_count = int(np.max(matrix))
    row_indices, col_indices = linear_sum_assignment(max_count - matrix)
    return [
        (int(row), int(col), int(matrix[int(row), int(col)]))
        for row, col in zip(row_indices, col_indices, strict=True)
    ]


def _compute_prototype_cluster_correspondence(
    *,
    prototype_labels: np.ndarray,
    cluster_labels: np.ndarray,
    num_prototypes: int,
    cluster_color_map: dict[int, str] | None,
) -> dict[str, Any]:
    prototype_ids, cluster_ids, prototype_counts, contingency = _build_contingency(
        prototype_labels=prototype_labels,
        cluster_labels=cluster_labels,
        num_prototypes=int(num_prototypes),
    )
    fallback_prototype_color_map = _build_cluster_color_map(
        np.arange(int(num_prototypes), dtype=int)
    )
    cluster_totals = contingency.sum(axis=0).astype(np.int64)
    nonnegative_cluster_sample_count = int(cluster_totals.sum())
    hungarian_pairs_pos = _hungarian_correspondence_pairs(contingency)
    hungarian_by_proto: dict[int, tuple[int, int]] = {}
    hungarian_by_cluster: dict[int, tuple[int, int]] = {}
    for row_pos, col_pos, count in hungarian_pairs_pos:
        proto_id = int(prototype_ids[int(row_pos)])
        cluster_id = int(cluster_ids[int(col_pos)])
        hungarian_by_proto[proto_id] = (cluster_id, int(count))
        hungarian_by_cluster[cluster_id] = (proto_id, int(count))

    best_by_proto: dict[int, tuple[int, int]] = {}
    prototype_rows: list[dict[str, Any]] = []
    prototype_color_map: dict[int, str] = {
        int(proto_id): str(fallback_prototype_color_map[int(proto_id)])
        for proto_id in prototype_ids
    }
    prototype_to_cluster: dict[int, int] = {}
    assigned_pairs: list[dict[str, Any]] = []
    assigned_total_overlap = 0
    for row_pos, proto_id in enumerate(prototype_ids):
        row = contingency[int(row_pos), :]
        prototype_sample_count = int(prototype_counts[int(row_pos)])
        nonnoise_overlap_count = int(row.sum())
        best_cluster_id: int | None = None
        best_count = 0
        if row.size > 0 and int(np.max(row)) > 0:
            best_count = int(np.max(row))
            best_col = int(np.flatnonzero(row == best_count)[0])
            best_cluster_id = int(cluster_ids[best_col])
            best_by_proto[int(proto_id)] = (best_cluster_id, best_count)

        hungarian_cluster_id: int | None = None
        hungarian_count = 0
        if int(proto_id) in hungarian_by_proto:
            hungarian_cluster_id, hungarian_count = hungarian_by_proto[int(proto_id)]

        assigned_cluster_id: int | None = None
        assigned_count = 0
        assignment_source = "none"
        if hungarian_cluster_id is not None and int(hungarian_count) > 0:
            assigned_cluster_id = int(hungarian_cluster_id)
            assigned_count = int(hungarian_count)
            assignment_source = "hungarian_one_to_one"
        elif best_cluster_id is not None and int(best_count) > 0:
            assigned_cluster_id = int(best_cluster_id)
            assigned_count = int(best_count)
            assignment_source = "best_overlap_fallback"

        assigned_color = ""
        if assigned_cluster_id is not None:
            prototype_to_cluster[int(proto_id)] = int(assigned_cluster_id)
            assigned_total_overlap += int(assigned_count)
            if cluster_color_map is not None and int(assigned_cluster_id) in cluster_color_map:
                assigned_color = str(cluster_color_map[int(assigned_cluster_id)])
                prototype_color_map[int(proto_id)] = assigned_color
            assigned_pairs.append(
                {
                    "prototype_id": int(proto_id),
                    "cluster_id": int(assigned_cluster_id),
                    "count": int(assigned_count),
                    "source": str(assignment_source),
                }
            )

        best_cluster_total = (
            int(cluster_totals[cluster_ids.index(int(best_cluster_id))])
            if best_cluster_id is not None
            else 0
        )
        prototype_rows.append(
            {
                "prototype_id": int(proto_id),
                "prototype_sample_count": int(prototype_sample_count),
                "nonnoise_overlap_count": int(nonnoise_overlap_count),
                "best_overlap_cluster_id": "" if best_cluster_id is None else int(best_cluster_id),
                "best_overlap_count": int(best_count),
                "best_overlap_fraction_of_prototype": (
                    float(best_count / prototype_sample_count)
                    if prototype_sample_count > 0
                    else float("nan")
                ),
                "best_overlap_fraction_of_cluster": (
                    float(best_count / best_cluster_total)
                    if best_cluster_total > 0
                    else float("nan")
                ),
                "hungarian_cluster_id": (
                    "" if hungarian_cluster_id is None else int(hungarian_cluster_id)
                ),
                "hungarian_overlap_count": int(hungarian_count),
                "assigned_cluster_id": (
                    "" if assigned_cluster_id is None else int(assigned_cluster_id)
                ),
                "assignment_source": str(assignment_source),
                "assigned_cluster_color": str(assigned_color),
                "prototype_color": str(prototype_color_map[int(proto_id)]),
            }
        )

    cluster_rows: list[dict[str, Any]] = []
    cluster_to_prototype: dict[int, int] = {}
    for col_pos, cluster_id in enumerate(cluster_ids):
        column = contingency[:, int(col_pos)]
        cluster_sample_count = int(cluster_totals[int(col_pos)])
        best_proto_id: int | None = None
        best_count = 0
        if column.size > 0 and int(np.max(column)) > 0:
            best_count = int(np.max(column))
            best_row = int(np.flatnonzero(column == best_count)[0])
            best_proto_id = int(prototype_ids[best_row])
            cluster_to_prototype[int(cluster_id)] = int(best_proto_id)
        hungarian_proto_id: int | None = None
        hungarian_count = 0
        if int(cluster_id) in hungarian_by_cluster:
            hungarian_proto_id, hungarian_count = hungarian_by_cluster[int(cluster_id)]
        cluster_rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_sample_count": int(cluster_sample_count),
                "best_overlap_prototype_id": "" if best_proto_id is None else int(best_proto_id),
                "best_overlap_count": int(best_count),
                "best_overlap_fraction_of_cluster": (
                    float(best_count / cluster_sample_count)
                    if cluster_sample_count > 0
                    else float("nan")
                ),
                "hungarian_prototype_id": (
                    "" if hungarian_proto_id is None else int(hungarian_proto_id)
                ),
                "hungarian_overlap_count": int(hungarian_count),
                "cluster_color": (
                    ""
                    if cluster_color_map is None or int(cluster_id) not in cluster_color_map
                    else str(cluster_color_map[int(cluster_id)])
                ),
            }
        )

    return {
        "sample_count": int(np.asarray(prototype_labels).reshape(-1).shape[0]),
        "nonnegative_cluster_sample_count": int(nonnegative_cluster_sample_count),
        "prototype_ids": [int(v) for v in prototype_ids],
        "cluster_ids": [int(v) for v in cluster_ids],
        "contingency": contingency.astype(int).tolist(),
        "prototype_rows": prototype_rows,
        "cluster_rows": cluster_rows,
        "prototype_to_cluster": {
            int(proto_id): int(cluster_id)
            for proto_id, cluster_id in prototype_to_cluster.items()
        },
        "cluster_to_prototype": {
            int(cluster_id): int(proto_id)
            for cluster_id, proto_id in cluster_to_prototype.items()
        },
        "best_overlap_prototype_to_cluster": {
            int(proto_id): int(cluster_id)
            for proto_id, (cluster_id, _count) in best_by_proto.items()
        },
        "hungarian_prototype_to_cluster": {
            int(proto_id): int(cluster_id)
            for proto_id, (cluster_id, count) in hungarian_by_proto.items()
            if int(count) > 0
        },
        "prototype_color_map": {
            int(proto_id): str(color) for proto_id, color in prototype_color_map.items()
        },
        "assigned_pairs": assigned_pairs,
        "hungarian_total_overlap": int(
            sum(int(count) for _row, _col, count in hungarian_pairs_pos)
        ),
        "assigned_total_overlap": int(assigned_total_overlap),
    }


def _write_assignment_csv(
    path: Path,
    assignments: dict[str, np.ndarray],
    cache: dict[str, np.ndarray],
    *,
    max_rows: int | None,
) -> dict[str, Any]:
    labels = np.asarray(assignments["labels"], dtype=int)
    confidence = np.asarray(assignments["confidence"], dtype=np.float32)
    margin = np.asarray(assignments["margin"], dtype=np.float32)
    entropy = np.asarray(assignments["entropy"], dtype=np.float32)
    coords = np.asarray(cache["coords"], dtype=np.float32)
    num_samples = int(labels.shape[0])
    phases = _optional_per_sample_cache_values(
        cache,
        "phases",
        num_samples=num_samples,
        dtype=np.int64,
        context="SwAV assignment CSV export",
    )
    instance_ids = _optional_per_sample_cache_values(
        cache,
        "instance_ids",
        num_samples=num_samples,
        dtype=np.int64,
        context="SwAV assignment CSV export",
    )
    has_phases = phases is not None
    instance_id_source = (
        "cache_instance_ids" if instance_ids is not None else "unavailable"
    )
    row_count = num_samples if max_rows is None else min(num_samples, int(max_rows))
    rows: list[dict[str, Any]] = []
    for idx in range(row_count):
        row = {
            "sample_index": int(idx),
            "prototype_id": int(labels[idx]),
            "confidence": float(confidence[idx]),
            "margin": float(margin[idx]),
            "entropy": float(entropy[idx]),
        }
        row.update(
            {
                "x": float(coords[idx, 0]),
                "y": float(coords[idx, 1]),
                "z": float(coords[idx, 2]),
                "instance_id": "" if instance_ids is None else int(instance_ids[idx]),
                "instance_id_source": str(instance_id_source),
            }
        )
        if has_phases:
            row["phase"] = int(phases[idx])
        rows.append(row)
    fieldnames = [
        "sample_index",
        "prototype_id",
        "confidence",
        "margin",
        "entropy",
        "x",
        "y",
        "z",
        "instance_id",
        "instance_id_source",
    ]
    if has_phases:
        fieldnames.append("phase")
    _write_dict_rows_csv(path, rows, fieldnames=fieldnames)
    return {
        "path": str(path),
        "rows_written": int(row_count),
        "truncated": bool(row_count < num_samples),
        "num_samples": int(num_samples),
        "instance_id_source": str(instance_id_source),
    }


def _representative_rows(
    assignments: dict[str, np.ndarray],
    cache: dict[str, np.ndarray],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    probs = np.asarray(assignments["probs"], dtype=np.float32)
    labels = np.asarray(assignments["labels"], dtype=int)
    confidence = np.asarray(assignments["confidence"], dtype=np.float32)
    coords = np.asarray(cache["coords"], dtype=np.float32)
    num_samples, num_prototypes = probs.shape
    phases = _optional_per_sample_cache_values(
        cache,
        "phases",
        num_samples=num_samples,
        dtype=np.int64,
        context="SwAV representative CSV export",
    )
    instance_ids = _optional_per_sample_cache_values(
        cache,
        "instance_ids",
        num_samples=num_samples,
        dtype=np.int64,
        context="SwAV representative CSV export",
    )
    has_phases = phases is not None
    instance_id_source = (
        "cache_instance_ids" if instance_ids is not None else "unavailable"
    )
    rows: list[dict[str, Any]] = []
    k = max(1, int(top_k))
    for proto_id in range(num_prototypes):
        order = np.argsort(probs[:, proto_id])[::-1][:k]
        for rank, sample_idx in enumerate(order, start=1):
            idx = int(sample_idx)
            row = {
                "prototype_id": int(proto_id),
                "rank": int(rank),
                "sample_index": int(idx),
                "score_probability": float(probs[idx, proto_id]),
                "argmax_prototype_id": int(labels[idx]),
                "argmax_confidence": float(confidence[idx]),
                "is_argmax": bool(int(labels[idx]) == int(proto_id)),
            }
            row.update(
                {
                    "x": float(coords[idx, 0]),
                    "y": float(coords[idx, 1]),
                    "z": float(coords[idx, 2]),
                    "instance_id": "" if instance_ids is None else int(instance_ids[idx]),
                    "instance_id_source": str(instance_id_source),
                }
            )
            if has_phases:
                row["phase"] = int(phases[idx])
            rows.append(row)
    return rows


def _frame_proportions(
    labels: np.ndarray,
    frame_groups: list[tuple[str, np.ndarray]] | None,
    *,
    num_prototypes: int,
) -> tuple[list[dict[str, Any]], np.ndarray | None, list[str]]:
    if not frame_groups:
        return [], None, []
    labels = np.asarray(labels, dtype=int)
    rows: list[dict[str, Any]] = []
    counts_matrix = np.zeros((len(frame_groups), int(num_prototypes)), dtype=np.int64)
    frame_names: list[str] = []
    for frame_pos, (frame_name, indices) in enumerate(frame_groups):
        idx = np.asarray(indices, dtype=int).reshape(-1)
        frame_labels = labels[idx]
        counts = np.bincount(frame_labels, minlength=int(num_prototypes)).astype(np.int64)
        counts_matrix[frame_pos, :] = counts
        total = int(frame_labels.shape[0])
        row: dict[str, Any] = {"frame": str(frame_name), "sample_count": total}
        for proto_id in range(int(num_prototypes)):
            row[f"prototype_{proto_id}_count"] = int(counts[proto_id])
            row[f"prototype_{proto_id}_fraction"] = float(counts[proto_id] / total) if total else float("nan")
        rows.append(row)
        frame_names.append(str(frame_name))
    return rows, counts_matrix, frame_names


def _transition_rows(
    labels: np.ndarray,
    instance_ids: np.ndarray,
    frame_groups: list[tuple[str, np.ndarray]] | None,
    *,
    num_prototypes: int,
) -> tuple[list[dict[str, Any]], str | None]:
    if not frame_groups:
        return [], "frame_groups_missing"
    labels = np.asarray(labels, dtype=int)
    ids = np.asarray(instance_ids).reshape(-1)
    if ids.size == 0:
        return [], "instance_ids_unavailable"
    if ids.shape[0] != labels.shape[0]:
        raise ValueError(
            "SwAV prototype transition export requires one instance_id per label, "
            f"got labels={labels.shape[0]} and instance_ids shape={tuple(ids.shape)}."
        )

    rows: list[dict[str, Any]] = []
    for pair_idx in range(len(frame_groups) - 1):
        frame_a, indices_a = frame_groups[pair_idx]
        frame_b, indices_b = frame_groups[pair_idx + 1]
        idx_a = np.asarray(indices_a, dtype=int).reshape(-1)
        idx_b = np.asarray(indices_b, dtype=int).reshape(-1)
        ids_a = ids[idx_a]
        ids_b = ids[idx_b]
        label_by_id_b = {int(atom_id): int(labels[int(idx_b[pos])]) for pos, atom_id in enumerate(ids_b)}
        counts = np.zeros((int(num_prototypes), int(num_prototypes)), dtype=np.int64)
        matched = 0
        for pos, atom_id in enumerate(ids_a):
            atom_id_int = int(atom_id)
            if atom_id_int not in label_by_id_b:
                continue
            proto_a = int(labels[int(idx_a[pos])])
            proto_b = int(label_by_id_b[atom_id_int])
            counts[proto_a, proto_b] += 1
            matched += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        for proto_a in range(int(num_prototypes)):
            for proto_b in range(int(num_prototypes)):
                count = int(counts[proto_a, proto_b])
                if count == 0:
                    continue
                denom = int(row_sums[proto_a, 0])
                rows.append(
                    {
                        "frame_from": str(frame_a),
                        "frame_to": str(frame_b),
                        "matched_instance_count": int(matched),
                        "prototype_from": int(proto_a),
                        "prototype_to": int(proto_b),
                        "count": count,
                        "row_fraction": float(count / denom) if denom > 0 else float("nan"),
                    }
                )
    if not rows:
        return [], "no_matched_instance_ids_between_adjacent_frames"
    return rows, None


def _plot_usage(
    path: Path,
    usage_rows: list[dict[str, Any]],
    *,
    prototype_color_map: dict[int, str] | None = None,
) -> None:
    proto_ids = [int(row["prototype_id"]) for row in usage_rows]
    hard = [float(row["hard_fraction"]) for row in usage_rows]
    soft = [float(row["mean_probability"]) for row in usage_rows]
    colors = [
        str(prototype_color_map[int(proto_id)])
        if prototype_color_map is not None and int(proto_id) in prototype_color_map
        else "#2878b5"
        for proto_id in proto_ids
    ]
    x = np.arange(len(proto_ids))
    width = 0.42
    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(proto_ids)), 4.0))
    ax.bar(x - width / 2.0, hard, width=width, label="argmax fraction", color=colors)
    ax.bar(
        x + width / 2.0,
        soft,
        width=width,
        label="mean probability",
        color=colors,
        alpha=0.42,
        edgecolor=colors,
        linewidth=1.0,
    )
    ax.set_xlabel("SwAV prototype")
    ax.set_ylabel("fraction")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in proto_ids])
    ax.set_ylim(0.0, max(1.0, max(hard + soft) * 1.1))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_confidence(path: Path, confidence: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(np.asarray(confidence, dtype=np.float32), bins=40, color="#2878b5", alpha=0.88)
    ax.set_xlabel("max prototype probability")
    ax.set_ylabel("sample count")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_frame_proportions(
    path: Path,
    counts_matrix: np.ndarray,
    frame_names: list[str],
    *,
    prototype_color_map: dict[int, str] | None = None,
    prototype_order: list[int] | None = None,
) -> None:
    counts = np.asarray(counts_matrix, dtype=np.float64)
    totals = counts.sum(axis=1, keepdims=True)
    fractions = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    x = np.arange(fractions.shape[0])
    order = (
        list(range(fractions.shape[1]))
        if prototype_order is None
        else [int(proto_id) for proto_id in prototype_order]
    )
    fractions_ordered = fractions[:, order]
    colors = [
        (
            str(prototype_color_map[int(proto_id)])
            if prototype_color_map is not None and int(proto_id) in prototype_color_map
            else None
        )
        for proto_id in order
    ]
    labels = [f"P{int(proto_id)}" for proto_id in order]
    fig, ax = plt.subplots(figsize=(11, 5), dpi=220)
    ax.stackplot(
        x,
        fractions_ordered.T,
        colors=colors,
        labels=labels,
        alpha=0.84,
        linewidth=0.0,
    )
    if len(x) > 1:
        ax.set_xlim(x[0], x[-1])
    else:
        ax.set_xlim(-0.5, 0.5)
    tick_stride = max(1, len(frame_names) // 10)
    tick_positions = x[::tick_stride]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([frame_names[int(i)] for i in tick_positions], rotation=35, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("frame")
    ax.set_ylabel("prototype fraction")
    ax.set_title("SwAV prototype proportions across time")
    if fractions.shape[1] <= 16:
        ax.legend(frameon=False, ncol=min(4, fractions.shape[1]), loc="upper right")
    ax.grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_contingency(
    path: Path,
    matrix: np.ndarray,
    *,
    title: str,
    prototype_ids: list[int] | None = None,
    cluster_ids: list[int] | None = None,
    assigned_pairs: list[dict[str, Any]] | None = None,
) -> None:
    matrix_arr = np.asarray(matrix, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    image = ax.imshow(matrix_arr, aspect="auto", cmap="viridis")
    if prototype_ids is not None and len(prototype_ids) <= 60:
        ax.set_yticks(np.arange(len(prototype_ids)))
        ax.set_yticklabels([str(int(v)) for v in prototype_ids])
    if cluster_ids is not None and len(cluster_ids) <= 60:
        ax.set_xticks(np.arange(len(cluster_ids)))
        ax.set_xticklabels([str(int(v)) for v in cluster_ids], rotation=35, ha="right")
    if assigned_pairs:
        proto_to_row = (
            {int(proto_id): int(pos) for pos, proto_id in enumerate(prototype_ids)}
            if prototype_ids is not None
            else {}
        )
        cluster_to_col = (
            {int(cluster_id): int(pos) for pos, cluster_id in enumerate(cluster_ids)}
            if cluster_ids is not None
            else {}
        )
        for pair in assigned_pairs:
            proto_id = int(pair["prototype_id"])
            cluster_id = int(pair["cluster_id"])
            if proto_id not in proto_to_row or cluster_id not in cluster_to_col:
                continue
            row_pos = proto_to_row[proto_id]
            col_pos = cluster_to_col[cluster_id]
            ax.add_patch(
                Rectangle(
                    (col_pos - 0.5, row_pos - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="white",
                    linewidth=1.8,
                )
            )
    ax.set_xlabel("post-hoc cluster")
    ax.set_ylabel("SwAV prototype")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_md_prototype_outputs(
    *,
    out_dir: Path,
    coords: np.ndarray,
    labels: np.ndarray,
    confidence: np.ndarray,
    prototype_color_map: dict[int, str] | None,
    max_points: int | None,
    sample_indices: np.ndarray | None = None,
    source_name: str | None = None,
) -> dict[str, Any]:
    coords_arr = np.asarray(coords, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=int).reshape(-1)
    confidence_arr = np.asarray(confidence, dtype=np.float32).reshape(-1)
    if sample_indices is not None:
        sample_indices_arr = np.asarray(sample_indices, dtype=int).reshape(-1)
        coords_arr = coords_arr[sample_indices_arr]
        labels_arr = labels_arr[sample_indices_arr]
        confidence_arr = confidence_arr[sample_indices_arr]
    title_source = "" if source_name is None else f" at {source_name}"
    title = f"MD local-structure SwAV prototypes{title_source} (n={len(labels_arr)})"

    md_dir = out_dir / "md_space"
    md_dir.mkdir(parents=True, exist_ok=True)
    static_path = md_dir / "md_space_swav_prototypes.png"
    save_md_space_clusters_plot(
        coords_arr,
        labels_arr,
        static_path,
        cluster_color_map=prototype_color_map,
        max_points=max_points,
        title=title,
        legend_title="prototype",
    )
    outputs: dict[str, Any] = {
        "static_png": str(static_path),
        "source_name": source_name,
        "num_points_before_max_points": int(len(labels_arr)),
    }
    interactive_path = md_dir / "md_space_swav_prototypes.html"
    save_interactive_md_plot(
        coords_arr,
        labels_arr,
        interactive_path,
        palette="tab10",
        cluster_color_map=prototype_color_map,
        max_points=max_points,
        marker_size=3.0,
        marker_line_width=0.0,
        title=title,
        label_prefix="Prototype",
        hover_values=confidence_arr,
        hover_label="confidence",
        aspect_mode="cube",
    )
    outputs["interactive_html"] = str(interactive_path)
    return outputs


def _compare_to_clustering(
    *,
    out_dir: Path,
    prototype_labels: np.ndarray,
    num_prototypes: int,
    cluster_labels_by_k: dict[int, np.ndarray] | None,
    cluster_color_maps_by_k: dict[int, dict[int, str]] | None,
    plots_enabled: bool,
) -> dict[str, Any]:
    if not cluster_labels_by_k:
        return {}
    comparisons: dict[str, Any] = {}
    proto_labels = np.asarray(prototype_labels, dtype=int).reshape(-1)
    for k_value, cluster_labels_raw in cluster_labels_by_k.items():
        cluster_labels = np.asarray(cluster_labels_raw, dtype=int).reshape(-1)
        mask = cluster_labels >= 0
        proto_vals = list(range(int(num_prototypes)))
        cluster_vals = sorted(int(v) for v in np.unique(cluster_labels[mask]))
        proto_to_row = {value: pos for pos, value in enumerate(proto_vals)}
        cluster_to_col = {value: pos for pos, value in enumerate(cluster_vals)}
        contingency = np.zeros((len(proto_vals), len(cluster_vals)), dtype=np.int64)
        for proto, cluster in zip(proto_labels[mask], cluster_labels[mask]):
            contingency[proto_to_row[int(proto)], cluster_to_col[int(cluster)]] += 1

        rows: list[dict[str, Any]] = []
        for row_pos, proto_id in enumerate(proto_vals):
            for col_pos, cluster_id in enumerate(cluster_vals):
                rows.append(
                    {
                        "prototype_id": int(proto_id),
                        "cluster_id": int(cluster_id),
                        "count": int(contingency[row_pos, col_pos]),
                    }
                )
        csv_path = out_dir / f"swav_vs_clustering_k{int(k_value)}_contingency.csv"
        _write_dict_rows_csv(csv_path, rows)
        correspondence = _compute_prototype_cluster_correspondence(
            prototype_labels=proto_labels,
            cluster_labels=cluster_labels,
            num_prototypes=int(num_prototypes),
            cluster_color_map=(
                None
                if cluster_color_maps_by_k is None
                else cluster_color_maps_by_k.get(int(k_value))
            ),
        )
        correspondence_csv = out_dir / f"swav_vs_clustering_k{int(k_value)}_correspondence.csv"
        correspondence_fieldnames = [
            "prototype_id",
            "prototype_sample_count",
            "nonnoise_overlap_count",
            "best_overlap_cluster_id",
            "best_overlap_count",
            "best_overlap_fraction_of_prototype",
            "best_overlap_fraction_of_cluster",
            "hungarian_cluster_id",
            "hungarian_overlap_count",
            "assigned_cluster_id",
            "assignment_source",
            "assigned_cluster_color",
            "prototype_color",
        ]
        _write_dict_rows_csv(
            correspondence_csv,
            list(correspondence["prototype_rows"]),
            fieldnames=correspondence_fieldnames,
        )
        cluster_correspondence_csv = out_dir / f"swav_vs_clustering_k{int(k_value)}_cluster_correspondence.csv"
        _write_dict_rows_csv(
            cluster_correspondence_csv,
            list(correspondence["cluster_rows"]),
            fieldnames=[
                "cluster_id",
                "cluster_sample_count",
                "best_overlap_prototype_id",
                "best_overlap_count",
                "best_overlap_fraction_of_cluster",
                "hungarian_prototype_id",
                "hungarian_overlap_count",
                "cluster_color",
            ],
        )
        plot_path = None
        if plots_enabled:
            plot_path = out_dir / f"swav_vs_clustering_k{int(k_value)}_contingency.png"
            _plot_contingency(
                plot_path,
                contingency,
                title=f"SwAV prototypes vs clustering k={int(k_value)}",
                prototype_ids=proto_vals,
                cluster_ids=cluster_vals,
                assigned_pairs=list(correspondence.get("assigned_pairs", [])),
            )
        comparisons[str(int(k_value))] = {
            "ari": float(adjusted_rand_score(proto_labels[mask], cluster_labels[mask])),
            "nmi": float(normalized_mutual_info_score(proto_labels[mask], cluster_labels[mask])),
            "sample_count": int(mask.sum()),
            "contingency_csv": str(csv_path),
            "contingency_png": None if plot_path is None else str(plot_path),
            "correspondence_csv": str(correspondence_csv),
            "cluster_correspondence_csv": str(cluster_correspondence_csv),
            "prototype_ids": proto_vals,
            "cluster_ids": cluster_vals,
            "prototype_to_cluster": correspondence["prototype_to_cluster"],
            "cluster_to_prototype": correspondence["cluster_to_prototype"],
            "best_overlap_prototype_to_cluster": correspondence[
                "best_overlap_prototype_to_cluster"
            ],
            "hungarian_prototype_to_cluster": correspondence[
                "hungarian_prototype_to_cluster"
            ],
            "prototype_color_map": correspondence["prototype_color_map"],
            "assigned_pairs": correspondence["assigned_pairs"],
            "hungarian_total_overlap": int(correspondence["hungarian_total_overlap"]),
            "assigned_total_overlap": int(correspondence["assigned_total_overlap"]),
        }
    return comparisons


def _prototype_color_map_from_primary_comparison(
    *,
    num_prototypes: int,
    comparisons: dict[str, Any],
    primary_k: int | None,
) -> tuple[dict[int, str], dict[str, Any] | None]:
    comparison: dict[str, Any] | None = None
    if primary_k is not None and str(int(primary_k)) in comparisons:
        comparison = comparisons[str(int(primary_k))]
    elif comparisons:
        first_key = sorted((int(k) for k in comparisons.keys()))[0]
        comparison = comparisons[str(first_key)]

    if comparison is None or "prototype_color_map" not in comparison:
        return _build_cluster_color_map(np.arange(int(num_prototypes), dtype=int)), comparison
    fallback = _build_cluster_color_map(np.arange(int(num_prototypes), dtype=int))
    resolved = {
        int(proto_id): str(color)
        for proto_id, color in dict(comparison["prototype_color_map"]).items()
    }
    for proto_id, color in fallback.items():
        resolved.setdefault(int(proto_id), str(color))
    return resolved, comparison


def _prototype_stack_order(
    *,
    num_prototypes: int,
    prototype_to_cluster: dict[int, int],
    cluster_ids: list[int] | None,
) -> list[int]:
    all_proto_ids = list(range(int(num_prototypes)))
    if not prototype_to_cluster:
        return all_proto_ids
    cluster_order = (
        sorted(int(cluster_id) for cluster_id in set(prototype_to_cluster.values()))
        if cluster_ids is None
        else [int(cluster_id) for cluster_id in cluster_ids]
    )
    cluster_rank = {int(cluster_id): int(pos) for pos, cluster_id in enumerate(cluster_order)}
    return sorted(
        all_proto_ids,
        key=lambda proto_id: (
            cluster_rank.get(
                int(prototype_to_cluster.get(int(proto_id), -1)),
                len(cluster_rank),
            ),
            int(proto_id),
        ),
    )


def _top_probability_fields(
    probs: np.ndarray,
    *,
    prototype_ids: list[int],
    top_n: int,
) -> list[dict[str, Any]]:
    values = np.asarray(probs, dtype=np.float64).reshape(-1)
    order = np.argsort(values)[::-1][: max(1, int(top_n))]
    rows: list[dict[str, Any]] = []
    for rank, proto_pos in enumerate(order, start=1):
        rows.append(
            {
                f"top{rank}_prototype_id": int(prototype_ids[int(proto_pos)]),
                f"top{rank}_probability": float(values[int(proto_pos)]),
            }
        )
    return rows


def _cluster_representatives_in_prototypes_rows(
    *,
    assignments: dict[str, np.ndarray],
    latents: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_color_map: dict[int, str] | None,
    prototype_color_map: dict[int, str],
    prototype_to_cluster: dict[int, int],
    top_n: int,
) -> tuple[list[dict[str, Any]], list[int], list[int], np.ndarray]:
    probs = np.asarray(assignments["probs"], dtype=np.float32)
    prototype_labels = np.asarray(assignments["labels"], dtype=int).reshape(-1)
    latents_arr = np.asarray(latents, dtype=np.float32)
    cluster_labels_arr = np.asarray(cluster_labels, dtype=int).reshape(-1)
    representatives = _compute_cluster_representative_indices(latents_arr, cluster_labels_arr)
    cluster_ids = sorted(int(cluster_id) for cluster_id in representatives.keys())
    prototype_ids = list(range(int(probs.shape[1])))
    matrix = np.zeros((len(cluster_ids), len(prototype_ids)), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for row_pos, cluster_id in enumerate(cluster_ids):
        sample_idx = int(representatives[int(cluster_id)])
        probabilities = probs[sample_idx, :]
        matrix[int(row_pos), :] = probabilities
        argmax_proto = int(prototype_labels[sample_idx])
        assigned_cluster = prototype_to_cluster.get(int(argmax_proto))
        row: dict[str, Any] = {
            "cluster_id": int(cluster_id),
            "representative_sample_index": int(sample_idx),
            "argmax_prototype_id": int(argmax_proto),
            "argmax_prototype_probability": float(probabilities[int(argmax_proto)]),
            "argmax_prototype_assigned_cluster_id": (
                "" if assigned_cluster is None else int(assigned_cluster)
            ),
            "argmax_prototype_assigned_to_this_cluster": bool(
                assigned_cluster is not None and int(assigned_cluster) == int(cluster_id)
            ),
            "cluster_color": (
                ""
                if cluster_color_map is None or int(cluster_id) not in cluster_color_map
                else str(cluster_color_map[int(cluster_id)])
            ),
            "argmax_prototype_color": str(prototype_color_map.get(int(argmax_proto), "#777777")),
        }
        for top_fields in _top_probability_fields(
            probabilities,
            prototype_ids=prototype_ids,
            top_n=int(top_n),
        ):
            row.update(top_fields)
        for proto_id in prototype_ids:
            row[f"prototype_{int(proto_id)}_probability"] = float(probabilities[int(proto_id)])
        rows.append(row)
    return rows, cluster_ids, prototype_ids, matrix


def _plot_cluster_representatives_in_prototypes(
    path: Path,
    *,
    cluster_ids: list[int],
    prototype_ids: list[int],
    probability_matrix: np.ndarray,
    prototype_color_map: dict[int, str],
    cluster_color_map: dict[int, str] | None,
    primary_k: int,
) -> None:
    probs = np.asarray(probability_matrix, dtype=np.float64)
    fig_height = max(4.2, 0.46 * len(cluster_ids) + 1.8)
    fig_width = max(7.2, 0.38 * len(prototype_ids) + 3.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    y = np.arange(len(cluster_ids), dtype=np.float64)
    left = np.zeros((len(cluster_ids),), dtype=np.float64)
    for proto_pos, proto_id in enumerate(prototype_ids):
        values = probs[:, int(proto_pos)]
        color = str(prototype_color_map.get(int(proto_id), "#777777"))
        ax.barh(
            y,
            values,
            left=left,
            height=0.72,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            label=f"P{int(proto_id)}",
        )
        left = left + values
    ax.set_xlim(0.0, max(1.0, float(np.max(left)) * 1.02))
    ax.set_yticks(y)
    ax.set_yticklabels([f"C{int(cluster_id)}" for cluster_id in cluster_ids])
    for tick, cluster_id in zip(ax.get_yticklabels(), cluster_ids, strict=True):
        if cluster_color_map is not None and int(cluster_id) in cluster_color_map:
            tick.set_color(str(cluster_color_map[int(cluster_id)]))
            tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel("SwAV prototype probability")
    ax.set_ylabel("cluster representative")
    ax.set_title(f"Cluster representatives described by SwAV prototypes (k={int(primary_k)})")
    if len(prototype_ids) <= 20:
        ax.legend(
            frameon=False,
            ncol=min(5, len(prototype_ids)),
            bbox_to_anchor=(0.5, -0.14),
            loc="upper center",
        )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_cluster_representatives_in_prototypes(
    *,
    out_root: Path,
    assignments: dict[str, np.ndarray],
    latents: np.ndarray,
    cluster_labels: np.ndarray | None,
    cluster_color_map: dict[int, str] | None,
    prototype_color_map: dict[int, str],
    prototype_to_cluster: dict[int, int],
    primary_k: int | None,
    plots_enabled: bool,
    top_n: int,
) -> dict[str, Any]:
    rows, cluster_ids, prototype_ids, probability_matrix = _cluster_representatives_in_prototypes_rows(
        assignments=assignments,
        latents=latents,
        cluster_labels=np.asarray(cluster_labels, dtype=int),
        cluster_color_map=cluster_color_map,
        prototype_color_map=prototype_color_map,
        prototype_to_cluster=prototype_to_cluster,
        top_n=int(top_n),
    )
    csv_path = out_root / f"cluster_representatives_described_by_prototypes_k{int(primary_k)}.csv"
    fieldnames = [
        "cluster_id",
        "representative_sample_index",
        "argmax_prototype_id",
        "argmax_prototype_probability",
        "argmax_prototype_assigned_cluster_id",
        "argmax_prototype_assigned_to_this_cluster",
        "cluster_color",
        "argmax_prototype_color",
    ]
    for rank in range(1, max(1, int(top_n)) + 1):
        fieldnames.extend([f"top{rank}_prototype_id", f"top{rank}_probability"])
    fieldnames.extend([f"prototype_{int(proto_id)}_probability" for proto_id in prototype_ids])
    _write_dict_rows_csv(csv_path, rows, fieldnames=fieldnames)
    result: dict[str, Any] = {
        "csv": str(csv_path),
        "cluster_ids": [int(v) for v in cluster_ids],
        "prototype_ids": [int(v) for v in prototype_ids],
    }
    if plots_enabled:
        plot_path = out_root / f"cluster_representatives_described_by_prototypes_k{int(primary_k)}.png"
        _plot_cluster_representatives_in_prototypes(
            plot_path,
            cluster_ids=cluster_ids,
            prototype_ids=prototype_ids,
            probability_matrix=probability_matrix,
            prototype_color_map=prototype_color_map,
            cluster_color_map=cluster_color_map,
            primary_k=int(primary_k),
        )
        result["png"] = str(plot_path)
    return result


def _visible_prototype_sets_from_cluster_sets(
    *,
    requested_cluster_sets: list[list[int]] | None,
    prototype_to_cluster: dict[int, int],
    available_prototype_ids: list[int],
) -> list[list[int]] | None:
    if not requested_cluster_sets or not prototype_to_cluster:
        return None
    available = {int(v) for v in available_prototype_ids}
    resolved: list[list[int]] = []
    for cluster_set in requested_cluster_sets:
        cluster_ids = {int(v) for v in cluster_set}
        prototype_ids = sorted(
            int(proto_id)
            for proto_id, cluster_id in prototype_to_cluster.items()
            if int(proto_id) in available and int(cluster_id) in cluster_ids
        )
        if prototype_ids:
            resolved.append(prototype_ids)
    return resolved or None


def _render_swav_prototype_figure_set(
    *,
    model: Any,
    assignments: dict[str, np.ndarray],
    main_latents: np.ndarray,
    out_root: Path,
    swav_cfg: Any,
    batch_size: int,
    assignment_method: str,
    sinkhorn_iterations: int | None,
    assignment_device: str,
    figure_settings: Any | None,
    figure_set_run_kwargs: dict[str, Any] | None,
    figure_dataloader: torch.utils.data.DataLoader | None,
    figure_snapshot_layout: Any | None,
    figure_analysis_source_names: list[str] | None,
    prototype_color_map: dict[int, str],
    prototype_to_cluster: dict[int, int],
    step: Any,
) -> dict[str, Any]:
    enabled = _cfg_bool(swav_cfg, "figure_set", True)
    if not enabled:
        return {"enabled": False}
    if figure_settings is None or not bool(getattr(figure_settings, "enabled", False)):
        return {"enabled": False, "skipped_reason": "figure_set_disabled"}

    figure_latents = np.asarray(figure_set_run_kwargs["latents"], dtype=np.float32)
    figure_coords = np.asarray(figure_set_run_kwargs["coords"], dtype=np.float32)
    main_latents_arr = np.asarray(main_latents, dtype=np.float32)
    if np.shares_memory(figure_latents, main_latents_arr):
        figure_assignments = assignments
        figure_arrays_path = None
    else:
        if step is not None:
            step("Evaluating SwAV prototypes for figure-set samples")
        figure_assignments = compute_swav_assignments(
            model,
            figure_latents,
            batch_size=int(batch_size),
            assignment_method=str(assignment_method),
            sinkhorn_iterations=sinkhorn_iterations,
            assignment_device=str(assignment_device),
        )
        figure_arrays_path = out_root / "swav_figure_assignment_arrays.npz"
        np.savez_compressed(figure_arrays_path, **figure_assignments)

    prototype_labels = np.asarray(figure_assignments["labels"], dtype=int).reshape(-1)
    num_prototypes = int(np.asarray(figure_assignments["probs"]).shape[1])
    available_prototype_ids = sorted(int(v) for v in np.unique(prototype_labels) if int(v) >= 0)
    visible_prototype_sets = _visible_prototype_sets_from_cluster_sets(
        requested_cluster_sets=getattr(figure_settings, "visible_cluster_sets", None),
        prototype_to_cluster=prototype_to_cluster,
        available_prototype_ids=available_prototype_ids,
    )
    figure_settings_for_prototypes = replace(
        figure_settings,
        k=int(num_prototypes),
        visible_cluster_sets=visible_prototype_sets,
        cluster_color_assignment={
            int(proto_id): str(color) for proto_id, color in prototype_color_map.items()
        },
    )
    prototype_run_kwargs = dict(figure_set_run_kwargs)
    prototype_run_kwargs["k_value"] = int(num_prototypes)
    prototype_run_kwargs["cluster_color_assignment"] = {
        int(proto_id): str(color) for proto_id, color in prototype_color_map.items()
    }
    prototype_run_kwargs["visible_cluster_sets"] = visible_prototype_sets
    cluster_figure_set, snapshot_figure_sets = render_cluster_figure_outputs(
        out_dir=out_root,
        dataloader=figure_dataloader,
        figure_settings=figure_settings_for_prototypes,
        figure_set_run_kwargs=prototype_run_kwargs,
        labels_for_k=prototype_labels,
        latents=figure_latents,
        coords=figure_coords,
        dataset_obj=figure_set_run_kwargs["dataset"],
        snapshot_layout=figure_snapshot_layout,
        analysis_source_names=figure_analysis_source_names,
        step=step,
    )
    return {
        "enabled": True,
        "num_prototypes": int(num_prototypes),
        "assignment_arrays_npz": None if figure_arrays_path is None else str(figure_arrays_path),
        "prototype_color_map": {
            int(proto_id): str(color) for proto_id, color in prototype_color_map.items()
        },
        "visible_prototype_sets": visible_prototype_sets,
        "cluster_figure_set": cluster_figure_set,
        "cluster_figure_sets_by_snapshot": snapshot_figure_sets,
    }


def run_swav_prototype_evaluation(
    *,
    model: Any,
    cache: dict[str, np.ndarray],
    out_dir: Path,
    analysis_cfg: Any,
    cluster_labels_by_k: dict[int, np.ndarray] | None = None,
    cluster_color_maps_by_k: dict[int, dict[int, str]] | None = None,
    primary_k: int | None = None,
    frame_groups: list[tuple[str, np.ndarray]] | None = None,
    proportion_frame_groups: list[tuple[str, np.ndarray]] | None = None,
    figure_settings: Any | None = None,
    figure_set_run_kwargs: dict[str, Any] | None = None,
    figure_dataloader: torch.utils.data.DataLoader | None = None,
    figure_snapshot_layout: Any | None = None,
    figure_analysis_source_names: list[str] | None = None,
    step=None,
) -> dict[str, Any]:
    swav_cfg = OmegaConf.select(analysis_cfg, "swav", default=None)
    enabled = bool(_cfg_bool(swav_cfg, "enabled", default=_swav_is_available(model)))
    if not enabled:
        return {}

    if step is not None:
        step("Evaluating SwAV prototypes")

    out_root = Path(out_dir) / "swav_prototypes"
    out_root.mkdir(parents=True, exist_ok=True)
    batch_size = _cfg_int(swav_cfg, "batch_size", 8192)
    assignment_method = str(_cfg_select(swav_cfg, "assignment_method", "sinkhorn")).strip().lower()
    assignment_device = str(_cfg_select(swav_cfg, "assignment_device", "auto")).strip().lower()
    sinkhorn_iterations_raw = _cfg_select(swav_cfg, "sinkhorn_iterations", None)
    sinkhorn_iterations = (
        None if sinkhorn_iterations_raw is None else int(sinkhorn_iterations_raw)
    )
    representative_top_k = _cfg_int(swav_cfg, "representative_top_k", 12)
    plots_enabled = _cfg_bool(swav_cfg, "plots", True)
    md_outputs_enabled = _cfg_bool(swav_cfg, "md_outputs", True)
    md_max_points_raw = _cfg_select(swav_cfg, "md_max_points", None)
    md_max_points = (
        None
        if md_max_points_raw is None or int(md_max_points_raw) <= 0
        else int(md_max_points_raw)
    )
    export_assignment_csv = _cfg_bool(swav_cfg, "export_assignment_csv", True)
    max_assignment_csv_rows_raw = _cfg_select(
        swav_cfg,
        "max_assignment_csv_rows",
        100000,
    )
    max_assignment_csv_rows = (
        None
        if max_assignment_csv_rows_raw is None or int(max_assignment_csv_rows_raw) <= 0
        else int(max_assignment_csv_rows_raw)
    )
    cluster_representative_top_n = _cfg_int(
        swav_cfg,
        "cluster_representative_top_prototypes",
        3,
    )

    assignments = compute_swav_assignments(
        model,
        np.asarray(cache["inv_latents"], dtype=np.float32),
        batch_size=batch_size,
        assignment_method=assignment_method,
        sinkhorn_iterations=sinkhorn_iterations,
        assignment_device=assignment_device,
    )
    arrays_path = out_root / "swav_assignment_arrays.npz"
    np.savez_compressed(arrays_path, **assignments)

    usage_metrics, usage_rows = _usage_summary(assignments)
    usage_csv = out_root / "prototype_usage.csv"
    _write_dict_rows_csv(usage_csv, usage_rows)

    assignment_csv_summary = None
    if export_assignment_csv:
        assignment_csv_summary = _write_assignment_csv(
            out_root / "prototype_assignments.csv",
            assignments,
            cache,
            max_rows=max_assignment_csv_rows,
        )

    representatives = _representative_rows(
        assignments,
        cache,
        top_k=representative_top_k,
    )
    representatives_csv = out_root / "prototype_representatives.csv"
    _write_dict_rows_csv(representatives_csv, representatives)

    comparisons = {}
    if _cfg_bool(swav_cfg, "compare_to_clustering", True):
        comparisons = _compare_to_clustering(
            out_dir=out_root,
            prototype_labels=assignments["labels"],
            num_prototypes=int(np.asarray(assignments["probs"]).shape[1]),
            cluster_labels_by_k=cluster_labels_by_k,
            cluster_color_maps_by_k=cluster_color_maps_by_k,
            plots_enabled=plots_enabled,
        )
    selected_comparison_key = None
    if primary_k is not None and str(int(primary_k)) in comparisons:
        selected_comparison_key = str(int(primary_k))
    elif comparisons:
        selected_comparison_key = str(sorted(int(k) for k in comparisons.keys())[0])
    prototype_color_map, primary_comparison = _prototype_color_map_from_primary_comparison(
        num_prototypes=int(np.asarray(assignments["probs"]).shape[1]),
        comparisons=comparisons,
        primary_k=primary_k,
    )
    prototype_to_cluster = (
        {}
        if primary_comparison is None
        else {
            int(proto_id): int(cluster_id)
            for proto_id, cluster_id in dict(
                primary_comparison.get("prototype_to_cluster", {})
            ).items()
        }
    )
    primary_cluster_labels = (
        None
        if primary_k is None or cluster_labels_by_k is None or int(primary_k) not in cluster_labels_by_k
        else np.asarray(cluster_labels_by_k[int(primary_k)], dtype=int)
    )
    primary_cluster_color_map = (
        None
        if primary_k is None or cluster_color_maps_by_k is None
        else cluster_color_maps_by_k.get(int(primary_k))
    )
    prototype_order = _prototype_stack_order(
        num_prototypes=int(np.asarray(assignments["probs"]).shape[1]),
        prototype_to_cluster=prototype_to_cluster,
        cluster_ids=(
            None
            if primary_comparison is None
            else [int(cluster_id) for cluster_id in primary_comparison.get("cluster_ids", [])]
        ),
    )

    frame_outputs: dict[str, Any] = {}
    if _cfg_bool(swav_cfg, "frame_proportions", True):
        frame_proportion_groups = (
            proportion_frame_groups
            if proportion_frame_groups is not None
            else frame_groups
        )
        frame_rows, counts_matrix, frame_names = _frame_proportions(
            assignments["labels"],
            frame_proportion_groups,
            num_prototypes=int(assignments["probs"].shape[1]),
        )
        if frame_rows:
            frame_csv = out_root / "prototype_proportions_by_frame.csv"
            _write_dict_rows_csv(frame_csv, frame_rows)
            frame_outputs["prototype_proportions_by_frame_csv"] = str(frame_csv)
            frame_outputs["num_frames"] = int(len(frame_names))
            frame_outputs["prototype_stack_order"] = [int(v) for v in prototype_order]
            if plots_enabled and counts_matrix is not None:
                frame_plot = out_root / "prototype_proportions_by_frame.png"
                _plot_frame_proportions(
                    frame_plot,
                    counts_matrix,
                    frame_names,
                    prototype_color_map=prototype_color_map,
                    prototype_order=prototype_order,
                )
                frame_outputs["prototype_proportions_by_frame_png"] = str(frame_plot)

    transition_outputs: dict[str, Any] = {}
    if _cfg_bool(swav_cfg, "transitions", True):
        transition_rows, skipped_reason = _transition_rows(
            assignments["labels"],
            np.asarray(cache["instance_ids"]),
            frame_groups,
            num_prototypes=int(assignments["probs"].shape[1]),
        )
        if transition_rows:
            transition_csv = out_root / "prototype_transitions_by_instance.csv"
            _write_dict_rows_csv(transition_csv, transition_rows)
            transition_outputs["prototype_transitions_by_instance_csv"] = str(transition_csv)
        elif skipped_reason is not None:
            transition_outputs["skipped_reason"] = str(skipped_reason)

    plot_outputs: dict[str, str] = {}
    if plots_enabled:
        usage_png = out_root / "prototype_usage.png"
        confidence_png = out_root / "prototype_confidence_hist.png"
        _plot_usage(
            usage_png,
            usage_rows,
            prototype_color_map=prototype_color_map,
        )
        _plot_confidence(confidence_png, assignments["confidence"])
        plot_outputs["prototype_usage_png"] = str(usage_png)
        plot_outputs["prototype_confidence_hist_png"] = str(confidence_png)

    prototype_figure_set = _render_swav_prototype_figure_set(
        model=model,
        assignments=assignments,
        main_latents=np.asarray(cache["inv_latents"], dtype=np.float32),
        out_root=out_root,
        swav_cfg=swav_cfg,
        batch_size=int(batch_size),
        assignment_method=str(assignment_method),
        sinkhorn_iterations=sinkhorn_iterations,
        assignment_device=str(assignment_device),
        figure_settings=figure_settings,
        figure_set_run_kwargs=figure_set_run_kwargs,
        figure_dataloader=figure_dataloader,
        figure_snapshot_layout=figure_snapshot_layout,
        figure_analysis_source_names=figure_analysis_source_names,
        prototype_color_map=prototype_color_map,
        prototype_to_cluster=prototype_to_cluster,
        step=step,
    )

    md_outputs: dict[str, Any] = {}
    if md_outputs_enabled:
        md_coords = np.asarray(cache["coords"], dtype=np.float32)
        md_labels = assignments["labels"]
        md_confidence = assignments["confidence"]
        md_frame_groups = (
            proportion_frame_groups
            if proportion_frame_groups is not None
            else frame_groups
        )
        md_sample_source = "main_temporal_inference"
        figure_assignment_arrays_path = prototype_figure_set.get("assignment_arrays_npz")
        if (
            figure_assignment_arrays_path is not None
            and figure_set_run_kwargs is not None
            and figure_snapshot_layout is not None
        ):
            md_coords = np.asarray(figure_set_run_kwargs["coords"], dtype=np.float32)
            with np.load(str(figure_assignment_arrays_path)) as figure_arrays:
                md_labels = np.asarray(figure_arrays["labels"], dtype=int)
                md_confidence = np.asarray(figure_arrays["confidence"], dtype=np.float32)
            md_frame_groups = figure_snapshot_layout.source_groups
            md_sample_source = "dense_snapshot_figure_set"
        md_frame_index = len(md_frame_groups) // 2 if md_frame_groups else None
        md_source_name, md_sample_indices = (
            md_frame_groups[int(md_frame_index)]
            if md_frame_index is not None
            else (None, None)
        )
        md_outputs = _save_md_prototype_outputs(
            out_dir=out_root,
            coords=md_coords,
            labels=md_labels,
            confidence=md_confidence,
            prototype_color_map=prototype_color_map,
            max_points=md_max_points,
            sample_indices=md_sample_indices,
            source_name=md_source_name,
        )
        if md_frame_index is not None:
            md_outputs["frame_index"] = int(md_frame_index)
        md_outputs["sample_source"] = str(md_sample_source)

    cluster_representatives_in_prototypes = _save_cluster_representatives_in_prototypes(
        out_root=out_root,
        assignments=assignments,
        latents=np.asarray(cache["inv_latents"], dtype=np.float32),
        cluster_labels=primary_cluster_labels,
        cluster_color_map=primary_cluster_color_map,
        prototype_color_map=prototype_color_map,
        prototype_to_cluster=prototype_to_cluster,
        primary_k=primary_k,
        plots_enabled=plots_enabled,
        top_n=cluster_representative_top_n,
    )

    summary: dict[str, Any] = {
        "enabled": True,
        "output_dir": str(out_root),
        "arrays_npz": str(arrays_path),
        "prototype_usage_csv": str(usage_csv),
        "prototype_representatives_csv": str(representatives_csv),
        "assignment_csv": assignment_csv_summary,
        "usage": usage_metrics,
        "primary_correspondence_k": (
            None if selected_comparison_key is None else int(selected_comparison_key)
        ),
        "prototype_to_cluster": prototype_to_cluster,
        "prototype_color_map": {
            int(proto_id): str(color) for proto_id, color in prototype_color_map.items()
        },
        "plots": plot_outputs,
        "md_outputs": md_outputs,
        "frame_outputs": frame_outputs,
        "transition_outputs": transition_outputs,
        "clustering_comparison": comparisons,
        "cluster_representatives_in_prototypes": cluster_representatives_in_prototypes,
        "prototype_figure_set": prototype_figure_set,
    }
    summary_path = out_root / "swav_prototype_summary.json"
    write_json(summary_path, summary)
    summary["summary_json"] = str(summary_path)
    return summary


__all__ = [
    "compute_swav_assignments",
    "run_swav_prototype_evaluation",
]
