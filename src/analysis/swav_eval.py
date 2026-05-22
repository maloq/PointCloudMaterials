from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.vis_tools.latent_analysis_vis import save_md_space_clusters_plot
from src.vis_tools.md_cluster_plot import save_interactive_md_plot


def _cfg_bool(cfg: Any, key: str, default: bool) -> bool:
    if cfg is None:
        return bool(default)
    return bool(OmegaConf.select(cfg, key, default=default))


def _cfg_int(cfg: Any, key: str, default: int) -> int:
    if cfg is None:
        return int(default)
    return int(OmegaConf.select(cfg, key, default=default))


def _cfg_select(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    return OmegaConf.select(cfg, key, default=default)


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
    try:
        return next(swav.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("SwAV module has no parameters; cannot evaluate prototypes.") from exc


def compute_swav_assignments(
    model: Any,
    latents: np.ndarray,
    *,
    batch_size: int = 8192,
    assignment_method: str = "sinkhorn",
    sinkhorn_iterations: int | None = None,
) -> dict[str, np.ndarray]:
    """Run the learned SwAV projector/prototype head on cached invariant latents."""
    swav = _resolve_swav(model)
    latents_arr = np.asarray(latents, dtype=np.float32)
    if latents_arr.ndim != 2:
        raise ValueError(
            "SwAV assignment evaluation expects latents with shape (num_samples, latent_dim), "
            f"got {tuple(latents_arr.shape)}."
        )
    if latents_arr.shape[0] == 0:
        raise ValueError("SwAV assignment evaluation received zero latents.")
    if int(batch_size) <= 0:
        raise ValueError(f"swav.batch_size must be > 0, got {batch_size}.")
    assignment_method = str(assignment_method).strip().lower()
    if assignment_method not in {"sinkhorn", "softmax"}:
        raise ValueError(
            "swav.assignment_method must be 'sinkhorn' or 'softmax', "
            f"got {assignment_method!r}."
        )
    if sinkhorn_iterations is not None and int(sinkhorn_iterations) <= 0:
        raise ValueError(f"swav.sinkhorn_iterations must be > 0, got {sinkhorn_iterations}.")

    if hasattr(swav, "normalize_prototypes"):
        swav.normalize_prototypes()

    temperature = float(getattr(swav, "temperature", 1.0))
    if temperature <= 0.0:
        raise ValueError(f"SwAV temperature must be > 0 for evaluation, got {temperature}.")

    device = _prototype_device(swav)
    logits_parts: list[torch.Tensor] = []
    softmax_probs_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, int(latents_arr.shape[0]), int(batch_size)):
            end = min(start + int(batch_size), int(latents_arr.shape[0]))
            batch = torch.from_numpy(latents_arr[start:end]).to(device=device)
            logits = swav._prototype_logits(batch)
            logits_parts.append(logits.detach().cpu())
            if assignment_method == "softmax":
                softmax_probs = F.softmax(logits / temperature, dim=1)
                softmax_probs_parts.append(softmax_probs.detach().cpu())

    logits_tensor = torch.cat(logits_parts, dim=0)
    logits_np = logits_tensor.numpy().astype(np.float32)
    if assignment_method == "sinkhorn":
        if not hasattr(swav, "_sinkhorn"):
            raise RuntimeError(
                "SwAV balanced assignment evaluation requires the model.swav module to expose "
                "_sinkhorn(logits, iterations=...). Use swav.assignment_method='softmax' only "
                "when intentionally inspecting unbalanced prototype logits."
            )
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

    probs_np = probs_tensor.numpy().astype(np.float32)
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
    coords = np.asarray(cache.get("coords", np.empty((0, 3))), dtype=np.float32)
    phases = np.asarray(cache.get("phases", np.empty((0,))), dtype=np.int64)
    instance_ids = np.asarray(cache.get("instance_ids", np.empty((0,))), dtype=np.int64)
    num_samples = int(labels.shape[0])
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
        if coords.shape == (num_samples, 3):
            row.update(
                {
                    "x": float(coords[idx, 0]),
                    "y": float(coords[idx, 1]),
                    "z": float(coords[idx, 2]),
                }
            )
        if phases.shape[0] == num_samples:
            row["phase"] = int(phases[idx])
        if instance_ids.shape[0] == num_samples:
            row["instance_id"] = int(instance_ids[idx])
        rows.append(row)
    fieldnames = ["sample_index", "prototype_id", "confidence", "margin", "entropy"]
    if coords.shape == (num_samples, 3):
        fieldnames += ["x", "y", "z"]
    if phases.shape[0] == num_samples:
        fieldnames.append("phase")
    if instance_ids.shape[0] == num_samples:
        fieldnames.append("instance_id")
    _write_dict_rows_csv(path, rows, fieldnames=fieldnames)
    return {
        "path": str(path),
        "rows_written": int(row_count),
        "truncated": bool(row_count < num_samples),
        "num_samples": int(num_samples),
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
    coords = np.asarray(cache.get("coords", np.empty((0, 3))), dtype=np.float32)
    phases = np.asarray(cache.get("phases", np.empty((0,))), dtype=np.int64)
    instance_ids = np.asarray(cache.get("instance_ids", np.empty((0,))), dtype=np.int64)
    num_samples, num_prototypes = probs.shape
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
            if coords.shape == (num_samples, 3):
                row.update(
                    {
                        "x": float(coords[idx, 0]),
                        "y": float(coords[idx, 1]),
                        "z": float(coords[idx, 2]),
                    }
                )
            if phases.shape[0] == num_samples:
                row["phase"] = int(phases[idx])
            if instance_ids.shape[0] == num_samples:
                row["instance_id"] = int(instance_ids[idx])
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
        if np.any(idx < 0) or np.any(idx >= labels.shape[0]):
            raise IndexError(
                "SwAV frame proportion indices are out of bounds: "
                f"frame={frame_name!r}, min={int(idx.min()) if idx.size else 'NA'}, "
                f"max={int(idx.max()) if idx.size else 'NA'}, num_samples={labels.shape[0]}."
            )
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
    if ids.shape[0] != labels.shape[0]:
        return [], "instance_ids_missing_or_wrong_length"

    rows: list[dict[str, Any]] = []
    for pair_idx in range(len(frame_groups) - 1):
        frame_a, indices_a = frame_groups[pair_idx]
        frame_b, indices_b = frame_groups[pair_idx + 1]
        idx_a = np.asarray(indices_a, dtype=int).reshape(-1)
        idx_b = np.asarray(indices_b, dtype=int).reshape(-1)
        ids_a = ids[idx_a]
        ids_b = ids[idx_b]
        if len(np.unique(ids_a)) != ids_a.shape[0] or len(np.unique(ids_b)) != ids_b.shape[0]:
            return [], (
                "duplicate_instance_ids_within_frame; prototype transitions require unique "
                "instance_ids per frame"
            )
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


def _plot_usage(path: Path, usage_rows: list[dict[str, Any]]) -> None:
    proto_ids = [int(row["prototype_id"]) for row in usage_rows]
    hard = [float(row["hard_fraction"]) for row in usage_rows]
    soft = [float(row["mean_probability"]) for row in usage_rows]
    x = np.arange(len(proto_ids))
    width = 0.42
    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(proto_ids)), 4.0))
    ax.bar(x - width / 2.0, hard, width=width, label="argmax fraction", color="#2878b5")
    ax.bar(x + width / 2.0, soft, width=width, label="mean probability", color="#d95f02")
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


def _plot_frame_proportions(path: Path, counts_matrix: np.ndarray, frame_names: list[str]) -> None:
    counts = np.asarray(counts_matrix, dtype=np.float64)
    totals = counts.sum(axis=1, keepdims=True)
    fractions = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    x = np.arange(fractions.shape[0])
    fig, ax = plt.subplots(figsize=(max(7.0, 0.35 * len(frame_names)), 4.6))
    bottom = np.zeros((fractions.shape[0],), dtype=np.float64)
    for proto_id in range(fractions.shape[1]):
        values = fractions[:, proto_id]
        ax.fill_between(x, bottom, bottom + values, step="mid", alpha=0.84, label=f"P{proto_id}")
        bottom = bottom + values
    tick_stride = max(1, len(frame_names) // 10)
    tick_positions = x[::tick_stride]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([frame_names[int(i)] for i in tick_positions], rotation=35, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("frame")
    ax.set_ylabel("prototype fraction")
    if fractions.shape[1] <= 16:
        ax.legend(frameon=False, ncol=min(4, fractions.shape[1]))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_contingency(path: Path, matrix: np.ndarray, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    image = ax.imshow(np.asarray(matrix, dtype=np.float64), aspect="auto", cmap="viridis")
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
    max_points: int | None,
) -> dict[str, Any]:
    coords_arr = np.asarray(coords, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=int).reshape(-1)
    if coords_arr.shape != (labels_arr.shape[0], 3):
        return {
            "skipped_reason": (
                "coords_missing_or_wrong_shape; expected "
                f"({labels_arr.shape[0]}, 3), got {tuple(coords_arr.shape)}"
            )
        }

    md_dir = out_dir / "md_space"
    md_dir.mkdir(parents=True, exist_ok=True)
    static_path = md_dir / "md_space_swav_prototypes.png"
    save_md_space_clusters_plot(
        coords_arr,
        labels_arr,
        static_path,
        max_points=max_points,
        title=f"MD local-structure SwAV prototypes (n={len(labels_arr)})",
    )
    outputs: dict[str, Any] = {"static_png": str(static_path)}
    try:
        interactive_path = md_dir / "md_space_swav_prototypes.html"
        save_interactive_md_plot(
            coords_arr,
            labels_arr,
            interactive_path,
            palette="tab10",
            max_points=max_points,
            marker_size=3.0,
            marker_line_width=0.0,
            title=f"MD local-structure SwAV prototypes (n={len(labels_arr)})",
            label_prefix="Prototype",
            hover_values=np.asarray(confidence, dtype=np.float32),
            hover_label="confidence",
            aspect_mode="cube",
        )
        outputs["interactive_html"] = str(interactive_path)
    except ImportError:
        outputs["interactive_html_skipped"] = "plotly_not_installed"
    return outputs


def _compare_to_clustering(
    *,
    out_dir: Path,
    prototype_labels: np.ndarray,
    cluster_labels_by_k: dict[int, np.ndarray] | None,
    plots_enabled: bool,
) -> dict[str, Any]:
    if not cluster_labels_by_k:
        return {}
    comparisons: dict[str, Any] = {}
    proto_labels = np.asarray(prototype_labels, dtype=int).reshape(-1)
    for k_value, cluster_labels_raw in cluster_labels_by_k.items():
        cluster_labels = np.asarray(cluster_labels_raw, dtype=int).reshape(-1)
        if cluster_labels.shape[0] != proto_labels.shape[0]:
            raise ValueError(
                "Cannot compare SwAV prototypes to clustering labels because lengths differ: "
                f"prototype_labels={proto_labels.shape[0]}, k={int(k_value)} labels={cluster_labels.shape[0]}."
            )
        mask = cluster_labels >= 0
        if not np.any(mask):
            comparisons[str(int(k_value))] = {"skipped": "no_nonnegative_cluster_labels"}
            continue
        proto_vals = sorted(int(v) for v in np.unique(proto_labels[mask]))
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
        plot_path = None
        if plots_enabled:
            plot_path = out_dir / f"swav_vs_clustering_k{int(k_value)}_contingency.png"
            _plot_contingency(
                plot_path,
                contingency,
                title=f"SwAV prototypes vs clustering k={int(k_value)}",
            )
        comparisons[str(int(k_value))] = {
            "ari": float(adjusted_rand_score(proto_labels[mask], cluster_labels[mask])),
            "nmi": float(normalized_mutual_info_score(proto_labels[mask], cluster_labels[mask])),
            "sample_count": int(mask.sum()),
            "contingency_csv": str(csv_path),
            "contingency_png": None if plot_path is None else str(plot_path),
            "prototype_ids": proto_vals,
            "cluster_ids": cluster_vals,
        }
    return comparisons


def run_swav_prototype_evaluation(
    *,
    model: Any,
    cache: dict[str, np.ndarray],
    out_dir: Path,
    analysis_cfg: Any,
    cluster_labels_by_k: dict[int, np.ndarray] | None = None,
    frame_groups: list[tuple[str, np.ndarray]] | None = None,
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

    assignments = compute_swav_assignments(
        model,
        np.asarray(cache["inv_latents"], dtype=np.float32),
        batch_size=batch_size,
        assignment_method=assignment_method,
        sinkhorn_iterations=sinkhorn_iterations,
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

    frame_outputs: dict[str, Any] = {}
    if _cfg_bool(swav_cfg, "frame_proportions", True):
        frame_rows, counts_matrix, frame_names = _frame_proportions(
            assignments["labels"],
            frame_groups,
            num_prototypes=int(assignments["probs"].shape[1]),
        )
        if frame_rows:
            frame_csv = out_root / "prototype_proportions_by_frame.csv"
            _write_dict_rows_csv(frame_csv, frame_rows)
            frame_outputs["prototype_proportions_by_frame_csv"] = str(frame_csv)
            if plots_enabled and counts_matrix is not None:
                frame_plot = out_root / "prototype_proportions_by_frame.png"
                _plot_frame_proportions(frame_plot, counts_matrix, frame_names)
                frame_outputs["prototype_proportions_by_frame_png"] = str(frame_plot)

    transition_outputs: dict[str, Any] = {}
    if _cfg_bool(swav_cfg, "transitions", True):
        transition_rows, skipped_reason = _transition_rows(
            assignments["labels"],
            np.asarray(cache.get("instance_ids", np.empty((0,)))),
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
        _plot_usage(usage_png, usage_rows)
        _plot_confidence(confidence_png, assignments["confidence"])
        plot_outputs["prototype_usage_png"] = str(usage_png)
        plot_outputs["prototype_confidence_hist_png"] = str(confidence_png)

    md_outputs: dict[str, Any] = {}
    if md_outputs_enabled:
        md_outputs = _save_md_prototype_outputs(
            out_dir=out_root,
            coords=np.asarray(cache.get("coords", np.empty((0, 3))), dtype=np.float32),
            labels=assignments["labels"],
            confidence=assignments["confidence"],
            max_points=md_max_points,
        )

    comparisons = {}
    if _cfg_bool(swav_cfg, "compare_to_clustering", True):
        comparisons = _compare_to_clustering(
            out_dir=out_root,
            prototype_labels=assignments["labels"],
            cluster_labels_by_k=cluster_labels_by_k,
            plots_enabled=plots_enabled,
        )

    summary: dict[str, Any] = {
        "enabled": True,
        "output_dir": str(out_root),
        "arrays_npz": str(arrays_path),
        "prototype_usage_csv": str(usage_csv),
        "prototype_representatives_csv": str(representatives_csv),
        "assignment_csv": assignment_csv_summary,
        "usage": usage_metrics,
        "plots": plot_outputs,
        "md_outputs": md_outputs,
        "frame_outputs": frame_outputs,
        "transition_outputs": transition_outputs,
        "clustering_comparison": comparisons,
    }
    summary_path = out_root / "swav_prototype_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    summary["summary_json"] = str(summary_path)
    return summary


__all__ = [
    "compute_swav_assignments",
    "run_swav_prototype_evaluation",
]
