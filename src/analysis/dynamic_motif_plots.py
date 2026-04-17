from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_parent(out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)


def _projection_2d(latents: np.ndarray) -> np.ndarray:
    values = np.asarray(latents, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    if centered.shape[1] < 2:
        padded = np.zeros((centered.shape[0], 2), dtype=np.float64)
        padded[:, : centered.shape[1]] = centered
        return padded
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    return centered @ basis


def plot_transition_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    out_file: Path,
) -> None:
    _ensure_parent(out_file)
    if matrix.size == 0:
        return
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    norm = matrix / max(float(matrix.sum()), 1.0)
    image = ax.imshow(norm, cmap="magma", interpolation="nearest")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_title("Transition probabilities")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_transition_flow(
    transition_counts: pd.DataFrame,
    *,
    out_file: Path,
    top_k: int,
) -> None:
    _ensure_parent(out_file)
    if transition_counts.empty:
        return
    top = transition_counts.head(int(top_k)).copy()
    labels = [
        f"{row.from_label} -> {row.to_label}"
        for row in top.itertuples(index=False)
    ]
    values = top["count"].to_numpy(dtype=np.float64)
    fig_height = max(4.0, 0.35 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(9.0, fig_height))
    y = np.arange(len(labels))
    ax.barh(y, values, color="#2a6f97", alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Transition count")
    ax.set_title("Top transitions")
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_motif_proportion_area(frame_df: pd.DataFrame, out_file: Path) -> None:
    _ensure_parent(out_file)
    if frame_df.empty:
        return
    stable = frame_df.loc[frame_df["motif_family"] == "stable"].copy()
    if stable.empty:
        return
    aggregated = (
        stable.groupby(["frame_index", "motif_id"], sort=True)["fraction"]
        .mean()
        .reset_index()
    )
    pivot = aggregated.pivot(index="frame_index", columns="motif_id", values="fraction").fillna(0.0)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = pivot.index.to_numpy(dtype=np.int64)
    y = [pivot[column].to_numpy(dtype=np.float64) for column in pivot.columns]
    labels = [f"S{int(column)}" for column in pivot.columns]
    ax.stackplot(x, y, labels=labels, alpha=0.92)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Mean motif fraction")
    ax.set_title("Stable motif proportions over time")
    if len(labels) <= 12:
        ax.legend(loc="upper left", ncols=min(4, len(labels)))
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_dwell_histograms(dwell_df: pd.DataFrame, out_file: Path) -> None:
    _ensure_parent(out_file)
    if dwell_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), sharey=True)
    for ax, family in zip(axes, ["stable", "bridge"], strict=True):
        subset = dwell_df.loc[dwell_df["motif_family"] == family]
        if subset.empty:
            ax.set_visible(False)
            continue
        ax.hist(subset["dwell_length"].to_numpy(dtype=np.int64), bins=20, color="#386641", alpha=0.85)
        ax.set_title(f"{family.title()} dwell lengths")
        ax.set_xlabel("Length")
    axes[0].set_ylabel("Run count")
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_survival_curves(survival_df: pd.DataFrame, out_file: Path) -> None:
    _ensure_parent(out_file)
    if survival_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    grouped = survival_df.groupby(["motif_family", "motif_id"], sort=True)
    for (family, motif_id), group in list(grouped)[:12]:
        group = group.sort_values("elapsed_dwell")
        ax.plot(
            group["elapsed_dwell"].to_numpy(dtype=np.int64),
            group["survival_probability"].to_numpy(dtype=np.float64),
            label=f"{family[0].upper()}{int(motif_id)}",
            linewidth=1.8,
        )
    ax.set_xlabel("Elapsed dwell length")
    ax.set_ylabel("Survival probability")
    ax.set_title("Motif survival curves")
    ax.set_ylim(0.0, 1.05)
    if grouped.ngroups <= 12:
        ax.legend(loc="upper right", ncols=min(4, grouped.ngroups))
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_recurrence_heatmap(revisit_df: pd.DataFrame, out_file: Path) -> None:
    _ensure_parent(out_file)
    if revisit_df.empty:
        return
    heatmap_df = revisit_df.copy()
    heatmap_df["motif_label"] = [
        f"{family[0].upper()}{int(motif_id)}"
        for family, motif_id in zip(
            heatmap_df["motif_family"].tolist(),
            heatmap_df["motif_id"].tolist(),
            strict=True,
        )
    ]
    heatmap_df["gap_bin"] = np.clip(heatmap_df["gap_frames"].to_numpy(dtype=np.int64), 0, 32)
    pivot = (
        heatmap_df.groupby(["motif_label", "gap_bin"], sort=True)
        .size()
        .reset_index(name="count")
        .pivot(index="motif_label", columns="gap_bin", values="count")
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(8.6, max(3.5, 0.35 * pivot.shape[0] + 1.2)))
    image = ax.imshow(pivot.to_numpy(dtype=np.float64), aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(list(pivot.index))
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([str(int(v)) for v in pivot.columns], rotation=90)
    ax.set_xlabel("Revisit gap (frames, clipped)")
    ax.set_title("Motif recurrence gaps")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_bridge_event_gallery(events: list[dict[str, Any]], out_file: Path) -> None:
    _ensure_parent(out_file)
    if not events:
        return
    fig_height = max(3.6, 0.8 * len(events) + 1.4)
    fig, ax = plt.subplots(figsize=(10.0, fig_height))
    y_positions = np.arange(len(events))
    for row_idx, event in enumerate(events):
        timeline = list(event["timeline"])
        x = [int(item["frame_index"]) for item in timeline]
        colors = ["#c1121f" if bool(item["bridge_active"]) else "#1d3557" for item in timeline]
        ax.scatter(x, np.full(len(x), row_idx), c=colors, s=60, edgecolors="none")
        for item in timeline:
            ax.text(
                int(item["frame_index"]),
                row_idx + 0.12,
                str(item["resolved_state_label"]),
                fontsize=7,
                rotation=35,
                ha="left",
                va="bottom",
            )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [
            f"{Path(str(event['source_path'])).stem}:atom{int(event['center_atom_id'])}"
            for event in events
        ]
    )
    ax.set_xlabel("Frame index")
    ax.set_title("Bridge event gallery")
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_motif_latent_2d(
    latents: np.ndarray,
    *,
    stable_ids: np.ndarray,
    bridge_active: np.ndarray | None,
    hazard_probs: np.ndarray | None,
    out_file: Path,
) -> np.ndarray:
    _ensure_parent(out_file)
    projection = _projection_2d(latents)
    if projection.size == 0:
        return projection
    stable = np.asarray(stable_ids, dtype=np.int64).reshape(-1)
    bridge_mask = (
        np.asarray(bridge_active, dtype=bool).reshape(-1)
        if bridge_active is not None
        else np.zeros(stable.shape[0], dtype=bool)
    )
    hazard = (
        np.asarray(hazard_probs, dtype=np.float64).reshape(-1)
        if hazard_probs is not None
        else np.zeros(stable.shape[0], dtype=np.float64)
    )
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    non_bridge = ~bridge_mask
    scatter = ax.scatter(
        projection[non_bridge, 0],
        projection[non_bridge, 1],
        c=stable[non_bridge],
        cmap="tab20",
        s=(10.0 + 18.0 * hazard[non_bridge]),
        alpha=0.72,
        edgecolors="none",
    )
    if np.any(bridge_mask):
        ax.scatter(
            projection[bridge_mask, 0],
            projection[bridge_mask, 1],
            c=stable[bridge_mask],
            cmap="tab20",
            s=(18.0 + 28.0 * hazard[bridge_mask]),
            alpha=0.92,
            edgecolors="#d00000",
            linewidths=0.6,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Motif latent projection")
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Stable motif")
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)
    return projection


def plot_hazard_calibration_curve(calibration_df: pd.DataFrame, out_file: Path) -> None:
    _ensure_parent(out_file)
    if calibration_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    for lag, group in calibration_df.groupby("lag", sort=True):
        ax.plot(
            group["predicted_mean"].to_numpy(dtype=np.float64),
            group["empirical_rate"].to_numpy(dtype=np.float64),
            marker="o",
            linewidth=1.8,
            label=f"lag {int(lag)}",
        )
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#555555", linewidth=1.0)
    ax.set_xlabel("Predicted hazard")
    ax.set_ylabel("Empirical change rate")
    ax.set_title("Hazard calibration")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_neighbor_influence_heatmap(field_df: pd.DataFrame, out_file: Path) -> None:
    _ensure_parent(out_file)
    if field_df.empty:
        return
    plot_df = field_df.loc[field_df["target_motif_id"] >= 0].copy()
    if plot_df.empty:
        plot_df = field_df.copy()
    pivot = (
        plot_df.pivot(index="lag", columns="target_motif_id", values="gain")
        .fillna(0.0)
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(8.0, max(3.8, 0.5 * pivot.shape[0] + 1.5)))
    image = ax.imshow(pivot.to_numpy(dtype=np.float64), aspect="auto", cmap="coolwarm", vmin=-np.max(np.abs(pivot.values)), vmax=np.max(np.abs(pivot.values)))
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([f"lag {int(v)}" for v in pivot.index])
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([str(int(v)) for v in pivot.columns], rotation=90)
    ax.set_xlabel("Target stable motif")
    ax.set_title("Neighbor-conditioned gain")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Accuracy gain")
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_representative_panel(
    projection_2d: np.ndarray,
    sample_df: pd.DataFrame,
    *,
    motif_family: str,
    motif_id: int,
    representative_sample_indices: list[int],
    out_file: Path,
) -> None:
    _ensure_parent(out_file)
    if projection_2d.size == 0 or sample_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    ax.scatter(
        projection_2d[:, 0],
        projection_2d[:, 1],
        c="#cccccc",
        s=8,
        alpha=0.22,
        edgecolors="none",
    )
    motif_mask = (
        (sample_df["resolved_family"] == str(motif_family))
        & (sample_df["resolved_motif_id"] == int(motif_id))
    ).to_numpy(dtype=bool)
    ax.scatter(
        projection_2d[motif_mask, 0],
        projection_2d[motif_mask, 1],
        c="#1f77b4" if motif_family == "stable" else "#d62828",
        s=18,
        alpha=0.82,
        edgecolors="none",
    )
    rep_mask = sample_df["sample_index"].isin(representative_sample_indices).to_numpy(dtype=bool)
    ax.scatter(
        projection_2d[rep_mask, 0],
        projection_2d[rep_mask, 1],
        c="#ffbe0b",
        s=70,
        marker="*",
        edgecolors="#000000",
        linewidths=0.5,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{motif_family.title()} motif {int(motif_id)} representatives")
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


__all__ = [
    "plot_bridge_event_gallery",
    "plot_dwell_histograms",
    "plot_hazard_calibration_curve",
    "plot_motif_latent_2d",
    "plot_motif_proportion_area",
    "plot_neighbor_influence_heatmap",
    "plot_recurrence_heatmap",
    "plot_representative_panel",
    "plot_survival_curves",
    "plot_transition_flow",
    "plot_transition_heatmap",
]
