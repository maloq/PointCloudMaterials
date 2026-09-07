"""Analysis utilities for time-resolved frozen GeoFrameTransformer embeddings."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from src.temporal_vamp.evaluation import CovariancePCA
from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache


ROLE_ORDER = ("transition_candidate", "liquid_control", "crystal_control")
ROLE_LABELS = {
    "transition_candidate": "transition candidate",
    "liquid_control": "liquid control",
    "crystal_control": "crystal control",
}
ROLE_COLORS = {
    "transition_candidate": "#7b2cbf",
    "liquid_control": "#1982c4",
    "crystal_control": "#d97706",
}


def assemble_time_series(cache: ShootingEmbeddingCache) -> np.ndarray:
    """Return ``(branch, frame, center, feature)`` including the shared t=0 parent."""

    future = np.asarray(cache.future_z, dtype=np.float32)
    branch_parent = np.asarray(cache.branch_parent_index, dtype=np.int64)
    parent = np.asarray(cache.parent_local_z, dtype=np.float32)
    series = np.empty(
        (future.shape[0], future.shape[1] + 1, future.shape[2], future.shape[3]),
        dtype=np.float32,
    )
    series[:, 0] = parent[branch_parent]
    series[:, 1:] = future
    return series


def embedding_rms_radius(values: np.ndarray, batch_size: int = 65536) -> float:
    """RMS Euclidean radius about the global embedding mean."""

    flat = values.reshape(-1, values.shape[-1])
    total = np.zeros(flat.shape[1], dtype=np.float64)
    total_sq = 0.0
    for start in range(0, flat.shape[0], int(batch_size)):
        chunk = np.asarray(flat[start : start + batch_size], dtype=np.float64)
        total += chunk.sum(axis=0)
        total_sq += float(np.square(chunk).sum())
    mean = total / float(flat.shape[0])
    variance_trace = total_sq / float(flat.shape[0]) - float(mean @ mean)
    return float(np.sqrt(max(variance_trace, 0.0)))


def _branch_groups(
    parent_temperatures: np.ndarray,
    parent_roles: np.ndarray,
    branch_parent: np.ndarray,
) -> list[tuple[str, str, np.ndarray]]:
    groups: list[tuple[str, str, np.ndarray]] = [
        ("all", "all", np.ones(branch_parent.size, dtype=bool))
    ]
    branch_temperatures = parent_temperatures[branch_parent]
    branch_roles = parent_roles[branch_parent]
    for temperature in sorted(set(parent_temperatures.tolist())):
        groups.append(
            (
                "temperature",
                f"{float(temperature):g} K",
                branch_temperatures == temperature,
            )
        )
    for role in ROLE_ORDER:
        if role in set(parent_roles.tolist()):
            groups.append(("role", role, branch_roles == role))
    for temperature in sorted(set(parent_temperatures.tolist())):
        for role in ROLE_ORDER:
            mask = (branch_temperatures == temperature) & (branch_roles == role)
            if np.any(mask):
                groups.append(
                    (
                        "temperature_role",
                        f"{float(temperature):g} K / {role}",
                        mask,
                    )
                )
    return groups


def _parent_decomposition(
    values: np.ndarray,
    branch_parent: np.ndarray,
    selected_branches: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean squared coherent displacement and within-parent shot dispersion."""

    coherent: list[np.ndarray] = []
    stochastic: list[np.ndarray] = []
    for parent_index in np.unique(branch_parent[selected_branches]).tolist():
        indices = selected_branches[branch_parent[selected_branches] == parent_index]
        if indices.size < 2:
            raise RuntimeError(
                f"Temporal decomposition needs at least two shots, parent={parent_index} "
                f"has {indices.size}."
            )
        delta = values[indices] - values[indices, :1]
        mean_delta = delta.mean(axis=0, dtype=np.float64)
        coherent.append(np.square(mean_delta).sum(axis=-1))
        stochastic.append(
            np.square(delta.astype(np.float64) - mean_delta[None, ...])
            .sum(axis=-1)
            .mean(axis=0)
        )
    return np.concatenate(coherent, axis=1), np.concatenate(stochastic, axis=1)


def compute_time_metrics(
    values: np.ndarray,
    *,
    times_ps: np.ndarray,
    branch_parent: np.ndarray,
    parent_temperatures: np.ndarray,
    parent_roles: np.ndarray,
    rms_radius: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_kind, group, mask in _branch_groups(
        parent_temperatures, parent_roles, branch_parent
    ):
        selected = np.flatnonzero(mask)
        group_rms_radius = embedding_rms_radius(values[selected])
        shot_counts = np.asarray(
            [np.sum(branch_parent[selected] == parent) for parent in np.unique(branch_parent[selected])],
            dtype=np.int64,
        )
        if np.unique(shot_counts).size != 1:
            raise RuntimeError(
                f"Finite-shot correction requires equal shots per selected parent, got {shot_counts.tolist()}."
            )
        shots_per_parent = int(shot_counts[0])
        delta = values[selected] - values[selected, :1]
        drift_norm = np.linalg.norm(delta, axis=-1)
        initial = values[selected, :1]
        denominator = np.linalg.norm(initial, axis=-1) * np.linalg.norm(
            values[selected], axis=-1
        )
        cosine = np.divide(
            np.sum(initial * values[selected], axis=-1),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0.0,
        )
        coherent_sq, stochastic_sq = _parent_decomposition(
            values, branch_parent, selected
        )
        for frame_index, time_ps in enumerate(times_ps.tolist()):
            drift = drift_norm[:, frame_index].reshape(-1)
            coherent_energy = float(coherent_sq[frame_index].mean())
            stochastic_energy = float(stochastic_sq[frame_index].mean())
            total_energy = coherent_energy + stochastic_energy
            conditional_variance_unbiased = (
                stochastic_energy * shots_per_parent / float(shots_per_parent - 1)
            )
            conditional_mean_energy_debiased = max(
                coherent_energy - stochastic_energy / float(shots_per_parent - 1),
                0.0,
            )
            corrected_total = (
                conditional_variance_unbiased + conditional_mean_energy_debiased
            )
            row: dict[str, Any] = {
                "group_kind": group_kind,
                "group": group,
                "time_ps": float(time_ps),
                "branch_count": int(selected.size),
                "sample_count": int(drift.size),
                "drift_mean": float(drift.mean()),
                "drift_rms": float(np.sqrt(np.square(drift).mean())),
                "drift_median": float(np.median(drift)),
                "drift_q10": float(np.quantile(drift, 0.10)),
                "drift_q90": float(np.quantile(drift, 0.90)),
                "drift_rms_over_static_radius": float(
                    np.sqrt(np.square(drift).mean()) / group_rms_radius
                ),
                "normalization_rms_radius": group_rms_radius,
                "cosine_to_start_mean": float(
                    cosine[:, frame_index].mean()
                ),
                "coherent_rms": float(np.sqrt(coherent_energy)),
                "sibling_dispersion_rms": float(np.sqrt(stochastic_energy)),
                "stochastic_energy_fraction": (
                    float(stochastic_energy / total_energy)
                    if total_energy > 0.0
                    else 0.0
                ),
                "conditional_variance_rms_unbiased": float(
                    np.sqrt(conditional_variance_unbiased)
                ),
                "conditional_mean_change_rms_debiased": float(
                    np.sqrt(conditional_mean_energy_debiased)
                ),
                "conditional_variance_energy_fraction_debiased": (
                    float(conditional_variance_unbiased / corrected_total)
                    if corrected_total > 0.0
                    else 0.0
                ),
            }
            if frame_index == 0:
                row["adjacent_step_rms"] = 0.0
                row["adjacent_step_mean"] = 0.0
            else:
                step = np.linalg.norm(
                    values[selected, frame_index]
                    - values[selected, frame_index - 1],
                    axis=-1,
                ).reshape(-1)
                row["adjacent_step_rms"] = float(np.sqrt(np.square(step).mean()))
                row["adjacent_step_mean"] = float(step.mean())
            rows.append(row)
    return rows


def compute_lag_metrics(
    values: np.ndarray,
    *,
    lag_frames: Sequence[int],
    frame_interval_ps: float,
    branch_parent: np.ndarray,
    parent_temperatures: np.ndarray,
    parent_roles: np.ndarray,
    rms_radius: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = _branch_groups(parent_temperatures, parent_roles, branch_parent)
    for group_kind, group, mask in groups:
        selected = np.flatnonzero(mask)
        group_rms_radius = embedding_rms_radius(values[selected])
        trajectories = values[selected].astype(np.float64)
        centered = trajectories - trajectories.mean(axis=1, keepdims=True)
        for lag in lag_frames:
            if lag <= 0 or lag >= values.shape[1]:
                raise ValueError(
                    f"Lag must be within [1, {values.shape[1] - 1}], got {lag}."
                )
            distance_parts: list[np.ndarray] = []
            cosine_parts: list[np.ndarray] = []
            centered_numerator = 0.0
            centered_left_energy = 0.0
            centered_right_energy = 0.0
            for branch_index in range(trajectories.shape[0]):
                left = trajectories[branch_index, :-lag]
                right = trajectories[branch_index, lag:]
                difference = right - left
                distance_parts.append(np.linalg.norm(difference, axis=-1).reshape(-1))
                denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(
                    right, axis=-1
                )
                cosine_parts.append(
                    np.divide(
                        np.sum(left * right, axis=-1),
                        denominator,
                        out=np.zeros_like(denominator),
                        where=denominator > 0.0,
                    ).reshape(-1)
                )
                centered_left = centered[branch_index, :-lag]
                centered_right = centered[branch_index, lag:]
                centered_numerator += float(np.sum(centered_left * centered_right))
                centered_left_energy += float(np.square(centered_left).sum())
                centered_right_energy += float(np.square(centered_right).sum())
            distance = np.concatenate(distance_parts)
            raw_cosine = np.concatenate(cosine_parts)
            centered_correlation = centered_numerator / np.sqrt(
                centered_left_energy * centered_right_energy
            )
            rows.append(
                {
                    "group_kind": group_kind,
                    "group": group,
                    "lag_frames": int(lag),
                    "lag_ps": float(lag * frame_interval_ps),
                    "pair_count": int(distance.size),
                    "distance_mean": float(distance.mean()),
                    "distance_rms": float(np.sqrt(np.square(distance).mean())),
                    "distance_median": float(np.median(distance)),
                    "distance_q10": float(np.quantile(distance, 0.10)),
                    "distance_q90": float(np.quantile(distance, 0.90)),
                    "distance_rms_over_static_radius": float(
                        np.sqrt(np.square(distance).mean()) / group_rms_radius
                    ),
                    "normalization_rms_radius": group_rms_radius,
                    "raw_cosine_mean": float(raw_cosine.mean()),
                    "trajectory_centered_correlation": float(centered_correlation),
                }
            )
    return rows


def compute_increment_alignment(values: np.ndarray) -> dict[str, float]:
    increments = np.diff(values.astype(np.float64), axis=1)
    left = increments[:, :-1]
    right = increments[:, 1:]
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    cosine = np.divide(
        np.sum(left * right, axis=-1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    ).reshape(-1)
    return {
        "mean": float(cosine.mean()),
        "median": float(np.median(cosine)),
        "q10": float(np.quantile(cosine, 0.10)),
        "q90": float(np.quantile(cosine, 0.90)),
        "positive_fraction": float(np.mean(cosine > 0.0)),
        "sample_count": int(cosine.size),
    }


def compute_smoothing_metrics(
    values: np.ndarray, windows_frames: Sequence[int], frame_interval_ps: float
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for window in windows_frames:
        width = int(window)
        if width <= 0 or width > values.shape[1]:
            raise ValueError(
                f"Smoothing window must be within [1, {values.shape[1]}], got {width}."
            )
        cumulative = np.concatenate(
            [
                np.zeros_like(values[:, :1], dtype=np.float64),
                np.cumsum(values.astype(np.float64), axis=1),
            ],
            axis=1,
        )
        smoothed = (cumulative[:, width:] - cumulative[:, :-width]) / float(width)
        steps = np.linalg.norm(np.diff(smoothed, axis=1), axis=-1)
        path_length = steps.sum(axis=1)
        net = np.linalg.norm(smoothed[:, -1] - smoothed[:, 0], axis=-1)
        tortuosity = np.divide(
            path_length,
            net,
            out=np.full_like(path_length, np.nan),
            where=net > 1.0e-12,
        )
        rows.append(
            {
                "window_frames": width,
                "window_ps": float(width * frame_interval_ps),
                "output_frames": int(smoothed.shape[1]),
                "adjacent_step_rms": float(np.sqrt(np.square(steps).mean())),
                "mean_path_length": float(path_length.mean()),
                "mean_net_displacement": float(net.mean()),
                "median_tortuosity": float(np.nanmedian(tortuosity)),
            }
        )
    return rows


def temporal_dimension_metrics(values: np.ndarray) -> list[dict[str, float | int]]:
    flat = values.reshape(-1, values.shape[-1]).astype(np.float64)
    static_variance = flat.var(axis=0)
    increments = np.diff(values.astype(np.float64), axis=1).reshape(-1, values.shape[-1])
    increment_variance = increments.var(axis=0)
    total_increment = float(increment_variance.sum())
    return [
        {
            "dimension": int(index),
            "static_variance": float(static_variance[index]),
            "increment_variance": float(increment_variance[index]),
            "increment_variance_fraction": float(
                increment_variance[index] / total_increment
            ),
            "increment_to_static_variance": float(
                increment_variance[index] / static_variance[index]
            ),
        }
        for index in range(values.shape[-1])
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty metrics table: {path}.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rows_for(
    rows: list[dict[str, Any]], *, group_kind: str, group: str
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["group_kind"] == group_kind and row["group"] == group
    ]


def _plot_time_metrics(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for role in ROLE_ORDER:
        selected = _rows_for(rows, group_kind="role", group=role)
        if not selected:
            continue
        time = np.asarray([row["time_ps"] for row in selected])
        mean = np.asarray([row["drift_mean"] for row in selected])
        q10 = np.asarray([row["drift_q10"] for row in selected])
        q90 = np.asarray([row["drift_q90"] for row in selected])
        axes[0].plot(time, mean, label=ROLE_LABELS[role], color=ROLE_COLORS[role])
        axes[0].fill_between(time, q10, q90, color=ROLE_COLORS[role], alpha=0.15)
    for temperature, color in (("400 K", "#2166ac"), ("450 K", "#762a83"), ("500 K", "#b2182b")):
        selected = _rows_for(rows, group_kind="temperature", group=temperature)
        time = [row["time_ps"] for row in selected]
        axes[1].plot(
            time,
            [row["drift_mean"] for row in selected],
            label=temperature,
            color=color,
        )
    axes[0].set_title("Displacement from the parent embedding")
    axes[1].set_title("Temperature dependence")
    for axis in axes:
        axis.set_xlabel("time after shooting (ps)")
        axis.set_ylabel("embedding distance")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.savefig(plot_dir / "embedding_displacement_vs_time.png", dpi=180)
    plt.close(fig)


def _plot_decomposition(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for role in ROLE_ORDER:
        selected = _rows_for(rows, group_kind="role", group=role)
        if not selected:
            continue
        time = [row["time_ps"] for row in selected]
        axes[0].plot(
            time,
            [row["sibling_dispersion_rms"] for row in selected],
            color=ROLE_COLORS[role],
            label=ROLE_LABELS[role],
        )
        axes[1].plot(
            time,
            [
                row["conditional_variance_energy_fraction_debiased"]
                for row in selected
            ],
            color=ROLE_COLORS[role],
            label=ROLE_LABELS[role],
        )
    axes[0].set_title("Divergence of four sibling futures")
    axes[0].set_ylabel("within-parent RMS dispersion")
    axes[1].set_title("Conditional-variance share (four-shot corrected)")
    axes[1].set_ylabel("conditional variance / total change energy")
    axes[1].set_ylim(-0.02, 1.02)
    for axis in axes:
        axis.set_xlabel("time after shooting (ps)")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.savefig(plot_dir / "sibling_future_divergence.png", dpi=180)
    plt.close(fig)


def _plot_lag_metrics(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for role in ROLE_ORDER:
        selected = _rows_for(rows, group_kind="role", group=role)
        if not selected:
            continue
        lag = [row["lag_ps"] for row in selected]
        axes[0].plot(
            lag,
            [row["distance_rms_over_static_radius"] for row in selected],
            "o-",
            color=ROLE_COLORS[role],
            label=ROLE_LABELS[role],
        )
        axes[1].plot(
            lag,
            [row["trajectory_centered_correlation"] for row in selected],
            "o-",
            color=ROLE_COLORS[role],
            label=ROLE_LABELS[role],
        )
    axes[0].set_title("Lag-dependent embedding change")
    axes[0].set_ylabel("RMS distance / group static RMS radius")
    axes[1].set_title("Within-trajectory autocorrelation")
    axes[1].set_ylabel("trajectory-centered correlation")
    axes[1].axhline(0.0, color="black", lw=0.8, alpha=0.5)
    for axis in axes:
        axis.set_xlabel("lag (ps)")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.savefig(plot_dir / "lag_dependence.png", dpi=180)
    plt.close(fig)


def _plot_pca_spectrum(
    global_pca: CovariancePCA, delta_pca: CovariancePCA, plot_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, pca, title in (
        (axes[0], global_pca, "Absolute embeddings"),
        (axes[1], delta_pca, "Changes from the parent"),
    ):
        ratio = pca.eigenvalues_ / pca.eigenvalues_.sum()
        count = min(32, ratio.size)
        axis.bar(np.arange(1, count + 1), ratio[:count], color="#4c78a8")
        axis.plot(
            np.arange(1, count + 1), np.cumsum(ratio[:count]), color="#e45756", marker="."
        )
        axis.set_title(title)
        axis.set_xlabel("principal component")
        axis.set_ylabel("variance fraction / cumulative")
        axis.grid(axis="y", alpha=0.2)
    fig.savefig(plot_dir / "pca_spectra.png", dpi=180)
    plt.close(fig)


def _plot_pca_scatter(
    values: np.ndarray,
    delta: np.ndarray,
    global_pca: CovariancePCA,
    delta_pca: CovariancePCA,
    times_ps: np.ndarray,
    parent_roles: np.ndarray,
    branch_parent: np.ndarray,
    plot_dir: Path,
    *,
    maximum_points: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(int(seed))
    branch_count, frame_count, center_count, feature_dim = values.shape
    total = branch_count * frame_count * center_count
    flat_indices = np.sort(rng.choice(total, size=min(maximum_points, total), replace=False))
    branch, remainder = np.divmod(flat_indices, frame_count * center_count)
    frame, center = np.divmod(remainder, center_count)
    absolute_values = values[branch, frame, center].reshape(-1, feature_dim)
    delta_values = delta[branch, frame, center].reshape(-1, feature_dim)
    absolute_xy = global_pca.transform(absolute_values, dimension=2)
    delta_xy = delta_pca.transform(delta_values, dimension=2)
    color_time = times_ps[frame]
    roles = parent_roles[branch_parent[branch]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    scatter = axes[0].scatter(
        absolute_xy[:, 0], absolute_xy[:, 1], c=color_time, s=2, alpha=0.25, cmap="viridis"
    )
    fig.colorbar(scatter, ax=axes[0], label="time (ps)")
    for role in ROLE_ORDER:
        mask = roles == role
        if np.any(mask):
            axes[1].scatter(
                delta_xy[mask, 0],
                delta_xy[mask, 1],
                s=3,
                alpha=0.25,
                color=ROLE_COLORS[role],
                label=ROLE_LABELS[role],
            )
    axes[0].set_title("Absolute embedding PCA")
    axes[1].set_title("PCA of change from parent")
    axes[1].legend(frameon=False, markerscale=3)
    for axis in axes:
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.grid(alpha=0.15)
    fig.savefig(plot_dir / "embedding_pca_overview.png", dpi=180)
    plt.close(fig)


def select_representative_trajectories(
    values: np.ndarray,
    *,
    parent_temperatures: np.ndarray,
    parent_roles: np.ndarray,
    branch_parent: np.ndarray,
    atom_ids: np.ndarray,
    parents: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    requests = [
        ("transition_candidate", 400.0),
        ("transition_candidate", 450.0),
        ("transition_candidate", 500.0),
        ("liquid_control", 450.0),
        ("crystal_control", 450.0),
    ]
    representatives: list[dict[str, Any]] = []
    for role, temperature in requests:
        candidates = np.flatnonzero(
            (parent_roles == role) & (parent_temperatures == temperature)
        )
        if candidates.size == 0:
            continue
        parent_scores = []
        for parent_index in candidates.tolist():
            branches = np.flatnonzero(branch_parent == parent_index)
            endpoint = np.linalg.norm(
                values[branches, -1] - values[branches, 0], axis=-1
            )
            parent_scores.append(float(endpoint.mean()))
        median_score = float(np.median(parent_scores))
        chosen_position = int(np.argmin(np.abs(np.asarray(parent_scores) - median_score)))
        parent_index = int(candidates[chosen_position])
        branches = np.flatnonzero(branch_parent == parent_index)
        center_scores = np.linalg.norm(
            values[branches, -1] - values[branches, 0], axis=-1
        ).mean(axis=0)
        center_target = float(np.median(center_scores))
        center_index = int(np.argmin(np.abs(center_scores - center_target)))
        representatives.append(
            {
                "role": role,
                "temperature_K": float(temperature),
                "parent_index": parent_index,
                "parent_id": str(parents[parent_index]["parent_id"]),
                "atom_index": center_index,
                "atom_id": int(atom_ids[center_index]),
                "branch_indices": branches.tolist(),
                "endpoint_distance_mean": float(center_scores[center_index]),
                "selection": "median parent and median atom by 24 ps embedding displacement",
            }
        )
    return representatives


def _plot_representative_trajectories(
    delta: np.ndarray,
    delta_pca: CovariancePCA,
    times_ps: np.ndarray,
    representatives: list[dict[str, Any]],
    plot_dir: Path,
) -> None:
    columns = 3
    rows = int(np.ceil(len(representatives) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(5.0 * columns, 4.4 * rows), constrained_layout=True
    )
    flat_axes = np.asarray(axes).reshape(-1)
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.05, 0.95, times_ps.size))
    for axis, record in zip(flat_axes, representatives):
        for shot_number, branch_index in enumerate(record["branch_indices"]):
            path = delta_pca.transform(
                delta[int(branch_index), :, int(record["atom_index"])], dimension=2
            )
            axis.plot(path[:, 0], path[:, 1], color="#555555", alpha=0.45, lw=0.9)
            axis.scatter(path[:, 0], path[:, 1], c=colors, s=9, alpha=0.9)
            axis.scatter(path[0, 0], path[0, 1], marker="*", s=80, color="black", zorder=4)
        axis.set_title(
            f"{ROLE_LABELS[record['role']]}, {record['temperature_K']:g} K\n"
            f"parent {record['parent_index']}, atom {record['atom_id']}"
        )
        axis.set_xlabel("change PC1")
        axis.set_ylabel("change PC2")
        axis.grid(alpha=0.15)
    for axis in flat_axes[len(representatives) :]:
        axis.set_visible(False)
    fig.savefig(plot_dir / "representative_embedding_trajectories.png", dpi=180)
    plt.close(fig)


def _plot_smoothed_representative_trajectories(
    delta: np.ndarray,
    delta_pca: CovariancePCA,
    times_ps: np.ndarray,
    representatives: list[dict[str, Any]],
    plot_dir: Path,
    *,
    window_frames: int = 5,
) -> None:
    columns = 3
    rows = int(np.ceil(len(representatives) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(5.0 * columns, 4.4 * rows), constrained_layout=True
    )
    flat_axes = np.asarray(axes).reshape(-1)
    cmap = plt.get_cmap("viridis")
    smoothed_times = np.convolve(
        times_ps, np.ones(window_frames) / float(window_frames), mode="valid"
    )
    normalization = plt.Normalize(float(smoothed_times[0]), float(smoothed_times[-1]))
    for axis, record in zip(flat_axes, representatives):
        for branch_index in record["branch_indices"]:
            raw = delta[int(branch_index), :, int(record["atom_index"])].astype(
                np.float64
            )
            cumulative = np.concatenate(
                [np.zeros((1, raw.shape[1]), dtype=np.float64), np.cumsum(raw, axis=0)],
                axis=0,
            )
            smoothed = (
                cumulative[window_frames:] - cumulative[:-window_frames]
            ) / float(window_frames)
            path = delta_pca.transform(smoothed, dimension=2)
            axis.plot(path[:, 0], path[:, 1], color="#555555", alpha=0.5, lw=1.0)
            axis.scatter(
                path[:, 0],
                path[:, 1],
                c=smoothed_times,
                norm=normalization,
                cmap=cmap,
                s=11,
                alpha=0.9,
            )
            axis.scatter(
                path[0, 0], path[0, 1], marker="*", s=75, color="black", zorder=4
            )
        axis.set_title(
            f"{ROLE_LABELS[record['role']]}, {record['temperature_K']:g} K\n"
            f"parent {record['parent_index']}, atom {record['atom_id']}"
        )
        axis.set_xlabel("change PC1")
        axis.set_ylabel("change PC2")
        axis.grid(alpha=0.15)
    for axis in flat_axes[len(representatives) :]:
        axis.set_visible(False)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=normalization, cmap=cmap),
        ax=[axis for axis in flat_axes if axis.get_visible()],
        label="time (ps)",
        shrink=0.8,
    )
    fig.savefig(
        plot_dir / "representative_embedding_trajectories_smoothed_1p5ps.png",
        dpi=180,
    )
    plt.close(fig)


def _role_pca_spectra(
    values: np.ndarray,
    *,
    parent_roles: np.ndarray,
    branch_parent: np.ndarray,
    dimension: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role in ROLE_ORDER:
        branch_mask = parent_roles[branch_parent] == role
        if not np.any(branch_mask):
            continue
        role_values = values[branch_mask].reshape(-1, values.shape[-1])
        pca = CovariancePCA.fit(role_values, dimension=int(dimension))
        ratio = pca.eigenvalues_ / pca.eigenvalues_.sum()
        for index in range(ratio.size):
            rows.append(
                {
                    "role": role,
                    "component": index + 1,
                    "variance_fraction": float(ratio[index]),
                    "cumulative_fraction": float(ratio[: index + 1].sum()),
                }
            )
    return rows


def _plot_role_pca_spectra(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.7), constrained_layout=True)
    for role in ROLE_ORDER:
        selected = [row for row in rows if row["role"] == role]
        count = min(24, len(selected))
        axis.plot(
            [row["component"] for row in selected[:count]],
            [row["cumulative_fraction"] for row in selected[:count]],
            "o-",
            color=ROLE_COLORS[role],
            label=ROLE_LABELS[role],
        )
    axis.set_xlabel("number of temporal-change PCs")
    axis.set_ylabel("cumulative temporal-change variance")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Temporal-change dimensionality by parent role")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.savefig(plot_dir / "temporal_change_pca_by_role.png", dpi=180)
    plt.close(fig)


def _plot_smoothing(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    windows = [row["window_ps"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    axes[0].plot(windows, [row["adjacent_step_rms"] for row in rows], "o-")
    axes[0].set_ylabel("adjacent-step RMS")
    axes[1].plot(windows, [row["median_tortuosity"] for row in rows], "o-")
    axes[1].set_ylabel("median path length / endpoint distance")
    for axis in axes:
        axis.set_xlabel("moving-average window (ps)")
        axis.grid(alpha=0.2)
    axes[0].set_title("High-frequency embedding motion")
    axes[1].set_title("Embedding-path tortuosity")
    fig.savefig(plot_dir / "temporal_smoothing.png", dpi=180)
    plt.close(fig)


def _plot_dimension_variability(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    static = np.asarray([row["static_variance"] for row in rows])
    temporal = np.asarray([row["increment_variance"] for row in rows])
    fig, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    axis.scatter(static, temporal, s=18, alpha=0.7)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("variance across all states")
    axis.set_ylabel("0.3 ps increment variance")
    axis.set_title("Which embedding dimensions move in time?")
    axis.grid(alpha=0.2)
    fig.savefig(plot_dir / "per_dimension_temporal_variability.png", dpi=180)
    plt.close(fig)


def analyze_temporal_embeddings(
    cache: ShootingEmbeddingCache,
    *,
    output_dir: str | Path,
    lag_frames: Sequence[int],
    smoothing_windows_frames: Sequence[int],
    pca_dimension: int,
    scatter_maximum_points: int,
    seed: int,
) -> dict[str, Any]:
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    plot_dir = target / "plots"
    plot_dir.mkdir(exist_ok=True)
    values = assemble_time_series(cache)
    times_ps = np.concatenate(
        [np.asarray([0.0], dtype=np.float64), np.asarray(cache.horizons_ps)]
    )
    spacing = np.diff(times_ps)
    if not np.allclose(spacing, spacing[0], rtol=0.0, atol=1.0e-10):
        raise RuntimeError(
            f"Fine-time analysis requires uniform frames, got times_ps={times_ps.tolist()}."
        )
    snapshot = cache.manifest["snapshot"]
    parents = snapshot["parents"]
    branches = snapshot["branches"]
    parent_temperatures = np.asarray(
        [float(parent["temperature_K"]) for parent in parents], dtype=np.float64
    )
    parent_roles = np.asarray(
        [str(parent["basin_role"]) for parent in parents], dtype=object
    )
    branch_parent = np.asarray(cache.branch_parent_index, dtype=np.int64)
    rms_radius = embedding_rms_radius(values)
    time_rows = compute_time_metrics(
        values,
        times_ps=times_ps,
        branch_parent=branch_parent,
        parent_temperatures=parent_temperatures,
        parent_roles=parent_roles,
        rms_radius=rms_radius,
    )
    lag_rows = compute_lag_metrics(
        values,
        lag_frames=lag_frames,
        frame_interval_ps=float(spacing[0]),
        branch_parent=branch_parent,
        parent_temperatures=parent_temperatures,
        parent_roles=parent_roles,
        rms_radius=rms_radius,
    )
    smoothing_rows = compute_smoothing_metrics(
        values, smoothing_windows_frames, float(spacing[0])
    )
    dimension_rows = temporal_dimension_metrics(values)
    _write_csv(target / "time_metrics.csv", time_rows)
    _write_csv(target / "lag_metrics.csv", lag_rows)
    _write_csv(target / "smoothing_metrics.csv", smoothing_rows)
    _write_csv(target / "dimension_metrics.csv", dimension_rows)

    flat = values.reshape(-1, values.shape[-1])
    delta = values - values[:, :1]
    global_pca = CovariancePCA.fit(flat, dimension=int(pca_dimension))
    delta_pca = CovariancePCA.fit(
        delta.reshape(-1, delta.shape[-1]), dimension=int(pca_dimension)
    )
    global_pca.save(target / "absolute_embedding_pca.npz")
    delta_pca.save(target / "temporal_change_pca.npz")
    global_ratio = global_pca.eigenvalues_ / global_pca.eigenvalues_.sum()
    delta_ratio = delta_pca.eigenvalues_ / delta_pca.eigenvalues_.sum()
    pca_rows = []
    for index in range(values.shape[-1]):
        pca_rows.append(
            {
                "component": index + 1,
                "absolute_variance_fraction": float(global_ratio[index]),
                "absolute_cumulative_fraction": float(global_ratio[: index + 1].sum()),
                "temporal_change_variance_fraction": float(delta_ratio[index]),
                "temporal_change_cumulative_fraction": float(
                    delta_ratio[: index + 1].sum()
                ),
            }
        )
    _write_csv(target / "pca_spectrum.csv", pca_rows)
    role_pca_rows = _role_pca_spectra(
        delta,
        parent_roles=parent_roles,
        branch_parent=branch_parent,
        dimension=int(pca_dimension),
    )
    _write_csv(target / "temporal_change_pca_by_role.csv", role_pca_rows)
    representatives = select_representative_trajectories(
        values,
        parent_temperatures=parent_temperatures,
        parent_roles=parent_roles,
        branch_parent=branch_parent,
        atom_ids=np.asarray(cache.atom_ids),
        parents=parents,
    )
    with (target / "representatives.json").open("w", encoding="utf-8") as handle:
        json.dump(representatives, handle, indent=2)

    _plot_time_metrics(time_rows, plot_dir)
    _plot_decomposition(time_rows, plot_dir)
    _plot_lag_metrics(lag_rows, plot_dir)
    _plot_pca_spectrum(global_pca, delta_pca, plot_dir)
    _plot_pca_scatter(
        values,
        delta,
        global_pca,
        delta_pca,
        times_ps,
        parent_roles,
        branch_parent,
        plot_dir,
        maximum_points=int(scatter_maximum_points),
        seed=int(seed),
    )
    _plot_representative_trajectories(
        delta, delta_pca, times_ps, representatives, plot_dir
    )
    _plot_smoothed_representative_trajectories(
        delta, delta_pca, times_ps, representatives, plot_dir
    )
    _plot_role_pca_spectra(role_pca_rows, plot_dir)
    _plot_smoothing(smoothing_rows, plot_dir)
    _plot_dimension_variability(dimension_rows, plot_dir)

    all_time = _rows_for(time_rows, group_kind="all", group="all")
    all_lag = _rows_for(lag_rows, group_kind="all", group="all")
    time_by_value = {float(row["time_ps"]): row for row in all_time}
    lag_by_value = {float(row["lag_ps"]): row for row in all_lag}
    selected_times = [value for value in (0.3, 3.0, 6.0, 12.0, 24.0) if value in time_by_value]
    selected_lags = [value for value in (0.3, 1.5, 3.0, 6.0, 12.0, 24.0) if value in lag_by_value]
    increment_alignment = compute_increment_alignment(values)
    summary = {
        "data": {
            "parents": len(parents),
            "branches": len(branches),
            "shots_per_parent": int(len(branches) // len(parents)),
            "center_atoms_per_parent": int(values.shape[2]),
            "frames": int(values.shape[1]),
            "frame_interval_ps": float(spacing[0]),
            "duration_ps": float(times_ps[-1]),
            "point_clouds_encoded": int(
                len(parents) * values.shape[2]
                + len(branches) * (values.shape[1] - 1) * values.shape[2]
            ),
            "temperatures_K": sorted(set(parent_temperatures.tolist())),
            "role_parent_counts": {
                role: int(np.sum(parent_roles == role)) for role in ROLE_ORDER
            },
        },
        "embedding": {
            "dimension": int(values.shape[-1]),
            "global_rms_radius": rms_radius,
            "deterministic_t0_contract": (
                "All shots of a parent share the exact parent coordinates and one cached "
                "deterministic parent embedding."
            ),
        },
        "time_checkpoints": {f"{time:g}ps": time_by_value[time] for time in selected_times},
        "lag_checkpoints": {f"{lag:g}ps": lag_by_value[lag] for lag in selected_lags},
        "increment_direction_alignment": increment_alignment,
        "smoothing": smoothing_rows,
        "pca": {
            "absolute_cumulative_2": float(global_ratio[:2].sum()),
            "absolute_cumulative_8": float(global_ratio[:8].sum()),
            "absolute_cumulative_16": float(global_ratio[:16].sum()),
            "temporal_change_cumulative_2": float(delta_ratio[:2].sum()),
            "temporal_change_cumulative_8": float(delta_ratio[:8].sum()),
            "temporal_change_cumulative_16": float(delta_ratio[:16].sum()),
            "temporal_change_cumulative_by_role": {
                role: {
                    str(count): float(
                        [
                            row
                            for row in role_pca_rows
                            if row["role"] == role and row["component"] == count
                        ][0]["cumulative_fraction"]
                    )
                    for count in (2, 8, 16)
                }
                for role in ROLE_ORDER
            },
        },
    }
    with (target / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def write_experiment_readme(
    output_dir: str | Path,
    *,
    metrics: dict[str, Any],
    checkpoint: Path,
    campaign_root: Path,
    config_path: Path,
) -> None:
    target = Path(output_dir).expanduser().resolve()
    time = metrics["time_checkpoints"]
    lag = metrics["lag_checkpoints"]
    pca = metrics["pca"]
    alignment = metrics["increment_direction_alignment"]
    lines = [
        "# Frozen GeoFrameTransformer embedding variability over 24 ps",
        "",
        "## Experiment",
        "",
        f"- Campaign: `{campaign_root}`",
        f"- Checkpoint: `{checkpoint}`",
        "- Representation: deterministic frozen VICReg projector output (128 dimensions).",
        f"- Data: {metrics['data']['parents']} parents, {metrics['data']['branches']} shooting branches, "
        f"{metrics['data']['center_atoms_per_parent']} fixed atoms, {metrics['data']['frames']} frames at "
        f"{metrics['data']['frame_interval_ps']:.1f} ps spacing through {metrics['data']['duration_ps']:.0f} ps.",
        f"- Point clouds encoded: {metrics['data']['point_clouds_encoded']:,}; each contains the nearest 160 atoms under PBC.",
        "- Absolute PCA and temporal-change PCA are separate. PCA is descriptive only and uses no phase labels.",
        f"- Resolved configuration: `{config_path}`",
        "",
        "## Main numerical observations",
        "",
    ]
    for label in ("0.3ps", "3ps", "6ps", "12ps", "24ps"):
        if label in time:
            row = time[label]
            lines.append(
                f"- At {label.replace('ps', ' ps')}: displacement RMS = {row['drift_rms']:.5f} "
                f"({100.0 * row['drift_rms_over_static_radius']:.2f}% of the static embedding RMS radius); "
                f"conditional-variance RMS = {row['conditional_variance_rms_unbiased']:.5f}; "
                f"finite-shot-corrected random share = "
                f"{100.0 * row['conditional_variance_energy_fraction_debiased']:.1f}%."
            )
    lines.extend(
        [
            f"- Consecutive 0.3 ps increments have mean directional cosine {alignment['mean']:.3f} "
            f"and {100.0 * alignment['positive_fraction']:.1f}% are aligned rather than reversing.",
            f"- Absolute PCA: 2/8/16 PCs explain {100*pca['absolute_cumulative_2']:.1f}% / "
            f"{100*pca['absolute_cumulative_8']:.1f}% / {100*pca['absolute_cumulative_16']:.1f}% of variance.",
            f"- Change PCA: 2/8/16 PCs explain {100*pca['temporal_change_cumulative_2']:.1f}% / "
            f"{100*pca['temporal_change_cumulative_8']:.1f}% / {100*pca['temporal_change_cumulative_16']:.1f}% of temporal-change variance.",
            "",
            "## Interpretation guide",
            "",
            "`drift` is distance from the shared parent embedding. `coherent_rms` is the RMS of the four-shot sample-mean displacement; `sibling_dispersion_rms` is spread around that sample mean. The reported conditional-variance RMS applies the Bessel correction, and the corrected random share also subtracts the expected finite-shot noise remaining in the sample mean. It estimates how much embedding-change energy is branch-random rather than a common conditional-mean motion.",
            "",
            "The trajectory-centered lag correlation removes each atom/shot trajectory's time mean. It is more sensitive to temporal memory than raw cosine similarity, which is dominated by persistent atom identity and static structure.",
            "",
            "## Files",
            "",
            "- `metrics.json`: compact headline metrics.",
            "- `time_metrics.csv`: time-resolved drift and sibling decomposition, stratified by temperature and parent role.",
            "- `lag_metrics.csv`: structure function and autocorrelation versus lag.",
            "- `smoothing_metrics.csv`: path roughness under moving-average windows.",
            "- `pca_spectrum.csv`, `absolute_embedding_pca.npz`, `temporal_change_pca.npz`: reusable PCA results.",
            "- `temporal_change_pca_by_role.csv`: phase-stratified change spectrum, preventing the crystal controls from dominating the global PCA.",
            "- `representatives.json`: exact parent/atom/branch provenance for trajectory panels.",
            "- `plots/`: publication-ready static visualizations.",
            "- `embeddings/`: reusable frozen embeddings; raw point clouds are not copied.",
            "- `representation_comparison/`: same-checkpoint raw-invariant-versus-VICReg diagnostic (created by the comparison entry point).",
            "",
        ]
    )
    if "24ps" in lag:
        lines.append(
            f"At a 24 ps lag, the RMS pair displacement is {lag['24ps']['distance_rms']:.5f} "
            f"and the trajectory-centered correlation is {lag['24ps']['trajectory_centered_correlation']:.3f}."
        )
    (target / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "analyze_temporal_embeddings",
    "assemble_time_series",
    "compute_increment_alignment",
    "compute_lag_metrics",
    "compute_smoothing_metrics",
    "compute_time_metrics",
    "embedding_rms_radius",
    "select_representative_trajectories",
    "temporal_dimension_metrics",
    "write_experiment_readme",
]
