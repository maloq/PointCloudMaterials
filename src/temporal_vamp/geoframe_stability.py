from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.temporal_vamp.evaluation import CovariancePCA


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    numerator = np.sum(a * b, axis=-1)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    if np.any(denominator <= 0.0):
        raise RuntimeError("GeoFrame stability cosine received a zero-norm embedding.")
    return numerator / denominator


def sibling_distances(
    embeddings: np.ndarray,
    branch_parent: np.ndarray,
) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    parents = np.asarray(branch_parent, dtype=np.int64)
    parent_count = int(parents.max()) + 1
    blocks: list[np.ndarray] = []
    for parent in range(parent_count):
        branches = np.flatnonzero(parents == parent)
        if branches.size != 4:
            raise RuntimeError(
                f"Finest-step stability requires four siblings per parent, "
                f"parent={parent}, observed={branches.size}."
            )
        for left, right in combinations(branches.tolist(), 2):
            blocks.append(np.linalg.norm(values[left] - values[right], axis=-1))
    return np.stack(blocks, axis=0)


def same_atom_retrieval(
    query: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, float]:
    current = np.asarray(query, dtype=np.float64)
    initial = np.asarray(reference, dtype=np.float64)
    if current.shape != initial.shape or current.ndim != 3:
        raise ValueError(
            "Same-atom retrieval expects matching [branch, center, feature] arrays, "
            f"got query={current.shape}, reference={initial.shape}."
        )
    correct = 0
    ranks: list[np.ndarray] = []
    center_count = current.shape[1]
    diagonal_index = np.arange(center_count, dtype=np.int64)
    for branch in range(current.shape[0]):
        difference = current[branch, :, None, :] - initial[branch, None, :, :]
        distances = np.sum(np.square(difference), axis=-1)
        predicted = np.argmin(distances, axis=1)
        correct += int(np.sum(predicted == diagonal_index))
        diagonal = distances[diagonal_index, diagonal_index]
        ranks.append(1 + np.sum(distances < diagonal[:, None], axis=1))
    return (
        float(correct / (current.shape[0] * center_count)),
        float(np.median(np.concatenate(ranks))),
    )


def temporal_stability_table(
    embeddings: np.ndarray,
    times_ps: np.ndarray,
    neighbor_retention: np.ndarray,
    cloud_rms_A: np.ndarray,
    branch_parent: np.ndarray,
    cross_state_distances: np.ndarray,
    *,
    seed: int,
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray]:
    values = np.asarray(embeddings, dtype=np.float32)
    times = np.asarray(times_ps, dtype=np.float64)
    retention = np.asarray(neighbor_retention, dtype=np.float32)
    rms = np.asarray(cloud_rms_A, dtype=np.float32)
    if values.ndim != 4:
        raise ValueError(
            f"Temporal stability embeddings must be [branch,time,center,feature], got {values.shape}."
        )
    expected_scalar_shape = values.shape[:3]
    if retention.shape != expected_scalar_shape or rms.shape != expected_scalar_shape:
        raise ValueError(
            "Temporal stability physical diagnostics changed shape: "
            f"expected={expected_scalar_shape}, retention={retention.shape}, rms={rms.shape}."
        )
    if times.shape != (values.shape[1],):
        raise ValueError(
            f"Temporal stability times must have shape {(values.shape[1],)}, got {times.shape}."
        )
    cross = np.asarray(cross_state_distances, dtype=np.float64)
    cross_scale = float(np.median(cross))
    if cross_scale <= 0.0:
        raise RuntimeError(f"Cross-state GeoFrame distance scale is invalid: {cross_scale}.")

    initial = values[:, 0]
    sibling = np.empty((values.shape[1],), dtype=np.float64)
    temporal_distances = np.empty(values.shape[:3], dtype=np.float32)
    rows: list[dict[str, float]] = []
    rng = np.random.default_rng(int(seed))
    for time_index, time_ps in enumerate(times.tolist()):
        distances = np.linalg.norm(
            values[:, time_index].astype(np.float64) - initial.astype(np.float64),
            axis=-1,
        )
        temporal_distances[:, time_index] = distances.astype(np.float32)
        cosine = row_cosine(values[:, time_index], initial)
        consecutive = (
            np.zeros_like(distances)
            if time_index == 0
            else np.linalg.norm(
                values[:, time_index].astype(np.float64)
                - values[:, time_index - 1].astype(np.float64),
                axis=-1,
            )
        )
        sibling_values = sibling_distances(values[:, time_index], branch_parent)
        sibling[time_index] = float(np.mean(sibling_values))
        top1, median_rank = same_atom_retrieval(values[:, time_index], initial)
        random_for_comparison = rng.choice(
            cross, size=distances.size, replace=True
        ).reshape(distances.shape)
        rows.append(
            {
                "time_ps": float(time_ps),
                "embedding_distance_mean": float(np.mean(distances)),
                "embedding_distance_median": float(np.median(distances)),
                "embedding_distance_q05": float(np.quantile(distances, 0.05)),
                "embedding_distance_q95": float(np.quantile(distances, 0.95)),
                "relative_to_cross_state_median": float(
                    np.median(distances) / cross_scale
                ),
                "cosine_mean": float(np.mean(cosine)),
                "cosine_median": float(np.median(cosine)),
                "cosine_q05": float(np.quantile(cosine, 0.05)),
                "consecutive_distance_mean": float(np.mean(consecutive)),
                "same_atom_top1": top1,
                "same_atom_median_rank": median_rank,
                "neighbor_retention_mean": float(np.mean(retention[:, time_index])),
                "cloud_rms_A_mean": float(np.mean(rms[:, time_index])),
                "sibling_distance_mean": float(sibling[time_index]),
                "probability_temporal_closer_than_random": float(
                    np.mean(distances < random_for_comparison)
                ),
            }
        )
    return rows, temporal_distances, sibling


def stratified_stability_rows(
    temporal_distances: np.ndarray,
    embeddings: np.ndarray,
    neighbor_retention: np.ndarray,
    cloud_rms_A: np.ndarray,
    times_ps: np.ndarray,
    branch_temperature_K: np.ndarray,
    branch_role: Sequence[str],
    cross_state_scale: float,
) -> list[dict[str, Any]]:
    values = np.asarray(embeddings, dtype=np.float32)
    initial = values[:, 0]
    roles = np.asarray([str(value) for value in branch_role])
    temperatures = np.asarray(branch_temperature_K, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for temperature in np.unique(temperatures):
        for role in np.unique(roles):
            selected = (temperatures == temperature) & (roles == role)
            if not np.any(selected):
                continue
            for time_index, time_ps in enumerate(np.asarray(times_ps).tolist()):
                distances = np.asarray(temporal_distances[selected, time_index])
                cosine = row_cosine(values[selected, time_index], initial[selected])
                rows.append(
                    {
                        "temperature_K": float(temperature),
                        "role": str(role),
                        "time_ps": float(time_ps),
                        "branch_count": int(np.sum(selected)),
                        "embedding_distance_median": float(np.median(distances)),
                        "relative_to_cross_state_median": float(
                            np.median(distances) / float(cross_state_scale)
                        ),
                        "cosine_mean": float(np.mean(cosine)),
                        "neighbor_retention_mean": float(
                            np.mean(neighbor_retention[selected, time_index])
                        ),
                        "cloud_rms_A_mean": float(
                            np.mean(cloud_rms_A[selected, time_index])
                        ),
                    }
                )
    return rows


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty stability table: {path}.")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_stability_summary(rows: list[dict[str, float]], path: str | Path) -> None:
    time = np.asarray([row["time_ps"] for row in rows])
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    axes[0, 0].plot(time, [row["cosine_mean"] for row in rows], marker="o")
    axes[0, 0].plot(time, [row["cosine_q05"] for row in rows], marker=".", label="5th percentile")
    axes[0, 0].set_ylabel("same-atom cosine")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(
        time,
        [row["relative_to_cross_state_median"] for row in rows],
        marker="o",
    )
    axes[0, 1].set_ylabel("median drift / cross-state median")
    axes[1, 0].plot(time, [row["neighbor_retention_mean"] for row in rows], marker="o")
    axes[1, 0].set_ylabel("initial-neighbor retention")
    axes[1, 1].plot(time, [row["cloud_rms_A_mean"] for row in rows], marker="o")
    axes[1, 1].set_ylabel("atom-matched local RMS (Å)")
    for axis in axes.flat:
        axis.set_xlabel("time from shot start (ps)")
        axis.grid(alpha=0.25)
    figure.suptitle("GeoFrame stability at the finest stored 30 fs interval")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_distance_distributions(
    temporal_distances: np.ndarray,
    cross_state_distances: np.ndarray,
    times_ps: np.ndarray,
    path: str | Path,
) -> None:
    times = np.asarray(times_ps, dtype=np.float64)
    requested = [1, len(times) - 1]
    cross = np.asarray(cross_state_distances, dtype=np.float64)
    high = float(
        np.quantile(
            np.concatenate(
                [cross, *[temporal_distances[:, index].reshape(-1) for index in requested]]
            ),
            0.995,
        )
    )
    bins = np.linspace(0.0, high, 80)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for axis, index in zip(axes, requested, strict=True):
        axis.hist(cross, bins=bins, density=True, alpha=0.45, label="different state")
        axis.hist(
            temporal_distances[:, index].reshape(-1),
            bins=bins,
            density=True,
            alpha=0.65,
            label=f"same atom, {times[index]:.2f} ps",
        )
        axis.set_xlabel("GeoFrame Euclidean distance")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    axes[0].set_ylabel("density")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_sibling_divergence(
    rows: list[dict[str, float]],
    cross_state_scale: float,
    path: str | Path,
) -> None:
    time = np.asarray([row["time_ps"] for row in rows])
    temporal = np.asarray([row["embedding_distance_mean"] for row in rows])
    sibling = np.asarray([row["sibling_distance_mean"] for row in rows])
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.plot(time, temporal / cross_state_scale, marker="o", label="same branch vs t=0")
    axis.plot(time, sibling / cross_state_scale, marker="o", label="sibling spread")
    axis.set_xlabel("time from shot start (ps)")
    axis.set_ylabel("mean distance / cross-state median")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_example_trajectories(
    embeddings: np.ndarray,
    times_ps: np.ndarray,
    branch_parent: np.ndarray,
    parent_temperature_K: np.ndarray,
    parent_role: Sequence[str],
    atom_ids: np.ndarray,
    path: str | Path,
) -> list[dict[str, Any]]:
    values = np.asarray(embeddings, dtype=np.float32)
    flattened = values.reshape(-1, values.shape[-1])
    pca = CovariancePCA.fit(flattened, dimension=2)
    coordinates = pca.transform(flattened).reshape(*values.shape[:-1], 2)
    roles = np.asarray([str(value) for value in parent_role])
    examples: list[dict[str, Any]] = []
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    for axis, temperature in zip(axes, [400.0, 450.0, 500.0], strict=True):
        candidates = np.flatnonzero(
            (np.asarray(parent_temperature_K) == temperature)
            & (roles == "transition_candidate")
        )
        best: tuple[float, int, int] | None = None
        for parent in candidates.tolist():
            branches = np.flatnonzero(np.asarray(branch_parent) == parent)
            terminal = values[branches, -1]
            spread = np.mean(
                np.sum(
                    np.square(terminal - terminal.mean(axis=0, keepdims=True)), axis=-1
                ),
                axis=0,
            )
            center = int(np.argmax(spread))
            candidate = (float(spread[center]), int(parent), center)
            if best is None or candidate > best:
                best = candidate
        if best is None:
            raise RuntimeError(f"No transition parent exists at {temperature:g} K.")
        spread, parent, center = best
        branches = np.flatnonzero(np.asarray(branch_parent) == parent)
        for shot, branch in enumerate(branches.tolist()):
            trajectory = coordinates[branch, :, center]
            axis.plot(trajectory[:, 0], trajectory[:, 1], marker="o", markersize=2.5, label=f"shot {shot}")
            axis.scatter(trajectory[0, 0], trajectory[0, 1], marker="*", s=80, color="black", zorder=5)
        axis.set_title(f"{temperature:g} K, atom {int(atom_ids[center])}")
        axis.set_xlabel("embedding PC1")
        axis.set_ylabel("embedding PC2")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=7)
        examples.append(
            {
                "temperature_K": temperature,
                "parent_index": parent,
                "center_index": center,
                "atom_id": int(atom_ids[center]),
                "terminal_sibling_dispersion": spread,
            }
        )
    figure.suptitle(
        f"Finest-step GeoFrame trajectories, 0–{float(np.max(times_ps)):.2f} ps"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return examples


__all__ = [
    "plot_distance_distributions",
    "plot_example_trajectories",
    "plot_sibling_divergence",
    "plot_stability_summary",
    "row_cosine",
    "same_atom_retrieval",
    "sibling_distances",
    "stratified_stability_rows",
    "temporal_stability_table",
    "write_csv",
]
