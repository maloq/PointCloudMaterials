from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.temporal_vamp.data import TemporalPairDataset
from src.temporal_vamp.embeddings import EmbeddingCache, FrozenEncoder
from src.temporal_vamp.linear_vamp import LinearVAMP


@dataclass
class CovariancePCA:
    mean_: np.ndarray
    components_: np.ndarray
    eigenvalues_: np.ndarray

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        *,
        dimension: int,
        batch_size: int = 65536,
    ) -> "CovariancePCA":
        n_samples, feature_dim = values.shape
        mean = np.zeros(feature_dim, dtype=np.float64)
        for start in range(0, n_samples, batch_size):
            mean += np.asarray(values[start : start + batch_size], dtype=np.float64).sum(axis=0)
        mean /= float(n_samples)
        covariance = np.zeros((feature_dim, feature_dim), dtype=np.float64)
        for start in range(0, n_samples, batch_size):
            centered = np.asarray(values[start : start + batch_size], dtype=np.float64) - mean
            covariance += centered.T @ centered
        covariance /= float(n_samples)
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
        order = np.argsort(eigenvalues)[::-1]
        dim = min(int(dimension), feature_dim)
        return cls(
            mean_=mean,
            components_=eigenvectors[:, order[:dim]],
            eigenvalues_=np.maximum(eigenvalues[order], 0.0),
        )

    def transform(self, values: np.ndarray, dimension: int | None = None) -> np.ndarray:
        dim = self.components_.shape[1] if dimension is None else int(dimension)
        if dim <= 0 or dim > self.components_.shape[1]:
            raise ValueError(
                f"PCA dimension must be in [1, {self.components_.shape[1]}], got {dim}."
            )
        return (np.asarray(values, dtype=np.float64) - self.mean_) @ self.components_[:, :dim]

    def save(self, path: str | Path) -> None:
        np.savez(
            Path(path),
            mean=self.mean_,
            components=self.components_,
            eigenvalues=self.eigenvalues_,
        )

    @classmethod
    def load(cls, path: str | Path) -> "CovariancePCA":
        with np.load(Path(path), allow_pickle=False) as payload:
            return cls(
                mean_=payload["mean"].copy(),
                components_=payload["components"].copy(),
                eigenvalues_=payload["eigenvalues"].copy(),
            )


def _group_codes(cache: EmbeddingCache) -> np.ndarray:
    pairs = np.stack(
        [np.asarray(cache.run_index, dtype=np.int64), np.asarray(cache.atom_id, dtype=np.int64)],
        axis=1,
    )
    _, codes = np.unique(pairs, axis=0, return_inverse=True)
    return codes.astype(np.int64, copy=False)


def _select_query_indices(sample_count: int, max_queries: int, seed: int) -> np.ndarray:
    if max_queries <= 0 or max_queries >= sample_count:
        return np.arange(sample_count, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(sample_count, size=max_queries, replace=False))


def _random_reference_indices(
    *,
    query_indices: np.ndarray,
    group_codes: np.ndarray,
    exclude_same_atom: bool,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    result = np.empty(query_indices.size, dtype=np.int64)
    sample_count = int(group_codes.size)
    for output_index, query_index in enumerate(query_indices.tolist()):
        while True:
            candidate = int(rng.integers(0, sample_count))
            same_sample = candidate == int(query_index)
            same_atom = int(group_codes[candidate]) == int(group_codes[query_index])
            if not same_sample and not (exclude_same_atom and same_atom):
                result[output_index] = candidate
                break
    return result


def future_neighbor_consistency(
    present_space: np.ndarray,
    future_embeddings: np.ndarray,
    cache: EmbeddingCache,
    *,
    neighbors: int,
    max_queries: int,
    exclude_same_atom: bool,
    seed: int,
    future_labels: np.ndarray | None = None,
) -> dict[str, float | int]:
    present = np.asarray(present_space)
    future = np.asarray(future_embeddings)
    if present.shape[0] != future.shape[0] or present.shape[0] != cache.z0.shape[0]:
        raise ValueError(
            "Present space, future embeddings, and metadata must have the same row count. "
            f"Got {present.shape[0]}, {future.shape[0]}, {cache.z0.shape[0]}."
        )
    k = int(neighbors)
    if k <= 0 or present.shape[0] <= k:
        raise ValueError(
            f"neighbors must satisfy 1 <= k < sample_count, got k={k}, n={present.shape[0]}."
        )
    group_codes = _group_codes(cache)
    query_indices = _select_query_indices(present.shape[0], int(max_queries), int(seed))
    if query_indices.size < 2:
        raise ValueError(
            f"Future-neighbor evaluation requires at least two queries, got {query_indices.size}."
        )
    if exclude_same_atom and np.unique(group_codes).size < 2:
        raise ValueError(
            "exclude_same_atom=true requires at least two distinct (run, atom) groups."
        )
    max_group_count = int(np.bincount(group_codes).max()) if exclude_same_atom else 1
    candidate_count = min(present.shape[0], k + max_group_count + 1)
    search = NearestNeighbors(n_neighbors=candidate_count, metric="euclidean")
    search.fit(present)
    candidate_indices = search.kneighbors(
        present[query_indices], return_distance=False
    )
    selected = np.empty((query_indices.size, k), dtype=np.int64)
    for row, query_index in enumerate(query_indices.tolist()):
        candidates = candidate_indices[row]
        keep = candidates != query_index
        if exclude_same_atom:
            keep &= group_codes[candidates] != group_codes[query_index]
        valid = candidates[keep]
        if valid.size < k:
            raise RuntimeError(
                "Nearest-neighbor filtering left too few candidates. "
                f"query_index={query_index}, required={k}, available={valid.size}, "
                f"candidate_count={candidate_count}, exclude_same_atom={exclude_same_atom}."
            )
        selected[row] = valid[:k]

    future_delta = future[selected] - future[query_indices, None, :]
    future_distances = np.linalg.norm(future_delta, axis=2)
    per_query = future_distances.mean(axis=1)
    random_indices = _random_reference_indices(
        query_indices=query_indices,
        group_codes=group_codes,
        exclude_same_atom=exclude_same_atom,
        seed=int(seed) + 7919,
    )
    random_distances = np.linalg.norm(
        future[random_indices] - future[query_indices], axis=1
    )
    result: dict[str, float | int] = {
        "queries": int(query_indices.size),
        "neighbors": k,
        "mean_future_embedding_distance": float(per_query.mean()),
        "sem_future_embedding_distance": float(
            per_query.std(ddof=1) / np.sqrt(per_query.size)
        ),
        "random_mean_future_embedding_distance": float(random_distances.mean()),
        "distance_over_random": float(per_query.mean() / random_distances.mean()),
    }
    if future_labels is not None:
        labels = np.asarray(future_labels)
        agreement = labels[selected] == labels[query_indices, None]
        random_agreement = labels[random_indices] == labels[query_indices]
        result["future_state_neighbor_agreement"] = float(agreement.mean())
        result["random_future_state_agreement"] = float(random_agreement.mean())
    return result


def fit_future_state_labels(
    train_future: np.ndarray,
    validation_future: np.ndarray,
    *,
    clusters: int,
    max_fit_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, KMeans]:
    train = np.asarray(train_future)
    if max_fit_samples > 0 and train.shape[0] > max_fit_samples:
        rng = np.random.default_rng(int(seed))
        fit_indices = np.sort(rng.choice(train.shape[0], size=max_fit_samples, replace=False))
        fit_values = train[fit_indices]
    else:
        fit_values = train
    model = KMeans(n_clusters=int(clusters), n_init=10, random_state=int(seed))
    model.fit(fit_values)
    return model.predict(train), model.predict(validation_future), model


def future_prediction_probes(
    train_spaces: Mapping[str, np.ndarray],
    validation_spaces: Mapping[str, np.ndarray],
    train_future_labels: np.ndarray,
    validation_future_labels: np.ndarray,
    *,
    max_train_samples: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    rng = np.random.default_rng(int(seed))
    train_count = len(train_future_labels)
    if max_train_samples > 0 and train_count > max_train_samples:
        indices = np.sort(rng.choice(train_count, size=max_train_samples, replace=False))
    else:
        indices = np.arange(train_count, dtype=np.int64)
    output: dict[str, dict[str, float | int]] = {}
    for name, train_values in train_spaces.items():
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=int(seed)),
        )
        probe.fit(np.asarray(train_values)[indices], train_future_labels[indices])
        prediction = probe.predict(np.asarray(validation_spaces[name]))
        output[name] = {
            "accuracy": float(accuracy_score(validation_future_labels, prediction)),
            "train_samples": int(indices.size),
        }
    return output


def encoder_sanity_checks(
    encoder: FrozenEncoder,
    vamp: LinearVAMP,
    dataset: TemporalPairDataset,
    *,
    samples: int,
    dimension: int,
) -> dict[str, float | int]:
    count = min(int(samples), len(dataset))
    if count <= 0:
        raise ValueError(f"Sanity-check samples must be > 0, got {samples}.")
    batch = dataset.__getitems__(range(count))
    points = batch["points0"]
    encoded_a = encoder.encode(points).cpu().numpy()
    encoded_b = encoder.encode(points).cpu().numpy()
    deterministic_delta = encoded_b - encoded_a

    rotation = np.asarray(
        [
            [-0.3333333333, -0.6666666667, 0.6666666667],
            [0.9333333333, -0.3333333333, 0.1333333333],
            [0.1333333333, 0.6666666667, 0.7333333333],
        ],
        dtype=np.float32,
    )
    rotated_points = points @ torch.from_numpy(rotation)
    rotated_embedding = encoder.encode(rotated_points).cpu().numpy()
    rotated_delta = rotated_embedding - encoded_a
    kinetic = vamp.transform(encoded_a, int(dimension))
    rotated_kinetic = vamp.transform(rotated_embedding, int(dimension))
    kinetic_delta = rotated_kinetic - kinetic
    return {
        "samples": count,
        "repeat_embedding_max_abs": float(np.abs(deterministic_delta).max()),
        "repeat_embedding_rmse": float(np.sqrt(np.mean(deterministic_delta**2))),
        "rotation_embedding_relative_rmse": float(
            np.sqrt(np.mean(rotated_delta**2))
            / max(np.sqrt(np.mean(encoded_a**2)), 1.0e-12)
        ),
        "rotation_kinetic_relative_rmse": float(
            np.sqrt(np.mean(kinetic_delta**2))
            / max(np.sqrt(np.mean(kinetic**2)), 1.0e-12)
        ),
    }


def regularization_sensitivity(
    train: EmbeddingCache,
    validation: EmbeddingCache,
    *,
    regularizations: Sequence[float],
    dimension: int,
    eigenvalue_cutoff: float,
    covariance_batch_size: int,
) -> dict[str, dict[str, Any]]:
    if not regularizations:
        raise ValueError("regularization_sensitivity requires at least one ridge value.")
    coordinates: dict[str, np.ndarray] = {}
    results: dict[str, dict[str, Any]] = {}
    for value in regularizations:
        model = LinearVAMP(
            regularization=float(value),
            eigenvalue_cutoff=float(eigenvalue_cutoff),
            covariance_batch_size=int(covariance_batch_size),
        ).fit(train.z0, train.z1)
        key = f"{float(value):.3e}"
        coordinates[key] = model.transform(validation.z0, int(dimension))
        results[key] = {
            "leading_singular_values": model.singular_values_[: int(dimension)].tolist(),
            "rank": model.rank,
        }
    reference_key = f"{float(regularizations[len(regularizations) // 2]):.3e}"
    reference, _ = np.linalg.qr(coordinates[reference_key] - coordinates[reference_key].mean(axis=0))
    for key, values in coordinates.items():
        basis, _ = np.linalg.qr(values - values.mean(axis=0))
        canonical = np.linalg.svd(reference.T @ basis, compute_uv=False)
        results[key]["minimum_subspace_canonical_correlation_to_middle"] = float(canonical.min())
    return results


def save_coordinate_archive(
    path: str | Path,
    cache: EmbeddingCache,
    *,
    kinetic: np.ndarray,
    pca: np.ndarray,
    future_state: np.ndarray | None,
) -> None:
    payload: dict[str, np.ndarray] = {
        "kinetic": np.asarray(kinetic, dtype=np.float32),
        "pca": np.asarray(pca, dtype=np.float32),
        "embedding": np.asarray(cache.z0, dtype=np.float32),
        "future_embedding": np.asarray(cache.z1, dtype=np.float32),
        "atom_id": np.asarray(cache.atom_id),
        "run_index": np.asarray(cache.run_index),
        "frame0": np.asarray(cache.frame0),
        "frame1": np.asarray(cache.frame1),
        "timestep0": np.asarray(cache.timestep0),
        "timestep1": np.asarray(cache.timestep1),
        "coords0": np.asarray(cache.coords0),
        "coords1": np.asarray(cache.coords1),
        "run_ids": np.asarray(cache.run_ids),
    }
    if future_state is not None:
        payload["future_state"] = np.asarray(future_state, dtype=np.int32)
    np.savez(Path(path), **payload)


def plot_singular_spectrum(
    spectra: Mapping[str, np.ndarray],
    path: str | Path,
    *,
    max_modes: int,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label, spectrum in spectra.items():
        shown = np.asarray(spectrum)[: int(max_modes)]
        ax.plot(np.arange(1, shown.size + 1), shown, marker="o", label=str(label))
    ax.set(xlabel="mode", ylabel="VAMP singular value", ylim=(0.0, 1.05))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(Path(path), dpi=180)
    plt.close(fig)


def plot_kinetic_coordinates(
    kinetic: np.ndarray,
    cache: EmbeddingCache,
    path: str | Path,
    *,
    color: str,
    future_state: np.ndarray | None,
    max_points: int,
    seed: int,
) -> None:
    coords = np.asarray(kinetic)
    if coords.shape[1] < 2:
        return
    indices = _select_query_indices(coords.shape[0], int(max_points), int(seed))
    if color == "time":
        values = np.asarray(cache.timestep0)[indices]
        color_label = "simulation timestep"
        cmap = "viridis"
    elif color == "run":
        values = np.asarray(cache.run_index)[indices]
        color_label = "run index"
        cmap = "tab10"
    elif color == "future_state":
        if future_state is None:
            return
        values = np.asarray(future_state)[indices]
        color_label = "future state"
        cmap = "tab10"
    else:
        raise ValueError(f"Unknown kinetic-coordinate color mode {color!r}.")
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    scatter = ax.scatter(
        coords[indices, 0],
        coords[indices, 1],
        c=values,
        cmap=cmap,
        s=5,
        alpha=0.55,
        linewidths=0,
    )
    ax.set(xlabel=r"$\xi_1$", ylabel=r"$\xi_2$")
    fig.colorbar(scatter, ax=ax, label=color_label)
    fig.tight_layout()
    fig.savefig(Path(path), dpi=180)
    plt.close(fig)


def plot_temporal_trajectories(
    kinetic: np.ndarray,
    cache: EmbeddingCache,
    path: str | Path,
    *,
    atom_ids: Sequence[int] | None,
    count: int,
) -> list[dict[str, int]]:
    if kinetic.shape[1] < 2:
        return []
    if atom_ids is None:
        pairs = np.stack([cache.run_index, cache.atom_id], axis=1)
        unique, counts = np.unique(pairs, axis=0, return_counts=True)
        chosen = unique[np.argsort(counts)[::-1][: int(count)]]
    else:
        chosen = np.asarray([[0, int(atom_id)] for atom_id in atom_ids], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    plotted: list[dict[str, int]] = []
    for run_index, atom_id in chosen.tolist():
        mask = (cache.run_index == run_index) & (cache.atom_id == atom_id)
        indices = np.flatnonzero(mask)
        if indices.size < 2:
            continue
        indices = indices[np.argsort(cache.frame0[indices])]
        ax.plot(
            kinetic[indices, 0],
            kinetic[indices, 1],
            marker="o",
            markersize=2.5,
            linewidth=1.0,
            alpha=0.8,
            label=f"{cache.run_ids[run_index]}: atom {atom_id}",
        )
        plotted.append({"run_index": int(run_index), "atom_id": int(atom_id)})
    ax.set(xlabel=r"$\xi_1$", ylabel=r"$\xi_2$")
    if plotted:
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(path), dpi=180)
    plt.close(fig)
    return plotted


def plot_future_neighbor_comparison(
    metrics: Mapping[str, Mapping[str, float | int]],
    path: str | Path,
) -> None:
    names = list(metrics)
    ratios = [float(metrics[name]["distance_over_random"]) for name in names]
    fig_width = max(6.4, 0.7 * len(names))
    fig, ax = plt.subplots(figsize=(fig_width, 4.4))
    ax.bar(np.arange(len(names)), ratios, color="#4477aa")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="random retrieval")
    ax.set(
        ylabel="future distance / random distance",
        xticks=np.arange(len(names)),
        xticklabels=names,
    )
    ax.tick_params(axis="x", rotation=40)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(Path(path), dpi=180)
    plt.close(fig)


def write_json(path: str | Path, value: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


__all__ = [
    "CovariancePCA",
    "encoder_sanity_checks",
    "fit_future_state_labels",
    "future_neighbor_consistency",
    "future_prediction_probes",
    "plot_future_neighbor_comparison",
    "plot_kinetic_coordinates",
    "plot_singular_spectrum",
    "plot_temporal_trajectories",
    "regularization_sensitivity",
    "save_coordinate_archive",
    "write_json",
]
