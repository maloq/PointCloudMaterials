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
from src.temporal_vamp.embeddings import (
    EmbeddingCache,
    FrozenEncoder,
    encode_spatial_context_state,
)
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


@dataclass(frozen=True)
class FutureNeighborCandidateFilter:
    """Metadata constraints for confound-controlled held-out retrieval."""

    exclude_same_run: bool
    match_temperature: bool
    relative_time_tolerance_ps: float | None = None
    crystalline_fraction_tolerance: float | None = None

    def __post_init__(self) -> None:
        if not self.exclude_same_run:
            raise ValueError(
                "Confound-controlled future-neighbor retrieval requires "
                "exclude_same_run=true."
            )
        if (
            self.relative_time_tolerance_ps is not None
            and self.relative_time_tolerance_ps < 0.0
        ):
            raise ValueError(
                "relative_time_tolerance_ps must be non-negative, got "
                f"{self.relative_time_tolerance_ps}."
            )
        if (
            self.crystalline_fraction_tolerance is not None
            and self.crystalline_fraction_tolerance < 0.0
        ):
            raise ValueError(
                "crystalline_fraction_tolerance must be non-negative, got "
                f"{self.crystalline_fraction_tolerance}."
            )

    def to_dict(self) -> dict[str, bool | float | None]:
        return {
            "exclude_same_run": self.exclude_same_run,
            "match_temperature": self.match_temperature,
            "relative_time_tolerance_ps": self.relative_time_tolerance_ps,
            "crystalline_fraction_tolerance": self.crystalline_fraction_tolerance,
        }


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


def _relative_event_times(cache: EmbeddingCache) -> np.ndarray:
    if cache.time_ps0 is None:
        raise ValueError(
            "Relative-event-time matching requires cached physical times (time_ps0)."
        )
    run_metadata = cache.manifest.get("run_metadata")
    if not isinstance(run_metadata, list) or len(run_metadata) != len(cache.run_ids):
        raise ValueError(
            "Relative-event-time matching requires one run_metadata entry per run; "
            f"got metadata={type(run_metadata).__name__}, runs={len(cache.run_ids)}."
        )
    nucleation_time_values = [metadata["nucleation_time_ps"] for metadata in run_metadata]
    if any(value is None for value in nucleation_time_values):
        missing_runs = [
            cache.run_ids[index]
            for index, value in enumerate(nucleation_time_values)
            if value is None
        ]
        raise ValueError(
            "Relative-event-time matching requires a detected nucleation time for "
            f"every cached run; missing for {missing_runs}."
        )
    nucleation_times = np.asarray(nucleation_time_values, dtype=np.float64)
    return np.asarray(cache.time_ps0, dtype=np.float64) - nucleation_times[
        np.asarray(cache.run_index, dtype=np.int64)
    ]


def _filtered_neighbor_indices(
    present: np.ndarray,
    cache: EmbeddingCache,
    *,
    query_indices: np.ndarray,
    neighbors: int,
    candidate_filter: FutureNeighborCandidateFilter,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return exact neighbors and matched random references for eligible queries."""
    run_index = np.asarray(cache.run_index, dtype=np.int64)
    if candidate_filter.match_temperature and cache.temperature_K is None:
        raise ValueError(
            "Temperature-matched retrieval requires cached temperature_K metadata."
        )
    temperature = (
        None
        if not candidate_filter.match_temperature
        else np.asarray(cache.temperature_K)
    )
    relative_time = (
        None
        if candidate_filter.relative_time_tolerance_ps is None
        else _relative_event_times(cache)
    )
    if (
        candidate_filter.crystalline_fraction_tolerance is not None
        and cache.crystalline_fraction0 is None
    ):
        raise ValueError(
            "Crystallinity-matched retrieval requires cached crystalline_fraction0 metadata."
        )
    crystallinity = (
        None
        if candidate_filter.crystalline_fraction_tolerance is None
        else np.asarray(cache.crystalline_fraction0)
    )

    grouped_queries: dict[tuple[int, float | None, float | None, float | None], list[int]] = {}
    for query_position, query_index in enumerate(query_indices.tolist()):
        key = (
            int(run_index[query_index]),
            None if temperature is None else float(temperature[query_index]),
            None if relative_time is None else float(relative_time[query_index]),
            None if crystallinity is None else float(crystallinity[query_index]),
        )
        grouped_queries.setdefault(key, []).append(query_position)

    k = int(neighbors)
    rng = np.random.default_rng(int(seed))
    candidate_counts = np.zeros(query_indices.size, dtype=np.int64)
    eligible_positions: list[np.ndarray] = []
    neighbor_blocks: list[np.ndarray] = []
    random_blocks: list[np.ndarray] = []
    for key, positions_list in grouped_queries.items():
        query_run, query_temperature, query_relative_time, query_crystallinity = key
        eligible = run_index != query_run
        if temperature is not None:
            assert query_temperature is not None
            eligible &= temperature == query_temperature
        if relative_time is not None:
            assert query_relative_time is not None
            assert candidate_filter.relative_time_tolerance_ps is not None
            eligible &= (
                np.abs(relative_time - query_relative_time)
                <= candidate_filter.relative_time_tolerance_ps
            )
        if crystallinity is not None:
            assert query_crystallinity is not None
            assert candidate_filter.crystalline_fraction_tolerance is not None
            eligible &= (
                np.abs(crystallinity - query_crystallinity)
                <= candidate_filter.crystalline_fraction_tolerance
            )
        candidates = np.flatnonzero(eligible)
        positions = np.asarray(positions_list, dtype=np.int64)
        candidate_counts[positions] = candidates.size
        if candidates.size < k:
            continue
        search = NearestNeighbors(
            n_neighbors=k,
            metric="euclidean",
            algorithm="brute",
        )
        search.fit(present[candidates])
        local_neighbors = search.kneighbors(
            present[query_indices[positions]], return_distance=False
        )
        eligible_positions.append(positions)
        neighbor_blocks.append(candidates[local_neighbors])
        random_blocks.append(
            np.stack(
                [rng.choice(candidates, size=k, replace=False) for _ in positions_list],
                axis=0,
            )
        )

    if not eligible_positions:
        raise RuntimeError(
            "No requested query has enough candidates for confound-controlled retrieval. "
            f"required_neighbors={k}, filter={candidate_filter.to_dict()}."
        )
    output_positions = np.concatenate(eligible_positions)
    order = np.argsort(output_positions)
    eligible_queries = query_indices[output_positions][order]
    selected = np.concatenate(neighbor_blocks, axis=0)[order]
    random_indices = np.concatenate(random_blocks, axis=0)[order]
    return eligible_queries, selected, random_indices, candidate_counts


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
    candidate_filter: FutureNeighborCandidateFilter | None = None,
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
    requested_query_indices = _select_query_indices(
        present.shape[0], int(max_queries), int(seed)
    )
    if requested_query_indices.size < 2:
        raise ValueError(
            "Future-neighbor evaluation requires at least two queries, got "
            f"{requested_query_indices.size}."
        )
    candidate_counts: np.ndarray | None = None
    if candidate_filter is not None:
        query_indices, selected, random_indices, candidate_counts = (
            _filtered_neighbor_indices(
                present,
                cache,
                query_indices=requested_query_indices,
                neighbors=k,
                candidate_filter=candidate_filter,
                seed=int(seed) + 7919,
            )
        )
        if exclude_same_atom:
            selected_group_codes = group_codes[selected]
            query_group_codes = group_codes[query_indices, None]
            if np.any(selected_group_codes == query_group_codes):
                raise RuntimeError(
                    "Cross-run candidate filtering unexpectedly retained a same-(run, atom) "
                    "neighbor."
                )
    else:
        query_indices = requested_query_indices
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
                    f"candidate_count={candidate_count}, "
                    f"exclude_same_atom={exclude_same_atom}."
                )
            selected[row] = valid[:k]
        random_indices = _random_reference_indices(
            query_indices=query_indices,
            group_codes=group_codes,
            exclude_same_atom=exclude_same_atom,
            seed=int(seed) + 7919,
        )

    future_delta = future[selected] - future[query_indices, None, :]
    future_distances = np.linalg.norm(future_delta, axis=2)
    per_query = future_distances.mean(axis=1)
    if random_indices.ndim == 1:
        random_distances = np.linalg.norm(
            future[random_indices] - future[query_indices], axis=1
        )
    else:
        random_distances = np.linalg.norm(
            future[random_indices] - future[query_indices, None, :], axis=2
        ).mean(axis=1)
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
    if candidate_counts is not None:
        eligible_counts = candidate_counts[candidate_counts >= k]
        result.update(
            {
                "requested_queries": int(requested_query_indices.size),
                "query_coverage": float(query_indices.size / requested_query_indices.size),
                "candidate_count_min": int(candidate_counts.min()),
                "candidate_count_median": float(np.median(candidate_counts)),
                "candidate_count_max": int(candidate_counts.max()),
                "eligible_candidate_count_min": int(eligible_counts.min()),
                "eligible_candidate_count_median": float(np.median(eligible_counts)),
                "eligible_candidate_count_max": int(eligible_counts.max()),
            }
        )
    if future_labels is not None:
        labels = np.asarray(future_labels)
        agreement = labels[selected] == labels[query_indices, None]
        if random_indices.ndim == 1:
            random_agreement = labels[random_indices] == labels[query_indices]
        else:
            random_agreement = labels[random_indices] == labels[query_indices, None]
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
    spatial_context_aggregation: str = "mean_std",
    point_cloud_batch_size: int = 2048,
) -> dict[str, float | int]:
    count = min(int(samples), len(dataset))
    if count <= 0:
        raise ValueError(f"Sanity-check samples must be > 0, got {samples}.")
    batch = dataset.__getitems__(range(count))
    points = batch["points0"]
    context = batch.get("context_points0")
    encoded_a_tensor, _ = encode_spatial_context_state(
        encoder,
        points,
        context_points=context,
        aggregation=spatial_context_aggregation,
        point_cloud_batch_size=point_cloud_batch_size,
    )
    encoded_b_tensor, _ = encode_spatial_context_state(
        encoder,
        points,
        context_points=context,
        aggregation=spatial_context_aggregation,
        point_cloud_batch_size=point_cloud_batch_size,
    )
    encoded_a = encoded_a_tensor.numpy()
    encoded_b = encoded_b_tensor.numpy()
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
    rotated_context = (
        None if context is None else context @ torch.from_numpy(rotation)
    )
    rotated_embedding_tensor, _ = encode_spatial_context_state(
        encoder,
        rotated_points,
        context_points=rotated_context,
        aggregation=spatial_context_aggregation,
        point_cloud_batch_size=point_cloud_batch_size,
    )
    rotated_embedding = rotated_embedding_tensor.numpy()
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
    if cache.local_z0 is not None:
        assert cache.local_z1 is not None
        payload["local_embedding"] = np.asarray(cache.local_z0, dtype=np.float32)
        payload["local_future_embedding"] = np.asarray(
            cache.local_z1, dtype=np.float32
        )
    for name in (
        "time_ps0",
        "time_ps1",
        "temperature_K",
        "pressure_GPa",
        "velocity_seed",
        "crystalline_fraction0",
        "crystalline_fraction1",
        "largest_crystalline_cluster_atoms0",
        "largest_crystalline_cluster_atoms1",
    ):
        values = getattr(cache, name)
        if values is not None:
            payload[name] = np.asarray(values)
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
        values = (
            np.asarray(cache.time_ps0)[indices]
            if cache.time_ps0 is not None
            else np.asarray(cache.timestep0)[indices]
        )
        color_label = (
            "simulation time (ps)"
            if cache.time_ps0 is not None
            else "simulation timestep"
        )
        cmap = "viridis"
    elif color == "temperature":
        if cache.temperature_K is None:
            return
        values = np.asarray(cache.temperature_K)[indices]
        color_label = "temperature (K)"
        cmap = "plasma"
    elif color == "crystallinity":
        if cache.crystalline_fraction0 is None:
            return
        values = np.asarray(cache.crystalline_fraction0)[indices]
        color_label = "global crystalline fraction"
        cmap = "cividis"
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
    "FutureNeighborCandidateFilter",
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
