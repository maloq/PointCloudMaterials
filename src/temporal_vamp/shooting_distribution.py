from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.temporal_vamp.evaluation import CovariancePCA
from src.temporal_vamp.shooting_ablation import _rows_for_parents
from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache
from src.temporal_vamp.shooting_multiscale import DynamicTargetData
from src.temporal_vamp.shooting_predictor import (
    _future_neighbor_metrics,
    _parent_split_indices,
)
from src.temporal_vamp.shooting_spatial import FittedSpatialContextPredictor


@dataclass(frozen=True)
class HorizonRFFParameters:
    delta_mean: np.ndarray
    delta_scale: np.ndarray
    pca: CovariancePCA
    median_distance: float
    bandwidths: np.ndarray
    frequencies: np.ndarray
    phases: np.ndarray


@dataclass(frozen=True)
class DistributionalTargetData:
    selected_horizon_indices: np.ndarray
    selected_horizons_ps: np.ndarray
    target_modes: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    distribution_signature: np.ndarray
    split_rows: dict[str, np.ndarray]
    parent_splits: dict[str, np.ndarray]
    horizon_parameters: tuple[HorizonRFFParameters, ...]
    diagnostics: dict[str, Any]


def _median_pair_distance(
    values: np.ndarray, *, maximum_samples: int, rng: np.random.Generator
) -> float:
    if values.shape[0] > int(maximum_samples):
        rows = rng.choice(values.shape[0], size=int(maximum_samples), replace=False)
        selected = values[rows]
    else:
        selected = values
    pair_count = min(100_000, selected.shape[0] * 32)
    left = rng.integers(0, selected.shape[0], size=pair_count)
    right = rng.integers(0, selected.shape[0], size=pair_count)
    distances = np.linalg.norm(selected[left] - selected[right], axis=1)
    positive = distances[distances > 1.0e-10]
    if positive.size == 0:
        raise RuntimeError(
            "Cannot choose an RFF bandwidth because every sampled future-change "
            "distance is zero."
        )
    return float(np.median(positive))


def _random_fourier_features(
    values: np.ndarray,
    frequencies: np.ndarray,
    phases: np.ndarray,
) -> np.ndarray:
    # frequencies: [bands, input_dim, features_per_band]
    blocks = [
        np.sqrt(2.0 / float(frequencies.shape[2]))
        * np.cos(values @ frequencies[band] + phases[band])
        for band in range(frequencies.shape[0])
    ]
    return np.concatenate(blocks, axis=-1)


def prepare_distributional_target_data(
    cache: ShootingEmbeddingCache,
    *,
    horizons_ps: Sequence[float],
    change_pca_dim: int,
    rff_features_per_bandwidth: int,
    bandwidth_multipliers: Sequence[float],
    selection_source_velocity_seeds: Sequence[int],
    seed: int,
) -> DistributionalTargetData:
    """Represent every parent's empirical future distribution by RFF kernel means."""

    parent_splits = _parent_split_indices(
        cache,
        selection_source_velocity_seeds=selection_source_velocity_seeds,
    )
    parent_count, center_count, embedding_dim = cache.parent_local_z.shape
    split_rows = {
        name: _rows_for_parents(indices, int(center_count))
        for name, indices in parent_splits.items()
    }
    available_horizons = np.asarray(cache.horizons_ps, dtype=np.float64)
    requested_horizons = np.asarray(
        [float(value) for value in horizons_ps], dtype=np.float64
    )
    selected_indices: list[int] = []
    for horizon in requested_horizons.tolist():
        matches = np.flatnonzero(
            np.isclose(available_horizons, horizon, rtol=0.0, atol=1.0e-9)
        )
        if matches.size != 1:
            raise ValueError(
                f"Requested horizon {horizon:g} ps is not uniquely available in "
                f"{available_horizons.tolist()}."
            )
        selected_indices.append(int(matches[0]))
    bandwidth_factors = np.asarray(
        [float(value) for value in bandwidth_multipliers], dtype=np.float64
    )
    if bandwidth_factors.size == 0 or np.any(bandwidth_factors <= 0.0):
        raise ValueError(
            "bandwidth_multipliers must be positive and nonempty; "
            f"got {bandwidth_factors.tolist()}."
        )
    features_per_band = int(rff_features_per_bandwidth)
    if features_per_band <= 0:
        raise ValueError(
            f"rff_features_per_bandwidth must be positive, got {features_per_band}."
        )

    branch_parent = np.asarray(cache.branch_parent_index, dtype=np.int64)
    optimization_branch_rows = np.flatnonzero(
        np.isin(branch_parent, parent_splits["optimization"])
    )
    rng = np.random.default_rng(int(seed))
    horizon_signatures: list[np.ndarray] = []
    horizon_parameters: list[HorizonRFFParameters] = []
    split_shot_diagnostics: dict[str, Any] = {}
    branch_counts: dict[int, int] = {}
    for parent_index in range(int(parent_count)):
        branch_counts[parent_index] = int(
            np.count_nonzero(branch_parent == parent_index)
        )
    for local_horizon_index, cache_horizon_index in enumerate(selected_indices):
        current_for_branches = np.asarray(
            cache.parent_local_z[branch_parent], dtype=np.float64
        )
        future = np.asarray(
            cache.future_z[:, cache_horizon_index], dtype=np.float64
        )
        delta = future - current_for_branches
        optimization_delta = delta[optimization_branch_rows].reshape(
            -1, int(embedding_dim)
        )
        delta_mean = optimization_delta.mean(axis=0)
        delta_scale = optimization_delta.std(axis=0)
        delta_scale = np.where(delta_scale <= 1.0e-10, 1.0, delta_scale)
        standardized_delta = (delta - delta_mean) / delta_scale
        pca = CovariancePCA.fit(
            standardized_delta[optimization_branch_rows].reshape(
                -1, int(embedding_dim)
            ),
            dimension=int(change_pca_dim),
        )
        projected = pca.transform(
            standardized_delta.reshape(-1, int(embedding_dim)),
            dimension=int(change_pca_dim),
        ).reshape(delta.shape[0], int(center_count), int(change_pca_dim))
        median_distance = _median_pair_distance(
            projected[optimization_branch_rows].reshape(-1, int(change_pca_dim)),
            maximum_samples=4096,
            rng=rng,
        )
        bandwidths = median_distance * bandwidth_factors
        frequencies = np.stack(
            [
                rng.normal(
                    scale=1.0 / bandwidth,
                    size=(int(change_pca_dim), features_per_band),
                )
                for bandwidth in bandwidths.tolist()
            ],
            axis=0,
        )
        phases = rng.uniform(
            0.0,
            2.0 * np.pi,
            size=(bandwidths.size, features_per_band),
        )
        branch_features = _random_fourier_features(projected, frequencies, phases)
        signature_dim = int(branch_features.shape[-1])
        parent_signature = np.empty(
            (int(parent_count), int(center_count), signature_dim), dtype=np.float64
        )
        split_a = np.empty_like(parent_signature)
        split_b = np.empty_like(parent_signature)
        for parent_index in range(int(parent_count)):
            branches = np.flatnonzero(branch_parent == parent_index)
            if branches.size < 2:
                raise RuntimeError(
                    f"Distributional shooting target requires at least two futures per "
                    f"parent; parent_index={parent_index}, branches={branches.size}."
                )
            ordered = np.sort(branches)
            a_rows = ordered[::2]
            b_rows = ordered[1::2]
            parent_signature[parent_index] = branch_features[ordered].mean(axis=0)
            split_a[parent_index] = branch_features[a_rows].mean(axis=0)
            split_b[parent_index] = branch_features[b_rows].mean(axis=0)
        horizon_signatures.append(
            parent_signature.reshape(int(parent_count) * int(center_count), signature_dim)
        )
        validation_rows = split_rows["validation"]
        split_distance = np.linalg.norm(
            split_a.reshape(-1, signature_dim)[validation_rows]
            - split_b.reshape(-1, signature_dim)[validation_rows],
            axis=1,
        )
        centered_a = split_a.reshape(-1, signature_dim)[validation_rows]
        centered_b = split_b.reshape(-1, signature_dim)[validation_rows]
        component_correlation = np.corrcoef(
            centered_a.reshape(-1), centered_b.reshape(-1)
        )[0, 1]
        split_shot_diagnostics[f"{float(requested_horizons[local_horizon_index]):g}ps"] = {
            "mean_split_signature_distance": float(split_distance.mean()),
            "flattened_split_signature_correlation": float(component_correlation),
            "minimum_branches_per_parent": int(min(branch_counts.values())),
            "maximum_branches_per_parent": int(max(branch_counts.values())),
        }
        horizon_parameters.append(
            HorizonRFFParameters(
                delta_mean=delta_mean,
                delta_scale=delta_scale,
                pca=pca,
                median_distance=median_distance,
                bandwidths=bandwidths,
                frequencies=frequencies,
                phases=phases,
            )
        )

    distribution_signature = np.stack(horizon_signatures, axis=1)
    flat_signature = distribution_signature.reshape(
        distribution_signature.shape[0], -1
    )
    target_mean = flat_signature[split_rows["optimization"]].mean(axis=0)
    target_scale = flat_signature[split_rows["optimization"]].std(axis=0)
    target_scale = np.where(target_scale <= 1.0e-10, 1.0, target_scale)
    standardized_target = ((flat_signature - target_mean) / target_scale).astype(
        np.float32
    )
    diagnostics = {
        "target": "multi-bandwidth RFF kernel mean of all sibling future changes",
        "change_pca_fit": "branch-level changes from optimization source runs only",
        "rff_seed": int(seed),
        "rff_features_per_bandwidth": features_per_band,
        "bandwidth_multipliers": bandwidth_factors.tolist(),
        "signature_dim_per_horizon": int(distribution_signature.shape[-1]),
        "median_distances": [
            float(parameters.median_distance) for parameters in horizon_parameters
        ],
        "bandwidths": [
            parameters.bandwidths.tolist() for parameters in horizon_parameters
        ],
        "split_shot_reliability": split_shot_diagnostics,
    }
    return DistributionalTargetData(
        selected_horizon_indices=np.asarray(selected_indices, dtype=np.int64),
        selected_horizons_ps=requested_horizons,
        target_modes=standardized_target,
        target_mean=target_mean,
        target_scale=target_scale,
        distribution_signature=distribution_signature,
        split_rows=split_rows,
        parent_splits=parent_splits,
        horizon_parameters=tuple(horizon_parameters),
        diagnostics=diagnostics,
    )


def evaluate_distributional_predictor(
    cache: ShootingEmbeddingCache,
    static_feature_variants: Mapping[str, np.ndarray],
    targets: DistributionalTargetData,
    fitted: FittedSpatialContextPredictor,
    *,
    static_pca_dim: int,
    neighbors: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    static_spaces: dict[str, np.ndarray] = {}
    for name in ("local", "old_mean_std_8", "mean_std_context", "multiscale_context"):
        values = np.asarray(static_feature_variants[name], dtype=np.float64)
        pca = CovariancePCA.fit(
            values[targets.split_rows["optimization"]], dimension=int(static_pca_dim)
        )
        static_spaces[f"{name}_pca_{int(static_pca_dim)}d"] = pca.transform(
            values, dimension=int(static_pca_dim)
        )
    standardized_prediction = fitted.predictions_by_seed[fitted.seed]
    raw_prediction = (
        standardized_prediction * targets.target_scale + targets.target_mean
    )
    representation = fitted.representations_by_seed[fitted.seed]
    horizon_count = int(targets.selected_horizons_ps.size)
    signature_dim = int(targets.distribution_signature.shape[-1])
    prediction_blocks = raw_prediction.reshape(-1, horizon_count, signature_dim)
    retrieval: dict[str, Any] = {}
    baseline_name = f"local_pca_{int(static_pca_dim)}d"
    for horizon_index, horizon in enumerate(targets.selected_horizons_ps.tolist()):
        spaces = {
            **static_spaces,
            "distributional_transformer_representation": representation,
            "predicted_kernel_mean": prediction_blocks[:, horizon_index],
        }
        values = _future_neighbor_metrics(
            spaces,
            targets.distribution_signature[:, horizon_index],
            cache,
            targets.parent_splits["validation"],
            neighbors=int(neighbors),
            seed=int(seed),
        )
        baseline_distance = float(
            values[baseline_name]["mean_ensemble_future_distance"]
        )
        values["gain_over_local_pca_percent"] = {
            name: float(
                100.0
                * (
                    1.0
                    - float(result["mean_ensemble_future_distance"])
                    / baseline_distance
                )
            )
            for name, result in values.items()
        }
        retrieval[f"{float(horizon):g}ps"] = values
    combined_spaces = {
        **static_spaces,
        "distributional_transformer_representation": representation,
        "predicted_kernel_mean": raw_prediction,
    }
    combined = _future_neighbor_metrics(
        combined_spaces,
        targets.distribution_signature.reshape(
            targets.distribution_signature.shape[0], -1
        ),
        cache,
        targets.parent_splits["validation"],
        neighbors=int(neighbors),
        seed=int(seed),
    )
    baseline_distance = float(
        combined[baseline_name]["mean_ensemble_future_distance"]
    )
    combined["gain_over_local_pca_percent"] = {
        name: float(
            100.0
            * (
                1.0
                - float(result["mean_ensemble_future_distance"])
                / baseline_distance
            )
        )
        for name, result in combined.items()
    }
    retrieval["all_horizons"] = combined
    return (
        {
            "scientific_contract": {
                "ablation": 3,
                "target": "empirical distribution of sibling future changes via RFF kernel mean",
                "training_horizons_ps": targets.selected_horizons_ps.tolist(),
                "query_split": "validation source runs only",
                "candidate_filter": "different source run, exact temperature, exact parent phase",
                "model_change_from_ablation_2": "prediction target only",
                "excluded_changes": "no neighbour KL, pretraining, velocity, or encoder fine-tuning",
            },
            "selected_seed": int(fitted.seed),
            "seed_metrics": {str(key): value for key, value in fitted.seed_metrics.items()},
            "target_diagnostics": targets.diagnostics,
            "future_neighbor_consistency": retrieval,
        },
        {
            "prediction": raw_prediction,
            "standardized_prediction": standardized_prediction,
            "representation": representation,
            "distribution_signature": targets.distribution_signature,
        },
    )


def save_distributional_preprocessing(
    targets: DistributionalTargetData, path: str | Path
) -> None:
    target = Path(path)
    payload: dict[str, np.ndarray] = {
        "target_mean": targets.target_mean,
        "target_scale": targets.target_scale,
        "selected_horizons_ps": targets.selected_horizons_ps,
    }
    for horizon_index, parameters in enumerate(targets.horizon_parameters):
        prefix = f"horizon_{horizon_index}"
        payload[f"{prefix}__delta_mean"] = parameters.delta_mean
        payload[f"{prefix}__delta_scale"] = parameters.delta_scale
        payload[f"{prefix}__pca_mean"] = parameters.pca.mean_
        payload[f"{prefix}__pca_components"] = parameters.pca.components_
        payload[f"{prefix}__pca_eigenvalues"] = parameters.pca.eigenvalues_
        payload[f"{prefix}__median_distance"] = np.asarray(
            parameters.median_distance
        )
        payload[f"{prefix}__bandwidths"] = parameters.bandwidths
        payload[f"{prefix}__frequencies"] = parameters.frequencies
        payload[f"{prefix}__phases"] = parameters.phases
    np.savez(target, **payload)


__all__ = [
    "DistributionalTargetData",
    "HorizonRFFParameters",
    "evaluate_distributional_predictor",
    "prepare_distributional_target_data",
    "save_distributional_preprocessing",
]
