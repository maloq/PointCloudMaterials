"""Short-horizon momentum control for position-conditioned shooting data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from matplotlib import pyplot as plt

from src.temporal_vamp.shooting_distribution import DistributionalTargetData
from src.temporal_vamp.shooting_dynamics import (
    ShootingDynamicalFeatureCache,
    _branch_future_neighbor_metrics,
    _branch_rows_for_parents,
    _prediction_metrics,
    fit_selected_ridge_residual,
    individual_branch_signatures,
)
from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache
from src.temporal_vamp.shooting_predictor import _future_neighbor_metrics


@dataclass(frozen=True)
class ShortHorizonVelocityResult:
    metrics: dict[str, Any]
    arrays: dict[str, np.ndarray]
    model_arrays: dict[str, np.ndarray]


def _prediction_metrics_by_horizon(
    prediction: np.ndarray,
    target: np.ndarray,
    rows: np.ndarray,
    *,
    horizon_count: int,
    signature_dim: int,
) -> dict[int, dict[str, float]]:
    prediction_blocks = prediction.reshape(-1, int(horizon_count), int(signature_dim))
    target_blocks = target.reshape(-1, int(horizon_count), int(signature_dim))
    return {
        horizon_index: {
            "mse": values[0],
            "r2": values[1],
        }
        for horizon_index in range(int(horizon_count))
        for values in (
            _prediction_metrics(
                prediction_blocks[:, horizon_index],
                target_blocks[:, horizon_index],
                rows,
            ),
        )
    }


def _aggregate_branch_predictions(
    prediction: np.ndarray,
    branch_parent: np.ndarray,
    *,
    parent_count: int,
    center_count: int,
) -> np.ndarray:
    reshaped = prediction.reshape(branch_parent.size, int(center_count), -1)
    output = np.empty(
        (int(parent_count), int(center_count), int(reshaped.shape[-1])),
        dtype=np.float64,
    )
    for parent_index in range(int(parent_count)):
        output[parent_index] = reshaped[branch_parent == parent_index].mean(axis=0)
    return output.reshape(int(parent_count) * int(center_count), -1)


def evaluate_short_horizon_velocity(
    cache: ShootingEmbeddingCache,
    targets: DistributionalTargetData,
    dynamics: ShootingDynamicalFeatureCache,
    position_arrays: Mapping[str, np.ndarray],
    *,
    velocity_pca_dimensions: Sequence[int],
    ridge_alphas: Sequence[float],
    neighbors: int,
    seed: int,
) -> ShortHorizonVelocityResult:
    """Measure incremental branch predictability from the exact time-zero velocities."""

    parent_count, center_count = cache.parent_local_z.shape[:2]
    parent_row_count = int(parent_count * center_count)
    branch_parent = np.asarray(cache.branch_parent_index, dtype=np.int64)
    branch_count = int(branch_parent.size)
    if not np.array_equal(branch_parent, dynamics.branch_parent_index):
        raise RuntimeError(
            "Short-horizon embedding and dynamical caches disagree on branch ordering."
        )
    base_standardized = np.asarray(
        position_arrays["standardized_prediction"], dtype=np.float64
    )
    base_raw = np.asarray(position_arrays["prediction"], dtype=np.float64)
    base_representation = np.asarray(position_arrays["representation"], dtype=np.float64)
    if base_standardized.shape != targets.target_modes.shape:
        raise RuntimeError(
            "Short-horizon position prediction shape changed: "
            f"prediction={base_standardized.shape}, target={targets.target_modes.shape}."
        )

    branch_signatures = individual_branch_signatures(cache, targets)
    horizon_count = int(targets.selected_horizons_ps.size)
    signature_dim = int(targets.distribution_signature.shape[-1])
    branch_target_raw = branch_signatures.transpose(0, 2, 1, 3).reshape(
        branch_count * center_count, horizon_count * signature_dim
    )
    branch_target = (
        branch_target_raw - targets.target_mean[None, :]
    ) / targets.target_scale[None, :]
    parent_rows = (
        branch_parent[:, None] * center_count
        + np.arange(center_count, dtype=np.int64)[None, :]
    ).reshape(-1)
    position_branch = base_standardized[parent_rows]
    velocity_raw = np.asarray(dynamics.velocity_features, dtype=np.float64).reshape(
        branch_count * center_count, -1
    )
    optimization_rows = _branch_rows_for_parents(
        branch_parent, targets.parent_splits["optimization"], center_count
    )
    selection_rows = _branch_rows_for_parents(
        branch_parent, targets.parent_splits["selection"], center_count
    )
    validation_rows = _branch_rows_for_parents(
        branch_parent, targets.parent_splits["validation"], center_count
    )
    velocity = fit_selected_ridge_residual(
        velocity_raw,
        position_branch,
        branch_target,
        optimization_rows=optimization_rows,
        selection_rows=selection_rows,
        validation_rows=validation_rows,
        dimensions=velocity_pca_dimensions,
        alphas=ridge_alphas,
    )
    velocity_raw_prediction = (
        velocity.prediction * targets.target_scale[None, :]
        + targets.target_mean[None, :]
    )
    position_raw_prediction = base_raw[parent_rows]

    prediction_metrics: dict[str, Any] = {}
    for name, prediction in {
        "position_only": position_branch,
        "velocity_conditioned": velocity.prediction,
    }.items():
        selection_mse, selection_r2 = _prediction_metrics(
            prediction, branch_target, selection_rows
        )
        validation_mse, validation_r2 = _prediction_metrics(
            prediction, branch_target, validation_rows
        )
        horizon_values = _prediction_metrics_by_horizon(
            prediction,
            branch_target,
            validation_rows,
            horizon_count=horizon_count,
            signature_dim=signature_dim,
        )
        prediction_metrics[name] = {
            "selection_mse": selection_mse,
            "selection_r2": selection_r2,
            "validation_mse": validation_mse,
            "validation_r2": validation_r2,
            "validation_by_horizon": {
                f"{float(targets.selected_horizons_ps[index]):g}ps": values
                for index, values in horizon_values.items()
            },
        }

    branch_predictions = {
        "position_only": position_raw_prediction,
        "velocity_conditioned": velocity_raw_prediction,
    }
    branch_retrieval: dict[str, Any] = {}
    for horizon_index, horizon in enumerate(targets.selected_horizons_ps.tolist()):
        spaces = {
            name: values.reshape(
                branch_count * center_count, horizon_count, signature_dim
            )[:, horizon_index]
            for name, values in branch_predictions.items()
        }
        spaces["velocity_descriptor_pca"] = velocity.features
        future = branch_target_raw.reshape(
            branch_count * center_count, horizon_count, signature_dim
        )[:, horizon_index]
        result = _branch_future_neighbor_metrics(
            spaces,
            future,
            cache,
            targets.parent_splits["validation"],
            neighbors=int(neighbors),
            seed=int(seed),
        )
        baseline = float(result["position_only"]["mean_individual_future_distance"])
        result["gain_over_position_only_percent"] = {
            name: float(
                100.0
                * (1.0 - float(values["mean_individual_future_distance"]) / baseline)
            )
            for name, values in result.items()
        }
        branch_retrieval[f"{float(horizon):g}ps"] = result
    combined_spaces = {
        **branch_predictions,
        "velocity_descriptor_pca": velocity.features,
    }
    combined = _branch_future_neighbor_metrics(
        combined_spaces,
        branch_target_raw,
        cache,
        targets.parent_splits["validation"],
        neighbors=int(neighbors),
        seed=int(seed),
    )
    baseline = float(combined["position_only"]["mean_individual_future_distance"])
    combined["gain_over_position_only_percent"] = {
        name: float(
            100.0
            * (1.0 - float(values["mean_individual_future_distance"]) / baseline)
        )
        for name, values in combined.items()
    }
    branch_retrieval["all_horizons"] = combined

    ensemble_predictions = {
        name: _aggregate_branch_predictions(
            values,
            branch_parent,
            parent_count=parent_count,
            center_count=center_count,
        )
        for name, values in branch_predictions.items()
    }
    ensemble_retrieval: dict[str, Any] = {}
    for horizon_index, horizon in enumerate(targets.selected_horizons_ps.tolist()):
        spaces = {
            name: values.reshape(parent_row_count, horizon_count, signature_dim)[
                :, horizon_index
            ]
            for name, values in ensemble_predictions.items()
        }
        result = _future_neighbor_metrics(
            spaces,
            targets.distribution_signature[:, horizon_index],
            cache,
            targets.parent_splits["validation"],
            neighbors=int(neighbors),
            seed=int(seed),
        )
        baseline = float(result["position_only"]["mean_ensemble_future_distance"])
        result["gain_over_position_only_percent"] = {
            name: float(
                100.0
                * (1.0 - float(values["mean_ensemble_future_distance"]) / baseline)
            )
            for name, values in result.items()
        }
        ensemble_retrieval[f"{float(horizon):g}ps"] = result
    combined = _future_neighbor_metrics(
        ensemble_predictions,
        targets.distribution_signature.reshape(parent_row_count, -1),
        cache,
        targets.parent_splits["validation"],
        neighbors=int(neighbors),
        seed=int(seed),
    )
    baseline = float(combined["position_only"]["mean_ensemble_future_distance"])
    combined["gain_over_position_only_percent"] = {
        name: float(
            100.0 * (1.0 - float(values["mean_ensemble_future_distance"]) / baseline)
        )
        for name, values in combined.items()
    }
    ensemble_retrieval["all_horizons"] = combined

    model_arrays: dict[str, np.ndarray] = {
        "selected_dimension": np.asarray(velocity.selected_dimension),
        "selected_alpha": np.asarray(velocity.selected_alpha),
        "coefficients": velocity.coefficients,
        "intercept": velocity.intercept,
    }
    for name, values in velocity.preprocessing.items():
        model_arrays[name] = values
    return ShortHorizonVelocityResult(
        metrics={
            "velocity_residual": {
                "selected_pca_dimension": velocity.selected_dimension,
                "selected_alpha": velocity.selected_alpha,
                "selection_mse": velocity.selection_mse,
                "validation_mse": velocity.validation_mse,
                "validation_r2": velocity.validation_r2,
            },
            "individual_branch_prediction": prediction_metrics,
            "individual_branch_retrieval": branch_retrieval,
            "ensemble_of_branch_predictions_retrieval": ensemble_retrieval,
            "counts": {
                "parents": int(parent_count),
                "branches": branch_count,
                "centers": int(center_count),
                "branch_rows": int(branch_count * center_count),
            },
        },
        arrays={
            "position_parent_prediction": base_raw.astype(np.float32),
            "velocity_branch_prediction": velocity_raw_prediction.astype(np.float32),
            "velocity_features": velocity.features.astype(np.float32),
            "position_representation": base_representation.astype(np.float32),
        },
        model_arrays=model_arrays,
    )


def plot_short_horizon_velocity(metrics: Mapping[str, Any], path: str | Path) -> None:
    prediction = metrics["individual_branch_prediction"]
    retrieval = metrics["individual_branch_retrieval"]
    horizons = [key for key in retrieval if key != "all_horizons"]
    position_r2 = [
        float(prediction["position_only"]["validation_by_horizon"][key]["r2"])
        for key in horizons
    ]
    velocity_r2 = [
        float(prediction["velocity_conditioned"]["validation_by_horizon"][key]["r2"])
        for key in horizons
    ]
    retrieval_gain = [
        float(retrieval[key]["gain_over_position_only_percent"]["velocity_conditioned"])
        for key in horizons
    ]
    descriptor_gain = [
        float(retrieval[key]["gain_over_position_only_percent"]["velocity_descriptor_pca"])
        for key in horizons
    ]
    x = np.arange(len(horizons), dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].plot(x, position_r2, marker="o", label="position only")
    axes[0].plot(x, velocity_r2, marker="o", label="velocity conditioned")
    axes[0].set(
        xticks=x,
        xticklabels=horizons,
        xlabel="future horizon",
        ylabel="held-out individual-branch R2",
    )
    axes[0].legend(frameon=False)
    axes[1].plot(x, retrieval_gain, marker="o", label="predicted signature")
    axes[1].plot(x, descriptor_gain, marker="o", label="velocity descriptor PCA")
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set(
        xticks=x,
        xticklabels=horizons,
        xlabel="future horizon",
        ylabel="retrieval gain over position only (%)",
    )
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(Path(path), dpi=180)
    plt.close(fig)


__all__ = [
    "ShortHorizonVelocityResult",
    "evaluate_short_horizon_velocity",
    "plot_short_horizon_velocity",
]
