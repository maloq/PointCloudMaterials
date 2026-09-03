from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import Ridge

from src.temporal_vamp.evaluation import CovariancePCA
from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache
from src.temporal_vamp.shooting_predictor import (
    _future_neighbor_metrics,
    _parent_future_means,
)


def _rows_for_parents(parent_indices: np.ndarray, center_count: int) -> np.ndarray:
    return np.concatenate(
        [
            np.arange(
                int(parent_index) * center_count,
                (int(parent_index) + 1) * center_count,
                dtype=np.int64,
            )
            for parent_index in np.asarray(parent_indices, dtype=np.int64).tolist()
        ]
    )


def _standardization(
    values: np.ndarray, optimization_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(values, dtype=np.float64)[optimization_rows]
    mean = selected.mean(axis=0)
    scale = selected.std(axis=0)
    scale = np.where(scale <= 1.0e-10, 1.0, scale)
    return mean, scale


def compute_dynamic_future_targets(
    current_local: np.ndarray,
    parent_future_mean: np.ndarray,
    *,
    optimization_rows: np.ndarray,
    selection_rows: np.ndarray,
    ridge_alphas: Sequence[float],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Construct future-change targets without using held-out validation rows.

    The repository-owned shooting arrays have shapes ``(P, C, D)`` for current
    local embeddings and ``(P, H, C, D)`` for sibling-mean future embeddings.
    Every horizon is standardized using optimization rows only. The residual
    target removes the best optimization-fitted linear prediction of the absolute
    future from the current local embedding; its ridge coefficient is selected on
    selection rows.
    """

    current = np.asarray(current_local, dtype=np.float64)
    future = np.asarray(parent_future_mean, dtype=np.float64)
    if current.ndim != 3:
        raise ValueError(
            f"current_local must have shape (P, C, D), got {current.shape}."
        )
    if future.ndim != 4:
        raise ValueError(
            f"parent_future_mean must have shape (P, H, C, D), got {future.shape}."
        )
    parent_count, center_count, embedding_dim = current.shape
    if (
        future.shape[0] != parent_count
        or future.shape[2] != center_count
        or future.shape[3] != embedding_dim
    ):
        raise ValueError(
            "Current and future shooting embeddings are not aligned: "
            f"current={current.shape}, future={future.shape}."
        )
    alphas = [float(value) for value in ridge_alphas]
    if not alphas or any(value <= 0.0 for value in alphas):
        raise ValueError(f"ridge_alphas must be positive and nonempty, got {alphas}.")

    row_count = parent_count * center_count
    x = current.reshape(row_count, embedding_dim)
    x_mean, x_scale = _standardization(x, optimization_rows)
    x_standardized = (x - x_mean) / x_scale
    horizon_count = int(future.shape[1])
    mean_delta = np.empty(
        (row_count, horizon_count, embedding_dim), dtype=np.float64
    )
    linear_residual = np.empty_like(mean_delta)
    diagnostics: dict[str, Any] = {"ridge_by_horizon": {}}

    for horizon_index in range(horizon_count):
        y = future[:, horizon_index].reshape(row_count, embedding_dim)
        delta = y - x
        delta_mean, delta_scale = _standardization(delta, optimization_rows)
        mean_delta[:, horizon_index] = (delta - delta_mean) / delta_scale

        y_mean, y_scale = _standardization(y, optimization_rows)
        y_standardized = (y - y_mean) / y_scale
        selection_mse: dict[str, float] = {}
        fitted_by_alpha: dict[float, Ridge] = {}
        for alpha in alphas:
            model = Ridge(alpha=alpha)
            model.fit(
                x_standardized[optimization_rows],
                y_standardized[optimization_rows],
            )
            prediction = model.predict(x_standardized[selection_rows])
            selection_mse[f"{alpha:g}"] = float(
                np.mean((prediction - y_standardized[selection_rows]) ** 2)
            )
            fitted_by_alpha[alpha] = model
        selected_alpha = min(
            alphas, key=lambda value: (selection_mse[f"{value:g}"], value)
        )
        selected_model = fitted_by_alpha[selected_alpha]
        prediction = selected_model.predict(x_standardized)
        linear_residual[:, horizon_index] = y_standardized - prediction
        diagnostics["ridge_by_horizon"][str(horizon_index)] = {
            "selected_alpha": float(selected_alpha),
            "selection_mse_by_alpha": selection_mse,
            "optimization_residual_mse": float(
                np.mean(
                    linear_residual[optimization_rows, horizon_index] ** 2
                )
            ),
            "selection_residual_mse": float(
                np.mean(linear_residual[selection_rows, horizon_index] ** 2)
            ),
        }

    return {
        "mean_delta": mean_delta,
        "linear_residual": linear_residual,
    }, diagnostics


def evaluate_saved_shooting_dynamic_targets(
    cache: ShootingEmbeddingCache,
    saved_arrays: Mapping[str, np.ndarray],
    *,
    split_parent_indices: Mapping[str, Sequence[int]],
    ridge_alphas: Sequence[float],
    neighbors: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    parent_count, center_count, input_dim = cache.parent_z.shape
    row_count = int(parent_count * center_count)
    required_arrays = {"coordinates", "prediction", "input_pca"}
    missing = sorted(required_arrays.difference(saved_arrays))
    if missing:
        raise KeyError(
            f"Saved shooting arrays are missing {missing}; available={sorted(saved_arrays)}."
        )
    for name in required_arrays:
        if np.asarray(saved_arrays[name]).shape[0] != row_count:
            raise ValueError(
                f"Saved array {name!r} has the wrong row count: "
                f"expected={row_count}, observed={np.asarray(saved_arrays[name]).shape}."
            )

    parent_splits = {
        name: np.asarray(split_parent_indices[name], dtype=np.int64)
        for name in ("optimization", "selection", "validation")
    }
    split_rows = {
        name: _rows_for_parents(indices, int(center_count))
        for name, indices in parent_splits.items()
    }
    parent_future_mean = _parent_future_means(cache)
    targets, target_diagnostics = compute_dynamic_future_targets(
        cache.parent_local_z,
        parent_future_mean,
        optimization_rows=split_rows["optimization"],
        selection_rows=split_rows["selection"],
        ridge_alphas=ridge_alphas,
    )

    bottleneck_dim = int(np.asarray(saved_arrays["coordinates"]).shape[1])
    flat_context = np.asarray(cache.parent_z, dtype=np.float64).reshape(
        row_count, int(input_dim)
    )
    flat_local = np.asarray(cache.parent_local_z, dtype=np.float64).reshape(
        row_count, int(cache.parent_local_z.shape[-1])
    )
    local_pca = CovariancePCA.fit(
        flat_local[split_rows["optimization"]], dimension=bottleneck_dim
    )
    spaces = {
        "local_encoder": flat_local,
        f"local_pca_{bottleneck_dim}d": local_pca.transform(
            flat_local, dimension=bottleneck_dim
        ),
        "context_encoder": flat_context,
        f"context_pca_{bottleneck_dim}d": np.asarray(
            saved_arrays["input_pca"], dtype=np.float64
        ),
        f"shooting_bottleneck_{bottleneck_dim}d": np.asarray(
            saved_arrays["coordinates"], dtype=np.float64
        ),
        "predicted_absolute_future": np.asarray(
            saved_arrays["prediction"], dtype=np.float64
        ),
    }

    horizon_values = np.asarray(cache.horizons_ps, dtype=np.float64)
    validation_parents = parent_splits["validation"]
    metrics_by_target: dict[str, Any] = {}
    flat_targets: dict[str, np.ndarray] = {}
    for target_name, target in targets.items():
        target_metrics: dict[str, Any] = {}
        for horizon_index, horizon_ps in enumerate(horizon_values.tolist()):
            future_values = target[:, horizon_index]
            target_metrics[f"{float(horizon_ps):g}ps"] = _future_neighbor_metrics(
                spaces,
                future_values,
                cache,
                validation_parents,
                neighbors=int(neighbors),
                seed=int(seed),
            )
        combined = target.reshape(row_count, -1)
        target_metrics["all_horizons"] = _future_neighbor_metrics(
            spaces,
            combined,
            cache,
            validation_parents,
            neighbors=int(neighbors),
            seed=int(seed),
        )
        context_pca_name = f"context_pca_{bottleneck_dim}d"
        bottleneck_name = f"shooting_bottleneck_{bottleneck_dim}d"
        target_metrics["bottleneck_gain_over_context_pca_percent"] = {
            horizon: float(
                100.0
                * (
                    1.0
                    - values[bottleneck_name]["mean_ensemble_future_distance"]
                    / values[context_pca_name]["mean_ensemble_future_distance"]
                )
            )
            for horizon, values in target_metrics.items()
            if horizon != "bottleneck_gain_over_context_pca_percent"
        }
        metrics_by_target[target_name] = target_metrics
        flat_targets[target_name] = combined

    metrics = {
        "scientific_contract": {
            "query_split": "validation source runs only",
            "candidate_filter": "different source run, exact temperature, exact parent phase",
            "mean_delta": "per-dimension standardized sibling-mean z(t+h)-z(t)",
            "linear_residual": (
                "standardized absolute future minus an optimization-fitted current-local "
                "ridge prediction; alpha selected on selection source runs"
            ),
            "static_representations_retrained": False,
        },
        "target_diagnostics": target_diagnostics,
        "future_neighbor_consistency": metrics_by_target,
    }
    return metrics, flat_targets


def load_saved_shooting_arrays(path: str | Path) -> dict[str, np.ndarray]:
    target = Path(path).expanduser().resolve()
    with np.load(target, allow_pickle=False) as payload:
        return {name: payload[name].copy() for name in payload.files}


__all__ = [
    "compute_dynamic_future_targets",
    "evaluate_saved_shooting_dynamic_targets",
    "load_saved_shooting_arrays",
]
