from __future__ import annotations

import copy
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from torch import nn

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.temporal_vamp.evaluation import CovariancePCA
from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache


class ShootingPredictiveBottleneck(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        bottleneck_dim: int,
        target_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.target_dim = int(target_dim)
        self.dropout = float(dropout)
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.bottleneck_dim),
        )
        self.bottleneck_norm = nn.LayerNorm(self.bottleneck_dim)
        self.prediction_head = nn.Linear(self.bottleneck_dim, self.target_dim)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.bottleneck_norm(self.backbone(values))

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coordinates = self.encode(values)
        return coordinates, self.prediction_head(coordinates)


@dataclass(frozen=True)
class ShootingTrainingData:
    input_mean: np.ndarray
    input_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    target_pca: CovariancePCA
    input_pca: CovariancePCA
    parent_future_mean: np.ndarray
    parent_target: np.ndarray
    split_rows: dict[str, np.ndarray]
    parent_splits: dict[str, np.ndarray]


@dataclass(frozen=True)
class FittedShootingPredictor:
    model: ShootingPredictiveBottleneck
    training_data: ShootingTrainingData
    seed: int
    histories: dict[int, dict[str, list[float]]]
    seed_metrics: dict[int, dict[str, float | int]]


def _parent_future_means(cache: ShootingEmbeddingCache) -> np.ndarray:
    parent_count = int(cache.parent_z.shape[0])
    output = np.empty(
        (
            parent_count,
            int(cache.future_z.shape[1]),
            int(cache.future_z.shape[2]),
            int(cache.future_z.shape[3]),
        ),
        dtype=np.float64,
    )
    for parent_index in range(parent_count):
        branch_rows = np.flatnonzero(cache.branch_parent_index == parent_index)
        if branch_rows.size == 0:
            raise RuntimeError(
                f"Shooting embedding cache has no future branch for parent_index={parent_index}."
            )
        output[parent_index] = np.asarray(
            cache.future_z[branch_rows], dtype=np.float64
        ).mean(axis=0)
    return output


def _parent_split_indices(
    cache: ShootingEmbeddingCache,
    *,
    selection_source_velocity_seeds: Sequence[int],
) -> dict[str, np.ndarray]:
    parents = cache.manifest["snapshot"]["parents"]
    selection_seeds = {int(value) for value in selection_source_velocity_seeds}
    optimization: list[int] = []
    selection: list[int] = []
    validation: list[int] = []
    for index, parent in enumerate(parents):
        source_split = str(parent["source_split"])
        source_seed = int(parent["source_velocity_seed"])
        if source_split == "validation":
            validation.append(index)
        elif source_seed in selection_seeds:
            selection.append(index)
        else:
            optimization.append(index)
    result = {
        "optimization": np.asarray(optimization, dtype=np.int64),
        "selection": np.asarray(selection, dtype=np.int64),
        "validation": np.asarray(validation, dtype=np.int64),
    }
    if any(values.size == 0 for values in result.values()):
        raise RuntimeError(
            "Shooting predictor requires nonempty optimization, selection, and validation "
            f"parent sets; got { {name: values.tolist() for name, values in result.items()} }."
        )
    optimization_sources = {
        str(parents[index]["source_run_id"]) for index in result["optimization"]
    }
    selection_sources = {
        str(parents[index]["source_run_id"]) for index in result["selection"]
    }
    validation_sources = {
        str(parents[index]["source_run_id"]) for index in result["validation"]
    }
    if (
        optimization_sources & selection_sources
        or optimization_sources & validation_sources
        or selection_sources & validation_sources
    ):
        raise RuntimeError(
            "Source-run leakage detected between shooting predictor splits."
        )
    return result


def prepare_shooting_training_data(
    cache: ShootingEmbeddingCache,
    *,
    target_pca_dim: int,
    input_pca_dim: int,
    selection_source_velocity_seeds: Sequence[int],
) -> ShootingTrainingData:
    parent_splits = _parent_split_indices(
        cache,
        selection_source_velocity_seeds=selection_source_velocity_seeds,
    )
    atom_count = int(cache.parent_z.shape[1])
    split_rows = {
        name: np.concatenate(
            [
                np.arange(index * atom_count, (index + 1) * atom_count, dtype=np.int64)
                for index in parent_indices.tolist()
            ]
        )
        for name, parent_indices in parent_splits.items()
    }
    optimization_parents = parent_splits["optimization"]
    optimization_branch_mask = np.isin(
        np.asarray(cache.branch_parent_index, dtype=np.int64), optimization_parents
    )
    optimization_future = np.asarray(
        cache.future_z[optimization_branch_mask], dtype=np.float64
    ).reshape(-1, int(cache.future_z.shape[-1]))
    target_pca = CovariancePCA.fit(
        optimization_future,
        dimension=int(target_pca_dim),
    )
    parent_future_mean = _parent_future_means(cache)
    parent_count, horizon_count, center_count, future_dim = parent_future_mean.shape
    transformed = target_pca.transform(
        parent_future_mean.reshape(-1, future_dim), dimension=int(target_pca_dim)
    ).reshape(parent_count, horizon_count, center_count, int(target_pca_dim))
    parent_target = transformed.transpose(0, 2, 1, 3).reshape(
        parent_count * center_count, horizon_count * int(target_pca_dim)
    )
    parent_inputs = np.asarray(cache.parent_z, dtype=np.float64).reshape(
        parent_count * center_count, -1
    )
    optimization_rows = split_rows["optimization"]
    input_mean = parent_inputs[optimization_rows].mean(axis=0)
    input_scale = parent_inputs[optimization_rows].std(axis=0)
    if np.any(input_scale <= 1.0e-10):
        input_scale = np.where(input_scale <= 1.0e-10, 1.0, input_scale)
    target_mean = parent_target[optimization_rows].mean(axis=0)
    target_scale = parent_target[optimization_rows].std(axis=0)
    if np.any(target_scale <= 1.0e-10):
        target_scale = np.where(target_scale <= 1.0e-10, 1.0, target_scale)
    input_pca = CovariancePCA.fit(
        parent_inputs[optimization_rows], dimension=int(input_pca_dim)
    )
    return ShootingTrainingData(
        input_mean=input_mean,
        input_scale=input_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        target_pca=target_pca,
        input_pca=input_pca,
        parent_future_mean=parent_future_mean,
        parent_target=parent_target,
        split_rows=split_rows,
        parent_splits=parent_splits,
    )


def _distance_geometry_loss(
    coordinates: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    coordinate_distances = torch.pdist(coordinates, p=2)
    target_distances = torch.pdist(targets.detach(), p=2)
    coordinate_scaled = coordinate_distances / coordinate_distances.mean().clamp_min(1.0e-8)
    target_scaled = target_distances / target_distances.mean().clamp_min(1.0e-8)
    return torch.mean((coordinate_scaled - target_scaled) ** 2)


def _evaluate_model_loss(
    model: ShootingPredictiveBottleneck,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    geometry_weight: float,
) -> tuple[float, float, float]:
    model.eval()
    with torch.no_grad():
        coordinates, prediction = model(inputs)
        prediction_loss = torch.mean((prediction - targets) ** 2)
        geometry_loss = _distance_geometry_loss(coordinates, targets)
        total = prediction_loss + float(geometry_weight) * geometry_loss
    return float(total), float(prediction_loss), float(geometry_loss)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def fit_shooting_predictive_bottleneck(
    cache: ShootingEmbeddingCache,
    *,
    device: str,
    hidden_dim: int,
    bottleneck_dim: int,
    target_pca_dim: int,
    input_pca_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    geometry_weight: float,
    batch_size: int,
    maximum_epochs: int,
    patience: int,
    seeds: Sequence[int],
    selection_source_velocity_seeds: Sequence[int],
) -> FittedShootingPredictor:
    training_data = prepare_shooting_training_data(
        cache,
        target_pca_dim=int(target_pca_dim),
        input_pca_dim=int(input_pca_dim),
        selection_source_velocity_seeds=selection_source_velocity_seeds,
    )
    inputs = np.asarray(cache.parent_z, dtype=np.float64).reshape(
        -1, int(cache.parent_z.shape[-1])
    )
    standardized_inputs = ((inputs - training_data.input_mean) / training_data.input_scale).astype(
        np.float32
    )
    standardized_targets = (
        (training_data.parent_target - training_data.target_mean)
        / training_data.target_scale
    ).astype(np.float32)
    torch_device = torch.device(device)
    input_tensor = torch.from_numpy(standardized_inputs).to(torch_device)
    target_tensor = torch.from_numpy(standardized_targets).to(torch_device)
    optimization_rows = torch.from_numpy(training_data.split_rows["optimization"]).to(
        torch_device
    )
    selection_rows = torch.from_numpy(training_data.split_rows["selection"]).to(torch_device)
    validation_rows = torch.from_numpy(training_data.split_rows["validation"]).to(torch_device)
    histories: dict[int, dict[str, list[float]]] = {}
    seed_metrics: dict[int, dict[str, float | int]] = {}
    fitted_models: dict[int, ShootingPredictiveBottleneck] = {}
    for seed_value in seeds:
        seed = int(seed_value)
        _seed_everything(seed)
        model = ShootingPredictiveBottleneck(
            input_dim=standardized_inputs.shape[1],
            hidden_dim=int(hidden_dim),
            bottleneck_dim=int(bottleneck_dim),
            target_dim=standardized_targets.shape[1],
            dropout=float(dropout),
        ).to(torch_device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        history = {"optimization": [], "selection": []}
        best_state: dict[str, torch.Tensor] | None = None
        best_selection = float("inf")
        best_epoch = -1
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        for epoch in range(int(maximum_epochs)):
            permutation = torch.randperm(
                optimization_rows.numel(), generator=generator
            ).to(torch_device)
            model.train()
            total_weighted_loss = 0.0
            for start in range(0, int(permutation.numel()), int(batch_size)):
                rows = optimization_rows[permutation[start : start + int(batch_size)]]
                coordinates, prediction = model(input_tensor[rows])
                prediction_loss = torch.mean((prediction - target_tensor[rows]) ** 2)
                geometry_loss = _distance_geometry_loss(
                    coordinates, target_tensor[rows]
                )
                loss = prediction_loss + float(geometry_weight) * geometry_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_weighted_loss += float(loss.detach()) * int(rows.numel())
            optimization_loss = total_weighted_loss / float(permutation.numel())
            selection_loss, _, _ = _evaluate_model_loss(
                model,
                input_tensor[selection_rows],
                target_tensor[selection_rows],
                geometry_weight=float(geometry_weight),
            )
            history["optimization"].append(optimization_loss)
            history["selection"].append(selection_loss)
            if selection_loss < best_selection:
                best_selection = selection_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            if epoch - best_epoch >= int(patience):
                break
        if best_state is None:
            raise RuntimeError(f"Training seed {seed} did not produce a checkpoint.")
        model.load_state_dict(best_state)
        selection_total, selection_prediction, selection_geometry = _evaluate_model_loss(
            model,
            input_tensor[selection_rows],
            target_tensor[selection_rows],
            geometry_weight=float(geometry_weight),
        )
        validation_total, validation_prediction, validation_geometry = _evaluate_model_loss(
            model,
            input_tensor[validation_rows],
            target_tensor[validation_rows],
            geometry_weight=float(geometry_weight),
        )
        histories[seed] = history
        seed_metrics[seed] = {
            "best_epoch": best_epoch,
            "epochs_run": len(history["selection"]),
            "selection_total_loss": selection_total,
            "selection_prediction_mse": selection_prediction,
            "selection_geometry_loss": selection_geometry,
            "validation_total_loss": validation_total,
            "validation_prediction_mse": validation_prediction,
            "validation_geometry_loss": validation_geometry,
        }
        fitted_models[seed] = model.cpu()
    best_seed = min(
        seed_metrics, key=lambda value: float(seed_metrics[value]["selection_total_loss"])
    )
    return FittedShootingPredictor(
        model=fitted_models[best_seed],
        training_data=training_data,
        seed=best_seed,
        histories=histories,
        seed_metrics=seed_metrics,
    )


def _standardized_inputs(
    cache: ShootingEmbeddingCache, training_data: ShootingTrainingData
) -> np.ndarray:
    values = np.asarray(cache.parent_z, dtype=np.float64).reshape(
        -1, int(cache.parent_z.shape[-1])
    )
    return ((values - training_data.input_mean) / training_data.input_scale).astype(
        np.float32
    )


def _model_outputs(
    fitted: FittedShootingPredictor,
    cache: ShootingEmbeddingCache,
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    model = fitted.model.to(torch.device(device)).eval()
    inputs = torch.from_numpy(_standardized_inputs(cache, fitted.training_data)).to(
        torch.device(device)
    )
    with torch.no_grad():
        coordinates, prediction = model(inputs)
    fitted.model.cpu()
    return coordinates.cpu().numpy(), prediction.cpu().numpy()


def _prediction_metrics(
    prediction: np.ndarray, target: np.ndarray, rows: np.ndarray
) -> dict[str, float]:
    pred = prediction[rows]
    truth = target[rows]
    mse = float(np.mean((pred - truth) ** 2))
    denominator = float(np.sum((truth - truth.mean(axis=0)) ** 2))
    r2 = float(1.0 - np.sum((pred - truth) ** 2) / denominator)
    return {"standardized_mse": mse, "r2": r2}


def _prediction_metrics_by_horizon(
    prediction: np.ndarray,
    target: np.ndarray,
    rows: np.ndarray,
    horizons_ps: np.ndarray,
) -> dict[str, dict[str, float]]:
    horizon_count = int(horizons_ps.size)
    if prediction.shape != target.shape or prediction.shape[1] % horizon_count != 0:
        raise ValueError(
            "Multi-horizon prediction and target must have identical [sample, "
            "horizon * target_mode] shapes; "
            f"prediction={prediction.shape}, target={target.shape}, "
            f"horizons={horizons_ps.tolist()}."
        )
    mode_count = prediction.shape[1] // horizon_count
    prediction_by_horizon = prediction.reshape(-1, horizon_count, mode_count)
    target_by_horizon = target.reshape(-1, horizon_count, mode_count)
    return {
        f"{float(horizon):g}ps": _prediction_metrics(
            prediction_by_horizon[:, horizon_index],
            target_by_horizon[:, horizon_index],
            rows,
        )
        for horizon_index, horizon in enumerate(horizons_ps.tolist())
    }


def _ensemble_future_variance_metrics(
    cache: ShootingEmbeddingCache,
    parent_future_mean: np.ndarray,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    parent_indices = np.unique(np.asarray(cache.branch_parent_index, dtype=np.int64))
    for horizon_index, horizon in enumerate(np.asarray(cache.horizons_ps).tolist()):
        within_parent_values: list[float] = []
        ensemble_mean_noise_values: list[float] = []
        for parent_index in parent_indices.tolist():
            branch_rows = np.flatnonzero(cache.branch_parent_index == parent_index)
            branch_values = np.asarray(
                cache.future_z[branch_rows, horizon_index], dtype=np.float64
            )
            parent_mean = branch_values.mean(axis=0)
            within_parent = float(
                np.mean(np.sum((branch_values - parent_mean) ** 2, axis=-1))
            )
            within_parent_values.append(within_parent)
            ensemble_mean_noise_values.append(within_parent / float(branch_rows.size))
        mean_futures = parent_future_mean[:, horizon_index]
        global_mean = mean_futures.reshape(-1, mean_futures.shape[-1]).mean(axis=0)
        between_state = float(
            np.mean(np.sum((mean_futures - global_mean) ** 2, axis=-1))
        )
        within_branch = float(np.mean(within_parent_values))
        ensemble_mean_noise = float(np.mean(ensemble_mean_noise_values))
        metrics[f"{float(horizon):g}ps"] = {
            "within_parent_branch_variance": within_branch,
            "between_parent_atom_variance": between_state,
            "between_fraction": between_state / (between_state + within_branch),
            "estimated_ensemble_mean_noise_variance": ensemble_mean_noise,
            "estimated_ensemble_mean_reliability": between_state
            / (between_state + ensemble_mean_noise),
        }
    return metrics


def _fit_ridge_baseline(
    inputs: np.ndarray,
    targets: np.ndarray,
    split_rows: Mapping[str, np.ndarray],
    alphas: Sequence[float],
) -> tuple[Ridge, dict[str, Any]]:
    models: dict[float, Ridge] = {}
    selection_mse: dict[float, float] = {}
    for raw_alpha in alphas:
        alpha = float(raw_alpha)
        model = Ridge(alpha=alpha).fit(
            inputs[split_rows["optimization"]], targets[split_rows["optimization"]]
        )
        prediction = model.predict(inputs[split_rows["selection"]])
        selection_mse[alpha] = float(
            np.mean((prediction - targets[split_rows["selection"]]) ** 2)
        )
        models[alpha] = model
    best_alpha = min(selection_mse, key=selection_mse.get)
    return models[best_alpha], {
        "selected_alpha": best_alpha,
        "selection_mse_by_alpha": {str(key): value for key, value in selection_mse.items()},
    }


def _temperature_phase_mean_prediction(
    cache: ShootingEmbeddingCache,
    targets: np.ndarray,
    training_data: ShootingTrainingData,
) -> np.ndarray:
    parents = cache.manifest["snapshot"]["parents"]
    center_count = int(cache.parent_z.shape[1])
    optimization_parents = set(
        training_data.parent_splits["optimization"].tolist()
    )
    groups: dict[tuple[float, str], list[int]] = defaultdict(list)
    for parent_index, parent in enumerate(parents):
        groups[(float(parent["temperature_K"]), str(parent["phase"]))].append(
            parent_index
        )
    prediction = np.empty_like(targets)
    assigned = np.zeros(targets.shape[0], dtype=bool)
    for group, parent_indices in groups.items():
        optimization_group = [
            index for index in parent_indices if index in optimization_parents
        ]
        if not optimization_group:
            raise RuntimeError(
                "Temperature/phase evaluation baseline has no optimization parent "
                f"for group={group}."
            )
        optimization_rows = np.concatenate(
            [
                np.arange(
                    index * center_count,
                    (index + 1) * center_count,
                    dtype=np.int64,
                )
                for index in optimization_group
            ]
        )
        group_rows = np.concatenate(
            [
                np.arange(
                    index * center_count,
                    (index + 1) * center_count,
                    dtype=np.int64,
                )
                for index in parent_indices
            ]
        )
        prediction[group_rows] = targets[optimization_rows].mean(axis=0)
        assigned[group_rows] = True
    if not np.all(assigned):
        raise RuntimeError(
            "Temperature/phase evaluation baseline did not assign every shooting row."
        )
    return prediction


def _future_neighbor_metrics(
    spaces: Mapping[str, np.ndarray],
    future: np.ndarray,
    cache: ShootingEmbeddingCache,
    parent_indices: np.ndarray,
    *,
    neighbors: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    parents = cache.manifest["snapshot"]["parents"]
    center_count = int(cache.parent_z.shape[1])
    selected_rows = np.concatenate(
        [
            np.arange(index * center_count, (index + 1) * center_count, dtype=np.int64)
            for index in parent_indices.tolist()
        ]
    )
    selected_parent_for_row = np.repeat(parent_indices, center_count)
    group_keys = [
        (
            str(parents[parent_index]["source_run_id"]),
            float(parents[parent_index]["temperature_K"]),
            str(parents[parent_index]["phase"]),
        )
        for parent_index in selected_parent_for_row.tolist()
    ]
    rng = np.random.default_rng(int(seed))
    random_neighbors = np.empty((selected_rows.size, int(neighbors)), dtype=np.int64)
    candidate_sets: list[np.ndarray] = []
    for query_position, (source_run, temperature, phase) in enumerate(group_keys):
        candidates = np.asarray(
            [
                position
                for position, (candidate_run, candidate_temperature, candidate_phase) in enumerate(group_keys)
                if candidate_run != source_run
                and candidate_temperature == temperature
                and candidate_phase == phase
            ],
            dtype=np.int64,
        )
        if candidates.size < int(neighbors):
            raise RuntimeError(
                "Shooting future-neighbor matching has too few cross-source candidates: "
                f"query={query_position}, source={source_run}, temperature={temperature}, "
                f"phase={phase}, candidates={candidates.size}, required={neighbors}."
            )
        candidate_sets.append(candidates)
        random_neighbors[query_position] = rng.choice(
            candidates, size=int(neighbors), replace=False
        )
    future_selected = future[selected_rows]
    random_distance = np.linalg.norm(
        future_selected[random_neighbors] - future_selected[:, None, :], axis=2
    ).mean(axis=1)
    results: dict[str, dict[str, float | int]] = {}
    for name, full_values in spaces.items():
        values = np.asarray(full_values)[selected_rows]
        selected = np.empty((selected_rows.size, int(neighbors)), dtype=np.int64)
        grouped: dict[tuple[str, float, str], list[int]] = defaultdict(list)
        for query_position, key in enumerate(group_keys):
            grouped[key].append(query_position)
        for key, positions_list in grouped.items():
            source_run, temperature, phase = key
            positions = np.asarray(positions_list, dtype=np.int64)
            candidates = np.asarray(
                [
                    position
                    for position, (candidate_run, candidate_temperature, candidate_phase) in enumerate(group_keys)
                    if candidate_run != source_run
                    and candidate_temperature == temperature
                    and candidate_phase == phase
                ],
                dtype=np.int64,
            )
            search = NearestNeighbors(
                n_neighbors=int(neighbors), metric="euclidean", algorithm="brute"
            ).fit(values[candidates])
            local = search.kneighbors(values[positions], return_distance=False)
            selected[positions] = candidates[local]
        future_distance = np.linalg.norm(
            future_selected[selected] - future_selected[:, None, :], axis=2
        ).mean(axis=1)
        results[name] = {
            "queries": int(selected_rows.size),
            "neighbors": int(neighbors),
            "mean_ensemble_future_distance": float(future_distance.mean()),
            "sem_ensemble_future_distance": float(
                future_distance.std(ddof=1) / np.sqrt(future_distance.size)
            ),
            "matched_random_mean_ensemble_future_distance": float(random_distance.mean()),
            "distance_over_matched_random": float(
                future_distance.mean() / random_distance.mean()
            ),
            "candidate_count": int(min(map(len, candidate_sets))),
        }
    return results


def evaluate_shooting_predictor(
    fitted: FittedShootingPredictor,
    cache: ShootingEmbeddingCache,
    *,
    device: str,
    ridge_alphas: Sequence[float],
    neighbors: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    training_data = fitted.training_data
    coordinates, prediction = _model_outputs(fitted, cache, device=device)
    standardized_target = (
        (training_data.parent_target - training_data.target_mean)
        / training_data.target_scale
    ).astype(np.float64)
    standardized_input = _standardized_inputs(cache, training_data).astype(np.float64)
    ridge, ridge_selection = _fit_ridge_baseline(
        standardized_input,
        standardized_target,
        training_data.split_rows,
        ridge_alphas,
    )
    ridge_prediction = ridge.predict(standardized_input)
    zero_prediction = np.zeros_like(standardized_target)
    temperature_phase_prediction = _temperature_phase_mean_prediction(
        cache, standardized_target, training_data
    )
    prediction_metrics: dict[str, Any] = {
        "neural": {
            split: _prediction_metrics(
                prediction, standardized_target, training_data.split_rows[split]
            )
            for split in ("selection", "validation")
        },
        "ridge": {
            **ridge_selection,
            **{
                split: _prediction_metrics(
                    ridge_prediction, standardized_target, training_data.split_rows[split]
                )
                for split in ("selection", "validation")
            },
        },
        "optimization_mean": {
            split: _prediction_metrics(
                zero_prediction, standardized_target, training_data.split_rows[split]
            )
            for split in ("selection", "validation")
        },
        "temperature_phase_mean": {
            split: _prediction_metrics(
                temperature_phase_prediction,
                standardized_target,
                training_data.split_rows[split],
            )
            for split in ("selection", "validation")
        },
    }
    horizons_ps = np.asarray(cache.horizons_ps, dtype=np.float64)
    for model_name, model_prediction in (
        ("neural", prediction),
        ("ridge", ridge_prediction),
        ("optimization_mean", zero_prediction),
        ("temperature_phase_mean", temperature_phase_prediction),
    ):
        prediction_metrics[model_name]["by_horizon"] = {
            split: _prediction_metrics_by_horizon(
                model_prediction,
                standardized_target,
                training_data.split_rows[split],
                horizons_ps,
            )
            for split in ("selection", "validation")
        }
    flat_inputs = np.asarray(cache.parent_z, dtype=np.float64).reshape(
        -1, int(cache.parent_z.shape[-1])
    )
    input_pca = training_data.input_pca.transform(
        flat_inputs, dimension=int(coordinates.shape[1])
    )
    spaces = {
        "context_encoder": flat_inputs,
        f"context_pca_{coordinates.shape[1]}d": input_pca,
        f"shooting_bottleneck_{coordinates.shape[1]}d": coordinates,
        "predicted_future": prediction,
    }
    parent_count, horizon_count, center_count, future_dim = (
        training_data.parent_future_mean.shape
    )
    neighbor_metrics: dict[str, Any] = {}
    validation_parents = training_data.parent_splits["validation"]
    for horizon_index, horizon in enumerate(np.asarray(cache.horizons_ps).tolist()):
        future = training_data.parent_future_mean[:, horizon_index].reshape(
            parent_count * center_count, future_dim
        )
        neighbor_metrics[f"{float(horizon):g}ps"] = _future_neighbor_metrics(
            spaces,
            future,
            cache,
            validation_parents,
            neighbors=int(neighbors),
            seed=int(seed),
        )
    combined_future = training_data.parent_future_mean.transpose(0, 2, 1, 3).reshape(
        parent_count * center_count, horizon_count * future_dim
    )
    neighbor_metrics["all_horizons"] = _future_neighbor_metrics(
        spaces,
        combined_future,
        cache,
        validation_parents,
        neighbors=int(neighbors),
        seed=int(seed),
    )
    metrics = {
        "selected_seed": fitted.seed,
        "seed_metrics": {str(key): value for key, value in fitted.seed_metrics.items()},
        "split_parent_indices": {
            name: values.tolist() for name, values in training_data.parent_splits.items()
        },
        "split_sample_counts": {
            name: int(values.size) for name, values in training_data.split_rows.items()
        },
        "prediction": prediction_metrics,
        "ensemble_future_variance": _ensemble_future_variance_metrics(
            cache, training_data.parent_future_mean
        ),
        "future_neighbor_consistency": neighbor_metrics,
    }
    arrays = {
        "coordinates": coordinates,
        "prediction": prediction,
        "ridge_prediction": ridge_prediction,
        "temperature_phase_mean_prediction": temperature_phase_prediction,
        "standardized_target": standardized_target,
        "input_pca": input_pca,
    }
    return metrics, arrays


def save_shooting_predictor(
    fitted: FittedShootingPredictor,
    path: str | Path,
) -> None:
    target = Path(path)
    payload = {
        "state_dict": fitted.model.state_dict(),
        "input_dim": fitted.model.input_dim,
        "hidden_dim": fitted.model.hidden_dim,
        "bottleneck_dim": fitted.model.bottleneck_dim,
        "target_dim": fitted.model.target_dim,
        "dropout": fitted.model.dropout,
        "seed": fitted.seed,
    }
    torch.save(payload, target)
    data = fitted.training_data
    np.savez(
        target.with_suffix(".preprocessing.npz"),
        input_mean=data.input_mean,
        input_scale=data.input_scale,
        target_mean=data.target_mean,
        target_scale=data.target_scale,
        target_pca_mean=data.target_pca.mean_,
        target_pca_components=data.target_pca.components_,
        target_pca_eigenvalues=data.target_pca.eigenvalues_,
        input_pca_mean=data.input_pca.mean_,
        input_pca_components=data.input_pca.components_,
        input_pca_eigenvalues=data.input_pca.eigenvalues_,
    )


def plot_shooting_training(
    fitted: FittedShootingPredictor,
    path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for seed, history in fitted.histories.items():
        ax.plot(history["selection"], label=f"seed {seed} selection")
    ax.set(xlabel="epoch", ylabel="prediction + geometry loss", yscale="log")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(path), dpi=180)
    plt.close(fig)


def plot_shooting_neighbor_metrics(
    metrics: Mapping[str, Mapping[str, Mapping[str, float | int]]],
    path: str | Path,
) -> None:
    horizons = list(metrics)
    space_names = list(metrics[horizons[0]])
    x = np.arange(len(horizons), dtype=np.float64)
    width = 0.8 / len(space_names)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.3 * len(horizons)), 4.7))
    for index, name in enumerate(space_names):
        values = [
            float(metrics[horizon][name]["distance_over_matched_random"])
            for horizon in horizons
        ]
        ax.bar(x + (index - (len(space_names) - 1) / 2.0) * width, values, width, label=name)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set(
        xticks=x,
        xticklabels=horizons,
        ylabel="ensemble-future distance / matched random",
        xlabel="prediction horizon",
    )
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(path), dpi=180)
    plt.close(fig)


def write_shooting_json(path: str | Path, value: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


__all__ = [
    "FittedShootingPredictor",
    "ShootingPredictiveBottleneck",
    "evaluate_shooting_predictor",
    "fit_shooting_predictive_bottleneck",
    "plot_shooting_neighbor_metrics",
    "plot_shooting_training",
    "prepare_shooting_training_data",
    "save_shooting_predictor",
    "write_shooting_json",
]
