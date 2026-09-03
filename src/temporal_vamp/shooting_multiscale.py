from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import nn

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.temporal_vamp.evaluation import CovariancePCA
from src.temporal_vamp.shooting_ablation import (
    _rows_for_parents,
    compute_dynamic_future_targets,
)
from src.temporal_vamp.shooting_context import ShootingContextTokenCache
from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache
from src.temporal_vamp.shooting_predictor import (
    _future_neighbor_metrics,
    _parent_future_means,
    _parent_split_indices,
)


class MultiscaleFuturePredictor(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        representation_dim: int,
        target_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.representation_dim = int(representation_dim)
        self.target_dim = int(target_dim)
        self.dropout = float(dropout)
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.representation_dim),
            nn.GELU(),
            nn.LayerNorm(self.representation_dim),
        )
        self.prediction_head = nn.Linear(self.representation_dim, self.target_dim)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.backbone(values)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.encode(values)
        return representation, self.prediction_head(representation)


@dataclass(frozen=True)
class DynamicTargetData:
    selected_horizon_indices: np.ndarray
    selected_horizons_ps: np.ndarray
    target_modes: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    mean_delta: np.ndarray
    target_pcas: tuple[CovariancePCA, ...]
    split_rows: dict[str, np.ndarray]
    parent_splits: dict[str, np.ndarray]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class FittedMultiscalePredictor:
    model: MultiscaleFuturePredictor
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    seed: int
    histories: dict[int, dict[str, list[float]]]
    seed_metrics: dict[int, dict[str, float | int]]
    predictions_by_seed: dict[int, np.ndarray]
    representations_by_seed: dict[int, np.ndarray]


def _weighted_mean_std(
    values: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    denominator = weights.sum(axis=2, keepdims=True)
    normalized = weights / denominator
    mean = np.sum(normalized[..., None] * values, axis=2)
    variance = np.sum(
        normalized[..., None] * (values - mean[:, :, None, :]) ** 2,
        axis=2,
    )
    return mean, np.sqrt(np.maximum(variance, 0.0))


def build_multiscale_feature_variants(
    base_cache: ShootingEmbeddingCache,
    context_cache: ShootingContextTokenCache,
    *,
    radial_scales_angstrom: Sequence[float],
) -> dict[str, np.ndarray]:
    central = np.asarray(base_cache.parent_local_z, dtype=np.float64)
    satellites = np.asarray(context_cache.satellite_z, dtype=np.float64)
    offsets = np.asarray(context_cache.satellite_offsets, dtype=np.float64)
    central_descriptors = np.asarray(
        context_cache.central_descriptors, dtype=np.float64
    )
    satellite_descriptors = np.asarray(
        context_cache.satellite_descriptors, dtype=np.float64
    )
    if satellites.shape[:2] != central.shape[:2] or satellites.shape[-1] != central.shape[-1]:
        raise ValueError(
            "Central and satellite GeoFrame embeddings are not aligned: "
            f"central={central.shape}, satellites={satellites.shape}."
        )
    radii = np.linalg.norm(offsets, axis=-1)
    scales = [float(value) for value in radial_scales_angstrom]
    if not scales or any(value <= 0.0 for value in scales):
        raise ValueError(
            f"radial_scales_angstrom must be positive and nonempty, got {scales}."
        )

    parent_count, center_count, embedding_dim = central.shape
    row_count = parent_count * center_count
    unweighted_z_mean = satellites.mean(axis=2)
    unweighted_z_std = satellites.std(axis=2)
    unweighted_descriptor_mean = satellite_descriptors.mean(axis=2)
    unweighted_descriptor_std = satellite_descriptors.std(axis=2)
    radial_summary = np.stack(
        [radii.mean(axis=2), radii.std(axis=2), radii.min(axis=2), radii.max(axis=2)],
        axis=-1,
    )
    mean_context = np.concatenate(
        [
            central,
            central_descriptors,
            unweighted_z_mean,
            unweighted_z_std,
            unweighted_descriptor_mean,
            unweighted_descriptor_std,
            radial_summary,
        ],
        axis=-1,
    )

    multiscale_parts = [central, central_descriptors]
    for scale in scales:
        weights = np.exp(-0.5 * np.square(radii / scale))
        z_mean, z_std = _weighted_mean_std(satellites, weights)
        descriptor_mean, descriptor_std = _weighted_mean_std(
            satellite_descriptors, weights
        )
        denominator = weights.sum(axis=2)
        weighted_radius = (weights * radii).sum(axis=2) / denominator
        weighted_radius_variance = (
            weights * np.square(radii - weighted_radius[:, :, None])
        ).sum(axis=2) / denominator
        scale_summary = np.stack(
            [
                denominator / float(satellites.shape[2]),
                weighted_radius,
                np.sqrt(np.maximum(weighted_radius_variance, 0.0)),
            ],
            axis=-1,
        )
        multiscale_parts.extend(
            [z_mean, z_std, descriptor_mean, descriptor_std, scale_summary]
        )
    multiscale = np.concatenate(multiscale_parts, axis=-1)
    old_context = np.asarray(base_cache.parent_z, dtype=np.float64)
    variants = {
        "local": central.reshape(row_count, embedding_dim),
        "local_q4_q6": np.concatenate(
            [central, central_descriptors], axis=-1
        ).reshape(row_count, -1),
        "old_mean_std_8": old_context.reshape(row_count, old_context.shape[-1]),
        "mean_std_context": mean_context.reshape(row_count, mean_context.shape[-1]),
        "multiscale_context": multiscale.reshape(row_count, multiscale.shape[-1]),
    }
    for name, values in variants.items():
        if not np.isfinite(values).all():
            raise RuntimeError(
                f"Non-finite values in multiscale feature variant {name!r}, "
                f"shape={values.shape}."
            )
    return variants


def prepare_dynamic_target_data(
    cache: ShootingEmbeddingCache,
    *,
    horizons_ps: Sequence[float],
    target_pca_dim: int,
    selection_source_velocity_seeds: Sequence[int],
    residual_ridge_alphas: Sequence[float],
) -> DynamicTargetData:
    parent_splits = _parent_split_indices(
        cache,
        selection_source_velocity_seeds=selection_source_velocity_seeds,
    )
    center_count = int(cache.parent_z.shape[1])
    split_rows = {
        name: _rows_for_parents(indices, center_count)
        for name, indices in parent_splits.items()
    }
    parent_future_mean = _parent_future_means(cache)
    dynamic_targets, diagnostics = compute_dynamic_future_targets(
        cache.parent_local_z,
        parent_future_mean,
        optimization_rows=split_rows["optimization"],
        selection_rows=split_rows["selection"],
        ridge_alphas=residual_ridge_alphas,
    )
    available_horizons = np.asarray(cache.horizons_ps, dtype=np.float64)
    requested = np.asarray([float(value) for value in horizons_ps], dtype=np.float64)
    selected_indices: list[int] = []
    for value in requested.tolist():
        matches = np.flatnonzero(np.isclose(available_horizons, value, rtol=0.0, atol=1.0e-9))
        if matches.size != 1:
            raise ValueError(
                f"Requested horizon {value:g} ps is not uniquely available in "
                f"{available_horizons.tolist()}."
            )
        selected_indices.append(int(matches[0]))
    if len(set(selected_indices)) != len(selected_indices):
        raise ValueError(f"Duplicate training horizons were requested: {requested.tolist()}.")

    mean_delta = dynamic_targets["mean_delta"][:, selected_indices]
    optimization_rows = split_rows["optimization"]
    pcas: list[CovariancePCA] = []
    mode_blocks: list[np.ndarray] = []
    for local_horizon_index in range(int(requested.size)):
        values = mean_delta[:, local_horizon_index]
        pca = CovariancePCA.fit(
            values[optimization_rows], dimension=int(target_pca_dim)
        )
        pcas.append(pca)
        mode_blocks.append(pca.transform(values, dimension=int(target_pca_dim)))
    target_modes = np.stack(mode_blocks, axis=1).reshape(
        mean_delta.shape[0], requested.size * int(target_pca_dim)
    )
    target_mean = target_modes[optimization_rows].mean(axis=0)
    target_scale = target_modes[optimization_rows].std(axis=0)
    target_scale = np.where(target_scale <= 1.0e-10, 1.0, target_scale)
    standardized_modes = (target_modes - target_mean) / target_scale
    diagnostics = {
        **diagnostics,
        "target": "sibling-mean standardized GeoFrame future change",
        "target_pca_fit": "optimization source runs only, independently per horizon",
        "target_pca_eigenvalues": [pca.eigenvalues_.tolist() for pca in pcas],
    }
    return DynamicTargetData(
        selected_horizon_indices=np.asarray(selected_indices, dtype=np.int64),
        selected_horizons_ps=requested,
        target_modes=standardized_modes.astype(np.float32),
        target_mean=target_mean,
        target_scale=target_scale,
        mean_delta=mean_delta,
        target_pcas=tuple(pcas),
        split_rows=split_rows,
        parent_splits=parent_splits,
        diagnostics=diagnostics,
    )


def _standardize_features(
    values: np.ndarray, optimization_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values[optimization_rows].mean(axis=0)
    scale = values[optimization_rows].std(axis=0)
    scale = np.where(scale <= 1.0e-10, 1.0, scale)
    standardized = ((values - mean) / scale).astype(np.float32)
    return standardized, mean, scale


def _metrics(prediction: np.ndarray, target: np.ndarray, rows: np.ndarray) -> dict[str, float]:
    selected_prediction = prediction[rows]
    selected_target = target[rows]
    error = float(np.sum((selected_prediction - selected_target) ** 2))
    denominator = float(
        np.sum((selected_target - selected_target.mean(axis=0)) ** 2)
    )
    return {
        "mse": float(np.mean((selected_prediction - selected_target) ** 2)),
        "r2": float(1.0 - error / denominator),
    }


def _metrics_by_horizon(
    prediction: np.ndarray,
    target: np.ndarray,
    rows: np.ndarray,
    horizons_ps: np.ndarray,
) -> dict[str, dict[str, float]]:
    horizon_count = int(horizons_ps.size)
    mode_count = int(target.shape[1] // horizon_count)
    prediction_blocks = prediction.reshape(-1, horizon_count, mode_count)
    target_blocks = target.reshape(-1, horizon_count, mode_count)
    return {
        f"{float(horizon):g}ps": _metrics(
            prediction_blocks[:, index], target_blocks[:, index], rows
        )
        for index, horizon in enumerate(horizons_ps.tolist())
    }


def fit_ridge_feature_variants(
    feature_variants: Mapping[str, np.ndarray],
    targets: DynamicTargetData,
    *,
    ridge_alphas: Sequence[float],
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    fitted_parameters: dict[str, dict[str, np.ndarray]] = {}
    for name, values in feature_variants.items():
        standardized, mean, scale = _standardize_features(
            np.asarray(values, dtype=np.float64), targets.split_rows["optimization"]
        )
        models: dict[float, Ridge] = {}
        selection_mse: dict[str, float] = {}
        for raw_alpha in ridge_alphas:
            alpha = float(raw_alpha)
            model = Ridge(alpha=alpha).fit(
                standardized[targets.split_rows["optimization"]],
                targets.target_modes[targets.split_rows["optimization"]],
            )
            selection_prediction = model.predict(
                standardized[targets.split_rows["selection"]]
            )
            selection_mse[f"{alpha:g}"] = float(
                np.mean(
                    (
                        selection_prediction
                        - targets.target_modes[targets.split_rows["selection"]]
                    )
                    ** 2
                )
            )
            models[alpha] = model
        best_alpha = min(
            models, key=lambda value: (selection_mse[f"{value:g}"], value)
        )
        prediction = models[best_alpha].predict(standardized)
        fitted_parameters[name] = {
            "feature_mean": mean,
            "feature_scale": scale,
            "coefficient": np.asarray(models[best_alpha].coef_),
            "intercept": np.asarray(models[best_alpha].intercept_),
            "alpha": np.asarray(best_alpha),
        }
        predictions[name] = prediction.astype(np.float32, copy=False)
        metrics[name] = {
            "selected_alpha": float(best_alpha),
            "selection_mse_by_alpha": selection_mse,
            "selection": _metrics(
                prediction,
                targets.target_modes,
                targets.split_rows["selection"],
            ),
            "validation": _metrics(
                prediction,
                targets.target_modes,
                targets.split_rows["validation"],
            ),
            "by_horizon": {
                split: _metrics_by_horizon(
                    prediction,
                    targets.target_modes,
                    targets.split_rows[split],
                    targets.selected_horizons_ps,
                )
                for split in ("selection", "validation")
            },
        }
    return predictions, metrics, fitted_parameters


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def fit_multiscale_mlp(
    multiscale_features: np.ndarray,
    targets: DynamicTargetData,
    *,
    device: str,
    hidden_dim: int,
    representation_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    maximum_epochs: int,
    patience: int,
    seeds: Sequence[int],
) -> FittedMultiscalePredictor:
    standardized, feature_mean, feature_scale = _standardize_features(
        np.asarray(multiscale_features, dtype=np.float64),
        targets.split_rows["optimization"],
    )
    torch_device = torch.device(device)
    inputs = torch.from_numpy(standardized).to(torch_device)
    target_tensor = torch.from_numpy(targets.target_modes).to(torch_device)
    optimization_rows = torch.from_numpy(targets.split_rows["optimization"]).to(
        torch_device
    )
    selection_rows = torch.from_numpy(targets.split_rows["selection"]).to(torch_device)
    histories: dict[int, dict[str, list[float]]] = {}
    seed_metrics: dict[int, dict[str, float | int]] = {}
    models: dict[int, MultiscaleFuturePredictor] = {}
    predictions: dict[int, np.ndarray] = {}
    representations: dict[int, np.ndarray] = {}
    for raw_seed in seeds:
        seed = int(raw_seed)
        _seed_everything(seed)
        model = MultiscaleFuturePredictor(
            input_dim=int(standardized.shape[1]),
            hidden_dim=int(hidden_dim),
            representation_dim=int(representation_dim),
            target_dim=int(targets.target_modes.shape[1]),
            dropout=float(dropout),
        ).to(torch_device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        history = {"optimization": [], "selection": []}
        best_selection = float("inf")
        best_epoch = -1
        best_state: dict[str, torch.Tensor] | None = None
        for epoch in range(int(maximum_epochs)):
            permutation = torch.randperm(
                optimization_rows.numel(), generator=generator
            ).to(torch_device)
            model.train()
            accumulated = 0.0
            for start in range(0, int(permutation.numel()), int(batch_size)):
                rows = optimization_rows[
                    permutation[start : start + int(batch_size)]
                ]
                _, prediction = model(inputs[rows])
                loss = torch.mean((prediction - target_tensor[rows]) ** 2)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                accumulated += float(loss.detach()) * int(rows.numel())
            optimization_loss = accumulated / float(permutation.numel())
            model.eval()
            with torch.no_grad():
                _, selection_prediction = model(inputs[selection_rows])
                selection_loss = float(
                    torch.mean(
                        (selection_prediction - target_tensor[selection_rows]) ** 2
                    )
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
            raise RuntimeError(f"Multiscale MLP seed {seed} produced no checkpoint.")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            representation, prediction = model(inputs)
        prediction_array = prediction.cpu().numpy()
        representation_array = representation.cpu().numpy()
        histories[seed] = history
        seed_metrics[seed] = {
            "best_epoch": int(best_epoch),
            "epochs_run": int(len(history["selection"])),
            "selection": _metrics(
                prediction_array,
                targets.target_modes,
                targets.split_rows["selection"],
            ),
            "validation": _metrics(
                prediction_array,
                targets.target_modes,
                targets.split_rows["validation"],
            ),
            "validation_by_horizon": _metrics_by_horizon(
                prediction_array,
                targets.target_modes,
                targets.split_rows["validation"],
                targets.selected_horizons_ps,
            ),
        }
        predictions[seed] = prediction_array
        representations[seed] = representation_array
        models[seed] = model.cpu()
        print(
            f"[shooting-multiscale] seed={seed} best_epoch={best_epoch} "
            f"selection_mse={best_selection:.6f} "
            f"validation_r2={seed_metrics[seed]['validation']['r2']:.6f}",
            flush=True,
        )
    selected_seed = min(
        seed_metrics,
        key=lambda value: (
            float(seed_metrics[value]["selection"]["mse"]),
            int(value),
        ),
    )
    return FittedMultiscalePredictor(
        model=models[selected_seed],
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        seed=selected_seed,
        histories=histories,
        seed_metrics=seed_metrics,
        predictions_by_seed=predictions,
        representations_by_seed=representations,
    )


def evaluate_multiscale_ablation(
    base_cache: ShootingEmbeddingCache,
    feature_variants: Mapping[str, np.ndarray],
    targets: DynamicTargetData,
    ridge_predictions: Mapping[str, np.ndarray],
    fitted: FittedMultiscalePredictor,
    *,
    static_pca_dim: int,
    neighbors: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    optimization_rows = targets.split_rows["optimization"]
    static_spaces: dict[str, np.ndarray] = {}
    for name in ("local", "old_mean_std_8", "mean_std_context", "multiscale_context"):
        values = np.asarray(feature_variants[name], dtype=np.float64)
        pca = CovariancePCA.fit(
            values[optimization_rows], dimension=int(static_pca_dim)
        )
        static_spaces[f"{name}_pca_{int(static_pca_dim)}d"] = pca.transform(
            values, dimension=int(static_pca_dim)
        )

    selected_prediction = fitted.predictions_by_seed[fitted.seed]
    selected_representation = fitted.representations_by_seed[fitted.seed]
    horizon_count = int(targets.selected_horizons_ps.size)
    mode_count = int(targets.target_modes.shape[1] // horizon_count)
    prediction_blocks = {
        name: np.asarray(values).reshape(-1, horizon_count, mode_count)
        for name, values in ridge_predictions.items()
    }
    mlp_blocks = selected_prediction.reshape(-1, horizon_count, mode_count)
    validation_parents = targets.parent_splits["validation"]
    retrieval: dict[str, Any] = {}
    baseline_name = f"local_pca_{int(static_pca_dim)}d"
    for local_horizon_index, horizon in enumerate(
        targets.selected_horizons_ps.tolist()
    ):
        spaces = dict(static_spaces)
        for name, values in prediction_blocks.items():
            spaces[f"ridge_{name}_predicted_change"] = values[
                :, local_horizon_index
            ]
        spaces["multiscale_mlp_representation"] = selected_representation
        spaces["multiscale_mlp_predicted_change"] = mlp_blocks[
            :, local_horizon_index
        ]
        horizon_metrics = _future_neighbor_metrics(
            spaces,
            targets.mean_delta[:, local_horizon_index],
            base_cache,
            validation_parents,
            neighbors=int(neighbors),
            seed=int(seed),
        )
        baseline_distance = float(
            horizon_metrics[baseline_name]["mean_ensemble_future_distance"]
        )
        horizon_metrics["gain_over_local_pca_percent"] = {
            name: float(
                100.0
                * (
                    1.0
                    - float(values["mean_ensemble_future_distance"])
                    / baseline_distance
                )
            )
            for name, values in horizon_metrics.items()
        }
        retrieval[f"{float(horizon):g}ps"] = horizon_metrics

    combined_spaces = dict(static_spaces)
    for name, values in ridge_predictions.items():
        combined_spaces[f"ridge_{name}_predicted_change"] = values
    combined_spaces["multiscale_mlp_representation"] = selected_representation
    combined_spaces["multiscale_mlp_predicted_change"] = selected_prediction
    combined_metrics = _future_neighbor_metrics(
        combined_spaces,
        targets.mean_delta.reshape(targets.mean_delta.shape[0], -1),
        base_cache,
        validation_parents,
        neighbors=int(neighbors),
        seed=int(seed),
    )
    baseline_distance = float(
        combined_metrics[baseline_name]["mean_ensemble_future_distance"]
    )
    combined_metrics["gain_over_local_pca_percent"] = {
        name: float(
            100.0
            * (
                1.0
                - float(values["mean_ensemble_future_distance"])
                / baseline_distance
            )
        )
        for name, values in combined_metrics.items()
    }
    retrieval["all_horizons"] = combined_metrics
    metrics = {
        "scientific_contract": {
            "ablation": 1,
            "encoder": "frozen GeoFrameTransformerV2",
            "target": "sibling-mean future embedding change",
            "training_horizons_ps": targets.selected_horizons_ps.tolist(),
            "excluded_horizons": "48 ps excluded from training as pre-registered",
            "query_split": "validation source runs only",
            "candidate_filter": "different source run, exact temperature, exact parent phase",
            "context_model": "radial multiscale mean/std only; no attention or angular layout",
        },
        "selected_mlp_seed": int(fitted.seed),
        "mlp_seed_metrics": {str(key): value for key, value in fitted.seed_metrics.items()},
        "future_neighbor_consistency": retrieval,
        "target_diagnostics": targets.diagnostics,
    }
    arrays = {
        **{f"feature_{name}": values for name, values in feature_variants.items()},
        **{f"ridge_prediction_{name}": values for name, values in ridge_predictions.items()},
        "mlp_prediction": selected_prediction,
        "mlp_representation": selected_representation,
        "standardized_target_modes": targets.target_modes,
        "mean_delta": targets.mean_delta,
    }
    return metrics, arrays


def save_multiscale_predictor(
    fitted: FittedMultiscalePredictor,
    targets: DynamicTargetData,
    path: str | Path,
) -> None:
    target = Path(path)
    torch.save(
        {
            "state_dict": fitted.model.state_dict(),
            "input_dim": fitted.model.input_dim,
            "hidden_dim": fitted.model.hidden_dim,
            "representation_dim": fitted.model.representation_dim,
            "target_dim": fitted.model.target_dim,
            "dropout": fitted.model.dropout,
            "seed": fitted.seed,
        },
        target,
    )
    np.savez(
        target.with_suffix(".preprocessing.npz"),
        feature_mean=fitted.feature_mean,
        feature_scale=fitted.feature_scale,
        target_mean=targets.target_mean,
        target_scale=targets.target_scale,
        selected_horizons_ps=targets.selected_horizons_ps,
        target_pca_means=np.stack([pca.mean_ for pca in targets.target_pcas]),
        target_pca_components=np.stack(
            [pca.components_ for pca in targets.target_pcas]
        ),
        target_pca_eigenvalues=np.stack(
            [pca.eigenvalues_ for pca in targets.target_pcas]
        ),
    )


def plot_multiscale_training(
    fitted: FittedMultiscalePredictor, path: str | Path
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for seed, history in fitted.histories.items():
        ax.plot(history["selection"], label=f"seed {seed}")
    ax.set(xlabel="epoch", ylabel="selection future-change MSE", yscale="log")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(path), dpi=180)
    plt.close(fig)


def plot_multiscale_retrieval(
    metrics: Mapping[str, Mapping[str, Mapping[str, float | int]]],
    path: str | Path,
) -> None:
    horizons = list(metrics)
    preferred = [
        name
        for name in (
            "local_pca_32d",
            "old_mean_std_8_pca_32d",
            "mean_std_context_pca_32d",
            "multiscale_context_pca_32d",
            "ridge_multiscale_context_predicted_change",
            "multiscale_mlp_representation",
            "multiscale_mlp_predicted_change",
            "spatial_transformer_representation",
            "spatial_transformer_predicted_change",
            "distributional_transformer_representation",
            "predicted_kernel_mean",
        )
        if name in metrics[horizons[0]]
    ]
    x = np.arange(len(horizons), dtype=np.float64)
    width = 0.84 / len(preferred)
    fig, ax = plt.subplots(figsize=(max(8.5, 1.5 * len(horizons)), 5.0))
    for index, name in enumerate(preferred):
        values = [
            float(metrics[horizon][name]["distance_over_matched_random"])
            for horizon in horizons
        ]
        ax.bar(
            x + (index - (len(preferred) - 1) / 2.0) * width,
            values,
            width,
            label=name,
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set(
        xticks=x,
        xticklabels=horizons,
        ylabel="future-change distance / matched random",
        xlabel="prediction horizon",
    )
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(Path(path), dpi=180)
    plt.close(fig)


def write_json(path: str | Path, value: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


__all__ = [
    "DynamicTargetData",
    "FittedMultiscalePredictor",
    "MultiscaleFuturePredictor",
    "build_multiscale_feature_variants",
    "evaluate_multiscale_ablation",
    "fit_multiscale_mlp",
    "fit_ridge_feature_variants",
    "plot_multiscale_retrieval",
    "plot_multiscale_training",
    "prepare_dynamic_target_data",
    "save_multiscale_predictor",
    "write_json",
]
