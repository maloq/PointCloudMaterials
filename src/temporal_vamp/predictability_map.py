from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.temporal_vamp.evaluation import CovariancePCA
from src.temporal_vamp.predictive_atlas import JointPathTargetData
from src.temporal_vamp.shooting_context import ShootingContextTokenCache
from src.temporal_vamp.shooting_distribution import DistributionalTargetData
from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache
from src.temporal_vamp.shooting_spatial import build_spatial_token_data


class DensePredictiveProbe(nn.Module):
    """Controlled nonlinear probe with an explicit predictive bottleneck."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        target_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.target_dim = int(target_dim)
        self.dropout = float(dropout)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.GELU(),
            nn.LayerNorm(self.latent_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.target_dim),
        )

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(values)
        return latent, self.head(latent)


@dataclass(frozen=True)
class FittedDenseProbe:
    model: DensePredictiveProbe
    input_mean: np.ndarray
    input_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    latent: np.ndarray
    prediction: np.ndarray
    selected_seed: int
    seed_metrics: dict[int, dict[str, Any]]
    histories: dict[int, dict[str, list[float]]]

    def transform(
        self,
        values: np.ndarray,
        *,
        device: str,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        inputs = np.asarray(values, dtype=np.float32)
        if inputs.ndim != 2 or inputs.shape[1] != self.model.input_dim:
            raise ValueError(
                "Dense predictive probe input shape changed: "
                f"expected=(*, {self.model.input_dim}), observed={inputs.shape}."
            )
        standardized = ((inputs - self.input_mean) / self.input_scale).astype(
            np.float32
        )
        model = self.model.to(device).eval()
        latent_blocks: list[np.ndarray] = []
        prediction_blocks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, standardized.shape[0], int(batch_size)):
                batch = torch.from_numpy(
                    standardized[start : start + int(batch_size)]
                ).to(device)
                latent, prediction = model(batch)
                latent_blocks.append(latent.cpu().numpy())
                prediction_blocks.append(prediction.cpu().numpy())
        model.cpu()
        latent_values = np.concatenate(latent_blocks).astype(np.float32, copy=False)
        standardized_prediction = np.concatenate(prediction_blocks)
        raw_prediction = (
            standardized_prediction * self.target_scale + self.target_mean
        ).astype(np.float32, copy=False)
        return latent_values, raw_prediction


def load_fitted_probe(path: str | Path) -> FittedDenseProbe:
    checkpoint_path = Path(path).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = DensePredictiveProbe(
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        latent_dim=int(payload["latent_dim"]),
        target_dim=int(payload["target_dim"]),
        dropout=float(payload["dropout"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    empty = np.empty((0, model.latent_dim), dtype=np.float32)
    return FittedDenseProbe(
        model=model,
        input_mean=np.asarray(payload["input_mean"], dtype=np.float32),
        input_scale=np.asarray(payload["input_scale"], dtype=np.float32),
        target_mean=np.asarray(payload["target_mean"], dtype=np.float32),
        target_scale=np.asarray(payload["target_scale"], dtype=np.float32),
        latent=empty,
        prediction=np.empty((0, model.target_dim), dtype=np.float32),
        selected_seed=int(payload["selected_seed"]),
        seed_metrics={int(key): value for key, value in payload["seed_metrics"].items()},
        histories={},
    )


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def regression_metrics(
    prediction: np.ndarray, target: np.ndarray, rows: np.ndarray
) -> dict[str, float]:
    selected_prediction = np.asarray(prediction, dtype=np.float64)[rows]
    selected_target = np.asarray(target, dtype=np.float64)[rows]
    residual = float(np.sum(np.square(selected_prediction - selected_target)))
    centered = selected_target - selected_target.mean(axis=0, keepdims=True)
    denominator = float(np.sum(np.square(centered)))
    if denominator <= 0.0:
        raise RuntimeError(
            f"Regression target has zero variance on {rows.size} selected rows."
        )
    return {
        "mse": float(np.mean(np.square(selected_prediction - selected_target))),
        "r2": float(1.0 - residual / denominator),
    }


def fit_dense_probe(
    inputs: np.ndarray,
    target: np.ndarray,
    split_rows: Mapping[str, np.ndarray],
    *,
    device: str,
    hidden_dim: int,
    latent_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    maximum_epochs: int,
    patience: int,
    seeds: Sequence[int],
) -> FittedDenseProbe:
    values = np.asarray(inputs, dtype=np.float32)
    labels = np.asarray(target, dtype=np.float32)
    if values.ndim != 2 or labels.ndim != 2 or values.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Probe requires aligned 2D arrays, inputs={values.shape}, target={labels.shape}."
        )
    optimization_rows = np.asarray(split_rows["optimization"], dtype=np.int64)
    selection_rows = np.asarray(split_rows["selection"], dtype=np.int64)
    validation_rows = np.asarray(split_rows["validation"], dtype=np.int64)
    input_mean = values[optimization_rows].mean(axis=0, dtype=np.float64)
    input_scale = values[optimization_rows].std(axis=0, dtype=np.float64)
    input_scale = np.where(input_scale <= 1.0e-10, 1.0, input_scale)
    target_mean = labels[optimization_rows].mean(axis=0, dtype=np.float64)
    target_scale = labels[optimization_rows].std(axis=0, dtype=np.float64)
    target_scale = np.where(target_scale <= 1.0e-10, 1.0, target_scale)
    standardized_inputs = ((values - input_mean) / input_scale).astype(np.float32)
    standardized_target = ((labels - target_mean) / target_scale).astype(np.float32)

    torch_device = torch.device(device)
    input_tensor = torch.from_numpy(standardized_inputs).to(torch_device)
    target_tensor = torch.from_numpy(standardized_target).to(torch_device)
    optimization_tensor = torch.from_numpy(optimization_rows).to(torch_device)
    selection_tensor = torch.from_numpy(selection_rows).to(torch_device)
    histories: dict[int, dict[str, list[float]]] = {}
    seed_metrics: dict[int, dict[str, Any]] = {}
    models: dict[int, DensePredictiveProbe] = {}
    predictions: dict[int, np.ndarray] = {}
    latents: dict[int, np.ndarray] = {}

    for raw_seed in seeds:
        seed = int(raw_seed)
        _seed_everything(seed)
        model = DensePredictiveProbe(
            input_dim=values.shape[1],
            hidden_dim=int(hidden_dim),
            latent_dim=int(latent_dim),
            target_dim=labels.shape[1],
            dropout=float(dropout),
        ).to(torch_device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        best_selection = math.inf
        best_epoch = -1
        best_state: dict[str, torch.Tensor] | None = None
        history = {"optimization": [], "selection": []}
        for epoch in range(int(maximum_epochs)):
            permutation = torch.randperm(
                optimization_tensor.numel(), generator=generator
            ).to(torch_device)
            model.train()
            accumulated = 0.0
            for start in range(0, permutation.numel(), int(batch_size)):
                rows = optimization_tensor[
                    permutation[start : start + int(batch_size)]
                ]
                _, prediction = model(input_tensor[rows])
                loss = torch.mean(torch.square(prediction - target_tensor[rows]))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                accumulated += float(loss.detach()) * int(rows.numel())
            model.eval()
            with torch.inference_mode():
                _, selection_prediction = model(input_tensor[selection_tensor])
                selection_loss = float(
                    torch.mean(
                        torch.square(
                            selection_prediction - target_tensor[selection_tensor]
                        )
                    )
                )
            history["optimization"].append(
                accumulated / float(optimization_tensor.numel())
            )
            history["selection"].append(selection_loss)
            if selection_loss < best_selection:
                best_selection = selection_loss
                best_epoch = int(epoch)
                best_state = copy.deepcopy(model.state_dict())
            if epoch - best_epoch >= int(patience):
                break
        if best_state is None:
            raise RuntimeError(f"Probe seed {seed} did not produce a checkpoint.")
        model.load_state_dict(best_state)
        model.eval()
        latent_blocks: list[np.ndarray] = []
        prediction_blocks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, values.shape[0], int(batch_size)):
                latent, prediction = model(
                    input_tensor[start : start + int(batch_size)]
                )
                latent_blocks.append(latent.cpu().numpy())
                prediction_blocks.append(prediction.cpu().numpy())
        latent_array = np.concatenate(latent_blocks).astype(np.float32, copy=False)
        standardized_prediction = np.concatenate(prediction_blocks)
        raw_prediction = (
            standardized_prediction * target_scale + target_mean
        ).astype(np.float32, copy=False)
        metrics = {
            "best_epoch": best_epoch,
            "epochs_run": len(history["selection"]),
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "selection": regression_metrics(raw_prediction, labels, selection_rows),
            "validation": regression_metrics(raw_prediction, labels, validation_rows),
        }
        histories[seed] = history
        seed_metrics[seed] = metrics
        models[seed] = model.cpu()
        predictions[seed] = raw_prediction
        latents[seed] = latent_array
        print(
            f"[predictability-map] seed={seed} input={values.shape[1]} "
            f"target={labels.shape[1]} best_epoch={best_epoch} "
            f"selection_r2={metrics['selection']['r2']:.6f} "
            f"validation_r2={metrics['validation']['r2']:.6f}",
            flush=True,
        )
    selected_seed = max(
        seed_metrics,
        key=lambda value: (
            float(seed_metrics[value]["selection"]["r2"]),
            -int(value),
        ),
    )
    return FittedDenseProbe(
        model=models[selected_seed],
        input_mean=input_mean.astype(np.float32),
        input_scale=input_scale.astype(np.float32),
        target_mean=target_mean.astype(np.float32),
        target_scale=target_scale.astype(np.float32),
        latent=latents[selected_seed],
        prediction=predictions[selected_seed],
        selected_seed=int(selected_seed),
        seed_metrics=seed_metrics,
        histories=histories,
    )


def save_fitted_probe(fitted: FittedDenseProbe, path: str | Path) -> None:
    model = fitted.model
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "latent_dim": model.latent_dim,
            "target_dim": model.target_dim,
            "dropout": model.dropout,
            "input_mean": fitted.input_mean,
            "input_scale": fitted.input_scale,
            "target_mean": fitted.target_mean,
            "target_scale": fitted.target_scale,
            "selected_seed": fitted.selected_seed,
            "seed_metrics": fitted.seed_metrics,
        },
        Path(path),
    )


def parent_metadata(cache: ShootingEmbeddingCache) -> dict[str, Any]:
    parents = cache.manifest["snapshot"]["parents"]
    campaign_type = str(cache.manifest["snapshot"]["campaign_type"])
    temperature = np.asarray([float(p["temperature_K"]) for p in parents])
    crystallinity = np.asarray(
        [float(p["source_crystalline_fraction"]) for p in parents]
    )
    cluster = np.asarray(
        [float(p["source_largest_crystalline_cluster_atoms"]) for p in parents]
    )
    if campaign_type == "position_conditioned_langevin_nvt_shooting":
        progress = np.asarray([float(p["parent_offset_ps"]) for p in parents])
        progress_label = "time relative to nucleation (ps)"
    elif campaign_type == "fixed_horizon_compatibility_from_nested_first_passage":
        progress = cluster.copy()
        progress_label = "largest crystalline cluster (atoms)"
    else:
        raise ValueError(
            f"Predictability-map metadata does not support campaign_type={campaign_type!r}."
        )
    source_time = np.asarray([float(p["source_frame_time_ps"]) for p in parents])
    phase = np.asarray([str(p["phase"]) for p in parents])
    source = np.asarray([str(p["source_run_id"]) for p in parents])
    center_count = int(cache.parent_local_z.shape[1])
    return {
        "temperature_parent": temperature,
        "crystallinity_parent": crystallinity,
        "cluster_parent": cluster,
        "offset_parent": progress,
        "source_time_parent": source_time,
        "phase_parent": phase,
        "source_parent": source,
        "temperature": np.repeat(temperature, center_count),
        "crystallinity": np.repeat(crystallinity, center_count),
        "cluster": np.repeat(cluster, center_count),
        "offset": np.repeat(progress, center_count),
        "progress_label": progress_label,
        "source_time": np.repeat(source_time, center_count),
        "phase": np.repeat(phase, center_count),
        "source": np.repeat(source, center_count),
    }


def build_input_ladder(
    cache: ShootingEmbeddingCache,
    context_cache: ShootingContextTokenCache,
    history_embeddings: np.ndarray,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    tokens = build_spatial_token_data(cache, context_cache)
    history = np.asarray(history_embeddings, dtype=np.float32)
    parent_count, center_count = cache.parent_local_z.shape[:2]
    expected_history = (
        parent_count,
        center_count,
        history.shape[2],
        tokens.embeddings.shape[1],
        tokens.embeddings.shape[2],
    )
    if history.ndim != 5 or history.shape != expected_history:
        raise ValueError(
            f"History cache shape does not match spatial tokens: "
            f"expected={expected_history}, observed={history.shape}."
        )
    rows = parent_count * center_count
    metadata = parent_metadata(cache)
    global_features = np.stack(
        [
            metadata["temperature"],
            metadata["crystallinity"],
            np.log1p(metadata["cluster"]),
            metadata["offset"],
            metadata["source_time"],
        ],
        axis=1,
    ).astype(np.float32)
    temperature = metadata["temperature"][:, None].astype(np.float32)
    local = np.asarray(cache.parent_local_z, dtype=np.float32).reshape(rows, -1)
    satellites = tokens.embeddings[:, 1:]
    satellite_descriptors = tokens.descriptors[:, 1:]
    radii = np.linalg.norm(tokens.offsets[:, 1:], axis=-1)
    summary = np.concatenate(
        [
            local,
            satellites.mean(axis=1),
            satellites.std(axis=1),
            tokens.descriptors[:, 0],
            satellite_descriptors.mean(axis=1),
            satellite_descriptors.std(axis=1),
            np.stack(
                [radii.mean(1), radii.std(1), radii.min(1), radii.max(1)], axis=1
            ),
        ],
        axis=1,
    ).astype(np.float32)
    spatial_full = np.concatenate(
        [
            tokens.embeddings.reshape(rows, -1),
            tokens.descriptors.reshape(rows, -1),
            tokens.offsets.reshape(rows, -1),
        ],
        axis=1,
    ).astype(np.float32)
    current = tokens.embeddings.reshape(
        parent_count, center_count, 1, tokens.embeddings.shape[1], -1
    )
    history_delta = history - current
    flat_history = history_delta.reshape(rows, -1).astype(np.float32)

    rng = np.random.default_rng(int(seed))
    shuffled_parent = np.arange(parent_count, dtype=np.int64)
    phase = metadata["phase_parent"]
    temperature_parent = metadata["temperature_parent"]
    for temperature_value in np.unique(temperature_parent):
        for phase_value in np.unique(phase):
            group = np.flatnonzero(
                (temperature_parent == temperature_value) & (phase == phase_value)
            )
            if group.size < 2:
                raise RuntimeError(
                    "Matched history shuffle requires at least two parents in every "
                    f"temperature/phase cell; T={temperature_value}, phase={phase_value}, "
                    f"parents={group.tolist()}."
                )
            permutation = np.roll(rng.permutation(group), 1)
            shuffled_parent[group] = permutation
    shuffled_history = history_delta.reshape(parent_count, center_count, -1)[
        shuffled_parent
    ].reshape(rows, -1)

    return {
        "metadata_only": global_features,
        "local": np.concatenate([temperature, local], axis=1),
        "local_global": np.concatenate([global_features, local], axis=1),
        "spatial_summary": np.concatenate([global_features, summary], axis=1),
        "spatial_full": np.concatenate([global_features, spatial_full], axis=1),
        "spatial_history": np.concatenate(
            [global_features, spatial_full, flat_history], axis=1
        ),
        "spatial_shuffled_history": np.concatenate(
            [global_features, spatial_full, shuffled_history], axis=1
        ),
    }


def branch_features_for_horizon(
    cache: ShootingEmbeddingCache,
    targets: DistributionalTargetData,
    horizon_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_horizon = int(targets.selected_horizon_indices[int(horizon_index)])
    parameters = targets.horizon_parameters[int(horizon_index)]
    branch_parent = np.asarray(cache.branch_parent_index, dtype=np.int64)
    current = np.asarray(cache.parent_local_z[branch_parent], dtype=np.float64)
    future = np.asarray(cache.future_z[:, cache_horizon], dtype=np.float64)
    standardized = (future - current - parameters.delta_mean) / parameters.delta_scale
    projected = parameters.pca.transform(
        standardized.reshape(-1, standardized.shape[-1]),
        dimension=parameters.frequencies.shape[1],
    ).reshape(standardized.shape[0], standardized.shape[1], -1)
    blocks = [
        np.sqrt(2.0 / float(parameters.frequencies.shape[2]))
        * np.cos(
            projected @ parameters.frequencies[band] + parameters.phases[band]
        )
        for band in range(parameters.frequencies.shape[0])
    ]
    rff = np.concatenate(blocks, axis=-1).astype(np.float32)
    return projected.astype(np.float32), rff, branch_parent


def aggregate_by_parent(
    values: np.ndarray, branch_parent: np.ndarray, parent_count: int
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.empty((parent_count, *values.shape[1:]), dtype=np.float32)
    variance = np.empty_like(mean)
    for parent_index in range(int(parent_count)):
        selected = np.asarray(values[branch_parent == parent_index], dtype=np.float32)
        mean[parent_index] = selected.mean(axis=0)
        variance[parent_index] = selected.var(axis=0, ddof=1)
    return mean, variance


def split_shot_noise_ceiling(
    projected: np.ndarray,
    rff: np.ndarray,
    branch_parent: np.ndarray,
    validation_parents: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    parent_count = int(branch_parent.max()) + 1
    center_count = int(projected.shape[1])
    validation_rows = np.concatenate(
        [
            np.arange(p * center_count, (p + 1) * center_count, dtype=np.int64)
            for p in validation_parents.tolist()
        ]
    )
    rng = np.random.default_rng(int(seed))
    correlations: dict[str, list[float]] = {
        "mean_future": [],
        "future_law": [],
        "log_variance": [],
    }
    distances: dict[str, list[float]] = {name: [] for name in correlations}
    for _ in range(int(repetitions)):
        partitions: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for parent_index in range(parent_count):
            branches = np.flatnonzero(branch_parent == parent_index)
            if branches.size % 2 != 0:
                raise RuntimeError(
                    f"Noise-ceiling split requires an even shot count, parent="
                    f"{parent_index}, shots={branches.size}."
                )
            shuffled = rng.permutation(branches)
            partitions[parent_index] = (
                shuffled[: branches.size // 2],
                shuffled[branches.size // 2 :],
            )
        split_values: dict[str, list[np.ndarray]] = {
            name: [] for name in correlations
        }
        for half in (0, 1):
            mean_projected = np.empty(
                (parent_count, center_count, projected.shape[-1]), dtype=np.float32
            )
            mean_rff = np.empty(
                (parent_count, center_count, rff.shape[-1]), dtype=np.float32
            )
            variance = np.empty_like(mean_projected)
            for parent_index in range(parent_count):
                selected = partitions[parent_index][half]
                mean_projected[parent_index] = projected[selected].mean(axis=0)
                mean_rff[parent_index] = rff[selected].mean(axis=0)
                variance[parent_index] = projected[selected].var(axis=0, ddof=1)
            split_values["mean_future"].append(
                mean_projected.reshape(-1, projected.shape[-1])[validation_rows]
            )
            split_values["future_law"].append(
                mean_rff.reshape(-1, rff.shape[-1])[validation_rows]
            )
            split_values["log_variance"].append(
                np.log(variance.reshape(-1, projected.shape[-1])[validation_rows] + 1e-6)
            )
        for name, halves in split_values.items():
            left, right = halves
            pooled = 0.5 * (left + right)
            component_mean = pooled.mean(axis=0, dtype=np.float64)
            component_scale = pooled.std(axis=0, dtype=np.float64)
            active = component_scale > 1.0e-8
            if not np.any(active):
                raise RuntimeError(
                    f"Split-shot target {name!r} has no state-dependent component."
                )
            standardized_left = (
                left[:, active] - component_mean[active]
            ) / component_scale[active]
            standardized_right = (
                right[:, active] - component_mean[active]
            ) / component_scale[active]
            correlations[name].append(
                float(
                    np.corrcoef(
                        standardized_left.reshape(-1),
                        standardized_right.reshape(-1),
                    )[0, 1]
                )
            )
            distances[name].append(float(np.linalg.norm(left - right, axis=1).mean()))
    output: dict[str, Any] = {}
    for name in correlations:
        values = np.asarray(correlations[name], dtype=np.float64)
        full_reliability = 2.0 * values / (1.0 + values)
        output[name] = {
            "half_shot_correlation_mean": float(values.mean()),
            "half_shot_correlation_ci95": np.quantile(values, [0.025, 0.975]).tolist(),
            "estimated_full_shot_reliability_mean": float(full_reliability.mean()),
            "estimated_full_shot_reliability_ci95": np.quantile(
                full_reliability, [0.025, 0.975]
            ).tolist(),
            "half_shot_mean_distance": float(np.mean(distances[name])),
            "repetitions": int(repetitions),
        }
    return output


def source_bootstrap_r2(
    prediction: np.ndarray,
    target: np.ndarray,
    validation_rows: np.ndarray,
    source_by_row: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rows = np.asarray(validation_rows, dtype=np.int64)
    sources = np.unique(source_by_row[rows])
    if sources.size < 2:
        raise RuntimeError(
            f"Source bootstrap requires at least two validation sources, got {sources}."
        )
    statistics: dict[str, tuple[int, float, np.ndarray, float]] = {}
    prediction_values = np.asarray(prediction, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    for source in sources.tolist():
        selected = rows[source_by_row[rows] == source]
        source_target = target_values[selected]
        statistics[str(source)] = (
            int(selected.size),
            float(np.sum(np.square(prediction_values[selected] - source_target))),
            source_target.sum(axis=0),
            float(np.sum(np.square(source_target))),
        )
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        sampled = rng.choice(sources, size=sources.size, replace=True)
        count = 0
        residual = 0.0
        target_sum = np.zeros(target_values.shape[1], dtype=np.float64)
        target_square_sum = 0.0
        for source in sampled.tolist():
            source_count, source_residual, source_sum, source_square_sum = statistics[
                str(source)
            ]
            count += source_count
            residual += source_residual
            target_sum += source_sum
            target_square_sum += source_square_sum
        denominator = target_square_sum - float(np.sum(np.square(target_sum))) / float(
            count
        )
        if denominator <= 0.0:
            raise RuntimeError(
                "Source-bootstrap resample has zero target variance: "
                f"sampled_sources={sampled.tolist()}."
            )
        values[index] = 1.0 - residual / denominator
    return {
        "validation_sources": int(sources.size),
        "samples": int(samples),
        "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
        "probability_positive": float(np.mean(values > 0.0)),
    }


def plot_noise_ceiling(
    metrics: Mapping[str, Mapping[str, Any]],
    horizons: Sequence[float],
    path: Path,
    *,
    shot_count: int,
) -> None:
    names = ["mean_future", "future_law", "log_variance"]
    labels = ["Mean future", "Future law", "Log variance"]
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for name, label in zip(names, labels, strict=True):
        values = [
            float(metrics[f"{float(h):g}ps"][name]["estimated_full_shot_reliability_mean"])
            for h in horizons
        ]
        axis.plot(horizons, values, marker="o", linewidth=2, label=label)
    axis.set_xlabel("forecast horizon (ps)")
    axis.set_ylabel(f"estimated {int(shot_count)}-shot reliability")
    axis.set_ylim(-0.05, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_predictability_heatmap(
    model_scores: Mapping[str, Mapping[str, float]],
    references: Mapping[str, Mapping[str, float]],
    horizons: Sequence[float],
    path: Path,
) -> None:
    targets = ["individual_future", "mean_future", "future_law", "log_variance"]
    labels = ["Individual future", "Conditional mean", "Future law", "Log variance"]
    model = np.asarray(
        [[model_scores[f"{float(h):g}ps"][name] for h in horizons] for name in targets]
    )
    reference = np.asarray(
        [[references[f"{float(h):g}ps"][name] for h in horizons] for name in targets]
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharey=True)
    for axis, values, title in zip(
        axes,
        [model, reference],
        ["Held-out model $R^2$", "Finite-shot reference / reliability"],
        strict=True,
    ):
        image = axis.imshow(values, vmin=-0.1, vmax=1.0, cmap="viridis", aspect="auto")
        axis.set_xticks(range(len(horizons)), [f"{h:g}" for h in horizons])
        axis.set_xlabel("horizon (ps)")
        axis.set_title(title)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                color = "white" if values[row, column] < 0.35 else "black"
                axis.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center", color=color, fontsize=8)
    axes[0].set_yticks(range(len(labels)), labels)
    figure.colorbar(image, ax=axes, fraction=0.025, pad=0.03)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_input_ladder(metrics: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    names = list(metrics)
    values = [float(metrics[name]["validation"]["r2"]) for name in names]
    low = [float(metrics[name]["bootstrap"]["ci95"][0]) for name in names]
    high = [float(metrics[name]["bootstrap"]["ci95"][1]) for name in names]
    y = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.barh(y, values, color="#4477aa")
    axis.errorbar(
        values,
        y,
        xerr=[np.asarray(values) - low, np.asarray(high) - values],
        fmt="none",
        color="black",
        capsize=3,
    )
    axis.set_yticks(y, [name.replace("_", " ") for name in names])
    axis.set_xlabel("held-out joint future-law $R^2$")
    axis.grid(axis="x", alpha=0.25)
    axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_atlas_metadata(
    coordinates_path: Path,
    metadata: Mapping[str, np.ndarray],
    path: Path,
) -> None:
    with np.load(coordinates_path, allow_pickle=False) as payload:
        coordinates = np.asarray(payload["atlas_latent_pca2"], dtype=np.float64)
    panels = [
        (metadata["temperature"], "temperature (K)", "plasma"),
        (metadata["crystallinity"], "crystalline fraction", "viridis"),
        (metadata["offset"], str(metadata["progress_label"]), "coolwarm"),
        (metadata["source_time"], "simulation time (ps)", "cividis"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for axis, (color, title, cmap) in zip(axes.flat, panels, strict=True):
        image = axis.scatter(
            coordinates[:, 0], coordinates[:, 1], c=color, s=5, alpha=0.55, cmap=cmap
        )
        axis.set_title(title)
        axis.set_xlabel("atlas PC1")
        axis.set_ylabel("atlas PC2")
        figure.colorbar(image, ax=axis, fraction=0.045)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def predictive_latent_dynamics(
    cache: ShootingEmbeddingCache,
    local_probe: FittedDenseProbe,
    split_rows: Mapping[str, np.ndarray],
    *,
    device: str,
    batch_size: int,
    horizon_indices: Sequence[int],
) -> dict[str, np.ndarray]:
    metadata = parent_metadata(cache)
    parent_count, center_count = cache.parent_local_z.shape[:2]
    present_input = np.concatenate(
        [
            metadata["temperature"][:, None].astype(np.float32),
            np.asarray(cache.parent_local_z, dtype=np.float32).reshape(
                parent_count * center_count, -1
            ),
        ],
        axis=1,
    )
    present_latent, _ = local_probe.transform(
        present_input, device=device, batch_size=int(batch_size)
    )
    pca = CovariancePCA.fit(
        present_latent[np.asarray(split_rows["optimization"], dtype=np.int64)],
        dimension=2,
    )
    present_2d = pca.transform(present_latent, dimension=2).astype(np.float32)
    branch_parent = np.asarray(cache.branch_parent_index, dtype=np.int64)
    branch_temperature = metadata["temperature_parent"][branch_parent]
    future_2d: list[np.ndarray] = []
    for horizon_index in horizon_indices:
        future = np.asarray(cache.future_z[:, int(horizon_index)], dtype=np.float32)
        future_input = np.concatenate(
            [
                np.repeat(branch_temperature, center_count)[:, None].astype(np.float32),
                future.reshape(-1, future.shape[-1]),
            ],
            axis=1,
        )
        future_latent, _ = local_probe.transform(
            future_input, device=device, batch_size=int(batch_size)
        )
        future_2d.append(
            pca.transform(future_latent, dimension=2)
            .reshape(future.shape[0], center_count, 2)
            .astype(np.float32)
        )
    return {
        "present_2d": present_2d,
        "future_2d": np.stack(future_2d, axis=1),
        "branch_parent_index": branch_parent,
        "pca_mean": pca.mean_,
        "pca_components": pca.components_[:2],
    }


def plot_shooting_fans(
    dynamics: Mapping[str, np.ndarray],
    cache: ShootingEmbeddingCache,
    horizons: Sequence[float],
    path: Path,
) -> list[dict[str, Any]]:
    present = dynamics["present_2d"]
    future = dynamics["future_2d"]
    branch_parent = dynamics["branch_parent_index"]
    parents = cache.manifest["snapshot"]["parents"]
    parent_count, center_count = cache.parent_local_z.shape[:2]
    validation_parents = [
        index
        for index, parent in enumerate(parents)
        if str(parent["source_split"]) in {"validation", "final_validation"}
    ]
    chosen_parents: list[int] = []
    for temperature in (400.0, 450.0, 500.0):
        candidates = [
            p for p in validation_parents if float(parents[p]["temperature_K"]) == temperature
        ]
        if not candidates:
            raise RuntimeError(f"No validation parent is available at {temperature:g} K.")
        chosen_parents.append(candidates[-1])
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    records: list[dict[str, Any]] = []
    atom_ids = np.asarray(cache.atom_ids, dtype=np.int64)
    for axis, parent_index in zip(axes, chosen_parents, strict=True):
        branches = np.flatnonzero(branch_parent == parent_index)
        terminal = future[branches, -1]
        dispersion = np.mean(
            np.sum(np.square(terminal - terminal.mean(axis=0, keepdims=True)), axis=-1),
            axis=0,
        )
        center = int(np.argmax(dispersion))
        start = present[parent_index * center_count + center]
        for local_branch, branch in enumerate(branches.tolist()):
            trajectory = np.concatenate(
                [start[None], future[branch, :, center]], axis=0
            )
            axis.plot(
                trajectory[:, 0], trajectory[:, 1], "-o", linewidth=0.9, markersize=2.8,
                alpha=0.58, color=plt.cm.tab20(local_branch % 20)
            )
        axis.scatter(start[0], start[1], s=70, marker="*", color="black", zorder=5)
        axis.set_title(
            f"{float(parents[parent_index]['temperature_K']):g} K; atom {atom_ids[center]}"
        )
        axis.set_xlabel("predictive latent PC1")
        axis.set_ylabel("predictive latent PC2")
        axis.grid(alpha=0.2)
        records.append(
            {
                "parent_index": int(parent_index),
                "parent_id": str(parents[parent_index]["parent_id"]),
                "temperature_K": float(parents[parent_index]["temperature_K"]),
                "center_index": center,
                "atom_id": int(atom_ids[center]),
                "terminal_latent_dispersion": float(dispersion[center]),
                "branches": int(branches.size),
            }
        )
    figure.suptitle(
        "Shooting fans in local predictive latent (map evaluated at future states)"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return records


def plot_latent_vector_field(
    dynamics: Mapping[str, np.ndarray],
    cache: ShootingEmbeddingCache,
    horizon_position: int,
    validation_rows: np.ndarray,
    path: Path,
) -> None:
    present = np.asarray(dynamics["present_2d"], dtype=np.float64)
    future = np.asarray(dynamics["future_2d"], dtype=np.float64)
    branch_parent = np.asarray(dynamics["branch_parent_index"], dtype=np.int64)
    center_count = int(cache.parent_local_z.shape[1])
    mean_future = np.empty_like(present)
    for parent_index in range(cache.parent_local_z.shape[0]):
        branches = np.flatnonzero(branch_parent == parent_index)
        mean_future[parent_index * center_count : (parent_index + 1) * center_count] = (
            future[branches, int(horizon_position)].mean(axis=0)
        )
    rows = np.asarray(validation_rows, dtype=np.int64)
    x = present[rows]
    delta = mean_future[rows] - x
    bins = 12
    x_edges = np.linspace(np.quantile(x[:, 0], 0.01), np.quantile(x[:, 0], 0.99), bins + 1)
    y_edges = np.linspace(np.quantile(x[:, 1], 0.01), np.quantile(x[:, 1], 0.99), bins + 1)
    centers: list[np.ndarray] = []
    vectors: list[np.ndarray] = []
    counts: list[int] = []
    for ix in range(bins):
        for iy in range(bins):
            selected = (
                (x[:, 0] >= x_edges[ix])
                & (x[:, 0] < x_edges[ix + 1])
                & (x[:, 1] >= y_edges[iy])
                & (x[:, 1] < y_edges[iy + 1])
            )
            if np.count_nonzero(selected) >= 5:
                centers.append(x[selected].mean(axis=0))
                vectors.append(delta[selected].mean(axis=0))
                counts.append(int(np.count_nonzero(selected)))
    centers_array = np.asarray(centers)
    vectors_array = np.asarray(vectors)
    figure, axis = plt.subplots(figsize=(6.5, 5.6))
    metadata = parent_metadata(cache)
    image = axis.scatter(
        x[:, 0], x[:, 1], c=metadata["temperature"][rows], cmap="plasma", s=8, alpha=0.3
    )
    axis.quiver(
        centers_array[:, 0], centers_array[:, 1], vectors_array[:, 0], vectors_array[:, 1],
        np.asarray(counts), cmap="viridis", angles="xy", scale_units="xy", scale=1.0,
        width=0.005
    )
    axis.set_xlabel("predictive latent PC1")
    axis.set_ylabel("predictive latent PC2")
    axis.set_title("Held-out conditional drift in predictive latent space")
    figure.colorbar(image, ax=axis, label="temperature (K)")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def static_predictive_disagreement(
    cache: ShootingEmbeddingCache,
    target: JointPathTargetData,
    prediction: np.ndarray,
    *,
    samples: int,
    seed: int,
    plot_path: Path,
) -> dict[str, Any]:
    local = np.asarray(cache.parent_local_z, dtype=np.float64).reshape(
        -1, cache.parent_local_z.shape[-1]
    )
    pca = CovariancePCA.fit(local[target.split_rows["optimization"]], dimension=32)
    static = pca.transform(local, dimension=32)
    rows = np.asarray(target.split_rows["validation"], dtype=np.int64)
    metadata = parent_metadata(cache)
    rng = np.random.default_rng(int(seed))
    left: list[int] = []
    right: list[int] = []
    attempts = 0
    while len(left) < int(samples) and attempts < int(samples) * 100:
        a, b = rng.choice(rows, size=2, replace=False).tolist()
        attempts += 1
        if (
            metadata["temperature"][a] == metadata["temperature"][b]
            and metadata["phase"][a] == metadata["phase"][b]
            and metadata["source"][a] != metadata["source"][b]
        ):
            left.append(a)
            right.append(b)
    if len(left) != int(samples):
        raise RuntimeError(
            f"Could only sample {len(left)} of {samples} cross-source matched pairs."
        )
    a = np.asarray(left, dtype=np.int64)
    b = np.asarray(right, dtype=np.int64)
    static_distance = np.linalg.norm(static[a] - static[b], axis=1)
    teacher_distance = np.linalg.norm(
        target.empirical_mean_embedding[a] - target.empirical_mean_embedding[b], axis=1
    )
    predicted_distance = np.linalg.norm(prediction[a] - prediction[b], axis=1)
    threshold = float(np.quantile(static_distance, 0.1))
    near = np.flatnonzero(static_distance <= threshold)
    far_future = int(near[np.argmax(teacher_distance[near])])
    close_future = int(near[np.argmin(teacher_distance[near])])
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    axes[0].hexbin(static_distance, teacher_distance, gridsize=45, mincnt=1, cmap="viridis")
    axes[0].scatter(
        static_distance[[close_future, far_future]],
        teacher_distance[[close_future, far_future]],
        c=["white", "red"], edgecolor="black", s=55,
    )
    axes[0].axvline(threshold, linestyle="--", color="gray")
    axes[0].set_xlabel("static GeoFrame PCA distance")
    axes[0].set_ylabel("held-out future-law distance")
    axes[0].set_title("Static similarity does not fix future similarity")
    axes[1].hexbin(predicted_distance, teacher_distance, gridsize=45, mincnt=1, cmap="magma")
    axes[1].set_xlabel("predicted future-law distance")
    axes[1].set_ylabel("held-out future-law distance")
    axes[1].set_title("Predictive distance calibration")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    def record(position: int) -> dict[str, Any]:
        center_count = int(cache.parent_local_z.shape[1])
        parents = cache.manifest["snapshot"]["parents"]
        return {
            "left_parent": str(parents[int(a[position] // center_count)]["parent_id"]),
            "right_parent": str(parents[int(b[position] // center_count)]["parent_id"]),
            "left_atom_id": int(cache.atom_ids[int(a[position] % center_count)]),
            "right_atom_id": int(cache.atom_ids[int(b[position] % center_count)]),
            "static_distance": float(static_distance[position]),
            "teacher_future_law_distance": float(teacher_distance[position]),
            "predicted_future_law_distance": float(predicted_distance[position]),
        }

    return {
        "sampled_pairs": int(samples),
        "static_vs_teacher_spearman": float(spearmanr(static_distance, teacher_distance).statistic),
        "predicted_vs_teacher_spearman": float(spearmanr(predicted_distance, teacher_distance).statistic),
        "static_near_threshold_decile": threshold,
        "static_near_future_close_example": record(close_future),
        "static_near_future_divergent_example": record(far_future),
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


__all__ = [
    "DensePredictiveProbe",
    "FittedDenseProbe",
    "aggregate_by_parent",
    "branch_features_for_horizon",
    "build_input_ladder",
    "fit_dense_probe",
    "json_ready",
    "load_fitted_probe",
    "parent_metadata",
    "plot_atlas_metadata",
    "plot_input_ladder",
    "plot_latent_vector_field",
    "plot_noise_ceiling",
    "plot_predictability_heatmap",
    "plot_shooting_fans",
    "predictive_latent_dynamics",
    "regression_metrics",
    "save_fitted_probe",
    "source_bootstrap_r2",
    "split_shot_noise_ceiling",
    "static_predictive_disagreement",
]
