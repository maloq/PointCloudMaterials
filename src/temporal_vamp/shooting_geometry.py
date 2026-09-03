from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from src.temporal_vamp.shooting_distribution import DistributionalTargetData
from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache
from src.temporal_vamp.shooting_multiscale import _metrics, _metrics_by_horizon
from src.temporal_vamp.shooting_spatial import (
    FittedSpatialContextPredictor,
    SpatialContextTransformer,
    SpatialTokenData,
    _standardize_tokens,
)


@dataclass(frozen=True)
class GeometryTeacher:
    probability: torch.Tensor
    valid_mask: torch.Tensor
    temperature: float
    minimum_candidates: int


def _optimization_neighbor_mask(
    cache: ShootingEmbeddingCache,
    optimization_rows: np.ndarray,
) -> np.ndarray:
    center_count = int(cache.parent_z.shape[1])
    parents = cache.manifest["snapshot"]["parents"]
    parent_for_row = optimization_rows // center_count
    source_run = np.asarray(
        [str(parents[index]["source_run_id"]) for index in parent_for_row.tolist()]
    )
    temperature = np.asarray(
        [float(parents[index]["temperature_K"]) for index in parent_for_row.tolist()]
    )
    phase = np.asarray(
        [str(parents[index]["phase"]) for index in parent_for_row.tolist()]
    )
    mask = (
        (source_run[:, None] != source_run[None, :])
        & (temperature[:, None] == temperature[None, :])
        & (phase[:, None] == phase[None, :])
    )
    candidate_counts = mask.sum(axis=1)
    if np.any(candidate_counts == 0):
        row = int(np.flatnonzero(candidate_counts == 0)[0])
        raise RuntimeError(
            "Future-neighbour distillation found no cross-source candidate for an "
            f"optimization row: local_row={row}, parent_index={parent_for_row[row]}, "
            f"temperature={temperature[row]}, phase={phase[row]}."
        )
    return mask


def build_geometry_teacher(
    cache: ShootingEmbeddingCache,
    targets: DistributionalTargetData,
    *,
    device: str,
    temperature_scale: float,
) -> GeometryTeacher:
    optimization_rows = targets.split_rows["optimization"]
    mask_numpy = _optimization_neighbor_mask(cache, optimization_rows)
    torch_device = torch.device(device)
    signature = torch.from_numpy(
        targets.distribution_signature.reshape(
            targets.distribution_signature.shape[0], -1
        )[optimization_rows].astype(np.float32)
    ).to(torch_device)
    valid_mask = torch.from_numpy(mask_numpy).to(torch_device)
    with torch.no_grad():
        squared_distance = torch.cdist(signature, signature).square()
        median_distance = torch.median(squared_distance[valid_mask])
        temperature = float(median_distance * float(temperature_scale))
        if temperature <= 1.0e-10:
            raise RuntimeError(
                "The distributional geometry teacher has a non-positive softmax "
                f"temperature: median_squared_distance={float(median_distance)}, "
                f"scale={temperature_scale}."
            )
        logits = -squared_distance / temperature
        logits = logits.masked_fill(~valid_mask, -torch.inf)
        probability = torch.softmax(logits, dim=1)
    return GeometryTeacher(
        probability=probability,
        valid_mask=valid_mask,
        temperature=temperature,
        minimum_candidates=int(mask_numpy.sum(axis=1).min()),
    )


def _neighbor_kl_loss(
    representation: torch.Tensor,
    teacher: GeometryTeacher,
    *,
    student_temperature: torch.Tensor,
) -> torch.Tensor:
    squared_distance = torch.cdist(representation, representation).square()
    logits = (-squared_distance / student_temperature).masked_fill(
        ~teacher.valid_mask, -torch.inf
    )
    log_probability = torch.log_softmax(logits, dim=1)
    log_probability = torch.where(
        teacher.valid_mask, log_probability, torch.zeros_like(log_probability)
    )
    teacher_probability = teacher.probability
    teacher_log_probability = torch.where(
        teacher_probability > 0.0,
        torch.log(teacher_probability.clamp_min(1.0e-30)),
        torch.zeros_like(teacher_probability),
    )
    kl = torch.sum(
        teacher_probability * (teacher_log_probability - log_probability), dim=1
    ).mean()
    return kl


def _vicreg_losses(representation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centered = representation - representation.mean(dim=0)
    standard_deviation = torch.sqrt(
        representation.var(dim=0, unbiased=False) + 1.0e-4
    )
    variance_loss = torch.relu(1.0 - standard_deviation).mean()
    covariance = centered.T @ centered / float(max(1, representation.shape[0] - 1))
    diagonal = torch.diagonal(covariance)
    covariance_loss = (
        covariance.square().sum() - diagonal.square().sum()
    ) / float(representation.shape[1])
    return variance_loss, covariance_loss


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def fit_distributional_geometry_transformer(
    cache: ShootingEmbeddingCache,
    tokens: SpatialTokenData,
    targets: DistributionalTargetData,
    *,
    device: str,
    hidden_dim: int,
    heads: int,
    blocks: int,
    rbf_dim: int,
    maximum_radius: float,
    representation_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    maximum_epochs: int,
    patience: int,
    seeds: Sequence[int],
    prediction_weight: float,
    neighbor_kl_weight: float,
    teacher_temperature_scale: float,
    student_temperature_scale: float,
    variance_weight: float,
    covariance_weight: float,
    initial_state_dict: dict[str, torch.Tensor] | None = None,
) -> tuple[FittedSpatialContextPredictor, dict[str, Any]]:
    (
        standardized,
        embedding_mean,
        embedding_scale,
        descriptor_mean,
        descriptor_scale,
    ) = _standardize_tokens(tokens, targets.split_rows["optimization"])
    torch_device = torch.device(device)
    embeddings = torch.from_numpy(standardized.embeddings).to(torch_device)
    descriptors = torch.from_numpy(standardized.descriptors).to(torch_device)
    offsets = torch.from_numpy(standardized.offsets).to(torch_device)
    target_tensor = torch.from_numpy(targets.target_modes).to(torch_device)
    optimization_rows = torch.from_numpy(targets.split_rows["optimization"]).to(
        torch_device
    )
    selection_rows = torch.from_numpy(targets.split_rows["selection"]).to(torch_device)
    teacher = build_geometry_teacher(
        cache,
        targets,
        device=device,
        temperature_scale=float(teacher_temperature_scale),
    )
    histories: dict[int, dict[str, list[float]]] = {}
    seed_metrics: dict[int, dict[str, Any]] = {}
    predictions: dict[int, np.ndarray] = {}
    representations: dict[int, np.ndarray] = {}
    models: dict[int, SpatialContextTransformer] = {}
    for raw_seed in seeds:
        seed = int(raw_seed)
        _seed_everything(seed)
        model = SpatialContextTransformer(
            embedding_dim=int(embeddings.shape[-1]),
            descriptor_dim=int(descriptors.shape[-1]),
            hidden_dim=int(hidden_dim),
            heads=int(heads),
            blocks=int(blocks),
            rbf_dim=int(rbf_dim),
            maximum_radius=float(maximum_radius),
            representation_dim=int(representation_dim),
            target_dim=int(targets.target_modes.shape[1]),
            dropout=float(dropout),
        ).to(torch_device)
        if initial_state_dict is not None:
            model.load_state_dict(initial_state_dict, strict=True)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        history = {
            "optimization": [],
            "selection": [],
            "prediction": [],
            "neighbor_kl": [],
            "variance": [],
            "covariance": [],
        }
        model.eval()
        with torch.no_grad():
            initial_representation, _ = model(
                embeddings[optimization_rows],
                descriptors[optimization_rows],
                offsets[optimization_rows],
            )
            initial_squared_distance = torch.cdist(
                initial_representation, initial_representation
            ).square()
            student_temperature = (
                torch.median(initial_squared_distance[teacher.valid_mask])
                * float(student_temperature_scale)
            ).clamp_min(1.0e-4)
            _, initial_selection_prediction = model(
                embeddings[selection_rows],
                descriptors[selection_rows],
                offsets[selection_rows],
            )
            best_selection = float(
                torch.mean(
                    (
                        initial_selection_prediction
                        - target_tensor[selection_rows]
                    )
                    ** 2
                )
            )
        best_epoch = -1
        best_state: dict[str, torch.Tensor] | None = copy.deepcopy(model.state_dict())
        for epoch in range(int(maximum_epochs)):
            model.train()
            representation, prediction = model(
                embeddings[optimization_rows],
                descriptors[optimization_rows],
                offsets[optimization_rows],
            )
            prediction_loss = torch.mean(
                (prediction - target_tensor[optimization_rows]) ** 2
            )
            neighbor_loss = _neighbor_kl_loss(
                representation,
                teacher,
                student_temperature=student_temperature,
            )
            variance_loss, covariance_loss = _vicreg_losses(representation)
            loss = (
                float(prediction_weight) * prediction_loss
                + float(neighbor_kl_weight) * neighbor_loss
                + float(variance_weight) * variance_loss
                + float(covariance_weight) * covariance_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                _, selection_prediction = model(
                    embeddings[selection_rows],
                    descriptors[selection_rows],
                    offsets[selection_rows],
                )
                selection_loss = float(
                    torch.mean(
                        (selection_prediction - target_tensor[selection_rows]) ** 2
                    )
                )
            history["optimization"].append(float(loss.detach()))
            history["selection"].append(selection_loss)
            history["prediction"].append(float(prediction_loss.detach()))
            history["neighbor_kl"].append(float(neighbor_loss.detach()))
            history["variance"].append(float(variance_loss.detach()))
            history["covariance"].append(float(covariance_loss.detach()))
            if selection_loss < best_selection:
                best_selection = selection_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            if epoch - best_epoch >= int(patience):
                break
        if best_state is None:
            raise RuntimeError(
                f"Distributional geometry seed {seed} produced no checkpoint."
            )
        model.load_state_dict(best_state)
        model.eval()
        all_predictions: list[np.ndarray] = []
        all_representations: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, int(embeddings.shape[0]), 256):
                representation, prediction = model(
                    embeddings[start : start + 256],
                    descriptors[start : start + 256],
                    offsets[start : start + 256],
                )
                all_predictions.append(prediction.cpu().numpy())
                all_representations.append(representation.cpu().numpy())
        prediction_array = np.concatenate(all_predictions)
        representation_array = np.concatenate(all_representations)
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
            "final_optimization_losses": {
                name: float(values[-1])
                for name, values in history.items()
                if name not in {"selection"}
            },
            "fixed_student_temperature": float(student_temperature),
            "representation_mean_dimension_std": float(
                representation_array[
                    targets.split_rows["optimization"]
                ].std(axis=0).mean()
            ),
        }
        predictions[seed] = prediction_array
        representations[seed] = representation_array
        models[seed] = model.cpu()
        print(
            f"[shooting-geometry] seed={seed} best_epoch={best_epoch} "
            f"selection_mse={best_selection:.6f} "
            f"validation_r2={seed_metrics[seed]['validation']['r2']:.6f}",
            flush=True,
        )
    selected_seed = min(
        seed_metrics,
        key=lambda value: (
            float(seed_metrics[value]["selection"]["mse"]), int(value)
        ),
    )
    fitted = FittedSpatialContextPredictor(
        model=models[selected_seed],
        embedding_mean=embedding_mean,
        embedding_scale=embedding_scale,
        descriptor_mean=descriptor_mean,
        descriptor_scale=descriptor_scale,
        seed=selected_seed,
        histories=histories,
        seed_metrics=seed_metrics,
        predictions_by_seed=predictions,
        representations_by_seed=representations,
    )
    diagnostics = {
        "teacher_temperature": float(teacher.temperature),
        "teacher_temperature_scale": float(teacher_temperature_scale),
        "student_temperature_scale": float(student_temperature_scale),
        "initialization": (
            "pretrained distributional model"
            if initial_state_dict is not None
            else "random"
        ),
        "minimum_cross_run_matched_candidates": int(teacher.minimum_candidates),
        "loss_weights": {
            "prediction": float(prediction_weight),
            "neighbor_kl": float(neighbor_kl_weight),
            "variance": float(variance_weight),
            "covariance": float(covariance_weight),
        },
    }
    return fitted, diagnostics


__all__ = [
    "GeometryTeacher",
    "build_geometry_teacher",
    "fit_distributional_geometry_transformer",
]
