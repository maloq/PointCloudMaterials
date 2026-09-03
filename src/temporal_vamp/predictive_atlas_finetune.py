"""Last-block present-encoder fine-tuning for the temporal predictive atlas."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from src.temporal_vamp.embeddings import FrozenEncoder
from src.temporal_vamp.predictive_atlas import (
    FittedPredictiveAtlas,
    JointPathTargetData,
    PredictiveAtlas,
    _latent_regularization,
)
from src.temporal_vamp.shooting_encoder_finetune import (
    ShootingGeoFrameActivationCache,
    _encode_last_block,
    _geoframe_v2_encoder,
    configure_last_geoframe_block_trainable,
)
from src.temporal_vamp.shooting_multiscale import _metrics
from src.temporal_vamp.shooting_spatial import SpatialTokenData


@dataclass(frozen=True)
class FittedAtlasEncoderFineTune:
    atlas: FittedPredictiveAtlas
    encoder_state: dict[str, torch.Tensor]
    trainable_encoder_parameter_names: tuple[str, ...]
    trainable_encoder_parameter_count: int
    initial_embedding_max_abs_error: float


def fit_predictive_atlas_last_geoframe_block(
    activation_cache: ShootingGeoFrameActivationCache,
    frozen_encoder: FrozenEncoder,
    tokens: SpatialTokenData,
    history_embeddings: np.ndarray,
    targets: JointPathTargetData,
    conditioning_values: np.ndarray,
    initial_atlas: FittedPredictiveAtlas,
    *,
    device: str,
    atlas_learning_rate: float,
    encoder_learning_rate: float,
    weight_decay: float,
    variance_weight: float,
    covariance_weight: float,
    batch_size: int,
    maximum_epochs: int,
    patience: int,
    gradient_clip_norm: float,
    mixed_precision: bool,
    seeds: Sequence[int],
) -> FittedAtlasEncoderFineTune:
    torch_device = torch.device(device)
    if frozen_encoder.device != torch_device:
        raise RuntimeError(
            f"Fine-tune encoder device={frozen_encoder.device} does not match "
            f"training device={torch_device}."
        )
    parent_count, center_count, activation_token_count = (
        activation_cache.tokens_before_last.shape[:3]
    )
    row_count, spatial_token_count, embedding_dim = tokens.embeddings.shape
    if (
        int(parent_count * center_count) != int(row_count)
        or int(activation_token_count) != 1
    ):
        raise RuntimeError(
            "Central activation cache does not align with atlas rows: "
            f"activations={parent_count}x{center_count}x{activation_token_count}, "
            f"tokens={tokens.embeddings.shape}."
        )
    history = np.asarray(history_embeddings, dtype=np.float32)
    expected_history_shape = (
        row_count,
        initial_atlas.model.history_lag_count,
        spatial_token_count,
        embedding_dim,
    )
    if tuple(history.shape) != expected_history_shape:
        raise RuntimeError(
            f"Fine-tune history shape changed: expected={expected_history_shape}, "
            f"observed={history.shape}."
        )
    if (
        initial_atlas.history_delta_mean is None
        or initial_atlas.history_delta_scale is None
    ):
        raise RuntimeError("Temporal atlas checkpoint has no history preprocessing.")

    standardized_embeddings = (
        (tokens.embeddings - initial_atlas.embedding_mean)
        / initial_atlas.embedding_scale
    ).astype(np.float32)
    standardized_descriptors = (
        (tokens.descriptors - initial_atlas.descriptor_mean)
        / initial_atlas.descriptor_scale
    ).astype(np.float32)
    history_delta = history - tokens.embeddings[:, None, :, :]
    standardized_history = (
        (history_delta - initial_atlas.history_delta_mean[None, :, None, :])
        / initial_atlas.history_delta_scale[None, :, None, :]
    ).astype(np.float32)
    conditioning = np.asarray(conditioning_values, dtype=np.float64)
    standardized_conditioning = (
        (conditioning - initial_atlas.conditioning_mean)
        / initial_atlas.conditioning_scale
    ).astype(np.float32)

    embeddings = torch.from_numpy(standardized_embeddings).to(torch_device)
    descriptors = torch.from_numpy(standardized_descriptors).to(torch_device)
    offsets = torch.from_numpy(np.asarray(tokens.offsets, dtype=np.float32)).to(
        torch_device
    )
    histories_tensor = torch.from_numpy(standardized_history).to(torch_device)
    conditions = torch.from_numpy(standardized_conditioning).to(torch_device)
    target_tensor = torch.from_numpy(targets.target_modes).to(torch_device)
    optimization_rows = np.asarray(targets.split_rows["optimization"], dtype=np.int64)
    selection_rows = np.asarray(targets.split_rows["selection"], dtype=np.int64)

    def activation_batch(
        rows: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        parent_rows = rows // int(center_count)
        center_rows = rows % int(center_count)
        token_values = np.asarray(
            activation_cache.tokens_before_last[parent_rows, center_rows, 0],
            dtype=np.float32,
        )
        bias_values = np.asarray(
            activation_cache.shared_attention_bias[parent_rows, center_rows, 0],
            dtype=np.float32,
        )
        geometry_values = np.asarray(
            activation_cache.value_geometry[parent_rows, center_rows, 0],
            dtype=np.float32,
        )
        return (
            torch.from_numpy(token_values).to(torch_device),
            torch.from_numpy(bias_values).to(torch_device),
            torch.from_numpy(geometry_values).to(torch_device),
        )

    embedding_mean = torch.as_tensor(
        initial_atlas.embedding_mean, device=torch_device, dtype=torch.float32
    )
    embedding_scale = torch.as_tensor(
        initial_atlas.embedding_scale, device=torch_device, dtype=torch.float32
    )

    def model_batch(
        model: PredictiveAtlas, rows: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_values, bias_values, geometry_values = activation_batch(rows)
        central_raw = _encode_last_block(
            frozen_encoder,
            token_values,
            bias_values,
            geometry_values,
        )
        central_standardized = (central_raw - embedding_mean) / embedding_scale
        token_batch = embeddings[rows].clone()
        token_batch[:, 0] = central_standardized
        return model(
            token_batch,
            descriptors[rows],
            offsets[rows],
            conditions[rows],
            histories_tensor[rows],
        )

    encoder = _geoframe_v2_encoder(frozen_encoder)
    base_encoder_state = copy.deepcopy(encoder.state_dict())
    encoder.load_state_dict(base_encoder_state, strict=True)
    configure_last_geoframe_block_trainable(encoder)
    encoder.eval()
    maximum_error = 0.0
    with torch.no_grad():
        for start in range(0, row_count, int(batch_size)):
            rows = np.arange(
                start, min(row_count, start + int(batch_size)), dtype=np.int64
            )
            token_values, bias_values, geometry_values = activation_batch(rows)
            reconstructed = _encode_last_block(
                frozen_encoder, token_values, bias_values, geometry_values
            )
            reference = torch.from_numpy(
                np.asarray(tokens.embeddings[rows, 0], dtype=np.float32)
            ).to(torch_device)
            maximum_error = max(
                maximum_error,
                float(torch.max(torch.abs(reconstructed - reference))),
            )
    if maximum_error > 5.0e-4:
        raise RuntimeError(
            "Central activation cache does not reconstruct the frozen embeddings: "
            f"max_abs_error={maximum_error:.8g}."
        )
    print(
        f"[predictive-atlas-finetune] initial embedding max_abs_error={maximum_error:.8g}",
        flush=True,
    )

    histories: dict[int, dict[str, list[float]]] = {}
    seed_metrics: dict[int, dict[str, Any]] = {}
    predictions_by_seed: dict[int, np.ndarray] = {}
    representations_by_seed: dict[int, np.ndarray] = {}
    atlas_states: dict[int, dict[str, torch.Tensor]] = {}
    encoder_states: dict[int, dict[str, torch.Tensor]] = {}
    trainable_names: tuple[str, ...] | None = None
    use_autocast = torch_device.type == "cuda" and bool(mixed_precision)
    for raw_seed in seeds:
        seed = int(raw_seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        encoder.load_state_dict(base_encoder_state, strict=True)
        trainable_names = configure_last_geoframe_block_trainable(encoder)
        encoder.eval()
        model = copy.deepcopy(initial_atlas.model).to(torch_device)
        encoder_parameters = [
            parameter for parameter in encoder.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": model.parameters(), "lr": float(atlas_learning_rate)},
                {
                    "params": encoder_parameters,
                    "lr": float(encoder_learning_rate),
                },
            ],
            weight_decay=float(weight_decay),
        )
        generator = np.random.default_rng(seed)
        model.eval()
        initial_selection_sum = 0.0
        with torch.no_grad():
            for start in range(0, selection_rows.size, int(batch_size)):
                rows = selection_rows[start : start + int(batch_size)]
                _, prediction = model_batch(model, rows)
                initial_selection_sum += float(
                    torch.sum((prediction - target_tensor[rows]) ** 2)
                )
        best_selection = initial_selection_sum / float(
            selection_rows.size * targets.target_modes.shape[1]
        )
        best_epoch = -1
        best_atlas_state = copy.deepcopy(model.state_dict())
        best_encoder_state = {
            name: value.detach().cpu().clone()
            for name, value in encoder.state_dict().items()
            if name in trainable_names
        }
        run_history = {
            "optimization": [],
            "selection": [],
            "variance": [],
            "covariance": [],
        }
        for epoch in range(int(maximum_epochs)):
            model.train()
            permutation = generator.permutation(optimization_rows)
            prediction_sum = 0.0
            variance_sum = 0.0
            covariance_sum = 0.0
            for start in range(0, permutation.size, int(batch_size)):
                rows = permutation[start : start + int(batch_size)]
                with torch.autocast(
                    device_type=torch_device.type,
                    dtype=torch.bfloat16 if use_autocast else torch.float32,
                    enabled=use_autocast,
                ):
                    latent, prediction = model_batch(model, rows)
                    prediction_loss = torch.mean(
                        (prediction - target_tensor[rows]) ** 2
                    )
                    variance_loss, covariance_loss = _latent_regularization(latent)
                    loss = (
                        prediction_loss
                        + float(variance_weight) * variance_loss
                        + float(covariance_weight) * covariance_loss
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    encoder_parameters, max_norm=float(gradient_clip_norm)
                )
                optimizer.step()
                count = int(rows.size)
                prediction_sum += float(prediction_loss.detach()) * count
                variance_sum += float(variance_loss.detach()) * count
                covariance_sum += float(covariance_loss.detach()) * count
            denominator = float(permutation.size)
            model.eval()
            selection_sum = 0.0
            with torch.no_grad():
                for start in range(0, selection_rows.size, int(batch_size)):
                    rows = selection_rows[start : start + int(batch_size)]
                    with torch.autocast(
                        device_type=torch_device.type,
                        dtype=torch.bfloat16 if use_autocast else torch.float32,
                        enabled=use_autocast,
                    ):
                        _, prediction = model_batch(model, rows)
                    selection_sum += float(
                        torch.sum((prediction - target_tensor[rows]) ** 2)
                    )
            selection_loss = selection_sum / float(
                selection_rows.size * targets.target_modes.shape[1]
            )
            run_history["optimization"].append(prediction_sum / denominator)
            run_history["selection"].append(selection_loss)
            run_history["variance"].append(variance_sum / denominator)
            run_history["covariance"].append(covariance_sum / denominator)
            if selection_loss < best_selection:
                best_selection = selection_loss
                best_epoch = epoch
                best_atlas_state = copy.deepcopy(model.state_dict())
                best_encoder_state = {
                    name: value.detach().cpu().clone()
                    for name, value in encoder.state_dict().items()
                    if name in trainable_names
                }
            if epoch - best_epoch >= int(patience):
                break

        model.load_state_dict(best_atlas_state)
        current_encoder_state = encoder.state_dict()
        current_encoder_state.update(best_encoder_state)
        encoder.load_state_dict(current_encoder_state, strict=True)
        model.eval()
        predictions: list[np.ndarray] = []
        representations: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, row_count, int(batch_size)):
                rows = np.arange(
                    start, min(row_count, start + int(batch_size)), dtype=np.int64
                )
                with torch.autocast(
                    device_type=torch_device.type,
                    dtype=torch.bfloat16 if use_autocast else torch.float32,
                    enabled=use_autocast,
                ):
                    latent, prediction = model_batch(model, rows)
                representations.append(latent.float().cpu().numpy())
                predictions.append(prediction.float().cpu().numpy())
        prediction_array = np.concatenate(predictions)
        representation_array = np.concatenate(representations)
        histories[seed] = run_history
        seed_metrics[seed] = {
            "best_epoch": int(best_epoch),
            "epochs_run": int(len(run_history["selection"])),
            "selection": _metrics(
                prediction_array, targets.target_modes, targets.split_rows["selection"]
            ),
            "validation": _metrics(
                prediction_array, targets.target_modes, targets.split_rows["validation"]
            ),
        }
        predictions_by_seed[seed] = prediction_array
        representations_by_seed[seed] = representation_array
        atlas_states[seed] = best_atlas_state
        encoder_states[seed] = best_encoder_state
        print(
            f"[predictive-atlas-finetune] seed={seed} best_epoch={best_epoch} "
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
    selected_model = copy.deepcopy(initial_atlas.model)
    selected_model.load_state_dict(atlas_states[selected_seed])
    selected_model.eval()
    fitted_atlas = FittedPredictiveAtlas(
        model=selected_model.cpu(),
        embedding_mean=initial_atlas.embedding_mean,
        embedding_scale=initial_atlas.embedding_scale,
        descriptor_mean=initial_atlas.descriptor_mean,
        descriptor_scale=initial_atlas.descriptor_scale,
        conditioning_mean=initial_atlas.conditioning_mean,
        conditioning_scale=initial_atlas.conditioning_scale,
        seed=selected_seed,
        histories=histories,
        seed_metrics=seed_metrics,
        predictions_by_seed=predictions_by_seed,
        representations_by_seed=representations_by_seed,
        history_delta_mean=initial_atlas.history_delta_mean,
        history_delta_scale=initial_atlas.history_delta_scale,
    )
    assert trainable_names is not None
    parameter_by_name = dict(encoder.named_parameters())
    trainable_count = sum(
        int(parameter_by_name[name].numel()) for name in trainable_names
    )
    return FittedAtlasEncoderFineTune(
        atlas=fitted_atlas,
        encoder_state=encoder_states[selected_seed],
        trainable_encoder_parameter_names=trainable_names,
        trainable_encoder_parameter_count=trainable_count,
        initial_embedding_max_abs_error=maximum_error,
    )


__all__ = [
    "FittedAtlasEncoderFineTune",
    "fit_predictive_atlas_last_geoframe_block",
]
