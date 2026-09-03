from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from src.temporal_vamp.evaluation import CovariancePCA
from src.temporal_vamp.shooting_context import ShootingContextTokenCache
from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache
from src.temporal_vamp.shooting_multiscale import (
    DynamicTargetData,
    _metrics,
    _metrics_by_horizon,
)
from src.temporal_vamp.shooting_predictor import _future_neighbor_metrics


class SpatialAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        heads: int,
        rbf_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if int(hidden_dim) % int(heads) != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by heads={heads}."
            )
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.head_dim = self.hidden_dim // self.heads
        self.pre_attention_norm = nn.LayerNorm(self.hidden_dim)
        self.qkv = nn.Linear(self.hidden_dim, 3 * self.hidden_dim)
        self.distance_bias = nn.Linear(int(rbf_dim), self.heads, bias=False)
        self.attention_dropout = nn.Dropout(float(dropout))
        self.output = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_dropout = nn.Dropout(float(dropout))
        self.pre_ffn_norm = nn.LayerNorm(self.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, 4 * self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(4 * self.hidden_dim, self.hidden_dim),
            nn.Dropout(float(dropout)),
        )

    def forward(self, tokens: torch.Tensor, distance_rbf: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, _ = tokens.shape
        normalized = self.pre_attention_norm(tokens)
        qkv = self.qkv(normalized).reshape(
            batch_size, token_count, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
            float(self.head_dim)
        )
        scores = scores + self.distance_bias(distance_rbf).permute(0, 3, 1, 2)
        weights = self.attention_dropout(torch.softmax(scores, dim=-1))
        attended = torch.matmul(weights, value).permute(0, 2, 1, 3).reshape(
            batch_size, token_count, self.hidden_dim
        )
        tokens = tokens + self.output_dropout(self.output(attended))
        return tokens + self.ffn(self.pre_ffn_norm(tokens))


class SpatialContextTransformer(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int,
        descriptor_dim: int,
        hidden_dim: int,
        heads: int,
        blocks: int,
        rbf_dim: int,
        maximum_radius: float,
        representation_dim: int,
        target_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.descriptor_dim = int(descriptor_dim)
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.block_count = int(blocks)
        self.rbf_dim = int(rbf_dim)
        self.maximum_radius = float(maximum_radius)
        self.representation_dim = int(representation_dim)
        self.target_dim = int(target_dim)
        self.dropout = float(dropout)
        self.token_projection = nn.Linear(
            self.embedding_dim + self.descriptor_dim + 1, self.hidden_dim
        )
        self.blocks = nn.ModuleList(
            [
                SpatialAttentionBlock(
                    hidden_dim=self.hidden_dim,
                    heads=self.heads,
                    rbf_dim=self.rbf_dim,
                    dropout=self.dropout,
                )
                for _ in range(self.block_count)
            ]
        )
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.representation = nn.Sequential(
            nn.Linear(self.hidden_dim, self.representation_dim),
            nn.GELU(),
            nn.LayerNorm(self.representation_dim),
        )
        self.prediction_head = nn.Linear(self.representation_dim, self.target_dim)
        centers = torch.linspace(0.0, self.maximum_radius * 2.0, self.rbf_dim)
        self.register_buffer("rbf_centers", centers, persistent=True)
        spacing = float(centers[1] - centers[0]) if self.rbf_dim > 1 else self.maximum_radius
        self.register_buffer(
            "rbf_inverse_width_squared",
            torch.tensor(1.0 / max(spacing**2, 1.0e-8)),
            persistent=True,
        )

    def _distance_rbf(self, offsets: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(offsets, offsets)
        return torch.exp(
            -0.5
            * torch.square(distances[..., None] - self.rbf_centers)
            * self.rbf_inverse_width_squared
        )

    def encode(
        self,
        token_embeddings: torch.Tensor,
        token_descriptors: torch.Tensor,
        token_offsets: torch.Tensor,
    ) -> torch.Tensor:
        central_flag = torch.zeros(
            (*token_embeddings.shape[:2], 1),
            dtype=token_embeddings.dtype,
            device=token_embeddings.device,
        )
        central_flag[:, 0, 0] = 1.0
        tokens = self.token_projection(
            torch.cat([token_embeddings, token_descriptors, central_flag], dim=-1)
        )
        distance_rbf = self._distance_rbf(token_offsets)
        for block in self.blocks:
            tokens = block(tokens, distance_rbf)
        return self.representation(self.final_norm(tokens[:, 0]))

    def forward(
        self,
        token_embeddings: torch.Tensor,
        token_descriptors: torch.Tensor,
        token_offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.encode(
            token_embeddings, token_descriptors, token_offsets
        )
        return representation, self.prediction_head(representation)


@dataclass(frozen=True)
class SpatialTokenData:
    embeddings: np.ndarray
    descriptors: np.ndarray
    offsets: np.ndarray


@dataclass(frozen=True)
class FittedSpatialContextPredictor:
    model: SpatialContextTransformer
    embedding_mean: np.ndarray
    embedding_scale: np.ndarray
    descriptor_mean: np.ndarray
    descriptor_scale: np.ndarray
    seed: int
    histories: dict[int, dict[str, list[float]]]
    seed_metrics: dict[int, dict[str, Any]]
    predictions_by_seed: dict[int, np.ndarray]
    representations_by_seed: dict[int, np.ndarray]


def build_spatial_token_data(
    base_cache: ShootingEmbeddingCache,
    context_cache: ShootingContextTokenCache,
) -> SpatialTokenData:
    central_z = np.asarray(base_cache.parent_local_z, dtype=np.float32)
    satellite_z = np.asarray(context_cache.satellite_z, dtype=np.float32)
    central_descriptors = np.asarray(
        context_cache.central_descriptors, dtype=np.float32
    )
    satellite_descriptors = np.asarray(
        context_cache.satellite_descriptors, dtype=np.float32
    )
    satellite_offsets = np.asarray(
        context_cache.satellite_offsets, dtype=np.float32
    )
    parent_count, center_count, embedding_dim = central_z.shape
    row_count = parent_count * center_count
    embeddings = np.concatenate(
        [central_z[:, :, None, :], satellite_z], axis=2
    ).reshape(row_count, satellite_z.shape[2] + 1, embedding_dim)
    descriptors = np.concatenate(
        [central_descriptors[:, :, None, :], satellite_descriptors], axis=2
    ).reshape(row_count, satellite_z.shape[2] + 1, central_descriptors.shape[-1])
    offsets = np.concatenate(
        [
            np.zeros((parent_count, center_count, 1, 3), dtype=np.float32),
            satellite_offsets,
        ],
        axis=2,
    ).reshape(row_count, satellite_z.shape[2] + 1, 3)
    return SpatialTokenData(
        embeddings=embeddings, descriptors=descriptors, offsets=offsets
    )


def _standardize_tokens(
    tokens: SpatialTokenData,
    optimization_rows: np.ndarray,
) -> tuple[SpatialTokenData, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (
        embedding_mean,
        embedding_scale,
        descriptor_mean,
        descriptor_scale,
    ) = fit_spatial_token_standardization(tokens, optimization_rows)
    standardized = SpatialTokenData(
        embeddings=((tokens.embeddings - embedding_mean) / embedding_scale).astype(
            np.float32
        ),
        descriptors=(
            (tokens.descriptors - descriptor_mean) / descriptor_scale
        ).astype(np.float32),
        offsets=tokens.offsets.astype(np.float32, copy=False),
    )
    return (
        standardized,
        embedding_mean,
        embedding_scale,
        descriptor_mean,
        descriptor_scale,
    )


def fit_spatial_token_standardization(
    tokens: SpatialTokenData,
    optimization_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected_embeddings = tokens.embeddings[optimization_rows].reshape(
        -1, tokens.embeddings.shape[-1]
    )
    embedding_mean = selected_embeddings.mean(axis=0)
    embedding_scale = selected_embeddings.std(axis=0)
    embedding_scale = np.where(embedding_scale <= 1.0e-10, 1.0, embedding_scale)
    selected_descriptors = tokens.descriptors[optimization_rows].reshape(
        -1, tokens.descriptors.shape[-1]
    )
    descriptor_mean = selected_descriptors.mean(axis=0)
    descriptor_scale = selected_descriptors.std(axis=0)
    descriptor_scale = np.where(descriptor_scale <= 1.0e-10, 1.0, descriptor_scale)
    return (
        embedding_mean,
        embedding_scale,
        descriptor_mean,
        descriptor_scale,
    )


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def fit_spatial_context_transformer(
    tokens: SpatialTokenData,
    targets: DynamicTargetData,
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
    batch_size: int,
    maximum_epochs: int,
    patience: int,
    seeds: Sequence[int],
    initial_backbone_states: Mapping[int, Mapping[str, torch.Tensor]] | None = None,
) -> FittedSpatialContextPredictor:
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
        if initial_backbone_states is not None:
            if seed not in initial_backbone_states:
                raise KeyError(
                    f"No ordinary-trajectory pretrained backbone was provided for seed={seed}."
                )
            incompatible = model.load_state_dict(
                dict(initial_backbone_states[seed]), strict=False
            )
            expected_missing = {"prediction_head.weight", "prediction_head.bias"}
            if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "Pretrained spatial-backbone state is incompatible with the shooting "
                    f"model for seed={seed}: missing={incompatible.missing_keys}, "
                    f"unexpected={incompatible.unexpected_keys}."
                )
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
                _, prediction = model(
                    embeddings[rows], descriptors[rows], offsets[rows]
                )
                loss = torch.mean((prediction - target_tensor[rows]) ** 2)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                accumulated += float(loss.detach()) * int(rows.numel())
            optimization_loss = accumulated / float(permutation.numel())
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
            history["optimization"].append(optimization_loss)
            history["selection"].append(selection_loss)
            if selection_loss < best_selection:
                best_selection = selection_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            if epoch - best_epoch >= int(patience):
                break
        if best_state is None:
            raise RuntimeError(f"Spatial transformer seed {seed} produced no checkpoint.")
        model.load_state_dict(best_state)
        model.eval()
        all_representations: list[np.ndarray] = []
        all_predictions: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, int(embeddings.shape[0]), int(batch_size)):
                representation, prediction = model(
                    embeddings[start : start + int(batch_size)],
                    descriptors[start : start + int(batch_size)],
                    offsets[start : start + int(batch_size)],
                )
                all_representations.append(representation.cpu().numpy())
                all_predictions.append(prediction.cpu().numpy())
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
        }
        histories[seed] = history
        predictions[seed] = prediction_array
        representations[seed] = representation_array
        models[seed] = model.cpu()
        print(
            f"[shooting-spatial] seed={seed} best_epoch={best_epoch} "
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
    return FittedSpatialContextPredictor(
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


def evaluate_spatial_context_transformer(
    base_cache: ShootingEmbeddingCache,
    static_feature_variants: Mapping[str, np.ndarray],
    targets: DynamicTargetData,
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
    prediction = fitted.predictions_by_seed[fitted.seed]
    representation = fitted.representations_by_seed[fitted.seed]
    horizon_count = int(targets.selected_horizons_ps.size)
    mode_count = int(targets.target_modes.shape[1] // horizon_count)
    prediction_blocks = prediction.reshape(-1, horizon_count, mode_count)
    baseline_name = f"local_pca_{int(static_pca_dim)}d"
    retrieval: dict[str, Any] = {}
    for horizon_index, horizon in enumerate(targets.selected_horizons_ps.tolist()):
        spaces = {
            **static_spaces,
            "spatial_transformer_representation": representation,
            "spatial_transformer_predicted_change": prediction_blocks[:, horizon_index],
        }
        values = _future_neighbor_metrics(
            spaces,
            targets.mean_delta[:, horizon_index],
            base_cache,
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
    spaces = {
        **static_spaces,
        "spatial_transformer_representation": representation,
        "spatial_transformer_predicted_change": prediction,
    }
    combined = _future_neighbor_metrics(
        spaces,
        targets.mean_delta.reshape(targets.mean_delta.shape[0], -1),
        base_cache,
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
                "ablation": 2,
                "encoder": "frozen GeoFrameTransformerV2",
                "target": "same sibling-mean future-change target as ablation 1",
                "training_horizons_ps": targets.selected_horizons_ps.tolist(),
                "query_split": "validation source runs only",
                "candidate_filter": "different source run, exact temperature, exact parent phase",
                "spatial_invariance": "pairwise distances between PBC-correct context offsets",
                "excluded_changes": "no RFF target, neighbour KL, pretraining, velocity, or encoder fine-tuning",
            },
            "selected_seed": int(fitted.seed),
            "seed_metrics": {str(key): value for key, value in fitted.seed_metrics.items()},
            "future_neighbor_consistency": retrieval,
        },
        {
            "prediction": prediction,
            "representation": representation,
            "standardized_target_modes": targets.target_modes,
            "mean_delta": targets.mean_delta,
        },
    )


def save_spatial_context_transformer(
    fitted: FittedSpatialContextPredictor,
    targets: DynamicTargetData,
    path: str | Path,
) -> None:
    target = Path(path)
    model = fitted.model
    torch.save(
        {
            "state_dict": model.state_dict(),
            "embedding_dim": model.embedding_dim,
            "descriptor_dim": model.descriptor_dim,
            "hidden_dim": model.hidden_dim,
            "heads": model.heads,
            "blocks": model.block_count,
            "rbf_dim": model.rbf_dim,
            "maximum_radius": model.maximum_radius,
            "representation_dim": model.representation_dim,
            "target_dim": model.target_dim,
            "dropout": model.dropout,
            "seed": fitted.seed,
        },
        target,
    )
    np.savez(
        target.with_suffix(".preprocessing.npz"),
        embedding_mean=fitted.embedding_mean,
        embedding_scale=fitted.embedding_scale,
        descriptor_mean=fitted.descriptor_mean,
        descriptor_scale=fitted.descriptor_scale,
        target_mean=targets.target_mean,
        target_scale=targets.target_scale,
        target_pca_means=np.stack([pca.mean_ for pca in targets.target_pcas]),
        target_pca_components=np.stack(
            [pca.components_ for pca in targets.target_pcas]
        ),
        target_pca_eigenvalues=np.stack(
            [pca.eigenvalues_ for pca in targets.target_pcas]
        ),
        selected_horizons_ps=targets.selected_horizons_ps,
    )


__all__ = [
    "FittedSpatialContextPredictor",
    "SpatialContextTransformer",
    "SpatialTokenData",
    "build_spatial_token_data",
    "evaluate_spatial_context_transformer",
    "fit_spatial_token_standardization",
    "fit_spatial_context_transformer",
    "save_spatial_context_transformer",
]
