from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from torch import nn

from src.temporal_vamp.evaluation import CovariancePCA
from src.temporal_vamp.linear_vamp import LinearVAMP
from src.temporal_vamp.shooting_ablation import _rows_for_parents
from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache
from src.temporal_vamp.shooting_multiscale import _metrics
from src.temporal_vamp.shooting_predictor import _parent_split_indices
from src.temporal_vamp.shooting_spatial import (
    SpatialAttentionBlock,
    SpatialTokenData,
    _standardize_tokens,
)


@dataclass(frozen=True)
class PathKernelParameters:
    selected_horizon_indices: np.ndarray
    selected_horizons_ps: np.ndarray
    delta_mean: np.ndarray
    delta_scale: np.ndarray
    horizon_weights: np.ndarray
    median_distance: float
    bandwidths: np.ndarray
    frequencies: np.ndarray
    phases: np.ndarray


@dataclass(frozen=True)
class JointPathTargetData:
    target_modes: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    empirical_mean_embedding: np.ndarray
    branch_paths: np.ndarray
    branch_features: np.ndarray
    branch_parent_index: np.ndarray
    split_rows: dict[str, np.ndarray]
    parent_splits: dict[str, np.ndarray]
    kernel: PathKernelParameters
    diagnostics: dict[str, Any]

    @property
    def selected_horizons_ps(self) -> np.ndarray:
        return self.kernel.selected_horizons_ps


@dataclass(frozen=True)
class FittedPredictiveAtlas:
    model: "PredictiveAtlas"
    embedding_mean: np.ndarray
    embedding_scale: np.ndarray
    descriptor_mean: np.ndarray
    descriptor_scale: np.ndarray
    conditioning_mean: np.ndarray
    conditioning_scale: np.ndarray
    seed: int
    histories: dict[int, dict[str, list[float]]]
    seed_metrics: dict[int, dict[str, Any]]
    predictions_by_seed: dict[int, np.ndarray]
    representations_by_seed: dict[int, np.ndarray]
    history_delta_mean: np.ndarray | None = None
    history_delta_scale: np.ndarray | None = None


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _selected_horizon_indices(
    available: np.ndarray, requested: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    requested_values = np.asarray([float(value) for value in requested], dtype=np.float64)
    indices: list[int] = []
    for horizon in requested_values.tolist():
        matches = np.flatnonzero(
            np.isclose(available, horizon, rtol=0.0, atol=1.0e-9)
        )
        if matches.size != 1:
            raise ValueError(
                f"Requested atlas horizon {horizon:g} ps is not uniquely available in "
                f"{available.tolist()}."
            )
        indices.append(int(matches[0]))
    if len(set(indices)) != len(indices):
        raise ValueError(
            f"Predictive-atlas horizons must be unique, got {requested_values.tolist()}."
        )
    return np.asarray(indices, dtype=np.int64), requested_values


def _median_pair_distance(
    values: np.ndarray, *, maximum_samples: int, rng: np.random.Generator
) -> float:
    selected = np.asarray(values, dtype=np.float32)
    if selected.shape[0] > int(maximum_samples):
        rows = rng.choice(selected.shape[0], size=int(maximum_samples), replace=False)
        selected = selected[rows]
    pair_count = min(100_000, int(selected.shape[0]) * 32)
    left = rng.integers(0, selected.shape[0], size=pair_count)
    right = rng.integers(0, selected.shape[0], size=pair_count)
    distances = np.linalg.norm(selected[left] - selected[right], axis=1)
    positive = distances[distances > 1.0e-10]
    if positive.size == 0:
        raise RuntimeError(
            "Cannot fit the predictive-atlas kernel because every sampled path "
            "distance is zero."
        )
    return float(np.median(positive))


def random_fourier_path_features(
    values: np.ndarray,
    frequencies: np.ndarray,
    phases: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    paths = np.asarray(values, dtype=np.float32)
    if paths.ndim != 2:
        raise ValueError(f"RFF path values must be 2D, got {paths.shape}.")
    omega = np.asarray(frequencies, dtype=np.float32)
    bias = np.asarray(phases, dtype=np.float32)
    if omega.ndim != 3 or omega.shape[1] != paths.shape[1]:
        raise ValueError(
            "RFF frequencies must have shape (bands, path_dim, features); "
            f"paths={paths.shape}, frequencies={omega.shape}."
        )
    if bias.shape != (omega.shape[0], omega.shape[2]):
        raise ValueError(
            f"RFF phase shape mismatch: expected={(omega.shape[0], omega.shape[2])}, "
            f"observed={bias.shape}."
        )
    torch_device = torch.device(device)
    omega_tensor = torch.from_numpy(omega).to(torch_device)
    bias_tensor = torch.from_numpy(bias).to(torch_device)
    scale = math.sqrt(2.0 / float(omega.shape[0] * omega.shape[2]))
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, paths.shape[0], int(batch_size)):
            batch = torch.from_numpy(paths[start : start + int(batch_size)]).to(
                torch_device
            )
            blocks = [
                scale * torch.cos(batch @ omega_tensor[band] + bias_tensor[band])
                for band in range(int(omega.shape[0]))
            ]
            outputs.append(torch.cat(blocks, dim=1).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)


def _parent_mean_embeddings(
    branch_features: np.ndarray,
    branch_parent_index: np.ndarray,
    *,
    parent_count: int,
) -> np.ndarray:
    output = np.empty(
        (int(parent_count), branch_features.shape[1], branch_features.shape[2]),
        dtype=np.float32,
    )
    for parent_index in range(int(parent_count)):
        rows = np.flatnonzero(branch_parent_index == parent_index)
        if rows.size < 2:
            raise RuntimeError(
                "Joint future-path mean embeddings require at least two branches per "
                f"parent; parent_index={parent_index}, branches={rows.size}."
            )
        output[parent_index] = branch_features[rows].mean(axis=0)
    return output


def _split_shot_diagnostics(
    branch_features: np.ndarray,
    branch_parent_index: np.ndarray,
    *,
    parent_count: int,
    validation_rows: np.ndarray,
    center_count: int,
) -> dict[str, float | int]:
    shape = (int(parent_count), int(center_count), branch_features.shape[-1])
    split_a = np.empty(shape, dtype=np.float32)
    split_b = np.empty(shape, dtype=np.float32)
    counts: list[int] = []
    for parent_index in range(int(parent_count)):
        rows = np.sort(np.flatnonzero(branch_parent_index == parent_index))
        counts.append(int(rows.size))
        a_rows = rows[::2]
        b_rows = rows[1::2]
        if b_rows.size == 0:
            raise RuntimeError(
                f"Parent {parent_index} has no second split-shot half."
            )
        split_a[parent_index] = branch_features[a_rows].mean(axis=0)
        split_b[parent_index] = branch_features[b_rows].mean(axis=0)
    flat_a = split_a.reshape(-1, branch_features.shape[-1])[validation_rows]
    flat_b = split_b.reshape(-1, branch_features.shape[-1])[validation_rows]
    distance = np.linalg.norm(flat_a - flat_b, axis=1)
    correlation = np.corrcoef(flat_a.reshape(-1), flat_b.reshape(-1))[0, 1]
    return {
        "mean_split_embedding_distance": float(distance.mean()),
        "flattened_split_embedding_correlation": float(correlation),
        "minimum_branches_per_parent": int(min(counts)),
        "maximum_branches_per_parent": int(max(counts)),
    }


def prepare_joint_path_target_data(
    cache: ShootingEmbeddingCache,
    *,
    horizons_ps: Sequence[float],
    horizon_weights: Sequence[float],
    rff_features_per_bandwidth: int,
    bandwidth_multipliers: Sequence[float],
    selection_source_velocity_seeds: Sequence[int],
    seed: int,
    rff_device: str,
    rff_batch_size: int,
) -> JointPathTargetData:
    parent_splits = _parent_split_indices(
        cache,
        selection_source_velocity_seeds=selection_source_velocity_seeds,
    )
    parent_count, center_count, embedding_dim = cache.parent_local_z.shape
    split_rows = {
        name: _rows_for_parents(indices, int(center_count))
        for name, indices in parent_splits.items()
    }
    horizon_indices, requested_horizons = _selected_horizon_indices(
        np.asarray(cache.horizons_ps, dtype=np.float64), horizons_ps
    )
    raw_weights = np.asarray([float(value) for value in horizon_weights], dtype=np.float64)
    if raw_weights.shape != requested_horizons.shape or np.any(raw_weights <= 0.0):
        raise ValueError(
            "target.horizon_weights must contain one positive value per horizon; "
            f"horizons={requested_horizons.tolist()}, weights={raw_weights.tolist()}."
        )
    normalized_weights = raw_weights / np.linalg.norm(raw_weights)
    bandwidth_factors = np.asarray(
        [float(value) for value in bandwidth_multipliers], dtype=np.float64
    )
    if bandwidth_factors.size == 0 or np.any(bandwidth_factors <= 0.0):
        raise ValueError(
            "target.bandwidth_multipliers must be positive and nonempty; "
            f"got {bandwidth_factors.tolist()}."
        )
    feature_count = int(rff_features_per_bandwidth)
    if feature_count <= 0:
        raise ValueError(
            f"rff_features_per_bandwidth must be positive, got {feature_count}."
        )

    branch_parent = np.asarray(cache.branch_parent_index, dtype=np.int64)
    optimization_branch_rows = np.flatnonzero(
        np.isin(branch_parent, parent_splits["optimization"])
    )
    current = np.asarray(cache.parent_local_z[branch_parent], dtype=np.float32)
    future = np.asarray(cache.future_z[:, horizon_indices], dtype=np.float32)
    delta = future - current[:, None, :, :]
    optimization_delta = delta[optimization_branch_rows]
    delta_mean = optimization_delta.mean(axis=(0, 2), dtype=np.float64)
    delta_scale = optimization_delta.std(axis=(0, 2), dtype=np.float64)
    scale_floor = np.maximum(
        1.0e-8,
        1.0e-6 * np.max(delta_scale, axis=1, keepdims=True),
    )
    redundant_dimensions = delta_scale <= scale_floor
    delta_scale = np.where(redundant_dimensions, 1.0, delta_scale)
    standardized = (delta - delta_mean[None, :, None, :]) / delta_scale[
        None, :, None, :
    ]
    weighted = standardized * normalized_weights[None, :, None, None]
    branch_paths = weighted.transpose(0, 2, 1, 3).reshape(
        future.shape[0], int(center_count), requested_horizons.size * int(embedding_dim)
    )
    branch_paths = branch_paths.astype(np.float32, copy=False)

    rng = np.random.default_rng(int(seed))
    optimization_paths = branch_paths[optimization_branch_rows].reshape(
        -1, branch_paths.shape[-1]
    )
    median_distance = _median_pair_distance(
        optimization_paths,
        maximum_samples=4096,
        rng=rng,
    )
    bandwidths = median_distance * bandwidth_factors
    frequencies = np.stack(
        [
            rng.normal(
                scale=1.0 / float(bandwidth),
                size=(branch_paths.shape[-1], feature_count),
            )
            for bandwidth in bandwidths.tolist()
        ],
        axis=0,
    ).astype(np.float32)
    phases = rng.uniform(
        0.0,
        2.0 * np.pi,
        size=(bandwidths.size, feature_count),
    ).astype(np.float32)
    flat_features = random_fourier_path_features(
        branch_paths.reshape(-1, branch_paths.shape[-1]),
        frequencies,
        phases,
        device=rff_device,
        batch_size=int(rff_batch_size),
    )
    branch_features = flat_features.reshape(
        branch_paths.shape[0], int(center_count), flat_features.shape[-1]
    )
    parent_embedding = _parent_mean_embeddings(
        branch_features,
        branch_parent,
        parent_count=int(parent_count),
    )
    flat_embedding = parent_embedding.reshape(
        int(parent_count) * int(center_count), parent_embedding.shape[-1]
    )
    target_mean = flat_embedding[split_rows["optimization"]].mean(
        axis=0, dtype=np.float64
    )
    target_scale = flat_embedding[split_rows["optimization"]].std(
        axis=0, dtype=np.float64
    )
    target_scale = np.where(target_scale <= 1.0e-10, 1.0, target_scale)
    target_modes = ((flat_embedding - target_mean) / target_scale).astype(np.float32)

    shuffled_standardized = weighted.copy()
    for parent_index in range(int(parent_count)):
        rows = np.sort(np.flatnonzero(branch_parent == parent_index))
        for horizon_index in range(1, requested_horizons.size):
            permutation = rng.permutation(rows.size)
            shuffled_standardized[rows, horizon_index] = weighted[
                rows[permutation], horizon_index
            ]
    shuffled_paths = shuffled_standardized.transpose(0, 2, 1, 3).reshape(
        branch_paths.shape
    )
    shuffled_features = random_fourier_path_features(
        shuffled_paths.reshape(-1, shuffled_paths.shape[-1]),
        frequencies,
        phases,
        device=rff_device,
        batch_size=int(rff_batch_size),
    ).reshape(branch_features.shape)
    shuffled_parent_embedding = _parent_mean_embeddings(
        shuffled_features,
        branch_parent,
        parent_count=int(parent_count),
    ).reshape(flat_embedding.shape)
    alignment_distance = np.linalg.norm(
        flat_embedding[split_rows["validation"]]
        - shuffled_parent_embedding[split_rows["validation"]],
        axis=1,
    )
    diagnostics = {
        "target": (
            "multi-bandwidth RFF conditional mean embedding of the same-branch "
            "joint GeoFrame future-change path"
        ),
        "path_representation": (
            "full 128-dimensional frozen target-encoder change at each horizon; "
            "standardized on optimization source runs"
        ),
        "path_dimension": int(branch_paths.shape[-1]),
        "mean_embedding_dimension": int(flat_embedding.shape[-1]),
        "horizons_ps": requested_horizons.tolist(),
        "horizon_weights": normalized_weights.tolist(),
        "near_constant_dimensions_by_horizon": np.sum(
            redundant_dimensions, axis=1
        ).tolist(),
        "median_distance": float(median_distance),
        "bandwidths": bandwidths.tolist(),
        "rff_features_per_bandwidth": int(feature_count),
        "split_shot_reliability": _split_shot_diagnostics(
            branch_features,
            branch_parent,
            parent_count=int(parent_count),
            validation_rows=split_rows["validation"],
            center_count=int(center_count),
        ),
        "same_branch_alignment": {
            "shuffle": (
                "horizons after the first were independently permuted among sibling "
                "branches within each parent"
            ),
            "validation_mean_true_vs_shuffled_embedding_distance": float(
                alignment_distance.mean()
            ),
            "validation_median_true_vs_shuffled_embedding_distance": float(
                np.median(alignment_distance)
            ),
        },
    }
    return JointPathTargetData(
        target_modes=target_modes,
        target_mean=target_mean,
        target_scale=target_scale,
        empirical_mean_embedding=flat_embedding,
        branch_paths=branch_paths,
        branch_features=branch_features,
        branch_parent_index=branch_parent,
        split_rows=split_rows,
        parent_splits=parent_splits,
        kernel=PathKernelParameters(
            selected_horizon_indices=horizon_indices,
            selected_horizons_ps=requested_horizons,
            delta_mean=delta_mean,
            delta_scale=delta_scale,
            horizon_weights=normalized_weights,
            median_distance=float(median_distance),
            bandwidths=bandwidths,
            frequencies=frequencies,
            phases=phases,
        ),
        diagnostics=diagnostics,
    )


def prepare_joint_path_target_data_from_kernel(
    cache: ShootingEmbeddingCache,
    *,
    kernel_path: str | Path,
    selection_source_velocity_seeds: Sequence[int],
    rff_device: str,
    rff_batch_size: int,
) -> JointPathTargetData:
    """Apply an already fitted future-path kernel to additional center atoms."""

    path = Path(kernel_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Predictive-atlas path kernel is missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "selected_horizon_indices",
            "selected_horizons_ps",
            "delta_mean",
            "delta_scale",
            "horizon_weights",
            "median_distance",
            "bandwidths",
            "frequencies",
            "phases",
            "target_mean",
            "target_scale",
        }
        missing = sorted(required.difference(payload.files))
        if missing:
            raise RuntimeError(f"Path kernel {path} is missing arrays: {missing}.")
        arrays = {name: payload[name].copy() for name in required}
    horizon_indices = np.asarray(arrays["selected_horizon_indices"], dtype=np.int64)
    horizons_ps = np.asarray(arrays["selected_horizons_ps"], dtype=np.float64)
    available = np.asarray(cache.horizons_ps, dtype=np.float64)
    if (
        np.any(horizon_indices < 0)
        or np.any(horizon_indices >= available.size)
        or not np.allclose(
            available[horizon_indices], horizons_ps, rtol=0.0, atol=1.0e-9
        )
    ):
        raise RuntimeError(
            f"Path kernel horizons {horizons_ps.tolist()} do not match cache "
            f"horizons {available.tolist()} at indices {horizon_indices.tolist()}."
        )
    delta_mean = np.asarray(arrays["delta_mean"], dtype=np.float64)
    delta_scale = np.asarray(arrays["delta_scale"], dtype=np.float64)
    horizon_weights = np.asarray(arrays["horizon_weights"], dtype=np.float64)
    frequencies = np.asarray(arrays["frequencies"], dtype=np.float32)
    phases = np.asarray(arrays["phases"], dtype=np.float32)
    target_mean = np.asarray(arrays["target_mean"], dtype=np.float64)
    target_scale = np.asarray(arrays["target_scale"], dtype=np.float64)

    parent_splits = _parent_split_indices(
        cache,
        selection_source_velocity_seeds=selection_source_velocity_seeds,
    )
    parent_count, center_count, embedding_dim = cache.parent_local_z.shape
    split_rows = {
        name: _rows_for_parents(indices, int(center_count))
        for name, indices in parent_splits.items()
    }
    branch_parent = np.asarray(cache.branch_parent_index, dtype=np.int64)
    current = np.asarray(cache.parent_local_z[branch_parent], dtype=np.float32)
    future = np.asarray(cache.future_z[:, horizon_indices], dtype=np.float32)
    delta = future - current[:, None, :, :]
    expected_statistics_shape = (horizons_ps.size, int(embedding_dim))
    if (
        delta_mean.shape != expected_statistics_shape
        or delta_scale.shape != expected_statistics_shape
        or horizon_weights.shape != (horizons_ps.size,)
    ):
        raise RuntimeError(
            "Path-kernel preprocessing shape does not match expanded embeddings: "
            f"mean={delta_mean.shape}, scale={delta_scale.shape}, "
            f"weights={horizon_weights.shape}, expected={expected_statistics_shape}."
        )
    standardized = (delta - delta_mean[None, :, None, :]) / delta_scale[
        None, :, None, :
    ]
    weighted = standardized * horizon_weights[None, :, None, None]
    branch_paths = weighted.transpose(0, 2, 1, 3).reshape(
        future.shape[0], int(center_count), horizons_ps.size * int(embedding_dim)
    )
    branch_paths = branch_paths.astype(np.float32, copy=False)
    flat_features = random_fourier_path_features(
        branch_paths.reshape(-1, branch_paths.shape[-1]),
        frequencies,
        phases,
        device=rff_device,
        batch_size=int(rff_batch_size),
    )
    branch_features = flat_features.reshape(
        branch_paths.shape[0], int(center_count), flat_features.shape[-1]
    )
    parent_embedding = _parent_mean_embeddings(
        branch_features,
        branch_parent,
        parent_count=int(parent_count),
    )
    flat_embedding = parent_embedding.reshape(
        int(parent_count) * int(center_count), parent_embedding.shape[-1]
    )
    if target_mean.shape != (flat_embedding.shape[1],) or target_scale.shape != (
        flat_embedding.shape[1],
    ):
        raise RuntimeError(
            f"Path-kernel target standardization shape changed: mean={target_mean.shape}, "
            f"scale={target_scale.shape}, embedding={flat_embedding.shape}."
        )
    target_modes = ((flat_embedding - target_mean) / target_scale).astype(np.float32)
    diagnostics = {
        "target": "fixed conditional mean-embedding kernel from the 64-center atlas",
        "kernel_path": str(path),
        "path_dimension": int(branch_paths.shape[-1]),
        "mean_embedding_dimension": int(flat_embedding.shape[-1]),
        "horizons_ps": horizons_ps.tolist(),
        "center_atoms_per_parent": int(center_count),
        "split_shot_reliability": _split_shot_diagnostics(
            branch_features,
            branch_parent,
            parent_count=int(parent_count),
            validation_rows=split_rows["validation"],
            center_count=int(center_count),
        ),
    }
    return JointPathTargetData(
        target_modes=target_modes,
        target_mean=target_mean,
        target_scale=target_scale,
        empirical_mean_embedding=flat_embedding,
        branch_paths=branch_paths,
        branch_features=branch_features,
        branch_parent_index=branch_parent,
        split_rows=split_rows,
        parent_splits=parent_splits,
        kernel=PathKernelParameters(
            selected_horizon_indices=horizon_indices,
            selected_horizons_ps=horizons_ps,
            delta_mean=delta_mean,
            delta_scale=delta_scale,
            horizon_weights=horizon_weights,
            median_distance=float(np.asarray(arrays["median_distance"]).item()),
            bandwidths=np.asarray(arrays["bandwidths"], dtype=np.float64),
            frequencies=frequencies,
            phases=phases,
        ),
        diagnostics=diagnostics,
    )


class PredictiveAtlas(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int,
        descriptor_dim: int,
        conditioning_dim: int,
        hidden_dim: int,
        heads: int,
        blocks: int,
        rbf_dim: int,
        maximum_radius: float,
        latent_dim: int,
        decoder_hidden_dim: int,
        target_dim: int,
        dropout: float,
        history_lag_count: int = 0,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.descriptor_dim = int(descriptor_dim)
        self.conditioning_dim = int(conditioning_dim)
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.block_count = int(blocks)
        self.rbf_dim = int(rbf_dim)
        self.maximum_radius = float(maximum_radius)
        self.latent_dim = int(latent_dim)
        self.decoder_hidden_dim = int(decoder_hidden_dim)
        self.target_dim = int(target_dim)
        self.dropout = float(dropout)
        self.history_lag_count = int(history_lag_count)
        self.token_projection = nn.Linear(
            self.embedding_dim + self.descriptor_dim + 1,
            self.hidden_dim,
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
        self.condition_projection = nn.Linear(
            self.conditioning_dim, self.hidden_dim, bias=False
        )
        if self.history_lag_count > 0:
            self.history_gru = nn.GRU(
                input_size=self.embedding_dim,
                hidden_size=self.hidden_dim,
                batch_first=True,
            )
            self.history_norm = nn.LayerNorm(self.hidden_dim)
            self.history_gate = nn.Parameter(torch.full((self.hidden_dim,), -4.0))
        else:
            self.history_gru = None
            self.history_norm = None
            self.register_parameter("history_gate", None)
        self.atlas_chart = nn.Sequential(
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.GELU(),
            nn.LayerNorm(self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.decoder_hidden_dim, self.decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(self.decoder_hidden_dim, self.target_dim),
        )
        centers = torch.linspace(0.0, self.maximum_radius * 2.0, self.rbf_dim)
        self.register_buffer("rbf_centers", centers, persistent=True)
        spacing = (
            float(centers[1] - centers[0])
            if self.rbf_dim > 1
            else self.maximum_radius
        )
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
        conditioning: torch.Tensor,
        history_deltas: torch.Tensor | None = None,
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
        if self.history_lag_count > 0:
            if history_deltas is None:
                raise RuntimeError(
                    "History-conditioned PredictiveAtlas requires history_deltas."
                )
            expected = (
                token_embeddings.shape[0],
                self.history_lag_count,
                token_embeddings.shape[1],
                self.embedding_dim,
            )
            if tuple(history_deltas.shape) != expected:
                raise ValueError(
                    "Predictive-atlas history shape changed: "
                    f"expected={expected}, observed={tuple(history_deltas.shape)}."
                )
            batch_size, lag_count, token_count, embedding_dim = history_deltas.shape
            sequences = history_deltas.permute(0, 2, 1, 3).reshape(
                batch_size * token_count, lag_count, embedding_dim
            )
            assert self.history_gru is not None
            assert self.history_norm is not None
            _, final_history = self.history_gru(sequences)
            temporal_tokens = self.history_norm(final_history[0]).reshape(
                batch_size, token_count, self.hidden_dim
            )
            tokens = tokens + torch.sigmoid(self.history_gate) * temporal_tokens
        elif history_deltas is not None:
            raise RuntimeError(
                "Position-only PredictiveAtlas received unexpected history_deltas."
            )
        distance_rbf = self._distance_rbf(token_offsets)
        for block in self.blocks:
            tokens = block(tokens, distance_rbf)
        pooled = self.final_norm(tokens[:, 0])
        pooled = pooled + self.condition_projection(conditioning)
        return self.atlas_chart(pooled)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        token_descriptors: torch.Tensor,
        token_offsets: torch.Tensor,
        conditioning: torch.Tensor,
        history_deltas: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(
            token_embeddings,
            token_descriptors,
            token_offsets,
            conditioning,
            history_deltas,
        )
        return latent, self.decode(latent)


def _latent_regularization(latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centered = latent - latent.mean(dim=0, keepdim=True)
    standard_deviation = torch.sqrt(latent.var(dim=0, unbiased=False) + 1.0e-4)
    variance_loss = torch.mean(torch.relu(1.0 - standard_deviation))
    covariance = centered.T @ centered / float(max(int(latent.shape[0]) - 1, 1))
    covariance = covariance - torch.diag(torch.diagonal(covariance))
    covariance_loss = torch.sum(covariance**2) / float(latent.shape[1])
    return variance_loss, covariance_loss


def _load_backbone_state(
    model: PredictiveAtlas,
    source_state: Mapping[str, torch.Tensor],
) -> None:
    prefixes = ("token_projection.", "blocks.", "final_norm.")
    model_state = model.state_dict()
    required = {name for name in model_state if name.startswith(prefixes)}
    selected = {
        name: value
        for name, value in source_state.items()
        if name in required and tuple(value.shape) == tuple(model_state[name].shape)
    }
    missing = sorted(required.difference(selected))
    if missing:
        raise RuntimeError(
            "The ordinary temporal-pretraining checkpoint is missing compatible atlas "
            f"backbone parameters: {missing}."
        )
    model.load_state_dict(selected, strict=False)


def fit_predictive_atlas(
    tokens: SpatialTokenData,
    targets: JointPathTargetData,
    conditioning_values: np.ndarray,
    *,
    device: str,
    hidden_dim: int,
    heads: int,
    blocks: int,
    rbf_dim: int,
    maximum_radius: float,
    latent_dim: int,
    decoder_hidden_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    variance_weight: float,
    covariance_weight: float,
    batch_size: int,
    maximum_epochs: int,
    patience: int,
    seeds: Sequence[int],
    initial_backbone_states: Mapping[int, Mapping[str, torch.Tensor]] | None,
    history_embeddings: np.ndarray | None = None,
    initial_model_state: Mapping[str, torch.Tensor] | None = None,
) -> FittedPredictiveAtlas:
    (
        standardized,
        embedding_mean,
        embedding_scale,
        descriptor_mean,
        descriptor_scale,
    ) = _standardize_tokens(tokens, targets.split_rows["optimization"])
    conditioning = np.asarray(conditioning_values, dtype=np.float64)
    if conditioning.shape[0] != standardized.embeddings.shape[0] or conditioning.ndim != 2:
        raise ValueError(
            "Atlas conditioning must have shape (state_rows, features); "
            f"tokens={standardized.embeddings.shape}, conditioning={conditioning.shape}."
        )
    conditioning_mean = conditioning[targets.split_rows["optimization"]].mean(axis=0)
    conditioning_scale = conditioning[targets.split_rows["optimization"]].std(axis=0)
    conditioning_scale = np.where(conditioning_scale <= 1.0e-10, 1.0, conditioning_scale)
    standardized_conditioning = (
        (conditioning - conditioning_mean) / conditioning_scale
    ).astype(np.float32)
    history_delta_mean: np.ndarray | None = None
    history_delta_scale: np.ndarray | None = None
    standardized_history: np.ndarray | None = None
    if history_embeddings is not None:
        history_values = np.asarray(history_embeddings, dtype=np.float32)
        expected = (
            standardized.embeddings.shape[0],
            history_values.shape[1],
            standardized.embeddings.shape[1],
            standardized.embeddings.shape[2],
        )
        if history_values.ndim != 4 or tuple(history_values.shape) != expected:
            raise ValueError(
                "History embeddings must have shape "
                "(state_rows, history_lags, spatial_tokens, embedding_dim); "
                f"expected={expected}, observed={history_values.shape}."
            )
        raw_current = np.asarray(tokens.embeddings, dtype=np.float32)
        history_delta = history_values - raw_current[:, None, :, :]
        selected_delta = history_delta[targets.split_rows["optimization"]]
        history_delta_mean = selected_delta.mean(axis=(0, 2), dtype=np.float64)
        history_delta_scale = selected_delta.std(axis=(0, 2), dtype=np.float64)
        history_delta_scale = np.where(
            history_delta_scale <= 1.0e-10, 1.0, history_delta_scale
        )
        standardized_history = (
            (history_delta - history_delta_mean[None, :, None, :])
            / history_delta_scale[None, :, None, :]
        ).astype(np.float32)

    torch_device = torch.device(device)
    embeddings = torch.from_numpy(standardized.embeddings).to(torch_device)
    descriptors = torch.from_numpy(standardized.descriptors).to(torch_device)
    offsets = torch.from_numpy(standardized.offsets).to(torch_device)
    conditions = torch.from_numpy(standardized_conditioning).to(torch_device)
    history_tensor = (
        None
        if standardized_history is None
        else torch.from_numpy(standardized_history).to(torch_device)
    )
    target_tensor = torch.from_numpy(targets.target_modes).to(torch_device)
    optimization_rows = torch.from_numpy(targets.split_rows["optimization"]).to(
        torch_device
    )
    selection_rows = torch.from_numpy(targets.split_rows["selection"]).to(torch_device)
    histories: dict[int, dict[str, list[float]]] = {}
    seed_metrics: dict[int, dict[str, Any]] = {}
    predictions: dict[int, np.ndarray] = {}
    representations: dict[int, np.ndarray] = {}
    models: dict[int, PredictiveAtlas] = {}
    for raw_seed in seeds:
        seed = int(raw_seed)
        _seed_everything(seed)
        model = PredictiveAtlas(
            embedding_dim=int(embeddings.shape[-1]),
            descriptor_dim=int(descriptors.shape[-1]),
            conditioning_dim=int(conditions.shape[-1]),
            hidden_dim=int(hidden_dim),
            heads=int(heads),
            blocks=int(blocks),
            rbf_dim=int(rbf_dim),
            maximum_radius=float(maximum_radius),
            latent_dim=int(latent_dim),
            decoder_hidden_dim=int(decoder_hidden_dim),
            target_dim=int(targets.target_modes.shape[1]),
            dropout=float(dropout),
            history_lag_count=(
                0 if history_tensor is None else int(history_tensor.shape[1])
            ),
        ).to(torch_device)
        if initial_backbone_states is not None:
            if seed not in initial_backbone_states:
                raise KeyError(
                    f"No ordinary pretrained backbone is available for atlas seed={seed}."
                )
            _load_backbone_state(model, initial_backbone_states[seed])
        if initial_model_state is not None:
            incompatible = model.load_state_dict(dict(initial_model_state), strict=False)
            allowed_missing = {
                name
                for name in model.state_dict()
                if name not in initial_model_state
            }
            unexpected_missing = sorted(
                name
                for name in allowed_missing
                if not name.startswith(
                    ("history_gru.", "history_norm.", "history_gate")
                )
            )
            if (
                set(incompatible.missing_keys) != allowed_missing
                or unexpected_missing
                or incompatible.unexpected_keys
            ):
                raise RuntimeError(
                    "Position-only atlas checkpoint is incompatible with temporal "
                    f"initialization: missing={incompatible.missing_keys}, "
                    f"unexpected={incompatible.unexpected_keys}."
                )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        history = {
            "optimization": [],
            "selection": [],
            "variance": [],
            "covariance": [],
        }
        best_selection = float("inf")
        best_epoch = -1
        best_state: dict[str, torch.Tensor] | None = None
        if initial_model_state is not None:
            model.eval()
            with torch.no_grad():
                _, initial_selection_prediction = model(
                    embeddings[selection_rows],
                    descriptors[selection_rows],
                    offsets[selection_rows],
                    conditions[selection_rows],
                    (
                        None
                        if history_tensor is None
                        else history_tensor[selection_rows]
                    ),
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
            best_state = copy.deepcopy(model.state_dict())
        for epoch in range(int(maximum_epochs)):
            permutation = torch.randperm(
                optimization_rows.numel(), generator=generator
            ).to(torch_device)
            model.train()
            accumulated_prediction = 0.0
            accumulated_variance = 0.0
            accumulated_covariance = 0.0
            for start in range(0, int(permutation.numel()), int(batch_size)):
                rows = optimization_rows[permutation[start : start + int(batch_size)]]
                latent, prediction = model(
                    embeddings[rows],
                    descriptors[rows],
                    offsets[rows],
                    conditions[rows],
                    None if history_tensor is None else history_tensor[rows],
                )
                prediction_loss = torch.mean((prediction - target_tensor[rows]) ** 2)
                variance_loss, covariance_loss = _latent_regularization(latent)
                loss = (
                    prediction_loss
                    + float(variance_weight) * variance_loss
                    + float(covariance_weight) * covariance_loss
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                row_count = int(rows.numel())
                accumulated_prediction += float(prediction_loss.detach()) * row_count
                accumulated_variance += float(variance_loss.detach()) * row_count
                accumulated_covariance += float(covariance_loss.detach()) * row_count
            denominator = float(permutation.numel())
            model.eval()
            with torch.no_grad():
                _, selection_prediction = model(
                    embeddings[selection_rows],
                    descriptors[selection_rows],
                    offsets[selection_rows],
                    conditions[selection_rows],
                    (
                        None
                        if history_tensor is None
                        else history_tensor[selection_rows]
                    ),
                )
                selection_loss = float(
                    torch.mean(
                        (selection_prediction - target_tensor[selection_rows]) ** 2
                    )
                )
            history["optimization"].append(accumulated_prediction / denominator)
            history["selection"].append(selection_loss)
            history["variance"].append(accumulated_variance / denominator)
            history["covariance"].append(accumulated_covariance / denominator)
            if selection_loss < best_selection:
                best_selection = selection_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            if epoch - best_epoch >= int(patience):
                break
        if best_state is None:
            raise RuntimeError(f"Predictive-atlas seed {seed} produced no checkpoint.")
        model.load_state_dict(best_state)
        model.eval()
        seed_predictions: list[np.ndarray] = []
        seed_representations: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, int(embeddings.shape[0]), int(batch_size)):
                representation, prediction = model(
                    embeddings[start : start + int(batch_size)],
                    descriptors[start : start + int(batch_size)],
                    offsets[start : start + int(batch_size)],
                    conditions[start : start + int(batch_size)],
                    (
                        None
                        if history_tensor is None
                        else history_tensor[start : start + int(batch_size)]
                    ),
                )
                seed_representations.append(representation.cpu().numpy())
                seed_predictions.append(prediction.cpu().numpy())
        prediction_array = np.concatenate(seed_predictions)
        representation_array = np.concatenate(seed_representations)
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
            "latent_component_std_validation": np.std(
                representation_array[targets.split_rows["validation"]], axis=0
            ).tolist(),
        }
        predictions[seed] = prediction_array
        representations[seed] = representation_array
        models[seed] = model.cpu()
        print(
            f"[predictive-atlas] seed={seed} best_epoch={best_epoch} "
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
    return FittedPredictiveAtlas(
        model=models[selected_seed],
        embedding_mean=embedding_mean,
        embedding_scale=embedding_scale,
        descriptor_mean=descriptor_mean,
        descriptor_scale=descriptor_scale,
        conditioning_mean=conditioning_mean,
        conditioning_scale=conditioning_scale,
        seed=selected_seed,
        histories=histories,
        seed_metrics=seed_metrics,
        predictions_by_seed=predictions,
        representations_by_seed=representations,
        history_delta_mean=history_delta_mean,
        history_delta_scale=history_delta_scale,
    )


def parent_temperature_conditioning(cache: ShootingEmbeddingCache) -> np.ndarray:
    temperatures = np.asarray(
        [
            float(parent["temperature_K"])
            for parent in cache.manifest["snapshot"]["parents"]
        ],
        dtype=np.float64,
    )
    return np.repeat(temperatures, int(cache.parent_z.shape[1]))[:, None]


def _fit_vamp_baseline(
    cache: ShootingEmbeddingCache,
    targets: JointPathTargetData,
    *,
    horizon_ps: float,
    dimension: int,
    regularization: float,
    eigenvalue_cutoff: float,
) -> tuple[np.ndarray, LinearVAMP]:
    matches = np.flatnonzero(
        np.isclose(
            np.asarray(cache.horizons_ps, dtype=np.float64),
            float(horizon_ps),
            rtol=0.0,
            atol=1.0e-9,
        )
    )
    if matches.size != 1:
        raise ValueError(
            f"VAMP baseline horizon {horizon_ps:g} ps is unavailable in "
            f"{np.asarray(cache.horizons_ps).tolist()}."
        )
    branch_parent = np.asarray(cache.branch_parent_index, dtype=np.int64)
    optimization_branches = np.flatnonzero(
        np.isin(branch_parent, targets.parent_splits["optimization"])
    )
    present = np.asarray(
        cache.parent_local_z[branch_parent[optimization_branches]], dtype=np.float64
    ).reshape(-1, int(cache.parent_local_z.shape[-1]))
    future = np.asarray(
        cache.future_z[optimization_branches, int(matches[0])], dtype=np.float64
    ).reshape(present.shape)
    vamp = LinearVAMP(
        regularization=float(regularization),
        eigenvalue_cutoff=float(eigenvalue_cutoff),
    ).fit(present, future)
    flat_present = np.asarray(cache.parent_local_z, dtype=np.float64).reshape(
        -1, int(cache.parent_local_z.shape[-1])
    )
    return vamp.transform(flat_present, int(dimension)), vamp


def build_atlas_baseline_spaces(
    cache: ShootingEmbeddingCache,
    targets: JointPathTargetData,
    feature_variants: Mapping[str, np.ndarray],
    fitted: FittedPredictiveAtlas,
    *,
    marginal_prediction: np.ndarray | None,
    static_pca_dim: int,
    vamp_horizon_ps: float,
    vamp_dimension: int,
    vamp_regularization: float,
    vamp_eigenvalue_cutoff: float,
) -> tuple[dict[str, np.ndarray], LinearVAMP]:
    optimization_rows = targets.split_rows["optimization"]
    local = np.asarray(feature_variants["local"], dtype=np.float64)
    context = np.asarray(feature_variants["mean_std_context"], dtype=np.float64)
    local_pca = CovariancePCA.fit(
        local[optimization_rows], dimension=int(static_pca_dim)
    )
    context_pca = CovariancePCA.fit(
        context[optimization_rows], dimension=int(static_pca_dim)
    )
    vamp_values, vamp = _fit_vamp_baseline(
        cache,
        targets,
        horizon_ps=float(vamp_horizon_ps),
        dimension=int(vamp_dimension),
        regularization=float(vamp_regularization),
        eigenvalue_cutoff=float(vamp_eigenvalue_cutoff),
    )
    standardized_prediction = fitted.predictions_by_seed[fitted.seed]
    raw_prediction = (
        standardized_prediction * targets.target_scale + targets.target_mean
    )
    spaces = {
        f"local_pca_{int(static_pca_dim)}d": local_pca.transform(
            local, dimension=int(static_pca_dim)
        ),
        f"context_mean_std_pca_{int(static_pca_dim)}d": context_pca.transform(
            context, dimension=int(static_pca_dim)
        ),
        f"vamp_{int(vamp_dimension)}d_{float(vamp_horizon_ps):g}ps": vamp_values,
        f"atlas_latent_{fitted.model.latent_dim}d": fitted.representations_by_seed[
            fitted.seed
        ],
        "predicted_joint_path_mean_embedding": raw_prediction,
        "empirical_joint_path_mean_embedding_oracle": targets.empirical_mean_embedding,
    }
    if marginal_prediction is not None:
        spaces["predicted_marginal_mean_embeddings"] = np.asarray(
            marginal_prediction, dtype=np.float64
        )
    return spaces, vamp


def _candidate_sets(
    cache: ShootingEmbeddingCache,
    targets: JointPathTargetData,
    *,
    static_values: np.ndarray,
    crystalline_fraction_tolerance: float,
    static_caliper_candidates: int,
    neighbors: int,
) -> tuple[np.ndarray, list[np.ndarray], list[tuple[str, float, str]]]:
    parent_indices = targets.parent_splits["validation"]
    center_count = int(cache.parent_z.shape[1])
    selected_rows = _rows_for_parents(parent_indices, center_count)
    parent_for_position = np.repeat(parent_indices, center_count)
    parents = cache.manifest["snapshot"]["parents"]
    keys = [
        (
            str(parents[parent_index]["source_run_id"]),
            float(parents[parent_index]["temperature_K"]),
            str(parents[parent_index]["phase"]),
        )
        for parent_index in parent_for_position.tolist()
    ]
    static_selected = np.asarray(static_values, dtype=np.float64)[selected_rows]
    candidate_sets: list[np.ndarray] = []
    for query_position, (source_run, temperature, phase) in enumerate(keys):
        query_parent = int(parent_for_position[query_position])
        query_crystallinity = float(
            parents[query_parent]["source_crystalline_fraction"]
        )
        candidates = np.asarray(
            [
                position
                for position, (candidate_run, candidate_temperature, candidate_phase)
                in enumerate(keys)
                if candidate_run != source_run
                and candidate_temperature == temperature
                and candidate_phase == phase
                and abs(
                    float(
                        parents[int(parent_for_position[position])][
                            "source_crystalline_fraction"
                        ]
                    )
                    - query_crystallinity
                )
                <= float(crystalline_fraction_tolerance)
            ],
            dtype=np.int64,
        )
        if candidates.size < int(neighbors):
            raise RuntimeError(
                "Predictive-atlas matched retrieval has too few candidates: "
                f"query={query_position}, source={source_run}, temperature={temperature}, "
                f"phase={phase}, crystallinity={query_crystallinity:.6g}, "
                f"tolerance={crystalline_fraction_tolerance}, candidates={candidates.size}, "
                f"required={neighbors}."
            )
        caliper_size = min(int(static_caliper_candidates), int(candidates.size))
        distances = np.linalg.norm(
            static_selected[candidates] - static_selected[query_position], axis=1
        )
        retained = candidates[np.argsort(distances, kind="stable")[:caliper_size]]
        if retained.size < int(neighbors):
            raise RuntimeError(
                "The predictive-atlas static caliper retained fewer candidates than k: "
                f"retained={retained.size}, required={neighbors}."
            )
        candidate_sets.append(retained)
    return selected_rows, candidate_sets, keys


def _bootstrap_gain(
    candidate_distance: np.ndarray,
    baseline_distance: np.ndarray,
    source_ids: Sequence[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    unique_sources = sorted(set(source_ids))
    positions = {
        source: np.asarray(
            [index for index, value in enumerate(source_ids) if value == source],
            dtype=np.int64,
        )
        for source in unique_sources
    }
    rng = np.random.default_rng(int(seed))
    gains = np.empty(int(samples), dtype=np.float64)
    for bootstrap_index in range(int(samples)):
        sampled_sources = rng.choice(
            unique_sources, size=len(unique_sources), replace=True
        )
        rows = np.concatenate([positions[str(source)] for source in sampled_sources])
        gains[bootstrap_index] = 100.0 * (
            1.0
            - float(candidate_distance[rows].mean())
            / float(baseline_distance[rows].mean())
        )
    return {
        "gain_percent": float(
            100.0
            * (
                1.0
                - float(candidate_distance.mean()) / float(baseline_distance.mean())
            )
        ),
        "source_bootstrap_ci95_low": float(np.quantile(gains, 0.025)),
        "source_bootstrap_ci95_high": float(np.quantile(gains, 0.975)),
        "source_bootstrap_probability_positive": float(np.mean(gains > 0.0)),
        "source_bootstrap_samples": int(samples),
    }


def _exact_rbf_mmd(
    left: np.ndarray,
    right: np.ndarray,
    bandwidths: np.ndarray,
) -> tuple[float, float]:
    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        squared = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return np.mean(
            np.stack(
                [
                    np.exp(-squared / (2.0 * float(bandwidth) ** 2))
                    for bandwidth in bandwidths.tolist()
                ],
                axis=0,
            ),
            axis=0,
        )

    kxx = kernel(left, left)
    kyy = kernel(right, right)
    kxy = kernel(left, right)
    biased = float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())
    if left.shape[0] < 2 or right.shape[0] < 2:
        raise RuntimeError(
            "Unbiased exact MMD requires at least two paths in each empirical law."
        )
    unbiased = float(
        (kxx.sum() - np.trace(kxx)) / (left.shape[0] * (left.shape[0] - 1))
        + (kyy.sum() - np.trace(kyy)) / (right.shape[0] * (right.shape[0] - 1))
        - 2.0 * kxy.mean()
    )
    return biased, unbiased


def _exact_mmd_diagnostics(
    cache: ShootingEmbeddingCache,
    targets: JointPathTargetData,
    selected_rows: np.ndarray,
    candidate_sets: Sequence[np.ndarray],
    *,
    maximum_pairs: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(int(seed))
    pairs = [
        (query, int(candidate))
        for query, candidates in enumerate(candidate_sets)
        for candidate in candidates.tolist()
    ]
    selected_pair_indices = rng.choice(
        len(pairs), size=min(int(maximum_pairs), len(pairs)), replace=False
    )
    center_count = int(cache.parent_z.shape[1])
    branch_parent = targets.branch_parent_index
    exact_biased: list[float] = []
    exact_unbiased: list[float] = []
    approximate: list[float] = []
    teacher = targets.empirical_mean_embedding[selected_rows]
    for pair_index in selected_pair_indices.tolist():
        query_position, candidate_position = pairs[pair_index]
        query_row = int(selected_rows[query_position])
        candidate_row = int(selected_rows[candidate_position])
        query_parent, query_center = divmod(query_row, center_count)
        candidate_parent, candidate_center = divmod(candidate_row, center_count)
        query_paths = targets.branch_paths[
            branch_parent == query_parent, query_center
        ]
        candidate_paths = targets.branch_paths[
            branch_parent == candidate_parent, candidate_center
        ]
        biased, unbiased = _exact_rbf_mmd(
            query_paths,
            candidate_paths,
            targets.kernel.bandwidths,
        )
        exact_biased.append(biased)
        exact_unbiased.append(unbiased)
        approximate.append(
            float(
                np.sum(
                    (teacher[query_position] - teacher[candidate_position]) ** 2
                )
            )
        )
    exact_array = np.asarray(exact_biased)
    approximate_array = np.asarray(approximate)
    denominator = np.maximum(np.abs(exact_array), 1.0e-8)
    return {
        "sampled_pairs": int(exact_array.size),
        "rff_vs_exact_biased_spearman": float(
            spearmanr(approximate_array, exact_array).statistic
        ),
        "rff_vs_exact_biased_mean_absolute_error": float(
            np.mean(np.abs(approximate_array - exact_array))
        ),
        "rff_vs_exact_biased_mean_relative_absolute_error": float(
            np.mean(np.abs(approximate_array - exact_array) / denominator)
        ),
        "exact_biased_mean": float(exact_array.mean()),
        "exact_unbiased_mean": float(np.mean(exact_unbiased)),
        "rff_biased_mean": float(approximate_array.mean()),
    }


def evaluate_predictive_atlas(
    cache: ShootingEmbeddingCache,
    targets: JointPathTargetData,
    spaces: Mapping[str, np.ndarray],
    *,
    static_space_name: str,
    neighbors: int,
    static_caliper_candidates: int,
    crystalline_fraction_tolerance: float,
    pairwise_samples: int,
    exact_mmd_pairs: int,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[np.ndarray]]:
    if static_space_name not in spaces:
        raise KeyError(
            f"Atlas static caliper space {static_space_name!r} is unavailable; "
            f"spaces={sorted(spaces)}."
        )
    selected_rows, candidate_sets, keys = _candidate_sets(
        cache,
        targets,
        static_values=spaces[static_space_name],
        crystalline_fraction_tolerance=float(crystalline_fraction_tolerance),
        static_caliper_candidates=int(static_caliper_candidates),
        neighbors=int(neighbors),
    )
    teacher = np.asarray(targets.empirical_mean_embedding)[selected_rows]
    rng = np.random.default_rng(int(seed))
    random_neighbors = np.stack(
        [
            rng.choice(candidates, size=int(neighbors), replace=False)
            for candidates in candidate_sets
        ],
        axis=0,
    )
    random_distance = np.linalg.norm(
        teacher[random_neighbors] - teacher[:, None, :], axis=2
    ).mean(axis=1)
    results: dict[str, Any] = {}
    query_distances: dict[str, np.ndarray] = {}
    neighbor_indices: dict[str, np.ndarray] = {}
    for name, full_values in spaces.items():
        values = np.asarray(full_values, dtype=np.float64)[selected_rows]
        selected = np.empty((selected_rows.size, int(neighbors)), dtype=np.int64)
        for query_position, candidates in enumerate(candidate_sets):
            search = NearestNeighbors(
                n_neighbors=int(neighbors), metric="euclidean", algorithm="brute"
            ).fit(values[candidates])
            local = search.kneighbors(
                values[query_position : query_position + 1], return_distance=False
            )[0]
            selected[query_position] = candidates[local]
        future_distance = np.linalg.norm(
            teacher[selected] - teacher[:, None, :], axis=2
        ).mean(axis=1)
        query_distances[name] = future_distance
        neighbor_indices[name] = selected
        results[name] = {
            "queries": int(selected_rows.size),
            "neighbors": int(neighbors),
            "mean_heldout_empirical_mmd_distance": float(future_distance.mean()),
            "sem_heldout_empirical_mmd_distance": float(
                future_distance.std(ddof=1) / np.sqrt(future_distance.size)
            ),
            "matched_random_mean_distance": float(random_distance.mean()),
            "distance_over_matched_random": float(
                future_distance.mean() / random_distance.mean()
            ),
            "minimum_candidates_after_caliper": int(
                min(value.size for value in candidate_sets)
            ),
            "maximum_candidates_after_caliper": int(
                max(value.size for value in candidate_sets)
            ),
        }

    baseline_distance = query_distances[static_space_name]
    source_ids = [key[0] for key in keys]
    for name, distances in query_distances.items():
        results[name]["gain_over_static_caliper_baseline"] = _bootstrap_gain(
            distances,
            baseline_distance,
            source_ids,
            samples=int(bootstrap_samples),
            seed=int(seed) + 17,
        )

    pair_pool = [
        (query, int(candidate))
        for query, candidates in enumerate(candidate_sets)
        for candidate in candidates.tolist()
    ]
    sampled_pair_indices = rng.choice(
        len(pair_pool),
        size=min(int(pairwise_samples), len(pair_pool)),
        replace=False,
    )
    sampled_pairs = np.asarray(
        [pair_pool[index] for index in sampled_pair_indices.tolist()], dtype=np.int64
    )
    teacher_distance = np.linalg.norm(
        teacher[sampled_pairs[:, 0]] - teacher[sampled_pairs[:, 1]], axis=1
    )
    distance_correlations: dict[str, Any] = {}
    for name, full_values in spaces.items():
        values = np.asarray(full_values, dtype=np.float64)[selected_rows]
        present_distance = np.linalg.norm(
            values[sampled_pairs[:, 0]] - values[sampled_pairs[:, 1]], axis=1
        )
        distance_correlations[name] = {
            "spearman_with_heldout_empirical_mmd": float(
                spearmanr(present_distance, teacher_distance).statistic
            ),
            "sampled_pairs": int(sampled_pairs.shape[0]),
        }
    metrics = {
        "scientific_contract": {
            "query_split": "validation source runs only",
            "candidate_filter": (
                "different source run, exact temperature and parent phase, global "
                "crystalline-fraction tolerance, then a static-GeoFrame PCA caliper"
            ),
            "teacher_distance": (
                "Euclidean distance between empirical RFF means of complete same-branch "
                "future paths; validation branches are unseen during fitting"
            ),
            "uncertainty_unit": "source MD run",
        },
        "retrieval": results,
        "distance_correlations": distance_correlations,
        "rff_approximation": _exact_mmd_diagnostics(
            cache,
            targets,
            selected_rows,
            candidate_sets,
            maximum_pairs=int(exact_mmd_pairs),
            seed=int(seed) + 31,
        ),
    }
    arrays = {
        "validation_rows": selected_rows,
        "random_future_distance": random_distance,
        "sampled_pair_positions": sampled_pairs,
        "sampled_pair_teacher_distance": teacher_distance,
    }
    for name, distances in query_distances.items():
        arrays[f"query_future_distance__{name}"] = distances
        arrays[f"neighbors__{name}"] = neighbor_indices[name]
    return metrics, arrays, candidate_sets


def select_atlas_witnesses(
    cache: ShootingEmbeddingCache,
    targets: JointPathTargetData,
    spaces: Mapping[str, np.ndarray],
    evaluation_arrays: Mapping[str, np.ndarray],
    *,
    static_space_name: str,
    atlas_space_name: str,
    count: int,
) -> list[dict[str, Any]]:
    selected_rows = np.asarray(evaluation_arrays["validation_rows"], dtype=np.int64)
    sampled_pairs = np.asarray(
        evaluation_arrays["sampled_pair_positions"], dtype=np.int64
    )
    teacher_distance = np.asarray(
        evaluation_arrays["sampled_pair_teacher_distance"], dtype=np.float64
    )
    static = np.asarray(spaces[static_space_name], dtype=np.float64)[selected_rows]
    atlas = np.asarray(spaces[atlas_space_name], dtype=np.float64)[selected_rows]
    static_distance = np.linalg.norm(
        static[sampled_pairs[:, 0]] - static[sampled_pairs[:, 1]], axis=1
    )
    atlas_distance = np.linalg.norm(
        atlas[sampled_pairs[:, 0]] - atlas[sampled_pairs[:, 1]], axis=1
    )
    static_threshold = float(np.quantile(static_distance, 0.25))
    eligible = np.flatnonzero(static_distance <= static_threshold)
    score = teacher_distance[eligible] * (
        atlas_distance[eligible] / np.maximum(static_distance[eligible], 1.0e-8)
    )
    selected = eligible[np.argsort(score, kind="stable")[-int(count) :][::-1]]
    parents = cache.manifest["snapshot"]["parents"]
    center_count = int(cache.parent_z.shape[1])
    atom_ids = np.asarray(cache.atom_ids, dtype=np.int64)
    branch_parent = targets.branch_parent_index
    predicted = np.asarray(spaces[atlas_space_name], dtype=np.float64)
    witnesses: list[dict[str, Any]] = []
    for pair_index in selected.tolist():
        left_position, right_position = sampled_pairs[pair_index]
        left_row = int(selected_rows[left_position])
        right_row = int(selected_rows[right_position])
        left_parent, left_center = divmod(left_row, center_count)
        right_parent, right_center = divmod(right_row, center_count)
        direction = predicted[left_row] - predicted[right_row]
        left_branches = np.flatnonzero(branch_parent == left_parent)
        right_branches = np.flatnonzero(branch_parent == right_parent)
        branch_indices = np.concatenate([left_branches, right_branches])
        features = np.concatenate(
            [
                targets.branch_features[left_branches, left_center],
                targets.branch_features[right_branches, right_center],
            ],
            axis=0,
        )
        witness_scores = features @ direction
        maximum_index = int(np.argmax(witness_scores))
        minimum_index = int(np.argmin(witness_scores))
        witnesses.append(
            {
                "left": {
                    "row": left_row,
                    "parent_index": left_parent,
                    "parent_id": str(parents[left_parent]["parent_id"]),
                    "atom_id": int(atom_ids[left_center]),
                },
                "right": {
                    "row": right_row,
                    "parent_index": right_parent,
                    "parent_id": str(parents[right_parent]["parent_id"]),
                    "atom_id": int(atom_ids[right_center]),
                },
                "static_distance": float(static_distance[pair_index]),
                "atlas_distance": float(atlas_distance[pair_index]),
                "heldout_empirical_mmd_distance": float(
                    teacher_distance[pair_index]
                ),
                "positive_witness": {
                    "branch_cache_index": int(branch_indices[maximum_index]),
                    "score": float(witness_scores[maximum_index]),
                },
                "negative_witness": {
                    "branch_cache_index": int(branch_indices[minimum_index]),
                    "score": float(witness_scores[minimum_index]),
                },
            }
        )
    return witnesses


def compute_pullback_spectrum(
    fitted: FittedPredictiveAtlas,
    *,
    rows: np.ndarray,
    device: str,
    batch_size: int,
    relative_eigenvalue_cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    model = fitted.model.to(torch.device(device)).eval()
    latent = torch.from_numpy(
        fitted.representations_by_seed[fitted.seed][rows].astype(np.float32)
    ).to(torch.device(device))
    jacobian_function = torch.func.jacfwd(model.decode)
    eigenvalues: list[np.ndarray] = []
    for start in range(0, int(latent.shape[0]), int(batch_size)):
        jacobian = torch.func.vmap(jacobian_function)(
            latent[start : start + int(batch_size)]
        )
        metric = torch.einsum("bmd,bme->bde", jacobian, jacobian)
        eigenvalues.append(torch.linalg.eigvalsh(metric).cpu().detach().numpy())
    model.cpu()
    values = np.concatenate(eigenvalues)
    maximum = np.maximum(values[:, -1], 1.0e-12)
    effective_rank = np.sum(
        values > float(relative_eigenvalue_cutoff) * maximum[:, None], axis=1
    )
    return values, effective_rank.astype(np.int64)


def save_path_kernel(targets: JointPathTargetData, path: str | Path) -> None:
    kernel = targets.kernel
    np.savez(
        Path(path),
        selected_horizon_indices=kernel.selected_horizon_indices,
        selected_horizons_ps=kernel.selected_horizons_ps,
        delta_mean=kernel.delta_mean,
        delta_scale=kernel.delta_scale,
        horizon_weights=kernel.horizon_weights,
        median_distance=np.asarray(kernel.median_distance),
        bandwidths=kernel.bandwidths,
        frequencies=kernel.frequencies,
        phases=kernel.phases,
        target_mean=targets.target_mean,
        target_scale=targets.target_scale,
    )


__all__ = [
    "FittedPredictiveAtlas",
    "JointPathTargetData",
    "PathKernelParameters",
    "PredictiveAtlas",
    "build_atlas_baseline_spaces",
    "compute_pullback_spectrum",
    "evaluate_predictive_atlas",
    "fit_predictive_atlas",
    "parent_temperature_conditioning",
    "prepare_joint_path_target_data",
    "prepare_joint_path_target_data_from_kernel",
    "random_fourier_path_features",
    "save_path_kernel",
    "select_atlas_witnesses",
]
