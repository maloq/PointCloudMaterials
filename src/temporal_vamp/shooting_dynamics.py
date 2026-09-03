"""Position-history and momentum upper bounds for shooting prediction."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors

from src.data_utils.shooting_binary import ShootingBinaryTrajectory
from src.data_utils.shooting_dataset import (
    ShootingCampaignSnapshot,
    ShootingPositionFrame,
    build_periodic_environment_batch,
    resolve_shooting_trajectory_path,
    shooting_snapshot_sha256,
)
from src.data_utils.temporal_lammps_binary import TemporalLAMMPSBinaryTrajectory
from src.temporal_vamp.embeddings import FrozenEncoder
from src.temporal_vamp.evaluation import CovariancePCA
from src.temporal_vamp.shooting_context import ShootingContextTokenCache
from src.temporal_vamp.shooting_distribution import (
    DistributionalTargetData,
    _random_fourier_features,
)
from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache
from src.temporal_vamp.shooting_predictor import _future_neighbor_metrics
from src.temporal_vamp.shooting_spatial import SpatialTokenData


VELOCITY_TOKEN_FEATURE_NAMES = (
    "center_speed",
    "local_mean_velocity_norm",
    "neighbor_speed_mean",
    "neighbor_speed_std",
    "relative_speed_mean",
    "relative_speed_std",
    "relative_speed_max",
    "radial_relative_mean",
    "radial_relative_std",
    "radial_relative_rms",
    "radial_relative_abs_mean",
    "tangential_relative_mean",
    "tangential_relative_std",
    "tangential_relative_rms",
    "center_radial_mean",
    "center_radial_std",
    "center_radial_rms",
    "velocity_covariance_trace",
    "velocity_covariance_frobenius",
    "velocity_covariance_determinant",
    "position_velocity_trace",
    "position_velocity_frobenius",
    "position_velocity_determinant",
    "angular_momentum_mean_norm",
    "angular_momentum_net_norm",
)


@dataclass(frozen=True)
class ShootingDynamicalFeatureCache:
    path: Path
    manifest: dict[str, Any]
    previous_token_z: np.ndarray
    context_center_atom_ids: np.ndarray
    velocity_features: np.ndarray
    branch_parent_index: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "ShootingDynamicalFeatureCache":
        root = Path(path).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Shooting dynamics manifest is missing: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        arrays = {
            name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in (
                "previous_token_z",
                "context_center_atom_ids",
                "velocity_features",
                "branch_parent_index",
            )
        }
        for name, values in arrays.items():
            expected = tuple(int(value) for value in manifest["array_shapes"][name])
            if values.shape != expected:
                raise RuntimeError(
                    f"Shooting dynamics array shape mismatch for {name}: "
                    f"expected={expected}, observed={values.shape}, root={root}."
                )
        return cls(path=root, manifest=manifest, **arrays)


@dataclass(frozen=True)
class SelectedRidgeResidual:
    prediction: np.ndarray
    residual_prediction: np.ndarray
    features: np.ndarray
    selected_dimension: int
    selected_alpha: float
    selection_mse: float
    validation_mse: float
    validation_r2: float
    preprocessing: dict[str, np.ndarray]
    coefficients: np.ndarray
    intercept: np.ndarray


@dataclass(frozen=True)
class DynamicalAblationResult:
    metrics: dict[str, Any]
    arrays: dict[str, np.ndarray]
    model_arrays: dict[str, np.ndarray]


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _position_frame(
    trajectory: TemporalLAMMPSBinaryTrajectory, frame_index: int
) -> ShootingPositionFrame:
    return ShootingPositionFrame(
        timestep=int(trajectory.timesteps[int(frame_index)]),
        atom_ids=trajectory.atom_ids,
        atom_types=trajectory.atom_types,
        positions=trajectory.positions[int(frame_index)],
        box_low=trajectory.box_low[int(frame_index)],
        box_high=trajectory.box_high[int(frame_index)],
    )


@torch.inference_mode()
def _encode_chunks(
    encoder: FrozenEncoder, points: torch.Tensor, *, chunk_size: int
) -> np.ndarray:
    chunks = [
        encoder.encode(points[start : start + int(chunk_size)]).cpu()
        for start in range(0, int(points.shape[0]), int(chunk_size))
    ]
    if not chunks:
        raise RuntimeError("Dynamical feature extraction received no point clouds.")
    return torch.cat(chunks).numpy().astype(np.float32, copy=False)


def invariant_velocity_token_features(
    relative_positions: np.ndarray,
    neighbor_velocities: np.ndarray,
    center_velocities: np.ndarray,
) -> np.ndarray:
    """Rotation-invariant velocity/structure couplings for fixed local tokens."""

    positions = np.asarray(relative_positions, dtype=np.float64)
    velocities = np.asarray(neighbor_velocities, dtype=np.float64)
    centers = np.asarray(center_velocities, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"relative_positions must have shape (M,N,3), got {positions.shape}.")
    if velocities.shape != positions.shape or centers.shape != (positions.shape[0], 3):
        raise ValueError(
            f"Velocity shapes must align with positions: positions={positions.shape}, "
            f"neighbors={velocities.shape}, centers={centers.shape}."
        )
    distance = np.linalg.norm(positions, axis=2)
    mask = distance > 1.0e-10
    count = mask.sum(axis=1)
    if np.any(count != positions.shape[1] - 1):
        raise RuntimeError(
            "Every repository local token must contain exactly one zero-distance center: "
            f"counts={np.unique(count, return_counts=True)}."
        )
    denominator = count.astype(np.float64)
    unit = np.zeros_like(positions)
    unit[mask] = positions[mask] / distance[mask, None]
    relative_velocity = velocities - centers[:, None, :]
    neighbor_speed = np.linalg.norm(velocities, axis=2)
    relative_speed = np.linalg.norm(relative_velocity, axis=2)
    radial = np.sum(relative_velocity * unit, axis=2)
    tangential = np.sqrt(np.maximum(relative_speed**2 - radial**2, 0.0))
    center_radial = np.sum(centers[:, None, :] * unit, axis=2)

    def mean(values: np.ndarray) -> np.ndarray:
        return np.sum(np.where(mask, values, 0.0), axis=1) / denominator

    def std(values: np.ndarray) -> np.ndarray:
        average = mean(values)
        return np.sqrt(
            np.sum(np.where(mask, (values - average[:, None]) ** 2, 0.0), axis=1)
            / denominator
        )

    def rms(values: np.ndarray) -> np.ndarray:
        return np.sqrt(mean(values**2))

    maximum_relative_speed = np.max(
        np.where(mask, relative_speed, -np.inf), axis=1
    )
    masked_relative = relative_velocity * mask[..., None]
    velocity_covariance = np.einsum(
        "mni,mnj->mij", masked_relative, masked_relative
    ) / denominator[:, None, None]
    position_velocity = np.einsum(
        "mni,mnj->mij", positions * mask[..., None], masked_relative
    ) / denominator[:, None, None]
    angular = np.cross(positions, relative_velocity) * mask[..., None]
    angular_norm = np.linalg.norm(angular, axis=2)
    local_mean_velocity = np.sum(
        velocities * mask[..., None], axis=1
    ) / denominator[:, None]
    features = np.stack(
        [
            np.linalg.norm(centers, axis=1),
            np.linalg.norm(local_mean_velocity, axis=1),
            mean(neighbor_speed),
            std(neighbor_speed),
            mean(relative_speed),
            std(relative_speed),
            maximum_relative_speed,
            mean(radial),
            std(radial),
            rms(radial),
            mean(np.abs(radial)),
            mean(tangential),
            std(tangential),
            rms(tangential),
            mean(center_radial),
            std(center_radial),
            rms(center_radial),
            np.trace(velocity_covariance, axis1=1, axis2=2),
            np.linalg.norm(velocity_covariance, axis=(1, 2)),
            np.linalg.det(velocity_covariance),
            np.trace(position_velocity, axis1=1, axis2=2),
            np.linalg.norm(position_velocity, axis=(1, 2)),
            np.linalg.det(position_velocity),
            mean(angular_norm),
            np.linalg.norm(
                np.sum(angular, axis=1) / denominator[:, None], axis=1
            ),
        ],
        axis=1,
    ).astype(np.float32)
    if not np.isfinite(features).all():
        raise RuntimeError("Invariant velocity descriptors contain non-finite values.")
    return features


def _token_neighbor_geometry(
    frame: ShootingPositionFrame,
    token_atom_ids: np.ndarray,
    *,
    num_points: int,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat_ids = np.asarray(token_atom_ids, dtype=np.int64).reshape(-1)
    center_indices = flat_ids - 1
    positions = np.asarray(frame.positions, dtype=np.float32)
    box_lengths = np.asarray(frame.box_lengths, dtype=np.float32)
    tree = cKDTree(positions, boxsize=box_lengths, balanced_tree=False)
    _, neighbor_indices = tree.query(positions[center_indices], k=int(num_points))
    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
    relative = positions[neighbor_indices] - positions[center_indices, None]
    relative -= box_lengths[None, None, :] * np.round(
        relative / box_lengths[None, None, :]
    )
    return (
        neighbor_indices,
        center_indices,
        (relative / float(radius)).astype(np.float32, copy=False),
    )


def _branch_path(snapshot: ShootingCampaignSnapshot, branch: dict[str, Any]) -> Path:
    root = (
        snapshot.root
        if len(snapshot.campaign_roots) == 1
        else Path(str(branch["campaign_root"]))
    )
    return resolve_shooting_trajectory_path(root, branch)


def extract_shooting_dynamical_feature_cache(
    snapshot: ShootingCampaignSnapshot,
    base_cache: ShootingEmbeddingCache,
    context_cache: ShootingContextTokenCache,
    *,
    encoder: FrozenEncoder,
    source_trajectory_root: str | Path,
    cache_path: str | Path,
    history_lag_frames: int,
    history_lag_ps: float,
    source_sample_interval_ps: float,
    num_points: int,
    radius: float,
    context_center_count: int,
    point_cloud_batch_size: int,
    force_recompute: bool,
) -> ShootingDynamicalFeatureCache:
    target = Path(cache_path).expanduser().resolve()
    source_root = Path(source_trajectory_root).expanduser().resolve()
    snapshot_hash = shooting_snapshot_sha256(snapshot)
    if snapshot_hash != str(base_cache.manifest["spec"]["snapshot_sha256"]):
        raise RuntimeError("Dynamics snapshot does not match the base embedding cache.")
    if snapshot_hash != str(context_cache.manifest["spec"]["snapshot_sha256"]):
        raise RuntimeError("Dynamics snapshot does not match the context-token cache.")
    if not np.isclose(
        int(history_lag_frames) * float(source_sample_interval_ps),
        float(history_lag_ps),
        atol=1.0e-9,
        rtol=0.0,
    ):
        raise ValueError(
            f"History lag contract is inconsistent: frames={history_lag_frames}, "
            f"interval_ps={source_sample_interval_ps}, lag_ps={history_lag_ps}."
        )
    checkpoint_stat = encoder.checkpoint_path.stat()
    spec = {
        "version": 2,
        "snapshot_sha256": snapshot_hash,
        "source_trajectory_root": str(source_root),
        "history_lag_frames": int(history_lag_frames),
        "history_lag_ps": float(history_lag_ps),
        "source_sample_interval_ps": float(source_sample_interval_ps),
        "checkpoint": str(encoder.checkpoint_path),
        "checkpoint_size_bytes": int(checkpoint_stat.st_size),
        "checkpoint_mtime_ns": int(checkpoint_stat.st_mtime_ns),
        "representation_source": encoder.representation_source,
        "center_atom_ids": np.asarray(base_cache.atom_ids, dtype=np.int64).tolist(),
        "num_points": int(num_points),
        "radius": float(radius),
        "context_center_count": int(context_center_count),
        "velocity_token_features": list(VELOCITY_TOKEN_FEATURE_NAMES),
        "velocity_aggregation": "central + satellite mean + satellite std",
        "current_reference": "exact shooting parent frame; source frame used only for history",
    }
    manifest_path = target / "manifest.json"
    if manifest_path.is_file() and not force_recompute:
        cached = ShootingDynamicalFeatureCache.load(target)
        if cached.manifest["spec"] != spec:
            raise RuntimeError(
                f"Dynamical feature cache specification changed at {target}; "
                "choose a new path or set force_recompute=true."
            )
        return cached
    if target.exists() and not force_recompute:
        raise RuntimeError(
            f"Dynamical cache exists without a complete manifest: {target}."
        )
    if force_recompute and target.exists():
        shutil.rmtree(target)
    shard_root = target.parent / f"{target.name}_shards"
    if force_recompute and shard_root.exists():
        shutil.rmtree(shard_root)
    parent_shards = shard_root / "parents"
    branch_shards = shard_root / "branches"
    parent_shards.mkdir(parents=True, exist_ok=True)
    branch_shards.mkdir(parents=True, exist_ok=True)
    spec_sha = _sha256_json(spec)
    atom_ids = np.asarray(base_cache.atom_ids, dtype=np.int64)
    frozen_current = np.concatenate(
        [
            np.asarray(base_cache.parent_local_z, dtype=np.float32)[:, :, None],
            np.asarray(context_cache.satellite_z, dtype=np.float32),
        ],
        axis=2,
    )
    branches_by_parent: dict[str, list[dict[str, Any]]] = {
        str(parent["parent_id"]): [] for parent in snapshot.parents
    }
    for branch in snapshot.branches:
        branches_by_parent[str(branch["parent_id"])].append(branch)
    parent_geometries: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    maximum_current_error = 0.0
    for parent_position, parent in enumerate(snapshot.parents):
        shard_path = parent_shards / f"parent_{parent_position:04d}.npz"
        source_path = source_root / str(parent["source_run_id"]) / "trajectory_binary_float32"
        trajectory = TemporalLAMMPSBinaryTrajectory.load(source_path)
        current_frame_index = int(parent["source_frame_index"])
        previous_frame_index = current_frame_index - int(history_lag_frames)
        if previous_frame_index < 0:
            raise RuntimeError(
                f"Parent={parent['parent_id']} has no requested history frame: "
                f"current={current_frame_index}, lag={history_lag_frames}."
            )
        source_current_frame = _position_frame(trajectory, current_frame_index)
        representative = sorted(
            branches_by_parent[str(parent["parent_id"])],
            key=lambda value: int(value["shot_index"]),
        )[0]
        shooting_trajectory = ShootingBinaryTrajectory.load(
            _branch_path(snapshot, representative)
        )
        current_frame = shooting_trajectory.load_position_frames([0])[0]
        source_position_error = float(
            np.max(
                np.abs(
                    np.asarray(source_current_frame.positions, dtype=np.float32)
                    - np.asarray(current_frame.positions, dtype=np.float32)
                )
            )
        )
        if source_position_error > 1.0e-4:
            raise RuntimeError(
                f"Shooting parent no longer matches its source frame for "
                f"parent={parent['parent_id']}: max_position_error={source_position_error}."
            )
        if int(trajectory.timesteps[current_frame_index]) - int(
            trajectory.timesteps[previous_frame_index]
        ) <= 0:
            raise RuntimeError(f"Source history timesteps are invalid for {source_path}.")
        if shard_path.is_file():
            with np.load(shard_path, allow_pickle=False) as payload:
                if str(payload["spec_sha256"].item()) != spec_sha:
                    raise RuntimeError(f"Parent dynamics shard changed: {shard_path}")
                context_ids = payload["context_center_atom_ids"].copy()
                maximum_current_error = max(
                    maximum_current_error, float(payload["current_embedding_max_abs_error"])
                )
        else:
            current = build_periodic_environment_batch(
                current_frame,
                center_atom_ids=atom_ids,
                num_points=int(num_points),
                radius=float(radius),
                spatial_context_center_count=int(context_center_count),
            )
            if current.context_points is None or current.context_center_atom_ids is None:
                raise RuntimeError(f"Current source frame returned no context for {parent['parent_id']}.")
            context_ids = current.context_center_atom_ids
            current_points = torch.cat(
                [current.points[:, None], current.context_points], dim=1
            )
            current_z = _encode_chunks(
                encoder,
                current_points.reshape(-1, int(num_points), 3),
                chunk_size=int(point_cloud_batch_size),
            ).reshape(atom_ids.size, int(context_center_count) + 1, -1)
            current_error = float(
                np.max(np.abs(current_z - frozen_current[parent_position]))
            )
            maximum_current_error = max(maximum_current_error, current_error)
            if current_error > 5.0e-4:
                raise RuntimeError(
                    f"Source history reconstruction disagrees with shooting tokens for "
                    f"parent={parent['parent_id']}: max_abs_error={current_error:.8g}."
                )
            token_ids = np.concatenate([atom_ids[:, None], context_ids], axis=1)
            previous = build_periodic_environment_batch(
                _position_frame(trajectory, previous_frame_index),
                center_atom_ids=token_ids.reshape(-1),
                num_points=int(num_points),
                radius=float(radius),
                spatial_context_center_count=0,
            )
            previous_z = _encode_chunks(
                encoder,
                previous.points,
                chunk_size=int(point_cloud_batch_size),
            ).reshape(atom_ids.size, int(context_center_count) + 1, -1)
            temporary = shard_path.with_suffix(".tmp.npz")
            np.savez(
                temporary,
                spec_sha256=np.asarray(spec_sha),
                previous_token_z=previous_z,
                context_center_atom_ids=context_ids,
                current_embedding_max_abs_error=np.asarray(current_error),
                source_position_max_abs_error=np.asarray(source_position_error),
            )
            os.replace(temporary, shard_path)
            print(
                f"[shooting-dynamics] history parent={parent_position + 1}/{len(snapshot.parents)}",
                flush=True,
            )
        token_ids = np.concatenate([atom_ids[:, None], context_ids], axis=1)
        parent_geometries[parent_position] = _token_neighbor_geometry(
            current_frame,
            token_ids,
            num_points=int(num_points),
            radius=float(radius),
        )

    branch_parent = np.asarray(base_cache.branch_parent_index, dtype=np.int64)
    if branch_parent.shape[0] != len(snapshot.branches):
        raise RuntimeError(
            f"Snapshot/cache branch mismatch: snapshot={len(snapshot.branches)}, "
            f"cache={branch_parent.shape[0]}."
        )
    feature_dim = 3 * len(VELOCITY_TOKEN_FEATURE_NAMES)
    for branch_position, branch in enumerate(snapshot.branches):
        shard_path = branch_shards / f"branch_{branch_position:05d}.npz"
        if shard_path.is_file():
            with np.load(shard_path, allow_pickle=False) as payload:
                if str(payload["spec_sha256"].item()) != spec_sha:
                    raise RuntimeError(f"Branch dynamics shard changed: {shard_path}")
            continue
        parent_position = int(branch_parent[branch_position])
        if str(branch["parent_id"]) != str(snapshot.parents[parent_position]["parent_id"]):
            raise RuntimeError(
                f"Branch ordering changed at position={branch_position}: "
                f"branch_parent={branch['parent_id']}, cache_parent={parent_position}."
            )
        trajectory = ShootingBinaryTrajectory.load(_branch_path(snapshot, branch))
        neighbor_indices, center_indices, relative = parent_geometries[parent_position]
        velocity = np.asarray(trajectory.velocities[0], dtype=np.float32)
        token_features = invariant_velocity_token_features(
            relative,
            velocity[neighbor_indices],
            velocity[center_indices],
        ).reshape(atom_ids.size, int(context_center_count) + 1, -1)
        aggregated = np.concatenate(
            [
                token_features[:, 0],
                token_features[:, 1:].mean(axis=1),
                token_features[:, 1:].std(axis=1),
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        if aggregated.shape != (atom_ids.size, feature_dim):
            raise RuntimeError(
                f"Velocity feature shape changed: {aggregated.shape}, expected={(atom_ids.size, feature_dim)}."
            )
        temporary = shard_path.with_suffix(".tmp.npz")
        np.savez(
            temporary,
            spec_sha256=np.asarray(spec_sha),
            velocity_features=aggregated,
            parent_index=np.asarray(parent_position, dtype=np.int64),
        )
        os.replace(temporary, shard_path)
        if (branch_position + 1) % 20 == 0 or branch_position + 1 == len(snapshot.branches):
            print(
                f"[shooting-dynamics] velocity branch={branch_position + 1}/{len(snapshot.branches)}",
                flush=True,
            )

    parent_count = len(snapshot.parents)
    center_count = int(atom_ids.size)
    token_count = int(context_center_count) + 1
    embedding_dim = int(base_cache.parent_local_z.shape[-1])
    branch_count = len(snapshot.branches)
    shapes = {
        "previous_token_z": (parent_count, center_count, token_count, embedding_dim),
        "context_center_atom_ids": (
            parent_count,
            center_count,
            int(context_center_count),
        ),
        "velocity_features": (branch_count, center_count, feature_dim),
        "branch_parent_index": (branch_count,),
    }
    dtypes = {
        "previous_token_z": np.float32,
        "context_center_atom_ids": np.int64,
        "velocity_features": np.float32,
        "branch_parent_index": np.int64,
    }
    building = target.parent / f".{target.name}.building-{os.getpid()}"
    if building.exists():
        raise FileExistsError(f"Dynamics cache build directory exists: {building}")
    building.mkdir(parents=True)
    arrays = {
        name: np.lib.format.open_memmap(
            building / f"{name}.npy", mode="w+", dtype=dtypes[name], shape=shape
        )
        for name, shape in shapes.items()
    }
    for parent_position in range(parent_count):
        with np.load(
            parent_shards / f"parent_{parent_position:04d}.npz", allow_pickle=False
        ) as payload:
            arrays["previous_token_z"][parent_position] = payload["previous_token_z"]
            arrays["context_center_atom_ids"][parent_position] = payload[
                "context_center_atom_ids"
            ]
    for branch_position in range(branch_count):
        with np.load(
            branch_shards / f"branch_{branch_position:05d}.npz", allow_pickle=False
        ) as payload:
            parent_index = int(payload["parent_index"])
            if parent_index != int(branch_parent[branch_position]):
                raise RuntimeError(f"Branch parent index changed in shard={branch_position}.")
            arrays["velocity_features"][branch_position] = payload["velocity_features"]
    arrays["branch_parent_index"][:] = branch_parent
    for values in arrays.values():
        values.flush()
    manifest = {
        "state": "complete",
        "spec": spec,
        "array_shapes": {name: list(shape) for name, shape in shapes.items()},
        "current_embedding_max_abs_error": maximum_current_error,
    }
    with (building / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(building, target)
    return ShootingDynamicalFeatureCache.load(target)


def _prediction_metrics(
    prediction: np.ndarray, target: np.ndarray, rows: np.ndarray
) -> tuple[float, float]:
    residual = target[rows] - prediction[rows]
    mse = float(np.mean(residual**2))
    denominator = float(np.sum((target[rows] - target[rows].mean(axis=0)) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / denominator)
    return mse, r2


def fit_selected_ridge_residual(
    raw_features: np.ndarray,
    base_prediction: np.ndarray,
    target: np.ndarray,
    *,
    optimization_rows: np.ndarray,
    selection_rows: np.ndarray,
    validation_rows: np.ndarray,
    dimensions: Sequence[int],
    alphas: Sequence[float],
) -> SelectedRidgeResidual:
    values = np.asarray(raw_features, dtype=np.float64)
    base = np.asarray(base_prediction, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    feature_mean = values[optimization_rows].mean(axis=0)
    feature_scale = values[optimization_rows].std(axis=0)
    feature_scale = np.where(feature_scale <= 1.0e-10, 1.0, feature_scale)
    standardized = (values - feature_mean) / feature_scale
    maximum_dimension = min(max(int(value) for value in dimensions), standardized.shape[1])
    pca = CovariancePCA.fit(
        standardized[optimization_rows], dimension=maximum_dimension
    )
    projected = pca.transform(standardized, dimension=maximum_dimension)
    residual_target = target_values - base
    candidates: list[tuple[float, int, float, Ridge]] = []
    for raw_dimension in dimensions:
        dimension = min(int(raw_dimension), maximum_dimension)
        for raw_alpha in alphas:
            alpha = float(raw_alpha)
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(projected[optimization_rows, :dimension], residual_target[optimization_rows])
            selection_prediction = base[selection_rows] + model.predict(
                projected[selection_rows, :dimension]
            )
            selection_mse = float(
                np.mean((selection_prediction - target_values[selection_rows]) ** 2)
            )
            candidates.append((selection_mse, dimension, alpha, model))
    selection_mse, dimension, alpha, model = min(
        candidates, key=lambda value: (value[0], value[1], value[2])
    )
    residual_prediction = model.predict(projected[:, :dimension])
    prediction = base + residual_prediction
    validation_mse, validation_r2 = _prediction_metrics(
        prediction, target_values, validation_rows
    )
    return SelectedRidgeResidual(
        prediction=prediction,
        residual_prediction=residual_prediction,
        features=projected[:, :dimension],
        selected_dimension=dimension,
        selected_alpha=alpha,
        selection_mse=selection_mse,
        validation_mse=validation_mse,
        validation_r2=validation_r2,
        preprocessing={
            "feature_mean": feature_mean,
            "feature_scale": feature_scale,
            "pca_mean": pca.mean_,
            "pca_components": pca.components_,
            "pca_eigenvalues": pca.eigenvalues_,
        },
        coefficients=np.asarray(model.coef_),
        intercept=np.asarray(model.intercept_),
    )


def individual_branch_signatures(
    cache: ShootingEmbeddingCache,
    targets: DistributionalTargetData,
) -> np.ndarray:
    branch_parent = np.asarray(cache.branch_parent_index, dtype=np.int64)
    current = np.asarray(cache.parent_local_z[branch_parent], dtype=np.float64)
    future = np.asarray(
        cache.future_z[:, targets.selected_horizon_indices], dtype=np.float64
    )
    blocks: list[np.ndarray] = []
    for horizon_index, parameters in enumerate(targets.horizon_parameters):
        delta = future[:, horizon_index] - current
        standardized = (delta - parameters.delta_mean) / parameters.delta_scale
        projected = parameters.pca.transform(
            standardized.reshape(-1, standardized.shape[-1]),
            dimension=parameters.pca.components_.shape[1],
        ).reshape(*standardized.shape[:2], -1)
        blocks.append(
            _random_fourier_features(
                projected, parameters.frequencies, parameters.phases
            )
        )
    signatures = np.stack(blocks, axis=1)
    parent_count = int(cache.parent_z.shape[0])
    center_count = int(cache.parent_z.shape[1])
    reconstructed = np.empty(
        (
            parent_count,
            center_count,
            signatures.shape[1],
            signatures.shape[-1],
        ),
        dtype=np.float64,
    )
    for parent_index in range(parent_count):
        reconstructed[parent_index] = signatures[branch_parent == parent_index].mean(
            axis=0
        ).transpose(1, 0, 2)
    maximum_error = float(
        np.max(
            np.abs(
                reconstructed.reshape(parent_count * center_count, signatures.shape[1], -1)
                - targets.distribution_signature
            )
        )
    )
    if maximum_error > 1.0e-10:
        raise RuntimeError(
            f"Individual branch signatures do not average to the parent target: {maximum_error}."
        )
    return signatures


def _branch_rows_for_parents(
    branch_parent: np.ndarray, parent_indices: np.ndarray, center_count: int
) -> np.ndarray:
    branches = np.flatnonzero(np.isin(branch_parent, parent_indices))
    return (
        branches[:, None] * int(center_count)
        + np.arange(int(center_count), dtype=np.int64)[None, :]
    ).reshape(-1)


def _branch_future_neighbor_metrics(
    spaces: Mapping[str, np.ndarray],
    future: np.ndarray,
    cache: ShootingEmbeddingCache,
    validation_parents: np.ndarray,
    *,
    neighbors: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    branch_parent = np.asarray(cache.branch_parent_index, dtype=np.int64)
    center_count = int(cache.parent_z.shape[1])
    rows = _branch_rows_for_parents(branch_parent, validation_parents, center_count)
    branch_for_row = rows // center_count
    parent_for_row = branch_parent[branch_for_row]
    parents = cache.manifest["snapshot"]["parents"]
    keys = [
        (
            str(parents[parent]["source_run_id"]),
            float(parents[parent]["temperature_K"]),
            str(parents[parent]["phase"]),
        )
        for parent in parent_for_row.tolist()
    ]
    grouped: dict[tuple[str, float, str], list[int]] = defaultdict(list)
    for position, key in enumerate(keys):
        grouped[key].append(position)
    rng = np.random.default_rng(int(seed))
    candidate_sets: list[np.ndarray] = []
    random_neighbors = np.empty((rows.size, int(neighbors)), dtype=np.int64)
    for position, (source_run, temperature, phase) in enumerate(keys):
        candidates = np.asarray(
            [
                candidate
                for candidate, (run, temp, candidate_phase) in enumerate(keys)
                if run != source_run and temp == temperature and candidate_phase == phase
            ],
            dtype=np.int64,
        )
        if candidates.size < int(neighbors):
            raise RuntimeError(
                f"Branch retrieval has too few cross-run candidates at query={position}: "
                f"observed={candidates.size}, required={neighbors}."
            )
        candidate_sets.append(candidates)
        random_neighbors[position] = rng.choice(candidates, size=int(neighbors), replace=False)
    selected_future = np.asarray(future)[rows]
    random_distance = np.linalg.norm(
        selected_future[random_neighbors] - selected_future[:, None], axis=2
    ).mean(axis=1)
    output: dict[str, dict[str, float | int]] = {}
    for name, full_values in spaces.items():
        values = np.asarray(full_values)[rows]
        selected = np.empty((rows.size, int(neighbors)), dtype=np.int64)
        for key, position_list in grouped.items():
            source_run, temperature, phase = key
            positions = np.asarray(position_list, dtype=np.int64)
            candidates = np.asarray(
                [
                    candidate
                    for candidate, (run, temp, candidate_phase) in enumerate(keys)
                    if run != source_run and temp == temperature and candidate_phase == phase
                ],
                dtype=np.int64,
            )
            search = NearestNeighbors(
                n_neighbors=int(neighbors), metric="euclidean", algorithm="brute"
            ).fit(values[candidates])
            local = search.kneighbors(values[positions], return_distance=False)
            selected[positions] = candidates[local]
        distance = np.linalg.norm(
            selected_future[selected] - selected_future[:, None], axis=2
        ).mean(axis=1)
        output[name] = {
            "queries": int(rows.size),
            "neighbors": int(neighbors),
            "mean_individual_future_distance": float(distance.mean()),
            "sem_individual_future_distance": float(
                distance.std(ddof=1) / np.sqrt(distance.size)
            ),
            "matched_random_mean_individual_future_distance": float(
                random_distance.mean()
            ),
            "distance_over_matched_random": float(
                distance.mean() / random_distance.mean()
            ),
            "candidate_count": int(min(map(len, candidate_sets))),
        }
    return output


def evaluate_dynamical_ablation(
    cache: ShootingEmbeddingCache,
    targets: DistributionalTargetData,
    frozen_tokens: SpatialTokenData,
    dynamics: ShootingDynamicalFeatureCache,
    *,
    ablation5_arrays_path: str | Path,
    history_pca_dimensions: Sequence[int],
    velocity_pca_dimensions: Sequence[int],
    ridge_alphas: Sequence[float],
    neighbors: int,
    seed: int,
) -> DynamicalAblationResult:
    with np.load(Path(ablation5_arrays_path), allow_pickle=False) as payload:
        base_standardized = payload["standardized_prediction"].copy()
        base_raw = payload["prediction"].copy()
        base_representation = payload["representation"].copy()
    parent_count, center_count = cache.parent_local_z.shape[:2]
    row_count = int(parent_count * center_count)
    if base_standardized.shape != targets.target_modes.shape:
        raise RuntimeError(
            f"Ablation-5 prediction shape changed: {base_standardized.shape} "
            f"versus target={targets.target_modes.shape}."
        )
    current_tokens = np.asarray(frozen_tokens.embeddings, dtype=np.float64).reshape(
        parent_count, center_count, frozen_tokens.embeddings.shape[1], -1
    )
    history_delta = current_tokens - np.asarray(dynamics.previous_token_z, dtype=np.float64)
    history_raw = history_delta.reshape(row_count, -1)
    history = fit_selected_ridge_residual(
        history_raw,
        base_standardized,
        targets.target_modes,
        optimization_rows=targets.split_rows["optimization"],
        selection_rows=targets.split_rows["selection"],
        validation_rows=targets.split_rows["validation"],
        dimensions=history_pca_dimensions,
        alphas=ridge_alphas,
    )
    base_parent_validation_mse, base_parent_validation_r2 = _prediction_metrics(
        base_standardized, targets.target_modes, targets.split_rows["validation"]
    )
    history_raw_prediction = (
        history.prediction * targets.target_scale + targets.target_mean
    )
    parent_spaces = {
        "ablation5_predicted_kernel_mean": base_raw,
        "history_residual_predicted_kernel_mean": history_raw_prediction,
    }
    parent_retrieval: dict[str, Any] = {}
    signature_dim = int(targets.distribution_signature.shape[-1])
    for horizon_index, horizon in enumerate(targets.selected_horizons_ps.tolist()):
        sliced_spaces = {
            name: value.reshape(row_count, len(targets.selected_horizons_ps), signature_dim)[
                :, horizon_index
            ]
            for name, value in parent_spaces.items()
        }
        result = _future_neighbor_metrics(
            sliced_spaces,
            targets.distribution_signature[:, horizon_index],
            cache,
            targets.parent_splits["validation"],
            neighbors=int(neighbors),
            seed=int(seed),
        )
        baseline = float(
            result["ablation5_predicted_kernel_mean"]["mean_ensemble_future_distance"]
        )
        result["gain_over_ablation5_percent"] = {
            name: float(
                100.0
                * (1.0 - float(value["mean_ensemble_future_distance"]) / baseline)
            )
            for name, value in result.items()
        }
        parent_retrieval[f"{float(horizon):g}ps"] = result
    result = _future_neighbor_metrics(
        parent_spaces,
        targets.distribution_signature.reshape(row_count, -1),
        cache,
        targets.parent_splits["validation"],
        neighbors=int(neighbors),
        seed=int(seed),
    )
    baseline = float(
        result["ablation5_predicted_kernel_mean"]["mean_ensemble_future_distance"]
    )
    result["gain_over_ablation5_percent"] = {
        name: float(100.0 * (1.0 - float(value["mean_ensemble_future_distance"]) / baseline))
        for name, value in result.items()
    }
    parent_retrieval["all_horizons"] = result

    branch_signatures = individual_branch_signatures(cache, targets)
    branch_count = int(branch_signatures.shape[0])
    branch_target_raw = branch_signatures.transpose(0, 2, 1, 3).reshape(
        branch_count * center_count, -1
    )
    branch_target = (
        (branch_target_raw - targets.target_mean) / targets.target_scale
    )
    branch_parent = np.asarray(cache.branch_parent_index, dtype=np.int64)
    parent_rows = (
        branch_parent[:, None] * center_count
        + np.arange(center_count, dtype=np.int64)[None]
    ).reshape(-1)
    base_branch = base_standardized[parent_rows]
    history_branch = history.prediction[parent_rows]
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
        base_branch,
        branch_target,
        optimization_rows=optimization_rows,
        selection_rows=selection_rows,
        validation_rows=validation_rows,
        dimensions=velocity_pca_dimensions,
        alphas=ridge_alphas,
    )
    combined = fit_selected_ridge_residual(
        velocity_raw,
        history_branch,
        branch_target,
        optimization_rows=optimization_rows,
        selection_rows=selection_rows,
        validation_rows=validation_rows,
        dimensions=velocity_pca_dimensions,
        alphas=ridge_alphas,
    )
    branch_prediction_metrics: dict[str, dict[str, float]] = {}
    for name, prediction in {
        "position_only_ablation5": base_branch,
        "position_history": history_branch,
        "velocity_conditioned": velocity.prediction,
        "history_velocity_conditioned": combined.prediction,
    }.items():
        selection_mse, selection_r2 = _prediction_metrics(
            prediction, branch_target, selection_rows
        )
        validation_mse, validation_r2 = _prediction_metrics(
            prediction, branch_target, validation_rows
        )
        branch_prediction_metrics[name] = {
            "selection_mse": selection_mse,
            "selection_r2": selection_r2,
            "validation_mse": validation_mse,
            "validation_r2": validation_r2,
        }
    branch_predictions = {
        "position_only_ablation5": base_branch
        * targets.target_scale
        + targets.target_mean,
        "position_history": history_branch * targets.target_scale + targets.target_mean,
        "velocity_conditioned": velocity.prediction
        * targets.target_scale
        + targets.target_mean,
        "history_velocity_conditioned": combined.prediction
        * targets.target_scale
        + targets.target_mean,
    }
    branch_retrieval: dict[str, Any] = {}
    for horizon_index, horizon in enumerate(targets.selected_horizons_ps.tolist()):
        spaces = {
            name: prediction.reshape(
                branch_count * center_count,
                len(targets.selected_horizons_ps),
                signature_dim,
            )[:, horizon_index]
            for name, prediction in branch_predictions.items()
        }
        future = branch_target_raw.reshape(
            branch_count * center_count,
            len(targets.selected_horizons_ps),
            signature_dim,
        )[:, horizon_index]
        result = _branch_future_neighbor_metrics(
            spaces,
            future,
            cache,
            targets.parent_splits["validation"],
            neighbors=int(neighbors),
            seed=int(seed),
        )
        baseline = float(
            result["position_only_ablation5"]["mean_individual_future_distance"]
        )
        result["gain_over_position_only_percent"] = {
            name: float(
                100.0
                * (1.0 - float(value["mean_individual_future_distance"]) / baseline)
            )
            for name, value in result.items()
        }
        branch_retrieval[f"{float(horizon):g}ps"] = result
    result = _branch_future_neighbor_metrics(
        branch_predictions,
        branch_target_raw,
        cache,
        targets.parent_splits["validation"],
        neighbors=int(neighbors),
        seed=int(seed),
    )
    baseline = float(
        result["position_only_ablation5"]["mean_individual_future_distance"]
    )
    result["gain_over_position_only_percent"] = {
        name: float(
            100.0 * (1.0 - float(value["mean_individual_future_distance"]) / baseline)
        )
        for name, value in result.items()
    }
    branch_retrieval["all_horizons"] = result

    ensemble_predictions: dict[str, np.ndarray] = {}
    for name, prediction in branch_predictions.items():
        reshaped = prediction.reshape(branch_count, center_count, -1)
        parent_prediction = np.empty((parent_count, center_count, reshaped.shape[-1]))
        for parent_index in range(parent_count):
            parent_prediction[parent_index] = reshaped[branch_parent == parent_index].mean(axis=0)
        ensemble_predictions[name] = parent_prediction.reshape(row_count, -1)
    ensemble_retrieval: dict[str, Any] = {}
    for horizon_index, horizon in enumerate(targets.selected_horizons_ps.tolist()):
        spaces = {
            name: prediction.reshape(
                row_count, len(targets.selected_horizons_ps), signature_dim
            )[:, horizon_index]
            for name, prediction in ensemble_predictions.items()
        }
        result = _future_neighbor_metrics(
            spaces,
            targets.distribution_signature[:, horizon_index],
            cache,
            targets.parent_splits["validation"],
            neighbors=int(neighbors),
            seed=int(seed),
        )
        baseline = float(
            result["position_only_ablation5"]["mean_ensemble_future_distance"]
        )
        result["gain_over_position_only_percent"] = {
            name: float(
                100.0
                * (1.0 - float(value["mean_ensemble_future_distance"]) / baseline)
            )
            for name, value in result.items()
        }
        ensemble_retrieval[f"{float(horizon):g}ps"] = result
    result = _future_neighbor_metrics(
        ensemble_predictions,
        targets.distribution_signature.reshape(row_count, -1),
        cache,
        targets.parent_splits["validation"],
        neighbors=int(neighbors),
        seed=int(seed),
    )
    baseline = float(
        result["position_only_ablation5"]["mean_ensemble_future_distance"]
    )
    result["gain_over_position_only_percent"] = {
        name: float(
            100.0 * (1.0 - float(value["mean_ensemble_future_distance"]) / baseline)
        )
        for name, value in result.items()
    }
    ensemble_retrieval["all_horizons"] = result

    metrics = {
        "history_parent_residual": {
            "selected_pca_dimension": history.selected_dimension,
            "selected_alpha": history.selected_alpha,
            "selection_mse": history.selection_mse,
            "validation_mse": history.validation_mse,
            "validation_r2": history.validation_r2,
            "base_ablation5_validation_mse": base_parent_validation_mse,
            "base_ablation5_validation_r2": base_parent_validation_r2,
        },
        "velocity_branch_residual": {
            "selected_pca_dimension": velocity.selected_dimension,
            "selected_alpha": velocity.selected_alpha,
            "selection_mse": velocity.selection_mse,
            "validation_mse": velocity.validation_mse,
            "validation_r2": velocity.validation_r2,
        },
        "history_velocity_branch_residual": {
            "selected_pca_dimension": combined.selected_dimension,
            "selected_alpha": combined.selected_alpha,
            "selection_mse": combined.selection_mse,
            "validation_mse": combined.validation_mse,
            "validation_r2": combined.validation_r2,
        },
        "individual_branch_prediction": branch_prediction_metrics,
        "parent_distribution_retrieval": parent_retrieval,
        "individual_branch_retrieval": branch_retrieval,
        "ensemble_of_branch_predictions_retrieval": ensemble_retrieval,
        "counts": {
            "parents": parent_count,
            "branches": branch_count,
            "centers": center_count,
            "parent_rows": row_count,
            "branch_rows": branch_count * center_count,
        },
    }
    arrays = {
        "history_parent_prediction": history_raw_prediction.astype(np.float32),
        "history_parent_features": history.features.astype(np.float32),
        "branch_target_signature": branch_target_raw.astype(np.float32),
        "base_branch_prediction": branch_predictions["position_only_ablation5"].astype(
            np.float32
        ),
        "velocity_branch_prediction": branch_predictions["velocity_conditioned"].astype(
            np.float32
        ),
        "combined_branch_prediction": branch_predictions[
            "history_velocity_conditioned"
        ].astype(np.float32),
        "base_representation": base_representation.astype(np.float32),
    }
    model_arrays: dict[str, np.ndarray] = {}
    for prefix, fitted in (
        ("history", history),
        ("velocity", velocity),
        ("combined", combined),
    ):
        model_arrays[f"{prefix}__selected_dimension"] = np.asarray(
            fitted.selected_dimension
        )
        model_arrays[f"{prefix}__selected_alpha"] = np.asarray(fitted.selected_alpha)
        model_arrays[f"{prefix}__coefficients"] = fitted.coefficients
        model_arrays[f"{prefix}__intercept"] = fitted.intercept
        for name, values in fitted.preprocessing.items():
            model_arrays[f"{prefix}__{name}"] = values
    return DynamicalAblationResult(metrics=metrics, arrays=arrays, model_arrays=model_arrays)


def plot_dynamical_retrieval(metrics: Mapping[str, Any], path: str | Path) -> None:
    panels = (
        ("parent_distribution_retrieval", "gain_over_ablation5_percent"),
        ("individual_branch_retrieval", "gain_over_position_only_percent"),
        ("ensemble_of_branch_predictions_retrieval", "gain_over_position_only_percent"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), sharey=False)
    for axis, (section, gain_key) in zip(axes, panels):
        values = metrics[section]
        horizons = list(values)
        names = list(values[horizons[0]][gain_key])
        x = np.arange(len(horizons), dtype=np.float64)
        width = 0.8 / len(names)
        for index, name in enumerate(names):
            axis.bar(
                x + (index - (len(names) - 1) / 2.0) * width,
                [values[horizon][gain_key][name] for horizon in horizons],
                width,
                label=name,
            )
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.set_xticks(x, horizons, rotation=30)
        axis.set_title(section.replace("_", " "))
        axis.set_ylabel("retrieval gain (%)")
        axis.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(Path(path), dpi=180)
    plt.close(fig)


__all__ = [
    "DynamicalAblationResult",
    "ShootingDynamicalFeatureCache",
    "VELOCITY_TOKEN_FEATURE_NAMES",
    "evaluate_dynamical_ablation",
    "extract_shooting_dynamical_feature_cache",
    "fit_selected_ridge_residual",
    "individual_branch_signatures",
    "invariant_velocity_token_features",
    "plot_dynamical_retrieval",
]
