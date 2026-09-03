"""Force-free point-cloud rollout diagnostic for shooting velocities."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.data_utils.shooting_binary_dataset import (
    ShootingBallisticEnvironmentDataset,
    make_shooting_environment_loader,
)
from src.data_utils.shooting_dataset import (
    ShootingCampaignSnapshot,
    shooting_snapshot_sha256,
)
from src.temporal_vamp.embeddings import FrozenEncoder
from src.temporal_vamp.shooting_distribution import (
    DistributionalTargetData,
    _random_fourier_features,
)
from src.temporal_vamp.shooting_dynamics import (
    _branch_future_neighbor_metrics,
    _branch_rows_for_parents,
    _prediction_metrics,
    fit_selected_ridge_residual,
    individual_branch_signatures,
)
from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache
from src.temporal_vamp.shooting_predictor import _future_neighbor_metrics
from src.temporal_vamp.shooting_short_horizon import (
    ShortHorizonVelocityResult,
    _aggregate_branch_predictions,
    _prediction_metrics_by_horizon,
)


@dataclass(frozen=True)
class ShootingBallisticEmbeddingCache:
    path: Path
    manifest: dict[str, Any]
    ballistic_future_z: np.ndarray
    branch_parent_index: np.ndarray
    horizons_ps: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "ShootingBallisticEmbeddingCache":
        root = Path(path).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Ballistic embedding manifest is missing: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        arrays = {
            name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in ("ballistic_future_z", "branch_parent_index", "horizons_ps")
        }
        for name, values in arrays.items():
            expected = tuple(int(value) for value in manifest["array_shapes"][name])
            if values.shape != expected:
                raise RuntimeError(
                    f"Ballistic cache shape changed for {name}: "
                    f"expected={expected}, observed={values.shape}, root={root}."
                )
        return cls(path=root, manifest=manifest, **arrays)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract_shooting_ballistic_embedding_cache(
    snapshot: ShootingCampaignSnapshot,
    reference_cache: ShootingEmbeddingCache,
    *,
    encoder: FrozenEncoder,
    cache_path: str | Path,
    horizons_ps: Sequence[float],
    num_points: int,
    radius: float,
    point_cloud_batch_size: int,
    environment_batch_size: int,
    environment_num_workers: int,
    force_recompute: bool,
) -> ShootingBallisticEmbeddingCache:
    target = Path(cache_path).expanduser().resolve()
    horizons = np.asarray([float(value) for value in horizons_ps], dtype=np.float64)
    snapshot_hash = shooting_snapshot_sha256(snapshot)
    if snapshot_hash != str(reference_cache.manifest["spec"]["snapshot_sha256"]):
        raise RuntimeError("Ballistic extraction snapshot does not match the shooting cache.")
    checkpoint_stat = encoder.checkpoint_path.stat()
    spec = {
        "version": 1,
        "snapshot_sha256": snapshot_hash,
        "checkpoint": str(encoder.checkpoint_path),
        "checkpoint_size_bytes": int(checkpoint_stat.st_size),
        "checkpoint_mtime_ns": int(checkpoint_stat.st_mtime_ns),
        "representation_source": encoder.representation_source,
        "encoder_repeats": int(encoder.repeats),
        "encoder_seed": int(encoder.seed),
        "horizons_ps": horizons.tolist(),
        "center_atom_ids": np.asarray(reference_cache.atom_ids, dtype=np.int64).tolist(),
        "num_points": int(num_points),
        "radius": float(radius),
        "rollout": "x_ballistic(t)=mod(x(0)+v(0)*t, box_lengths), LAMMPS metal units",
    }
    manifest_path = target / "manifest.json"
    if manifest_path.is_file() and not force_recompute:
        cached = ShootingBallisticEmbeddingCache.load(target)
        if cached.manifest["spec"] != spec:
            raise RuntimeError(
                f"Ballistic cache specification changed at {target}; choose a new path."
            )
        return cached
    if target.exists() and not force_recompute:
        raise RuntimeError(f"Ballistic cache exists without a complete manifest: {target}.")
    if force_recompute and target.exists():
        shutil.rmtree(target)
    shard_root = target.parent / f"{target.name}_shards"
    if force_recompute and shard_root.exists():
        shutil.rmtree(shard_root)
    shard_root.mkdir(parents=True, exist_ok=True)
    spec_sha = _sha256_json(spec)

    payloads: list[np.ndarray | None] = [None for _ in snapshot.branches]
    pending: list[dict[str, Any]] = []
    pending_records: list[tuple[int, Path]] = []
    for branch_index, branch in enumerate(snapshot.branches):
        shard = shard_root / f"branch_{branch_index:05d}.npz"
        if shard.is_file():
            with np.load(shard, allow_pickle=False) as values:
                observed = str(values["spec_sha256"].item())
                if observed != spec_sha:
                    raise RuntimeError(f"Ballistic shard specification changed: {shard}")
                payloads[branch_index] = values["ballistic_future_z"].copy()
            continue
        pending.append(branch)
        pending_records.append((branch_index, shard))

    if pending:
        dataset = ShootingBallisticEnvironmentDataset(
            snapshot,
            branches=pending,
            horizons_ps=horizons.tolist(),
            center_atom_ids=np.asarray(reference_cache.atom_ids, dtype=np.int64),
            num_points=int(num_points),
            radius=float(radius),
        )
        loader = make_shooting_environment_loader(
            dataset,
            batch_size=int(environment_batch_size),
            num_workers=int(environment_num_workers),
            pin_memory=encoder.device.type == "cuda",
        )
        for batch in loader:
            points = batch["points"]
            batch_count, horizon_count, center_count, point_count, coordinate_dim = (
                points.shape
            )
            flat = points.reshape(
                batch_count * horizon_count * center_count,
                point_count,
                coordinate_dim,
            )
            chunks = []
            with torch.inference_mode():
                for start in range(0, int(flat.shape[0]), int(point_cloud_batch_size)):
                    chunks.append(
                        encoder.encode(flat[start : start + int(point_cloud_batch_size)]).cpu()
                    )
            encoded = torch.cat(chunks).numpy().reshape(
                batch_count, horizon_count, center_count, encoder.output_dim
            )
            for batch_index, dataset_index_tensor in enumerate(batch["dataset_index"]):
                dataset_index = int(dataset_index_tensor.item())
                branch_index, shard = pending_records[dataset_index]
                values = encoded[batch_index].astype(np.float32, copy=False)
                temporary = shard.with_suffix(".tmp.npz")
                np.savez(
                    temporary,
                    spec_sha256=np.asarray(spec_sha),
                    ballistic_future_z=values,
                )
                os.replace(temporary, shard)
                payloads[branch_index] = values
                if (branch_index + 1) % 20 == 0 or branch_index + 1 == len(snapshot.branches):
                    print(
                        f"[shooting-ballistic] branch={branch_index + 1}/{len(snapshot.branches)}",
                        flush=True,
                    )

    if any(values is None for values in payloads):
        raise RuntimeError("Ballistic extraction left an unwritten branch shard.")
    complete = [values for values in payloads if values is not None]
    shapes = {
        "ballistic_future_z": (
            len(snapshot.branches),
            int(horizons.size),
            int(reference_cache.atom_ids.size),
            int(encoder.output_dim),
        ),
        "branch_parent_index": (len(snapshot.branches),),
        "horizons_ps": (int(horizons.size),),
    }
    building = target.parent / f".{target.name}.building-{os.getpid()}"
    building.mkdir(parents=True)
    ballistic_array = np.lib.format.open_memmap(
        building / "ballistic_future_z.npy",
        mode="w+",
        dtype=np.float32,
        shape=shapes["ballistic_future_z"],
    )
    for index, values in enumerate(complete):
        ballistic_array[index] = values
    ballistic_array.flush()
    branch_parent_array = np.lib.format.open_memmap(
        building / "branch_parent_index.npy",
        mode="w+",
        dtype=np.int32,
        shape=shapes["branch_parent_index"],
    )
    branch_parent_array[:] = reference_cache.branch_parent_index
    branch_parent_array.flush()
    horizons_array = np.lib.format.open_memmap(
        building / "horizons_ps.npy",
        mode="w+",
        dtype=np.float64,
        shape=shapes["horizons_ps"],
    )
    horizons_array[:] = horizons
    horizons_array.flush()
    del ballistic_array, branch_parent_array, horizons_array
    manifest = {
        "state": "complete",
        "spec": spec,
        "array_shapes": {name: list(shape) for name, shape in shapes.items()},
    }
    with (building / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(building, target)
    return ShootingBallisticEmbeddingCache.load(target)


def _ballistic_signatures(
    cache: ShootingEmbeddingCache,
    targets: DistributionalTargetData,
    ballistic: ShootingBallisticEmbeddingCache,
) -> np.ndarray:
    if not np.array_equal(cache.branch_parent_index, ballistic.branch_parent_index):
        raise RuntimeError("Ballistic and actual-future caches disagree on branch order.")
    if not np.allclose(
        targets.selected_horizons_ps, ballistic.horizons_ps, rtol=0.0, atol=1.0e-9
    ):
        raise RuntimeError("Ballistic and actual-future caches disagree on horizons.")
    branch_parent = np.asarray(cache.branch_parent_index, dtype=np.int64)
    current = np.asarray(cache.parent_local_z[branch_parent], dtype=np.float64)
    blocks: list[np.ndarray] = []
    for horizon_index, parameters in enumerate(targets.horizon_parameters):
        delta = np.asarray(
            ballistic.ballistic_future_z[:, horizon_index], dtype=np.float64
        ) - current
        standardized = (delta - parameters.delta_mean) / parameters.delta_scale
        projected = parameters.pca.transform(
            standardized.reshape(-1, standardized.shape[-1]),
            dimension=parameters.pca.components_.shape[1],
        ).reshape(*standardized.shape[:2], -1)
        blocks.append(
            _random_fourier_features(projected, parameters.frequencies, parameters.phases)
        )
    return np.stack(blocks, axis=1)


def evaluate_ballistic_rollout(
    cache: ShootingEmbeddingCache,
    targets: DistributionalTargetData,
    ballistic: ShootingBallisticEmbeddingCache,
    position_arrays: Mapping[str, np.ndarray],
    velocity_result: ShortHorizonVelocityResult,
    *,
    ballistic_pca_dimensions: Sequence[int],
    ridge_alphas: Sequence[float],
    neighbors: int,
    seed: int,
) -> ShortHorizonVelocityResult:
    parent_count, center_count = cache.parent_local_z.shape[:2]
    parent_rows_count = int(parent_count * center_count)
    branch_parent = np.asarray(cache.branch_parent_index, dtype=np.int64)
    branch_count = int(branch_parent.size)
    horizon_count = int(targets.selected_horizons_ps.size)
    signature_dim = int(targets.distribution_signature.shape[-1])
    actual_signatures = individual_branch_signatures(cache, targets)
    actual_raw = actual_signatures.transpose(0, 2, 1, 3).reshape(
        branch_count * center_count, -1
    )
    actual = (actual_raw - targets.target_mean) / targets.target_scale
    ballistic_signatures = _ballistic_signatures(cache, targets, ballistic)
    ballistic_raw = ballistic_signatures.transpose(0, 2, 1, 3).reshape(
        branch_count * center_count, -1
    )
    ballistic_standardized = (
        ballistic_raw - targets.target_mean
    ) / targets.target_scale
    parent_row = (
        branch_parent[:, None] * center_count
        + np.arange(center_count, dtype=np.int64)[None]
    ).reshape(-1)
    position_parent_raw = np.asarray(position_arrays["prediction"], dtype=np.float64)
    position_raw = position_parent_raw[parent_row]
    position = (
        position_raw - targets.target_mean
    ) / targets.target_scale
    velocity_raw = np.asarray(
        velocity_result.arrays["velocity_branch_prediction"], dtype=np.float64
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
    calibrated = fit_selected_ridge_residual(
        ballistic_standardized,
        position,
        actual,
        optimization_rows=optimization_rows,
        selection_rows=selection_rows,
        validation_rows=validation_rows,
        dimensions=ballistic_pca_dimensions,
        alphas=ridge_alphas,
    )
    calibrated_raw = calibrated.prediction * targets.target_scale + targets.target_mean
    standardized_predictions = {
        "position_only": position,
        "aggregated_velocity": (
            velocity_raw - targets.target_mean
        ) / targets.target_scale,
        "ballistic_direct": ballistic_standardized,
        "ballistic_calibrated": calibrated.prediction,
    }
    raw_predictions = {
        "position_only": position_raw,
        "aggregated_velocity": velocity_raw,
        "ballistic_direct": ballistic_raw,
        "ballistic_calibrated": calibrated_raw,
    }
    prediction_metrics: dict[str, Any] = {}
    for name, prediction in standardized_predictions.items():
        validation_mse, validation_r2 = _prediction_metrics(
            prediction, actual, validation_rows
        )
        by_horizon = _prediction_metrics_by_horizon(
            prediction,
            actual,
            validation_rows,
            horizon_count=horizon_count,
            signature_dim=signature_dim,
        )
        prediction_metrics[name] = {
            "validation_mse": validation_mse,
            "validation_r2": validation_r2,
            "validation_by_horizon": {
                f"{float(targets.selected_horizons_ps[index]):g}ps": values
                for index, values in by_horizon.items()
            },
        }

    branch_retrieval: dict[str, Any] = {}
    for horizon_index, horizon in enumerate(targets.selected_horizons_ps.tolist()):
        spaces = {
            name: values.reshape(-1, horizon_count, signature_dim)[:, horizon_index]
            for name, values in raw_predictions.items()
        }
        future = actual_raw.reshape(-1, horizon_count, signature_dim)[:, horizon_index]
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
    combined = _branch_future_neighbor_metrics(
        raw_predictions,
        actual_raw,
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
        for name, values in raw_predictions.items()
    }
    ensemble_retrieval: dict[str, Any] = {}
    for horizon_index, horizon in enumerate(targets.selected_horizons_ps.tolist()):
        spaces = {
            name: values.reshape(parent_rows_count, horizon_count, signature_dim)[
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
        targets.distribution_signature.reshape(parent_rows_count, -1),
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
        "selected_dimension": np.asarray(calibrated.selected_dimension),
        "selected_alpha": np.asarray(calibrated.selected_alpha),
        "coefficients": calibrated.coefficients,
        "intercept": calibrated.intercept,
    }
    for name, values in calibrated.preprocessing.items():
        model_arrays[name] = values
    return ShortHorizonVelocityResult(
        metrics={
            "ballistic_calibration": {
                "selected_pca_dimension": calibrated.selected_dimension,
                "selected_alpha": calibrated.selected_alpha,
                "selection_mse": calibrated.selection_mse,
                "validation_mse": calibrated.validation_mse,
                "validation_r2": calibrated.validation_r2,
            },
            "individual_branch_prediction": prediction_metrics,
            "individual_branch_retrieval": branch_retrieval,
            "ensemble_of_branch_predictions_retrieval": ensemble_retrieval,
        },
        arrays={
            "ballistic_calibrated_prediction": calibrated_raw.astype(np.float32),
            "ballistic_selected_features": calibrated.features.astype(np.float32),
        },
        model_arrays=model_arrays,
    )


def plot_ballistic_rollout(metrics: Mapping[str, Any], path: str | Path) -> None:
    prediction = metrics["individual_branch_prediction"]
    retrieval = metrics["individual_branch_retrieval"]
    horizons = [key for key in retrieval if key != "all_horizons"]
    names = ("position_only", "aggregated_velocity", "ballistic_direct", "ballistic_calibrated")
    x = np.arange(len(horizons), dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3))
    for name in names:
        axes[0].plot(
            x,
            [float(prediction[name]["validation_by_horizon"][key]["r2"]) for key in horizons],
            marker="o",
            label=name,
        )
        axes[1].plot(
            x,
            [
                float(retrieval[key]["gain_over_position_only_percent"][name])
                for key in horizons
            ],
            marker="o",
            label=name,
        )
    axes[0].set(
        xticks=x,
        xticklabels=horizons,
        xlabel="future horizon",
        ylabel="held-out individual-branch R2",
    )
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set(
        xticks=x,
        xticklabels=horizons,
        xlabel="future horizon",
        ylabel="retrieval gain over position only (%)",
    )
    for axis in axes:
        axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(path), dpi=180)
    plt.close(fig)


__all__ = [
    "ShootingBallisticEmbeddingCache",
    "evaluate_ballistic_rollout",
    "extract_shooting_ballistic_embedding_cache",
    "plot_ballistic_rollout",
]
