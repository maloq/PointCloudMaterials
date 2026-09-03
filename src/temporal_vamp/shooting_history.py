"""Deterministic past-context embeddings for shooting parent states."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from numpy.lib.format import open_memmap

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
from src.temporal_vamp.shooting_context import ShootingContextTokenCache
from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache


@dataclass(frozen=True)
class ShootingHistoryEmbeddingCache:
    path: Path
    manifest: dict[str, Any]
    history_z: np.ndarray
    context_center_atom_ids: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "ShootingHistoryEmbeddingCache":
        root = Path(path).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Shooting history manifest is missing: {manifest_path}"
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("state") != "complete":
            raise RuntimeError(
                f"Shooting history cache is not complete: {manifest_path}"
            )
        arrays = {
            name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in ("history_z", "context_center_atom_ids")
        }
        for name, values in arrays.items():
            expected = tuple(int(value) for value in manifest["array_shapes"][name])
            if tuple(values.shape) != expected:
                raise RuntimeError(
                    f"Shooting history shape changed for {name}: expected={expected}, "
                    f"observed={values.shape}, root={root}."
                )
        return cls(path=root, manifest=manifest, **arrays)


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


def _branch_path(
    snapshot: ShootingCampaignSnapshot, branch: dict[str, Any]
) -> Path:
    root = (
        snapshot.root
        if len(snapshot.campaign_roots) == 1
        else Path(str(branch["campaign_root"]))
    )
    return resolve_shooting_trajectory_path(root, branch)


@torch.inference_mode()
def _encode_chunks(
    encoder: FrozenEncoder, points: torch.Tensor, *, batch_size: int
) -> np.ndarray:
    outputs = [
        encoder.encode(points[start : start + int(batch_size)]).cpu()
        for start in range(0, int(points.shape[0]), int(batch_size))
    ]
    if not outputs:
        raise RuntimeError("Shooting history extraction produced no point clouds.")
    return torch.cat(outputs).numpy().astype(np.float32, copy=False)


def extract_shooting_history_embedding_cache(
    snapshot: ShootingCampaignSnapshot,
    base_cache: ShootingEmbeddingCache,
    context_cache: ShootingContextTokenCache,
    *,
    encoder: FrozenEncoder,
    source_trajectory_root: str | Path,
    cache_path: str | Path,
    lag_frames: Sequence[int],
    lag_times_ps: Sequence[float],
    source_sample_interval_ps: float,
    num_points: int,
    radius: float,
    context_center_count: int,
    point_cloud_batch_size: int,
    force_recompute: bool,
) -> ShootingHistoryEmbeddingCache:
    target = Path(cache_path).expanduser().resolve()
    source_root = Path(source_trajectory_root).expanduser().resolve()
    frame_offsets = np.asarray([int(value) for value in lag_frames], dtype=np.int64)
    lag_ps = np.asarray([float(value) for value in lag_times_ps], dtype=np.float64)
    if frame_offsets.shape != lag_ps.shape or frame_offsets.size == 0:
        raise ValueError(
            "History lag_frames and lag_times_ps must have the same nonzero length: "
            f"frames={frame_offsets.tolist()}, times={lag_ps.tolist()}."
        )
    if np.any(frame_offsets <= 0) or np.any(np.diff(frame_offsets) >= 0):
        raise ValueError(
            "History lag_frames must be strictly decreasing so the encoded sequence "
            f"runs oldest to newest, got {frame_offsets.tolist()}."
        )
    expected_lag_ps = frame_offsets.astype(np.float64) * float(
        source_sample_interval_ps
    )
    if not np.allclose(lag_ps, expected_lag_ps, rtol=0.0, atol=1.0e-9):
        raise ValueError(
            "History physical lags disagree with source frame spacing: "
            f"expected={expected_lag_ps.tolist()}, observed={lag_ps.tolist()}."
        )

    snapshot_hash = shooting_snapshot_sha256(snapshot)
    for name, manifest in (
        ("base", base_cache.manifest),
        ("context", context_cache.manifest),
    ):
        observed = str(manifest["spec"]["snapshot_sha256"])
        if observed != snapshot_hash:
            raise RuntimeError(
                f"History snapshot does not match the {name} cache: "
                f"snapshot={snapshot_hash}, cache={observed}."
            )
    checkpoint_stat = encoder.checkpoint_path.stat()
    spec = {
        "version": 1,
        "snapshot_sha256": snapshot_hash,
        "source_trajectory_root": str(source_root),
        "lag_frames_oldest_to_newest": frame_offsets.tolist(),
        "lag_times_ps_oldest_to_newest": lag_ps.tolist(),
        "source_sample_interval_ps": float(source_sample_interval_ps),
        "checkpoint": str(encoder.checkpoint_path),
        "checkpoint_size_bytes": int(checkpoint_stat.st_size),
        "checkpoint_mtime_ns": int(checkpoint_stat.st_mtime_ns),
        "representation_source": encoder.representation_source,
        "center_atom_ids": np.asarray(base_cache.atom_ids, dtype=np.int64).tolist(),
        "num_points": int(num_points),
        "radius": float(radius),
        "context_center_count": int(context_center_count),
        "token_identity": (
            "current central atom plus current-frame farthest-point satellite IDs, "
            "tracked backward through the source trajectory"
        ),
    }
    if (target / "manifest.json").is_file() and not force_recompute:
        cached = ShootingHistoryEmbeddingCache.load(target)
        if cached.manifest["spec"] != spec:
            raise RuntimeError(
                f"Shooting history cache specification changed at {target}; use a "
                "new cache path or set force_recompute=true."
            )
        return cached
    if target.exists() and not force_recompute:
        raise RuntimeError(
            f"Shooting history cache exists without a complete manifest: {target}."
        )
    if force_recompute and target.exists():
        shutil.rmtree(target)

    shard_root = target.parent / f"{target.name}_shards"
    if force_recompute and shard_root.exists():
        shutil.rmtree(shard_root)
    shard_root.mkdir(parents=True, exist_ok=True)
    spec_sha = _sha256_json(spec)
    branches_by_parent: dict[str, list[dict[str, Any]]] = {
        str(parent["parent_id"]): [] for parent in snapshot.parents
    }
    for branch in snapshot.branches:
        branches_by_parent[str(branch["parent_id"])].append(branch)

    atom_ids = np.asarray(base_cache.atom_ids, dtype=np.int64)
    frozen_current = np.concatenate(
        [
            np.asarray(base_cache.parent_local_z, dtype=np.float32)[:, :, None],
            np.asarray(context_cache.satellite_z, dtype=np.float32),
        ],
        axis=2,
    )
    maximum_current_embedding_error = 0.0
    maximum_source_position_error = 0.0
    for parent_position, parent in enumerate(snapshot.parents):
        shard_path = shard_root / f"parent_{parent_position:04d}.npz"
        if shard_path.is_file():
            with np.load(shard_path, allow_pickle=False) as payload:
                if str(payload["spec_sha256"].item()) != spec_sha:
                    raise RuntimeError(
                        f"Shooting history shard specification changed: {shard_path}"
                    )
                maximum_current_embedding_error = max(
                    maximum_current_embedding_error,
                    float(payload["current_embedding_max_abs_error"]),
                )
                maximum_source_position_error = max(
                    maximum_source_position_error,
                    float(payload["source_position_max_abs_error"]),
                )
            continue

        source_path = (
            source_root
            / str(parent["source_run_id"])
            / "trajectory_binary_float32"
        )
        source = TemporalLAMMPSBinaryTrajectory.load(source_path)
        current_index = int(parent["source_frame_index"])
        if current_index < int(frame_offsets.max()):
            raise RuntimeError(
                f"Parent {parent['parent_id']} has insufficient source history: "
                f"current_frame={current_index}, requested_offsets={frame_offsets.tolist()}."
            )
        representative = sorted(
            branches_by_parent[str(parent["parent_id"])],
            key=lambda value: int(value["shot_index"]),
        )[0]
        branch = ShootingBinaryTrajectory.load(_branch_path(snapshot, representative))
        parent_frame = branch.load_position_frames([0])[0]
        source_current = _position_frame(source, current_index)
        source_position_error = float(
            np.max(
                np.abs(
                    np.asarray(source_current.positions, dtype=np.float32)
                    - np.asarray(parent_frame.positions, dtype=np.float32)
                )
            )
        )
        maximum_source_position_error = max(
            maximum_source_position_error, source_position_error
        )
        if source_position_error > 1.0e-4:
            raise RuntimeError(
                f"Shooting parent does not match source history for "
                f"parent={parent['parent_id']}: max_abs_error={source_position_error}."
            )

        current = build_periodic_environment_batch(
            parent_frame,
            center_atom_ids=atom_ids,
            num_points=int(num_points),
            radius=float(radius),
            spatial_context_center_count=int(context_center_count),
        )
        if current.context_points is None or current.context_center_atom_ids is None:
            raise RuntimeError(
                f"Current frame produced no context tokens for {parent['parent_id']}."
            )
        context_ids = current.context_center_atom_ids
        current_points = torch.cat(
            [current.points[:, None], current.context_points], dim=1
        )
        current_z = _encode_chunks(
            encoder,
            current_points.reshape(-1, int(num_points), 3),
            batch_size=int(point_cloud_batch_size),
        ).reshape(atom_ids.size, int(context_center_count) + 1, -1)
        current_error = float(
            np.max(np.abs(current_z - frozen_current[parent_position]))
        )
        maximum_current_embedding_error = max(
            maximum_current_embedding_error, current_error
        )
        if current_error > 5.0e-4:
            raise RuntimeError(
                f"History extraction disagrees with cached current tokens for "
                f"parent={parent['parent_id']}: max_abs_error={current_error}."
            )

        token_ids = np.concatenate([atom_ids[:, None], context_ids], axis=1)
        encoded_history: list[np.ndarray] = []
        timestep_gaps: list[int] = []
        for frame_offset in frame_offsets.tolist():
            previous_index = current_index - int(frame_offset)
            timestep_gaps.append(
                int(source.timesteps[current_index])
                - int(source.timesteps[previous_index])
            )
            previous = build_periodic_environment_batch(
                _position_frame(source, previous_index),
                center_atom_ids=token_ids.reshape(-1),
                num_points=int(num_points),
                radius=float(radius),
                spatial_context_center_count=0,
            )
            encoded_history.append(
                _encode_chunks(
                    encoder,
                    previous.points,
                    batch_size=int(point_cloud_batch_size),
                ).reshape(atom_ids.size, int(context_center_count) + 1, -1)
            )
        history_z = np.stack(encoded_history, axis=1)
        temporary = shard_path.with_suffix(".tmp.npz")
        np.savez(
            temporary,
            spec_sha256=np.asarray(spec_sha),
            history_z=history_z,
            context_center_atom_ids=context_ids,
            source_timestep_gaps=np.asarray(timestep_gaps, dtype=np.int64),
            current_embedding_max_abs_error=np.asarray(current_error),
            source_position_max_abs_error=np.asarray(source_position_error),
        )
        os.replace(temporary, shard_path)
        print(
            f"[shooting-history] parent={parent_position + 1}/{len(snapshot.parents)} "
            f"{parent['parent_id']}",
            flush=True,
        )

    parent_count = len(snapshot.parents)
    center_count = int(atom_ids.size)
    lag_count = int(frame_offsets.size)
    token_count = int(context_center_count) + 1
    embedding_dim = int(base_cache.parent_local_z.shape[-1])
    shapes = {
        "history_z": (
            parent_count,
            center_count,
            lag_count,
            token_count,
            embedding_dim,
        ),
        "context_center_atom_ids": (
            parent_count,
            center_count,
            int(context_center_count),
        ),
    }
    building = target.parent / f".{target.name}.building-{os.getpid()}"
    if building.exists():
        raise FileExistsError(f"History cache build directory exists: {building}")
    building.mkdir(parents=True)
    history_array = open_memmap(
        building / "history_z.npy",
        mode="w+",
        dtype=np.float32,
        shape=shapes["history_z"],
    )
    context_id_array = open_memmap(
        building / "context_center_atom_ids.npy",
        mode="w+",
        dtype=np.int64,
        shape=shapes["context_center_atom_ids"],
    )
    timestep_gap_patterns: set[tuple[int, ...]] = set()
    for parent_position in range(parent_count):
        with np.load(
            shard_root / f"parent_{parent_position:04d}.npz", allow_pickle=False
        ) as payload:
            history_array[parent_position] = payload["history_z"]
            context_id_array[parent_position] = payload["context_center_atom_ids"]
            timestep_gap_patterns.add(
                tuple(int(value) for value in payload["source_timestep_gaps"])
            )
    history_array.flush()
    context_id_array.flush()
    del history_array, context_id_array
    manifest = {
        "state": "complete",
        "spec": spec,
        "array_shapes": {name: list(shape) for name, shape in shapes.items()},
        "source_timestep_gap_patterns": [
            list(pattern) for pattern in sorted(timestep_gap_patterns)
        ],
        "current_embedding_max_abs_error": maximum_current_embedding_error,
        "source_position_max_abs_error": maximum_source_position_error,
    }
    with (building / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(building, target)
    return ShootingHistoryEmbeddingCache.load(target)


__all__ = [
    "ShootingHistoryEmbeddingCache",
    "extract_shooting_history_embedding_cache",
]
