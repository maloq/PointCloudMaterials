from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.lib.format import open_memmap

from src.data_utils.shooting_binary_dataset import (
    ShootingBinaryEnvironmentDataset,
    make_shooting_environment_loader,
)
from src.data_utils.shooting_dataset import (
    ShootingCampaignSnapshot,
    shooting_snapshot_sha256,
)
from src.temporal_vamp.embeddings import (
    FrozenEncoder,
    encode_spatial_context_state,
)


@dataclass(frozen=True)
class ShootingEmbeddingCache:
    path: Path
    manifest: dict[str, Any]
    parent_z: np.ndarray
    parent_local_z: np.ndarray
    parent_coords: np.ndarray
    future_z: np.ndarray
    branch_parent_index: np.ndarray
    atom_ids: np.ndarray
    horizons_ps: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "ShootingEmbeddingCache":
        root = Path(path).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Shooting embedding manifest is missing: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        arrays = {
            name: np.load(root / f"{name}.npy", mmap_mode="r")
            for name in (
                "parent_z",
                "parent_local_z",
                "parent_coords",
                "future_z",
                "branch_parent_index",
                "atom_ids",
                "horizons_ps",
            )
        }
        expected = {
            "parent_z": tuple(manifest["array_shapes"]["parent_z"]),
            "parent_local_z": tuple(manifest["array_shapes"]["parent_local_z"]),
            "parent_coords": tuple(manifest["array_shapes"]["parent_coords"]),
            "future_z": tuple(manifest["array_shapes"]["future_z"]),
            "branch_parent_index": tuple(manifest["array_shapes"]["branch_parent_index"]),
            "atom_ids": tuple(manifest["array_shapes"]["atom_ids"]),
            "horizons_ps": tuple(manifest["array_shapes"]["horizons_ps"]),
        }
        for name, values in arrays.items():
            if tuple(values.shape) != expected[name]:
                raise RuntimeError(
                    f"Shooting embedding array shape mismatch for {name}: "
                    f"expected={expected[name]}, observed={tuple(values.shape)}, path={root}."
                )
        return cls(path=root, manifest=manifest, **arrays)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_shard(path: Path, *, expected_sha256: str) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as payload:
        observed = str(payload["spec_sha256"].item())
        if observed != expected_sha256:
            raise RuntimeError(
                f"Shooting embedding shard specification changed: path={path}, "
                f"expected={expected_sha256}, observed={observed}. Remove the derived shard "
                "or set cache.force_recompute=true."
            )
        return {name: payload[name].copy() for name in payload.files if name != "spec_sha256"}


def _write_shard(path: Path, *, spec_sha256: str, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, spec_sha256=np.asarray(spec_sha256), **arrays)
    os.replace(temporary, path)


def _branch_uid(
    snapshot: ShootingCampaignSnapshot, branch: dict[str, Any]
) -> str:
    if len(snapshot.campaign_roots) == 1:
        return str(branch["branch_id"])
    return str(branch["branch_uid"])


def _resolve_horizon_timesteps(
    snapshot: ShootingCampaignSnapshot,
    horizons_ps: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    protocol = snapshot.manifest["protocol"]
    timestep_ps = float(protocol["timestep_fs"]) / 1000.0
    sample_interval_steps = int(protocol["sample_interval_steps"])
    requested = np.asarray([float(value) for value in horizons_ps], dtype=np.float64)
    if requested.ndim != 1 or requested.size == 0 or np.any(requested <= 0.0):
        raise ValueError(f"shooting.horizons_ps must be positive and nonempty, got {requested}.")
    steps_float = requested / timestep_ps
    steps = np.rint(steps_float).astype(np.int64)
    if not np.allclose(steps_float, steps, rtol=0.0, atol=1.0e-9):
        raise ValueError(
            f"Shooting horizons do not align with timestep_ps={timestep_ps}: "
            f"horizons={requested.tolist()}, steps={steps_float.tolist()}."
        )
    if np.any(steps % sample_interval_steps != 0):
        raise ValueError(
            f"Shooting horizons must align with sample_interval_steps={sample_interval_steps}; "
            f"resolved steps={steps.tolist()}."
        )
    if np.any(steps > int(protocol["run_steps"])):
        raise ValueError(
            f"Shooting horizons exceed run_steps={protocol['run_steps']}: {steps.tolist()}."
        )
    if len(set(steps.tolist())) != steps.size:
        raise ValueError(f"Shooting horizons resolve to duplicate timesteps: {steps.tolist()}.")
    return requested, steps


def extract_shooting_embedding_cache(
    snapshot: ShootingCampaignSnapshot,
    *,
    encoder: FrozenEncoder,
    cache_path: str | Path,
    horizons_ps: Sequence[float],
    center_atom_count: int,
    center_selection_seed: int,
    num_points: int,
    radius: float,
    spatial_context_center_count: int,
    spatial_context_aggregation: str,
    point_cloud_batch_size: int,
    environment_batch_size: int,
    environment_num_workers: int,
    force_recompute: bool,
) -> ShootingEmbeddingCache:
    target = Path(cache_path).expanduser().resolve()
    resolved_horizons, horizon_timesteps = _resolve_horizon_timesteps(
        snapshot, horizons_ps
    )
    atom_count = int(snapshot.manifest["atom_count"])
    center_count = int(center_atom_count)
    if center_count <= 0 or center_count > atom_count:
        raise ValueError(
            f"center_atom_count must be in [1, {atom_count}], got {center_count}."
        )
    rng = np.random.default_rng(int(center_selection_seed))
    atom_ids = np.sort(
        rng.choice(
            np.arange(1, atom_count + 1, dtype=np.int64),
            size=center_count,
            replace=False,
        )
    )
    local_dim = int(encoder.output_dim)
    context_count = int(spatial_context_center_count)
    input_dim = local_dim * (3 if context_count > 0 else 1)
    checkpoint_stat = encoder.checkpoint_path.stat()
    cache_spec = {
        "version": 1,
        "snapshot_sha256": shooting_snapshot_sha256(snapshot),
        "campaign_root": str(snapshot.root),
        "campaign_roots": [str(root) for root in snapshot.campaign_roots],
        "checkpoint": str(encoder.checkpoint_path),
        "checkpoint_size_bytes": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "representation_source": encoder.representation_source,
        "encoder_repeats": encoder.repeats,
        "encoder_seed": encoder.seed,
        "horizons_ps": resolved_horizons.tolist(),
        "horizon_timesteps": horizon_timesteps.tolist(),
        "center_atom_count": center_count,
        "center_selection_seed": int(center_selection_seed),
        "center_atom_ids": atom_ids.tolist(),
        "num_points": int(num_points),
        "radius": float(radius),
        "spatial_context_center_count": context_count,
        "spatial_context_aggregation": str(spatial_context_aggregation),
        "point_cloud_batch_size": int(point_cloud_batch_size),
    }
    manifest_path = target / "manifest.json"
    if target.exists() and manifest_path.is_file() and not force_recompute:
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing["spec"] != cache_spec:
            raise RuntimeError(
                f"Shooting embedding cache specification changed: {target}. "
                "Set cache.force_recompute=true or choose another output directory."
            )
        return ShootingEmbeddingCache.load(target)
    if target.exists() and not manifest_path.is_file() and not force_recompute:
        raise RuntimeError(
            f"Shooting embedding cache exists without a final manifest: {target}. "
            "Set cache.force_recompute=true to rebuild it."
        )
    if force_recompute and target.exists():
        shutil.rmtree(target)

    shard_root = target.parent / f"{target.name}_shards"
    if force_recompute and shard_root.exists():
        shutil.rmtree(shard_root)
    (shard_root / "parents").mkdir(parents=True, exist_ok=True)
    (shard_root / "branches").mkdir(parents=True, exist_ok=True)

    parent_index_by_id = {
        str(parent["parent_id"]): index for index, parent in enumerate(snapshot.parents)
    }
    branches_by_parent: dict[str, list[dict[str, Any]]] = {
        parent_id: [] for parent_id in parent_index_by_id
    }
    for branch in snapshot.branches:
        branches_by_parent[str(branch["parent_id"])].append(branch)

    parent_payloads: list[dict[str, np.ndarray] | None] = [
        None for _ in snapshot.parents
    ]
    pending_parent_branches: list[dict[str, Any]] = []
    pending_parent_records: list[tuple[int, str, Path, str]] = []
    for parent_position, parent in enumerate(snapshot.parents):
        parent_id = str(parent["parent_id"])
        representative = branches_by_parent[parent_id][0]
        spec = {
            "cache_spec": cache_spec,
            "parent": parent,
            "representative_branch_uid": _branch_uid(snapshot, representative),
        }
        spec_sha = _sha256_json(spec)
        shard_path = shard_root / "parents" / f"{parent_id}.npz"
        payload = _load_shard(shard_path, expected_sha256=spec_sha)
        if payload is not None:
            parent_payloads[parent_position] = payload
            print(
                f"[shooting-embeddings] parent {parent_position + 1}/{len(snapshot.parents)} "
                f"{parent_id} (cached)",
                flush=True,
            )
            continue
        pending_parent_branches.append(representative)
        pending_parent_records.append((parent_position, parent_id, shard_path, spec_sha))

    if pending_parent_branches:
        parent_dataset = ShootingBinaryEnvironmentDataset(
            snapshot,
            branches=pending_parent_branches,
            timesteps=[0],
            center_atom_ids=atom_ids,
            num_points=int(num_points),
            radius=float(radius),
            spatial_context_center_count=context_count,
        )
        parent_loader = make_shooting_environment_loader(
            parent_dataset,
            batch_size=int(environment_batch_size),
            num_workers=int(environment_num_workers),
            pin_memory=encoder.device.type == "cuda",
        )
        for batch in parent_loader:
            points = batch["points"][:, 0]
            batch_count, center_batch_count, point_count, coordinate_dim = points.shape
            flat_points = points.reshape(
                batch_count * center_batch_count, point_count, coordinate_dim
            )
            context_points = batch.get("context_points")
            flat_context = None
            if context_points is not None:
                flat_context = context_points[:, 0].reshape(
                    batch_count * center_batch_count,
                    context_count,
                    point_count,
                    coordinate_dim,
                )
            contextual, local = encode_spatial_context_state(
                encoder,
                flat_points,
                context_points=flat_context,
                aggregation=str(spatial_context_aggregation),
                point_cloud_batch_size=int(point_cloud_batch_size),
            )
            contextual_values = contextual.numpy().reshape(
                batch_count, center_batch_count, input_dim
            )
            local_values = local.numpy().reshape(
                batch_count, center_batch_count, local_dim
            )
            coordinate_values = batch["center_positions"][:, 0].numpy()
            for batch_index, dataset_index_tensor in enumerate(batch["dataset_index"]):
                dataset_index = int(dataset_index_tensor.item())
                parent_position, parent_id, shard_path, spec_sha = (
                    pending_parent_records[dataset_index]
                )
                payload = {
                    "z": contextual_values[batch_index].astype(np.float32, copy=False),
                    "local_z": local_values[batch_index].astype(np.float32, copy=False),
                    "coords": coordinate_values[batch_index].astype(np.float32, copy=False),
                }
                _write_shard(shard_path, spec_sha256=spec_sha, **payload)
                parent_payloads[parent_position] = payload
                print(
                    f"[shooting-embeddings] parent {parent_position + 1}/"
                    f"{len(snapshot.parents)} {parent_id}",
                    flush=True,
                )

    branch_payloads: list[dict[str, np.ndarray] | None] = [
        None for _ in snapshot.branches
    ]
    pending_branches: list[dict[str, Any]] = []
    pending_branch_records: list[tuple[int, str, Path, str]] = []
    for branch_position, branch in enumerate(snapshot.branches):
        branch_uid = _branch_uid(snapshot, branch)
        spec = {"cache_spec": cache_spec, "branch": branch}
        spec_sha = _sha256_json(spec)
        shard_path = shard_root / "branches" / f"{branch_uid}.npz"
        payload = _load_shard(shard_path, expected_sha256=spec_sha)
        if payload is not None:
            branch_payloads[branch_position] = payload
            print(
                f"[shooting-embeddings] branch {branch_position + 1}/"
                f"{len(snapshot.branches)} {branch_uid} (cached)",
                flush=True,
            )
            continue
        pending_branches.append(branch)
        pending_branch_records.append((branch_position, branch_uid, shard_path, spec_sha))

    if pending_branches:
        branch_dataset = ShootingBinaryEnvironmentDataset(
            snapshot,
            branches=pending_branches,
            timesteps=horizon_timesteps.tolist(),
            center_atom_ids=atom_ids,
            num_points=int(num_points),
            radius=float(radius),
            spatial_context_center_count=0,
        )
        branch_loader = make_shooting_environment_loader(
            branch_dataset,
            batch_size=int(environment_batch_size),
            num_workers=int(environment_num_workers),
            pin_memory=encoder.device.type == "cuda",
        )
        for batch in branch_loader:
            points = batch["points"]
            batch_count, horizon_count, center_batch_count, point_count, coordinate_dim = (
                points.shape
            )
            _, local = encode_spatial_context_state(
                encoder,
                points.reshape(
                    batch_count * horizon_count * center_batch_count,
                    point_count,
                    coordinate_dim,
                ),
                context_points=None,
                aggregation=str(spatial_context_aggregation),
                point_cloud_batch_size=int(point_cloud_batch_size),
            )
            future_values = local.numpy().reshape(
                batch_count, horizon_count, center_batch_count, local_dim
            )
            for batch_index, dataset_index_tensor in enumerate(batch["dataset_index"]):
                dataset_index = int(dataset_index_tensor.item())
                branch_position, branch_uid, shard_path, spec_sha = (
                    pending_branch_records[dataset_index]
                )
                payload = {
                    "future_z": future_values[batch_index].astype(np.float32, copy=False)
                }
                _write_shard(shard_path, spec_sha256=spec_sha, **payload)
                branch_payloads[branch_position] = payload
                print(
                    f"[shooting-embeddings] branch {branch_position + 1}/"
                    f"{len(snapshot.branches)} {branch_uid}",
                    flush=True,
                )

    if any(payload is None for payload in parent_payloads) or any(
        payload is None for payload in branch_payloads
    ):
        raise RuntimeError("Shooting embedding extraction left an unwritten parent or branch shard.")
    complete_parent_payloads = [payload for payload in parent_payloads if payload is not None]
    complete_branch_payloads = [payload for payload in branch_payloads if payload is not None]

    building = target.with_name(f"{target.name}.building-{os.getpid()}")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    shapes = {
        "parent_z": (len(snapshot.parents), center_count, input_dim),
        "parent_local_z": (len(snapshot.parents), center_count, local_dim),
        "parent_coords": (len(snapshot.parents), center_count, 3),
        "future_z": (
            len(snapshot.branches),
            resolved_horizons.size,
            center_count,
            local_dim,
        ),
        "branch_parent_index": (len(snapshot.branches),),
        "atom_ids": (center_count,),
        "horizons_ps": (resolved_horizons.size,),
    }
    arrays = {
        "parent_z": open_memmap(building / "parent_z.npy", mode="w+", dtype=np.float32, shape=shapes["parent_z"]),
        "parent_local_z": open_memmap(building / "parent_local_z.npy", mode="w+", dtype=np.float32, shape=shapes["parent_local_z"]),
        "parent_coords": open_memmap(building / "parent_coords.npy", mode="w+", dtype=np.float32, shape=shapes["parent_coords"]),
        "future_z": open_memmap(building / "future_z.npy", mode="w+", dtype=np.float32, shape=shapes["future_z"]),
        "branch_parent_index": open_memmap(building / "branch_parent_index.npy", mode="w+", dtype=np.int32, shape=shapes["branch_parent_index"]),
        "atom_ids": open_memmap(building / "atom_ids.npy", mode="w+", dtype=np.int64, shape=shapes["atom_ids"]),
        "horizons_ps": open_memmap(building / "horizons_ps.npy", mode="w+", dtype=np.float64, shape=shapes["horizons_ps"]),
    }
    for index, payload in enumerate(complete_parent_payloads):
        arrays["parent_z"][index] = payload["z"]
        arrays["parent_local_z"][index] = payload["local_z"]
        arrays["parent_coords"][index] = payload["coords"]
    for index, (branch, payload) in enumerate(zip(snapshot.branches, complete_branch_payloads)):
        arrays["future_z"][index] = payload["future_z"]
        arrays["branch_parent_index"][index] = parent_index_by_id[str(branch["parent_id"])]
    arrays["atom_ids"][:] = atom_ids
    arrays["horizons_ps"][:] = resolved_horizons
    for values in arrays.values():
        values.flush()
    del arrays

    final_manifest = {
        "version": 1,
        "spec": cache_spec,
        "loader": {
            "format": "float32_binary",
            "environment_batch_size": int(environment_batch_size),
            "environment_num_workers": int(environment_num_workers),
        },
        "snapshot": snapshot.to_dict(),
        "local_embedding_dim": local_dim,
        "input_embedding_dim": input_dim,
        "array_shapes": {name: list(shape) for name, shape in shapes.items()},
    }
    with (building / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(final_manifest, handle, indent=2, sort_keys=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(building, target)
    return ShootingEmbeddingCache.load(target)


__all__ = ["ShootingEmbeddingCache", "extract_shooting_embedding_cache"]
