from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.lib.format import open_memmap

from src.baselines.descriptor_baselines import SteinhardtDescriptorBaseline
from src.data_utils.shooting_binary_dataset import (
    ShootingBinaryEnvironmentDataset,
    make_shooting_environment_loader,
)
from src.data_utils.shooting_dataset import (
    ShootingCampaignSnapshot,
    resolve_shooting_trajectory_path,
    shooting_snapshot_sha256,
)
from src.temporal_vamp.embeddings import FrozenEncoder
from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache


@dataclass(frozen=True)
class ShootingContextTokenCache:
    path: Path
    manifest: dict[str, Any]
    satellite_z: np.ndarray
    satellite_offsets: np.ndarray
    central_descriptors: np.ndarray
    satellite_descriptors: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "ShootingContextTokenCache":
        root = Path(path).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Shooting context-token manifest is missing: {manifest_path}"
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        arrays = {
            name: np.load(root / f"{name}.npy", mmap_mode="r")
            for name in (
                "satellite_z",
                "satellite_offsets",
                "central_descriptors",
                "satellite_descriptors",
            )
        }
        for name, values in arrays.items():
            expected = tuple(manifest["array_shapes"][name])
            if tuple(values.shape) != expected:
                raise RuntimeError(
                    f"Shooting context-token array shape mismatch for {name}: "
                    f"expected={expected}, observed={tuple(values.shape)}, path={root}."
                )
        return cls(path=root, manifest=manifest, **arrays)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _branch_trajectory_path(
    snapshot: ShootingCampaignSnapshot, branch: dict[str, Any]
) -> Path:
    root = (
        snapshot.root
        if len(snapshot.campaign_roots) == 1
        else Path(str(branch["campaign_root"]))
    )
    return resolve_shooting_trajectory_path(root, branch)


def _load_shard(path: Path, *, expected_sha256: str) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as payload:
        observed = str(payload["spec_sha256"].item())
        if observed != expected_sha256:
            raise RuntimeError(
                f"Context-token shard specification changed: path={path}, "
                f"expected={expected_sha256}, observed={observed}. Remove the derived "
                "shard or use force_recompute=true."
            )
        return {
            name: payload[name].copy()
            for name in payload.files
            if name != "spec_sha256"
        }


def _write_shard(path: Path, *, spec_sha256: str, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, spec_sha256=np.asarray(spec_sha256), **arrays)
    os.replace(temporary, path)


@torch.inference_mode()
def _encode_context_points(
    encoder: FrozenEncoder,
    context_points: torch.Tensor,
    *,
    point_cloud_batch_size: int,
) -> np.ndarray:
    batch_size, context_count, point_count, coordinate_dim = context_points.shape
    flat = context_points.reshape(
        batch_size * context_count, point_count, coordinate_dim
    )
    chunks = [
        encoder.encode(flat[start : start + int(point_cloud_batch_size)]).cpu()
        for start in range(0, int(flat.shape[0]), int(point_cloud_batch_size))
    ]
    if not chunks:
        raise RuntimeError("Context-token extraction produced an empty point-cloud batch.")
    return (
        torch.cat(chunks, dim=0)
        .reshape(batch_size, context_count, -1)
        .numpy()
        .astype(np.float32, copy=False)
    )


def extract_shooting_context_token_cache(
    snapshot: ShootingCampaignSnapshot,
    base_cache: ShootingEmbeddingCache,
    *,
    encoder: FrozenEncoder,
    cache_path: str | Path,
    num_points: int,
    radius: float,
    context_center_count: int,
    point_cloud_batch_size: int,
    environment_batch_size: int,
    environment_num_workers: int,
    steinhardt_shell_min_neighbors: int,
    steinhardt_shell_max_neighbors: int,
    force_recompute: bool,
) -> ShootingContextTokenCache:
    """Cache individual GeoFrame satellite tokens and invariant local descriptors.

    Only the immutable parent configurations are encoded. Future branch embeddings
    remain in ``ShootingEmbeddingCache`` and are not recomputed for this ablation.
    """

    target = Path(cache_path).expanduser().resolve()
    if int(context_center_count) <= 0 or int(context_center_count) >= int(num_points):
        raise ValueError(
            "context_center_count must be positive and smaller than num_points; "
            f"got context_center_count={context_center_count}, num_points={num_points}."
        )
    snapshot_hash = shooting_snapshot_sha256(snapshot)
    base_snapshot_hash = str(base_cache.manifest["spec"]["snapshot_sha256"])
    if snapshot_hash != base_snapshot_hash:
        raise RuntimeError(
            "The context-token snapshot does not match the existing shooting embedding "
            f"cache: snapshot={snapshot_hash}, base_cache={base_snapshot_hash}."
        )
    atom_ids = np.asarray(base_cache.atom_ids, dtype=np.int64)
    checkpoint_stat = encoder.checkpoint_path.stat()
    spec = {
        "version": 1,
        "snapshot_sha256": snapshot_hash,
        "base_embedding_cache": str(base_cache.path),
        "base_embedding_spec_sha256": _sha256_json(base_cache.manifest["spec"]),
        "checkpoint": str(encoder.checkpoint_path),
        "checkpoint_size_bytes": int(checkpoint_stat.st_size),
        "checkpoint_mtime_ns": int(checkpoint_stat.st_mtime_ns),
        "representation_source": encoder.representation_source,
        "encoder_repeats": int(encoder.repeats),
        "encoder_seed": int(encoder.seed),
        "center_atom_ids": atom_ids.tolist(),
        "num_points": int(num_points),
        "radius": float(radius),
        "context_center_count": int(context_center_count),
        "point_cloud_batch_size": int(point_cloud_batch_size),
        "descriptors": {
            "names": ["q4", "q6", "first_shell_size"],
            "shell_min_neighbors": int(steinhardt_shell_min_neighbors),
            "shell_max_neighbors": int(steinhardt_shell_max_neighbors),
        },
    }
    manifest_path = target / "manifest.json"
    if target.exists() and manifest_path.is_file() and not force_recompute:
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing["spec"] != spec:
            raise RuntimeError(
                f"Shooting context-token cache specification changed: {target}. "
                "Use force_recompute=true or choose a new cache path."
            )
        return ShootingContextTokenCache.load(target)
    if target.exists() and not manifest_path.is_file() and not force_recompute:
        raise RuntimeError(
            f"Context-token cache exists without a final manifest: {target}. "
            "Use force_recompute=true to rebuild it."
        )
    if force_recompute and target.exists():
        shutil.rmtree(target)

    shard_root = target.parent / f"{target.name}_shards"
    if force_recompute and shard_root.exists():
        shutil.rmtree(shard_root)
    shard_root.mkdir(parents=True, exist_ok=True)
    branches_by_parent: dict[str, list[dict[str, Any]]] = {
        str(parent["parent_id"]): [] for parent in snapshot.parents
    }
    for branch in snapshot.branches:
        branches_by_parent[str(branch["parent_id"])].append(branch)

    descriptor = SteinhardtDescriptorBaseline(
        l_values=[4, 6],
        center_atom_tolerance=1.0e-6,
        shell_min_neighbors=int(steinhardt_shell_min_neighbors),
        shell_max_neighbors=int(steinhardt_shell_max_neighbors),
        append_shell_size=True,
    )
    payloads: list[dict[str, np.ndarray] | None] = [None for _ in snapshot.parents]
    pending_branches: list[dict[str, Any]] = []
    pending_records: list[tuple[int, str, Path, str]] = []
    for parent_position, parent in enumerate(snapshot.parents):
        parent_id = str(parent["parent_id"])
        representative = branches_by_parent[parent_id][0]
        parent_spec = {
            "cache_spec": spec,
            "parent": parent,
            "representative_trajectory": str(
                _branch_trajectory_path(snapshot, representative)
            ),
        }
        spec_sha = _sha256_json(parent_spec)
        shard_path = shard_root / f"{parent_id}.npz"
        payload = _load_shard(shard_path, expected_sha256=spec_sha)
        if payload is not None:
            payloads[parent_position] = payload
            print(
                f"[shooting-context] parent {parent_position + 1}/{len(snapshot.parents)} "
                f"{parent_id} (cached)",
                flush=True,
            )
            continue
        pending_branches.append(representative)
        pending_records.append((parent_position, parent_id, shard_path, spec_sha))

    if pending_branches:
        dataset = ShootingBinaryEnvironmentDataset(
            snapshot,
            branches=pending_branches,
            timesteps=[0],
            center_atom_ids=atom_ids,
            num_points=int(num_points),
            radius=float(radius),
            spatial_context_center_count=int(context_center_count),
        )
        loader = make_shooting_environment_loader(
            dataset,
            batch_size=int(environment_batch_size),
            num_workers=int(environment_num_workers),
            pin_memory=encoder.device.type == "cuda",
        )
        for batch in loader:
            points = batch["points"][:, 0]
            context_points = batch["context_points"][:, 0]
            offsets = batch["context_center_offsets"][:, 0]
            batch_count, center_batch_count, point_count, coordinate_dim = points.shape
            observed_context_count = int(context_points.shape[2])
            if observed_context_count != int(context_center_count):
                raise RuntimeError(
                    "Shooting context loader returned the wrong satellite count: "
                    f"expected={context_center_count}, observed={observed_context_count}."
                )
            satellite_z = _encode_context_points(
                encoder,
                context_points.reshape(
                    batch_count * center_batch_count,
                    observed_context_count,
                    point_count,
                    coordinate_dim,
                ),
                point_cloud_batch_size=int(point_cloud_batch_size),
            ).reshape(batch_count, center_batch_count, observed_context_count, -1)
            central_descriptors = descriptor.transform(
                points.reshape(
                    batch_count * center_batch_count, point_count, coordinate_dim
                ).numpy()
            ).reshape(batch_count, center_batch_count, -1)
            satellite_descriptors = descriptor.transform(
                context_points.reshape(
                    batch_count * center_batch_count * observed_context_count,
                    point_count,
                    coordinate_dim,
                ).numpy()
            ).reshape(batch_count, center_batch_count, observed_context_count, -1)
            for batch_index, dataset_index_tensor in enumerate(batch["dataset_index"]):
                dataset_index = int(dataset_index_tensor.item())
                parent_position, parent_id, shard_path, spec_sha = pending_records[
                    dataset_index
                ]
                payload = {
                    "satellite_z": satellite_z[batch_index].astype(
                        np.float32, copy=False
                    ),
                    "satellite_offsets": offsets[batch_index].numpy().astype(
                        np.float32, copy=False
                    ),
                    "central_descriptors": central_descriptors[batch_index].astype(
                        np.float32, copy=False
                    ),
                    "satellite_descriptors": satellite_descriptors[batch_index].astype(
                        np.float32, copy=False
                    ),
                }
                _write_shard(shard_path, spec_sha256=spec_sha, **payload)
                payloads[parent_position] = payload
                print(
                    f"[shooting-context] parent {parent_position + 1}/"
                    f"{len(snapshot.parents)} {parent_id}",
                    flush=True,
                )

    if any(payload is None for payload in payloads):
        raise RuntimeError("Shooting context extraction left an unwritten parent shard.")
    complete_payloads = [payload for payload in payloads if payload is not None]

    parent_count = len(snapshot.parents)
    center_count = int(atom_ids.size)
    context_count = int(context_center_count)
    embedding_dim = int(base_cache.parent_local_z.shape[-1])
    descriptor_dim = 3
    shapes = {
        "satellite_z": (
            parent_count,
            center_count,
            context_count,
            embedding_dim,
        ),
        "satellite_offsets": (parent_count, center_count, context_count, 3),
        "central_descriptors": (parent_count, center_count, descriptor_dim),
        "satellite_descriptors": (
            parent_count,
            center_count,
            context_count,
            descriptor_dim,
        ),
    }
    building = target.with_name(f"{target.name}.building-{os.getpid()}")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    arrays = {
        name: open_memmap(
            building / f"{name}.npy",
            mode="w+",
            dtype=np.float32,
            shape=shape,
        )
        for name, shape in shapes.items()
    }
    for parent_index, payload in enumerate(complete_payloads):
        for name in arrays:
            arrays[name][parent_index] = payload[name]
    for values in arrays.values():
        values.flush()
    del arrays
    final_manifest = {
        "version": 1,
        "spec": spec,
        "loader": {
            "format": "float32_binary",
            "environment_batch_size": int(environment_batch_size),
            "environment_num_workers": int(environment_num_workers),
        },
        "array_shapes": {name: list(shape) for name, shape in shapes.items()},
    }
    with (building / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(final_manifest, handle, indent=2, sort_keys=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(building, target)
    return ShootingContextTokenCache.load(target)


__all__ = [
    "ShootingContextTokenCache",
    "extract_shooting_context_token_cache",
]
