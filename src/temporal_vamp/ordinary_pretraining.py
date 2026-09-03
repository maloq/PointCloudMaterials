"""Ordinary-trajectory cache and temporal pretraining for shooting ablation 5."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import Subset

from src.data_utils.temporal_binary_context_dataset import (
    TemporalBinaryContextDataset,
    make_temporal_binary_context_loader,
)
from src.temporal_vamp.embeddings import FrozenEncoder
from src.temporal_vamp.evaluation import CovariancePCA
from src.temporal_vamp.shooting_spatial import SpatialContextTransformer, SpatialTokenData


@dataclass(frozen=True)
class OrdinaryContextEmbeddingCache:
    path: Path
    manifest: dict[str, Any]
    token_z: np.ndarray
    token_descriptors: np.ndarray
    token_offsets: np.ndarray
    future_z: np.ndarray
    run_index: np.ndarray
    center_atom_id: np.ndarray
    anchor_frame: np.ndarray
    temperature_K: np.ndarray
    velocity_seed: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "OrdinaryContextEmbeddingCache":
        root = Path(path).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Ordinary context cache manifest is missing: {manifest_path}"
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        arrays = {
            name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in (
                "token_z",
                "token_descriptors",
                "token_offsets",
                "future_z",
                "run_index",
                "center_atom_id",
                "anchor_frame",
                "temperature_K",
                "velocity_seed",
            )
        }
        row_count = int(manifest["row_count"])
        for name, values in arrays.items():
            if int(values.shape[0]) != row_count:
                raise RuntimeError(
                    f"Ordinary cache row mismatch for {name}: expected={row_count}, "
                    f"observed={values.shape[0]}, root={root}."
                )
        return cls(path=root, manifest=manifest, **arrays)

    @property
    def tokens(self) -> SpatialTokenData:
        return SpatialTokenData(
            embeddings=self.token_z,
            descriptors=self.token_descriptors,
            offsets=self.token_offsets,
        )


@dataclass(frozen=True)
class OrdinaryPretrainingTargets:
    target_modes: np.ndarray
    split_rows: dict[str, np.ndarray]
    target_mean: np.ndarray
    target_scale: np.ndarray
    delta_mean: np.ndarray
    delta_scale: np.ndarray
    pcas: tuple[CovariancePCA, ...]


@dataclass(frozen=True)
class PretrainedSpatialBackbones:
    states: dict[int, dict[str, torch.Tensor]]
    histories: dict[int, dict[str, list[float]]]
    metrics: dict[int, dict[str, float | int]]


def _sha256_json(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@torch.inference_mode()
def _encode_chunks(
    encoder: FrozenEncoder, points: torch.Tensor, *, chunk_size: int
) -> np.ndarray:
    chunks = [
        encoder.encode(points[start : start + int(chunk_size)]).cpu()
        for start in range(0, int(points.shape[0]), int(chunk_size))
    ]
    if not chunks:
        raise RuntimeError("Ordinary temporal extraction received no point clouds.")
    return torch.cat(chunks, dim=0).numpy().astype(np.float32, copy=False)


def _write_shard(path: Path, *, spec_sha256: str, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, spec_sha256=np.asarray(spec_sha256), **arrays)
    os.replace(temporary, path)


def extract_ordinary_context_embedding_cache(
    dataset: TemporalBinaryContextDataset,
    *,
    encoder: FrozenEncoder,
    cache_path: str | Path,
    point_cloud_batch_size: int,
    environment_batch_size: int,
    environment_num_workers: int,
    force_recompute: bool,
) -> OrdinaryContextEmbeddingCache:
    target = Path(cache_path).expanduser().resolve()
    checkpoint_stat = encoder.checkpoint_path.stat()
    spec = {
        "version": 1,
        "runs": [entry.to_dict() for entry in dataset.entries],
        "center_atom_ids": dataset.center_atom_ids.tolist(),
        "horizons_ps": dataset.horizons_ps.tolist(),
        "anchor_stride_frames": dataset.anchor_stride_frames,
        "num_points": dataset.num_points,
        "radius": dataset.radius,
        "context_center_count": dataset.context_center_count,
        "checkpoint": str(encoder.checkpoint_path),
        "checkpoint_size_bytes": int(checkpoint_stat.st_size),
        "checkpoint_mtime_ns": int(checkpoint_stat.st_mtime_ns),
        "representation_source": encoder.representation_source,
        "encoder_repeats": encoder.repeats,
        "encoder_seed": encoder.seed,
    }
    spec_sha = _sha256_json(spec)
    manifest_path = target / "manifest.json"
    if manifest_path.is_file() and not force_recompute:
        cached = OrdinaryContextEmbeddingCache.load(target)
        if cached.manifest["spec"] != spec:
            raise RuntimeError(
                f"Ordinary cache specification changed at {target}; choose a new path "
                "or set force_recompute=true."
            )
        return cached
    if target.exists() and not force_recompute:
        raise RuntimeError(
            f"Ordinary cache exists without a complete manifest: {target}. "
            "Set force_recompute=true after inspecting it."
        )
    if force_recompute and target.exists():
        shutil.rmtree(target)
    shard_root = target.parent / f"{target.name}_shards"
    if force_recompute and shard_root.exists():
        shutil.rmtree(shard_root)
    shard_root.mkdir(parents=True, exist_ok=True)

    pending: list[int] = []
    for item_index in range(len(dataset)):
        shard = shard_root / f"{item_index:06d}.npz"
        if not shard.is_file():
            pending.append(item_index)
            continue
        with np.load(shard, allow_pickle=False) as payload:
            if str(payload["spec_sha256"].item()) != spec_sha:
                raise RuntimeError(
                    f"Ordinary extraction shard has a different specification: {shard}."
                )
    if pending:
        loader = make_temporal_binary_context_loader(
            Subset(dataset, pending),  # type: ignore[arg-type]
            batch_size=int(environment_batch_size),
            num_workers=int(environment_num_workers),
            pin_memory=encoder.device.type == "cuda",
        )
        completed = len(dataset) - len(pending)
        for batch in loader:
            token_points = batch["token_points"]
            future_points = batch["future_points"]
            batch_count, center_count, token_count, point_count, _ = token_points.shape
            horizon_count = int(future_points.shape[1])
            token_z = _encode_chunks(
                encoder,
                token_points.reshape(-1, point_count, 3),
                chunk_size=int(point_cloud_batch_size),
            ).reshape(batch_count, center_count, token_count, -1)
            future_z = _encode_chunks(
                encoder,
                future_points.permute(0, 2, 1, 3, 4).reshape(-1, point_count, 3),
                chunk_size=int(point_cloud_batch_size),
            ).reshape(batch_count, center_count, horizon_count, -1)
            dataset_indices = np.asarray(batch["dataset_index"], dtype=np.int64)
            for position, dataset_index in enumerate(dataset_indices.tolist()):
                _write_shard(
                    shard_root / f"{dataset_index:06d}.npz",
                    spec_sha256=spec_sha,
                    token_z=token_z[position],
                    token_descriptors=np.asarray(
                        batch["token_descriptors"][position], dtype=np.float32
                    ),
                    token_offsets=np.asarray(
                        batch["token_offsets"][position], dtype=np.float32
                    ),
                    future_z=future_z[position],
                    run_index=np.asarray(batch["run_index"][position], dtype=np.int32),
                    anchor_frame=np.asarray(
                        batch["anchor_frame"][position], dtype=np.int64
                    ),
                    temperature_K=np.asarray(
                        batch["temperature_K"][position], dtype=np.float32
                    ),
                    velocity_seed=np.asarray(
                        batch["velocity_seed"][position], dtype=np.int64
                    ),
                )
            completed += int(batch_count)
            print(
                f"[ordinary-pretraining] extracted anchors={completed}/{len(dataset)}",
                flush=True,
            )

    building = target.parent / f".{target.name}.building-{os.getpid()}"
    if building.exists():
        raise FileExistsError(f"Ordinary cache build directory already exists: {building}")
    building.mkdir(parents=True)
    row_count = len(dataset) * int(dataset.center_atom_ids.size)
    embedding_dim = int(encoder.output_dim)
    token_count = int(dataset.context_center_count + 1)
    horizon_count = int(dataset.horizons_ps.size)
    shapes = {
        "token_z": (row_count, token_count, embedding_dim),
        "token_descriptors": (row_count, token_count, 3),
        "token_offsets": (row_count, token_count, 3),
        "future_z": (row_count, horizon_count, embedding_dim),
        "run_index": (row_count,),
        "center_atom_id": (row_count,),
        "anchor_frame": (row_count,),
        "temperature_K": (row_count,),
        "velocity_seed": (row_count,),
    }
    dtypes = {
        "token_z": np.float32,
        "token_descriptors": np.float32,
        "token_offsets": np.float32,
        "future_z": np.float32,
        "run_index": np.int32,
        "center_atom_id": np.int64,
        "anchor_frame": np.int64,
        "temperature_K": np.float32,
        "velocity_seed": np.int64,
    }
    arrays = {
        name: open_memmap(building / f"{name}.npy", mode="w+", dtype=dtypes[name], shape=shape)
        for name, shape in shapes.items()
    }
    center_count = int(dataset.center_atom_ids.size)
    for item_index, record in enumerate(dataset.records):
        shard_path = shard_root / f"{item_index:06d}.npz"
        with np.load(shard_path, allow_pickle=False) as payload:
            if str(payload["spec_sha256"].item()) != spec_sha:
                raise RuntimeError(f"Shard specification changed during consolidation: {shard_path}")
            rows = slice(item_index * center_count, (item_index + 1) * center_count)
            for name in ("token_z", "token_descriptors", "token_offsets", "future_z"):
                arrays[name][rows] = payload[name]
            arrays["run_index"][rows] = int(payload["run_index"])
            arrays["center_atom_id"][rows] = dataset.center_atom_ids
            arrays["anchor_frame"][rows] = int(payload["anchor_frame"])
            arrays["temperature_K"][rows] = float(payload["temperature_K"])
            arrays["velocity_seed"][rows] = int(payload["velocity_seed"])
    for values in arrays.values():
        values.flush()
    manifest = {
        "state": "complete",
        "spec": spec,
        "spec_sha256": spec_sha,
        "row_count": row_count,
        "anchor_count": len(dataset),
        "run_ids": [entry.run_id for entry in dataset.entries],
        "array_shapes": {name: list(shape) for name, shape in shapes.items()},
    }
    with (building / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(building, target)
    return OrdinaryContextEmbeddingCache.load(target)


def prepare_ordinary_pretraining_targets(
    cache: OrdinaryContextEmbeddingCache,
    *,
    optimization_velocity_seeds: Sequence[int],
    selection_velocity_seeds: Sequence[int],
    pca_dim_per_horizon: int,
) -> OrdinaryPretrainingTargets:
    velocity_seed = np.asarray(cache.velocity_seed, dtype=np.int64)
    optimization_rows = np.flatnonzero(
        np.isin(velocity_seed, np.asarray(optimization_velocity_seeds, dtype=np.int64))
    )
    selection_rows = np.flatnonzero(
        np.isin(velocity_seed, np.asarray(selection_velocity_seeds, dtype=np.int64))
    )
    if optimization_rows.size == 0 or selection_rows.size == 0:
        raise RuntimeError(
            f"Ordinary pretraining split is empty: optimization={optimization_rows.size}, "
            f"selection={selection_rows.size}."
        )
    central = np.asarray(cache.token_z[:, 0], dtype=np.float64)
    future = np.asarray(cache.future_z, dtype=np.float64)
    delta = future - central[:, None, :]
    delta_mean = delta[optimization_rows].mean(axis=0)
    delta_scale = delta[optimization_rows].std(axis=0)
    delta_scale = np.where(delta_scale <= 1.0e-10, 1.0, delta_scale)
    blocks: list[np.ndarray] = []
    pcas: list[CovariancePCA] = []
    for horizon_index in range(delta.shape[1]):
        standardized = (delta[:, horizon_index] - delta_mean[horizon_index]) / delta_scale[
            horizon_index
        ]
        pca = CovariancePCA.fit(
            standardized[optimization_rows], dimension=int(pca_dim_per_horizon)
        )
        blocks.append(pca.transform(standardized, dimension=int(pca_dim_per_horizon)))
        pcas.append(pca)
    raw_target = np.concatenate(blocks, axis=1)
    target_mean = raw_target[optimization_rows].mean(axis=0)
    target_scale = raw_target[optimization_rows].std(axis=0)
    target_scale = np.where(target_scale <= 1.0e-10, 1.0, target_scale)
    target_modes = ((raw_target - target_mean) / target_scale).astype(np.float32)
    return OrdinaryPretrainingTargets(
        target_modes=target_modes,
        split_rows={"optimization": optimization_rows, "selection": selection_rows},
        target_mean=target_mean,
        target_scale=target_scale,
        delta_mean=delta_mean,
        delta_scale=delta_scale,
        pcas=tuple(pcas),
    )


def _backbone_state(model: SpatialContextTransformer) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("prediction_head.")
    }


def pretrain_spatial_context_backbones(
    tokens: SpatialTokenData,
    targets: OrdinaryPretrainingTargets,
    *,
    embedding_mean: np.ndarray,
    embedding_scale: np.ndarray,
    descriptor_mean: np.ndarray,
    descriptor_scale: np.ndarray,
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
) -> PretrainedSpatialBackbones:
    standardized_embeddings = np.asarray(
        (tokens.embeddings - embedding_mean) / embedding_scale, dtype=np.float32
    )
    standardized_descriptors = np.asarray(
        (tokens.descriptors - descriptor_mean) / descriptor_scale, dtype=np.float32
    )
    torch_device = torch.device(device)
    optimization_rows = np.asarray(targets.split_rows["optimization"], dtype=np.int64)
    selection_rows = torch.from_numpy(targets.split_rows["selection"]).to(torch_device)
    selection_target = torch.from_numpy(targets.target_modes).to(torch_device)[selection_rows]
    histories: dict[int, dict[str, list[float]]] = {}
    metrics: dict[int, dict[str, float | int]] = {}
    states: dict[int, dict[str, torch.Tensor]] = {}
    for raw_seed in seeds:
        seed = int(raw_seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = SpatialContextTransformer(
            embedding_dim=int(tokens.embeddings.shape[-1]),
            descriptor_dim=int(tokens.descriptors.shape[-1]),
            hidden_dim=int(hidden_dim),
            heads=int(heads),
            blocks=int(blocks),
            rbf_dim=int(rbf_dim),
            maximum_radius=float(maximum_radius),
            representation_dim=int(representation_dim),
            target_dim=int(targets.target_modes.shape[1]),
            dropout=float(dropout),
        ).to(torch_device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
        )
        generator = np.random.default_rng(seed)
        best_state: dict[str, torch.Tensor] | None = None
        best_selection = float("inf")
        best_epoch = -1
        history = {"optimization": [], "selection": []}
        for epoch in range(int(maximum_epochs)):
            permutation = generator.permutation(optimization_rows)
            model.train()
            total_loss = 0.0
            for start in range(0, permutation.size, int(batch_size)):
                rows_np = permutation[start : start + int(batch_size)]
                rows = torch.from_numpy(rows_np).to(torch_device)
                embeddings = torch.from_numpy(standardized_embeddings[rows_np]).to(torch_device)
                descriptors = torch.from_numpy(standardized_descriptors[rows_np]).to(torch_device)
                offsets = torch.from_numpy(np.asarray(tokens.offsets[rows_np], dtype=np.float32)).to(
                    torch_device
                )
                target = torch.from_numpy(targets.target_modes[rows_np]).to(torch_device)
                _, prediction = model(embeddings, descriptors, offsets)
                loss = torch.mean((prediction - target) ** 2)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * int(rows.numel())
            optimization_loss = total_loss / float(permutation.size)
            model.eval()
            selection_loss_sum = 0.0
            with torch.no_grad():
                for start in range(0, int(selection_rows.numel()), int(batch_size)):
                    selected = selection_rows[start : start + int(batch_size)]
                    selected_np = selected.cpu().numpy()
                    _, prediction = model(
                        torch.from_numpy(standardized_embeddings[selected_np]).to(torch_device),
                        torch.from_numpy(standardized_descriptors[selected_np]).to(torch_device),
                        torch.from_numpy(
                            np.asarray(tokens.offsets[selected_np], dtype=np.float32)
                        ).to(torch_device),
                    )
                    selection_loss_sum += float(
                        torch.sum((prediction - selection_target[start : start + selected.numel()]) ** 2)
                    )
            selection_loss = selection_loss_sum / float(
                selection_rows.numel() * targets.target_modes.shape[1]
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
            raise RuntimeError(f"Ordinary temporal pretraining seed={seed} made no checkpoint.")
        model.load_state_dict(best_state)
        states[seed] = _backbone_state(model)
        histories[seed] = history
        metrics[seed] = {
            "best_epoch": best_epoch,
            "epochs_run": len(history["selection"]),
            "best_selection_mse": best_selection,
        }
        print(
            f"[ordinary-pretraining] seed={seed} best_epoch={best_epoch} "
            f"selection_mse={best_selection:.6f}",
            flush=True,
        )
    return PretrainedSpatialBackbones(states=states, histories=histories, metrics=metrics)


def save_pretrained_spatial_backbones(
    fitted: PretrainedSpatialBackbones,
    targets: OrdinaryPretrainingTargets,
    path: str | Path,
) -> None:
    torch.save(
        {
            "backbone_states": fitted.states,
            "histories": fitted.histories,
            "metrics": fitted.metrics,
            "target_mean": targets.target_mean,
            "target_scale": targets.target_scale,
            "delta_mean": targets.delta_mean,
            "delta_scale": targets.delta_scale,
            "target_pca_means": np.stack([value.mean_ for value in targets.pcas]),
            "target_pca_components": np.stack([value.components_ for value in targets.pcas]),
            "target_pca_eigenvalues": np.stack([value.eigenvalues_ for value in targets.pcas]),
        },
        Path(path),
    )


__all__ = [
    "OrdinaryContextEmbeddingCache",
    "OrdinaryPretrainingTargets",
    "PretrainedSpatialBackbones",
    "extract_ordinary_context_embedding_cache",
    "prepare_ordinary_pretraining_targets",
    "pretrain_spatial_context_backbones",
    "save_pretrained_spatial_backbones",
]
