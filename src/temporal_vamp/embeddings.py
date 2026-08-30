from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from numpy.lib.format import open_memmap
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.models.encoders.runtime import EncoderAdapter, resolve_encoder_output_dim
from src.temporal_vamp.data import TemporalPairDataset, identity_batch_collate
from src.training_methods.registry import resolve_training_method
from src.utils.model_utils import load_model_from_checkpoint, resolve_config_path


_BASE_CACHE_ARRAYS = {
    "z0": np.float32,
    "z1": np.float32,
    "atom_id": np.int64,
    "run_index": np.int32,
    "frame0": np.int64,
    "frame1": np.int64,
    "timestep0": np.int64,
    "timestep1": np.int64,
    "coords0": np.float32,
    "coords1": np.float32,
}

_METADATA_CACHE_ARRAYS = {
    "time_ps0": np.float64,
    "time_ps1": np.float64,
    "temperature_K": np.float32,
    "pressure_GPa": np.float32,
    "velocity_seed": np.int64,
    "crystalline_fraction0": np.float32,
    "crystalline_fraction1": np.float32,
    "largest_crystalline_cluster_atoms0": np.int64,
    "largest_crystalline_cluster_atoms1": np.int64,
}

_SPATIAL_CONTEXT_CACHE_ARRAYS = {
    "local_z0": np.float32,
    "local_z1": np.float32,
}


@dataclass(frozen=True)
class EmbeddingCache:
    path: Path
    manifest: dict[str, Any]
    z0: np.ndarray
    z1: np.ndarray
    atom_id: np.ndarray
    run_index: np.ndarray
    frame0: np.ndarray
    frame1: np.ndarray
    timestep0: np.ndarray
    timestep1: np.ndarray
    coords0: np.ndarray
    coords1: np.ndarray
    time_ps0: np.ndarray | None = None
    time_ps1: np.ndarray | None = None
    temperature_K: np.ndarray | None = None
    pressure_GPa: np.ndarray | None = None
    velocity_seed: np.ndarray | None = None
    crystalline_fraction0: np.ndarray | None = None
    crystalline_fraction1: np.ndarray | None = None
    largest_crystalline_cluster_atoms0: np.ndarray | None = None
    largest_crystalline_cluster_atoms1: np.ndarray | None = None
    local_z0: np.ndarray | None = None
    local_z1: np.ndarray | None = None

    @property
    def run_ids(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.manifest["run_ids"])

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingCache":
        cache_path = Path(path).expanduser().resolve()
        manifest_path = cache_path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Embedding cache manifest is missing: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        arrays: dict[str, np.ndarray] = {}
        array_specs = dict(_BASE_CACHE_ARRAYS)
        if bool(manifest.get("has_simulation_metadata", False)):
            array_specs.update(_METADATA_CACHE_ARRAYS)
        if bool(manifest.get("has_spatial_context", False)):
            array_specs.update(_SPATIAL_CONTEXT_CACHE_ARRAYS)
        for name in array_specs:
            array_path = cache_path / f"{name}.npy"
            if not array_path.is_file():
                raise FileNotFoundError(
                    f"Embedding cache array {name!r} is missing: {array_path}"
                )
            arrays[name] = np.load(array_path, mmap_mode="r")
        expected_samples = int(manifest["sample_count"])
        for name, values in arrays.items():
            if int(values.shape[0]) != expected_samples:
                raise RuntimeError(
                    f"Embedding cache {name}.npy has {values.shape[0]} rows, "
                    f"manifest declares {expected_samples}: {cache_path}"
                )
        return cls(path=cache_path, manifest=manifest, **arrays)


@dataclass
class FrozenEncoder:
    model: torch.nn.Module
    checkpoint_config: DictConfig
    checkpoint_path: Path
    device: torch.device
    repeats: int
    deterministic: bool
    seed: int
    representation_source: str

    @property
    def output_dim(self) -> int:
        if self.representation_source == "vicreg_projector":
            return int(self.model.vicreg.embed_dim)
        return int(resolve_encoder_output_dim(self.model.encoder))

    @torch.inference_mode()
    def encode(self, points: torch.Tensor) -> torch.Tensor:
        inputs = points.to(device=self.device, dtype=torch.float32, non_blocking=True)
        inputs = self.model._prepare_model_input(inputs)
        cuda_devices = (
            [self.device.index if self.device.index is not None else torch.cuda.current_device()]
            if self.device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(self.seed))
            accumulated: torch.Tensor | None = None
            for _ in range(int(self.repeats)):
                encoded = self.model.encoder_io.encode(inputs)
                encoder_features = self.model._contrastive_invariant_latent(
                    encoded.invariant,
                    encoded.equivariant,
                )
                representation = self.model._output_representation(encoder_features).to(
                    torch.float32
                )
                accumulated = (
                    representation if accumulated is None else accumulated + representation
                )
        assert accumulated is not None
        return accumulated / float(self.repeats)


def load_frozen_encoder(
    checkpoint_path: str | Path,
    *,
    device: str,
    repeats: int = 1,
    seed: int = 0,
    representation_source: str = "checkpoint",
) -> FrozenEncoder:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    config_dir, config_name = resolve_config_path(str(checkpoint))
    config_path = Path(config_dir) / f"{config_name}.yaml"
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    encoder_name = str(cfg.encoder.name)
    if encoder_name != "GeoFrameTransformer":
        raise ValueError(
            "The temporal VAMP prototype is configured to use GeoFrameTransformer, "
            f"but checkpoint {checkpoint} declares encoder.name={encoder_name!r}."
        )
    if int(repeats) <= 0:
        raise ValueError(f"embedding repeats must be > 0, got {repeats}.")

    method = resolve_training_method(cfg)
    module_class = method.load_module_class()
    model = load_model_from_checkpoint(
        str(checkpoint),
        cfg,
        device=device,
        module=module_class,
    )
    compiled_encoder = model.encoder
    if hasattr(compiled_encoder, "_orig_mod"):
        model.encoder = compiled_encoder._orig_mod
        model.encoder_io = EncoderAdapter(model.encoder)
    model.requires_grad_(False)
    model.eval()

    resolved_representation = str(representation_source).strip().lower()
    if resolved_representation == "checkpoint":
        resolved_representation = str(
            OmegaConf.select(cfg, "representation_source", default="encoder")
        ).strip().lower()
    if resolved_representation not in {"encoder", "vicreg_projector"}:
        raise ValueError(
            "encoder.representation_source must be checkpoint, encoder, or "
            f"vicreg_projector; got {representation_source!r}."
        )
    if resolved_representation == "vicreg_projector" and model.vicreg.projector is None:
        raise ValueError(
            f"Checkpoint {checkpoint} was requested with vicreg_projector representation, "
            "but its VICReg projector is not active."
        )
    model.representation_source = resolved_representation

    group = model.encoder.token_encoder.group_divider
    deterministic = bool(group.deterministic_fps)
    if not deterministic and int(repeats) == 1:
        raise ValueError(
            "This GeoFrameTransformer checkpoint uses non-deterministic eval-time FPS. "
            "Set embedding.repeats > 1 to average repeated encodings, or use a checkpoint "
            "trained with encoder.kwargs.deterministic_fps=true."
        )
    return FrozenEncoder(
        model=model,
        checkpoint_config=cfg,
        checkpoint_path=checkpoint,
        device=torch.device(device),
        repeats=int(repeats),
        deterministic=deterministic,
        seed=int(seed),
        representation_source=resolved_representation,
    )


def _allocate_cache_arrays(
    directory: Path,
    *,
    sample_count: int,
    embedding_dim: int,
    local_embedding_dim: int,
    has_simulation_metadata: bool,
    has_spatial_context: bool,
) -> dict[str, np.memmap]:
    shapes = {
        "z0": (sample_count, embedding_dim),
        "z1": (sample_count, embedding_dim),
        "atom_id": (sample_count,),
        "run_index": (sample_count,),
        "frame0": (sample_count,),
        "frame1": (sample_count,),
        "timestep0": (sample_count,),
        "timestep1": (sample_count,),
        "coords0": (sample_count, 3),
        "coords1": (sample_count, 3),
    }
    if has_simulation_metadata:
        shapes.update(
            {
                "time_ps0": (sample_count,),
                "time_ps1": (sample_count,),
                "temperature_K": (sample_count,),
                "pressure_GPa": (sample_count,),
                "velocity_seed": (sample_count,),
                "crystalline_fraction0": (sample_count,),
                "crystalline_fraction1": (sample_count,),
                "largest_crystalline_cluster_atoms0": (sample_count,),
                "largest_crystalline_cluster_atoms1": (sample_count,),
            }
        )
    if has_spatial_context:
        shapes.update(
            {
                "local_z0": (sample_count, local_embedding_dim),
                "local_z1": (sample_count, local_embedding_dim),
            }
        )
    array_specs = dict(_BASE_CACHE_ARRAYS)
    if has_simulation_metadata:
        array_specs.update(_METADATA_CACHE_ARRAYS)
    if has_spatial_context:
        array_specs.update(_SPATIAL_CONTEXT_CACHE_ARRAYS)
    return {
        name: open_memmap(
            directory / f"{name}.npy",
            mode="w+",
            dtype=array_specs[name],
            shape=shape,
        )
        for name, shape in shapes.items()
    }


def _encode_point_clouds_in_chunks(
    encoder: FrozenEncoder,
    points: torch.Tensor,
    *,
    point_cloud_batch_size: int,
) -> torch.Tensor:
    chunk_size = int(point_cloud_batch_size)
    if chunk_size <= 0:
        raise ValueError(
            f"embedding.point_cloud_batch_size must be > 0, got {point_cloud_batch_size}."
        )
    encoded = [
        encoder.encode(points[start : start + chunk_size]).cpu()
        for start in range(0, int(points.shape[0]), chunk_size)
    ]
    if not encoded:
        raise ValueError("Cannot encode an empty point-cloud batch.")
    return torch.cat(encoded, dim=0)


def encode_spatial_context_state(
    encoder: FrozenEncoder,
    points: torch.Tensor,
    *,
    context_points: torch.Tensor | None,
    aggregation: str,
    point_cloud_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode one time slice and append invariant satellite summary statistics."""
    local = _encode_point_clouds_in_chunks(
        encoder,
        points,
        point_cloud_batch_size=point_cloud_batch_size,
    )
    if context_points is None:
        return local, local
    if aggregation != "mean_std":
        raise ValueError(
            "data.spatial_context.aggregation currently supports exactly 'mean_std', "
            f"got {aggregation!r}."
        )
    if context_points.ndim != 4 or context_points.shape[0] != points.shape[0]:
        raise ValueError(
            "Context point clouds must have shape (B, K, N, 3) aligned with central clouds, "
            f"got central={tuple(points.shape)}, context={tuple(context_points.shape)}."
        )
    batch_size, context_count, point_count, coordinate_dim = context_points.shape
    if context_count <= 0 or coordinate_dim != 3:
        raise ValueError(
            f"Context point clouds require K>0 and xyz coordinates, got {tuple(context_points.shape)}."
        )
    context_local = _encode_point_clouds_in_chunks(
        encoder,
        context_points.reshape(batch_size * context_count, point_count, coordinate_dim),
        point_cloud_batch_size=point_cloud_batch_size,
    ).reshape(batch_size, context_count, -1)
    context_mean = context_local.mean(dim=1)
    context_std = context_local.std(dim=1, unbiased=False)
    return torch.cat([local, context_mean, context_std], dim=1), local


def extract_embedding_cache(
    datasets: Sequence[TemporalPairDataset],
    *,
    encoder: FrozenEncoder,
    cache_path: str | Path,
    cache_spec: dict[str, Any],
    batch_size: int,
    point_cloud_batch_size: int,
    num_workers: int,
    force_recompute: bool,
    spatial_context_aggregation: str = "mean_std",
) -> EmbeddingCache:
    target = Path(cache_path).expanduser().resolve()
    if int(batch_size) <= 0 or int(num_workers) < 0:
        raise ValueError(
            f"Expected batch_size>0 and num_workers>=0, got {batch_size}, {num_workers}."
        )
    if not datasets:
        raise ValueError("Embedding extraction requires at least one temporal-pair dataset.")
    sample_count = sum(len(dataset) for dataset in datasets)
    run_ids = [dataset.run_id for dataset in datasets]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError(f"run_id values must be unique within a split, got {run_ids}.")
    metadata_flags = {dataset.metadata is not None for dataset in datasets}
    if len(metadata_flags) != 1:
        raise ValueError(
            "A temporal embedding split cannot mix trajectories with and without simulation metadata."
        )
    has_simulation_metadata = metadata_flags.pop()
    context_counts = {
        int(dataset.dataset.spatial_context_center_count) for dataset in datasets
    }
    if len(context_counts) != 1:
        raise ValueError(
            f"All datasets in an embedding split must use one context-center count, got {context_counts}."
        )
    context_center_count = context_counts.pop()
    has_spatial_context = context_center_count > 0
    aggregation = str(spatial_context_aggregation).strip().lower()
    if has_spatial_context and aggregation != "mean_std":
        raise ValueError(
            "Spatial context currently supports exactly aggregation='mean_std', "
            f"got {aggregation!r}."
        )
    local_embedding_dim = int(encoder.output_dim)
    embedding_dim = local_embedding_dim * (3 if has_spatial_context else 1)
    run_metadata = [
        dataset.metadata.to_dict() if dataset.metadata is not None else None
        for dataset in datasets
    ]

    expected_manifest = {
        "version": 3 if has_spatial_context else 2,
        "sample_count": int(sample_count),
        "embedding_dim": embedding_dim,
        "run_ids": run_ids,
        "has_simulation_metadata": has_simulation_metadata,
        "run_metadata": run_metadata,
        "spec": cache_spec,
    }
    if has_spatial_context:
        expected_manifest.update(
            {
                "has_spatial_context": True,
                "local_embedding_dim": local_embedding_dim,
                "spatial_context_center_count": context_center_count,
                "spatial_context_aggregation": aggregation,
                "point_cloud_batch_size": int(point_cloud_batch_size),
            }
        )
    manifest_path = target / "manifest.json"
    if target.exists() and not force_recompute:
        if not manifest_path.is_file():
            raise RuntimeError(
                f"Embedding cache directory exists without a manifest: {target}. "
                "Set cache.force_recompute=true to rebuild it."
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != expected_manifest:
            raise RuntimeError(
                f"Embedding cache specification changed for {target}. "
                "Set cache.force_recompute=true or choose a new output directory."
            )
        return EmbeddingCache.load(target)

    if target.exists():
        shutil.rmtree(target)
    building = target.with_name(f"{target.name}.building-{os.getpid()}")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True, exist_ok=False)
    arrays = _allocate_cache_arrays(
        building,
        sample_count=sample_count,
        embedding_dim=embedding_dim,
        local_embedding_dim=local_embedding_dim,
        has_simulation_metadata=has_simulation_metadata,
        has_spatial_context=has_spatial_context,
    )

    cursor = 0
    pin_memory = encoder.device.type == "cuda"
    for run_index, dataset in enumerate(datasets):
        loader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=pin_memory,
            persistent_workers=int(num_workers) > 0,
            collate_fn=identity_batch_collate,
        )
        for batch in loader:
            points0 = batch["points0"]
            points1 = batch["points1"]
            batch_rows = int(points0.shape[0])
            context0 = batch.get("context_points0")
            context1 = batch.get("context_points1")
            embeddings0, local0 = encode_spatial_context_state(
                encoder,
                points0,
                context_points=context0,
                aggregation=aggregation,
                point_cloud_batch_size=point_cloud_batch_size,
            )
            embeddings1, local1 = encode_spatial_context_state(
                encoder,
                points1,
                context_points=context1,
                aggregation=aggregation,
                point_cloud_batch_size=point_cloud_batch_size,
            )
            row_slice = slice(cursor, cursor + batch_rows)
            arrays["z0"][row_slice] = embeddings0.numpy()
            arrays["z1"][row_slice] = embeddings1.numpy()
            if has_spatial_context:
                arrays["local_z0"][row_slice] = local0.numpy()
                arrays["local_z1"][row_slice] = local1.numpy()
            arrays["atom_id"][row_slice] = batch["atom_id"].numpy()
            arrays["run_index"][row_slice] = int(run_index)
            for name in ("frame0", "frame1", "timestep0", "timestep1", "coords0", "coords1"):
                arrays[name][row_slice] = batch[name].numpy()
            if has_simulation_metadata:
                for name in _METADATA_CACHE_ARRAYS:
                    arrays[name][row_slice] = batch[name].numpy()
            cursor += batch_rows
    if cursor != sample_count:
        raise RuntimeError(
            f"Embedding extraction wrote {cursor} rows, expected {sample_count}."
        )
    for values in arrays.values():
        values.flush()
    del arrays
    with (building / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(expected_manifest, handle, indent=2, sort_keys=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(building, target)
    return EmbeddingCache.load(target)


__all__ = [
    "EmbeddingCache",
    "FrozenEncoder",
    "encode_spatial_context_state",
    "extract_embedding_cache",
    "load_frozen_encoder",
]
