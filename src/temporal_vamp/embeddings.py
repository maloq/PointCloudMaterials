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


_CACHE_ARRAYS = {
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
        for name in _CACHE_ARRAYS:
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

    @property
    def output_dim(self) -> int:
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
                encoded = self.model.encoder_io.encode(inputs).invariant.to(torch.float32)
                accumulated = encoded if accumulated is None else accumulated + encoded
        assert accumulated is not None
        return accumulated / float(self.repeats)


def load_frozen_encoder(
    checkpoint_path: str | Path,
    *,
    device: str,
    repeats: int = 1,
    seed: int = 0,
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
    )


def _allocate_cache_arrays(
    directory: Path,
    *,
    sample_count: int,
    embedding_dim: int,
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
    return {
        name: open_memmap(
            directory / f"{name}.npy",
            mode="w+",
            dtype=_CACHE_ARRAYS[name],
            shape=shape,
        )
        for name, shape in shapes.items()
    }


def extract_embedding_cache(
    datasets: Sequence[TemporalPairDataset],
    *,
    encoder: FrozenEncoder,
    cache_path: str | Path,
    cache_spec: dict[str, Any],
    batch_size: int,
    num_workers: int,
    force_recompute: bool,
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

    expected_manifest = {
        "version": 1,
        "sample_count": int(sample_count),
        "embedding_dim": int(encoder.output_dim),
        "run_ids": run_ids,
        "spec": cache_spec,
    }
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
        embedding_dim=encoder.output_dim,
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
            joined = torch.cat([points0, points1], dim=0)
            embeddings = encoder.encode(joined).cpu().numpy()
            row_slice = slice(cursor, cursor + batch_rows)
            arrays["z0"][row_slice] = embeddings[:batch_rows]
            arrays["z1"][row_slice] = embeddings[batch_rows:]
            arrays["atom_id"][row_slice] = batch["atom_id"].numpy()
            arrays["run_index"][row_slice] = int(run_index)
            for name in ("frame0", "frame1", "timestep0", "timestep1", "coords0", "coords1"):
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
    "extract_embedding_cache",
    "load_frozen_encoder",
]
