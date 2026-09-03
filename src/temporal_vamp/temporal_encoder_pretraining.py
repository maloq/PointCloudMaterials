"""Temporal fine-tuning of GeoFrameV2 on ordinary MD trajectories.

The future target is produced by an immutable copy of the static checkpoint.
Only a configured tail of the present encoder and a disposable prediction head
are optimized.  This keeps the encoder usable by every existing extraction path
while teaching its invariant embedding to retain multi-horizon dynamical signal.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch import nn

from src.data_utils.temporal_binary_context_dataset import (
    TemporalBinaryContextDataset,
    make_temporal_binary_context_loader,
)
from src.models.encoders.geo_frame_transformer_v2 import GeoFrameTransformerV2Encoder
from src.temporal_vamp.embeddings import FrozenEncoder
from src.temporal_vamp.ordinary_pretraining import (
    OrdinaryContextEmbeddingCache,
    OrdinaryPretrainingTargets,
)
from src.temporal_vamp.shooting_encoder_finetune import _geoframe_v2_encoder


@dataclass(frozen=True)
class OrdinaryGeoFrameActivationCache:
    path: Path
    manifest: dict[str, Any]
    tokens_before_tail: np.ndarray
    shared_attention_bias: np.ndarray
    value_geometry: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "OrdinaryGeoFrameActivationCache":
        root = Path(path).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Ordinary GeoFrame activation manifest is missing: {manifest_path}"
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("state") != "complete":
            raise RuntimeError(f"Ordinary activation cache is incomplete: {manifest_path}")
        arrays = {
            name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in (
                "tokens_before_tail",
                "shared_attention_bias",
                "value_geometry",
            )
        }
        for name, values in arrays.items():
            expected = tuple(int(value) for value in manifest["array_shapes"][name])
            if values.shape != expected or values.dtype != np.dtype("float32"):
                raise RuntimeError(
                    f"Ordinary activation array changed for {name}: expected={expected} "
                    f"float32, observed={values.shape} {values.dtype}, root={root}."
                )
        return cls(path=root, manifest=manifest, **arrays)


@dataclass(frozen=True)
class FittedTemporalGeoFrameEncoder:
    encoder_state: dict[str, torch.Tensor]
    trainable_parameter_names: tuple[str, ...]
    trainable_parameter_count: int
    selected_seed: int
    histories: dict[int, dict[str, list[float]]]
    metrics: dict[int, dict[str, Any]]
    initial_embedding_max_abs_error: float


class TemporalChangeHead(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, target_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(embedding_dim) + 1, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(target_dim)),
        )

    def forward(self, embedding: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((embedding, temperature[:, None]), dim=1))


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def configure_geoframe_tail_trainable(
    encoder: GeoFrameTransformerV2Encoder,
    *,
    trainable_tail_blocks: int,
) -> tuple[str, ...]:
    encoder.requires_grad_(False)
    layers = encoder.token_encoder.transformer.layers
    count = int(trainable_tail_blocks)
    if count <= 0 or count > len(layers):
        raise ValueError(
            f"trainable_tail_blocks must be in [1, {len(layers)}], got {count}."
        )
    first = len(layers) - count
    for layer in layers[first:]:
        layer.requires_grad_(True)
    encoder.token_encoder.transformer.norm.requires_grad_(True)
    names = tuple(name for name, value in encoder.named_parameters() if value.requires_grad)
    prefixes = tuple(
        f"token_encoder.transformer.layers.{index}."
        for index in range(first, len(layers))
    ) + ("token_encoder.transformer.norm.",)
    unexpected = [name for name in names if not name.startswith(prefixes)]
    if not names or unexpected:
        raise RuntimeError(
            f"Temporal GeoFrame tail boundary is invalid: names={names}, "
            f"unexpected={unexpected}."
        )
    return names


@torch.inference_mode()
def _upstream_activations(
    frozen: FrozenEncoder,
    points: torch.Tensor,
    *,
    trainable_tail_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoder = _geoframe_v2_encoder(frozen)
    inputs = points.to(device=frozen.device, dtype=torch.float32, non_blocking=True)
    inputs = frozen.model._prepare_model_input(inputs)
    centered = encoder._center_points(inputs)
    neighborhood, centers = encoder.token_encoder.group_points(centered)
    tokens, attention_bias, value_geometry, _ = encoder.token_encoder._prepare_tokens(
        neighborhood, centers, return_state=False
    )
    boundary = len(encoder.token_encoder.transformer.layers) - int(
        trainable_tail_blocks
    )
    for layer in encoder.token_encoder.transformer.layers[:boundary]:
        tokens = layer(tokens, attention_bias, value_geometry)
    return (
        tokens.to(torch.float32),
        attention_bias.to(torch.float32),
        value_geometry.to(torch.float32),
    )


def _encode_tail(
    frozen: FrozenEncoder,
    tokens: torch.Tensor,
    attention_bias: torch.Tensor,
    value_geometry: torch.Tensor,
    *,
    trainable_tail_blocks: int,
) -> torch.Tensor:
    encoder = _geoframe_v2_encoder(frozen)
    transformer = encoder.token_encoder.transformer
    first = len(transformer.layers) - int(trainable_tail_blocks)
    for layer in transformer.layers[first:]:
        tokens = layer(tokens, attention_bias, value_geometry)
    tokens = transformer.norm(tokens)
    features = encoder._pool_tokens(tokens)
    invariant = frozen.model._contrastive_invariant_latent(features, None)
    return frozen.model._output_representation(invariant).to(torch.float32)


def _write_shard(path: Path, *, spec_sha256: str, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, spec_sha256=np.asarray(spec_sha256), **arrays)
    os.replace(temporary, path)


def extract_ordinary_geoframe_activation_cache(
    dataset: TemporalBinaryContextDataset,
    ordinary_cache: OrdinaryContextEmbeddingCache,
    *,
    encoder: FrozenEncoder,
    cache_path: str | Path,
    trainable_tail_blocks: int,
    point_cloud_batch_size: int,
    environment_batch_size: int,
    environment_num_workers: int,
    force_recompute: bool,
) -> OrdinaryGeoFrameActivationCache:
    target = Path(cache_path).expanduser().resolve()
    checkpoint_stat = encoder.checkpoint_path.stat()
    ordinary_spec_hash = _sha256_json(ordinary_cache.manifest["spec"])
    spec = {
        "version": 1,
        "ordinary_cache": str(ordinary_cache.path),
        "ordinary_spec_sha256": ordinary_spec_hash,
        "checkpoint": str(encoder.checkpoint_path),
        "checkpoint_size_bytes": int(checkpoint_stat.st_size),
        "checkpoint_mtime_ns": int(checkpoint_stat.st_mtime_ns),
        "representation_source": encoder.representation_source,
        "trainable_tail_blocks": int(trainable_tail_blocks),
        "row_order": "dataset anchor major, configured center atom minor",
    }
    manifest_path = target / "manifest.json"
    if manifest_path.is_file() and not force_recompute:
        cached = OrdinaryGeoFrameActivationCache.load(target)
        if cached.manifest["spec"] != spec:
            raise RuntimeError(
                f"Ordinary activation specification changed at {target}; use a new "
                "path or set force_recompute=true."
            )
        return cached
    if target.exists() and not force_recompute:
        raise RuntimeError(f"Incomplete ordinary activation cache exists: {target}")
    if force_recompute and target.exists():
        shutil.rmtree(target)
    shard_root = target.parent / f"{target.name}_shards"
    if force_recompute and shard_root.exists():
        shutil.rmtree(shard_root)
    shard_root.mkdir(parents=True, exist_ok=True)
    spec_sha = _sha256_json(spec)

    pending = []
    for index in range(len(dataset)):
        shard = shard_root / f"{index:06d}.npz"
        if not shard.is_file():
            pending.append(index)
            continue
        with np.load(shard, allow_pickle=False) as payload:
            if str(payload["spec_sha256"].item()) != spec_sha:
                raise RuntimeError(f"Ordinary activation shard has another spec: {shard}")
    if pending:
        subset = torch.utils.data.Subset(dataset, pending)
        loader = make_temporal_binary_context_loader(
            subset,  # type: ignore[arg-type]
            batch_size=int(environment_batch_size),
            num_workers=int(environment_num_workers),
            pin_memory=encoder.device.type == "cuda",
        )
        completed = len(dataset) - len(pending)
        for batch in loader:
            central = batch["token_points"][:, :, 0]
            batch_count, center_count, point_count, _ = central.shape
            flat = central.reshape(-1, point_count, 3)
            token_chunks: list[torch.Tensor] = []
            bias_chunks: list[torch.Tensor] = []
            geometry_chunks: list[torch.Tensor] = []
            for start in range(0, int(flat.shape[0]), int(point_cloud_batch_size)):
                token, bias, geometry = _upstream_activations(
                    encoder,
                    flat[start : start + int(point_cloud_batch_size)],
                    trainable_tail_blocks=int(trainable_tail_blocks),
                )
                token_chunks.append(token.cpu())
                bias_chunks.append(bias.cpu())
                geometry_chunks.append(geometry.cpu())
            token_array = torch.cat(token_chunks).reshape(
                batch_count, center_count, *token_chunks[0].shape[1:]
            ).numpy()
            bias_array = torch.cat(bias_chunks).reshape(
                batch_count, center_count, *bias_chunks[0].shape[1:]
            ).numpy()
            geometry_array = torch.cat(geometry_chunks).reshape(
                batch_count, center_count, *geometry_chunks[0].shape[1:]
            ).numpy()
            dataset_indices = np.asarray(batch["dataset_index"], dtype=np.int64)
            for position, dataset_index in enumerate(dataset_indices.tolist()):
                _write_shard(
                    shard_root / f"{dataset_index:06d}.npz",
                    spec_sha256=spec_sha,
                    tokens_before_tail=token_array[position].astype(np.float32, copy=False),
                    shared_attention_bias=bias_array[position].astype(
                        np.float32, copy=False
                    ),
                    value_geometry=geometry_array[position].astype(np.float32, copy=False),
                )
            completed += int(batch_count)
            print(
                f"[temporal-encoder] extracted anchors={completed}/{len(dataset)}",
                flush=True,
            )

    first_path = shard_root / "000000.npz"
    with np.load(first_path, allow_pickle=False) as first:
        per_anchor_shapes = {
            name: first[name].shape
            for name in (
                "tokens_before_tail",
                "shared_attention_bias",
                "value_geometry",
            )
        }
    shapes = {
        name: (len(dataset), *shape) for name, shape in per_anchor_shapes.items()
    }
    building = target.parent / f".{target.name}.building-{os.getpid()}"
    if building.exists():
        raise FileExistsError(f"Ordinary activation build directory exists: {building}")
    building.mkdir(parents=True)
    arrays = {
        name: open_memmap(
            building / f"{name}.npy", mode="w+", dtype=np.float32, shape=shape
        )
        for name, shape in shapes.items()
    }
    for anchor_index in range(len(dataset)):
        shard_path = shard_root / f"{anchor_index:06d}.npz"
        if not shard_path.is_file():
            raise FileNotFoundError(f"Ordinary activation shard is missing: {shard_path}")
        with np.load(shard_path, allow_pickle=False) as payload:
            if str(payload["spec_sha256"].item()) != spec_sha:
                raise RuntimeError(f"Ordinary activation shard changed: {shard_path}")
            for name in arrays:
                arrays[name][anchor_index] = payload[name]
    for values in arrays.values():
        values.flush()
    manifest = {
        "state": "complete",
        "spec": spec,
        "array_shapes": {name: list(shape) for name, shape in shapes.items()},
    }
    with (building / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(building, target)
    return OrdinaryGeoFrameActivationCache.load(target)


def _metric(prediction: np.ndarray, target: np.ndarray, rows: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction[rows], dtype=np.float64) - np.asarray(
        target[rows], dtype=np.float64
    )
    mse = float(np.mean(error**2))
    target_values = np.asarray(target[rows], dtype=np.float64)
    denominator = float(np.sum((target_values - target_values.mean(axis=0)) ** 2))
    r2 = 1.0 - float(np.sum(error**2)) / denominator
    return {"mse": mse, "r2": r2}


def fit_temporal_geoframe_encoder(
    activation_cache: OrdinaryGeoFrameActivationCache,
    ordinary_cache: OrdinaryContextEmbeddingCache,
    targets: OrdinaryPretrainingTargets,
    frozen_encoder: FrozenEncoder,
    *,
    device: str,
    trainable_tail_blocks: int,
    hidden_dim: int,
    batch_size: int,
    head_learning_rate: float,
    head_maximum_epochs: int,
    head_patience: int,
    encoder_learning_rate: float,
    joint_head_learning_rate: float,
    joint_maximum_epochs: int,
    joint_patience: int,
    weight_decay: float,
    distillation_weight: float,
    gradient_clip_norm: float,
    seeds: Sequence[int],
) -> FittedTemporalGeoFrameEncoder:
    torch_device = torch.device(device)
    if frozen_encoder.device != torch_device:
        raise RuntimeError(
            f"Encoder device={frozen_encoder.device} differs from training device={device}."
        )
    encoder = _geoframe_v2_encoder(frozen_encoder)
    base_state = copy.deepcopy(encoder.state_dict())
    center_count = int(ordinary_cache.manifest["spec"]["center_atom_ids"].__len__())
    anchor_count = int(activation_cache.tokens_before_tail.shape[0])
    row_count = int(ordinary_cache.token_z.shape[0])
    if anchor_count * center_count != row_count:
        raise RuntimeError(
            f"Ordinary activation row layout changed: anchors={anchor_count}, "
            f"centers={center_count}, rows={row_count}."
        )
    reference = np.asarray(ordinary_cache.token_z[:, 0], dtype=np.float32)
    optimization_rows = np.asarray(targets.split_rows["optimization"], dtype=np.int64)
    selection_rows = np.asarray(targets.split_rows["selection"], dtype=np.int64)
    embedding_mean = reference[optimization_rows].mean(axis=0)
    embedding_scale = reference[optimization_rows].std(axis=0)
    embedding_scale = np.where(embedding_scale <= 1.0e-8, 1.0, embedding_scale)
    temperature = np.asarray(ordinary_cache.temperature_K, dtype=np.float32)
    temperature_mean = float(temperature[optimization_rows].mean())
    temperature_scale = float(temperature[optimization_rows].std())
    if temperature_scale <= 1.0e-8:
        raise RuntimeError("Ordinary optimization rows contain no temperature variation.")
    standardized_temperature = (temperature - temperature_mean) / temperature_scale
    target_tensor = torch.from_numpy(np.asarray(targets.target_modes, dtype=np.float32)).to(
        torch_device
    )
    mean_tensor = torch.from_numpy(embedding_mean.astype(np.float32)).to(torch_device)
    scale_tensor = torch.from_numpy(embedding_scale.astype(np.float32)).to(torch_device)
    reference_tensor = torch.from_numpy(
        ((reference - embedding_mean) / embedding_scale).astype(np.float32)
    ).to(torch_device)
    activation_tensors = tuple(
        torch.from_numpy(np.array(values, dtype=np.float32, copy=True))
        .reshape(row_count, *values.shape[2:])
        .to(torch_device)
        for values in (
            activation_cache.tokens_before_tail,
            activation_cache.shared_attention_bias,
            activation_cache.value_geometry,
        )
    )

    def activation_batch(rows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.from_numpy(rows).to(torch_device)
        return tuple(values[indices] for values in activation_tensors)  # type: ignore[return-value]

    def embed(rows: np.ndarray) -> torch.Tensor:
        token, bias, geometry = activation_batch(rows)
        values = _encode_tail(
            frozen_encoder,
            token,
            bias,
            geometry,
            trainable_tail_blocks=int(trainable_tail_blocks),
        )
        return (values - mean_tensor) / scale_tensor

    encoder.load_state_dict(base_state, strict=True)
    configure_geoframe_tail_trainable(
        encoder, trainable_tail_blocks=int(trainable_tail_blocks)
    )
    encoder.eval()
    maximum_error = 0.0
    with torch.no_grad():
        for start in range(0, row_count, int(batch_size)):
            rows = np.arange(start, min(row_count, start + int(batch_size)), dtype=np.int64)
            reconstructed = embed(rows) * scale_tensor + mean_tensor
            expected = torch.from_numpy(reference[rows]).to(torch_device)
            maximum_error = max(
                maximum_error, float(torch.max(torch.abs(reconstructed - expected)))
            )
    if maximum_error > 5.0e-4:
        raise RuntimeError(
            "Ordinary upstream activations do not reconstruct the teacher embedding: "
            f"max_abs_error={maximum_error:.8g}."
        )
    print(
        f"[temporal-encoder] initial embedding max_abs_error={maximum_error:.8g}",
        flush=True,
    )

    histories: dict[int, dict[str, list[float]]] = {}
    metrics: dict[int, dict[str, Any]] = {}
    encoder_states: dict[int, dict[str, torch.Tensor]] = {}
    names: tuple[str, ...] | None = None
    for raw_seed in seeds:
        seed = int(raw_seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        encoder.load_state_dict(base_state, strict=True)
        names = configure_geoframe_tail_trainable(
            encoder, trainable_tail_blocks=int(trainable_tail_blocks)
        )
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        head = TemporalChangeHead(
            embedding_dim=int(reference.shape[1]),
            hidden_dim=int(hidden_dim),
            target_dim=int(targets.target_modes.shape[1]),
        ).to(torch_device)
        generator = np.random.default_rng(seed)
        head_optimizer = torch.optim.AdamW(
            head.parameters(), lr=float(head_learning_rate), weight_decay=float(weight_decay)
        )
        best_head_state: dict[str, torch.Tensor] | None = None
        best_head_selection = float("inf")
        best_head_epoch = -1
        head_history: list[float] = []
        for epoch in range(int(head_maximum_epochs)):
            head.train()
            permutation = generator.permutation(optimization_rows)
            for start in range(0, permutation.size, int(batch_size)):
                rows = permutation[start : start + int(batch_size)]
                embedding = reference_tensor[rows]
                temp = torch.from_numpy(standardized_temperature[rows]).to(torch_device)
                prediction = head(embedding, temp)
                loss = torch.mean((prediction - target_tensor[rows]) ** 2)
                head_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                head_optimizer.step()
            head.eval()
            selection_sum = 0.0
            with torch.no_grad():
                for start in range(0, selection_rows.size, int(batch_size)):
                    rows = selection_rows[start : start + int(batch_size)]
                    prediction = head(
                        embed(rows),
                        torch.from_numpy(standardized_temperature[rows]).to(torch_device),
                    )
                    selection_sum += float(
                        torch.sum((prediction - target_tensor[rows]) ** 2)
                    )
            selection_loss = selection_sum / float(
                selection_rows.size * targets.target_modes.shape[1]
            )
            head_history.append(selection_loss)
            if selection_loss < best_head_selection:
                best_head_selection = selection_loss
                best_head_epoch = epoch
                best_head_state = copy.deepcopy(head.state_dict())
            if epoch - best_head_epoch >= int(head_patience):
                break
        if best_head_state is None:
            raise RuntimeError(f"Temporal baseline head seed={seed} made no checkpoint.")
        head.load_state_dict(best_head_state)

        encoder.load_state_dict(base_state, strict=True)
        names = configure_geoframe_tail_trainable(
            encoder, trainable_tail_blocks=int(trainable_tail_blocks)
        )
        encoder_parameters = [
            parameter for parameter in encoder.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": head.parameters(), "lr": float(joint_head_learning_rate)},
                {"params": encoder_parameters, "lr": float(encoder_learning_rate)},
            ],
            weight_decay=float(weight_decay),
        )
        best_selection = best_head_selection
        best_epoch = -1
        best_encoder_state = {
            name: base_state[name].detach().cpu().clone() for name in names
        }
        best_joint_head_state = copy.deepcopy(best_head_state)
        joint_optimization: list[float] = []
        joint_selection: list[float] = []
        for epoch in range(int(joint_maximum_epochs)):
            permutation = generator.permutation(optimization_rows)
            head.train()
            total = 0.0
            for start in range(0, permutation.size, int(batch_size)):
                rows = permutation[start : start + int(batch_size)]
                embedding = embed(rows)
                temp = torch.from_numpy(standardized_temperature[rows]).to(torch_device)
                prediction = head(embedding, temp)
                reference_batch = reference_tensor[rows]
                prediction_loss = torch.mean((prediction - target_tensor[rows]) ** 2)
                distillation_loss = torch.mean((embedding - reference_batch) ** 2)
                loss = prediction_loss + float(distillation_weight) * distillation_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    encoder_parameters, max_norm=float(gradient_clip_norm)
                )
                optimizer.step()
                total += float(prediction_loss.detach()) * int(rows.size)
            joint_optimization.append(total / float(permutation.size))
            head.eval()
            selection_sum = 0.0
            with torch.no_grad():
                for start in range(0, selection_rows.size, int(batch_size)):
                    rows = selection_rows[start : start + int(batch_size)]
                    prediction = head(
                        embed(rows),
                        torch.from_numpy(standardized_temperature[rows]).to(torch_device),
                    )
                    selection_sum += float(
                        torch.sum((prediction - target_tensor[rows]) ** 2)
                    )
            selection_loss = selection_sum / float(
                selection_rows.size * targets.target_modes.shape[1]
            )
            joint_selection.append(selection_loss)
            if selection_loss < best_selection:
                best_selection = selection_loss
                best_epoch = epoch
                best_encoder_state = {
                    name: value.detach().cpu().clone()
                    for name, value in encoder.state_dict().items()
                    if name in names
                }
                best_joint_head_state = copy.deepcopy(head.state_dict())
            if epoch - best_epoch >= int(joint_patience):
                break
        current_state = encoder.state_dict()
        current_state.update(best_encoder_state)
        encoder.load_state_dict(current_state, strict=True)
        head.load_state_dict(best_joint_head_state)
        prediction_chunks: list[np.ndarray] = []
        head.eval()
        with torch.no_grad():
            for start in range(0, row_count, int(batch_size)):
                rows = np.arange(start, min(row_count, start + int(batch_size)), dtype=np.int64)
                prediction = head(
                    embed(rows),
                    torch.from_numpy(standardized_temperature[rows]).to(torch_device),
                )
                prediction_chunks.append(prediction.cpu().numpy())
        predictions = np.concatenate(prediction_chunks)
        histories[seed] = {
            "baseline_selection": head_history,
            "joint_optimization": joint_optimization,
            "joint_selection": joint_selection,
        }
        metrics[seed] = {
            "baseline_head_best_epoch": best_head_epoch,
            "baseline_selection_mse": best_head_selection,
            "joint_best_epoch": best_epoch,
            "joint_epochs_run": len(joint_selection),
            "selection": _metric(predictions, targets.target_modes, selection_rows),
            "optimization": _metric(predictions, targets.target_modes, optimization_rows),
        }
        encoder_states[seed] = best_encoder_state
        print(
            f"[temporal-encoder] seed={seed} baseline={best_head_selection:.6f} "
            f"best_epoch={best_epoch} selection={best_selection:.6f}",
            flush=True,
        )
    selected_seed = min(
        metrics,
        key=lambda seed: (float(metrics[seed]["selection"]["mse"]), int(seed)),
    )
    assert names is not None
    trainable_count = sum(int(dict(encoder.named_parameters())[name].numel()) for name in names)
    return FittedTemporalGeoFrameEncoder(
        encoder_state=encoder_states[selected_seed],
        trainable_parameter_names=names,
        trainable_parameter_count=trainable_count,
        selected_seed=int(selected_seed),
        histories=histories,
        metrics=metrics,
        initial_embedding_max_abs_error=maximum_error,
    )


def write_temporal_encoder_checkpoint(
    source_checkpoint: str | Path,
    destination: str | Path,
    fitted: FittedTemporalGeoFrameEncoder,
    *,
    metadata: dict[str, Any],
) -> None:
    source = Path(source_checkpoint).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    replaced: list[str] = []
    for name, value in fitted.encoder_state.items():
        candidates = (f"encoder._orig_mod.{name}", f"encoder.{name}")
        matches = [key for key in candidates if key in state]
        if len(matches) != 1:
            raise RuntimeError(
                f"Cannot map temporal encoder parameter {name!r} into {source}: "
                f"matches={matches}."
            )
        state[matches[0]] = value.detach().cpu().clone()
        replaced.append(matches[0])
    payload["temporal_encoder_pretraining"] = {
        **metadata,
        "selected_seed": fitted.selected_seed,
        "trainable_parameter_names": list(fitted.trainable_parameter_names),
        "replaced_checkpoint_keys": replaced,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    source_config = source.parent / ".hydra" / "config.yaml"
    if not source_config.is_file():
        raise FileNotFoundError(
            f"Source checkpoint training configuration is missing: {source_config}"
        )
    target_config = target.parent / ".hydra" / "config.yaml"
    target_config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_config, target_config)
    temporary = target.with_suffix(".tmp.ckpt")
    torch.save(payload, temporary)
    os.replace(temporary, target)


__all__ = [
    "FittedTemporalGeoFrameEncoder",
    "OrdinaryGeoFrameActivationCache",
    "TemporalChangeHead",
    "configure_geoframe_tail_trainable",
    "extract_ordinary_geoframe_activation_cache",
    "fit_temporal_geoframe_encoder",
    "write_temporal_encoder_checkpoint",
]
