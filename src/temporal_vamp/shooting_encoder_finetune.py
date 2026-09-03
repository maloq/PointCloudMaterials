"""Last-block GeoFrameV2 fine-tuning for shooting ablation 6."""

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

from src.data_utils.shooting_binary_dataset import (
    ShootingBinaryEnvironmentDataset,
    make_shooting_environment_loader,
)
from src.data_utils.shooting_dataset import (
    ShootingCampaignSnapshot,
    shooting_snapshot_sha256,
)
from src.models.encoders.geo_frame_transformer_v2 import GeoFrameTransformerV2Encoder
from src.temporal_vamp.embeddings import FrozenEncoder
from src.temporal_vamp.shooting_context import ShootingContextTokenCache
from src.temporal_vamp.shooting_distribution import DistributionalTargetData
from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache
from src.temporal_vamp.shooting_multiscale import _metrics, _metrics_by_horizon
from src.temporal_vamp.shooting_spatial import (
    FittedSpatialContextPredictor,
    SpatialContextTransformer,
    SpatialTokenData,
    fit_spatial_token_standardization,
)


@dataclass(frozen=True)
class ShootingGeoFrameActivationCache:
    path: Path
    manifest: dict[str, Any]
    tokens_before_last: np.ndarray
    shared_attention_bias: np.ndarray
    value_geometry: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "ShootingGeoFrameActivationCache":
        root = Path(path).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Shooting GeoFrame activation manifest is missing: {manifest_path}"
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        arrays = {
            name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in (
                "tokens_before_last",
                "shared_attention_bias",
                "value_geometry",
            )
        }
        expected = manifest["array_shapes"]
        for name, values in arrays.items():
            expected_shape = tuple(int(value) for value in expected[name])
            if values.shape != expected_shape or values.dtype != np.dtype("float32"):
                raise RuntimeError(
                    f"GeoFrame activation cache mismatch for {name}: "
                    f"expected_shape={expected_shape}, observed_shape={values.shape}, "
                    f"expected_dtype=float32, observed_dtype={values.dtype.name}, root={root}."
                )
        return cls(path=root, manifest=manifest, **arrays)


@dataclass(frozen=True)
class FittedLastBlockPredictor:
    spatial: FittedSpatialContextPredictor
    encoder_state: dict[str, torch.Tensor]
    trainable_encoder_parameter_names: tuple[str, ...]
    trainable_encoder_parameter_count: int
    initial_embedding_max_abs_error: float


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _geoframe_v2_encoder(frozen: FrozenEncoder) -> GeoFrameTransformerV2Encoder:
    encoder = frozen.model.encoder
    if not isinstance(encoder, GeoFrameTransformerV2Encoder):
        raise TypeError(
            "Ablation 6 requires the concrete GeoFrameTransformerV2Encoder, got "
            f"{type(encoder).__name__}."
        )
    if len(encoder.token_encoder.transformer.layers) != 6:
        raise RuntimeError(
            "The selected checkpoint no longer has the registered six-layer GeoFrameV2 "
            f"contract: observed={len(encoder.token_encoder.transformer.layers)}."
        )
    return encoder


def configure_last_geoframe_block_trainable(
    encoder: GeoFrameTransformerV2Encoder,
) -> tuple[str, ...]:
    encoder.requires_grad_(False)
    final_layer_index = len(encoder.token_encoder.transformer.layers) - 1
    encoder.token_encoder.transformer.layers[final_layer_index].requires_grad_(True)
    encoder.token_encoder.transformer.norm.requires_grad_(True)
    names = tuple(name for name, value in encoder.named_parameters() if value.requires_grad)
    prefixes = (
        f"token_encoder.transformer.layers.{final_layer_index}.",
        "token_encoder.transformer.norm.",
    )
    unexpected = [name for name in names if not name.startswith(prefixes)]
    if not names or unexpected:
        raise RuntimeError(
            f"Last-block parameter boundary is invalid: names={names}, unexpected={unexpected}."
        )
    return names


@torch.inference_mode()
def _upstream_activations(
    frozen: FrozenEncoder,
    points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoder = _geoframe_v2_encoder(frozen)
    inputs = points.to(device=frozen.device, dtype=torch.float32, non_blocking=True)
    inputs = frozen.model._prepare_model_input(inputs)
    centered = encoder._center_points(inputs)
    neighborhood, centers = encoder.token_encoder.group_points(centered)
    tokens, shared_attention_bias, value_geometry, _ = (
        encoder.token_encoder._prepare_tokens(
            neighborhood, centers, return_state=False
        )
    )
    for layer in encoder.token_encoder.transformer.layers[:-1]:
        tokens = layer(tokens, shared_attention_bias, value_geometry)
    return (
        tokens.to(torch.float32),
        shared_attention_bias.to(torch.float32),
        value_geometry.to(torch.float32),
    )


def _encode_last_block(
    frozen: FrozenEncoder,
    tokens_before_last: torch.Tensor,
    shared_attention_bias: torch.Tensor,
    value_geometry: torch.Tensor,
) -> torch.Tensor:
    encoder = _geoframe_v2_encoder(frozen)
    transformer = encoder.token_encoder.transformer
    tokens = transformer.layers[-1](
        tokens_before_last, shared_attention_bias, value_geometry
    )
    tokens = transformer.norm(tokens)
    encoder_features = encoder._pool_tokens(tokens)
    invariant = frozen.model._contrastive_invariant_latent(encoder_features, None)
    return frozen.model._output_representation(invariant).to(torch.float32)


def _write_shard(path: Path, *, spec_sha256: str, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, spec_sha256=np.asarray(spec_sha256), **arrays)
    os.replace(temporary, path)


def extract_shooting_geoframe_activation_cache(
    snapshot: ShootingCampaignSnapshot,
    base_cache: ShootingEmbeddingCache,
    context_cache: ShootingContextTokenCache,
    *,
    encoder: FrozenEncoder,
    cache_path: str | Path,
    num_points: int,
    radius: float,
    context_center_count: int,
    point_cloud_batch_size: int,
    environment_batch_size: int,
    environment_num_workers: int,
    force_recompute: bool,
) -> ShootingGeoFrameActivationCache:
    target = Path(cache_path).expanduser().resolve()
    _geoframe_v2_encoder(encoder)
    snapshot_hash = shooting_snapshot_sha256(snapshot)
    if snapshot_hash != str(base_cache.manifest["spec"]["snapshot_sha256"]):
        raise RuntimeError("Activation-cache snapshot does not match the base embedding cache.")
    if snapshot_hash != str(context_cache.manifest["spec"]["snapshot_sha256"]):
        raise RuntimeError("Activation-cache snapshot does not match the context-token cache.")
    checkpoint_stat = encoder.checkpoint_path.stat()
    spec = {
        "version": 1,
        "snapshot_sha256": snapshot_hash,
        "checkpoint": str(encoder.checkpoint_path),
        "checkpoint_size_bytes": int(checkpoint_stat.st_size),
        "checkpoint_mtime_ns": int(checkpoint_stat.st_mtime_ns),
        "representation_source": encoder.representation_source,
        "center_atom_ids": np.asarray(base_cache.atom_ids, dtype=np.int64).tolist(),
        "num_points": int(num_points),
        "radius": float(radius),
        "context_center_count": int(context_center_count),
        "fine_tune_boundary": "token_encoder.transformer.layers.5 + transformer.norm",
        "storage_dtype": "float32",
    }
    manifest_path = target / "manifest.json"
    if manifest_path.is_file() and not force_recompute:
        cached = ShootingGeoFrameActivationCache.load(target)
        if cached.manifest["spec"] != spec:
            raise RuntimeError(
                f"Shooting activation cache specification changed at {target}; "
                "choose a new path or set force_recompute=true."
            )
        return cached
    if target.exists() and not force_recompute:
        raise RuntimeError(
            f"Activation cache exists without a final manifest: {target}. "
            "Set force_recompute=true after inspecting it."
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
    pending_branches: list[dict[str, Any]] = []
    pending_parent_positions: list[int] = []
    for parent_position, parent in enumerate(snapshot.parents):
        shard_path = shard_root / f"parent_{parent_position:04d}.npz"
        if shard_path.is_file():
            with np.load(shard_path, allow_pickle=False) as payload:
                if str(payload["spec_sha256"].item()) != spec_sha:
                    raise RuntimeError(
                        f"Activation shard specification changed: {shard_path}."
                    )
            continue
        representative = sorted(
            branches_by_parent[str(parent["parent_id"])],
            key=lambda value: int(value["shot_index"]),
        )[0]
        pending_branches.append(representative)
        pending_parent_positions.append(parent_position)

    if pending_branches:
        dataset = ShootingBinaryEnvironmentDataset(
            snapshot,
            branches=pending_branches,
            timesteps=[0],
            center_atom_ids=np.asarray(base_cache.atom_ids, dtype=np.int64),
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
            central = batch["points"][:, 0]
            batch_count, center_count, point_count, _ = central.shape
            if int(context_center_count) == 0:
                context_count = 0
                token_points = central[:, :, None]
            else:
                satellite = batch["context_points"][:, 0]
                observed = satellite.shape[2]
                if int(observed) != int(context_center_count):
                    raise RuntimeError(
                        "Activation context-token count changed: "
                        f"configured={context_center_count}, observed={observed}."
                    )
                context_count = int(observed)
                token_points = torch.cat([central[:, :, None], satellite], dim=2)
            flat = token_points.reshape(-1, point_count, 3)
            token_chunks: list[torch.Tensor] = []
            bias_chunks: list[torch.Tensor] = []
            value_chunks: list[torch.Tensor] = []
            for start in range(0, int(flat.shape[0]), int(point_cloud_batch_size)):
                token_values, bias_values, value_values = _upstream_activations(
                    encoder, flat[start : start + int(point_cloud_batch_size)]
                )
                token_chunks.append(token_values.cpu())
                bias_chunks.append(bias_values.cpu())
                value_chunks.append(value_values.cpu())
            token_array = torch.cat(token_chunks).reshape(
                batch_count, center_count, context_count + 1, *token_chunks[0].shape[1:]
            ).numpy()
            bias_array = torch.cat(bias_chunks).reshape(
                batch_count, center_count, context_count + 1, *bias_chunks[0].shape[1:]
            ).numpy()
            value_array = torch.cat(value_chunks).reshape(
                batch_count, center_count, context_count + 1, *value_chunks[0].shape[1:]
            ).numpy()
            dataset_indices = np.asarray(batch["dataset_index"], dtype=np.int64)
            for batch_position, dataset_index in enumerate(dataset_indices.tolist()):
                parent_position = pending_parent_positions[dataset_index]
                _write_shard(
                    shard_root / f"parent_{parent_position:04d}.npz",
                    spec_sha256=spec_sha,
                    tokens_before_last=token_array[batch_position].astype(
                        np.float32, copy=False
                    ),
                    shared_attention_bias=bias_array[batch_position].astype(
                        np.float32, copy=False
                    ),
                    value_geometry=value_array[batch_position].astype(
                        np.float32, copy=False
                    ),
                )
                print(
                    f"[shooting-finetune] activation parent={parent_position + 1}/"
                    f"{len(snapshot.parents)}",
                    flush=True,
                )

    first_shard_path = shard_root / "parent_0000.npz"
    with np.load(first_shard_path, allow_pickle=False) as first:
        per_parent_shapes = {
            name: first[name].shape
            for name in (
                "tokens_before_last",
                "shared_attention_bias",
                "value_geometry",
            )
        }
    parent_count = len(snapshot.parents)
    shapes = {
        name: (parent_count, *shape) for name, shape in per_parent_shapes.items()
    }
    building = target.parent / f".{target.name}.building-{os.getpid()}"
    if building.exists():
        raise FileExistsError(f"Activation cache build directory exists: {building}")
    building.mkdir(parents=True)
    arrays = {
        name: open_memmap(
            building / f"{name}.npy", mode="w+", dtype=np.float32, shape=shape
        )
        for name, shape in shapes.items()
    }
    for parent_position in range(parent_count):
        shard_path = shard_root / f"parent_{parent_position:04d}.npz"
        if not shard_path.is_file():
            raise FileNotFoundError(f"Activation shard is missing: {shard_path}")
        with np.load(shard_path, allow_pickle=False) as payload:
            if str(payload["spec_sha256"].item()) != spec_sha:
                raise RuntimeError(f"Activation shard changed during consolidation: {shard_path}")
            for name in arrays:
                arrays[name][parent_position] = payload[name]
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
    return ShootingGeoFrameActivationCache.load(target)


def _spatial_model(
    tokens: SpatialTokenData,
    targets: DistributionalTargetData,
    *,
    hidden_dim: int,
    heads: int,
    blocks: int,
    rbf_dim: int,
    maximum_radius: float,
    representation_dim: int,
    dropout: float,
) -> SpatialContextTransformer:
    return SpatialContextTransformer(
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
    )


def fit_last_geoframe_block_predictor(
    activation_cache: ShootingGeoFrameActivationCache,
    frozen_encoder: FrozenEncoder,
    frozen_tokens: SpatialTokenData,
    targets: DistributionalTargetData,
    *,
    initial_backbone_states: Mapping[int, Mapping[str, torch.Tensor]],
    device: str,
    hidden_dim: int,
    heads: int,
    blocks: int,
    rbf_dim: int,
    maximum_radius: float,
    representation_dim: int,
    dropout: float,
    context_learning_rate: float,
    encoder_learning_rate: float,
    weight_decay: float,
    batch_size: int,
    maximum_epochs: int,
    patience: int,
    gradient_clip_norm: float,
    mixed_precision: bool,
    seeds: Sequence[int],
) -> FittedLastBlockPredictor:
    torch_device = torch.device(device)
    if frozen_encoder.device != torch_device:
        raise RuntimeError(
            f"Frozen encoder device={frozen_encoder.device} does not match training device={torch_device}."
        )
    encoder = _geoframe_v2_encoder(frozen_encoder)
    base_encoder_state = copy.deepcopy(encoder.state_dict())
    (
        embedding_mean,
        embedding_scale,
        descriptor_mean,
        descriptor_scale,
    ) = fit_spatial_token_standardization(
        frozen_tokens, targets.split_rows["optimization"]
    )
    descriptors = torch.from_numpy(
        ((frozen_tokens.descriptors - descriptor_mean) / descriptor_scale).astype(
            np.float32
        )
    ).to(torch_device)
    offsets = torch.from_numpy(np.asarray(frozen_tokens.offsets, dtype=np.float32)).to(
        torch_device
    )
    target_tensor = torch.from_numpy(targets.target_modes).to(torch_device)
    optimization_rows = np.asarray(targets.split_rows["optimization"], dtype=np.int64)
    selection_rows = np.asarray(targets.split_rows["selection"], dtype=np.int64)
    row_count = int(frozen_tokens.embeddings.shape[0])
    parent_count, center_count, context_count = activation_cache.tokens_before_last.shape[:3]
    token_count = int(context_count)
    if row_count != int(parent_count * center_count):
        raise RuntimeError(
            f"Activation/token row mismatch: activations={parent_count}*{center_count}, "
            f"tokens={row_count}."
        )

    def activation_batch(rows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        parent_rows = rows // int(center_count)
        center_rows = rows % int(center_count)
        token_values = np.asarray(
            activation_cache.tokens_before_last[parent_rows, center_rows], dtype=np.float32
        )
        bias_values = np.asarray(
            activation_cache.shared_attention_bias[parent_rows, center_rows], dtype=np.float32
        )
        geometry_values = np.asarray(
            activation_cache.value_geometry[parent_rows, center_rows], dtype=np.float32
        )
        return (
            torch.from_numpy(token_values).to(torch_device),
            torch.from_numpy(bias_values).to(torch_device),
            torch.from_numpy(geometry_values).to(torch_device),
        )

    def embed_rows(rows: np.ndarray) -> torch.Tensor:
        token_values, bias_values, geometry_values = activation_batch(rows)
        flat_shape = (int(rows.size) * token_count,)
        embeddings = _encode_last_block(
            frozen_encoder,
            token_values.reshape(*flat_shape, *token_values.shape[-2:]),
            bias_values.reshape(*flat_shape, *bias_values.shape[-3:]),
            geometry_values.reshape(*flat_shape, *geometry_values.shape[-2:]),
        ).reshape(int(rows.size), token_count, -1)
        mean = torch.as_tensor(embedding_mean, device=torch_device, dtype=embeddings.dtype)
        scale = torch.as_tensor(embedding_scale, device=torch_device, dtype=embeddings.dtype)
        return (embeddings - mean) / scale

    encoder.load_state_dict(base_encoder_state, strict=True)
    configure_last_geoframe_block_trainable(encoder)
    encoder.eval()
    with torch.no_grad():
        maximum_error = 0.0
        for start in range(0, row_count, int(batch_size)):
            rows = np.arange(start, min(row_count, start + int(batch_size)), dtype=np.int64)
            reconstructed = embed_rows(rows) * torch.as_tensor(
                embedding_scale, device=torch_device
            ) + torch.as_tensor(embedding_mean, device=torch_device)
            reference = torch.from_numpy(
                np.asarray(frozen_tokens.embeddings[rows], dtype=np.float32)
            ).to(torch_device)
            maximum_error = max(
                maximum_error, float(torch.max(torch.abs(reconstructed - reference)))
            )
    if maximum_error > 5.0e-4:
        raise RuntimeError(
            "Cached layer-4 activations do not reproduce the frozen shooting embeddings: "
            f"max_abs_error={maximum_error:.8g}."
        )
    print(
        f"[shooting-finetune] initial embedding max_abs_error={maximum_error:.8g}",
        flush=True,
    )

    histories: dict[int, dict[str, list[float]]] = {}
    seed_metrics: dict[int, dict[str, Any]] = {}
    predictions_by_seed: dict[int, np.ndarray] = {}
    representations_by_seed: dict[int, np.ndarray] = {}
    spatial_states: dict[int, dict[str, torch.Tensor]] = {}
    encoder_states: dict[int, dict[str, torch.Tensor]] = {}
    trainable_names: tuple[str, ...] | None = None
    use_autocast = torch_device.type == "cuda" and bool(mixed_precision)
    for raw_seed in seeds:
        seed = int(raw_seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        encoder.load_state_dict(base_encoder_state, strict=True)
        trainable_names = configure_last_geoframe_block_trainable(encoder)
        encoder.eval()
        spatial = _spatial_model(
            frozen_tokens,
            targets,
            hidden_dim=hidden_dim,
            heads=heads,
            blocks=blocks,
            rbf_dim=rbf_dim,
            maximum_radius=maximum_radius,
            representation_dim=representation_dim,
            dropout=dropout,
        ).to(torch_device)
        incompatible = spatial.load_state_dict(
            dict(initial_backbone_states[seed]), strict=False
        )
        if set(incompatible.missing_keys) != {
            "prediction_head.weight",
            "prediction_head.bias",
        } or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Ordinary-pretrained spatial state is incompatible for seed={seed}: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}."
            )
        encoder_parameters = [
            parameter for parameter in encoder.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": spatial.parameters(),
                    "lr": float(context_learning_rate),
                },
                {
                    "params": encoder_parameters,
                    "lr": float(encoder_learning_rate),
                },
            ],
            weight_decay=float(weight_decay),
        )
        generator = np.random.default_rng(seed)
        best_selection = float("inf")
        best_epoch = -1
        best_spatial_state: dict[str, torch.Tensor] | None = None
        best_encoder_state: dict[str, torch.Tensor] | None = None
        history = {"optimization": [], "selection": []}
        for epoch in range(int(maximum_epochs)):
            spatial.train()
            permutation = generator.permutation(optimization_rows)
            accumulated = 0.0
            for start in range(0, permutation.size, int(batch_size)):
                rows = permutation[start : start + int(batch_size)]
                with torch.autocast(
                    device_type=torch_device.type,
                    dtype=torch.bfloat16 if use_autocast else torch.float32,
                    enabled=use_autocast,
                ):
                    embeddings = embed_rows(rows)
                    _, prediction = spatial(
                        embeddings, descriptors[rows], offsets[rows]
                    )
                    loss = torch.mean((prediction - target_tensor[rows]) ** 2)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    encoder_parameters, max_norm=float(gradient_clip_norm)
                )
                optimizer.step()
                accumulated += float(loss.detach()) * int(rows.size)
            optimization_loss = accumulated / float(permutation.size)
            spatial.eval()
            selection_sum = 0.0
            with torch.no_grad():
                for start in range(0, selection_rows.size, int(batch_size)):
                    rows = selection_rows[start : start + int(batch_size)]
                    with torch.autocast(
                        device_type=torch_device.type,
                        dtype=torch.bfloat16 if use_autocast else torch.float32,
                        enabled=use_autocast,
                    ):
                        embeddings = embed_rows(rows)
                        _, prediction = spatial(
                            embeddings, descriptors[rows], offsets[rows]
                        )
                        selection_sum += float(
                            torch.sum((prediction - target_tensor[rows]) ** 2)
                        )
            selection_loss = selection_sum / float(
                selection_rows.size * targets.target_modes.shape[1]
            )
            history["optimization"].append(optimization_loss)
            history["selection"].append(selection_loss)
            if selection_loss < best_selection:
                best_selection = selection_loss
                best_epoch = epoch
                best_spatial_state = copy.deepcopy(spatial.state_dict())
                best_encoder_state = {
                    name: value.detach().cpu().clone()
                    for name, value in encoder.state_dict().items()
                    if name in trainable_names
                }
            if epoch - best_epoch >= int(patience):
                break
        if best_spatial_state is None or best_encoder_state is None:
            raise RuntimeError(f"Last-block fine-tuning seed={seed} made no checkpoint.")
        spatial.load_state_dict(best_spatial_state)
        current_encoder_state = encoder.state_dict()
        current_encoder_state.update(best_encoder_state)
        encoder.load_state_dict(current_encoder_state, strict=True)
        spatial.eval()
        predictions: list[np.ndarray] = []
        representations: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, row_count, int(batch_size)):
                rows = np.arange(start, min(row_count, start + int(batch_size)), dtype=np.int64)
                with torch.autocast(
                    device_type=torch_device.type,
                    dtype=torch.bfloat16 if use_autocast else torch.float32,
                    enabled=use_autocast,
                ):
                    embedding_values = embed_rows(rows)
                    representation, prediction = spatial(
                        embedding_values, descriptors[rows], offsets[rows]
                    )
                representations.append(representation.float().cpu().numpy())
                predictions.append(prediction.float().cpu().numpy())
        prediction_array = np.concatenate(predictions)
        representation_array = np.concatenate(representations)
        histories[seed] = history
        seed_metrics[seed] = {
            "best_epoch": best_epoch,
            "epochs_run": len(history["selection"]),
            "selection": _metrics(
                prediction_array, targets.target_modes, targets.split_rows["selection"]
            ),
            "validation": _metrics(
                prediction_array, targets.target_modes, targets.split_rows["validation"]
            ),
            "validation_by_horizon": _metrics_by_horizon(
                prediction_array,
                targets.target_modes,
                targets.split_rows["validation"],
                targets.selected_horizons_ps,
            ),
        }
        predictions_by_seed[seed] = prediction_array
        representations_by_seed[seed] = representation_array
        spatial_states[seed] = best_spatial_state
        encoder_states[seed] = best_encoder_state
        print(
            f"[shooting-finetune] seed={seed} best_epoch={best_epoch} "
            f"selection_mse={best_selection:.6f} "
            f"validation_r2={seed_metrics[seed]['validation']['r2']:.6f}",
            flush=True,
        )
    selected_seed = min(
        seed_metrics,
        key=lambda value: (float(seed_metrics[value]["selection"]["mse"]), int(value)),
    )
    selected_spatial = _spatial_model(
        frozen_tokens,
        targets,
        hidden_dim=hidden_dim,
        heads=heads,
        blocks=blocks,
        rbf_dim=rbf_dim,
        maximum_radius=maximum_radius,
        representation_dim=representation_dim,
        dropout=dropout,
    )
    selected_spatial.load_state_dict(spatial_states[selected_seed])
    selected_spatial.eval()
    assert trainable_names is not None
    trainable_count = sum(
        int(dict(encoder.named_parameters())[name].numel()) for name in trainable_names
    )
    fitted_spatial = FittedSpatialContextPredictor(
        model=selected_spatial.cpu(),
        embedding_mean=embedding_mean,
        embedding_scale=embedding_scale,
        descriptor_mean=descriptor_mean,
        descriptor_scale=descriptor_scale,
        seed=selected_seed,
        histories=histories,
        seed_metrics=seed_metrics,
        predictions_by_seed=predictions_by_seed,
        representations_by_seed=representations_by_seed,
    )
    return FittedLastBlockPredictor(
        spatial=fitted_spatial,
        encoder_state=encoder_states[selected_seed],
        trainable_encoder_parameter_names=trainable_names,
        trainable_encoder_parameter_count=trainable_count,
        initial_embedding_max_abs_error=maximum_error,
    )


__all__ = [
    "FittedLastBlockPredictor",
    "ShootingGeoFrameActivationCache",
    "configure_last_geoframe_block_trainable",
    "extract_shooting_geoframe_activation_cache",
    "fit_last_geoframe_block_predictor",
]
