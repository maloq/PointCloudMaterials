from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch


def _broadcast_metadata(value: Any, *, batch_size: int) -> list[Any]:
    if value is None:
        return [None] * int(batch_size)
    if torch.is_tensor(value):
        if value.ndim == 0:
            values = [value.item()]
        else:
            values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, str):
        values = [value]
    elif isinstance(value, Iterable):
        values = list(value)
    else:
        values = [value]
    if len(values) == int(batch_size):
        return values
    if len(values) == 1 and int(batch_size) > 1:
        return values * int(batch_size)
    raise ValueError(
        "Could not broadcast batch metadata to batch size. "
        f"batch_size={batch_size}, actual_length={len(values)}, value_type={type(value)!r}."
    )


def _to_numpy_1d(values: list[Any], *, dtype: Any) -> np.ndarray:
    if dtype is str:
        return np.asarray([("" if value is None else str(value)) for value in values], dtype=str)
    converted: list[Any] = []
    for value in values:
        if value is None:
            converted.append(-1)
        else:
            converted.append(dtype(value))
    return np.asarray(converted, dtype=dtype)


def _slice_batch_tensor(
    value: Any,
    *,
    batch_size: int,
    anchor_index: int,
) -> torch.Tensor | None:
    if value is None:
        return None
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if value.ndim == 0:
        value = value.reshape(1)
    if value.shape[0] != int(batch_size):
        raise ValueError(
            "Batch metadata tensor must have the batch dimension first. "
            f"Expected batch_size={batch_size}, got shape={tuple(value.shape)}."
        )
    if value.ndim == 2:
        return value[:, anchor_index]
    if value.ndim == 1:
        return value
    raise ValueError(
        "Expected metadata tensor with shape (B,) or (B, T), "
        f"got shape={tuple(value.shape)}."
    )


def _slice_anchor_coords(
    batch: dict[str, Any],
    *,
    batch_size: int,
    anchor_index: int,
) -> torch.Tensor:
    center_positions = batch.get("center_positions")
    if center_positions is not None:
        if not torch.is_tensor(center_positions):
            center_positions = torch.as_tensor(center_positions)
        if center_positions.ndim != 3 or center_positions.shape[0] != int(batch_size) or center_positions.shape[2] != 3:
            raise ValueError(
                "Expected batch['center_positions'] with shape (B, T, 3), "
                f"got {tuple(center_positions.shape)}."
            )
        return center_positions[:, anchor_index, :].detach().cpu().to(dtype=torch.float32)

    coords = batch.get("coords")
    if coords is None:
        raise KeyError(
            "Temporal motif cache collection requires either batch['center_positions'] "
            "or batch['coords'] to be present."
        )
    if not torch.is_tensor(coords):
        coords = torch.as_tensor(coords)
    if coords.ndim == 1:
        coords = coords.unsqueeze(0)
    if coords.ndim != 2 or coords.shape[0] != int(batch_size) or coords.shape[1] != 3:
        raise ValueError(
            "Expected batch['coords'] with shape (B, 3), "
            f"got {tuple(coords.shape)}."
        )
    return coords.detach().cpu().to(dtype=torch.float32)


def _seeded_forward_sequence(model: Any, batch: dict[str, Any], seed: int) -> dict[str, Any]:
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        return model.forward_sequence(batch)
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _append_numpy(target: dict[str, list[np.ndarray]], key: str, value: np.ndarray) -> None:
    target.setdefault(key, []).append(np.asarray(value))


def _concat_cache_chunks(chunks: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    for key, values in chunks.items():
        if not values:
            continue
        first = np.asarray(values[0])
        if first.ndim == 0:
            raise ValueError(
                "Dynamic motif cache only supports per-sample arrays. "
                f"Key {key!r} produced a scalar array."
            )
        cache[key] = np.concatenate(values, axis=0)
    return cache


def cache_has_dynamic_motif_outputs(cache: dict[str, np.ndarray]) -> bool:
    return any(
        key in cache
        for key in (
            "stable_ids",
            "stable_probs",
            "bridge_ids",
            "bridge_probs",
            "hazard_probs_lag_1",
        )
    )


def collect_tmf_inference_cache(
    model: Any,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    *,
    max_batches: int | None,
    max_samples_total: int | None,
    seed_base: int | None,
    progress_every_batches: int,
    verbose: bool,
) -> dict[str, np.ndarray]:
    if not hasattr(model, "forward_sequence"):
        raise TypeError(
            "collect_tmf_inference_cache requires a model exposing forward_sequence(batch). "
            f"Got model type {type(model)!r}."
        )
    anchor_index = getattr(model, "anchor_index", None)
    if anchor_index is None:
        raise AttributeError(
            "collect_tmf_inference_cache requires model.anchor_index so anchor metadata can be aligned."
        )
    anchor_index = int(anchor_index)
    max_samples = None if max_samples_total is None else max(1, int(max_samples_total))
    every = max(1, int(progress_every_batches))
    chunks: dict[str, list[np.ndarray]] = {}
    collected = 0

    model = model.to(device)
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.inference_mode():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= int(max_batches):
                    break
                if not isinstance(batch, dict):
                    raise TypeError(
                        "TMF inference cache collection expects temporal dict batches. "
                        f"Got batch type {type(batch)!r} at batch_idx={batch_idx}. "
                        "If you set inputs.data_config to a static dataset override, remove it "
                        "or point it to a temporal_lammps config for TMF evaluation."
                    )
                points = batch.get("points")
                if points is None:
                    raise KeyError(
                        f"Temporal batch at batch_idx={batch_idx} is missing required key 'points'."
                    )
                if not torch.is_tensor(points):
                    points = torch.as_tensor(points)
                if points.ndim != 4 or points.shape[-1] != 3:
                    raise ValueError(
                        "TMF inference cache collection expects batch['points'] with shape (B, T, N, 3), "
                        f"got {tuple(points.shape)} at batch_idx={batch_idx}."
                    )
                batch_size = int(points.shape[0])

                outputs = (
                    model.forward_sequence(batch)
                    if seed_base is None
                    else _seeded_forward_sequence(model, batch, int(seed_base) + batch_idx)
                )
                teacher_targets = outputs.get("teacher_targets")
                if teacher_targets is None:
                    raise KeyError(
                        "TMF forward_sequence(batch) did not return teacher_targets; "
                        "dynamic analysis requires them for future/hazard evaluation."
                    )

                take = batch_size
                if max_samples is not None:
                    remaining = int(max_samples) - int(collected)
                    if remaining <= 0:
                        break
                    take = min(take, remaining)
                if take <= 0:
                    break

                z_anchor = outputs["z_anchor"].detach().cpu().to(dtype=torch.float32)[:take]
                _append_numpy(chunks, "inv_latents", z_anchor.numpy())
                _append_numpy(chunks, "eq_latents", np.empty((0,), dtype=np.float32))
                _append_numpy(chunks, "phases", np.empty((0,), dtype=np.int64))

                coords = _slice_anchor_coords(batch, batch_size=batch_size, anchor_index=anchor_index)[:take]
                _append_numpy(chunks, "coords", coords.numpy())

                instance_id = _slice_batch_tensor(
                    batch.get("instance_id", batch.get("center_atom_id")),
                    batch_size=batch_size,
                    anchor_index=anchor_index,
                )
                if instance_id is None:
                    instance_np = np.arange(collected, collected + take, dtype=np.int64)
                else:
                    instance_np = instance_id.detach().cpu().numpy().astype(np.int64, copy=False)[:take]
                _append_numpy(chunks, "instance_ids", instance_np)

                stable_probs = outputs["stable_probs_anchor"].detach().cpu().to(dtype=torch.float32)[:take]
                stable_ids = stable_probs.argmax(dim=-1).numpy().astype(np.int64, copy=False)
                stable_conf = stable_probs.max(dim=-1).values.numpy().astype(np.float32, copy=False)
                _append_numpy(chunks, "stable_probs", stable_probs.numpy())
                _append_numpy(chunks, "stable_ids", stable_ids)
                _append_numpy(chunks, "stable_confidence", stable_conf)

                residual_anchor = outputs["residual_anchor"].detach().cpu().to(dtype=torch.float32)[:take]
                residual_norm = torch.linalg.norm(residual_anchor, dim=-1).numpy().astype(np.float32, copy=False)
                _append_numpy(chunks, "residual_norm", residual_norm)

                bridge_output = outputs.get("bridge_output")
                if bridge_output is not None:
                    bridge_probs = bridge_output.probs.detach().cpu().to(dtype=torch.float32)[:take]
                    bridge_ids = bridge_probs.argmax(dim=-1).numpy().astype(np.int64, copy=False)
                    bridge_conf = bridge_probs.max(dim=-1).values.numpy().astype(np.float32, copy=False)
                    _append_numpy(chunks, "bridge_probs", bridge_probs.numpy())
                    _append_numpy(chunks, "bridge_ids", bridge_ids)
                    _append_numpy(chunks, "bridge_confidence", bridge_conf)
                bridge_gate = outputs.get("bridge_gate")
                if bridge_gate is not None:
                    gate_np = (
                        bridge_gate.detach().cpu().reshape(-1).to(dtype=torch.float32).numpy()[:take]
                    )
                    _append_numpy(chunks, "bridge_gate", gate_np)

                frame_indices = _slice_batch_tensor(
                    batch.get("frame_indices"),
                    batch_size=batch_size,
                    anchor_index=anchor_index,
                )
                if frame_indices is not None:
                    _append_numpy(
                        chunks,
                        "frame_index",
                        frame_indices.detach().cpu().numpy().astype(np.int64, copy=False)[:take],
                    )

                timesteps = _slice_batch_tensor(
                    batch.get("timesteps"),
                    batch_size=batch_size,
                    anchor_index=anchor_index,
                )
                if timesteps is not None:
                    _append_numpy(
                        chunks,
                        "timestep",
                        timesteps.detach().cpu().numpy().astype(np.int64, copy=False)[:take],
                    )

                center_atom_ids = _broadcast_metadata(batch.get("center_atom_id"), batch_size=batch_size)
                _append_numpy(
                    chunks,
                    "center_atom_id",
                    _to_numpy_1d(center_atom_ids[:take], dtype=np.int64),
                )
                source_paths = _broadcast_metadata(batch.get("source_path"), batch_size=batch_size)
                _append_numpy(
                    chunks,
                    "source_path",
                    _to_numpy_1d(source_paths[:take], dtype=str),
                )

                future_logits = outputs.get("future_stable_logits", {})
                hazard_logits = outputs.get("hazard_logits", {})
                target_probs_by_lag = teacher_targets.get("target_stable_probs", {})
                change_targets_by_lag = teacher_targets.get("change_targets", {})
                for lag, pred_logits in future_logits.items():
                    if lag not in target_probs_by_lag:
                        raise KeyError(
                            f"TMF teacher_targets is missing target_stable_probs[{lag}] required for analysis."
                        )
                    pred_logits_cpu = pred_logits.detach().cpu().to(dtype=torch.float32)[:take]
                    pred_probs_cpu = torch.softmax(pred_logits_cpu, dim=-1)
                    target_probs_cpu = (
                        target_probs_by_lag[lag].detach().cpu().to(dtype=torch.float32)[:take]
                    )
                    future_nll = (
                        -(target_probs_cpu * torch.log(pred_probs_cpu.clamp_min(1e-8))).sum(dim=-1)
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    _append_numpy(
                        chunks,
                        f"future_pred_ids_lag_{int(lag)}",
                        pred_probs_cpu.argmax(dim=-1).numpy().astype(np.int64, copy=False),
                    )
                    _append_numpy(
                        chunks,
                        f"future_target_ids_lag_{int(lag)}",
                        target_probs_cpu.argmax(dim=-1).numpy().astype(np.int64, copy=False),
                    )
                    _append_numpy(chunks, f"future_nll_lag_{int(lag)}", future_nll)

                    if lag in hazard_logits:
                        hazard_prob = (
                            torch.sigmoid(hazard_logits[lag].detach().cpu().reshape(-1).to(dtype=torch.float32))[:take]
                            .numpy()
                            .astype(np.float32, copy=False)
                        )
                        _append_numpy(chunks, f"hazard_probs_lag_{int(lag)}", hazard_prob)
                    if lag in change_targets_by_lag:
                        change_target = (
                            change_targets_by_lag[lag].detach().cpu().reshape(-1).to(dtype=torch.float32)[:take]
                            .numpy()
                            .astype(np.float32, copy=False)
                        )
                        _append_numpy(chunks, f"change_target_lag_{int(lag)}", change_target)

                field_logits = outputs.get("field_logits")
                field_lag = getattr(model, "field_lag", None)
                if field_logits is not None and field_lag is not None:
                    field_probs = torch.softmax(
                        field_logits.detach().cpu().to(dtype=torch.float32)[:take],
                        dim=-1,
                    )
                    if int(field_lag) not in target_probs_by_lag:
                        raise KeyError(
                            "TMF field head returned logits, but teacher_targets did not expose "
                            f"target_stable_probs for field lag {field_lag}."
                        )
                    field_target = target_probs_by_lag[int(field_lag)].detach().cpu().to(dtype=torch.float32)[:take]
                    field_nll = (
                        -(field_target * torch.log(field_probs.clamp_min(1e-8))).sum(dim=-1)
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    _append_numpy(
                        chunks,
                        f"field_pred_ids_lag_{int(field_lag)}",
                        field_probs.argmax(dim=-1).numpy().astype(np.int64, copy=False),
                    )
                    _append_numpy(chunks, f"field_nll_lag_{int(field_lag)}", field_nll)

                collected += int(take)
                if verbose and ((batch_idx + 1) % every == 0 or take != batch_size):
                    print(
                        f"[analysis][tmf collect] batch={batch_idx + 1} "
                        f"samples={collected}"
                        + (f"/{max_samples}" if max_samples is not None else "")
                    )
                if max_samples is not None and collected >= int(max_samples):
                    if verbose:
                        print(f"[analysis][tmf collect] reached sample cap: {collected}")
                    break
    finally:
        if was_training:
            model.train()

    cache = _concat_cache_chunks(chunks)
    num_samples = int(cache["inv_latents"].shape[0]) if "inv_latents" in cache else 0
    cache["sample_index"] = np.arange(num_samples, dtype=np.int64)
    if "center_atom_id" not in cache and "instance_ids" in cache:
        cache["center_atom_id"] = np.asarray(cache["instance_ids"], dtype=np.int64)
    return cache


__all__ = [
    "cache_has_dynamic_motif_outputs",
    "collect_tmf_inference_cache",
]
