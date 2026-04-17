from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch
import torch.nn as nn


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    *,
    num_layers: int = 2,
    dropout: float = 0.0,
    activation: str = "gelu",
    output_bias: bool = True,
) -> nn.Sequential:
    if input_dim <= 0:
        raise ValueError(f"build_mlp requires input_dim > 0, got {input_dim}.")
    if hidden_dim <= 0:
        raise ValueError(f"build_mlp requires hidden_dim > 0, got {hidden_dim}.")
    if output_dim <= 0:
        raise ValueError(f"build_mlp requires output_dim > 0, got {output_dim}.")
    if num_layers <= 0:
        raise ValueError(f"build_mlp requires num_layers > 0, got {num_layers}.")

    activation_name = str(activation).strip().lower()
    if activation_name == "relu":
        activation_layer = nn.ReLU
    elif activation_name == "gelu":
        activation_layer = nn.GELU
    elif activation_name == "silu":
        activation_layer = nn.SiLU
    else:
        raise ValueError(
            "activation must be one of {'relu', 'gelu', 'silu'}, "
            f"got {activation!r}."
        )

    dims = [int(input_dim)]
    if num_layers == 1:
        dims.append(int(output_dim))
    else:
        dims.extend([int(hidden_dim)] * (int(num_layers) - 1))
        dims.append(int(output_dim))

    layers: list[nn.Module] = []
    for idx in range(len(dims) - 1):
        in_dim = int(dims[idx])
        out_dim = int(dims[idx + 1])
        is_last = idx == len(dims) - 2
        layers.append(nn.Linear(in_dim, out_dim, bias=(output_bias or not is_last)))
        if not is_last:
            layers.append(activation_layer())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
    return nn.Sequential(*layers)


def lag_to_key(lag: int) -> str:
    return f"lag_{int(lag)}"


def resolve_lag_weights(lags: Sequence[int], lag_weights: Sequence[float] | None) -> dict[int, float]:
    resolved_lags = [int(lag) for lag in lags]
    if lag_weights is None:
        return {lag: 1.0 for lag in resolved_lags}

    weights = [float(weight) for weight in lag_weights]
    if len(weights) != len(resolved_lags):
        raise ValueError(
            "tmf.temporal.lag_weights must have the same length as tmf.lags. "
            f"Got lags={resolved_lags}, lag_weights={weights}."
        )
    return {lag: weight for lag, weight in zip(resolved_lags, weights)}


def as_python_list(value: Any, *, expected_length: int | None = None) -> list[Any]:
    if value is None:
        if expected_length is None:
            return []
        return [None] * int(expected_length)

    if torch.is_tensor(value):
        if value.ndim == 0:
            result = [value.item()]
        else:
            result = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, str):
        result = [value]
    elif isinstance(value, Iterable):
        result = list(value)
    else:
        result = [value]

    if expected_length is None:
        return result

    if len(result) == int(expected_length):
        return result
    if len(result) == 1 and int(expected_length) > 1:
        return result * int(expected_length)
    raise ValueError(
        "Unable to broadcast batch metadata to the requested batch size. "
        f"expected_length={expected_length}, actual_length={len(result)}, value_type={type(value)}."
    )


def build_sample_keys_from_batch(batch: dict[str, Any], *, batch_size: int) -> list[tuple[str, Any, Any]]:
    source_paths = as_python_list(batch.get("source_path"), expected_length=batch_size)
    center_atom_ids = as_python_list(batch.get("center_atom_id"), expected_length=batch_size)
    anchor_frame_indices = as_python_list(batch.get("anchor_frame_index"), expected_length=batch_size)
    instance_ids = as_python_list(batch.get("instance_id"), expected_length=batch_size)

    keys: list[tuple[str, Any, Any]] = []
    for idx in range(int(batch_size)):
        source_path = source_paths[idx]
        center_atom_id = center_atom_ids[idx]
        anchor_frame_index = anchor_frame_indices[idx]
        instance_id = instance_ids[idx]

        if source_path is not None and center_atom_id is not None and anchor_frame_index is not None:
            keys.append(
                (
                    f"src::{source_path}",
                    int(center_atom_id),
                    int(anchor_frame_index),
                )
            )
            continue

        if instance_id is not None:
            keys.append(("instance", int(instance_id), 0))
            continue

        keys.append(("batch_index", int(idx), 0))
    return keys


def mean_by_offsets(values: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(values):
        raise TypeError(f"values must be a torch.Tensor, got {type(values)}.")
    if not torch.is_tensor(offsets):
        raise TypeError(f"offsets must be a torch.Tensor, got {type(offsets)}.")
    if offsets.ndim != 1:
        raise ValueError(f"offsets must have shape (B + 1,), got {tuple(offsets.shape)}.")
    if int(offsets.shape[0]) < 2:
        raise ValueError(f"offsets must contain at least two entries, got {tuple(offsets.shape)}.")
    if int(offsets[0].item()) != 0:
        raise ValueError(f"offsets must start at 0, got first value {int(offsets[0].item())}.")
    if int(offsets[-1].item()) != int(values.shape[0]):
        raise ValueError(
            "offsets[-1] must match the flattened ragged value count. "
            f"offsets[-1]={int(offsets[-1].item())}, values.shape[0]={int(values.shape[0])}."
        )

    if int(values.shape[0]) == 0:
        feature_shape = tuple(values.shape[1:])
        batch_size = int(offsets.shape[0]) - 1
        return values.new_zeros((batch_size, *feature_shape))

    pooled: list[torch.Tensor] = []
    for start, stop in zip(offsets[:-1].tolist(), offsets[1:].tolist()):
        if int(stop) <= int(start):
            pooled.append(torch.zeros_like(values[0]))
            continue
        pooled.append(values[int(start) : int(stop)].mean(dim=0))
    return torch.stack(pooled, dim=0)


def detach_to_float(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return value.detach().to(dtype=torch.float32)
