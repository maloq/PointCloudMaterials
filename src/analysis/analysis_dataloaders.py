from __future__ import annotations

import bisect
from typing import Any

import numpy as np
import torch
from torch.utils.data import default_collate


def _analysis_prefetch_factor(num_workers: int) -> int | None:
    if int(num_workers) <= 0:
        return None
    return 4


def _analysis_dataloader_kwargs(
    *,
    batch_size: int,
    dataloader_num_workers: int,
    collate_fn: Any | None = None,
) -> dict[str, Any]:
    num_workers = int(dataloader_num_workers)
    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "num_workers": num_workers,
        "shuffle": False,
        "drop_last": False,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": bool(num_workers > 0),
    }
    prefetch_factor = _analysis_prefetch_factor(num_workers)
    if prefetch_factor is not None:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    return kwargs


def _fetch_batched_dataset_items(dataset: Any, indices: list[int]) -> Any:
    if not indices:
        raise ValueError("Cannot fetch an empty batched dataset slice.")
    if hasattr(dataset, "__getitems__"):
        batch = dataset.__getitems__([int(v) for v in indices])
        if not isinstance(batch, list):
            return batch
        return default_collate(batch)
    return default_collate([dataset[int(index)] for index in indices])


def _concat_batched_values(values: list[Any], *, key_path: str) -> Any:
    if not values:
        raise ValueError(f"Cannot concatenate zero batched values at {key_path}.")
    first = values[0]
    if torch.is_tensor(first):
        return torch.cat(values, dim=0)
    if isinstance(first, np.ndarray):
        return np.concatenate(values, axis=0)
    if isinstance(first, list):
        merged: list[Any] = []
        for value in values:
            if not isinstance(value, list):
                raise TypeError(
                    f"Inconsistent batched value types at {key_path}: "
                    f"expected list, got {type(value)!r}."
                )
            merged.extend(value)
        return merged
    if isinstance(first, tuple):
        return tuple(
            _concat_batched_values(
                [value[item_idx] for value in values],
                key_path=f"{key_path}[{item_idx}]",
            )
            for item_idx in range(len(first))
        )
    if isinstance(first, dict):
        keys = list(first.keys())
        for value in values[1:]:
            if not isinstance(value, dict) or list(value.keys()) != keys:
                raise TypeError(
                    f"Inconsistent batched dict structure at {key_path}: "
                    f"first_keys={keys}, value_type={type(value)!r}, "
                    f"value_keys={list(value.keys()) if isinstance(value, dict) else None}."
                )
        return {
            key: _concat_batched_values(
                [value[key] for value in values],
                key_path=f"{key_path}.{key}",
            )
            for key in keys
        }
    raise TypeError(
        f"Unsupported batched value type at {key_path}: {type(first)!r}."
    )


class _BatchedConcatDataset(torch.utils.data.ConcatDataset):
    """ConcatDataset variant that preserves child datasets' batched __getitems__."""

    def __getitems__(self, indices: list[int]) -> Any:
        if not indices:
            raise ValueError("Cannot fetch an empty batch from _BatchedConcatDataset.")
        batches: list[Any] = []
        current_dataset_idx: int | None = None
        current_local_indices: list[int] = []

        def _flush_current_run() -> None:
            if current_dataset_idx is None:
                return
            batches.append(
                _fetch_batched_dataset_items(
                    self.datasets[int(current_dataset_idx)],
                    current_local_indices,
                )
            )

        for raw_idx in indices:
            idx = int(raw_idx)
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError(
                    f"_BatchedConcatDataset index {raw_idx} is out of range for len={len(self)}."
                )
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
            sample_idx = (
                idx
                if dataset_idx == 0
                else idx - int(self.cumulative_sizes[dataset_idx - 1])
            )
            if current_dataset_idx is not None and dataset_idx != current_dataset_idx:
                _flush_current_run()
                current_local_indices = []
            current_dataset_idx = int(dataset_idx)
            current_local_indices.append(int(sample_idx))

        _flush_current_run()
        return _concat_batched_values(batches, key_path="batch")
