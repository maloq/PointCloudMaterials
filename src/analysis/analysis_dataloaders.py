from __future__ import annotations

import bisect
from typing import Any

import torch


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


class _BatchedConcatDataset(torch.utils.data.ConcatDataset):
    """Batch the temporal dataset (or its Subset) across source boundaries."""

    def __getitems__(self, indices):
        runs = []
        for index in indices:
            dataset_index = bisect.bisect_right(self.cumulative_sizes, index)
            offset = 0 if dataset_index == 0 else self.cumulative_sizes[dataset_index - 1]
            if not runs or runs[-1][0] != dataset_index:
                runs.append((dataset_index, []))
            runs[-1][1].append(index - offset)
        batches = [self.datasets[i].__getitems__(rows) for i, rows in runs]
        return {
            key: sum((b[key] for b in batches), [])
            if key == "source_path"
            else torch.cat([b[key] for b in batches])
            for key in batches[0]
        }
