import math
from collections import deque

import torch
from torch.utils.data import BatchSampler, Sampler
from torch.utils.data._utils.collate import default_collate


def _temporal_identity_or_default_collate(batch):
    if isinstance(batch, dict):
        return batch
    return default_collate(batch)


class _EpochTrackingSampler(Sampler[int]):
    """Sampler wrapper that exposes epoch state even for non-distributed samplers."""

    def __init__(self, sampler):
        self._sampler = sampler
        self.epoch = int(getattr(sampler, "epoch", 0) or 0)
        self.seed = int(getattr(sampler, "seed", 0) or 0)
        for attr in ("num_replicas", "rank", "drop_last", "shuffle"):
            if hasattr(sampler, attr):
                setattr(self, attr, getattr(sampler, attr))

    def __iter__(self):
        return iter(self._sampler)

    def __len__(self) -> int:
        return len(self._sampler)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if hasattr(self._sampler, "set_epoch"):
            self._sampler.set_epoch(epoch)
        if hasattr(self._sampler, "epoch"):
            self._sampler.epoch = self.epoch


class TemporalWindowBatchSampler(BatchSampler):
    def __init__(
        self,
        sampler,
        batch_size: int,
        drop_last: bool,
        *,
        dataset=None,
        shuffle_windows: bool = False,
        shuffle_centers: bool = False,
        mixed_windows_per_batch: int | None = None,
        shuffle_window_block_size: int | None = None,
        shared_center_order: bool = False,
    ) -> None:
        super().__init__(sampler=sampler, batch_size=int(batch_size), drop_last=bool(drop_last))
        self.sampler = _EpochTrackingSampler(self.sampler)
        self.dataset = dataset if dataset is not None else self._resolve_dataset_from_sampler(sampler)
        self.shuffle_windows = bool(shuffle_windows)
        self.shuffle_centers = bool(shuffle_centers)
        self.shared_center_order = bool(shared_center_order)
        self.mixed_windows_per_batch = (
            None if mixed_windows_per_batch is None else int(mixed_windows_per_batch)
        )
        self.shuffle_window_block_size = (
            None if shuffle_window_block_size is None else int(shuffle_window_block_size)
        )

        if self.mixed_windows_per_batch is not None and self.mixed_windows_per_batch <= 0:
            raise ValueError(
                "mixed_windows_per_batch must be > 0 when provided, "
                f"got {self.mixed_windows_per_batch}."
            )
        if self.shuffle_window_block_size is not None and self.shuffle_window_block_size <= 0:
            raise ValueError(
                "shuffle_window_block_size must be > 0 when provided, "
                f"got {self.shuffle_window_block_size}."
            )
        if self.dataset is None:
            raise TypeError(
                "TemporalWindowBatchSampler requires either dataset=... or a sampler exposing "
                "a dataset via .dataset or .data_source."
            )
        if not hasattr(self.dataset, "window_count") or not hasattr(self.dataset, "center_count"):
            raise TypeError(
                "TemporalWindowBatchSampler requires a dataset exposing window_count and center_count. "
                f"Got dataset={type(self.dataset)}."
            )
        self.window_count = int(self.dataset.window_count)
        self.center_count = int(self.dataset.center_count)
        self.total_samples = int(len(self.dataset))
        if self.window_count <= 0 or self.center_count <= 0:
            raise ValueError(
                "TemporalWindowBatchSampler requires a non-empty temporal dataset. "
                f"window_count={self.window_count}, center_count={self.center_count}."
            )
        expected_total_samples = self.window_count * self.center_count
        if self.total_samples != expected_total_samples:
            raise ValueError(
                "TemporalWindowBatchSampler requires a dense window-major temporal dataset layout. "
                f"Expected len(dataset) == window_count * center_count, got len(dataset)={self.total_samples}, "
                f"window_count={self.window_count}, center_count={self.center_count}."
            )

    @staticmethod
    def _resolve_dataset_from_sampler(sampler):
        if hasattr(sampler, "dataset"):
            return sampler.dataset
        if hasattr(sampler, "data_source"):
            return sampler.data_source
        return None

    def _rank_and_world_size(self) -> tuple[int, int]:
        world_size = int(getattr(self.sampler, "num_replicas", 1) or 1)
        rank = int(getattr(self.sampler, "rank", 0) or 0)
        if world_size <= 1:
            return 0, 1
        if rank < 0 or rank >= world_size:
            raise ValueError(
                f"TemporalWindowBatchSampler received invalid distributed rank/world_size: "
                f"rank={rank}, world_size={world_size}."
            )
        return rank, world_size

    def _epoch_seed(self, *, salt: int = 0) -> int:
        base_seed = int(getattr(self.sampler, "seed", 0) or 0)
        epoch = int(getattr(self.sampler, "epoch", 0) or 0)
        return base_seed + epoch + int(salt)

    def _generator(self, *, salt: int = 0) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(self._epoch_seed(salt=salt))
        return generator

    def _local_window_count(self) -> int:
        rank, world_size = self._rank_and_world_size()
        if world_size == 1:
            return self.window_count
        if self.drop_last:
            usable = (self.window_count // world_size) * world_size
            if usable <= 0:
                return 0
            return usable // world_size
        return math.ceil(self.window_count / world_size)

    def _local_sample_count(self) -> int:
        return self._local_window_count() * self.center_count

    def __len__(self) -> int:
        local_samples = self._local_sample_count()
        if self.drop_last:
            return local_samples // self.batch_size
        return math.ceil(local_samples / self.batch_size)

    def __iter__(self):
        mix_windows = self._resolve_mixed_windows_per_batch()
        if mix_windows is not None:
            yield from self._iter_mixed_window_batches(mix_windows)
            return

        yield from self._iter_window_grouped_batches()

    def _iter_window_grouped_batches(self):
        window_order = self._ordered_window_slots()
        center_order_cache: dict[int, torch.Tensor] = {}

        batch: list[int] = []
        for window_slot in window_order:
            window_offset = 0
            while window_offset < self.center_count:
                take = min(self.batch_size - len(batch), self.center_count - window_offset)
                batch.extend(
                    self._window_sample_indices(
                        window_slot,
                        offset=window_offset,
                        take=take,
                        center_order_cache=center_order_cache,
                    )
                )
                window_offset += take
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

        if batch and not self.drop_last:
            yield batch

    def _resolve_mixed_windows_per_batch(self) -> int | None:
        if not (self.shuffle_windows and self.shuffle_centers):
            return None
        if self.mixed_windows_per_batch is None:
            return None
        resolved = min(self.window_count, self.batch_size, int(self.mixed_windows_per_batch))
        if resolved <= 1:
            return None
        return resolved

    def _ordered_window_slots(self) -> list[int]:
        window_slots = list(range(self.window_count))
        if self.shuffle_windows:
            window_slots = self._blockwise_shuffled_window_slots(
                window_slots,
                generator=self._generator(),
            )
        rank, world_size = self._rank_and_world_size()
        if world_size == 1:
            return window_slots

        if self.drop_last:
            usable = (len(window_slots) // world_size) * world_size
            if usable <= 0:
                return []
            sharded_slots = window_slots[:usable]
        else:
            total = math.ceil(len(window_slots) / world_size) * world_size
            sharded_slots = list(window_slots)
            if len(sharded_slots) < total:
                sharded_slots.extend(sharded_slots[: total - len(sharded_slots)])
        return [int(window_slot) for window_slot in sharded_slots[rank::world_size]]

    def _blockwise_shuffled_window_slots(
        self,
        window_slots: list[int],
        *,
        generator: torch.Generator,
    ) -> list[int]:
        ordered_slots = sorted(int(window_slot) for window_slot in window_slots)
        block_size = self.shuffle_window_block_size
        if block_size is None or block_size <= 1:
            permutation = torch.randperm(len(ordered_slots), generator=generator).tolist()
            return [int(ordered_slots[idx]) for idx in permutation]

        blocks = [
            ordered_slots[pos : pos + block_size]
            for pos in range(0, len(ordered_slots), block_size)
        ]
        block_permutation = torch.randperm(len(blocks), generator=generator).tolist()
        shuffled: list[int] = []
        for block_idx in block_permutation:
            shuffled.extend(int(window_slot) for window_slot in blocks[block_idx])
        return shuffled

    def _window_sample_indices(
        self,
        window_slot: int,
        *,
        offset: int,
        take: int,
        center_order_cache: dict[int, torch.Tensor],
    ) -> list[int]:
        if take <= 0:
            return []
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}.")
        if offset >= self.center_count:
            return []

        end = min(self.center_count, offset + take)
        base_index = int(window_slot) * self.center_count
        if not self.shuffle_centers:
            return list(range(base_index + offset, base_index + end))

        cache_key = "__shared__" if self.shared_center_order else int(window_slot)
        center_order = center_order_cache.get(cache_key)
        if center_order is None:
            salt = 1_000_003 if self.shared_center_order else 1_000_003 + int(window_slot)
            center_order = torch.randperm(
                self.center_count,
                generator=self._generator(salt=salt),
            )
            center_order_cache[cache_key] = center_order
        return (base_index + center_order[offset:end]).tolist()

    def _iter_mixed_window_batches(
        self,
        mix_windows: int,
    ):
        window_order = self._ordered_window_slots()
        resolved_mix_windows = min(len(window_order), int(mix_windows))
        if resolved_mix_windows <= 1:
            yield from self._iter_window_grouped_batches()
            return

        window_queue: deque[int] = deque(int(window_slot) for window_slot in window_order)
        window_offsets = {int(window_slot): 0 for window_slot in window_order}
        center_order_cache: dict[int, torch.Tensor] = {}
        produced_samples = 0
        total_local_samples = len(window_order) * self.center_count

        while produced_samples < total_local_samples:
            remaining_local = total_local_samples - produced_samples
            target_batch_size = min(self.batch_size, remaining_local)
            if target_batch_size < self.batch_size and self.drop_last:
                break

            batch: list[int] = []
            safety_rounds = 0
            while len(batch) < target_batch_size:
                active_windows: list[int] = []
                while window_queue and len(active_windows) < resolved_mix_windows:
                    active_windows.append(int(window_queue.popleft()))
                if not active_windows:
                    raise RuntimeError(
                        "TemporalWindowBatchSampler could not assemble a mixed-window batch "
                        f"with target_batch_size={target_batch_size}, produced={produced_samples}, "
                        f"window_count={self.window_count}, center_count={self.center_count}."
                    )

                need = target_batch_size - len(batch)
                share, remainder = divmod(need, len(active_windows))
                for active_pos, window_slot in enumerate(active_windows):
                    take_target = share + (1 if active_pos < remainder else 0)
                    if take_target <= 0:
                        if window_offsets[window_slot] < self.center_count:
                            window_queue.append(window_slot)
                        continue

                    offset = int(window_offsets[window_slot])
                    remaining_centers = self.center_count - offset
                    take = min(take_target, remaining_centers)
                    if take > 0:
                        batch.extend(
                            self._window_sample_indices(
                                window_slot,
                                offset=offset,
                                take=take,
                                center_order_cache=center_order_cache,
                            )
                        )
                        window_offsets[window_slot] = offset + take

                    if window_offsets[window_slot] < self.center_count:
                        window_queue.append(window_slot)

                safety_rounds += 1
                if safety_rounds > len(window_order) + 1:
                    raise RuntimeError(
                        "TemporalWindowBatchSampler exceeded the expected number of refill rounds "
                        f"while assembling a batch. target_batch_size={target_batch_size}, "
                        f"current_batch_size={len(batch)}, mix_windows={mix_windows}."
                    )

            if len(batch) != target_batch_size:
                raise RuntimeError(
                    "TemporalWindowBatchSampler assembled an unexpected batch size. "
                    f"expected={target_batch_size}, got={len(batch)}, produced={produced_samples}."
                )

            produced_samples += len(batch)
            yield batch


__all__ = ["TemporalWindowBatchSampler", "_temporal_identity_or_default_collate"]
