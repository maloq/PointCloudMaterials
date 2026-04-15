import math

import torch
import pytorch_lightning as pl
from torch.utils.data import BatchSampler, DataLoader, Sampler, SequentialSampler, random_split
from torch.utils.data._utils.collate import default_collate
from collections import defaultdict, deque
from src.data_utils.data_load import (
    PointCloudDataset,
    SyntheticPointCloudDataset,
)
from src.data_utils.temporal_lammps_dataset import (
    TemporalLAMMPSDumpDataset,
    estimate_lammps_dump_cutoff_radius,
)
import time
import logging
from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf

from src.utils.logging_config import setup_logging
logger = setup_logging()


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


def _resolve_split_seed(cfg, *, default: int = 42) -> int:
    data_cfg = getattr(cfg, "data", None)
    raw_seed = getattr(data_cfg, "split_seed", None) if data_cfg is not None else None
    if raw_seed is None:
        raw_seed = getattr(cfg, "split_seed", default)
    try:
        seed = int(raw_seed)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"split_seed must be an integer >= 0, got {raw_seed!r}"
        ) from exc
    if seed < 0:
        raise ValueError(f"split_seed must be >= 0, got {seed}")
    return seed


def _seeded_random_split(dataset, lengths: list[int], *, seed: int, context: str):
    if not hasattr(dataset, "__len__"):
        raise TypeError(
            f"{context}: dataset must define __len__, got {type(dataset)}"
        )
    split_lengths = [int(v) for v in lengths]
    if any(v < 0 for v in split_lengths):
        raise ValueError(
            f"{context}: split lengths must be non-negative, got {split_lengths}"
        )
    total = int(sum(split_lengths))
    n_items = int(len(dataset))
    if total != n_items:
        raise ValueError(
            f"{context}: split lengths must sum to dataset length, "
            f"got sum(lengths)={total}, len(dataset)={n_items}, lengths={split_lengths}."
        )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return random_split(dataset, split_lengths, generator=generator)


def _resolve_temporal_window_start_frames(
    *,
    frame_count: int,
    sequence_length: int,
    frame_stride: int,
    frame_start: int,
    frame_stop: int | None,
    window_stride: int,
) -> list[int]:
    frame_count = int(frame_count)
    sequence_length = int(sequence_length)
    frame_stride = int(frame_stride)
    frame_start = int(frame_start)
    window_stride = int(window_stride)
    stop = frame_count if frame_stop is None else int(frame_stop)

    if frame_count <= 0:
        raise ValueError(f"frame_count must be > 0, got {frame_count}")
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be > 0, got {sequence_length}")
    if frame_stride <= 0:
        raise ValueError(f"frame_stride must be > 0, got {frame_stride}")
    if window_stride <= 0:
        raise ValueError(f"window_stride must be > 0, got {window_stride}")
    if frame_start < 0 or frame_start >= frame_count:
        raise ValueError(
            f"frame_start must satisfy 0 <= frame_start < frame_count, got frame_start={frame_start}, "
            f"frame_count={frame_count}."
        )
    if stop <= frame_start:
        raise ValueError(
            f"frame_stop must be > frame_start, got frame_start={frame_start}, frame_stop={stop}."
        )
    if stop > frame_count:
        raise ValueError(
            f"frame_stop must be <= frame_count, got frame_stop={stop}, frame_count={frame_count}."
        )

    last_required_frame = frame_start + (sequence_length - 1) * frame_stride
    if last_required_frame >= stop:
        return []
    max_start = stop - (sequence_length - 1) * frame_stride
    return list(range(frame_start, max_start, window_stride))


def _split_temporal_window_start_frames(
    anchor_frames: list[int],
    *,
    train_ratio: float,
    seed: int,
    context: str,
) -> tuple[list[int], list[int]]:
    if not anchor_frames:
        raise ValueError(f"{context}: anchor_frames must be non-empty")

    train_ratio = float(train_ratio)
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"{context}: train_ratio must be in (0, 1), got {train_ratio}")

    num_frames = len(anchor_frames)
    train_size = int(train_ratio * num_frames)
    val_size = num_frames - train_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError(
            f"{context}: temporal split produced an empty subset. "
            f"num_windows={num_frames}, train_ratio={train_ratio}, "
            f"train_size={train_size}, val_size={val_size}."
        )

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    perm = torch.randperm(num_frames, generator=generator).tolist()
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]
    train_frames = sorted(int(anchor_frames[idx]) for idx in train_idx)
    val_frames = sorted(int(anchor_frames[idx]) for idx in val_idx)
    return train_frames, val_frames


class RealPointCloudDataModule(pl.LightningDataModule):
    def __init__(self, cfg, *, return_coords: bool = False):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.max_samples = cfg.max_samples
        self.split_seed = _resolve_split_seed(cfg)
        self.return_coords = bool(return_coords)
        self._datasets_initialized = False

    def setup(self, stage=None):
        start_time = time.time()
        initialized_now = False
        if not self._datasets_initialized:
            self.train_dataset, self.val_dataset = self._setup_real_dataset()
            # Test dataset is same as val for metrics.
            self.test_dataset = self.val_dataset

            if self.max_samples > 0:
                max_train = min(self.max_samples, len(self.train_dataset))
                max_val = min(self.max_samples, len(self.val_dataset))
                self.train_dataset = torch.utils.data.Subset(self.train_dataset, range(max_train))
                self.val_dataset = torch.utils.data.Subset(self.val_dataset, range(max_val))
                self.test_dataset = self.val_dataset
            self._datasets_initialized = True
            initialized_now = True

        elapsed_time = time.time() - start_time
        if not initialized_now:
            logger.print(
                f"Reusing existing real dataset split for stage={stage!r} "
                f"(split_seed={self.split_seed})."
            )
        logger.print(f"Train dataset size: {len(self.train_dataset)}")
        logger.print(f"Val dataset size: {len(self.val_dataset)}")
        logger.print(f"Test dataset size: {len(self.test_dataset)}")
        logger.print(f"Dataloader took {elapsed_time:.4f} seconds")

    def _setup_real_dataset(self):
        data_cfg = self.cfg.data

        data_sources_raw = getattr(data_cfg, "data_sources", None)
        data_files_raw = getattr(data_cfg, "data_files", None)
        auto_cutoff_cfg = _to_container(getattr(data_cfg, "auto_cutoff", None))
        dataset_common_kwargs = dict(
            radius=data_cfg.radius,
            sample_type=getattr(data_cfg, "sample_type", "regular"),
            overlap_fraction=getattr(data_cfg, "overlap_fraction", 0.0),
            n_samples=getattr(data_cfg, "n_samples", 1000),
            num_points=getattr(data_cfg, "num_points", 100),
            return_coords=self.return_coords,
            drop_edge_samples=bool(getattr(data_cfg, "drop_edge_samples", True)),
            edge_drop_layers=getattr(data_cfg, "edge_drop_layers", None),
            pre_normalize=bool(getattr(data_cfg, "pre_normalize", True)),
            normalize=bool(getattr(data_cfg, "normalize", True)),
            sampling_method=getattr(data_cfg, "sampling_method", "drop_farthest"),
            auto_cutoff_config=auto_cutoff_cfg,
        )

        if data_sources_raw is not None:
            data_sources = _to_container(data_sources_raw)
            if not isinstance(data_sources, list) or not data_sources:
                raise ValueError("data_sources must be a non-empty list of {data_path, data_files} dicts")
            full_dataset = PointCloudDataset(
                data_sources=data_sources,
                **dataset_common_kwargs,
            )
        elif data_files_raw is not None:
            file_list = _to_container(data_files_raw)
            if isinstance(file_list, str):
                file_list = [file_list]
            data_path = getattr(data_cfg, "data_path", None)
            if not data_path:
                raise ValueError("data_path is required when using data_files (single-source mode)")
            full_dataset = PointCloudDataset(
                root=data_path,
                data_files=file_list,
                **dataset_common_kwargs,
            )
        else:
            raise ValueError(
                "Data config must provide either 'data_sources' (multi-material) "
                "or 'data_files' + 'data_path' (single-material)"
            )

        train_ratio = getattr(data_cfg, "train_ratio", 0.8)
        train_size = int(train_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        if train_size <= 0 or val_size <= 0:
            raise ValueError("Dataset split resulted in empty train or val set")
        train_ds, val_ds = _seeded_random_split(
            full_dataset,
            [train_size, val_size],
            seed=self.split_seed,
            context="RealPointCloudDataModule._setup_real_dataset",
        )
        return train_ds, val_ds

    def train_dataloader(self):
        print(f"Using {self.num_workers} workers for train dataloader")
        main = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
        return main

    def val_dataloader(self):
        print(f"Using {self.num_workers} workers for val dataloader")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        print(f"Using {self.num_workers} workers for test dataloader")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


class TemporalLAMMPSDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.max_samples = cfg.max_samples
        self.split_seed = _resolve_split_seed(cfg)
        self._datasets_initialized = False

    def setup(self, stage=None):
        start_time = time.time()
        initialized_now = False
        if not self._datasets_initialized:
            self.train_dataset, self.val_dataset = self._build_datasets()
            self.test_dataset = self.val_dataset

            if self.max_samples > 0:
                max_train = min(self.max_samples, len(self.train_dataset))
                max_val = min(self.max_samples, len(self.val_dataset))
                self.train_dataset = torch.utils.data.Subset(self.train_dataset, range(max_train))
                self.val_dataset = torch.utils.data.Subset(self.val_dataset, range(max_val))
                self.test_dataset = self.val_dataset
            self._datasets_initialized = True
            initialized_now = True

        elapsed_time = time.time() - start_time
        if not initialized_now:
            logger.print(
                f"Reusing existing temporal LAMMPS dataset split for stage={stage!r} "
                f"(split_seed={self.split_seed})."
            )
        logger.print(f"Temporal train dataset size: {len(self.train_dataset)}")
        logger.print(f"Temporal val dataset size: {len(self.val_dataset)}")
        logger.print(f"Temporal test dataset size: {len(self.test_dataset)}")
        logger.print(f"Temporal dataloader prep took {elapsed_time:.4f} seconds")

    def _build_datasets(self):
        data_cfg = self.cfg.data
        dump_file = getattr(data_cfg, "dump_file", None)
        if dump_file is None or str(dump_file).strip() == "":
            raise ValueError("Temporal LAMMPS data configuration must provide data.dump_file")
        cache_dir = getattr(data_cfg, "cache_dir", None)

        sequence_length = int(getattr(data_cfg, "sequence_length", 0))
        num_points = int(getattr(data_cfg, "num_points", 0))
        frame_stride = int(getattr(data_cfg, "frame_stride", 1))
        window_stride = int(getattr(data_cfg, "window_stride", 1))
        frame_start = int(getattr(data_cfg, "frame_start", 0))
        frame_stop = getattr(data_cfg, "frame_stop", None)
        frame_stop = None if frame_stop is None else int(frame_stop)

        if sequence_length <= 0:
            raise ValueError(f"data.sequence_length must be > 0, got {sequence_length}")
        if num_points <= 0:
            raise ValueError(f"data.num_points must be > 0, got {num_points}")

        scan = TemporalLAMMPSDumpDataset.scan_dump_file(dump_file, cache_dir=cache_dir)
        radius = self._resolve_radius(
            dump_file=dump_file,
            data_cfg=data_cfg,
            frame_start=frame_start,
            num_points=num_points,
        )
        anchor_frames = _resolve_temporal_window_start_frames(
            frame_count=int(scan.frame_count),
            sequence_length=sequence_length,
            frame_stride=frame_stride,
            frame_start=frame_start,
            frame_stop=frame_stop,
            window_stride=window_stride,
        )
        train_anchor_frames, val_anchor_frames = _split_temporal_window_start_frames(
            anchor_frames,
            train_ratio=float(getattr(data_cfg, "train_ratio", 0.8)),
            seed=self.split_seed,
            context="TemporalLAMMPSDataModule._build_datasets",
        )

        common_kwargs = dict(
            dump_file=dump_file,
            sequence_length=sequence_length,
            num_points=num_points,
            radius=radius,
            frame_stride=frame_stride,
            window_stride=window_stride,
            frame_start=frame_start,
            frame_stop=frame_stop,
            center_selection_mode=getattr(data_cfg, "center_selection_mode", None),
            center_atom_ids=_to_container(getattr(data_cfg, "center_atom_ids", None)),
            center_atom_stride=getattr(data_cfg, "center_atom_stride", None),
            max_center_atoms=getattr(data_cfg, "max_center_atoms", None),
            center_selection_seed=int(getattr(data_cfg, "center_selection_seed", 0)),
            center_grid_overlap=getattr(data_cfg, "center_grid_overlap", None),
            center_grid_reference_frame_index=getattr(data_cfg, "center_grid_reference_frame_index", None),
            normalize=bool(getattr(data_cfg, "normalize", True)),
            center_neighborhoods=bool(getattr(data_cfg, "center_neighborhoods", True)),
            selection_method=str(getattr(data_cfg, "selection_method", "closest")),
            cache_dir=cache_dir,
            rebuild_cache=bool(getattr(data_cfg, "rebuild_cache", False)),
            tree_cache_size=int(getattr(data_cfg, "tree_cache_size", 4)),
            precompute_neighbor_indices=bool(getattr(data_cfg, "precompute_neighbor_indices", False)),
            build_lock_timeout_sec=float(getattr(data_cfg, "build_lock_timeout_sec", 7200.0)),
            build_lock_stale_sec=float(getattr(data_cfg, "build_lock_stale_sec", 86400.0)),
        )

        train_ds = TemporalLAMMPSDumpDataset(
            anchor_frame_indices=train_anchor_frames,
            **common_kwargs,
        )
        val_ds = TemporalLAMMPSDumpDataset(
            anchor_frame_indices=val_anchor_frames,
            **common_kwargs,
        )
        return train_ds, val_ds

    def _resolve_radius(self, *, dump_file, data_cfg, frame_start: int, num_points: int) -> float:
        radius_raw = getattr(data_cfg, "radius", None)
        auto_cutoff_cfg = PointCloudDataset._resolve_auto_cutoff_config(
            _to_container(getattr(data_cfg, "auto_cutoff", None)),
            default_target_points=int(num_points),
            default_radius=float(radius_raw) if radius_raw is not None else 0.0,
        )
        if auto_cutoff_cfg is not None:
            reference_frame_index = int(auto_cutoff_cfg.get("reference_frame_index", frame_start))
            estimation = estimate_lammps_dump_cutoff_radius(
                dump_file,
                reference_frame_index=reference_frame_index,
                target_points=max(
                    int(num_points),
                    int(auto_cutoff_cfg.get("target_points", num_points)),
                ),
                quantile=float(auto_cutoff_cfg.get("quantile", 1.0)),
                estimation_samples=int(auto_cutoff_cfg.get("estimation_samples_per_file", 4096)),
                seed=int(auto_cutoff_cfg.get("seed", 0)),
                safety_factor=float(auto_cutoff_cfg.get("safety_factor", 1.0)),
                boundary_margin=auto_cutoff_cfg.get("boundary_margin", None),
                periodic=False,
            )
            radius_value = float(estimation["estimated_radius"])
            if radius_raw is not None:
                logger.print(
                    "Temporal LAMMPS radius: data.auto_cutoff.enabled=true, "
                    f"so ignoring explicit data.radius={float(radius_raw):.6f}."
                )
            logger.print(
                "Temporal LAMMPS radius: "
                f"{radius_value:.6f} (source=auto_cutoff_static_style, "
                f"reference_frame_index={reference_frame_index}, "
                f"coverage={float(estimation['coverage']):.4f}, "
                f"boundary_margin={estimation['boundary_margin']})"
            )
            return radius_value

        if radius_raw is None:
            raise ValueError(
                "Temporal LAMMPS data requires either data.radius or data.auto_cutoff.enabled=true."
            )

        radius_value = float(radius_raw)
        if radius_value <= 0.0:
            raise ValueError(f"data.radius must be > 0, got {radius_value}.")
        logger.print(f"Temporal LAMMPS radius: {radius_value:.6f} (source=data.radius)")
        return radius_value

    def _temporal_loader(
        self,
        dataset,
        *,
        shuffle_windows: bool,
        shuffle_centers: bool,
        drop_last: bool,
        mixed_windows_per_batch: int | None = None,
        shuffle_window_block_size: int | None = None,
        shared_center_order: bool = False,
    ):
        batch_sampler = TemporalWindowBatchSampler(
            SequentialSampler(dataset),
            batch_size=self.batch_size,
            drop_last=drop_last,
            dataset=dataset,
            shuffle_windows=shuffle_windows,
            shuffle_centers=shuffle_centers,
            mixed_windows_per_batch=mixed_windows_per_batch,
            shuffle_window_block_size=shuffle_window_block_size,
            shared_center_order=shared_center_order,
        )
        loader_kwargs = dict(
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=_temporal_identity_or_default_collate,
        )
        prefetch_factor = getattr(self.cfg, "temporal_prefetch_factor", None)
        if prefetch_factor is not None:
            resolved_prefetch_factor = int(prefetch_factor)
            if resolved_prefetch_factor <= 0:
                raise ValueError(
                    "temporal_prefetch_factor must be > 0 when provided, "
                    f"got {resolved_prefetch_factor}."
                )
            if self.num_workers <= 0:
                raise ValueError(
                    "temporal_prefetch_factor requires num_workers > 0, "
                    f"got num_workers={self.num_workers}."
                )
            loader_kwargs["prefetch_factor"] = resolved_prefetch_factor
        return DataLoader(
            dataset,
            **loader_kwargs,
        )

    def train_dataloader(self):
        mixed_windows_per_batch_raw = getattr(self.cfg, "temporal_train_windows_per_batch", None)
        if mixed_windows_per_batch_raw is None:
            mixed_windows_per_batch = min(
                int(self.batch_size),
                int(getattr(self.train_dataset, "window_count", self.batch_size)),
            )
        else:
            mixed_windows_per_batch = int(mixed_windows_per_batch_raw)
        shuffle_window_block_size = getattr(
            self.cfg,
            "temporal_train_window_shuffle_block_size",
            None,
        )
        shared_center_order = bool(
            getattr(self.cfg, "temporal_train_shared_center_order", False)
        )
        return self._temporal_loader(
            self.train_dataset,
            shuffle_windows=True,
            shuffle_centers=True,
            drop_last=True,
            mixed_windows_per_batch=mixed_windows_per_batch,
            shuffle_window_block_size=shuffle_window_block_size,
            shared_center_order=shared_center_order,
        )

    def val_dataloader(self):
        return self._temporal_loader(
            self.val_dataset,
            shuffle_windows=False,
            shuffle_centers=False,
            drop_last=False,
            mixed_windows_per_batch=None,
        )

    def test_dataloader(self):
        return self._temporal_loader(
            self.test_dataset,
            shuffle_windows=False,
            shuffle_centers=False,
            drop_last=False,
            mixed_windows_per_batch=None,
        )


class SyntheticPointCloudDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.max_samples = cfg.max_samples
        self.split_seed = _resolve_split_seed(cfg)
        self._phase_info_logged = False
        self._datasets_initialized = False

    def setup(self, stage=None):
        start_time = time.time()
        initialized_now = False
        if not self._datasets_initialized:
            self.train_dataset, self.val_dataset = self._build_datasets()
            # Test dataset is same as val for metrics.
            self.test_dataset = self.val_dataset

            if self.max_samples > 0:
                max_train = min(self.max_samples, len(self.train_dataset))
                max_val = min(self.max_samples, len(self.val_dataset))
                self.train_dataset = torch.utils.data.Subset(self.train_dataset, range(max_train))
                self.val_dataset = torch.utils.data.Subset(self.val_dataset, range(max_val))
                self.test_dataset = self.val_dataset
            self._datasets_initialized = True
            initialized_now = True

        if (stage is None or stage == 'fit') and not self._phase_info_logged:
            self._log_class_mapping()

        elapsed_time = time.time() - start_time
        if not initialized_now:
            logger.print(
                f"Reusing existing synthetic dataset split for stage={stage!r} "
                f"(split_seed={self.split_seed})."
            )
        logger.print(f"Synth train dataset size: {len(self.train_dataset)}")
        logger.print(f"Synth val dataset size: {len(self.val_dataset)}")
        logger.print(f"Synth test dataset size: {len(self.test_dataset)}")
        logger.print(f"Synth dataloader prep took {elapsed_time:.4f} seconds")

    def _build_datasets(self):
        data_cfg = self.cfg.data
        synth_cfg = getattr(data_cfg, "synthetic", None)
        synth_dict = _to_container(synth_cfg) if synth_cfg is not None else {}

        aug_cfg = getattr(data_cfg, "augmentation", None)
        if aug_cfg is None:
            aug_cfg = getattr(self.cfg, "augmentation", None)
        rotation_scale = getattr(aug_cfg, "rotation_scale", 0.0) if aug_cfg else 0.0
        noise_scale = getattr(aug_cfg, "noise_scale", 0.0) if aug_cfg else 0.0
        jitter_scale = getattr(aug_cfg, "jitter_scale", 0.0) if aug_cfg else 0.0
        scaling_range = getattr(aug_cfg, "scaling_range", 0.0) if aug_cfg else 0.0
        track_augmentation = getattr(aug_cfg, "track_augmentation", False) if aug_cfg else False

        dataset_type = self._get_param("dataset_type", data_cfg, synth_dict, default="synthetic_env")
        auto_cutoff_cfg = _to_container(self._get_param("auto_cutoff", data_cfg, synth_dict, default=None))
        model_type = str(getattr(self.cfg, "model_type", "")).lower()
        disable_dataset_aug_for_ssl = bool(getattr(self.cfg, "disable_dataset_augmentation_for_ssl", True))
        uses_ssl_views = (
            model_type == "vicreg"
            or bool(getattr(self.cfg, "vicreg_enabled", False))
        )
        if uses_ssl_views and disable_dataset_aug_for_ssl:
            has_dataset_aug = any(
                float(v or 0.0) != 0.0 for v in (rotation_scale, noise_scale, jitter_scale, scaling_range)
            )
            if has_dataset_aug:
                logger.print(
                    "Disabling dataset-level geometric augmentation for contrastive SSL; "
                    "view augmentation is applied in the contrastive loss."
                )
            rotation_scale = 0.0
            noise_scale = 0.0
            jitter_scale = 0.0
            scaling_range = 0.0
            track_augmentation = False

        if dataset_type != "synthetic_env":
            raise ValueError(
                f"Unsupported dataset_type {dataset_type!r}. "
                "SyntheticPointCloudDataModule only supports 'synthetic_env'."
            )

        env_dirs = self._resolve_env_dirs(data_cfg, synth_dict)
        radius = self._get_param("radius", data_cfg, synth_dict, required=True)
        sample_type = self._get_param("sample_type", data_cfg, synth_dict, default="regular")
        overlap_fraction = self._get_param("overlap_fraction", data_cfg, synth_dict, default=0.0)
        n_samples = self._get_param("n_samples", data_cfg, synth_dict, default=0)
        num_points = self._get_param("num_points", data_cfg, synth_dict, required=True)
        drop_edge_samples = self._get_param("drop_edge_samples", data_cfg, synth_dict, default=True)
        pre_normalize = self._get_param("pre_normalize", data_cfg, synth_dict, default=True)
        normalize = self._get_param("normalize", data_cfg, synth_dict, default=True)
        dataset_max = self._get_param("dataset_max_samples", data_cfg, synth_dict, default=None)
        classes = self._get_param("classes", data_cfg, synth_dict, default=None)
        if classes is not None:
             classes = _to_container(classes)
             if isinstance(classes, str):
                 classes = [classes]

        dataset = SyntheticPointCloudDataset(
            env_dirs=env_dirs,
            radius=float(radius),
            sample_type=str(sample_type),
            overlap_fraction=float(overlap_fraction),
            n_samples=int(n_samples),
            num_points=int(num_points),
            drop_edge_samples=bool(drop_edge_samples),
            pre_normalize=bool(pre_normalize),
            normalize=bool(normalize),
            max_samples=int(dataset_max) if dataset_max is not None else None,
            rotation_scale=rotation_scale,
            noise_scale=noise_scale,
            jitter_scale=jitter_scale,
            scaling_range=scaling_range,
            track_augmentation=track_augmentation,
            allowed_classes=classes,
            auto_cutoff_config=auto_cutoff_cfg,
        )

        train_ratio = float(self._get_param("train_ratio", data_cfg, synth_dict, default=0.8))
        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        if train_size <= 0 or val_size <= 0:
            raise ValueError("Synthetic dataset split resulted in empty train or val set")
        return _seeded_random_split(
            dataset,
            [train_size, val_size],
            seed=self.split_seed,
            context="SyntheticPointCloudDataModule._build_datasets(synthetic)",
        )

    def _resolve_env_dirs(self, data_cfg, synth_dict):
        env_dirs = []
        if isinstance(synth_dict, dict):
            if "data_dirs" in synth_dict and synth_dict["data_dirs"]:
                env_dirs = synth_dict["data_dirs"]
            elif synth_dict.get("data_dir"):
                env_dirs = [synth_dict["data_dir"]]
            elif synth_dict.get("root_dir"):
                root = Path(synth_dict["root_dir"])
                if not root.exists():
                    raise FileNotFoundError(f"Synthetic root_dir {root} does not exist")
                candidates = sorted([p for p in root.iterdir() if p.is_dir()])
                num_env = synth_dict.get("num_environments")
                if num_env is not None:
                    candidates = candidates[:int(num_env)]
                env_dirs = candidates
        if not env_dirs and hasattr(data_cfg, "data_path"):
            env_dirs = [getattr(data_cfg, "data_path")]
        if not env_dirs:
            raise ValueError("No synthetic environment directories configured")
        resolved = []
        for path in env_dirs:
            if isinstance(path, (Path, str)):
                resolved.append(Path(path))
            else:
                raise TypeError(f"Unsupported path type {type(path)} in synthetic env_dirs")
        return resolved

    @staticmethod
    def _get_param(name, data_cfg, synth_dict, *, required=False, default=None):
        value = getattr(data_cfg, name, None)
        if value is None and isinstance(synth_dict, dict):
            value = synth_dict.get(name, None)
        if value is None:
            if default is not None:
                return default
            if required:
                raise ValueError(f"Synthetic data configuration must provide '{name}'")
        return value

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        use_persistent = self.num_workers > 0
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=use_persistent,
        )

    def test_dataloader(self):
        use_persistent = self.num_workers > 0
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=use_persistent,
        )

    def _resolve_synthetic_dataset_with_indices(self):
        dataset = getattr(self, 'train_dataset', None)
        indices = None
        visited = set()
        while dataset is not None and id(dataset) not in visited:
            visited.add(id(dataset))
            if isinstance(dataset, torch.utils.data.Subset):
                current = list(dataset.indices)
                if indices is None:
                    indices = current
                else:
                    indices = [current[i] for i in indices]
                dataset = dataset.dataset
                continue
            if isinstance(dataset, SyntheticPointCloudDataset):
                return dataset, indices
            next_dataset = getattr(dataset, 'dataset', None)
            if next_dataset is not None and next_dataset is not dataset:
                dataset = next_dataset
                continue
            base_dataset = getattr(dataset, 'base_dataset', None)
            if base_dataset is not None and base_dataset is not dataset:
                dataset = base_dataset
                continue
            break
        return None, None

    @staticmethod
    def _anisotropy_from_points(points: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if points.dim() != 3 or points.shape[-1] != 3:
            raise ValueError(f"Expected points of shape (B,N,3), got {tuple(points.shape)}")
        centered = points - points.mean(dim=1, keepdim=True)
        cov = (centered.transpose(1, 2) @ centered) / float(centered.shape[1])
        eigvals = torch.linalg.eigvalsh(cov.to(dtype=torch.float32))
        eigvals, _ = torch.sort(eigvals, dim=-1, descending=True)
        lam1, lam2, lam3 = eigvals.unbind(dim=-1)
        return (lam1 - lam3) / (lam1 + lam2 + lam3 + float(eps))

    @staticmethod
    def _class_id_to_name_map(dataset) -> dict[int, str]:
        class_names = getattr(dataset, "class_names", None)
        if class_names is None:
            return {}
        return {int(k): str(v) for k, v in dict(class_names).items()}

    def _log_train_anisotropy_by_class(self, dataset, subset_indices: list[int] | None) -> None:
        class_ids_raw = getattr(dataset, "_class_ids", None)
        points_raw = getattr(dataset, "samples", None)
        if points_raw is None:
            points_raw = getattr(dataset, "points", None)
        class_name_map = self._class_id_to_name_map(dataset)

        sums = defaultdict(float)
        counts = defaultdict(int)

        if class_ids_raw is not None and points_raw is not None:
            class_ids = class_ids_raw if torch.is_tensor(class_ids_raw) else torch.as_tensor(class_ids_raw)
            class_ids = class_ids.to(dtype=torch.long).view(-1).cpu()
            total = int(class_ids.shape[0])
            if subset_indices is None:
                selected = list(range(total))
            else:
                selected = [int(i) for i in subset_indices if 0 <= int(i) < total]

            chunk_size = 512
            for start in range(0, len(selected), chunk_size):
                idx_list = selected[start:start + chunk_size]
                if not idx_list:
                    continue
                idx = torch.as_tensor(idx_list, dtype=torch.long)
                if torch.is_tensor(points_raw):
                    pts = points_raw.index_select(0, idx).to(dtype=torch.float32)
                else:
                    pts = torch.stack([points_raw[int(i)] for i in idx_list], dim=0).to(dtype=torch.float32)
                anis = self._anisotropy_from_points(pts).to(dtype=torch.float64).cpu()
                cls = class_ids.index_select(0, idx)
                valid = cls >= 0
                if not valid.any():
                    continue
                cls = cls[valid]
                anis = anis[valid]
                for cid in torch.unique(cls):
                    mask = cls == cid
                    cid_int = int(cid.item())
                    sums[cid_int] += float(anis[mask].sum().item())
                    counts[cid_int] += int(mask.sum().item())
        else:
            train_ds = getattr(self, "train_dataset", None)
            if train_ds is not None:
                for idx in range(len(train_ds)):
                    sample = train_ds[idx]
                    if not isinstance(sample, dict):
                        continue
                    cls = sample.get("class_id", None)
                    pts = sample.get("points", None)
                    if cls is None or pts is None:
                        continue
                    cls_val = int(cls.item()) if torch.is_tensor(cls) else int(cls)
                    if cls_val < 0:
                        continue
                    anis = float(self._anisotropy_from_points(pts.unsqueeze(0).to(dtype=torch.float32)).item())
                    sums[cls_val] += anis
                    counts[cls_val] += 1

        logger.print("Train anisotropy by class (mean over train split):")
        if not counts:
            logger.print("  unavailable (no class_id labels found)")
            return
        for cid in sorted(counts):
            name = class_name_map.get(cid, f"class_{cid}")
            mean_anis = sums[cid] / max(1, counts[cid])
            logger.print(f"  {name}: {mean_anis:.6f} (n={counts[cid]})")

    def _log_class_mapping(self):
        """Log the class name to ID mapping for the dataset."""
        base_dataset, subset_indices = self._resolve_synthetic_dataset_with_indices()
        if base_dataset is None or not hasattr(base_dataset, '_class_to_idx'):
            logger.print("Class mapping unavailable; dataset not initialized yet")
            return

        if not base_dataset._class_to_idx:
            logger.print("Dataset did not expose any class labels")
            return

        domain = getattr(base_dataset, 'domain', 'unknown')
        logger.print(f"Dataset domain: {domain}")
        logger.print("Class labels:")
        for class_name, class_idx in sorted(base_dataset._class_to_idx.items(), key=lambda item: item[1]):
            logger.print(f"  Class {class_idx}: {class_name}")

        self._log_train_anisotropy_by_class(base_dataset, subset_indices)
        self._phase_info_logged = True


class PointCloudDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        kind = getattr(cfg.data, "kind", "real")
        if kind == "synthetic":
            self.impl = SyntheticPointCloudDataModule(cfg)
        elif kind == "temporal_lammps":
            self.impl = TemporalLAMMPSDataModule(cfg)
        else:
            self.impl = RealPointCloudDataModule(cfg)

    def setup(self, stage=None):
        return self.impl.setup(stage)

    def train_dataloader(self):
        return self.impl.train_dataloader()

    def val_dataloader(self):
        return self.impl.val_dataloader()

    def test_dataloader(self):
        return self.impl.test_dataloader()


# Module-level helper
def _to_container(cfg):
    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg

SynthPointCloudDataModule = SyntheticPointCloudDataModule
TemporalPointCloudDataModule = TemporalLAMMPSDataModule
