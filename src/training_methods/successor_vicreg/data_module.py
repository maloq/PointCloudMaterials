import time

import pytorch_lightning as pl
from torch.utils.data import DataLoader, SequentialSampler

from src.data_utils.data_load import PointCloudDataset
from src.data_utils.data_module import (
    TemporalWindowBatchSampler,
    _resolve_temporal_window_start_frames,
    _resolve_split_seed,
    _temporal_identity_or_default_collate,
    _to_container,
)
from src.data_utils.temporal_lammps_dataset import (
    TemporalLAMMPSDumpDataset,
    estimate_lammps_dump_cutoff_radius,
)
from src.utils.logging_config import setup_logging


logger = setup_logging()


def resolve_temporal_lammps_radius(*, dump_file, data_cfg, frame_start: int, num_points: int) -> float:
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
                "Successor-VICReg temporal radius: data.auto_cutoff.enabled=true, "
                f"so ignoring explicit data.radius={float(radius_raw):.6f}."
            )
        logger.print(
            "Successor-VICReg temporal radius: "
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
    logger.print(f"Successor-VICReg temporal radius: {radius_value:.6f} (source=data.radius)")
    return radius_value


def split_contiguous_anchor_frames(
    anchor_frames: list[int],
    *,
    train_ratio: float,
    val_ratio: float | None,
    context: str,
) -> tuple[list[int], list[int], list[int]]:
    if not anchor_frames:
        raise ValueError(f"{context}: anchor_frames must be non-empty.")

    total = int(len(anchor_frames))
    train_fraction = float(train_ratio)
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"{context}: train_ratio must be in (0, 1), got {train_ratio}.")

    train_size = int(train_fraction * total)
    if train_size <= 0:
        raise ValueError(
            f"{context}: train split is empty. total_windows={total}, train_ratio={train_ratio}."
        )

    if val_ratio is None:
        val_size = total - train_size
        if val_size <= 0:
            raise ValueError(
                f"{context}: validation split is empty. total_windows={total}, train_size={train_size}."
            )
        train_frames = [int(v) for v in anchor_frames[:train_size]]
        val_frames = [int(v) for v in anchor_frames[train_size:]]
        return train_frames, val_frames, list(val_frames)

    val_fraction = float(val_ratio)
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"{context}: val_ratio must be in (0, 1), got {val_ratio}.")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError(
            f"{context}: train_ratio + val_ratio must be < 1.0, got "
            f"{train_fraction + val_fraction:.6f}."
        )

    val_size = int(val_fraction * total)
    test_size = total - train_size - val_size
    if val_size <= 0 or test_size <= 0:
        raise ValueError(
            f"{context}: contiguous temporal split produced an empty subset. "
            f"total_windows={total}, train_size={train_size}, val_size={val_size}, "
            f"test_size={test_size}."
        )

    train_stop = train_size
    val_stop = train_stop + val_size
    train_frames = [int(v) for v in anchor_frames[:train_stop]]
    val_frames = [int(v) for v in anchor_frames[train_stop:val_stop]]
    test_frames = [int(v) for v in anchor_frames[val_stop:]]
    return train_frames, val_frames, test_frames


class SuccessorTemporalLAMMPSDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.max_samples = int(getattr(cfg, "max_samples", 0))
        self.split_seed = _resolve_split_seed(cfg)
        self._datasets_initialized = False

    def setup(self, stage=None):
        start_time = time.time()
        initialized_now = False
        if not self._datasets_initialized:
            self.train_dataset, self.val_dataset, self.test_dataset = self._build_datasets()
            if self.max_samples > 0:
                max_train = min(self.max_samples, len(self.train_dataset))
                max_val = min(self.max_samples, len(self.val_dataset))
                max_test = min(self.max_samples, len(self.test_dataset))
                self.train_dataset = self._truncate_dataset(self.train_dataset, max_train)
                self.val_dataset = self._truncate_dataset(self.val_dataset, max_val)
                self.test_dataset = self._truncate_dataset(self.test_dataset, max_test)
            self._datasets_initialized = True
            initialized_now = True

        elapsed_time = time.time() - start_time
        if not initialized_now:
            logger.print(
                f"Reusing existing Successor-VICReg temporal dataset split for stage={stage!r} "
                f"(split_seed={self.split_seed})."
            )
        logger.print(f"Successor-VICReg train dataset size: {len(self.train_dataset)}")
        logger.print(f"Successor-VICReg val dataset size: {len(self.val_dataset)}")
        logger.print(f"Successor-VICReg test dataset size: {len(self.test_dataset)}")
        logger.print(f"Successor-VICReg dataloader prep took {elapsed_time:.4f} seconds")

    @staticmethod
    def _truncate_dataset(dataset, max_items: int):
        if max_items <= 0:
            raise ValueError(f"max_items must be > 0 when truncating, got {max_items}.")
        import torch

        return torch.utils.data.Subset(dataset, range(int(max_items)))

    def _build_datasets(self):
        data_cfg = self.cfg.data
        dump_file = getattr(data_cfg, "dump_file", None)
        if dump_file is None or str(dump_file).strip() == "":
            raise ValueError("Successor-VICReg requires data.dump_file to be set.")
        cache_dir = getattr(data_cfg, "cache_dir", None)

        sequence_length = int(getattr(data_cfg, "sequence_length", 0))
        num_points = int(getattr(data_cfg, "num_points", 0))
        frame_stride = int(getattr(data_cfg, "frame_stride", 1))
        window_stride = int(getattr(data_cfg, "window_stride", 1))
        frame_start = int(getattr(data_cfg, "frame_start", 0))
        frame_stop = getattr(data_cfg, "frame_stop", None)
        frame_stop = None if frame_stop is None else int(frame_stop)

        if sequence_length <= 0:
            raise ValueError(f"data.sequence_length must be > 0, got {sequence_length}.")
        if num_points <= 0:
            raise ValueError(f"data.num_points must be > 0, got {num_points}.")

        scan = TemporalLAMMPSDumpDataset.scan_dump_file(dump_file, cache_dir=cache_dir)
        radius = resolve_temporal_lammps_radius(
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
        train_anchor_frames, val_anchor_frames, test_anchor_frames = split_contiguous_anchor_frames(
            anchor_frames,
            train_ratio=float(getattr(data_cfg, "train_ratio", 0.8)),
            val_ratio=getattr(data_cfg, "val_ratio", None),
            context="SuccessorTemporalLAMMPSDataModule._build_datasets",
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
        test_ds = TemporalLAMMPSDumpDataset(
            anchor_frame_indices=test_anchor_frames,
            **common_kwargs,
        )
        return train_ds, val_ds, test_ds

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
        )

    def test_dataloader(self):
        return self._temporal_loader(
            self.test_dataset,
            shuffle_windows=False,
            shuffle_centers=False,
            drop_last=False,
        )

