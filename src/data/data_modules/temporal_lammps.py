import time

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, SequentialSampler

from src.data.static_sources import resolve_auto_cutoff_config
from src.data.data_modules.common import (
    _resolve_split_seed,
    _resolve_temporal_window_start_frames,
    _split_temporal_window_start_frames,
    _to_container,
    logger,
)
from src.data.data_modules.temporal_window import (
    TemporalWindowBatchSampler,
    _identity_batch_collate,
)
from src.data.temporal import (
    TemporalLAMMPSDumpDataset,
    estimate_lammps_dump_cutoff_radius,
)


class TemporalLAMMPSDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.split_seed = _resolve_split_seed(cfg)
        self._datasets_initialized = False

    def setup(self, stage=None):
        start_time = time.time()
        initialized_now = False
        if not self._datasets_initialized:
            self.train_dataset, self.val_dataset = self._build_datasets()
            self.test_dataset = self.val_dataset

            self._datasets_initialized = True
            initialized_now = True

        elapsed_time = time.time() - start_time
        if not initialized_now:
            logger.print(
                "Reusing existing temporal LAMMPS dataset split for"
                f" stage={stage!r} (split_seed={self.split_seed})."
            )
        logger.print(f"Temporal train dataset size: {len(self.train_dataset)}")
        logger.print(f"Temporal val dataset size: {len(self.val_dataset)}")
        logger.print(f"Temporal test dataset size: {len(self.test_dataset)}")
        logger.print(
            f"Temporal dataloader prep took {elapsed_time:.4f} seconds"
        )

    def _build_datasets(self):
        data_cfg = self.cfg.data
        dump_file = data_cfg.dump_file
        cache_dir = data_cfg.get("cache_dir", None)

        sequence_length = int(data_cfg.sequence_length)
        num_points = int(data_cfg.num_points)
        frame_stride = int(data_cfg.get("frame_stride", 1))
        window_stride = int(data_cfg.get("window_stride", 1))
        frame_start = int(data_cfg.get("frame_start", 0))
        frame_stop_raw = data_cfg.get("frame_stop", None)
        frame_stop = None if frame_stop_raw is None else int(frame_stop_raw)

        scan = TemporalLAMMPSDumpDataset.scan_dump_file(
            dump_file, cache_dir=cache_dir
        )
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
        train_anchor_frames, val_anchor_frames = (
            _split_temporal_window_start_frames(
                anchor_frames,
                train_ratio=float(data_cfg.train_ratio),
                seed=self.split_seed,
                context="TemporalLAMMPSDataModule._build_datasets",
            )
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
            center_selection_mode=data_cfg.center_selection_mode,
            center_atom_ids=_to_container(
                data_cfg.get("center_atom_ids", None)
            ),
            center_atom_stride=data_cfg.get("center_atom_stride", None),
            max_center_atoms=data_cfg.get("max_center_atoms", None),
            center_selection_seed=int(
                data_cfg.get("center_selection_seed", 0)
            ),
            center_grid_overlap=data_cfg.get("center_grid_overlap", None),
            center_grid_reference_frame_index=data_cfg.get(
                "center_grid_reference_frame_index", None
            ),
            normalize=bool(data_cfg.get("normalize", True)),
            center_neighborhoods=bool(
                data_cfg.get("center_neighborhoods", True)
            ),
            selection_method=str(data_cfg.get("selection_method", "closest")),
            cache_dir=cache_dir,
            rebuild_cache=bool(data_cfg.get("rebuild_cache", False)),
            tree_cache_size=int(data_cfg.get("tree_cache_size", 4)),
            precompute_neighbor_indices=bool(
                data_cfg.get("precompute_neighbor_indices", False)
            ),
            build_lock_timeout_sec=float(
                data_cfg.get("build_lock_timeout_sec", 7200.0)
            ),
            build_lock_stale_sec=float(
                data_cfg.get("build_lock_stale_sec", 86400.0)
            ),
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

    def _resolve_radius(
        self, *, dump_file, data_cfg, frame_start: int, num_points: int
    ) -> float:
        radius_raw = data_cfg.get("radius", None)
        auto_cutoff_cfg = resolve_auto_cutoff_config(
            _to_container(data_cfg.get("auto_cutoff", None)),
        )
        if auto_cutoff_cfg is not None:
            if radius_raw is not None:
                raise ValueError(
                    "Temporal LAMMPS data: data.radius and"
                    " data.auto_cutoff.enabled=true are mutually exclusive —"
                    " set exactly one. Got"
                    f" data.radius={float(radius_raw)!r} and"
                    f" data.auto_cutoff={dict(auto_cutoff_cfg)!r}."
                )
            reference_frame_index = int(
                auto_cutoff_cfg["reference_frame_index"]
            )
            estimation = estimate_lammps_dump_cutoff_radius(
                dump_file,
                reference_frame_index=reference_frame_index,
                target_points=max(
                    int(num_points),
                    int(auto_cutoff_cfg["target_points"]),
                ),
                quantile=float(auto_cutoff_cfg["quantile"]),
                estimation_samples=int(
                    auto_cutoff_cfg["estimation_samples_per_file"]
                ),
                seed=int(auto_cutoff_cfg["seed"]),
                safety_factor=float(auto_cutoff_cfg["safety_factor"]),
                boundary_margin=auto_cutoff_cfg["boundary_margin"],
                periodic=False,
            )
            radius_value = float(estimation["estimated_radius"])
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
                "Temporal LAMMPS data requires either data.radius or"
                " data.auto_cutoff.enabled=true."
            )

        radius_value = float(radius_raw)
        logger.print(
            f"Temporal LAMMPS radius: {radius_value:.6f} (source=data.radius)"
        )
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
            collate_fn=_identity_batch_collate,
        )
        prefetch_factor = getattr(self.cfg, "temporal_prefetch_factor", None)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
        return DataLoader(
            dataset,
            **loader_kwargs,
        )

    def train_dataloader(self):
        mixed_windows_per_batch_raw = getattr(
            self.cfg, "temporal_train_windows_per_batch", None
        )
        if mixed_windows_per_batch_raw is None:
            mixed_windows_per_batch = min(
                int(self.batch_size),
                int(self.train_dataset.window_count),
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


TemporalPointCloudDataModule = TemporalLAMMPSDataModule
