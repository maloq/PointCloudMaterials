import time

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, SequentialSampler

from src.data_utils.data_load import PointCloudDataset
from src.data_utils.data_modules.common import (
    _cfg_get,
    _resolve_split_seed,
    _resolve_temporal_window_start_frames,
    _split_temporal_window_start_frames,
    _to_container,
    logger,
)
from src.data_utils.data_modules.temporal_window import (
    TemporalWindowBatchSampler,
    _temporal_identity_or_default_collate,
)
from src.data_utils.temporal_lammps_dataset import (
    TemporalLAMMPSDumpDataset,
    estimate_lammps_dump_cutoff_radius,
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
        ctx = "TemporalLAMMPSDataModule.data"
        dump_file = _cfg_get(data_cfg, "dump_file", context=ctx)
        if dump_file is None or str(dump_file).strip() == "":
            raise ValueError("Temporal LAMMPS data configuration must provide data.dump_file")
        cache_dir = _cfg_get(data_cfg, "cache_dir", default=None, context=ctx)

        sequence_length = int(_cfg_get(data_cfg, "sequence_length", context=ctx))
        num_points = int(_cfg_get(data_cfg, "num_points", context=ctx))
        frame_stride = int(_cfg_get(data_cfg, "frame_stride", default=1, context=ctx))
        window_stride = int(_cfg_get(data_cfg, "window_stride", default=1, context=ctx))
        frame_start = int(_cfg_get(data_cfg, "frame_start", default=0, context=ctx))
        frame_stop_raw = _cfg_get(data_cfg, "frame_stop", default=None, context=ctx)
        frame_stop = None if frame_stop_raw is None else int(frame_stop_raw)

        # Scalar invariants (sequence_length>0, num_points>0, frame_start>=0, stride>0) are
        # validated inside `TemporalLAMMPSDumpDataset.__init__` / `_resolve_temporal_window_start_frames`.

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
            train_ratio=float(_cfg_get(data_cfg, "train_ratio", context=ctx)),
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
            center_selection_mode=_cfg_get(data_cfg, "center_selection_mode", default=None, context=ctx),
            center_atom_ids=_to_container(
                _cfg_get(data_cfg, "center_atom_ids", default=None, context=ctx)
            ),
            center_atom_stride=_cfg_get(data_cfg, "center_atom_stride", default=None, context=ctx),
            max_center_atoms=_cfg_get(data_cfg, "max_center_atoms", default=None, context=ctx),
            center_selection_seed=int(
                _cfg_get(data_cfg, "center_selection_seed", default=0, context=ctx)
            ),
            center_grid_overlap=_cfg_get(data_cfg, "center_grid_overlap", default=None, context=ctx),
            center_grid_reference_frame_index=_cfg_get(
                data_cfg, "center_grid_reference_frame_index", default=None, context=ctx
            ),
            normalize=bool(_cfg_get(data_cfg, "normalize", default=True, context=ctx)),
            center_neighborhoods=bool(
                _cfg_get(data_cfg, "center_neighborhoods", default=True, context=ctx)
            ),
            selection_method=str(
                _cfg_get(data_cfg, "selection_method", default="closest", context=ctx)
            ),
            cache_dir=cache_dir,
            rebuild_cache=bool(_cfg_get(data_cfg, "rebuild_cache", default=False, context=ctx)),
            tree_cache_size=int(_cfg_get(data_cfg, "tree_cache_size", default=4, context=ctx)),
            precompute_neighbor_indices=bool(
                _cfg_get(data_cfg, "precompute_neighbor_indices", default=False, context=ctx)
            ),
            build_lock_timeout_sec=float(
                _cfg_get(data_cfg, "build_lock_timeout_sec", default=7200.0, context=ctx)
            ),
            build_lock_stale_sec=float(
                _cfg_get(data_cfg, "build_lock_stale_sec", default=86400.0, context=ctx)
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

    def _resolve_radius(self, *, dump_file, data_cfg, frame_start: int, num_points: int) -> float:
        ctx = "TemporalLAMMPSDataModule.data"
        radius_raw = _cfg_get(data_cfg, "radius", default=None, context=ctx)
        auto_cutoff_cfg = PointCloudDataset._resolve_auto_cutoff_config(
            _to_container(_cfg_get(data_cfg, "auto_cutoff", default=None, context=ctx)),
            default_target_points=int(num_points),
            default_radius=float(radius_raw) if radius_raw is not None else 0.0,
        )
        if auto_cutoff_cfg is not None:
            if radius_raw is not None:
                raise ValueError(
                    "Temporal LAMMPS data: data.radius and data.auto_cutoff.enabled=true are "
                    "mutually exclusive — set exactly one. "
                    f"Got data.radius={float(radius_raw)!r} and data.auto_cutoff={dict(auto_cutoff_cfg)!r}."
                )
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


TemporalPointCloudDataModule = TemporalLAMMPSDataModule
