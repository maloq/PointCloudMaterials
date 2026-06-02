import time

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data_utils.data_load import PointCloudDataset
from src.data_utils.data_modules.common import (
    _cfg_get,
    _resolve_split_seed,
    _seeded_random_split,
    _to_container,
    logger,
)


class StaticPointCloudDataModule(pl.LightningDataModule):
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
            self.train_dataset, self.val_dataset = self._setup_static_dataset()
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
                f"Reusing existing static dataset split for stage={stage!r} "
                f"(split_seed={self.split_seed})."
            )
        logger.print(f"Train dataset size: {len(self.train_dataset)}")
        logger.print(f"Val dataset size: {len(self.val_dataset)}")
        logger.print(f"Test dataset size: {len(self.test_dataset)}")
        logger.print(f"Dataloader took {elapsed_time:.4f} seconds")

    def _setup_static_dataset(self):
        data_cfg = self.cfg.data
        ctx = "StaticPointCloudDataModule.data"

        data_sources_raw = _cfg_get(data_cfg, "data_sources", default=None, context=ctx)
        data_files_raw = _cfg_get(data_cfg, "data_files", default=None, context=ctx)
        auto_cutoff_cfg = _to_container(_cfg_get(data_cfg, "auto_cutoff", default=None, context=ctx))
        dataset_common_kwargs = dict(
            radius=_cfg_get(data_cfg, "radius", context=ctx),
            sample_type=_cfg_get(data_cfg, "sample_type", context=ctx),
            overlap_fraction=_cfg_get(data_cfg, "overlap_fraction", default=0.0, context=ctx),
            n_samples=_cfg_get(data_cfg, "n_samples", default=1000, context=ctx),
            num_points=_cfg_get(data_cfg, "num_points", context=ctx),
            return_coords=self.return_coords,
            drop_edge_samples=bool(_cfg_get(data_cfg, "drop_edge_samples", default=True, context=ctx)),
            edge_drop_layers=_cfg_get(data_cfg, "edge_drop_layers", default=None, context=ctx),
            pre_normalize=bool(_cfg_get(data_cfg, "pre_normalize", default=True, context=ctx)),
            normalize=bool(_cfg_get(data_cfg, "normalize", default=True, context=ctx)),
            sampling_method=_cfg_get(data_cfg, "sampling_method", default="drop_farthest", context=ctx),
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
            data_path = _cfg_get(data_cfg, "data_path", context=ctx)
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

        train_ratio = float(_cfg_get(data_cfg, "train_ratio", default=0.8, context=ctx))
        train_size = int(train_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        if train_size <= 0 or val_size <= 0:
            raise ValueError("Dataset split resulted in empty train or val set")
        train_ds, val_ds = _seeded_random_split(
            full_dataset,
            [train_size, val_size],
            seed=self.split_seed,
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


RealPointCloudDataModule = StaticPointCloudDataModule
