import time

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data.static import PointCloudDataset
from src.data.data_modules.common import (
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
                self.train_dataset = torch.utils.data.Subset(
                    self.train_dataset, range(max_train)
                )
                self.val_dataset = torch.utils.data.Subset(
                    self.val_dataset, range(max_val)
                )
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

        data_sources = _to_container(data_cfg.get("data_sources"))
        full_dataset = PointCloudDataset(
            data_sources=data_sources,
            root=data_cfg.data_path if data_sources is None else "",
            data_files=_to_container(data_cfg.data_files)
            if data_sources is None
            else None,
            radius=data_cfg.radius,
            sample_type=data_cfg.sample_type,
            overlap_fraction=data_cfg.get("overlap_fraction", 0.0),
            n_samples=data_cfg.get("n_samples", 1000),
            num_points=data_cfg.num_points,
            return_coords=self.return_coords,
            drop_edge_samples=bool(data_cfg.get("drop_edge_samples", True)),
            edge_drop_layers=data_cfg.get("edge_drop_layers", None),
            pre_normalize=bool(data_cfg.get("pre_normalize", True)),
            normalize=bool(data_cfg.get("normalize", True)),
            sampling_method=data_cfg.get("sampling_method", "drop_farthest"),
            auto_cutoff_config=_to_container(data_cfg.get("auto_cutoff")),
            sample_cache_config=_to_container(data_cfg.get("sample_cache")),
            atomic_context=_to_container(data_cfg.get("atomic_context", None)),
        )

        train_ratio = float(data_cfg.get("train_ratio", 0.8))
        train_size = int(train_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        if train_size <= 0 or val_size <= 0:
            raise ValueError(
                "Dataset split resulted in empty train or val set"
            )
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
