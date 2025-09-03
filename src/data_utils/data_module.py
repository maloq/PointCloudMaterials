import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset, random_split
from src.data_utils.data_load import PointCloudDataset
from src.data_utils.neighbor_pairs import NeighborPairDataset
import time
import logging
from src.utils.logging_config import setup_logging
logger = setup_logging()



class PointCloudDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.max_samples = cfg.max_samples
        # Neighbor-pair configuration (optional)
        self.neighbor_loss_scale = float(getattr(cfg, 'neighbor_loss_scale', 0.0))
        self.neighbor_radius = float(getattr(cfg, 'neighbor_radius', 1.0))
        self.neighbor_max_neighbors = int(getattr(cfg, 'neighbor_max_neighbors', 0) or 0)
        self.neighbor_pair_batch_size = int(getattr(cfg, 'neighbor_pair_batch_size', max(1, cfg.batch_size // 2)))
        
    def setup(self, stage=None):
        start_time = time.time()

        if self.cfg.data.data_files:
            # If neighbor loss is enabled, ensure coords are available
            return_coords = bool(self.neighbor_loss_scale > 0)
            full_dataset = PointCloudDataset(
                root=self.cfg.data.data_path,
                data_files=self.cfg.data.data_files,
                radius=self.cfg.data.radius,
                sample_type=self.cfg.data.sample_type,
                overlap_fraction=self.cfg.data.overlap_fraction,
                n_samples=self.cfg.data.n_samples,
                num_points=self.cfg.data.num_points,
                return_coords=return_coords)   
        else:
            raise ValueError("No dataset under data_files files provided")
        
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        if self.max_samples>0:
            self.train_dataset = torch.utils.data.Subset(self.train_dataset, range(self.max_samples))
            self.val_dataset = torch.utils.data.Subset(self.val_dataset, range(self.max_samples))

        # Optional: build neighbor pair dataset on the train split
        self.train_pair_dataset = None
        if self.neighbor_loss_scale > 0:
            max_nbrs = self.neighbor_max_neighbors if self.neighbor_max_neighbors > 0 else None
            self.train_pair_dataset = NeighborPairDataset(
                self.train_dataset,
                radius=self.neighbor_radius,
                max_neighbors=max_nbrs,
                directed=True,
            )

        elapsed_time = time.time() - start_time
        logger.print(f"Train dataset size: {len(self.train_dataset)}")
        logger.print(f"Val dataset size: {len(self.val_dataset)}")
        logger.print(f"Dataloader took {elapsed_time:.4f} seconds")

    def train_dataloader(self):
        print(f"Using {self.num_workers} workers for train dataloader")
        main = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
        )
        if self.train_pair_dataset is None:
            return main
        pairs = DataLoader(
            self.train_pair_dataset,
            batch_size=self.neighbor_pair_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
        )
        return [main, pairs]
    
    def val_dataloader(self):
        print(f"Using {self.num_workers} workers for val dataloader")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
    
