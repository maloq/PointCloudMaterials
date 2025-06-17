import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset, random_split
from src.data_utils.data_load import PointCloudClsDataset
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
        
    def setup(self, stage=None):
        start_time = time.time()

        common_params = {
            "root": self.cfg.data.data_path,
            "return_coords": False,
            "radius": self.cfg.data.radius,
            "sample_type": self.cfg.data.sample_type,
            "overlap_fraction": self.cfg.data.overlap_fraction,
            "n_samples": self.cfg.data.n_samples,
            "num_points": self.cfg.data.num_points,
        }
        
        dataset_configs = [
            ('liquid_files', 0),
            ('crystal_files', 1),
        ]

        datasets = []
        for files_attr, label in dataset_configs:
            if data_files := getattr(self.cfg.data, files_attr, None):
                datasets.append(PointCloudClsDataset(
                    data_files=data_files,
                    label=label,
                    **common_params
                ))
            else:
                print(f"No {files_attr.split('_')[0]} dataset")

        if not datasets:
            raise ValueError("No datasets found. Please provide at least one of liquid or crystal files.")
            
        full_dataset = ConcatDataset(datasets)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        # self.train_dataset = torch.utils.data.Subset(self.train_dataset, range(2))
        # self.val_dataset = torch.utils.data.Subset(self.val_dataset, range(2))

        elapsed_time = time.time() - start_time
        logger.print(f"Train dataset size: {len(self.train_dataset)}")
        logger.print(f"Val dataset size: {len(self.val_dataset)}")
        logger.print(f"Dataloader took {elapsed_time:.4f} seconds")

    def train_dataloader(self):
        print(f"Using {self.num_workers} workers for train dataloader")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                        num_workers=self.num_workers, shuffle=True, drop_last=True, pin_memory=True, persistent_workers=True)
    
    def val_dataloader(self):
        print(f"Using {self.num_workers} workers for val dataloader")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
    