import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset, random_split
from src.data_utils.data_load import AtomicDataset


class PointCloudDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.training.batch_size
        self.num_workers = cfg.training.num_workers
        
    def setup(self, stage=None):
        
        liquid_dataset = AtomicDataset(
            root=self.cfg.data.data_path,
            data_files=self.cfg.data.liquid_files,
            radius=self.cfg.data.radius,
            sample_type=self.cfg.data.sample_type,
            sample_shape=self.cfg.data.sample_shape,
            cube_size=self.cfg.data.cube_size,
            overlap_fraction=self.cfg.data.overlap_fraction,
            n_samples=self.cfg.data.n_samples,
            point_size=self.cfg.data.point_size,
            label=0
        )
        
        crystal_dataset = AtomicDataset(
            root=self.cfg.data.data_path,
            data_files=self.cfg.data.crystal_files,
            radius=self.cfg.data.radius,
            sample_type=self.cfg.data.sample_type,
            sample_shape=self.cfg.data.sample_shape,
            cube_size=self.cfg.data.cube_size,
            overlap_fraction=self.cfg.data.overlap_fraction,
            n_samples=self.cfg.data.n_samples,
            point_size=self.cfg.data.point_size,
            label=1
        )
        
        # liquid_dataset = torch.utils.data.Subset(liquid_dataset, range(100))
        #crystal_dataset = torch.utils.data.Subset(crystal_dataset, range(100))

        full_dataset = ConcatDataset([liquid_dataset, crystal_dataset])
        train_size = int(0.5 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        # self.train_dataset = torch.utils.data.Subset(self.train_dataset, range(100))
        # self.val_dataset = torch.utils.data.Subset(self.val_dataset, range(100))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                        num_workers=self.num_workers, shuffle=True, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers)