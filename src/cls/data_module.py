import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset, random_split
from src.data_utils.data_load import AtomicDataset


class PointCloudDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128, num_workers=10):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        liquid_dataset = AtomicDataset(
            root="/home/teshbek/Work/PhD/PointCloudMaterials/datasets/Al/inherent_configurations_off",
            data_files=["166ps.off"],
            cube_size=16,
            n_samples=10000,
            num_point=256,
            label=0
        )
        
        crystal_dataset = AtomicDataset(
            root="/home/teshbek/Work/PhD/PointCloudMaterials/datasets/Al/inherent_configurations_off",
            data_files=["240ps.off"],
            cube_size=16,
            n_samples=10000,
            num_point=256,
            label=1
        )
        
        full_dataset = ConcatDataset([liquid_dataset, crystal_dataset])
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                        num_workers=self.num_workers, shuffle=True, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers)