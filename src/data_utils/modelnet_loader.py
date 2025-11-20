import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional, List, Union
from src.data_utils.prepare_data import read_off_file, drop_points_farthest

class ModelNetDataset(Dataset):
    def __init__(self, 
                 root_dir: str, 
                 metadata_file: str, 
                 split: str = 'train', 
                 classes: Optional[List[str]] = None,
                 n_points: int = 1024,
                 cache: bool = True,
                 preload: bool = True):
        """
        Args:
            root_dir: Path to ModelNet40 root directory.
            metadata_file: Path to metadata_modelnet40.csv.
            split: 'train' or 'test'.
            classes: List of class names to include. If None, include all.
            n_points: Number of points to sample per object.
            cache: Whether to cache loaded .off files as .npy.
            preload: Whether to preload all data into RAM.
        """
        self.root_dir = root_dir
        self.n_points = n_points
        self.cache = cache
        self.preload = preload
        
        # Load metadata
        df = pd.read_csv(metadata_file)
        
        # Filter by split
        df = df[df['split'] == split]
        
        # Filter by classes if specified
        if classes is not None:
            df = df[df['class'].isin(classes)]
            
        self.metadata = df.reset_index(drop=True)
        
        # Create class mapping
        self.class_to_idx = {cls: i for i, cls in enumerate(sorted(df['class'].unique()))}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}

        self.data = []
        if self.preload:
            print(f"Preloading {len(self.metadata)} {split} samples into RAM...")
            # Simple progress indication
            for idx in range(len(self.metadata)):
                self.data.append(self._load_sample(idx))
                if (idx + 1) % 1000 == 0:
                    print(f"Loaded {idx + 1}/{len(self.metadata)}")
            print("Preloading complete.")
        
    def _load_sample(self, idx):
        row = self.metadata.iloc[idx]
        rel_path = row['object_path']
        full_path = os.path.join(self.root_dir, rel_path)
        class_name = row['class']
        label = self.class_to_idx[class_name]
        
        # Load points
        try:
            points = read_off_file(full_path, verbose=False, cache=self.cache)
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            points = np.zeros((self.n_points, 3), dtype=np.float32)

        # Resample points
        points = drop_points_farthest(points, self.n_points)
        
        # Normalize to unit sphere
        points = points - np.mean(points, axis=0)
        dist = np.max(np.linalg.norm(points, axis=1))
        if dist > 0:
            points = points / dist
            
        points_tensor = torch.from_numpy(points).float()
        
        return points_tensor, label, class_name

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        if self.preload:
            return self.data[idx]
        return self._load_sample(idx)

class ModelNetDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.data_path = cfg.data.data_path
        self.metadata_file = cfg.data.metadata_file
        self.classes = getattr(cfg.data, 'classes', None)
        self.n_points = getattr(cfg.data, 'num_points', 1024)
        
    def setup(self, stage=None):
        self.train_dataset = ModelNetDataset(
            root_dir=self.data_path,
            metadata_file=self.metadata_file,
            split='train',
            classes=self.classes,
            n_points=self.n_points
        )
        self.val_dataset = ModelNetDataset(
            root_dir=self.data_path,
            metadata_file=self.metadata_file,
            split='test',
            classes=self.classes,
            n_points=self.n_points
        )
        self.test_dataset = self.val_dataset
        
        print(f"ModelNet Train size: {len(self.train_dataset)}")
        print(f"ModelNet Val/Test size: {len(self.val_dataset)}")
        if self.classes:
            print(f"Classes: {self.classes}")
        else:
            print(f"Classes: All {len(self.train_dataset.class_to_idx)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )
