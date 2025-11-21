import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional, List, Union
from src.data_utils.prepare_data import read_off_file, farthest_point_sample, drop_points_farthest
from concurrent.futures import as_completed


class ModelNetDataset(Dataset):
    def __init__(self, 
                 root_dir: str, 
                 metadata_file: str, 
                 split: str = 'train', 
                 classes: Optional[List[str]] = None,
                 n_points: int = 512,
                 cache: bool = True,
                 preload: bool = True,
                 sampling_method: str = "fps"):

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

        self.sampling_method = sampling_method
        
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

        self.data = [None] * len(self.metadata)
        
        if self.preload:

            print(f"Preloading {len(self.metadata)} {split} samples into RAM using {os.cpu_count()} workers...")
            from concurrent.futures import ThreadPoolExecutor
            try:
                from tqdm import tqdm
            except ImportError:
                def tqdm(x, **kwargs): return x

            with ThreadPoolExecutor() as executor:
                # Submit all tasks
                futures = {executor.submit(self._load_sample, i): i for i in range(len(self.metadata))}
                
                # Collect results as they complete
                for future in tqdm(as_completed(futures), total=len(self.metadata), desc=f"Loading {split}"):
                    idx = futures[future]
                    try:
                        self.data[idx] = future.result()
                    except Exception as e:
                        print(f"Failed to load sample {idx}: {e}")
                        # Fallback or keep None (will error on access if not handled)
                        # Re-try synchronously or just fill with dummy
                        self.data[idx] = (torch.zeros((self.n_points, 3)), 0, "error")
            
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
        if len(points) >= self.n_points:
            if self.sampling_method == "fps":
                # Downsample using FPS to preserve shape
                points = farthest_point_sample(points, self.n_points)
            elif self.sampling_method == "drop_farthest":
                # Drop farthest points (keeps center, loses wings)
                points = drop_points_farthest(points, self.n_points)
            else:
                raise ValueError(f"Unknown sampling method: {self.sampling_method}")
        else:
            # Upsample using random duplication
            # (Simple random choice with replacement)
            indices = np.random.choice(len(points), self.n_points, replace=True)
            points = points[indices]
        
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
        if self.classes is not None:
            self.classes = list(self.classes)

        self.n_points = getattr(cfg.data, 'num_points', 1024)

        self.sampling_method = getattr(cfg.data, 'sampling_method', 'fps')
        
    def setup(self, stage=None):
        # Only create train dataset if not in test stage
        if stage != 'test':
            self.train_dataset = ModelNetDataset(
                root_dir=self.data_path,
                metadata_file=self.metadata_file,
                split='train',
                classes=self.classes,
                n_points=self.n_points,
                sampling_method=self.sampling_method
            )
        
        # Always create val/test dataset
        self.val_dataset = ModelNetDataset(
            root_dir=self.data_path,
            metadata_file=self.metadata_file,
            split='test',
            classes=self.classes,
            n_points=self.n_points,
            sampling_method=self.sampling_method
        )

        self.test_dataset = self.val_dataset
        
        # Print dataset sizes
        if stage != 'test':
            print(f"ModelNet Train size: {len(self.train_dataset)}")
        print(f"ModelNet Val/Test size: {len(self.val_dataset)}")
        if self.classes:
            print(f"Classes: {self.classes}")
        else:
            # Get class count from available dataset
            dataset = self.val_dataset if stage == 'test' else self.train_dataset
            print(f"Classes: All {len(dataset.class_to_idx)}")

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
