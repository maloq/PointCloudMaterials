import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional, List, Union
import numpy as np

class ModelNetFastDataset(Dataset):
    def __init__(self, 
                 root_dir: str, 
                 split: str = 'train', 
                 classes: Optional[List[str]] = None,
                 n_points: int = 2048):
        """
        Args:
            root_dir: Path to directory containing .pt files (e.g., datasets/ModelNet40_fast).
            split: 'train' or 'test'.
            classes: List of class names to include. If None, include all found.
            n_points: Number of points to return per sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.n_points = n_points
        
        self.points = []
        self.labels = []
        self.class_names = []
        
        # Find available files
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory {root_dir} does not exist. Did you run the conversion script?")
            
        all_files = [f for f in os.listdir(root_dir) if f.endswith(f"_{split}.pt")]
        
        # Filter by classes if specified
        if classes is not None:
            # Create a set for faster lookup
            classes_set = set(classes)
            # Filter files: filename is {class_name}_{split}.pt
            # We need to extract class_name. 
            # Assumption: class_name does not contain '_train' or '_test' suffix except at the end.
            # Actually, split is known.
            suffix = f"_{split}.pt"
            filtered_files = []
            for f in all_files:
                if f.endswith(suffix):
                    c_name = f[:-len(suffix)]
                    if c_name in classes_set:
                        filtered_files.append(f)
            all_files = filtered_files
            
        if not all_files:
            print(f"Warning: No files found for split={split} in {root_dir}")
            
        # Sort for deterministic order
        all_files.sort()
        
        # Load data
        # We need a consistent class-to-idx mapping. 
        # Ideally, this should be consistent with the original loader.
        # We can infer it from the sorted list of all unique classes in the directory, 
        # or pass it in. For now, let's build it from the files we found (or all files if classes is None).
        
        # To ensure consistency across train/test, we should probably scan all classes first.
        # But if we provide `classes` argument, we can use that.
        # If `classes` is None, we might get different mappings if train and test have different classes (unlikely for ModelNet).
        
        # Let's just use the sorted list of classes found in the current split for now, 
        # or better, scan for all classes in the dir to build the map.
        
        # Scan for all classes to build global map
        all_possible_files = os.listdir(root_dir)
        all_class_names = set()
        for f in all_possible_files:
            if f.endswith("_train.pt"):
                all_class_names.add(f[:-9])
            elif f.endswith("_test.pt"):
                all_class_names.add(f[:-8])
        
        self.class_to_idx = {cls: i for i, cls in enumerate(sorted(list(all_class_names)))}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
        
        print(f"Loading {split} data from {len(all_files)} files...")
        
        for f in all_files:
            path = os.path.join(root_dir, f)
            data = torch.load(path)
            
            # data is dict: {'points': tensor, 'class_name': str, ...}
            pts = data['points'] # (N_samples, Total_Points, 3)
            c_name = data['class_name']
            label = self.class_to_idx[c_name]
            
            num_samples = pts.shape[0]
            
            self.points.append(pts)
            self.labels.append(torch.full((num_samples,), label, dtype=torch.long))
            self.class_names.extend([c_name] * num_samples)
            
        if self.points:
            self.points = torch.cat(self.points, dim=0) # (Total_Samples, Total_Points, 3)
            self.labels = torch.cat(self.labels, dim=0)
        else:
            self.points = torch.empty(0, n_points, 3)
            self.labels = torch.empty(0, dtype=torch.long)
            
        print(f"Loaded {len(self.points)} samples.")

    def __len__(self):
        return len(self.points)
    
    def __getitem__(self, idx):
        # Points are already loaded.
        # We might need to downsample if stored points > n_points
        # or upsample if stored points < n_points (unlikely if we converted with 2048)
        
        pts = self.points[idx] # (Total_Points, 3)
        label = self.labels[idx]
        class_name = self.class_names[idx]
        
        curr_n = pts.shape[0]
        if curr_n > self.n_points:
            # Random choice or take first N?
            # FPS is expensive to do on the fly. 
            # Random choice is fast.
            # Or just take the first N if we assume they are shuffled or FPS-ordered?
            # The conversion used FPS, so the first N points are a good FPS subset!
            # This is a key property of FPS.
            pts = pts[:self.n_points]
        elif curr_n < self.n_points:
            # Upsample with replacement
            choice = np.random.choice(curr_n, self.n_points, replace=True)
            pts = pts[choice]
            
        return pts, label, class_name

class ModelNetFastDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        # Use the fast directory. If not specified in cfg, assume default relative to root
        # But cfg.data.data_path might point to the old one.
        # We can add a new config option or just append '_fast'
        
        self.data_path = getattr(cfg.data, 'fast_data_path', 'datasets/ModelNet40_fast')
        self.classes = getattr(cfg.data, 'classes', None)
        if self.classes is not None:
            self.classes = list(self.classes)
            
        self.n_points = getattr(cfg.data, 'num_points', 1024)
        
    def setup(self, stage=None):
        if stage != 'test':
            self.train_dataset = ModelNetFastDataset(
                root_dir=self.data_path,
                split='train',
                classes=self.classes,
                n_points=self.n_points
            )
            
        self.val_dataset = ModelNetFastDataset(
            root_dir=self.data_path,
            split='test',
            classes=self.classes,
            n_points=self.n_points
        )
        self.test_dataset = self.val_dataset
        
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
