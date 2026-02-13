import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


_FAST_FILE_RE = re.compile(r"^(?P<class_name>.+)_(?P<split>train|test)\.pt$")


def _dedupe_keep_order(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        name = str(value)
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _scan_fast_files(root_dir: Path) -> Tuple[Dict[str, Dict[str, Path]], List[str]]:
    per_split: Dict[str, Dict[str, Path]] = {"train": {}, "test": {}}
    discovered_classes = set()

    for file_path in root_dir.iterdir():
        if not file_path.is_file() or file_path.suffix != ".pt":
            continue
        match = _FAST_FILE_RE.match(file_path.name)
        if match is None:
            continue
        class_name = match.group("class_name")
        split = match.group("split")
        per_split[split][class_name] = file_path
        discovered_classes.add(class_name)

    return per_split, sorted(discovered_classes)

class ModelNetFastDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        classes: Optional[Sequence[str]] = None,
        n_points: int = 2048,
        strict_classes: bool = False,
    ):
        """
        Args:
            root_dir: Path to directory containing .pt files (e.g., datasets/ModelNet40_fast).
            split: 'train' or 'test'.
            classes: List of class names to include. If None, include all found.
            n_points: Number of points to return per sample.
            strict_classes: If True, raise when any requested class is missing.
        """
        self.root_dir = root_dir
        self.split = split
        self.n_points = n_points
        self.strict_classes = bool(strict_classes)

        self.points = []
        self.labels = []
        self.class_names = []

        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        root_path = Path(root_dir)
        if not root_path.exists():
            raise FileNotFoundError(
                f"Directory {root_dir} does not exist. Did you run the conversion script?"
            )

        split_to_files, discovered_classes = _scan_fast_files(root_path)
        if not discovered_classes:
            raise RuntimeError(
                f"No ModelNet fast files found under {root_dir}. "
                "Expected files named <class>_train.pt or <class>_test.pt."
            )

        self.requested_classes = (
            _dedupe_keep_order(classes) if classes is not None else None
        )
        self.missing_requested_classes: List[str] = []

        if self.requested_classes is None:
            selected_classes = discovered_classes
        else:
            discovered_set = set(discovered_classes)
            self.missing_requested_classes = [
                c for c in self.requested_classes if c not in discovered_set
            ]
            if self.missing_requested_classes:
                msg = (
                    "Requested classes were not found in the fast dataset root "
                    f"{root_dir}: {self.missing_requested_classes}"
                )
                if self.strict_classes:
                    raise ValueError(msg)
                print(f"Warning: {msg}")

            selected_classes = [
                c for c in self.requested_classes if c in discovered_set
            ]
            if not selected_classes:
                raise RuntimeError(
                    "No requested classes are available in the fast dataset root. "
                    f"Requested={self.requested_classes}, discovered={discovered_classes}"
                )

        self.class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}

        split_files = split_to_files[self.split]
        missing_in_split = [c for c in selected_classes if c not in split_files]
        if missing_in_split:
            msg = (
                f"Classes requested for split={self.split!r} are missing files in {root_dir}: "
                f"{missing_in_split}"
            )
            if self.strict_classes:
                raise ValueError(msg)
            print(f"Warning: {msg}")

        files_to_load = [
            (class_name, split_files[class_name])
            for class_name in selected_classes
            if class_name in split_files
        ]

        if not files_to_load:
            print(f"Warning: No files found for split={split} in {root_dir}")

        print(f"Loading {split} data from {len(files_to_load)} files...")

        for class_name, path in files_to_load:
            data = torch.load(path, map_location="cpu")
            if "points" not in data:
                raise KeyError(f"Missing 'points' key in {path}")

            pts = data["points"]
            if not torch.is_tensor(pts):
                pts = torch.as_tensor(pts, dtype=torch.float32)
            else:
                pts = pts.to(dtype=torch.float32)

            if pts.dim() != 3 or pts.shape[-1] != 3:
                raise ValueError(
                    f"Expected points tensor with shape (N, P, 3) in {path}, got {tuple(pts.shape)}"
                )

            file_class_name = str(data.get("class_name", class_name))
            if file_class_name != class_name:
                print(
                    f"Warning: class name mismatch in {path.name}: "
                    f"filename={class_name}, payload={file_class_name}. Using filename."
                )

            label = self.class_to_idx[class_name]
            num_samples = int(pts.shape[0])

            self.points.append(pts)
            self.labels.append(torch.full((num_samples,), label, dtype=torch.long))
            self.class_names.extend([class_name] * num_samples)

        if self.points:
            self.points = torch.cat(self.points, dim=0)
            self.labels = torch.cat(self.labels, dim=0)
        else:
            self.points = torch.empty((0, n_points, 3), dtype=torch.float32)
            self.labels = torch.empty(0, dtype=torch.long)

        print(f"Loaded {len(self.points)} samples.")

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        pts = self.points[idx]
        label = self.labels[idx]
        class_name = self.class_names[idx]

        curr_n = pts.shape[0]
        if curr_n > self.n_points:
            pts = pts[:self.n_points]
        elif curr_n < self.n_points:
            if curr_n == 0:
                pts = torch.zeros((self.n_points, 3), dtype=self.points.dtype)
            else:
                choice = torch.randint(curr_n, (self.n_points,), device=pts.device)
                pts = pts.index_select(0, choice)

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
        self.strict_classes = bool(getattr(cfg.data, "strict_classes", False))

        self.n_points = getattr(cfg.data, 'num_points', 1024)

    def setup(self, stage=None):
        if stage != 'test':
            self.train_dataset = ModelNetFastDataset(
                root_dir=self.data_path,
                split='train',
                classes=self.classes,
                n_points=self.n_points,
                strict_classes=self.strict_classes,
            )

        self.val_dataset = ModelNetFastDataset(
            root_dir=self.data_path,
            split='test',
            classes=self.classes,
            n_points=self.n_points,
            strict_classes=self.strict_classes,
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
