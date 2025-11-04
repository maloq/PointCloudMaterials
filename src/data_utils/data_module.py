import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from src.data_utils.data_load import PointCloudDataset, SyntheticPointCloudDataset
import time
import logging
from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf

from src.utils.logging_config import setup_logging
logger = setup_logging()


class RealPointCloudDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.max_samples = cfg.max_samples

    def setup(self, stage=None):
        start_time = time.time()
        self.train_dataset, self.val_dataset = self._setup_real_dataset()

        if self.max_samples > 0:
            max_train = min(self.max_samples, len(self.train_dataset))
            max_val = min(self.max_samples, len(self.val_dataset))
            self.train_dataset = torch.utils.data.Subset(self.train_dataset, range(max_train))
            self.val_dataset = torch.utils.data.Subset(self.val_dataset, range(max_val))

        elapsed_time = time.time() - start_time
        logger.print(f"Train dataset size: {len(self.train_dataset)}")
        logger.print(f"Val dataset size: {len(self.val_dataset)}")
        logger.print(f"Dataloader took {elapsed_time:.4f} seconds")

    def _setup_real_dataset(self):
        data_cfg = self.cfg.data
        data_files = getattr(data_cfg, "data_files", None)
        if not data_files:
            raise ValueError("No dataset under data_files files provided")

        file_list = _to_container(data_files)
        if isinstance(file_list, str):
            file_list = [file_list]
        full_dataset = PointCloudDataset(
            root=data_cfg.data_path,
            data_files=file_list,
            radius=data_cfg.radius,
            sample_type=data_cfg.sample_type,
            overlap_fraction=data_cfg.overlap_fraction,
            n_samples=data_cfg.n_samples,
            num_points=data_cfg.num_points,
        )

        train_ratio = getattr(data_cfg, "train_ratio", 0.8)
        train_size = int(train_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        if train_size <= 0 or val_size <= 0:
            raise ValueError("Dataset split resulted in empty train or val set")
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
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
            persistent_workers=True,
        )
        return main

    def val_dataloader(self):
        print(f"Using {self.num_workers} workers for val dataloader")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )


class SyntheticPointCloudDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.max_samples = cfg.max_samples

    def setup(self, stage=None):
        start_time = time.time()
        self.train_dataset, self.val_dataset = self._build_datasets()

        if self.max_samples > 0:
            max_train = min(self.max_samples, len(self.train_dataset))
            max_val = min(self.max_samples, len(self.val_dataset))
            self.train_dataset = torch.utils.data.Subset(self.train_dataset, range(max_train))
            self.val_dataset = torch.utils.data.Subset(self.val_dataset, range(max_val))

        elapsed_time = time.time() - start_time
        logger.print(f"Synth train dataset size: {len(self.train_dataset)}")
        logger.print(f"Synth val dataset size: {len(self.val_dataset)}")
        logger.print(f"Synth dataloader prep took {elapsed_time:.4f} seconds")

    def _build_datasets(self):
        data_cfg = self.cfg.data
        synth_cfg = getattr(data_cfg, "synthetic", None)
        synth_dict = _to_container(synth_cfg) if synth_cfg is not None else {}

        env_dirs = self._resolve_env_dirs(data_cfg, synth_dict)
        radius = self._get_param("radius", data_cfg, synth_dict, required=True)
        sample_type = self._get_param("sample_type", data_cfg, synth_dict, default="regular")
        overlap_fraction = self._get_param("overlap_fraction", data_cfg, synth_dict, default=0.0)
        n_samples = self._get_param("n_samples", data_cfg, synth_dict, default=0)
        num_points = self._get_param("num_points", data_cfg, synth_dict, required=True)
        drop_edge_samples = self._get_param("drop_edge_samples", data_cfg, synth_dict, default=True)
        pre_normalize = self._get_param("pre_normalize", data_cfg, synth_dict, default=True)
        normalize = self._get_param("normalize", data_cfg, synth_dict, default=True)
        dataset_max = self._get_param("dataset_max_samples", data_cfg, synth_dict, default=None)

        dataset = SyntheticPointCloudDataset(
            env_dirs=env_dirs,
            radius=float(radius),
            sample_type=str(sample_type),
            overlap_fraction=float(overlap_fraction),
            n_samples=int(n_samples),
            num_points=int(num_points),
            drop_edge_samples=bool(drop_edge_samples),
            pre_normalize=bool(pre_normalize),
            normalize=bool(normalize),
            max_samples=int(dataset_max) if dataset_max is not None else None,
        )

        train_ratio = float(self._get_param("train_ratio", data_cfg, synth_dict, default=0.8))
        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        if train_size <= 0 or val_size <= 0:
            raise ValueError("Synthetic dataset split resulted in empty train or val set")
        return random_split(dataset, [train_size, val_size])

    def _resolve_env_dirs(self, data_cfg, synth_dict):
        env_dirs = []
        if isinstance(synth_dict, dict):
            if "data_dirs" in synth_dict and synth_dict["data_dirs"]:
                env_dirs = synth_dict["data_dirs"]
            elif synth_dict.get("data_dir"):
                env_dirs = [synth_dict["data_dir"]]
            elif synth_dict.get("root_dir"):
                root = Path(synth_dict["root_dir"])
                if not root.exists():
                    raise FileNotFoundError(f"Synthetic root_dir {root} does not exist")
                candidates = sorted([p for p in root.iterdir() if p.is_dir()])
                num_env = synth_dict.get("num_environments")
                if num_env is not None:
                    candidates = candidates[:int(num_env)]
                env_dirs = candidates
        if not env_dirs and hasattr(data_cfg, "data_path"):
            env_dirs = [getattr(data_cfg, "data_path")]
        if not env_dirs:
            raise ValueError("No synthetic environment directories configured")
        resolved = []
        for path in env_dirs:
            if isinstance(path, (Path, str)):
                resolved.append(Path(path))
            else:
                raise TypeError(f"Unsupported path type {type(path)} in synthetic env_dirs")
        return resolved

    @staticmethod
    def _get_param(name, data_cfg, synth_dict, *, required=False, default=None):
        value = getattr(data_cfg, name, None)
        if value is None and isinstance(synth_dict, dict):
            value = synth_dict.get(name, None)
        if value is None:
            if default is not None:
                return default
            if required:
                raise ValueError(f"Synthetic data configuration must provide '{name}'")
        return value

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )


class PointCloudDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        kind = getattr(cfg.data, "kind", "real")
        if kind == "synthetic":
            self.impl = SyntheticPointCloudDataModule(cfg)
        else:
            self.impl = RealPointCloudDataModule(cfg)

    def setup(self, stage=None):
        return self.impl.setup(stage)

    def train_dataloader(self):
        return self.impl.train_dataloader()

    def val_dataloader(self):
        return self.impl.val_dataloader()


# Module-level helper
def _to_container(cfg):
    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg

SynthPointCloudDataModule = SyntheticPointCloudDataModule
