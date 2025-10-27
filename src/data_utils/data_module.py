import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from src.data_utils.data_load import PointCloudDataset
import time
import logging
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict

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



# Module-level helpers shared by modules above
def _to_container(cfg) -> Any:
    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


def _resolve_synthetic_dataset_config(spec: Dict[str, Any]) -> DatasetConfig:
    if "config" in spec and spec["config"]:
        cfg_dict = _to_container(spec["config"])
        if not isinstance(cfg_dict, dict):
            raise ValueError("`data.synthetic.config` must be a mapping")
        return _dataset_config_from_dict(cfg_dict)

    preset = spec.get("preset", "baseline").lower()
    if preset == "baseline":
        cfg = deepcopy(BASELINE_PRESET)
    elif preset == "imbalanced":
        kwargs = spec.get("preset_kwargs", {}) or {}
        rare_phase = kwargs.get("rare_phase", "icosa")
        rare_fraction = float(kwargs.get("rare_fraction", 0.05))
        cfg = imbalanced_phase_preset(rare_phase=rare_phase, rare_fraction=rare_fraction)
    elif preset == "voronoi":
        cfg = voronoi_anisotropic_preset()
    else:
        raise ValueError(f"Unknown synthetic preset '{preset}'")
    return cfg


def _dataset_config_from_dict(cfg_dict: Dict[str, Any]) -> DatasetConfig:
    data = deepcopy(cfg_dict)

    grain_radius = data.get("grain_radius_dist")
    if grain_radius is not None:
        params = grain_radius.get("params") or {}
        data["grain_radius_dist"] = GrainRadiusDistSpec(kind=grain_radius["kind"], params=dict(params))

    noise_cfg = data.get("noise")
    if noise_cfg is not None:
        anisotropic = noise_cfg.get("anisotropic_scale")
        if isinstance(anisotropic, list):
            anisotropic = tuple(anisotropic)
        data["noise"] = NoiseSpec(
            jitter_sigma=noise_cfg["jitter_sigma"],
            anisotropic_scale=anisotropic,
            missing_rate=noise_cfg.get("missing_rate", 0.0),
            outlier_rate=noise_cfg.get("outlier_rate", 0.0),
            density_gradient=noise_cfg.get("density_gradient"),
        )

    sampler = data.get("env_center_sampler")
    if sampler is not None:
        data["env_center_sampler"] = EnvCenterSamplerSpec(
            name=sampler["name"],
            boundary_band_fraction=sampler.get("boundary_band_fraction"),
            oversample_factor=sampler.get("oversample_factor", 1.0),
        )

    splits_cfg = data.get("splits")
    if splits_cfg is not None:
        ratios = splits_cfg.get("ratios")
        if ratios is None:
            raise ValueError("`data.synthetic.config.splits` must include `ratios`")
        data["splits"] = SplitSpec(
            ratios=dict(ratios),
            holdout_unseen_rotations=splits_cfg.get("holdout_unseen_rotations", True),
            boundary_band_fraction=splits_cfg.get("boundary_band_fraction", 0.02),
        )

    return DatasetConfig(**data)
