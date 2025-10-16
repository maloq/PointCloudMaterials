import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from src.data_utils.data_load import PointCloudDataset
from src.data_utils.neighbor_pairs import NeighborPairDataset
import time
import logging
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict

from omegaconf import DictConfig, ListConfig, OmegaConf

from src.data_utils.synthetic import (
    BASELINE_PRESET,
    SyntheticPointCloudDataset,
    export_scene,
    generate_scene,
    load_scene,
    make_splits,
    imbalanced_phase_preset,
    voronoi_anisotropic_preset,
)
from src.data_utils.synthetic.config import (
    DatasetConfig,
    EnvCenterSamplerSpec,
    GrainRadiusDistSpec,
    NoiseSpec,
    SplitSpec,
)
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

        data_kind = getattr(self.cfg.data, "kind", "off")
        if data_kind == "synthetic":
            self.train_dataset, self.val_dataset = self._setup_synthetic_dataset()
        else:
            self.train_dataset, self.val_dataset = self._setup_off_dataset()

        if self.max_samples > 0:
            max_train = min(self.max_samples, len(self.train_dataset))
            max_val = min(self.max_samples, len(self.val_dataset))
            self.train_dataset = torch.utils.data.Subset(self.train_dataset, range(max_train))
            self.val_dataset = torch.utils.data.Subset(self.val_dataset, range(max_val))

        # Optional: build neighbor pair dataset on the train split
        self.train_pair_dataset = None
        if self.neighbor_loss_scale > 0 and len(self.train_dataset) > 0:
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

    def _setup_off_dataset(self):
        data_cfg = self.cfg.data
        data_files = getattr(data_cfg, "data_files", None)
        if not data_files:
            raise ValueError("No dataset under data_files files provided")

        file_list = self._to_container(data_files)
        if isinstance(file_list, str):
            file_list = [file_list]
        return_coords = bool(self.neighbor_loss_scale > 0)
        full_dataset = PointCloudDataset(
            root=data_cfg.data_path,
            data_files=file_list,
            radius=data_cfg.radius,
            sample_type=data_cfg.sample_type,
            overlap_fraction=data_cfg.overlap_fraction,
            n_samples=data_cfg.n_samples,
            num_points=data_cfg.num_points,
            return_coords=return_coords,
        )

        train_ratio = getattr(data_cfg, "train_ratio", 0.8)
        train_size = int(train_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        if train_size <= 0 or val_size <= 0:
            raise ValueError("Dataset split resulted in empty train or val set")
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
        return train_ds, val_ds

    def _setup_synthetic_dataset(self):
        data_cfg = self.cfg.data
        synth_cfg = getattr(data_cfg, "synthetic", None)
        if synth_cfg is None:
            raise ValueError("Synthetic dataset requested but `data.synthetic` config is missing")

        spec = self._to_container(synth_cfg)
        if not isinstance(spec, dict):
            raise ValueError("`data.synthetic` must be a mapping")

        dataset_cfg = self._resolve_synthetic_dataset_config(spec)
        dataset_cfg = deepcopy(dataset_cfg)

        # Keep Hydra num_points authoritative for decoder expectations
        dataset_cfg.M = int(getattr(data_cfg, "num_points", dataset_cfg.M))

        seed_override = spec.get("seed")
        if seed_override is not None:
            dataset_cfg.seed = int(seed_override)

        dataset_cfg.validate()

        num_env = int(spec.get("num_environments", 0))
        if num_env <= 0:
            raise ValueError("`data.synthetic.num_environments` must be > 0")

        cache_path = spec.get("cache_path")
        regenerate = bool(spec.get("regenerate", False))
        scene = None
        if cache_path:
            cache_path = Path(cache_path)
            if cache_path.exists() and not regenerate:
                logger.print(f"Loading synthetic scene from {cache_path}")
                scene = load_scene(cache_path)
            else:
                if regenerate and cache_path.exists():
                    logger.print(f"Regenerating synthetic scene at {cache_path}")
                cache_path.parent.mkdir(parents=True, exist_ok=True)

        if scene is None:
            logger.print(f"Generating synthetic scene with {num_env} environments")
            scene = generate_scene(dataset_cfg, num_env)
            if cache_path:
                export_scene(scene, cache_path)
        else:
            if scene.points.shape[0] < num_env:
                logger.print(
                    f"Loaded scene has {scene.points.shape[0]} environments (< requested {num_env}); proceeding with available data"
                )

        scene_cfg = deepcopy(scene.config)
        points_per_env = scene.points.shape[1]
        expected_points = int(getattr(data_cfg, "num_points", points_per_env))
        if points_per_env != expected_points:
            raise ValueError(
                "Synthetic scene point count does not match `data.num_points` ("
                f"scene has {points_per_env}, config requests {expected_points})."
            )
        scene_cfg.M = points_per_env

        splits = make_splits(scene, scene_cfg)
        train_split = spec.get("train_split", "train")
        val_split = spec.get("val_split", "val")

        if train_split not in splits:
            raise ValueError(f"Synthetic scene splits do not contain '{train_split}'")
        if val_split not in splits:
            raise ValueError(f"Synthetic scene splits do not contain '{val_split}'")

        dataset_kwargs = {
            "return_orientation": bool(spec.get("return_orientation", True)),
            "return_meta": bool(spec.get("return_meta", False)),
        }
        device_str = spec.get("device")
        if device_str:
            dataset_kwargs["device"] = torch.device(device_str)

        train_dataset = SyntheticPointCloudDataset(splits[train_split], **dataset_kwargs)
        val_dataset = SyntheticPointCloudDataset(splits[val_split], **dataset_kwargs)
        return train_dataset, val_dataset

    @staticmethod
    def _to_container(cfg) -> Any:
        if isinstance(cfg, (DictConfig, ListConfig)):
            return OmegaConf.to_container(cfg, resolve=True)
        return cfg

    def _resolve_synthetic_dataset_config(self, spec: Dict[str, Any]) -> DatasetConfig:
        if "config" in spec and spec["config"]:
            cfg_dict = self._to_container(spec["config"])
            if not isinstance(cfg_dict, dict):
                raise ValueError("`data.synthetic.config` must be a mapping")
            return self._dataset_config_from_dict(cfg_dict)

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

    def _dataset_config_from_dict(self, cfg_dict: Dict[str, Any]) -> DatasetConfig:
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
    
