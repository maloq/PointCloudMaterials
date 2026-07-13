import time
from collections import defaultdict
from pathlib import Path

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data_utils.data_load import SyntheticPointCloudDataset
from src.data_utils.data_modules.common import (
    _resolve_split_seed,
    _seeded_random_split,
    _to_container,
    logger,
)


class SyntheticPointCloudDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.max_samples = cfg.max_samples
        self.split_seed = _resolve_split_seed(cfg)
        self._phase_info_logged = False
        self._datasets_initialized = False

    def setup(self, stage=None):
        start_time = time.time()
        initialized_now = False
        if not self._datasets_initialized:
            self.train_dataset, self.val_dataset = self._build_datasets()
            # Test dataset is same as val for metrics.
            self.test_dataset = self.val_dataset

            if self.max_samples > 0:
                max_train = min(self.max_samples, len(self.train_dataset))
                max_val = min(self.max_samples, len(self.val_dataset))
                self.train_dataset = torch.utils.data.Subset(self.train_dataset, range(max_train))
                self.val_dataset = torch.utils.data.Subset(self.val_dataset, range(max_val))
                self.train_indices = self.train_indices[:max_train]
                self.test_dataset = self.val_dataset
            self._datasets_initialized = True
            initialized_now = True

        if (stage is None or stage == 'fit') and not self._phase_info_logged:
            self._log_class_mapping()

        elapsed_time = time.time() - start_time
        if not initialized_now:
            logger.print(
                f"Reusing existing synthetic dataset split for stage={stage!r} "
                f"(split_seed={self.split_seed})."
            )
        logger.print(f"Synth train dataset size: {len(self.train_dataset)}")
        logger.print(f"Synth val dataset size: {len(self.val_dataset)}")
        logger.print(f"Synth test dataset size: {len(self.test_dataset)}")
        logger.print(f"Synth dataloader prep took {elapsed_time:.4f} seconds")

    def _build_datasets(self):
        data_cfg = self.cfg.data
        synth_dict = _to_container(data_cfg.synthetic)

        aug_cfg = OmegaConf.select(self.cfg, "augmentation", default=None)
        if aug_cfg is None:
            rotation_scale = 0.0
            noise_scale = 0.0
            jitter_scale = 0.0
            scaling_range = 0.0
            track_augmentation = False
        else:
            rotation_scale = aug_cfg.rotation_scale
            noise_scale = aug_cfg.noise_scale
            jitter_scale = aug_cfg.jitter_scale
            scaling_range = aug_cfg.scaling_range
            track_augmentation = aug_cfg.track_augmentation

        auto_cutoff_raw = OmegaConf.select(data_cfg, "auto_cutoff", default=None)
        auto_cutoff_cfg = _to_container(auto_cutoff_raw)
        model_type = OmegaConf.select(self.cfg, "model_type", default=None)
        disable_dataset_aug_for_ssl = bool(
            OmegaConf.select(self.cfg, "disable_dataset_augmentation_for_ssl", default=True)
        )
        uses_ssl_views = (
            model_type in {"vicreg", "visreg"}
            or bool(OmegaConf.select(self.cfg, "vicreg_enabled", default=False))
        )
        if uses_ssl_views and disable_dataset_aug_for_ssl:
            has_dataset_aug = any(
                float(v) != 0.0 for v in (rotation_scale, noise_scale, jitter_scale, scaling_range)
            )
            if has_dataset_aug:
                logger.print(
                    "Disabling dataset-level geometric augmentation for contrastive SSL; "
                    "view augmentation is applied in the contrastive loss."
                )
            rotation_scale = 0.0
            noise_scale = 0.0
            jitter_scale = 0.0
            scaling_range = 0.0
            track_augmentation = False

        env_dirs = self._resolve_env_dirs(synth_dict)
        classes = synth_dict["classes"]

        dataset = SyntheticPointCloudDataset(
            env_dirs=env_dirs,
            radius=float(data_cfg.radius),
            sample_type=data_cfg.sample_type,
            overlap_fraction=float(data_cfg.overlap_fraction),
            n_samples=int(data_cfg.n_samples),
            num_points=int(data_cfg.num_points),
            drop_edge_samples=bool(data_cfg.drop_edge_samples),
            pre_normalize=bool(data_cfg.pre_normalize),
            normalize=bool(data_cfg.normalize),
            max_samples=(
                int(data_cfg.dataset_max_samples)
                if data_cfg.dataset_max_samples is not None
                else None
            ),
            rotation_scale=rotation_scale,
            noise_scale=noise_scale,
            jitter_scale=jitter_scale,
            scaling_range=scaling_range,
            track_augmentation=track_augmentation,
            allowed_classes=classes,
            auto_cutoff_config=auto_cutoff_cfg,
        )

        train_ratio = float(data_cfg.train_ratio)
        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        if train_size <= 0 or val_size <= 0:
            raise ValueError("Synthetic dataset split resulted in empty train or val set")
        train_dataset, val_dataset = _seeded_random_split(
            dataset,
            [train_size, val_size],
            seed=self.split_seed,
        )
        self.dataset = dataset
        self.train_indices = list(train_dataset.indices)
        return train_dataset, val_dataset

    def _resolve_env_dirs(self, synth_dict):
        root = Path(synth_dict["root_dir"])
        if not root.is_dir():
            raise FileNotFoundError(f"Synthetic root_dir is missing or not a directory: {root}")
        env_dirs = sorted(path for path in root.iterdir() if path.is_dir())
        if not env_dirs:
            raise ValueError(f"Synthetic root_dir contains no environment directories: {root}")
        return env_dirs

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        use_persistent = self.num_workers > 0
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=use_persistent,
        )

    def test_dataloader(self):
        use_persistent = self.num_workers > 0
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=use_persistent,
        )

    @staticmethod
    def _anisotropy_from_points(points: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if points.dim() != 3 or points.shape[-1] != 3:
            raise ValueError(f"Expected points of shape (B,N,3), got {tuple(points.shape)}")
        centered = points - points.mean(dim=1, keepdim=True)
        cov = (centered.transpose(1, 2) @ centered) / float(centered.shape[1])
        eigvals = torch.linalg.eigvalsh(cov.to(dtype=torch.float32))
        eigvals, _ = torch.sort(eigvals, dim=-1, descending=True)
        lam1, lam2, lam3 = eigvals.unbind(dim=-1)
        return (lam1 - lam3) / (lam1 + lam2 + lam3 + float(eps))

    def _log_train_anisotropy_by_class(
        self,
        dataset: SyntheticPointCloudDataset,
        subset_indices: list[int],
    ) -> None:
        class_name_map = dataset.class_names
        sums = defaultdict(float)
        counts = defaultdict(int)

        class_ids = torch.as_tensor(dataset._class_ids, dtype=torch.long)
        chunk_size = 512
        for start in range(0, len(subset_indices), chunk_size):
            idx_list = subset_indices[start:start + chunk_size]
            idx = torch.as_tensor(idx_list, dtype=torch.long)
            pts = torch.stack([dataset.samples[index] for index in idx_list], dim=0)
            anis = self._anisotropy_from_points(pts).to(dtype=torch.float64).cpu()
            cls = class_ids.index_select(0, idx)
            for cid in torch.unique(cls):
                mask = cls == cid
                cid_int = int(cid.item())
                sums[cid_int] += float(anis[mask].sum().item())
                counts[cid_int] += int(mask.sum().item())

        logger.print("Train anisotropy by class (mean over train split):")
        for cid in sorted(counts):
            name = class_name_map[cid]
            mean_anis = sums[cid] / counts[cid]
            logger.print(f"  {name}: {mean_anis:.6f} (n={counts[cid]})")

    def _log_class_mapping(self):
        """Log the class name to ID mapping for the dataset."""
        base_dataset = self.dataset
        if not base_dataset._class_to_idx:
            raise RuntimeError("SyntheticPointCloudDataset produced no class labels.")

        logger.print(f"Dataset domain: {base_dataset.domain}")
        logger.print("Class labels:")
        for class_name, class_idx in sorted(base_dataset._class_to_idx.items(), key=lambda item: item[1]):
            logger.print(f"  Class {class_idx}: {class_name}")

        self._log_train_anisotropy_by_class(base_dataset, self.train_indices)
        self._phase_info_logged = True


SynthPointCloudDataModule = SyntheticPointCloudDataModule
