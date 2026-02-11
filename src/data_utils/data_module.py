import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from collections import defaultdict
from src.data_utils.data_load import (
    PointCloudDataset,
    SyntheticPointCloudDataset,
    CenteredModelNetDataset,
)
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
        # Test dataset is same as val for metrics
        self.test_dataset = self.val_dataset

        if self.max_samples > 0:
            max_train = min(self.max_samples, len(self.train_dataset))
            max_val = min(self.max_samples, len(self.val_dataset))
            self.train_dataset = torch.utils.data.Subset(self.train_dataset, range(max_train))
            self.val_dataset = torch.utils.data.Subset(self.val_dataset, range(max_val))
            self.test_dataset = self.val_dataset

        elapsed_time = time.time() - start_time
        logger.print(f"Train dataset size: {len(self.train_dataset)}")
        logger.print(f"Val dataset size: {len(self.val_dataset)}")
        logger.print(f"Test dataset size: {len(self.test_dataset)}")
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
            persistent_workers=self.num_workers > 0,
        )
        return main

    def val_dataloader(self):
        print(f"Using {self.num_workers} workers for val dataloader")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        print(f"Using {self.num_workers} workers for test dataloader")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


class SyntheticPointCloudDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.max_samples = cfg.max_samples
        self._phase_info_logged = False

    def setup(self, stage=None):
        start_time = time.time()
        self.train_dataset, self.val_dataset = self._build_datasets()
        # Test dataset is same as val for metrics
        self.test_dataset = self.val_dataset

        if self.max_samples > 0:
            max_train = min(self.max_samples, len(self.train_dataset))
            max_val = min(self.max_samples, len(self.val_dataset))
            self.train_dataset = torch.utils.data.Subset(self.train_dataset, range(max_train))
            self.val_dataset = torch.utils.data.Subset(self.val_dataset, range(max_val))
            self.test_dataset = self.val_dataset

        if (stage is None or stage == 'fit') and not self._phase_info_logged:
            self._log_class_mapping()

        elapsed_time = time.time() - start_time
        logger.print(f"Synth train dataset size: {len(self.train_dataset)}")
        logger.print(f"Synth val dataset size: {len(self.val_dataset)}")
        logger.print(f"Synth test dataset size: {len(self.test_dataset)}")
        logger.print(f"Synth dataloader prep took {elapsed_time:.4f} seconds")

    def _build_datasets(self):
        data_cfg = self.cfg.data
        synth_cfg = getattr(data_cfg, "synthetic", None)
        synth_dict = _to_container(synth_cfg) if synth_cfg is not None else {}

        aug_cfg = getattr(data_cfg, "augmentation", None)
        if aug_cfg is None:
            aug_cfg = getattr(self.cfg, "augmentation", None)
        rotation_scale = getattr(aug_cfg, "rotation_scale", 0.0) if aug_cfg else 0.0
        noise_scale = getattr(aug_cfg, "noise_scale", 0.0) if aug_cfg else 0.0
        jitter_scale = getattr(aug_cfg, "jitter_scale", 0.0) if aug_cfg else 0.0
        scaling_range = getattr(aug_cfg, "scaling_range", 0.0) if aug_cfg else 0.0
        track_augmentation = getattr(aug_cfg, "track_augmentation", False) if aug_cfg else False

        dataset_type = self._get_param("dataset_type", data_cfg, synth_dict, default="synthetic_env")
        if dataset_type == "modelnet_objects":
            root_dir = self._get_param("data_path", data_cfg, synth_dict, required=True)
            num_points = self._get_param("num_points", data_cfg, synth_dict, required=True)
            pre_normalize = self._get_param("pre_normalize", data_cfg, synth_dict, default=True)
            normalize = self._get_param("normalize", data_cfg, synth_dict, default=False)
            dataset_max = self._get_param("dataset_max_samples", data_cfg, synth_dict, default=None)
            modelnet_split = self._get_param("modelnet_split", data_cfg, synth_dict, default="train")
            add_center_point = self._get_param("add_center_point", data_cfg, synth_dict, default=True)
            fps_cache = self._get_param("fps_cache", data_cfg, synth_dict, default=True)
            fps_cache_dir = self._get_param("fps_cache_dir", data_cfg, synth_dict, default=None)
            fps_use_gpu = self._get_param("fps_use_gpu", data_cfg, synth_dict, default=True)
            classes = self._get_param("classes", data_cfg, synth_dict, default=None)
            if classes is not None:
                classes = _to_container(classes)
                if isinstance(classes, str):
                    classes = [classes]

            dataset = CenteredModelNetDataset(
                root_dir=root_dir,
                split=str(modelnet_split),
                classes=classes,
                num_points=int(num_points),
                add_center_point=bool(add_center_point),
                pre_normalize=bool(pre_normalize),
                normalize=bool(normalize),
                max_samples=int(dataset_max) if dataset_max is not None else None,
                fps_cache=bool(fps_cache),
                fps_cache_dir=fps_cache_dir,
                fps_use_gpu=bool(fps_use_gpu),
                rotation_scale=rotation_scale,
                noise_scale=noise_scale,
                jitter_scale=jitter_scale,
                scaling_range=scaling_range,
                track_augmentation=track_augmentation,
            )

            train_ratio = float(self._get_param("train_ratio", data_cfg, synth_dict, default=0.8))
            train_size = int(train_ratio * len(dataset))
            val_size = len(dataset) - train_size
            if train_size <= 0 or val_size <= 0:
                raise ValueError("Synthetic dataset split resulted in empty train or val set")
            return random_split(dataset, [train_size, val_size])

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
        classes = self._get_param("classes", data_cfg, synth_dict, default=None)
        if classes is not None:
             classes = _to_container(classes)
             if isinstance(classes, str):
                 classes = [classes]

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
            rotation_scale=rotation_scale,
            noise_scale=noise_scale,
            jitter_scale=jitter_scale,
            scaling_range=scaling_range,
            track_augmentation=track_augmentation,
            allowed_classes=classes,
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

    def _resolve_synthetic_dataset_with_indices(self):
        dataset = getattr(self, 'train_dataset', None)
        indices = None
        visited = set()
        while dataset is not None and id(dataset) not in visited:
            visited.add(id(dataset))
            if isinstance(dataset, torch.utils.data.Subset):
                current = list(dataset.indices)
                if indices is None:
                    indices = current
                else:
                    indices = [current[i] for i in indices]
                dataset = dataset.dataset
                continue
            if isinstance(dataset, (SyntheticPointCloudDataset, CenteredModelNetDataset)):
                return dataset, indices
            next_dataset = getattr(dataset, 'dataset', None)
            if next_dataset is not None and next_dataset is not dataset:
                dataset = next_dataset
                continue
            base_dataset = getattr(dataset, 'base_dataset', None)
            if base_dataset is not None and base_dataset is not dataset:
                dataset = base_dataset
                continue
            break
        return None, None

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

    @staticmethod
    def _class_id_to_name_map(dataset) -> dict[int, str]:
        class_names = getattr(dataset, "class_names", None)
        if class_names is None:
            return {}
        return {int(k): str(v) for k, v in dict(class_names).items()}

    def _log_train_anisotropy_by_class(self, dataset, subset_indices: list[int] | None) -> None:
        class_ids_raw = getattr(dataset, "_class_ids", None)
        points_raw = getattr(dataset, "samples", None)
        if points_raw is None:
            points_raw = getattr(dataset, "points", None)
        class_name_map = self._class_id_to_name_map(dataset)

        sums = defaultdict(float)
        counts = defaultdict(int)

        if class_ids_raw is not None and points_raw is not None:
            class_ids = class_ids_raw if torch.is_tensor(class_ids_raw) else torch.as_tensor(class_ids_raw)
            class_ids = class_ids.to(dtype=torch.long).view(-1).cpu()
            total = int(class_ids.shape[0])
            if subset_indices is None:
                selected = list(range(total))
            else:
                selected = [int(i) for i in subset_indices if 0 <= int(i) < total]

            chunk_size = 512
            for start in range(0, len(selected), chunk_size):
                idx_list = selected[start:start + chunk_size]
                if not idx_list:
                    continue
                idx = torch.as_tensor(idx_list, dtype=torch.long)
                if torch.is_tensor(points_raw):
                    pts = points_raw.index_select(0, idx).to(dtype=torch.float32)
                else:
                    pts = torch.stack([points_raw[int(i)] for i in idx_list], dim=0).to(dtype=torch.float32)
                anis = self._anisotropy_from_points(pts).to(dtype=torch.float64).cpu()
                cls = class_ids.index_select(0, idx)
                valid = cls >= 0
                if not valid.any():
                    continue
                cls = cls[valid]
                anis = anis[valid]
                for cid in torch.unique(cls):
                    mask = cls == cid
                    cid_int = int(cid.item())
                    sums[cid_int] += float(anis[mask].sum().item())
                    counts[cid_int] += int(mask.sum().item())
        else:
            train_ds = getattr(self, "train_dataset", None)
            if train_ds is not None:
                for idx in range(len(train_ds)):
                    sample = train_ds[idx]
                    if not isinstance(sample, dict):
                        continue
                    cls = sample.get("class_id", None)
                    pts = sample.get("points", None)
                    if cls is None or pts is None:
                        continue
                    cls_val = int(cls.item()) if torch.is_tensor(cls) else int(cls)
                    if cls_val < 0:
                        continue
                    anis = float(self._anisotropy_from_points(pts.unsqueeze(0).to(dtype=torch.float32)).item())
                    sums[cls_val] += anis
                    counts[cls_val] += 1

        logger.print("Train anisotropy by class (mean over train split):")
        if not counts:
            logger.print("  unavailable (no class_id labels found)")
            return
        for cid in sorted(counts):
            name = class_name_map.get(cid, f"class_{cid}")
            mean_anis = sums[cid] / max(1, counts[cid])
            logger.print(f"  {name}: {mean_anis:.6f} (n={counts[cid]})")

    def _log_class_mapping(self):
        """Log the class name to ID mapping for the dataset."""
        base_dataset, subset_indices = self._resolve_synthetic_dataset_with_indices()
        if base_dataset is None or not hasattr(base_dataset, '_class_to_idx'):
            logger.print("Class mapping unavailable; dataset not initialized yet")
            return

        if not base_dataset._class_to_idx:
            logger.print("Dataset did not expose any class labels")
            return

        domain = getattr(base_dataset, 'domain', 'unknown')
        logger.print(f"Dataset domain: {domain}")
        logger.print("Class labels:")
        for class_name, class_idx in sorted(base_dataset._class_to_idx.items(), key=lambda item: item[1]):
            logger.print(f"  Class {class_idx}: {class_name}")

        self._log_train_anisotropy_by_class(base_dataset, subset_indices)
        self._phase_info_logged = True


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

    def test_dataloader(self):
        return self.impl.test_dataloader()


# Module-level helper
def _to_container(cfg):
    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg

SynthPointCloudDataModule = SyntheticPointCloudDataModule
