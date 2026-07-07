import time

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data_utils.data_modules.common import (
    _cfg_get,
    _resolve_split_seed,
    _to_container,
    logger,
)
from src.data_utils.data_modules.temporal_window import _temporal_identity_or_default_collate
from src.data_utils.data_load import PointCloudDataset
from src.data_utils.line_static_dataset import LineStaticPointCloudDataset


class LineStaticDataModule(pl.LightningDataModule):
    """DataModule for Line-JEPA over static .npy/.off point-cloud files."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.max_samples = int(cfg.max_samples)
        self.split_seed = _resolve_split_seed(cfg)
        self._datasets_initialized = False

    def setup(self, stage=None):
        start_time = time.time()
        initialized_now = False
        if not self._datasets_initialized:
            self.train_dataset, self.val_dataset = self._build_datasets()
            self.test_dataset = self.val_dataset

            if self.max_samples > 0:
                max_train = min(self.max_samples, len(self.train_dataset))
                max_val = min(self.max_samples, len(self.val_dataset))
                self.train_dataset = torch.utils.data.Subset(self.train_dataset, range(max_train))
                self.val_dataset = torch.utils.data.Subset(self.val_dataset, range(max_val))
                self.test_dataset = self.val_dataset
            self._datasets_initialized = True
            initialized_now = True

        elapsed_time = time.time() - start_time
        if not initialized_now:
            logger.print(
                f"Reusing existing static Line-JEPA dataset split for stage={stage!r} "
                f"(split_seed={self.split_seed})."
            )
        logger.print(f"Static Line-JEPA train dataset size: {len(self.train_dataset)}")
        logger.print(f"Static Line-JEPA val dataset size: {len(self.val_dataset)}")
        logger.print(f"Static Line-JEPA test dataset size: {len(self.test_dataset)}")
        logger.print(f"Static Line-JEPA dataloader prep took {elapsed_time:.4f} seconds")

    def _line_samples_per_file(self, data_cfg) -> int:
        ctx = "LineStaticDataModule.data"
        raw = _cfg_get(data_cfg, "line_samples_per_file", context=ctx)
        samples = int(raw)
        if samples <= 0:
            raise ValueError(f"line_samples_per_file must be > 0, got {samples}.")
        return samples

    def _source_file_count(self, data_cfg) -> int:
        ctx = "LineStaticDataModule.data"
        data_sources_raw = _cfg_get(data_cfg, "data_sources", default=None, context=ctx)
        data_files_raw = _cfg_get(data_cfg, "data_files", default=None, context=ctx)
        if data_sources_raw is not None:
            data_sources = _to_container(data_sources_raw)
            sources = PointCloudDataset._resolve_sources("", None, data_sources)
        elif data_files_raw is not None:
            file_list = _to_container(data_files_raw)
            data_path = _cfg_get(data_cfg, "data_path", context=ctx)
            sources = PointCloudDataset._resolve_sources(data_path, file_list, None)
        else:
            raise ValueError(
                "Static Line-JEPA data config must provide either 'data_sources' "
                "or 'data_files' + 'data_path'."
            )
        return sum(len(source["files"]) for source in sources)

    def _split_indices(self, *, total_samples: int, train_ratio: float) -> tuple[list[int], list[int]]:
        total_samples = int(total_samples)
        train_ratio = float(train_ratio)
        if total_samples <= 1:
            raise ValueError(f"Static Line-JEPA requires at least 2 samples, got {total_samples}.")
        if not (0.0 < train_ratio < 1.0):
            raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}.")
        train_size = int(train_ratio * total_samples)
        val_size = total_samples - train_size
        if train_size <= 0 or val_size <= 0:
            raise ValueError(
                "Static Line-JEPA split produced an empty subset. "
                f"total_samples={total_samples}, train_ratio={train_ratio}, "
                f"train_size={train_size}, val_size={val_size}."
            )
        generator = torch.Generator()
        generator.manual_seed(int(self.split_seed))
        perm = torch.randperm(total_samples, generator=generator).tolist()
        return perm[:train_size], perm[train_size:]

    def _build_datasets(self):
        data_cfg = self.cfg.data
        ctx = "LineStaticDataModule.data"
        line_samples_per_file = self._line_samples_per_file(data_cfg)
        source_file_count = self._source_file_count(data_cfg)
        total_samples = source_file_count * line_samples_per_file
        train_indices, val_indices = self._split_indices(
            total_samples=total_samples,
            train_ratio=float(_cfg_get(data_cfg, "train_ratio", default=0.8, context=ctx)),
        )

        data_sources_raw = _cfg_get(data_cfg, "data_sources", default=None, context=ctx)
        data_files_raw = _cfg_get(data_cfg, "data_files", default=None, context=ctx)
        common_kwargs = dict(
            radius=float(_cfg_get(data_cfg, "radius", context=ctx)),
            num_points=int(_cfg_get(data_cfg, "num_points", context=ctx)),
            line_atoms=int(_cfg_get(data_cfg, "line_atoms", context=ctx)),
            line_candidate_atoms=int(_cfg_get(data_cfg, "line_candidate_atoms", context=ctx)),
            line_min_separation_radius_factor=float(
                _cfg_get(data_cfg, "line_min_separation_radius_factor", default=0.0, context=ctx)
            ),
            line_anchor_views_enabled=bool(
                _cfg_get(data_cfg, "line_anchor_views_enabled", default=False, context=ctx)
            ),
            line_anchor_view_min_radius_factor=float(
                _cfg_get(data_cfg, "line_anchor_view_min_radius_factor", default=0.0, context=ctx)
            ),
            line_anchor_view_max_radius_factor=_cfg_get(
                data_cfg,
                "line_anchor_view_max_radius_factor",
                default=None,
                context=ctx,
            ),
            line_anchor_view_selection=str(
                _cfg_get(data_cfg, "line_anchor_view_selection", default="closest", context=ctx)
            ),
            line_samples_per_file=line_samples_per_file,
            normalize=bool(_cfg_get(data_cfg, "normalize", default=True, context=ctx)),
            center_neighborhoods=bool(
                _cfg_get(data_cfg, "center_neighborhoods", default=True, context=ctx)
            ),
            drop_edge_samples=bool(_cfg_get(data_cfg, "drop_edge_samples", default=True, context=ctx)),
            edge_drop_layers=_cfg_get(data_cfg, "edge_drop_layers", default=None, context=ctx),
            line_selection_method=str(
                _cfg_get(data_cfg, "line_selection_method", default="closest", context=ctx)
            ),
            line_seed=int(_cfg_get(data_cfg, "line_seed", default=0, context=ctx)),
            auto_cutoff_config=_to_container(_cfg_get(data_cfg, "auto_cutoff", default=None, context=ctx)),
        )
        if data_sources_raw is not None:
            data_sources = _to_container(data_sources_raw)
            source_kwargs = dict(data_sources=data_sources)
        elif data_files_raw is not None:
            file_list = _to_container(data_files_raw)
            if isinstance(file_list, str):
                file_list = [file_list]
            source_kwargs = dict(
                root=_cfg_get(data_cfg, "data_path", context=ctx),
                data_files=file_list,
            )
        else:
            raise RuntimeError("Static Line-JEPA source configuration disappeared during dataset build.")

        train_ds = LineStaticPointCloudDataset(
            sample_indices=train_indices,
            deterministic_lines=False,
            **source_kwargs,
            **common_kwargs,
        )
        val_ds = LineStaticPointCloudDataset(
            sample_indices=val_indices,
            deterministic_lines=True,
            **source_kwargs,
            **common_kwargs,
        )
        return train_ds, val_ds

    def _loader(self, dataset, *, shuffle: bool, drop_last: bool):
        loader_kwargs = dict(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=bool(shuffle),
            drop_last=bool(drop_last),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=_temporal_identity_or_default_collate,
        )
        prefetch_factor = getattr(self.cfg, "line_prefetch_factor", None)
        if prefetch_factor is not None:
            resolved_prefetch_factor = int(prefetch_factor)
            if resolved_prefetch_factor <= 0:
                raise ValueError(
                    "line_prefetch_factor must be > 0 when provided, "
                    f"got {resolved_prefetch_factor}."
                )
            if self.num_workers <= 0:
                raise ValueError(
                    "line_prefetch_factor requires num_workers > 0, "
                    f"got num_workers={self.num_workers}."
                )
            loader_kwargs["prefetch_factor"] = resolved_prefetch_factor
        return DataLoader(**loader_kwargs)

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False, drop_last=False)


__all__ = ["LineStaticDataModule"]
