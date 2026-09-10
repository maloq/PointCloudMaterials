"""Prepared anchor, spatial-neighbor and same-atom temporal views for VICReg."""

import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset, Subset


def periodic_tree(positions: np.ndarray, lengths: np.ndarray):
    # Imported Al/Mg coordinates are float16 global positions. Rounding can put
    # boundary atoms at L, so decode to float32 and wrap before building the tree.
    points = np.mod(positions.astype(np.float32), lengths)
    return points, cKDTree(points, boxsize=lengths, balanced_tree=False)


def local_views(points, tree, lengths, center_rows, *, num_points, radius):
    _, indices = tree.query(points[center_rows], k=num_points, workers=1)
    offsets = points[indices] - points[center_rows, None]
    offsets -= lengths * np.round(offsets / lengths)
    return (offsets / radius).astype(np.float32)


class SpatiotemporalViewDataset(Dataset):
    def __init__(self, root: Path, split: str, temporal_lag_steps: int | None = None, tda_cache: Path | None = None):
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["state"] != "complete":
            raise RuntimeError(f"View preparation is incomplete: {root}")
        shards = [s for s in manifest['shards'] if s['split']==split]
        self.shards = [np.load(root / s['views'], mmap_mode='r') for s in shards]
        self.rows = [np.arange(len(x)) if temporal_lag_steps is None else
                     np.flatnonzero(np.load(root/s['pairs'],mmap_mode='r')[:,3]==temporal_lag_steps)
                     for s,x in zip(shards,self.shards)]
        self.ends = np.cumsum([len(rows) for rows in self.rows])
        self.tda = None
        if tda_cache is not None:
            targets = json.loads((tda_cache/'manifest.json').read_text())
            protocol = targets['protocol']
            if targets['state'] != 'complete' or protocol['source_root'] != str(root.resolve()) or protocol['temporal_lag_steps'] != temporal_lag_steps:
                raise ValueError(f'TDA cache does not match the selected view producer: {tda_cache}')
            records = {s['source_views']:s for s in targets['shards']}
            self.tda = []
            for shard, rows in zip(shards, self.rows):
                target = records[shard['views']]
                np.testing.assert_array_equal(rows, np.load(tda_cache/target['rows']))
                self.tda.append(np.load(tda_cache/target['targets'], mmap_mode='r'))

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        shard = int(np.searchsorted(self.ends, index, side="right"))
        offset = index - (int(self.ends[shard - 1]) if shard else 0)
        views = torch.from_numpy(self.shards[shard][self.rows[shard][offset]].astype(np.float32))
        result = dict(points=views[0], spatial_points=views[1], temporal_points=views[2])
        if self.tda is not None:
            result['tda_targets'] = torch.from_numpy(self.tda[shard][offset].copy())
        return result


class SpatiotemporalViewDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size

    def setup(self, stage=None):
        root = Path(self.cfg.data.cache_dir)
        lag = self.cfg.data.temporal_lag_steps
        tda_path = OmegaConf.select(self.cfg, 'tda.cache_dir') if OmegaConf.select(self.cfg, 'tda.enabled', default=False) else None
        tda_cache = Path(tda_path) if tda_path is not None else None
        self.train_dataset = SpatiotemporalViewDataset(root, "train", lag, tda_cache)
        self.val_dataset = SpatiotemporalViewDataset(root, "val", lag, tda_cache)
        if getattr(self.cfg, "spatiotemporal_mix_validation", False):
            # Use the same material-mixed batches every epoch and for both losses.
            order = torch.randperm(len(self.val_dataset), generator=torch.Generator().manual_seed(self.cfg.seed_everything))
            self.val_dataset = Subset(self.val_dataset, order.tolist())
        self.test_dataset = self.val_dataset

    def _loader(self, dataset, *, shuffle, drop_last, batch_size):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          drop_last=drop_last, num_workers=self.cfg.num_workers,
                          pin_memory=True, persistent_workers=self.cfg.num_workers > 0)

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True, drop_last=True, batch_size=self.batch_size)

    def val_dataloader(self):
        batch_size = getattr(self.cfg, "spatiotemporal_validation_batch_size", self.batch_size)
        return self._loader(self.val_dataset, shuffle=False, drop_last=False, batch_size=batch_size)

    def test_dataloader(self):
        return self.val_dataloader()
