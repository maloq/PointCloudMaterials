"""Prepared anchor, spatial-neighbor and same-atom temporal views for VICReg."""

import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset, Subset
from src.project_runtime.paths import resolve_path


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
        root = resolve_path(root)
        tda_cache = resolve_path(tda_cache) if tda_cache is not None else None
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
            if targets['state'] != 'complete' or resolve_path(protocol['source_root']).resolve() != root.resolve() or protocol['temporal_lag_steps'] != temporal_lag_steps:
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


def prepare_branch(task):
    source, output, branch_index, seed = task
    from numpy.lib.format import open_memmap
    from src.data.trajectories.lammps import TemporalLAMMPSBinaryTrajectory
    trajectory = TemporalLAMMPSBinaryTrajectory.load(source["path"])
    if trajectory.frame_count != source["frame_count"]:
        raise ValueError(f"Unexpected frame count: {source['path']}")
    expected_step = source["timestep_stride"]
    if not np.array_equal(trajectory.timesteps, np.arange(source["frame_count"]) * expected_step):
        raise ValueError(f"Unexpected timestep grid: {source['path']}")
    rng = np.random.default_rng(seed + branch_index)
    shards = []
    for selection in source["splits"]:
        split = selection["split"]
        anchors = selection["anchors"]
        centers = selection["centers"]
        count = len(anchors) * centers
        stem = f"{source['material']}_{source['snapshot']}_{split}"
        views_file, pairs_file = f"{stem}.views.npy", f"{stem}.pairs.npy"
        views = open_memmap(output / views_file, mode="w+", dtype=np.float16, shape=(count, 3, 80, 3))
        pairs = open_memmap(output / pairs_file, mode="w+", dtype=np.int64, shape=(count, 4))
        rows = np.arange(trajectory.atom_count)
        pool = rows[rows % 5 != 0] if split == "train" else rows[rows % 5 == 0]
        for j, anchor in enumerate(anchors):
            selected = rng.choice(pool, size=centers, replace=False)
            lengths = trajectory.box_high[anchor] - trajectory.box_low[anchor]
            points, tree = periodic_tree(trajectory.positions[anchor], lengths)
            _, nearest = tree.query(points[selected], k=9, workers=1)
            candidates = nearest[nearest != selected[:, None]].reshape(centers, 8)
            spatial = candidates[np.arange(centers), rng.integers(0, 8, size=centers)]
            batch = slice(j * centers, (j + 1) * centers)
            views[batch, 0] = local_views(points, tree, lengths, selected, num_points=80, radius=source["radius"])
            views[batch, 1] = local_views(points, tree, lengths, spatial, num_points=80, radius=source["radius"])
            lags = np.resize(np.asarray(source["lags"], dtype=np.int64), centers)
            rng.shuffle(lags)
            for lag in source["lags"]:
                frame = anchor + lag
                lengths_t = trajectory.box_high[frame] - trajectory.box_low[frame]
                points_t, tree_t = periodic_tree(trajectory.positions[frame], lengths_t)
                mask = lags == lag
                views[j * centers + np.flatnonzero(mask), 2] = local_views(
                    points_t, tree_t, lengths_t, selected[mask], num_points=80, radius=source["radius"],
                )
            pairs[batch] = np.column_stack((trajectory.atom_ids[selected], trajectory.atom_ids[spatial], np.full(centers, anchor), lags))
        views.flush()
        pairs.flush()
        shards.append(dict(material=source["material"], snapshot=source["snapshot"], split=split,
                           samples=count, views=views_file, pairs=pairs_file))
        print(f"Prepared {stem}: {count} triplets", flush=True)
    return shards


def prepare_expanded(cfg):
    """Append declared trajectories to the original normalized view cache."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from src.experiment_runner.registry import sha256, write_json

    output = Path(cfg['output'])
    output.mkdir(parents=True, exist_ok=True)
    base = Path(cfg['base_cache'])
    original = json.loads((base/'manifest.json').read_text())
    if original['state'] != 'complete':
        raise RuntimeError(f'Original view cache is incomplete: {base}')
    protocol = dict(config=cfg, base_manifest_sha256=sha256(base/'manifest.json'))
    if (output/'manifest.json').exists():
        saved = json.loads((output/'manifest.json').read_text())
        if saved['protocol'] != protocol:
            raise ValueError(f'Expanded view protocol changed: {output}')
        for name, digest in saved['checksums'].items():
            if sha256(output/name) != digest:
                raise ValueError(f'Expanded view checksum changed: {output/name}')
        print(f'Verified completed expanded cache: {output}', flush=True)
        return
    shards = list(original['shards'])
    checksums = {}
    for shard in shards:
        for key in ('views', 'pairs'):
            name = shard[key]
            target = output/name
            if not target.is_symlink():
                target.symlink_to((base/name).resolve())
            if target.resolve() != (base/name).resolve():
                raise ValueError(f'Incorrect original-cache link: {target}')
            checksums[name] = sha256(target)
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        pending = {}
        for index, source in enumerate(cfg['sources']):
            source_manifest = Path(source['path'])/'manifest.json'
            if sha256(source_manifest) != source['manifest_sha256']:
                raise ValueError(f'Source manifest changed: {source_manifest}')
            part = output/f"{source['material']}_{source['snapshot']}.json"
            if part.exists():
                saved = json.loads(part.read_text())
                if saved['source'] != source:
                    raise ValueError(f'Prepared source configuration changed: {part}')
                for name, digest in saved['checksums'].items():
                    if sha256(output/name) != digest:
                        raise ValueError(f'Prepared view checksum changed: {output/name}')
                shards.extend(saved['shards'])
                checksums.update(saved['checksums'])
            else:
                future = pool.submit(prepare_branch, (source, output, index, cfg['seed']))
                pending[future] = (source, part)
        for future in as_completed(pending):
            source, part = pending[future]
            prepared = future.result()
            digests = {s[k]: sha256(output/s[k]) for s in prepared for k in ('views','pairs')}
            write_json(part, dict(source=source, shards=prepared, checksums=digests))
            shards.extend(prepared)
            checksums.update(digests)
            write_json(output/'status.json', dict(state='preparing', completed_shards=len(shards)))
    counts = {}
    for split in ('train','val'):
        counts[split] = {material: sum(int(np.count_nonzero(np.load(output/s['pairs'], mmap_mode='r')[:,3] == 1))
            for s in shards if s['split']==split and s['material']==material) for material in cfg['materials']}
    report = dict(state='complete', protocol=protocol, sources=original['sources']+cfg['sources'],
        shards=shards, checksums=checksums, selected_lag_samples=counts,
        split_note=cfg['split_note'], cutoff_estimation=cfg['cutoff_estimation'],
        duplicate_trajectories_excluded=cfg['duplicate_trajectories_excluded'],
        num_points=80, views=original['views'], pair_columns=original['pair_columns'],
        temporal_lags_ps=[0.1], cache_storage_dtype='float16')
    write_json(output/'manifest.json', report)
    write_json(output/'status.json', dict(state='complete', selected_lag_samples=counts))
    print(json.dumps(counts), flush=True)
