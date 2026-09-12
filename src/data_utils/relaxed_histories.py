"""Prepared MEAM histories and relaxed-anchor targets for the original VICReg module."""

import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from scipy.spatial import cKDTree
import torch
from torch.utils.data import DataLoader, Dataset

from src.data_utils.mace_history import history_clouds
from src.data_utils.shooting_binary import ShootingBinaryTrajectory
from src.data_utils.topology_targets import fit_targets, transform_target
from src.experiment_runner.registry import sha256, write_json


def prepare(cfg):
    root = Path(cfg.data.cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    source_path = Path(cfg.data.relaxed_manifest)
    source = json.loads(source_path.read_text())
    if source['protocol'] != 'denoising80':
        raise ValueError(f'Expected prepared denoising80 MEAM targets: {source_path}')
    protocol = dict(source_manifest=str(source_path.resolve()), source_sha256=sha256(source_path),
        radius_A=float(cfg.data.normalization_radius_A), frame_offsets_ps=list(cfg.data.frame_offsets_ps),
        temporal_lag_steps=int(cfg.data.temporal_lag_steps), neighbor_k=int(cfg.vicreg_neighbor_k),
        seed=int(cfg.data.preparation_seed), tda_components=32,
        block_scale_floor_fraction=float(cfg.tda.block_scale_floor_fraction), producer_sha256=sha256(Path(__file__)))
    if (root/'manifest.json').exists():
        saved = json.loads((root/'manifest.json').read_text())
        if saved['protocol'] != protocol:
            raise ValueError(f'Prepared history protocol changed: {root}; choose a new cache.')
        for name, digest in saved['checksums'].items():
            if sha256(root/name) != digest:
                raise ValueError(f'Changed prepared history artifact: {root/name}')
        print(f'Verified prepared VICReg histories: {root}', flush=True)
        return

    records, raw_targets = [], []
    trajectory = None
    for context, original in enumerate(source['shards']):
        origin = Path(original['directory'])
        for name, digest in original['checksums'].items():
            if sha256(origin/name) != digest:
                raise ValueError(f'Changed relaxed-target input: {origin/name}')
        source_cfg = original['provenance']['source']
        if trajectory is None or trajectory.root != Path(source_cfg['path']):
            trajectory = ShootingBinaryTrajectory.load(source_cfg['path'])
            trajectory.verify_checksums()
        frame = original['frame']
        steps = np.rint(np.array(cfg.data.frame_offsets_ps)/source_cfg['cadence_ps']).astype(np.int64)
        np.testing.assert_allclose(steps*source_cfg['cadence_ps'], cfg.data.frame_offsets_ps, rtol=0, atol=1e-9)
        centers = np.load(origin/'centers.npy')
        anchor_ids = np.load(origin/'neighbor_ids.npy')
        anchor = np.load(origin/'histories.npy')
        np.testing.assert_array_equal(anchor_ids[:, 0], trajectory.atom_ids[centers])
        rng = np.random.default_rng(np.random.SeedSequence([cfg.data.preparation_seed, context]))

        low = trajectory.box_low[frame].astype(np.float64)
        lengths = trajectory.box_high[frame].astype(np.float64)-low
        points = np.mod(trajectory.positions[frame].astype(np.float64)-low, lengths)
        tree = cKDTree(points, boxsize=lengths)
        candidates = tree.query(points[centers], k=cfg.vicreg_neighbor_k+1, workers=1)[1][:, 1:]
        spatial_centers = candidates[np.arange(len(centers)), rng.integers(0, cfg.vicreg_neighbor_k, len(centers))]
        spatial_neighbors = tree.query(points[spatial_centers], k=80, workers=1)[1]
        spatial, spatial_error = history_clouds(trajectory, spatial_centers,
            trajectory.atom_ids[spatial_neighbors], frame, steps)

        temporal_frame = frame-cfg.data.temporal_lag_steps
        low = trajectory.box_low[temporal_frame].astype(np.float64)
        lengths = trajectory.box_high[temporal_frame].astype(np.float64)-low
        points = np.mod(trajectory.positions[temporal_frame].astype(np.float64)-low, lengths)
        temporal_neighbors = cKDTree(points, boxsize=lengths).query(points[centers], k=80, workers=1)[1]
        temporal, temporal_error = history_clouds(trajectory, centers,
            trajectory.atom_ids[temporal_neighbors], temporal_frame, steps)
        views = np.stack((anchor, spatial, temporal), axis=1)
        path = root/f'{context:03d}.views.npy'
        np.save(path, views)
        raw_targets.append(np.load(origin/'targets.npy'))
        records.append(dict(views=path.name, split=original['split'], samples=len(centers),
            source=original['source_index'], context=context, frame=frame,
            temperature_K=original['temperature_K'], source_directory=str(origin),
            additional_quantization_max_A=max(spatial_error, temporal_error)))
        write_json(root/'status.json', dict(state='preparing', contexts=len(records), total=len(source['shards'])))
        print(f'VICREG_HISTORY {len(records)}/{len(source["shards"])} {original["name"]}', flush=True)

    targets = np.concatenate(raw_targets)
    train = np.concatenate([np.full(r['samples'], r['split']=='train') for r in records])
    scaling = fit_targets(targets[train], 32, cfg.tda.block_scale_floor_fraction)
    np.save(root/'targets.npy', targets)
    np.savez(root/'scaling.npz', **scaling)
    names = [r['views'] for r in records]+['targets.npy', 'scaling.npz']
    write_json(root/'manifest.json', dict(state='complete', protocol=protocol, shards=records,
        checksums={name:sha256(root/name) for name in names},
        input_definition='Physical float16 offsets; loader divides by the declared source radius. '
            'Each view has its own complete 80-atom neighborhood; atom IDs are fixed within each history.',
        target_definition='Original relaxed target for the anchor view only. Spatial and earlier views '
            'have no assigned relaxed target and participate only in the original VICReg pairs.',
        scaling_fit='All training anchors only; no validation or test rows.'))
    write_json(root/'status.json', dict(state='complete', contexts=len(records), examples=len(targets)))


class RelaxedHistoryDataset(Dataset):
    def __init__(self, cfg, split):
        root = Path(cfg.data.cache_dir)
        manifest = json.loads((root/'manifest.json').read_text())
        if manifest['state'] != 'complete':
            raise RuntimeError(f'History preparation incomplete: {root}')
        self.records = manifest['shards']
        self.contexts = np.concatenate([np.full(r['samples'], i, dtype=np.int64) for i,r in enumerate(self.records)])
        self.indices = np.flatnonzero(np.isin(self.contexts, [i for i,r in enumerate(self.records) if r['split']==split]))
        self.starts = np.cumsum([0]+[r['samples'] for r in self.records])
        self.views = {i:np.load(root/r['views'], mmap_mode='r') for i,r in enumerate(self.records) if r['split']==split}
        self.scaling = dict(np.load(root/'scaling.npz'))
        self.raw_targets = np.load(root/'targets.npy')
        self.targets = transform_target(self.raw_targets, self.scaling, cfg.tda.target).astype(np.float32)
        self.radius = float(cfg.data.normalization_radius_A)
        self.input_mode = cfg.data.input_mode
        if self.input_mode not in ('anchor', 'history'):
            raise ValueError(f'Unknown prepared-history input mode: {self.input_mode}')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.indices[index]
        context = self.contexts[row]
        record = self.records[context]
        views = self.views[context][row-self.starts[context]].astype(np.float32)/self.radius
        if self.input_mode == 'anchor':
            views = views[:, -1]
        return dict(points=torch.from_numpy(views[0]), spatial_points=torch.from_numpy(views[1]),
            temporal_points=torch.from_numpy(views[2]), tda_targets=torch.from_numpy(self.targets[row]),
            row=row, context=context, source=record['source'], frame=record['frame'],
            temperature_K=record['temperature_K'])


class RelaxedHistoryDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size

    def setup(self, stage=None):
        self.train_dataset = RelaxedHistoryDataset(self.cfg, 'train')
        self.val_dataset = RelaxedHistoryDataset(self.cfg, 'val')
        self.test_dataset = RelaxedHistoryDataset(self.cfg, 'test')

    def _loader(self, dataset, train):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=train, drop_last=train,
            num_workers=self.cfg.num_workers, pin_memory=True, persistent_workers=self.cfg.num_workers>0)

    def train_dataloader(self):
        return self._loader(self.train_dataset, True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, False)
