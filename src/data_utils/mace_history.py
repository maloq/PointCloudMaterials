"""Causal, identity-matched histories for the repository's relaxed MACE targets."""

import json
from pathlib import Path

import numpy as np
import torch

from src.data_utils.temporal_campaign import write_json
from src.data_utils.temporal_lammps_binary import TemporalLAMMPSBinaryTrajectory
from src.simulation.relaxation import sha256


def history_clouds(trajectory, centers, neighbor_ids, anchor, steps):
    """Keep the target's hot-selected IDs in every frame; decode before geometry."""
    frames = anchor + steps
    if frames[0] < 0 or frames[-1] >= trajectory.frame_count:
        raise ValueError(f"History {frames.tolist()} exceeds trajectory {trajectory.root}.")
    # TemporalLAMMPSBinaryTrajectory verifies exact ordered IDs 1..N on load.
    neighbors = neighbor_ids.astype(np.int64) - 1
    np.testing.assert_array_equal(neighbors[:, 0], centers)
    result = []
    error = 0.0
    for frame in frames:
        low = trajectory.box_low[frame].astype(np.float64)
        lengths = trajectory.box_high[frame].astype(np.float64) - low
        positions = np.mod(trajectory.positions[frame, neighbors].astype(np.float64) - low, lengths)
        origin = np.mod(trajectory.positions[frame, centers].astype(np.float64) - low, lengths)
        offsets = positions - origin[:, None]
        offsets -= lengths * np.round(offsets / lengths)
        stored = offsets.astype(np.float16)
        error = max(error, float(np.abs(stored.astype(np.float64) - offsets).max()))
        result.append(stored)
    return np.stack(result, axis=1), error


def prepare(cfg):
    root = Path(cfg['cache'])
    root.mkdir(parents=True, exist_ok=True)
    source = Path(cfg['paired_manifest'])
    manifest = json.loads(source.read_text())
    if manifest['protocol'] != 'thermal80':
        raise ValueError(f"History supervision requires the verified thermal80 cache: {source}")
    offsets = np.array(cfg['encoder']['frame_offsets_ps'])
    records = []
    for record in manifest['shards']:
        source_dir = Path(record['directory'])
        for name, digest in record['checksums'].items():
            if sha256(source_dir / name) != digest:
                raise ValueError(f"Changed paired source: {source_dir / name}")
        directory = root / record['name']
        directory.mkdir(exist_ok=True)
        signature = dict(paired_manifest_sha256=sha256(source_dir / 'manifest.json'),
                         frame_offsets_ps=offsets.tolist())
        completion = directory / 'manifest.json'
        if completion.exists():
            saved = json.loads(completion.read_text())
            if saved['signature'] != signature:
                raise ValueError(f"History configuration/source changed: {directory}")
            for name, digest in saved['checksums'].items():
                if sha256(directory / name) != digest:
                    raise ValueError(f"Changed history artifact: {directory / name}")
            records.append(saved)
            continue
        write_json(root / 'status.json', dict(state='preparing', shard=record['name']))
        trajectory = TemporalLAMMPSBinaryTrajectory.load(record['path'])
        steps = np.rint(offsets / record['cadence_ps']).astype(np.int64)
        np.testing.assert_allclose(steps * record['cadence_ps'], offsets, rtol=0, atol=1e-9)
        if steps[-1] != 0 or np.any(np.diff(steps) <= 0):
            raise ValueError(f"History steps must increase to the anchor: {steps}")
        ids = np.load(source_dir / 'ids.npy')
        frames = np.load(source_dir / 'frames.npy')
        neighbors = np.load(source_dir / 'neighbor_ids.npy')
        hot = np.load(source_dir / 'clouds.npy', mmap_mode='r')
        history = np.lib.format.open_memmap(directory / 'histories.npy', mode='w+',
            dtype=np.float16, shape=(len(ids), 4, len(steps), 80, 3))
        error = 0.0
        for view in range(4):
            for anchor in np.unique(frames[:, view]):
                rows = np.flatnonzero(frames[:, view] == anchor)
                values, quantization = history_clouds(
                    trajectory, ids[rows, view], neighbors[rows, view], int(anchor), steps)
                np.testing.assert_array_equal(values[:, -1], hot[rows, view])
                history[rows, view] = values
                error = max(error, quantization)
        history.flush()
        np.save(directory / 'targets.npy', np.load(source_dir / 'tda.npy')[:, :4])
        np.save(directory / 'frames.npy', frames[:, :, None] + steps)
        np.save(directory / 'ids.npy', ids)
        np.save(directory / 'neighbor_ids.npy', neighbors)
        positions = np.stack([trajectory.positions[f, i].astype(np.float32) for f, i in zip(frames[:, 0], ids[:, 0])])
        np.save(directory / 'center_positions.npy', positions)
        filenames = ('histories.npy', 'targets.npy', 'frames.npy', 'ids.npy', 'neighbor_ids.npy', 'center_positions.npy')
        saved = dict(name=record['name'], directory=str(directory), source=record['path'],
            split=record['split'], material=record['material'], count=len(ids),
            signature=signature, trajectory_manifest_sha256=sha256(Path(record['path']) / 'manifest.json'),
            cadence_ps=record['cadence_ps'], frame_min=int(frames.min()+steps[0]), frame_max=int(frames.max()),
            local_quantization_max_A=error, anchor_matches_paired_cache_exactly=True,
            checksums={name: sha256(directory / name) for name in filenames})
        write_json(completion, saved)
        records.append(saved)
        print('HISTORY_SHARD', json.dumps(saved), flush=True)
    # Ta shares a source across splits: verify whole observed windows are disjoint.
    for train in (r for r in records if r['split'] == 'train'):
        for val in (r for r in records if r['split'] == 'val' and r['source'] == train['source']):
            if train['frame_max'] >= val['frame_min']:
                raise ValueError(f"Shared-source history windows overlap: {train['name']}, {val['name']}")
    write_json(root / 'manifest.json', dict(protocol='temporal80', shards=records,
        paired_manifest=str(source), paired_manifest_sha256=sha256(source),
        frame_offsets_ps=offsets.tolist(), training_views=[0], analysis_views=[0, 1, 2, 3]))
    write_json(root / 'status.json', dict(state='complete', shards=len(records), anchors=sum(r['count'] for r in records)))


class Histories:
    """GPU-resident float16 histories; float32 geometry and standardized targets."""

    def __init__(self, cfg, scaling=None, device='cuda'):
        self.records = json.loads((Path(cfg['cache']) / 'manifest.json').read_text())['shards']
        self.raw_targets = np.concatenate([np.load(Path(r['directory']) / 'targets.npy') for r in self.records])
        self.materials = np.concatenate([np.full(r['count'], r['material'], np.int64) for r in self.records])
        self.sources = np.concatenate([np.full(r['count'], i, np.int64) for i, r in enumerate(self.records)])
        self.indices = {split: np.flatnonzero(np.isin(self.sources, [i for i, r in enumerate(self.records) if r['split'] == split]))
                        for split in ('train', 'val')}
        self.clouds = torch.from_numpy(np.concatenate([
            np.load(Path(r['directory']) / 'histories.npy') for r in self.records])).to(device)
        self.material = torch.from_numpy(self.materials).to(device)
        self.targets = None
        if scaling is not None:
            self.set_scaling(scaling)

    def set_scaling(self, scaling):
        targets = ((self.raw_targets - scaling['tda_mean']) @ scaling['tda_components'].T) / scaling['tda_std']
        self.targets = torch.from_numpy(targets.astype(np.float32)).to(self.clouds.device)

    def get(self, indices, view=0):
        rows = torch.as_tensor(indices, dtype=torch.long, device=self.clouds.device)
        return self.clouds[rows, view].float(), self.targets[rows, view], self.material[rows]
