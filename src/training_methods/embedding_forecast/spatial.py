"""Observed-frame neighbors among cached centers, without duplicating embeddings."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.spatial import cKDTree
import torch

from src.data_utils.shooting_binary import ShootingBinaryTrajectory
from src.experiment_runner.registry import sha256, write_json
from src.project_runtime.paths import load_json
from .resident import ResidentWindows
from .data import WindowDataset
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler


class SpatialWindowDataset(WindowDataset):
    def __init__(self, *args, spatial_root):
        super().__init__(*args)
        self.spatial_root = Path(spatial_root)


class SpatialResidentLoader:
    def __init__(self, dataset, batch_size, shuffle, seed, device):
        self.dataset = SpatialResidentWindows(dataset, device, dataset.spatial_root)
        generator = torch.Generator().manual_seed(seed)
        sampler = RandomSampler(dataset, generator=generator) if shuffle else SequentialSampler(dataset)
        self.sampler = BatchSampler(sampler, batch_size, drop_last=False)

    def __iter__(self):
        for indices in self.sampler:
            yield self.dataset[indices]

    def __len__(self):
        return len(self.sampler)


def pool_neighbors(values, neighbors):
    """Pool repository neighbor rows at their own time; retain the cache dtype."""
    count, frames, _ = neighbors.shape
    result = torch.empty((count, frames, values.shape[-1]), dtype=values.dtype, device=values.device)
    # Bound the gather for broad neighborhoods; the original K=8 / D=256 batch is unchanged.
    batch_frames = min(8192, max(1, 2**24//(neighbors.shape[-1]*values.shape[-1])))
    for start in range(0, count*frames, batch_frames):
        flat = torch.arange(start, min(start+batch_frames, count*frames), device=values.device)
        center, time_index = flat // frames, flat % frames
        selected = values[neighbors[center, time_index], time_index[:, None]].float().mean(dim=1)
        result[center, time_index] = selected.to(result.dtype)
    return result


def prepare_source(config, record, source):
    torch.set_num_threads(1)
    started = time.monotonic()
    cache = Path(config['embedding_cache']) / record['directory']
    output = Path(config['output']) / record['directory']
    output.mkdir(parents=True)
    if source['name'] != record['name'] or source['preparation_seed'] != record['preparation_seed']:
        raise ValueError(f'Spatial source identity differs from embedding producer: {record["directory"]}')
    trajectory = ShootingBinaryTrajectory.load(source['path'])
    ids = np.load(cache / 'atom_ids.npy')
    rows = np.searchsorted(trajectory.atom_ids, ids)
    np.testing.assert_array_equal(trajectory.atom_ids[rows], ids)
    frames = np.load(cache / 'frames.npy')
    np.testing.assert_array_equal(trajectory.timesteps[frames], np.load(cache / 'timesteps.npy'))
    neighbors = np.empty((record['centers'], len(frames), config['neighbors']), dtype=np.uint16)
    radii = np.empty((record['centers'], len(frames), 2), dtype=np.float32)
    for column, frame in enumerate(frames):
        low = trajectory.box_low[frame].astype(np.float64)
        lengths = trajectory.box_high[frame].astype(np.float64)-low
        points = np.mod(trajectory.positions[frame, rows].astype(np.float64)-low, lengths)
        tree = cKDTree(points, boxsize=lengths)
        distance, indices = tree.query(points, k=config['neighbors']+1, workers=1)
        np.testing.assert_array_equal(indices[:, 0], np.arange(len(rows)))
        neighbors[:, column] = indices[:, 1:]
        radii[:, column, 0] = distance[:, 1:].mean(axis=1)
        radii[:, column, 1] = distance[:, -1]
    np.save(output / 'neighbors.npy', neighbors)
    np.save(output / 'radii_A.npy', radii)
    files = ['neighbors.npy', 'radii_A.npy']
    if config.get('prepool_embeddings', False):
        path = cache/'embeddings.npy'
        if sha256(path) != record['checksums']['embeddings.npy']:
            raise ValueError(f'Spatial pooling embedding values changed: {path}')
        values = torch.from_numpy(np.array(np.load(path, mmap_mode='r'), copy=True))
        pooled = pool_neighbors(values, torch.from_numpy(neighbors.astype(np.int64)))
        np.save(output/'pooled_embeddings.npy', pooled.numpy())
        files.append('pooled_embeddings.npy')
    metadata = dict(directory=record['directory'], source_index=record['source_index'],
        split=record['split'], centers=record['centers'], frames=record['frames'],
        atom_ids_sha256=sha256(cache / 'atom_ids.npy'),
        source_manifest_sha256=sha256(Path(source['path']) / 'manifest.json'),
        neighbors=config['neighbors'],
        checksums={name:sha256(output/name) for name in files},
        mean_neighbor_distance_A=float(radii[:, :, 0].mean()),
        median_outer_distance_A=float(np.median(radii[:, :, 1])), elapsed_s=time.monotonic()-started)
    write_json(output / 'complete.json', metadata)
    return metadata


def _prepare(config):
    root = Path(config['output']); root.mkdir(parents=True, exist_ok=True)
    if (root / 'manifest.json').exists():
        raise FileExistsError(f'Spatial cache already completed: {root}')
    manifest_path = Path(config['embedding_cache']) / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['state'] != 'complete' or any(r['centers'] > 65535 for r in manifest['shards']):
        raise ValueError('Spatial cache requires a completed embedding producer with uint16-addressable centers.')
    if any(not 1 <= config['neighbors'] < r['centers'] for r in manifest['shards']):
        raise ValueError('Spatial neighbors must be positive and exclude the tracked center.')
    sources = json.loads(Path(config['sources_config']).read_text())['sources']
    records = []
    write_json(root / 'config.json', config)
    with ProcessPoolExecutor(max_workers=config['workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(prepare_source, config, r, sources[r['source_index']]) for r in manifest['shards']]
        for future in as_completed(futures):
            record = future.result(); records.append(record)
            print(f'Spatial cache {len(records)}/{len(futures)}: source {record["source_index"]}, '
                  f'{record["elapsed_s"]:.1f} s, outer median {record["median_outer_distance_A"]:.2f} A', flush=True)
            write_json(root / 'progress.json', dict(completed=len(records), total=len(futures)))
    records.sort(key=lambda r:r['source_index'])
    write_json(root / 'manifest.json', dict(state='complete', config=config, records=records,
        base_cache_manifest_sha256=sha256(manifest_path),
        semantics=f"{config['neighbors']} nearest OTHER cached centers, reselected at each observed frame; no future positions used"))


def prepare(config):
    root = Path(config['output']); root.mkdir(parents=True, exist_ok=True)
    if (root/'manifest.json').exists():
        raise FileExistsError(f'Spatial cache already completed: {root}')
    write_json(root/'status.json', dict(state='preparing', neighbors=config['neighbors']))
    try:
        _prepare(config)
    except BaseException:
        write_json(root/'status.json', dict(state='failed', error=traceback.format_exc()))
        raise
    write_json(root/'status.json', dict(state='complete', neighbors=config['neighbors']))


class SpatialResidentWindows(ResidentWindows):
    """Pool same-frame neighbors once per GPU split; preserve base window identities."""

    def __init__(self, dataset, device, spatial_root):
        super().__init__(dataset, device)
        spatial_root = Path(spatial_root)
        manifest = json.loads((spatial_root / 'manifest.json').read_text())
        if manifest['state'] != 'complete' or manifest['base_cache_manifest_sha256'] != sha256(dataset.root / 'manifest.json'):
            raise ValueError(f'Spatial and embedding cache identity mismatch: {spatial_root}')
        records = {r['directory']:r for r in manifest['records']}
        self.spatial = torch.empty_like(self.embeddings)
        self.radii = torch.empty((*self.embeddings.shape[:2], 2), device=device, dtype=torch.float32)
        base = 0
        for record in dataset.records:
            directory = spatial_root / record['directory']
            for name, digest in records[record['directory']]['checksums'].items():
                if sha256(directory / name) != digest:
                    raise ValueError(f'Spatial artifact changed: {directory/name}')
            n = record['centers']; frames = record['frames']
            values = self.embeddings[base:base+n]
            if 'pooled_embeddings.npy' in records[record['directory']]['checksums']:
                pooled = np.load(directory/'pooled_embeddings.npy')
                if pooled.shape != tuple(values.shape) or torch.from_numpy(pooled).dtype != values.dtype:
                    raise ValueError(f'Prepooled spatial shape/dtype differs from embeddings: {directory}')
                self.spatial[base:base+n].copy_(torch.from_numpy(pooled))
            else:
                neighbors = torch.from_numpy(np.load(directory / 'neighbors.npy').astype(np.int64)).to(device)
                self.spatial[base:base+n] = pool_neighbors(values, neighbors)
            self.radii[base:base+n].copy_(torch.from_numpy(np.load(directory / 'radii_A.npy')))
            base += n
        print(f'Resident spatial means: {self.spatial.numel()*self.spatial.element_size()/2**30:.3f} GiB', flush=True)

    def __getitem__(self, indices):
        batch = super().__getitem__(indices)
        rows = torch.as_tensor(indices, device=self.embeddings.device)
        centers = rows // self.windows
        columns = self.source.history_skip+(rows % self.windows)[:, None]*self.source.stride+self.offsets[:self.source.history_steps]
        batch['spatial'] = self.spatial[centers[:, None], columns].float()
        batch['spatial_radii_A'] = self.radii[centers[:, None], columns]
        return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--stage', choices=('neighbors','geometry'), default='neighbors')
    args = parser.parse_args()
    if args.stage == 'geometry':
        from .attention_data import prepare_geometry
        prepare_geometry(load_json(args.config))
    else:
        prepare(load_json(args.config))


if __name__ == '__main__':
    main()
