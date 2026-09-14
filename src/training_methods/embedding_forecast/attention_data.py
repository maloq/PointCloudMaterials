"""Individual cached neighbors and periodic relative geometry for learned attention."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler

from src.data_utils.shooting_binary import ShootingBinaryTrajectory
from src.experiment_runner.registry import sha256, write_json
from .data import WindowDataset
from .resident import ResidentWindows


def prepare_geometry_source(config, record, source):
    start = time.monotonic()
    cache = Path(config['embedding_cache'])/record['directory']
    output = Path(config['output'])/record['directory']; output.mkdir(parents=True)
    if (source['name'], source['preparation_seed']) != (record['name'], record['preparation_seed']):
        raise ValueError(f'Geometry source differs from embedding producer: {record["directory"]}')
    trajectory = ShootingBinaryTrajectory.load(source['path'])
    ids = np.load(cache/'atom_ids.npy'); rows = np.searchsorted(trajectory.atom_ids, ids)
    np.testing.assert_array_equal(trajectory.atom_ids[rows], ids)
    frames = np.load(cache/'frames.npy')
    np.testing.assert_array_equal(trajectory.timesteps[frames], np.load(cache/'timesteps.npy'))
    low = trajectory.box_low[frames].astype(np.float64)
    lengths = trajectory.box_high[frames].astype(np.float64)-low
    positions = trajectory.positions[frames[:, None], rows[None]].astype(np.float64)
    positions = np.mod(positions-low[:, None], lengths[:, None]).transpose(1, 0, 2)
    np.save(output/'center_positions_A.npy', positions.astype(np.float32))
    np.save(output/'box_lengths_A.npy', lengths.astype(np.float32))
    metadata = dict(directory=record['directory'], source_index=record['source_index'],
        atom_ids_sha256=sha256(cache/'atom_ids.npy'),
        trajectory_manifest_sha256=sha256(Path(source['path'])/'manifest.json'),
        checksums={name:sha256(output/name) for name in ('center_positions_A.npy','box_lengths_A.npy')},
        elapsed_s=time.monotonic()-start)
    write_json(output/'complete.json', metadata)
    return metadata


def prepare_geometry(config):
    root = Path(config['output']); root.mkdir(parents=True, exist_ok=True)
    if (root/'manifest.json').exists():
        raise FileExistsError(f'Geometry cache already complete: {root}')
    write_json(root/'status.json', dict(state='preparing'))
    try:
        base = Path(config['embedding_cache'])/'manifest.json'
        manifest = json.loads(base.read_text())
        if manifest['state'] != 'complete':
            raise ValueError(f'Geometry requires completed embeddings: {base}')
        sources = json.loads(Path(config['sources_config']).read_text())['sources']
        write_json(root/'config.json', config)
        records = []
        with ProcessPoolExecutor(max_workers=config['workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(prepare_geometry_source, config, r, sources[r['source_index']]) for r in manifest['shards']]
            for future in as_completed(futures):
                record = future.result(); records.append(record)
                print(f'Geometry {len(records)}/{len(futures)}: source {record["source_index"]}, {record["elapsed_s"]:.1f}s', flush=True)
                write_json(root/'progress.json', dict(completed=len(records), total=len(futures)))
        records.sort(key=lambda r:r['source_index'])
        write_json(root/'manifest.json', dict(state='complete', config=config, records=records,
            base_cache_manifest_sha256=sha256(base),
            semantics='Float32 periodic center positions and box lengths at each cached frame; attention gathers observed frames only.'))
        write_json(root/'status.json', dict(state='complete'))
    except BaseException:
        write_json(root/'status.json', dict(state='failed', error=traceback.format_exc()))
        raise


def gather_neighbors(embeddings, neighbors, positions, box_lengths, centers, columns, base_centers, source_rows):
    """Indices refer to OTHER centers of this source at this exact observed frame."""
    other = neighbors[centers[:, None], columns].long()+base_centers[centers, None, None]
    values = embeddings[other, columns[:, :, None]]
    relative = positions[other, columns[:, :, None]]-positions[centers[:, None], columns][:, :, None]
    lengths = box_lengths[source_rows[centers, None], columns][:, :, None]
    relative = relative-lengths*torch.round(relative/lengths)
    return values, relative


class AttentionWindowDataset(WindowDataset):
    def __init__(self, *args, spatial_root, geometry_root):
        super().__init__(*args)
        self.spatial_root = Path(spatial_root)
        self.geometry_root = Path(geometry_root)


class AttentionResidentWindows(ResidentWindows):
    """Store each embedding once and gather individual neighbors only for a microbatch."""
    tensor_names = ('embeddings','offsets','neighbors','positions_A','box_lengths_A','base_centers','source_rows')

    def __init__(self, dataset, device):
        super().__init__(dataset, device)
        base_digest = sha256(dataset.root/'manifest.json')
        manifests = {}
        for kind, root in [('spatial',dataset.spatial_root),('geometry',dataset.geometry_root)]:
            manifest = json.loads((root/'manifest.json').read_text())
            if manifest['state'] != 'complete' or manifest['base_cache_manifest_sha256'] != base_digest:
                raise ValueError(f'Attention {kind} identity mismatch: {root}')
            manifests[kind] = manifest
        k = manifests['spatial']['config']['neighbors']
        count, frames, _ = self.embeddings.shape
        self.neighbors = torch.empty((count,frames,k), dtype=torch.int16, device=device)
        self.positions_A = torch.empty((count,frames,3), dtype=torch.float32, device=device)
        self.box_lengths_A = torch.empty((len(dataset.records),frames,3), dtype=torch.float32, device=device)
        self.base_centers = torch.empty(count,dtype=torch.long,device=device)
        self.source_rows = torch.tensor(self.shards,dtype=torch.long,device=device)
        records = {kind:{r['directory']:r for r in manifest['records']} for kind,manifest in manifests.items()}
        base = 0
        for source, record in enumerate(dataset.records):
            name = record['directory']; n = record['centers']
            if n > 32767:
                raise ValueError(f'Attention resident indices require int16-addressable source centers: {name}')
            paths = {'spatial':dataset.spatial_root/name, 'geometry':dataset.geometry_root/name}
            for kind, files in [('spatial',('neighbors.npy',)),('geometry',('center_positions_A.npy','box_lengths_A.npy'))]:
                for file in files:
                    if sha256(paths[kind]/file) != records[kind][name]['checksums'][file]:
                        raise ValueError(f'Attention input changed: {paths[kind]/file}')
            neighbors = np.load(paths['spatial']/'neighbors.npy')
            if neighbors.shape != (n,frames,k) or neighbors.dtype != np.uint16 or neighbors.max() >= n:
                raise ValueError(f'Invalid same-source neighbor indices: {paths["spatial"]}')
            self.neighbors[base:base+n].copy_(torch.from_numpy(neighbors.astype(np.int16)))
            positions = np.load(paths['geometry']/'center_positions_A.npy')
            boxes = np.load(paths['geometry']/'box_lengths_A.npy')
            if positions.shape != (n,frames,3) or boxes.shape != (frames,3) or positions.dtype != np.float32 or boxes.dtype != np.float32:
                raise ValueError(f'Invalid center geometry arrays: {paths["geometry"]}')
            self.positions_A[base:base+n].copy_(torch.from_numpy(positions))
            self.box_lengths_A[source].copy_(torch.from_numpy(boxes))
            self.base_centers[base:base+n] = base
            base += n
        size = sum(getattr(self,name).numel()*getattr(self,name).element_size() for name in self.tensor_names)/2**30
        print(f'Resident attention inputs: {size:.3f} GiB on {device}',flush=True)

    def __getitem__(self, indices):
        batch = super().__getitem__(indices)
        rows = torch.as_tensor(indices,device=self.embeddings.device)
        batch['spatial_centers'] = rows//self.windows
        batch['spatial_columns'] = (self.source.history_skip+(rows%self.windows)[:,None]*self.source.stride
                                    +self.offsets[:self.source.history_steps])
        batch['spatial_store'] = self
        return batch

    def neighbor_batch(self, centers, columns):
        return gather_neighbors(self.embeddings,self.neighbors,self.positions_A,self.box_lengths_A,
                                centers,columns,self.base_centers,self.source_rows)


class AttentionResidentLoader:
    def __init__(self, dataset, batch_size, shuffle, seed, device):
        self.dataset = AttentionResidentWindows(dataset,device)
        generator = torch.Generator().manual_seed(seed)
        sampler = RandomSampler(dataset,generator=generator) if shuffle else SequentialSampler(dataset)
        self.sampler = BatchSampler(sampler,batch_size,drop_last=False)

    def __iter__(self):
        for indices in self.sampler:
            yield self.dataset[indices]

    def __len__(self):
        return len(self.sampler)


class AttentionSource:
    """The local physical assay retains all cached centers while querying labeled rows."""
    def __init__(self, embeddings, neighbors, positions_A, box_lengths_A):
        self.embeddings = embeddings
        self.neighbors = neighbors
        self.positions_A = positions_A
        self.box_lengths_A = box_lengths_A[None]
        self.base_centers = torch.zeros(len(embeddings),device=embeddings.device,dtype=torch.long)
        self.source_rows = torch.zeros_like(self.base_centers)

    def neighbor_batch(self, centers, columns):
        return gather_neighbors(self.embeddings,self.neighbors,self.positions_A,self.box_lengths_A,
                                centers,columns,self.base_centers,self.source_rows)
