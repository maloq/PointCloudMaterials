"""Pack an existing fitting subset once; globally permute its rows each epoch."""
import fcntl
from pathlib import Path
import shutil

import numpy as np

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.project_runtime.paths import resolve_path
from .data import config
from .rich_multimaterial_data import CachedPatches


def pack(c):
    reference = resolve_path(c['fitting_reference']) / 'technical'
    root = resolve_path(c['loader']['packed_cache'])
    root.mkdir(parents=True, exist_ok=True)
    expected = config(reference / 'batch-plan.json')
    ids_path = reference / 'training-pool-row-ids.npy'
    transform = reference / 'target-standardization.npz'
    binding = dict(dataset=c['prepared_identity'], rows=c['data']['training_rows'],
                   subset_sha256=sha(ids_path), transform_sha256=sha(transform),
                   descriptor_manifest_sha256=sha(resolve_path(c['cache']) / 'manifest.json'))
    if (binding['subset_sha256'] != expected['subset_sha256'] or
            binding['transform_sha256'] != expected['transform_sha256']):
        raise ValueError('Reference fitting rows or standardization changed')
    identity = digest(binding)
    with (root / 'pack.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / 'manifest.json').exists():
            saved = config(root / 'manifest.json')
            if saved['identity'] != identity:
                raise ValueError('Packed cache belongs to another fitting subset')
            for name, checksum in saved['files'].items():
                if sha(root / name) != checksum:
                    raise ValueError(f'Changed packed file: {name}')
            return saved
        data = CachedPatches(c, 'train')
        ids = np.load(ids_path)
        if len(ids) != binding['rows'] or data.identity != binding['dataset']:
            raise ValueError('Fitting reference does not match requested release')
        # select verifies the shard hashes and recomputes moments of these exact rows.
        data.select(ids)
        with np.load(transform) as frozen:
            for name in ('mean', 'scale', 'active', 'loss_weight'):
                if not np.array_equal(frozen[name], getattr(data, name)):
                    raise ValueError(f'Packed fitting transform differs: {name}')
        arrays = {
            'positions': np.lib.format.open_memmap(root / 'positions.npy', mode='w+',
                                                  dtype=np.float32, shape=(len(ids), 80, 3)),
            'features': np.lib.format.open_memmap(root / 'features.npy', mode='w+',
                                                 dtype=np.float32, shape=(len(ids), len(data.mean))),
        }
        shards = []
        for i, shard in enumerate(data.shards):
            values = data._arrays(i)
            selected = shard['selected_indices']
            for name, dest in arrays.items():
                dest[data.starts[i]:data.ends[i]] = values[name][selected]
            # Row order and audit metadata stay identical to the original selected view.
            shards.append(dict(task=shard['task'], rows=int(shard['rows'])))
        for dest in arrays.values():
            dest.flush()
            dest._mmap.close()
        data.close()
        shutil.copy2(ids_path, root / 'training-pool-row-ids.npy')
        shutil.copy2(transform, root / 'target-standardization.npz')
        files = {name: sha(root / name) for name in
                 ('positions.npy', 'features.npy', 'training-pool-row-ids.npy', 'target-standardization.npz')}
        saved = dict(identity=identity, binding=binding, files=files, shards=shards,
                     material_rows=expected['material_rows'], pool_rows=expected['pool_rows'],
                     source_count=expected['source_count'], columns=data.columns)
        write_json(root / 'manifest.json', saved)
        return saved


class PackedPatches(CachedPatches):
    """Float32 positions and raw targets in RAM (~2.7 GiB); no shard I/O per batch."""

    def __init__(self, c):
        self.root = resolve_path(c['loader']['packed_cache'])
        m = config(self.root / 'manifest.json')
        if m['binding']['dataset'] != c['prepared_identity'] or m['binding']['rows'] != c['data']['training_rows']:
            raise ValueError('Packed release does not match configuration')
        self.identity = m['binding']['dataset']
        self.packed_identity = m['identity']
        self.role = 'train'
        self.columns = m['columns']
        self.shards = m['shards']
        self.ends = np.cumsum([s['rows'] for s in self.shards])
        self.starts = np.r_[0, self.ends[:-1]]
        for name, checksum in m['files'].items():
            if sha(self.root / name) != checksum:
                raise ValueError(f'Changed packed file: {self.root / name}')
        with np.load(self.root / 'target-standardization.npz') as a:
            for name in ('mean', 'scale', 'active', 'loss_weight'):
                setattr(self, name, a[name].copy())
        self.positions = np.load(self.root / 'positions.npy')
        self.features = np.load(self.root / 'features.npy')
        if self.positions.shape != (len(self), 80, 3) or self.features.shape != (len(self), len(self.mean)):
            raise ValueError('Packed array shapes differ from manifest')

    def select(self, ids):
        if not np.array_equal(ids, np.load(self.root / 'training-pool-row-ids.npy')):
            raise ValueError('Packed rows cannot be resampled')

    def batch(self, ids):
        ids = np.asarray(ids, np.int64)
        if ids.ndim != 1 or not len(ids) or ids.min() < 0 or ids.max() >= len(self):
            raise IndexError('Invalid packed row IDs')
        return dict(positions=self.positions[ids], target=(self.features[ids] - self.mean) / self.scale)

    def epoch(self, batch_size, epoch, seed, start_batch=0):
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(self))
        for step, begin in enumerate(range(0, len(order), batch_size)):
            if step >= start_batch:
                yield step, order[begin:begin + batch_size]

    def close(self):
        self.positions = self.features = None


def pack_pilot(c, destination, fraction):
    """Nested uniform draw from the packed fitting rows; refit only its moments."""
    source = resolve_path(c['loader']['packed_cache'])
    parent = config(source / 'manifest.json')
    root = resolve_path(destination)
    root.mkdir(parents=True, exist_ok=True)
    n = int(round(parent['binding']['rows'] * fraction))
    selected = np.sort(np.random.default_rng(c['seed'] + 97).choice(parent['binding']['rows'], n, replace=False))
    binding = dict(dataset=c['prepared_identity'], rows=n, parent=parent['identity'],
                   fraction=fraction, seed=c['seed'] + 97)
    identity = digest(binding)
    if (root / 'manifest.json').exists():
        saved = config(root / 'manifest.json')
        if saved['identity'] != identity:
            raise ValueError('Pilot subset definition changed')
        for name, checksum in saved['files'].items():
            if sha(root / name) != checksum:
                raise ValueError(f'Changed pilot packed file: {name}')
        return saved
    for name in ('positions', 'features'):
        raw = np.load(source / f'{name}.npy', mmap_mode='r')
        np.save(root / f'{name}.npy', raw[selected])
        raw._mmap.close()
    ids = np.load(source / 'training-pool-row-ids.npy')[selected]
    np.save(root / 'training-pool-row-ids.npy', ids)
    np.save(root / 'parent-row-ids.npy', selected)
    values = np.load(root / 'features.npy').astype(np.float64)
    mean, std = values.mean(0), values.std(0)
    active = std >= 1e-4
    families = np.array([col['family'] for col in parent['columns']])
    weight = np.zeros(len(mean), np.float32)
    for family in ('geometry', 'bond_order', 'cna', 'tda'):
        mask = (families == family) & active
        if not mask.any():
            raise ValueError(f'No active pilot targets in {family}')
        weight[mask] = 1 / (4 * mask.sum())
    np.savez(root / 'target-standardization.npz', mean=mean.astype(np.float32),
             scale=std.clip(1e-4).astype(np.float32), active=active, loss_weight=weight)
    shards, begin = [], 0
    for shard in parent['shards']:
        end = begin + shard['rows']
        lo, hi = np.searchsorted(selected, [begin, end])
        if hi > lo:
            shards.append(dict(task=shard['task'], rows=int(hi - lo)))
        begin = end
    files = {name: sha(root / name) for name in
             ('positions.npy', 'features.npy', 'training-pool-row-ids.npy',
              'parent-row-ids.npy', 'target-standardization.npz')}
    binding.update(subset_sha256=files['training-pool-row-ids.npy'],
                   transform_sha256=files['target-standardization.npz'])
    saved = dict(identity=identity, binding=binding, files=files, shards=shards,
                 material_rows={material: sum(s['rows'] for s in shards if s['task']['material'] == material)
                                for material in parent['material_rows']},
                 pool_rows=parent['pool_rows'], source_count=len({s['task']['source'] for s in shards}),
                 columns=parent['columns'])
    write_json(root / 'manifest.json', saved)
    return saved
