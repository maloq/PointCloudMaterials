"""Memory-mapped data access with separate scientific inputs, labels and audit metadata."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from src.project_runtime.paths import resolve_path
from .protocol import ROLES, assert_prediction_rows, audit_sources, digest, sha


def read_release(root):
    root = resolve_path(root)
    seal = json.loads((root / 'manifest.json').read_text())
    if seal['state'] != 'complete':
        raise ValueError('Dataset preparation is not complete')
    for name, checksum in seal['files'].items():
        if sha(root / name) != checksum:
            raise ValueError(f'Changed fixed release contract: {name}')
    plan = json.loads((root / 'plan.json').read_text())
    if digest({k: v for k, v in plan.items() if k != 'identity'}) != seal['identity']:
        raise ValueError('Fixed release identity mismatch')
    audit_sources(plan['sources'], plan['config']['expected_source_counts'], plan['config']['centers_per_source'])
    return root, plan


def verify_release(root):
    """Offline integrity audit; performs no training, simulation or online logging."""
    root, plan = read_release(root)
    for section in ('benchmark', 'structural'):
        manifest = json.loads((root / section / 'manifest.json').read_text())
        if manifest['identity'] != plan['identity']:
            raise ValueError(f'Changed {section} identity')
        for source in manifest['sources']:
            for name, checksum in source['files'].items():
                path = root / section / 'sources' / str(source['id']) / name
                if sha(path) != checksum:
                    raise ValueError(f'Changed observation: {path}')
    manifest = json.loads((root / 'benchmark/manifest.json').read_text())
    for name in ('population', 'legacy_order'):
        suffix = '.npz' if name == 'population' else '.npy'
        if sha(root / 'benchmark' / (name + suffix)) != manifest[name + '_sha256']:
            raise ValueError(f'Changed benchmark {name}')
    return plan['identity']


class PatchAccess:
    @lru_cache(maxsize=32)
    def array(self, section, sid, name):
        return np.load(self.root / section / 'sources' / str(sid) / f'{name}.npy', mmap_mode='r')

    def patch(self, section, sid, name, index):
        value = self.array(section, sid, name)
        centers = self.plan['config']['centers_per_source']
        return np.array(value[index // centers, index % centers], copy=True)

    def geometry(self, positions):
        return dict(positions=positions, atomic_numbers=np.full(len(positions), 13, dtype=np.int64),
                    mask=np.linalg.norm(positions, axis=-1) < self.plan['config']['radius_A'])


class OnsetDataset(PatchAccess):
    def __init__(self, root, role, domain='hot', motion=False, track='all64'):
        self.root, self.plan = read_release(root)
        if role not in ROLES or domain not in ('hot', 'cold') or track not in ('all64', 'legacy16'):
            raise ValueError('Require a declared role, hot/cold domain and all64/legacy16 track')
        if motion and domain != 'hot':
            raise ValueError('Velocities belong to observed MD, not minimized configurations')
        manifest = json.loads((self.root / 'benchmark/manifest.json').read_text())
        if sha(self.root / 'benchmark/population.npz') != manifest['population_sha256']:
            raise ValueError('Changed fixed benchmark population')
        with np.load(self.root / 'benchmark/population.npz') as arrays:
            self.population = {k: arrays[k] for k in arrays.files}
        if track == 'all64':
            indices = np.arange(len(self.population['source']))
        else:
            path = self.root / 'benchmark/legacy_order.npy'
            if sha(path) != manifest['legacy_order_sha256']:
                raise ValueError('Changed legacy evaluation row order')
            indices = np.load(path)
        self.indices = indices[self.population['role'][indices] == role]
        self.domain, self.motion, self.role = domain, motion, role
        self.sample_ids = self.population['sample_id'][self.indices]
        assert_prediction_rows(self.sample_ids, self.sample_ids)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.indices[index]
        sid, patch = self.population['source'][row], self.population['patch_index'][row]
        inputs = self.geometry(self.patch('benchmark', sid, self.domain, patch))
        if self.motion:
            inputs['velocities'] = self.patch('benchmark', sid, 'velocity', patch)
        # IDs and targets are separate from the only dictionary passed to the encoder.
        return dict(inputs=inputs, event=self.population['event'][row], sample_id=self.sample_ids[index])

    def validate_predictions(self, sample_ids):
        assert_prediction_rows(self.sample_ids, sample_ids)


class StructuralDataset(PatchAccess):
    def __init__(self, root, role='train', paired=False):
        self.root, self.plan = read_release(root)
        if role not in ('train', 'selection'):
            raise ValueError('Structural pretraining has train/selection roles only; no held-out access')
        manifest = json.loads((self.root / 'structural/manifest.json').read_text())
        self.sources = [s for s in manifest['sources'] if s['role'] == role]
        allowed = {s['id'] for s in self.plan['sources'] if s['role'] == role}
        if {s['id'] for s in self.sources} != allowed:
            raise ValueError('Structural source list disagrees with the fixed source split')
        self.ends = np.cumsum([s['paired_rows' if paired else 'rows'] for s in self.sources])
        self.paired, self.role = paired, role

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        source_index = np.searchsorted(self.ends, index, side='right')
        offset = index - (self.ends[source_index - 1] if source_index else 0)
        sid = self.sources[source_index]['id']
        if self.paired:
            return dict(inputs=self.geometry(self.patch('benchmark', sid, 'hot', offset)),
                        teacher=self.geometry(self.patch('benchmark', sid, 'cold', offset)))
        return dict(inputs=self.geometry(self.patch('structural', sid, 'positions', offset)))
