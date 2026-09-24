"""Frozen identities, root splits and reusable checkpoint features."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from src.project_runtime.paths import resolve_path
from src.research.bcr_pilot.compare import checkpoint, load_model
from src.training_methods.bcr.data import balanced_subset, pack, corrupt
from src.training_methods.bcr.runtime import load_data, identity


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def remaining(deadline):
    if deadline is not None and time.time() >= deadline - 120:
        raise TimeoutError('Allocation reserve reached; resume this stage from its saved state')


def root_split(records):
    training = [i for i, r in enumerate(records) if r['split'] == 'train']
    development = [i for i, r in enumerate(records) if r['split'] == 'development']
    train_roots = sorted({records[i]['root'] for i in training})
    dev_roots = {records[i]['root'] for i in development}
    if len(train_roots) != 12 or len(dev_roots) != 6 or set(train_roots) & dev_roots:
        raise ValueError('Follow-up requires the frozen independent 12-train/6-development-root pilot')
    tuning = set(train_roots[-2:])
    return dict(train=training, fit=[i for i in training if records[i]['root'] not in tuning],
                tune=[i for i in training if records[i]['root'] in tuning], development=development,
                tuning_roots=sorted(tuning))


class Study:
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text())
        self.pilot = json.loads(resolve_path(self.config['pilot']).read_text())
        self.original = resolve_path(self.pilot['output'])
        self.root = resolve_path(self.config['output'])
        self.technical = self.root / 'technical'
        self.technical.mkdir(parents=True, exist_ok=True)
        self.patches, self.manifest = load_data(resolve_path(self.pilot['data']))
        self.records = self.manifest['records']
        self.split = root_split(self.records)
        self.chosen = balanced_subset(self.records, self.split['development'], self.pilot['evaluation_anchors'])
        if self.config['evaluation_chunk'] != 16:
            raise ValueError('Retain the original comparison chunk=16 for exact replay of its corruption bank')
        source_files = [*Path('src/research/bcr_followup').glob('*.py'), *Path('src/training_methods/bcr').glob('*.py'),
                        Path('src/research/bcr_pilot/compare.py'), Path('src/research/bcr_pilot/data.py'),
                        Path('src/data/trajectories/lammps.py'), Path('src/data/trajectories/shooting.py'),
                        Path('docs/metrics/bcr_followup.md')]
        self.receipt = dict(config=self.config, data_identity=self.manifest['identity'],
                            relaxed_plan_sha256=file_hash(resolve_path(self.config['relaxed']['plan'])),
                            manifest_sha256=file_hash(resolve_path(self.pilot['data']) / 'manifest.json'),
                            implementation={str(p): file_hash(p) for p in source_files},
                            checkpoints={str(step): file_hash(checkpoint(self.original, 'bcr', step))
                                         for step in self.config['feature_steps']},
                            descriptors_sha256=file_hash(self.original / 'technical/descriptors.npz'),
                            tuning_roots=self.split['tuning_roots'],
                            evaluation_keys=[[self.records[i][k] for k in ('root', 'frame', 'center_atom_id')]
                                             for i in self.chosen])
        if self.config['vicreg_checkpoint'] is not None:
            self.receipt['vicreg_checkpoint_sha256'] = file_hash(resolve_path(self.config['vicreg_checkpoint']))
        self.identity = identity(self.receipt)
        path = self.technical / 'identity.json'
        if path.exists() and json.loads(path.read_text()) != self.receipt:
            raise ValueError(f'Follow-up inputs/configuration/implementation changed: {path}; use a new output')
        write_json(path, self.receipt)

    def model(self, step, device):
        return load_model(checkpoint(self.original, 'bcr', step), self.manifest, device)

    def features(self, step):
        path = self.technical / 'features' / f'{step:06d}.npz'
        with np.load(path) as data:
            if str(data['identity']) != self.identity:
                raise ValueError(f'Feature cache identity changed: {path}')
            return {k: data[k] for k in ('pooled', 'exported')}

    def descriptors(self):
        with np.load(self.original / 'technical/descriptors.npz') as data:
            return {k: data[k] for k in ('radial', 'angular', 'rich')}, data['covariates']


def extract(study, device, deadline=None):
    for step in study.config['feature_steps']:
        destination = study.technical / 'features' / f'{step:06d}.npz'
        if destination.exists():
            study.features(step)
            continue
        remaining(deadline)
        model = study.model(step, device)
        pooled, exported = [], []
        with torch.no_grad():
            for start in range(0, len(study.patches), 16):
                remaining(deadline)
                p = model.encoder.pooled(pack(study.patches[start:start+16], device))
                pooled.append(p.cpu().numpy())
                exported.append(model.encoder.readout(p).cpu().numpy())
        p, z = np.concatenate(pooled), np.concatenate(exported)
        old = np.load(study.original / f'technical/evaluations/{step:06d}/bcr-features.npy')
        np.testing.assert_allclose(z, old, atol=3e-5, rtol=2e-4,
                                   err_msg=f'Checkpoint {step} exported features differ from original assay')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.with_suffix('.tmp').open('wb') as stream:
            np.savez(stream, pooled=p, exported=z, identity=study.identity)
        destination.with_suffix('.tmp').replace(destination)
        print(f'Extracted step {step}: pooled {p.shape}, exported {z.shape}', flush=True)


def fixed_batch(patches, start, stop, level, draw, levels, d0, device, seed=731):
    """Replay the original evaluator's exact anchor keys, shape and corruption call."""
    clean = pack(patches[start:stop], device)
    positions, epsilons, sigmas = [], [], []
    for i in range(start, stop):
        one = {k: v[i-start:i-start+1] for k, v in clean.items()}
        noisy, epsilon, sigma, _ = corrupt(one, levels, d0,
            torch.Generator().manual_seed(seed + 100003*i + 1009*level + draw), torch.tensor([level]))
        positions.append(noisy['positions']); epsilons.append(epsilon); sigmas.append(sigma)
    return clean, dict(clean, positions=torch.cat(positions)), torch.cat(epsilons), torch.cat(sigmas)


def decode(decoder, noisy, code, sigma, d0):
    return decoder(noisy['positions'], noisy['species'], noisy['center'], noisy['mask'], code, torch.log(sigma/d0))
