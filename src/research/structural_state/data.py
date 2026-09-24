"""Verified reuse of paired full-radius patches and original-MD physical labels."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from src.project_runtime.paths import resolve_path
from src.training_methods.bcr.probes import descriptors
from src.research.forecast_crystallization.local_metrics import first_sustained_onset, risk_windows
from .common import sha, digest, write_json

DOMAINS = ('observed', 'relaxed')
BLOCKS = {'radial': slice(0, 17), 'l2': slice(17, 53), 'l4': slice(53, 89)}


def geometry(targets, domain):
    return np.concatenate((targets[domain + '_radial'], targets[domain + '_rich'][:, 36:108]), axis=1)


def splits(records):
    result = {s: np.array([i for i, r in enumerate(records) if r['split'] == s], dtype=np.int64)
              for s in ('fit', 'tune', 'development')}
    roots = [{records[i]['root'] for i in ids} for ids in result.values()]
    if any(not len(v) for v in result.values()) or any(roots[i] & roots[j] for i in range(3) for j in range(i)):
        raise ValueError('Require three nonempty, ancestry-disjoint source splits')
    keys = [(r['root'], r['frame'], r['center_atom_id']) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate root/frame/center observations')
    return result


def graph_arrays(patches, cutoff):
    edges = []
    for i, x in enumerate(patches):
        if not np.isfinite(x).all() or not np.array_equal(x[0], np.zeros(3)):
            raise ValueError(f'Invalid centered geometry at patch {i}')
        # Match the original encoder's float32 cutoff, directed edges and order.
        candidates = cKDTree(x).query_pairs(cutoff + 1e-5, output_type='ndarray')
        distance = np.linalg.norm(x[candidates[:, 1]] - x[candidates[:, 0]], axis=1)
        pairs = candidates[distance < cutoff]
        if np.any(distance == 0) or not len(pairs):
            raise ValueError(f'Coincident atoms or empty spatial graph at patch {i}')
        directed = np.concatenate((pairs, pairs[:, ::-1]))
        directed = directed[np.lexsort((directed[:, 1], directed[:, 0]))]
        edges.append(directed.T.astype(np.int32))
    return dict(positions=np.concatenate(patches).astype(np.float32),
                offsets=np.r_[0, np.cumsum([len(p) for p in patches])],
                edges=np.concatenate(edges, axis=1),
                edge_offsets=np.r_[0, np.cumsum([e.shape[1] for e in edges])])


def prepare(study):
    config = study.config
    paired = resolve_path(config['paired_cache'])
    selection_path = resolve_path(config['paired_selection'])
    parent = json.loads((paired / 'manifest.json').read_text())
    selection = json.loads(selection_path.read_text())
    if parent['state'] != 'complete' or parent['identity'] != config['paired_identity'] or selection['identity'] != parent['identity']:
        raise ValueError('Require the pinned, completed 45-root paired audit')
    if parent['radius_A'] != config['encoder']['radius']:
        raise ValueError('Input radius and native encoder support differ')
    input_receipt = dict(paired_sha256=sha(paired / 'manifest.json'), selection_sha256=sha(selection_path),
                         assay_plan_sha256=sha(resolve_path(config['assay_plan'])),
                         producer_sha256=sha(__file__), descriptors_sha256=sha('src/training_methods/bcr/probes.py'),
                         cutoff=config['encoder']['cutoff'], radius=config['encoder']['radius'],
                         future_ps=config['future_ps'])
    manifest_path = study.cache / 'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous['inputs'] != input_receipt:
            raise ValueError('Prepared structural-state inputs/producer changed; use a fresh cache')
        for name, checksum in previous['files'].items():
            if sha(study.cache / name) != checksum:
                raise ValueError(f'Changed structural-state cache file: {name}')
        return previous
    study.cache.mkdir(parents=True, exist_ok=True)
    patches = {d: [] for d in DOMAINS}
    records = []
    for cell in parent['cells']:
        path = paired / 'cells' / cell['key'] / 'patches.npz'
        if sha(path) != cell['patches_sha256'] or cell['identity'] != parent['identity']:
            raise ValueError(f'Paired patch provenance mismatch: {path}')
        with np.load(path) as data:
            for domain in DOMAINS:
                x, offsets = data[domain + '_positions'], data[domain + '_offsets']
                patches[domain].extend([x[offsets[i]:offsets[i+1]].copy() for i in range(len(offsets)-1)])
            np.testing.assert_array_equal(data['center_atom_ids'], cell['center_atom_ids'])
            records.extend(dict(root=cell['root'], source=cell['source'], frame=cell['frame'],
                                center_atom_id=int(i), split=cell['split'], temperature_K=cell['temperature_K'])
                           for i in data['center_atom_ids'])
    split = splits(records)
    if [len({records[i]['root'] for i in split[s]}) for s in split] != [25, 5, 15]:
        raise ValueError('Expected the frozen 25 fitting / 5 tuning / 15 development roots')
    n = len(records)
    targets = {}
    for domain in DOMAINS:
        values, cov = descriptors(patches[domain], parent['radius_A'])
        targets.update({domain + '_' + k: v.astype(np.float32) for k, v in values.items()})
        targets[domain + '_covariates'] = cov.astype(np.float32)
        np.savez(study.cache / f'{domain}-graphs.npz', **graph_arrays(patches[domain], config['encoder']['cutoff']))
    # These labels are produced independently of our training descriptors. They
    # never enter encoder optimization, except the current coarse PTM stratum.
    plan = json.loads(resolve_path(config['assay_plan']).read_text())
    source_by_id = {s['id']: s for s in selection['sources']}
    plan_by_id = {s['id']: s for s in plan['sources']}
    ids = np.array([r['source'] for r in records])
    targets.update(current_order=np.empty((n, 8), np.float32), phase=np.empty(n, np.int64),
                   at_risk=np.empty(n, bool), event_bin=np.full(n, -1, np.int64), delay_ps=np.empty(n, np.float32))
    for lag in config['future_ps']:
        targets[f'future_order_{lag:g}'] = np.empty((n, 8), np.float32)
    assay_hashes = {}
    for sid in np.unique(ids):
        source = source_by_id[int(sid)]
        original = plan_by_id[int(sid)]
        if source['lineage'] != original['lineage'] or source['manifest_sha256'] != original['manifest_sha256']:
            raise ValueError(f'Original-MD ancestry mismatch: {sid}')
        path = resolve_path(plan['config']['assay_cache']) / original['shard']
        checksum = sha(path)
        if checksum != original['shard_sha256']:
            raise ValueError(f'Original-MD physical assay changed: {sid}')
        assay_hashes[str(sid)] = checksum
        ix = np.flatnonzero(ids == sid)
        frames = np.array([records[i]['frame'] for i in ix])
        centers = np.array([records[i]['center_atom_id'] for i in ix])
        with np.load(path) as data:
            np.testing.assert_array_equal(data['atom_ids'], original['center_atom_ids'])
            ci = np.searchsorted(data['atom_ids'], centers)
            np.testing.assert_array_equal(data['atom_ids'][ci], centers)
            np.testing.assert_allclose(data['times_ps'], np.arange(801) * .75, atol=1e-8, rtol=0)
            if np.max(frames) + 16 + 2 >= len(data['times_ps']):
                raise ValueError(f'Insufficient 12 ps onset-confirmation follow-up: {sid}')
            crystal = np.isin(data['labels'], [1, 2, 3])
            onset = first_sustained_onset(crystal, 3)
            risk = risk_windows(crystal, onset, np.unique(frames), 3)
            active = risk[ci, np.searchsorted(np.unique(frames), frames)]
            delay = (onset[ci] - frames) * .75
            targets['current_order'][ix] = data['order'][ci, frames]
            targets['phase'][ix] = crystal[ci, frames]
            targets['at_risk'][ix] = active
            targets['delay_ps'][ix] = delay
            targets['event_bin'][ix[active]] = np.searchsorted([.75, 3., 6., 9., 12.], delay[active])
            for lag in config['future_ps']:
                offset = int(round(lag / .75))
                if offset * .75 != lag:
                    raise ValueError(f'Future horizon is not on the actual saved cadence: {lag}')
                targets[f'future_order_{lag:g}'][ix] = data['order'][ci, frames + offset]
    if any(not np.isfinite(v).all() for v in targets.values()):
        raise ValueError('Nonfinite fixed targets')
    np.savez(study.cache / 'targets.npz', **targets)
    write_json(study.cache / 'records.json', records)
    fit_patches = [patches['observed'][i] for i in split['fit']]
    manifest = dict(state='complete', inputs=input_receipt, rows=n,
                    split_rows={k: len(v) for k, v in split.items()}, assay_sha256=assay_hashes,
                    n_ref=float(np.mean([len(p) for p in fit_patches])),
                    d0=float(np.median([np.linalg.norm(p[1:], axis=1).min() for p in fit_patches])),
                    radius_A=parent['radius_A'], potential_sha256=parent['potential_sha256'],
                    parent_identity=parent['identity'],
                    files={p.name: sha(p) for p in study.cache.iterdir() if p.suffix in ('.npz', '.json')},
                    target_training='observed/relaxed radial17 + moment l2/l4; all other targets evaluation only',
                    precision=parent['precision'], new_simulations=0,
                    onset_counts={s: dict(at_risk=int(targets['at_risk'][v].sum()),
                        events_12ps=int(((targets['event_bin'][v] >= 0) & (targets['event_bin'][v] < 5)).sum()))
                                  for s, v in split.items()})
    manifest['identity'] = digest(manifest)
    write_json(manifest_path, manifest)
    return manifest


class Corpus:
    def __init__(self, study):
        self.manifest = json.loads((study.cache / 'manifest.json').read_text())
        for name, checksum in self.manifest['files'].items():
            if sha(study.cache / name) != checksum:
                raise ValueError(f'Cache checksum mismatch: {name}')
        self.records = json.loads((study.cache / 'records.json').read_text())
        self.split = splits(self.records)
        with np.load(study.cache / 'targets.npz') as data:
            self.targets = dict(data)
        self.geometry = {d: geometry(self.targets, d) for d in DOMAINS}
        self.scalers = {}
        for domain, values in self.geometry.items():
            train = values[self.split['fit']].astype(np.float64)
            self.scalers[domain] = dict(mean=train.mean(0).astype(np.float32),
                                        scale=train.std(0).clip(1e-6).astype(np.float32))


class PairStream:
    """Same stream in every arm: balanced anchor, then same-T/current-PTM partner.

    Partners come from a different fitting root. No future labels enter sampling.
    """
    def __init__(self, records, fit, phase, seed):
        self.rng = np.random.default_rng(seed)
        self.records = records
        self.roots = {}
        self.strata = {}
        for i in fit:
            r = records[i]
            self.roots.setdefault(r['root'], {}).setdefault(r['frame'], []).append(int(i))
            self.strata.setdefault((r['temperature_K'], int(phase[i])), {}).setdefault(r['root'], []).append(int(i))
        self.phase = phase
        for key, roots in self.strata.items():
            if len(roots) < 2:
                raise ValueError(f'No cross-root relational control in current-condition stratum {key}')
        self.exposures = 0

    def draw(self, count):
        if count % 2:
            raise ValueError('Pair sampling requires an even batch size')
        result = []
        for _ in range(count // 2):
            root = self.rng.choice(list(self.roots))
            frame = self.rng.choice(list(self.roots[root]))
            first = int(self.rng.choice(self.roots[root][frame]))
            r = self.records[first]
            partners = self.strata[(r['temperature_K'], int(self.phase[first]))]
            other = self.rng.choice([p for p in partners if p != root])
            result.extend((first, int(self.rng.choice(partners[other]))))
        self.exposures += count
        return np.array(result, dtype=np.int64)

    def state_dict(self):
        return dict(rng=self.rng.bit_generator.state, exposures=self.exposures)

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state['rng']
        self.exposures = state['exposures']
