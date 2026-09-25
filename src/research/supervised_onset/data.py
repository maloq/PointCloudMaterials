"""Reuse the producer's paired assay, preserving original MD events and ancestry."""
import json
from pathlib import Path
import numpy as np

from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.support import REFERENCE_RADIUS
from src.research.structural_state.data import graph_arrays
from src.research.local_predictability.metrics import source_weights
from .common import sha, digest, write_json

ROLES = ('train', 'selection', 'calibration', 'test')
DOMAINS = ('hot', 'cold')
HORIZONS = np.array([.75, 3., 6., 9., 12.])


def audit_population(pop, sources):
    """Reject cross-role ancestors and mislabeled or duplicated prospective rows."""
    by_id = {s['id']: s for s in sources}
    if len(by_id) != len(sources) or len({s['lineage'] for s in sources}) != len(sources):
        raise ValueError('Assay requires one independent lineage per source')
    if set(np.unique(pop['role'])) != set(ROLES):
        raise ValueError('Require separate train, selection, calibration and test populations')
    if np.any(pop['delay'] <= 0) or not np.isfinite(pop['delay']).all():
        raise ValueError('Onset assay must contain only prospective, currently at-risk windows')
    if not np.array_equal(pop['event'], np.searchsorted(HORIZONS, pop['delay'])):
        raise ValueError('Producer event bins disagree with physical delay in ps')
    if len(np.unique(np.c_[pop['source'], pop['graph']], axis=0)) != len(pop['source']):
        raise ValueError('Duplicate source/graph observation in the population')
    for sid in np.unique(pop['source']):
        source = by_id[int(sid)]
        if set(pop['role'][pop['source'] == sid]) != {source.get('validation_role', source['split'])}:
            raise ValueError(f'Role mismatch for source {sid}')
    result = {}
    for role in ROLES:
        ids = np.flatnonzero(pop['role'] == role)
        result[role] = dict(windows=len(ids), sources=len(np.unique(pop['source'][ids])), events={})
        for horizon in (3., 6., 12.):
            positive = ids[pop['delay'][ids] <= horizon]
            if not len(positive) or len(positive) == len(ids):
                raise ValueError(f'{role} lacks both classes at {horizon} ps')
            result[role]['events'][str(horizon)] = dict(positives=len(positive),
                positive_sources=len(np.unique(pop['source'][positive])))
    return result


def prepare(study):
    if 'fixed_dataset' in study.config:
        from .fixed_data import prepare as prepare_fixed
        return prepare_fixed(study)
    c = study.config
    parent = resolve_path(c['parent_output']) / 'technical'
    plan = json.loads((parent / 'plan.json').read_text())
    pop = dict(np.load(parent / 'assay/population.npz'))
    counts = audit_population(pop, plan['sources'])
    receipt = dict(plan=sha(parent / 'plan.json'), population=sha(parent / 'assay/population.npz'),
                   producer=sha(Path(__file__)), cutoff_A=c['encoder']['cutoff'], radius_A=c['encoder']['radius'])
    manifest = study.cache / 'manifest.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved['inputs'] != receipt:
            raise ValueError('Prepared supervised data changed; use a new cache')
        for name, checksum in saved['files'].items():
            if sha(study.cache / name) != checksum:
                raise ValueError(f'Changed supervised cache: {name}')
        return saved
    study.cache.mkdir(parents=True, exist_ok=True)
    folder = resolve_path(plan['config']['cache']) / 'assay'
    source_receipts, stats = {}, {}
    # The old cache stores normalized coordinates. Convert explicitly back to A;
    # rebuild edges at the NEW physical cutoff, never reuse old scaled edges.
    factor = plan['config']['scale'] / REFERENCE_RADIUS
    for domain in DOMAINS:
        patches = [None] * len(pop['source'])
        for sid in np.unique(pop['source']):
            d = folder / domain / str(sid)
            complete = json.loads((d / 'complete.json').read_text())
            if complete['identity'] != plan['identity']:
                raise ValueError(f'Paired release mismatch: {d}')
            hashes = {k: sha(d / f'{k}.npy') for k in ('positions', 'offsets')}
            if any(hashes[k] != complete['hashes'][k] for k in hashes):
                raise ValueError(f'Paired coordinate checksum failed: {d}')
            source_receipts[f'{domain}/{sid}'] = hashes
            x, offsets = np.load(d / 'positions.npy'), np.load(d / 'offsets.npy')
            for row in np.flatnonzero(pop['source'] == sid):
                graph = pop['graph'][row]
                p = (x[offsets[graph]:offsets[graph+1]] * factor).astype(np.float32)
                p = p[np.linalg.norm(p, axis=1) < c['encoder']['radius']]
                if len(p) < 13 or np.any(p[0]) or not np.isfinite(p).all():
                    raise ValueError(f'Invalid centered physical patch: {domain}/{sid}/{graph}')
                patches[row] = p
        arrays = graph_arrays(patches, c['encoder']['cutoff'])
        for name, value in arrays.items():
            np.save(study.cache / f'{domain}-{name}.npy', value)
        fit = np.flatnonzero(pop['role'] == 'train')
        stats[domain] = dict(nodes=len(arrays['positions']), edges=arrays['edges'].shape[1],
            n_ref=float(np.mean([len(patches[i]) for i in fit])),
            d0=float(np.median([np.linalg.norm(patches[i][1:], axis=1).min() for i in fit])))
        descriptor = np.load(parent / f'assay/{domain}-descriptors.npy')
        if descriptor.shape != (len(pop['source']), 237) or not np.isfinite(descriptor).all():
            raise ValueError('Unexpected relaxed-assay descriptor schema')
        np.save(study.cache / f'{domain}-descriptors.npy', descriptor)
        source_receipts[f'{domain}-descriptors'] = sha(parent / f'assay/{domain}-descriptors.npy')
        print(json.dumps(dict(stage='prepared_domain', domain=domain, **stats[domain])), flush=True)
    # Frame indices refer to the ORIGINAL assay anchors, not the coarser quench grid.
    original = json.loads(resolve_path(plan['config']['assay_plan']).read_text())
    pop['frame'] = np.asarray(original['anchors'])[pop['rows'][:, 1]]
    by_id = {s['id']: s for s in plan['sources']}
    pop['atom'] = np.array([by_id[int(s)]['center_atom_ids'][i] for s, i in zip(pop['source'], pop['rows'][:, 2], strict=True)])
    np.savez(study.cache / 'population.npz', **pop)
    result = dict(state='complete', inputs=receipt, source_receipts=source_receipts, counts=counts,
        domains=stats, sources=[dict(id=s['id'], lineage=s['lineage'],
            role=s.get('validation_role', s['split']), manifest_sha256=s['manifest_sha256']) for s in plan['sources']],
        coordinate_units='Angstrom; producer normalized coordinates inverted by scale/REFERENCE_RADIUS',
        support='Existing paired nearest-80 observations, cropped at 8 A, no extra context or new MD',
        labels='Original MD sustained local onset, not relaxed-state crystallinity',
        test_status='Reused historical test sources, not a new untouched test',
        potential_sha256=plan['config']['potential_sha256'], new_simulations=0,
        files={p.name: sha(p) for p in study.cache.iterdir() if p.suffix in ('.npy', '.npz')})
    result['identity'] = digest(result)
    write_json(manifest, result)
    return result


class Corpus:
    def __init__(self, study):
        self.manifest = json.loads((study.cache / 'manifest.json').read_text())
        self.pop = dict(np.load(study.cache / 'population.npz'))
        self.split = {role: np.flatnonzero(self.pop['role'] == role) for role in ROLES}
        self.arrays = {d: {k: np.load(study.cache / f'{d}-{k}.npy', mmap_mode='r')
            for k in ('positions', 'offsets', 'edges', 'edge_offsets')} for d in DOMAINS}
        self.descriptors = {}
        self.scalers = {}
        if 'fixed_dataset' in study.config:
            if self.manifest['fixed_dataset'] != study.config['fixed_dataset']:
                raise ValueError('Graph corpus disagrees with the declared fixed dataset')
            return  # No descriptor baselines were prepared for the context comparison.
        fit = self.split['train']
        weights = source_weights(self.pop['source'][fit])
        for d in DOMAINS:
            x = np.load(study.cache / f'{d}-descriptors.npy')
            mean = weights @ x[fit].astype(float)
            scale = np.sqrt(weights @ (x[fit] - mean)**2).clip(1e-5)
            self.descriptors[d] = ((x - mean) / scale).astype(np.float32)
            self.scalers[d] = dict(mean=mean.tolist(), scale=scale.tolist())


def sampling_distribution(source, event, positive_fraction):
    """Event-enriched training with exact importance correction to natural risk."""
    p = source_weights(source)
    positive = event <= 2  # By 6 ps; both target horizons receive positive examples.
    if not positive.any() or positive.all() or not 0 <= positive_fraction < 1:
        raise ValueError('Require both classes and a mixture fraction in [0,1)')
    q = (1 - positive_fraction) * p + positive_fraction * p * positive / (p @ positive)
    return q, p / q
