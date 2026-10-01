"""One sealed target map, fitted exclusively to historical training sources."""
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.project_runtime.paths import resolve_path
from src.research.shooting_laws.common import read, plan
from src.research.shooting_laws.data import load as shooting_load
from src.research.shooting_laws.features import bank


def output(c):
    return resolve_path(c['output'])


def cache(c):
    return resolve_path(c['cache'])


def moments(x, w, floor=1e-5):
    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    mean = w @ np.asarray(x, dtype=np.float64)
    scale = np.sqrt(w @ (x - mean)**2).clip(floor)
    return mean, scale


def prepare(c):
    base = read(resolve_path(c['shooting_config']))
    data, manifest = shooting_load(base)
    p = plan(base)
    if (manifest['identity'] != c['shooting_identity'] or p['identity'] != c['shooting_plan_identity']
            or manifest['plan_identity'] != p['identity']):
        raise ValueError('Historical shooting population changed')
    if base['horizons_ps'] != c['target']['horizons_ps']:
        raise ValueError('Future observations must use exact existing 3/6/12 ps horizons')
    parents = p['parents']
    sources = np.array([parents[int(i)]['source'] for i in data['parent']])
    roles = np.array([parents[int(i)]['role'] for i in data['parent']])
    temperature = np.array([parents[int(i)]['temperature_K'] for i in data['parent']])
    weights = data['weights'].copy()
    for source in np.unique(sources):
        ix = sources == source
        if len(np.unique(roles[ix])) != 1:
            raise ValueError(f'Source crosses roles: {source}')
        weights[ix] /= weights[ix].sum()
    tr = roles == 'train'
    indices = [manifest['path_descriptors'].index(k) for k in c['target']['observables'][1:]]
    y = np.concatenate((data['crystalline_fraction'][..., None], data['future'][..., indices]), -1)
    y = y.reshape(len(sources), 12, 9).astype(np.float64)
    if data['positions'].shape != (len(y), 80, 3) or not np.all(data['positions'][:, 0] == 0):
        raise ValueError('Expected 80-atom center-relative observation with center at index 0')
    wshot = np.repeat(weights[tr] / 12, 12)
    flat = y[tr].reshape(-1, 9)
    mean, scale = moments(flat, wshot, c['target']['normalization_floor'])
    normalized = (y - mean) / scale
    rng = np.random.default_rng(c['target']['seed'])
    samples = normalized[tr].reshape(-1, 9)
    chosen = rng.choice(len(samples), c['target']['bandwidth_samples'], replace=True, p=wshot / wshot.sum())
    bandwidth = float(np.median(pdist(samples[chosen])))
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError(f'Degenerate training-only kernel bandwidth: {bandwidth}')
    count = c['target']['rff_features']
    omega = rng.normal(size=(9, count)) / bandwidth
    phase = rng.uniform(0, 2*np.pi, count)
    phi = np.concatenate((normalized, normalized**2, np.sqrt(2/count)*np.cos(normalized @ omega + phase)), -1)
    flat_phi = phi[tr].reshape(-1, 18 + count)
    center, _ = moments(flat_phi, wshot)
    coordinate_variance = np.average((flat_phi - center)**2, axis=0, weights=wshot)
    metric = np.empty(18 + count)
    traces = []
    for (lo, hi), block_weight in zip(((0, 9), (9, 18), (18, 18+count)), c['target']['block_weights'], strict=True):
        trace = float(coordinate_variance[lo:hi].sum())
        if trace <= 0 or not np.isfinite(trace):
            raise ValueError(f'Degenerate feature block {lo}:{hi}: {trace}')
        traces.append(trace)
        metric[lo:hi] = block_weight / trace
    features = ((phi - center) * np.sqrt(metric)).astype(np.float32)
    root = cache(c)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'manifest.json').exists():
        raise FileExistsError(f'Already sealed; use existing target bank: {root}')
    arrays = dict(positions=data['positions'], y=y.astype(np.float32), features=features,
        target=features.mean(1), descriptors=data['descriptors'], sources=sources,
        roles=roles, weights=weights, temperature=temperature, parent=data['parent'],
        atom_ids=data['atom_ids'], strata=data['strata'],
        extra_future=data['future'][..., 2:].mean(1).reshape(len(y), -1))
    for name in c['frozen_controls']:
        arrays[name] = bank(base, data, manifest, dict(bank=name))
    files = {}
    for name, value in arrays.items():
        if value.dtype.kind in 'fc' and not np.isfinite(value).all():
            raise FloatingPointError(f'Nonfinite {name}')
        np.save(root / f'{name}.npy', value)
        files[f'{name}.npy'] = sha(root / f'{name}.npy')
    np.savez(root / 'feature_map.npz', y_mean=mean, y_scale=scale, omega=omega, phase=phase,
             bandwidth=bandwidth, center=center, metric=metric, block_variance_trace=traces,
             bandwidth_training_shot_indices=chosen)
    files['feature_map.npz'] = sha(root / 'feature_map.npz')
    binding = dict(protocol=c['protocol'], target=c['target'], shooting_identity=manifest['identity'],
        plan_identity=p['identity'], source_files=manifest['files'], files=files,
        producer_sha256=sha(Path(__file__)), frozen_encoders=base['encoders'])
    record = dict(identity=digest(binding), binding=binding, files=files,
        observations=len(y), parent_count=len(parents), shots_per_parent=12,
        splits={role: dict(rows=int((roles == role).sum()), sources=len(np.unique(sources[roles == role])))
                for role in ('train', 'selection', 'test')},
        columns=manifest['columns'], strata=manifest['strata'],
        target_columns=[f'{name}@{h}ps' for h in c['target']['horizons_ps'] for name in c['target']['observables']],
        extra_future_columns=[f'{name}@{h}ps' for h in c['target']['horizons_ps'] for name in manifest['path_descriptors'][2:]],
        parents=[{k: v for k, v in item.items() if k != 'shots'} for item in parents])
    write_json(root / 'manifest.json', record)
    write_json(output(c) / 'technical/target-manifest.json', record)
    write_json(output(c) / 'technical/prediction-context.json', dict(
        track=c['track'], encoder_inputs=dict(coordinates='minimum-image Al Angstrom; fixed Al multiplier 1',
            atoms=80, radius_A=8, edge_cutoff_A=5, halo=None, constant_atom_channel=True,
            center_indicator=True, history=0, motion=False, conditions=[], relaxation=False, training_only_teachers=[]),
        predictor_inputs=dict(joint='z128 only', descriptors='442 present geometry descriptors',
            frozen=c['frozen_controls'], conditions=[], history=0, motion=False, relaxation=False),
        frozen_pretraining='Recorded VICReg/Epi Al64 train-only observed/relaxed pairs; no shooting source overlap per sealed plan',
        sampling='Present PTM strata; population inclusion weights normalized within source; equal source mass',
        continuation=p['protocol'],
        interpretation='Local observation and pooled 400/450/500K variable-cell parent population; not full-q fixed-condition sufficiency'))
    print(record['splits'], flush=True)
    return record


def load(c):
    root = cache(c)
    manifest = read(root / 'manifest.json')
    for filename, expected in manifest['files'].items():
        if sha(root / filename) != expected:
            raise ValueError(f'Changed baseline input: {filename}')
    data = {Path(name).stem: np.load(root / name, mmap_mode='r') for name in manifest['files'] if name.endswith('.npy')}
    return data, manifest
