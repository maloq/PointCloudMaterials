"""Explicit input identities and bounded scientific evaluation caches."""
import json
from pathlib import Path
from types import SimpleNamespace

from src.project_runtime.paths import resolve_path
from src.research.structural_state.common import sha, digest, write_json


def load(path):
    config = json.loads(Path(path).read_text())
    if config['protocol'] != 'native_mace_quality_v1':
        raise ValueError('Undeclared encoder-quality protocol')
    if config['external_inputs'] != []:
        raise ValueError('No explicit temperature/time inputs are permitted')
    for key in ('output', 'population_cache', 'reference', 'native_inputs', 'dense_inputs',
                'dense_observed', 'feature_cache'):
        config[key] = str(resolve_path(config[key]))
    for model in config['models']:
        for key in ('checkpoint', 'origin', 'producer'):
            model[key] = str(resolve_path(model[key]))
    return config


def bind(config):
    root = Path(config['output']); technical = root/'technical'
    technical.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[3]
    paths = list((repo/'src/research/encoder_quality').glob('*.py'))
    for family in ('supervised_onset', 'encoder_context', 'geoframe_evolution',
                   'encoder_parameter_search', 'trajectory_stability'):
        paths += list((repo/'src/research'/family).glob('*.py'))
    paths += [repo/p for p in ('src/models/encoders/spatial_mace.py',
        'src/models/encoders/mace_backend.py', 'src/models/encoders/graph_bank.py',
        'src/research/equivariant_context/cache.py', 'src/research/local_predictability/metrics.py')]
    record = dict(config=config, implementation={str(p.relative_to(repo)):sha(p) for p in paths},
                  population_sha256=sha(Path(config['population_cache'])/'population.npz'))
    identity = digest(record)
    dest = technical/'identity.json'
    if dest.exists() and json.loads(dest.read_text()) != record:
        raise ValueError('Evaluation identity changed; use a fresh output')
    write_json(dest, record)
    return identity


def corpus(config):
    import numpy as np
    root = Path(config['population_cache'])
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['fixed_dataset'] != config['fixed_dataset']:
        raise ValueError('Prediction release/sample track changed')
    pop = dict(np.load(root/'population.npz'))
    arrays = {d: {k: np.load(root/f'{d}-{k}.npy', mmap_mode='r')
                     for k in ('positions', 'offsets', 'edges', 'edge_offsets')}
              for d in ('hot', 'cold')}
    return SimpleNamespace(pop=pop, split={r:np.flatnonzero(pop['role']==r)
        for r in ('train','selection','calibration','test')}, arrays=arrays, manifest=manifest)


def static_coordinate_agreement(rebuilt, original):
    """Equal-distance neighbor ties may reorder the same unordered point set."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    if rebuilt.shape!=original.shape:raise ValueError('Static patch shapes differ')
    changed=np.flatnonzero(~np.isclose(rebuilt,original,atol=2e-6,rtol=2e-6).all((1,2)))
    aligned=rebuilt.copy()
    for i in changed:
        rows,cols=linear_sum_assignment(cdist(original[i],rebuilt[i]))
        aligned[i,rows]=rebuilt[i,cols]
    np.testing.assert_allclose(aligned,original,atol=2e-6,rtol=2e-6)
    return dict(reordered_patches=changed.tolist(),max_aligned_abs=float(abs(aligned-original).max()),
                comparison='Unordered point sets; minimum-cost bijection only for reordered patches')


def probe_study(config, identity, name, input_record):
    root = Path(config['output'])/'technical/models'/name
    technical = root/'technical'; technical.mkdir(parents=True, exist_ok=True)
    settings = dict(seed=config['seed'], branch='crystallization_supervised', arms=[], tracking_scope='diagnostic',
        baselines=config['probes'], wandb=dict(config['wandb'],
        display_name=f'Frozen MACE quality | {name}'), bootstrap=config['bootstrap'])
    study = SimpleNamespace(root=root, technical=technical, config=settings,
        identity=digest(dict(campaign=identity,model=name,inputs=input_record)))
    write_json(technical/'prediction-context.json', input_record)
    return study
