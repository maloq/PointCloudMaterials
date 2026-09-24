"""Immutable study identities and atomic research artifacts."""
import hashlib
import json
from pathlib import Path
import time

import torch

from src.project_runtime.paths import resolve_path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def remaining(deadline, reserve=120):
    if deadline is not None and time.time() >= deadline - reserve:
        raise TimeoutError('Allocation checkpoint reserve reached')


def implementation():
    paths = list(Path(__file__).parent.glob('*.py')) + [
        Path('src/training_methods/bcr/model.py'), Path('src/training_methods/bcr/data.py'),
        Path('src/training_methods/bcr/probes.py'), Path('src/models/encoders/mace_causal.py'),
        Path('src/models/encoders/mace_backend.py'), Path('src/research/bcr_followup/readouts.py'),
        Path('src/research/forecast_crystallization/local_metrics.py'),
        Path('src/research/local_predictability/metrics.py')]
    paths += [Path('src/research/liquid_geometry/metrics.py'),Path('src/research/structural_state_onset_review.py')]
    base = Path(__file__).resolve().parents[3]
    return {str(p.resolve().relative_to(base)): sha(p) for p in paths}


class Study:
    def __init__(self, path):
        self.config_path = Path(path).resolve()
        self.config = json.loads(self.config_path.read_text())
        if self.config['protocol'] not in ('fixed_geometry_relaxed_relation_v2', 'fixed_geometry_future_relation_v3', 'fixed_geometry_parameter_search_v4'):
            raise ValueError('Require structural-state v2/v3/v4; reproduce v1 from frozen code')
        if self.config['protocol'] == 'fixed_geometry_future_relation_v3':
            c = self.config
            if c['dynamics']['lag_ps'] != 9. or 9. not in c['future_ps']:
                raise ValueError('This predeclared factorial uses the cached9ps horizon')
            if any(a['input'] != 'relaxed' or a['relaxed_weight'] != 0 for a in c['arms']):
                raise ValueError('The v3 factorial holds relaxed input domain fixed')
            if any(a['current_weight'] != .25 for a in c['arms']):
                raise ValueError('Current-order supervision must be identical in every arm')
            if len(c['arms']) != 4 or {(a['relation_weight'], a['future_weight']) for a in c['arms']} != {(0.,0.),(.1,0.),(0.,.25),(.1,.25)}:
                raise ValueError('Require the four predeclared distance/future combinations')
        if self.config['protocol'] == 'fixed_geometry_parameter_search_v4':
            c = self.config
            if c['dynamics']['lag_ps'] != 9. or len(c['arms']) != 4:
                raise ValueError('Parameter search requires the four declared learning-rate/distance arms')
            combinations = {(a['encoder_lr'], a['relation_weight']) for a in c['arms']}
            if combinations != {(1e-5, 0.), (1e-4, 0.), (1e-5, 1.), (1e-4, 1.)}:
                raise ValueError(f'Changed v4 factorial: {combinations}')
            if any((a['input'], a['relaxed_weight'], a['current_weight'], a['future_weight']) !=
                   ('relaxed', 0., .25, 0.) for a in c['arms']):
                raise ValueError('v4 holds relaxed input and current-order supervision fixed; no future loss')
        self.root = resolve_path(self.config['output'])
        self.technical = self.root / 'technical'
        self.technical.mkdir(parents=True, exist_ok=True)
        self.cache = resolve_path(self.config['cache'])

    def bind(self):
        manifest = self.cache / 'manifest.json'
        receipt = dict(config=self.config, data_sha256=sha(manifest), implementation=implementation())
        self.identity = digest(receipt)
        path = self.technical / 'identity.json'
        if path.exists() and json.loads(path.read_text()) != receipt:
            raise ValueError(f'Study implementation/configuration/data changed: {path}')
        if not path.exists():
            write_json(path, receipt)
        return self.identity

    def arm(self, name):
        return next(a for a in self.config['arms'] if a['name'] == name)
