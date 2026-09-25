"""Frozen input/configuration/source identities for the new scientific protocol."""
import json
from pathlib import Path
from src.project_runtime.paths import resolve_path
from src.research.structural_state.common import sha, digest, write_json, save_checkpoint, remaining


class Study:
    protocol = 'robust_onset_v1'
    queue_module = 'src.research.robust_onset.queue'
    metric_family = 'robust_onset'

    def __init__(self, path):
        self.config_path = Path(path).resolve()
        self.config = json.loads(self.config_path.read_text())
        if self.config['protocol'] != self.protocol:
            raise ValueError(f'Require {self.protocol}')
        from .metrics import horizon_index
        horizon_index(self.config['primary_horizon_ps'])
        self.root = resolve_path(self.config['output'])
        self.technical = self.root/'technical'
        self.technical.mkdir(parents=True, exist_ok=True)
        self.cache = resolve_path(self.config['cache'])
        self.augmented = resolve_path(self.config['augmented_cache'])

    def arm(self, name):
        return next(a for a in self.config['arms'] if a['name'] == name)

    def prepare(self):
        from .data import prepare
        return prepare(self)

    def make_model(self, encoder_config, arm, temperatures):
        from .model import Model
        return Model(encoder_config, arm['tensor_pool'], temperatures)

    def graph_arrays(self, arm):
        import numpy as np
        with np.load(self.cache/f'{arm["input"]}-graphs.npz') as a:
            return dict(a)

    def noise_patches(self, arm, arrays):
        from .data import patches
        return patches(arrays)

    def identity_extras(self):
        return {}

    def additional_diagnostics(self, model, corpus, features, device, deadline):
        return {}

    def bind(self):
        base = Path(__file__).resolve().parents[3]
        files = list((base/'src/research/robust_onset').glob('*.py'))
        files += list((base/'src/research/structural_state').glob('*.py'))
        files += [base/p for p in ['src/training_methods/bcr/model.py', 'src/training_methods/bcr/data.py',
            'src/models/encoders/mace_causal.py','src/models/encoders/mace_backend.py',
            'src/research/local_predictability/metrics.py','src/research/trajectory_stability/spectrum.py']]
        receipt = dict(config=self.config, cache=sha(self.cache/'manifest.json'),
            augmentations=sha(self.augmented/'manifest.json'),
            implementation={str(p.relative_to(base)):sha(p) for p in files},
            extensions=self.identity_extras())
        self.identity = digest(receipt)
        path = self.technical/'identity.json'
        if path.exists() and json.loads(path.read_text()) != receipt:
            raise ValueError('Study changed: use a new output to preserve the frozen experiment')
        write_json(path, receipt)
        return self.identity
