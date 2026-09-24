"""Frozen input/configuration/source identities for the new scientific protocol."""
import json
from pathlib import Path
from src.project_runtime.paths import resolve_path
from src.research.structural_state.common import sha, digest, write_json, save_checkpoint, remaining


class Study:
    def __init__(self, path):
        self.config_path = Path(path).resolve()
        self.config = json.loads(self.config_path.read_text())
        if self.config['protocol'] != 'robust_onset_v1':
            raise ValueError('Require robust_onset_v1')
        self.root = resolve_path(self.config['output'])
        self.technical = self.root/'technical'
        self.technical.mkdir(parents=True, exist_ok=True)
        self.cache = resolve_path(self.config['cache'])
        self.augmented = resolve_path(self.config['augmented_cache'])

    def arm(self, name):
        return next(a for a in self.config['arms'] if a['name'] == name)

    def bind(self):
        base = Path(__file__).resolve().parents[3]
        files = list((base/'src/research/robust_onset').glob('*.py'))
        files += list((base/'src/research/structural_state').glob('*.py'))
        files += [base/p for p in ['src/training_methods/bcr/model.py', 'src/training_methods/bcr/data.py',
            'src/models/encoders/mace_causal.py','src/models/encoders/mace_backend.py',
            'src/research/local_predictability/metrics.py','src/research/trajectory_stability/spectrum.py']]
        receipt = dict(config=self.config, cache=sha(self.cache/'manifest.json'),
            augmentations=sha(self.augmented/'manifest.json'),
            implementation={str(p.relative_to(base)):sha(p) for p in files})
        self.identity = digest(receipt)
        path = self.technical/'identity.json'
        if path.exists() and json.loads(path.read_text()) != receipt:
            raise ValueError('Study changed: use a new output to preserve the frozen experiment')
        write_json(path, receipt)
        return self.identity
