"""Immutable encoder-treatment campaign and source/sample contract."""
import json
from pathlib import Path
from src.project_runtime.paths import resolve_path
from src.research.structural_state.common import sha,digest,write_json
from src.research.supervised_onset.tracking import require_online


class Study:
    def __init__(self,path):
        self.path=Path(path).resolve();self.config=json.loads(self.path.read_text());c=self.config
        if c['protocol']!='encoder_context_epochs_v1':raise ValueError('Unknown encoder comparison protocol')
        if c['methods']!=['scratch','physical','vicreg','epi_variance']:raise ValueError('Require the declared four treatments')
        if c['pretraining']['epochs']!=12 or c['supervised_epochs']<12 or c['predictor_epochs']<12:
            raise ValueError('Require twelve structural epochs and at least twelve complete supervised/predictor epochs')
        if c['variants']!=['vector_messages','harmonic_hierarchy']:raise ValueError('Undeclared context predictors')
        require_online(c['wandb'])
        self.root=resolve_path(c['output']);self.technical=self.root/'technical';self.technical.mkdir(parents=True,exist_ok=True)
        self.cache=resolve_path(c['cache'])

    def bind(self):
        from src.data.fixed_cohort.dataset import read_release
        root,plan=read_release(self.config['fixed_dataset']['root'])
        if plan['identity']!=self.config['fixed_dataset']['identity']:raise ValueError('Release identity differs')
        repo=Path(__file__).resolve().parents[3]
        paths=[]
        for package in ('encoder_context','equivariant_context','supervised_onset','trajectory_stability'):
            paths+=list((repo/'src/research'/package).glob('*.py'))
        paths+=list((repo/'src/data/fixed_cohort').glob('*.py'))
        paths += [repo/p for p in ('src/research/mace_epi/objective.py',
            'src/training_methods/neighborhood_jepa/regularization/objective.py',
            'src/training_methods/structural_pretraining/objective.py',
            'src/models/encoders/spatial_mace.py','src/models/encoders/graph_bank.py',
            'src/models/encoders/mace_backend.py','src/research/local_predictability/metrics.py',
            'src/research/robust_onset/metrics.py','src/research/structural_state/data.py')]
        record=dict(config=self.config,release_identity=plan['identity'],
            implementation={str(p.relative_to(repo)):sha(p) for p in paths},
            recipes={str(p.relative_to(self.path.parent)):sha(p) for p in self.path.parent.rglob('*.json')})
        self.identity=digest(record);path=self.technical/'identity.json'
        if path.exists():
            if json.loads(path.read_text())!=record:raise ValueError('Frozen encoder study changed; use a fresh output')
        else:write_json(path,record)
        return self.identity
