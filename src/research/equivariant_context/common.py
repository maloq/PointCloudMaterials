"""Frozen configuration, data ancestry and implementation identity."""
import json
from pathlib import Path
from src.project_runtime.paths import resolve_path
from src.research.structural_state.common import sha, digest, write_json
from src.research.supervised_onset.tracking import require_online
from .model import VARIANTS


class Study:
    def __init__(self,path):
        self.path=Path(path).resolve();self.config=json.loads(self.path.read_text());c=self.config
        if c['protocol']!='equivariant_context_v1':raise ValueError('Unknown context protocol')
        if (not c['variants'] or len(set(c['variants']))!=len(c['variants'])
            or any(v not in VARIANTS for v in c['variants']) or c['external_inputs']!=[]):
            raise ValueError('Distinct declared predictor variants and no external conditions required')
        if c['objective']!='event_nll' or c['selection_metric']!='event_nll':
            raise ValueError('Predictive likelihood training and selection required; AP is evaluation only')
        require_online(c['wandb'])
        c['extraction']=dict(chunk=c['microbatch'],workers=2,prefetch=2,compile=True)|c.get('extraction',{})
        if any(c['extraction'][k]<1 for k in ('chunk','workers','prefetch')):
            raise ValueError('Positive extraction chunk, workers and prefetch capacity required')
        self.root=resolve_path(c['output']);self.cache=resolve_path(c['cache']);self.technical=self.root/'technical'
        self.technical.mkdir(parents=True,exist_ok=True)

    def identity_record(self):
        base=Path(__file__).resolve().parents[3]
        paths=[]
        for package in ('equivariant_context','supervised_onset'):
            paths.extend((base/'src/research'/package).glob('*.py'))
        paths.extend((base/'src/data/fixed_cohort').glob('*.py'))
        paths.extend((base/'src/research/encoder_context').glob('*.py'))
        paths.extend(base/p for p in ('src/models/encoders/spatial_mace.py','src/models/encoders/graph_bank.py',
            'src/models/encoders/mace_backend.py','src/research/structured_context/geometry.py',
            'src/research/local_predictability/metrics.py','src/research/structural_state/data.py',
            'src/training_methods/bcr/data.py','src/research/structural_state/common.py',
            'src/data/trajectories/shooting.py','src/data/trajectories/lammps.py'))
        return dict(config=self.config,implementation={str(p.relative_to(base)):sha(p) for p in paths},
            bases={d:sha(resolve_path(p)) for d,p in self.config['base_configs'].items()},
            data={k:sha(resolve_path(self.config['population_cache'])/k) for k in ('manifest.json','population.npz')},
            parent=sha(resolve_path(self.config['parent_plan'])))

    def bind(self):
        record=self.identity_record();self.identity=digest(record)
        path=self.technical/'identity.json'
        if path.exists():
            if json.loads(path.read_text())!=record:
                raise ValueError(f'Prepared study changed; use a fresh output or explicitly rebuild an unlaunched plan: {path}')
        else:
            write_json(path,record)
        return self.identity
