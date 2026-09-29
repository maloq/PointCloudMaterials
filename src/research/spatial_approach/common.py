import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from src.data.fixed_cohort.dataset import read_release
from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.project_runtime.paths import resolve_path
from src.experiment_runner.artifacts import result_folders


def study(path):
    c = json.loads(Path(path).read_text())
    root = result_folders(resolve_path(c['output']))
    fixed, plan = read_release(c['fixed_dataset']['root'])
    if plan['identity'] != c['fixed_dataset']['identity']:
        raise ValueError('Fixed source/sample release changed')
    pointer = json.loads(resolve_path(c['feature_pointer']).read_text())
    features = Path(pointer['path'])
    checkpoint = resolve_path(c['checkpoint'])
    if sha(checkpoint) != pointer['encoder_sha256']:
        raise ValueError('Cached features belong to another checkpoint')
    repo = Path(__file__).resolve().parents[3]
    dependencies = list(Path(__file__).parent.glob('*.py'))
    for package in ('equivariant_context', 'crystallization_origin'):
        dependencies += list((repo / 'src/research' / package).glob('*.py'))
    dependencies += [repo / p for p in (
        'src/research/supervised_onset/model.py', 'src/research/supervised_onset/tracking.py',
        'src/models/encoders/spatial_mace.py', 'src/research/structured_context/geometry.py',
        'src/research/encoder_context/geometry.py', 'src/data/fixed_cohort/protocol.py')]
    record = dict(config=c, release_identity=plan['identity'], checkpoint_sha256=sha(checkpoint),
                  feature_manifest_sha256=sha(features / 'manifest.json'),
                  population_sha256=sha(fixed / 'benchmark/population.npz'),
                  audit_config_sha256=sha(resolve_path(c['audit_config'])),
                  audit_summary_sha256=sha(resolve_path(c['audit']) / 'technical/summary.json'),
                  implementation={str(p.relative_to(repo)): sha(p) for p in dependencies})
    identity = digest(record)
    dest = root / 'technical/identity.json'
    if dest.exists() and json.loads(dest.read_text()) != record:
        raise ValueError('Scientific contract changed; use a new run directory')
    if not dest.exists():
        write_json(dest, record)
    with np.load(fixed / 'benchmark/population.npz') as a:
        # Temporal outcome and time/temperature covariates never enter this task.
        pop = {k: a[k] for k in ('source', 'role', 'frame', 'atom', 'sample_id', 'legacy_row')}
    return SimpleNamespace(config=c, root=root, technical=root / 'technical', identity=identity,
                           features=features, pointer=pointer, checkpoint=checkpoint, plan=plan, pop=pop)
