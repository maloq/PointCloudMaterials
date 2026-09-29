import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from src.data.fixed_cohort.dataset import read_release
from src.data.fixed_cohort.protocol import sha, digest, write_json
from src.project_runtime.paths import resolve_path
from src.experiment_runner.artifacts import result_folders


def study(path):
    c = json.loads(Path(path).read_text())
    parent = resolve_path(c['parent_run'])
    fixed, plan = read_release(c['fixed_dataset']['root'])
    if plan['identity'] != c['fixed_dataset']['identity']:
        raise ValueError('Fixed cohort changed')
    pointer = json.loads(resolve_path(c['feature_pointer']).read_text())
    original = json.loads((parent/'technical/identity.json').read_text())
    for key, target in (('audit_config_sha256', resolve_path(c['audit_config'])),
                        ('audit_summary_sha256', resolve_path(c['audit'])/'technical/summary.json')):
        if sha(target) != original[key]:
            raise ValueError(f'Audit no longer matches retained spatial labels: {target}')
    checkpoint = resolve_path(c['checkpoint'])
    if sha(checkpoint) != pointer['encoder_sha256']:
        raise ValueError('Encoder checkpoint mismatch')
    repo = Path(__file__).resolve().parents[3]
    dependencies = []
    for package in ('spatial_distance', 'spatial_approach', 'equivariant_context', 'crystallization_origin'):
        dependencies += list((repo/'src/research'/package).glob('*.py'))
    dependencies += [repo/p for p in ('src/research/supervised_onset/model.py',
        'src/research/supervised_onset/tracking.py', 'src/research/structured_context/geometry.py',
        'src/research/encoder_context/geometry.py', 'src/models/encoders/spatial_mace.py')]
    parent_receipts = {p.name: sha(p) for p in (parent/'technical').glob('*.json')
                       if p.name in ('identity.json','prepared.json','scan-features.json')}
    binding = dict(config=c, release_identity=plan['identity'],
        parent_receipts=parent_receipts, checkpoint_sha256=sha(checkpoint),
        population_sha256=sha(fixed/'benchmark/population.npz'),
        original_feature_pointer_sha256=sha(resolve_path(c['feature_pointer'])),
        fixed_geometry_pointer_sha256=sha(resolve_path(c['fixed_geometry_pointer'])),
        fixed_geometry_metadata=json.loads((Path(json.loads(resolve_path(c['fixed_geometry_pointer']).read_text())['path'])/'entry.json').read_text())['metadata'],
        implementation={str(p.relative_to(repo)):sha(p) for p in dependencies})
    identity = digest(binding)
    root = result_folders(resolve_path(c['output'])); dest = root/'technical/identity.json'
    if dest.exists() and json.loads(dest.read_text()) != binding:
        raise ValueError('Distance protocol changed: use a new run directory')
    if not dest.exists(): write_json(dest,binding)
    with np.load(fixed/'benchmark/population.npz') as a:
        pop = {k:a[k] for k in ('source','role','frame','atom','sample_id','legacy_row')}
    geometry = resolve_path(c['geometry_cache'])/identity
    geometry.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(config=c, root=root, technical=root/'technical', parent=parent,
        identity=identity, pointer=pointer, features=Path(pointer['path']), checkpoint=checkpoint,
        plan=plan, pop=pop, geometry=geometry)
