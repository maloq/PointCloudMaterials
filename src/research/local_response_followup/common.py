"""Bind new protocols to the unchanged local80 inputs, labels and source roles."""
from pathlib import Path
import shutil
import torch

from src.project_runtime.paths import resolve_path
from src.research.local_response.common import read, sha, digest, save, write_json

FAMILY = 'local_response_followup'


def root(c):
    return resolve_path(c['output'])


def reference(c):
    return resolve_path(c['reference'])


def source_config(c):
    return read(reference(c)/'technical/binding.json')['config']


def load(c):
    receipt = read(reference(c)/'technical/data-complete.json')
    if receipt['identity'] != c['reference_identity'] or sha(receipt['cache']) != receipt['sha256']:
        raise ValueError('Original local80 label bank changed')
    return torch.load(receipt['cache'], map_location='cpu', weights_only=False)


def bind(c):
    parent = read(reference(c)/'technical/binding.json')
    if parent['identity'] != c['reference_identity']:
        raise ValueError('Reference experiment identity changed')
    for name, expected in parent['implementation'].items():
        if sha(Path(name)) != expected:
            raise ValueError(f'Original numerical producer changed: {name}')
    receipt = read(reference(c)/'technical/data-complete.json')
    if sha(receipt['cache']) != receipt['sha256']:
        raise ValueError('Original response bank changed')
    parents_path = reference(c)/'technical/parents.pt'
    if sha(parents_path) != read(reference(c)/'technical/parents.json')['sha256']:
        raise ValueError('Original simulator inputs changed')
    repo = Path(__file__).resolve().parents[3]
    implementation = dict(parent['implementation'])
    implementation.update({str(p.relative_to(repo)):sha(p) for p in Path(__file__).parent.glob('*.py')})
    record = dict(config=c, implementation=implementation, data_sha256=receipt['sha256'],
                  parents_sha256=sha(parents_path))
    record['identity'] = digest(record)
    path = root(c)/'technical/binding.json'
    if path.exists() and read(path) != record:
        raise ValueError('Follow-up configuration/producer changed; use a new revision')
    write_json(path, record)
    context = read(reference(c)/'technical/prediction-context.json')
    context.update(followup=dict(reference_identity=parent['identity'],
        training_inputs='Exactly the original90/15/30 local80 tensors and original training normalizers',
        changed='Response weight/schedule and fixed400-update optimization budget only',
        finite_forward='Train-only fixed-identity local displacements; surrounding atoms move; no teacher conditions enter predictors',
        evaluation='All30 original held-out sources retained; reused exploratory cohort, not a newly untouched test set'))
    write_json(root(c)/'technical/prediction-context.json',context)
    return record


def archive_payload(c, name, payload):
    source=resolve_path(c['simulation_scratch'])/name
    dest=resolve_path(c['simulation_archive'])/name
    save(source,payload);dest.parent.mkdir(parents=True,exist_ok=True)
    temporary=dest.with_suffix('.copying.pt');shutil.copy2(source,temporary)
    if sha(source)!=sha(temporary):raise IOError(f'Archive copy mismatch: {source}')
    temporary.replace(dest)
    write_json(dest.with_suffix('.json'),dict(identity=read(root(c)/'technical/binding.json')['identity'],
        sha256=sha(dest),state='complete'))
    return dest


def read_payload(c, name):
    path=resolve_path(c['simulation_archive'])/name
    receipt=path.with_suffix('.json')
    if not receipt.exists():return None
    rec=read(receipt)
    if rec['identity']!=read(root(c)/'technical/binding.json')['identity'] or sha(path)!=rec['sha256']:
        raise ValueError(f'Changed forward payload: {path}')
    return torch.load(path,map_location='cpu',weights_only=False)
