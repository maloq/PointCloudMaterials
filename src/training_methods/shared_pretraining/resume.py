"""Explicit, exact-source continuation for execution-only optimizations."""
import json
from pathlib import Path

from src.data.structural_pretraining.prepare import digest, file_hash, save_json
from src.project_runtime.paths import resolve_path

EXECUTION_FILES = {
    'src/training_methods/shared_pretraining/runtime.py',
    'src/training_methods/shared_pretraining/parallel_mace.py',
    'src/training_methods/shared_pretraining/input_pipeline.py',
    'src/training_methods/shared_pretraining/mixed.py',
    'src/training_methods/shared_pretraining/resume.py',
}
EXECUTION_SETTINGS = {'preparation_workers', 'preparation_processes', 'prefetch_batches'}


def canonical_files(files):
    # Earlier checkpoints used __file__ for runtime.py, including the frozen root.
    return {'src/'+p.rsplit('/src/', 1)[1] if '/src/' in p else p: h for p, h in files.items()}


def verify_execution_change(previous, replacement):
    left, right = dict(previous), dict(replacement)
    old_code, new_code = dict(left.pop('implementation')), dict(right.pop('implementation'))
    old_config, new_config = left.pop('config'), right.pop('config')
    if left != right or ({k:v for k,v in old_config.items() if k not in EXECUTION_SETTINGS}
                       != {k:v for k,v in new_config.items() if k not in EXECUTION_SETTINGS}):
        raise ValueError('Execution continuation changed data, model, losses, schedule or sampling settings')
    old_files = canonical_files(old_code.pop('files'))
    new_files = canonical_files(new_code.pop('files'))
    if old_code != new_code:
        raise ValueError('Execution continuation changed library versions')
    changed = {p for p in old_files.keys() | new_files.keys() if old_files.get(p) != new_files.get(p)}
    if not changed <= EXECUTION_FILES:
        raise ValueError(f'Execution continuation changed protected implementations: {sorted(changed-EXECUTION_FILES)}')
    return sorted(changed)


def check_resume(previous, replacement, transition_path, technical):
    if previous == replacement:
        return
    if transition_path is None:
        raise ValueError('Changed training identity requires an explicit tested execution transition')
    path = resolve_path(transition_path)
    receipt = json.loads(path.read_text())
    if receipt['previous_identity_sha256'] != digest(previous) or receipt['replacement_identity_sha256'] != digest(replacement):
        raise ValueError('Execution transition does not match the exact previous/replacement identity')
    if receipt['checkpoint_sha256'] != file_hash(Path(technical)/'last.pt'):
        raise ValueError('Execution transition does not match the checkpoint being resumed')
    changed = verify_execution_change(previous, replacement)
    save_json(Path(technical)/'implementation_transition.json', dict(transition=receipt,
        receipt_sha256=file_hash(path), changed_files=changed,
        preserved='model, objective, optimizer, scheduler, sample order, RNG, best score and validation history'))
