"""Explicit input contracts for new prediction studies, separate from audit metadata."""
from pathlib import Path
import json


def write_context(root, record):
    required = {'protocol', 'identity', 'branch', 'encoder_inputs', 'predictor_inputs',
                'external_inputs', 'spatial', 'history', 'velocities', 'arms', 'probes', 'controls'}
    missing = required - record.keys()
    if missing:
        raise ValueError(f'Missing prediction-context fields: {sorted(missing)}')
    if record['external_inputs'] or record['history']['time_features']:
        raise ValueError('User policy forbids temperature/explicit time context in new experiments')
    target = Path(root) / 'technical/prediction-context.json'
    if target.exists() and json.loads(target.read_text()) != record:
        raise ValueError(f'Prediction context changed within a frozen experiment: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
