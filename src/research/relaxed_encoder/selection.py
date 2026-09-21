"""Explicit encoder exclusions without changing frozen data or training plans."""
import json

from src.project_runtime.paths import resolve_path


def evaluation_exclusions(plan):
    path = resolve_path(plan['config']['output'])/'technical/evaluation-exclusions.json'
    if not path.exists():
        return {}
    record = json.loads(path.read_text())
    if record['plan_identity'] != plan['identity']:
        raise ValueError(f'Evaluation exclusions belong to another plan: {path}')
    excluded = record['excluded']
    known = {r['name'] for r in plan['config']['runs']}
    if set(excluded) - known:
        raise ValueError(f'Unknown excluded encoders: {set(excluded) - known}')
    for name, receipt in excluded.items():
        if not receipt['reason']:
            raise ValueError(f'Encoder exclusion needs an explicit reason: {name}')
    return excluded


def selected_runs(plan):
    excluded = evaluation_exclusions(plan)
    return [r for r in plan['config']['runs'] if r['name'] not in excluded]
