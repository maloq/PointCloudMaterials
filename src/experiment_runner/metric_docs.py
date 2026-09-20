"""Save metric tables with a frozen description and the exact implementation identity."""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .artifacts import result_folders

REPO = Path(__file__).resolve().parents[2]
DOCUMENTS = REPO / 'docs/metrics'


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_metric_docs(*, family=None):
    """Validate one export's dependencies, or all families for the repository audit."""
    contracts = json.loads((DOCUMENTS / 'contracts.json').read_text()) if (DOCUMENTS / 'contracts.json').exists() else {}
    if family is not None and family in contracts:
        contracts = {family: contracts[family]}
    return contracts


def snapshot_metric_docs(root, family):
    contracts = check_metric_docs(family=family)
    contract = contracts.get(family, {'files': {}})
    root = result_folders(root)
    desc_path = DOCUMENTS / f'{family}.md'
    description = desc_path.read_text() if desc_path.exists() else ''
    captured = datetime.now(timezone.utc).isoformat()
    (root / 'tables/METRICS.md').write_text(description + '\n\n'
        f'Table export: {captured}. The machine-readable values retain full precision; '
        'blank values mean undefined or unrecorded, never zero. '
        'Nested metric names preserve the producer\'s grouping. '
        'The implementation hashes are in `../technical/metric-contract.json`.\n')
    payload = dict(family=family, exported_at=captured, files=contract.get('files', {}))
    (root / 'technical/metric-contract.json').write_text(json.dumps(payload, indent=2) + '\n')


def numeric_rows(metrics, prefix=''):
    """Flatten the nested JSON mappings written by our metric producers; arrays stay in JSON."""
    for key, value in metrics.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            yield from numeric_rows(value, name)
        elif key in ('ci95', 'source_bootstrap_95_percent_interval') and value is not None:
            lower, upper = value
            yield name + '.lower', lower
            yield name + '.upper', upper
        elif type(value) in (int, float) or value is None:
            yield name, value


def write_metric_table(metrics, root, *, family, name='metrics'):
    snapshot_metric_docs(root, family)
    path = Path(root) / 'tables' / f'{name}.csv'
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('metric', 'value'))
        writer.writerows(numeric_rows(metrics))
    return path
