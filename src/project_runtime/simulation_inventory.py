"""Readable location and producer-record indexes for the dataset catalog."""

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from .paths import catalog, dataset_path


def export_simulations(output):
    """Index collections and outcome evidence without claiming independent counts."""
    output = Path(output)
    collections, records = [], []
    observed = datetime.now(timezone.utc).isoformat()
    for identifier, entry in sorted(catalog().items()):
        if entry['kind'] != 'simulation' and not entry.get('simulation_fixture', False):
            continue
        root = dataset_path(identifier).resolve()
        available = root.is_dir()
        collections.append((identifier, entry['kind'], str(root), available, observed))
        if not available:
            continue
        for directory, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = sorted(name for name in dirs if not Path(directory, name).is_symlink())
            if 'outcome.json' not in names:
                continue
            path = Path(directory, 'outcome.json')
            if path.is_symlink():
                continue
            raw = path.read_bytes()
            outcome = json.loads(raw)
            records.append((identifier, str(path.relative_to(root)), outcome['state'],
                            str(path), hashlib.sha256(raw).hexdigest(), observed))
    output.mkdir(parents=True, exist_ok=True)
    for name, columns, rows in (
        ('collections.csv', ('dataset_id', 'kind', 'location', 'available', 'observed_at'), collections),
        ('run_records.csv', ('dataset_id', 'record', 'recorded_state', 'location', 'sha256', 'observed_at'), records),
    ):
        path = output / name
        temporary = path.with_suffix('.csv.building')
        with temporary.open('w', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(columns)
            writer.writerows(rows)
        temporary.replace(path)
    return dict(collections=len(collections), outcome_records=len(records), output=str(output),
                note='Producer evidence includes duplicates/attempts; not independent or newly verified simulation counts.')
