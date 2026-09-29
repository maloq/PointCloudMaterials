"""Save metric tables with a frozen description and the exact implementation identity."""

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil

from .artifacts import result_folders, write_json

REPO = Path(__file__).resolve().parents[2]
DOCUMENTS = REPO / 'docs/metrics'


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_metric_docs(*, family=None):
    """Validate one export's dependencies, or all families for the repository audit."""
    contracts = json.loads((DOCUMENTS / 'contracts.json').read_text())
    if family is not None:
        if family not in contracts:
            raise ValueError(f'Unregistered metric family {family!r}: {DOCUMENTS / "contracts.json"}')
        contracts = {family: contracts[family]}
    failures = []
    observed = {}
    for name, contract in contracts.items():
        description = f'docs/metrics/{name}.md'
        if description not in contract['files']:
            failures.append(f'{name}: description is not included in the contract: {description}')
        for filename, expected in contract['files'].items():
            path = REPO / filename
            if not path.is_file():
                failures.append(f'{name}: missing {filename}')
                continue
            if filename not in observed:
                observed[filename] = fingerprint(path)
            if observed[filename] != expected:
                failures.append(f'{name}: {filename}: expected {expected}, observed {observed[filename]}')
    if failures:
        raise ValueError('Metric contract validation failed; review code and definitions together:\n' + '\n'.join(failures))
    return contracts


def snapshot_metric_docs(root, family, *, generated_catalogue=False):
    contracts = check_metric_docs(family=family)
    contract = contracts[family]
    root = result_folders(root)
    desc_path = DOCUMENTS / f'{family}.md'
    description = desc_path.read_text()
    scoped = root / f'technical/metric-contracts/{family}.json'
    scoped_doc = root / f'tables/metric-definitions/{family}.md'
    original = root / 'technical/metric-contract.json'
    if not original.exists() and (root/'tables/METRICS.md').exists():
        raise ValueError(f'{root}: existing definitions have no frozen contract; use a new numerical export revision')
    previous = scoped if scoped.exists() else original
    replacing_catalogue = False
    if previous.exists():
        saved = json.loads(previous.read_text())
        if saved['family'] == family:
            if saved['files'] != contract['files']:
                if not generated_catalogue or family != 'encoder_research':
                    raise ValueError(f'{previous}: calculation contract changed; export to a new analysis revision. '
                                     'Use publication-only rendering for historical results.')
                # This is the generated inventory's contract, never a scientific
                # producer's. Preserve its old tables and definitions as a unit.
                files = [p for p in (root/'tables').rglob('*') if p.is_file()]
                files += [p for p in (original,scoped,root/'technical/coverage.json') if p.is_file()]
                hashes = {str(p.relative_to(root)):fingerprint(p) for p in files}
                version = hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
                archive = root/'technical/catalogue-revisions'/version
                for source in files:
                    target = archive/source.relative_to(root)
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if not target.exists():shutil.copy2(source,target)
                    if fingerprint(target)!=hashes[str(source.relative_to(root))]:
                        raise ValueError(f'Catalogue definition archive differs: {target}')
                write_json(archive/'files.json',hashes)
                replacing_catalogue = True
            original_doc = scoped_doc if scoped_doc.exists() else root / 'tables/METRICS.md'
            if not original_doc.is_file():
                raise FileNotFoundError(f'Frozen metric description missing: {original_doc}')
            if not scoped.exists() and not replacing_catalogue:
                write_json(scoped, saved)
                scoped_doc.parent.mkdir(parents=True, exist_ok=True)
                scoped_doc.write_bytes(original_doc.read_bytes())
            if not replacing_catalogue:
                return scoped
    captured = datetime.now(timezone.utc).isoformat()
    rendered = description + '\n\n' + (
        f'Table export: {captured}. The machine-readable values retain full precision; '
        'blank values mean undefined or unrecorded, never zero. '
        'Nested metric names preserve the producer\'s grouping. '
        f'The implementation hashes are in `technical/metric-contracts/{family}.json` relative to the analysis root.\n')
    payload = dict(family=family, exported_at=captured, files=contract['files'], verified=True)
    write_json(scoped, payload)
    scoped_doc.parent.mkdir(parents=True, exist_ok=True)
    scoped_doc.write_text(rendered)
    if not original.exists() or replacing_catalogue:
        write_json(original, payload)
        (root / 'tables/METRICS.md').write_text(rendered)
    families = sorted((root / 'technical/metric-contracts').glob('*.json'))
    (root / 'tables/METRIC_INDEX.md').write_text('# Metric definitions by family\n\n' + '\n'.join(
        f'- [{p.stem}](metric-definitions/{p.stem}.md) · [contract](../technical/metric-contracts/{p.name})'
        for p in families) + '\n')
    return scoped


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
    path = Path(root) / 'tables' / f'{name}.csv'
    binding = Path(root) / f'technical/table-contracts/{name}.json'
    prior = binding if binding.exists() else Path(root)/'technical/metric-contract.json'
    if path.exists() and prior.exists() and json.loads(prior.read_text())['family']!=family:
        raise ValueError(f'{path}: table belongs to another metric family; choose a distinct table name or analysis')
    contract = snapshot_metric_docs(root, family)
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(('metric', 'value'))
    writer.writerows(numeric_rows(metrics))
    temporary = path.with_suffix('.csv.building')
    temporary.write_text(stream.getvalue())
    temporary.replace(path)
    write_json(Path(root) / f'technical/table-contracts/{name}.json', dict(
        table=f'tables/{name}.csv', sha256=fingerprint(path), family=family,
        contract=str(contract.relative_to(root)),
        definitions=f'tables/metric-definitions/{family}.md'))
    return path
