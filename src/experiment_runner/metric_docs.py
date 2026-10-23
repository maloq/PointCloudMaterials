"""Capture metric definitions and source identity without gating experiments on hashes."""

import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil

from .artifacts import file_hash, json_digest, result_folders, write_json

REPO = Path(__file__).resolve().parents[2]
DOCUMENTS = REPO / 'docs/metrics'


def fingerprint(path):
    return file_hash(path)


def check_metric_docs(*, family=None):
    """Load declared dependencies; explicit audits check hashes, never training preflights.

    Keep the declared maps unchanged for historical plans that pin them. These
    maps describe the registry, not the implementation currently on disk.
    """
    contracts = json.loads((DOCUMENTS / 'contracts.json').read_text())
    if family is None:
        return contracts
    if family not in contracts:
        raise ValueError(f'Unregistered metric family {family!r}: {DOCUMENTS / "contracts.json"}')
    return {family: contracts[family]}


def audit_metric_docs(*, family=None):
    """Report registry drift for review without altering or endorsing its baselines."""
    contracts = check_metric_docs(family=family)
    issues = []
    observed = {}
    for name, contract in contracts.items():
        description = f'docs/metrics/{name}.md'
        files = dict(contract['files'])
        if description not in files:
            issues.append(dict(family=name, file=description, status='undeclared_description',
                               expected=None, observed=None))
        if contract.get('status') == 'retired':
            # Removed source is expected for retired families. Audit their frozen record instead.
            retirement = contract['retirement']
            files = {description: files.get(description), retirement['record']: retirement['sha256']}
        for filename, expected in files.items():
            if filename not in observed:
                path = REPO / filename
                observed[filename] = fingerprint(path) if path.is_file() else None
            actual = observed[filename]
            if actual is None or actual != expected:
                issues.append(dict(family=name, file=filename, expected=expected, observed=actual,
                                   status='missing' if actual is None else 'changed'))
        if contract.get('status') == 'retired' and observed[retirement['record']] == retirement['sha256']:
            record = json.loads((REPO / retirement['record']).read_text())
            if record['family'] != name or record['contract']['files'] != contract['files']:
                issues.append(dict(family=name, file=retirement['record'], status='retirement_mismatch',
                                   expected=contract['files'], observed=record['contract']['files']))
    return dict(ok=not issues, families=list(contracts), issues=issues)


def _archive(root, paths, kind):
    """Preserve existing bytes together; archive collisions remain real integrity errors."""
    files = sorted({p for p in paths if p.is_file()})
    if not files:
        return
    hashes = {str(p.relative_to(root)): fingerprint(p) for p in files}
    archive = root / f'technical/{kind}-revisions' / json_digest(hashes)
    for source in files:
        relative = source.relative_to(root)
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        if fingerprint(target) != hashes[str(relative)]:
            raise ValueError(f'Metric archive differs: {target}')
    manifest = archive / 'files.json'
    if not manifest.exists():
        write_json(manifest, hashes)


def _table_evidence(root, table):
    paths = [table]
    binding = root / 'technical/table-contracts' / f'{table.stem}.json'
    if binding.is_file():
        record = json.loads(binding.read_text())
        paths.extend([binding, root / record['contract'], root / record['definitions']])
    else:
        paths.extend([root / 'technical/metric-contract.json', root / 'tables/METRICS.md'])
    return paths


def _family_tables(root, family, current):
    original = root / 'technical/metric-contract.json'
    for table in (root / 'tables').glob('*.csv'):
        binding = root / 'technical/table-contracts' / f'{table.stem}.json'
        owner = binding if binding.is_file() else original
        if not owner.is_file():
            continue
        record = json.loads(owner.read_text())
        if record['family'] != family or (binding.is_file() and record['contract'] == str(current.relative_to(root))):
            continue
        yield from _table_evidence(root, table)
        if not binding.is_file():
            # A legacy direct writer supplied no table binding. Preserve every
            # available capture rather than guessing which version produced it.
            yield from (root / 'technical/metric-contracts').glob(f'{family}.*.json')
            yield from (root / 'tables/metric-definitions').glob(f'{family}.*.md')
            yield root / 'tables/METRIC_INDEX.md'


def snapshot_metric_docs(root, family, *, generated_catalogue=False):
    """Capture actual bytes in an immutable receipt; registry drift is informational."""
    contract = check_metric_docs(family=family)[family]
    if contract.get('status') == 'retired':
        raise ValueError(f'Metric family {family!r} is retired. Use its recorded frozen producer '
                         'for numerical reproduction, or publication-only rendering for existing results.')
    root = result_folders(root)
    description_name = f'docs/metrics/{family}.md'
    observed = {name: fingerprint(REPO / name) if (REPO / name).is_file() else None
                for name in sorted(set(contract['files']) | {description_name})}
    identity = dict(family=family, files={name: value for name, value in observed.items() if value is not None},
                    missing_files=[name for name, value in observed.items() if value is None],
                    expected_files=contract['files'])
    stem = f'{family}.{json_digest(identity)}'
    scoped = root / f'technical/metric-contracts/{family}.json'
    scoped_doc = root / f'tables/metric-definitions/{family}.md'
    receipt = root / f'technical/metric-contracts/{stem}.json'
    definitions = root / f'tables/metric-definitions/{stem}.md'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if any(saved[key] != value for key, value in identity.items()) or fingerprint(definitions) != saved['definitions_sha256']:
            raise ValueError(f'Captured metric provenance changed: {receipt}')

    original = root / 'technical/metric-contract.json'
    original_doc = root / 'tables/METRICS.md'
    changed_tables = list(_family_tables(root, family, receipt))
    if changed_tables and generated_catalogue and family == 'encoder_research':
        paths = list((root / 'tables').rglob('*')) + list((root / 'technical/table-contracts').glob('*.json'))
        paths += list((root / 'technical/metric-contracts').glob('*.json'))
        paths += [original, root / 'technical/coverage.json']
        _archive(root, paths, 'catalogue')
    else:
        _archive(root, changed_tables, 'metric')
    if receipt.exists():
        return receipt

    # Preserve legacy aliases exactly; new tables bind the versioned receipt below.
    if not scoped.exists() and original.exists():
        saved = json.loads(original.read_text())
        if saved['family'] == family:
            scoped.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, scoped)
            if original_doc.exists() and not scoped_doc.exists():
                scoped_doc.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original_doc, scoped_doc)

    captured = datetime.now(timezone.utc).isoformat()
    desc_path = DOCUMENTS / f'{family}.md'
    description = desc_path.read_text() if desc_path.is_file() else f'# {family}\n\nNo metric description was available at export.\n'
    rendered = description + '\n\n' + (
        f'Table export: {captured}. Source and definition hashes were captured from files on disk; '
        'capture is not scientific validation or review. Blank table values mean undefined or '
        'unrecorded, never zero. Nested metric names preserve the producer\'s grouping. '
        f'Implementation receipt: `technical/metric-contracts/{stem}.json` relative to the analysis root.\n')
    payload = dict(identity, exported_at=captured, schema_version=2, verified=False,
                   verification='captured', matches_registry=(observed == contract['files']),
                   definitions=str(definitions.relative_to(root)))
    definitions.parent.mkdir(parents=True, exist_ok=True)
    definitions.write_text(rendered)
    payload['definitions_sha256'] = fingerprint(definitions)
    write_json(receipt, payload)
    if not scoped.exists():
        shutil.copy2(receipt, scoped)
        if not scoped_doc.exists():
            shutil.copy2(definitions, scoped_doc)
    if not original.exists() and not original_doc.exists():
        shutil.copy2(receipt, original)
        shutil.copy2(definitions, original_doc)
    versions = sorted((root / 'technical/metric-contracts').glob('*.*.json'))
    (root / 'tables/METRIC_INDEX.md').write_text('# Captured metric definitions\n\n'
        'Each table binding identifies its exact version. Unsuffixed files retain the first export.\n\n' + '\n'.join(
        f'- [{p.stem}](metric-definitions/{p.stem}.md) · [receipt](../technical/metric-contracts/{p.name})'
        for p in versions) + '\n')
    return receipt


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


def _table_contract(root, family, name):
    root = Path(root)
    path = Path(root) / 'tables' / f'{name}.csv'
    binding = Path(root) / f'technical/table-contracts/{name}.json'
    prior = binding if binding.exists() else Path(root)/'technical/metric-contract.json'
    if path.exists() and prior.exists() and json.loads(prior.read_text())['family']!=family:
        raise ValueError(f'{path}: table belongs to another metric family; choose a distinct table name or analysis')
    contract = snapshot_metric_docs(root, family)
    return path, contract


def _publish_table(path, contract, family, content):
    temporary = path.with_suffix('.csv.building')
    temporary.write_text(content)
    return _finish_table(path, temporary, contract, family)


def _finish_table(path, temporary, contract, family):
    root = path.parent.parent
    binding = root / 'technical/table-contracts' / f'{path.stem}.json'
    if path.exists() and (not binding.exists() or
            json.loads(binding.read_text())['contract'] != str(contract.relative_to(root))):
        _archive(root, _table_evidence(root, path), 'metric')
    temporary.replace(path)
    return _bind_table(path, contract, family)


def _bind_table(path, contract, family):
    root = path.parent.parent
    name = path.stem
    write_json(Path(root) / f'technical/table-contracts/{name}.json', dict(
        table=f'tables/{name}.csv', sha256=fingerprint(path), family=family,
        contract=str(contract.relative_to(root)),
        definitions=json.loads(contract.read_text())['definitions']))
    return path


def write_metric_table(metrics, root, *, family, name='metrics'):
    path, contract = _table_contract(root, family, name)
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(('metric', 'value'))
    writer.writerows(numeric_rows(metrics))
    return _publish_table(path, contract, family, stream.getvalue())


def write_metric_rows(rows, root, *, family, name, columns=None):
    """Bind a row table to frozen definitions, preserving field order and precision.

    Nonempty producers may retain their first row's declared field order. Empty
    tables require explicit columns so their schema remains meaningful.
    """
    rows = iter(rows)
    first = next(rows, None)
    if columns is None:
        if first is None:
            raise ValueError(f'{family}/{name}: empty metric table requires columns')
        columns = tuple(first)
    path, contract = _table_contract(root, family, name)
    temporary = path.with_suffix('.csv.building')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        if first is not None:
            writer.writerow(first)
        writer.writerows(rows)
    return _finish_table(path, temporary, contract, family)
