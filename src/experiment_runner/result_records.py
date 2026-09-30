"""Explicit research identities shared by producers and both catalogue views.

Receipts are durable evidence. Registration files are a small rebuildable queue of
receipt locations, not another manually curated scientific database.
"""
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import sqlite3
from urllib.parse import quote
from urllib.parse import unquote, urlsplit
from html.parser import HTMLParser
from functools import lru_cache

from src.project_runtime.paths import REPO, machine, resolve_path
from .artifacts import file_hash, json_digest as identity, write_json

KINDS = {'research', 'operations', 'simulation', 'dataset', 'unclassified'}


@lru_cache(maxsize=1)
def storage_roots():
    return machine()['roots']


def location(path):
    path = Path(path).absolute()
    for role, root in sorted(storage_roots().items(),key=lambda item:-len(item[1])):
        if path.is_relative_to(root):
            return '${storage:'+role+'}/'+str(path.relative_to(root))
    return str(path)


def resolve_reference(value):
    if str(value).startswith('${storage:'):
        token, suffix = str(value).split('}',1)
        return Path(storage_roots()[token[len('${storage:'):]])/suffix.lstrip('/')
    return resolve_path(value)


def href(path, base):
    return quote(os.path.relpath(resolve_reference(path), Path(base).absolute()), safe='/')


def validate_record(record):
    required = {'schema_version', 'id', 'kind', 'activity', 'title', 'study',
                'execution', 'evidence', 'interpretation', 'analyses', 'components'}
    if missing := required - record.keys():
        raise ValueError(f'Result record missing {sorted(missing)}')
    if record['schema_version'] != 1 or record['kind'] not in KINDS:
        raise ValueError(f'Unsupported result record: {record["id"]}')
    for dimension in ('execution', 'evidence', 'interpretation'):
        if not isinstance(record[dimension], dict) or 'state' not in record[dimension]:
            raise ValueError(f'{record["id"]}: {dimension} requires a separate state record')
    ids = [a['id'] for a in record['analyses']]
    if len(ids) != len(set(ids)):
        raise ValueError(f'{record["id"]}: repeated evaluation IDs')
    for analysis in record['analyses']:
        for key in ('protocol', 'inputs', 'population', 'selection', 'stages', 'artifacts'):
            if key not in analysis:
                raise ValueError(f'{record["id"]}/{analysis["id"]}: missing {key}')
        for stage, receipt in analysis['stages'].items():
            if receipt['state'] == 'complete' and not receipt.get('evidence'):
                raise ValueError(f'{analysis["id"]}/{stage}: completion requires evidence')
    return record


def save_record(root, record, *, refresh=False):
    """Preserve scientific identity and authored interpretation across progress updates."""
    root = Path(root)
    validate_record(record)
    path = root / 'run.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old['id'] != record['id']:
            raise ValueError(f'{path}: already belongs to {old["id"]}; choose another result directory')
        record['interpretation'] = old['interpretation']
        # Keep revised/earlier evaluations accessible instead of replacing them.
        current = {a['id'] for a in record['analyses']}
        record['analyses'] += [a for a in old['analyses'] if a['id'] not in current]
    write_json(path, record)
    register_record(path)
    if refresh:
        try:
            refresh_results()
        except (OSError, ValueError, sqlite3.Error) as error:
            write_json(root / 'technical/indexing-status.json', dict(state='failed', error=str(error)))
            print(f'[results] Scientific receipt saved, catalogue refresh failed: {error}', flush=True)
        else:
            write_json(root / 'technical/indexing-status.json', dict(state='complete'))
    return path


def register_record(path):
    path = Path(path).absolute()
    record = validate_record(json.loads(path.read_text()))
    target = REPO / 'output/registry/registrations' / f'{identity(record["id"])}.json'
    if target.exists():
        previous = json.loads(target.read_text())
        if resolve_reference(previous['record']).resolve() != path.resolve():
            raise ValueError(f'{record["id"]}: already registered at {previous["record"]}')
    write_json(target, dict(id=record['id'], record=location(path)))


def registered_records():
    """Read tiny explicit receipts only; unavailable storage remains visible."""
    result = []
    for path in sorted((REPO / 'output/registry/registrations').glob('*.json')):
        entry = json.loads(path.read_text())
        receipt = resolve_reference(entry['record'])
        if receipt.is_file():
            record = validate_record(json.loads(receipt.read_text()))
            if record['id'] != entry['id']:
                raise ValueError(f'{path}: registered identity differs from {receipt}')
            result.append(dict(entry, availability='available', record_data=record,
                               sha256=file_hash(receipt)))
        else:
            result.append(dict(entry, availability='unavailable', record_data=None, sha256=None))
    return result


def verify_record(path):
    """User-facing evidence audit; never recompute scientific quantities."""
    record = validate_record(json.loads(Path(path).read_text()))
    counts = dict(artifacts=0,hashed=0,large_unhashed=0,retired_caches=0,external_html_assets=0,
                  component_receipts=0)
    failures = []
    for component in record['components']:
        if component.get('receipt_sha256'):
            receipt = resolve_reference(component['receipt'])
            if not receipt.is_file():
                failures.append(f'Missing component receipt: {receipt}')
            elif file_hash(receipt) != component['receipt_sha256']:
                failures.append(f'Changed component receipt: {receipt}')
            counts['component_receipts'] += 1
    class Assets(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag not in {'script','link','img','iframe','source'}:
                return
            values = dict(attrs)
            value = values.get('src') or (values.get('href') if tag=='link' else None)
            if value:
                self.assets.append(value)
    seen = set()
    for analysis in record['analyses']:
        for artifact in analysis['artifacts']:
            if artifact['id'] in seen:
                continue
            seen.add(artifact['id']);counts['artifacts']+=1
            original = resolve_reference(artifact['path'])
            if not original.is_file():
                if artifact['role']=='cache':
                    counts['retired_caches']+=1
                    continue
                failures.append(f'Missing {artifact["role"]}: {original}')
                continue
            if artifact.get('sha256'):
                if file_hash(original)!=artifact['sha256']:
                    failures.append(f'Changed evidence: {original}')
                counts['hashed']+=1
            else:
                counts['large_unhashed']+=1
            if 'published_path' in artifact:
                alias = resolve_reference(artifact['published_path'])
                if artifact.get('publication_mode') == 'html_launcher':
                    if not alias.is_file() or file_hash(alias)!=artifact['published_sha256']:
                        failures.append(f'Changed or missing interactive launcher: {alias}')
                elif not alias.is_file() or alias.resolve()!=original.resolve():
                    failures.append(f'Broken or redirected publication link: {alias}')
            if original.suffix=='.html' and artifact['role']=='figure':
                parser=Assets();parser.assets=[]
                with original.open() as stream:
                    for block in iter(lambda:stream.read(65536),''):
                        parser.feed(block)
                for asset in parser.assets:
                    url=urlsplit(asset)
                    if url.scheme in {'http','https'} or url.netloc:
                        counts['external_html_assets']+=1
                    elif not url.scheme and url.path and not (original.parent/unquote(url.path)).exists():
                        failures.append(f'{original}: missing relative asset {asset}')
    if failures:
        raise ValueError('Result evidence verification failed:\n'+'\n'.join(failures))
    return dict(id=record['id'],state='verified',**counts,
                note='Recorded hashes and links checked; this is not a scientific recalculation or a claim of scheduler liveness.')


def ingest_records(con, entries):
    """Extend the existing evidence SQLite database without modifying raw records."""
    con.executescript('''
      CREATE TABLE IF NOT EXISTS result_runs(
        id TEXT PRIMARY KEY, kind TEXT, activity TEXT, title TEXT, study_json TEXT,
        execution_json TEXT, evidence_json TEXT, interpretation_json TEXT,
        receipt TEXT, receipt_sha256 TEXT, availability TEXT);
      CREATE TABLE IF NOT EXISTS result_components(
        id TEXT, run_id TEXT REFERENCES result_runs(id), payload_json TEXT,
        PRIMARY KEY(id,run_id));
      CREATE TABLE IF NOT EXISTS result_evaluations(
        id TEXT, run_id TEXT REFERENCES result_runs(id), payload_json TEXT,
        PRIMARY KEY(id,run_id));
      CREATE TABLE IF NOT EXISTS result_artifacts(
        id TEXT, evaluation_id TEXT, run_id TEXT, role TEXT, path TEXT,
        sha256 TEXT, payload_json TEXT, PRIMARY KEY(id,evaluation_id,run_id),
        FOREIGN KEY(evaluation_id,run_id) REFERENCES result_evaluations(id,run_id));
    ''')
    for entry in entries:
        if entry['record_data'] is None:
            con.execute('INSERT INTO result_runs(id,receipt,availability) VALUES(?,?,?) '
                        'ON CONFLICT(id) DO UPDATE SET availability=excluded.availability',
                        (entry['id'],entry['record'],'unavailable'))
            continue
        r = entry['record_data']
        con.execute('INSERT INTO result_runs VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET '
                    'kind=excluded.kind,activity=excluded.activity,title=excluded.title,study_json=excluded.study_json,'
                    'execution_json=excluded.execution_json,evidence_json=excluded.evidence_json,'
                    'interpretation_json=excluded.interpretation_json,receipt=excluded.receipt,'
                    'receipt_sha256=excluded.receipt_sha256,availability=excluded.availability',
                    (r['id'],r['kind'],r['activity'],r['title'],json.dumps(r['study']),
                     json.dumps(r['execution']),json.dumps(r['evidence']),json.dumps(r['interpretation']),
                     entry['record'],entry['sha256'],entry['availability']))
        for component in r['components']:
            con.execute('INSERT INTO result_components VALUES(?,?,?) ON CONFLICT(id,run_id) DO UPDATE SET '
                        'payload_json=excluded.payload_json',
                        (component['id'],r['id'],json.dumps(component)))
        for a in r['analyses']:
            con.execute('INSERT INTO result_evaluations VALUES(?,?,?) ON CONFLICT(id,run_id) DO UPDATE SET '
                        'payload_json=excluded.payload_json', (a['id'],r['id'],json.dumps(a)))
            for artifact in a['artifacts']:
                con.execute('INSERT INTO result_artifacts VALUES(?,?,?,?,?,?,?) '
                            'ON CONFLICT(id,evaluation_id,run_id) DO UPDATE SET role=excluded.role,'
                            'path=excluded.path,sha256=excluded.sha256,payload_json=excluded.payload_json',
                            (artifact['id'],a['id'],r['id'],artifact['role'],artifact['path'],
                             artifact.get('sha256'),json.dumps(artifact)))


def refresh_results():
    entries = registered_records()
    database = REPO / 'output/encoder_research/catalogue/technical/results.sqlite'
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database, timeout=60) as con:
        con.execute('PRAGMA foreign_keys=ON')
        ingest_records(con, entries)
        errors = con.execute('PRAGMA foreign_key_check').fetchall()
        if errors:
            raise ValueError(f'Result catalogue foreign key errors: {errors}')
    root = REPO / 'output/registry'
    refresh_status = root/'technical/historical-refresh-status.json'
    historical_status = ''
    if refresh_status.exists():
        status = json.loads(refresh_status.read_text())
        if status['state'] == 'failed':
            historical_status = ('<aside><strong>Historical catalogue refresh failed</strong><p>'
                + html.escape(status['error']) + '</p><p>The linked historical catalogue has its own capture date. '
                '<a href="technical/historical-refresh-status.json">Refresh receipt</a></p></aside>')
    cards = []
    inventory_summary = root/'technical/inventory-coverage.json'
    if inventory_summary.exists():
        summary=json.loads(inventory_summary.read_text())
        for kind,count in summary['counts'].items():
            cards.append(f'<article data-kind="{html.escape(kind)}"><h2>{html.escape(kind.title())} inventory</h2>'
                f'<p>{count} filesystem groups · observed {html.escape(summary["observed_at"])}. '
                'These groups are not counts of independent fits or completed science.</p>'
                f'<a href="inventory.html#kind={quote(kind)}">Browse retained inventory</a></article>')
    for entry in entries:
        r = entry['record_data']
        if r is None:
            cards.append(f'<article data-kind="unclassified"><h2>{html.escape(entry["id"])}</h2><p>Source unavailable: '
                         f'{html.escape(entry["record"])}</p></article>')
            continue
        links = []
        for a in r['analyses']:
            if 'page' in a:
                links.append(f'<li><a href="{href(a["page"],root)}">{html.escape(a["title"])}</a> · '
                             f'{len(a["artifacts"])} artifacts</li>')
        study = r['study']
        protocol = (f'<a href="{href(study["record"],root)}">Scientific protocol</a> · '
                    if study.get('record') else '')
        status = ' · '.join(f'{key}: {r[key]["state"]}' for key in ('execution','evidence','interpretation'))
        cards.append(f'<article data-kind="{r["kind"]}"><small>{html.escape(r["activity"])}</small>'
                     f'<h2>{html.escape(r["title"])}</h2><p>{html.escape(status)}</p><p>{protocol}'
                     f'<a href="{href(entry["record"],root)}">Result receipt</a></p><ul>{"".join(links)}</ul>'
                     f'<p>{len(r["components"])} declared components · {len(r["analyses"])} recorded evaluations</p>'
                     f'<p>{html.escape(r["interpretation"].get("note", ""))}</p></article>')
    now = datetime.now(timezone.utc).isoformat()
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Materials research results</title><style>
body{font:16px system-ui;max-width:1120px;margin:40px auto;padding:0 24px;background:#f4f6f8;color:#172a3a}
a{color:#1553a1}article{background:white;border:1px solid #d5dfe8;padding:22px;margin:18px 0;border-radius:10px}
input,select{font:inherit;padding:10px;margin-right:10px}p{line-height:1.6}li{margin:10px 0}small{color:#516678}
article[hidden]{display:none}</style><h1>Materials research results</h1>
<p>Questions, evaluations and their evidence. Execution, evidence coverage and interpretation are separate.</p>
__HISTORICAL_STATUS__
<p><a href="../encoder_research/catalogue/index.html">Historical evidence catalogue</a> ·
<a href="inventory.html">Filesystem inventory</a> · <a href="../../experiments/README.md">Study index</a> ·
<a href="../../docs/research_results_system.md">How results are organized</a></p>
<p>__COVERAGE__</p><input id="search" placeholder="Search question, run or analysis">
<select id="kind"><option value="research">Research</option><option value="operations">Operations</option>
<option value="simulation">Simulations</option><option value="dataset">Datasets</option>
<option value="unclassified">Unclassified / unavailable</option><option value="">All records</option></select>
__CARDS__<script>const s=document.querySelector('#search'),k=document.querySelector('#kind');
function filter(){for(const c of document.querySelectorAll('article'))c.hidden=Boolean((k.value&&c.dataset.kind!==k.value)||!c.textContent.toLowerCase().includes(s.value.toLowerCase()));}
s.oninput=k.onchange=filter;filter();</script></html>'''
    coverage = dict(generated_at=now,registered=len(entries),available=sum(e['availability']=='available' for e in entries),
                    scope='Explicit registrations only; historical evidence and filesystem inventory have separate coverage.')
    write_json(root/'technical/result-coverage.json',coverage)
    write_json(root/'technical/registered-results.json',entries)
    if (root/'index.html').exists() and not (root/'inventory.html').exists():
        (root/'inventory.html').write_bytes((root/'index.html').read_bytes())
    temporary = root / f'.index.{os.getpid()}.html'
    temporary.write_text(page.replace('__HISTORICAL_STATUS__',historical_status).replace('__CARDS__',''.join(cards)).replace('__COVERAGE__',html.escape(
        f'{coverage["available"]}/{len(entries)} registered records available · refreshed {now}. '+coverage['scope'])))
    temporary.replace(root/'index.html')
    historical = REPO/'output/encoder_research/catalogue/index.html'
    if historical.exists():
        page = historical.read_text()
        if '../../registry/index.html' not in page:
            page = page.replace('<h1>Encoder research catalogue</h1>',
                '<h1>Encoder research catalogue</h1><p><a href="../../registry/index.html">'
                'Registered runs and analysis bundles</a></p>')
            temporary = historical.with_name(f'.index.{os.getpid()}.html')
            temporary.write_text(page);temporary.replace(historical)
    return coverage
