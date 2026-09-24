"""Read-only, provenance-preserving catalogue of encoder research evidence.

CSV strings, JSON objects and Markdown table cells retain their producer's units.
This collector does not infer scientific equivalence, completion or metric direction.
"""
from __future__ import annotations

import csv
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import os
from pathlib import Path
import sqlite3
from urllib.parse import quote

from src.project_runtime.paths import resolve_path
from .metric_docs import snapshot_metric_docs

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / 'docs/encoder_research'
OUTPUT = REPO / 'output/encoder_research/catalogue'


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csv_records(data: bytes, path: str):
    """Preserve column order, repeated/empty headers, blanks and literal NaNs."""
    rows = csv.reader(io.StringIO(data.decode('utf-8-sig'), newline=''))
    header = next(rows, [])
    records = []
    for ordinal, row in enumerate(rows, 1):
        if not row:  # Empty physical lines are not CSV records.
            continue
        if len(row) != len(header):
            raise ValueError(f'{path}: CSV record {ordinal}: {len(row)} fields, header has {len(header)}')
        records.append((ordinal, row))
    return header, records


def markdown_records(text: str):
    """Index literal table lines, retaining raw text when pipes occur in cells."""
    header = None
    previous = None
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.startswith('|'):
            previous = header = None
            continue
        cells = [x.strip() for x in line.strip().strip('|').split('|')]
        if cells and all(x and set(x) <= set('-: ') for x in cells):
            header = previous
        elif header is not None:
            yield line_number, {'header_line': header, 'row_line': line}
        previous = line


def artifact_kind(path: Path):
    if path.suffix.lower() == '.csv':
        return 'csv'
    if path.suffix.lower() == '.md':
        return 'report'
    if path.suffix.lower() == '.json' and (
        path.name in {'metrics.json', 'analysis_metrics.json', 'summary.json', 'results.json', 'comparison.json'}
        or path.name.endswith(('_metrics.json', '_summary.json'))
    ):
        return 'json_result'
    if path.suffix.lower() == '.html':
        return 'gallery'
    return None


def walk_collection(root: Path, excluded: set[str]):
    """Traverse linked analysis directories once per collection, never link cycles."""
    seen = set()
    for directory, dirs, files in os.walk(root, followlinks=True):
        real = Path(directory).resolve()
        if real in seen:
            dirs[:] = []
            continue
        seen.add(real)
        dirs[:] = sorted(d for d in dirs if d not in excluded and not d.startswith('.'))
        for name in sorted(files):
            path = Path(directory) / name
            if artifact_kind(path):
                yield path


def source_link(source: str, path: str):
    return 'storage/' + quote(source) + '/' + quote(path, safe='/')


def validate_highlight(row, sources):
    """A curated value must carry an exact, checked quotation of its local evidence."""
    path = sources[row['source']] / row['path']
    data = path.read_bytes()
    if row['evidence_text'] not in data.decode('utf-8'):
        raise ValueError(f'Headline {row["id"]}: evidence text changed or is missing in {path}')
    return digest(data)


def render_studies(manifest):
    lines = ['# Encoder research study index', '',
             '[Handbook](README.md) · [Searchable artifact catalogue](../../output/encoder_research/catalogue/index.html)', '',
             'Generated from registered study records. Dates identify the research record,',
             'not necessarily a completed fit. Read its findings/status before treating a',
             'planned arm as evidence. Active records supersede the archived study entry;',
             'original archive artifacts remain indexed separately.', '']
    for family in manifest['families']:
        studies = [s for s in manifest['studies'] if s['family'] == family['id']]
        if not studies:
            continue
        lines += ['## ' + family['title'], '', family['description'], '',
                  '| Study | Location |', '| --- | --- |']
        for study in studies:
            target = ('../../' + study['path'] if study['source'] == 'repo' else
                      '../../output/encoder_research/catalogue/' + source_link(study['source'], study['path']))
            lines.append(f'| [{study["title"]}]({target}) | {study["source"]}: `{study["id"]}` |')
        lines.append('')
    lines += ['Additional GeoFrame shooting/predictive-atlas and older sweep collections',
              'do not all have dated study READMEs. Their artifacts remain searchable',
              'under the corresponding family; absence here does not imply no experiment.', '']
    return '\n'.join(lines)


def make_database(path: Path):
    con = sqlite3.connect(path)
    con.executescript('''
      PRAGMA foreign_keys=ON;
      CREATE TABLE families(id TEXT PRIMARY KEY, title TEXT, description TEXT);
      CREATE TABLE studies(id TEXT PRIMARY KEY, family TEXT REFERENCES families(id),
        title TEXT, source TEXT, path TEXT, evidence TEXT, status TEXT);
      CREATE TABLE record_sets(id TEXT PRIMARY KEY);
      CREATE TABLE artifacts(id TEXT PRIMARY KEY, family TEXT REFERENCES families(id),
        source TEXT, path TEXT, kind TEXT, sha256 TEXT, bytes INTEGER,
        row_count INTEGER, import_status TEXT, columns_json TEXT, definitions TEXT,
        implementation_contract TEXT, link TEXT, recordset_id TEXT REFERENCES record_sets(id));
      CREATE TABLE record_data(recordset_id TEXT REFERENCES record_sets(id), ordinal INTEGER,
        kind TEXT, payload_json TEXT, PRIMARY KEY(recordset_id,ordinal));
      CREATE VIEW records AS
        SELECT a.id AS artifact_id, r.ordinal, r.kind, r.payload_json
        FROM artifacts a JOIN record_data r ON a.recordset_id=r.recordset_id;
      CREATE TABLE headlines(id TEXT PRIMARY KEY, comparison_group TEXT,
        family TEXT REFERENCES families(id), model TEXT, metric TEXT, value REAL,
        unit TEXT, direction TEXT, population TEXT, split TEXT, horizon_ps TEXT,
        seeds TEXT, evidence TEXT, source TEXT, path TEXT, evidence_text TEXT,
        caveat TEXT, sha256 TEXT);
      CREATE INDEX artifact_family ON artifacts(family,kind);
      CREATE INDEX artifact_hash ON artifacts(sha256);
      CREATE VIEW csv_cells AS
        SELECT r.artifact_id, r.ordinal, h.key AS column_index,
               h.value AS column_name, v.value AS value
        FROM records r JOIN artifacts a ON a.id=r.artifact_id,
             json_each(a.columns_json) h, json_each(r.payload_json) v
        WHERE r.kind='csv' AND h.key=v.key;
      CREATE VIEW duplicate_artifacts AS
        SELECT sha256, COUNT(*) AS copies FROM artifacts
        WHERE sha256 IS NOT NULL GROUP BY sha256 HAVING COUNT(*)>1;
    ''')
    return con


def nearby_contract(path, root):
    for directory in [path.parent, *path.parent.parents]:
        if not directory.is_relative_to(root):
            break
        description = directory / 'tables/METRICS.md'
        if directory.name == 'tables':
            description = directory / 'METRICS.md'
        contract = directory / 'technical/metric-contract.json'
        if directory.name == 'tables':
            contract = directory.parent / 'technical/metric-contract.json'
        if description.is_file():
            return str(description.relative_to(root)), str(contract.relative_to(root)) if contract.is_file() else ''
    return '', ''


def render_dashboard(manifest, artifacts, headlines, stats, output):
    # Embedded data works with file://; no network, JS libraries or server needed.
    navigation = [{k: a[k] for k in ('family','source','path','kind','row_count',
                                     'import_status','definitions','link')} for a in artifacts]
    payload = json.dumps({'families': manifest['families'], 'studies': manifest['studies'],
                          'artifacts': navigation, 'headlines': headlines}, ensure_ascii=False).replace('<', '\\u003c')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Encoder research catalogue</title><style>
body{font:16px system-ui;max-width:1450px;margin:30px auto;padding:0 24px;color:#17263b;background:#f5f7fa}a{color:#1553a1}h1{margin-bottom:8px}p{max-width:1000px;line-height:1.5}input,select,button{padding:10px;margin:4px;border:1px solid #a6b5c5;border-radius:5px;background:white}input{width:320px}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{text-align:left;padding:10px;border-bottom:1px solid #dae1e9;vertical-align:top;overflow-wrap:anywhere}th{position:sticky;top:0;background:#e4edf6}small{color:#4c5c70}.bar{position:sticky;top:0;background:#f5f7fa;padding:10px 0;z-index:2}.limit{color:#8a3c11}button{cursor:pointer}
</style><h1>Encoder research catalogue</h1>
<p>__STATS__ · <a href="../../../docs/encoder_research/README.md">Research guide</a> · <a href="technical/results.sqlite">SQLite database</a> · <a href="tables/headlines.csv">Curated results CSV</a> · <a href="tables/artifacts.csv">All artifacts CSV</a></p>
<p class="limit">Different cohorts and metrics are not a leaderboard. Historical, smoke, superseded and reported-only evidence remains visible. Repeated files and report tables are not independent fits. Source files are linked, never modified.</p>
<div class="bar"><label>Search <input id="search" placeholder="model, protocol, metric or path"></label><label>Family <select id="family"><option value="">All families</option></select></label><label>View <select id="view"><option value="headlines">Curated results</option><option value="studies">Study records</option><option value="artifacts">Reports, tables and galleries</option></select></label><label>Kind <select id="kind"><option value="">All kinds</option><option>csv</option><option>report</option><option>json_result</option><option>gallery</option></select></label><button id="previous">Previous</button><button id="next">Next</button><span id="count"></span></div><div id="table"></div>
<script id="data" type="application/json">__PAYLOAD__</script><script>
const data=JSON.parse(document.getElementById('data').textContent);let page=0;const size=80;
const el=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const link=r=>r.link||('storage/'+encodeURIComponent(r.source)+'/'+r.path.split('/').map(encodeURIComponent).join('/'));
for(const f of data.families){const o=document.createElement('option');o.value=f.id;o.textContent=f.title;el('family').append(o)}
function draw(){const view=el('view').value,q=el('search').value.toLowerCase(),f=el('family').value,k=el('kind').value;const rows=data[view].filter(r=>(!f||r.family===f)&&(!k||view!=='artifacts'||r.kind===k)&&JSON.stringify(r).toLowerCase().includes(q));page=Math.min(page,Math.max(0,Math.ceil(rows.length/size)-1));el('count').textContent=` ${rows.length} matches · page ${page+1}/${Math.max(1,Math.ceil(rows.length/size))}`;
const columns=view==='headlines'?['comparison_group','model','metric','value','unit','population','split','seeds','evidence','caveat']:view==='studies'?['family','title','evidence','status']:['family','kind','path','row_count','import_status','definitions'];
el('table').innerHTML='<table><thead><tr>'+columns.map(c=>'<th>'+esc(c)+'</th>').join('')+'<th>Source</th></tr></thead><tbody>'+rows.slice(page*size,(page+1)*size).map(r=>'<tr>'+columns.map(c=>'<td>'+esc(r[c])+'</td>').join('')+'<td><a href="'+esc(link(r))+'">Open</a></td></tr>').join('')+'</tbody></table>';el('previous').disabled=page===0;el('next').disabled=(page+1)*size>=rows.length}
for(const id of ['search','family','view','kind'])el(id).addEventListener('input',()=>{page=0;draw()});el('previous').onclick=()=>{page--;draw()};el('next').onclick=()=>{page++;draw()};draw();
</script></html>'''
    summary = f'{stats["studies"]} study records · {stats["artifacts"]:,} artifacts · {stats["records"]:,} imported records · captured {stats["captured_at"]}'
    (output / 'index.html').write_text(page.replace('__STATS__', html.escape(summary)).replace('__PAYLOAD__', payload))


def build(repo=REPO, *, manifest_path=None, output=None, highlights_path=None):
    manifest_path = Path(manifest_path or repo / 'docs/encoder_research/catalogue.json')
    output = Path(output or repo / 'output/encoder_research/catalogue')
    manifest = json.loads(manifest_path.read_text())
    if manifest['schema_version'] != 1:
        raise ValueError('Unsupported encoder catalogue schema')
    sources = {name: resolve_path(path) for name, path in manifest['sources'].items()}
    for name, path in sources.items():
        if not path.is_dir():
            raise FileNotFoundError(f'Catalogue source {name} unavailable: {path}; configure machine.local.yaml before refresh')
    highlights_path = Path(highlights_path or repo / 'docs/encoder_research/highlights.json')
    headlines = json.loads(highlights_path.read_text())
    for row in headlines:
        row['sha256'] = validate_highlight(row, sources)
    for study in manifest['studies']:
        if not (sources[study['source']] / study['path']).is_file():
            raise FileNotFoundError(f'Missing study record: {study}')
    for part in ('technical', 'tables', 'storage'):
        (output / part).mkdir(parents=True, exist_ok=True)
    for name, path in sources.items():
        link = output / 'storage' / name
        if link.is_symlink():
            if link.resolve() != path.resolve():
                raise ValueError(f'Wrong catalogue storage link: {link}')
        elif link.exists():
            raise ValueError(f'Expected a storage symlink, found {link}')
        else:
            link.symlink_to(path, target_is_directory=True)
    temporary = output / 'technical/results.building.sqlite'
    if temporary.exists():
        temporary.unlink()  # Only this collector's incomplete database.
    with closing(make_database(temporary)) as con:
        for row in manifest['families']:
            con.execute('INSERT INTO families VALUES(?,?,?)', (row['id'], row['title'], row['description']))
        for row in manifest['studies']:
            con.execute('INSERT INTO studies VALUES(?,?,?,?,?,?,?)', tuple(row[k] for k in ('id','family','title','source','path','evidence','status')))
        headline_columns = [r[1] for r in con.execute('PRAGMA table_info(headlines)')]
        for row in headlines:
            con.execute('INSERT INTO headlines VALUES('+','.join('?' for _ in headline_columns)+')', [row[k] for k in headline_columns])
        artifacts, seen, recordsets = [], set(), set()
        collections = list(manifest['collections'])
        # Include study report tables even when old output metrics are missing.
        collections += [{'source':s['source'], 'path':str(Path(s['path']).parent), 'family':s['family']} for s in manifest['studies']]
        for collection in collections:
            source = collection['source']; base = sources[source]; root = base / collection['path']
            if not root.is_dir():
                raise FileNotFoundError(f'Registered evidence collection missing: {root}')
            for path in walk_collection(root, set(manifest['skip_directories'])):
                relative = str(path.relative_to(base)); identifier = source + ':' + relative
                if identifier in seen:
                    continue
                seen.add(identifier)
                kind = artifact_kind(path); header = []; records = []; reason = 'imported'
                size = path.stat().st_size
                # Galleries are linked; they can embed enormous point clouds.
                if kind == 'gallery':
                    data = None; reason = 'linked_gallery'
                elif size > manifest['max_import_bytes']:
                    data = None; reason = 'index_only:over_import_byte_limit'
                elif path.name in manifest['index_only_csv']:
                    data = None; reason = 'index_only:per_observation_coordinates'
                else:
                    data = path.read_bytes(); size = len(data)
                    try:
                        if kind == 'csv':
                            header, records = csv_records(data, identifier)
                        elif kind == 'report':
                            records = list(markdown_records(data.decode('utf-8')))
                        else:
                            # Preserve legacy nonfinite JSON tokens as strings, not valid numeric scores.
                            value = json.loads(data, parse_constant=lambda token: token)
                            records = [(1, value)]
                    except (UnicodeError, ValueError) as exc:
                        raise ValueError(f'Cannot catalogue {identifier}: {exc}') from exc
                definitions, contract = nearby_contract(path, base)
                content_hash = digest(data) if data is not None else None
                recordset = kind + ':' + content_hash if records else None
                if recordset is not None and recordset not in recordsets:
                    con.execute('INSERT INTO record_sets VALUES(?)', (recordset,))
                    con.executemany('INSERT INTO record_data VALUES(?,?,?,?)',
                                    ((recordset,n,kind,json.dumps(v,ensure_ascii=False,allow_nan=False)) for n,v in records))
                    recordsets.add(recordset)
                row = dict(id=identifier, family=collection['family'], source=source, path=relative,
                           kind=kind, sha256=content_hash, bytes=size,
                           row_count=len(records), import_status=reason, columns_json=json.dumps(header),
                           definitions=definitions, implementation_contract=contract, link=source_link(source,relative),
                           recordset_id=recordset)
                con.execute('INSERT INTO artifacts VALUES('+','.join('?' for _ in row)+')', tuple(row.values()))
                artifacts.append(row)
        count = sum(a['row_count'] for a in artifacts)
        stored_count = con.execute('SELECT count(*) FROM record_data').fetchone()[0]
        if con.execute('PRAGMA integrity_check').fetchone()[0] != 'ok' or con.execute('PRAGMA foreign_key_check').fetchall():
            raise RuntimeError('Encoder catalogue SQLite integrity check failed')
        con.commit()
    temporary.replace(output / 'technical/results.sqlite')
    stats = {'captured_at':datetime.now(timezone.utc).isoformat(), 'families':len(manifest['families']),
             'studies':len(manifest['studies']), 'collections':len(manifest['collections']),
             'artifacts':len(artifacts), 'records':count, 'stored_records':stored_count, 'headlines':len(headlines),
             'manifest_sha256':digest(manifest_path.read_bytes()), 'highlights_sha256':digest(highlights_path.read_bytes()),
             'missing_metric_definition_artifacts':sum(a['kind'] in ('csv','json_result') and not a['definitions'] for a in artifacts),
             'index_only_artifacts':sum(a['import_status'].startswith('index_only:') for a in artifacts),
             'warning':'Records include duplicate exports, diagnostics and historical evidence. Counts are not numbers of fits or independent observations.'}
    for name, rows in [('artifacts',artifacts), ('headlines',headlines), ('studies',manifest['studies'])]:
        with (output / f'tables/{name}.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    snapshot_metric_docs(output, 'encoder_research')
    write_json(output / 'technical/coverage.json',stats)
    write_json(output / 'technical/catalogue-snapshot.json',manifest)
    render_dashboard(manifest,artifacts,headlines,stats,output)
    if manifest_path == repo / 'docs/encoder_research/catalogue.json':
        (manifest_path.parent / 'studies.md').write_text(render_studies(manifest))
    (output / 'README.md').write_text('# Encoder research evidence catalogue\n\n'
        '[Search reports, tables and galleries](index.html) · [Research guide](../../../docs/encoder_research/README.md)\n\n'
        +f'{stats["studies"]} study records, {stats["artifacts"]:,} artifacts, {count:,} imported records and {len(headlines)} curated result entries.\n\n'
        'These counts include copies and historical diagnostic rows; they are not independent fits. '
        'See [coverage and capture time](technical/coverage.json), [definitions](tables/METRICS.md), '
        '[SQLite database](technical/results.sqlite), and [all artifacts](tables/artifacts.csv).\n')
    print(json.dumps(stats,indent=2))
    return stats
