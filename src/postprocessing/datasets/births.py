"""Publish and read the cumulative Al nucleus-birth dataset without relabelling MD.

Completed v3 searches supply immutable event shards; a consolidated SQLite index
and copied source/frame manifests make consumers independent of dated studies.
Updates replace the current index. Existing readers keep their open database.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import fnmatch
from functools import lru_cache
import json
import os
from pathlib import Path
import shutil
import sqlite3

import numpy as np

from src.experiment_runner.artifacts import file_hash as sha, json_digest as digest, write_json
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_rows
from src.project_runtime.paths import REPO, catalog, resolve_path, storage_path
from src.postprocessing.datasets.cadence import select_views, view_id, retain_views

DEFAULT_CONFIG = REPO / 'configs/data/nuclei_birth.json'
FAMILY = 'nuclei_birth_dataset'
SCHEMA = 1
SELECTIONS = {
    'established': "e.record_kind='root' AND e.origin='emergence_from_liquid' AND e.established=1",
    'appearances': "e.record_kind='root' AND e.origin='emergence_from_liquid'",
    'dissolved': "e.record_kind='root' AND e.origin='emergence_from_liquid' AND e.established=0 AND e.terminal_status='dissolved'",
    'all': '1=1',
}


def read(path):
    return json.loads(Path(path).read_text())


def compact(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False)


def checked_plan(path):
    p = read(path)
    if digest({k: v for k, v in p.items() if k != 'identity'}) != p['identity']:
        raise ValueError(f'Changed frozen plan: {path}')
    return p


def scientific_producers(c, files):
    """Normalize only explicitly audited implementations for stable event IDs.

    The original paths are scientific identity keys, not live import locations.
    A split implementation must match its entire reviewed dependency set.
    Actual execution files remain untouched in the search's provenance.
    """
    producer = {name: c.get('producer_equivalences', {}).get(name, {}).get(value, value)
                for name, value in files.items()}
    for revision in c.get('execution_equivalences', []):
        if all(files.get(name) == value for name, value in revision['files'].items()):
            producer.update(revision['scientific_producers'])
    return producer


def discover(c):
    """Registered searches only; never infer production purpose from a folder name."""
    searches, pending = [], []
    for key, entry in sorted(catalog().items()):
        if not any(fnmatch.fnmatchcase(key, pattern) for pattern in c['search_dataset_patterns']):
            continue
        root = storage_path(entry['root']) / entry['path']
        complete = root / 'technical/complete.json'
        if not complete.is_file():
            pending.append(dict(dataset=key, path=str(root), state='search_not_complete'))
            continue
        p = checked_plan(root / 'technical/plan.json')
        done = read(complete)
        if done['identity'] != p['identity'] or done['jobs'] != len(p['jobs']):
            raise ValueError(f'Incomplete search receipt: {root}')
        parent_path = resolve_path(p['config']['parent_output']) / 'technical/plan.json'
        if sha(parent_path) != p['parent_plan_sha256']:
            raise ValueError(f'Changed extraction plan: {parent_path}')
        parent = checked_plan(parent_path)
        if parent['identity'] != p['parent_identity']:
            raise ValueError(f'Extraction identity differs: {root}')
        # Only explicitly audited, output-equivalent execution revisions may
        # share this scientific protocol. Keep actual hashes in each search's
        # provenance; an unknown implementation still fails below.
        producer = scientific_producers(c, p['producer_files'])
        for expected, observed in [(c['lineage'], p['config']), (c['structure'], parent['config']),
                                   (c['scientific_producers'], producer)]:
            changed = [k for k, v in expected.items() if observed.get(k) != v]
            if changed:
                raise ValueError(f'Incompatible birth protocol in {key}: {changed}. Use a distinct dataset protocol.')
        # A native 0.10-ps addition need not manufacture a 0.30-ps view.
        # Validate every supplied view; select_views applies each dataset's
        # exact/native saved-cadence policy independently at publication.
        supplied = parent['config']['cadences']
        if not set(supplied).intersection(c['cadences']):
            raise ValueError(f'No supported cadence in {key}')
        for cadence, dt in supplied.items():
            # Historical searches also retain retired 0.75-ps labels. They
            # remain upstream and are never selected into these dataset IDs.
            if cadence in c['cadences'] and c['cadences'][cadence] != dt:
                raise ValueError(f'Wrong cadence in {key}: {cadence}')
        backend = parent.get('descriptor_backend')
        if backend is not None:
            if backend not in c.get('accepted_descriptor_backends', []):
                raise ValueError(f'Unreviewed descriptor backend in {key}')
            evidence = read(parent_path.parent / 'gpu-complete.json')
            if (evidence['parent_identity'] != parent['identity']
                    or evidence['state'] != 'complete'
                    or set(evidence['sources']) != {s['id'] for s in parent['sources']}):
                raise ValueError(f'Incomplete GPU extraction provenance in {key}')
        searches.append(dict(dataset=key, root=root, plan=p, parent=parent,
                             plan_sha256=sha(root / 'technical/plan.json'),
                             complete_sha256=sha(complete)))
    if not searches:
        raise ValueError('No completed compatible registered nucleus searches')
    return searches, pending


def inventory(searches, pending):
    """Expose unsearched Al holdings; coverage is not a prediction-negative label."""
    included = defaultdict(set)
    for search in searches:
        for s in search['parent']['sources']:
            included[s['dataset']].update(s['views'])
    rows = []
    for key, e in sorted(catalog().items()):
        m = e.get('metadata', {})
        if 'Al' not in m.get('materials', []):
            continue
        role = m.get('role', 'unknown')
        state = ('searched' if key in included else
                 'temporal_search_not_in_dataset' if role in ('raw_dynamics', 'simulation', 'storage_conversion')
                 else 'paired_or_nonproduction_holding')
        rows.append(dict(dataset=key, state=state, registered_state=e.get('state'),
                         role=role, cadences=','.join(sorted(included[key])),
                         potential=compact(m.get('potential_ids', []))))
    return dict(holdings=rows, pending_searches=pending)


def connect(path, readonly=False):
    db = sqlite3.connect(path.as_uri() + '?mode=ro' if readonly else str(path), uri=readonly)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA foreign_keys=ON')
    return db


def create_schema(db):
    db.executescript('''
      CREATE TABLE sources (
        source_id TEXT PRIMARY KEY, source TEXT NOT NULL, dataset TEXT NOT NULL,
        ancestor TEXT NOT NULL, role TEXT NOT NULL, source_origin TEXT NOT NULL,
        native_cadence_ps REAL NOT NULL, payload TEXT NOT NULL);
      CREATE TABLE views (
        view_id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources,
        cadence TEXT NOT NULL, cadence_ps REAL NOT NULL, frame_count INTEGER NOT NULL,
        first_time_ps REAL NOT NULL, last_time_ps REAL NOT NULL,
        UNIQUE(source_id,cadence));
      CREATE TABLE tasks (
        task_id TEXT PRIMARY KEY, view_id TEXT NOT NULL REFERENCES views,
        detector TEXT NOT NULL, search_dataset TEXT NOT NULL, search_identity TEXT NOT NULL,
        folder TEXT NOT NULL, files TEXT NOT NULL, event_count INTEGER NOT NULL,
        UNIQUE(view_id,detector));
      CREATE TABLE events (
        event_id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks,
        event_row INTEGER NOT NULL, track INTEGER NOT NULL, episode INTEGER NOT NULL,
        record_kind TEXT NOT NULL, origin TEXT NOT NULL, established INTEGER NOT NULL,
        terminal_status TEXT NOT NULL, fate TEXT NOT NULL, peak_atoms INTEGER NOT NULL,
        first_frame INTEGER NOT NULL, first_time_ps REAL NOT NULL,
        establishment_confirmation_ps REAL, growth_confirmation_ps REAL,
        UNIQUE(task_id,track,episode));
      CREATE TABLE ancestry (
        ancestor TEXT PRIMARY KEY, historical_roles TEXT NOT NULL,
        role_conflict INTEGER NOT NULL, source_records INTEGER NOT NULL);
      CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
      CREATE INDEX events_selection ON events(origin,record_kind,established,task_id);
      CREATE INDEX events_task ON events(task_id);
      CREATE INDEX views_cadence ON views(cadence);
    ''')


def add_source(db, s, c, cadences):
    sid = s['input_identity']
    old = db.execute('SELECT payload FROM sources WHERE source_id=?', (sid,)).fetchone()
    if old:
        previous = json.loads(old['payload'])
        # Duplicate exports may have different paths/IDs, but not different science.
        keys = ('times_ps', 'frames', 'views', 'ancestry_group', 'role', 'potential', 'native_cadence_ps')
        if any(previous[k] != s[k] for k in keys):
            raise ValueError(f'Conflicting duplicate source {sid}: {previous["id"]} vs {s["id"]}')
    else:
        db.execute('INSERT INTO sources VALUES (?,?,?,?,?,?,?,?)',
               (sid, s['id'], s['dataset'], s['ancestry_group'], s['role'], s['origin'],
                s['native_cadence_ps'], compact(s)))
    times = np.asarray(s['times_ps'], dtype=float)
    if len(times) != len(s['frames']) or np.any(np.diff(times) <= 0):
        raise ValueError(f'Invalid frame timeline: {s["id"]}')
    for cadence in cadences:
        dt = c['cadences'][cadence]
        frames = np.asarray(s['views'][cadence], dtype=int)
        if len(frames) < 2 or not np.allclose(np.diff(times[frames]), dt, rtol=0, atol=1e-8):
            raise ValueError(f'Nonexact {dt} ps view: {s["id"]}')
        vid = digest(dict(input_identity=sid, cadence=cadence, frames=frames.tolist()))
        db.execute('INSERT OR IGNORE INTO views VALUES (?,?,?,?,?,?,?)',
                   (vid, sid, cadence, dt, len(frames), times[frames[0]], times[frames[-1]]))


def add_task(db, search, task, source, protocol_identity):
    folder = search['root'] / 'technical/sources' / task['source'] / task['cadence'] / task['detector']
    receipt = read(folder / 'complete.json')
    if receipt['identity'] != digest(dict(plan=search['plan']['identity'], task=task)):
        raise ValueError(f'Wrong tracking receipt: {folder}')
    view = db.execute('SELECT view_id FROM views WHERE source_id=? AND cadence=?',
                      (source['input_identity'], task['cadence'])).fetchone()['view_id']
    tid = digest(dict(view=view, detector=task['detector'], protocol=protocol_identity))
    old = db.execute('SELECT * FROM tasks WHERE task_id=?', (tid,)).fetchone()
    if old:
        if json.loads(old['files']) != receipt['files']:
            raise ValueError(f'Conflicting labels for existing source/cadence/detector: {folder}')
        return tid, False
    for name, expected in receipt['files'].items():
        if sha(folder / name) != expected:
            raise ValueError(f'Changed lineage artifact: {folder / name}')
    es = read(folder / 'events.json')
    if len(es) != receipt['counts']['events']:
        raise ValueError(f'Wrong event count: {folder}')
    db.execute('INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?)',
               (tid, view, task['detector'], search['dataset'], search['plan']['identity'],
                str(folder), compact(receipt['files']), len(es)))

    def rows():
        for i, e in enumerate(es):
            if (e['source'], e['detector'], e['ancestor'], e['role']) != (
                    source['id'], task['detector'], source['ancestry_group'], source['role']):
                raise ValueError(f'Event/source mismatch: {folder}, row {i}')
            eid = f'{tid}:{e["track"]}:{e["episode"]}'
            yield (eid, tid, i, e['track'], e['episode'], e['record_kind'], e['origin'],
                   int(e['establishment_confirmation_ps'] is not None), e['terminal_status'],
                   e['fate'], e['peak_atoms'], e['first_frame'], e['first_time_ps'],
                   e['establishment_confirmation_ps'], e['growth_confirmation_ps'])
    db.executemany('INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', rows())
    return tid, True


def summarise(db):
    counts = [dict(r) for r in db.execute('''
      SELECT v.cadence,v.cadence_ps,t.detector,e.record_kind,e.origin,
        count(*) AS records,sum(e.established) AS established,
        sum(e.established=0 AND e.terminal_status='dissolved') AS dissolved_before_establishment,
        sum(e.growth_confirmation_ps IS NOT NULL) AS grew_to_128
      FROM events e JOIN tasks t USING(task_id) JOIN views v USING(view_id)
      GROUP BY v.cadence,t.detector,e.record_kind,e.origin
      ORDER BY v.cadence,t.detector,e.record_kind,e.origin''')]
    views = [dict(r) for r in db.execute('''
      SELECT v.cadence,v.cadence_ps,count(*) AS source_views,
        count(DISTINCT s.ancestor) AS recorded_ancestry_groups,sum(v.frame_count) AS frames,
        sum(v.last_time_ps-v.first_time_ps) AS observed_span_ps
      FROM views v JOIN sources s USING(source_id) GROUP BY v.cadence ORDER BY v.cadence''')]
    return dict(counts=counts, cadences=views,
                indexed_events=db.execute('SELECT count(*) FROM events').fetchone()[0],
                indexed_tasks=db.execute('SELECT count(*) FROM tasks').fetchone()[0])


def refresh(c):
    root = resolve_path(c['output'])
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'update.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        contract = check_metric_docs(family=FAMILY)[FAMILY]
        searches, pending = discover(c)
        coverage = inventory(searches, pending)
        bindings = [dict(dataset=x['dataset'], identity=x['plan']['identity'],
                         plan_sha256=x['plan_sha256'], complete_sha256=x['complete_sha256'],
                         parent_plan_sha256=x['plan']['parent_plan_sha256']) for x in searches]
        identity = digest(dict(schema=SCHEMA, config=c, searches=bindings, producer=contract['files']))
        old = read(root / 'manifest.json') if (root / 'manifest.json').exists() else None
        if old and old['fingerprint'] == identity:
            manifest = old
            if sha(root / 'catalogue.sqlite') != manifest['files']['catalogue.sqlite']:
                raise ValueError(f'Changed published database: {root}')
        else:
            stage = root / f'.updating-{os.getpid()}'
            stage.mkdir(parents=True)
            dbpath = stage / 'catalogue.sqlite'
            if old:
                previous = {x['dataset']: x for x in old['searches']}
                current = {x['dataset']: x for x in bindings}
                if any(current.get(k) != v for k, v in previous.items()):
                    raise ValueError('Update removes or changes indexed searches; explicitly review the source list')
                science = ('lineage', 'structure', 'scientific_producers')
                if old['schema_version'] == SCHEMA and all(old['config'][k] == c[k] for k in science):
                    if sha(root / 'catalogue.sqlite') != old['files']['catalogue.sqlite']:
                        raise ValueError('Changed previous database')
                    shutil.copyfile(root / 'catalogue.sqlite', dbpath)
            db = connect(dbpath)
            if not db.execute("SELECT name FROM sqlite_master WHERE name='events'").fetchone():
                create_schema(db)
            protocol = digest(dict(lineage=c['lineage'], structure=c['structure'], producers=c['scientific_producers']))
            selected, eligibility, raw_cache = {}, [], {}
            for search in searches:
                for s in search['parent']['sources']:
                    cadences, rows = select_views(s, c, raw_cache)
                    selected[(search['dataset'], s['id'])] = cadences
                    eligibility.extend(dict(row, search_dataset=search['dataset']) for row in rows)
            kept = {view_id(s, cadence) for search in searches for s in search['parent']['sources']
                    for cadence in selected[(search['dataset'], s['id'])]}
            removed_views = retain_views(db, kept)
            seen, added = set(), 0
            for search in searches:
                source_map = {s['id']: s for s in search['parent']['sources']}
                for s in source_map.values():
                    cadences = selected[(search['dataset'], s['id'])]
                    if cadences:
                        add_source(db, s, c, cadences)
                tasks = [t for t in search['plan']['jobs']
                         if t['cadence'] in selected[(search['dataset'], t['source'])]]
                for i, task in enumerate(tasks):
                    tid, new = add_task(db, search, task, source_map[task['source']], protocol)
                    seen.add(tid); added += new
                    if (i + 1) % 100 == 0:
                        db.commit()
                        print(compact(dict(search=search['dataset'], indexed=i+1, total=len(tasks))), flush=True)
                db.commit()
            if len(seen) != db.execute('SELECT count(*) FROM tasks').fetchone()[0]:
                raise ValueError('Unexpected removed tracking tasks during cumulative update')
            db.execute('DELETE FROM ancestry')
            groups = defaultdict(lambda: dict(roles=set(), sources=0))
            for s in db.execute('SELECT ancestor,role FROM sources'):
                groups[s['ancestor']]['roles'].add(s['role']); groups[s['ancestor']]['sources'] += 1
            db.executemany('INSERT INTO ancestry VALUES (?,?,?,?)',
                           ((a, compact(sorted(v['roles'])), int(len(v['roles']) > 1), v['sources'])
                            for a, v in sorted(groups.items())))
            db.execute("INSERT OR REPLACE INTO metadata VALUES ('fingerprint',?)", (identity,))
            db.commit()
            if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok' or db.execute('PRAGMA foreign_key_check').fetchall():
                raise ValueError('Published SQLite index failed integrity verification')
            summary = summarise(db)
            dest = stage / 'analyses/catalogue'
            write_metric_rows(summary['counts'], dest, family=FAMILY, name='counts')
            write_metric_rows(summary['cadences'], dest, family=FAMILY, name='cadences')
            write_metric_rows(eligibility, dest, family=FAMILY, name='source-eligibility')
            write_metric_rows((dict(r) for r in db.execute('SELECT * FROM ancestry ORDER BY ancestor')),
                              dest, family=FAMILY, name='ancestry')
            source_columns = ('source_id','source','dataset','ancestor','role','source_origin','native_cadence_ps')
            write_metric_rows((dict(r) for r in db.execute('SELECT '+','.join(source_columns)+' FROM sources ORDER BY source_id')),
                              dest, family=FAMILY, name='sources')
            birth_columns = ('event_id','source_id','source','dataset','ancestor','role','source_origin',
                             'cadence','cadence_ps','detector','track','first_frame','first_time_ps',
                             'establishment_confirmation_ps','growth_confirmation_ps','peak_atoms','fate')
            write_metric_rows((dict(r) for r in db.execute('''
                SELECT e.event_id,v.source_id,s.source,s.dataset,s.ancestor,s.role,s.source_origin,
                v.cadence,v.cadence_ps,t.detector,e.track,e.first_frame,e.first_time_ps,
                e.establishment_confirmation_ps,e.growth_confirmation_ps,e.peak_atoms,e.fate
                FROM events e JOIN tasks t USING(task_id) JOIN views v USING(view_id)
                JOIN sources s USING(source_id)
                WHERE e.record_kind='root' AND e.origin='emergence_from_liquid' AND e.established=1
                ORDER BY v.cadence,t.detector,v.source_id,e.first_time_ps''')),
                dest, family=FAMILY, name='established-births', columns=birth_columns)
            write_json(stage / 'technical/coverage-at-publication.json', coverage)
            write_json(stage / 'technical/searches.json', [dict(dataset=x['dataset'], root=str(x['root']),
                       config=x['plan']['config'], producer=x['plan']['producer_files'],
                       extraction_coverage=x['parent']['coverage'],
                       descriptor_backend=x['parent'].get('descriptor_backend', {'name': 'historical_cpu'})) for x in searches])
            db.close()
            manifest = dict(schema_version=SCHEMA, dataset=c['dataset_id'], fingerprint=identity,
                created_at=datetime.now(timezone.utc).isoformat(), config=c, searches=bindings,
                producer=contract['files'], summary=summary, files={},
                input_contract=dict(geometry='Referenced original periodic MD; exact frame maps and atom identities',
                    relaxation='No relaxation; use the shared relaxation index for paired views',
                    encoder=None, predictor=None, conditions='Audit metadata only; never model inputs',
                    population='Compatible completed v3 Al searches, filtered by declared saved-cadence policy; prediction eligibility not applied',
                    sampling_policy=c['sampling_policy'],
                    split='Historical roles unchanged; group by ancestor and review role conflicts before any fit'))
            for cadence, dt in c['cadences'].items():
                write_json(stage / 'cadences' / cadence / 'manifest.json', dict(dataset=f'{c["dataset_id"]}-{cadence}',
                    fingerprint=identity, cadence=cadence, cadence_ps=dt, database='../../catalogue.sqlite',
                    sampling_policy=c['sampling_policy'][cadence],
                    default_detector='ptm', default_selection='established'))
            lines = ['# Al nuclei birth dataset', '',
                     'Stable IDs: `al-nuclei-birth-010` and `al-nuclei-birth-030`.', '',
                     f'Dataset fingerprint: `{identity}`.', '',
                     '0.10 ps: exact saved-frame views, including explicitly recorded finer native data.',
                     '0.30 ps: only native 0.30 ps saved trajectories; no finer or mixed-cadence raw parts.',
                     'Established births, dissolved embryos,',
                     'composites and unresolved origins retain their separate labels and original ancestry.', '',
                     '| Cadence ps | Source views | PTM established liquid births | Bond order established liquid births |',
                     '|---|---:|---:|---:|']
            for v in summary['cadences']:
                n = {r['detector']: r['established'] for r in summary['counts'] if
                     r['cadence'] == v['cadence'] and r['record_kind'] == 'root' and r['origin'] == 'emergence_from_liquid'}
                lines.append(f'| {v["cadence_ps"]} | {v["source_views"]} | {n.get("ptm",0)} | {n.get("coherent",0)} |')
            lines += ['', 'Views, detectors, related descendants and historical prefixes are not independent nuclei or melts.',
                      'No controls or crystal-free prediction histories are synthesized by publication.',
                      'Precursor tails and first-crystal labels are separate supplements with their own completion manifests.', '',
                      '[Counts](analyses/catalogue/tables/counts.csv) · [Sources](analyses/catalogue/tables/sources.csv) ·',
                      '[Ancestry](analyses/catalogue/tables/ancestry.csv) · [Definitions](analyses/catalogue/tables/METRICS.md)', '']
            (stage / 'README.md').write_text('\n'.join(lines))
            manifest['files'] = {str(p.relative_to(stage)): sha(p) for p in sorted(stage.rglob('*')) if p.is_file()}
            write_json(stage / 'manifest.json', manifest)
            # Readers take this same lock while opening. Existing SQLite readers
            # keep the old inode until close; no historical dataset copies remain.
            for p in sorted(stage.rglob('*')):
                if p.is_file() and p.name != 'manifest.json':
                    target = root / p.relative_to(stage)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    p.replace(target)
            for cadence in c['cadences']:
                target = root / 'cadences' / cadence / 'manifest.json'
                target.parent.mkdir(parents=True, exist_ok=True)
                (stage / 'cadences' / cadence / 'manifest.json').replace(target)
            (stage / 'manifest.json').replace(root / 'manifest.json')
            shutil.rmtree(stage)
            print(compact(dict(new_tasks=added, removed_views=removed_views, fingerprint=identity)), flush=True)
        write_json(root / 'status.json', dict(observed_at=datetime.now(timezone.utc).isoformat(),
                    fingerprint=identity, **coverage))
        return dict(fingerprint=identity, path=str(root), **manifest['summary'])


class NucleiBirthDataset:
    """Read the current cadence. Metadata is label/audit data, not a feature tensor."""
    def __init__(self, cadence, *, expected_fingerprint=None, config=DEFAULT_CONFIG):
        self.config = read(config)
        if cadence not in self.config['cadences']:
            raise ValueError(f'Choose an exact cadence: {list(self.config["cadences"])}')
        self.root = resolve_path(self.config['output'])
        with (self.root / 'update.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            self.manifest = read(self.root / 'manifest.json')
            self.cadence_manifest = read(self.root / 'cadences' / cadence / 'manifest.json')
            self.manifest_sha256 = sha(self.root / 'manifest.json')
            self.fingerprint = self.manifest['fingerprint']
            if self.cadence_manifest['fingerprint'] != self.fingerprint:
                raise ValueError('Interrupted cadence publication; rerun refresh before opening')
            if expected_fingerprint is not None and expected_fingerprint != self.fingerprint:
                raise ValueError('Dataset changed since this experiment was prepared; prepare a new cohort explicitly')
            self.db = connect(self.root / 'catalogue.sqlite', readonly=True)
            self.db.execute('BEGIN')
            if self.db.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()[0] != self.fingerprint:
                self.db.close()
                raise ValueError('Interrupted dataset publication; rerun refresh before opening')
        self.cadence = cadence

    def close(self):
        self.db.close()
        self._events.cache_clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def events(self, *, selection='established', detector='ptm', source_origin=None):
        if selection not in SELECTIONS or detector not in self.config['structure']['detectors']:
            raise ValueError('Unknown selection or structural detector')
        sql = '''SELECT e.*,v.cadence,v.cadence_ps,v.source_id,t.detector,
          s.source,s.dataset,s.ancestor,s.role,s.source_origin,s.native_cadence_ps,a.role_conflict
          FROM events e JOIN tasks t USING(task_id) JOIN views v USING(view_id)
          JOIN sources s USING(source_id) JOIN ancestry a USING(ancestor)
          WHERE v.cadence=? AND t.detector=? AND ''' + SELECTIONS[selection]
        params = [self.cadence, detector]
        if source_origin is not None:
            sql += ' AND s.source_origin=?'; params.append(source_origin)
        for row in self.db.execute(sql + ' ORDER BY v.source_id,e.first_time_ps,e.track', params):
            yield dict(row)

    @lru_cache(maxsize=1)
    def _events(self, task_id):
        t = self.db.execute('SELECT * FROM tasks WHERE task_id=?', (task_id,)).fetchone()
        path = Path(t['folder']) / 'events.json'
        if sha(path) != json.loads(t['files'])['events.json']:
            raise ValueError(f'Changed event shard: {path}')
        return read(path)

    def event(self, event_id):
        r = self.db.execute('''SELECT e.*,v.cadence,v.source_id FROM events e
            JOIN tasks t USING(task_id) JOIN views v USING(view_id) WHERE event_id=?''', (event_id,)).fetchone()
        if r is None or r['cadence'] != self.cadence:
            raise KeyError(f'Event absent from this cadence: {event_id}')
        return dict(self._events(r['task_id'])[r['event_row']], event_id=event_id,
                    source_id=r['source_id'], task_id=r['task_id'], fingerprint=self.fingerprint)

    def source(self, source_id):
        r = self.db.execute('''SELECT payload FROM sources JOIN views USING(source_id)
            WHERE source_id=? AND cadence=?''', (source_id, self.cadence)).fetchone()
        if r is None:
            raise KeyError(source_id)
        return json.loads(r['payload'])

    def pin(self, path):
        """Write this exact identity into the consuming experiment's provenance."""
        write_json(path, dict(dataset=f'{self.config["dataset_id"]}-{self.cadence}',
            fingerprint=self.fingerprint, cadence_ps=self.config['cadences'][self.cadence],
            manifest_sha256=self.manifest_sha256,
            database_sha256=self.manifest['files']['catalogue.sqlite']))

    def verify(self, *, upstream=False):
        for name, expected in self.manifest['files'].items():
            if sha(self.root / name) != expected:
                raise ValueError(f'Changed dataset artifact: {self.root / name}')
        if self.db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok' or self.db.execute('PRAGMA foreign_key_check').fetchall():
            raise ValueError('Dataset database integrity failure')
        if upstream:
            for t in self.db.execute('SELECT folder,files FROM tasks'):
                for name, expected in json.loads(t['files']).items():
                    if sha(Path(t['folder']) / name) != expected:
                        raise ValueError(f'Changed upstream artifact: {t["folder"]}/{name}')
        return dict(fingerprint=self.fingerprint, verified=True, upstream=upstream)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['refresh', 'status', 'verify'])
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--upstream', action='store_true')
    args = parser.parse_args()
    c = read(args.config)
    if args.stage == 'refresh':
        result = refresh(c)
    elif args.stage == 'verify':
        with NucleiBirthDataset('010', config=args.config) as data:
            result = data.verify(upstream=args.upstream)
    else:
        searches, pending = discover(c)
        coverage = inventory(searches, pending)
        root = resolve_path(c['output'])
        result = dict(observed_at=datetime.now(timezone.utc).isoformat(),
                      completed_searches=[x['dataset'] for x in searches], **coverage)
        if (root / 'manifest.json').exists():
            m = read(root / 'manifest.json')
            result.update(fingerprint=m['fingerprint'], summary=m['summary'])
        write_json(root / 'status.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
