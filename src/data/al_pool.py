"""Unified Al inventory and selectable, checksum-bound appearance/fate inputs.

Data preparation only: this module neither trains nor changes historical splits.
Metadata never becomes predictor input. Sealed observations retain native labels.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import getpass
import json
from pathlib import Path
import subprocess

import numpy as np

from src.experiment_runner.artifacts import file_hash as sha, json_digest as digest, write_json
from src.project_runtime.paths import REPO, catalog, resolve_path


def read(path):
    return json.loads(Path(path).read_text())


def rows_at(root):
    with np.load(root / 'rows.npz', allow_pickle=False) as a:
        return [{k: a[k][i].tolist() for k in a.files} for i in range(len(a['id']))]


def verify(root, files):
    for name, expected in files.items():
        if sha(root / name) != expected:
            raise ValueError(f'Changed upstream input: {root / name}')


def source_record(s, pool):
    earlier = pool == 'earlier'
    return dict(source=f'earlier:{s["id"]}' if earlier else s['id'], pool=pool,
        dataset=s['dataset'] if earlier or pool == 'dense010' else s['campaign'],
        ancestor=s['lineage'] if earlier else s['ancestry_group'],
        parent_sha256=None if earlier else s['parent_sha256'],
        historical_role=s['role'], material='Al' if earlier else s['material'],
        potential='al-lee2003-meam' if earlier else s['potential'],
        label_cadence_ps=.75 if earlier else s['cadence_ps'],
        temperature_K=s['audit_metadata']['temperature_K'] if earlier else s['temperature_K'],
        atom_count=s['atom_count'], frame_count=s['frame_count'],
        provenance={k: v for k, v in s.items() if k not in
                    ('frames', 'times_ps', 'center_atom_ids', 'legacy_center_atom_ids')})


class Builder:
    def __init__(self, config):
        self.config = config
        self.cache = resolve_path(config['cache'])
        self.sources, self.events, self.banks, self.rows = {}, {}, {}, []
        self.columns = None

    def event(self, source, event, stratum, fate, evidence):
        key = f'{source}/event/{event}'
        value = dict(id=key, source=source, stratum=stratum, fate=fate, evidence=evidence)
        if key in self.events and self.events[key] != value:
            raise ValueError(f'Conflicting event labels: {key}')
        self.events[key] = value
        return key

    def bank(self, name, features, columns, provenance, offsets, domain):
        if features.ndim != 3 or features.shape[1:] != (len(offsets), len(columns)):
            raise ValueError(f'Unexpected feature dimensions: {name}: {features.shape}')
        if not np.isfinite(features).all():
            raise ValueError(f'Nonfinite descriptors: {name}')
        if self.columns is not None and self.columns != columns:
            raise ValueError(f'Descriptor column definitions differ: {name}')
        self.columns = columns
        binding = dict(provenance=provenance, offsets_ps=offsets, domain=domain,
                       shape=list(features.shape), columns=columns)
        identity = digest(binding)
        path = self.cache / 'banks' / identity / 'features.npy'
        receipt = path.with_name('complete.json')
        if receipt.exists():
            done = read(receipt)
            if done['binding'] != binding or sha(path) != done['sha256']:
                raise ValueError(f'Changed pooled feature bank: {path}')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, features)
            done = dict(binding=binding, sha256=sha(path))
            write_json(receipt, done)
        self.banks[name] = dict(path=str(path), sha256=done['sha256'], identity=identity,
                               **binding)

    def add(self, row, bank, index, domain, offsets, source, fate, stratum):
        s = self.sources[source]
        event = f'{source}/event/{row["event"]}'
        if event not in self.events:
            raise ValueError(f'Missing audited event: {event}')
        # Appearance anchor and atom identify a paired raw/relaxed sample.
        sample = f'{source}/atom/{row["atom"]}/anchor/{row["appearance_frame"]}'
        label = int(row['label'])
        resolved = label == 1 and fate in ('dissolved', 'grew_then_dissolved', 'continued_growth')
        self.rows.append(dict(id=f'{sample}/{domain}', sample_id=sample,
            source=source, pool=s['pool'], dataset=s['dataset'], ancestor=s['ancestor'],
            parent_sha256=s['parent_sha256'], historical_role=s['historical_role'],
            material=s['material'], potential=s['potential'], temperature_K=s['temperature_K'],
            label_cadence_ps=s['label_cadence_ps'], domain=domain,
            stratum=stratum, appearance=label, fate=fate if label else 'liquid_control',
            dissolution=int(fate != 'continued_growth') if resolved else None,
            fate_resolved=resolved, matched_event=event, event=event if label else None,
            pair=f'{source}/pair/{row["pair"]}', atom=int(row['atom']),
            appearance_frame=int(row['appearance_frame']), offsets_ps=offsets,
            control_followup_ps=float(row['control_followup_ps']),
            bank=bank, bank_row=index, upstream_row_id=row['id']))


def assemble(c):
    """Merge completed physical inputs; keep incomplete collections in coverage."""
    from src.research.birth_prediction.appearance import original_config
    from src.research.birth_prediction.appearance_data import cache_config

    a = read(resolve_path(c['appearance_config']))
    sc = read(resolve_path(c['shooting_config']))
    dc = read(resolve_path(c['dense_config']))
    ap = read(resolve_path(a['output']) / 'technical/data-plan.json')
    sp = read(resolve_path(sc['output']) / 'technical/data-plan.json')
    dp = read(resolve_path(dc['output']) / 'technical/plan.json')
    for label, p in [('earlier', ap), ('shooting', sp), ('dense010', dp)]:
        if digest({k:v for k,v in p.items() if k != 'identity'}) != p['identity']:
            raise ValueError(f'Changed input plan: {label}')
    if ap['config'] != a or sp['config'] != sc or dp['config'] != dc:
        raise ValueError('Prepared cohort configuration changed; use its recorded recipe')
    if read(resolve_path(a['fate_output']) / 'technical/complete.json')['identity'] != a['fate_identity']:
        raise ValueError('Changed earlier fate audit identity')
    if sp['audit']['identity'] != sc['shooting_identity']:
        raise ValueError('Changed shooting audit identity')
    b = Builder(c)
    for pool, sources in [('earlier', ap['original_plan']['sources']),
                          ('shooting', sp['audit']['sources']), ('dense010', dp['sources'])]:
        for s in sources:
            r = source_record(s, pool)
            b.sources[r['source']] = r
    # Include screened-out candidates too: zero eligible histories is not no event.
    fate_root = resolve_path(a['fate_output']) / 'technical/sources'
    for s in ap['original_plan']['sources']:
        folder = fate_root / str(s['id'])
        receipt = read(folder / 'complete.json')
        verify(folder, {'established-fates.json': receipt['files']['established-fates.json']})
        for e in read(folder / 'established-fates.json'):
            if e['kind'] == 'isolated_establishment':
                b.event(f'earlier:{s["id"]}', str(e['event']), 'established', e['fate_label'], e)
    for e in ap['candidates']:
        b.event(f'earlier:{e["source"]}', str(-e['first_node']),
                'failed_strong' if e['primary_failed_candidate'] else 'failed_other', 'dissolved', e)
    for e in sp['candidates']:
        b.event(e['source'], e['event'], e['stratum'], e['fate'], e)
    for domain in ('original', 'relaxed'):
        for stratum, producer in [('established', original_config), ('failed', cache_config)]:
            config = producer(a, domain)
            root = resolve_path(config['cache'])
            manifest, desc = read(root / 'manifest.json'), read(root / 'descriptor-manifest.json')
            if not manifest['supported'] or desc['dataset_identity'] != manifest['identity']:
                raise ValueError(f'Unsealed descriptor cohort: {root}')
            verify(root, {'rows.npz': manifest['files']['rows.npz'], 'descriptors.npy': desc['sha256']})
            rows = rows_at(root)
            features = np.load(root / 'descriptors.npy', mmap_mode='r')
            bank = f'earlier-{stratum}-{domain}'
            offsets = c['protocols']['legacy075']['offsets_ps']
            b.bank(bank, features[np.asarray([r['indices'] for r in rows])], desc['columns'],
                   dict(root=str(root), manifest=manifest, descriptors=desc,
                        input_domain=config.get('input_domain')), offsets, domain)
            for i, r in enumerate(rows):
                if r['appearance_frame'] - r['end_frame'] != 1 or r['end_frame'] - r['start_frame'] != 7:
                    raise ValueError(f'Changed legacy observation timeline: {r["id"]}')
                source = f'earlier:{r["source"]}'
                e = b.events[f'{source}/event/{r["event"]}']
                r['control_followup_ps'] = (r['confirmation_frame'] - r['end_frame']) * .75
                b.add(r, bank, i, domain, offsets, source, e['fate'], e['stratum'])
    # This bank contains an older-data replay; include only its new shooting rows.
    root = resolve_path(sc['cache'])
    manifest = read(root / 'manifest.json')
    if manifest['data_plan'] != sp['identity']:
        raise ValueError('Changed shooting dataset binding')
    verify(root, manifest['files'])
    rows = rows_at(root)
    ids = [i for i, r in enumerate(rows) if r['pool'] == 'shooting']
    offsets = sc['offsets_ps']
    b.bank('shooting-original', np.load(root / 'features.npy', mmap_mode='r')[ids],
           sp['descriptor_columns'], dict(root=str(root), manifest=manifest), offsets, 'original')
    for index, i in enumerate(ids):
        r = rows[i]
        b.add(r, 'shooting-original', index, 'original', offsets, r['source'], r['fate'], r['stratum'])
    dense_root = resolve_path(dc['cache'])
    dense_ready = (dense_root / 'manifest.json').exists()
    if dense_ready:
        manifest = read(dense_root / 'manifest.json')
        if manifest['plan'] != dp['identity']:
            raise ValueError('Changed dense data-plan binding')
        verify(dense_root, manifest['files'])
        for s in dp['sources']:
            hp = read(dense_root / 'sources' / s['id'] / 'input-plan.json')
            if hp['audit']['identity'] != dp['identity']:
                raise ValueError(f'Changed dense source binding: {s["id"]}')
            for e in hp['candidates']:
                b.event(e['source'], e['event'], e['stratum'], e['fate'], e)
        b.bank('dense010-original', np.load(dense_root / 'features.npy', mmap_mode='r'),
               hp['descriptor_columns'], dict(root=str(dense_root), manifest=manifest),
               dc['offsets_ps'], 'original')
        for i, r in enumerate(rows_at(dense_root)):
            b.add(r, 'dense010-original', i, 'original', dc['offsets_ps'], r['source'], r['fate'], r['stratum'])
    ids = [r['id'] for r in b.rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate physical input/domain in unified pool')
    roles = defaultdict(set)
    for r in b.rows:
        roles[r['ancestor']].add(r['historical_role'])
    if any(len(v) != 1 for v in roles.values()):
        raise ValueError(f'Historical ancestry split conflict: {[(k,v) for k,v in roles.items() if len(v)>1]}')
    cv = read(resolve_path(sc['output']) / 'technical/design.json')
    for r in b.rows:
        r['historical_shooting_cv_fold'] = cv['outer'].get(r['ancestor'])
    eligible = defaultdict(set)
    for r in b.rows:
        if r['appearance']:
            eligible[r['event']].add(r['domain'])
    for e in b.events.values():
        e['eligible_domains'] = sorted(eligible[e['id']])
    return b, dict(dense010='sealed' if dense_ready else 'preparing; no released training rows',
        unprepared='Other Al collections are inventoried; task-specific eligibility remains unassessed.')


def inventory():
    """Include every registered Al collection; do not invent unknown materials."""
    registry = read(REPO / 'docs/datasets/registry.json')
    scanned = {d['id']: d for d in registry['datasets']}
    all_al, unknown = [], []
    for identifier, entry in catalog().items():
        materials = entry.get('metadata', {}).get('materials', [])
        if 'Al' not in materials:
            if not materials:
                unknown.append(identifier)
            continue
        record = dict(id=identifier, registration=entry, inventory=scanned.get(identifier),
                      scan_state='scanned' if identifier in scanned else 'registered_since_scan')
        all_al.append(record)
    return dict(registry_generated_at=registry['generated_at'], collections=all_al,
                unknown_material=unknown, remote_reported_copies=registry['remote_holdings'],
                count_policy='Collection entries include containers, derivatives and aliases; not independent sources.')


def status(c):
    studies = []
    for s in c['studies']:
        root = resolve_path(s['root'])
        receipt = root / s['receipt']
        studies.append(dict(id=s['id'], root=str(root), complete=receipt.exists(),
            completion_receipt=str(receipt), receipt_sha256=sha(receipt) if receipt.exists() else None))
    result = subprocess.run(['squeue', '--noheader', '--user', getpass.getuser(),
                             '--format=%i|%j|%T|%M|%R'], capture_output=True, text=True, check=True)
    jobs = [dict(zip(('job', 'name', 'state', 'elapsed', 'location'), line.split('|', 4)))
            for line in result.stdout.splitlines() if line.strip()]
    launch = resolve_path(c['simulation_launch'])
    plan = read(launch / 'manifest.json')
    counts = Counter()
    for r in plan['runs']:
        path = Path(plan['root']) / r['run_dir'] / 'status.json'
        counts[read(path)['state'] if path.exists() else 'no_status'] += 1
    return dict(checked_at=datetime.now(timezone.utc).isoformat(), studies=studies,
        user_jobs=jobs, dense_simulations=dict(total=len(plan['runs']), states=dict(counts), launch=str(launch)),
        overall='partial: finished studies, active preparation/simulation, and unassessed collections are distinct')


def refresh(c):
    b, coverage = assemble(c)
    root = resolve_path(c['output'])
    value = dict(schema_version=1, protocols=c['protocols'], rows=b.rows,
        sources=list(b.sources.values()), events=list(b.events.values()), banks=b.banks,
        descriptor_columns=b.columns, coverage=coverage, collections=inventory(),
        implementation_sha256=sha(Path(__file__)))
    value['identity'] = digest(value)
    path = root / 'releases' / (value['identity'] + '.json')
    write_json(path, value)
    write_json(root / 'current.json', dict(identity=value['identity'], manifest=str(path), sha256=sha(path)))
    progress = status(c)
    write_json(root / 'status.json', progress)
    lines = ['# All Al data pool', '', f'Updated: {progress["checked_at"]}.', '',
        f'Pool release: `{value["identity"]}`. [Manifest]({path}).', '',
        'Appearance and fate are labels on the same records. Raw/relaxed observations are paired views.',
        'All registered Al collections are included in the inventory; only eligible sealed inputs are training-ready.', '',
        '| Pool | Domain | Rows | Positive events | Ancestors |', '|---|---|---:|---:|---:|']
    for pool, domain in sorted({(r['pool'], r['domain']) for r in b.rows}):
        rr = [r for r in b.rows if (r['pool'], r['domain']) == (pool, domain)]
        lines.append(f'| {pool} | {domain} | {len(rr)} | {len({r["event"] for r in rr if r["appearance"]})} | {len({r["ancestor"] for r in rr})} |')
    lines += ['', 'Do not sum paired views or shooting branches as independent events/ancestors.', '',
        f'Native 0.1 ps preparation: **{coverage["dense010"]}**.',
        f'Dense simulation states: `{progress["dense_simulations"]["states"]}`.', '',
        '## Study completion', '', '| Study | State |', '|---|---|']
    for s in progress['studies']:
        lines.append(f'| [{s["id"]}]({s["root"]}) | {"finished" if s["complete"] else "not finished (no final receipt)"} |')
    lines += ['', '## Scheduler snapshot', '',
              'All user jobs are shown, including interactive allocations; names alone do not prove Al coverage.', '',
              '| Job | Name | State | Location/reason |', '|---|---|---|---|']
    for job in progress['user_jobs']:
        lines.append(f'| {job["job"]} | {job["name"]} | {job["state"]} | {job["location"]} |')
    lines += ['', f'[Selection instructions]({REPO / "docs/al_research_pool.md"}) · '
        f'[All dataset cards]({REPO / "docs/datasets/index.html"}) · '
        f'[Historical research results]({REPO / "output/registry/index.html"})', '',
        'Historical metrics are not recomputed or combined into a new aggregate score.']
    (root / 'README.md').write_text('\n'.join(lines) + '\n')
    return dict(identity=value['identity'], rows=len(b.rows), samples=len({r['sample_id'] for r in b.rows}),
        eligible_events=len({r['event'] for r in b.rows if r['appearance']}),
        registered_al_collections=len(value['collections']['collections']), overview=str(root / 'README.md'), coverage=coverage)


def load_pool(c, release=None):
    if release is None:
        pointer = read(resolve_path(c['output']) / 'current.json')
        release = Path(pointer['manifest'])
        if sha(release) != pointer['sha256']:
            raise ValueError('Changed pool release file')
    value = read(release)
    if digest({k:v for k,v in value.items() if k != 'identity'}) != value['identity']:
        raise ValueError('Changed pool manifest identity')
    return value


def select(pool, *, protocol, domain, task, filters=None):
    """Return selected metadata, N×T×442 geometry descriptors, labels, event weights.

    Source metadata/conditions are never concatenated to x. Historical splits and
    CV assignments remain metadata; use ancestor for every future partition.
    """
    if task not in ('appearance', 'dissolution') or domain not in ('original', 'relaxed'):
        raise ValueError(f'Unknown task/domain: {task}/{domain}')
    wanted = np.asarray(pool['protocols'][protocol]['offsets_ps'])
    rows, values, banks = [], [], {}
    keys = set(pool['rows'][0])
    if set(filters or {}) - keys:
        raise ValueError(f'Unknown cohort selectors: {set(filters)-keys}')
    for r in pool['rows']:
        if r['domain'] != domain or any(r[k] not in allowed for k, allowed in (filters or {}).items()):
            continue
        if task == 'dissolution' and not r['fate_resolved']:
            continue
        have = np.asarray(r['offsets_ps'])
        matches = np.isclose(wanted[:, None], have[None, :], rtol=0, atol=1e-8)
        if not np.all(matches.sum(1) == 1):
            continue
        bank = r['bank']
        if bank not in banks:
            p = pool['banks'][bank]
            if sha(p['path']) != p['sha256']:
                raise ValueError(f'Changed pooled features: {p["path"]}')
            banks[bank] = np.load(p['path'], mmap_mode='r')
        rows.append(dict(r, selected_offsets_ps=wanted.tolist()))
        values.append(banks[bank][r['bank_row'], matches.argmax(1)])
    if not rows:
        raise ValueError(f'No sealed observations match {protocol}/{domain}/{task}/{filters}')
    counts = Counter(r['matched_event'] for r in rows)
    return (rows, np.asarray(values), np.asarray([r[task] for r in rows], np.int64),
            np.asarray([1/counts[r['matched_event']] for r in rows]))


def submit_refresh(c):
    """Publish sealed dense inputs and final status without changing its jobs."""
    from src.experiment_runner.execution import ExecutionBundle, SlurmQueue
    import os
    dc = read(resolve_path(c['dense_config']))
    upstream = read(resolve_path(dc['output']) / 'technical/launch.json')
    technical = resolve_path(c['output']) / 'technical/dense-refresh'
    receipt = technical / 'launch.json'
    if receipt.exists():
        raise FileExistsError(f'Pool refresh jobs already submitted; see {receipt}')
    bundle = ExecutionBundle.freeze(REPO, technical / 'code', c, directories=('src', 'configs'))
    record = dict(upstream=upstream['jobs'], jobs={}, code=str(bundle.root))
    environment = dict(PCM_PROJECT_ROOT=str(REPO),
        PCM_MACHINE_CONFIG=os.environ.get('PCM_MACHINE_CONFIG', str(REPO / 'machine.local.yaml')),
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    queue = SlurmQueue(technical, bundle, 'src.data.al_pool', environment, receipt, record, 'AL-POOL')
    with queue.submission():
        sealed = queue.submit('sealed', ['--cpus-per-task=1', '--mem=6G', '--time=00:30:00'],
            dependency='afterany:' + upstream['jobs']['seal'], command_stage='refresh')
        queue.submit('evaluated', ['--cpus-per-task=1', '--mem=6G', '--time=00:30:00'],
            dependency='afterany:' + upstream['jobs']['fit'] + ':' + sealed, command_stage='refresh')
    return record


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=('refresh', 'select', 'status', 'submit-refresh'))
    parser.add_argument('--config', default='configs/al_research_pool.json')
    parser.add_argument('--release', type=Path)
    parser.add_argument('--protocol', choices=('legacy075', 'common150', 'dense010'))
    parser.add_argument('--domain', choices=('original', 'relaxed'))
    parser.add_argument('--task', choices=('appearance', 'dissolution'), default='appearance')
    parser.add_argument('--where', action='append', default=[], metavar='FIELD=VALUE')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    c = read(resolve_path(args.config))
    if args.action == 'refresh':
        result = refresh(c)
    elif args.action == 'submit-refresh':
        result = submit_refresh(c)
    elif args.action == 'status':
        result = status(c)
        write_json(resolve_path(c['output']) / 'status.json', result)
    else:
        if not args.protocol or not args.domain or not args.output:
            parser.error('select requires --protocol, --domain and --output')
        filters = defaultdict(list)
        for item in args.where:
            key, text = item.split('=', 1)
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                value = text
            filters[key].append(value)
        pool = load_pool(c, args.release)
        rows, x, y, weights = select(pool, protocol=args.protocol, domain=args.domain,
                                      task=args.task, filters=filters)
        out = args.output.resolve()
        if out.exists():
            raise FileExistsError(f'Preserve the frozen cohort; choose a new destination: {out}')
        out.mkdir(parents=True)
        np.save(out / 'features.npy', x)
        np.savez_compressed(out / 'targets.npz', label=y, weight=weights,
            ancestor=np.asarray([r['ancestor'] for r in rows]),
            historical_role=np.asarray([r['historical_role'] for r in rows]))
        write_json(out / 'rows.json', rows)
        result = dict(pool_identity=pool['identity'], protocol=args.protocol, domain=args.domain,
            task=args.task, filters=dict(filters), rows=len(rows), shape=list(x.shape),
            predictor_inputs='442 geometry/bond-order/CNA/TDA descriptors only', conditions=[],
            descriptor_columns=pool['descriptor_columns'],
            files={n:sha(out/n) for n in ('features.npy', 'targets.npz', 'rows.json')})
        result['input_context'] = dict(encoder=None, predictor=dict(
            bank='442 fixed geometry/bond-order/CNA/TDA descriptors', nearest_atoms=80,
            descriptor_radius_A=8., clearance_radius_A=8., offsets_ps=pool['protocols'][args.protocol]['offsets_ps'],
            velocities=False, species=False, conditions=[], relaxation=args.domain == 'relaxed'),
            training_only_teachers=None)
        write_json(out / 'manifest.json', result)
        result = {k:result[k] for k in ('pool_identity', 'protocol', 'domain', 'task', 'filters', 'rows', 'shape')}
        result['output'] = str(out)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
