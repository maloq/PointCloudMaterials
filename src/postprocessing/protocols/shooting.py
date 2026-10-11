"""Detached full-cell birth/fate search over the registered Al shooting archive."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_rows, snapshot_metric_docs
from src.project_runtime.paths import resolve_path
from src.postprocessing.protocols.shooting_data import read, prepare, implementation
from src.postprocessing.protocols.shooting_fates import process

FAMILY = 'shooting_births'


def load(c):
    p = read(resolve_path(c['output']) / 'technical/plan.json')
    bound = {k: v for k,v in p.items() if k != 'identity'}
    if digest(bound) != p['identity'] or p['config'] != c or p['implementation'] != implementation():
        raise ValueError('Shooting population/configuration/producer changed; use the frozen execution bundle')
    return p


def initialize(path):
    global PLAN
    PLAN = load(read(path))


def worker_source(item):
    with recorded_stage(resolve_path(PLAN['config']['output']) / f'technical/sources/{item["id"]}/stage.json',
                        source=item['id'], job=os.environ.get('SLURM_JOB_ID')):
        return process(PLAN, item)


def worker(path, index):
    c = read(path)
    p = load(c)
    selected = sorted((s for s in p['sources'] if s['lane'] == index), key=lambda s: (-s['frame_count'], s['id']))
    with recorded_stage(resolve_path(c['output']) / f'technical/lane-{index}.json',
                        total=len(selected), completed=0, node=os.uname().nodename,
                        job=os.environ.get('SLURM_JOB_ID')) as progress:
        with ProcessPoolExecutor(max_workers=c['workers_per_lane'], mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize, initargs=(path,)) as pool:
            pending = [pool.submit(worker_source, item) for item in selected]
            completed = 0
            for future in as_completed(pending):
                r = future.result()
                completed += 1
                progress.update(completed=completed, last_source=r['source'], last_counts=r['counts'])
                print(json.dumps(dict(source=r['source'], seconds=r['seconds'], counts=r['counts'],
                    completed=completed, total=len(selected))), flush=True)


def collect(c):
    p = load(c)
    out = resolve_path(c['output'])
    transient, established, branches = [], [], []
    for item in p['sources']:
        folder = out / 'technical/sources' / item['id']
        receipt = read(folder / 'complete.json')
        expected = digest(dict(plan=p['identity'], source=item['id']))
        if receipt['identity'] != expected or any(sha(folder / name) != h for name,h in receipt['files'].items()):
            raise ValueError(f'Incomplete/changed branch result: {item["id"]}')
        transient.extend(read(folder / 'transient-episodes.json'))
        established.extend(read(folder / 'established-fates.json'))
        branches.append(dict(source=item['id'], campaign=item['campaign'], branch_id=item['branch_id'],
            parent_id=item['parent_id'], parent_sha256=item['parent_sha256'], lineage=item['ancestry_group'],
            role=item['role'], dynamics=item['dynamics'], frames=item['frame_count'],
            duration_ps=item['branch_observed_duration_ps'], **receipt['actual_durations'], **receipt['counts']))
    destination = out / 'analyses/births-fates-v1'
    snapshot_metric_docs(destination, FAMILY)
    for name, rows in [('transient-episodes',transient), ('established-fates',established),
                       ('branches',branches), ('unavailable-or-duplicate',p['unavailable'])]:
        if rows:
            columns = ('campaign', 'branch_id', 'reason', 'retained_source') if name == 'unavailable-or-duplicate' else None
            write_metric_rows(rows, destination, family=FAMILY, name=name, columns=columns)
    summary = []
    populations = [('establishments', established, 'kind'), ('established_fates', established, 'fate_label'),
        ('unestablished_episodes', transient, 'terminal_status'),
        ('brief_failed_candidates', [r for r in transient if r['brief_failed_candidate']], 'terminal_status'),
        ('sustained_failed_candidates', [r for r in transient if r['primary_failed_candidate']], 'terminal_status')]
    for campaign in ['all', *sorted({s['campaign'] for s in p['sources']})]:
        for population, rows, field in populations:
            subset = [r for r in rows if campaign == 'all' or r['campaign'] == campaign]
            for label in sorted({r[field] for r in subset}):
                selected = [r for r in subset if r[field] == label]
                summary.append(dict(campaign=campaign, population=population, label=label, count=len(selected),
                    branches=len({r['source'] for r in selected}), parents=len({r['parent_sha256'] for r in selected}),
                    ancestral_sources=len({r['lineage'] for r in selected})))
    if summary:
        write_metric_rows(summary, destination, family=FAMILY, name='counts')
    lines = ['# Al shooting: births and fates', '',
        f'Complete: {len(branches)} unique futures, {len({s["parent_sha256"] for s in branches})} parent configurations, '
        f'{len({s["lineage"] for s in branches})} ancestral source runs.', '',
        'Full-cell search on the exact 0.30 ps grid. Original 0.75 ps Al64 results remain a separate protocol.',
        'Parent-present clusters, new isolated establishments, interface-associated establishments, failed episodes and uncertain endings are separate.',
        'Counts are correlated branch-local events, not independent samples or nucleation rates. Nested original and continuation segments count once.', '',
        '| Population | Label | Count | Branches | Parents | Ancestors |', '|---|---|---:|---:|---:|---:|']
    lines += [f'| {r["population"]} | {r["label"]} | {r["count"]} | {r["branches"]} | {r["parents"]} | {r["ancestral_sources"]} |'
              for r in summary if r['campaign'] == 'all']
    lines += ['', 'These are harvest candidates. Strict crystal-free predictor histories and liquid controls still require a separate eligibility pass.',
              'Original MD coordinates only; no relaxation, simulation, fitting or training/test reassignment.', '',
              '[Counts](tables/counts.csv) · [Established fates](tables/established-fates.csv) · '
              '[Transient episodes](tables/transient-episodes.csv) · [Definitions](tables/METRICS.md)', '']
    (destination / 'README.md').write_text('\n'.join(lines))
    write_json(out / 'technical/complete.json', dict(identity=p['identity'], branches=len(branches), counts=summary))
    (out / 'README.md').write_text('# Al shooting birth search\n\n[Completed analysis](analyses/births-fates-v1/README.md)\n')


def submit(path):
    c = read(path)
    check_metric_docs(family=FAMILY)
    p = load(c)
    technical = resolve_path(c['output']) / 'technical'
    if (technical / 'launch.json').exists():
        raise ValueError('Already submitted; resume the failed lane from its frozen bundle')
    repo = Path(__file__).resolve().parents[3]
    bundle = ExecutionBundle.freeze(repo, technical / 'code', c, directories=('src', 'configs', 'docs/metrics'))
    record = dict(identity=p['identity'], code=str(bundle.root), branches=len(p['sources']), jobs={})
    queue = SlurmQueue(technical, bundle, 'src.postprocessing.protocols.shooting',
        dict(PCM_PROJECT_ROOT=str(repo), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
             OVITO_THREAD_COUNT='1', QT_QPA_PLATFORM='offscreen'), technical / 'launch.json', record, 'SHOT-BIRTH')
    with queue.submission():
        lanes = queue.submit('worker', [f'--array=0-{c["lanes"]-1}',
            f'--cpus-per-task={c["workers_per_lane"]}', '--mem=32G', f'--time={c["time_limit"]}'])
        queue.submit('collect', ['--cpus-per-task=1', '--mem=16G', '--time=01:00:00'], dependency='afterok:'+lanes)
    return record


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['prepare', 'source', 'worker', 'collect', 'submit'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--source-id')
    parser.add_argument('--index', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', '0')))
    args = parser.parse_args()
    c = read(args.config)
    if args.stage == 'prepare':
        check_metric_docs(family=FAMILY)
        p = prepare(c)
        print(json.dumps(read(resolve_path(c['output']) / 'technical/inventory.json'), indent=2))
    elif args.stage == 'submit':
        print(json.dumps(submit(args.config), indent=2))
    elif args.stage == 'worker':
        worker(args.config, args.index)
    elif args.stage == 'source':
        p = load(c)
        item = next(s for s in p['sources'] if s['id'] == args.source_id)
        print(json.dumps(process(p, item), indent=2))
    else:
        with recorded_stage(resolve_path(c['output']) / 'technical/collector.json', job=os.environ.get('SLURM_JOB_ID')):
            collect(c)


if __name__ == '__main__':
    main()
