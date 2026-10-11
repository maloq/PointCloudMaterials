"""Nested ancestry-held-out CV of the established/failed appearance assay."""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import yaml

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_rows
from src.project_runtime.paths import REPO, resolve_path
from .appearance import inputs, specs
from .data import read
from .merged_fit import fit

FAMILY = 'birth_appearance_cv'
STRATA = ('established', 'failed_strong', 'failed_other')
ROW_KEYS = ('id', 'source', 'event', 'pair', 'role', 'atom', 'label', 'weight', 'stratum')


def row_identity(rows):
    return digest({key: rows[key].tolist() for key in ROW_KEYS})


def feature_identity(x):
    return dict(shape=list(x.shape), dtype=str(x.dtype),
                sha256=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest())


def appearance_config(c):
    a = read(resolve_path(c['appearance_config']))
    bound = read(resolve_path(a['output']) / 'technical/prepared.json')
    if bound['config'] != a or bound['identity'] != c['appearance_identity']:
        raise ValueError('Appearance cohort/configuration changed')
    return a


def balance(rows, groups, eligible, folds, seed):
    """Greedy event/row balance, using no descriptors, predictions or conditions."""
    names = sorted(set(groups[eligible]))
    rng = np.random.default_rng(seed)
    names = [names[i] for i in rng.permutation(len(names))]
    vectors = []
    for name in names:
        ids = np.flatnonzero(eligible & (groups == name))
        counts = [len(set(zip(rows['source'][ids][(rows['label'][ids] == 1) &
            (rows['stratum'][ids] == s)], rows['event'][ids][(rows['label'][ids] == 1) &
            (rows['stratum'][ids] == s)]))) for s in STRATA]
        vectors.append([*counts, int(counts[1] > 0), len(ids), 1])
    vectors = np.asarray(vectors, float)
    totals = vectors.sum(0)
    scaled = vectors / np.maximum(totals, 1)
    order = sorted(range(len(names)), key=lambda i: (-scaled[i].max(), -scaled[i].sum()))
    loads = np.zeros((folds, scaled.shape[1]))
    assignments = {}
    for i in order:
        costs = [np.sum((loads + np.eye(folds)[:, k, None] * scaled[i])**2)
                 for k in range(folds)]
        chosen = int(np.argmin(costs))
        assignments[names[i]] = chosen
        loads[chosen] += scaled[i]
    if set(assignments.values()) != set(range(folds)):
        raise ValueError('Empty ancestry fold')
    return assignments


def support(rows, groups, assignment, folds):
    values = np.asarray([assignment[str(g)] for g in groups])
    records = []
    for fold in range(folds):
        for stratum in (*STRATA, 'combined'):
            mask = (values == fold) & ((rows['stratum'] == stratum) if stratum != 'combined' else True)
            positive = mask & (rows['label'] == 1)
            records.append(dict(fold=fold, stratum=stratum, rows=int(mask.sum()),
                sources=len(set(rows['source'][mask])), ancestors=len(set(groups[mask])),
                events=len(set(zip(rows['source'][positive], rows['event'][positive])))))
    return records


def prepare(c):
    contracts = check_metric_docs(family=FAMILY)
    a = appearance_config(c)
    b, rows, x, _, context, _ = inputs(a, next(specs(a)))
    if c['outer_folds'] != 5 or c['folds'] != 5 or c['seed'] != a['seed']:
        raise ValueError('This protocol declares five outer/five inner folds and one retained seed')
    sources = read(resolve_path(a['output']) / 'technical/data-plan.json')['original_plan']['sources']
    source_ancestry = {str(s['id']): s['lineage'] for s in sources}
    groups = np.asarray([source_ancestry[str(s)] for s in rows['source']])
    outer = balance(rows, groups, np.ones(len(groups), bool), c['outer_folds'], c['seed'])
    inner = {}
    for fold in range(c['outer_folds']):
        mask = np.asarray([outer[str(g)] != fold for g in groups])
        inner[str(fold)] = balance(rows, groups, mask, c['folds'], c['seed'])
        for pool in ('original', 'combined'):
            train = mask & ((rows['stratum'] == 'established') if pool == 'original' else True)
            assigned = np.asarray([inner[str(fold)][str(g)] for g in groups[train]])
            if set(assigned) != set(range(c['folds'])):
                raise ValueError(f'Missing inner-fold support: {fold}/{pool}')
            for k in range(c['folds']):
                for selected in (assigned == k, assigned != k):
                    if set(rows['label'][train][selected]) != {0, 1}:
                        raise ValueError(f'Missing inner outcome: {fold}/{pool}/{k}')
    coverage = support(rows, groups, outer, c['outer_folds'])
    if any(r['events'] == 0 for r in coverage if r['stratum'] == 'failed_strong'):
        raise ValueError('An outer fold contains no stronger failed event')
    record = dict(config=c, appearance=a, rows_identity=row_identity(rows), rows=len(x),
        source_ancestry=source_ancestry, outer=outer, inner=inner, support=coverage,
        readout_recipe={k: b[k] for k in ('linear_C', 'catboost')},
        input_context=context, implementation=contracts[FAMILY]['files'])
    record['identity'] = digest(record)
    root = resolve_path(c['output'])
    path = root / 'technical/design.json'
    if path.exists() and read(path) != record:
        raise ValueError('CV design changed; preserve this run and choose a new revision')
    write_json(path, record)
    write_metric_rows(coverage, root / 'analyses/coverage-v1', family=FAMILY, name='outer-fold-support')
    return record


def load(c):
    p = read(resolve_path(c['output']) / 'technical/design.json')
    if (p['config'] != c or digest({k: v for k, v in p.items() if k != 'identity'}) != p['identity']
            or p['implementation'] != check_metric_docs(family=FAMILY)[FAMILY]['files']):
        raise ValueError('Nested-CV design/producer changed; use the frozen execution bundle')
    return p


def fit_root(c, domain, spec, pool, fold):
    return (resolve_path(c['output']) / 'analyses/fits-v1' / domain / spec['arm'] /
            spec['observation']['name'] / pool / f'outer-{fold}')


def partition(p, rows, fold, pool):
    if row_identity(rows) != p['rows_identity']:
        raise ValueError('CV models/domains must use identical rows and weights')
    groups = np.asarray([p['source_ancestry'][str(s)] for s in rows['source']])
    held = np.asarray([p['outer'][str(g)] == fold for g in groups])
    training = ~held & ((rows['stratum'] == 'established') if pool == 'original' else True)
    if set(groups[training]) & set(groups[held]):
        raise ValueError('Ancestry crossed an outer split')
    ids = np.flatnonzero(training | held)
    local = {k: v[ids].copy() for k, v in rows.items()}
    local['role'] = np.where(training[ids], 'train', 'outer_test')
    inner = np.asarray([p['inner'][str(fold)][str(g)] for g in groups[training]])
    for k in range(p['config']['folds']):
        if set(groups[training][inner == k]) & set(groups[training][inner != k]):
            raise ValueError('Ancestry crossed an inner split')
    return ids, local, inner


def worker(c, domain, progress):
    p = load(c)
    a = p['appearance']
    for spec in specs(a):
        if spec['domain'] != domain:
            continue
        b, rows, x, _, ctx, _ = inputs(a, spec)
        if {k: b[k] for k in p['readout_recipe']} != p['readout_recipe']:
            raise ValueError('Readout hyperparameter recipe changed')
        features = feature_identity(x)
        feature_path = resolve_path(c['output']) / f'technical/inputs-{domain}-{spec["observation"]["name"]}.json'
        if feature_path.exists() and read(feature_path) != features:
            raise ValueError('Bound CV descriptor values changed')
        write_json(feature_path, features)
        for fold in range(c['outer_folds']):
            for pool in ('original', 'combined'):
                progress.update(arm=spec['arm'], observation=spec['observation']['name'],
                                outer_fold=fold, training_pool=pool)
                ids, local, inner = partition(p, rows, fold, pool)
                context = dict(ctx, selector='Inner ancestry-grouped five-fold binary NLL',
                    split='Exploratory nested five-fold CV across all eligible original source roles',
                    outer_fold=fold, training_pool=pool, cv_design_identity=p['identity'],
                    input_features=features,
                    label=('Established birth versus liquid; failed appearance is transfer evaluation'
                           if pool == 'original' else ctx['label']),
                    calibration='none; alarm threshold from inner training OOF negatives only')
                destination = fit_root(c, domain, spec, pool, fold)
                fit(c, b, local, x[ids], spec, destination, inner, context, family=FAMILY)
                write_json(destination / 'technical/outer-split.json', dict(design=p['identity'],
                    global_indices=ids.tolist(), original_role=rows['role'][ids].tolist(),
                    outer_fold=fold, training_pool=pool,
                    role_mapping={'train': 'outer_training', 'training_selection_cv': 'inner_selection',
                                  'merged_test': 'outer_test'}))
    from .appearance_cv_analysis import collect
    collect(c, p, domain, progress)


def submit(path):
    c = read(path)
    p = prepare(c)
    technical = resolve_path(c['output']) / 'technical'
    if (technical / 'launch.json').exists():
        raise ValueError('Already submitted; resume the recorded frozen worker')
    bundle = ExecutionBundle.freeze(Path(__file__).resolve().parents[3], technical / 'code', c,
        directories=('src', 'configs', 'docs/metrics'))
    # Freeze path resolution too: registry edits must not interrupt a running readout.
    machine_path = Path(os.environ.get('PCM_MACHINE_CONFIG', REPO / 'machine.local.yaml'))
    machine = yaml.safe_load(machine_path.read_text())
    machine['catalog'] = str(bundle.root / 'configs/datasets.json')
    (technical / 'machine.yaml').write_text(yaml.safe_dump(machine))
    record = dict(identity=p['identity'], code=str(bundle.root), jobs={}, outer_fits=80,
                  inner_folds=5, seed=c['seed'], relaxed_ready_job=c['relaxed_ready_job'])
    queue = SlurmQueue(technical, bundle, 'src.research.birth_prediction.appearance_cv',
        dict(PCM_PROJECT_ROOT=str(bundle.root), PCM_MACHINE_CONFIG=str(technical / 'machine.yaml'),
             OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'),
        technical / 'launch.json', record, 'APPEARANCE-CV')
    with queue.submission():
        for domain in p['appearance']['domains']:
            dependency = 'afterok:' + c['relaxed_ready_job'] if domain == 'relaxed' else None
            queue.submit(domain, ['--cpus-per-task=8', '--mem=24G', '--gpus=1', '--time=12:00:00',
                f'--exclude={p["appearance"]["excluded_gpu_node"]}'], dependency=dependency,
                partition=p['appearance']['gpu_partition'], command_stage='worker',
                arguments=('--domain', domain))
    return record


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('prepare', 'submit', 'worker', 'collect'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--domain', choices=('original', 'relaxed'))
    args = parser.parse_args()
    c = read(args.config)
    if args.stage == 'prepare':
        print(prepare(c)['support'])
    elif args.stage == 'submit':
        print(submit(args.config))
    else:
        if args.domain is None:
            parser.error('--domain is required for worker/collect')
        with recorded_stage(resolve_path(c['output']) / f'technical/{args.stage}-{args.domain}.json',
                            job=os.environ.get('SLURM_JOB_ID')) as progress:
            if args.stage == 'worker':
                worker(c, args.domain, progress)
            else:
                from .appearance_cv_analysis import collect
                collect(c, load(c), args.domain, progress)


if __name__ == '__main__':
    main()
