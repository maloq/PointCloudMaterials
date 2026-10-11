"""Matched-cadence Al appearance transfer and preappearance nucleus-fate readouts."""
import argparse
from collections import Counter
import os
from pathlib import Path

import numpy as np
import yaml

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_rows
from src.project_runtime.paths import REPO, resolve_path
from . import shooting_appearance_data as data
from .appearance import original_config
from .appearance_cv import balance, feature_identity
from .data import read
from .merged_fit import fit

FAMILY = 'birth_shooting_appearance'
POOLS = ('earlier', 'shooting', 'combined')


def tasks():
    for task in ('appearance', 'fate'):
        for model, arm in (('linear', 'rich_linear'), ('catboost', 'rich_gbdt')):
            for observation in ('snapshot', 'history4'):
                for pool in POOLS if task == 'appearance' else ('combined',):
                    yield dict(task=task, model=model, arm=arm, pool=pool,
                               observation=dict(name=observation))


def cohort(rows, features, task):
    if task == 'appearance':
        return rows, features
    ids = np.flatnonzero((rows['label'] == 1) & np.isin(rows['fate'],
        ('dissolved', 'grew_then_dissolved', 'continued_growth')))
    local = {k: v[ids].copy() for k, v in rows.items()}
    local['label'] = np.isin(local['fate'], ('dissolved', 'grew_then_dissolved')).astype(int)
    counts = Counter(zip(local['source'], local['event']))
    local['weight'] = np.asarray([1/counts[key] for key in zip(local['source'], local['event'])])
    return local, features[ids]


def packet(features, name):
    if name == 'snapshot':
        return features[:, -1].copy()
    return np.concatenate((features.reshape(len(features), -1), features.mean(1), features.std(1),
                           features[:, -1]-features[:, 0]), axis=1)


def support(rows, assignment, task):
    output = []
    folds = np.asarray([assignment[g] for g in rows['ancestor']])
    for pool in POOLS:
        for fold in range(5):
            for label in (0, 1):
                mask = (folds == fold) & (rows['label'] == label)
                if pool != 'combined': mask &= rows['pool'] == pool
                output.append(dict(task=task, pool=pool, fold=fold, label=label, rows=int(mask.sum()),
                    events=len(set(zip(rows['source'][mask], rows['event'][mask]))),
                    sources=len(set(rows['source'][mask])), ancestors=len(set(rows['ancestor'][mask]))))
    return output


def seal(c):
    plan = read(resolve_path(c['output']) / 'technical/data-plan.json')
    if plan['config'] != c or sha(Path(data.__file__)) != plan['implementation']:
        raise ValueError('Data producer/configuration changed')
    rows, manifest = data.seal(c, plan)
    previous = read(resolve_path(c['earlier_cv']) / 'technical/design.json')
    if (previous['identity'] != c['earlier_cv_identity'] or
            digest({k: v for k, v in previous.items() if k != 'identity'}) != previous['identity']):
        raise ValueError('Earlier outer-fold design changed')
    old = set(rows['ancestor'][rows['pool'] == 'earlier'])
    new = set(rows['ancestor'][rows['pool'] == 'shooting'])
    if old & new:
        raise ValueError('Shared earlier/shooting ancestors require an explicit joint fold assignment')
    outer = {g: previous['outer'][g] for g in sorted(old)}
    outer.update(balance(rows, rows['ancestor'], rows['pool'] == 'shooting', 5, c['seed']))
    # All patches sharing a parent must have the same ancestor, including other protocols.
    for parent in set(rows['parent']) - {''}:
        if len(set(rows['ancestor'][rows['parent'] == parent])) != 1:
            raise ValueError('A shooting parent maps to multiple ancestry groups')
    inner, coverage = {}, []
    # Separate inner assignments per task and training pool; outer evaluation is shared.
    for task in ('appearance', 'fate'):
        local, _ = cohort(rows, np.zeros((len(rows['id']), 0)), task)
        coverage += support(local, outer, task)
        groups = local['ancestor']
        balancing = dict(local)
        if task == 'fate':
            balancing['stratum'] = np.where(local['label'] == 1, 'failed_strong', 'established')
            balancing['label'] = np.ones(len(groups), int)
        for pool in POOLS if task == 'appearance' else ('combined',):
            for fold in range(5):
                held = np.asarray([outer[g] == fold for g in groups])
                train = ~held & ((local['pool'] == pool) if pool != 'combined' else True)
                mapping = balance(balancing, groups, train, 5, c['seed'])
                assignment = np.asarray([mapping[g] for g in groups[train]])
                if set(local['label'][held]) != {0, 1}:
                    raise ValueError(f'Insufficient outer class support: {task}/{pool}/{fold}')
                for k in range(5):
                    for mask in (assignment == k, assignment != k):
                        if set(local['label'][train][mask]) != {0, 1}:
                            raise ValueError(f'Insufficient inner class support: {task}/{pool}/{fold}/{k}')
                inner[f'{task}/{pool}/{fold}'] = mapping
    b = original_config(plan['earlier'], 'original')
    design = dict(config=c, dataset=manifest['identity'], outer=outer, inner=inner,
        support=coverage, recipe={k: b[k] for k in ('linear_C', 'catboost')},
        implementation=check_metric_docs(family=FAMILY)[FAMILY]['files'])
    design['identity'] = digest(design)
    path = resolve_path(c['output']) / 'technical/design.json'
    if path.exists() and read(path) != design: raise ValueError('CV design changed')
    write_json(path, design)
    write_metric_rows(coverage, resolve_path(c['output']) / 'analyses/coverage-v1',
                      family=FAMILY, name='fold-support')
    # Censored/ambiguous nuclei remain appearance examples but not binary-fate labels.
    fate_counts = []
    for pool in ('earlier', 'shooting'):
        for fate in sorted(set(rows['fate'][rows['label'] == 1])):
            mask = (rows['pool'] == pool) & (rows['label'] == 1) & (rows['fate'] == fate)
            fate_counts.append(dict(pool=pool, fate=fate, centers=int(mask.sum()),
                events=len(set(zip(rows['source'][mask], rows['event'][mask]))),
                ancestors=len(set(rows['ancestor'][mask]))))
    write_metric_rows(fate_counts, resolve_path(c['output']) / 'analyses/coverage-v1',
                      family=FAMILY, name='fate-labels')
    return design


def load(c):
    p = read(resolve_path(c['output']) / 'technical/design.json')
    if (p['config'] != c or digest({k: v for k, v in p.items() if k != 'identity'}) != p['identity']
            or p['implementation'] != check_metric_docs(family=FAMILY)[FAMILY]['files']):
        raise ValueError('Changed frozen scientific producer/design')
    return p


def fit_root(c, spec, fold):
    return (resolve_path(c['output']) / 'analyses/fits-v1' / spec['task'] / spec['arm'] /
            spec['observation']['name'] / spec['pool'] / f'outer-{fold}')


def worker(c, progress):
    design = load(c)
    all_rows, features, manifest = data.dataset(c)
    if manifest['identity'] != design['dataset']: raise ValueError('Changed dataset')
    for spec in tasks():
        rows, frames = cohort(all_rows, features, spec['task'])
        x = packet(frames, spec['observation']['name'])
        features_id = feature_identity(x)
        groups = rows['ancestor']
        for fold in range(5):
            held = np.asarray([design['outer'][g] == fold for g in groups])
            train = ~held & ((rows['pool'] == spec['pool']) if spec['pool'] != 'combined' else True)
            ids = np.flatnonzero(train | held)
            local = {k: v[ids].copy() for k, v in rows.items()}
            local['role'] = np.where(train[ids], 'train', 'outer_test')
            mapping = design['inner'][f'{spec["task"]}/{spec["pool"]}/{fold}']
            folds = np.asarray([mapping[g] for g in groups[train]])
            context = dict(encoder=None, predictor=dict(bank='442 fixed geometry/bond-order/CNA/TDA descriptors',
                spatial_support='nearest80 atoms; local appearance and control screen radius8A',
                computational_halo=None, observation=spec['observation']['name'],
                offsets_ps=c['offsets_ps'] if spec['observation']['name'] == 'history4' else [-1.5],
                conditions=[], velocity=False, species=False, relaxation=False,
                temporal_features='chronological values, mean, std, last-first' if spec['observation']['name'] == 'history4' else 'current descriptors'),
                training_only_teachers=None, future_geometry_as_input=False,
                target='any local crystal appearance versus matched persistent liquid' if spec['task'] == 'appearance'
                       else 'verified dissolution versus documented continued growth; resolved nuclei only',
                design=design['identity'], input_features=features_id, outer_fold=fold,
                training_pool=spec['pool'], selector='inner ancestry-held-out binary NLL',
                calibration='none', retrospective_site_selection=True)
            progress.update(task=spec['task'], model=spec['model'], pool=spec['pool'],
                            observation=spec['observation']['name'], outer_fold=fold)
            destination = fit_root(c, spec, fold)
            fit(c, design['recipe'], local, x[ids], spec, destination, folds, context, family=FAMILY)
            write_json(destination / 'technical/outer-split.json', dict(design=design['identity'],
                global_indices=ids.tolist(), original_role=rows['role'][ids].tolist(),
                train_prior=float(np.average(rows['label'][train], weights=rows['weight'][train])),
                outer_fold=fold, task=spec['task'], training_pool=spec['pool']))
    from .shooting_appearance_analysis import collect
    collect(c, design, progress)


def submit(path):
    c = read(path)
    check_metric_docs(family=FAMILY)
    p = data.prepare(c)
    technical = resolve_path(c['output']) / 'technical'
    if (technical / 'launch.json').exists(): raise ValueError('Already submitted')
    bundle = ExecutionBundle.freeze(REPO, technical / 'code', c,
        directories=('src', 'configs', 'docs/metrics'))
    machine = yaml.safe_load(Path(os.environ.get('PCM_MACHINE_CONFIG', REPO / 'machine.local.yaml')).read_text())
    machine['catalog'] = str(bundle.root / 'configs/datasets.json')
    (technical / 'machine.yaml').write_text(yaml.safe_dump(machine))
    record = dict(data_plan=p['identity'], code=str(bundle.root), jobs={}, outer_fits=80,
                  seed=c['seed'], candidates=Counter(e['stratum'] for e in p['candidates']))
    queue = SlurmQueue(technical, bundle, 'src.research.birth_prediction.shooting_appearance',
        dict(PCM_PROJECT_ROOT=str(bundle.root), PCM_MACHINE_CONFIG=str(technical / 'machine.yaml'),
             OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'),
        technical / 'launch.json', record, 'AL-APPEAR')
    with queue.submission():
        preparation = queue.submit('prepare-sources', [f'--array=0-{c["workers"]-1}',
            '--cpus-per-task=1', '--mem=8G', '--time=06:00:00'])
        sealing = queue.submit('seal', ['--cpus-per-task=1', '--mem=12G', '--time=01:00:00'],
                               dependency='afterok:'+preparation)
        queue.submit('fit', ['--cpus-per-task=8', '--mem=24G', '--gpus=1', '--time=12:00:00',
            '--exclude='+p['earlier']['excluded_gpu_node']], dependency='afterok:'+sealing,
            partition=p['earlier']['gpu_partition'])
    return record


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('prepare', 'prepare-sources', 'seal', 'fit', 'collect', 'submit'))
    parser.add_argument('--config', required=True)
    args = parser.parse_args(); c = read(args.config)
    if args.stage == 'prepare':
        print(data.prepare(c)['identity']); return
    if args.stage == 'submit':
        print(submit(args.config)); return
    index = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
    with recorded_stage(resolve_path(c['output']) / f'technical/{args.stage}-{index}.json',
                        job=os.environ.get('SLURM_JOB_ID')) as progress:
        if args.stage == 'prepare-sources':
            p = read(resolve_path(c['output']) / 'technical/data-plan.json')
            if p['config'] != c or p['implementation'] != sha(Path(data.__file__)):
                raise ValueError('Data producer changed')
            for item in p['sources'][index::c['workers']]:
                progress.update(source=item['id'])
                print(data.source(c, p, item), flush=True)
        elif args.stage == 'seal': seal(c)
        elif args.stage == 'fit': worker(c, progress)
        else:
            from .shooting_appearance_analysis import collect
            collect(c, load(c), progress)


if __name__ == '__main__':
    main()
