"""Matched descriptor-family refits and reliance on original/relaxed birth inputs."""
import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs
from src.project_runtime.paths import resolve_path
from .analysis import predictions
from .data import load, read
from .features import load_bank
from .fit import fit
from .temporal import design, partition

FAMILY = 'birth_feature_families'
FAMILIES = ('geometry', 'bond_order', 'cna', 'tda')
MODELS = {'linear': 'rich_linear', 'catboost': 'rich_gbdt'}
OBSERVATION = dict(name='current', kind='snapshot', frames=[7])


def variants():
    return [dict(name='full', families=list(FAMILIES))] + [
        dict(name=f'{mode}_{family}', families=[family] if mode == 'only' else
             [f for f in FAMILIES if f != family])
        for mode in ('only', 'without') for family in FAMILIES]


def domain_config(c, domain):
    path = resolve_path(domain['run']) / 'analyses/fits-v1/fixed-test/rich_linear/current/technical/binding.json'
    if sha(path) != domain['binding_sha256']:
        raise ValueError(f'Changed retained input/fitting contract: {path}')
    binding = read(path)
    if binding['dataset_identity'] != domain['dataset_identity']:
        raise ValueError(f'Changed domain identity: {domain["name"]}')
    b = copy.deepcopy(binding['config'])
    if b['seed'] != c['seed'] or b['cadence_ps'] != .75 or b['history_frames'] != 8:
        raise ValueError('Family comparison changed seed or observed timeline')
    return b


def fit_path(c, domain, model, variant, index):
    scope = 'fixed-test' if index == 0 else f'cv/fold-{index-1}'
    if variant == 'full':
        return resolve_path(domain['run']) / 'analyses/fits-v1' / scope / MODELS[model] / 'current'
    return resolve_path(c['output']) / 'analyses/fits-v1' / domain['name'] / scope / model / variant


def prepare(c):
    check_metric_docs(family=FAMILY)
    design(c)
    if [d['name'] for d in c['domains']] != ['original', 'relaxed'] or c['folds'] != 5:
        raise ValueError('This matched protocol requires original/relaxed domains and the five retained folds')
    records, reference_rows, reference_columns = [], None, None
    for domain in c['domains']:
        b = domain_config(c, domain)
        _, rows, manifest = load(b)
        bank, columns = load_bank(b, 'descriptors')
        if manifest['identity'] != domain['dataset_identity']:
            raise ValueError(f'Changed input release: {domain["name"]}')
        if bank.shape[1] != len(columns) or not np.isfinite(bank).all():
            raise ValueError('Invalid retained descriptor bank')
        if set(col.split('/')[0] for col in columns) != set(FAMILIES):
            raise ValueError('Unexpected descriptor family schema')
        if reference_rows is None:
            reference_rows, reference_columns = rows, columns
        elif (columns != reference_columns or rows.keys() != reference_rows.keys() or
              any(not np.array_equal(v, reference_rows[k]) for k, v in rows.items())):
            raise ValueError('Original and relaxed rows, targets, weights, atom indices or feature names differ')
        for index in range(c['folds'] + 1):
            scope, split, evaluation = partition(c, rows, index)
            for model, arm in MODELS.items():
                root = fit_path(c, domain, model, 'full', index)
                binding = read(root / 'technical/binding.json')
                expected_arm = next(a for a in b['arms'] if a['name'] == arm)
                expected_partition = digest({k: rows['id'][v].tolist() for k, v in split.items()})
                expected = dict(dataset_identity=manifest['identity'], partition_sha256=expected_partition,
                    observation=OBSERVATION, arm=expected_arm, fit_sha256=domain['reference_fit_sha256'],
                    temporal_input_sha256=sha(Path(__file__).with_name('temporal_inputs.py')))
                differences = {k: dict(expected=v, observed=binding[k]) for k, v in expected.items() if binding[k] != v}
                if differences:
                    raise ValueError(f'Full-feature reference has a different protocol: {root}: {differences}')
                reference_fitter = resolve_path(domain['run']) / 'technical/code/src/research/birth_prediction/fit.py'
                if sha(reference_fitter) != domain['reference_fit_sha256']:
                    raise ValueError(f'Changed historical fitting source: {reference_fitter}')
                if read(root / 'technical/complete.json')['identity'] != digest(binding):
                    raise ValueError(f'Changed full-feature fitting binding: {root}')
                for key in ('catboost', 'linear_C', 'seed', 'patience', 'fit_threads'):
                    if binding['config'][key] != b[key]:
                        raise ValueError(f'Full-feature reference changed {key}: {root}')
                _, receipt = predictions(root, rows['id'][split[evaluation]], evaluation)
                model_path = root / 'technical' / ('model.joblib' if model == 'linear' else 'model.cbm')
                records.append(dict(domain=domain['name'], model=model, scope=scope,
                                    model_sha256=sha(model_path), **receipt))
        cache = resolve_path(b['cache'])
        records.append(dict(domain=domain['name'], descriptor_sha256=sha(cache / 'descriptors.npy'),
            descriptor_manifest_sha256=sha(cache / 'descriptor-manifest.json'), dataset_identity=manifest['identity'],
            columns=columns, family_dimensions={f: sum(x.startswith(f+'/') for x in columns) for f in FAMILIES}))
    record = dict(config=c, references=records, row_count=len(reference_rows['id']),
        variants=variants(), new_fits=2*2*8*(c['folds']+1), reused_fits=2*2*(c['folds']+1))
    record['identity'] = digest(record)
    path = resolve_path(c['output']) / 'technical/prepared.json'
    if path.exists() and read(path) != record:
        raise ValueError('Prepared feature-family study changed; use a new output revision')
    write_json(path, record)
    return record


def task(c, domain, model, variant, index):
    b = domain_config(c, domain)
    _, rows, _ = load(b)
    scope, split, _ = partition(c, rows, index)
    arm = dict(name=f'{model}_{variant["name"]}', bank='descriptors', model=model,
               families=variant['families'])
    b.update(arms=[arm], encoders=[])
    fit(b, arm['name'], 0, split=split,
        destination=fit_path(c, domain, model, variant['name'], index), family=FAMILY,
        observation=OBSERVATION, evaluation=dict(protocol=c['protocol'], scope=scope,
            domain=domain['name'], variant=variant, study_config=c,
            fold_identity=c['fold_identity'], original_source_roles=True,
            interpretation='Retrospective matched birth-site classification; no natural-incidence claim'))


def lane(c, model, worker, progress):
    workers = c['cpu_workers'] if model == 'linear' else 1
    tasks = [(domain, index, variant) for domain in c['domains']
             for index in range(c['folds'] + 1) for variant in variants()[1:]]
    for number in range(worker, len(tasks), workers):
        domain, index, variant = tasks[number]
        progress.update(task=number, domain=domain['name'], model=model,
                        fold=index-1, variant=variant['name'])
        task(c, domain, model, variant, index)


def submit(path):
    c = read(path)
    prepared = prepare(c)
    tech = resolve_path(c['output']) / 'technical'
    if (tech / 'launch.json').exists():
        raise ValueError('Already submitted; resume using the frozen bundle')
    repo = Path(__file__).resolve().parents[3]
    bundle = ExecutionBundle.freeze(repo, tech / 'code', c, directories=('src', 'docs/metrics', 'configs'))
    record = dict(protocol=c['protocol'], prepared_identity=prepared['identity'], jobs={},
        code=str(bundle.root), new_fits=prepared['new_fits'], reused_fits=prepared['reused_fits'],
        tracking='local descriptor diagnostics; no encoder fitting or new W&B runs')
    queue = SlurmQueue(tech, bundle, 'src.research.birth_prediction.feature_families',
        dict(PCM_PROJECT_ROOT=str(repo), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
             MKL_NUM_THREADS='1', NUMBA_NUM_THREADS='1'), tech / 'launch.json', record, 'BIRTH-FAMILY')
    with queue.submission():
        linear = queue.submit('linear', [f'--array=0-{c["cpu_workers"]-1}',
            '--cpus-per-task=8', '--mem=12G', '--time=03:00:00'])
        boost = queue.submit('boost', ['--gpus=1', '--cpus-per-task=8', '--mem=12G',
            '--time=04:00:00'], partition=c['gpu_partition'])
        # CatBoost replay shares the GPU lane's modern CPU after its last fit;
        # it does not occupy a separate GPU allocation just for attribution.
        queue.submit('collect', ['--cpus-per-task=4', '--mem=12G', '--time=01:00:00'],
                     dependency='afterok:' + ':'.join((linear, boost)))
    return record


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('stage', choices=['prepare', 'submit', 'linear', 'boost', 'explain', 'collect'])
    p.add_argument('--config', required=True)
    p.add_argument('--index', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', '0')))
    args = p.parse_args()
    c = read(args.config)
    if args.stage == 'submit':
        print(json.dumps(submit(args.config), indent=2)); return
    with recorded_stage(resolve_path(c['output']) / f'technical/{args.stage}-{args.index}.json',
                        job=os.environ.get('SLURM_JOB_ID')) as progress:
        if args.stage == 'prepare':
            progress.update(identity=prepare(c)['identity'])
        elif args.stage in ('linear', 'boost'):
            lane(c, 'linear' if args.stage == 'linear' else 'catboost', args.index, progress)
            if args.stage == 'boost':
                from .feature_family_analysis import explain
                explain(c, progress)
        elif args.stage == 'explain':
            from .feature_family_analysis import explain
            explain(c, progress)
        else:
            from .feature_family_analysis import collect
            collect(c, progress)


if __name__ == '__main__':
    main()
