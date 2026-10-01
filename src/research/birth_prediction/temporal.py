"""Fixed-endpoint temporal controls and same-site onset diagnostics on retained data."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs
from src.project_runtime.paths import resolve_path
from .data import load, read, ROLES
from .extension import base
from .features import load_bank
from .fit import fit
from .temporal_inputs import packet, context

FAMILY = 'birth_prediction_temporal'


def design(c):
    record = read(resolve_path(c['fold_design']))
    if record['identity'] != c['fold_identity']:
        raise ValueError('The retained readout CV assignment changed')
    if digest(record['binding']) != record['identity']:
        raise ValueError('Corrupted source-fold binding')
    return record


def partition(c, rows, lane):
    if lane == 0:
        return 'fixed-test', {r: np.flatnonzero(rows['role'] == r) for r in ROLES}, 'test'
    holdout = [int(s) for s, f in design(c)['binding']['source_folds'].items() if f == lane - 1]
    train = rows['role'] == 'train'
    evaluate = train & np.isin(rows['source'], holdout)
    split = dict(train=np.flatnonzero(train & ~evaluate),
        selection=np.flatnonzero(rows['role'] == 'selection'),
        calibration=np.flatnonzero(rows['role'] == 'calibration'),
        cv_evaluation=np.flatnonzero(evaluate))
    return f'cv/fold-{lane - 1}', split, 'cv_evaluation'


def fit_root(c, scope, arm, observation):
    return resolve_path(c['output']) / 'analyses/fits-v1' / scope / arm / observation


def prepare(c):
    check_metric_docs(family=FAMILY)
    b = base(c)
    _, rows, manifest = load(b)
    folds = design(c)
    if b['seed'] != c['seed'] or b['cadence_ps'] != .75 or rows['indices'].shape[1] != 8:
        raise ValueError('Temporal protocol changed seed, cadence or observation support')
    names = [o['name'] for o in c['observations']]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate observation name')
    for left, right in c['contrasts']:
        if left not in names or right not in names:
            raise ValueError(f'Undeclared contrast {left}/{right}')
    bindings = []
    for bank_name in ['descriptors', 'mace_rich', 'mace_vicreg']:
        bank, columns = load_bank(b, bank_name)
        for observation in c['observations']:
            x = packet(bank, rows, np.arange(len(columns)), observation, c['seed'])
            bindings.append(dict(bank=bank_name, observation_name=observation['name'],
                rows=len(x), columns=x.shape[1], **context(observation, b['cadence_ps'])))
            del x
        del bank
    support = []
    for lane in range(c['folds'] + 1):
        scope, split, evaluation = partition(c, rows, lane)
        source_sets = [set(rows['source'][ids]) for ids in split.values()]
        if any(a & z for i, a in enumerate(source_sets) for z in source_sets[i + 1:]):
            raise ValueError(f'Source crossing in {scope}')
        for role, ids in split.items():
            if set(rows['label'][ids]) != {0, 1}:
                raise ValueError(f'Missing outcome in {scope}/{role}')
        ids = split[evaluation]
        support.append(dict(scope=scope, rows=len(ids), sources=len(np.unique(rows['source'][ids])),
            births=len(set(zip(rows['source'][ids][rows['label'][ids] == 1].tolist(),
                               rows['event'][ids][rows['label'][ids] == 1].tolist())))))
    record = dict(config=c, dataset_identity=manifest['identity'], fold_identity=folds['identity'],
        rows=len(rows['id']), input_bindings=bindings, support=support,
        implementation={str(p.name): sha(p) for p in [Path(__file__), Path(__file__).with_name('temporal_inputs.py'),
            Path(__file__).with_name('temporal_analysis.py'), Path(__file__).with_name('fit.py')]})
    record['identity'] = digest(record)
    target = resolve_path(c['output']) / 'technical/prepared.json'
    if target.exists() and read(target) != record:
        raise ValueError('Prepared temporal study changed; use a new output revision')
    write_json(target, record)
    return record


def lane(c, boosted, index, progress):
    b = base(c)
    _, rows, _ = load(b)
    scope, split, _ = partition(c, rows, index)
    for name in (['prior'] if not boosted else []) + c['arms']:
        arm = next(a for a in b['arms'] if a['name'] == name)
        if (arm['model'] == 'catboost') != boosted:
            continue
        observations = [None] if name == 'prior' else c['observations']
        for observation in observations:
            label = observation['name'] if observation is not None else 'prior'
            progress.update(scope=scope, arm=name, observation=label)
            fit(b, name, 0, split=split, destination=fit_root(c, scope, name, label),
                family=FAMILY, observation=observation,
                evaluation=dict(protocol=c['protocol'], scope=scope, study_config=c,
                    original_source_roles=True, fold_identity=c['fold_identity'],
                    encoder_pretraining_exposed=(index > 0 and arm['bank'].startswith('mace_')),
                    interpretation='Retrospective selected sites; event-enriched probability population'))


def submit(path):
    c = read(path)
    prepared = prepare(c)
    tech = resolve_path(c['output']) / 'technical'
    if (tech / 'launch.json').exists():
        raise ValueError('Already submitted; resume using the frozen bundle')
    repo = Path(__file__).resolve().parents[3]
    bundle = ExecutionBundle.freeze(repo, tech / 'code', c, directories=('src', 'docs/metrics', 'configs'))
    record = dict(protocol=c['protocol'], prepared_identity=prepared['identity'], jobs={}, code=str(bundle.root),
        fits=(len(c['arms']) * len(c['observations']) + 1) * (c['folds'] + 1),
        tracking='local descriptor controls and frozen readouts; no encoder training')
    queue = SlurmQueue(tech, bundle, 'src.research.birth_prediction.temporal',
        dict(PCM_PROJECT_ROOT=str(repo), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
             NUMBA_NUM_THREADS='1', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'), tech / 'launch.json', record, 'BIRTH-TIME')
    with queue.submission():
        cpu = queue.submit('linear', [f'--array=0-{c["cpu_workers"]-1}',
            '--cpus-per-task=8', '--mem=16G', '--time=08:00:00'])
        gpu = queue.submit('boost', [f'--array=0-{c["gpu_workers"]-1}',
            '--gpus=1', '--cpus-per-task=8', '--mem=16G', '--time=08:00:00'], partition=c['gpu_partition'])
        queue.submit('collect', ['--cpus-per-task=4', '--mem=16G', '--time=02:00:00'],
            dependency='afterok:' + ':'.join((cpu, gpu)))
    return record


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('stage', choices=['prepare', 'submit', 'linear', 'boost', 'collect', 'site-descriptors'])
    p.add_argument('--config', required=True)
    p.add_argument('--index', type=int)
    args = p.parse_args()
    c = read(args.config)
    if args.stage == 'submit':
        print(json.dumps(submit(args.config), indent=2))
        return
    index = args.index if args.index is not None else int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
    with recorded_stage(resolve_path(c['output']) / f'technical/{args.stage}-{index}.json',
                        job=os.environ.get('SLURM_JOB_ID')) as progress:
        if args.stage == 'prepare':
            progress.update(prepared_identity=prepare(c)['identity'])
        elif args.stage in ['linear', 'boost']:
            workers = c['gpu_workers'] if args.stage == 'boost' else c['cpu_workers']
            for fit_index in range(index, c['folds'] + 1, workers):
                lane(c, args.stage == 'boost', fit_index, progress)
        else:
            from .temporal_analysis import collect, site_descriptors
            site_descriptors(c, progress)
            if args.stage == 'collect':
                collect(c, progress)


if __name__ == '__main__':
    main()
