"""Extend a frozen birth cohort to one frame and cross-validate its readouts."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_rows
from src.project_runtime.paths import resolve_path
from .data import load, read, ROLES
from .features import history_packet, load_bank
from .fit import fit
from .queue import resolve

FAMILY = 'birth_prediction_extension'


def base(c):
    root = resolve_path(c['parent_run'])
    path = root / 'technical/code/config.json'
    if sha(path) != c['parent_config_sha256']:
        raise ValueError('Changed frozen parent recipe')
    b = resolve(read(path))
    _, _, manifest = load(b)
    if manifest['identity'] != c['dataset_identity']:
        raise ValueError('Extended readouts changed the existing birth cohort')
    return b


def prepare(c):
    check_metric_docs(family=FAMILY)
    b = base(c)
    _, rows, manifest = load(b)
    if c['remove_frames'] != list(range(b['history_frames'])):
        raise ValueError('This protocol requires every endpoint from eight frames to one')
    parent_plan = read(resolve_path(b['cache']) / 'plan.json')
    groups = defaultdict(list)
    for source in parent_plan['sources']:
        if source['role'] == 'train':
            groups[source['lineage']].append(source['id'])
    # Keep melt ancestry together, including sources with no eligible examples.
    rng = np.random.default_rng(c['fold_seed'])
    order = rng.permutation(list(groups)).tolist()
    def births(group):
        ids = np.isin(rows['source'], groups[group]) & (rows['label'] == 1)
        return len(set(zip(rows['source'][ids].tolist(), rows['event'][ids].tolist())))
    order.sort(key=births, reverse=True)  # random tie order fixed above
    assignments = {}
    event_mass = np.zeros(c['folds'], int)
    group_counts = np.zeros(c['folds'], int)
    for group in order:
        n = births(group)
        fold = min(range(c['folds']), key=lambda k: (event_mass[k], group_counts[k], k) if n else (group_counts[k], k))
        assignments[group] = fold
        event_mass[fold] += n
        group_counts[fold] += 1
    source_folds = {str(source): assignments[group] for group, sources in groups.items() for source in sources}
    summary = []
    for fold in range(c['folds']):
        sources = [int(s) for s, f in source_folds.items() if f == fold]
        ids = np.flatnonzero(np.isin(rows['source'], sources))
        if len(np.unique(rows['label'][ids])) != 2 or event_mass[fold] < 3:
            raise ValueError(f'Insufficient birth/liquid support in outer fold {fold}')
        summary.append(dict(fold=fold, registered_sources=len(sources), observed_sources=len(np.unique(rows['source'][ids])),
                            births=int(event_mass[fold]), rows=len(ids), positives=int(rows['label'][ids].sum())))
    binding = dict(config=c, dataset_identity=manifest['identity'], source_folds=source_folds,
                   implementation_sha256=sha(Path(__file__)))
    record = dict(identity=digest(binding), binding=binding, summary=summary,
        protocol='Outer CV is within original training sources; original selection and calibration sources stay fixed.',
        limitation='Frozen encoders saw these original training sources during pretraining; this is readout CV, not end-to-end encoder CV.')
    root = resolve_path(c['output'])
    dest = root / 'technical/folds.json'
    if dest.exists() and read(dest) != record:
        raise ValueError('Changed CV release; use a new output revision')
    write_json(dest, record)
    # Verify resident feature identities and the one-frame packet without re-exporting.
    banks = {}
    for name in sorted({a['bank'] for a in b['arms']} - {'none'}):
        bank, names = load_bank(b, name)
        packet = history_packet(bank, rows, 7, np.arange(len(names)))
        if len(packet) != len(rows['id']) or not np.isfinite(packet).all():
            raise ValueError(f'Invalid one-frame packet: {name}')
        banks[name] = dict(patches=len(bank), dimensions=len(names), one_frame_dimensions=packet.shape[1])
    write_json(root / 'technical/feature-readiness.json', dict(dataset_identity=manifest['identity'], banks=banks))
    write_metric_rows(summary, root / 'analyses/fold-design-v1', family=FAMILY, name='fold-support')
    return record


def task(c, arm, remove, fold=None):
    b = base(c)
    _, rows, _ = load(b)
    root = resolve_path(c['output'])
    if fold is None:
        if remove in b['remove_frames']:
            raise ValueError('Original fixed-test fits are referenced, never refitted')
        split = {role: np.flatnonzero(rows['role'] == role) for role in ROLES}
        scope = 'fixed-test-v1'
        context = dict(scope='fixed original test', original_source_roles=True)
    else:
        design = read(root / 'technical/folds.json')
        holdout = [int(s) for s, f in design['binding']['source_folds'].items() if f == fold]
        train = rows['role'] == 'train'
        evaluation = train & np.isin(rows['source'], holdout)
        split = dict(train=np.flatnonzero(train & ~evaluation),
                     selection=np.flatnonzero(rows['role'] == 'selection'),
                     calibration=np.flatnonzero(rows['role'] == 'calibration'),
                     cv_evaluation=np.flatnonzero(evaluation))
        scope = f'cv-v1/fold-{fold}'
        context = dict(scope='outer frozen-readout CV on original training sources', fold=fold,
                       fold_identity=design['identity'],
                       encoder_pretraining_exposed=arm in {a['name'] for a in b['arms'] if a['bank'].startswith('mace_')},
                       original_test_evaluated=False, fixed_selection_calibration_sources=True)
    fit(b, arm, remove, split=split, destination=root / 'analyses' / scope / arm / f'minus-{remove}',
        family=FAMILY, evaluation=context)


def lane(c, boosted, fold=None, progress=None):
    b = base(c)
    removed = c['remove_frames'] if fold is not None else [r for r in c['remove_frames'] if r not in b['remove_frames']]
    for arm in b['arms']:
        if (arm['model'] == 'catboost') == boosted:
            for remove in removed:
                if progress is not None:
                    progress.update(arm=arm['name'], removed_frames=remove, fold=fold)
                task(c, arm['name'], remove, fold)


def submit(path):
    c = read(path)
    design = prepare(c)
    root = resolve_path(c['output']) / 'technical'
    if (root / 'launch.json').exists():
        raise ValueError('Extension already submitted; resume against its frozen bundle')
    repo = Path(__file__).resolve().parents[3]
    bundle = ExecutionBundle.freeze(repo, root / 'code', c, directories=('src', 'docs/metrics', 'configs'))
    record = dict(protocol=c['protocol'], dataset_identity=c['dataset_identity'], folds_identity=design['identity'],
                  jobs={}, code=str(bundle.root), original_fits_reused=44, added_fixed_test_fits=44,
                  cv_fits=c['folds'] * len(c['remove_frames']) * len(base(c)['arms']))
    queue = SlurmQueue(root, bundle, 'src.research.birth_prediction.extension',
        dict(PCM_PROJECT_ROOT=str(repo), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
             NUMBA_NUM_THREADS='1', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'), root / 'launch.json', record, 'BIRTH-EXT')
    with queue.submission():
        queue.submit('plot-existing', ['--cpus-per-task=2', '--mem=8G', '--time=00:30:00'])
        cpu = queue.submit('linear', ['--cpus-per-task=8', '--mem=16G', '--time=04:00:00'])
        gpu = queue.submit('boost', ['--gpus=1', '--cpus-per-task=8', '--mem=16G', '--time=04:00:00'], partition=c['gpu_partition'])
        cv_cpu = queue.submit('cv-linear', [f'--array=0-{c["folds"]-1}%{c["cpu_fold_concurrency"]}',
                              '--cpus-per-task=8', '--mem=16G', '--time=04:00:00'])
        cv_gpu = queue.submit('cv-boost', [f'--array=0-{c["folds"]-1}%{c["gpu_fold_concurrency"]}',
                              '--gpus=1', '--cpus-per-task=8', '--mem=16G', '--time=04:00:00'], partition=c['gpu_partition'])
        queue.submit('collect', ['--cpus-per-task=4', '--mem=16G', '--time=01:00:00'],
                     'afterok:' + ':'.join((cpu, gpu, cv_cpu, cv_gpu)))
    return record


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['prepare', 'submit', 'fit', 'linear', 'boost', 'cv-linear', 'cv-boost', 'plot-existing', 'collect'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--arm')
    parser.add_argument('--remove', type=int)
    parser.add_argument('--fold', type=int)
    args = parser.parse_args()
    c = read(args.config)
    if args.stage == 'submit':
        print(json.dumps(submit(args.config), indent=2))
        return
    fold = args.fold if args.fold is not None else int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    state = resolve_path(c['output']) / 'technical' / f'{args.stage}-{fold}.json'
    with recorded_stage(state, job=os.environ.get('SLURM_JOB_ID')) as progress:
        if args.stage == 'prepare':
            progress.update(summary=prepare(c)['summary'])
        elif args.stage == 'fit':
            task(c, args.arm, args.remove, args.fold)
        elif args.stage in ('linear', 'boost', 'cv-linear', 'cv-boost'):
            lane(c, args.stage.endswith('boost'), fold if args.stage.startswith('cv-') else None, progress)
        else:
            from .analysis import collect
            collect(c, existing_only=args.stage == 'plot-existing')


if __name__ == '__main__':
    main()
