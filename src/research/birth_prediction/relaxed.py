"""Detached full-cell quench, feature export and matched birth-readout queue."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from src.data.fixed_cohort.protocol import write_json
from src.data.relaxed_targets.worker import lock
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs
from src.project_runtime.paths import resolve_path
from .data import read
from . import features, temporal
from .extension import base as original_base
from .relaxed_data import freeze, study, cell, seal, INPUT


def validate_parent(c):
    p = freeze(c)
    expected = dict(original_base(c), cache=c['cache'], output=c['output'],
                    prepare_tasks=c['descriptor_workers'], input_domain=INPUT)
    path = resolve_path(c['output']) / 'technical/derived-parent/technical/code/config.json'
    if read(path) != expected:
        raise ValueError('Derived parent recipe changed from its original frozen cohort/encoder contract')
    return p


def worker(c, index, progress):
    p = validate_parent(c)
    errors = []
    tasks = p['tasks'][index::c['relaxation_workers']]
    for task in tasks:
        progress.update(task=task['id'])
        with lock(resolve_path(c['cache']) / 'cells' / task['id'] / 'worker.lock') as acquired:
            if not acquired:
                raise RuntimeError(f'Concurrent relaxed input producer: {task["id"]}')
            try:
                result = cell(p, task)
                print(json.dumps(dict(task=task['id'], state='complete', seconds=result['seconds'])), flush=True)
            except Exception:
                record = dict(task=task['id'], traceback=traceback.format_exc())
                write_json(resolve_path(c['output']) / 'technical/failures' / f'{task["id"]}.json', record)
                errors.append(task['id'])
                print(json.dumps(record), flush=True)
    progress.update(tasks=len(tasks), errors=errors)
    if errors:
        raise RuntimeError(f'Quench failures archived; no cohort exclusions or fitting: {errors}')


def group(c, path, index):
    cpus = sorted(os.sched_getaffinity(0))
    n = c['workers_per_group']
    if len(cpus) < n * c['ranks']:
        raise RuntimeError(f'Need {n*c["ranks"]} CPUs; allocated affinity is {cpus}')
    root = resolve_path(c['output']) / 'technical/workers'
    root.mkdir(parents=True, exist_ok=True)
    children = []
    for slot in range(n):
        worker_index = index * n + slot
        if worker_index >= c['relaxation_workers']:
            break
        mask = cpus[slot*c['ranks']:(slot+1)*c['ranks']]
        command = ['taskset', '-c', ','.join(map(str, mask)), sys.executable, '-u', '-m',
                   'src.research.birth_prediction.relaxed', 'worker', '--config', str(path), '--index', str(worker_index)]
        with (root / f'{worker_index}.log').open('a') as log:
            children.append((worker_index, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)))
    codes = {i: process.wait() for i, process in children}
    if any(codes.values()):
        raise RuntimeError(f'Relaxation workers failed: {codes}; see preserved per-cell failures')


def submit(path):
    c = read(path)
    p = validate_parent(c)
    for family in ['birth_prediction_temporal', 'birth_prediction_relaxed']:
        check_metric_docs(family=family)
    tech = resolve_path(c['output']) / 'technical'
    if (tech / 'launch.json').exists():
        raise ValueError('Already submitted; resume the frozen workers with their receipts')
    # One real converged cell validates the source-coordinate and atom-ID path.
    from .relaxed_data import checked
    checked(p, p['tasks'][0])
    preflight = read(tech / 'preflight/complete.json')
    if preflight != dict(passed=True, dataset_identity=p['identity'], cell=p['tasks'][0]['id']):
        raise ValueError('Require the actual relaxed descriptor and both frozen-encoder preflight')
    repo = Path(__file__).resolve().parents[3]
    bundle = ExecutionBundle.freeze(repo, tech / 'code', c, directories=('src', 'docs/metrics', 'configs'))
    record = dict(protocol=c['protocol'], dataset_identity=p['identity'], jobs={}, code=str(bundle.root),
        cells=len(p['tasks']), patches=p['patch_count'], fits=(len(c['arms'])*len(c['observations'])+1)*(c['folds']+1),
        tracking='local descriptor controls and frozen probes; no encoder fitting or new W&B runs')
    queue = SlurmQueue(tech, bundle, 'src.research.birth_prediction.relaxed',
        dict(PCM_PROJECT_ROOT=str(repo), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
             NUMBA_NUM_THREADS='1', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'),
        tech / 'launch.json', record, 'BIRTH-RELAX')
    groups = (c['relaxation_workers'] + c['workers_per_group'] - 1) // c['workers_per_group']
    with queue.submission():
        quench = queue.submit('group', [f'--array=0-{groups-1}%{groups}',
            f'--cpus-per-task={c["workers_per_group"]*c["ranks"]}', '--mem=24G', '--time=16:00:00'])
        sealed = queue.submit('seal', ['--cpus-per-task=2', '--mem=8G', '--time=01:00:00'], 'afterok:'+quench)
        desc = queue.submit('descriptors', [f'--array=0-{c["descriptor_workers"]-1}',
            '--cpus-per-task=2', '--mem=12G', '--time=04:00:00'], 'afterok:'+sealed)
        ds = queue.submit('seal-descriptors', ['--cpus-per-task=2', '--mem=8G', '--time=01:00:00'], 'afterok:'+desc)
        enc = queue.submit('encode', ['--array=0-1%2', '--gpus=1', '--cpus-per-task=2', '--mem=24G',
            '--time=02:00:00'], 'afterok:'+sealed, partition=c['gpu_partition'])
        ready = queue.submit('ready', ['--cpus-per-task=2', '--mem=16G', '--time=00:30:00'], 'afterok:'+ds+':'+enc)
        cpu = queue.submit('linear', [f'--array=0-{c["cpu_workers"]-1}',
            '--cpus-per-task=8', '--mem=16G', '--time=04:00:00'], 'afterok:'+ready)
        gpu = queue.submit('boost', ['--gpus=1', '--cpus-per-task=8', '--mem=16G', '--time=02:00:00'],
            'afterok:'+ready, partition=c['gpu_partition'])
        queue.submit('collect', ['--cpus-per-task=4', '--mem=16G', '--time=02:00:00'], 'afterok:'+cpu+':'+gpu)
    return record


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['freeze', 'submit', 'worker', 'group', 'seal', 'descriptors',
        'seal-descriptors', 'encode', 'ready', 'linear', 'boost', 'collect'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    c = read(args.config)
    validate_parent(c)
    if args.stage == 'submit':
        print(json.dumps(submit(args.config), indent=2))
        return
    index = args.index if args.index is not None else int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
    with recorded_stage(resolve_path(c['output']) / f'technical/{args.stage}-{index}.json',
                        job=os.environ.get('SLURM_JOB_ID')) as progress:
        if args.stage == 'freeze':
            progress.update(identity=freeze(c)['identity'])
        elif args.stage == 'worker':
            worker(c, index, progress)
        elif args.stage == 'group':
            group(c, args.config, index)
        elif args.stage == 'seal':
            progress.update(identity=seal(c)['identity'])
        elif args.stage in ['descriptors', 'seal-descriptors', 'encode']:
            b = temporal.base(study(c))
            if args.stage == 'descriptors':
                features.descriptors(b, index)
            elif args.stage == 'seal-descriptors':
                features.seal_descriptors(b)
            else:
                features.encode(b, index)
        elif args.stage == 'ready':
            progress.update(identity=temporal.prepare(study(c))['identity'])
        elif args.stage in ['linear', 'boost']:
            workers = c['cpu_workers'] if args.stage == 'linear' else c['gpu_workers']
            for lane in range(index, c['folds'] + 1, workers):
                temporal.lane(study(c), args.stage == 'boost', lane, progress)
        else:
            from .temporal_analysis import collect, site_descriptors
            from .relaxed_report import compare
            sc = study(c)
            site_descriptors(sc, progress)
            collect(sc, progress)
            compare(c, sc, progress)


if __name__ == '__main__':
    main()
