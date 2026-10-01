"""Frozen CPU preparation, GPU inference/boosting and local linear diagnostics."""
import argparse
import json
import os
from pathlib import Path
import sys

from src.data.fixed_cohort.protocol import sha, write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs
from src.project_runtime.paths import resolve_path
from .data import read, plan


def resolve(c):
    for model in c['encoders']:
        for name in ('checkpoint', 'producer'):
            model[name] = str(resolve_path(model[name]))
        if sha(Path(model['checkpoint'])) != model['checkpoint_sha256']:
            raise ValueError(f'Changed {model["name"]} checkpoint')
        for name, h in model['inference_dependencies'].items():
            if sha(Path(model['producer']) / name) != h:
                raise ValueError(f'Changed frozen model source: {model["name"]}/{name}')
    return c


def bind(c):
    p = plan(c)
    root = resolve_path(c['cache'])
    root.mkdir(parents=True, exist_ok=True)
    dest = root / 'plan.json'
    if dest.exists() and read(dest) != p:
        raise ValueError('Study binding changed; use a new release/output')
    write_json(dest, p)
    return p


def submit(path):
    c = resolve(read(path))
    check_metric_docs(family='birth_prediction')
    p = bind(c)
    root = resolve_path(c['output'])
    tech = root / 'technical'
    tech.mkdir(parents=True, exist_ok=True)
    if (tech / 'launch.json').exists():
        raise ValueError('Study already submitted; use retained stage receipts to resume')
    receipt = read(tech / 'preflight.json')
    if receipt['plan_identity'] != p['identity'] or not receipt['passed']:
        raise ValueError('Require the actual training-source / frozen-encoder preflight')
    repo = Path(__file__).resolve().parents[3]
    bundle = ExecutionBundle.freeze(repo, tech / 'code', c, directories=('src', 'docs/metrics', 'configs'))
    record = dict(protocol=c['protocol'], dataset_identity=p['identity'], jobs={}, code=str(bundle.root))
    queue = SlurmQueue(tech, bundle, 'src.research.birth_prediction.queue',
        dict(PCM_PROJECT_ROOT=str(repo), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
             NUMBA_NUM_THREADS='1', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
             PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'),
        tech / 'launch.json', record, 'BIRTH')
    n = c['prepare_tasks']
    with queue.submission():
        prep = queue.submit('prepare', [f'--array=0-{n-1}%{n}', '--cpus-per-task=2', '--mem=16G', '--time=08:00:00'])
        sealed = queue.submit('seal', ['--cpus-per-task=2', '--mem=16G', '--time=01:00:00'], 'afterok:' + prep)
        desc = queue.submit('descriptors', [f'--array=0-{n-1}%{n}', '--cpus-per-task=2', '--mem=16G', '--time=08:00:00'], 'afterok:' + sealed)
        desc_seal = queue.submit('seal-descriptors', ['--cpus-per-task=2', '--mem=16G', '--time=01:00:00'], 'afterok:' + desc)
        enc = queue.submit('encode', [f'--array=0-{len(c["encoders"])-1}%1', '--gpus=1', '--cpus-per-task=2',
                                    '--mem=24G', '--time=04:00:00'], 'afterok:' + sealed, partition=c['gpu_partition'])
        cpu = queue.submit('linear', ['--cpus-per-task=8', '--mem=24G', '--time=08:00:00'],
                           'afterok:' + desc_seal + ':' + enc)
        gpu = queue.submit('boost', ['--gpus=1', '--cpus-per-task=8', '--mem=24G', '--time=08:00:00'],
                           'afterok:' + desc_seal + ':' + enc, partition=c['gpu_partition'])
        queue.submit('collect', ['--cpus-per-task=4', '--mem=16G', '--time=02:00:00'], 'afterok:' + cpu + ':' + gpu)
    return record


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['bind', 'submit', 'prepare', 'seal', 'descriptors', 'seal-descriptors',
                                         'encode', 'linear', 'boost', 'fit', 'collect'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--index', type=int)
    parser.add_argument('--source', type=int)
    parser.add_argument('--arm')
    parser.add_argument('--remove', type=int)
    args = parser.parse_args()
    c = resolve(read(args.config))
    if args.stage == 'bind':
        print(json.dumps(dict(identity=bind(c)['identity'])))
        return
    if args.stage == 'submit':
        print(json.dumps(submit(args.config), indent=2))
        return
    index = args.index if args.index is not None else int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
    state = resolve_path(c['output']) / 'technical' / f'{args.stage}-{args.source or args.arm or index}.json'
    with recorded_stage(state, job=os.environ.get('SLURM_JOB_ID')) as progress:
        if args.stage == 'prepare':
            from .data import prepare
            prepare(c, index, source_id=args.source)
        elif args.stage == 'seal':
            from .data import seal
            progress.update(summary=seal(c))
        elif args.stage == 'descriptors':
            from .features import descriptors
            descriptors(c, index)
        elif args.stage == 'seal-descriptors':
            from .features import seal_descriptors
            seal_descriptors(c)
        elif args.stage == 'encode':
            from .features import encode
            encode(c, index)
        elif args.stage == 'fit':
            from .fit import fit
            fit(c, args.arm, args.remove)
        elif args.stage in ('linear', 'boost'):
            from .fit import fit
            for arm in c['arms']:
                if (arm['model'] == 'catboost') == (args.stage == 'boost'):
                    for remove in c['remove_frames']:
                        progress.update(arm=arm['name'], removed_frames=remove)
                        fit(c, arm['name'], remove)
        elif args.stage == 'collect':
            from .fit import collect
            collect(c)


if __name__ == '__main__':
    main()
