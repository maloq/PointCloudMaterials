"""Freeze this baseline once and submit detached controls, fits and collection."""
import argparse
import json
import os
from pathlib import Path

from src.data.fixed_cohort.protocol import write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs
from src.research.shooting_laws.common import read
from .data import load, output, prepare


def submit(c):
    _, manifest = load(c)
    check_metric_docs(family='predictive_baseline')
    root = output(c)
    tech = root/'technical'
    gate = read(tech/'preflight.json')
    if gate['state'] != 'complete' or gate['dataset'] != manifest['identity']:
        raise ValueError('Run the local full-batch numerical preflight first')
    if (tech/'launch.json').exists():
        raise FileExistsError('Already submitted; continue through recorded frozen commands')
    repo = Path(__file__).resolve().parents[3]
    bundle = ExecutionBundle.freeze(repo, tech/'code', c,
        directories=('src', 'docs/metrics', 'configs/predictive_baseline'),
        files=((repo/c['shooting_config'], c['shooting_config']),))
    receipt = dict(jobs={}, dataset=manifest['identity'], code=str(bundle.root),
        protocol=c['protocol'], fit_seeds=c['fit_seeds'], allocation_hours=12,
        preflight=gate, scientific_training_runs=3)
    q = SlurmQueue(tech, bundle, 'src.research.predictive_baseline.queue',
        dict(PCM_PROJECT_ROOT=str(repo), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
             TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1', WANDB_MODE='online'), tech/'launch.json', receipt, 'PRED274')
    with q.submission():
        control = q.submit('controls', ['--cpus-per-task=4', '--mem=16G', '--time=01:00:00'])
        fits = q.submit('fit', ['--gpus=1', '--cpus-per-task=4', '--mem=32G', '--time=12:00:00',
            '--signal=B:USR1@300', f'--array=0-{len(c["fit_seeds"])-1}%{c["fit_concurrency"]}'], partition=c['gpu_partition'])
        q.submit('collect', ['--cpus-per-task=4', '--mem=16G', '--time=02:00:00'], f'afterok:{control}:{fits}')
    local = repo/'output/predictive_baseline'/root.name
    local.parent.mkdir(parents=True, exist_ok=True)
    if not local.exists():
        local.symlink_to(root, target_is_directory=True)
    (root/'README.md').write_text(
        '# Local predictive future-statistic baseline\n\n'
        'Three native geometry-only MACE128 fits learn the fixed 274-coordinate 3/6/12 ps shooting target. '
        'Historical Al480 sources and all 7,661 rows are preserved.\n\n'
        'See `technical/launch.json` for detached jobs, `technical/prediction-context.json` for actual inputs, '
        'and `technical/target-manifest.json` for the sealed target population. '
        'Per-seed progress, resumable checkpoints and W&B receipts are in `analyses/joint-seed-*/technical/`.\n\n'
        'The collector writes `analyses/comparison-v1/tables/comparison.csv`, physical moments, frozen readouts, '
        'source-paired uncertainty and scientific plots. Results must be read with their frozen `tables/METRICS.md`.\n\n'
        'This is a local, pooled-condition historical assay. It does not establish full-state predictive sufficiency '
        'or represent 7,661 independent parent configurations.\n')
    write_json(root/'technical/protocol.json', c)
    return receipt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('prepare', 'preflight', 'submit', 'controls', 'fit', 'collect'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    c = read(args.config)
    if args.stage == 'submit':
        print(json.dumps(submit(c)), flush=True)
        return
    from . import train, evaluate
    index = args.index if args.index is not None else int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
    stages = dict(prepare=lambda: prepare(c), preflight=lambda: train.preflight(c),
                  controls=lambda: evaluate.controls(c), fit=lambda: train.fit(c, index),
                  collect=lambda: evaluate.collect(c))
    suffix = f'-{index}' if args.stage == 'fit' else ''
    with recorded_stage(output(c)/'technical'/f'stage-{args.stage}{suffix}.json', job=os.environ.get('SLURM_JOB_ID')):
        stages[args.stage]()


if __name__ == '__main__':
    main()
