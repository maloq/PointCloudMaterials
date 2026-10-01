"""Detached factorial study with immutable source and dependency receipts."""
import argparse
import json
import os
from pathlib import Path

from src.data.fixed_cohort.protocol import write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs
from .common import FAMILY, prepare, encode_mm, jobs, output, read


def submit(c):
    binding = prepare(c)
    check_metric_docs(family=FAMILY)
    root = output(c)
    tech = root/'technical'
    gate = read(tech/'preflight.json')
    encoded = read(tech/'mm-encoding-complete.json')
    if gate['binding'] != binding['identity'] or gate['state'] != 'complete' or encoded['state'] != 'complete':
        raise ValueError('Matching numerical and MM-TDA inference gates required')
    if (tech/'launch.json').exists():
        raise FileExistsError('Follow-up already submitted; use frozen commands to resume')
    repo = Path(__file__).resolve().parents[3]
    bundle = ExecutionBundle.freeze(repo,tech/'code',c,
        directories=('src','docs/metrics','configs/predictive_baseline'),
        files=((repo/c['shooting_config'],c['shooting_config']),
               (repo/c['encoders'][0]['pretraining_recipe'],c['encoders'][0]['pretraining_recipe'])))
    receipt = dict(jobs={},binding=binding['identity'],code=str(bundle.root),
        joint_arms=jobs(c,'joint'),probe_arms=jobs(c,'probe'),
        scientific_online_runs=len(jobs(c,'joint')),local_diagnostic_heads=len(jobs(c,'probe')))
    q = SlurmQueue(tech,bundle,'src.research.predictive_followup.queue',
        dict(PCM_PROJECT_ROOT=str(repo),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
             WANDB_MODE='online',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'),tech/'launch.json',receipt,'PRED-FOLLOW')
    with q.submission():
        controls = q.submit('controls',['--cpus-per-task=4','--mem=16G','--time=00:30:00'])
        probes = q.submit('probe',['--cpus-per-task=4','--mem=12G','--time=00:30:00',
            f'--array=0-{len(jobs(c,"probe"))-1}%{c["probe_concurrency"]}'])
        joint = q.submit('joint',['--gpus=1','--cpus-per-task=4','--mem=24G',
            '--time='+c['training']['time_limit'],'--signal=B:USR1@120',
            '--exclude='+','.join(c['exclude_nodes']),
            f'--array=0-{len(jobs(c,"joint"))-1}%{c["fit_concurrency"]}'],partition=c['gpu_partition'])
        q.submit('collect',['--cpus-per-task=4','--mem=16G','--time=01:00:00'],f'afterok:{controls}:{probes}:{joint}')
    local = repo/'output/predictive_baseline'/root.name
    local.parent.mkdir(parents=True,exist_ok=True)
    if not local.exists():
        local.symlink_to(root,target_is_directory=True)
    (root/'README.md').write_text(
        '# Predictive baseline: heads, targets and MM-TDA\n\n'
        '12 native MACE128 scientific fits and 36 local frozen-head diagnostics form a matched '
        '2x2 target/variance design with three seeds. Frozen encoders include VICReg128, Epi128 '
        'and the exact MM-TDA-BLOCK-DIRECT-FULL epoch20 z256 export.\n\n'
        'All inputs, source roles, shooting branches and target normalization are inherited from '
        'the sealed historical Al480 baseline. Every new fit selects on the same 18-moment Gaussian '
        'feature likelihood. Original results remain separate.\n\n'
        'Execution: `technical/launch.json`. Actual inputs: `technical/prediction-context.json`. '
        'Per-fit progress and checkpoints: `analyses/*/technical/`. Final tables, paired comparisons '
        'and plots: `analyses/comparison-v1/`. Metric exports include frozen definitions.\n\n'
        'Full-feature errors are unavailable for moment-only heads. Frozen linear readouts of each '
        'selected joint embedding separately assess retained full-target information. '
        'MM-TDA has greater capacity and additional pretraining data; archived ancestry limitations '
        'are retained in the binding. No new simulations.\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('prepare','encode','preflight','submit','joint','probe','controls','collect'))
    parser.add_argument('--config',required=True)
    parser.add_argument('--index',type=int)
    args = parser.parse_args()
    c = read(args.config)
    if args.stage == 'submit':
        print(json.dumps(submit(c)),flush=True)
        return
    index = args.index if args.index is not None else int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))
    if args.stage in ('joint','probe','preflight'):
        from .train import fit,preflight
        action = (lambda: preflight(c)) if args.stage=='preflight' else lambda: fit(c,jobs(c,args.stage)[index])
    elif args.stage in ('controls','collect'):
        from . import evaluate
        action = lambda: getattr(evaluate,args.stage)(c)
    else:
        action = (lambda: prepare(c)) if args.stage=='prepare' else lambda: encode_mm(c)
    suffix = f'-{index}' if args.stage in ('joint','probe') else ''
    with recorded_stage(output(c)/'technical'/f'stage-{args.stage}{suffix}.json',job=os.environ.get('SLURM_JOB_ID')):
        action()


if __name__ == '__main__':
    main()
