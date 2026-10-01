"""One frozen queued allocation for the three response-atlas feasibility stages."""
import argparse
import os
import signal
from pathlib import Path

from src.data.fixed_cohort.protocol import write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs
from src.project_runtime.paths import resolve_path
from .common import read, output, bind


def submit(c):
    binding = bind(c)
    check_metric_docs(family='response_atlas')
    root = output(c) / 'technical'
    if (root / 'launch.json').exists():
        raise ValueError('Already submitted: use the recorded frozen worker')
    physical_gate = read(root / 'atomistic-gate.json')
    if not read(root / 'numerical.json')['passed'] or not physical_gate['passed'] or physical_gate['binding'] != binding['identity']:
        raise ValueError('Both actual numerical preflights must pass')
    from src.research.shooting_laws.common import result
    shooting = read(resolve_path(c['shooting_config']))
    launch = read(result(shooting) / 'technical/launch.json')
    repo = Path(__file__).resolve().parents[3]
    bundle = ExecutionBundle.freeze(repo, root / 'code', c, directories=('src', 'docs/metrics'))
    receipt = dict(jobs={}, protocol_identity=binding['identity'], code=str(bundle.root),
        dependency=dict(shooting_seal=launch['jobs']['seal']),
        stages=['existing-shot reliability', 'analytic mechanisms and online toy fits', 'atomistic gate and pilot'],
        deferred=['five-arm active atomistic training', 'large-cell MEAM bridge'])
    q = SlurmQueue(root, bundle, 'src.research.response_atlas.queue',
        dict(PCM_PROJECT_ROOT=str(repo), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
             TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1', WANDB_MODE='online'), root / 'launch.json', receipt, 'RESP')
    with q.submission():
        cpu = q.submit('cpu', ['--cpus-per-task=4', '--mem=16G', '--time=03:00:00'],
                       'afterok:' + launch['jobs']['seal'])
        q.submit('gpu', ['--gpus=1', '--cpus-per-task=4', '--mem=48G', '--time=24:00:00', '--signal=B:TERM@180'],
                 'afterok:' + cpu, partition=c['gpu_partition'])
    return receipt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('numerical', 'gate', 'submit', 'cpu', 'gpu', 'reliability', 'mechanisms', 'atomistic'))
    parser.add_argument('--config', required=True)
    args = parser.parse_args(); c = read(args.config)
    from . import mechanisms, atomistic, reliability
    stages = dict(numerical=lambda: mechanisms.numerical(c), gate=lambda: atomistic.gate(c),
        reliability=lambda: reliability.run(c),
        mechanisms=lambda: (mechanisms.acquisition(c), mechanisms.basin(c), mechanisms.learn(c)),
        atomistic=lambda: atomistic.run(c))
    if args.stage == 'submit':
        print(submit(c)); return
    if args.stage in ('cpu', 'gpu'):
        def stopping(signum, frame):
            raise RuntimeError(f'Scheduler signal {signum}: preserve partial query bundles before stopping')
        signal.signal(signal.SIGTERM, stopping)
        bind(c)
        names = ('numerical', 'reliability', 'mechanisms') if args.stage == 'cpu' else ('gate', 'atomistic')
        for name in names:
            path = output(c) / 'technical' / f'stage-{name}.json'
            if path.exists() and read(path)['state'] == 'complete':
                continue
            with recorded_stage(path, job=os.environ.get('SLURM_JOB_ID')):
                stages[name]()
        write_json(output(c) / 'technical' / f'{args.stage}-complete.json', dict(state='complete', stages=list(names)))
        return
    with recorded_stage(output(c) / 'technical' / f'preflight-{args.stage}.json', job=os.environ.get('SLURM_JOB_ID')):
        stages[args.stage]()


if __name__ == '__main__':
    main()
