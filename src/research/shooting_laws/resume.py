"""Continue frozen shooting fits with one immutable plan read per process."""
import argparse
import os
import time
from pathlib import Path

from src.data.fixed_cohort.protocol import sha, write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, recorded_stage
from .common import read, result, plan


def submit(c):
    root = result(c) / 'technical/continuation-20261001'
    if (root / 'launch.json').exists():
        raise ValueError('Continuation already submitted')
    original = result(c) / 'technical/code'
    if read(original / 'config.json') != c:
        raise ValueError('Continuation must retain the frozen scientific config')
    pending = []
    for index in range(len(c['arms']) * len(c['fit_seeds'])):
        arm = c['arms'][index // len(c['fit_seeds'])]['name']
        seed = c['fit_seeds'][index % len(c['fit_seeds'])]
        path = result(c) / 'analyses/readouts-v1' / arm / str(seed) / 'complete.json'
        if path.exists():
            receipt = read(path)
            if receipt['config'] != c or receipt['producer_sha256'] != sha(original / 'src/research/shooting_laws/fit.py'):
                raise ValueError(f'Incompatible completed fit: {path}')
        else:
            pending.append(index)
    if not pending:
        raise ValueError('No unfinished fits')
    bundle = ExecutionBundle.freeze(original, root / 'code', c, directories=('src', 'docs/metrics'),
        files=((Path(__file__), 'src/research/shooting_laws/resume.py'),))
    receipt = dict(jobs={}, pending=pending, original=str(original),
        adapter_sha256=sha(Path(__file__)), original_fit_sha256=sha(original / 'src/research/shooting_laws/fit.py'),
        change='Read and validate immutable parent plan once per fit process; original fit producer and math unchanged')
    repo = Path(__file__).resolve().parents[3]
    queue = SlurmQueue(root, bundle, 'src.research.shooting_laws.resume',
        dict(PCM_PROJECT_ROOT=str(repo), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'),
        root / 'launch.json', receipt, 'LAW-R')
    with queue.submission():
        queue.submit('fit', ['--array=' + ','.join(map(str, pending)) + '%3',
                            '--cpus-per-task=4', '--mem=12G', '--time=04:00:00'])
    return receipt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('submit', 'fit'))
    parser.add_argument('--config', required=True)
    args = parser.parse_args(); c = read(args.config)
    if args.stage == 'submit':
        print(submit(c)); return
    from . import fit
    start = time.monotonic()
    saved_plan = plan(c)
    def cached_plan(request):
        if request != c:
            raise ValueError('Fit requested a different protocol')
        return saved_plan
    fit.plan = cached_plan
    index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    path = result(c) / 'technical/continuation-20261001' / f'fit-{index}.json'
    with recorded_stage(path, job=os.environ['SLURM_JOB_ID'], plan_load_seconds=time.monotonic() - start,
                        adapter_sha256=sha(Path(__file__)), original_fit_sha256=sha(Path(fit.__file__))) as record:
        record.update(result=fit.fit(c, index))


if __name__ == '__main__':
    main()
