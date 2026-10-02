"""Parallel execution adapter for an already frozen response-training experiment.

Run this file by absolute path so scientific imports come only from --bundle.
Parent locks cover the entire frozen collector call, including failure publication.
The adapter changes task assignment and progress destinations, not numerical code.
"""
import argparse
from contextlib import contextmanager, ExitStack
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback


@contextmanager
def lock(path, *, blocking=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
        else:
            try:
                yield True
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def parent_collector(data, c, state, binding, progress):
    """Scope the original collector to one locked parent; defer global sealing."""
    original = {name: getattr(data, name) for name in
                ('bind', 'prepare', 'seal', 'write_metric_rows', 'write_json')}

    def same_config(value):
        if value != c:
            raise ValueError('Execution adapter received a changed scientific configuration')

    def bound(value):
        same_config(value)
        return binding

    def prepared(value):
        same_config(value)
        return [state]

    def defer_seal(value):
        same_config(value)

    def defer_costs(rows, destination, *, family, name):
        if name != 'oracle-cost' or len(rows) != 1 or rows[0]['parent'] != state['index']:
            raise ValueError('Frozen collection export contract changed')

    def write(path, value):
        if Path(path) == data.root(c)/'technical/collection-progress.json':
            # Counts here refer to this lane's single parent, never the full run.
            path = progress
            value = dict(value, lane_parent_count=1, updated_at=time.time())
        original['write_json'](path, value)

    data.bind, data.prepare, data.seal = bound, prepared, defer_seal
    data.write_metric_rows, data.write_json = defer_costs, write
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(data, name, value)


def worker(args):
    bundle = args.bundle.resolve()
    os.chdir(bundle)
    sys.path.insert(0, str(bundle))
    from src.research.response_training import common, data
    from src.experiment_runner.execution import allocation_deadline
    from src.project_runtime.paths import resolve_path
    import torch

    if Path(data.__file__).resolve() != bundle/'src/research/response_training/data.py':
        raise ValueError('Scientific producer was not imported from the frozen bundle')
    c = common.read(bundle/'config.json')
    tech = common.root(c)/'technical'
    execution = tech/'parallel-v1'
    lane = execution/args.lane
    lane.mkdir(parents=True, exist_ok=True)
    if args.prior_job:
        active = subprocess.check_output(
            ['squeue', '-h', '-u', str(os.getuid()), '-o', '%i %T'], text=True)
        previous = [line for line in active.splitlines() if line.split()[0] == args.prior_job]
        if previous:
            raise RuntimeError(f'Previous single-writer job still active: {previous}')
    torch.set_num_threads(2)
    context = dict(lane=args.lane, host=socket.gethostname(), pid=os.getpid(),
        job=os.environ.get('SLURM_JOB_ID'), coordinator=args.coordinator,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        device=torch.cuda.get_device_name(), bundle=str(bundle),
        execution_sha256=common.sha(__file__), started_at=time.time())
    with lock(execution/'prepare.lock', blocking=True):
        binding = common.bind(c)
        gate = common.read(tech/'preflight.json')
        if gate['state'] != 'complete' or gate['identity'] != binding['identity']:
            raise ValueError('Matching numerical preflight required')
        states = data.prepare(c)
    context['identity'] = binding['identity']
    archive = resolve_path(c['simulation_archive'])
    deadline = allocation_deadline(reserve_seconds=300)

    def status(state, **fields):
        common.write_json(lane/'status.json', dict(context, state=state,
                          updated_at=time.time(), **fields))
        if args.coordinator:
            complete = sum((archive/f'parent-{s["index"]:03d}'/'complete.json').exists()
                           for s in states)
            common.write_json(tech/'collection-progress.json', dict(
                state=state, completed_parents=complete, total_parents=len(states),
                execution='parallel-v1', lane_progress=str(execution),
                updated_at=time.time(), **fields))

    def validate_complete(index):
        folder = archive/f'parent-{index:03d}'
        receipt = common.read(folder/'complete.json')
        if receipt['identity'] != binding['identity'] or common.sha(folder/'query.pt') != receipt['sha256']:
            raise ValueError(f'Changed completed parent {index}')
        return receipt

    def check_time():
        if time.time() >= deadline:
            raise TimeoutError('Allocation reserve reached; locked work can resume on another lane')

    def stop(signum, frame):
        raise TimeoutError(f'Scheduler signal {signum}; completed branches remain archived')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGUSR1, stop)
    status('running')
    try:
        with lock(lane/'worker.lock') as acquired:
            if not acquired:
                raise RuntimeError(f'Lane already running: {args.lane}')
            # Every lane can reclaim unfinished work after another allocation ends.
            # Helpers start from the end to leave the existing partial parent to
            # the L40S coordinator. Locks, rather than ordering, ensure exclusion.
            order = states if args.coordinator else list(reversed(states))
            while True:
                pending = [s for s in order if not
                           (archive/f'parent-{s["index"]:03d}'/'complete.json').exists()]
                if not pending:
                    break
                advanced = False
                for state in pending:
                    check_time()
                    index = state['index']
                    with lock(execution/'locks'/f'parent-{index:03d}.lock') as acquired:
                        if not acquired:
                            continue
                        if (archive/f'parent-{index:03d}'/'complete.json').exists():
                            validate_complete(index)
                            continue
                        status('collecting', parent=index, role=state['role'])
                        owner = dict(context, parent=index, acquired_at=time.time())
                        common.write_json(execution/'owners'/f'parent-{index:03d}.json', owner)
                        with parent_collector(data, c, state, binding, lane/'collection-progress.json'):
                            data.collect(c)
                        validate_complete(index)
                        common.write_json(execution/'owners'/f'parent-{index:03d}.json',
                                          dict(owner, state='complete', finished_at=time.time()))
                        advanced = True
                if not advanced:
                    status('waiting_for_other_lanes', pending_parents=[s['index'] for s in pending])
                    check_time()
                    time.sleep(20)
            if not args.coordinator:
                status('complete', scope='collection helper')
                return
            with lock(execution/'coordinator.lock') as acquired:
                if not acquired:
                    raise RuntimeError('Another coordinator owns sealing/training')
                # A complete receipt precedes the frozen producer's final scratch
                # copy. Wait for every owner to release its lock before reading.
                with ExitStack() as held:
                    for state in states:
                        held.enter_context(lock(execution/'locks'/f'parent-{state["index"]:03d}.lock',
                                                blocking=True))
                    costs = [validate_complete(s['index'])['cost'] for s in states]
                    if not (tech/'data-complete.json').exists():
                        status('sealing')
                        fast = 'simulation_profile' in c
                        data.write_metric_rows(costs, common.root(c)/('analyses/collection-v2' if fast else 'analyses/collection-v1'),
                                               family=common.metric_family(c) if fast else common.FAMILY, name='oracle-cost')
                        data.seal(c)
                # Preserve the original arm order, selector, optimizer resume,
                # W&B identity, metric producers and nine-fit evaluation.
                from src.research.response_training.train import fit
                from src.research.response_training.evaluate import collect as evaluate
                for index, seed in enumerate(c['fit_seeds']):
                    for arm in c['arms'][index:]+c['arms'][:index]:
                        check_time()
                        status('training', arm=arm, seed=seed)
                        fit(c, arm, seed)
                status('evaluating')
                evaluate(c)
                status('complete', scope='collection, nine fits and paired evaluation')
    except TimeoutError as error:
        status('paused', error=repr(error), reason='allocation/signal; durable branch or optimizer continuation')
        job = os.environ.get('SLURM_JOB_ID')
        restarts = int(os.environ.get('SLURM_RESTART_COUNT', '0'))
        if (args.coordinator and job and restarts < 2 and
                time.time() >= allocation_deadline(reserve_seconds=360)):
            # This option is used only by the dedicated batch coordinator,
            # never the two helpers inside the user's interactive allocation.
            spec = json.loads(subprocess.check_output(
                ['scontrol', 'show', 'job', job, '--json'], text=True))['jobs'][0]
            if not spec['batch_flag']:
                raise RuntimeError('Refusing to requeue an interactive allocation') from error
            common.write_json(execution/f'requeue-{restarts+1}.json',
                              dict(job=job, requested_at=time.time(), next_restart=restarts+1))
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            subprocess.run(['scontrol', 'requeue', job], check=True)
        elif args.coordinator:
            raise
    except BaseException as error:
        status('failed', error=repr(error), traceback=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--lane', required=True)
    parser.add_argument('--coordinator', action='store_true')
    parser.add_argument('--prior-job')
    args = parser.parse_args()
    worker(args)


if __name__ == '__main__':
    main()
