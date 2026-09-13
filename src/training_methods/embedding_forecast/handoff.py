"""Replace a forecast process inside its existing Slurm batch allocation.

The immutable original batch launcher must stay alive while the new step runs.
It is stopped temporarily and released only after the replacement exits. Its
original child is intentionally interrupted, so collection must use afterany
and verify the new forecast completion artifacts rather than the old exit code.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import time

import torch

from src.experiment_runner.registry import sha256, write_json
from src.experiment_runner.tracking import execute_spec
from .runtime import check_resume_implementation, implementation_hashes


def process_state(pid):
    fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(state=fields[0], group=int(fields[2]), started=fields[19])


def wait_state(pid, desired, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not (Path('/proc') / str(pid)).exists():
            if 'Z' in desired:
                return
            raise ProcessLookupError(f'Process {pid} exited before reaching state {desired}.')
        if process_state(pid)['state'] in desired:
            return
        time.sleep(0.05)
    raise TimeoutError(f'Process {pid} did not reach state {desired} within {timeout} seconds.')


@contextmanager
def hold_original_launcher(record):
    """Validate process identity, then freeze the old launcher and its child group."""
    tracker = record['pid']
    observed = process_state(tracker)
    if (record['state'] != 'running' or record['host'] != socket.gethostname() or
            observed['started'] != record['pid_start_ticks']):
        raise RuntimeError(f'Original tracked process identity/state changed: {record}')
    children = (Path('/proc') / str(tracker) / 'task' / str(tracker) / 'children').read_text().split()
    if len(children) != 1:
        raise RuntimeError(f'Expected one original tracked forecast child, found {children}; tracker={tracker}')
    child = int(children[0])
    command = (Path('/proc') / str(child) / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
    if command != record['command'] or process_state(child)['group'] != child:
        raise RuntimeError(f'Original forecast command/process group changed: pid={child}, command={command}')
    os.kill(tracker, signal.SIGSTOP)
    try:
        wait_state(tracker, {'T'})
        os.killpg(child, signal.SIGSTOP)
        wait_state(child, {'T'})
        yield child
    finally:
        if (Path('/proc') / str(child)).exists() and process_state(child)['state'] == 'T':
            os.killpg(child, signal.SIGCONT)
        os.kill(tracker, signal.SIGCONT)


def handoff(plan_path, variant):
    plan = json.loads(Path(plan_path).read_text())
    run = plan['runs'][variant]
    if os.environ['SLURM_JOB_ID'] != run['job_id'] or socket.gethostname() != run['node']:
        raise RuntimeError(f'Handoff must run inside job {run["job_id"]} on {run["node"]}.')
    directory = Path(run['output'])
    directory.mkdir(parents=True, exist_ok=True)
    retained = directory / 'retained_checkpoints'
    retained.mkdir(exist_ok=False)
    record = json.loads(Path(run['original_record']).read_text())
    scientific = json.loads(Path(plan['config']).read_text())
    artifacts = Path(run['artifacts'])
    status = dict(state='preparing', variant=variant, job_id=run['job_id'],
        step_id=os.environ['SLURM_STEP_ID'], started_at=datetime.now(timezone.utc).isoformat())
    write_json(directory / 'handoff.json', status)
    try:
        with hold_original_launcher(record) as child:
            last = torch.load(artifacts / 'last.pt', map_location='cpu', weights_only=False)
            best = torch.load(artifacts / 'best.pt', map_location='cpu', weights_only=False)
            expected = next(v for v in scientific['variants'] if v['name'] == variant)
            if last['config'] != scientific or last['variant'] != expected or last['seed'] != run['seed']:
                raise ValueError('Original checkpoint differs from the declared scientific configuration.')
            if best['epoch'] > last['epoch']:
                raise RuntimeError('Original process was between best/last checkpoint saves; it will be resumed. Retry after the epoch finishes.')
            if last['epoch'] + 1 >= scientific['training']['epochs']:
                raise RuntimeError('Training epochs already finished; leave the original final evaluation running.')
            check_resume_implementation(last, implementation_hashes(), plan['transition'], retained)
            for name in ('best.pt', 'last.pt', 'config.json', 'training.jsonl', 'status.json'):
                # Checkpoints are atomically replaced by the producer; hard links preserve their exact bytes.
                if name.endswith('.pt'):
                    os.link(artifacts / name, retained / name)
                else:
                    (retained / name).write_bytes((artifacts / name).read_bytes())
            status.update(state='checkpoint_retained', original_child_pid=child,
                completed_epochs=last['epoch'] + 1, step=last['step'],
                last_sha256=sha256(retained / 'last.pt'), best_sha256=sha256(retained / 'best.pt'))
            write_json(directory / 'handoff.json', status)
            del last, best
            os.killpg(child, signal.SIGTERM)
            os.killpg(child, signal.SIGCONT)
            wait_state(child, {'Z'})
            status['state'] = 'optimized_training'
            write_json(directory / 'handoff.json', status)
            execute_spec(Path(run['spec']))
            if json.loads((artifacts / 'status.json').read_text())['state'] != 'complete':
                raise RuntimeError(f'Optimized command exited without complete forecast artifacts: {artifacts}')
            status.update(state='complete', finished_at=datetime.now(timezone.utc).isoformat())
            write_json(directory / 'handoff.json', status)
    except BaseException as error:
        status.update(state='failed', error=repr(error), finished_at=datetime.now(timezone.utc).isoformat())
        write_json(directory / 'handoff.json', status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--variant', required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f'Forecast handoff interrupted by signal {signum}')
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        handoff(args.plan, args.variant)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    main()
