"""Use an existing allocation until the queued local-response worker takes over.

This operational adapter calls the immutable scientific collector. Holding its
existing gate lock prevents the queued coordinator from entering collection (and
sealing) before the helper has finished and archived its current parent.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--queued-job', required=True)
    args = parser.parse_args()
    bundle = Path(args.bundle).resolve()
    os.chdir(bundle)
    sys.path.insert(0, str(bundle))
    from src.research.local_response import collection
    from src.research.local_response.common import read, root, bind, write_json, sha
    from src.research.local_response.data import prepare
    c = read(bundle/'config.json')
    identity = bind(c)['identity']
    states = prepare(c)
    lane = 'current-' + os.environ['SLURM_JOB_ID']
    receipt = root(c)/'technical/execution-v1'/f'{lane}.json'
    details = dict(identity=identity, lane=lane, pid=os.getpid(), bundle=str(bundle),
        queued_job=args.queued_job, started_at=time.time(),
        adapter_sha256=sha(Path(__file__)), scientific_producer_changed=False)
    write_json(receipt, dict(details, state='starting'))

    def queued_state():
        value = subprocess.check_output(
            ['squeue', '-h', '-j', args.queued_job, '-o', '%T'], text=True).strip()
        if value not in ('PENDING', 'RUNNING', 'CONFIGURING'):
            raise RuntimeError(f'Unexpected queued coordinator state {value!r}; inspect job {args.queued_job}')
        return value

    def stop(signum, frame):
        raise TimeoutError(f'Allocation signal {signum}; completed batches already archived')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGUSR1, stop)
    try:
        with (root(c)/'technical/gate.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if queued_state() != 'PENDING':
                write_json(receipt, dict(details, state='complete', reason='queued worker already starting', finished_at=time.time()))
                return
            gate = read(root(c)/'technical/gates/complete.json')
            if gate['state'] != 'complete' or gate['identity'] != identity:
                raise ValueError('Completed scientific gates required')

            def pending_parents():
                for state in states:
                    if queued_state() != 'PENDING':
                        print('Queued worker starting; current parent archived, releasing gate lock.', flush=True)
                        return
                    write_json(receipt, dict(details, state='running', next_parent=state['index'], updated_at=time.time()))
                    yield state

            # Scope only scheduling, never numerical inputs, branches or collection.
            collection.prepare = lambda config: pending_parents()
            collection.collect(c, lane)
        write_json(receipt, dict(details, state='complete', reason='handoff or all parents collected', finished_at=time.time()))
    except BaseException:
        write_json(receipt, dict(details, state='failed', finished_at=time.time(), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
