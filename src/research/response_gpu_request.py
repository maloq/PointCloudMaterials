"""Submit two response-collection helpers; retry the Slurm job-count limit only."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def save(path, value):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def request(execution, deadline):
    receipt = execution/'launch.json'
    status_path = execution/'gpu-request-status.json'
    attempt = 0
    with (execution/'gpu-request.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while time.time() < deadline:
            attempt += 1
            if (execution.parent/'data-complete.json').exists():
                save(status_path, dict(state='no_longer_needed', reason='collection already sealed'))
                return
            with (execution/'submission.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                record = json.loads(receipt.read_text())
                wrapper = Path(record['wrapper'])
                if hashlib.sha256(wrapper.read_bytes()).hexdigest() != record['wrapper_sha256']:
                    raise ValueError('Recorded execution adapter changed')
                for number in (1, 2):
                    lane = f'l40s-helper{number}'
                    if lane in record['jobs']:
                        continue
                    command = [os.sys.executable, '-u', str(wrapper), '--bundle', record['bundle'],
                               '--lane', lane, '--prior-job', record['prior_job']]
                    script = execution/(lane+'.sbatch')
                    script.write_text('\n'.join([
                        '#!/bin/bash', f'#SBATCH --job-name=RESP-TRAIN-{lane}',
                        '#SBATCH --partition=L40S', '#SBATCH --nodes=1', '#SBATCH --ntasks=1',
                        '#SBATCH --gpus=1', '#SBATCH --cpus-per-task=4', '#SBATCH --mem=64G',
                        '#SBATCH --time=12:00:00', '#SBATCH --signal=B:TERM@240',
                        '#SBATCH --exclude=node52', f'#SBATCH --output={execution}/{lane}-%j.log',
                        'set -euo pipefail', 'ulimit -n 4096', 'cd '+shlex.quote(record['bundle']),
                        'exec env '+shlex.join([f'{k}={v}' for k,v in record['environment'].items()]+command), '']))
                    result = subprocess.run(['sbatch', '--parsable', str(script)],
                                            text=True, capture_output=True)
                    if result.returncode:
                        status = dict(state='waiting_for_submission_slot', attempt=attempt,
                            error=result.stderr.strip(), updated_at=time.time(), deadline=deadline,
                            accepted_jobs={k:v for k,v in record['jobs'].items() if k.startswith('l40s-helper')})
                        if 'QOSMaxSubmitJobPerUserLimit' not in result.stderr:
                            status['state'] = 'failed'
                            save(status_path, status)
                            raise RuntimeError(result.stderr)
                        save(status_path, status)
                        print(json.dumps(status), flush=True)
                        break
                    job = result.stdout.strip().split(';')[0]
                    if not job.isdigit():
                        raise ValueError(f'Unexpected sbatch receipt: {result.stdout!r}')
                    record['jobs'][lane] = job
                    record['helpers'].append(dict(lane=lane, job=job, partition='L40S',
                        script=str(script), command=command, submitted_at=time.time(),
                        log=str(execution/f'{lane}-{job}.log')))
                    save(receipt, record)
                    print(json.dumps(dict(lane=lane, job=job, state='submitted')), flush=True)
                else:
                    save(status_path, dict(state='submitted', updated_at=time.time(),
                        jobs={k:v for k,v in record['jobs'].items() if k.startswith('l40s-helper')}))
                    return
            time.sleep(min(60, max(0, deadline-time.time())))
        save(status_path, dict(state='expired', reason='requester allocation ending',
                              updated_at=time.time(), deadline=deadline))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--execution', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    request(args.execution, args.deadline)
