"""Execute explicit forecast/evaluation commands serially in an existing allocation."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

from src.project_runtime.paths import load_json


def dependency_state(spec):
    state = json.loads(Path(spec['path']).read_text())
    if state['state'] == spec['success_state']:
        return True
    if state['state'] not in spec['pending_states']:
        raise RuntimeError(f"Forecast queue dependency cannot succeed: {spec['path']}: {state}")
    return False


def execute(plan):
    root = Path(plan['output'])/'technical'
    root.mkdir(parents=True, exist_ok=True)
    progress = Path(plan['status_path']) if 'status_path' in plan else root/'allocation-status.json'
    if progress.exists():
        raise FileExistsError(f'Allocation plan was already started: {progress}; inspect its retained attempt.')
    if socket.gethostname() != plan['node'] or os.environ['SLURM_JOB_ID'] != str(plan['allocation']):
        raise RuntimeError(f"Plan requires {plan['node']} allocation {plan['allocation']}; "
                           f"got {socket.gethostname()} allocation {os.environ['SLURM_JOB_ID']}")
    deadline = datetime.fromisoformat(plan['deadline_utc']).timestamp()

    def status(**fields):
        temporary = progress.with_suffix('.building')
        temporary.write_text(json.dumps(dict(pid=os.getpid(), node=plan['node'],
            allocation=plan['allocation'], updated_at=datetime.now(timezone.utc).isoformat(), **fields), indent=2)+'\n')
        temporary.replace(progress)

    bootstrap = ('import runpy,sys;sys.path.insert(0,sys.argv.pop(1));'
                 'runpy.run_module(sys.argv.pop(1),run_name="__main__",alter_sys=True)')
    try:
        for index, step in enumerate(plan['steps']):
            status(state='waiting', current=step['name'], step=index, steps=len(plan['steps']))
            while not all([dependency_state(spec) for spec in step['dependencies']]):
                if time.time() >= deadline:
                    raise TimeoutError(f"Allocation dependency deadline reached before {step['name']}")
                time.sleep(10)
            if deadline-time.time() < step['minimum_remaining_seconds']:
                raise TimeoutError(f"Insufficient allocation time to start {step['name']}; inspect completed fits before rescheduling.")
            command = [sys.executable, '-u', '-c', bootstrap, step['source'], step['module'], *step['arguments']]
            status(state='running', current=step['name'], step=index, steps=len(plan['steps']), command=command)
            print(f"Starting {step['name']}", flush=True)
            with (root/(step['name']+'.log')).open('x') as log:
                subprocess.run(command, cwd=plan['cwd'], stdout=log, stderr=subprocess.STDOUT, check=True)
            for spec in step['completion']:
                if not dependency_state(spec):
                    raise RuntimeError(f"Command exited without its declared completion: {step['name']}: {spec}")
            print(f"Completed {step['name']}", flush=True)
        status(state='complete', steps=len(plan['steps']))
    except BaseException:
        status(state='failed', error=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    execute(load_json(args.plan))


if __name__ == '__main__':
    main()
