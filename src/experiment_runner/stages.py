"""Locked subprocess stages with bounded retries and durable attempt receipts.

Workflows own task ordering, prerequisites, continuation policy and path
resolution. This module only executes an explicitly named stage.
"""
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .artifacts import write_json


def run_stage(output, bundle, lane, deadline, arm_name, name, marker, *,
              module, max_attempts, hard_deadline=False, folder=None):
    """Return ``done``, ``paused``, ``failed``, or None for an unclaimed stage.

    Exit 75 pauses without exhausting retries. Completion is established by the
    workflow's marker, not the exit code. A hard deadline terminates only the
    process group created here. ``folder`` supports source caches whose stage
    artifacts live outside the run's analyses directory.
    """
    import torch

    output = Path(output)
    folder = output / 'analyses' / arm_name / 'technical' if folder is None else Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / marker).exists() or (folder / (name + '-failed.json')).exists():
        return None
    with (folder / 'worker.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        if (folder / marker).exists():
            return None
        torch.cuda.empty_cache()
        for attempt in range(max_attempts):
            command = [sys.executable, '-u', '-m', module, name, '--config', str(bundle.config_path),
                       '--name', arm_name, '--deadline', str(deadline)]
            log = folder / f'{name}-{time.time_ns()}.log'
            with log.open('a') as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=hard_deadline)
                record = dict(stage=name, arm=arm_name, lane=lane, pid=process.pid, attempt=attempt+1,
                              log=str(log), command=command, started=time.time())
                write_json(output / 'technical' / f'lane-{lane}.json', record)
                try:
                    code = process.wait(timeout=max(1, deadline-time.time()) if hard_deadline else None)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    code = 75
                    record['hard_deadline_reached'] = True
            record.update(exit_code=code, finished=time.time())
            write_json(folder / f'{name}-attempt-{time.time_ns()}.json', record)
            if code == 75:
                return 'paused'
            if (folder / marker).exists():
                return 'done'
        write_json(folder / (name + '-failed.json'), record)
        return 'failed'
