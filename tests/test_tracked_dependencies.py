import json
import os
from pathlib import Path
import socket
import time

import pytest

from src.experiment_runner.tracking import wait_for_dependencies


def record(path, state='running', pid=None):
    pid = os.getpid() if pid is None else pid
    path.write_text(json.dumps(dict(state=state, host=socket.gethostname(), pid=pid,
        pid_start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19])))


def test_queue_waits_for_tracked_analysis_completion(tmp_path, monkeypatch):
    dependency = tmp_path/'predecessor.json'
    status = tmp_path/'queue.json'
    record(dependency)
    def analysis_finishes(seconds):
        assert json.loads(status.read_text())['state'] == 'waiting_for_dependencies'
        record(dependency, 'command_succeeded')
    monkeypatch.setattr('src.experiment_runner.tracking.time.sleep', analysis_finishes)
    wait_for_dependencies([str(dependency)], until=time.time()+60, status_path=status)
    assert json.loads(status.read_text())['state'] == 'dependencies_complete'


@pytest.mark.parametrize('state,pid', [('failed', None), ('running', 999999999)])
def test_queue_rejects_failed_or_dead_predecessor(tmp_path, state, pid):
    dependency = tmp_path/'predecessor.json'
    status = tmp_path/'queue.json'
    record(dependency, state, pid)
    with pytest.raises(RuntimeError, match='Dependency cannot complete'):
        wait_for_dependencies([str(dependency)], until=time.time()+60, status_path=status)
    assert json.loads(status.read_text())['state'] == 'failed'


def test_queue_deadline_is_enforced(tmp_path):
    dependency = tmp_path/'predecessor.json'
    record(dependency)
    with pytest.raises(TimeoutError, match='deadline'):
        wait_for_dependencies([str(dependency)], until=time.time()-1, status_path=tmp_path/'queue.json')


@pytest.mark.parametrize('alive', [True, False])
def test_remote_dependency_requires_verified_process_identity(tmp_path, monkeypatch, alive):
    dependency = tmp_path/'remote.json'
    value = dict(state='running', host='another-slurm-node', pid=123,
                 pid_start_ticks='456', slurm_job_id='789', python='/python')
    dependency.write_text(json.dumps(value))
    observed = []
    def remote(record):
        observed.append(record['pid_start_ticks'])
        return 'process alive' if alive else 'interrupted/stale: PID exited or was reused'
    monkeypatch.setattr('src.experiment_runner.tracking.remote_process_observation', remote)
    def finishes(seconds):
        value['state'] = 'command_succeeded'
        dependency.write_text(json.dumps(value))
    monkeypatch.setattr('src.experiment_runner.tracking.time.sleep', finishes)
    if alive:
        wait_for_dependencies([str(dependency)], until=time.time()+60, status_path=tmp_path/'status.json')
    else:
        with pytest.raises(RuntimeError, match='Dependency cannot complete'):
            wait_for_dependencies([str(dependency)], until=time.time()+60, status_path=tmp_path/'status.json')
    assert observed == ['456']
