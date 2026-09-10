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
