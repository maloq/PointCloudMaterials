"""Exercise live launcher holding without allocating or cancelling Slurm jobs."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from src.training_methods.embedding_forecast.handoff import hold_original_launcher, process_state, wait_state


@pytest.fixture
def original_launcher(tmp_path):
    record_path = tmp_path / 'record.json'
    code = '''import json,os,socket,subprocess,sys
from pathlib import Path
command=[sys.executable,'-c','import time; time.sleep(60)']
child=subprocess.Popen(command,start_new_session=True)
pid=os.getpid()
record=dict(pid=pid,pid_start_ticks=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19],
            state='running',host=socket.gethostname(),command=command,child=child.pid)
Path(sys.argv[1]).write_text(json.dumps(record))
sys.exit(0 if child.wait()==0 else 1)
'''
    launcher = subprocess.Popen([sys.executable, '-c', code, str(record_path)])
    deadline = time.monotonic() + 10
    while not record_path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    record = json.loads(record_path.read_text())
    try:
        yield launcher, record
    finally:
        if launcher.poll() is None:
            child = record['child']
            if (Path('/proc') / str(child)).exists() and process_state(child)['state'] != 'Z':
                os.killpg(child, signal.SIGTERM)
                os.killpg(child, signal.SIGCONT)
            os.kill(launcher.pid, signal.SIGCONT)
            launcher.wait(timeout=10)


def test_handoff_error_restores_original_processes(original_launcher):
    launcher, record = original_launcher
    with pytest.raises(RuntimeError, match='preflight failed'):
        with hold_original_launcher(record) as child:
            assert process_state(launcher.pid)['state'] == 'T'
            assert process_state(child)['state'] == 'T'
            raise RuntimeError('preflight failed')
    wait_state(record['child'], {'S', 'R'})
    assert launcher.poll() is None


def test_interrupted_child_does_not_release_allocation_until_replacement_exits(original_launcher):
    launcher, record = original_launcher
    with hold_original_launcher(record) as child:
        os.killpg(child, signal.SIGTERM)
        os.killpg(child, signal.SIGCONT)
        wait_state(child, {'Z'})
        assert launcher.poll() is None
        assert process_state(launcher.pid)['state'] == 'T'
    assert launcher.wait(timeout=10) == 1


def test_handoff_rejects_reused_process_identity(original_launcher):
    launcher, record = original_launcher
    with pytest.raises(RuntimeError, match='identity'):
        with hold_original_launcher(dict(record, pid_start_ticks='0')):
            raise AssertionError('Invalid identity must not enter the handoff.')
    assert launcher.poll() is None
