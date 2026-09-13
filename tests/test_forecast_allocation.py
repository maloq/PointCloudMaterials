"""Explicit allocation commands, failed prerequisites and deadline enforcement."""

from datetime import datetime, timedelta, timezone
import json
import socket

import pytest

from src.training_methods.embedding_forecast.allocation import dependency_state, execute


def test_failed_dependency_is_loud(tmp_path):
    path = tmp_path/'dependency.json'
    path.write_text(json.dumps(dict(state='failed', error='out of memory')))
    with pytest.raises(RuntimeError, match='out of memory'):
        dependency_state(dict(path=str(path), success_state='complete', pending_states=['training']))


def test_allocation_executes_command_and_rejects_duplicate_attempt(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '42')
    source = tmp_path/'source.json'; source.write_text('{"state":"complete"}')
    target = tmp_path/'completed.json'
    plan = dict(output=str(tmp_path/'output'), node=socket.gethostname(), allocation=42,
        deadline_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(), cwd=str(tmp_path),
        steps=[dict(name='write-completion', dependencies=[], source=str(tmp_path), module='json.tool',
            arguments=[str(source), str(target)], minimum_remaining_seconds=1,
            completion=[dict(path=str(target), success_state='complete', pending_states=[])])])
    execute(plan)
    status = json.loads((tmp_path/'output/technical/allocation-status.json').read_text())
    assert status['state'] == 'complete' and target.is_file()
    with pytest.raises(FileExistsError, match='already started'):
        execute(plan)


def test_expired_allocation_does_not_start_command(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '42')
    plan = dict(output=str(tmp_path), node=socket.gethostname(), allocation=42,
        deadline_utc=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat(),
        steps=[dict(name='must-not-start', dependencies=[], minimum_remaining_seconds=1)])
    with pytest.raises(TimeoutError, match='Insufficient allocation time'):
        execute(plan)
    assert json.loads((tmp_path/'technical/allocation-status.json').read_text())['state'] == 'failed'
