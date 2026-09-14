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


@pytest.mark.parametrize('custom_status', [False, True])
def test_allocation_executes_command_and_rejects_duplicate_attempt(tmp_path, monkeypatch, custom_status):
    monkeypatch.setenv('SLURM_JOB_ID', '42')
    source = tmp_path/'source.json'; source.write_text('{"state":"complete"}')
    target = tmp_path/'completed.json'
    plan = dict(output=str(tmp_path/'output'), node=socket.gethostname(), allocation=42,
        deadline_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(), cwd=str(tmp_path),
        steps=[dict(name='write-completion', dependencies=[], source=str(tmp_path), module='json.tool',
            arguments=[str(source), str(target)], minimum_remaining_seconds=1,
            completion=[dict(path=str(target), success_state='complete', pending_states=[])])])
    status_path = tmp_path/'controller.json' if custom_status else tmp_path/'output/technical/allocation-status.json'
    if custom_status:
        plan['status_path'] = str(status_path)
    execute(plan)
    status = json.loads(status_path.read_text())
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


def test_allocation_module_can_spawn_pickled_workers(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '42')
    (tmp_path/'worker_command.py').write_text(
        'from concurrent.futures import ProcessPoolExecutor\n'
        'import multiprocessing\n'
        'import json\n'
        'from pathlib import Path\n'
        'def worker(value):\n'
        '    return value * 2\n'
        'if __name__ == "__main__":\n'
        '    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:\n'
        '        result = pool.submit(worker, 21).result()\n'
        '    assert result == 42\n'
        '    Path("completed.json").write_text(json.dumps({"state": "complete"}))\n')
    target = tmp_path/'completed.json'
    execute(dict(output=str(tmp_path/'output'), node=socket.gethostname(), allocation=42,
        deadline_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(), cwd=str(tmp_path),
        steps=[dict(name='spawn-workers', dependencies=[], source=str(tmp_path), module='worker_command',
            arguments=[], minimum_remaining_seconds=1,
            completion=[dict(path=str(target), success_state='complete', pending_states=[])])]))
    assert json.loads(target.read_text())['state'] == 'complete'
