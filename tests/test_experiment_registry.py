"""Destructive cleanup boundaries and durable run provenance."""
import json
from pathlib import Path
import sys

import pytest

from src.experiment_runner.registry import prune, sha256, checked_path, run_id, pack_logs
from src.experiment_runner.tracking import tracked_run, observed_record, execute_spec


def item(path, repo, reason='regenerable cache'):
    return {'path': str(path.relative_to(repo)), 'bytes': path.stat().st_size,
            'sha256': sha256(path), 'reason': reason}


def test_prune_preflights_whole_plan_and_preserves_prerequisite(tmp_path):
    root = tmp_path / 'output'
    (root / 'registry').mkdir(parents=True)
    first, second, checkpoint = [root / name for name in ['cache1.npy', 'cache2.npy', 'best.pt']]
    for path in (first, second, checkpoint):
        path.write_bytes(path.name.encode())
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'remove': [item(first, tmp_path), item(second, tmp_path)],
                                'required': [item(checkpoint, tmp_path)]}))
    original = second.read_bytes()
    second.write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='changed'):
        prune(tmp_path, plan, True)
    assert first.exists() and checkpoint.exists()
    second.write_bytes(original)
    assert prune(tmp_path, plan, False)['state'] == 'preview'
    assert first.exists()
    assert prune(tmp_path, plan, True)['state'] == 'complete'
    assert not first.exists() and not second.exists() and checkpoint.exists()
    assert len((root / 'registry/cleanup_applied.jsonl').read_text().splitlines()) == 2


def test_prune_rejects_external_symlink_and_required_overlap(tmp_path):
    output = tmp_path / 'output'
    output.mkdir()
    outside = tmp_path / 'dataset.npy'
    outside.write_bytes(b'irreplaceable')
    (output / 'data').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        checked_path(tmp_path, 'output/data/dataset.npy')
    with pytest.raises(ValueError, match='inside output'):
        checked_path(tmp_path, 'output/../dataset.npy')
    kept = output / 'best.pt'
    kept.write_bytes(b'model')
    plan = tmp_path / 'plan.json'
    record = item(kept, tmp_path)
    plan.write_text(json.dumps({'remove': [record], 'required': [record]}))
    with pytest.raises(ValueError, match='prerequisite'):
        prune(tmp_path, plan, True)
    assert kept.exists()


def test_tracking_records_failure_and_keeps_each_attempt(tmp_path):
    config = tmp_path / 'config.json'
    config.write_text('{"seed": 123}')
    output = tmp_path / 'run'
    with pytest.raises(RuntimeError, match='scientific failure'):
        with tracked_run(output, kind='training', configs=[config], command=[sys.executable, 'train.py']):
            raise RuntimeError('scientific failure')
    failed = json.loads((output / 'run_record.json').read_text())
    assert failed['state'] == 'failed'
    assert 'scientific failure' in failed['traceback']
    assert Path(failed['configs'][0]['snapshot']).read_text() == config.read_text()
    with tracked_run(output, kind='analysis', configs=[config], command=[sys.executable, 'analysis.py']):
        assert observed_record(output / 'run_record.json')['observation'] == 'process alive'
    assert json.loads((output / 'run_record.json').read_text())['state'] == 'command_succeeded'
    assert json.loads((Path(failed['attempt']) / 'run_record.json').read_text())['state'] == 'failed'
    assert run_id(Path('2026-03-13/00-58-07/analysis/metrics.json')) == '2026-03-13/00-58-07'


def test_old_logs_are_recoverable_and_simulation_logs_are_excluded(tmp_path):
    import tarfile
    root = tmp_path / 'output'
    (root / 'registry').mkdir(parents=True)
    entries = []
    for kind, name in [('experiment', 'old_training'), ('dataset', 'synthetic_data')]:
        directory = root / name
        directory.mkdir()
        log = directory / 'train.log'
        log.write_bytes(b'epoch=1 validation_loss=0.2\n')
        entries.append({'id': name, 'kind': kind, 'artifacts': [
            {'path': str(log.relative_to(tmp_path)), 'kind': 'logs'}]})
    (root / 'registry/experiments.json').write_text(json.dumps({'experiments': entries}))
    result = pack_logs(tmp_path, '2100-01-01T00:00:00+00:00', True)
    assert len(result) == 1
    assert not (root / 'old_training/train.log').exists()
    assert (root / 'synthetic_data/train.log').exists()
    with tarfile.open(tmp_path / result[0]['archive']) as archive:
        assert archive.extractfile('train.log').read() == b'epoch=1 validation_loss=0.2\n'


def test_execution_failure_keeps_log_and_refuses_implicit_retry(tmp_path):
    import subprocess
    output = tmp_path / 'run'
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({'kind': 'analysis', 'question': 'failure fixture',
        'output': str(output), 'cwd': str(tmp_path), 'configs': [], 'dependencies': [],
        'completion': 'process_exit',
        'command': [sys.executable, '-c', 'print("diagnostic evidence"); raise SystemExit(7)']}))
    with pytest.raises(subprocess.CalledProcessError):
        execute_spec(spec)
    assert 'diagnostic evidence' in (output / 'command.log').read_text()
    assert json.loads((output / 'execution/run_record.json').read_text())['state'] == 'failed'
    with pytest.raises(FileExistsError, match='already tracked'):
        execute_spec(spec)
