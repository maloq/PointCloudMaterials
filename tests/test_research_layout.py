"""Output readability, frozen metric definitions and destructive cleanup boundaries."""

import csv
import json
from pathlib import Path

import pytest

from src.experiment_runner.artifacts import analysis_artifacts
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_table
from src.experiment_runner.storage import clean_caches


def cache_fixture(repo):
    analysis = repo / 'outputs/example/technical'
    analysis.mkdir(parents=True)
    checkpoint = repo / 'output/model/best.ckpt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'selected checkpoint')
    (analysis / 'analysis_metrics.json').write_text('{"clustering": {"primary_k": 7}}')
    (analysis / 'test_predictions.npz').write_bytes(b'paired observations')
    data = analysis / 'analysis_inference_cache.npz'
    data.write_bytes(b'regenerable arrays')
    metadata = data.with_suffix('.npz.meta.json')
    stat = checkpoint.stat()
    metadata.write_text(json.dumps(dict(spec=dict(version=8, checkpoint=dict(
        path=str(checkpoint), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)))))
    return analysis, checkpoint, data, metadata


def test_cache_cleanup_preserves_results_and_reconstruction_provenance(tmp_path):
    analysis, checkpoint, data, metadata = cache_fixture(tmp_path)
    content = metadata.read_bytes()
    preview = clean_caches(tmp_path, ['outputs/example'])
    assert preview['files'] == 2 and data.exists() and metadata.exists()
    with pytest.raises(ValueError, match='--inactive'):
        clean_caches(tmp_path, ['outputs/example'], apply=True)
    clean_caches(tmp_path, ['outputs/example'], apply=True, inactive=True)
    assert not data.exists() and not metadata.exists()
    assert checkpoint.exists() and (analysis / 'test_predictions.npz').exists()
    copies = list((tmp_path / 'output/maintenance').glob('*/technical/retained-cache-metadata/**/*.meta.json'))
    assert copies and all(p.read_bytes() == content for p in copies)
    # An archived sidecar must not be mistaken for an orphan live cache on the next pass.
    assert clean_caches(tmp_path, None)['files'] == 0


def test_cache_with_changed_checkpoint_is_kept(tmp_path):
    _, checkpoint, data, metadata = cache_fixture(tmp_path)
    checkpoint.write_bytes(b'new model')
    assert clean_caches(tmp_path, ['outputs/example'])['files'] == 0
    assert data.exists() and metadata.exists()


def test_cleanup_rejects_redirected_storage(tmp_path):
    (tmp_path / 'output').mkdir()
    outside = tmp_path / 'dataset'
    outside.mkdir()
    (tmp_path / 'output/cache').symlink_to(outside)
    with pytest.raises(ValueError, match='symlink'):
        clean_caches(tmp_path, ['output/cache'])


def test_metric_documents_match_implementation_and_travel_with_table(tmp_path):
    assert set(check_metric_docs()) == {'analysis', 'topology', 'forecast', 'forecast_context', 'forecast_crystallization', 'forecast_spatial_mixture', 'aggregation', 'mace_encoder_diagnostics', 'mace_tda_ridge_audit', 'mace_context', 'mace_context_smoothness', 'mace_context_recovery', 'mace_velocity', 'mace_data_amount', 'mace_causal', 'mace_causal_comparison', 'hardware_benchmark', 'mace_causal_runtime', 'predictive_memory'}
    path = write_metric_table({'test': {'balanced_mse': 0.125, 'undefined': None, 'ci95': [0.1, 0.2]}},
                              tmp_path, family='topology')
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [{'metric': 'test.balanced_mse', 'value': '0.125'},
                    {'metric': 'test.undefined', 'value': ''},
                    {'metric': 'test.ci95.lower', 'value': '0.1'},
                    {'metric': 'test.ci95.upper', 'value': '0.2'}]
    assert '4,000 paired resamples' in (tmp_path / 'tables/METRICS.md').read_text()
    recorded = json.loads((tmp_path / 'technical/metric-contract.json').read_text())
    assert recorded['files']['src/analysis/topology_metrics.py']


def test_analysis_paths_keep_legacy_caches_and_hide_new_artifacts(tmp_path):
    assert analysis_artifacts(tmp_path) == tmp_path / 'technical'
    (tmp_path / 'analysis_metrics.json').write_text('{}')
    assert analysis_artifacts(tmp_path) == tmp_path


def test_experiment_implementations_do_not_import_dated_recipes():
    import ast
    for path in Path('src').rglob('*.py'):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith('experiments.'), path
