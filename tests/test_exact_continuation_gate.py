import json
from pathlib import Path

import pytest

from src.simulation.campaigns import predictive_dynamics_15ps as runner


def test_missing_smoke_blocks_extension_before_dynamics(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, '_verify_manifest', lambda root: {'design_kind': 'single_parent_preproduction_smoke_8x2'})
    def forbidden(*args, **kwargs):
        raise AssertionError('Unvalidated extension must not run')
    monkeypatch.setattr(runner, '_run_extension', forbidden)
    with pytest.raises(RuntimeError, match='no passing smoke'):
        runner.extend_branch(tmp_path, 0, selection_reason='test', selection_probability=1.0)
    assert not (tmp_path / 'extended_24ps_branches.json').exists()


def test_explicit_topup_gate_blocks_extension(tmp_path):
    (tmp_path / 'extension_gate.json').write_text(json.dumps({'exact_15_to_24ps_extensions_allowed': False}))
    with pytest.raises(RuntimeError, match='blocked by'):
        runner._require_exact_continuation(tmp_path, {'design_kind': 'legacy_40_parent_topup_from_12_to_16'})


def test_accepted_extension_records_comparison_provenance(tmp_path, monkeypatch):
    proof = dict(state='complete', positions_bitwise_equal_after_text_quantization=True,
                 velocities_bitwise_equal_after_text_quantization=True, mpi_ranks=24,
                 thermostat_style='temp/csld', short_duration_ps=15.0, extended_duration_ps=24.0)
    path = tmp_path / 'continuation_smoke_test.json'
    path.write_text(json.dumps(proof))
    monkeypatch.setattr(runner, '_verify_manifest', lambda root: {'design_kind': 'single_parent_preproduction_smoke_8x2'})
    monkeypatch.setattr(runner, '_run_extension', lambda *args, **kwargs: {'state': 'complete', 'branch_id': 'branch_0', 'branch_index': 0})
    accepted = runner.extend_branch(tmp_path, 0, selection_reason='test', selection_probability=1.0)
    index = json.loads((tmp_path / 'extended_24ps_branches.json').read_text())
    assert index['records'] == [accepted]
    assert accepted['classification'] == 'exact_continuation'
    assert accepted['acceptance_proof_sha256'] == runner.campaign._sha256_file(path)
    proof['velocities_bitwise_equal_after_text_quantization'] = False
    path.write_text(json.dumps(proof))
    with pytest.raises(RuntimeError, match='did not satisfy'):
        runner.extend_branch(tmp_path, 1, selection_reason='test', selection_probability=1.0)
