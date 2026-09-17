"""Reject scientifically unmatched timings, populations and gate reuse."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from src.data.predictive_memory.prepare import file_hash, write_json
from src.research import backbone_repeats as repeats


def profile():
    return dict(gpu='H100', variant='history12', precision='float32', effective_batch=8, microbatch=8,
        rows=[1, 2], observation_shapes=[[17, 100], [17, 103]], update_seconds=[2., 2., 2., 2.],
        validation_seconds=4., cold_preparation_seconds=3., peak_allocated_bytes=2*1024**3,
        identity=dict(data={'rows': 38400}, config={'seed': 20260919}, implementation={'a.py': 'hash'}))


def test_speedup_is_ratio_of_matched_mean_durations():
    a = profile(); b = copy.deepcopy(a); b['update_seconds'] = [1., 1., 1., 1.]; b['validation_seconds'] = 2.
    result = repeats.compare_profiles(a, b)
    assert result['gatr_training_speedup'] == 2 and result['gatr_windows_per_second'] == 8
    assert result['gatr_validation_speedup'] == 2 and result['mace_peak_allocated_gib'] == 2


@pytest.mark.parametrize('key,value', [('microbatch', 1), ('rows', [2, 3]), ('precision', 'bfloat16'), ('gpu', 'RTX6000')])
def test_mismatched_speedup_is_rejected(key, value):
    a = profile(); b = copy.deepcopy(a); b[key] = value
    with pytest.raises(ValueError, match='Unmatched'):
        repeats.compare_profiles(a, b)


def test_same_length_does_not_mean_same_prediction_population():
    a = {key: np.arange(5) for key in ('indices', 'source', 'center', 'anchor')}
    b = copy.deepcopy(a); b['anchor'][0] = 42
    with pytest.raises(AssertionError, match='anchor'):
        repeats.assert_paired(a, b)


def test_gate_copy_preserves_bytes_and_rejects_different_experiment(tmp_path, monkeypatch):
    parent = {'output': str(tmp_path/'parent'), 'width': 8}
    child = {'output': str(tmp_path/'child'), 'width': 8}
    data = SimpleNamespace()
    monkeypatch.setattr(repeats, 'identity', lambda config, *args: {'width': config['width']})
    monkeypatch.setattr(repeats, 'verify_gate', lambda *args: 'checked-by-trainer')
    source = repeats.gate_path(parent, 'axial_gatr'); source.parent.mkdir(parents=True)
    (source.parent/'best.pt').write_bytes(b'passing immutable model')
    write_json(source, {'checkpoint_sha256': file_hash(source.parent/'best.pt')})
    repeats.copy_verified_gate(parent, child, data)
    target = repeats.gate_path(child, 'axial_gatr')
    assert target.read_bytes() == source.read_bytes()
    assert (target.parent/'best.pt').read_bytes() == (source.parent/'best.pt').read_bytes()
    repeats.copy_verified_gate(parent, child, data)
    with pytest.raises(ValueError, match='scientific configuration'):
        repeats.copy_verified_gate(parent, {**child, 'width': 16}, data)
    (source.parent/'best.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint changed'):
        repeats.copy_verified_gate(parent, child, data)
