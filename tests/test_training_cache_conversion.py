"""Storage conversion must preserve layout and keep unsafe inputs intact."""
import numpy as np
import pytest

from src.data_utils.conversion.training_cache import convert_file
from src.data_utils.pretrained_mace import Quadruplets


def test_local_cache_conversion_decodes_float32_and_is_verifiably_repeatable(tmp_path):
    path = tmp_path / 'clouds.npy'
    values = np.random.default_rng(17).uniform(-12, 12, (3, 4, 8, 3)).astype(np.float32)
    np.save(path, values)
    record = convert_file(path)
    stored = np.load(path, mmap_mode='r')
    assert stored.dtype == np.float16
    np.testing.assert_array_equal(stored, values.astype(np.float16))
    assert record['max_absolute_error'] <= 0.004
    assert convert_file(path)['float16_sha256'] == record['float16_sha256']
    # Exercise the training batch boundary, retaining float32 targets/conditions.
    data = object.__new__(Quadruplets)
    data.clouds = [stored]
    data.tda = [np.zeros((3, 4, 144), dtype=np.float32)]
    data.conditions = [np.zeros((3, 5), dtype=np.float32)]
    data.records = [{'material': 0}]
    x, t, c, material = data.get([(0, 1)])
    assert x.dtype == t.dtype == c.dtype == np.float32
    assert material.dtype == np.int64
    np.testing.assert_array_equal(x[0], stored[1].astype(np.float32))


@pytest.mark.parametrize('bad_value', [np.inf, 100000.0])
def test_unsafe_float16_cache_leaves_original_untouched(tmp_path, bad_value):
    path = tmp_path / 'clouds.npy'
    np.save(path, np.full((2, 4, 3), bad_value, dtype=np.float32))
    before = path.read_bytes()
    with pytest.raises(ValueError, match='Non-finite or float16-overflowing'):
        convert_file(path)
    assert path.read_bytes() == before
    assert not path.with_suffix('.float16.json').exists()
    assert not path.with_name(path.name + '.float16-building').exists()
