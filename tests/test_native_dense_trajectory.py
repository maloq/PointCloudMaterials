"""Protect tracked centers and nearest-neighbor identities during dense reuse."""
import numpy as np
import pytest

from src.research.trajectory_stability.native_dense import pack_observations


def test_dense_packing_keeps_ids_and_center_after_context_crop():
    rng = np.random.default_rng(20)
    x = rng.normal(size=(100, 3)).astype(np.float32)
    x[19] = 0
    x[0] = [11, 0, 0]
    ids = rng.permutation(np.arange(100))
    near = np.lexsort((ids, np.square(x.astype(float)).sum(1)))[:80]
    observations = dict(offsets=np.array([0, 100]), center_indices=np.array([19]), nearest_ids=ids[near][None])
    packed = pack_observations(x, ids, observations)
    assert packed['centers'].tolist() == [18]
    np.testing.assert_array_equal(packed['positions'][18], np.zeros(3))
    np.testing.assert_array_equal(packed['nearest80'][0], x[near])
    assert packed['offsets'].tolist() == [0, 99]
    observations['nearest_ids'] = observations['nearest_ids'][:, ::-1]
    with pytest.raises(AssertionError):
        pack_observations(x, ids, observations)


def test_dense_packing_fails_when_saved_support_is_incomplete():
    x = np.arange(90.)[:, None]*np.array([[.2, 0, 0]])
    observations = dict(offsets=np.array([0, 90]), center_indices=np.array([0]), nearest_ids=np.arange(80)[None])
    with pytest.raises(ValueError, match='Incomplete nearest-80'):
        pack_observations(x, np.arange(90), observations)
