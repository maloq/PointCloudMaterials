import numpy as np
import torch
from src.data.predictive_memory.observations import frame_observation
from src.research.local_predictability.native_data import (
    BoundedCache, NativeWindows, SourceSampler, conditions,
)


def test_bounded_cache_evicts_and_rejects_oversize():
    cache = BoundedCache(16)
    cache.put('a', torch.ones(3)); cache.put('b', torch.ones(2))
    assert cache.get('a') is None and cache.bytes == 8
    cache.put('large', torch.ones(100))
    assert cache.bytes == 8 and cache.get('large') is None


def test_raw_observation_access_is_causal_and_repeat_uses_anchor_only():
    window = NativeWindows.__new__(NativeWindows)
    window.rows = [dict(row_id='1:1:80', source_id=1, center_id=1, anchor=80)]
    window.observations = BoundedCache(0); window.device = torch.device('cpu')
    accessed = []
    rng = np.random.default_rng(8); position = rng.normal(size=(10, 3))
    frame = frame_observation(position, position, np.full(3, 100.), np.arange(1, 11), 1, 17., 5.)
    def read(source, center, index):
        accessed.append(index)
        return frame
    window.frame = read
    window.observation(0, 'history12')
    assert accessed == list(range(64, 81))
    accessed.clear(); repeated = window.observation(0, 'repeat12')
    assert accessed == [80] * 17
    assert torch.equal(repeated.positions[0], repeated.positions[-1])
    accessed.clear(); window.observation(0, 'snapshot')
    assert accessed == [80]


def test_periodic_center_relative_identity_and_velocity():
    position = np.array([[99., 0., 0.], [1., 0., 0.], [50., 0., 0.]])
    velocity = np.array([[2., 0., 0.], [5., 0., 0.], [9., 0., 0.]])
    result = frame_observation(position, velocity, np.full(3, 100.), np.array([4, 8, 12]), 4, 17., 5.)
    assert result['ids'].tolist() == [4, 8]
    torch.testing.assert_close(result['positions'][1], torch.tensor([2., 0., 0.]))
    torch.testing.assert_close(result['velocities'][1], torch.tensor([3., 0., 0.]))


def test_conditions_never_fit_heldout_rows():
    rows = [dict(split='train', temperature_K=400, anchor=i) for i in [64, 104]]
    rows.append(dict(split='test', temperature_K=520, anchor=664))
    _, stats = conditions(rows)
    rows[-1]['anchor'] = 1
    _, updated = conditions(rows)
    assert stats == updated


def test_source_sampler_excludes_heldout_and_resumes_exactly():
    rows = [dict(source_id=i, split='train' if i < 9 else 'test') for i in range(10)]
    sampler = SourceSampler(rows)
    assert set(sampler.batch()) <= set(range(9))
    state = sampler.state_dict(); first = sampler.batch()
    sampler.load_state_dict(state)
    assert sampler.batch() == first
