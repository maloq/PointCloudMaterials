"""Prefetch preserves sampling, observation tensors, losses and cache lifetimes."""
import copy

import numpy as np
import pytest
import torch

from src.research.local_predictability.native_data import BoundedCache, SourceSampler
from src.research.local_predictability.native_runtime import ObservationPrefetcher, peek_batch, evaluate
from src.research.local_predictability.supervised import evaluate as sequential_evaluate
from src.research.local_predictability.onset_model import OnsetModel
from test_local_predictability_native import observation


class Windows:
    def __init__(self):
        self.device = torch.device('cuda')
        self.frames, self.observations = BoundedCache(1024), BoundedCache(1)
        self.rows = [dict(source_id=i, split='train', row_id=str(i)) for i in range(16)]
        self.value = observation()

    def observation(self, index, variant):
        if index == 99:
            raise ValueError('broken source 99')
        return self.value.to(self.device)


def test_lookahead_never_advances_checkpointed_sampler():
    sampler = SourceSampler([dict(source_id=i,split='train') for i in range(16)])
    state = copy.deepcopy(sampler.state_dict())
    expected = peek_batch(sampler)
    assert sampler.state_dict() == state
    assert sampler.batch() == expected
    state = copy.deepcopy(sampler.state_dict())
    for _ in range(3):
        peek_batch(sampler)
    assert sampler.state_dict() == state


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA stream lifetime test')
def test_prefetch_values_order_short_batch_and_worker_failure():
    windows = Windows()
    with ObservationPrefetcher(windows) as inputs:
        batches = list(inputs.batches([5,0,5,2,1], 'history12', 2))
        assert [list(batch) for batch,_ in batches] == [[5,0],[5,2],[1]]
        for _,values in batches:
            for value in values:
                for name in ('positions','velocities','weights','offsets_ps','atom_ids'):
                    torch.testing.assert_close(getattr(value,name).cpu(),getattr(windows.value,name),atol=0,rtol=0)
                for a,b in zip(value.edges,windows.value.edges,strict=True):
                    torch.testing.assert_close(a.cpu(),b,atol=0,rtol=0)
        future = inputs.submit([1], 'snapshot')
        with pytest.raises(ValueError, match='order'):
            inputs.take(future,[2])
    # Force a loader failure after dispatch, with an otherwise valid row key.
    windows.rows.extend([dict(source_id=i,split='train',row_id=str(i)) for i in range(16,100)])
    with ObservationPrefetcher(windows) as inputs:
        with pytest.raises(ValueError, match='broken source 99'):
            inputs.take(inputs.submit([99],'snapshot'),[99])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA evaluator parity')
def test_prefetched_evaluation_matches_sequential():
    torch.set_num_threads(2)
    windows = Windows()
    model = OnsetModel('snapshot',activation_checkpoint=False).cuda()
    cond = torch.zeros(16,7,device='cuda')
    event = torch.arange(16,device='cuda') % 7
    indices = np.array([5,0,1,3,8])
    expected = sequential_evaluate(model,windows,indices,cond,event,2)
    actual = evaluate(model,windows,indices,cond,event,2)
    assert actual[0] == pytest.approx(expected[0], abs=1e-6,rel=1e-5)
    for a,b in zip(actual[1:],expected[1:],strict=True):
        torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA producer/consumer handoff')
def test_drain_finishes_copy_stream_before_cache_handoff():
    windows = Windows()
    ready = torch.cuda.Event()
    with ObservationPrefetcher(windows) as inputs:
        def copy_work():
            with torch.cuda.stream(inputs.stream):
                torch.cuda._sleep(10000000)
                ready.record()
        inputs.worker.submit(copy_work)
        inputs.drain()
        assert ready.query()
