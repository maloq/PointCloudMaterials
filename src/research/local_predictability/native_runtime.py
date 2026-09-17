"""One-batch lookahead of immutable observations, overlapping CPU work with CUDA."""
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import replace
import threading
import time

import numpy as np
import torch

from .metrics import hazard_loss, source_weights
from .native_data import BoundedCache


def peek_batch(sampler):
    """Read the next draw without advancing the checkpointed sampling stream."""
    state = copy.deepcopy(sampler.state_dict())
    try:
        return sampler.batch()
    finally:
        sampler.load_state_dict(state)


class ObservationPrefetcher:
    """CPU workers own separate frame caches; one dispatcher owns the GPU cache.

    The copy stream overlaps graph construction/transfers with current compute.
    Events and record_stream protect readiness and allocator lifetimes even when
    the bounded GPU cache evicts an observation still used by the current batch.
    No atom features, labels, gradients, or future-frame inputs are prefetched.
    """
    def __init__(self, windows, workers=3):
        if windows.device.type != 'cuda':
            raise ValueError('Native observation prefetch requires a CUDA loader')
        self.windows = windows
        self.stream = torch.cuda.Stream(device=windows.device)
        self.stream.wait_stream(torch.cuda.current_stream(windows.device))
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='atomic-inputs')
        self.cpu_workers = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='atomic-frames')
        self.local = threading.local()
        self.workers = workers
        self.wait_seconds = 0.

    def _cpu_observation(self, index, variant):
        if not hasattr(self.local, 'windows'):
            # Immutable cohort/row metadata is shared. Mutable LRU caches and
            # trajectory handles have exactly one CPU-worker owner each.
            window = copy.copy(self.windows)
            window.device = torch.device('cpu')
            window.raw = {}
            window.frames = BoundedCache(self.windows.frames.maximum_bytes // self.workers)
            window.observations = BoundedCache(0)
            self.local.windows = window
        observed = self.local.windows.observation(index, variant)
        return replace(observed, **{name: value.pin_memory() if isinstance(value, torch.Tensor)
                                    else tuple(t.pin_memory() for t in value)
                                    for name,value in vars(observed).items()
                                    if name not in ('radius_A', 'cutoff_A')})

    def _prepare(self, indices, variant):
        keys = [(self.windows.rows[i]['row_id'], variant) for i in indices]
        cached = [self.windows.observations.get(key) for key in keys]
        pending = {i: self.cpu_workers.submit(self._cpu_observation, index, variant)
                   for i,(index,value) in enumerate(zip(indices,cached,strict=True)) if value is None}
        with torch.cuda.device(self.windows.device), torch.cuda.stream(self.stream):
            for i,future in pending.items():
                observed = future.result()
                value = replace(observed, **{
                    name: tensor.to(self.windows.device, non_blocking=True) if isinstance(tensor,torch.Tensor)
                    else tuple(t.to(self.windows.device,non_blocking=True) for t in tensor)
                    for name,tensor in vars(observed).items() if name not in ('radius_A','cutoff_A')})
                cached[i] = self.windows.observations.put(keys[i], value)
            ready = torch.cuda.Event()
            ready.record(self.stream)
        return indices, cached, ready

    def submit(self, indices, variant):
        return self.worker.submit(self._prepare, list(map(int, indices)), variant)

    def take(self, future, indices):
        start = time.monotonic()
        actual, observations, ready = future.result()
        self.wait_seconds += time.monotonic()-start
        if actual != list(map(int, indices)):
            raise ValueError('Prefetched observation order differs from sampled row order')
        stream = torch.cuda.current_stream(self.windows.device)
        stream.wait_event(ready)
        for observation in observations:
            for value in vars(observation).values():
                if isinstance(value, torch.Tensor):
                    value.record_stream(stream)
                elif isinstance(value, tuple):
                    for tensor in value:
                        tensor.record_stream(stream)
        return observations

    def batches(self, indices, variant, size):
        if not len(indices):
            raise ValueError('Observation batches must be nonempty')
        future = self.submit(indices[:size], variant)
        for start in range(0, len(indices), size):
            batch = indices[start:start+size]
            observed = self.take(future, batch)
            if start+size < len(indices):
                future = self.submit(indices[start+size:start+2*size], variant)
            yield batch, observed

    def drain(self):
        # Finish loader/cache mutations before another evaluator uses the loader.
        self.worker.submit(lambda: None).result()
        self.stream.synchronize()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.worker.shutdown(wait=True, cancel_futures=True)
        self.cpu_workers.shutdown(wait=True, cancel_futures=True)
        self.stream.synchronize()


@torch.no_grad()
def evaluate(model, windows, indices, cond, event, microbatch):
    """The supervised evaluator's same rows, weights and losses with input lookahead."""
    model.eval()
    logits, states = [], []
    with ObservationPrefetcher(windows) as inputs:
        for batch, observations in inputs.batches(indices, model.encoder.variant, microbatch):
            result = model(observations, cond[batch])
            logits.append(result['logits'].cpu()); states.append(result['state'].cpu())
    logits, states = torch.cat(logits), torch.cat(states)
    losses = hazard_loss(logits, event[indices].cpu()).numpy()
    weights = source_weights(np.array([windows.rows[int(i)]['source_id'] for i in indices]))
    model.train()
    return float(weights@losses), logits, states
