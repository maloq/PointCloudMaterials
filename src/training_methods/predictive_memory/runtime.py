"""Immutable observation residency and batched targets for the memory trainer."""
from collections import Counter, OrderedDict
import math

import torch


def batch_settings(training):
    batch = training['batch_size']
    micro = training.get('micro_batch_size', batch)
    evaluation = training.get('evaluation_batch_size', micro)
    for name, value in [('batch_size', batch), ('micro_batch_size', micro), ('evaluation_batch_size', evaluation)]:
        if type(value) is not int or value < 1:
            raise ValueError(f'training.{name} must be a positive integer, got {value!r}')
    if micro > batch:
        raise ValueError('micro_batch_size cannot exceed the effective batch_size')
    return batch, micro, evaluation


def sample_batch(sampler, indices, batch_size):
    return sampler.choice(indices, size=batch_size, replace=True).tolist()


def observation_bytes(observation):
    # Expanded views, e.g. repeated histories, can share underlying storage.
    tensors = [observation.positions, observation.velocities, observation.weights,
               observation.offsets_ps, observation.atom_ids, *observation.edges]
    storage = {tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes() for tensor in tensors}
    return sum(storage.values())


class MemoryRuntime:
    def __init__(self, dataset, device, settings=None):
        settings = {} if settings is None else settings
        unknown = set(settings)-{'observation_cache_gib', 'cache_scope', 'torch_threads'}
        if unknown:
            raise ValueError(f'Unknown memory runtime settings: {sorted(unknown)}')
        gib = settings.get('observation_cache_gib', 0.)
        if not isinstance(gib, (int, float)) or not math.isfinite(gib) or gib < 0:
            raise ValueError('observation_cache_gib must be finite and nonnegative')
        self.scope = settings.get('cache_scope', 'train')
        if self.scope not in ('train', 'all'):
            raise ValueError('cache_scope must be train or all')
        self.dataset, self.device = dataset, torch.device(device)
        self.maximum_bytes = int(gib*2**30)
        self.cache = OrderedDict()
        self.bytes = self.hits = self.misses = 0
        counts = Counter(dataset.rows[i]['source_id'] for i in dataset.indices['train'])
        if len(set(counts.values())) != 1:
            raise ValueError('Uniform memory-window sampling requires equal training windows per source')
        self.present, self.future, self.condition = dataset.targets(list(range(len(dataset.rows))), self.device)

    def normalize(self, normalizer):
        center, scale = normalizer
        self.standard_present = (self.present-center)/scale
        self.standard_future = (self.future-center)/scale

    def observation(self, index):
        if index in self.cache:
            self.hits += 1
            value = self.cache.pop(index)
            self.cache[index] = value
            return value[0]
        self.misses += 1
        value = self.dataset.observation(index).to(self.device)
        if self.maximum_bytes and (self.scope == 'all' or self.dataset.rows[index]['split'] == 'train'):
            size = observation_bytes(value)
            if size <= self.maximum_bytes:
                while self.cache and self.bytes+size > self.maximum_bytes:
                    _, (_, old_size) = self.cache.popitem(last=False)
                    self.bytes -= old_size
                self.cache[index] = (value, size)
                self.bytes += size
        return value

    def observations(self, indices):
        return [self.observation(i) for i in indices]

    def statistics(self):
        return dict(observation_cache_bytes=self.bytes, observation_cache_limit_bytes=self.maximum_bytes,
                    cache_entries=len(self.cache), cache_hits=self.hits, cache_misses=self.misses,
                    cache_scope=self.scope)
