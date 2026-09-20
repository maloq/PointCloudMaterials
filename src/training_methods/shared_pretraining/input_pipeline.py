"""Ordered, bounded process prefetch for the mixed snapshot protocol.

Workers build the exact same sampled batches without touching CUDA. The parent
pins completed batches on a separate thread. Step-derived sampling makes worker
completion order and discarded lookahead irrelevant to checkpoint continuation.
"""
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from pathlib import Path
import resource
from threading import Lock

import numpy as np
import torch

from src.data.structural_pretraining.batches import Release
from .mixed import prepare


class ShardArrays(dict):
    def __init__(self, root):
        super().__init__()
        self.root = Path(root)

    def __missing__(self, name):
        arrays = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False)
                  for p in (self.root/'shards'/name).glob('*.npy')}
        if not arrays:
            raise FileNotFoundError(f'Missing structural shard: {self.root}/shards/{name}')
        self[name] = arrays
        return arrays


class PackingRelease(Release):
    """Use the parent's exact row index; lazily open arrays without refitting moments."""
    def __init__(self, root, rows, groups, group_keys, group_weights):
        self.root = Path(root)
        self.rows, self.groups = rows, groups
        self.group_keys, self.group_weights = group_keys, group_weights
        self.arrays = ShardArrays(root)
        self.graphs = OrderedDict()
        self.graph_bytes = 0
        self.max_graph_bytes = 256*2**20
        self.graph_lock = Lock()


_release = None
_config = None


def initialize_worker(spec, config):
    global _release, _config
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 65536), hard))
    torch.set_num_threads(1)
    _release = PackingRelease(*spec)
    _config = dict(config, preparation_workers=1)


def prepare_step(step):
    return prepare(_release, step, _config, pin_memory=False)


class ProcessPrefetch:
    def __init__(self, release, config, start, total, *, pin_memory=True):
        self.depth = config['prefetch_batches']
        workers = config['preparation_processes']
        if workers < 1 or self.depth < workers:
            raise ValueError('Process prefetch requires positive workers and depth >= workers')
        self.total, self.next_step, self.pin_memory = total, start, pin_memory
        spec = (str(release.root), release.rows, release.groups, release.group_keys, release.group_weights)
        # Spawn explicitly: the main process already owns CUDA contexts.
        self.pool = ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_worker, initargs=(spec, config))
        self.pin_pool = ThreadPoolExecutor(max_workers=1)
        self.pending = {step: self.pool.submit(prepare_step, step)
                        for step in range(start, min(start+self.depth, total))}
        self.ready = self.pin_pool.submit(self._receive, start) if start < total else None

    def _receive(self, step):
        batches, *rest = self.pending[step].result()
        if self.pin_memory:
            batches = [{k: v.pin_memory() for k, v in batch.items()} for batch in batches]
        return (batches, *rest)

    def take(self, step):
        if step != self.next_step or step >= self.total:
            raise ValueError(f'Expected prefetched update {self.next_step}, received {step}')
        result = self.ready.result()
        del self.pending[step]
        following = step+self.depth
        if following < self.total:
            self.pending[following] = self.pool.submit(prepare_step, following)
        self.next_step += 1
        self.ready = (self.pin_pool.submit(self._receive, self.next_step)
                      if self.next_step < self.total else None)
        return result

    def close(self):
        self.pin_pool.shutdown(wait=True, cancel_futures=True)
        self.pool.shutdown(wait=True, cancel_futures=True)
