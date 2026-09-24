"""Reuse the verified native paired graphs and frozen random-MACE reservoir."""
import json
import math
from pathlib import Path

import numpy as np

from src.data.structural_pretraining.prepare import file_hash
from src.project_runtime.paths import resolve_path
from src.training_methods.neighborhood_jepa.regularization.data import Data as NativeData


PAIR_SPEC = dict(regularizer='epi', neighbors=6, future_weight=0.,
                 family_weights=dict(present_neighbors=0., future_neighbors=0., future_center=0.))


class Data(NativeData):
    def __init__(self, config):
        super().__init__(config, PAIR_SPEC)
        if self.manifest['identity'] != config['data_identity']:
            raise ValueError('Native Al data identity changed')
        if self.plan.views != ((1, 0), (2, 0)) or self.manifest['lag_ps'] != .75:
            raise ValueError('Require same-center current/next views at 0.75 ps')
        if self.train_size != 32768 or len(self.selection) != 480:
            raise ValueError('Expected the declared 32768 train / 480 development anchors')
        path = resolve_path(config['order_cache'])/'reservoir.json'
        if file_hash(path) != config['reservoir_manifest_sha256']:
            raise ValueError('Frozen random reservoir manifest changed')
        receipt = json.loads(path.read_text())
        for shard, checksum in receipt['files'].items():
            if file_hash(self.extra/'reservoir'/f'{shard}.npy') != checksum:
                raise ValueError(f'Frozen random reservoir changed: {shard}')
        # The inherited loader includes physical/order metadata, but the objective
        # consumes only index and reservoir. No physical targets enter optimization.


class PassBatches:
    """Exactly one shuffled exposure per training anchor per complete pass."""
    def __init__(self, indices, size, seed, start, stop):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.size, self.seed, self.start, self.stop = size, seed, start, stop
        self.steps_per_pass = math.ceil(len(self.indices)/size)
        if len(self.indices) % size == 1:
            raise ValueError('A singleton last batch cannot define VICReg/Epi statistics')

    def __iter__(self):
        previous, order = None, None
        for step in range(self.start, self.stop):
            epoch, batch = divmod(step, self.steps_per_pass)
            if epoch != previous:
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch]))
                order = rng.permutation(self.indices)
                previous = epoch
            yield order[batch*self.size:(batch+1)*self.size].tolist()

    def __len__(self):
        return self.stop-self.start
