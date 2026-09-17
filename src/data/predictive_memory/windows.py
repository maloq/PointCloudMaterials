"""Matched windows from an audited immutable release; no outcome-label store."""
from functools import lru_cache
import json
from pathlib import Path
import torch
from .observations import assemble
from .prepare import file_hash


class MemoryDataset:
    def __init__(self, config, history_ps):
        self.config, self.history_ps = config, history_ps
        root = Path(config['cache'])
        complete = json.loads((root/'complete.json').read_text())
        if complete['state'] != 'complete' or file_hash(root/'dataset_release.json') != complete['release_sha256']:
            raise RuntimeError('Incomplete or modified predictive-memory release')
        self.release_sha256 = complete['release_sha256']
        self.release = json.loads((root/'dataset_release.json').read_text())
        if history_ps not in (0., 12., 48.):
            raise ValueError('Pilot supports exact full-resolution H=0,12,48 ps')
        for name in ('radius_A', 'cutoff_A', 'future_lags_ps', 'anchor_frames', 'protocol'):
            if config[name] != self.release['config'][name]:
                raise ValueError(f'Dataset configuration mismatch: {name}')
        self.shards, self.rows = [], []
        for source in self.release['sources']:
            path = root/f"source-{source['trajectory_id']:04d}.pt"
            if file_hash(path) != source['shard_sha256']:
                raise RuntimeError(f'Modified source cache: {path}')
            shard = torch.load(path, weights_only=True, mmap=True)
            index = len(self.shards)
            self.shards.append(shard)
            for sample in shard['samples']:
                self.rows.append(dict(source_index=index, source_id=shard['source_id'], split=shard['split'],
                    lineage=shard['lineage'], center_id=shard['center_id'], temperature_K=shard['temperature_K'], **sample))
        self.indices = {split: [i for i, r in enumerate(self.rows) if r['split'] == split]
                        for split in ('train', 'val', 'test')}
        if any(not values for values in self.indices.values()):
            raise ValueError('Require nonempty inherited train/val/test splits')

    @lru_cache(maxsize=450)
    def observation(self, index):
        row = self.rows[index]
        shard = self.shards[row['source_index']]
        anchor = row['anchor']
        frames = list(range(anchor-int(self.history_ps/.75), anchor+1))
        return assemble([shard['frames'][f] for f in frames], [shard['times_ps'][f] for f in frames],
                        radius=self.config['radius_A'], cutoff=self.config['cutoff_A'])

    def targets(self, indices, device):
        present = torch.stack([self.rows[i]['present'] for i in indices]).to(device)
        future = torch.stack([self.rows[i]['future'] for i in indices]).to(device)
        condition = torch.tensor([[self.rows[i]['temperature_K']/500.] for i in indices], device=device)
        return present, future, condition
