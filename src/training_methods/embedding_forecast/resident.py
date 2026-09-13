"""Keep stored embedding shards on the compute device; gather identical windows."""

import numpy as np
import torch
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler


class ResidentWindows:
    """The repository producer gives every segment the same number of frames."""

    def __init__(self, dataset, device):
        self.source = dataset
        frames = dataset.records[0]['frames']
        if any(r['frames'] != frames for r in dataset.records):
            raise ValueError('Resident windows require equal-length producer segments within a split.')
        self.windows = int(dataset.windows[0])
        centers = sum(r['centers'] for r in dataset.records)
        first = dataset.arrays(0)['embeddings']
        dtype = torch.from_numpy(np.empty(0, dtype=first.dtype)).dtype
        self.embeddings = torch.empty((centers, frames, dataset.dim), dtype=dtype, device=device)
        self.frames = np.stack([dataset.arrays(i)['frames'] for i in range(len(dataset.records))])
        self.atoms = np.concatenate([dataset.arrays(i)['atom_ids'] for i in range(len(dataset.records))])
        self.shards = np.repeat(np.arange(len(dataset.records)), [r['centers'] for r in dataset.records])
        self.sources = np.array([r['source_index'] for r in dataset.records], dtype=np.int64)[self.shards]
        self.temperatures = np.array([r['temperature_K'] for r in dataset.records], dtype=np.float32)[self.shards]
        start = 0
        for index, record in enumerate(dataset.records):
            stop = start + record['centers']
            # Own one host shard while copying; never mutate the read-only mmap.
            values = np.array(dataset.arrays(index)['embeddings'], copy=True)
            self.embeddings[start:stop].copy_(torch.from_numpy(values))
            start = stop
        self.offsets = torch.arange(dataset.history_steps + dataset.future_steps, device=device)
        print(f'Resident {dataset.records[0]["split"]} embeddings: '
              f'{self.embeddings.numel() * self.embeddings.element_size() / 2**30:.3f} GiB on {device}', flush=True)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, indices):
        rows = np.asarray(indices, dtype=np.int64)
        centers, windows = np.divmod(rows, self.windows)
        starts = self.source.history_skip + windows * self.source.stride
        device_rows = torch.from_numpy(rows).to(self.embeddings.device)
        device_centers = device_rows // self.windows
        columns = (self.source.history_skip + (device_rows % self.windows)[:, None] *
                   self.source.stride + self.offsets)
        # The existing mmap loader expands stored float16 to float32 before normalization.
        values = self.embeddings[device_centers[:, None], columns].float()
        anchor = starts + self.source.history_steps - 1
        return dict(history=values[:, :self.source.history_steps],
                    future=values[:, self.source.history_steps:],
                    source=torch.from_numpy(self.sources[centers]),
                    atom_id=torch.from_numpy(self.atoms[centers]),
                    anchor_frame=torch.from_numpy(self.frames[self.shards[centers], anchor]),
                    temperature_K=torch.from_numpy(self.temperatures[centers]))


class ResidentLoader:
    """Retain the original CPU RandomSampler and its exact checkpointed RNG state."""

    def __init__(self, dataset, batch_size, shuffle, seed, device):
        self.dataset = ResidentWindows(dataset, device)
        generator = torch.Generator().manual_seed(seed)
        sampler = RandomSampler(dataset, generator=generator) if shuffle else SequentialSampler(dataset)
        self.sampler = BatchSampler(sampler, batch_size, drop_last=False)

    def __iter__(self):
        for indices in self.sampler:
            yield self.dataset[indices]

    def __len__(self):
        return len(self.sampler)
