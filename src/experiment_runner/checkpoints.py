"""Atomic training state with the producer's explicit checkpoint fields."""

from dataclasses import dataclass
from pathlib import Path
import os

import torch


@dataclass
class TrainingState:
    payload: dict

    @classmethod
    def capture(cls, model, optimizer, sampler, *, identity,
                capture_torch_rng=False, **fields):
        payload = dict(fields, identity=identity, model=model.state_dict(),
                       optimizer=optimizer.state_dict(), rng=sampler.bit_generator.state)
        if capture_torch_rng:
            payload.update(torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
        return cls(payload)

    @classmethod
    def read(cls, path, *, identity, device):
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload['identity'] != identity:
            raise ValueError(f'Training checkpoint identity changed: {path}')
        return cls(payload)

    def restore(self, model, optimizer, sampler, *, restore_torch_rng=False):
        model.load_state_dict(self.payload['model'])
        optimizer.load_state_dict(self.payload['optimizer'])
        sampler.bit_generator.state = self.payload['rng']
        if restore_torch_rng:
            torch.set_rng_state(self.payload['torch_rng'].cpu())
            torch.cuda.set_rng_state(self.payload['cuda_rng'].cpu())

    def save(self, path):
        path = Path(path)
        temporary = path.with_name(f'.{path.name}.{os.getpid()}.building')
        torch.save(self.payload, temporary)
        temporary.replace(path)
