"""Five-femtosecond evolution with the original 0.1-ps observation memory.

Past observation slots stay fixed inside a 0.1-ps interval. Only the present
code evolves; completed predictions replace history slots at native boundaries.
No interpolated input frames or future teacher resets are used.
"""
import math
import torch
from .model import Latent


class FineLatent(Latent):
    def __init__(self, arm):
        super().__init__(arm)
        transition_kind = arm.get('transition_kind', 'mlp')
        if transition_kind == 'transformer':
            from .transformer import TemporalTransformer
            self.transition = TemporalTransformer(self.d, self.memory, arm['transformer'])
        elif transition_kind != 'mlp':
            raise ValueError('Unknown latent transition architecture')
        self.step_fs = arm['neural_step_fs']
        if self.step_fs <= 0 or 100 % self.step_fs:
            raise ValueError('Neural timestep must exactly divide the native 100 fs interval')
        self.substeps = 100 // self.step_fs

    def microstep(self, past, current, noise):
        inputs = torch.cat((past, current.unsqueeze(-2)), -2)
        drift, raw = self.transition(inputs.flatten(-2)).chunk(2, -1)
        # The legacy noise amplitude describes a 0.1-ps step. Scale its
        # variance linearly with dt, while the drift scales linearly with dt.
        return (current + self.step_fs / 1000 * drift
                + math.sqrt(self.step_fs / 100) * torch.nn.functional.softplus(raw) * noise)

    def paths(self, x, intervals, particles, generator, *, dense):
        history = self.encode(x[:, -self.memory:])
        history = history[:, None].expand(-1, particles, -1, -1)
        outputs = []
        for _ in range(intervals):
            current = history[..., -1, :]
            noises = torch.randn(self.substeps, *current.shape, device=x.device, generator=generator)
            for j in range(self.substeps):
                current = self.microstep(history[..., :-1, :], current, noises[j])
                if dense or j == self.substeps - 1:
                    outputs.append(self.emit(current))
            history = torch.cat((history[..., 1:, :], current.unsqueeze(-2)), -2)
        return tuple(torch.stack(part, 2) for part in zip(*outputs))

    def rollout(self, x, steps, particles, generator):
        # Public steps remain native 0.1-ps intervals for the existing assay.
        return self.paths(x, steps, particles, generator, dense=False)

    def dense_rollout(self, x, particles, generator):
        return self.paths(x, 1, particles, generator, dense=True)
