"""Normalized residual readout for rich descriptors of one exported patch state."""
import math

import torch
from torch import nn


class NormalizedResidualHead(nn.Module):
    def __init__(self, inputs, outputs, config):
        super().__init__()
        if config['kind'] != 'normalized_residual_v1':
            raise ValueError(f'Unknown descriptor head: {config["kind"]}')
        if config['precision'] != 'float32':
            raise ValueError('The repaired descriptor head requires float32 computation')
        width, depth = config['width'], config['blocks']
        expanded = width * config['expansion']
        self.input_norm = nn.LayerNorm(inputs, elementwise_affine=False)
        self.skip = nn.Linear(inputs, outputs)
        self.stem = nn.Linear(inputs, width)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(width, elementwise_affine=False),
                nn.Linear(width, expanded),
                # Normalize immediately before SiLU so all units cannot drift
                # together into its negative saturated region.
                nn.LayerNorm(expanded, elementwise_affine=False),
                nn.SiLU(),
                nn.Linear(expanded, width),
            ) for _ in range(depth)
        ])
        self.residual_scale = 1 / math.sqrt(depth)
        self.output_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.output = nn.Linear(width, outputs)
        with torch.no_grad():
            for layer in (self.skip, self.output, *(block[-1] for block in self.blocks)):
                layer.weight.mul_(.01)
                layer.bias.zero_()

    def forward(self, state):
        # The small head stays in FP32; the expensive MACE encoder retains BF16.
        with torch.autocast(device_type=state.device.type, enabled=False):
            x = self.input_norm(state.float())
            h = self.stem(x)
            for block in self.blocks:
                h = h + self.residual_scale * block(h)
            return self.skip(x) + self.output(self.output_norm(h))
