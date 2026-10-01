"""Matched distance mixtures and a distance-only equivariant context readout."""

import math
import torch
from torch import nn
from src.research.crystal_vector.model import distance_head, mixture_combiner, distance_parts
from src.research.crystal_vector.trunk import SpatialContextTrunk, TypedPatchTrunk
from .rich_descriptor_head import NormalizedResidualHead
from src.research.spatial_distance.model import parameters


class DescriptorMixture(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.register_buffer('mean', torch.zeros(224))
        self.register_buffer('scale', torch.ones(224))
        if kind == 'prior':
            self.raw = nn.Parameter(torch.zeros(25, 4))
        elif kind == 'linear':
            self.net = nn.Linear(224, 100)
        elif kind == 'mlp':
            self.net = nn.Sequential(
                nn.Linear(224, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 100)
            )
        else:
            raise ValueError(kind)

    def forward(self, batch):
        n = len(batch['distance'])
        raw = (
            self.raw[None].expand(n, -1, -1)
            if self.kind == 'prior'
            else self.net((batch['features'] - self.mean) / self.scale).reshape(n, 25, 4)
        )
        return dict(parts=parameters(raw[..., :3].float(), raw[..., 3].float()))


class DistanceMACE(SpatialContextTrunk):
    architecture = 'distance_mace_v2'

    def __init__(self, encoder_config, config):
        super().__init__(encoder_config, config)
        self.distance_head = distance_head(config['predictor']['width'])
        self.combine = mixture_combiner()

    def forward(self, batch):
        if 'z' in batch:
            scalar, vector = batch['z'], batch['v']
        else:
            scalar, vector = self.encode(batch['positions'])
            scalar = scalar[batch['inverse']]
            vector = vector[batch['inverse']]
        state, _ = self.context(scalar, vector, batch['actual'])
        parts = distance_parts(self.distance_head, self.combine, state, batch['actual'])
        return dict(parts=parts, z=scalar.float(), v=vector.float(), state=state.float())


class ControlMACE(SpatialContextTrunk):
    architecture = 'control_mace_v2'

    def __init__(self, enc, c, outputs):
        super().__init__(enc, c)
        # The only decoder input is one exported context state. Every patch
        # shares the same MACE, and no target descriptor is an input feature.
        self.context_export = nn.Sequential(
            nn.LayerNorm(c['predictor']['width']),
            nn.Linear(c['predictor']['width'], self.latent_dim),
            nn.SiLU(),
        )
        self.readout = nn.Sequential(
            nn.Linear(self.latent_dim, 2 * self.latent_dim),
            nn.SiLU(),
            nn.Linear(2 * self.latent_dim, outputs),
        )
        self.register_buffer('output_mask', torch.ones(outputs))

    def forward(self, batch):
        z, v = self.encode(batch['positions'])
        z = z[batch['inverse']]
        v = v[batch['inverse']]
        s, _ = self.context(z, v, batch['actual'])
        state = self.context_export(s.mean(1))
        return dict(
            prediction=self.readout(state).float() * self.output_mask,
            z=z.float(),
            v=v.float(),
            state=state.float(),
        )


class RichPatchMACE(TypedPatchTrunk):
    architecture = 'rich_patch_mace_objectives_v2'

    def __init__(self, c, outputs):
        super().__init__(c['encoder_config'], c, vector_channels=c['vector_channels'])
        self.register_buffer('output_mask', torch.ones(outputs))
        from .rich_objectives import StructuredDescriptorHead
        head = (StructuredDescriptorHead if c['descriptor_head']['kind'] == 'structured_residual_v1'
                else NormalizedResidualHead)
        self.readout = head(self.latent_dim, outputs, c['descriptor_head'])
        # Always construct these after the encoder/head, so all VCReg arms have
        # identical shared initialization and capacity; only placement changes.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(c['seed'] + 53)
            self.regularization_projector = nn.Sequential(
                nn.Linear(self.latent_dim, 512), nn.LayerNorm(512), nn.SiLU(),
                nn.Linear(512, self.latent_dim))
            self.vector_regularization_projector = nn.Linear(c['vector_channels'], c['vector_channels'], bias=False)

    def forward(self, positions):
        z, v = self.encode(positions)
        result = dict(
            prediction=self.readout(z).float() * self.output_mask,
            state=z.float(),
            z=z[:, None].float(),
            v=v[:, None].float(),
        )
        with torch.autocast(device_type=z.device.type, enabled=False):
            if self.config['regularization']['placement'] == 'projector':
                result['regularization_z'] = self.regularization_projector(z.float())[:, None]
                result['regularization_v'] = self.vector_regularization_projector(
                    v.float().transpose(-1, -2)).transpose(-1, -2)[:, None]
            elif self.config['regularization']['placement'] == 'embedding':
                result['regularization_z'], result['regularization_v'] = result['z'], result['v']
            else:
                raise ValueError(self.config['regularization']['placement'])
        return result


def initialize_descriptor(model, features, distance, weights):
    import numpy as np

    if model.kind != 'prior':
        mean = weights @ features
        std = np.sqrt(weights @ (features - mean) ** 2).clip(1e-4)
        model.mean.copy_(torch.as_tensor(mean, device=model.mean.device))
        model.scale.copy_(torch.as_tensor(std, device=model.scale.device))
    order = np.argsort(distance)
    quant = np.interp(np.linspace(0.02, 0.98, 25), np.cumsum(weights[order]), distance[order])
    raw = torch.zeros((25, 4), device=model.mean.device)
    raw[:, 0] = -12
    raw[:, 1] = 4 * torch.atanh(
        torch.as_tensor((np.log(quant) - math.log(16)) / 4, device=raw.device).clamp(-0.99, 0.99)
    )
    raw[:, 2] = math.log((0.3 - 0.15) / (2 - 0.3))
    with torch.no_grad():
        if model.kind == 'prior':
            model.raw.copy_(raw)
        else:
            last = model.net if model.kind == 'linear' else model.net[-1]
            last.weight.mul_(0.01)
            last.bias.copy_(raw.flatten())
