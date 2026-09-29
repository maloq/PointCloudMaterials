"""End-to-end typed MACE and shared vector-message localization mixtures."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.research.equivariant_context.model import radial_basis
from .trunk import SpatialContextTrunk
from src.research.spatial_distance.model import parameters, component_cdf
from src.research.distance_encoder.model import loss_terms


def distance_head(width):
    return nn.Sequential(
        nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 3)
    )


def mixture_combiner():
    combine = nn.Sequential(nn.Linear(14, 64), nn.SiLU(), nn.Linear(64, 1))
    nn.init.zeros_(combine[-1].weight)
    nn.init.zeros_(combine[-1].bias)
    return combine


def distance_parts(head, combine, state, actual):
    raw = head(state).float()
    provisional = parameters(raw, torch.zeros_like(raw[..., 0]))
    risk = component_cdf(provisional, raw.new_tensor([4, 8, 12, 20, 32, 64]))
    offsets = radial_basis(actual.norm(dim=-1), 28.0)
    scores = combine(torch.cat((risk, offsets), -1)).squeeze(-1).float()
    return parameters(raw, scores)


class JointCrystalVector(SpatialContextTrunk):
    architecture = 'joint_crystal_vector_v1'

    def __init__(self, encoder_config, config):
        super().__init__(encoder_config, config)
        width = config['predictor']['width']
        # Keep the original order and flat names for historical joint checkpoints.
        self.distance_head = distance_head(width)
        self.direction_channels = nn.Linear(self.layout.vectors, 1, bias=False)
        self.direction_offset = nn.Linear(width, 1)
        self.combine = mixture_combiner()

    def forward(self, batch):
        scalar, vector = self.encode(batch['positions'])
        scalar = scalar[batch['inverse']]
        vector = vector[batch['inverse']]
        state, fields = self.context(scalar, vector, batch['actual'])
        parts = distance_parts(self.distance_head, self.combine, state, batch['actual'])
        direction = self.direction_channels(fields[1].transpose(-1, -2)).squeeze(-1)
        direction = direction - self.direction_offset(state) * batch['actual'] / 24.0
        # Norm below one permits a uniform spherical density at zero.
        direction = direction.float() / torch.sqrt(
            direction.float().square().sum(-1, keepdim=True) + 1e-8
        )
        return dict(
            parts=parts,
            direction=direction,
            z=scalar.float(),
            v=vector.float(),
            state=state.float(),
        )


def component_log_distance(parts, distance, cap):
    _, zero, mu, sigma = parts
    d = distance.clamp(1e-6, cap)[:, None]
    positive = F.logsigmoid(-zero)
    density = (
        positive
        - d.log()
        - sigma.log()
        - 0.5 * math.log(2 * math.pi)
        - 0.5 * ((d.log() - mu) / sigma).square()
    )
    survival = positive + torch.special.log_ndtr((mu - math.log(cap)) / sigma)
    return torch.where(
        distance[:, None] == 0,
        F.logsigmoid(zero),
        torch.where(distance[:, None] >= cap, survival, density),
    )


def directional_log_density(direction, target, distance, config):
    k = config['direction_kappa'] / (
        1 + (distance.clamp(max=config['distance_cap']) / config['direction_scale_A']).square()
    )
    eta = direction * k[:, None, None]
    k2 = eta.square().sum(-1)
    norm = k2.clamp_min(1e-12).sqrt()
    # log(sinh(k)/k), with its analytic small-k limit and finite unused branches.
    safe = norm.clamp_min(0.01)
    log_sinhc = torch.where(
        k2 < 1e-4, k2 / 6 - k2.square() / 180, torch.sinh(safe).log() - safe.log()
    )
    return (eta * target[:, None]).sum(-1) - math.log(4 * math.pi) - log_sinhc


def objective(output, batch, config, directional):
    parts = output['parts']
    d = batch['distance']
    total, nll, early = loss_terms(parts, d, config)
    log_d = component_log_distance(parts, d, config['distance_cap'])
    log_joint = torch.logsumexp(
        parts[0]
        + log_d
        + directional_log_density(output['direction'], batch['direction'], d, config),
        1,
    )
    angular = torch.where(batch['valid'], -log_joint - nll, torch.zeros_like(nll))
    return dict(
        objective=total + config['direction_weight'] * angular if directional else total,
        distance_nll=nll,
        direction_nll=angular,
        proximity_log_loss=early,
    )


def vcreg(output, config):
    from .parallel import global_covariance

    # Independent target-population draws: every query has equal mass. Each has
    # 25 patches; average their sufficient statistics without claiming independence.
    terms = []
    stats = {}
    for name, field in [('scalar', output['z']), ('vector', output['v'])]:
        x = field.flatten(0, 1)
        if name == 'scalar':
            x = x[..., None]
        covariance = global_covariance(x)
        std = (covariance.diagonal() + config['epsilon']).sqrt()
        variance = F.relu(config['std_floor'] - std).mean()
        off = covariance - torch.diag_embed(covariance.diagonal())
        redundancy = off.square().sum() / (len(off) * (len(off) - 1))
        terms.append(
            config['variance_weight'] * variance + config['covariance_weight'] * redundancy
        )
        stats[f'{name}_variance_penalty'] = variance.detach()
        stats[f'{name}_covariance_penalty'] = redundancy.detach()
        stats[f'{name}_minimum_std'] = std.min().detach()
    return sum(terms), stats
