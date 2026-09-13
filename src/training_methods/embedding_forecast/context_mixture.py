"""Spatially conditioned forecasts and finite mixtures of entire future paths."""

import math
import torch
from torch import nn
from torch.nn import functional as F

from .model import EmbeddingForecaster


class ContextMixtureForecaster(EmbeddingForecaster):
    """One mixture component is shared across all future times and embedding channels."""

    def __init__(self, dim, history_steps, cadence_ps, horizons_ps, config):
        base = dict(config, distribution='deterministic')
        super().__init__(dim, history_steps, cadence_ps, horizons_ps, base)
        if config['architecture'] != 'mean_residual_gru' or config['target'] != 'trajectory':
            raise ValueError('Context/mixture experiments use the direct mean-residual GRU full-path decoder.')
        self.config = config
        self.distribution = config['distribution']
        self.spatial_neighbors = config['spatial_neighbors']
        if self.spatial_neighbors:
            self.spatial_input = nn.Linear(dim+2, config['width'], bias=False)
            nn.init.normal_(self.spatial_input.weight, std=.001)
        if self.distribution == 'trajectory_mixture':
            self.components = config['mixture_components']
            if self.components < 1 or config['minimum_std'] <= 0:
                raise ValueError('Trajectory mixtures require at least one component and a positive scale floor.')
            self.component_head = nn.Linear(config['width'], self.components*dim)
            self.scale_head = nn.Linear(config['width'], self.components*dim)
            self.gate = nn.Linear(config['width'], self.components)
            nn.init.normal_(self.component_head.weight, std=.002)
            nn.init.zeros_(self.component_head.bias)
            nn.init.normal_(self.scale_head.weight, std=.001)
            nn.init.constant_(self.scale_head.bias, math.log(math.expm1(.5)))
        elif self.distribution != 'deterministic':
            raise ValueError(f'Unsupported context model distribution: {self.distribution}')

    def forward(self, history, spatial=None, spatial_radii_A=None):
        if not self.spatial_neighbors and self.distribution == 'deterministic':
            return super().forward(history)
        if self.history_mode != 'real':
            raise ValueError('Context/mixture pilots require the explicit observed history.')
        delta = torch.cat((torch.zeros_like(history[:, :1]), history[:, 1:]-history[:, :-1]), 1)
        time = self.past_time[None, :, None].expand(len(history), -1, -1)
        tokens = self.input(torch.cat((history, delta, time), -1))
        if self.spatial_neighbors:
            if spatial is None or spatial_radii_A is None:
                raise ValueError('Spatial model needs same-frame neighbor means and radii in angstrom.')
            tokens = tokens+self.spatial_input(torch.cat((spatial-history, spatial_radii_A/self.config['spatial_distance_unit_A']), -1))
        _, hidden = self.history(tokens)
        context = hidden[-1]
        decoded = self.decoder(torch.cat((context[:, None].expand(-1, self.output_steps, -1),
                         self.future_time[None].expand(len(history), -1, -1)), -1))
        mean = history.mean(1, keepdim=True)+self.mean_head(decoded)
        if self.distribution == 'deterministic':
            return {'mean': mean}
        shape = (len(history), self.output_steps, self.components, self.dim)
        component_means = mean[:, None]+self.component_head(decoded).reshape(shape).permute(0, 2, 1, 3)
        std = (F.softplus(self.scale_head(decoded))+self.config['minimum_std']).reshape(shape).permute(0, 2, 1, 3)
        logits = self.gate(context)
        probability = logits.softmax(-1)
        return dict(mean=(component_means*probability[:, :, None, None]).sum(1),
                    component_means=component_means, component_std=std, mixture_logits=logits)


def build_forecaster(dim, history_steps, cadence_ps, horizons_ps, variant):
    cls = ContextMixtureForecaster if 'spatial_neighbors' in variant else EmbeddingForecaster
    return cls(dim, history_steps, cadence_ps, horizons_ps, variant)


def call_forecaster(model, history, batch, mean, scale):
    if isinstance(model, ContextMixtureForecaster) and model.spatial_neighbors:
        spatial = (batch['spatial'].to(history.device)-mean)/scale
        return model(history, spatial, batch['spatial_radii_A'].to(history.device))
    return model(history)


def component_log_prob(output, target):
    residual = (target[:, None]-output['component_means'])/output['component_std']
    return (-.5*residual.square()-output['component_std'].log()-.5*math.log(2*math.pi)).sum((-2, -1))


def mixture_nll(output, target):
    # Sum a complete component path likelihood BEFORE mixing; do not mix each frame.
    joint = component_log_prob(output, target)+output['mixture_logits'].log_softmax(-1)
    return -torch.logsumexp(joint, dim=-1)/target[0].numel()


def sample_trajectories(output, samples):
    probability = output['mixture_logits'].softmax(-1)
    selected = torch.multinomial(probability, samples, replacement=True).T
    batch = torch.arange(len(probability), device=probability.device)[None]
    means = output['component_means'][batch, selected]
    std = output['component_std'][batch, selected]
    return means+std*torch.randn_like(means)


def normal_absolute_mean(delta, std):
    z = delta/std
    return std*math.sqrt(2/math.pi)*torch.exp(-.5*z.square())+delta*torch.erf(z/math.sqrt(2))


def mixture_metrics(output, target, sample_paths):
    means, std = output['component_means'], output['component_std']
    probability = output['mixture_logits'].softmax(-1)
    residual = (target[:, None]-means)/std
    cdf = (.5*(1+torch.erf(residual/math.sqrt(2)))*probability[:, :, None, None]).sum(1)
    crps = (normal_absolute_mean(target[:, None]-means, std)*probability[:, :, None, None]).sum(1)
    for i in range(means.shape[1]):
        for j in range(means.shape[1]):
            crps -= .5*probability[:, i, None, None]*probability[:, j, None, None]*normal_absolute_mean(
                means[:, i]-means[:, j], torch.sqrt(std[:, i].square()+std[:, j].square()))
    posterior = (component_log_prob(output, target)+output['mixture_logits'].log_softmax(-1)).softmax(-1)
    entropy = -(probability*output['mixture_logits'].log_softmax(-1)).sum(-1)
    result = dict(nll=mixture_nll(output, target), marginal_crps=crps.mean((1, 2)),
                  coverage90=((cdf >= .05)&(cdf <= .95)).float().mean((1, 2)),
                  mixture_entropy=entropy, mixture_effective_components=entropy.exp(),
                  component_weight=probability, component_responsibility=posterior)
    if sample_paths:
        samples = sample_trajectories(output, 16)
        result['energy_score'] = ((samples-target[None]).flatten(2).norm(dim=-1).mean(0)-
                                  .5*(samples[:8]-samples[8:]).flatten(2).norm(dim=-1).mean(0))/math.sqrt(target[0].numel())
    return result


def crystal_probability(output, projected_weight, projected_bias):
    """Marginal P(linear crystal margin >= 0) at each future frame, not event probability."""
    mu = output['component_means'].double()@projected_weight+projected_bias
    variance = output['component_std'].double().square()@projected_weight.square()
    cdf = .5*(1+torch.erf(mu/torch.sqrt(2*variance)))
    return (cdf*output['mixture_logits'].double().softmax(-1)[:, :, None]).sum(1)
