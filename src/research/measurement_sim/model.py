"""Stochastic evolution of all244 measured coordinates; exact observed anchor."""
import math
import torch
from torch import nn
from src.research.ni_latent_sim.transformer import TemporalTransformer


class DirectProcess(nn.Module):
    def __init__(self, c, rate):
        super().__init__()
        self.d = 244; self.memory = 4
        self.dt = c['model']['step_fs']/1000
        self.substeps = 100//c['model']['step_fs']
        self.transition = TemporalTransformer(self.d, self.memory, c['model'])
        nn.init.zeros_(self.transition.head.weight); nn.init.zeros_(self.transition.head.bias)
        self.register_buffer('decay', torch.exp(-rate*self.dt))
        self.register_buffer('noise_scale', (1-self.decay.square()).sqrt())
        self.register_buffer('active', (rate > 0).float())

    def interval(self, history, noises, dense=False):
        current = history[..., -1, :]
        values = []
        for noise in noises:
            tokens = torch.cat((history[..., :-1, :], current.unsqueeze(-2)), -2)
            drift, diffusion = self.transition(tokens.flatten(-2)).float().chunk(2, -1)
            # Stable exact diagonal-OU prior + bounded learned residuals. The
            # state remains244D FP32; BF16 only applies to neural matrix work.
            current = self.decay*current + self.active*self.dt*20*torch.tanh(drift/20)
            current = current+self.noise_scale*torch.exp(2*torch.tanh(diffusion/2))*noise
            if dense: values.append(current)
        history = torch.cat((history[..., 1:, :], current.unsqueeze(-2)), -2)
        return history, torch.stack(values, 2) if dense else current

    def forward(self, history, intervals, particles, seed, *, dense=False):
        generator = torch.Generator(device=history.device).manual_seed(seed)
        memory = history[:, None].expand(-1, particles, -1, -1)
        outputs = []
        for _ in range(intervals):
            noises = torch.randn(self.substeps, len(history), particles, self.d,
                                  generator=generator, device=history.device)
            memory, value = self.interval(memory, noises, dense)
            outputs.append(value)
        return torch.cat(outputs, 2) if dense else torch.stack(outputs, 2)


def energy(samples, target, weights):
    """Unbiased ensemble energy score on a joint path (not time marginals)."""
    def distance(delta):
        return (delta.square()*weights).sum(-1).mean(-1).clamp_min(1e-12).sqrt()
    first = distance(samples-target[:, None]).mean(1)
    flat = (samples*weights.sqrt()).flatten(2)/math.sqrt(samples.shape[2])
    pairwise = torch.cdist(flat, flat, compute_mode='donot_use_mm_for_euclid_dist')
    i, j = torch.tril_indices(samples.shape[1], samples.shape[1], offset=-1, device=samples.device)
    return first-.5*pairwise[:, i, j].clamp_min(1e-6).mean(1)


def predictive_scores(samples, target, initial, weights, increment_scale, bandwidth):
    # Composite likelihood of scalar Gaussian mixtures, family balanced. Joint
    # path dependence is scored below rather than selecting one of only four
    # particles by a244-dimensional narrow kernel. Observation bandwidth is fixed.
    logp = -.5*(((target[:, None]-samples)/bandwidth).square()
                +2*math.log(bandwidth)+math.log(2*math.pi))
    marginal = -(torch.logsumexp(logp, 1)-math.log(samples.shape[1]))
    nll = (marginal*weights).sum(-1).mean(-1)
    path = energy(samples, target, weights)
    truth = torch.cat((initial[:, None], target), 1)
    sampled = torch.cat((initial[:, None, None].expand(-1, samples.shape[1], -1, -1), samples), 2)
    dtrue = (truth[:, 1:]-truth[:, :-1])/increment_scale
    dpred = (sampled[:, :, 1:]-sampled[:, :, :-1])/increment_scale
    increments = energy(dpred, dtrue, weights)
    # Ordered lag products retain persistence/reversal of the *same* particle.
    actual, predicted = [], []
    for lag in [1, 5]:
        actual.append((dtrue[:, lag:]*dtrue[:, :-lag]).mean(1))
        predicted.append((dpred[:, :, lag:]*dpred[:, :, :-lag]).mean(2))
    correlations = energy(torch.stack(predicted, 2), torch.stack(actual, 1), weights)
    return dict(nll=nll, path_energy=path, increment_energy=increments, correlation_energy=correlations)


def feature_weights(scale):
    """Equal weight to BO, TDA, density, velocity; train-constant channels off."""
    weights = torch.zeros_like(scale)
    for start, stop in [(0, 40), (40, 153), (153, 227), (227, 244)]:
        active = (scale[start:stop] > 1e-4).float()
        weights[start:stop] = active/active.sum().clamp_min(1)/4
    return weights
