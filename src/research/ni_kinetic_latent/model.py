"""Kinetic representations and an autoregressive stochastic latent propagator."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def mlp(inputs, width, outputs):
    return nn.Sequential(nn.Linear(inputs, width), nn.SiLU(),
                         nn.Linear(width, width), nn.SiLU(), nn.Linear(width, outputs))


def vamp_score(present, future, ridge):
    """Regularized nonreversible VAMP-2, divided by the latent dimension."""
    a = present.double() - present.double().mean(0)
    b = future.double() - future.double().mean(0)
    c00 = a.T @ a / len(a)
    c11 = b.T @ b / len(b)
    c01 = a.T @ b / len(a)
    eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
    left = torch.linalg.cholesky(c00 + ridge * eye)
    right = torch.linalg.cholesky(c11 + ridge * eye)
    white = torch.linalg.solve_triangular(left, c01, upper=False)
    white = torch.linalg.solve_triangular(right, white.T, upper=False).T
    return white.square().sum() / a.shape[-1]


class Transition(nn.Module):
    """Mixture density of standardized latent increments, with raw-latent API."""
    def __init__(self, dimension, width, mixtures):
        super().__init__()
        self.dimension, self.mixtures = dimension, mixtures
        self.network = mlp(4 * dimension, width, mixtures * (1 + 2 * dimension))
        self.register_buffer('location', torch.zeros(dimension))
        self.register_buffer('scale', torch.ones(dimension))

    def distribution(self, history):
        normalized = (history - self.location) / self.scale
        values = self.network(normalized.flatten(-2))
        logits = values[..., :self.mixtures]
        means, scales = values[..., self.mixtures:].reshape(
            *values.shape[:-1], self.mixtures, 2 * self.dimension).chunk(2, -1)
        return logits, means, F.softplus(scales) + .02

    def log_prob(self, history, following):
        logits, means, scales = self.distribution(history)
        delta = (following - history[..., -1, :]) / self.scale
        gaussian = -.5 * (((delta[..., None, :] - means) / scales).square()
                           + 2 * scales.log() + math.log(2 * math.pi)).sum(-1)
        return torch.logsumexp(logits.log_softmax(-1) + gaussian, -1)

    def sample(self, history, generator):
        logits, means, scales = self.distribution(history)
        component = torch.multinomial(logits.softmax(-1).reshape(-1, self.mixtures),
                                      1, generator=generator).flatten()
        index = torch.arange(len(component), device=history.device)
        mean = means.reshape(-1, self.mixtures, self.dimension)[index, component]
        scale = scales.reshape(-1, self.mixtures, self.dimension)[index, component]
        noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype,
                            generator=generator)
        delta = (mean + scale * noise).reshape(*history.shape[:-2], self.dimension)
        return history[..., -1, :] + self.scale * delta

    def mean(self, history):
        logits, means, _ = self.distribution(history)
        delta = (logits.softmax(-1)[..., None] * means).sum(-2)
        return history[..., -1, :] + self.scale * delta


class Model(nn.Module):
    """All encoders export the same raw 32-dimensional latent interface."""
    def __init__(self, c, kind, input_dimension=244):
        super().__init__()
        self.kind = kind
        width, dimension = c['model']['width'], c['model']['latent']
        self.dimension = dimension
        self.input_dimension = input_dimension
        if kind in ('pca', 'linear_vamp'):
            self.register_buffer('linear_mean', torch.zeros(input_dimension))
            self.register_buffer('linear_matrix', torch.zeros(input_dimension, dimension))
        else:
            self.encoder = nn.Sequential(nn.Linear(input_dimension, width), nn.SiLU(),
                                         nn.Linear(width, width), nn.SiLU())
            self.location = nn.Linear(width, dimension)
            if kind == 'vae':
                self.log_variance = nn.Linear(width, dimension)
        self.decoder = mlp(dimension, width, 244)
        self.transition = Transition(dimension, width, c['model']['mixtures'])

    def posterior(self, x):
        hidden = self.encoder(x)
        return self.location(hidden), self.log_variance(hidden).clamp(-12, 8)

    def encode(self, x):
        if self.kind in ('pca', 'linear_vamp'):
            return (x - self.linear_mean) @ self.linear_matrix
        return self.location(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    @torch.no_grad()
    def rollout(self, historyx, steps, particles, seed, deterministic=False):
        """Return future-only raw latents and decoded measurements [B,P,T,D]."""
        history = self.encode(historyx)
        history = history[:, None].expand(-1, particles, -1, -1).flatten(0, 1)
        generator = torch.Generator(device=history.device).manual_seed(seed)
        latents = []
        for _ in range(steps):
            following = (self.transition.mean(history) if deterministic else
                         self.transition.sample(history, generator))
            latents.append(following)
            history = torch.cat((history[:, 1:], following[:, None]), 1)
        z = torch.stack(latents, 1).reshape(len(historyx), particles, steps, -1)
        return z, self.decode(z)
