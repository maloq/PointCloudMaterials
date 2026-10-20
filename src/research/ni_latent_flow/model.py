"""Whole-segment conditional flows and matched finite-history transition models.

The flow integration coordinate generates samples; it is not physical MD time.
Every public state interface uses the supplied raw latent/measurement basis.
Train-only location and scale buffers standardize that basis internally.
LaM-SLidE algorithm reference: ml-jku/LaM-SLidE commit
3f7d7fa8f26ba6c4fa5ec4f9c95c85a743ff1ee3, MD17 second-stage recipe.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.research.ni_kinetic_latent.model import Transition, mlp


def _noise_like(value, generator=None):
    return torch.randn(value.shape, device=value.device, dtype=value.dtype,
                       generator=generator)


class _ParallelAttention(nn.Module):
    """LaM-SLidE parallel attention/MLP block, with RoPE and Q/K RMSNorm."""

    def __init__(self, width, heads):
        super().__init__()
        self.width, self.heads = width, heads
        self.linear1 = nn.Linear(width, 5 * width)
        self.linear2 = nn.Linear(3 * width, width)
        self.query_scale = nn.Parameter(torch.ones(width // heads))
        self.key_scale = nn.Parameter(torch.ones(width // heads))
        self.register_buffer('frequency', torch.exp(
            -math.log(10000.) * torch.arange(0, width // heads, 2) / (width // heads)))

    def _rope(self, value):
        angles = torch.arange(value.shape[-2], device=value.device)[:, None] * self.frequency
        even, odd = value.float()[..., 0::2], value.float()[..., 1::2]
        return torch.stack((even * angles.cos() - odd * angles.sin(),
                            even * angles.sin() + odd * angles.cos()), -1).flatten(-2).to(value.dtype)

    def forward(self, tokens):
        qkv, hidden = self.linear1(tokens).split((3 * self.width, 2 * self.width), -1)
        query, key, value = qkv.reshape(len(tokens), tokens.shape[1], 3,
                                       self.heads, -1).permute(2, 0, 3, 1, 4)
        query = (query.float() * torch.rsqrt(query.float().square().mean(-1, keepdim=True) + 1e-6)
                 * self.query_scale).to(value.dtype)
        key = (key.float() * torch.rsqrt(key.float().square().mean(-1, keepdim=True) + 1e-6)
               * self.key_scale).to(value.dtype)
        attention = F.scaled_dot_product_attention(self._rope(query), self._rope(key), value)
        attention = attention.transpose(1, 2).reshape(*tokens.shape)
        return self.linear2(torch.cat((attention, F.gelu(hidden, approximate='tanh')), -1))


class _LatentLayer(nn.Module):
    """Original alternating entity/time blocks specialized to one latent token."""

    def __init__(self, width, heads):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        self.spatial = _ParallelAttention(width, heads)
        self.temporal = _ParallelAttention(width, heads)

    def forward(self, tokens, embedding):
        spatial_shift, spatial_scale, spatial_gate, temporal_shift, temporal_scale, temporal_gate = (
            self.modulation(embedding)[:, None].chunk(6, -1))
        hidden = self.norm(tokens) * (1 + spatial_scale) + spatial_shift
        hidden = self.spatial(hidden.reshape(-1, 1, tokens.shape[-1])).reshape_as(tokens)
        tokens = tokens + spatial_gate * hidden
        hidden = self.norm(tokens) * (1 + temporal_scale) + temporal_shift
        return tokens + temporal_gate * self.temporal(hidden)


class BlockFlow(nn.Module):
    """LaM-SLidE data-prediction interpolant for a complete conditioned block.

    ``config`` declares history, horizon, width, layers, heads, flow_steps and
    history_noise. Here history is exactly four observations. History corruption
    is a separate robustness adaptation used in training and stochastic inference.
    One corrupted context is held fixed throughout each sampled flow.

    Reference: arXiv:2502.12128v5, Appendix C equations 8--11 and Appendix E.3.
    The released MD17 recipe uses GVP interpolation, clean-data MSE, and Euler
    integration on [.001,.999]. ``flow_steps`` counts grid points, as its code
    does; ten grid points imply nine model evaluations. The one-token state,
    frozen existing representation and train-only whitening are Ni adaptations.
    """

    def __init__(self, dimension, config):
        super().__init__()
        self.dimension = dimension
        self.history = config['history']
        self.horizon = config['horizon']
        self.flow_steps = config['flow_steps']
        self.history_noise = config['history_noise']
        width = config['width']
        if self.history != 4 or width % config['heads'] or (width // config['heads']) % 2:
            raise ValueError('BlockFlow requires four history frames and an even per-head dimension')
        if self.flow_steps < 2 or self.horizon < 1 or self.history_noise < 0:
            raise ValueError('Invalid flow horizon, integration count or context-noise scale')
        self.register_buffer('location', torch.zeros(dimension))
        self.register_buffer('scale', torch.ones(dimension))
        self.register_buffer('flow_frequencies', torch.exp(-math.log(10000.) * torch.arange(128) / 128))
        self.register_buffer('conditioning_mask', torch.cat((torch.ones(self.history),
                                                             torch.zeros(self.horizon))).long())
        self.input_projection = nn.Linear(dimension, width)
        self.condition_projection = nn.Linear(dimension, width)
        self.flow_embedding = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))
        self.token_kind = nn.Embedding(2, width)
        self.layers = nn.ModuleList([_LatentLayer(width, config['heads'])
                                     for _ in range(config['layers'])])
        self.output_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.output_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.output = nn.Linear(width, dimension)

    def _context(self, history, generator=None, corrupt=True):
        context = (history.float() - self.location) / self.scale
        if corrupt and self.history_noise:
            context = context + self.history_noise * _noise_like(context, generator)
        future_context = context.mean(1, keepdim=True).expand(-1, self.horizon, -1)
        return torch.cat((context, future_context), 1)

    def _prediction(self, interpolated, flow_time, context):
        phase = 1000 * flow_time.float()[:, None] * self.flow_frequencies[None]
        embedding = self.flow_embedding(torch.cat((phase.cos(), phase.sin()), -1))
        tokens = (self.input_projection(interpolated) + self.condition_projection(context)
                  + self.token_kind(self.conditioning_mask)[None])
        for layer in self.layers:
            tokens = layer(tokens, embedding)
        shift, scale = self.output_modulation(embedding)[:, None].chunk(2, -1)
        return self.output(self.output_norm(tokens) * (1 + scale) + shift).float()

    def _velocity(self, interpolated, flow_time, context):
        """Stable algebraic form of the paper's data-to-score-to-drift formula."""
        angle = .5 * math.pi * flow_time[:, None, None]
        prediction = self._prediction(interpolated, flow_time, context)
        return .5 * math.pi * (prediction - angle.sin() * interpolated) / angle.cos()

    def training_prediction(self, history, target, weights=None):
        """Return clean future prediction, data loss, and differentiable metrics.

        The returned prediction is from the same noisy target/interpolation used
        in the data-prediction objective. Decode it for auxiliary losses; it is not
        an autonomous forecast. ``weights`` sum over dimensions, then losses are
        averaged over examples and all 64 observed/future frames, as in the
        reference data objective. Omitting weights uses 1/D.
        """
        clean = (torch.cat((history, target), 1).float() - self.location) / self.scale
        source = torch.randn_like(clean)
        flow_time = .001 + .998 * torch.rand(len(history), device=history.device)
        angle = .5 * math.pi * flow_time[:, None, None]
        interpolated = angle.sin() * clean + angle.cos() * source
        estimated = self._prediction(interpolated, flow_time, self._context(history))
        squared = (estimated - clean).square()
        error = squared.mean(-1) if weights is None else (squared * weights).sum(-1)
        loss = error.mean()
        prediction = self.location + self.scale * estimated[:, self.history:]
        return prediction, loss, dict(data_prediction_mse=loss,
                                      clean_prediction_mse=(prediction - target).square().mean())

    def loss(self, history, target, weights=None):
        _, loss, metrics = self.training_prediction(history, target, weights)
        return loss, metrics

    @torch.no_grad()
    def sample(self, history, particles, seed, deterministic=False):
        """Future-only states [B,P,T,D], without resets or state clipping.

        ``deterministic=True`` is the clean-context, zero-initial-noise flow
        trajectory. It is not the conditional mean of the generated distribution.
        """
        batch = len(history)
        expanded = history[:, None].expand(-1, particles, -1, -1).flatten(0, 1)
        generator = torch.Generator(device=history.device).manual_seed(seed)
        context = self._context(expanded, generator, corrupt=not deterministic)
        shape = (len(expanded), self.history + self.horizon, self.dimension)
        generated = torch.zeros(shape, device=history.device, dtype=torch.float32)
        if not deterministic:
            generated = _noise_like(generated, generator)
        step = .998 / (self.flow_steps - 1)
        for index in range(self.flow_steps - 1):
            flow_time = torch.full((len(expanded),), .001 + index * step, device=history.device)
            generated = generated + step * self._velocity(generated, flow_time, context)
        future = self.location + self.scale * generated[:, self.history:]
        return future.reshape(batch, particles, self.horizon, self.dimension)


class MemoryTransition(Transition):
    """Matched mixture increments with present state and finite-history features.

    Modes: current, four_frame, gru and filters. All consume the same four native
    observations; GRU/filters use only their three consecutive increments. Filter
    states start at zero at the first observed increment and are normalized by
    their finite observed weight. This does not supply additional physical past.

    As in the existing Transition, log_prob is a density of standardized
    increments (the fixed raw-scale Jacobian is omitted). With history_noise,
    it is conditional on one corrupted context, not marginalized over noise.
    """

    def __init__(self, dimension, width, mixtures, mode, history_noise=0.,
                 filter_decays=(.5, .8, .95)):
        super().__init__(dimension, width, mixtures)
        self.mode = mode
        self.history_noise = history_noise
        outputs = mixtures * (1 + 2 * dimension)
        if history_noise < 0:
            raise ValueError('Context-noise scale must be nonnegative')
        if mode == 'current':
            self.network = mlp(dimension, width, outputs)
        elif mode == 'gru':
            self.memory = nn.GRU(dimension, width, batch_first=True)
            self.network = mlp(dimension + width, width, outputs)
        elif mode == 'filters':
            decays = torch.tensor(filter_decays, dtype=torch.float32)
            if not len(decays) or not ((decays > 0) & (decays < 1)).all():
                raise ValueError('Exponential filter decays must lie strictly between zero and one')
            weights = decays[:, None] ** torch.arange(2, -1, -1)
            self.register_buffer('filter_weights', weights / weights.sum(-1, keepdim=True))
            self.network = mlp((1 + len(decays)) * dimension, width, outputs)
        elif mode != 'four_frame':
            raise ValueError('Unknown finite-history transition: ' + mode)

    def distribution(self, history, generator=None, corrupt=True):
        normalized = (history.float() - self.location) / self.scale
        if corrupt and self.history_noise:
            normalized = normalized + self.history_noise * _noise_like(normalized, generator)
        current = normalized[..., -1, :]
        if self.mode == 'current':
            features = current
        elif self.mode == 'four_frame':
            features = normalized.flatten(-2)
        else:
            increments = normalized[..., 1:, :] - normalized[..., :-1, :]
            if self.mode == 'gru':
                _, memory = self.memory(increments.reshape(-1, 3, self.dimension))
                memory = memory[-1].reshape(*current.shape[:-1], -1)
            else:
                memory = torch.einsum('kt,...td->...kd', self.filter_weights, increments).flatten(-2)
            features = torch.cat((current, memory), -1)
        values = self.network(features).float()
        logits = values[..., :self.mixtures]
        means, scales = values[..., self.mixtures:].reshape(
            *values.shape[:-1], self.mixtures, 2 * self.dimension).chunk(2, -1)
        return logits, means, F.softplus(scales) + .02

    def sample(self, history, generator):
        logits, means, scales = self.distribution(history, generator)
        component = torch.multinomial(logits.softmax(-1).reshape(-1, self.mixtures),
                                      1, generator=generator).flatten()
        index = torch.arange(len(component), device=history.device)
        mean = means.reshape(-1, self.mixtures, self.dimension)[index, component]
        scale = scales.reshape(-1, self.mixtures, self.dimension)[index, component]
        delta = (mean + scale * _noise_like(mean, generator)).reshape(
            *history.shape[:-2], self.dimension)
        return history[..., -1, :] + self.scale * delta

    def mean(self, history):
        """Clean-context mixture mean; context-noise averaging is not performed."""
        logits, means, _ = self.distribution(history, corrupt=False)
        delta = (logits.softmax(-1)[..., None] * means).sum(-2)
        return history[..., -1, :] + self.scale * delta
