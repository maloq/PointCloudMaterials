"""Direct and autoregressive residual decoders for embedding futures."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .data import frame_steps


def bin_edges(horizons_ps, cadence_ps):
    edges = [0] + [frame_steps(h, cadence_ps) for h in horizons_ps]
    if any(b <= a for a, b in zip(edges[:-1], edges[1:])):
        raise ValueError(f'Future mean-bin endpoints must strictly increase: {horizons_ps}')
    return edges


def bin_means(path, edges):
    """path[:, 0] is t+dt, so edges [0,4,8,12] mean (0,3], (3,6], (6,9]."""
    return torch.stack([path[:, a:b].mean(1) for a, b in zip(edges[:-1], edges[1:])], 1)


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width),
                                 nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * width, width))

    def forward(self, x):
        return x + self.net(x)


class EmbeddingForecaster(nn.Module):
    def __init__(self, dim, history_steps, cadence_ps, horizons_ps, config):
        super().__init__()
        self.config = config
        self.dim = dim
        self.history_steps = history_steps
        self.edges = bin_edges(horizons_ps, cadence_ps)
        self.target = config['target']
        if self.target not in ('bin_means', 'trajectory'):
            raise ValueError(f'Unknown forecast target: {self.target}')
        self.output_steps = len(horizons_ps) if self.target == 'bin_means' else self.edges[-1]
        self.distribution = config['distribution']
        if self.distribution not in ('deterministic', 'low_rank_gaussian'):
            raise ValueError(f'Unknown forecast distribution: {self.distribution}')
        self.history_mode = config['history_mode']
        if self.history_mode not in ('real', 'anchor', 'mean'):
            raise ValueError(f'Unknown history control: {self.history_mode}')
        width, dropout = config['width'], config['dropout']
        self.kind = config['architecture']
        if self.kind == 'autoregressive_gru':
            if self.target != 'trajectory' or self.distribution != 'deterministic':
                raise ValueError('Autoregressive GRU predicts a deterministic full trajectory; '
                                 'derive time-bin means from its rolled-out frames.')
            settings = config['autoregressive']
            if settings['initial_state'] not in ('anchor', 'history_mean'):
                raise ValueError(f"Unknown autoregressive initial state: {settings['initial_state']}")
            if settings['training'] not in ('rollout', 'teacher_forcing'):
                raise ValueError(f"Unknown autoregressive training protocol: {settings['training']}")
            if settings['training'] == 'teacher_forcing' and any(
                    config['loss'][key] != 0 for key in ('bin_mse', 'increment_mse', 'nll')):
                raise ValueError('Teacher forcing uses one-step MSE; bin/increment losses require '
                                 'a self-conditioned trajectory. Set these weights to zero.')
        self.input = nn.Linear(2 * dim + 1, width)
        if self.kind == 'mlp':
            self.history = nn.Sequential(nn.Flatten(1), nn.Linear(history_steps * width, width),
                                         nn.GELU(), ResidualBlock(width, dropout))
        elif self.kind in ('gru', 'mean_residual_gru', 'autoregressive_gru'):
            self.history = nn.GRU(width, width, num_layers=config['layers'], batch_first=True,
                                  dropout=dropout if config['layers'] > 1 else 0)
        elif self.kind == 'transformer':
            layer = nn.TransformerEncoderLayer(width, config['heads'], 4 * width, dropout,
                                               batch_first=True, norm_first=True, activation='gelu')
            self.history = nn.TransformerEncoder(layer, config['layers'], enable_nested_tensor=False)
        else:
            raise ValueError(f'Unknown history architecture: {self.kind}')
        times = torch.arange(1, self.edges[-1] + 1) * cadence_ps
        if self.target == 'bin_means':
            times = torch.stack([times[a:b].mean() for a, b in zip(self.edges[:-1], self.edges[1:])])
        t = times / horizons_ps[-1]
        self.register_buffer('future_time', torch.stack((t, t**2, torch.sin(math.pi*t), torch.cos(math.pi*t)), -1))
        self.register_buffer('past_time', (torch.arange(history_steps) - history_steps + 1) * cadence_ps / horizons_ps[-1])
        self.decoder = nn.Sequential(nn.Linear(width + 4, width), nn.GELU(), ResidualBlock(width, dropout))
        self.mean_head = nn.Linear(width, dim)
        # A small residual starts close to persistence without disconnecting historical gradients.
        nn.init.normal_(self.mean_head.weight, std=0.001)
        nn.init.zeros_(self.mean_head.bias)
        if self.kind == 'autoregressive_gru':
            self.step_cell = nn.GRUCell(dim + width + 4, width)
        if self.distribution == 'low_rank_gaussian':
            self.rank = config['covariance_rank']
            if self.rank < 1 or config['minimum_std'] <= 0:
                raise ValueError('Joint Gaussian requires positive covariance_rank and minimum_std.')
            self.std_head = nn.Linear(width, dim)
            self.factor_head = nn.Linear(width, dim * self.rank)
            nn.init.normal_(self.factor_head.weight, std=0.001)
            nn.init.zeros_(self.factor_head.bias)

    def forward(self, history):
        """Inference always rolls out predictions; this API accepts no future targets."""
        return self._forecast(history, teacher_future=None)

    def teacher_forced(self, history, future):
        """Training-only one-step predictions, conditioned on observed future prefixes."""
        if self.kind != 'autoregressive_gru' or not self.training:
            raise ValueError('Teacher forcing is only available while training an autoregressive GRU; '
                             'validation/test must use model(history) for a genuine rollout.')
        if self.config['autoregressive']['training'] != 'teacher_forcing':
            raise ValueError('This autoregressive experiment declares rollout training, not teacher forcing.')
        return self._forecast(history, teacher_future=future)

    def _forecast(self, history, teacher_future):
        anchor = history[:, -1:]
        if self.history_mode == 'anchor':
            history = anchor.expand_as(history)
        elif self.history_mode == 'mean':
            history = history.mean(1, keepdim=True).expand_as(history)
        delta = torch.cat((torch.zeros_like(history[:, :1]), history[:, 1:] - history[:, :-1]), 1)
        t = self.past_time[None, :, None].expand(len(history), -1, -1)
        tokens = self.input(torch.cat((history, delta, t), -1))
        if self.kind in ('gru', 'mean_residual_gru', 'autoregressive_gru'):
            _, hidden = self.history(tokens)
            context = hidden[-1]
        elif self.kind == 'transformer':
            # All tokens are observed past; no future tokens or teacher forcing enter attention.
            context = self.history(tokens)[:, -1]
        else:
            context = self.history(tokens)
        if self.kind == 'autoregressive_gru':
            initial = self.config['autoregressive']['initial_state']
            previous = history.mean(1) if initial == 'history_mean' else history[:, -1]
            state = context
            predictions = []
            for step in range(self.output_steps):
                time = self.future_time[step][None].expand(len(history), -1)
                state = self.step_cell(torch.cat((previous, context, time), -1), state)
                decoded = self.decoder(torch.cat((state, time), -1))
                predicted = previous + self.mean_head(decoded)
                predictions.append(predicted)
                # Keep the full feedback graph: later losses train earlier predictions too.
                # With teacher forcing, target k is consumed only AFTER predicting k.
                previous = predicted if teacher_future is None else teacher_future[:, step]
            return dict(mean=torch.stack(predictions, 1))
        decoded = self.decoder(torch.cat((context[:, None].expand(-1, self.output_steps, -1),
            self.future_time[None].expand(len(history), -1, -1)), -1))
        base = history.mean(1, keepdim=True) if self.kind == 'mean_residual_gru' else anchor
        mean = base + self.mean_head(decoded)
        output = dict(mean=mean)
        if self.distribution == 'low_rank_gaussian':
            output['std'] = F.softplus(self.std_head(decoded)) + self.config['minimum_std']
            output['factor'] = self.factor_head(decoded).reshape(len(history), -1, self.rank) / math.sqrt(self.rank)
        return output

    def target_values(self, future):
        return bin_means(future, self.edges) if self.target == 'bin_means' else future


def joint_distribution(output):
    return torch.distributions.LowRankMultivariateNormal(output['mean'].flatten(1),
        output['factor'], output['std'].flatten(1).square())


def forecast_loss(model, output, future, anchor, weights):
    target = model.target_values(future)
    mse = F.mse_loss(output['mean'], target)
    terms = dict(mse=mse)
    loss = weights['mse'] * mse
    if model.target == 'trajectory':
        terms['bin_mse'] = F.mse_loss(bin_means(output['mean'], model.edges), bin_means(future, model.edges))
        predicted_steps = torch.diff(torch.cat((anchor[:, None], output['mean']), 1), dim=1)
        actual_steps = torch.diff(torch.cat((anchor[:, None], future), 1), dim=1)
        terms['increment_mse'] = F.mse_loss(predicted_steps, actual_steps)
        loss = loss + weights['bin_mse'] * terms['bin_mse'] + weights['increment_mse'] * terms['increment_mse']
    elif weights['bin_mse'] != 0 or weights['increment_mse'] != 0:
        raise ValueError('Bin-mean model has no within-bin trajectory; set auxiliary path weights to zero.')
    if model.distribution == 'low_rank_gaussian':
        terms['nll'] = -joint_distribution(output).log_prob(target.flatten(1)).mean() / target[0].numel()
        loss = loss + weights['nll'] * terms['nll']
    elif weights['nll'] != 0:
        raise ValueError('NLL requires a probabilistic forecast distribution.')
    terms['loss'] = loss
    return loss, terms
