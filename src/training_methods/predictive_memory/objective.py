"""One mixture component describes the entire future path, not individual lags."""
import torch
from torch import nn
from torch.nn import functional as F
from src.data.predictive_memory.targets import BLOCKS


class PathHeads(nn.Module):
    def __init__(self, state_dim, lags, *, components=4, rank=2, hidden=128, minimum_std=.05):
        super().__init__()
        self.components, self.rank, self.dimension, self.minimum_std = components, rank, lags*128, minimum_std
        self.present = nn.Sequential(nn.Linear(state_dim, hidden), nn.SiLU(), nn.Linear(hidden, 128))
        self.future = nn.Sequential(nn.Linear(state_dim+1, hidden), nn.SiLU(),
                                    nn.Linear(hidden, components*(1+self.dimension*(2+rank))))

    def forward(self, state, condition):
        raw = self.future(torch.cat((state, condition), -1)).reshape(len(state), self.components, -1)
        logits = raw[..., 0]
        mean, scale, factor = torch.split(raw[..., 1:], [self.dimension, self.dimension, self.dimension*self.rank], -1)
        covariance_diag = (F.softplus(scale)+self.minimum_std).square()
        factor = .1*factor.reshape(len(state), self.components, self.dimension, self.rank)
        return dict(present=self.present(state), logits=logits, mean=mean, diagonal=covariance_diag, factor=factor)


def joint_nll(prediction, future):
    """Nats per physical channel per lag in train-standardized coordinates."""
    truth = future.flatten(1)
    distribution = torch.distributions.LowRankMultivariateNormal(
        prediction['mean'], prediction['factor'], prediction['diagonal'])
    component_log_probability = distribution.log_prob(truth[:, None])
    return -torch.logsumexp(F.log_softmax(prediction['logits'], -1)+component_log_probability, -1)/truth.shape[-1]


def mixture_mean(prediction):
    return (prediction['logits'].softmax(-1)[..., None]*prediction['mean']).sum(1)


def fit_scaler(present, future):
    # Present and future samples are both training-only. Equal packet definition
    # at all times; one coordinate system is shared across horizons and models.
    values = torch.cat((present, future.flatten(0, 1)), 0)
    return values.mean(0), values.std(0, unbiased=False).clamp_min(1e-4)


def mean_path_scores(mean, future):
    """Common physical-coordinate errors for mixture means and diagnostic readouts."""
    scores = dict(future_mse=(mean-future).square().mean((1, 2)))
    for name, (start, end) in BLOCKS.items():
        scores[f'future_mse_{name}'] = (mean[..., start:end]-future[..., start:end]).square().mean((1, 2))
    for lag in range(future.shape[1]):
        scores[f'future_mse_lag{lag}'] = (mean[:, lag]-future[:, lag]).square().mean(1)
    return scores


def physical_scores(prediction, present, future):
    scores = dict(joint_nll=joint_nll(prediction, future),
        present_mse=(prediction['present']-present).square().mean(1),
        persistence_mse=(present[:, None]-future).square().mean((1, 2)))
    scores.update(mean_path_scores(mixture_mean(prediction).reshape_as(future), future))
    return scores
