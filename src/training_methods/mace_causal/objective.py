"""Physical supervision, censored local hazards, and constrained checkpoint choice."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.research.mace_velocity.train import GROUPS


def hazard_label(crystal, times, anchor, bin_edges_ps, persistence):
    """Local first sustained onset; unfinished confirmation shortens censor time.

    Eligibility uses only the observed prefix. Each event bin is (left, right].
    Censored examples contribute only fully observed bins, never a false negative
    for an unconfirmed onset near the end of the retained trajectory.
    """
    crystal = np.asarray(crystal, dtype=bool)
    times = np.asarray(times, dtype=float)
    edges = np.asarray(bin_edges_ps, dtype=float)
    if (crystal.shape != times.shape or persistence < 1 or persistence > len(times)
            or not 0 <= anchor < len(times) or np.any(np.diff(times) <= 0)
            or len(edges) == 0 or np.any(edges <= 0) or np.any(np.diff(edges) <= 0)):
        raise ValueError('Invalid sustained-onset timeline, confirmation length or hazard bins')
    eligible = not crystal[anchor]
    starts = np.flatnonzero(np.lib.stride_tricks.sliding_window_view(crystal, persistence).all(-1))
    if np.any(starts+persistence-1 <= anchor):
        eligible = False
    onset = starts[starts > anchor]
    delay = times[onset[0]]-times[anchor] if len(onset) else float('inf')
    event = int(np.searchsorted(edges, delay, side='left')) if delay <= edges[-1] else -1
    followup = times[len(times)-persistence]-times[anchor]
    observed = int(np.sum(edges <= followup+1e-8))
    if event >= 0:
        observed = event+1
    return dict(event_bin=event, observed_bins=observed, at_risk=eligible)


def hazard_nll(logits, event_bin, observed_bins):
    """Per-example log likelihood, stable even for extreme hazard logits."""
    bins = torch.arange(logits.shape[-1], device=logits.device)[None]
    event = bins == event_bin[:, None]
    survival = (bins < observed_bins[:, None]) & ~event
    if (torch.any(event_bin < -1) or torch.any(event_bin >= logits.shape[-1])
            or torch.any(observed_bins < 0) or torch.any(observed_bins > logits.shape[-1])
            or torch.any((event_bin >= 0) & (observed_bins != event_bin+1))):
        raise ValueError('Invalid event/censor bin contract')
    return (F.softplus(logits)*survival+F.softplus(-logits)*event).sum(-1)


def cumulative_risk(logits):
    return -torch.expm1(F.logsigmoid(-logits).cumsum(-1))


class PhysicalHeads(nn.Module):
    """All tasks decode the exported state; temperature is a known condition."""
    def __init__(self, state_dim, future_lags_ps, *, hidden=96, probabilistic=False, event_bins_ps=()):
        super().__init__()
        self.probabilistic = probabilistic
        self.register_buffer('lags', torch.tensor(future_lags_ps, dtype=torch.float32))
        self.present = nn.Sequential(nn.Linear(state_dim, hidden), nn.SiLU(), nn.Linear(hidden, 169))
        # Endpoint, physical delta and mean/std qbar6 over the intervening path.
        self.future = nn.Sequential(nn.Linear(state_dim+2, hidden), nn.SiLU(),
                                    nn.Linear(hidden, 340*(2 if probabilistic else 1)))
        self.hazard = (nn.Sequential(nn.Linear(state_dim+1, hidden), nn.SiLU(),
                                    nn.Linear(hidden, len(event_bins_ps))) if event_bins_ps else None)

    def forward(self, z, temperature_K):
        b, h = len(z), len(self.lags)
        condition = temperature_K.reshape(b, 1)/1000.
        inp = torch.cat((z[:, None].expand(-1, h, -1), condition[:, None].expand(-1, h, -1),
                         self.lags[None, :, None].expand(b, -1, -1)), -1)
        predicted = self.future(inp)
        if self.probabilistic:
            mean, raw_scale = predicted.chunk(2, -1)
            scale = .02+F.softplus(raw_scale)
        else:
            mean, scale = predicted, None
        return dict(present=self.present(z), future=mean, scale=scale,
                    hazard=None if self.hazard is None else self.hazard(torch.cat((z, condition), -1)))


def target_normalization(samples):
    """Equal-source moments of train endpoints; no validation/test calibration."""
    groups = {}
    for sample in samples:
        if sample['split'] != 'train':
            continue
        groups.setdefault(sample['source_id'], []).append(np.vstack((sample['present'], sample['future'])))
    if not groups:
        raise ValueError('No training sources for target normalization')
    arrays = [np.concatenate(v).astype(np.float64) for v in groups.values()]
    mean = np.mean([a.mean(0) for a in arrays], 0)
    var = np.mean([np.mean((a-mean)**2, 0) for a in arrays], 0)
    scale = np.sqrt(var)
    for key, section in GROUPS.items():
        if 'TDA' in key:
            scale[section] = np.sqrt(var[section].mean())
    # A declared numerical floor, recorded with the fitted scales and constant IDs.
    return dict(mean=mean.astype(np.float32), scale=np.maximum(scale, 1e-6).astype(np.float32),
                constant_columns=np.flatnonzero(var == 0).tolist(), scale_floor=1e-6)


def targets(samples, norm, device):
    mean = torch.as_tensor(norm['mean'], device=device)
    scale = torch.as_tensor(norm['scale'], device=device)
    present = torch.as_tensor(np.stack([s['present'] for s in samples]), device=device)
    future = torch.as_tensor(np.stack([s['future'] for s in samples]), device=device)
    path = torch.as_tensor(np.stack([s['path'] for s in samples]), device=device)
    path = torch.stack(((path[..., 0]-mean[4])/scale[4], path[..., 1]/scale[4]), -1)
    return dict(present=(present-mean)/scale,
                future=torch.cat(((future-mean)/scale, (future-present[:, None])/scale, path), -1),
                temperature=torch.tensor([s['temperature_K'] for s in samples], device=device),
                event_bin=torch.tensor([s['event_bin'] for s in samples], device=device),
                observed_bins=torch.tensor([s['observed_bins'] for s in samples], device=device),
                at_risk=torch.tensor([s['at_risk'] for s in samples], device=device))


def block_errors(error):
    return {name: error[..., section].mean(-1) for name, section in GROUPS.items()}


def task_objective(pred, target, *, future_weight=1., delta_weight=.25, path_weight=.1, hazard_weight=.1):
    present = torch.stack(list(block_errors((pred['present']-target['present']).square()).values()), -1).mean()
    residual = pred['future']-target['future']
    if pred['scale'] is None:
        error = residual.square()
    else:
        error = .5*(residual/pred['scale']).square()+pred['scale'].log()+.5*math.log(2*math.pi)
    endpoint = torch.stack(list(block_errors(error[..., :169]).values()), -1).mean()
    delta = torch.stack(list(block_errors(error[..., 169:338]).values()), -1).mean()
    path = error[..., 338:].mean()
    hazard = present.new_zeros(())
    risk = target['at_risk'] & (target['observed_bins'] > 0)
    if pred['hazard'] is not None and risk.any():
        hazard = hazard_nll(pred['hazard'][risk], target['event_bin'][risk], target['observed_bins'][risk]).mean()
    total = present+future_weight*(endpoint+delta_weight*delta+path_weight*path)+hazard_weight*hazard
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite present/future physical objective')
    return total, dict(present=float(present.detach()), future=float(endpoint.detach()),
                       delta=float(delta.detach()), path=float(path.detach()), hazard=float(hazard.detach()))


def admissible(metrics, reference, relative_tolerance, absolute_tolerance):
    """Every declared physical task must meet its information-loss constraint."""
    if not reference or set(metrics) != set(reference):
        raise ValueError('Constraint tasks must exactly match the informative reference')
    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError('Information-loss tolerances must be nonnegative')
    return all(np.isfinite(metrics[k]) and metrics[k] <= v*(1+relative_tolerance)+absolute_tolerance
               for k, v in reference.items())
