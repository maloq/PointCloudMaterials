"""Zero-inflated lognormal mixtures with right-censored distance likelihood."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.research.equivariant_context.model import ContextPredictor, geometry, radial_basis


def parameters(raw, scores):
    return (F.log_softmax(scores, -1), raw[..., 0],
            math.log(16.) + 4*torch.tanh(raw[..., 1]/4),
            .15 + 1.85*torch.sigmoid(raw[..., 2]))


def component_cdf(parts, radii):
    _, zero, mu, sigma = parts
    z = (radii.clamp_min(1e-12).log()-mu[..., None])/sigma[..., None]
    return zero.sigmoid()[..., None] + (-zero).sigmoid()[..., None]*torch.special.ndtr(z)


def cdf(parts, radii):
    return (parts[0].exp()[..., None]*component_cdf(parts, radii)).sum(1)


def log_likelihood(parts, distance, cap=64.):
    log_weight, zero, mu, sigma = parts
    # Replace censored/zero distances before evaluating either branch. No inf
    # may enter autograd, even in an unselected torch.where branch.
    safe = distance.clamp(min=1e-6, max=cap)[:, None]
    positive = F.logsigmoid(-zero)
    density = positive - safe.log() - sigma.log() - .5*math.log(2*math.pi) - .5*((safe.log()-mu)/sigma).square()
    survival = positive + torch.special.log_ndtr((mu-math.log(cap))/sigma)
    value = torch.where(distance[:, None] == 0, F.logsigmoid(zero),
                        torch.where(distance[:, None] >= cap, survival, density))
    return torch.logsumexp(log_weight+value, 1)


def capped_mean(parts, cap=64.):
    weights, zero, mu, sigma = parts
    boundary = math.log(cap)
    expectation = torch.exp(mu+.5*sigma.square())*torch.special.ndtr((boundary-mu-sigma.square())/sigma)
    expectation += cap*torch.special.ndtr((mu-boundary)/sigma)
    return (weights.exp()*(-zero).sigmoid()*expectation).sum(1)


@torch.no_grad()
def capped_median(parts, cap=64.):
    lo = torch.zeros_like(parts[0][:, 0]); hi = lo+cap
    for _ in range(24):
        mid = (lo+hi)/2
        below = cdf(parts, mid[:, None, None]).squeeze(-1) < .5
        lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
    return (lo+hi)/2


class LocalDistance(nn.Module):
    def __init__(self, variant):
        super().__init__(); self.variant = variant
        self.field = 'visibility' if variant == 'visibility_only' else 'z'
        width = 2 if self.field == 'visibility' else 128
        hidden = 16 if self.field == 'visibility' else 128
        self.layers = nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3))

    def forward(self, batch):
        x = batch[self.field] if self.field == 'visibility' else batch['z'][:, 0]
        raw = self.layers(x)[:, None]
        return parameters(raw, raw[..., 0]*0)


class ContextDistance(ContextPredictor):
    def __init__(self, variant, **kwargs):
        super().__init__(variant, **kwargs)
        width = kwargs['width']
        self.node_head[-1] = nn.Linear(width, 3)

    def forward(self, batch):
        g = geometry(batch['actual'], batch['nominal'])
        s = self.stem(batch['z'])+self.geometry(g['node'])
        fields = {int(l): layer(batch[f'f{l}'].transpose(-1,-2)).transpose(-1,-2) for l,layer in self.fields.items()}
        for block in self.blocks:
            s, fields = block(s, fields, g)
        raw = self.node_head(s)
        provisional = parameters(raw, raw[..., 0]*0)
        predictions = component_cdf(provisional, raw.new_tensor([4,8,12,20,32,64]))
        scores = self.combine(torch.cat((predictions, radial_basis(batch['actual'].norm(dim=-1),28.)), -1)).squeeze(-1)
        return parameters(raw, scores)
