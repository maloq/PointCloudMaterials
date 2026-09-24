"""Frozen dual-domain context and a censored short-horizon auxiliary objective."""
import torch
from torch import nn
from torch.nn import functional as F
from src.research.structured_context.model import StructuredForecaster, StructuredHead
from src.research.crystallization_paths.model import block_loss, event_nll
from src.research.crystallization_paths.runtime import selected_indices
from .data import EXTRA_DIM, extra_mask


def censored_nll(logits, event, bins=16):
    """Events beyond bins are right-censored there, never relabeled as events."""
    x = logits.flatten(-2)[..., :bins].float()
    t = torch.arange(bins, device=x.device)
    return (F.softplus(x) * (t < event[..., None])).sum(-1) + (
        F.softplus(-x) * (t == event[..., None])).sum(-1)


class FollowupForecaster(StructuredForecaster):
    def __init__(self, spec):
        if spec['method'] != 'ar_mse':
            raise ValueError('This controlled follow-up uses deterministic AR only')
        super().__init__(spec)
        w = spec['head_width']
        # Identical auxiliary parameter budget across all single-branch variants.
        self.extra_head = nn.Sequential(nn.Linear(EXTRA_DIM, w), nn.SiLU(), nn.Linear(w, w))
        nn.init.zeros_(self.extra_head[-1].weight); nn.init.zeros_(self.extra_head[-1].bias)
        self.register_buffer('extra_mean', torch.zeros(EXTRA_DIM))
        self.register_buffer('extra_scale', torch.ones(EXTRA_DIM))
        self.register_buffer('extra_mask', extra_mask(spec, 'cpu'))
        if spec['secondary_domain'] != 'off':
            self.secondary_context = StructuredHead(spec)
            self.fusion = nn.Sequential(nn.Linear(2*w, w), nn.LayerNorm(w), nn.SiLU())

    @torch.no_grad()
    def initialize_information(self, data):
        super().initialize_information(data)
        self.extra_mean.copy_(data.extra_mean); self.extra_scale.copy_(data.extra_scale)
        if self.spec['secondary_domain'] != 'off':
            ids = selected_indices(data.corpus, 'train', 8, data.plan['config']['seed'])
            observed = data.observed(ids)
            x, w = self.secondary_context.inputs(observed['secondary_features'], observed['secondary_geometry'])
            self.secondary_context.normalization.calibrate(x, w)

    def encode(self, observed):
        main = {k: observed[k] for k in ('features', 'geometry', 'condition', 'information')}
        value = super().encode(main)
        if self.spec['secondary_domain'] != 'off':
            self.secondary_context.normalization.eval()
            secondary = self.secondary_context(observed['secondary_features'],
                                                observed['secondary_geometry'], observed['condition'])
            value = self.fusion(torch.cat((value, secondary), -1))
        x = (observed['extra'] - self.extra_mean) / self.extra_scale
        return value + self.extra_head(x * self.extra_mask)

    def loss(self, observed, target, teacher_probability):
        context = self.encode(observed); anchor = self.anchor(observed, context)
        mean, _, _, hazard, _ = self.recurrent_refined(context, anchor, target['state'], teacher_probability)
        error = (mean - target['state']).square()
        present = self.spec['present_weight'] * block_loss((anchor[:, None] - target['present'][:, None]).square())
        base = self.spec['state_weight'] * block_loss(error) + event_nll(hazard, target['event']) + present
        return base + self.spec['short_weight'] * (
            censored_nll(hazard, target['event']) + self.spec['state_weight'] * block_loss(error[:, :4]))
