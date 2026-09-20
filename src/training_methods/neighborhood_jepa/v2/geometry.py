"""Smooth fixed multiscale solid-harmonic targets, with no learned inverse decoder."""
import torch
from e3nn import o3
from .contracts import LAYOUT


def blocks(eq):
    return [x.reshape(*eq.shape[:-1], LAYOUT.channels, 2*l+1)
            for x,l in zip(eq.split(LAYOUT.widths, -1), LAYOUT.degrees)]


def moments(positions, graph, count):
    """C2 compact support; solid harmonics are regular at the center and carry parity.

    Each radial channel uses unit solid harmonics x/R, multiplied by a C2 taper
    between .75R and R, divided by 1 + the weighted coordination count. The center
    contributes to the denominator but all ell>0 solid harmonics are exactly zero.
    """
    x = positions.float()
    radius = x.norm(dim=-1)
    result = [[] for _ in LAYOUT.degrees]
    for support in LAYOUT.radii:
        u = ((radius/support-.75)/.25).clamp(0, 1)
        weight = 1-10*u**3+15*u**4-6*u**5
        denominator = x.new_ones(count).index_add(0, graph, weight)
        harmonics = o3.spherical_harmonics(list(LAYOUT.degrees), x/support,
                                          normalize=False, normalization='component')
        for i, h in enumerate(harmonics.split([2*l+1 for l in LAYOUT.degrees], -1)):
            value = x.new_zeros(count, h.shape[-1]).index_add(0, graph, weight[:,None]*h)
            result[i].append(value/denominator[:,None])
    return torch.cat([torch.stack(channels, 1).flatten(1) for channels in result], -1)


def scaled_error(prediction, target, scales):
    """Equal mean across degree/radius blocks; never independently scale m."""
    return torch.stack([((p-t)/scales[i,:,None]).square().mean((-2,-1))
                        for i,(p,t) in enumerate(zip(blocks(prediction), blocks(target)))], -1)
