"""244 fixed measurements: original BO40/TDA113, smooth density74, velocity17."""
import numpy as np
import torch
from torch import nn
from src.models.encoders.smooth_density import cutoff
from src.research.ni_latent_sim.measurements import VelocityMeasurements
from src.research.liquid_predictability.descriptors import patch_descriptor_batch


class SmoothMeasurements(nn.Module):
    """The existing density power spectrum in native tensor algebra for JVPs."""
    def __init__(self):
        super().__init__()
        self.velocity = VelocityMeasurements()
        self.register_buffer('centers', torch.linspace(.15, .95, 4))
        a, b = torch.triu_indices(4, 4)
        self.register_buffer('pair_a', a); self.register_buffer('pair_b', b)

    @property
    def columns(self):
        return ([f'density/radial{a}' for a in range(4)]
                + [f'density/l{ell}_{a}_{b}' for ell in range(7)
                   for a in range(4) for b in range(a, 4)]
                + ['velocity/'+n for n in self.velocity.columns])

    def density(self, state):
        q = state[:, 1:, :3]-state[:, :1, :3]
        r = q.norm(dim=-1)
        unit = q/r[..., None]
        cosine = unit @ unit.transpose(-1, -2)
        radial = torch.exp(-.5*((r[..., None]/5.5-self.centers)/.12).square())
        radial = radial*cutoff(r/5.5)[..., None]/12.
        previous, current = torch.ones_like(cosine), cosine
        polynomials = [previous, current]
        for ell in range(2, 7):
            previous, current = current, ((2*ell-1)*cosine*current-(ell-1)*previous)/ell
            polynomials.append(current)
        powers = [radial.sum(1)]
        for polynomial in polynomials:
            gram = radial.transpose(-1, -2) @ (polynomial @ radial)
            powers.append(gram[:, self.pair_a, self.pair_b])
        return torch.cat(powers, -1)

    def forward(self, state):
        return torch.cat((self.density(state), self.velocity(state)), -1)


def structural(q, names, factor):
    """Historical BO/TDA use their original fixed material length convention."""
    x = np.asarray(q, np.float64)*factor
    x = x-x[:, :1]
    value, _ = patch_descriptor_batch(x, families=('bond_order', 'tda'),
                alpha_backend='native', alpha_precision='safe', threads=1,
                selected_columns=names)
    return value
