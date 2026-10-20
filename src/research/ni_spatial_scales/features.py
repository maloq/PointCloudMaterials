"""Linear-cost, rotationally invariant regional density and motion measurements.

The producer supplies every atom inside the observed periodic ball, excluding
the central atom. Q and V are relative to that same central atom. Invalid padded
entries are moved outside the compact support before either shared measurement
implementation is called; no artificial origin density or shell weight remains.
"""
import torch
from torch import nn

from src.models.encoders.smooth_density import SmoothDensity
from src.research.ni_latent_sim.measurements import VelocityMeasurements


class RegionalMeasurements(nn.Module):
    def __init__(self, radius_A, shell_fractions=(.35, .6, .85), shell_width_fraction=.1):
        super().__init__()
        self.radius_A = float(radius_A)
        self.density = SmoothDensity(radial_channels=4, max_ell=6)
        self.velocity = VelocityMeasurements(
            cutoff_A=radius_A,
            shell_centers_A=tuple(radius_A*f for f in shell_fractions),
            shell_width_A=radius_A*shell_width_fraction)
        self.register_buffer('padding_position', torch.tensor([2*radius_A, 0., 0.]))

    @property
    def columns(self):
        return ([f'density/radial{a}' for a in range(4)]
                + [f'density/l{ell}_{a}_{b}' for ell in range(7)
                   for a in range(4) for b in range(a, 4)]
                + ['velocity/'+name for name in self.velocity.columns[:9]]
                + ['velocity/'+name for name in self.velocity.columns[-4:]])

    def forward(self, q, v, valid):
        q = torch.where(valid[..., None], q, self.padding_position.to(q))
        v = torch.where(valid[..., None], v, torch.zeros_like(v))
        density = self.density.power(self.density(q/self.radius_A))
        shells, flow = self.velocity.kinematic(q, v)
        return torch.cat((density, shells, flow), -1)
