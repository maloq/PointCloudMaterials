"""Fixed velocity measurements of a central-atom Q,V neighborhood.

Input is [batch, atoms, 6], centre at index zero, positions in Angstrom and
velocities in Angstrom/ps. Positions must use consistent periodic images.
Subtracting the central Q,V removes translations and uniform velocity boosts.
The caller supplies all atoms within the cutoff; this module never sorts or
changes membership. Rates hold the instantaneous box and atom list fixed.

The 17 outputs are 9 shell moments, 4 smooth bond-order-power rates and 4
affine-flow summaries. They are new fixed targets, not the historical BO/CNA
definitions, and contain no learned parameters or temperature/time inputs.
"""
import torch
from torch import nn
from src.models.encoders.smooth_density import cutoff


class VelocityMeasurements(nn.Module):
    def __init__(self, cutoff_A=5.5, shell_centers_A=(2.5, 3.5, 4.5),
                 shell_width_A=.45, affine_ridge_A2=1e-4):
        super().__init__()
        self.cutoff_A = cutoff_A
        self.shell_width_A = shell_width_A
        self.affine_ridge_A2 = affine_ridge_A2
        self.register_buffer('shell_centers', torch.tensor(shell_centers_A))
        self.register_buffer('identity', torch.eye(3))

    @property
    def columns(self):
        return ([f'shell{k}/{name}' for k in range(len(self.shell_centers))
                 for name in ('radial_mean', 'radial_second', 'tangential_second')]
                + [f'bond_power_rate/l{ell}' for ell in (2, 4, 6, 8)]
                + ['flow/dilation', 'flow/shear_squared', 'flow/rotation_squared',
                   'flow/residual_second'])

    def weights(self, q):
        radius = q.norm(dim=-1)
        taper = cutoff(radius/self.cutoff_A)
        shells = torch.exp(-.5*((radius[..., None]-self.shell_centers)
                                /self.shell_width_A).square())*taper[..., None]
        return radius, taper, shells

    def bond_power(self, q):
        """Four inner-shell squared bond-order magnitudes, smooth on support."""
        weight = self.weights(q)[2][..., 0]
        direction = q/q.norm(dim=-1, keepdim=True)
        cosine = direction @ direction.transpose(-1, -2)
        weight = weight/(weight.sum(1, keepdim=True)+1e-12)
        pairs = weight[:, :, None]*weight[:, None, :]
        # Addition theorem: squared bond order = sum_ab w_a w_b P_l(n_a.n_b).
        # Native tensor recurrence supports forward AD and torch.compile; the
        # installed e3nn TorchScript harmonic kernel does not support repeated
        # torch.func.jvp calls reliably. No alternate target or finite difference.
        previous, current = torch.ones_like(cosine), cosine
        powers = []
        for ell in range(2, 9):
            previous, current = current, ((2*ell-1)*cosine*current-(ell-1)*previous)/ell
            if ell % 2 == 0:
                powers.append((pairs*current).sum((-2, -1)))
        return torch.stack(powers, -1)

    def kinematic(self, q, v):
        """Shell and affine statistics from centre-relative Q,V (no centre)."""
        radius, taper, shells = self.weights(q)
        radial = (q*v).sum(-1)/radius
        tangential = v-radial[..., None]*q/radius[..., None]
        moments = torch.stack((radial, radial.square(), tangential.square().sum(-1)), -1)
        shell_moments = shells.transpose(-1, -2) @ moments
        shell_moments = shell_moments/(shells.sum(1)[..., None]+1e-12)

        # Ridge regression of v = A q. Fixed ridge is part of the descriptor,
        # not a conditional fallback for ill-conditioned neighborhoods.
        weight = taper/(taper.sum(1, keepdim=True)+1e-12)
        weighted_q = q*weight[..., None]
        covariance = q.transpose(-1, -2) @ weighted_q
        cross = weighted_q.transpose(-1, -2) @ v
        transpose_A, _ = torch.linalg.solve_ex(
            covariance+self.affine_ridge_A2*self.identity, cross)
        A = transpose_A.transpose(-1, -2)
        dilation = A.diagonal(dim1=-2, dim2=-1).sum(-1)
        symmetric = .5*(A+transpose_A)-dilation[:, None, None]*self.identity/3
        antisymmetric = .5*(A-transpose_A)
        residual = v-q @ transpose_A
        flow = torch.stack((dilation, symmetric.square().sum((-2, -1)),
                            antisymmetric.square().sum((-2, -1)),
                            (weight*residual.square().sum(-1)).sum(-1)), -1)
        return shell_moments.flatten(1), flow

    def forward(self, state):
        q = state[:, 1:, :3]-state[:, :1, :3]
        v = state[:, 1:, 3:]-state[:, :1, 3:]
        # One forward directional derivative; no dense coordinate Jacobian.
        _, rates = torch.func.jvp(self.bond_power, (q,), (v,))
        shells, flow = self.kinematic(q, v)
        return torch.cat((shells, rates, flow), -1)
