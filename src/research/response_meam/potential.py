"""PyTorch specialization of the repository's exact Lee2003 pure-Al 2NN MEAM.

Energy equations and spline construction follow LAMMPS stable_22Jul2025,
src/MEAM/{meam_setup_done.cpp,meam_dens_init.cpp,meam_dens_final.cpp,meam.h}.
LAMMPS copyright (2003) Sandia Corporation, GNU GPL; this derived implementation
is distributed under GPL-2.0-or-later. See this package's NOTICE.md.

This is deliberately not a general MEAM parser: both original parameter files
are hash checked. All forces and force-direction derivatives are obtained from
this energy by PyTorch AD. Geometry is open/nonperiodic; independent replicas
have shape (batch, 3*atoms). The discrete neighbor/screening support is selected
outside AD, while every nonconstant geometric screening term stays inside AD.
"""
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


FILE_HASHES = {
    'library': 'f72f19b5185e6da9c4e4c26029346b9210296b289ba791178dee1e923281835e',
    'parameters': 'b1ba33a29d8884692aeb4a1f0c78df51146f6f68d281121135dfca3207506e6a',
}
RC, DELR, RE, EC, EMBED = 4.5, 0.1, 2.86, 3.36, 1.16
ALPHA, CMIN, CMAX = 4.68559768, 0.49, 2.80
BETA, T = (3.2, 2.6, 6.0, 2.6), (3.05, 0.51, 7.75)
EBOUND = CMAX**2 / (4 * (CMAX - 1))
DR = 1.1 * RC / 1000


def fcut(x):
    """LAMMPS MEAM's polynomial, with its original piecewise boundaries."""
    interior = (1 - (1 - x.clamp(0, 1)).pow(4)).square()
    return torch.where(x <= 0, torch.zeros_like(x),
                       torch.where(x >= 1, torch.ones_like(x), interior))


def pair_coefficients():
    """Reproduce the fixed 1000-point 2NN pair table and cubic interpolation."""
    ratio = math.sqrt(2.)
    c = 4 / (ratio * ratio) - 1
    screening = (1 - (1 - (c - CMIN) / (CMAX - CMIN))**4)**8
    rho_ref = 12 + 6 * screening * math.exp(-BETA[0] * (ratio - 1))

    def phi(r):
        density = (12 * math.exp(-BETA[0] * (r / RE - 1))
                   + 6 * screening * math.exp(-BETA[0] * (ratio * r / RE - 1)))
        if density <= 1e-14 or r == 0:
            return 0.
        rhobar = density / rho_ref
        embedding = EMBED * EC * rhobar * math.log(rhobar)
        astar = ALPHA * (r / RE - 1)
        rose = -EC * (1 + astar + .05 * astar**3) * math.exp(-astar)
        return (rose - embedding) / 6

    values = np.empty(1000, dtype=np.float64)
    b2 = -screening / 2
    for j in range(1000):
        r = j * DR
        values[j] = phi(r)
        for n in range(1, 11):
            term = b2**n * phi(r * ratio**n)
            if abs(term) < 1e-20:
                break
            values[j] += term
    slope = np.zeros_like(values)
    slope[0] = values[1] - values[0]
    slope[1] = .5 * (values[2] - values[0])
    slope[-2] = .5 * (values[-1] - values[-3])
    slope[2:-2] = ((values[:-4] - values[4:]) + 8 * (values[3:-1] - values[1:-3])) / 12
    quadratic, cubic = np.zeros_like(values), np.zeros_like(values)
    quadratic[:-1] = 3 * np.diff(values) - 2 * slope[:-1] - slope[1:]
    cubic[:-1] = slope[:-1] + slope[1:] - 2 * np.diff(values)
    return np.stack([values, slope, quadratic, cubic], -1), rho_ref


class Lee2003Al(nn.Module):
    """Exact pinned pure-Al parameterization, with neighbor-list skin reuse."""
    def __init__(self, files, atoms, *, device='cuda', dtype=torch.float64, skin=.6):
        super().__init__()
        for key, expected in FILE_HASHES.items():
            actual = hashlib.sha256(Path(files[key]).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f'Lee2003 Al {key} differs: {actual} != {expected}')
        if skin <= 0:
            raise ValueError('A positive neighbor skin is required')
        self.atoms, self.skin = atoms, skin
        table, self.rho_ref = pair_coefficients()
        for name, value in {
            'table': table, 'beta': BETA, 'weights': T,
            'v2': [1, 2, 2, 1, 2, 1], 'v3': [1, 3, 3, 3, 6, 3, 1, 3, 3, 1],
            'parity': [1] + [-1]*3 + [1]*7 + [-1]*13,
        }.items():
            self.register_buffer(name, torch.as_tensor(value, device=device, dtype=dtype))
        self.reference = None
        self.rebuilds = 0

    @torch.no_grad()
    def support(self, q):
        """Find nonzero pairs and partial screeners; retain all their derivatives."""
        batch = len(q)
        x = q.reshape(batch, self.atoms, 3)
        if (self.reference is None or self.reference.shape != x.shape
                or bool(2 * (x-self.reference).norm(dim=-1).max() >= self.skin)):
            cutoff = RC * math.sqrt(EBOUND) + self.skin
            chunks = []
            for start in range(0, self.atoms, 256):
                distance2 = (x[:, start:start+256, None] - x[:, None]).square().sum(-1)
                b, i, j = ((distance2 < cutoff**2) & (distance2 > 0)).nonzero(as_tuple=True)
                chunks.append(torch.stack([b*self.atoms+i+start, b*self.atoms+j]))
            all_i, all_j = torch.cat(chunks, 1)
            order = torch.argsort(all_i, stable=True)
            all_i, all_j = all_i[order], all_j[order]
            counts = torch.bincount(all_i, minlength=batch*self.atoms)
            width = int(counts.max())
            starts = counts.cumsum(0) - counts
            slots = torch.arange(len(all_i), device=q.device)-starts[all_i]
            neighbors = torch.full((batch*self.atoms, width), -1, device=q.device, dtype=torch.long)
            neighbors[all_i, slots] = all_j
            half = all_i < all_j
            self.pair_i, self.pair_j = all_i[half], all_j[half]
            self.neighbors = neighbors
            self.reference = x.clone()
            self.rebuilds += 1
        flat = x.flatten(0, 1)
        i, j = self.pair_i, self.pair_j
        distance2 = (flat[j]-flat[i]).square().sum(-1)
        inside = distance2 < RC**2
        i, j, distance2 = i[inside], j[inside], distance2[inside]
        k = self.neighbors[i]
        ik2 = (flat[k.clamp_min(0)]-flat[i, None]).square().sum(-1)
        jk2 = (flat[k.clamp_min(0)]-flat[j, None]).square().sum(-1)
        a = 1 - ((ik2-jk2)/distance2[:, None]).square()
        valid = ((k >= 0) & (k != j[:, None]) & (a > 0)
                 & (ik2 <= EBOUND*distance2[:, None]) & (jk2 <= EBOUND*distance2[:, None]))
        c = (2*(ik2+jk2)/distance2[:, None] + a - 2) / torch.where(valid, a, 1.)
        alive = ~(valid & (c <= CMIN)).any(-1)
        i, j, k, c, valid = i[alive], j[alive], k[alive], c[alive], valid[alive]
        pair, slot = (valid & (c > CMIN) & (c < CMAX)).nonzero(as_tuple=True)
        return i, j, pair, k[pair, slot]

    def energy_with_support(self, q, support):
        i, j, pair, k = support
        x = q.reshape(-1, 3)
        vector = x[j]-x[i]
        r2 = vector.square().sum(-1)
        r = r2.sqrt()
        ik2 = (x[k]-x[i[pair]]).square().sum(-1)
        jk2 = (x[k]-x[j[pair]]).square().sum(-1)
        a = 1 - ((ik2-jk2)/r2[pair]).square()
        c = (2*(ik2+jk2)/r2[pair]+a-2)/a
        partial = fcut((c-CMIN)/(CMAX-CMIN))
        log_screen = r.new_zeros(len(i)).index_add(0, pair, partial.log())
        screen = log_screen.exp() * fcut((RC-r)/DELR)
        radial = torch.exp(-(r[:, None]/RE-1)*self.beta)*screen[:, None]
        unit = vector/r[:, None]
        u, v, w = unit.unbind(-1)
        order2 = torch.stack([u*u, u*v, u*w, v*v, v*w, w*w], -1)
        order3 = torch.stack([u*u*u, u*u*v, u*u*w, u*v*v, u*v*w,
                              u*w*w, v*v*v, v*v*w, v*w*w, w*w*w], -1)
        contributions = torch.cat([
            radial[:, :1], radial[:, 1:2]*unit, radial[:, 2:3],
            radial[:, 2:3]*order2, radial[:, 3:4]*unit, radial[:, 3:4]*order3,
        ], -1)
        density = x.new_zeros(len(x), 24).index_add(0, i, contributions).index_add(
            0, j, contributions*self.parity)
        rho0 = density[:, 0]
        rho1 = density[:, 1:4].square().sum(-1)
        rho2 = (density[:, 5:11].square()*self.v2).sum(-1)-density[:, 4].square()/3
        rho3 = (density[:, 14:24].square()*self.v3).sum(-1)-.6*density[:, 11:14].square().sum(-1)
        gamma = (torch.stack([rho1, rho2, rho3], -1)*self.weights).sum(-1)
        gamma = gamma / torch.where(rho0 > 0, rho0.square(), 1.)
        rhobar = rho0*(2*torch.sigmoid(gamma))/self.rho_ref
        # Isolated atoms have zero embedding energy; avoid log(0) in either AD branch.
        embedding = EMBED*EC*rhobar*torch.where(rhobar > 0, rhobar, 1.).log()
        coordinate = r/DR
        index = coordinate.detach().long().clamp_max(998)
        fraction = (coordinate-index).clamp_max(1.)
        coeff = self.table[index]
        pair_energy = ((coeff[:, 3]*fraction+coeff[:, 2])*fraction+coeff[:, 1])*fraction+coeff[:, 0]
        return embedding.sum() + (pair_energy*screen).sum()

    def forward(self, q):
        if q.ndim != 2 or q.shape[1] != 3*self.atoms:
            raise ValueError(f'Expected (replicas,{3*self.atoms}), got {tuple(q.shape)}')
        return self.energy_with_support(q, self.support(q))
