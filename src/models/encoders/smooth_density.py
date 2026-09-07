"""Smooth central atomic densities and a local MACE product-basis encoder.

Inputs are repository-prepared offsets divided by a fixed cutoff radius. The
central atom is excluded; the producer must include every neighbor in support.
There is no nearest-neighbor selection, centering, or count normalization here.
"""
import numpy as np
import torch
from torch import nn

# Import MACE before e3nn, as in the existing adapter (trusted weight setup).
from mace.modules.blocks import EquivariantProductBasisBlock
from e3nn import o3


def cutoff_numpy(r):
    u = np.clip((r - 0.8) / 0.2, 0.0, 1.0)
    return 1 - u**3 * (10 - u * (15 - 6 * u))


def cutoff(r):
    u = ((r - 0.8) / 0.2).clamp(0, 1)
    return 1 - u.pow(3) * (10 - u * (15 - 6 * u))


class SmoothDensity(nn.Module):
    def __init__(self, radial_channels=8, max_ell=6, radial_width=0.12, density_scale=12.0):
        super().__init__()
        self.radial_channels = radial_channels
        self.max_ell = max_ell
        self.radial_width = radial_width
        self.density_scale = density_scale
        self.register_buffer("radial_centers", torch.linspace(0.15, 0.95, radial_channels))
        self.harmonics = o3.SphericalHarmonics(list(range(max_ell + 1)), normalize=True, normalization="component")
        a, b = torch.triu_indices(radial_channels, radial_channels)
        self.register_buffer("pair_a", a)
        self.register_buffer("pair_b", b)
        self.power_dim = radial_channels + (max_ell + 1) * len(a)

    def forward(self, offsets):
        r = offsets.norm(dim=-1)
        radial = torch.exp(-0.5 * ((r[..., None] - self.radial_centers) / self.radial_width).square())
        radial = radial * cutoff(r)[..., None] / self.density_scale
        return radial.transpose(-1, -2) @ self.harmonics(offsets)

    def power(self, moments):
        result = [moments[..., 0]]
        for ell in range(self.max_ell + 1):
            block = moments[..., ell * ell:(ell + 1) ** 2]
            gram = block @ block.transpose(-1, -2) / (2 * ell + 1)
            result.append(gram[..., self.pair_a, self.pair_b])
        return torch.cat(result, dim=-1)


class DensityEncoder(nn.Module):
    """Central MACE products plus density powers; no message-passing claim.

The cached radial basis is fixed. A learned channel mixing precedes the reference
MACE symmetric contraction. The power branch retains angular orders 4 and 6.
"""
    def __init__(self, density, kind, channels=32, product_ell=4, correlation=3,
                 latent_dim=128, hidden_dim=256):
        super().__init__()
        if kind not in ("power_mlp", "mace_product"):
            raise ValueError(f"Unknown density encoder: {kind}")
        self.density = density
        self.kind = kind
        self.product_ell = product_ell
        self.register_buffer("power_mean", torch.zeros(density.power_dim))
        self.register_buffer("power_std", torch.ones(density.power_dim))
        self.register_buffer("moment_scale", torch.ones(density.radial_channels))
        extra = 0
        if kind == "mace_product":
            self.mix = nn.Linear(density.radial_channels, channels, bias=False)
            irreps = o3.Irreps([(channels, (ell, (-1)**ell)) for ell in range(product_ell + 1)])
            self.product = EquivariantProductBasisBlock(
                node_feats_irreps=irreps, target_irreps=o3.Irreps(f"{channels}x0e"),
                correlation=correlation, num_elements=1, use_sc=False,
            )
            extra = channels
        self.head = nn.Sequential(nn.Linear(density.power_dim + extra, hidden_dim), nn.SiLU(),
                                  nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, latent_dim))

    def forward_moments(self, moments):
        features = [(self.density.power(moments) - self.power_mean) / self.power_std]
        if self.kind == "mace_product":
            block = moments[..., :(self.product_ell + 1)**2] / self.moment_scale[None, :, None]
            mixed = self.mix(block.transpose(-1, -2)).transpose(-1, -2)
            attributes = torch.ones((len(moments), 1), dtype=moments.dtype, device=moments.device)
            features.append(self.product(mixed, None, attributes))
        return self.head(torch.cat(features, dim=-1))

    def forward(self, offsets):
        return self.forward_moments(self.density(offsets))


class TensorTransport(nn.Module):
    """Projected tensor-power action, defined for rotations and general matrices.

Gauss-Legendre x Fourier quadrature integrates polynomials through degree 2*lmax
exactly. Solid harmonics give STF transport without choosing a canonical frame.
"""
    def __init__(self, max_ell=6):
        super().__init__()
        self.max_ell = max_ell
        z, weight = np.polynomial.legendre.leggauss(max_ell + 1)
        phi = np.arange(2 * max_ell + 1) * (2 * np.pi / (2 * max_ell + 1))
        zz, pp = np.meshgrid(z, phi, indexing="ij")
        xyz = np.stack((np.sqrt(1-zz**2)*np.cos(pp), np.sqrt(1-zz**2)*np.sin(pp), zz), -1).reshape(-1, 3)
        w = np.repeat(weight / (2 * len(phi)), len(phi))
        self.register_buffer("nodes", torch.tensor(xyz, dtype=torch.float64))
        self.register_buffer("weights", torch.tensor(w, dtype=torch.float64))
        self.harmonics = o3.SphericalHarmonics(list(range(max_ell + 1)), normalize=False, normalization="component")
        self.register_buffer("basis", self.harmonics(self.nodes))

    def matrices(self, transport):
        nodes = self.nodes.to(transport)
        values = self.harmonics(nodes @ transport.transpose(-1, -2))
        weighted = self.basis.to(transport) * self.weights.to(transport)[:, None]
        return [values[..., ell*ell:(ell+1)**2].transpose(-1, -2) @ weighted[:, ell*ell:(ell+1)**2]
                for ell in range(self.max_ell + 1)]

    def forward(self, moments, matrices):
        return torch.cat([moments[..., ell*ell:(ell+1)**2] @ matrix.transpose(-1, -2)
                          for ell, matrix in enumerate(matrices)], -1)


def motion_matrix(cross, method, epsilon):
    """Cross is sum(y x^T w) in dimensionless local coordinates."""
    c = cross.double()
    if method == "kabsch":
        u, _, vh = torch.linalg.svd(c)
        signs = torch.ones_like(c[..., 0, :])
        signs[..., -1] = torch.linalg.det(u @ vh)
        matrix = (u * signs[..., None, :]) @ vh
    elif method == "smooth":
        identity = torch.eye(3, device=c.device, dtype=c.dtype)
        values, vectors = torch.linalg.eigh(c.transpose(-1, -2) @ c + epsilon**2 * identity)
        inverse_root = (vectors * values.rsqrt()[..., None, :]) @ vectors.transpose(-1, -2)
        matrix = c @ inverse_root
    else:
        raise ValueError(f"Unknown transport method: {method}")
    return matrix.to(cross.dtype)
