import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import Decoder
from ..registry import register_decoder


# ----- Small helpers -----
def mlp(sizes: List[int], act=nn.ReLU, bn: bool = True, last_bn: bool = False) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            if bn:
                layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(act(inplace=True))
        else:
            if last_bn and bn:
                layers.append(nn.BatchNorm1d(sizes[i + 1]))
    return nn.Sequential(*layers)


def safe_softplus(x, beta=1.0):
    # Slightly “softer” softplus to avoid exploding gradients on large positives
    return F.softplus(x, beta=beta)


def spherical_fibonacci_points(n: int, device: torch.device) -> torch.Tensor:
    """
    Deterministic quasi-uniform directions on S^2. Returns (n, 3).
    """
    # https://doi.org/10.1145/206558.206560
    i = torch.arange(n, dtype=torch.float32, device=device) + 0.5
    phi = 2.0 * math.pi * i / ((1.0 + 5.0 ** 0.5) * 0.5)  # 2π i / φ
    cos_theta = 1.0 - 2.0 * i / n
    sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta ** 2, min=0.0))
    x = torch.cos(phi) * sin_theta
    y = torch.sin(phi) * sin_theta
    z = cos_theta
    dirs = torch.stack([x, y, z], dim=-1)  # (n, 3)
    return dirs


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    x: (B, N, 3)
    returns idx: (B, N, k) indices of k nearest neighbors (excluding self).
    """
    B, N, _ = x.shape
    with torch.no_grad():
        # Pairwise squared distances
        d2 = torch.cdist(x, x, p=2.0) ** 2  # (B, N, N)
        # Exclude self by adding a large number to diagonal
        eye = torch.eye(N, device=x.device).unsqueeze(0)
        d2 = d2 + eye * 1e6
        _, idx = torch.topk(d2, k=k, dim=-1, largest=False, sorted=False)
    return idx  # (B, N, k)


# ----- Experts -----
class LatticeExpert(nn.Module):
    """
    Decodes a lattice template in canonical pose:
      - Upper-triangular cell matrix A (positive diagonal via softplus)
      - Small basis of M fractional coordinates in [0,1)^3
      - Replicates basis over a [-r..r]^3 integer grid, maps with A, then crops N nearest to origin.
    """
    def __init__(
        self,
        latent_size: int,
        num_points: int,
        basis_size: int = 4,
        replicate_radius: int = 2,
        hidden: int = 256,
    ):
        super().__init__()
        self.n = num_points
        self.m = basis_size
        self.r = replicate_radius

        # Predict upper-triangular A params and a global scale
        self.head_A = mlp([latent_size, hidden, hidden, 7], bn=True)
        # Predict M basis fractional coords in [0,1)
        self.head_basis = mlp([latent_size, hidden, hidden, self.m * 3], bn=True)

        # Precompute integer translations
        t = torch.arange(-self.r, self.r + 1)
        grid = torch.stack(torch.meshgrid(t, t, t, indexing="ij"), dim=-1).reshape(-1, 3)  # (T,3)
        self.register_buffer("translations", grid.float(), persistent=False)  # (T,3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, D)  ->  pts: (B, N, 3) in canonical pose
        """
        B = z.size(0)

        # A params: [a11, a22, a33, a12, a13, a23, log_scale]
        a_params = self.head_A(z)  # (B,7)
        a11 = safe_softplus(a_params[:, 0]) + 1e-3
        a22 = safe_softplus(a_params[:, 1]) + 1e-3
        a33 = safe_softplus(a_params[:, 2]) + 1e-3
        a12, a13, a23 = a_params[:, 3], a_params[:, 4], a_params[:, 5]
        scale = safe_softplus(a_params[:, 6]) + 0.3  # keep cells not too tiny

        # Upper triangular A, shape (B,3,3)
        A = torch.zeros(B, 3, 3, device=z.device, dtype=z.dtype)
        A[:, 0, 0] = a11
        A[:, 0, 1] = a12
        A[:, 0, 2] = a13
        A[:, 1, 1] = a22
        A[:, 1, 2] = a23
        A[:, 2, 2] = a33
        A = A * scale.unsqueeze(-1).unsqueeze(-1)

        # Basis fractional coords in [0,1)
        basis = torch.sigmoid(self.head_basis(z)).view(B, self.m, 3)  # (B,M,3)

        # Replicate basis over integer grid
        T = self.translations.shape[0]
        trans = self.translations.view(1, T, 1, 3)  # (1,T,1,3)
        basis_exp = basis.view(B, 1, self.m, 3)     # (B,1,M,3)
        # Fractional coords shifted by integer translations
        frac = trans + basis_exp                    # (B,T,M,3)
        frac = frac.view(B, T * self.m, 3)          # (B, TM, 3)

        # Map to real coords: X = frac @ A
        X = torch.einsum("bqc,bcd->bqd", frac, A)   # (B, TM, 3)

        # Crop N nearest to origin
        d2 = (X ** 2).sum(-1)                       # (B, TM)
        _, idx = torch.topk(d2, k=self.n, dim=-1, largest=False, sorted=False)  # (B,N)
        batch_idx = torch.arange(B, device=z.device).unsqueeze(-1).expand(B, self.n)
        pts = X[batch_idx, idx, :]                  # (B,N,3)

        # Center to zero mean (optional but typically desired)
        pts = pts - pts.mean(dim=1, keepdim=True)
        return pts


class AmorphousExpert(nn.Module):
    """
    Decodes a radially-parameterized amorphous template:
      - Predicts per-point radii r_i >= 0 (B,N)
      - Projects onto deterministic spherical-fibonacci directions (N,3)
      - Adds small Gaussian noise only during training
    """
    def __init__(
        self,
        latent_size: int,
        num_points: int,
        hidden: int = 256,
        noise_std: float = 0.02,
    ):
        super().__init__()
        self.n = num_points
        self.noise_std = noise_std

        self.head_r = mlp([latent_size, hidden, hidden, self.n], bn=True, last_bn=False)
        self.head_scale = mlp([latent_size, hidden, 1], bn=True, last_bn=False)

        # directions buffer is lazily created on first forward (so it picks the right device)

        self.register_buffer("_dirs_ready", torch.tensor([0], dtype=torch.uint8), persistent=False)
        self.register_buffer("_dirs", torch.empty(0), persistent=False)

    def _ensure_dirs(self, device):
        if self._dirs.numel() == 0 or self._dirs.device != device:
            dirs = spherical_fibonacci_points(self.n, device=device)  # (N,3)
            self._dirs = dirs
            self._dirs_ready = torch.tensor([1], dtype=torch.uint8, device=device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, D) -> pts: (B, N, 3)
        """
        B = z.size(0)
        self._ensure_dirs(z.device)

        r = safe_softplus(self.head_r(z)) + 1e-4     # (B,N)
        s = safe_softplus(self.head_scale(z)).clamp_min(0.1)  # (B,1)

        dirs = self._dirs.unsqueeze(0).expand(B, -1, -1)      # (B,N,3)
        pts = dirs * r.unsqueeze(-1) * s.unsqueeze(-1)        # (B,N,3)

        # Small noise during training only (keeps eval deterministic)
        if self.training and self.noise_std > 0:
            pts = pts + torch.randn_like(pts) * self.noise_std

        pts = pts - pts.mean(dim=1, keepdim=True)
        return pts


# ----- Equivariant residual refiner -----
class EquivariantRefiner(nn.Module):
    """
    Updates points by message passing with scalar edge weights over relative vectors:
        Δx_i = sum_j w_ij * (x_j - x_i)
    w_ij = MLP([||x_i - x_j||, global_context(z)])  -> scalar

    This is E(3)-equivariant (vectors transform correctly; scalars are invariant).
    """
    def __init__(
        self,
        latent_size: int,
        hidden_ctx: int = 64,
        edge_hidden: int = 64,
        k: int = 12,
        n_layers: int = 2,
        step_size: float = 1.0,
    ):
        super().__init__()
        self.k = k
        self.n_layers = n_layers
        self.step_size = step_size

        self.ctx = mlp([latent_size, hidden_ctx], bn=False)
        # Edge MLP takes [dist, ctx] → weight
        self.edge_mlp = mlp([1 + hidden_ctx, edge_hidden, edge_hidden, 1], bn=False)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B,N,3), z: (B,D) -> x_refined: (B,N,3), residual_norm: (B,)
        """
        B, N, _ = x.shape
        ctx = self.ctx(z)  # (B,C)
        residual_accum = 0.0
        for _ in range(self.n_layers):
            idx = knn(x, self.k)  # (B,N,k)
            # Gather neighbor positions
            batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand_as(idx)
            nbrs = x[batch_idx, idx, :]  # (B,N,k,3)
            xi = x.unsqueeze(2)          # (B,N,1,3)
            rij = nbrs - xi              # (B,N,k,3)
            dij = torch.norm(rij + 1e-12, dim=-1, keepdim=True)  # (B,N,k,1)

            # Build edge features: [dist, ctx]
            c = ctx.unsqueeze(1).unsqueeze(2)  # (B,1,1,C)
            c = c.expand(B, N, self.k, -1)
            edge_feat = torch.cat([dij, c], dim=-1)  # (B,N,k,1+C)
            w = self.edge_mlp(edge_feat).squeeze(-1)  # (B,N,k)
            w = torch.tanh(w)  # keep weights bounded

            # Normalize over neighbors
            w = w / (w.abs().sum(dim=-1, keepdim=True) + 1e-8)  # (B,N,k)

            # Δx_i
            delta = (w.unsqueeze(-1) * rij).sum(dim=2)  # (B,N,3)
            x = x + self.step_size * delta
            residual_accum = residual_accum + (delta ** 2).sum(dim=(1, 2))  # (B,)

        residual_norm = torch.sqrt(residual_accum + 1e-12)  # (B,)
        # Re-center
        x = x - x.mean(dim=1, keepdim=True)
        return x, residual_norm


# ----- MoE Decoder (main class) -----
@register_decoder("MoETemplate")
class MoETemplateResidualDecoder(Decoder):
    """
    Mixture-of-Experts template decoder with equivariant residual refinement.

    Args:
        num_points: N
        latent_size: D (size of z_inv)
        n_experts: number of experts (2 or 3 supported here)
        topk: how many experts can be active per sample
        gate_hidden: hidden size for gate MLP
        gate_temp: temperature for softmax (lower => peakier)
        use_residual: enable residual refiner
        residual_kwargs: dict forwarded to EquivariantRefiner
        lattice_kwargs, amorphous_kwargs: dicts forwarded to experts
    """
    def __init__(
        self,
        num_points: int,
        latent_size: int,
        n_experts: int = 2,              # {Lattice, Amorphous} by default
        topk: int = 2,
        gate_hidden: int = 256,
        gate_temp: float = 1.0,
        use_residual: bool = True,
        lattice_kwargs: Optional[dict] = None,
        amorphous_kwargs: Optional[dict] = None,
        residual_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        assert n_experts in (2, 3), "Only 2 or 3 experts supported in this implementation."
        assert 1 <= topk <= n_experts

        self.n = num_points
        self.d = latent_size
        self.n_experts = n_experts
        self.topk = topk
        self.gate_temp = gate_temp
        self.use_residual = use_residual

        lattice_kwargs = lattice_kwargs or {}
        amorphous_kwargs = amorphous_kwargs or {}
        residual_kwargs = residual_kwargs or {}

        # Experts
        self.lattice = LatticeExpert(latent_size, num_points, **lattice_kwargs)
        self.amorphous = AmorphousExpert(latent_size, num_points, **amorphous_kwargs)
        self.mlp_fallback = None
        if n_experts == 3:
            # A tiny fallback MLP expert (acts like a low-capacity folding)
            self.mlp_fallback = mlp([latent_size, 256, 256, num_points * 3], bn=True)

        # Gating network
        self.gate = mlp([latent_size, gate_hidden, gate_hidden, n_experts], bn=True, last_bn=False)

        # Residual refiner (equivariant)
        self.refiner = None
        if use_residual:
            self.refiner = EquivariantRefiner(latent_size, **residual_kwargs)

        # Expose some useful stats from last forward (optional)
        self.last_aux = {}

    def _mix(self, z: torch.Tensor, templates: List[torch.Tensor]) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            z: (B,D)
            templates: list of expert outputs, each (B,N,3), length = n_experts
        Returns:
            mixed: (B,N,3)
            aux: dict with 'weights' (B,K), 'topk_idx' (B,topk)
        """
        B = z.size(0)
        logits = self.gate(z)  # (B, K)
        K = logits.size(1)

        # Top-k mask
        topk_vals, topk_idx = torch.topk(logits, k=self.topk, dim=-1)  # (B, topk)
        mask = torch.zeros_like(logits).scatter_(1, topk_idx, 1.0)
        weights = F.softmax(logits / self.gate_temp, dim=-1) * mask
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)  # renormalize

        # Stack templates and mix with weights
        T = torch.stack(templates, dim=1)  # (B, K, N, 3)
        mixed = torch.einsum("bk,bknc->bnc", weights, T)  # (B,N,3)

        aux = {"weights": weights.detach(), "topk_idx": topk_idx.detach()}
        return mixed, aux

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, D) -> (B, N, 3)
        """
        # Expert templates
        outs = [self.lattice(z), self.amorphous(z)]
        if self.n_experts == 3:
            mlp_pts = self.mlp_fallback(z).view(z.size(0), self.n, 3)
            mlp_pts = mlp_pts - mlp_pts.mean(dim=1, keepdim=True)
            outs.append(mlp_pts)

        mixed, aux = self._mix(z, outs)

        # Residual refinement
        residual_norm = None
        if self.use_residual and self.refiner is not None:
            mixed, residual_norm = self.refiner(mixed, z)

        # Save some diagnostics for external regularization/monitoring
        self.last_aux = {
            **aux,
            "residual_norm": residual_norm.detach() if residual_norm is not None else None,
        }

        return mixed


# ----- Minimal sanity check -----
if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, D = 4, 80, 96
    z = torch.randn(B, D)

    dec = MoETemplateResidualDecoder(
        num_points=N,
        latent_size=D,
        n_experts=2,              # {Lattice, Amorphous}
        topk=1,                   # force decisive gating
        gate_temp=0.7,
        lattice_kwargs=dict(basis_size=6, replicate_radius=2),
        amorphous_kwargs=dict(noise_std=0.01),
        residual_kwargs=dict(k=12, n_layers=2, hidden_ctx=64, edge_hidden=64, step_size=0.7),
    )

    Xhat = dec(z)  # (B,N,3)
    print("Output shape:", Xhat.shape)
    print("Last aux keys:", dec.last_aux.keys())
