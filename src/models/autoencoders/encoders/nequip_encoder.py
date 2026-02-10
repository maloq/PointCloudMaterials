from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..base import Encoder
from ..registry import register_encoder


def _gather_neighbors(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather x[B, N, ...] at neighbor indices idx[B, N, K] -> [B, N, K, ...]."""
    if x.dim() < 2:
        raise ValueError(f"Expected tensor with shape [B, N, ...], got {tuple(x.shape)}")
    if idx.dim() != 3:
        raise ValueError(f"Expected idx shape [B, N, K], got {tuple(idx.shape)}")
    if x.shape[0] != idx.shape[0] or x.shape[1] != idx.shape[1]:
        raise ValueError(
            f"Batch/point mismatch between x={tuple(x.shape)} and idx={tuple(idx.shape)}"
        )

    b, n, k = idx.shape
    batch_idx = torch.arange(b, device=x.device).view(b, 1, 1).expand(b, n, k)
    return x[batch_idx, idx]


def _knn_graph(points: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build kNN graph for points[B, N, 3].
    Returns:
      - idx: [B, N, K]
      - rel: [B, N, K, 3], neighbor - center
      - dist: [B, N, K]
    """
    if points.dim() != 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected points shape [B, N, 3], got {tuple(points.shape)}")

    b, n, _ = points.shape
    if n < 1:
        raise ValueError("Point cloud must contain at least one point")

    if n == 1:
        idx = torch.zeros((b, 1, 1), dtype=torch.long, device=points.device)
        rel = torch.zeros((b, 1, 1, 3), dtype=points.dtype, device=points.device)
        dist = torch.zeros((b, 1, 1), dtype=points.dtype, device=points.device)
        return idx, rel, dist

    k_eff = max(1, min(int(k), n - 1))

    # Distance matrix in fp32 for numerical stability under mixed precision.
    pdist = torch.cdist(points.float(), points.float(), p=2.0)
    eye = torch.eye(n, dtype=torch.bool, device=points.device).unsqueeze(0)
    pdist = pdist.masked_fill(eye, float("inf"))

    dist, idx = torch.topk(pdist, k=k_eff, dim=-1, largest=False, sorted=False)
    neighbors = _gather_neighbors(points, idx)
    rel = neighbors - points.unsqueeze(2)
    return idx, rel, dist.to(dtype=points.dtype)


class GaussianRadialBasis(nn.Module):
    def __init__(self, num_basis: int, cutoff: float):
        super().__init__()
        if num_basis <= 0:
            raise ValueError(f"num_basis must be > 0, got {num_basis}")
        if cutoff <= 0:
            raise ValueError(f"cutoff must be > 0, got {cutoff}")

        self.num_basis = int(num_basis)
        self.cutoff = float(cutoff)
        centers = torch.linspace(0.0, self.cutoff, self.num_basis)
        step = self.cutoff / max(1, self.num_basis - 1)
        gamma = 1.0 / (step * step + 1e-8)

        self.register_buffer("centers", centers, persistent=False)
        self.register_buffer("gamma", torch.tensor(gamma), persistent=False)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        # distances: [B, N, K]
        d = distances.to(dtype=torch.float32)
        diff = d.unsqueeze(-1) - self.centers
        rbf = torch.exp(-self.gamma * (diff * diff))

        x = d / self.cutoff
        envelope = 0.5 * (torch.cos(math.pi * x.clamp(0.0, 1.0)) + 1.0)
        envelope = torch.where(d < self.cutoff, envelope, torch.zeros_like(envelope))
        rbf = rbf * envelope.unsqueeze(-1)
        return rbf.to(dtype=distances.dtype)


class VectorLayerNorm(nn.Module):
    """LayerNorm over vector-channel magnitudes, preserving equivariance."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.norm = nn.LayerNorm(int(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C, 3]
        mag = torch.linalg.norm(x, dim=-1).clamp_min(self.eps)
        mag_norm = self.norm(mag)
        scale = (mag_norm / mag).unsqueeze(-1)
        return x * scale


def _vector_dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    if p <= 0.0 or not training:
        return x
    keep_prob = 1.0 - p
    mask = (torch.rand(x.shape[:-1], device=x.device) < keep_prob).to(dtype=x.dtype)
    mask = mask / keep_prob
    return x * mask.unsqueeze(-1)


class NequIPInteractionBlock(nn.Module):
    """
    Simplified NequIP-style interaction block with scalar/vector channels.
    Uses radial basis edge features and keeps equivariance by combining:
      - scalar gates
      - vector projections onto relative directions
      - scalar-to-vector directional lifting
    """

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        radial_dim: int,
        hidden_dim: int,
        cutoff: float,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.scalar_dim = int(scalar_dim)
        self.vector_dim = int(vector_dim)
        self.cutoff = float(cutoff)
        self.dropout_rate = float(dropout_rate)

        edge_in = 2 * self.scalar_dim + int(radial_dim)
        edge_out = 2 * self.scalar_dim + 2 * self.vector_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_out),
        )

        self.vec_to_scalar = nn.Linear(self.vector_dim, self.scalar_dim, bias=False)
        self.scalar_to_vec = nn.Linear(self.scalar_dim, self.vector_dim, bias=False)

        self.scalar_update = nn.Sequential(
            nn.Linear(self.scalar_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.scalar_dim),
        )
        self.vector_mix = nn.Linear(self.vector_dim, self.vector_dim, bias=False)

        self.scalar_norm = nn.LayerNorm(self.scalar_dim)
        self.vector_norm = VectorLayerNorm(self.vector_dim)

    def forward(
        self,
        s: torch.Tensor,
        v: torch.Tensor,
        *,
        neighbor_idx: torch.Tensor,
        rel_unit: torch.Tensor,
        distances: torch.Tensor,
        rbf: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # s: [B, N, Cs], v: [B, N, Cv, 3]
        s_j = _gather_neighbors(s, neighbor_idx)  # [B, N, K, Cs]
        v_j = _gather_neighbors(v, neighbor_idx)  # [B, N, K, Cv, 3]
        s_i = s.unsqueeze(2).expand_as(s_j)       # [B, N, K, Cs]

        edge_input = torch.cat([s_i, s_j, rbf], dim=-1)
        gates = self.edge_mlp(edge_input)
        g_ss, g_sv, g_vv, g_vs = torch.split(
            gates,
            [self.scalar_dim, self.scalar_dim, self.vector_dim, self.vector_dim],
            dim=-1,
        )

        # Project neighbor vectors onto relative direction to obtain invariant scalars.
        v_proj = (v_j * rel_unit.unsqueeze(-2)).sum(dim=-1)   # [B, N, K, Cv]
        v_proj = self.vec_to_scalar(v_proj)                    # [B, N, K, Cs]
        scalar_edge = g_ss * s_j + g_sv * v_proj              # [B, N, K, Cs]

        # Lift neighbor scalars to vectors along relative direction.
        s_to_v = self.scalar_to_vec(s_j)                       # [B, N, K, Cv]
        v_from_s = s_to_v.unsqueeze(-1) * rel_unit.unsqueeze(-2)  # [B, N, K, Cv, 3]
        vector_edge = g_vv.unsqueeze(-1) * v_j + g_vs.unsqueeze(-1) * v_from_s

        edge_mask = (distances < self.cutoff).to(dtype=s.dtype)
        denom = edge_mask.sum(dim=2, keepdim=True).clamp_min(1.0)

        scalar_msg = (scalar_edge * edge_mask.unsqueeze(-1)).sum(dim=2) / denom
        vector_msg = (
            vector_edge * edge_mask.unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=2) / denom.unsqueeze(-1)

        s_out = self.scalar_norm(s + self.scalar_update(scalar_msg))

        vector_msg = self.vector_mix(vector_msg.transpose(-1, -2)).transpose(-1, -2)
        vector_msg = _vector_dropout(vector_msg, p=self.dropout_rate, training=self.training)
        v_out = self.vector_norm(v + vector_msg)
        return s_out, v_out


@register_encoder("NequIP_Backbone")
class NequIPBackboneEncoder(Encoder):
    """
    NequIP-style equivariant encoder for point clouds.

    Input:
      x: [B, N, 3]
    Output:
      inv_z: [B, latent_size]
      eq_z: [B, latent_size, 3]
      center: [B, 3]
    """

    def __init__(
        self,
        latent_size: int = 128,
        scalar_dim: int = 128,
        vector_dim: int = 64,
        num_layers: int = 4,
        num_neighbors: int = 16,
        radial_basis: int = 16,
        cutoff: float = 2.0,
        hidden_dim: int = 256,
        pooling: str = "mean",
        dropout_rate: float = 0.0,
        invariant_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if latent_size <= 0:
            raise ValueError(f"latent_size must be > 0, got {latent_size}")
        if scalar_dim <= 0 or vector_dim <= 0:
            raise ValueError(f"scalar_dim/vector_dim must be > 0, got {scalar_dim}/{vector_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        if num_neighbors <= 0:
            raise ValueError(f"num_neighbors must be > 0, got {num_neighbors}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        if not (0.0 <= float(dropout_rate) < 1.0):
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")

        pooling = str(pooling).lower()
        if pooling not in {"mean", "norm_max"}:
            raise ValueError(f"Unsupported pooling={pooling!r}; expected 'mean' or 'norm_max'")

        self.latent_size = int(latent_size)
        self.scalar_dim = int(scalar_dim)
        self.vector_dim = int(vector_dim)
        self.num_layers = int(num_layers)
        self.num_neighbors = int(num_neighbors)
        self.cutoff = float(cutoff)
        self.pooling = pooling
        self.invariant_eps = float(invariant_eps)

        self.scalar_embed = nn.Sequential(
            nn.Linear(1, self.scalar_dim),
            nn.SiLU(),
            nn.Linear(self.scalar_dim, self.scalar_dim),
        )
        self.vector_scale = nn.Parameter(torch.ones(self.vector_dim))

        self.rbf = GaussianRadialBasis(num_basis=int(radial_basis), cutoff=self.cutoff)
        self.blocks = nn.ModuleList(
            [
                NequIPInteractionBlock(
                    scalar_dim=self.scalar_dim,
                    vector_dim=self.vector_dim,
                    radial_dim=int(radial_basis),
                    hidden_dim=int(hidden_dim),
                    cutoff=self.cutoff,
                    dropout_rate=float(dropout_rate),
                )
                for _ in range(self.num_layers)
            ]
        )

        # Channel projection preserving vector structure: [B, C, 3] -> [B, latent, 3]
        self.eq_head = nn.Linear(self.vector_dim, self.latent_size, bias=False)
        self.eq_scale = nn.Parameter(torch.tensor(1.0))

        self.inv_head = nn.Sequential(
            nn.Linear(self.scalar_dim + self.latent_size, int(hidden_dim)),
            nn.SiLU(),
            nn.Dropout(float(dropout_rate)),
            nn.Linear(int(hidden_dim), self.latent_size),
        )

    def _pool(self, s: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # s: [B, N, Cs], v: [B, N, Cv, 3]
        if self.pooling == "mean":
            return s.mean(dim=1), v.mean(dim=1)

        # norm_max: choose per-channel vector from point with max norm
        s_global = s.max(dim=1).values
        mags = torch.linalg.norm(v, dim=-1)  # [B, N, Cv]
        idx = mags.argmax(dim=1)             # [B, Cv]
        b = torch.arange(v.shape[0], device=v.device).unsqueeze(1)
        c = torch.arange(v.shape[2], device=v.device).unsqueeze(0)
        v_global = v[b, idx, c]              # [B, Cv, 3]
        return s_global, v_global

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 3 or x.shape[-1] != 3:
            raise ValueError(f"Expected input shape [B, N, 3], got {tuple(x.shape)}")

        center = x.mean(dim=1)                  # [B, 3]
        pos = x - center.unsqueeze(1)           # translation-invariant coordinates

        point_norm = torch.linalg.norm(pos, dim=-1, keepdim=True)  # [B, N, 1]
        s = self.scalar_embed(point_norm)       # [B, N, Cs]
        v = pos.unsqueeze(2) * self.vector_scale.view(1, 1, -1, 1)  # [B, N, Cv, 3]

        neighbor_idx, rel, dist = _knn_graph(pos, self.num_neighbors)
        rel_unit = rel / dist.unsqueeze(-1).clamp_min(1e-8)
        rbf = self.rbf(dist)

        for block in self.blocks:
            s, v = block(
                s,
                v,
                neighbor_idx=neighbor_idx,
                rel_unit=rel_unit,
                distances=dist,
                rbf=rbf,
            )

        s_global, v_global = self._pool(s, v)  # [B, Cs], [B, Cv, 3]

        eq_z = self.eq_head(v_global.transpose(1, 2)).transpose(1, 2) * self.eq_scale
        eq_norm = torch.sqrt((eq_z * eq_z).sum(dim=-1) + self.invariant_eps)
        inv_input = torch.cat([s_global, eq_norm], dim=-1)
        inv_z = self.inv_head(inv_input)

        return inv_z, eq_z, center
