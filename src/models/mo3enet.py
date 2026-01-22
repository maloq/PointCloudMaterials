# mo3enet.py
# Torch-only reference implementation of a Mo3ENet-style multi-type point cloud autoencoder.
#
# Notes:
# - Designed for "molecule-like" point clouds (small N), so the EGNN uses dense O(N^2) pairwise ops.
# - End-to-end O(3) equivariant in coordinates: rotations/reflections applied to input positions
#   rotate/reflect the latent vectors and decoded positions accordingly.
# - Types/weights are invariant (depend only on scalarized invariants of the latent).
#
# Paper: "Multi-Type Point Cloud Autoencoder: A Complete Equivariant Embedding for Molecule Conformation and Pose"
# (Mo3ENet), Kilgour et al. (arXiv:2405.13791)

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Utilities
# -------------------------

def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """
    logits: [..., N, ...]
    mask:   broadcastable boolean or {0,1} mask with same N on dim
    """
    mask = mask.to(dtype=torch.bool)
    neg_inf = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(~mask, neg_inf)
    return torch.softmax(logits, dim=dim)


def pairwise_diffs(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: (B, N, 3)
    returns:
      dx:   (B, N, N, 3) where dx[b,i,j] = x[b,i] - x[b,j]
      dist2:(B, N, N, 1)
    """
    dx = x[:, :, None, :] - x[:, None, :, :]
    dist2 = (dx * dx).sum(dim=-1, keepdim=True)
    return dx, dist2


class GaussianRBF(nn.Module):
    def __init__(self, num_rbfs: int, cutoff: float):
        super().__init__()
        self.num_rbfs = int(num_rbfs)
        self.cutoff = float(cutoff)
        centers = torch.linspace(0.0, cutoff, steps=num_rbfs)
        self.register_buffer("centers", centers)
        # width so neighboring bases overlap reasonably
        self.gamma = 1.0 / (centers[1] - centers[0]).clamp_min(1e-6).item()

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        dist: (..., 1) in same units as cutoff
        returns: (..., num_rbfs)
        """
        # dist[..., 1] -> dist[..., num_rbfs]
        d = dist
        c = self.centers.view(*([1] * (d.ndim - 1)), -1)
        rbf = torch.exp(-0.5 * ((d - c) * self.gamma) ** 2)
        return rbf


def one_hot_types(types: torch.Tensor, num_types: int) -> torch.Tensor:
    """
    types: (B,N) integer
    returns: (B,N,C) float
    """
    return F.one_hot(types.clamp_min(0), num_classes=num_types).to(dtype=torch.float32)


# -------------------------
# EGNN (E(n)-equivariant GNN)
# -------------------------

class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        n_hidden: int = 2,
        dropout: float = 0.0,
        act: nn.Module = nn.GELU(),
    ):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EGNNLayer(nn.Module):
    """
    Dense EGNN layer (O(B*N^2)).
    """

    def __init__(
        self,
        h_dim: int,
        msg_dim: int,
        num_rbfs: int,
        cutoff: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cutoff = float(cutoff)
        self.rbf = GaussianRBF(num_rbfs=num_rbfs, cutoff=cutoff)

        # edge message φ_e([h_i, h_j, rbf(||x_i-x_j||)])
        self.edge_mlp = MLP(
            in_dim=2 * h_dim + num_rbfs,
            out_dim=msg_dim,
            hidden_dim=msg_dim,
            n_hidden=2,
            dropout=dropout,
        )
        # node update φ_h([h_i, Σ_j m_ij])
        self.node_mlp = MLP(
            in_dim=h_dim + msg_dim,
            out_dim=h_dim,
            hidden_dim=h_dim,
            n_hidden=2,
            dropout=dropout,
        )
        # coordinate gate φ_x(m_ij) -> scalar
        self.coord_mlp = MLP(
            in_dim=msg_dim,
            out_dim=1,
            hidden_dim=msg_dim,
            n_hidden=1,
            dropout=dropout,
        )

        self.norm = nn.LayerNorm(h_dim)

    def forward(
        self,
        x: torch.Tensor,          # (B,N,3)
        h: torch.Tensor,          # (B,N,H)
        node_mask: torch.Tensor,  # (B,N) bool/0-1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape
        node_mask_b = node_mask.to(dtype=torch.bool)
        # pairwise geometry
        dx, dist2 = pairwise_diffs(x)                 # (B,N,N,3), (B,N,N,1)
        dist = torch.sqrt(dist2 + 1e-8)               # (B,N,N,1)

        # neighbor mask (radius cutoff, exclude self, respect padding mask)
        # mask_ij: (B,N,N,1) bool
        cutoff_mask = dist <= self.cutoff
        self_mask = ~torch.eye(N, device=x.device, dtype=torch.bool).view(1, N, N, 1)
        ni = node_mask_b[:, :, None, None]  # (B,N,1,1)
        nj = node_mask_b[:, None, :, None]  # (B,1,N,1)
        mask_ij = cutoff_mask & self_mask & ni & nj

        # build edge features
        rbf = self.rbf(dist)  # (B,N,N,R)
        hi = h[:, :, None, :].expand(B, N, N, h.shape[-1])
        hj = h[:, None, :, :].expand(B, N, N, h.shape[-1])
        e_in = torch.cat([hi, hj, rbf], dim=-1)  # (B,N,N,2H+R)

        m_ij = self.edge_mlp(e_in)  # (B,N,N,M)
        m_ij = m_ij.masked_fill(~mask_ij.expand_as(m_ij), 0.0)

        # aggregate messages
        m_i = m_ij.sum(dim=2)  # (B,N,M)

        # node update (residual)
        h_up = self.node_mlp(torch.cat([h, m_i], dim=-1))
        h = self.norm(h + h_up)

        # coordinate update (equivariant): x_i += Σ_j (x_i-x_j) * φ_x(m_ij)
        g_ij = self.coord_mlp(m_ij)  # (B,N,N,1)
        g_ij = g_ij.masked_fill(~mask_ij, 0.0)

        # normalize by neighbor count to stabilize
        denom = mask_ij.to(x.dtype).sum(dim=2).clamp_min(1.0)  # (B,N,1)
        delta = (dx * g_ij).sum(dim=2) / denom  # (B,N,3)
        x = x + delta

        # keep padded nodes fixed
        x = torch.where(node_mask_b[:, :, None], x, x.detach())
        h = torch.where(node_mask_b[:, :, None], h, h.detach())
        return x, h


class EGNNEncoder(nn.Module):
    """
    Mo3ENet-style encoder:
      (x, types) -> EGNN -> attention pool -> latent vectors g: (B,K,3)
    """

    def __init__(
        self,
        num_types: int,
        type_emb_dim: int = 32,
        h_dim: int = 256,
        msg_dim: int = 128,
        latent_k: int = 128,
        n_layers: int = 4,
        num_rbfs: int = 32,
        cutoff: float = 14.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.num_types = int(num_types)
        self.latent_k = int(latent_k)

        self.type_emb = nn.Embedding(num_embeddings=num_types, embedding_dim=type_emb_dim)

        # initial scalar features: [type_emb, radial_distance]
        self.in_proj = nn.Linear(type_emb_dim + 1, h_dim)

        self.layers = nn.ModuleList([
            EGNNLayer(h_dim=h_dim, msg_dim=msg_dim, num_rbfs=num_rbfs, cutoff=cutoff, dropout=dropout)
            for _ in range(n_layers)
        ])

        # pooling scores over nodes for each latent channel k
        self.pool_score = nn.Linear(h_dim, latent_k)

    def forward(
        self,
        x: torch.Tensor,           # (B,N,3) centered & normalized to ~unit sphere
        types: torch.Tensor,       # (B,N) int
        node_mask: torch.Tensor,   # (B,N) bool/0-1
    ) -> torch.Tensor:
        node_mask_b = node_mask.to(dtype=torch.bool)

        # radial distances (invariant scalar)
        r = torch.linalg.norm(x, dim=-1, keepdim=True)  # (B,N,1)
        t_emb = self.type_emb(types.clamp_min(0))       # (B,N,E)
        h = self.in_proj(torch.cat([t_emb, r], dim=-1)) # (B,N,H)
        h = torch.where(node_mask_b[:, :, None], h, 0.0)

        # EGNN stack
        for layer in self.layers:
            x, h = layer(x=x, h=h, node_mask=node_mask_b)

        # attention pooling to fixed-size latent vectors g (B,K,3):
        # weights over atoms per latent channel
        scores = self.pool_score(h)  # (B,N,K)
        # mask over atoms (broadcast to K)
        w = masked_softmax(scores, node_mask_b[:, :, None], dim=1)  # (B,N,K)
        g = torch.einsum("bnk,bnd->bkd", w, x)  # (B,K,3)
        return g


# -------------------------
# Equivariant decoder (vector attention)
# -------------------------

class EquivariantSwarmDecoder(nn.Module):
    """
    Decoder:
      latent vectors g (B,K,3) -> output swarm:
        y: (B,M,3)
        type_logits: (B,M,C)
        weight_logits: (B,M)  (later softmax over M, scaled by N_atoms)

    Key idea:
      - Build invariant per-latent scalars inv_k = ||g_k||.
      - Cross-attend from learned scalar queries (one per output point) to inv_k (keys),
        producing weights A[b,m,k] (invariant).
      - Output vectors y[b,m] = Σ_k A[b,m,k] * g[b,k] (equivariant).
    """

    def __init__(
        self,
        num_types: int,
        latent_k: int,
        swarm_m: int = 512,
        attn_dim: int = 128,
        query_dim: int = 128,
        hidden_s: int = 256,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.num_types = int(num_types)
        self.latent_k = int(latent_k)
        self.swarm_m = int(swarm_m)

        # learned scalar query per output swarm point
        self.query = nn.Parameter(torch.randn(swarm_m, query_dim) * 0.02)

        # attention projections (all scalar/invariant)
        self.q_proj = nn.Linear(query_dim, attn_dim)
        self.k_proj = nn.Linear(1, attn_dim)  # from inv_k (a scalar) to key
        self.v_proj = nn.Identity()           # vectors are g itself

        # global invariant summary from inv_k
        self.global_mlp = MLP(
            in_dim=latent_k, out_dim=hidden_s, hidden_dim=hidden_s, n_hidden=2, dropout=dropout
        )

        # per-point invariant head (types + weights), conditioned on global + query + attended inv
        self.point_mlp = MLP(
            in_dim=hidden_s + query_dim + 1,  # global + query + attended_inv
            out_dim=hidden_s,
            hidden_dim=hidden_s,
            n_hidden=2,
            dropout=dropout
        )
        self.type_head = nn.Linear(hidden_s, num_types)
        self.weight_head = nn.Linear(hidden_s, 1)

        # optional scalar gate on coordinates (still equivariant)
        self.coord_gate = nn.Linear(hidden_s, 1)

    def forward(self, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        g: (B,K,3)
        returns: y (B,M,3), type_logits (B,M,C), weight_logits (B,M)
        """
        B, K, _ = g.shape
        assert K == self.latent_k

        inv = torch.linalg.norm(g, dim=-1, keepdim=True)  # (B,K,1) invariant

        # attention weights A[b,m,k]
        Q = self.q_proj(self.query).unsqueeze(0).expand(B, -1, -1)  # (B,M,D)
        Kk = self.k_proj(inv)                                       # (B,K,D)
        scores = torch.einsum("bmd,bkd->bmk", Q, Kk) / math.sqrt(Q.shape[-1])
        A = torch.softmax(scores, dim=-1)                           # (B,M,K), invariant

        # equivariant coordinate output
        y = torch.einsum("bmk,bkd->bmd", A, g)  # (B,M,3)

        # invariant per-point summary
        inv_flat = inv.squeeze(-1)                                  # (B,K)
        global_s = self.global_mlp(inv_flat)                         # (B,hidden_s)
        global_s = global_s[:, None, :].expand(B, self.swarm_m, -1)  # (B,M,hidden_s)

        attended_inv = torch.einsum("bmk,bk->bm", A, inv_flat).unsqueeze(-1)  # (B,M,1)
        q_raw = self.query[None, :, :].expand(B, -1, -1)                       # (B,M,query_dim)

        s_in = torch.cat([global_s, q_raw, attended_inv], dim=-1)  # (B,M,*)
        s = self.point_mlp(s_in)                                   # (B,M,hidden_s)

        type_logits = self.type_head(s)                            # (B,M,C)
        weight_logits = self.weight_head(s).squeeze(-1)            # (B,M)

        gate = torch.sigmoid(self.coord_gate(s)).clamp(0.0, 1.0)   # (B,M,1)
        y = y * gate                                               # still equivariant

        return y, type_logits, weight_logits


# -------------------------
# Mo3ENet wrapper + GM loss
# -------------------------

@dataclass
class GMLossConfig:
    sigma: float = 1.05          # Gaussian width (paper decreases during training)
    type_scale: float = 2.0      # "type distance" scale factor (paper: ~interatomic distance order)
    use_self_target: bool = True # early training target = input self-overlap (then tends to 1 as sigma shrinks)
    loss_mode: str = "paper"     # "paper" (MSE, 1/(2*sigma^2)) or "mxtal" (SmoothL1, 1/sigma^2)
    sphere_penalty: float = 1.0  # constraint that decoded points remain in unit sphere
    minw_penalty: float = 1.0    # constraint discouraging vanishing weights
    minw_frac: float = 0.01      # 1% of mean weight
    weight_temperature: float = 1.0     # softmax temperature for node weights (MXtalTools)
    type_loss_weight: float = 0.0       # optional BCE loss on per-graph type distribution
    minw_log10_threshold: float = 2.0   # penalize weights < 10^-threshold of mean weight
    smooth_l1_beta: float = 1.0         # SmoothL1 beta for MXtalTools-style loss
    matching_eps: float = 0.01          # threshold for "matching nodes" metric


class Mo3ENet(nn.Module):
    def __init__(
        self,
        num_types: int,
        # encoder
        type_emb_dim: int = 32,
        h_dim: int = 256,
        msg_dim: int = 128,
        latent_k: int = 128,
        n_layers: int = 4,
        num_rbfs: int = 32,
        cutoff: float = 14.0,
        dropout: float = 0.05,
        # decoder
        swarm_m: int = 512,
        attn_dim: int = 128,
        query_dim: int = 128,
        hidden_s: int = 256,
        # normalization
        radius_norm: float = 1.0,  # set to dataset max radius so coords map into ~unit sphere
    ):
        super().__init__()
        self.num_types = int(num_types)
        self.radius_norm = float(radius_norm)

        self.encoder = EGNNEncoder(
            num_types=num_types,
            type_emb_dim=type_emb_dim,
            h_dim=h_dim,
            msg_dim=msg_dim,
            latent_k=latent_k,
            n_layers=n_layers,
            num_rbfs=num_rbfs,
            cutoff=cutoff,
            dropout=dropout,
        )
        self.decoder = EquivariantSwarmDecoder(
            num_types=num_types,
            latent_k=latent_k,
            swarm_m=swarm_m,
            attn_dim=attn_dim,
            query_dim=query_dim,
            hidden_s=hidden_s,
            dropout=dropout,
        )

    @staticmethod
    def center_and_normalize(
        x: torch.Tensor,
        node_mask: torch.Tensor,
        radius_norm: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Center each point cloud by its masked centroid and divide by radius_norm.
        Returns centered x and centroid (for possible de-normalization).
        """
        m = node_mask.to(dtype=x.dtype)  # (B,N)
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B,1)
        centroid = (x * m[:, :, None]).sum(dim=1) / denom  # (B,3)
        xc = x - centroid[:, None, :]
        xc = xc / max(radius_norm, 1e-8)
        return xc, centroid

    def forward(
        self,
        x: torch.Tensor,          # (B,N,3)
        types: torch.Tensor,      # (B,N)
        node_mask: torch.Tensor,  # (B,N)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          g: latent vectors (B,K,3)
          y: decoded swarm coords (B,M,3) in normalized centered space
          type_logits: (B,M,C)
          weight_logits: (B,M)
        """
        x_norm, _ = self.center_and_normalize(x, node_mask, self.radius_norm)
        g = self.encoder(x_norm, types, node_mask)
        y, type_logits, weight_logits = self.decoder(g)
        return g, y, type_logits, weight_logits


def gaussian_mixture_recon_loss(
    x_in: torch.Tensor,            # (B,N,3) normalized & centered
    t_in: torch.Tensor,            # (B,N) int
    node_mask: torch.Tensor,       # (B,N) bool/0-1
    y_out: torch.Tensor,           # (B,M,3) normalized & centered
    type_logits: torch.Tensor,     # (B,M,C)
    weight_logits: torch.Tensor,   # (B,M)
    cfg: GMLossConfig,
    return_stats: bool = False,
) -> torch.Tensor:
    """
    Mo3ENet-style GM overlap reconstruction loss:
      overlap(i,j) = w_j * exp( - (||x_i - y_j||^2 + type_scale^2 * ||onehot(t_i) - p_j||^2) / (2 sigma^2) )

    For each input point i, sum_j overlap(i,j) should match:
      - early training: input self-overlap (depends on sigma)
      - late training: tends to 1 as sigma shrinks (self-overlap -> 1)

    If return_stats=True, returns (loss, stats_dict).
    """
    sigma = float(cfg.sigma)
    type_scale = float(cfg.type_scale)
    B, N, _ = x_in.shape
    M = y_out.shape[1]
    C = type_logits.shape[-1]

    node_mask_b = node_mask.to(dtype=torch.bool)
    n_atoms = node_mask_b.sum(dim=1).to(dtype=x_in.dtype)  # (B,)

    # predicted per-swarm-point type probabilities (invariant)
    p = torch.softmax(type_logits, dim=-1)  # (B,M,C)

    # predicted swarm weights: softmax over M (optionally tempered), scaled so sum weights = N_atoms
    weight_temp = float(cfg.weight_temperature)
    weight_temp = weight_temp if weight_temp > 0 else 1.0
    w_raw = torch.softmax(weight_logits / weight_temp, dim=-1)  # (B,M)
    w = w_raw * n_atoms[:, None]                                # (B,M)

    # input one-hot types
    onehot = one_hot_types(t_in, num_types=C).to(device=x_in.device)  # (B,N,C)
    onehot = torch.where(node_mask_b[:, :, None], onehot, 0.0)

    # pairwise distances input->output
    d2_xyz = ((x_in[:, :, None, :] - y_out[:, None, :, :]) ** 2).sum(dim=-1)  # (B,N,M)
    d2_t = ((onehot[:, :, None, :] - p[:, None, :, :]) ** 2).sum(dim=-1)      # (B,N,M)
    d2 = d2_xyz + (type_scale ** 2) * d2_t

    # overlaps
    if cfg.loss_mode.lower() == "mxtal":
        overlap = torch.exp(-d2 / (sigma ** 2)) * w[:, None, :]  # (B,N,M)
    else:
        overlap = torch.exp(-d2 / (2.0 * sigma ** 2)) * w[:, None, :]  # (B,N,M)
    out_sum = overlap.sum(dim=-1)  # (B,N)

    # compute input self-overlap target (depends on sigma)
    if cfg.use_self_target:
        dx, dist2 = pairwise_diffs(x_in)
        d2_xyz_self = dist2.squeeze(-1)  # (B,N,N)
        d2_t_self = ((onehot[:, :, None, :] - onehot[:, None, :, :]) ** 2).sum(dim=-1)  # (B,N,N)
        d2_self = d2_xyz_self + (type_scale ** 2) * d2_t_self
        if cfg.loss_mode.lower() == "mxtal":
            self_ov = torch.exp(-d2_self / (sigma ** 2))  # (B,N,N)
        else:
            self_ov = torch.exp(-d2_self / (2.0 * sigma ** 2))  # (B,N,N)

        # mask padded nodes
        ni = node_mask_b[:, :, None]
        nj = node_mask_b[:, None, :]
        self_ov = self_ov * (ni & nj).to(self_ov.dtype)

        target = self_ov.sum(dim=-1).detach()  # (B,N)
    else:
        target = torch.ones_like(out_sum)

    # masked reconstruction loss per input point
    mask_f = node_mask_b.to(dtype=out_sum.dtype)
    if cfg.loss_mode.lower() == "mxtal":
        recon_per = F.smooth_l1_loss(out_sum, target, reduction="none", beta=cfg.smooth_l1_beta)
    else:
        recon_per = (out_sum - target) ** 2
    recon_graph = (recon_per * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
    recon = recon_graph.mean()

    # stabilizer 1: keep decoded points within unit sphere (since inputs are normalized)
    r = torch.linalg.norm(y_out, dim=-1)  # (B,M)
    if cfg.loss_mode.lower() == "mxtal":
        sphere = F.relu(r - 1.0)
    else:
        sphere = F.relu(r - 1.0) ** 2
    sphere = sphere.mean()

    # stabilizer 2: discourage vanishing weights
    mean_w = (n_atoms / float(M)).clamp_min(1e-8)  # (B,)
    if cfg.loss_mode.lower() == "mxtal":
        ratio = (w / mean_w[:, None]).clamp_min(1e-12)
        minw = F.relu(-torch.log10(ratio) - cfg.minw_log10_threshold)
    else:
        min_w = cfg.minw_frac * mean_w[:, None]        # (B,M)
        minw = F.relu(min_w - w) ** 2
    minw = minw.mean()

    # optional type distribution loss (per-graph)
    type_loss = torch.tensor(0.0, device=x_in.device, dtype=x_in.dtype)
    if cfg.type_loss_weight > 0:
        true_types = (onehot * mask_f[:, :, None]).sum(dim=1) / n_atoms[:, None].clamp_min(1.0)
        pred_types = (p * w_raw[:, :, None]).sum(dim=1)
        with torch.autocast(device_type=x_in.device.type, enabled=False):
            pred_types_f = pred_types.float()
            true_types_f = true_types.float()
            type_loss = F.binary_cross_entropy(pred_types_f, true_types_f) - F.binary_cross_entropy(true_types_f, true_types_f)

    total = recon + cfg.sphere_penalty * sphere + cfg.minw_penalty * minw + cfg.type_loss_weight * type_loss

    mean_self_overlap = (target * mask_f).sum() / mask_f.sum().clamp_min(1.0)
    matching = (recon_per < cfg.matching_eps) & node_mask_b
    matching_frac = matching.to(mask_f.dtype).sum() / mask_f.sum().clamp_min(1.0)

    stats = {
        "recon": recon.detach(),
        "sphere": sphere.detach(),
        "minw": minw.detach(),
        "type_loss": type_loss.detach(),
        "mean_self_overlap": mean_self_overlap.detach(),
        "matching_nodes_fraction": matching_frac.detach(),
    }

    if return_stats:
        return total, stats
    return total


# -------------------------
# Simplified interface for single-type point clouds
# -------------------------

class Mo3ENetSingleType(nn.Module):
    """
    Wrapper for Mo3ENet that works with single-type point clouds.
    Automatically creates dummy type tensors and masks.
    """
    def __init__(
        self,
        # encoder
        type_emb_dim: int = 32,
        h_dim: int = 256,
        msg_dim: int = 128,
        latent_k: int = 128,
        n_layers: int = 4,
        num_rbfs: int = 32,
        cutoff: float = 14.0,
        dropout: float = 0.05,
        # decoder
        swarm_m: int = 512,
        attn_dim: int = 128,
        query_dim: int = 128,
        hidden_s: int = 256,
        # normalization
        radius_norm: float = 1.0,
    ):
        super().__init__()
        # Single type: num_types=1
        self.model = Mo3ENet(
            num_types=1,
            type_emb_dim=type_emb_dim,
            h_dim=h_dim,
            msg_dim=msg_dim,
            latent_k=latent_k,
            n_layers=n_layers,
            num_rbfs=num_rbfs,
            cutoff=cutoff,
            dropout=dropout,
            swarm_m=swarm_m,
            attn_dim=attn_dim,
            query_dim=query_dim,
            hidden_s=hidden_s,
            radius_norm=radius_norm,
        )
        self.latent_k = latent_k
        self.swarm_m = swarm_m
        self.radius_norm = radius_norm

    def forward(
        self,
        x: torch.Tensor,  # (B, N, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Point cloud (B, N, 3)
            
        Returns:
            g: latent vectors (B, K, 3)
            y: decoded swarm coords (B, M, 3)
            type_logits: (B, M, 1) - single type
            weight_logits: (B, M)
        """
        B, N, _ = x.shape
        device = x.device
        
        # Create dummy types (all zeros) and full mask
        types = torch.zeros(B, N, dtype=torch.long, device=device)
        node_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        
        return self.model(x, types, node_mask)

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get only the latent representation."""
        g, _, _, _ = self.forward(x)
        return g

    def get_invariant_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get invariant latent (norms of equivariant latent vectors)."""
        g = self.get_latent(x)
        return torch.linalg.norm(g, dim=-1)  # (B, K)


# -------------------------
# Example usage + equivariance sanity check
# -------------------------

def random_rotation_matrix(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Generates a random 3x3 rotation matrix using QR decomposition.
    """
    A = torch.randn(3, 3, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    # ensure right-handed
    if torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


@torch.no_grad()
def equivariance_check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    B, N = 2, 12
    num_types = 5

    # toy batch with padding mask
    x = torch.randn(B, N, 3, device=device, dtype=dtype)
    types = torch.randint(0, num_types, (B, N), device=device)
    mask = torch.ones(B, N, device=device, dtype=torch.bool)
    mask[1, -3:] = False  # pad last 3 atoms of sample 2

    model = Mo3ENet(
        num_types=num_types,
        h_dim=128,
        msg_dim=64,
        latent_k=64,
        n_layers=3,
        swarm_m=128,
        radius_norm=3.0,  # pretend dataset max radius ~3
    ).to(device=device, dtype=dtype)
    model.eval()

    g1, y1, _, _ = model(x, types, mask)

    R = random_rotation_matrix(device, dtype)
    xR = x @ R.T

    g2, y2, _, _ = model(xR, types, mask)

    # latent and decoded should rotate by R
    g1R = g1 @ R.T
    y1R = y1 @ R.T

    print("latent equiv err:", (g2 - g1R).abs().max().item())
    print("decode equiv err:", (y2 - y1R).abs().max().item())


if __name__ == "__main__":
    equivariance_check()
