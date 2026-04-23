from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Encoder
from .registry import register_encoder


@dataclass(frozen=True)
class RIGSGeometry:
    radii: torch.Tensor
    distances: torch.Tensor
    cosines: torch.Tensor
    angular_valid: torch.Tensor
    edge_visible: torch.Tensor
    not_self: torch.Tensor


@dataclass(frozen=True)
class SparseRIGSGraph:
    radii: torch.Tensor
    edge_index: torch.Tensor
    edge_distance: torch.Tensor
    edge_cosine: torch.Tensor


def compute_rigs_geometry(
    points: torch.Tensor,
    *,
    local_k: int = 16,
    eps: float = 1e-8,
) -> RIGSGeometry:
    if not torch.is_tensor(points):
        raise TypeError(f"RIGS geometry expects a torch.Tensor, got {type(points)}.")
    if points.dim() != 3 or points.shape[-1] != 3:
        raise ValueError(
            "RIGS geometry expects centered local structures with shape (B, N, 3), "
            f"got {tuple(points.shape)}."
        )

    coords = points.to(dtype=torch.float32)
    batch_size, num_points, _ = coords.shape
    radii = coords.norm(dim=-1)
    distances = torch.cdist(coords, coords, p=2)

    radius_i = radii.unsqueeze(2)
    radius_j = radii.unsqueeze(1)
    not_self = ~torch.eye(num_points, device=coords.device, dtype=torch.bool).unsqueeze(0)
    not_self = not_self.expand(batch_size, -1, -1)
    angular_valid = (radius_i > eps) & (radius_j > eps) & not_self
    denom = (2.0 * radius_i * radius_j).clamp_min(float(eps))
    cosines = (radius_i.square() + radius_j.square() - distances.square()) / denom
    cosines = cosines.clamp(-1.0, 1.0)
    cosines = torch.where(angular_valid, cosines, torch.zeros_like(cosines))

    if num_points <= 1:
        edge_visible = torch.zeros(
            (batch_size, num_points, num_points),
            dtype=torch.bool,
            device=coords.device,
        )
    else:
        effective_k = min(max(int(local_k), 1), num_points - 1)
        masked_distances = distances.masked_fill(~not_self, float("inf"))
        nearest = masked_distances.topk(k=effective_k, dim=-1, largest=False).indices
        edge_visible = torch.zeros_like(not_self)
        edge_visible.scatter_(2, nearest, True)
        edge_visible = (edge_visible | edge_visible.transpose(1, 2)) & not_self

    return RIGSGeometry(
        radii=radii,
        distances=distances,
        cosines=cosines,
        angular_valid=angular_valid,
        edge_visible=edge_visible,
        not_self=not_self,
    )


def compute_sparse_rigs_graph(
    points: torch.Tensor,
    *,
    local_k: int = 16,
    eps: float = 1e-8,
) -> SparseRIGSGraph:
    if not torch.is_tensor(points):
        raise TypeError(f"Sparse RIGS graph expects a torch.Tensor, got {type(points)}.")
    if points.dim() != 3 or points.shape[-1] != 3:
        raise ValueError(
            "Sparse RIGS graph expects centered local structures with shape (B, N, 3), "
            f"got {tuple(points.shape)}."
        )

    coords = points.to(dtype=torch.float32)
    batch_size, num_points, _ = coords.shape
    if num_points < 2:
        raise ValueError(
            "Sparse RIGS graph requires at least 2 points per local structure, "
            f"got num_points={num_points}."
        )

    effective_k = min(max(int(local_k), 1), num_points - 1)
    radii = coords.norm(dim=-1)
    distances = torch.cdist(coords, coords, p=2)
    not_self = ~torch.eye(num_points, device=coords.device, dtype=torch.bool).unsqueeze(0)
    not_self = not_self.expand(batch_size, -1, -1)
    masked_distances = distances.masked_fill(~not_self, float("inf"))
    edge_distance, edge_index = masked_distances.topk(k=effective_k, dim=-1, largest=False)

    flat_index = edge_index.reshape(batch_size, -1)
    neigh_radii = radii.gather(1, flat_index).reshape(batch_size, num_points, effective_k)
    center_radii = radii.unsqueeze(-1)
    angular_valid = (center_radii > eps) & (neigh_radii > eps)
    denom = (2.0 * center_radii * neigh_radii).clamp_min(float(eps))
    edge_cosine = (center_radii.square() + neigh_radii.square() - edge_distance.square()) / denom
    edge_cosine = edge_cosine.clamp(-1.0, 1.0)
    edge_cosine = torch.where(angular_valid, edge_cosine, torch.zeros_like(edge_cosine))

    return SparseRIGSGraph(
        radii=radii,
        edge_index=edge_index.to(dtype=torch.long),
        edge_distance=edge_distance,
        edge_cosine=edge_cosine,
    )


def reconstruct_points_from_distance_matrix(
    distances: torch.Tensor,
    *,
    out_dim: int = 3,
    eps: float = 1e-10,
) -> torch.Tensor:
    if not torch.is_tensor(distances):
        raise TypeError(f"distances must be a torch.Tensor, got {type(distances)}.")
    if distances.dim() != 3:
        raise ValueError(
            "Distance-matrix reconstruction expects shape (B, N, N), "
            f"got {tuple(distances.shape)}."
        )
    if distances.shape[-1] != distances.shape[-2]:
        raise ValueError(
            "Distance-matrix reconstruction requires square matrices, "
            f"got {tuple(distances.shape)}."
        )

    dist2 = distances.to(dtype=torch.float64).square()
    _, num_points, _ = dist2.shape
    eye = torch.eye(num_points, device=dist2.device, dtype=dist2.dtype)
    centering = eye - (1.0 / float(num_points)) * torch.ones_like(eye)
    gram = -0.5 * centering.unsqueeze(0) @ dist2 @ centering.unsqueeze(0)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.flip(dims=(-1,))
    eigenvectors = eigenvectors.flip(dims=(-1,))
    kept_values = eigenvalues[..., :out_dim].clamp_min(float(eps))
    kept_vectors = eigenvectors[..., :out_dim]
    coords = kept_vectors * kept_values.sqrt().unsqueeze(1)
    return coords.to(dtype=distances.dtype)


def orthogonal_procrustes_rmsd(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    allow_reflection: bool,
) -> torch.Tensor:
    if source.shape != target.shape:
        raise ValueError(
            "Procrustes RMSD requires source and target with matching shape, "
            f"got {tuple(source.shape)} and {tuple(target.shape)}."
        )
    if source.dim() != 3 or source.shape[-1] != 3:
        raise ValueError(
            "Procrustes RMSD expects shape (B, N, 3), "
            f"got {tuple(source.shape)}."
        )

    src = source.to(dtype=torch.float64)
    tgt = target.to(dtype=torch.float64)
    src_centered = src - src.mean(dim=1, keepdim=True)
    tgt_centered = tgt - tgt.mean(dim=1, keepdim=True)

    covariance = src_centered.transpose(1, 2) @ tgt_centered
    u, _, vh = torch.linalg.svd(covariance)
    rotation = u @ vh
    if not allow_reflection:
        det = torch.det(rotation)
        reflection_mask = det < 0
        if bool(reflection_mask.any().item()):
            correction = torch.eye(3, device=rotation.device, dtype=rotation.dtype).unsqueeze(0).repeat(
                rotation.shape[0], 1, 1
            )
            correction[reflection_mask, -1, -1] = -1.0
            rotation = u @ correction @ vh

    aligned = src_centered @ rotation
    sq_error = (aligned - tgt_centered).square().mean(dim=(1, 2))
    return sq_error.sqrt().to(dtype=source.dtype)


class GaussianRBF(nn.Module):
    def __init__(self, start: float, stop: float, num_centers: int) -> None:
        super().__init__()
        if num_centers <= 0:
            raise ValueError(f"num_centers must be > 0, got {num_centers}.")
        centers = torch.linspace(float(start), float(stop), int(num_centers))
        if int(num_centers) == 1:
            gamma = torch.tensor(1.0)
        else:
            spacing = float(centers[1] - centers[0])
            gamma = torch.tensor(1.0 / max(spacing * spacing, 1e-12))
        self.register_buffer("centers", centers)
        self.register_buffer("gamma", gamma)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        diff = values.unsqueeze(-1) - self.centers
        return torch.exp(-self.gamma * diff.square())


class PairBiasedSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, edge_dim: int, dropout: float) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}.")
        if heads <= 0:
            raise ValueError(f"heads must be > 0, got {heads}.")
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads}).")

        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.edge_bias = nn.Sequential(
            nn.Linear(edge_dim, dim),
            nn.GELU(),
            nn.Linear(dim, heads),
        )
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, nodes: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, _ = nodes.shape
        q = self.q_proj(nodes).reshape(batch_size, num_points, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(nodes).reshape(batch_size, num_points, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(nodes).reshape(batch_size, num_points, self.heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        bias = self.edge_bias(edge_features).permute(0, 3, 1, 2)
        attn = F.softmax(logits + bias, dim=-1)
        attn = self.attn_drop(attn)
        mixed = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, num_points, self.dim)
        return self.out_drop(self.out_proj(mixed))


class RIGSTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, edge_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = PairBiasedSelfAttention(dim=dim, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.norm_edge = nn.LayerNorm(dim)
        self.edge_message = nn.Sequential(
            nn.Linear(edge_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.edge_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, nodes: torch.Tensor, edge_features: torch.Tensor, edge_visible: torch.Tensor) -> torch.Tensor:
        nodes = nodes + self.attn(self.norm_attn(nodes), edge_features)

        normed = self.norm_edge(nodes)
        node_messages = normed.unsqueeze(1).expand(-1, normed.shape[1], -1, -1)
        edge_messages = self.edge_message(edge_features)
        local_messages = node_messages + edge_messages
        masked_messages = local_messages * edge_visible.unsqueeze(-1).to(dtype=local_messages.dtype)
        counts = edge_visible.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype=local_messages.dtype)
        pooled = masked_messages.sum(dim=2) / counts
        nodes = nodes + self.edge_out(pooled)

        nodes = nodes + self.ffn(self.norm_ffn(nodes))
        return nodes


class RIGSInvariantEdgeConv(nn.Module):
    """Scalar edge convolution over the RIGS node/edge descriptor graph."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        edge_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}.")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be > 0, got {out_channels}.")
        if edge_dim <= 0:
            raise ValueError(f"edge_dim must be > 0, got {edge_dim}.")

        hidden = max(in_channels, out_channels)
        self.edge_mlp = nn.Sequential(
            nn.Linear((2 * in_channels) + edge_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.LayerNorm(out_channels),
            )
        )
        self.out_norm = nn.LayerNorm(out_channels)
        self.out_drop = nn.Dropout(float(dropout))

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        if nodes.dim() != 3:
            raise ValueError(f"RIGSInvariantEdgeConv expects nodes (B, N, C), got {tuple(nodes.shape)}.")
        if edge_features.dim() != 4:
            raise ValueError(
                "RIGSInvariantEdgeConv expects edge_features (B, N, K, Ce), "
                f"got {tuple(edge_features.shape)}."
            )
        if edge_index.shape != edge_features.shape[:3]:
            raise ValueError(
                "edge_index must match the first 3 dimensions of edge_features. "
                f"edge_index={tuple(edge_index.shape)}, edge_features={tuple(edge_features.shape)}."
            )

        batch_size, num_points, num_edges = edge_index.shape
        flat_index = edge_index.reshape(batch_size, -1)
        neigh_feat = nodes.gather(
            1,
            flat_index.unsqueeze(-1).expand(-1, -1, nodes.shape[-1]),
        ).reshape(batch_size, num_points, num_edges, nodes.shape[-1])
        center_feat = nodes.unsqueeze(2).expand(-1, -1, num_edges, -1)
        pair_input = torch.cat([center_feat, neigh_feat - center_feat, edge_features], dim=-1)
        messages = self.edge_mlp(pair_input)
        pooled = messages.mean(dim=2)
        out = pooled + self.shortcut(nodes)
        out = self.out_norm(out)
        return F.silu(self.out_drop(out))


@register_encoder("RIGS")
class RIGSEncoder(Encoder):
    expects_channel_first = False

    def __init__(
        self,
        latent_size: int = 240,
        *,
        node_dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        r_max: float = 1.0,
        num_radial: int = 32,
        num_density: int = 16,
        local_k: int = 16,
        dropout: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if latent_size <= 0:
            raise ValueError(f"RIGSEncoder latent_size must be > 0, got {latent_size}.")
        if node_dim <= 0:
            raise ValueError(f"RIGSEncoder node_dim must be > 0, got {node_dim}.")
        if depth <= 0:
            raise ValueError(f"RIGSEncoder depth must be > 0, got {depth}.")
        if heads <= 0:
            raise ValueError(f"RIGSEncoder heads must be > 0, got {heads}.")
        if r_max <= 0:
            raise ValueError(f"RIGSEncoder r_max must be > 0, got {r_max}.")
        if num_radial <= 0:
            raise ValueError(f"RIGSEncoder num_radial must be > 0, got {num_radial}.")
        if num_density <= 0:
            raise ValueError(f"RIGSEncoder num_density must be > 0, got {num_density}.")
        if local_k <= 0:
            raise ValueError(f"RIGSEncoder local_k must be > 0, got {local_k}.")

        self.invariant_dim = int(latent_size)
        self.local_k = int(local_k)
        self.eps = max(float(eps), 1e-12)
        self.radial_rbf = GaussianRBF(0.0, float(r_max), int(num_radial))
        self.density_rbf = GaussianRBF(0.0, 2.0 * float(r_max), int(num_density))
        self.node_embed = nn.Sequential(
            nn.Linear(int(num_radial) + int(num_density), int(node_dim)),
            nn.LayerNorm(int(node_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.blocks = nn.ModuleList(
            [
                RIGSTransformerBlock(
                    dim=int(node_dim),
                    heads=int(heads),
                    edge_dim=6,
                    dropout=float(dropout),
                )
                for _ in range(int(depth))
            ]
        )
        self.pool_norm = nn.LayerNorm(int(node_dim))
        self.pool_gate = nn.Linear(int(node_dim), 1)
        self.out = nn.Sequential(
            nn.LayerNorm(int(node_dim)),
            nn.Linear(int(node_dim), int(node_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(node_dim), int(latent_size)),
        )

    def _node_features(self, geometry: RIGSGeometry) -> torch.Tensor:
        radial = self.radial_rbf(geometry.radii)
        density = self.density_rbf(geometry.distances)
        density = density * geometry.not_self.unsqueeze(-1).to(dtype=density.dtype)
        counts = geometry.not_self.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype=density.dtype)
        density_hist = density.sum(dim=2) / counts
        return torch.cat([radial, density_hist], dim=-1)

    def _edge_features(self, geometry: RIGSGeometry) -> torch.Tensor:
        radius_i = geometry.radii.unsqueeze(2)
        radius_j = geometry.radii.unsqueeze(1)
        return torch.stack(
            [
                geometry.distances,
                geometry.cosines,
                (radius_i - radius_j).abs(),
                geometry.angular_valid.to(dtype=geometry.distances.dtype),
                geometry.edge_visible.to(dtype=geometry.distances.dtype),
                geometry.not_self.to(dtype=geometry.distances.dtype),
            ],
            dim=-1,
        )

    def _model_dtype(self) -> torch.dtype:
        return self.node_embed[0].weight.dtype

    def forward(self, x: torch.Tensor):
        geometry = compute_rigs_geometry(
            x,
            local_k=self.local_k,
            eps=self.eps,
        )
        model_dtype = self._model_dtype()
        nodes = self.node_embed(self._node_features(geometry).to(dtype=model_dtype))
        edge_features = self._edge_features(geometry).to(dtype=model_dtype)
        for block in self.blocks:
            nodes = block(nodes, edge_features, geometry.edge_visible)

        pooled_weights = F.softmax(self.pool_gate(self.pool_norm(nodes)).squeeze(-1), dim=1)
        pooled = (pooled_weights.unsqueeze(-1) * nodes).sum(dim=1)
        latent = self.out(pooled)
        return latent, None, None


@register_encoder("RIGS_NN")
class RIGSNNEncoder(Encoder):
    """Dedicated neural model over RIGS scalar descriptors using local edge convolutions."""

    expects_channel_first = False
    supports_precomputed_input = True

    def __init__(
        self,
        latent_size: int = 240,
        *,
        node_dim: int = 128,
        depth: int = 4,
        r_max: float = 1.0,
        num_radial: int = 32,
        num_density: int = 16,
        local_k: int = 16,
        dropout: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if latent_size <= 0:
            raise ValueError(f"RIGSNNEncoder latent_size must be > 0, got {latent_size}.")
        if node_dim <= 0:
            raise ValueError(f"RIGSNNEncoder node_dim must be > 0, got {node_dim}.")
        if depth <= 0:
            raise ValueError(f"RIGSNNEncoder depth must be > 0, got {depth}.")
        if r_max <= 0:
            raise ValueError(f"RIGSNNEncoder r_max must be > 0, got {r_max}.")
        if num_radial <= 0:
            raise ValueError(f"RIGSNNEncoder num_radial must be > 0, got {num_radial}.")
        if num_density <= 0:
            raise ValueError(f"RIGSNNEncoder num_density must be > 0, got {num_density}.")
        if local_k <= 0:
            raise ValueError(f"RIGSNNEncoder local_k must be > 0, got {local_k}.")

        self.invariant_dim = int(latent_size)
        self.local_k = int(local_k)
        self.eps = max(float(eps), 1e-12)
        self.radial_rbf = GaussianRBF(0.0, float(r_max), int(num_radial))
        self.density_rbf = GaussianRBF(0.0, 2.0 * float(r_max), int(num_density))
        self.input_proj = nn.Sequential(
            nn.Linear(int(num_radial) + int(num_density), int(node_dim)),
            nn.LayerNorm(int(node_dim)),
            nn.SiLU(inplace=True),
            nn.Dropout(float(dropout)),
        )
        self.layers = nn.ModuleList(
            [
                RIGSInvariantEdgeConv(
                    int(node_dim),
                    int(node_dim),
                    edge_dim=6,
                    dropout=float(dropout),
                )
                for _ in range(int(depth))
            ]
        )
        fused_dim = (int(depth) + 1) * int(node_dim)
        self.point_fuse = nn.Sequential(
            nn.Linear(fused_dim, int(node_dim)),
            nn.LayerNorm(int(node_dim)),
            nn.SiLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(node_dim), int(node_dim)),
            nn.LayerNorm(int(node_dim)),
            nn.SiLU(inplace=True),
        )
        self.pool_gate = nn.Linear(int(node_dim), 1)
        self.inv_head = nn.Sequential(
            nn.Linear((2 * int(node_dim)) + 6, int(node_dim)),
            nn.LayerNorm(int(node_dim)),
            nn.SiLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(node_dim), int(latent_size)),
        )

    def _coerce_graph(self, x) -> SparseRIGSGraph:
        if isinstance(x, SparseRIGSGraph):
            return x
        if torch.is_tensor(x):
            return compute_sparse_rigs_graph(
                x,
                local_k=self.local_k,
                eps=self.eps,
            )
        if not isinstance(x, dict):
            raise TypeError(
                "RIGSNNEncoder expects either points tensor, SparseRIGSGraph, or a dict containing a precomputed "
                f"graph, got {type(x)}."
            )
        required_keys = {"radii", "edge_index", "edge_distance", "edge_cosine"}
        missing = sorted(required_keys - set(x.keys()))
        if missing:
            raise KeyError(
                "Precomputed RIGS graph dict is missing required keys. "
                f"Missing={missing}, available={sorted(x.keys())}."
            )
        edge_index = x["edge_index"]
        if not torch.is_tensor(edge_index):
            raise TypeError(f"Precomputed edge_index must be a torch.Tensor, got {type(edge_index)}.")
        return SparseRIGSGraph(
            radii=x["radii"],
            edge_index=edge_index.to(dtype=torch.long),
            edge_distance=x["edge_distance"],
            edge_cosine=x["edge_cosine"],
        )

    def _node_features(self, graph: SparseRIGSGraph) -> torch.Tensor:
        radial = self.radial_rbf(graph.radii)
        density = self.density_rbf(graph.edge_distance)
        density_hist = density.mean(dim=2)
        return torch.cat([radial, density_hist], dim=-1)

    def _edge_features(self, graph: SparseRIGSGraph) -> torch.Tensor:
        batch_size, num_points, num_edges = graph.edge_index.shape
        flat_index = graph.edge_index.reshape(batch_size, -1)
        radius_i = graph.radii.unsqueeze(-1)
        radius_j = graph.radii.gather(1, flat_index).reshape(batch_size, num_points, num_edges)
        angular_valid = (radius_i > self.eps) & (radius_j > self.eps)
        return torch.stack(
            [
                graph.edge_distance,
                graph.edge_cosine,
                (radius_i - radius_j).abs(),
                angular_valid.to(dtype=graph.edge_distance.dtype),
                torch.ones_like(graph.edge_distance),
                torch.ones_like(graph.edge_distance),
            ],
            dim=-1,
        )

    @staticmethod
    def _edge_summary(edge_features: torch.Tensor) -> torch.Tensor:
        return edge_features.mean(dim=(1, 2))

    def _model_dtype(self) -> torch.dtype:
        return self.input_proj[0].weight.dtype

    def forward(self, x):
        graph = self._coerce_graph(x)
        model_dtype = self._model_dtype()
        edge_index = graph.edge_index
        edge_features = self._edge_features(graph).to(dtype=model_dtype)
        nodes = self.input_proj(self._node_features(graph).to(dtype=model_dtype))

        point_feats = [nodes]
        for layer in self.layers:
            nodes = layer(nodes, edge_index, edge_features)
            point_feats.append(nodes)

        point_feat = self.point_fuse(torch.cat(point_feats, dim=-1))
        pooled_mean = point_feat.mean(dim=1)
        pooled_attn = (
            F.softmax(self.pool_gate(point_feat).squeeze(-1), dim=1).unsqueeze(-1) * point_feat
        ).sum(dim=1)
        edge_summary = self._edge_summary(edge_features)
        inv_input = torch.cat([pooled_mean, pooled_attn, edge_summary.to(dtype=pooled_mean.dtype)], dim=-1)
        latent = self.inv_head(inv_input)
        return latent, None, None


__all__ = [
    "RIGSGeometry",
    "SparseRIGSGraph",
    "RIGSEncoder",
    "RIGSNNEncoder",
    "compute_rigs_geometry",
    "compute_sparse_rigs_graph",
    "orthogonal_procrustes_rmsd",
    "reconstruct_points_from_distance_matrix",
]
