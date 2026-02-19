from __future__ import annotations

import torch
import torch.nn as nn

from ..base import Encoder
from ..registry import register_encoder
from .vn_encoders import index_points, knn


def _validate_layer_inputs(
    h: torch.Tensor,
    x: torch.Tensor,
    edge_idx: torch.Tensor,
    *,
    layer_name: str,
) -> tuple[int, int, int]:
    if h.dim() != 3:
        raise ValueError(f"{layer_name}: expected h with shape (B,N,H), got {tuple(h.shape)}")
    if x.dim() != 3:
        raise ValueError(f"{layer_name}: expected x with shape (B,N,3), got {tuple(x.shape)}")
    if x.shape[-1] != 3:
        raise ValueError(f"{layer_name}: expected x last dimension to be 3, got shape {tuple(x.shape)}")
    if edge_idx.dim() != 3:
        raise ValueError(
            f"{layer_name}: expected edge_idx with shape (B,N,K), got {tuple(edge_idx.shape)}"
        )
    if edge_idx.dtype != torch.long:
        raise TypeError(
            f"{layer_name}: edge_idx must be torch.long for indexing, got dtype={edge_idx.dtype}"
        )

    b, n, hidden_dim = h.shape
    if x.shape[0] != b or x.shape[1] != n:
        raise ValueError(
            f"{layer_name}: h/x batch or point mismatch, h={tuple(h.shape)}, x={tuple(x.shape)}"
        )
    if edge_idx.shape[0] != b or edge_idx.shape[1] != n:
        raise ValueError(
            f"{layer_name}: edge_idx must match batch and points of h/x, "
            f"h={tuple(h.shape)}, edge_idx={tuple(edge_idx.shape)}"
        )
    k = int(edge_idx.shape[2])
    if k <= 0:
        raise ValueError(f"{layer_name}: edge_idx must have at least one neighbor, got k={k}")
    return b, n, hidden_dim


class EGNNLayer(nn.Module):
    """
    E(n)-equivariant message-passing layer on a fixed neighborhood graph.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError(f"EGNNLayer: hidden_dim must be > 0, got {hidden_dim}")
        hidden_dim = int(hidden_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, _ = _validate_layer_inputs(h, x, edge_idx, layer_name=self.__class__.__name__)
        k = int(edge_idx.shape[2])

        h_neigh = index_points(h, edge_idx)  # (B, N, K, H)
        x_neigh = index_points(x, edge_idx)  # (B, N, K, 3)

        h_i = h.unsqueeze(2).expand(-1, -1, k, -1)
        x_i = x.unsqueeze(2).expand(-1, -1, k, -1)

        sq_dist = ((x_i - x_neigh) ** 2).sum(dim=-1, keepdim=True)
        edge_feat = torch.cat([h_i, h_neigh, sq_dist], dim=-1)
        m_ij = self.edge_mlp(edge_feat)

        coord_weights = self.coord_mlp(m_ij)
        x_diff = x_i - x_neigh
        x_update = (x_diff * coord_weights).mean(dim=2)
        x_new = x + x_update

        m_i = m_ij.sum(dim=2)
        h_new = self.node_mlp(torch.cat([h, m_i], dim=-1)) + h
        return h_new, x_new


class FastEGNNLayer(nn.Module):
    """
    EGNN layer with factorized edge projection to reduce memory/FLOPs.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError(f"FastEGNNLayer: hidden_dim must be > 0, got {hidden_dim}")
        hidden_dim = int(hidden_dim)

        # Factorized equivalent of first edge linear:
        # Linear([h_i, h_j, dist2]) == W_i h_i + W_j h_j + W_d dist2 + b
        self.edge_proj_i = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_proj_j = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_proj_dist = nn.Linear(1, hidden_dim, bias=True)

        self.edge_mlp_rest = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, _ = _validate_layer_inputs(h, x, edge_idx, layer_name=self.__class__.__name__)

        h_i_proj = self.edge_proj_i(h).unsqueeze(2)  # (B, N, 1, H)
        h_j_proj = index_points(self.edge_proj_j(h), edge_idx)  # (B, N, K, H)

        x_neigh = index_points(x, edge_idx)  # (B, N, K, 3)
        x_diff = x.unsqueeze(2) - x_neigh
        sq_dist = (x_diff ** 2).sum(dim=-1, keepdim=True)
        dist_proj = self.edge_proj_dist(sq_dist)

        m_ij = h_i_proj + h_j_proj + dist_proj
        m_ij = self.edge_mlp_rest(m_ij)

        coord_weights = self.coord_mlp(m_ij)
        x_update = (x_diff * coord_weights).mean(dim=2)
        x_new = x + x_update

        m_i = m_ij.sum(dim=2)
        h_new = self.node_mlp(torch.cat([h, m_i], dim=-1)) + h
        return h_new, x_new


@register_encoder("EGNN_Encoder")
class EGNNEncoder(Encoder):
    """
    E(3)-equivariant encoder that returns:
      - inv_z: (B, latent_size)
      - eq_z: (B, latent_size, 3)
      - center_loc: (B, 3)
    """

    def __init__(
        self,
        latent_size: int = 256,
        n_knn: int = 20,
        num_layers: int = 6,
        hidden_dim: int = 128,
        pooling: str = "mean",
        use_fast_layer: bool = False,
    ) -> None:
        super().__init__()

        latent_size = int(latent_size)
        n_knn = int(n_knn)
        num_layers = int(num_layers)
        hidden_dim = int(hidden_dim)
        pooling = str(pooling).lower()

        if latent_size <= 0:
            raise ValueError(f"EGNNEncoder: latent_size must be > 0, got {latent_size}")
        if n_knn <= 0:
            raise ValueError(f"EGNNEncoder: n_knn must be > 0, got {n_knn}")
        if num_layers <= 0:
            raise ValueError(f"EGNNEncoder: num_layers must be > 0, got {num_layers}")
        if hidden_dim <= 0:
            raise ValueError(f"EGNNEncoder: hidden_dim must be > 0, got {hidden_dim}")
        if pooling not in {"mean", "max"}:
            raise ValueError(f"EGNNEncoder: unsupported pooling='{pooling}', expected 'mean' or 'max'")

        self.n_knn = n_knn
        self.pooling = pooling
        self.use_fast_layer = bool(use_fast_layer)

        self.h_init = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        layer_cls: type[nn.Module] = FastEGNNLayer if self.use_fast_layer else EGNNLayer
        self.layers = nn.ModuleList([layer_cls(hidden_dim) for _ in range(num_layers)])

        self.out_inv_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim * 2, latent_size),
        )
        self.eq_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, latent_size),
        )

    @staticmethod
    def _coerce_input_points(x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"EGNNEncoder: expected input with 3 dims, got shape {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError(
                f"EGNNEncoder: expected floating-point input tensor, got dtype={x.dtype} shape={tuple(x.shape)}"
            )
        if x.shape[-1] == 3:
            return x
        if x.shape[1] == 3:
            return x.transpose(1, 2).contiguous()
        raise ValueError(
            "EGNNEncoder: unable to infer point-cloud layout. "
            f"Expected shape (B,N,3) or (B,3,N), got {tuple(x.shape)}"
        )

    @staticmethod
    def _validate_edge_idx(edge_idx: torch.Tensor, *, num_points: int) -> None:
        if edge_idx.dtype != torch.long:
            raise TypeError(f"EGNNEncoder: knn returned dtype={edge_idx.dtype}, expected torch.long")
        if edge_idx.numel() == 0:
            raise ValueError("EGNNEncoder: knn returned an empty edge index tensor.")
        min_idx = int(edge_idx.min().item())
        max_idx = int(edge_idx.max().item())
        if min_idx < 0 or max_idx >= num_points:
            raise ValueError(
                "EGNNEncoder: knn produced out-of-range neighbor indices: "
                f"min={min_idx}, max={max_idx}, valid=[0,{num_points - 1}]"
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        points = self._coerce_input_points(x)
        batch_size, num_points, _ = points.shape
        if num_points <= 0:
            raise ValueError(f"EGNNEncoder: point cloud must contain at least one point, got {num_points}")
        if self.n_knn > num_points:
            raise ValueError(
                f"EGNNEncoder: n_knn ({self.n_knn}) cannot exceed number of input points ({num_points})."
            )

        center_loc = points.mean(dim=1, keepdim=True)
        x_centered = points - center_loc

        h = (x_centered ** 2).sum(dim=-1, keepdim=True)
        h = self.h_init(h)

        x_coord_knn = points.transpose(1, 2).contiguous()
        edge_idx = knn(x_coord_knn, k=self.n_knn)
        self._validate_edge_idx(edge_idx, num_points=num_points)

        for layer in self.layers:
            h, x_centered = layer(h, x_centered, edge_idx)

        if self.pooling == "max":
            h_pooled = h.max(dim=1).values
        else:
            h_pooled = h.mean(dim=1)
        inv_z = self.out_inv_mlp(h_pooled)

        weights = self.eq_projector(h)
        eq_z = torch.einsum("bnc,bnd->bcd", weights, x_centered) / float(num_points)

        expected_shape = (batch_size, inv_z.shape[1], 3)
        if tuple(eq_z.shape) != expected_shape:
            raise RuntimeError(
                "EGNNEncoder: unexpected eq_z shape after projection, "
                f"got {tuple(eq_z.shape)}, expected {expected_shape}."
            )
        return inv_z, eq_z, center_loc.squeeze(1)
