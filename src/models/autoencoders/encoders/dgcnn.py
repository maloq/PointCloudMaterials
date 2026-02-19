from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import Encoder
from ..registry import register_encoder


def _knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Return indices of k nearest neighbors for each point."""
    # x: (B, C, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = xx.transpose(2, 1) - 2.0 * torch.matmul(x.transpose(2, 1), x) + xx
    idx = pairwise_distance.topk(k=k, dim=-1, largest=False, sorted=False).indices
    return idx


def _get_graph_feature(x: torch.Tensor, k: int, idx: torch.Tensor | None = None) -> torch.Tensor:
    """Construct edge features [x_j - x_i, x_i]."""
    # x: (B, C, N)
    batch_size, channels, num_points = x.size()
    if idx is None:
        idx = _knn(x, k=k)

    idx_base = torch.arange(batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).view(-1)

    x_t = x.transpose(2, 1).contiguous()  # (B, N, C)
    neighbors = x_t.view(batch_size * num_points, channels)[idx, :]
    neighbors = neighbors.view(batch_size, num_points, k, channels)
    center = x_t.view(batch_size, num_points, 1, channels).expand(-1, -1, k, -1)

    feature = torch.cat((neighbors - center, center), dim=3)  # (B, N, k, 2C)
    return feature.permute(0, 3, 1, 2).contiguous()  # (B, 2C, N, k)


def _norm2d(channels: int, use_batchnorm: bool) -> nn.Module:
    return nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()


def _norm1d(channels: int, use_batchnorm: bool) -> nn.Module:
    return nn.BatchNorm1d(channels) if use_batchnorm else nn.Identity()


@register_encoder("DGCNN")
class DGCNNEncoder(Encoder):
    """Regular (non-VN) DGCNN encoder producing only invariant latent code."""

    expects_channel_first = True

    def __init__(
        self,
        latent_size: int = 96,
        *,
        n_knn: int = 20,
        feature_dims: Tuple[int, int, int, int] = (64, 64, 128, 256),
        emb_dims: int = 512,
        pooling: str = "mean",
        dropout_rate: float = 0.2,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.n_knn = int(n_knn)
        self.pooling = str(pooling).lower()
        if self.pooling not in {"max", "mean"}:
            raise ValueError(f"Unsupported pooling='{pooling}'. Expected 'max' or 'mean'.")

        c1, c2, c3, c4 = [int(v) for v in feature_dims]
        if min(c1, c2, c3, c4, emb_dims, latent_size) <= 0:
            raise ValueError("All channel sizes must be positive.")

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, c1, kernel_size=1, bias=False),
            _norm2d(c1, use_batchnorm),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1 * 2, c2, kernel_size=1, bias=False),
            _norm2d(c2, use_batchnorm),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(c2 * 2, c3, kernel_size=1, bias=False),
            _norm2d(c3, use_batchnorm),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(c3 * 2, c4, kernel_size=1, bias=False),
            _norm2d(c4, use_batchnorm),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(c1 + c2 + c3 + c4, emb_dims, kernel_size=1, bias=False),
            _norm1d(emb_dims, use_batchnorm),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        hidden = max(256, emb_dims // 2)
        self.head = nn.Sequential(
            nn.Linear(emb_dims * 2, hidden, bias=False),
            _norm1d(hidden, use_batchnorm),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(p=float(dropout_rate)),
            nn.Linear(hidden, latent_size),
        )

    def _neighbor_pool(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N, k)
        if self.pooling == "mean":
            return x.mean(dim=-1)
        return x.max(dim=-1).values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is channel-first point cloud: (B, 3, N)
        if x.dim() != 3 or x.shape[1] != 3:
            raise ValueError(f"DGCNN encoder expects input shape (B,3,N), got {tuple(x.shape)}")

        x1 = self._neighbor_pool(self.conv1(_get_graph_feature(x, k=self.n_knn)))
        x2 = self._neighbor_pool(self.conv2(_get_graph_feature(x1, k=self.n_knn)))
        x3 = self._neighbor_pool(self.conv3(_get_graph_feature(x2, k=self.n_knn)))
        x4 = self._neighbor_pool(self.conv4(_get_graph_feature(x3, k=self.n_knn)))

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        x_feat = self.conv5(x_cat)

        x_max = F.adaptive_max_pool1d(x_feat, 1).squeeze(-1)
        x_avg = F.adaptive_avg_pool1d(x_feat, 1).squeeze(-1)
        global_feat = torch.cat((x_max, x_avg), dim=1)
        latent = self.head(global_feat)
        return latent
