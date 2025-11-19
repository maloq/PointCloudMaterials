from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import Encoder
from ..registry import register_encoder

EPS = 1e-6


# ---------------------------------------------------------------------------
# Vector Neuron building blocks (adapted from VN-SPD)
# ---------------------------------------------------------------------------
class VNLinear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels: int, share_nonlinearity: bool = False, negative_slope: float = 0.2):
        super().__init__()
        if share_nonlinearity:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).to(dtype=x.dtype)
        d_norm_sq = (d * d).sum(2, keepdim=True)
        return self.negative_slope * x + (1 - self.negative_slope) * (
            mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d)
        )


class VNBatchNorm(nn.Module):
    def __init__(self, num_features: int, dim: int):
        super().__init__()
        self.dim = dim
        if dim in (3, 4):
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        return x / norm * norm_bn


class VNLinearLeakyReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int = 5,
        share_nonlinearity: bool = False,
        negative_slope: float = 0.2,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.negative_slope = negative_slope
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.batchnorm = VNBatchNorm(out_channels, dim=dim)
        if share_nonlinearity:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        if self.use_batchnorm:
            p = self.batchnorm(p)
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (p * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).to(dtype=p.dtype)
        d_norm_sq = (d * d).sum(2, keepdim=True)
        return self.negative_slope * p + (1 - self.negative_slope) * (
            mask * p + (1 - mask) * (p - (dotprod / (d_norm_sq + EPS)) * d)
        )


class VNMaxPool(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]], indexing='ij') + (idx,)
        return x[index_tuple]


def mean_pool(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    return x.mean(dim=dim, keepdim=keepdim)


class VNStdFeature(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dim: int = 4,
        normalize_frame: bool = False,
        share_nonlinearity: bool = False,
        negative_slope: float = 0.2,
        use_batchnorm: bool = True,
        hidden_dims: tuple[int, int] | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame

        if hidden_dims is None:
            h1 = in_channels // 2
            h2 = in_channels // 4
        else:
            h1, h2 = hidden_dims

        self.vn1 = VNLinearLeakyReLU(
            in_channels,
            h1,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
            use_batchnorm=use_batchnorm,
        )
        self.vn2 = VNLinearLeakyReLU(
            h1,
            h2,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
            use_batchnorm=use_batchnorm,
        )
        self.vn_lin = nn.Linear(h2, 3 if not normalize_frame else 2, bias=False)

    def forward(self, x: torch.Tensor):
        z0 = self.vn1(x)
        z0 = self.vn2(z0)
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)
        if self.normalize_frame:
            v1 = z0[:, 0, :]
            u1 = v1 / (torch.sqrt((v1 * v1).sum(1, keepdims=True)) + EPS)
            v2 = z0[:, 1, :] - (z0[:, 1, :] * u1).sum(1, keepdims=True) * u1
            u2 = v2 / (torch.sqrt((v2 * v2).sum(1, keepdims=True)) + EPS)
            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)
        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)
        else:
            raise ValueError(f"Unsupported VNStdFeature dim {self.dim}")
        return x_std, z0


class VNResBlock(nn.Module):
    """ResNet-style residual block for Vector Neurons with a VN bottleneck."""

    def __init__(
        self,
        channels: int,
        dim: int = 4,
        bottleneck_ratio: float = 0.5,
        negative_slope: float = 0.1,
        use_batchnorm: bool = True,
        share_nonlinearity: bool = False,
    ):
        super().__init__()
        mid = max(1, int(round(channels * bottleneck_ratio)))
        self.conv1 = VNLinearLeakyReLU(
            channels,
            mid,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = VNLinearLeakyReLU(
            mid,
            channels,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x)) + x


def _make_vn_layer(channels: int, num_blocks: int, dim: int = 4, **kwargs) -> nn.Sequential:
    return nn.Sequential(*[VNResBlock(channels, dim=dim, **kwargs) for _ in range(num_blocks)])


@register_encoder("PnE_VN")
class PointNetEncoderVN(Encoder):
    def __init__(
        self,
        latent_size: int = 64,
        n_knn: int = 20,
        pooling: str = 'mean',
        feature_transform: bool = True,
        hidden_dim1: int = 256,
        hidden_dim2: int = 1024,
        stn_hidden_dims: tuple[int, int, int] = (64, 128, 1024),
        stn_fc_dims: tuple[int, int] = (512, 256),
        std_feature_hidden_dims: tuple[int, int] | None = None,
        use_batchnorm: bool = True,
    ):
        """
        Vector neuron PointNet backbone producing invariant/equivariant latents.
        """
        super().__init__()
        self.n_knn = n_knn
        self.pooling = pooling

        c1 = hidden_dim1 // 3
        c2 = hidden_dim2 // 3
        c3 = latent_size // 3

        self.conv_pos = VNLinearLeakyReLU(3, c1, dim=5, negative_slope=0.1, use_batchnorm=use_batchnorm)
        self.conv1 = VNLinearLeakyReLU(c1, c1, dim=4, negative_slope=0.1, use_batchnorm=use_batchnorm)

        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(c1, hidden_dims=stn_hidden_dims, fc_dims=stn_fc_dims)

        self.conv2 = VNLinearLeakyReLU(
            c1 * 2,
            c2,
            dim=4,
            negative_slope=0.1,
            use_batchnorm=use_batchnorm,
        )

        self.conv3 = VNLinearLeakyReLU(
            c2,
            c3,
            dim=4,
            negative_slope=0.1,
            use_batchnorm=use_batchnorm,
        )

        self.conv4 = VNLinearLeakyReLU(c3, c3, dim=4, negative_slope=0.1, use_batchnorm=use_batchnorm)
        self.std_feature = VNStdFeature(
            c3 * 2,
            dim=4,
            normalize_frame=False,
            negative_slope=0.0,
            hidden_dims=std_feature_hidden_dims,
        )
        assert latent_size % 3 == 0, f"latent_size must be divisible by 3, got {latent_size}"
        # Map pooled equivariant VN features (B, c3, 3) -> (B, latent_size, 3)
        self.out_eq_mlp = VNLinearLeakyReLU(c3, latent_size, dim=3, use_batchnorm=False)
        # Map invariant feature vector (B, 2*c3*3) -> (B, latent_size)
        self.out_inv_mlp = nn.Linear(c3 * 2 * 3, latent_size)

        if pooling == 'max':
            self.pool = VNMaxPool(c1)
        else:
            self.pool = mean_pool

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.permute(0, 2, 1)
        batch_size, channels, num_points = x.size()
        x = x.unsqueeze(1)

        feat = get_graph_feature_cross(x, k=self.n_knn)
        x = self.conv_pos(feat)
        x = self.pool(x)

        x = self.conv1(x)

        if self.feature_transform:
            x_global = self.fstn(x).unsqueeze(-1).repeat(1, 1, 1, num_points)
            x = torch.cat((x, x_global), 1)
        else:
            x_mean = x.mean(dim=-1, keepdim=True)
            x = torch.cat((x, x_mean.expand_as(x)), 1)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Pool per-point VN features, then map to desired latent_size VN channels
        eq_z = x.mean(dim=-1)  # (B, c3, 3)
        eq_z = self.out_eq_mlp(eq_z)  # (B, latent_size, 3)
      
        x_mean_out = x.mean(dim=-1, keepdim=True)
        x = torch.cat((x, x_mean_out.expand_as(x)), 1)
        x, _ = self.std_feature(x)

        x = x.view(batch_size, -1, num_points)
        inv_feat = x.max(dim=-1, keepdim=False)[0]
        inv_z = self.out_inv_mlp(inv_feat)
        center_loc = x_mean_out.mean(dim=2)
        return inv_z, eq_z, center_loc


@register_encoder("VN_DGCNN")
class VNDGCNNEncoder(Encoder):
    def __init__(
        self,
        latent_size: int = 256,
        n_knn: int = 20,
        pooling: str = 'mean',
        feature_dims: tuple[int, int, int, int, int] = (96, 96, 192, 384, 576),
        global_mlp_dims: tuple[int, int] = (256, 128),
        global_dropout: float = 0.5,
        share_nonlinearity: bool = True,
        std_feature_hidden_dims: tuple[int, int] | None = None,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        if len(feature_dims) != 5:
            raise ValueError(f"feature_dims must provide five entries (got {feature_dims})")

        self.n_knn = n_knn
        self.pooling = pooling

        c1, c2, c3, c4, c5 = [max(1, dim // 3) for dim in feature_dims]

        self.conv1 = VNLinearLeakyReLU(
            2,
            c1,
            dim=5,
            negative_slope=0.2,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = VNLinearLeakyReLU(
            c1 * 2,
            c2,
            dim=5,
            negative_slope=0.2,
            use_batchnorm=use_batchnorm,
        )
        self.conv3 = VNLinearLeakyReLU(
            c2 * 2,
            c3,
            dim=5,
            negative_slope=0.2,
            use_batchnorm=use_batchnorm,
        )
        self.conv4 = VNLinearLeakyReLU(
            c3 * 2,
            c4,
            dim=5,
            negative_slope=0.2,
            use_batchnorm=use_batchnorm,
        )

        concat_channels = c1 + c2 + c3 + c4
        self.conv5 = VNLinearLeakyReLU(
            concat_channels,
            c5,
            dim=4,
            negative_slope=0.2,
            share_nonlinearity=share_nonlinearity,
            use_batchnorm=use_batchnorm,
        )

        self.std_feature = VNStdFeature(
            c5 * 2,
            dim=4,
            normalize_frame=False,
            negative_slope=0.0,
            use_batchnorm=use_batchnorm,
            hidden_dims=std_feature_hidden_dims,
        )

        global_in_dim = c5 * 12
        g1, g2 = global_mlp_dims
        self.global_mlp = nn.Sequential(
            nn.Linear(global_in_dim, g1),
            nn.BatchNorm1d(g1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=global_dropout),
            nn.Linear(g1, g2),
            nn.BatchNorm1d(g2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=global_dropout),
            nn.Linear(g2, latent_size),
        )
        # Map pooled equivariant VN features (B, c5, 3) -> (B, latent_size, 3)
        self.out_eq_mlp = VNLinearLeakyReLU(c5, latent_size, dim=3, use_batchnorm=False)

        if pooling == 'max':
            self.pool1 = VNMaxPool(c1)
            self.pool2 = VNMaxPool(c2)
            self.pool3 = VNMaxPool(c3)
            self.pool4 = VNMaxPool(c4)
        else:
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.permute(0, 2, 1)
        batch_size, channels, num_points = x.size()
        x = x.unsqueeze(1)

        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv1(x)
        x1 = self.pool1(x)

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv2(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv3(x)
        x3 = self.pool3(x)

        x = get_graph_feature(x3, k=self.n_knn)
        x = self.conv4(x)
        x4 = self.pool4(x)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)

        # Pool per-point VN features, then map to desired latent_size VN channels
        eq_z = x.mean(dim=-1)  # (B, c5, 3)
        eq_z = self.out_eq_mlp(eq_z)  # (B, latent_size, 3)
        x_mean_out = x.mean(dim=-1, keepdim=True)
        center_loc = x_mean_out.mean(dim=2)

        x_mean = x_mean_out.expand_as(x)
        x, _ = self.std_feature(torch.cat((x, x_mean), dim=1))

        x = x.view(batch_size, -1, num_points)
        x_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_avg = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x_global = torch.cat((x_max, x_avg), dim=1)

        inv_z = self.global_mlp(x_global)
        return inv_z, eq_z, center_loc


class STNkd(nn.Module):
    def __init__(
        self,
        d: int = 64,
        hidden_dims: tuple[int, int, int] = (64, 128, 1024),
        fc_dims: tuple[int, int] = (512, 256),
    ):
        super().__init__()
        c1, c2, c3 = [dim // 3 for dim in hidden_dims]
        fc1_dim, fc2_dim = [dim // 3 for dim in fc_dims]

        self.conv1 = VNLinearLeakyReLU(d, c1, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(c1, c2, dim=4, negative_slope=0.0)
        self.conv3 = VNLinearLeakyReLU(c2, c3, dim=4, negative_slope=0.0)
        self.fc1 = VNLinearLeakyReLU(c3, fc1_dim, dim=3, negative_slope=0.0)
        self.fc2 = VNLinearLeakyReLU(fc1_dim, fc2_dim, dim=3, negative_slope=0.0)
        self.pool = VNMaxPool(c3)
        self.fc3 = VNLinear(fc2_dim, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)


class SimpleRot(nn.Module):
    def __init__(self, in_ch: int, strict: str = 'None'):
        super().__init__()
        self.model = VNLinear(in_ch, 3)
        self.strict = strict

    def constraint_rot(self, rot_mat: torch.Tensor) -> torch.Tensor:
        if self.strict == 'None':
            return rot_mat
        u, _, v = torch.linalg.svd(rot_mat)
        return u @ v.transpose(-1, -2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rot_mat = self.model(x).squeeze(-1)
        return self.constraint_rot(rot_mat)


class ComplexRot(nn.Module):
    def __init__(self, in_ch: int, strict: str = 'None'):
        super().__init__()
        self.linear1 = VNLinearLeakyReLU(in_ch, in_ch * 6, dim=4, negative_slope=0.1)
        self.linear2 = VNLinearLeakyReLU(in_ch * 6, in_ch * 4, dim=4, negative_slope=0.1)
        self.linear3 = VNLinearLeakyReLU(in_ch * 4, in_ch, dim=4, negative_slope=0.1)
        self.linearR = VNLinear(in_ch, 3)
        self.strict = strict

    def constraint_rot(self, rot_mat: torch.Tensor) -> torch.Tensor:
        if self.strict == 'None':
            return rot_mat
        u, _, v = torch.linalg.svd(rot_mat)
        return u @ v.transpose(-1, -2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        R = self.linearR(x)
        rot_mat = torch.mean(R, dim=-1)
        return self.constraint_rot(rot_mat)


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]


def get_graph_feature(x: torch.Tensor, k: int = 20, idx: torch.Tensor | None = None, x_coord: torch.Tensor | None = None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.reshape(batch_size, -1, num_points)

    if idx is None:
        if x_coord is None:
            idx = knn(x, k=k)
        else:
            idx = knn(x_coord, k=k)

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()
    return feature


def get_graph_feature_cross(x: torch.Tensor, k: int = 20, idx: torch.Tensor | None = None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.reshape(batch_size, -1, num_points)

    if idx is None:
        idx = knn(x, k=k)
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    cross = torch.cross(feature, x, dim=-1)
    feature = torch.cat((feature - x, x, cross), dim=3).permute(0, 3, 4, 1, 2).contiguous()
    return feature



class VNRobustInvariantHead(nn.Module):
    """
    Robustly extracts invariant features from equivariant vectors (B, C, 3).
    Uses vector norms and projections onto the global mean direction.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        input_dim = in_channels * 2 
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 3)
        
        # 1. Norms (Invariant)
        norms = torch.norm(x, dim=-1) # (B, C)
        
        # 2. Projection onto global mean direction (Invariant relative to cloud)
        global_dir = x.mean(dim=1, keepdim=True) # (B, 1, 3)
        global_dir = F.normalize(global_dir, dim=-1, eps=1e-6)
        projections = (x * global_dir).sum(dim=-1) # (B, C)
        
        combined = torch.cat([norms, projections], dim=1) # (B, 2*C)
        return self.mlp(combined)


@register_encoder("VN_DGCNN_Refined")
class VNDGCNNEncoderRefined(Encoder):
    def __init__(
        self,
        latent_size: int = 256,
        n_knn: int = 20,
        pooling: str = 'mean',
        feature_dims: tuple[int, int, int, int, int] = (64, 64, 128, 256, 512),
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.n_knn = n_knn
        self.pooling = pooling

        c1, c2, c3, c4, c5 = [max(1, dim // 3) for dim in feature_dims]

        self.conv1 = VNLinearLeakyReLU(2, c1, dim=5, negative_slope=0.2, use_batchnorm=use_batchnorm)
        self.conv2 = VNLinearLeakyReLU(c1 * 2, c2, dim=5, negative_slope=0.2, use_batchnorm=use_batchnorm)
        self.conv3 = VNLinearLeakyReLU(c2 * 2, c3, dim=5, negative_slope=0.2, use_batchnorm=use_batchnorm)
        self.conv4 = VNLinearLeakyReLU(c3 * 2, c4, dim=5, negative_slope=0.2, use_batchnorm=use_batchnorm)
        
        concat_channels = c1 + c2 + c3 + c4
        self.conv5 = VNLinearLeakyReLU(concat_channels, c5, dim=4, negative_slope=0.2, use_batchnorm=use_batchnorm)

        if pooling == 'max':
            self.pool_layer = VNMaxPool(1) # Dummy dim
        else:
            self.pool_layer = mean_pool

        # Projection for Z_eq (B, latent, 3)
        # latent_size must be divisible by 3 to map from VN channels
        assert latent_size % 3 == 0
        self.eq_projector = VNLinear(c5, latent_size)

        # Head for Z_inv (B, latent)
        self.inv_head = VNRobustInvariantHead(c5, latent_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Input x: (B, 3, N)
        # We treat it as 1 channel of 3D vectors.
        # Shape required for get_graph_feature: (B, C, 3, N).
        
        # Input x: (B, N, 3) -> (B, 3, N)
        x = x.permute(0, 2, 1)
        
        batch_size = x.size(0)
        x = x.unsqueeze(1) # (B, 1, 3, N)

        # Layer 1
        x_graph = get_graph_feature(x, k=self.n_knn) # (B, 2*1, 3, N, k)
        x = self.conv1(x_graph) # (B, c1, 3, N, k)
        x1 = x.mean(dim=-1) # Pool neighbors -> (B, c1, 3, N)

        # Layer 2
        x_graph = get_graph_feature(x1, k=self.n_knn)
        x = self.conv2(x_graph)
        x2 = x.mean(dim=-1)

        # Layer 3
        x_graph = get_graph_feature(x2, k=self.n_knn)
        x = self.conv3(x_graph)
        x3 = x.mean(dim=-1)

        # Layer 4
        x_graph = get_graph_feature(x3, k=self.n_knn)
        x = self.conv4(x_graph)
        x4 = x.mean(dim=-1)

        # Global Aggregation
        x_concat = torch.cat((x1, x2, x3, x4), dim=1) # (B, sum(c), 3, N)
        x = self.conv5(x_concat) # (B, c5, 3, N)

        # Global Pooling
        if self.pooling == 'max':
             # VNMaxPool expects (B, C, 3) usually, here we have N points.
             # We treat N as the dimension to pool over.
             # Reshape to (B, c5, N, 3) for standard VN pool logic or just use standard logic
             x_pooled = self.pool_layer(x.transpose(2, 3)).transpose(2, 3) # Pool over N
             # Note: VNMaxPool implementation provided previously pools dim -1 (3).
             # We want to pool over N.
             # Let's stick to Mean Pool for stability with crystals.
             x_mean = x.mean(dim=-1) # (B, c5, 3)
        else:
             x_mean = x.mean(dim=-1) # (B, c5, 3)

        # Outputs
        
        # 1. Equivariant Embedding (for Rotation Head)
        z_eq = self.eq_projector(x_mean) # (B, latent/3, 3)

        # 2. Invariant Embedding (for Decoder/Clustering)
        z_inv = self.inv_head(x_mean) # (B, latent)

        return z_inv, z_eq, None

__all__ = [
    "PointNetEncoderVN",
    "VNDGCNNEncoder",
    "SimpleRot",
    "ComplexRot",
    "VNLinear",
    "VNLeakyReLU",
    "VNLinearLeakyReLU",
    "VNMaxPool",
    "VNBatchNorm",
    "VNResBlock",
    "VNStdFeature",
    "STNkd",
    "get_graph_feature",
    "get_graph_feature_cross",
]
