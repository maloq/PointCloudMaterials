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
def _softmax_fp32(scores: torch.Tensor, dim: int) -> torch.Tensor:
    if scores.dtype in (torch.float16, torch.bfloat16):
        return torch.softmax(scores.float(), dim=dim).to(scores.dtype)
    return torch.softmax(scores, dim=dim)


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
        index_tuple = torch.meshgrid([torch.arange(j, device=x.device) for j in x.size()[:-1]], indexing='ij') + (idx,)
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
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.5,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        input_dim = in_channels * 2 
        norm_layer = nn.BatchNorm1d(out_channels) if use_batchnorm else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, out_channels),
            norm_layer,
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=dropout_rate),
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
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.n_knn = n_knn
        self.pooling = pooling
        self.dropout_rate = dropout_rate

        c1, c2, c3, c4, c5 = [max(1, dim // 3) for dim in feature_dims]

        self.conv1 = VNLinearLeakyReLU(2, c1, dim=5, negative_slope=0.2, use_batchnorm=use_batchnorm)
        self.conv2 = VNLinearLeakyReLU(c1 * 2, c2, dim=5, negative_slope=0.2, use_batchnorm=use_batchnorm)
        self.conv3 = VNLinearLeakyReLU(c2 * 2, c3, dim=5, negative_slope=0.2, use_batchnorm=use_batchnorm)
        self.conv4 = VNLinearLeakyReLU(c3 * 2, c4, dim=5, negative_slope=0.2, use_batchnorm=use_batchnorm)
        
        concat_channels = c1 + c2 + c3 + c4
        self.conv5 = VNLinearLeakyReLU(concat_channels, c5, dim=4, negative_slope=0.2, use_batchnorm=use_batchnorm)

        if pooling == 'max':
            self.pool_layer = VNMaxPool(c5) 
        else:
            self.pool_layer = mean_pool

        # Projection for Z_eq (B, latent, 3)
        # latent_size must be divisible by 3 to map from VN channels
        assert latent_size % 3 == 0
        self.eq_projector = VNLinear(c5, latent_size)
        # Initialize with larger gain to produce useful equivariant outputs
        nn.init.xavier_uniform_(self.eq_projector.map_to_feat.weight, gain=2.0)

        # Head for Z_inv (B, latent)
        self.inv_head = VNRobustInvariantHead(
            c5,
            latent_size,
            dropout_rate=dropout_rate,
            use_batchnorm=use_batchnorm,
        )

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
             # VNMaxPool pools over the last dimension (N)
             x_mean = self.pool_layer(x) # (B, c5, 3)
        else:
             x_mean = x.mean(dim=-1) # (B, c5, 3)

        # Outputs
        
        # 1. Equivariant Embedding (for Rotation Head)
        z_eq = self.eq_projector(x_mean) # (B, latent/3, 3)

        # 2. Invariant Embedding (for Decoder/Clustering)
        z_inv = self.inv_head(x_mean) # (B, latent)

        return z_inv, z_eq, None


import torch
import torch.nn as nn
import torch.nn.functional as F
from .vn_encoders import VNLinearLeakyReLU, VNLinear, VNStdFeature, VNBatchNorm

# ---------------------------------------------------------------------------
# 1. Invariant Geometric Feature Extractor
# ---------------------------------------------------------------------------
class InvariantGeometricFeatures(nn.Module):
    """
    Explicitly computes topological/geometric invariants:
    1. Pairwise distances (Edge lengths)
    2. Local densities
    3. (Optional) Triplet angles
    """
    def __init__(self, num_points, n_knn=20):
        super().__init__()
        self.n_knn = n_knn
        # Project geometric features to a higher dim
        self.dist_embed = nn.Sequential(
            nn.Conv2d(1, 16, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        # x: (B, 3, N) - Input Points
        B, C, N = x.shape
        
        # 1. Compute Pairwise Distance Matrix (Invariant to Rotation)
        # (B, N, N)
        dist_mat = torch.cdist(x.transpose(1, 2), x.transpose(1, 2))
        
        # 2. Extract k-NN distances (Local Topology)
        # We sort distances to get 'canonical' local neighborhood descriptions
        knn_dists, _ = torch.topk(dist_mat, k=self.n_knn, dim=-1, largest=False)
        
        # (B, 1, N, k)
        knn_dists = knn_dists.unsqueeze(1) 
        
        # Embed these scalar invariants
        feat = self.dist_embed(knn_dists) # (B, 16, N, k)
        
        # Pool to get per-point invariant descriptors
        feat = feat.max(dim=-1)[0] # (B, 16, N)
        return feat

# ---------------------------------------------------------------------------
# 2. Vector Neuron Attention (VN-Attention)
# ---------------------------------------------------------------------------
class VNAttention(nn.Module):
    """
    Rotation Equivariant Self-Attention.
    Allows the model to learn long-range topology without fixed k-NN.
    """
    def __init__(self, in_channels, num_heads=1, dim=4):
        super().__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        self.q_conv = VNLinear(in_channels, in_channels)
        self.k_conv = VNLinear(in_channels, in_channels)
        self.v_conv = VNLinear(in_channels, in_channels)
        self.out_conv = VNLinear(in_channels, in_channels)
        self.dim = dim

    def forward(self, x):
        # x: (B, C, 3, N)
        B, C, D, N = x.size()
        H = self.num_heads
        
        Q = self.q_conv(x) # (B, C, 3, N)
        K = self.k_conv(x) # (B, C, 3, N)
        V = self.v_conv(x) # (B, C, 3, N)

        # Reshape for heads: (B, H, C/H, 3, N) -> (B, H, N, C/H, 3)
        q_reshaped = Q.view(B, H, self.head_dim, 3, N).permute(0, 1, 4, 2, 3)
        k_reshaped = K.view(B, H, self.head_dim, 3, N).permute(0, 1, 4, 2, 3)
        
        # Energy: dot product over C/H and 3
        # (B, H, N, C/H, 3) * (B, H, M, C/H, 3) -> (B, H, N, M)
        energy = torch.einsum('bhnci,bhmci->bhnm', q_reshaped, k_reshaped)

        attention = _softmax_fp32(energy / (self.head_dim**0.5), dim=-1) # (B, H, N, N)

        # Apply attention to V
        # V: (B, C, 3, N) -> (B, H, N, C/H, 3)
        v_reshaped = V.view(B, H, self.head_dim, 3, N).permute(0, 1, 4, 2, 3)
        
        # (B, H, N, M) * (B, H, M, C/H, 3) -> (B, H, N, C/H, 3)
        out = torch.einsum('bhnm,bhmci->bhnci', attention, v_reshaped)
        
        # Reshape back: (B, H, N, C/H, 3) -> (B, H, C/H, 3, N) -> (B, C, 3, N)
        out = out.permute(0, 1, 3, 4, 2).reshape(B, C, 3, N)
        
        out = self.out_conv(out)
        return out + x # Residual

class VNFeedForward(nn.Module):
    def __init__(self, in_channels, expansion_factor=2, dim=4):
        super().__init__()
        hidden_channels = in_channels * expansion_factor
        self.net = nn.Sequential(
            VNLinearLeakyReLU(in_channels, hidden_channels, dim=dim, negative_slope=0.0),
            VNLinear(hidden_channels, in_channels)
        )

    def forward(self, x):
        return self.net(x)

class VNTransformerBlock(nn.Module):
    def __init__(self, channels, num_heads=1, dim=4, expansion_factor=2):
        super().__init__()
        self.attn = VNAttention(channels, num_heads=num_heads, dim=dim)
        self.norm1 = VNBatchNorm(channels, dim=dim)
        self.ffn = VNFeedForward(channels, expansion_factor=expansion_factor, dim=dim)
        self.norm2 = VNBatchNorm(channels, dim=dim)

    def forward(self, x):
        # Post-Norm architecture
        # 1. Attention + Residual (VNAttention handles residual internally? No, wait, let's check VNAttention)
        # Checking VNAttention code above: "return out + x # Residual"
        # So VNAttention DOES add residual.
        
        x = self.norm1(self.attn(x))
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x

# ---------------------------------------------------------------------------
# 3. Geometry-Aware VN-Transformer Encoder
# ---------------------------------------------------------------------------

@register_encoder("VN_Transformer")
class VNTransformerEncoder(nn.Module):
    def __init__(self, latent_size=128, num_points=80, n_knn=20, hidden_dim=64, num_layers=1, num_heads=1, size_preset=None):
        super().__init__()
        
        if size_preset is not None:
            presets = {
                'tiny': {'hidden_dim': 32, 'num_layers': 1, 'num_heads': 2},
                'small': {'hidden_dim': 64, 'num_layers': 2, 'num_heads': 4},
                'base': {'hidden_dim': 128, 'num_layers': 4, 'num_heads': 8},
                'large': {'hidden_dim': 256, 'num_layers': 6, 'num_heads': 16},
            }
            if size_preset in presets:
                config = presets[size_preset]
                hidden_dim = config['hidden_dim']
                num_layers = config['num_layers']
                num_heads = config['num_heads']
            else:
                raise ValueError(f"Unknown preset {size_preset}. Available: {list(presets.keys())}")
        
        # 1. Explicit Topology Extractor (Invariant)
        self.geo_extractor = InvariantGeometricFeatures(num_points, n_knn)
        
        # 2. Embedding Layers (Equivariant)
        self.conv_pos = VNLinearLeakyReLU(1, hidden_dim // 2, dim=4, negative_slope=0.0)
        self.conv1 = VNLinearLeakyReLU(hidden_dim // 2, hidden_dim, dim=4, negative_slope=0.0)
        
        # 3. Transformer Block (Global Topology)
        self.transformer = nn.Sequential(*[
            VNTransformerBlock(hidden_dim, num_heads=num_heads, dim=4, expansion_factor=2) 
            for _ in range(num_layers)
        ])
        
        # 4. Std Feature (Mixing invariant and equivariant)
        # We augment the standard VN feature extraction with our explicit geometric features
        self.std_feature = VNStdFeature(
            hidden_dim * 2, # Doubled because of concatenation in forward
            dim=4,
            normalize_frame=False,
            hidden_dims=(hidden_dim * 2, hidden_dim)
        )
        
        # Fusion MLP for Invariant Latent
        # Input: (StdFeature Invariants) + (Geometric Explicit Invariants)
        inv_input_dim = (hidden_dim * 2 * 3) + 16 
        self.inv_mlp = nn.Sequential(
            nn.Linear(inv_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, latent_size)
        )
        
        # Head for Equivariant Latent (Orientation)
        # Output shape: (B, latent_size, 3) to match VNRotationHead expectations
        self.eq_mlp = VNLinear(hidden_dim, latent_size) 

        # Head for Crystallinity detection (0=Liquid, 1=Crystal)
        self.crystallinity_head = nn.Sequential(
            nn.Linear(latent_size, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Input x: (B, N, 3) -> (B, 3, N)
        x = x.permute(0, 2, 1)
        
        # A. Explicit Invariant Geometry (The "Topology" help)
        geo_feats = self.geo_extractor(x) # (B, 16, N)
        geo_global = geo_feats.mean(dim=-1) # (B, 16) Global descriptor
        
        # B. Equivariant Stream
        x_eq = x.unsqueeze(1) # (B, 1, 3, N)
        # Create initial graph feature to get local normals/directions
        # (Assuming you have get_graph_feature from vn.txt)
        # For simplicity in Transformer, we just embed position first
        x_eq = self.conv_pos(x_eq.repeat(1, 1, 1, 1)) # Dummy expansion or use graph
        
        # Note: To strictly use your vn_encoders 'get_graph_feature', you would do:
        # from .vn_encoders import get_graph_feature
        # x_graph = get_graph_feature(x_eq, k=20)
        # x_eq = self.conv_pos(x_graph).mean(dim=-1)
        
        # Let's assume simple VN embedding for the transformer demo:
        x_eq = self.conv1(x_eq) # (B, 64, 3, N)
        
        # Transformer mixes information globally
        x_eq = self.transformer(x_eq) # (B, 64, 3, N)
        
        # C. Feature Aggregation
        x_mean = x_eq.mean(dim=-1, keepdim=True)
        x_concat = torch.cat((x_eq, x_mean.expand_as(x_eq)), dim=1)
        
        # D. Get Invariant Features (V) and Equivariant Frames (Z0)
        # V: (B, C, N), Z0: (B, C, 3)
        V, Z0 = self.std_feature(x_concat) 
        
        # E. Latent Construction
        # 1. Invariant Latent (Shape/Topology)
        V = V.view(V.size(0), -1, V.size(-1))
        V_pooled = V.max(dim=-1)[0] # (B, C_inv)
        # FUSE explicit geometry with learned VN invariants
        V_final = torch.cat([V_pooled, geo_global], dim=1)
        z_inv = self.inv_mlp(V_final)
        
        # 2. Equivariant Latent (Orientation)
        z_eq = self.eq_mlp(x_mean.squeeze(-1)) # (B, latent/3, 3)
        
        # 3. Crystallinity Score
        crystallinity = self.crystallinity_head(z_inv)
        
        return z_inv, z_eq, crystallinity


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
    "VNTransformerEncoder",
]



# ---------------------------------------------------------------------------
# REVNET-inspired hierarchical VN anchor backbone encoder (autoencoder-friendly)
# ---------------------------------------------------------------------------
# This follows the paper's key decoder/encoder-side ideas:
# - Hierarchical downsampling to "anchors" (FPS) + neighborhood grouping (VN-SA)
# - ZCA-based normalization for VN features
# - Channel-wise subtraction attention (CWSA) on anchor tokens
# See REVNET Sec. 3.1.2, 3.2, 3.4. (arXiv:2601.08558v1)

import math


class VNZCALayerNorm(nn.Module):
    """ZCA-whitening LayerNorm for VN features (B, C, 3, ...)."""

    def __init__(self, channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        else:
            self.register_buffer("gamma", torch.ones(1, channels, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 3:
            raise ValueError(f"Expected VN tensor with >=3 dims, got shape {tuple(x.shape)}")
        B, C = x.shape[:2]
        if x.shape[2] != 3:
            raise ValueError(f"VNZCALayerNorm expects vector-dim=3, got {x.shape[2]}")

        x_flat = x.reshape(B, C, 3, -1).permute(0, 2, 1, 3).reshape(B, 3, -1)
        # Whitening in float32 avoids bf16 eigh/rsqrt issues and improves stability.
        x_flat_f = x_flat.to(torch.float32)
        mu = x_flat_f.mean(dim=-1, keepdim=True)
        xc = x_flat_f - mu

        M = xc.shape[-1]
        cov = (xc @ xc.transpose(1, 2)) / (float(M) + EPS)
        cov = cov + self.eps * torch.eye(3, device=x.device, dtype=xc.dtype).unsqueeze(0)

        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = eigvals.clamp_min(self.eps)
        inv_sqrt = torch.rsqrt(eigvals)
        W = eigvecs @ torch.diag_embed(inv_sqrt) @ eigvecs.transpose(1, 2)
        xw = W @ xc

        xw = xw.reshape(B, 3, C, -1).permute(0, 2, 1, 3).reshape_as(x).to(x.dtype)

        gamma = self.gamma
        while gamma.dim() < xw.dim():
            gamma = gamma.unsqueeze(-1)
        return xw * gamma.to(xw.dtype)


def farthest_point_sample(
    xyz: torch.Tensor,
    npoint: int,
    *,
    deterministic: bool = False,
) -> torch.Tensor:
    """FPS on xyz (B, N, 3) -> idx (B, npoint). Rotation-invariant via distances."""
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device, dtype=xyz.dtype)
    if deterministic:
        farthest = torch.zeros((B,), dtype=torch.long, device=device)
    else:
        farthest = torch.randint(0, N, (B,), device=device)
    batch_indices = torch.arange(B, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = distance.max(-1)[1]
    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points (B, N, D) by idx (B, ...). -> (B, ..., D)."""
    B = points.shape[0]
    idx_flat = idx.reshape(B, -1)
    out = points.gather(1, idx_flat.unsqueeze(-1).expand(B, idx_flat.shape[1], points.shape[-1]))
    return out.view(B, *idx.shape[1:], points.shape[-1])


def index_points_vn(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather VN features (B, C, 3, N) by idx (B, ...). -> (B, C, 3, ...)."""
    B, C, _, N = x.shape
    idx_flat = idx.reshape(B, -1)
    out = x.gather(3, idx_flat.unsqueeze(1).unsqueeze(2).expand(B, C, 3, idx_flat.shape[1]))
    return out.view(B, C, 3, *idx.shape[1:])


def knn_point(k: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """kNN from new_xyz to xyz using Euclidean distance. Returns idx (B, M, k)."""
    # xyz: (B, N, 3), new_xyz: (B, M, 3)
    dist = torch.cdist(new_xyz, xyz)  # (B, M, N)
    return dist.topk(k=k, dim=-1, largest=False)[1]


class VNSetAbstraction(nn.Module):
    """
    VN Set Abstraction (VN-SA) style block:
    - FPS downsample to M anchors
    - group kNN neighbors
    - edge feature: [neighbor - anchor, anchor, rel_xyz]
    - VN MLP + pooling over neighbors
    """

    def __init__(
        self,
        npoint: int,
        k: int,
        in_channels: int,
        out_channels: int,
        pooling: str = "mean",
        use_zca_norm: bool = True,
        negative_slope: float = 0.1,
        use_batchnorm: bool = True,
        deterministic_fps: bool = False,
    ):
        super().__init__()
        self.npoint = npoint
        self.k = k
        self.pooling = pooling
        self.deterministic_fps = deterministic_fps

        edge_in = 2 * in_channels + 1
        self.edge_mlp = VNLinearLeakyReLU(
            edge_in,
            out_channels,
            dim=5,
            negative_slope=negative_slope,
            use_batchnorm=use_batchnorm,
        )

        if pooling == "max":
            self.pool = VNMaxPool(out_channels)
        else:
            self.pool = None

        self.norm = VNZCALayerNorm(out_channels) if use_zca_norm else VNBatchNorm(out_channels, dim=4)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        xyz: (B, N, 3)
        feat: (B, C, 3, N)
        returns: new_xyz (B, M, 3), new_feat (B, Cout, 3, M)
        """
        B, N, _ = xyz.shape
        M = self.npoint

        fps_idx = farthest_point_sample(xyz, M, deterministic=self.deterministic_fps)  # (B, M)
        new_xyz = index_points(xyz, fps_idx)             # (B, M, 3)
        anchor_feat = index_points_vn(feat, fps_idx)     # (B, C, 3, M)

        group_idx = knn_point(self.k, xyz, new_xyz)      # (B, M, k)

        neigh_xyz = index_points(xyz, group_idx)         # (B, M, k, 3)
        rel_xyz = neigh_xyz - new_xyz.unsqueeze(2)       # (B, M, k, 3)
        rel_xyz_vn = rel_xyz.permute(0, 3, 1, 2).unsqueeze(1)  # (B, 1, 3, M, k)

        neigh_feat = index_points_vn(feat, group_idx)    # (B, C, 3, M, k)
        anchor_feat_rep = anchor_feat.unsqueeze(-1).expand_as(neigh_feat)
        edge = torch.cat([neigh_feat - anchor_feat_rep, anchor_feat_rep, rel_xyz_vn], dim=1)  # (B, 2C+1, 3, M, k)

        h = self.edge_mlp(edge)  # (B, Cout, 3, M, k)

        if self.pooling == "max":
            new_feat = self.pool(h)          # (B, Cout, 3, M)
        else:
            new_feat = h.mean(dim=-1)        # (B, Cout, 3, M)

        new_feat = self.norm(new_feat)
        return new_xyz, new_feat


class VNChannelWiseSubtractionAttention(nn.Module):
    """Channel-wise subtraction attention (CWSA) over anchor tokens."""

    def __init__(self, channels: int, hidden: int = 64, use_pos: bool = True):
        super().__init__()
        self.channels = channels
        self.use_pos = use_pos
        self.to_q = VNLinear(channels, channels)
        self.to_k = VNLinear(channels, channels)
        self.to_v = VNLinear(channels, channels)
        self.to_out = VNLinear(channels, channels)

        in_mlp = 2 if use_pos else 1
        self.score_mlp = nn.Sequential(
            nn.Linear(in_mlp, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, C, 3, K), pos: (B, K, 3)
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        qk = q.permute(0, 3, 1, 2)  # (B,K,C,3)
        kk = k.permute(0, 3, 1, 2)
        vv = v.permute(0, 3, 1, 2)

        rel = qk[:, :, None, :, :] - kk[:, None, :, :, :]     # (B,Kq,Kk,C,3)
        rel_norm = torch.norm(rel, dim=-1)                    # (B,Kq,Kk,C)
        rel_mean = rel_norm.mean(dim=-1, keepdim=True)        # (B,Kq,Kk,1)

        if self.use_pos:
            if pos is None:
                raise ValueError("pos must be provided when use_pos=True")
            rel_pos = pos[:, :, None, :] - pos[:, None, :, :]  # (B,Kq,Kk,3)
            rel_pos_norm = torch.norm(rel_pos, dim=-1, keepdim=True)
            score_in = torch.cat([rel_mean, rel_pos_norm], dim=-1)
        else:
            score_in = rel_mean

        scores = self.score_mlp(score_in)                     # (B,Kq,Kk,C)
        attn = _softmax_fp32(scores, dim=2)

        out = torch.einsum("bijk,bjkd->bikd", attn, vv)        # (B,K,C,3)
        out = out.permute(0, 2, 3, 1).contiguous()             # (B,C,3,K)
        return self.to_out(out)


class VNAnchorTransformerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        mlp_ratio: int = 2,
        attn_hidden: int = 64,
        use_zca_norm: bool = True,
        use_pos: bool = True,
        negative_slope: float = 0.1,
        use_batchnorm: bool = False,
    ):
        super().__init__()
        self.attn = VNChannelWiseSubtractionAttention(channels, hidden=attn_hidden, use_pos=use_pos)
        self.norm1 = VNZCALayerNorm(channels) if use_zca_norm else VNBatchNorm(channels, dim=4)
        self.ffn = nn.Sequential(
            VNLinearLeakyReLU(
                channels,
                channels * mlp_ratio,
                dim=4,
                negative_slope=negative_slope,
                use_batchnorm=use_batchnorm,
            ),
            VNLinear(channels * mlp_ratio, channels),
        )
        self.norm2 = VNZCALayerNorm(channels) if use_zca_norm else VNBatchNorm(channels, dim=4)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attn(x, pos))
        x = self.norm2(x + self.ffn(x))
        return x


@register_encoder("VN_REVNET_Backbone")
class VNRevnetBackboneEncoder(Encoder):
    """
    Encoder meant to replace VNDGCNNEncoderRefined when you want an easier-to-train,
    fully SO(3)-equivariant VN autoencoder.

    Output:
      inv_z: (B, latent_size)          (optional invariant head, for compatibility)
      eq_z:  (B, latent_size, 3)       equivariant VN latent for equivariant decoders
      center: (B, 3)                   mean input location (like other encoders here)
    """

    def __init__(
        self,
        latent_size: int = 256,
        k_embed: int = 20,
        embed_channels: int = 48,
        sa_channels: tuple[int, int] = (96, 192),
        sa_npoints: tuple[int, int] = (256, 64),
        sa_knn: tuple[int, int] = (32, 32),
        res_blocks_per_stage: int = 1,
        transformer_depth: int = 2,
        attn_hidden: int = 128,
        mlp_ratio: int = 2,
        pooling: str = "mean",
        global_pooling: str = "mean",
        use_zca_norm: bool = True,
        use_batchnorm: bool = True,
        dropout_rate: float = 0.0,
        deterministic_fps: bool = False,
    ):
        super().__init__()
        if len(sa_channels) != 2 or len(sa_npoints) != 2 or len(sa_knn) != 2:
            raise ValueError("This encoder currently expects 2-stage SA: sa_channels/npoints/knn must be length 2.")

        self.dropout_rate = dropout_rate

        # VN input embedding using VN-EdgeConv style lifting
        self.embed = VNLinearLeakyReLU(
            2,  # [neighbor - center, center] on 1-channel input -> 2 channels
            embed_channels,
            dim=5,
            negative_slope=0.1,
            use_batchnorm=use_batchnorm,
        )

        self.sa1 = VNSetAbstraction(
            npoint=sa_npoints[0],
            k=sa_knn[0],
            in_channels=embed_channels,
            out_channels=sa_channels[0],
            pooling=pooling,
            use_zca_norm=use_zca_norm,
            use_batchnorm=use_batchnorm,
            deterministic_fps=deterministic_fps,
        )
        self.sa2 = VNSetAbstraction(
            npoint=sa_npoints[1],
            k=sa_knn[1],
            in_channels=sa_channels[0],
            out_channels=sa_channels[1],
            pooling=pooling,
            use_zca_norm=use_zca_norm,
            use_batchnorm=use_batchnorm,
            deterministic_fps=deterministic_fps,
        )

        self.res1 = nn.Sequential(*[VNResBlock(sa_channels[0], dim=4, use_batchnorm=use_batchnorm) for _ in range(res_blocks_per_stage)])
        self.res2 = nn.Sequential(*[VNResBlock(sa_channels[1], dim=4, use_batchnorm=use_batchnorm) for _ in range(res_blocks_per_stage)])

        self.blocks = nn.ModuleList(
            [
                VNAnchorTransformerBlock(
                    channels=sa_channels[1],
                    mlp_ratio=mlp_ratio,
                    attn_hidden=attn_hidden,
                    use_zca_norm=use_zca_norm,
                    use_pos=True,
                )
                for _ in range(transformer_depth)
            ]
        )

        global_pooling = str(global_pooling).lower()
        if global_pooling not in {"mean", "max"}:
            raise ValueError(f"Unsupported global_pooling={global_pooling!r}")
        self.global_pooling = global_pooling
        self.global_pool = VNMaxPool(sa_channels[1]) if global_pooling == "max" else None

        # Equivariant latent head: (B, C, 3) -> (B, latent_size, 3)
        self.out_eq = VNLinearLeakyReLU(sa_channels[1], latent_size, dim=3, use_batchnorm=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, N, 3) expected
        if x.dim() != 3 or x.shape[-1] != 3:
            raise ValueError(f"Expected input (B, N, 3), got {tuple(x.shape)}")

        B, N, _ = x.shape
        center = x.mean(dim=1)  # (B, 3)

        # Build 1-channel VN feature from coordinates: (B, 1, 3, N)
        x_vn = x.permute(0, 2, 1).unsqueeze(1)

        # VN-EdgeConv style neighbor lifting on coordinates (kNN in coord space)
        x_coord = x.permute(0, 2, 1)  # (B, 3, N)
        edge = get_graph_feature(x_vn, k=20, x_coord=x_coord)  # (B, 2, 3, N, k)
        feat = self.embed(edge).mean(dim=-1)                   # (B, Cembed, 3, N)

        # Stage 1 SA
        xyz1, feat1 = self.sa1(x, feat)                        # (B, M1, 3), (B, C1, 3, M1)
        feat1 = self.res1(feat1)

        # Stage 2 SA
        xyz2, feat2 = self.sa2(xyz1, feat1)                    # (B, M2, 3), (B, C2, 3, M2)
        feat2 = self.res2(feat2)

        # Anchor transformer on final anchors
        for blk in self.blocks:
            feat2 = blk(feat2, xyz2)

        # Equivariant latent from pooled anchors
        if self.global_pooling == "max":
            pooled = self.global_pool(feat2)                   # (B, C2, 3)
        else:
            pooled = feat2.mean(dim=-1)                        # (B, C2, 3)
        eq_z = self.out_eq(pooled)                             # (B, latent_size, 3)

        # Invariant latent: L2 norm of each equivariant vector (no learned params)
        inv_z = eq_z.norm(dim=-1)                              # (B, latent_size)

        return inv_z, eq_z, center
