from __future__ import annotations

import warnings

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


@register_encoder("PnE_VN")
class PointNetEncoderVN(Encoder):
    def __init__(
        self,
        latent_size: int = 64,
        n_knn: int = 20,
        pooling: str = 'mean',
        feature_transform: bool = True,
        hidden_dim1: int = 320,
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
        self._warned_low_precision = False

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
        if x.dtype in (torch.float16, torch.bfloat16) and not self._warned_low_precision:
            warnings.warn(
                "PointNetEncoderVN in float16/bfloat16 can lose rotational equivariance; use 32-bit precision.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_low_precision = True
        x = x.permute(0, 2, 1)
        batch_size, _, num_points = x.size()
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
        feature_dims: tuple[int, int, int, int, int] = (128, 128, 256, 512, 1024),
        global_mlp_dims: tuple[int, int] = (512, 256),
        global_dropout: float = 0.05,
        share_nonlinearity: bool = True,
        std_feature_hidden_dims: tuple[int, int] | None = None,
        use_batchnorm: bool = True,
        use_cross_product: bool = False,
    ):
        super().__init__()
        if len(feature_dims) != 5:
            raise ValueError(f"feature_dims must provide five entries (got {feature_dims})")

        self.n_knn = n_knn
        self.pooling = pooling
        self.use_cross_product = use_cross_product
        
        c_mult = 3 if use_cross_product else 2

        c1, c2, c3, c4, c5 = [max(1, dim // 3) for dim in feature_dims]

        self.conv1 = VNLinearLeakyReLU(
            c_mult,
            c1,
            dim=5,
            negative_slope=0.2,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = VNLinearLeakyReLU(
            c1 * c_mult,
            c2,
            dim=5,
            negative_slope=0.2,
            use_batchnorm=use_batchnorm,
        )
        self.conv3 = VNLinearLeakyReLU(
            c2 * c_mult,
            c3,
            dim=5,
            negative_slope=0.2,
            use_batchnorm=use_batchnorm,
        )
        self.conv4 = VNLinearLeakyReLU(
            c3 * c_mult,
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
            negative_slope=0.2,
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
        if x.dim() != 3 or x.shape[-1] != 3:
            raise ValueError(f"VNDGCNNEncoder expects input shape (B,N,3), got {tuple(x.shape)}")
        center_loc = x.mean(dim=1)
        x = x.permute(0, 2, 1).contiguous()
        batch_size, _, num_points = x.size()
        x = x.unsqueeze(1)

        x = _get_graph_feature_with_mode(x, k=self.n_knn, use_cross_product=self.use_cross_product)
        x = self.conv1(x)
        x1 = self.pool1(x)

        x = _get_graph_feature_with_mode(x1, k=self.n_knn, use_cross_product=self.use_cross_product)
        x = self.conv2(x)
        x2 = self.pool2(x)

        x = _get_graph_feature_with_mode(x2, k=self.n_knn, use_cross_product=self.use_cross_product)
        x = self.conv3(x)
        x3 = self.pool3(x)

        x = _get_graph_feature_with_mode(x3, k=self.n_knn, use_cross_product=self.use_cross_product)
        x = self.conv4(x)
        x4 = self.pool4(x)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)

        # Pool per-point VN features, then map to desired latent_size VN channels
        eq_z = x.mean(dim=-1)  # (B, c5, 3)
        eq_z = self.out_eq_mlp(eq_z)  # (B, latent_size, 3)
        x_mean_out = x.mean(dim=-1, keepdim=True)

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


def _project_to_rotation_group(rot_mat: torch.Tensor, strict: str) -> torch.Tensor:
    if strict == 'None':
        return rot_mat
    u, _, v = torch.linalg.svd(rot_mat)
    return u @ v.transpose(-1, -2)


class SimpleRot(nn.Module):
    def __init__(self, in_ch: int, strict: str = 'None'):
        super().__init__()
        self.model = VNLinear(in_ch, 3)
        self.strict = strict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rot_mat = self.model(x).squeeze(-1)
        return _project_to_rotation_group(rot_mat, self.strict)


class ComplexRot(nn.Module):
    def __init__(self, in_ch: int, strict: str = 'None'):
        super().__init__()
        self.linear1 = VNLinearLeakyReLU(in_ch, in_ch * 6, dim=4, negative_slope=0.1)
        self.linear2 = VNLinearLeakyReLU(in_ch * 6, in_ch * 4, dim=4, negative_slope=0.1)
        self.linear3 = VNLinearLeakyReLU(in_ch * 4, in_ch, dim=4, negative_slope=0.1)
        self.linearR = VNLinear(in_ch, 3)
        self.strict = strict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        R = self.linearR(x)
        rot_mat = torch.mean(R, dim=-1)
        return _project_to_rotation_group(rot_mat, self.strict)


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



def _get_graph_feature_with_mode(
    x: torch.Tensor,
    k: int,
    *,
    use_cross_product: bool,
    idx: torch.Tensor | None = None,
    x_coord: torch.Tensor | None = None,
) -> torch.Tensor:
    if use_cross_product:
        if x_coord is not None:
            raise ValueError("x_coord is not supported when use_cross_product=True")
        return get_graph_feature_cross(x, k=k, idx=idx)
    return get_graph_feature(x, k=k, idx=idx, x_coord=x_coord)


class VNRobustInvariantHead(nn.Module):
    """
    Robustly extracts invariant features from equivariant vectors (B, C, 3).
    Uses vector norms and projections onto the global mean direction.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.2,
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
        dropout_rate: float = 0.2,
        use_cross_product: bool = False,
    ):
        super().__init__()
        self.n_knn = n_knn
        self.pooling = pooling
        self.dropout_rate = dropout_rate
        self.use_cross_product = use_cross_product

        c_mult = 3 if use_cross_product else 2

        c1, c2, c3, c4, c5 = [max(1, dim // 3) for dim in feature_dims]

        self.conv1 = VNLinearLeakyReLU(c_mult, c1, dim=5, negative_slope=0.2, use_batchnorm=use_batchnorm)
        self.conv2 = VNLinearLeakyReLU(c1 * c_mult, c2, dim=5, negative_slope=0.2, use_batchnorm=use_batchnorm)
        self.conv3 = VNLinearLeakyReLU(c2 * c_mult, c3, dim=5, negative_slope=0.2, use_batchnorm=use_batchnorm)
        self.conv4 = VNLinearLeakyReLU(c3 * c_mult, c4, dim=5, negative_slope=0.2, use_batchnorm=use_batchnorm)
        
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
        x_graph = _get_graph_feature_with_mode(x, k=self.n_knn, use_cross_product=self.use_cross_product)
        x = self.conv1(x_graph) # (B, c1, 3, N, k)
        x1 = x.mean(dim=-1) # Pool neighbors -> (B, c1, 3, N)

        # Layer 2
        x_graph = _get_graph_feature_with_mode(x1, k=self.n_knn, use_cross_product=self.use_cross_product)
        x = self.conv2(x_graph)
        x2 = x.mean(dim=-1)

        # Layer 3
        x_graph = _get_graph_feature_with_mode(x2, k=self.n_knn, use_cross_product=self.use_cross_product)
        x = self.conv3(x_graph)
        x3 = x.mean(dim=-1)

        # Layer 4
        x_graph = _get_graph_feature_with_mode(x3, k=self.n_knn, use_cross_product=self.use_cross_product)
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
      inv_z: (B, latent_size + P)      (invariant head: norms + selected Gram dots)
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
        dropout_rate: float = 0.2,
        gram_pairs: int = 128,
        gram_pair_mode: str = "nearest",
        gram_pair_seed: int = 0,
        invariant_eps: float = 1e-6,
        deterministic_fps: bool = False,
    ):
        super().__init__()
        if len(sa_channels) != 2 or len(sa_npoints) != 2 or len(sa_knn) != 2:
            raise ValueError("This encoder currently expects 2-stage SA: sa_channels/npoints/knn must be length 2.")

        self.dropout_rate = dropout_rate
        self.k_embed = int(k_embed)

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
        
        # Learnable scale for equivariant output to match input data scale
        # VNBatchNorm normalizes features resulting in small vector norms (~0.1)
        # For unit-sphere normalized data with RMS ~0.7, we need to scale up significantly
        # Initialize to 5.0 as a reasonable starting point (can learn from there)
        self.eq_z_scale = nn.Parameter(torch.tensor(5.0))

        self.inv_eps = float(invariant_eps)
        pair_indices = self._build_gram_pairs(
            num_channels=latent_size,
            num_pairs=gram_pairs,
            mode=gram_pair_mode,
            seed=gram_pair_seed,
        )
        self.register_buffer("gram_pairs", pair_indices, persistent=False)
        self.num_gram_pairs = int(pair_indices.shape[0])
        self.inv_dim = latent_size + self.num_gram_pairs
        self.inv_norm = nn.LayerNorm(self.inv_dim)
        self.inv_norm_pooled = nn.LayerNorm(self.inv_dim * 2)

    @staticmethod
    def _build_gram_pairs(
        num_channels: int,
        num_pairs: int,
        mode: str,
        seed: int,
    ) -> torch.Tensor:
        num_channels = int(num_channels)
        num_pairs = int(num_pairs)
        if num_pairs <= 0 or num_channels < 2:
            return torch.empty((0, 2), dtype=torch.long)

        max_pairs = num_channels * (num_channels - 1) // 2
        num_pairs = min(num_pairs, max_pairs)
        mode = str(mode).lower()

        if mode == "nearest":
            pairs = []
            for offset in range(1, num_channels):
                for i in range(num_channels - offset):
                    pairs.append((i, i + offset))
                    if len(pairs) >= num_pairs:
                        break
                if len(pairs) >= num_pairs:
                    break
            return torch.tensor(pairs, dtype=torch.long)

        if mode == "random":
            g = torch.Generator()
            g.manual_seed(int(seed))
            idx = torch.triu_indices(num_channels, num_channels, offset=1)
            perm = torch.randperm(idx.shape[1], generator=g)
            selected = perm[:num_pairs]
            return idx[:, selected].T.contiguous()

        raise ValueError(f"Unsupported gram_pair_mode={mode!r}")

    def _invariant_features(self, z_eq: torch.Tensor) -> torch.Tensor:
        if z_eq.dim() == 3:
            norms = torch.sqrt((z_eq * z_eq).sum(dim=-1) + self.inv_eps)  # (B, C)
            if self.num_gram_pairs > 0:
                pairs = self.gram_pairs
                v1 = z_eq[:, pairs[:, 0], :]
                v2 = z_eq[:, pairs[:, 1], :]
                dots = (v1 * v2).sum(dim=-1)  # (B, P)
                feat = torch.cat([norms, dots], dim=-1)
            else:
                feat = norms
            return self.inv_norm(feat)

        if z_eq.dim() == 4:
            norms = torch.sqrt((z_eq * z_eq).sum(dim=-1) + self.inv_eps)  # (B, N, C)
            if self.num_gram_pairs > 0:
                pairs = self.gram_pairs
                v1 = z_eq[:, :, pairs[:, 0], :]
                v2 = z_eq[:, :, pairs[:, 1], :]
                dots = (v1 * v2).sum(dim=-1)  # (B, N, P)
                feat = torch.cat([norms, dots], dim=-1)
            else:
                feat = norms
            feat_mean = feat.mean(dim=1)
            feat_max = feat.max(dim=1).values
            pooled = torch.cat([feat_mean, feat_max], dim=-1)
            return self.inv_norm_pooled(pooled)

        raise ValueError(f"Expected z_eq with shape (B,C,3) or (B,N,C,3), got {tuple(z_eq.shape)}")

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
        k0 = min(self.k_embed, N)
        edge = get_graph_feature(x_vn, k=k0, x_coord=x_coord)  # (B, 2, 3, N, k)
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
        
        # Apply learnable scale to match input data scale
        eq_z = eq_z * self.eq_z_scale

        # Invariant latent: channel norms + selected Gram dot products
        inv_z = self._invariant_features(eq_z)

        return inv_z, eq_z, center


# ---------------------------------------------------------------------------
# Atomic REVNET: GPU-friendly VN encoder for local atomic environments (64-256 pts)
# - No FPS (no Python loops)
# - Cross-product VN embedding (better plane/normal cues)
# - Multi-scale kNN EdgeConv (reuse single kNN at max_k)
# - Cheap invariant "moment" + radial-RBF head (crystal vs liquid signal)
# ---------------------------------------------------------------------------

class AtomicGeometricInvariantHead(nn.Module):
    """
    Computes cheap rotation-invariant geometric descriptors from a local point patch.

    Input:
      xyz: (B, N, 3)
      idx_knn_max: optional (B, N, Kmax) kNN indices (including self), computed once upstream.
    Output:
      geom: (B, out_dim)  (learned projection of raw invariants)
    """
    def __init__(
        self,
        k_geom: int = 16,
        rbf_bins: int = 16,
        rbf_max: float = 2.0,
        rbf_sigma: float = 0.15,
        proj_dim: int = 32,
        use_dir_moments: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.k_geom = int(k_geom)
        self.rbf_bins = int(rbf_bins)
        self.rbf_max = float(rbf_max)
        self.rbf_sigma = float(rbf_sigma)
        self.use_dir_moments = bool(use_dir_moments)
        self.eps = float(eps)

        # centers on normalized distance (d / mean(d)) in [0, rbf_max]
        mu = torch.linspace(0.0, self.rbf_max, self.rbf_bins)
        self.register_buffer("rbf_mu", mu, persistent=False)

        raw_dim = 0
        # global covariance invariants: tr, fro2/tr^2, det/tr^3
        raw_dim += 3
        # local covariance invariants (mean+std): tr, anis, vol => 6
        raw_dim += 6
        if self.use_dir_moments:
            # same on unit directions (mean+std): tr_dir, anis_dir => 4
            raw_dim += 4
        # radial histogram bins
        raw_dim += self.rbf_bins

        h = max(proj_dim, 8)
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, h),
            nn.LayerNorm(h),
            nn.SiLU(inplace=True),
            nn.Linear(h, proj_dim),
        )

    @staticmethod
    def _det3x3(C: torch.Tensor) -> torch.Tensor:
        # C: (..., 3, 3)
        a = C[..., 0, 0]
        b = C[..., 0, 1]
        c = C[..., 0, 2]
        d = C[..., 1, 0]
        e = C[..., 1, 1]
        f = C[..., 1, 2]
        g = C[..., 2, 0]
        h = C[..., 2, 1]
        i = C[..., 2, 2]
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

    def _cov_invariants(self, C: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # C: (..., 3, 3) symmetric PSD-ish
        tr = C[..., 0, 0] + C[..., 1, 1] + C[..., 2, 2]
        fro2 = (C * C).sum(dim=(-1, -2))
        det = self._det3x3(C).clamp_min(0.0)
        anis = fro2 / (tr * tr + self.eps)
        vol = det / (tr * tr * tr + self.eps)
        return tr, anis, vol

    def forward(self, xyz: torch.Tensor, idx_knn_max: torch.Tensor | None = None) -> torch.Tensor:
        if xyz.dim() != 3 or xyz.size(-1) != 3:
            raise ValueError(f"Expected xyz (B, N, 3), got {tuple(xyz.shape)}")
        B, N, _ = xyz.shape

        # float32 for stable invariants (tiny overhead for N<=256)
        xyz_f = xyz.float()

        # ---- Global covariance invariants (patch-level) ----
        centered = xyz_f - xyz_f.mean(dim=1, keepdim=True)  # (B,N,3)
        Cg = torch.einsum("bni,bnj->bij", centered, centered) / (N + self.eps)  # (B,3,3)
        tr_g, anis_g, vol_g = self._cov_invariants(Cg)
        global_feat = torch.stack([tr_g, anis_g, vol_g], dim=-1)  # (B,3)

        # ---- kNN (reuse idx if provided) ----
        k_needed = min(self.k_geom + 1, N)
        if idx_knn_max is None:
            x_coord = xyz_f.permute(0, 2, 1)  # (B,3,N)
            idx_knn_max = knn(x_coord, k=k_needed)  # includes self
        else:
            idx_knn_max = idx_knn_max[:, :, :k_needed]

        # drop self neighbor (first one if present)
        if idx_knn_max.size(-1) >= 2:
            idx = idx_knn_max[:, :, 1 : 1 + min(self.k_geom, idx_knn_max.size(-1) - 1)]  # (B,N,k)
        else:
            # degenerate: no neighbors besides self
            idx = idx_knn_max

        neigh = index_points(xyz_f, idx)  # (B,N,k,3)
        anchor = xyz_f[:, :, None, :]     # (B,N,1,3)
        r = neigh - anchor                # (B,N,k,3)
        d2 = (r * r).sum(dim=-1)          # (B,N,k)
        d = torch.sqrt(d2 + self.eps)

        # Smooth weights; sigma tied to mean neighbor distance (scale-robust)
        d_mean = d.mean(dim=(1, 2), keepdim=True).clamp_min(self.eps)  # (B,1,1)
        sigma = 0.75 * d_mean
        w = torch.exp(-d2 / (2.0 * sigma * sigma + self.eps))  # (B,N,k)

        # ---- Local covariance invariants (per point) ----
        rw = r * w[..., None]
        denom = w.sum(dim=-1, keepdim=True)[..., None] + self.eps
        C = torch.einsum("bnki,bnkj->bnij", rw, r) / denom  # (B,N,3,3)
        tr, anis, vol = self._cov_invariants(C)             # (B,N)

        local_stats = torch.stack(
            [tr.mean(dim=1), tr.std(dim=1),
             anis.mean(dim=1), anis.std(dim=1),
             vol.mean(dim=1), vol.std(dim=1)],
            dim=-1,
        )  # (B,6)

        dir_stats = []
        if self.use_dir_moments:
            u = r / (d[..., None] + self.eps)               # (B,N,k,3)
            uw = u * w[..., None]
            Cd = torch.einsum("bnki,bnkj->bnij", uw, u) / denom
            trd = Cd[..., 0, 0] + Cd[..., 1, 1] + Cd[..., 2, 2]
            fro2d = (Cd * Cd).sum(dim=(-1, -2))
            anisd = fro2d / (trd * trd + self.eps)
            dir_stats = torch.stack(
                [trd.mean(dim=1), trd.std(dim=1), anisd.mean(dim=1), anisd.std(dim=1)],
                dim=-1,
            )  # (B,4)

        # ---- Radial soft histogram on normalized distances ----
        dn = (d / d_mean).clamp_min(0.0)  # (B,N,k)
        mu = self.rbf_mu.to(dn.device, dn.dtype)[None, None, None, :]  # (1,1,1,M)
        rbf = torch.exp(-0.5 * ((dn[..., None] - mu) / self.rbf_sigma) ** 2)  # (B,N,k,M)
        rbf = (rbf * w[..., None]).mean(dim=(1, 2))  # (B,M)

        raw = [global_feat, local_stats]
        if self.use_dir_moments:
            raw.append(dir_stats)
        raw.append(rbf)
        raw = torch.cat(raw, dim=-1)  # (B, raw_dim)

        # Invariants are computed in float32 for robustness; cast back to the
        # projection dtype so bf16-true runs don't hit matmul dtype mismatches.
        proj_dtype = self.proj[0].weight.dtype
        if raw.dtype != proj_dtype:
            raw = raw.to(dtype=proj_dtype)

        return self.proj(raw)


class VNEdgeConvMS(nn.Module):
    """
    GPU-friendly VN EdgeConv with optional multi-scale kNN fusion and optional cross-product feature.

    Input:
      feat: (B, C, 3, N)
      idx_knn_max: (B, N, Kmax) indices for max K among k_list
    Output:
      out: (B, Cout, 3, N)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k_list: tuple[int, ...] = (16,),
        use_cross: bool = False,
        pooling: str = "mean",
        negative_slope: float = 0.1,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.k_list = tuple(int(k) for k in k_list)
        self.use_cross = bool(use_cross)

        pool = str(pooling).lower()
        if pool not in {"mean", "max"}:
            raise ValueError(f"Unsupported pooling={pool!r}")
        self.pooling = pool

        mult = 3 if self.use_cross else 2
        self.edge_mlp = VNLinearLeakyReLU(
            mult * in_channels,
            out_channels,
            dim=5,
            negative_slope=negative_slope,
            use_batchnorm=use_batchnorm,
        )
        self.norm = VNZCALayerNorm(out_channels)

        self.pool = VNMaxPool(out_channels) if self.pooling == "max" else None

    def forward(self, feat: torch.Tensor, idx_knn_max: torch.Tensor) -> torch.Tensor:
        outs = []
        Kmax = idx_knn_max.size(-1)

        for k in self.k_list:
            k = min(int(k), Kmax)
            idx = idx_knn_max[:, :, :k]
            if self.use_cross:
                edge = get_graph_feature_cross(feat, k=k, idx=idx)
            else:
                edge = get_graph_feature(feat, k=k, idx=idx)

            y = self.edge_mlp(edge)
            y = self.pool(y) if self.pool is not None else y.mean(dim=-1)  # (B,Cout,3,N)
            outs.append(y)

        out = outs[0] if len(outs) == 1 else torch.stack(outs, dim=0).mean(dim=0)
        return self.norm(out)


@register_encoder("VN_REVNET_Atomic")
class VNAtomicRevnetBackboneEncoder(Encoder):
    """
    Atomic REVNET: VN encoder tuned for local atomic neighborhoods (64-256 points).

    Output:
      inv_z: (B, latent_size + P + geom_dim)
      eq_z:  (B, latent_size, 3)
      center:(B, 3)
    """
    def __init__(
        self,
        latent_size: int = 256,
        k_embed: int = 24,
        k_list: tuple[int, ...] = (16, 32),
        embed_channels: int = 48,
        hidden_channels: tuple[int, int] = (96, 192),
        geom_k: int = 16,
        geom_dim: int = 32,
        gram_pairs: int = 128,
        gram_pair_mode: str = "nearest",
        gram_pair_seed: int = 0,
        global_pooling: str = "mean",
        use_batchnorm: bool = True,
        invariant_eps: float = 1e-6,
    ):
        super().__init__()
        self.latent_size = int(latent_size)
        self.k_embed = int(k_embed)
        self.k_list = tuple(int(k) for k in k_list)
        self.invariant_eps = float(invariant_eps)

        # Cross-product embedding on raw coordinates: [neighbor-center, center, cross]
        self.embed = VNLinearLeakyReLU(
            3,  # from get_graph_feature_cross on 1-channel coords
            embed_channels,
            dim=5,
            negative_slope=0.1,
            use_batchnorm=use_batchnorm,
        )

        # Two VN EdgeConv stages (no FPS)
        self.ec1 = VNEdgeConvMS(
            in_channels=embed_channels,
            out_channels=hidden_channels[0],
            k_list=self.k_list,
            use_cross=True,      # extra plane/normal cues
            pooling="mean",
            use_batchnorm=use_batchnorm,
        )
        self.ec2 = VNEdgeConvMS(
            in_channels=hidden_channels[0],
            out_channels=hidden_channels[1],
            k_list=(max(self.k_list),),
            use_cross=False,
            pooling="mean",
            use_batchnorm=use_batchnorm,
        )

        gp = str(global_pooling).lower()
        if gp not in {"mean", "max"}:
            raise ValueError(f"Unsupported global_pooling={gp!r}")
        self.global_pooling = gp
        self.global_pool = VNMaxPool(hidden_channels[1]) if gp == "max" else None

        # Equivariant latent
        self.out_eq = VNLinear(hidden_channels[1], latent_size)
        self.eq_z_scale = nn.Parameter(torch.tensor(5.0))

        # Invariant geometry head (moments + radial histogram)
        self.geom_head = AtomicGeometricInvariantHead(k_geom=geom_k, proj_dim=geom_dim)

        # Robust invariants from eq_z: norms + selected Gram dot products
        pair_indices = self._build_gram_pairs(
            num_channels=latent_size,
            num_pairs=gram_pairs,
            mode=gram_pair_mode,
            seed=gram_pair_seed,
        )
        self.register_buffer("gram_pairs", pair_indices, persistent=False)
        self.num_gram_pairs = int(pair_indices.shape[0])

        inv_dim = latent_size + self.num_gram_pairs + int(geom_dim)
        self.inv_norm = nn.LayerNorm(inv_dim)

    @staticmethod
    def _build_gram_pairs(num_channels: int, num_pairs: int, mode: str, seed: int) -> torch.Tensor:
        num_channels = int(num_channels)
        num_pairs = int(num_pairs)
        if num_pairs <= 0 or num_channels < 2:
            return torch.empty((0, 2), dtype=torch.long)

        max_pairs = num_channels * (num_channels - 1) // 2
        num_pairs = min(num_pairs, max_pairs)
        mode = str(mode).lower()

        if mode == "nearest":
            pairs = []
            for offset in range(1, num_channels):
                for i in range(num_channels - offset):
                    pairs.append((i, i + offset))
                    if len(pairs) >= num_pairs:
                        break
                if len(pairs) >= num_pairs:
                    break
            return torch.tensor(pairs, dtype=torch.long)

        if mode == "random":
            g = torch.Generator()
            g.manual_seed(int(seed))
            idx = torch.triu_indices(num_channels, num_channels, offset=1)
            perm = torch.randperm(idx.shape[1], generator=g)
            selected = perm[:num_pairs]
            return idx[:, selected].T.contiguous()

        raise ValueError(f"Unsupported gram_pair_mode={mode!r}")

    def _inv_from_eq(self, z_eq: torch.Tensor) -> torch.Tensor:
        # z_eq: (B, C, 3)
        norms = torch.sqrt((z_eq * z_eq).sum(dim=-1) + self.invariant_eps)  # (B, C)
        if self.num_gram_pairs > 0:
            pairs = self.gram_pairs
            v1 = z_eq[:, pairs[:, 0], :]
            v2 = z_eq[:, pairs[:, 1], :]
            dots = (v1 * v2).sum(dim=-1)  # (B, P)
            return torch.cat([norms, dots], dim=-1)
        return norms

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, N, 3)
        if x.dim() != 3 or x.shape[-1] != 3:
            raise ValueError(f"Expected input (B, N, 3), got {tuple(x.shape)}")

        B, N, _ = x.shape
        center = x.mean(dim=1)  # (B, 3)

        # Precompute kNN once (max_k across embed, edgeconv, geom (+1 for self-drop))
        max_k = max(self.k_embed, max(self.k_list), self.geom_head.k_geom + 1)
        max_k = min(int(max_k), N)

        x_coord = x.permute(0, 2, 1)          # (B, 3, N)
        idx_knn_max = knn(x_coord, k=max_k)   # (B, N, max_k), includes self

        # VN coordinate input: (B,1,3,N)
        x_vn = x.permute(0, 2, 1).unsqueeze(1)

        # Cross-product embed on coordinates
        k0 = min(self.k_embed, max_k)
        idx_embed = idx_knn_max[:, :, :k0]
        edge0 = get_graph_feature_cross(x_vn, k=k0, idx=idx_embed)  # (B, 3, 3, N, k)
        feat = self.embed(edge0).mean(dim=-1)                       # (B, Cembed, 3, N)

        # EdgeConv stages (no FPS)
        feat = self.ec1(feat, idx_knn_max)  # (B, C1, 3, N)
        feat = self.ec2(feat, idx_knn_max)  # (B, C2, 3, N)

        # Global pooling
        pooled = self.global_pool(feat) if self.global_pool is not None else feat.mean(dim=-1)  # (B, C2, 3)

        # Equivariant latent
        eq_z = self.out_eq(pooled) * self.eq_z_scale  # (B, latent_size, 3)

        # Invariant latent = eq invariants + geometric invariants
        inv_eq = self._inv_from_eq(eq_z)                       # (B, latent + P)
        inv_geom = self.geom_head(x, idx_knn_max=idx_knn_max)  # (B, geom_dim)
        inv_z = self.inv_norm(torch.cat([inv_eq, inv_geom], dim=-1))

        return inv_z, eq_z, center


class InvariantEdgeConv(nn.Module):
    """Scalar edge convolution over rotation-invariant local descriptors."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        pooling: str = "mean",
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        pool = str(pooling).lower()
        if pool not in {"mean", "max"}:
            raise ValueError(f"Unsupported pooling={pool!r}")
        self.pooling = pool

        hidden = max(in_channels, out_channels)
        edge_dim = (2 * in_channels) + 5
        self.edge_mlp = nn.Sequential(
            nn.Conv2d(edge_dim, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden) if use_batchnorm else nn.Identity(),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False),
                nn.LayerNorm(out_channels),
            )
        )

    def forward(self, feat: torch.Tensor, xyz_centered: torch.Tensor, idx_knn: torch.Tensor) -> torch.Tensor:
        if feat.dim() != 3:
            raise ValueError(f"InvariantEdgeConv expects feat (B,N,C), got {tuple(feat.shape)}")
        if xyz_centered.dim() != 3 or xyz_centered.shape[-1] != 3:
            raise ValueError(
                "InvariantEdgeConv expects xyz_centered with shape (B,N,3), "
                f"got {tuple(xyz_centered.shape)}"
            )

        neigh_feat = index_points(feat, idx_knn)                       # (B,N,k,C)
        center_feat = feat.unsqueeze(2).expand_as(neigh_feat)          # (B,N,k,C)

        center_xyz = xyz_centered.unsqueeze(2)                         # (B,N,1,3)
        neigh_xyz = index_points(xyz_centered, idx_knn)                # (B,N,k,3)
        rel_xyz = neigh_xyz - center_xyz                               # (B,N,k,3)

        center_r = torch.linalg.norm(xyz_centered, dim=-1, keepdim=True)
        center_r = center_r.unsqueeze(2).expand(-1, -1, idx_knn.size(-1), -1)
        neigh_r = torch.linalg.norm(neigh_xyz, dim=-1, keepdim=True)
        rel_r = torch.linalg.norm(rel_xyz, dim=-1, keepdim=True)
        dot = (center_xyz * neigh_xyz).sum(dim=-1, keepdim=True)
        cos = dot / (center_r * neigh_r + EPS)

        edge = torch.cat(
            [center_feat, neigh_feat - center_feat, center_r, neigh_r, rel_r, dot, cos],
            dim=-1,
        )
        edge = edge.permute(0, 3, 1, 2).contiguous()                   # (B,Ce,N,k)
        edge = self.edge_mlp(edge)

        if self.pooling == "max":
            pooled = edge.max(dim=-1).values
        else:
            pooled = edge.mean(dim=-1)
        pooled = pooled.transpose(1, 2).contiguous()                   # (B,N,Cout)
        return F.silu(pooled + self.shortcut(feat))


@register_encoder("REVNET_InvariantFast")
class REVNETInvariantFastEncoder(Encoder):
    """
    Fast REVNET-inspired encoder for contrastive/supervised use.

    The local backbone is fully invariant and uses only scalar neighborhood
    descriptors. An optional final head can emit a cheap equivariant latent by
    applying invariant scalar weights to centered coordinates.
    """

    def __init__(
        self,
        latent_size: int = 256,
        n_knn: int = 24,
        feature_dims: tuple[int, int, int] = (64, 128, 256),
        pooling: str = "mean",
        dropout_rate: float = 0.1,
        use_batchnorm: bool = True,
        use_geom_head: bool = False,
        geom_dim: int = 32,
        emit_eq_latent: bool = False,
        eq_latent_size: int | None = None,
        eq_softmax_temperature: float = 1.0,
    ) -> None:
        super().__init__()

        latent_size = int(latent_size)
        n_knn = int(n_knn)
        feature_dims = tuple(int(v) for v in feature_dims)
        pooling = str(pooling).lower()
        geom_dim = int(geom_dim)
        eq_channels = latent_size if eq_latent_size is None else int(eq_latent_size)
        eq_temperature = float(eq_softmax_temperature)

        if latent_size <= 0:
            raise ValueError(f"latent_size must be > 0, got {latent_size}")
        if n_knn <= 0:
            raise ValueError(f"n_knn must be > 0, got {n_knn}")
        if len(feature_dims) == 0 or min(feature_dims) <= 0:
            raise ValueError(f"feature_dims must contain positive channel sizes, got {feature_dims}")
        if pooling not in {"mean", "max"}:
            raise ValueError(f"Unsupported pooling={pooling!r}")
        if eq_temperature <= 0.0:
            raise ValueError(
                "eq_softmax_temperature must be > 0 so the cheap equivariant head remains well-defined, "
                f"got {eq_temperature}"
            )
        if eq_channels <= 0:
            raise ValueError(f"eq_latent_size must be > 0 when emit_eq_latent=True, got {eq_channels}")

        self.n_knn = n_knn
        self.pooling = pooling
        self.use_geom_head = bool(use_geom_head)
        self.emit_eq_latent = bool(emit_eq_latent)
        self.eq_softmax_temperature = eq_temperature

        first_dim = feature_dims[0]
        self.input_proj = nn.Sequential(
            nn.Linear(1, first_dim),
            nn.LayerNorm(first_dim),
            nn.SiLU(inplace=True),
            nn.Linear(first_dim, first_dim),
            nn.LayerNorm(first_dim),
            nn.SiLU(inplace=True),
        )

        dims = (first_dim,) + feature_dims
        self.layers = nn.ModuleList(
            [
                InvariantEdgeConv(
                    dims[i],
                    dims[i + 1],
                    pooling=pooling,
                    use_batchnorm=use_batchnorm,
                )
                for i in range(len(feature_dims))
            ]
        )

        point_dim = feature_dims[-1]
        self.point_fuse = nn.Sequential(
            nn.Linear(sum(dims), point_dim),
            nn.LayerNorm(point_dim),
            nn.SiLU(inplace=True),
            nn.Linear(point_dim, point_dim),
            nn.LayerNorm(point_dim),
            nn.SiLU(inplace=True),
        )

        if self.use_geom_head:
            self.geom_head = AtomicGeometricInvariantHead(k_geom=min(n_knn, 16), proj_dim=geom_dim)
            inv_input_dim = (2 * point_dim) + geom_dim
        else:
            self.geom_head = None
            inv_input_dim = 2 * point_dim

        inv_hidden = max(point_dim, latent_size)
        self.inv_head = nn.Sequential(
            nn.Linear(inv_input_dim, inv_hidden),
            nn.LayerNorm(inv_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(p=float(dropout_rate)),
            nn.Linear(inv_hidden, latent_size),
        )

        if self.emit_eq_latent:
            self.eq_projector = nn.Sequential(
                nn.Linear(point_dim, point_dim),
                nn.LayerNorm(point_dim),
                nn.SiLU(inplace=True),
                nn.Linear(point_dim, eq_channels),
            )
            self.eq_z_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.eq_projector = None
            self.register_buffer("eq_z_scale", torch.tensor(1.0), persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if x.dim() != 3 or x.shape[-1] != 3:
            raise ValueError(f"Expected input (B, N, 3), got {tuple(x.shape)}")

        B, N, _ = x.shape
        center = x.mean(dim=1)
        x_centered = x - center.unsqueeze(1)

        k = min(self.n_knn, N)
        x_coord = x.permute(0, 2, 1).contiguous()
        idx_knn = knn(x_coord, k=k)

        radius = torch.linalg.norm(x_centered, dim=-1, keepdim=True)
        feat = self.input_proj(radius)                                 # (B,N,C0)

        point_feats = [feat]
        for layer in self.layers:
            feat = layer(feat, x_centered, idx_knn)
            point_feats.append(feat)

        point_feat = self.point_fuse(torch.cat(point_feats, dim=-1))   # (B,N,C)
        pooled_mean = point_feat.mean(dim=1)
        pooled_max = point_feat.max(dim=1).values
        inv_input = torch.cat([pooled_mean, pooled_max], dim=-1)

        if self.geom_head is not None:
            inv_geom = self.geom_head(x, idx_knn_max=idx_knn)
            inv_input = torch.cat([inv_input, inv_geom.to(inv_input.dtype)], dim=-1)

        inv_z = self.inv_head(inv_input)

        eq_z = None
        if self.eq_projector is not None:
            weights = self.eq_projector(point_feat) / self.eq_softmax_temperature
            weights = torch.softmax(weights, dim=1)
            eq_z = torch.einsum("bnc,bnd->bcd", weights, x_centered)
            eq_z = eq_z * self.eq_z_scale.to(eq_z.dtype)

        return inv_z, eq_z, center
