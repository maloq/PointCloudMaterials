import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6

# ---------------------------------------------------------------------------
# Vector Neuron building blocks (adapted from VN-SPD)
# ---------------------------------------------------------------------------
class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super().__init__()
        if share_nonlinearity:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        return self.negative_slope * x + (1 - self.negative_slope) * (
            mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d)
        )


class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
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
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, negative_slope=0.2, use_batchnorm=True):
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
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        return self.negative_slope * p + (1 - self.negative_slope) * (
            mask * p + (1 - mask) * (p - (dotprod / (d_norm_sq + EPS)) * d)
        )


class VNMaxPool(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]], indexing='ij') + (idx,)
        return x[index_tuple]


def mean_pool(x: torch.Tensor, dim=-1, keepdim=False) -> torch.Tensor:
    return x.mean(dim=dim, keepdim=keepdim)


class VNStdFeature(nn.Module):
    def __init__(self, in_channels, dim=4, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2, use_batchnorm=True):
        super().__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame
        self.vn1 = VNLinearLeakyReLU(in_channels, in_channels // 2, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope, use_batchnorm=use_batchnorm)
        self.vn2 = VNLinearLeakyReLU(in_channels // 2, in_channels // 4, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope, use_batchnorm=use_batchnorm)
        self.vn_lin = nn.Linear(in_channels // 4, 3 if not normalize_frame else 2, bias=False)

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
        return x_std, z0


# ---------------------------------------------------------------------------
# VN PointNet-style encoder and rotation head
# ---------------------------------------------------------------------------
class PointNetEncoder(nn.Module):
    def __init__(self, latent_size=256, n_knn=20, pooling='mean', feature_transform=False):
        super().__init__()
        self.n_knn = n_knn
        self.pooling = pooling
        self.conv_pos = VNLinearLeakyReLU(3, 64 // 3, dim=5, negative_slope=0.0)
        self.conv1 = VNLinearLeakyReLU(64 // 3, 64 // 3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64 // 3 * 2, 128 // 3, dim=4, negative_slope=0.0)
        self.conv3 = VNLinear(128 // 3, latent_size // 3)
        self.bn3 = VNBatchNorm(latent_size // 3, dim=4)
        self.std_feature = VNStdFeature(latent_size // 3 * 2, dim=4, normalize_frame=False, negative_slope=0.0)
        if pooling == 'max':
            self.pool = VNMaxPool(64 // 3)
        else:
            self.pool = mean_pool
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(latent_size)

    def forward(self, x: torch.Tensor):
        # x: (B, N, 3)
        x = x.permute(0, 2, 1)  # (B,3,N)
        B, D, N = x.size()
        x = x.unsqueeze(1)
        feat = get_graph_feature_cross(x, k=self.n_knn)
        x = self.conv_pos(feat)
        x = self.pool(x)
        x = self.conv1(x)
        if self.feature_transform:
            x_global = self.fstn(x).unsqueeze(-1).repeat(1, 1, 1, N)
            x = torch.cat((x, x_global), 1)
        x = self.conv2(x)
        x = self.bn3(self.conv3(x))
        x_mean_out = x.mean(dim=-1, keepdim=True)
        x = torch.cat((x, x_mean_out.expand_as(x)), 1)
        x, trans = self.std_feature(x)
        x = x.view(B, -1, N)
        inv_z = x.max(dim=-1, keepdim=False)[0]
        center_loc = x_mean_out.mean(dim=2)
        return inv_z, x_mean_out, center_loc


class STNkd(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.conv1 = VNLinearLeakyReLU(d, 64 // 3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 128 // 3, dim=4, negative_slope=0.0)
        self.conv3 = VNLinearLeakyReLU(128 // 3, 1024 // 3, dim=4, negative_slope=0.0)
        self.fc1 = VNLinearLeakyReLU(1024 // 3, 512 // 3, dim=3, negative_slope=0.0)
        self.fc2 = VNLinearLeakyReLU(512 // 3, 256 // 3, dim=3, negative_slope=0.0)
        self.pool = VNMaxPool(1024 // 3)
        self.fc3 = VNLinear(256 // 3, d)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)


class SimpleRot(nn.Module):
    def __init__(self, in_ch, strict='None'):
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


# Utilities
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]


def get_graph_feature_cross(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
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
