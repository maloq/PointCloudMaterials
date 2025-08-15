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
    def __init__(self, in_channels, dim=4, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2, use_batchnorm=True, hidden_dims=None):
        super().__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame

        if hidden_dims is None:
            # Maintain original behavior if hidden_dims is not provided
            h1 = in_channels // 2
            h2 = in_channels // 4
        else:
            h1, h2 = hidden_dims

        self.vn1 = VNLinearLeakyReLU(in_channels, h1, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope, use_batchnorm=use_batchnorm)
        self.vn2 = VNLinearLeakyReLU(h1, h2, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope, use_batchnorm=use_batchnorm)
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
        return x_std, z0




class VNResBlock(nn.Module):
    """
    ResNet-style residual block for Vector Neurons.
    Uses a bottleneck (1x 'channels' -> mid -> 'channels') in VN space.
    """
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
            channels, mid, dim=dim,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = VNLinearLeakyReLU(
            mid, channels, dim=dim,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x)) + x


def _make_vn_layer(channels: int, num_blocks: int, dim: int = 4, **kwargs) -> nn.Sequential:
    return nn.Sequential(*[VNResBlock(channels, dim=dim, **kwargs) for _ in range(num_blocks)])


class PointNetEncoderVN(nn.Module):
    def __init__(self, latent_size=64, n_knn=20, pooling='mean',
                 feature_transform=True, hidden_dim1=256, hidden_dim2=1024,
                 stn_hidden_dims=(64, 128, 1024), stn_fc_dims=(512, 256),
                 std_feature_hidden_dims=None,
                 use_batchnorm=True,
                 # NEW: residual options
                 residual=True, blocks=(2, 2, 2), bottleneck_ratio=0.5):
        """
        blocks: number of residual blocks at each stage (c1, c2, c3).
        bottleneck_ratio: bottleneck width inside each VNResBlock.
        Set residual=False to disable residual stacks (keeps original depth).
        """
        super().__init__()
        self.n_knn = n_knn
        self.pooling = pooling

        # As before: channels are divided by 3 for VN (vector) channels.
        c1 = hidden_dim1 // 3
        c2 = hidden_dim2 // 3
        c3 = latent_size // 3

        # Stem on 5D tensor from graph features
        self.conv_pos = VNLinearLeakyReLU(3, c1, dim=5, negative_slope=0.1, use_batchnorm=use_batchnorm)

        # Stage 1 (dim=4)
        self.conv1 = VNLinearLeakyReLU(c1, c1, dim=4, negative_slope=0.1, use_batchnorm=use_batchnorm)

        # Residual stacks (all dim=4)
        self.residual = residual
        if residual:
            self.res1 = _make_vn_layer(
                c1, blocks[0], dim=4, bottleneck_ratio=bottleneck_ratio,
                negative_slope=0.1, use_batchnorm=use_batchnorm
            )
        else:
            self.res1 = nn.Identity()

        # STN is unchanged
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(c1, hidden_dims=stn_hidden_dims, fc_dims=stn_fc_dims)

        # Stage 2 projection & residuals
        self.conv2 = VNLinearLeakyReLU(c1 * 2, c2, dim=4, negative_slope=0.1, use_batchnorm=use_batchnorm)
        if residual:
            self.res2 = _make_vn_layer(
                c2, blocks[1], dim=4, bottleneck_ratio=bottleneck_ratio,
                negative_slope=0.1, use_batchnorm=use_batchnorm
            )
        else:
            self.res2 = nn.Identity()

        # Stage 3 projection & residuals
        self.conv3 = VNLinearLeakyReLU(c2, c3, dim=4, negative_slope=0.1, use_batchnorm=use_batchnorm)
        if residual:
            self.res3 = _make_vn_layer(
                c3, blocks[2], dim=4, bottleneck_ratio=bottleneck_ratio,
                negative_slope=0.1, use_batchnorm=use_batchnorm
            )
        else:
            self.res3 = nn.Identity()

        # Keep the original head conv
        self.conv4 = VNLinearLeakyReLU(c3, c3, dim=4, negative_slope=0.1, use_batchnorm=use_batchnorm)

        # Frame/std feature is unchanged
        self.std_feature = VNStdFeature(
            c3 * 2, dim=4, normalize_frame=False, negative_slope=0.0,
            hidden_dims=std_feature_hidden_dims
        )

        # Pooling choice for the 5D graph features right after conv_pos
        if pooling == 'max':
            self.pool = VNMaxPool(c1)
        else:
            self.pool = mean_pool

    def forward(self, x: torch.Tensor):
        # x: (B, N, 3)
        x = x.permute(0, 2, 1)  # (B,3,N)
        B, D, N = x.size()
        x = x.unsqueeze(1)

        # Build local graph and lift to VN features (5D)
        feat = get_graph_feature_cross(x, k=self.n_knn)
        x = self.conv_pos(feat)               # (B, c1, 3, N, k)
        x = self.pool(x)                      # (B, c1, 3, N)

        # Stage 1
        x = self.conv1(x)                     # (B, c1, 3, N)
        x = self.res1(x)                      # deeper @ c1

        # Optional STN (feature transform), identical placement as before
        if self.feature_transform:
            x_global = self.fstn(x).unsqueeze(-1).repeat(1, 1, 1, N)
            x = torch.cat((x, x_global), 1)
        else:
            x_mean = x.mean(dim=-1, keepdim=True)
            x = torch.cat((x, x_mean.expand_as(x)), 1)

        x = self.conv2(x)                     # (B, c2, 3, N)
        x = self.res2(x)                      # deeper @ c2

        x = self.conv3(x)                     # (B, c3, 3, N)
        x = self.res3(x)                      # deeper @ c3
        x = self.conv4(x)                     # (B, c3, 3, N)

        eq_z = x.view(B, -1, D, N)

        x_mean_out = x.mean(dim=-1, keepdim=True)      # (B, c3, 3, 1)
        x = torch.cat((x, x_mean_out.expand_as(x)), 1) # (B, 2*c3, 3, N)
        x, trans = self.std_feature(x)                 # (B, c3, 3, N), frame

        x = x.view(B, -1, N)
        inv_z = x.max(dim=-1, keepdim=False)[0]
        center_loc = x_mean_out.mean(dim=2)
        return inv_z, eq_z, center_loc



class STNkd(nn.Module):
    def __init__(self, d=64, hidden_dims=(64, 128, 1024), fc_dims=(512, 256)):
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


class ComplexRot(nn.Module):
    def __init__(self, in_ch, strict='None'):
        super().__init__()
        self.linear1 = VNLinearLeakyReLU(in_ch, in_ch * 6, dim=4, negative_slope=0.1)
        self.linear2 = VNLinearLeakyReLU(in_ch*6 , in_ch*4, dim=4, negative_slope=0.1)
        self.linear3 = VNLinearLeakyReLU(in_ch*4, in_ch , dim=4, negative_slope=0.1)
        self.linearR = VNLinear(in_ch , 3)
        self.strict = strict

    def constraint_rot(self, rot_mat: torch.Tensor) -> torch.Tensor:
        if self.strict == 'None':
            return rot_mat
        u, _, v = torch.linalg.svd(rot_mat)
        return u @ v.transpose(-1, -2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        R = self.linearR(x)
        rot_mat = torch.mean(R, dim=-1)
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
