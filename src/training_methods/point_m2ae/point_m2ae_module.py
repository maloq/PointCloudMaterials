from __future__ import annotations

from collections.abc import Sequence

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training_methods.contrastive_learning.supervised_cache import (
    cache_limit_for_stage,
    cache_supervised_batch,
    init_supervised_cache,
    log_supervised_metrics,
    reset_supervised_cache,
)
from src.utils.pointcloud_ops import crop_to_num_points
from src.utils.spd_utils import cached_sample_count, get_optimizers_and_scheduler


def _to_bn3(points: torch.Tensor) -> torch.Tensor:
    if points.dim() != 3:
        raise ValueError(
            f"Expected point cloud with shape (B, N, 3) or (B, 3, N), got {tuple(points.shape)}"
        )
    if points.shape[-1] == 3:
        return points
    if points.shape[1] == 3:
        return points.transpose(1, 2).contiguous()
    raise ValueError(f"Expected point cloud with 3 coordinates, got {tuple(points.shape)}")


def _chamfer_l2_squared(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = _to_bn3(pred)
    target = _to_bn3(target)

    pred_f = pred.to(torch.float32)
    target_f = target.to(torch.float32)

    pred_sq = (pred_f ** 2).sum(dim=-1, keepdim=True)
    target_sq = (target_f ** 2).sum(dim=-1).unsqueeze(1)
    cross = -2.0 * torch.bmm(pred_f, target_f.transpose(1, 2))
    d2 = (pred_sq + target_sq + cross).clamp_min_(0.0)

    min_pred = d2.min(dim=2).values
    min_tgt = d2.min(dim=1).values
    return (min_pred.mean(dim=1) + min_tgt.mean(dim=1)).mean()


def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        raise TypeError(f"Expected torch.Tensor for trunc normal init, got {type(tensor)}")
    try:
        return nn.init.trunc_normal_(tensor, std=std)
    except AttributeError:
        with torch.no_grad():
            return tensor.normal_(0.0, std)


def _drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob <= 0.0 or not training:
        return x
    if drop_prob >= 1.0:
        raise ValueError(f"drop_prob must be < 1.0, got {drop_prob}")
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)
        if self.drop_prob < 0.0 or self.drop_prob >= 1.0:
            raise ValueError(f"drop_prob must be in [0, 1), got {self.drop_prob}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)


def farthest_point_sample(
    xyz: torch.Tensor,
    npoint: int,
    *,
    deterministic: bool = False,
) -> torch.Tensor:
    xyz = _to_bn3(xyz)
    bsz, n_pts, _ = xyz.shape
    npoint = int(npoint)
    if npoint <= 0:
        raise ValueError(f"npoint must be > 0, got {npoint}")
    if npoint > n_pts:
        raise ValueError(f"npoint ({npoint}) cannot exceed number of points ({n_pts})")

    device = xyz.device
    xyz_f = xyz.to(torch.float32)

    centroids = torch.zeros((bsz, npoint), dtype=torch.long, device=device)
    distance = torch.full((bsz, n_pts), 1e10, dtype=xyz_f.dtype, device=device)
    if deterministic:
        farthest = torch.zeros((bsz,), dtype=torch.long, device=device)
    else:
        farthest = torch.randint(0, n_pts, (bsz,), device=device)

    batch_idx = torch.arange(bsz, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz_f[batch_idx, farthest, :].view(bsz, 1, 3)
        dist = ((xyz_f - centroid) ** 2).sum(dim=-1)
        update = dist < distance
        distance[update] = dist[update]
        farthest = distance.max(dim=-1).indices
    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    if points.dim() != 3:
        raise ValueError(f"points must have shape (B, N, C), got {tuple(points.shape)}")
    if idx.dim() < 2:
        raise ValueError(f"idx must have shape (B, ...), got {tuple(idx.shape)}")
    if points.shape[0] != idx.shape[0]:
        raise ValueError(
            "Batch size mismatch between points and idx: "
            f"points.shape={tuple(points.shape)}, idx.shape={tuple(idx.shape)}."
        )

    bsz = points.shape[0]
    idx_flat = idx.reshape(bsz, -1)
    gathered = points.gather(1, idx_flat.unsqueeze(-1).expand(-1, -1, points.shape[-1]))
    return gathered.reshape(bsz, *idx.shape[1:], points.shape[-1])


def knn_point(k: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    xyz = _to_bn3(xyz)
    new_xyz = _to_bn3(new_xyz)
    n_pts = xyz.shape[1]
    k_eff = min(int(k), int(n_pts))
    if k_eff <= 0:
        raise ValueError(f"k must be positive, got {k}")
    dist = torch.cdist(new_xyz.to(torch.float32), xyz.to(torch.float32))
    return dist.topk(k=k_eff, dim=-1, largest=False).indices


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, bias: bool = True) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(
        self,
        channel: int,
        kernel_size: int = 1,
        groups: int = 1,
        res_expansion: float = 1.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if channel <= 0:
            raise ValueError(f"channel must be > 0, got {channel}")
        if groups <= 0:
            raise ValueError(f"groups must be > 0, got {groups}")

        mid = int(channel * float(res_expansion))
        if mid <= 0:
            raise ValueError(f"res_expansion produced non-positive channels: {mid}")

        self.act = nn.GELU()
        self.net1 = nn.Sequential(
            nn.Conv1d(channel, mid, kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(mid),
            self.act,
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(mid, channel, kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(channel, channel, kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(mid, channel, kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net2(self.net1(x)) + x)


class PosExtraction(nn.Module):
    def __init__(
        self,
        channels: int,
        blocks: int = 1,
        groups: int = 1,
        res_expansion: float = 1.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        layers = [
            ConvBNReLURes1D(
                channels,
                groups=groups,
                res_expansion=res_expansion,
                bias=bias,
            )
            for _ in range(int(blocks))
        ]
        self.operation = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.operation(x)


class PointNetFeaturePropagation(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        *,
        blocks: int = 1,
        groups: int = 1,
        res_expansion: float = 1.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.fuse = ConvBNReLU1D(in_channel, out_channel, kernel_size=1, bias=bias)
        self.extraction = PosExtraction(
            out_channel,
            blocks=blocks,
            groups=groups,
            res_expansion=res_expansion,
            bias=bias,
        )

    def forward(
        self,
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
        points1: torch.Tensor,
        points2: torch.Tensor,
    ) -> torch.Tensor:
        xyz1 = _to_bn3(xyz1)
        xyz2 = _to_bn3(xyz2)
        if points1.dim() != 3 or points2.dim() != 3:
            raise ValueError(
                "points1/points2 must have shape (B, N, C) and (B, S, C), got "
                f"points1={tuple(points1.shape)}, points2={tuple(points2.shape)}."
            )
        if points1.shape[0] != xyz1.shape[0] or points2.shape[0] != xyz2.shape[0]:
            raise ValueError(
                "Batch size mismatch in PointNetFeaturePropagation: "
                f"xyz1={tuple(xyz1.shape)}, xyz2={tuple(xyz2.shape)}, "
                f"points1={tuple(points1.shape)}, points2={tuple(points2.shape)}."
            )

        bsz, n_pts, _ = xyz1.shape
        _, s_pts, _ = xyz2.shape
        if s_pts <= 0:
            raise ValueError(f"xyz2 must contain at least one point, got shape={tuple(xyz2.shape)}")

        if s_pts == 1:
            interpolated_points = points2.repeat(1, n_pts, 1)
        else:
            dists = torch.cdist(xyz1.to(torch.float32), xyz2.to(torch.float32))
            dists, idx = dists.sort(dim=-1)
            dists = dists[:, :, :3]
            idx = idx[:, :, :3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = dist_recip.sum(dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.unsqueeze(-1),
                dim=2,
            )

        new_points = torch.cat([points1, interpolated_points], dim=-1)
        new_points = new_points.permute(0, 2, 1).contiguous()
        new_points = self.fuse(new_points)
        new_points = self.extraction(new_points)
        return new_points.permute(0, 2, 1).contiguous()


class TokenEmbed(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.in_c = int(in_c)
        self.out_c = int(out_c)
        if self.in_c <= 0 or self.out_c <= 0:
            raise ValueError(f"in_c/out_c must be > 0, got in_c={self.in_c}, out_c={self.out_c}")

        if self.in_c == 3:
            self.first_conv = nn.Sequential(
                nn.Conv1d(self.in_c, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 256, 1),
            )
            self.second_conv = nn.Sequential(
                nn.Conv1d(512, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, self.out_c, 1),
            )
        else:
            self.first_conv = nn.Sequential(
                nn.Conv1d(self.in_c, self.in_c, 1),
                nn.BatchNorm1d(self.in_c),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.in_c, self.in_c, 1),
            )
            self.second_conv = nn.Sequential(
                nn.Conv1d(self.in_c * 2, self.out_c, 1),
                nn.BatchNorm1d(self.out_c),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.out_c, self.out_c, 1),
            )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        if point_groups.dim() != 4:
            raise ValueError(
                f"point_groups must have shape (B, G, S, C), got {tuple(point_groups.shape)}"
            )
        bsz, num_group, group_size, channels = point_groups.shape
        if channels != self.in_c:
            raise ValueError(
                f"TokenEmbed expected input channels={self.in_c}, got {channels} "
                f"for shape={tuple(point_groups.shape)}."
            )

        x = point_groups.reshape(bsz * num_group, group_size, channels)
        feat = self.first_conv(x.transpose(2, 1).contiguous())
        global_feat = feat.max(dim=2, keepdim=True).values
        feat = torch.cat([global_feat.expand(-1, -1, group_size), feat], dim=1)
        feat = self.second_conv(feat)
        global_feat = feat.max(dim=2).values
        return global_feat.reshape(bsz, num_group, self.out_c)


class Group(nn.Module):
    """FPS + kNN grouping used by Point-M2AE."""

    def __init__(
        self,
        num_group: int,
        group_size: int,
        *,
        deterministic_fps: bool = False,
    ) -> None:
        super().__init__()
        self.num_group = int(num_group)
        self.group_size = int(group_size)
        self.deterministic_fps = bool(deterministic_fps)

        if self.num_group <= 0:
            raise ValueError(f"num_group must be > 0, got {self.num_group}")
        if self.group_size <= 0:
            raise ValueError(f"group_size must be > 0, got {self.group_size}")

    def forward(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xyz = _to_bn3(xyz)
        bsz, n_pts, _ = xyz.shape

        if n_pts < self.num_group:
            raise ValueError(
                f"Number of points ({n_pts}) must be >= num_group ({self.num_group})."
            )
        if n_pts < self.group_size:
            raise ValueError(
                f"Number of points ({n_pts}) must be >= group_size ({self.group_size})."
            )

        fps_idx = farthest_point_sample(
            xyz,
            self.num_group,
            deterministic=self.deterministic_fps,
        )
        center = index_points(xyz, fps_idx)
        group_idx = knn_point(self.group_size, xyz, center)
        neighborhood = index_points(xyz, group_idx)
        neighborhood = neighborhood - center.unsqueeze(2)

        if group_idx.shape != (bsz, self.num_group, self.group_size):
            raise RuntimeError(
                "Unexpected kNN index shape in Group.forward: "
                f"expected {(bsz, self.num_group, self.group_size)}, got {tuple(group_idx.shape)}."
            )
        return neighborhood.contiguous(), center.contiguous(), group_idx.contiguous()


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = int(hidden_features) if hidden_features is not None else int(in_features)
        out = int(out_features) if out_features is not None else int(in_features)
        self.fc1 = nn.Linear(int(in_features), hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out)
        self.drop = nn.Dropout(float(drop))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {self.num_heads}")
        dim = int(dim)
        if dim % self.num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({self.num_heads})")
        head_dim = dim // self.num_heads
        self.scale = float(qk_scale) if qk_scale is not None else head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=bool(qkv_bias))
        self.attn_drop = nn.Dropout(float(attn_drop))
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(float(proj_drop))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seq_len, channels = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            if mask.shape != (bsz, seq_len, seq_len):
                raise ValueError(
                    "Attention mask shape mismatch: "
                    f"expected {(bsz, seq_len, seq_len)}, got {tuple(mask.shape)}."
                )
            mask_f = mask.to(dtype=attn.dtype)
            attn = attn + mask_f.unsqueeze(1) * -100000.0
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(bsz, seq_len, channels)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        dim = int(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=int(num_heads),
            qkv_bias=bool(qkv_bias),
            qk_scale=qk_scale,
            attn_drop=float(attn_drop),
            proj_drop=float(drop),
        )
        self.drop_path = DropPath(float(drop_path)) if float(drop_path) > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * float(mlp_ratio))
        self.mlp = Mlp(in_features=dim, hidden_features=hidden, out_features=dim, drop=float(drop))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        depth: int,
        num_heads: int,
        drop_path_rate: list[float] | float = 0.0,
    ) -> None:
        super().__init__()
        depth = int(depth)
        if depth <= 0:
            raise ValueError(f"EncoderBlock depth must be > 0, got {depth}")

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=int(embed_dim),
                    num_heads=int(num_heads),
                    drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else float(drop_path_rate),
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor, vis_mask: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x + pos, vis_mask)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        depth: int,
        num_heads: int,
        drop_path_rate: list[float] | float = 0.0,
    ) -> None:
        super().__init__()
        depth = int(depth)
        if depth <= 0:
            raise ValueError(f"DecoderBlock depth must be > 0, got {depth}")

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=int(embed_dim),
                    num_heads=int(num_heads),
                    drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else float(drop_path_rate),
                )
                for i in range(depth)
            ]
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x + pos)
        return x


class HierarchicalEncoder(nn.Module):
    def __init__(
        self,
        *,
        mask_ratio: float,
        encoder_depths: Sequence[int],
        encoder_dims: Sequence[int],
        local_radius: Sequence[float],
        num_heads: int,
        drop_path_rate: float,
    ) -> None:
        super().__init__()
        self.mask_ratio = float(mask_ratio)
        self.encoder_depths = [int(v) for v in encoder_depths]
        self.encoder_dims = [int(v) for v in encoder_dims]
        self.local_radius = [float(v) for v in local_radius]
        if not (0.0 <= self.mask_ratio < 1.0):
            raise ValueError(f"mask_ratio must be in [0, 1), got {self.mask_ratio}")
        if not self.encoder_depths:
            raise ValueError("encoder_depths must be non-empty")
        if len(self.encoder_depths) != len(self.encoder_dims):
            raise ValueError(
                "encoder_depths and encoder_dims must have the same length, got "
                f"{len(self.encoder_depths)} and {len(self.encoder_dims)}."
            )
        if len(self.local_radius) != len(self.encoder_dims):
            raise ValueError(
                "local_radius must match number of encoder scales, got "
                f"{len(self.local_radius)} vs {len(self.encoder_dims)}."
            )

        self.token_embed = nn.ModuleList()
        self.encoder_pos_embeds = nn.ModuleList()
        for i, dim in enumerate(self.encoder_dims):
            in_c = 3 if i == 0 else self.encoder_dims[i - 1]
            self.token_embed.append(TokenEmbed(in_c=in_c, out_c=dim))
            self.encoder_pos_embeds.append(
                nn.Sequential(
                    nn.Linear(3, dim),
                    nn.GELU(),
                    nn.Linear(dim, dim),
                )
            )

        total_depth = sum(self.encoder_depths)
        dpr = torch.linspace(0, float(drop_path_rate), total_depth).tolist()
        self.encoder_blocks = nn.ModuleList()
        depth_cursor = 0
        for i, depth in enumerate(self.encoder_depths):
            self.encoder_blocks.append(
                EncoderBlock(
                    embed_dim=self.encoder_dims[i],
                    depth=depth,
                    num_heads=int(num_heads),
                    drop_path_rate=dpr[depth_cursor: depth_cursor + depth],
                )
            )
            depth_cursor += depth

        self.encoder_norms = nn.ModuleList([nn.LayerNorm(dim) for dim in self.encoder_dims])
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m) -> None:
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def rand_mask(self, center: torch.Tensor) -> torch.Tensor:
        bsz, num_group, _ = center.shape
        num_mask = int(self.mask_ratio * num_group)
        num_mask = min(max(num_mask, 0), num_group)
        if num_mask == 0:
            return torch.zeros((bsz, num_group), dtype=torch.bool, device=center.device)

        noise = torch.rand((bsz, num_group), device=center.device)
        ids_shuffle = noise.argsort(dim=1)
        mask = torch.zeros((bsz, num_group), dtype=torch.bool, device=center.device)
        mask.scatter_(1, ids_shuffle[:, :num_mask], True)
        return mask

    @staticmethod
    def local_att_mask(
        xyz: torch.Tensor,
        radius: float,
        dist: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if dist is None or dist.shape[1] != xyz.shape[1]:
            dist = torch.cdist(xyz.to(torch.float32), xyz.to(torch.float32), p=2)
        mask = dist >= float(radius)
        return mask, dist

    def forward(
        self,
        neighborhoods: list[torch.Tensor],
        centers: list[torch.Tensor],
        idxs: list[torch.Tensor],
        *,
        eval: bool = False,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        num_scales = len(neighborhoods)
        if num_scales != len(centers) or num_scales != len(idxs):
            raise ValueError(
                "neighborhoods/centers/idxs length mismatch: "
                f"{len(neighborhoods)}, {len(centers)}, {len(idxs)}."
            )
        if num_scales != len(self.encoder_dims):
            raise ValueError(
                "Number of scales passed to HierarchicalEncoder does not match encoder config: "
                f"scales={num_scales}, encoder_scales={len(self.encoder_dims)}."
            )

        bool_masked_pos: list[torch.Tensor] = []
        if eval:
            bsz, g_top, _ = centers[-1].shape
            bool_masked_pos.append(torch.zeros((bsz, g_top), dtype=torch.bool, device=centers[-1].device))
        else:
            bool_masked_pos.append(self.rand_mask(centers[-1]))

        # Multi-scale masking by back-propagation from coarse to fine.
        for i in range(num_scales - 1, 0, -1):
            parent_mask = bool_masked_pos[-1]  # (B, G_parent), True=masked
            idx = idxs[i]  # (B, G_parent, K)
            if idx.dim() != 3:
                raise ValueError(
                    f"idxs[{i}] must have shape (B, G, K), got {tuple(idx.shape)}."
                )
            bsz, g_parent, _ = idx.shape
            child_count = int(centers[i - 1].shape[1])
            if parent_mask.shape != (bsz, g_parent):
                raise ValueError(
                    f"Mask/index shape mismatch at scale {i}: "
                    f"parent_mask={tuple(parent_mask.shape)}, idx={tuple(idx.shape)}."
                )

            child_mask = torch.ones((bsz, child_count), dtype=torch.bool, device=idx.device)
            for b in range(bsz):
                visible_parent = ~parent_mask[b]
                if not bool(visible_parent.any()):
                    continue
                visible_child_idx = idx[b, visible_parent].reshape(-1)
                child_mask[b, visible_child_idx] = False
            bool_masked_pos.append(child_mask)

        bool_masked_pos.reverse()

        x_vis_list: list[torch.Tensor] = []
        mask_vis_list: list[torch.Tensor] = []
        xyz_dist: torch.Tensor | None = None
        tokens_for_next_level: torch.Tensor | None = None

        for i in range(num_scales):
            if i == 0:
                group_input_tokens = self.token_embed[i](neighborhoods[0])
            else:
                if tokens_for_next_level is None:
                    raise RuntimeError(
                        f"Internal error: tokens_for_next_level is None at encoder scale {i}."
                    )
                x_vis_neighborhoods = index_points(tokens_for_next_level, idxs[i])
                group_input_tokens = self.token_embed[i](x_vis_neighborhoods)

            bool_vis_pos = ~bool_masked_pos[i]
            batch_size, _, channels = group_input_tokens.shape
            vis_tokens_len = bool_vis_pos.long().sum(dim=1)
            max_tokens_len = int(vis_tokens_len.max().item()) if vis_tokens_len.numel() > 0 else 0
            if max_tokens_len <= 0:
                raise RuntimeError(
                    "All tokens are masked at encoder scale "
                    f"{i}. Increase visible token count by lowering mask_ratio "
                    f"(current mask_ratio={self.mask_ratio})."
                )

            x_vis = group_input_tokens.new_zeros((batch_size, max_tokens_len, channels))
            masked_center = centers[i].new_zeros((batch_size, max_tokens_len, 3))
            mask_vis = torch.ones(
                (batch_size, max_tokens_len, max_tokens_len),
                dtype=torch.bool,
                device=group_input_tokens.device,
            )

            for bz in range(batch_size):
                vis_len = int(vis_tokens_len[bz].item())
                if vis_len <= 0:
                    continue
                vis_tokens = group_input_tokens[bz][bool_vis_pos[bz]]
                vis_centers = centers[i][bz][bool_vis_pos[bz]]
                x_vis[bz, :vis_len] = vis_tokens
                masked_center[bz, :vis_len] = vis_centers
                mask_vis[bz, :vis_len, :vis_len] = False

            if self.local_radius[i] > 0:
                mask_radius, xyz_dist = self.local_att_mask(masked_center, self.local_radius[i], xyz_dist)
                mask_vis_att = mask_radius | mask_vis
            else:
                mask_vis_att = mask_vis

            pos = self.encoder_pos_embeds[i](masked_center)
            x_vis = self.encoder_blocks[i](x_vis, pos, mask_vis_att)

            valid_visible = ~mask_vis[:, :, 0]
            x_vis_list.append(x_vis)
            mask_vis_list.append(valid_visible)

            if i < num_scales - 1:
                full_tokens = group_input_tokens.clone()
                expected_visible = int(bool_vis_pos.sum().item())
                encoded_visible = int(valid_visible.sum().item())
                if expected_visible != encoded_visible:
                    raise RuntimeError(
                        "Visible-token accounting mismatch in HierarchicalEncoder. "
                        f"scale={i}, expected_visible={expected_visible}, "
                        f"encoded_visible={encoded_visible}, "
                        f"group_input_tokens.shape={tuple(group_input_tokens.shape)}, "
                        f"x_vis.shape={tuple(x_vis.shape)}."
                    )
                full_tokens[bool_vis_pos] = x_vis[valid_visible]
                tokens_for_next_level = full_tokens

        for i in range(len(x_vis_list)):
            x_vis_list[i] = self.encoder_norms[i](x_vis_list[i])

        return x_vis_list, mask_vis_list, bool_masked_pos


class PointM2AEBackbone(nn.Module):
    """PyTorch Point-M2AE backbone adapted from the official implementation."""

    def __init__(
        self,
        *,
        mask_ratio: float,
        group_sizes: Sequence[int],
        num_groups: Sequence[int],
        encoder_depths: Sequence[int],
        encoder_dims: Sequence[int],
        local_radius: Sequence[float],
        decoder_depths: Sequence[int],
        decoder_dims: Sequence[int],
        decoder_up_blocks: Sequence[int],
        num_heads: int,
        drop_path_rate: float,
        deterministic_fps: bool,
    ) -> None:
        super().__init__()

        self.mask_ratio = float(mask_ratio)
        self.group_sizes = [int(v) for v in group_sizes]
        self.num_groups = [int(v) for v in num_groups]
        self.encoder_depths = [int(v) for v in encoder_depths]
        self.encoder_dims = [int(v) for v in encoder_dims]
        self.local_radius = [float(v) for v in local_radius]
        self.decoder_depths = [int(v) for v in decoder_depths]
        self.decoder_dims = [int(v) for v in decoder_dims]
        self.decoder_up_blocks = [int(v) for v in decoder_up_blocks]
        self.num_heads = int(num_heads)
        self.drop_path_rate = float(drop_path_rate)
        self.deterministic_fps = bool(deterministic_fps)

        if not (0.0 <= self.mask_ratio < 1.0):
            raise ValueError(f"mask_ratio must be in [0, 1), got {self.mask_ratio}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {self.num_heads}")
        if self.drop_path_rate < 0.0:
            raise ValueError(f"drop_path_rate must be >= 0, got {self.drop_path_rate}")
        if not self.group_sizes:
            raise ValueError("group_sizes must be non-empty")
        if len(self.group_sizes) < 2:
            raise ValueError(
                "Point-M2AE requires at least 2 grouping scales, got "
                f"{len(self.group_sizes)}."
            )
        if len(self.group_sizes) != len(self.num_groups):
            raise ValueError(
                "group_sizes and num_groups must have the same length, got "
                f"{len(self.group_sizes)} and {len(self.num_groups)}."
            )
        if any(v <= 0 for v in self.group_sizes):
            raise ValueError(f"group_sizes must be positive, got {self.group_sizes}")
        if any(v <= 0 for v in self.num_groups):
            raise ValueError(f"num_groups must be positive, got {self.num_groups}")
        if any(self.num_groups[i] > self.num_groups[i - 1] for i in range(1, len(self.num_groups))):
            raise ValueError(
                "num_groups must be non-increasing across scales, got "
                f"{self.num_groups}."
            )
        if len(self.encoder_depths) != len(self.group_sizes):
            raise ValueError(
                "encoder_depths length must match number of scales, got "
                f"{len(self.encoder_depths)} vs {len(self.group_sizes)}."
            )
        if len(self.encoder_dims) != len(self.group_sizes):
            raise ValueError(
                "encoder_dims length must match number of scales, got "
                f"{len(self.encoder_dims)} vs {len(self.group_sizes)}."
            )
        if len(self.local_radius) != len(self.group_sizes):
            raise ValueError(
                "local_radius length must match number of scales, got "
                f"{len(self.local_radius)} vs {len(self.group_sizes)}."
            )
        if len(self.decoder_depths) != len(self.decoder_dims):
            raise ValueError(
                "decoder_depths and decoder_dims must have equal lengths, got "
                f"{len(self.decoder_depths)} and {len(self.decoder_dims)}."
            )
        expected_decoder_levels = len(self.group_sizes) - 1
        if len(self.decoder_dims) != expected_decoder_levels:
            raise ValueError(
                "decoder_dims/decoder_depths must have length num_scales-1. "
                f"Got decoder_levels={len(self.decoder_dims)}, num_scales={len(self.group_sizes)}."
            )
        if len(self.decoder_up_blocks) != len(self.decoder_dims) - 1:
            raise ValueError(
                "decoder_up_blocks must have length decoder_levels-1, got "
                f"{len(self.decoder_up_blocks)} vs {len(self.decoder_dims) - 1}."
            )

        self.group_dividers = nn.ModuleList(
            [
                Group(
                    num_group=self.num_groups[i],
                    group_size=self.group_sizes[i],
                    deterministic_fps=self.deterministic_fps,
                )
                for i in range(len(self.group_sizes))
            ]
        )

        self.h_encoder = HierarchicalEncoder(
            mask_ratio=self.mask_ratio,
            encoder_depths=self.encoder_depths,
            encoder_dims=self.encoder_dims,
            local_radius=self.local_radius,
            num_heads=self.num_heads,
            drop_path_rate=self.drop_path_rate,
        )

        self.mask_token = nn.Parameter(torch.zeros(1, self.decoder_dims[0]))
        _trunc_normal_(self.mask_token, std=0.02)

        self.h_decoder = nn.ModuleList()
        self.decoder_pos_embeds = nn.ModuleList()
        self.token_prop = nn.ModuleList()

        total_decoder_depth = sum(self.decoder_depths)
        dpr = torch.linspace(0, self.drop_path_rate, total_decoder_depth).tolist()
        depth_cursor = 0
        for i in range(len(self.decoder_dims)):
            depth = self.decoder_depths[i]
            self.h_decoder.append(
                DecoderBlock(
                    embed_dim=self.decoder_dims[i],
                    depth=depth,
                    num_heads=self.num_heads,
                    drop_path_rate=dpr[depth_cursor: depth_cursor + depth],
                )
            )
            depth_cursor += depth

            self.decoder_pos_embeds.append(
                nn.Sequential(
                    nn.Linear(3, self.decoder_dims[i]),
                    nn.GELU(),
                    nn.Linear(self.decoder_dims[i], self.decoder_dims[i]),
                )
            )
            if i > 0:
                self.token_prop.append(
                    PointNetFeaturePropagation(
                        self.decoder_dims[i] + self.decoder_dims[i - 1],
                        self.decoder_dims[i],
                        blocks=self.decoder_up_blocks[i - 1],
                        groups=1,
                        res_expansion=1.0,
                        bias=True,
                    )
                )

        self.decoder_norm = nn.LayerNorm(self.decoder_dims[-1])
        self.rec_head = nn.Conv1d(self.decoder_dims[-1], 3 * self.group_sizes[0], 1)

    def _build_hierarchical_groups(
        self,
        points: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        neighborhoods: list[torch.Tensor] = []
        centers: list[torch.Tensor] = []
        idxs: list[torch.Tensor] = []

        source = points
        for i, divider in enumerate(self.group_dividers):
            neighborhood, center, idx = divider(source)
            if i > 0:
                prev_groups = centers[i - 1].shape[1]
                if int(idx.max().item()) >= prev_groups:
                    raise RuntimeError(
                        f"Grouping index out of range at scale {i}: "
                        f"idx.max={int(idx.max().item())}, prev_groups={prev_groups}."
                    )
            neighborhoods.append(neighborhood)
            centers.append(center)
            idxs.append(idx)
            source = center

        return neighborhoods, centers, idxs

    def forward_pretrain(
        self,
        points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        neighborhoods, centers, idxs = self._build_hierarchical_groups(points)
        x_vis_list, mask_vis_list, masks = self.h_encoder(
            neighborhoods,
            centers,
            idxs,
            eval=False,
        )

        centers_dec = list(reversed(centers))
        neighborhoods_dec = list(reversed(neighborhoods))
        x_vis_dec = list(reversed(x_vis_list))
        masks_dec = list(reversed(masks))
        mask_vis_dec = list(reversed(mask_vis_list))

        x_full: torch.Tensor | None = None
        center_0: torch.Tensor | None = None
        for i in range(len(self.decoder_dims)):
            center = centers_dec[i]
            if i == 0:
                x_full = x_vis_dec[i]
                mask = masks_dec[i]
                if x_full.dim() != 3:
                    raise RuntimeError(
                        f"Expected x_full to be 3D at decoder level 0, got {tuple(x_full.shape)}."
                    )
                bsz, _, channels = x_full.shape
                if mask.shape != (bsz, center.shape[1]):
                    raise RuntimeError(
                        "Mask/center shape mismatch at decoder level 0: "
                        f"mask={tuple(mask.shape)}, center={tuple(center.shape)}."
                    )

                center_vis = center[~mask].reshape(bsz, -1, 3)
                center_mask = center[mask].reshape(bsz, -1, 3)
                center_0 = torch.cat([center_vis, center_mask], dim=1)

                pos_vis = self.decoder_pos_embeds[i](center[~mask]).reshape(bsz, -1, channels)
                pos_mask = self.decoder_pos_embeds[i](center[mask]).reshape(bsz, -1, channels)
                pos_full = torch.cat([pos_vis, pos_mask], dim=1)

                num_mask = int(pos_mask.shape[1])
                mask_token = self.mask_token.expand(bsz, num_mask, -1)
                x_full = torch.cat([x_full, mask_token], dim=1)
            else:
                if x_full is None or center_0 is None:
                    raise RuntimeError(
                        f"Decoder state is not initialized before level {i}."
                    )
                x_vis = x_vis_dec[i]
                bool_vis_pos = ~masks_dec[i]
                mask_vis = mask_vis_dec[i]
                bsz, num_tokens, _ = center.shape
                channels = x_vis.shape[-1]
                x_full_en = x_vis.new_zeros((bsz, num_tokens, channels))
                expected_visible = int(bool_vis_pos.sum().item())
                encoded_visible = int(mask_vis.sum().item())
                if expected_visible != encoded_visible:
                    raise RuntimeError(
                        "Visible-token accounting mismatch in Point-M2AE decoder. "
                        f"decoder_level={i}, expected_visible={expected_visible}, "
                        f"encoded_visible={encoded_visible}, "
                        f"bool_vis_pos.shape={tuple(bool_vis_pos.shape)}, "
                        f"mask_vis.shape={tuple(mask_vis.shape)}, "
                        f"x_vis.shape={tuple(x_vis.shape)}."
                    )
                x_full_en[bool_vis_pos] = x_vis[mask_vis]

                if i == 1:
                    x_full = self.token_prop[i - 1](center, center_0, x_full_en, x_full)
                else:
                    x_full = self.token_prop[i - 1](center, centers_dec[i - 1], x_full_en, x_full)
                pos_full = self.decoder_pos_embeds[i](center)

            x_full = self.h_decoder[i](x_full, pos_full)

        if x_full is None:
            raise RuntimeError("Decoder produced no output tensor.")
        x_full = self.decoder_norm(x_full)

        if len(masks_dec) < 2 or len(neighborhoods_dec) < 2:
            raise RuntimeError(
                "Point-M2AE reconstruction requires at least 2 scales; got "
                f"masks={len(masks_dec)}, neighborhoods={len(neighborhoods_dec)}."
            )
        target_mask = masks_dec[-2]
        x_rec = x_full[target_mask].reshape(-1, x_full.shape[-1])
        if x_rec.shape[0] <= 0:
            raise RuntimeError(
                "Point-M2AE selected zero masked tokens for reconstruction. "
                f"mask_ratio={self.mask_ratio}, target_mask_shape={tuple(target_mask.shape)}."
            )

        rec_points = self.rec_head(x_rec.unsqueeze(-1)).reshape(x_rec.shape[0], -1, 3)
        gt_points = neighborhoods_dec[-2][target_mask].reshape(x_rec.shape[0], -1, 3)
        loss = _chamfer_l2_squared(rec_points, gt_points)

        top_tokens = x_vis_list[-1]
        global_latent = top_tokens.mean(dim=1) + top_tokens.max(dim=1).values
        mask_fraction = masks_dec[0].to(dtype=loss.dtype).mean()
        return loss, global_latent, mask_fraction, centers[0]

    @torch.no_grad()
    def forward_eval(self, points: torch.Tensor) -> torch.Tensor:
        neighborhoods, centers, idxs = self._build_hierarchical_groups(points)
        x_vis_list, _, _ = self.h_encoder(neighborhoods, centers, idxs, eval=True)
        x_vis = x_vis_list[-1]
        return x_vis.mean(dim=1) + x_vis.max(dim=1).values

    def forward(self, points: torch.Tensor, *, eval: bool = False):
        if eval:
            return self.forward_eval(points)
        return self.forward_pretrain(points)


def _as_number_list(
    value,
    *,
    field_name: str,
    cast,
) -> list:
    if value is None:
        raise ValueError(f"{field_name} is required and cannot be None.")
    if isinstance(value, (str, bytes)):
        raise ValueError(
            f"{field_name} must be a sequence of numbers, got string-like value {value!r}."
        )
    if isinstance(value, Sequence):
        raw = list(value)
    elif hasattr(value, "__iter__"):
        raw = list(value)
    else:
        raise TypeError(
            f"{field_name} must be a sequence of numbers, got type={type(value)}."
        )
    if not raw:
        raise ValueError(f"{field_name} must be non-empty.")
    out = []
    for idx, item in enumerate(raw):
        try:
            out.append(cast(item))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field_name}[{idx}] must be {cast.__name__}-convertible, got {item!r}."
            ) from exc
    return out


class PointM2AEModule(pl.LightningModule):
    """Lightning wrapper for Point-M2AE pretraining."""

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        pm_cfg = getattr(cfg, "point_m2ae", None)

        def _pm_get(name: str, default):
            if pm_cfg is not None and hasattr(pm_cfg, name):
                return getattr(pm_cfg, name)
            return getattr(cfg, f"point_m2ae_{name}", default)

        self.model = PointM2AEBackbone(
            mask_ratio=float(_pm_get("mask_ratio", 0.8)),
            group_sizes=_as_number_list(_pm_get("group_sizes", [16, 8, 8]), field_name="point_m2ae.group_sizes", cast=int),
            num_groups=_as_number_list(_pm_get("num_groups", [512, 256, 64]), field_name="point_m2ae.num_groups", cast=int),
            encoder_depths=_as_number_list(_pm_get("encoder_depths", [5, 5, 5]), field_name="point_m2ae.encoder_depths", cast=int),
            encoder_dims=_as_number_list(_pm_get("encoder_dims", [96, 192, 384]), field_name="point_m2ae.encoder_dims", cast=int),
            local_radius=_as_number_list(_pm_get("local_radius", [0.32, 0.64, 1.28]), field_name="point_m2ae.local_radius", cast=float),
            decoder_depths=_as_number_list(_pm_get("decoder_depths", [1, 1]), field_name="point_m2ae.decoder_depths", cast=int),
            decoder_dims=_as_number_list(_pm_get("decoder_dims", [384, 192]), field_name="point_m2ae.decoder_dims", cast=int),
            decoder_up_blocks=_as_number_list(_pm_get("decoder_up_blocks", [1]), field_name="point_m2ae.decoder_up_blocks", cast=int),
            num_heads=int(_pm_get("num_heads", 6)),
            drop_path_rate=float(_pm_get("drop_path_rate", 0.1)),
            deterministic_fps=bool(_pm_get("deterministic_fps", False)),
        )

        legacy_crop_mode = str(_pm_get("crop_mode", "random")).strip()
        self.crop_mode_train = str(_pm_get("crop_mode_train", legacy_crop_mode)).strip()
        self.crop_mode_eval = str(_pm_get("crop_mode_eval", legacy_crop_mode)).strip()
        if not self.crop_mode_train:
            raise ValueError("point_m2ae crop_mode_train must be a non-empty string.")
        if not self.crop_mode_eval:
            raise ValueError("point_m2ae crop_mode_eval must be a non-empty string.")
        self.center_input = bool(_pm_get("center_input", True))

        data_cfg = getattr(cfg, "data", None)
        model_points = getattr(data_cfg, "model_points", None) if data_cfg is not None else None
        if model_points is None:
            model_points = getattr(cfg, "model_points", None)
        self.model_points = int(model_points) if model_points is not None and int(model_points) > 0 else None

        init_supervised_cache(self, cfg)
        self.cache_train_supervised_metrics = bool(getattr(cfg, "cache_train_supervised_metrics", False))

    @staticmethod
    def _unpack_batch(batch) -> tuple[torch.Tensor, dict]:
        if isinstance(batch, dict):
            points = batch["points"]
            meta = {
                "class_id": batch.get("class_id"),
                "instance_id": batch.get("instance_id"),
                "rotation": batch.get("rotation"),
            }
            return points, meta
        if isinstance(batch, (tuple, list)):
            return batch[0], {}
        return batch, {}

    def _resolve_crop_mode(self, stage: str | None = None) -> str:
        if stage is None:
            return self.crop_mode_train if self.training else self.crop_mode_eval
        stage_name = str(stage).strip().lower()
        if stage_name == "train":
            return self.crop_mode_train
        if stage_name in {"val", "test", "predict"}:
            return self.crop_mode_eval
        raise ValueError(
            f"Unsupported stage {stage!r} for crop-mode resolution. "
            "Expected one of {'train', 'val', 'test', 'predict'} or None."
        )

    def _prepare_points(self, batch_points: torch.Tensor, *, stage: str | None = None) -> torch.Tensor:
        points = _to_bn3(batch_points)
        points = points.to(device=self.device, dtype=self.dtype, non_blocking=True)
        if self.model_points is not None:
            points = crop_to_num_points(
                points,
                self.model_points,
                mode=self._resolve_crop_mode(stage),
            )
        if self.center_input:
            points = points - points.mean(dim=1, keepdim=True)
        return points

    @torch.no_grad()
    def _extract_supervised_features_from_batch(self, batch):
        points_raw, meta = self._unpack_batch(batch)
        class_id = meta.get("class_id")
        if class_id is None:
            return None, None
        points = self._prepare_points(points_raw, stage="test")
        features = self.model.forward_eval(points)
        return features.detach().to(torch.float32), class_id

    def forward(self, batch_points: torch.Tensor):
        points = self._prepare_points(batch_points)
        features = self.model.forward_eval(points)
        if not torch.is_tensor(features):
            raise RuntimeError(
                "Point-M2AE forward_eval returned non-tensor features; "
                f"got type={type(features)}."
            )
        if features.dim() != 2:
            raise RuntimeError(
                "Point-M2AE forward_eval must return a 2D feature tensor (B, D), "
                f"got shape={tuple(features.shape)}."
            )
        return features, features, None

    def _step(self, batch, stage: str) -> torch.Tensor:
        points_raw, meta = self._unpack_batch(batch)
        points = self._prepare_points(points_raw, stage=stage)

        loss, global_latent, mask_fraction, patch_centroids = self.model(points, eval=False)

        should_cache = stage in self._supervised_cache and (
            stage != "train" or self.cache_train_supervised_metrics
        )
        if should_cache and global_latent is not None:
            limit = self._cache_limit_for_stage(stage)
            cache = self._supervised_cache.get(stage)
            already_cached = cached_sample_count(cache) if cache is not None else 0
            if limit is None or already_cached < limit:
                cached_latents = global_latent
                cached_features = global_latent
                if stage in {"val", "test"}:
                    with torch.no_grad():
                        cached_features = self.model.forward_eval(points)
                    cached_latents = cached_features
                self._cache_supervised_batch(
                    stage,
                    cached_latents,
                    meta,
                    encoder_features=cached_features,
                )

        with torch.no_grad():
            latent_std = global_latent.to(torch.float32).std(dim=0).mean()
            patch_spread = patch_centroids.to(torch.float32).std(dim=1).mean()

        self._log_metric(stage, "loss", loss, prog_bar=True, batch_size=points.shape[0])
        self._log_metric(stage, "mask_fraction", mask_fraction, batch_size=points.shape[0])
        self._log_metric(stage, "latent_std", latent_std, batch_size=points.shape[0])
        self._log_metric(stage, "patch_spread", patch_spread, batch_size=points.shape[0])
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        return get_optimizers_and_scheduler(self.hparams, self.parameters())

    def _handle_epoch_boundary(self, stage: str, is_start: bool) -> None:
        if is_start:
            self._reset_supervised_cache(stage)
        else:
            self._log_supervised_metrics(stage)

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self._handle_epoch_boundary("train", True)

    def on_train_epoch_end(self) -> None:
        self._handle_epoch_boundary("train", False)
        super().on_train_epoch_end()

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self._handle_epoch_boundary("val", True)

    def on_validation_epoch_end(self) -> None:
        self._handle_epoch_boundary("val", False)
        super().on_validation_epoch_end()

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self._handle_epoch_boundary("test", True)

    def on_test_epoch_end(self) -> None:
        self._handle_epoch_boundary("test", False)
        super().on_test_epoch_end()

    def _reset_supervised_cache(self, stage: str) -> None:
        reset_supervised_cache(self, stage)

    def _cache_limit_for_stage(self, stage: str):
        return cache_limit_for_stage(self, stage)

    def _cache_supervised_batch(
        self,
        stage: str,
        z: torch.Tensor,
        meta: dict,
        encoder_features: torch.Tensor | None = None,
    ) -> None:
        cache_supervised_batch(self, stage, z, meta, encoder_features=encoder_features)

    def _log_supervised_metrics(self, stage: str) -> None:
        log_supervised_metrics(self, stage)

    def _log_metric(self, stage: str, name: str, value, *, on_step=None, on_epoch=None, **kwargs) -> None:
        if on_step is None:
            on_step = stage == "train"
        if on_epoch is None:
            on_epoch = stage != "train"

        log_kwargs = dict(kwargs)
        if "sync_dist" not in log_kwargs and stage != "train":
            log_kwargs["sync_dist"] = True

        self.log(
            f"{stage}/{name}",
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            **log_kwargs,
        )


__all__ = ["PointM2AEModule", "PointM2AEBackbone"]
