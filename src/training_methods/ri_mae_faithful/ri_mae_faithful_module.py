from __future__ import annotations

import copy
import math
import random
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
from src.training_methods.point_m2ae.point_m2ae_module import DropPath
from src.training_methods.pointgpt.pointgpt_module import EncoderSmall
from src.utils.pointcloud_ops import crop_to_num_points
from src.utils.spd_utils import cached_sample_count, get_optimizers_and_scheduler


def _to_bn3(points: torch.Tensor) -> torch.Tensor:
    if points.dim() != 3:
        raise ValueError(f"Expected point cloud with shape (B, N, 3) or (B, 3, N), got {tuple(points.shape)}")
    if points.shape[-1] == 3:
        return points
    if points.shape[1] == 3:
        return points.transpose(1, 2).contiguous()
    raise ValueError(f"Expected point cloud with 3 coordinates, got {tuple(points.shape)}")


def farthest_point_sample(
    xyz: torch.Tensor,
    npoint: int,
    *,
    deterministic: bool = False,
) -> torch.Tensor:
    xyz = _to_bn3(xyz)
    bsz, n_pts, _ = xyz.shape
    if npoint <= 0:
        raise ValueError(f"npoint must be > 0, got {npoint}")
    if npoint > n_pts:
        raise ValueError(f"npoint ({npoint}) cannot exceed number of points ({n_pts})")

    xyz_f = xyz.to(torch.float32)
    centroids = torch.zeros((bsz, npoint), dtype=torch.long, device=xyz.device)
    distance = torch.full((bsz, n_pts), 1e10, dtype=xyz_f.dtype, device=xyz.device)
    farthest = (
        torch.zeros((bsz,), dtype=torch.long, device=xyz.device)
        if deterministic
        else torch.randint(0, n_pts, (bsz,), device=xyz.device)
    )
    batch_idx = torch.arange(bsz, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz_f[batch_idx, farthest].view(bsz, 1, 3)
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
            f"Batch size mismatch between points {tuple(points.shape)} and idx {tuple(idx.shape)}."
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


class Group(nn.Module):
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

    def forward(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xyz = _to_bn3(xyz)
        bsz, n_pts, _ = xyz.shape
        if self.num_group > n_pts:
            raise ValueError(f"num_group ({self.num_group}) cannot exceed number of points ({n_pts})")
        if self.group_size > n_pts:
            raise ValueError(f"group_size ({self.group_size}) cannot exceed number of points ({n_pts})")
        fps_idx = farthest_point_sample(xyz, self.num_group, deterministic=self.deterministic_fps)
        center = index_points(xyz, fps_idx)
        group_idx = knn_point(self.group_size, xyz, center)
        neighborhood = index_points(xyz, group_idx)
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood.contiguous(), center.contiguous()


def content_orientation_disentanglement(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if data.dim() != 4 or data.shape[-1] != 3:
        raise ValueError(f"Expected patch tensor with shape (B, G, M, 3), got {tuple(data.shape)}")
    batch_size, num_group, _, _ = data.shape
    data_flat = data.reshape(batch_size * num_group, data.shape[2], 3)
    centered = data_flat.transpose(1, 2)
    centered = centered - centered.mean(dim=-1, keepdim=True)
    gram = centered @ centered.transpose(1, 2)
    basis, _, _ = torch.linalg.svd(gram, full_matrices=True)

    canonical = data_flat @ basis
    center = canonical.mean(dim=1, keepdim=True)
    sign = torch.sign(center)
    sign[sign == 0] = 1
    canonical = canonical * sign

    basis = basis * sign
    basis = basis.transpose(1, 2)
    return canonical.reshape(batch_size, num_group, data.shape[2], 3), basis.reshape(batch_size, num_group, 3, 3)


def _apply_scale_and_translate(
    points: torch.Tensor,
    *,
    scale_low: float,
    scale_high: float,
    translate_range: float,
) -> torch.Tensor:
    if scale_low <= 0 or scale_high <= 0 or scale_high < scale_low:
        raise ValueError(
            f"Invalid scale range: scale_low={scale_low}, scale_high={scale_high}."
        )
    batch_size = points.shape[0]
    scale = torch.empty((batch_size, 1, 3), device=points.device, dtype=points.dtype).uniform_(
        float(scale_low),
        float(scale_high),
    )
    translate = torch.empty((batch_size, 1, 3), device=points.device, dtype=points.dtype).uniform_(
        -float(translate_range),
        float(translate_range),
    )
    return points * scale + translate


def _mask_ratio_range(mask_ratio) -> tuple[float, float]:
    if isinstance(mask_ratio, Sequence) and not isinstance(mask_ratio, (str, bytes)):
        if len(mask_ratio) != 2:
            raise ValueError(f"mask_ratio sequence must have length 2, got {mask_ratio!r}")
        low = float(mask_ratio[0])
        high = float(mask_ratio[1])
    else:
        low = float(mask_ratio)
        high = float(mask_ratio)
    if not (0.0 <= low <= high < 1.0):
        raise ValueError(f"mask_ratio must satisfy 0 <= low <= high < 1, got {(low, high)}")
    return low, high


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None, drop: float = 0.0) -> None:
        super().__init__()
        hidden = int(hidden_features) if hidden_features is not None else int(in_features)
        out = int(out_features) if out_features is not None else int(in_features)
        self.fc1 = nn.Linear(int(in_features), hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out)
        self.drop = nn.Dropout(float(drop))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class RIAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.dim = int(dim)
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by num_heads ({self.num_heads})")
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=bool(qkv_bias))
        self.attn_drop = nn.Dropout(float(attn_drop))
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(float(proj_drop))

    def forward(self, x: torch.Tensor, rioe: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        if rioe.shape != (batch_size, seq_len, seq_len, channels):
            raise ValueError(
                f"Expected rioe shape {(batch_size, seq_len, seq_len, channels)}, got {tuple(rioe.shape)}"
            )
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        eq = rioe.reshape(batch_size, seq_len, seq_len, self.num_heads, self.head_dim)
        eq = eq.permute(0, 3, 1, 2, 4).contiguous()
        rioe_bias_q = (q.unsqueeze(-2) @ eq.transpose(-2, -1)).squeeze(-2)

        attn = (q @ k.transpose(-2, -1) + rioe_bias_q) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, channels)
        out = self.proj_drop(self.proj(out))
        return out


class RIBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(int(dim))
        self.attn = RIAttention(
            int(dim),
            num_heads=int(num_heads),
            qkv_bias=bool(qkv_bias),
            attn_drop=float(attn_drop),
            proj_drop=float(drop),
        )
        self.drop_path = DropPath(float(drop_path)) if float(drop_path) > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(int(dim))
        hidden = int(int(dim) * float(mlp_ratio))
        self.mlp = Mlp(int(dim), hidden_features=hidden, out_features=int(dim), drop=float(drop))

    def forward(self, x: torch.Tensor, rioe: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), rioe))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class RITransformerEncoder(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: list[float] | float = 0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                RIBlock(
                    embed_dim,
                    num_heads=int(num_heads),
                    mlp_ratio=float(mlp_ratio),
                    drop=float(drop),
                    attn_drop=float(attn_drop),
                    drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else float(drop_path_rate),
                )
                for i in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(int(embed_dim))

    def forward(self, x: torch.Tensor, pos: torch.Tensor, rioe: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x + pos, rioe)
        return self.norm(x)


class PatchEncoderTransformer(nn.Module):
    def __init__(
        self,
        *,
        encoder_channel: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        drop_path_rate: list[float] | float,
    ) -> None:
        super().__init__()
        self.patch_encoder = EncoderSmall(encoder_channel)
        self.reduce_dim = (
            nn.Identity()
            if int(encoder_channel) == int(embed_dim)
            else nn.Linear(int(encoder_channel), int(embed_dim))
        )
        self.transformer = RITransformerEncoder(
            embed_dim=int(embed_dim),
            depth=int(depth),
            num_heads=int(num_heads),
            mlp_ratio=float(mlp_ratio),
            drop_path_rate=drop_path_rate,
        )

    def forward(self, neighborhood: torch.Tensor, pos: torch.Tensor, rioe: torch.Tensor) -> torch.Tensor:
        tokens = self.reduce_dim(self.patch_encoder(neighborhood))
        return self.transformer(tokens, pos, rioe)


class TransformerPredictor(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        drop_path_rate: list[float] | float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                RIBlock(
                    int(embed_dim),
                    num_heads=int(num_heads),
                    mlp_ratio=float(mlp_ratio),
                    drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else float(drop_path_rate),
                )
                for i in range(int(depth))
            ]
        )

    def forward(self, x: torch.Tensor, rioe: torch.Tensor, return_token_num: int = 0) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, rioe)
        if return_token_num > 0:
            x = x[:, -return_token_num:]
        return x


class SimpleEMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float) -> None:
        super().__init__()
        if not (0.0 < float(decay) < 1.0):
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.model = copy.deepcopy(model)
        self.model.requires_grad_(False)
        self.decay = float(decay)
        self.num_updates = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.get_feature(x)

    @staticmethod
    def get_annealed_rate(start: float, end: float, curr_step: int, total_steps: int) -> float:
        if total_steps <= 0:
            return float(end)
        delta = float(end) - float(start)
        pct_remaining = 1.0 - float(curr_step) / float(total_steps)
        return float(end) - delta * pct_remaining

    @torch.no_grad()
    def step(self, new_model: nn.Module) -> None:
        ema_state = {}
        ema_params = self.model.state_dict()
        for key, param in new_model.state_dict().items():
            ema_param = ema_params[key].float()
            ema_param.mul_(self.decay)
            ema_param.add_(param.to(dtype=ema_param.dtype), alpha=1.0 - self.decay)
            ema_state[key] = ema_param
        self.model.load_state_dict(ema_state, strict=False)
        self.num_updates += 1


class RIMAEStudent(nn.Module):
    def __init__(
        self,
        *,
        num_group: int,
        group_size: int,
        encoder_dims: int,
        trans_dim: int,
        depth_encoder: int,
        depth_predictor: int,
        num_heads: int,
        mask_ratio,
        mask_rand: bool,
        mlp_ratio: float,
        drop_path_rate: float,
        deterministic_fps: bool,
    ) -> None:
        super().__init__()
        self.num_group = int(num_group)
        self.group_size = int(group_size)
        self.encoder_dims = int(encoder_dims)
        self.trans_dim = int(trans_dim)
        self.mask_ratio = _mask_ratio_range(mask_ratio)
        self.mask_rand = bool(mask_rand)

        dpr_encoder = torch.linspace(0.0, float(drop_path_rate), int(depth_encoder)).tolist()
        dpr_predictor = torch.linspace(0.0, float(drop_path_rate), int(depth_predictor)).tolist()

        self.group_divider = Group(
            num_group=self.num_group,
            group_size=self.group_size,
            deterministic_fps=bool(deterministic_fps),
        )
        self.encoder = PatchEncoderTransformer(
            encoder_channel=self.encoder_dims,
            embed_dim=self.trans_dim,
            depth=int(depth_encoder),
            num_heads=int(num_heads),
            mlp_ratio=float(mlp_ratio),
            drop_path_rate=dpr_encoder,
        )
        self.de_mask_token = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )
        self.rel_ori_embed = nn.Sequential(
            nn.Linear(9, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )
        self.predictor = TransformerPredictor(
            embed_dim=self.trans_dim,
            depth=int(depth_predictor),
            num_heads=int(num_heads),
            mlp_ratio=float(mlp_ratio),
            drop_path_rate=dpr_predictor,
        )
        self.norm = nn.LayerNorm(self.trans_dim)
        self.projector = nn.Sequential(
            nn.Linear(self.trans_dim, self.trans_dim),
            nn.LayerNorm(self.trans_dim),
        )
        nn.init.normal_(self.de_mask_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0.0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def _mask_center_block(self, center: torch.Tensor) -> torch.Tensor:
        batch_size, num_group, _ = center.shape
        low, high = self.mask_ratio
        if high <= 0.0:
            return torch.zeros((batch_size, num_group), dtype=torch.bool, device=center.device)

        masks = []
        for sample in center:
            start = random.randint(0, num_group - 1)
            distance = torch.norm(sample[start].view(1, 3) - sample, dim=-1)
            idx = torch.argsort(distance, dim=-1, descending=False)
            ratio = random.uniform(low, high)
            mask_num = int(ratio * num_group)
            mask_num = max(1, min(num_group - 1, mask_num))
            mask = torch.zeros(num_group, dtype=torch.bool, device=center.device)
            mask[idx[:mask_num]] = True
            masks.append(mask)
        return torch.stack(masks, dim=0)

    def _mask_center_all_rand(self, center: torch.Tensor) -> torch.Tensor:
        batch_size, num_group, _ = center.shape
        ratio = self.mask_ratio[0]
        num_masks = int(num_group * ratio)
        num_masks = max(1, min(num_group - 1, num_masks))
        masks = []
        for _ in range(batch_size):
            perm = torch.randperm(num_group, device=center.device)
            mask = torch.zeros(num_group, dtype=torch.bool, device=center.device)
            mask[perm[:num_masks]] = True
            masks.append(mask)
        return torch.stack(masks, dim=0)

    @staticmethod
    def _mixup_pc(neighborhood: torch.Tensor, center: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mixup_ratio = torch.rand(neighborhood.size(0), device=neighborhood.device)
        mixup_mask = torch.rand(neighborhood.shape[:2], device=neighborhood.device) < mixup_ratio.unsqueeze(-1)
        mixup_mask = mixup_mask.to(dtype=neighborhood.dtype)
        mixed_neighborhood = (
            neighborhood * mixup_mask.unsqueeze(-1).unsqueeze(-1)
            + neighborhood.flip(0) * (1.0 - mixup_mask.unsqueeze(-1).unsqueeze(-1))
        )
        mixed_center = center * mixup_mask.unsqueeze(-1) + center.flip(0) * (1.0 - mixup_mask.unsqueeze(-1))
        return mixup_ratio, mixed_neighborhood, mixed_center

    def get_rioe(self, ori: torch.Tensor) -> torch.Tensor:
        batch_size, num_group, _, _ = ori.shape
        ori_1 = ori.unsqueeze(1).repeat(1, num_group, 1, 1, 1)
        ori_2 = ori.unsqueeze(2).repeat(1, 1, num_group, 1, 1)
        rel_ori = ori_2 @ ori_1.transpose(-2, -1)
        rel_ori = rel_ori.transpose(2, 1).reshape(batch_size, num_group * num_group, 9)
        return self.rel_ori_embed(rel_ori).reshape(batch_size, num_group, num_group, -1)

    def get_feature(self, pts: torch.Tensor) -> torch.Tensor:
        neighborhood, center = self.group_divider(pts)
        neighborhood, ori = content_orientation_disentanglement(neighborhood)
        batch_size, num_group, _, _ = neighborhood.shape

        center_t = center.reshape(batch_size * num_group, 1, 3)
        ori_t = ori.reshape(batch_size * num_group, 3, 3)
        center_ri = (center_t @ ori_t.transpose(-2, -1)).reshape(batch_size, num_group, 3)

        pos = self.pos_embed(center_ri)
        ori_embed = self.get_rioe(ori)
        x = self.encoder(neighborhood, pos, ori_embed)
        return self.norm(x)

    def forward_eval(self, pts: torch.Tensor) -> torch.Tensor:
        tokens = self.get_feature(pts)
        max_features = tokens.max(dim=1).values
        mean_features = tokens.mean(dim=1)
        return torch.cat([max_features, mean_features], dim=-1)

    def forward(self, pts: torch.Tensor, *, cutmix: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        neighborhood, center = self.group_divider(pts)
        neighborhood, ori_mat = content_orientation_disentanglement(neighborhood)
        batch_size, num_group, group_size, channels = neighborhood.shape

        center_t = center.reshape(batch_size * num_group, 1, 3)
        ori_t = ori_mat.reshape(batch_size * num_group, 3, 3)
        center_ri = (center_t @ ori_t.transpose(-2, -1)).reshape(batch_size, num_group, 3)

        if cutmix:
            _, neighborhood, center_ri = self._mixup_pc(neighborhood, center_ri)

        bool_masked_pos = (
            self._mask_center_all_rand(center_ri)
            if self.mask_rand
            else self._mask_center_block(center_ri)
        )
        bool_masked_pos = bool_masked_pos.to(center_ri.device, non_blocking=True)

        center_masked = center_ri[bool_masked_pos].reshape(batch_size, -1, 3)
        neighborhood_masked = neighborhood[bool_masked_pos].reshape(batch_size, -1, group_size, channels)
        ori_masked = ori_mat[bool_masked_pos].reshape(batch_size, -1, 3, 3)
        center_vis = center_ri[~bool_masked_pos].reshape(batch_size, -1, 3)
        neighborhood_vis = neighborhood[~bool_masked_pos].reshape(batch_size, -1, group_size, channels)
        ori_vis = ori_mat[~bool_masked_pos].reshape(batch_size, -1, 3, 3)

        pos_vis = self.pos_embed(center_vis)
        pos_masked = self.pos_embed(center_masked)
        ori_vis_embed = self.get_rioe(ori_vis)
        x = self.encoder(neighborhood_vis, pos_vis, ori_vis_embed)
        x = self.norm(x)

        feat_all = torch.cat([x, self.de_mask_token + pos_masked], dim=1)
        ori_all = torch.cat([ori_vis, ori_masked], dim=1)
        ori_all_embed = self.get_rioe(ori_all)
        output = self.predictor(feat_all, ori_all_embed, center_masked.shape[1])
        return self.projector(output), bool_masked_pos, center


class RIMAEFaithfulBackbone(nn.Module):
    def __init__(
        self,
        *,
        num_group: int,
        group_size: int,
        encoder_dims: int,
        trans_dim: int,
        depth_encoder: int,
        depth_predictor: int,
        num_heads: int,
        mask_ratio,
        mask_rand: bool,
        mlp_ratio: float,
        drop_path_rate: float,
        ema_decay: float,
        ema_end_decay: float,
        ema_anneal_end_step: int,
        deterministic_fps: bool,
    ) -> None:
        super().__init__()
        self.student = RIMAEStudent(
            num_group=int(num_group),
            group_size=int(group_size),
            encoder_dims=int(encoder_dims),
            trans_dim=int(trans_dim),
            depth_encoder=int(depth_encoder),
            depth_predictor=int(depth_predictor),
            num_heads=int(num_heads),
            mask_ratio=mask_ratio,
            mask_rand=bool(mask_rand),
            mlp_ratio=float(mlp_ratio),
            drop_path_rate=float(drop_path_rate),
            deterministic_fps=bool(deterministic_fps),
        )
        self.teacher = SimpleEMA(self.student, float(ema_decay))
        self.ema_decay = float(ema_decay)
        self.ema_end_decay = float(ema_end_decay)
        self.ema_anneal_end_step = int(ema_anneal_end_step)

    @torch.no_grad()
    def ema_step(self) -> None:
        if self.ema_decay != self.ema_end_decay:
            if self.teacher.num_updates >= self.ema_anneal_end_step:
                decay = self.ema_end_decay
            else:
                decay = self.teacher.get_annealed_rate(
                    self.ema_decay,
                    self.ema_end_decay,
                    self.teacher.num_updates,
                    self.ema_anneal_end_step,
                )
            self.teacher.decay = float(decay)
        if self.teacher.decay < 1.0:
            self.teacher.step(self.student)

    def forward(self, pts: torch.Tensor, *, cutmix: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred, mask, centers = self.student(pts, cutmix=cutmix)
        with torch.no_grad():
            self.teacher.eval()
            target = self.teacher(pts)

        batch_size, _, trans_dim = pred.shape
        target = target[mask].reshape(batch_size, -1, trans_dim)
        if target.shape != pred.shape:
            raise RuntimeError(
                f"Teacher/pred shape mismatch: target={tuple(target.shape)}, pred={tuple(pred.shape)}"
            )

        loss = F.mse_loss(target.to(torch.float32), pred.to(torch.float32))
        global_latent = self.student.forward_eval(pts)
        mask_fraction = mask.to(dtype=loss.dtype).mean()
        return loss, global_latent, mask_fraction, centers

    @torch.no_grad()
    def forward_eval(self, pts: torch.Tensor) -> torch.Tensor:
        return self.student.forward_eval(pts)


class RIMAEFaithfulModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        ri_cfg = getattr(cfg, "ri_mae_faithful", None)

        def _ri_get(name: str, default):
            if ri_cfg is not None and hasattr(ri_cfg, name):
                return getattr(ri_cfg, name)
            return getattr(cfg, f"ri_mae_faithful_{name}", default)

        self.model = RIMAEFaithfulBackbone(
            num_group=int(_ri_get("num_group", 24)),
            group_size=int(_ri_get("group_size", 12)),
            encoder_dims=int(_ri_get("encoder_dims", 256)),
            trans_dim=int(_ri_get("trans_dim", 384)),
            depth_encoder=int(_ri_get("depth_encoder", 12)),
            depth_predictor=int(_ri_get("depth_predictor", 1)),
            num_heads=int(_ri_get("num_heads", 6)),
            mask_ratio=_ri_get("mask_ratio", [0.75, 0.75]),
            mask_rand=bool(_ri_get("mask_rand", True)),
            mlp_ratio=float(_ri_get("mlp_ratio", 4.0)),
            drop_path_rate=float(_ri_get("drop_path_rate", 0.1)),
            ema_decay=float(_ri_get("ema_decay", 0.99)),
            ema_end_decay=float(_ri_get("ema_end_decay", 0.9996)),
            ema_anneal_end_step=int(_ri_get("ema_anneal_end_step", 10000)),
            deterministic_fps=bool(_ri_get("deterministic_fps", False)),
        )

        legacy_crop_mode = str(_ri_get("crop_mode", "random")).strip()
        self.crop_mode_train = str(_ri_get("crop_mode_train", legacy_crop_mode)).strip()
        self.crop_mode_eval = str(_ri_get("crop_mode_eval", "center")).strip()
        if not self.crop_mode_train:
            raise ValueError("ri_mae_faithful crop_mode_train must be a non-empty string.")
        if not self.crop_mode_eval:
            raise ValueError("ri_mae_faithful crop_mode_eval must be a non-empty string.")
        self.center_input = bool(_ri_get("center_input", False))
        self.use_cutmix = bool(_ri_get("use_cutmix", False))
        self.scale_translate = bool(_ri_get("scale_translate", True))
        self.scale_low = float(_ri_get("scale_low", 2.0 / 3.0))
        self.scale_high = float(_ri_get("scale_high", 3.0 / 2.0))
        self.translate_range = float(_ri_get("translate_range", 0.2))

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
            points = crop_to_num_points(points, self.model_points, mode=self._resolve_crop_mode(stage))
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
            raise RuntimeError(f"RI-MAE faithful forward_eval returned non-tensor features: {type(features)}")
        if features.dim() != 2:
            raise RuntimeError(f"RI-MAE faithful forward_eval must return a 2D tensor, got {tuple(features.shape)}")
        return features, features, None

    def _step(self, batch, stage: str) -> torch.Tensor:
        points_raw, meta = self._unpack_batch(batch)
        points = self._prepare_points(points_raw, stage=stage)
        if stage == "train" and self.scale_translate:
            points = _apply_scale_and_translate(
                points,
                scale_low=self.scale_low,
                scale_high=self.scale_high,
                translate_range=self.translate_range,
            )

        loss, global_latent, mask_fraction, patch_centers = self.model(
            points,
            cutmix=(stage == "train" and self.use_cutmix),
        )

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
            patch_spread = patch_centers.to(torch.float32).std(dim=1).mean()

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

    def on_after_backward(self) -> None:
        super().on_after_backward()
        if self.training:
            self.model.ema_step()

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
        self.log(f"{stage}/{name}", value, on_step=on_step, on_epoch=on_epoch, **log_kwargs)


__all__ = ["RIMAEFaithfulModule"]
