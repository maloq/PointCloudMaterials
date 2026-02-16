from __future__ import annotations

import math

import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.loss.reconstruction_loss import chamfer_distance
from src.training_methods.contrastive_learning.supervised_cache import (
    cache_limit_for_stage,
    cache_supervised_batch,
    init_supervised_cache,
    log_supervised_metrics,
    reset_supervised_cache,
)
from src.utils.pointcloud_ops import crop_to_num_points
from src.utils.spd_utils import get_optimizers_and_scheduler


def _to_bn3(points: torch.Tensor) -> torch.Tensor:
    if points.dim() != 3:
        raise ValueError(f"Expected point cloud with shape (B, N, 3) or (B, 3, N), got {tuple(points.shape)}")
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


def farthest_point_sample(
    xyz: torch.Tensor,
    npoint: int,
    *,
    deterministic: bool = False,
) -> torch.Tensor:
    """FPS on xyz (B, N, 3) -> idx (B, npoint)."""
    xyz = _to_bn3(xyz)
    bsz, n_pts, _ = xyz.shape
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
    """Gather points (B, N, D) by idx (B, ...) -> (B, ..., D)."""
    points = _to_bn3(points)
    bsz = points.shape[0]
    idx_flat = idx.reshape(bsz, -1)
    gathered = points.gather(1, idx_flat.unsqueeze(-1).expand(-1, -1, points.shape[-1]))
    return gathered.view(bsz, *idx.shape[1:], points.shape[-1])


def knn_point(k: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """kNN from new_xyz to xyz using Euclidean distance; returns idx (B, M, k)."""
    xyz = _to_bn3(xyz)
    new_xyz = _to_bn3(new_xyz)
    n_pts = xyz.shape[1]
    k_eff = min(int(k), int(n_pts))
    if k_eff <= 0:
        raise ValueError(f"k must be positive, got {k}")

    dist = torch.cdist(new_xyz.to(torch.float32), xyz.to(torch.float32))
    return dist.topk(k=k_eff, dim=-1, largest=False).indices


class Group(nn.Module):
    """Split cloud into FPS-centered local groups and order them autoregressively."""

    def __init__(
        self,
        num_group: int,
        group_size: int,
        *,
        deterministic_fps: bool = False,
        sorting_mode: str = "nearest",
    ) -> None:
        super().__init__()
        self.num_group = int(num_group)
        self.group_size = int(group_size)
        self.deterministic_fps = bool(deterministic_fps)
        self.sorting_mode = str(sorting_mode).lower()

        if self.num_group <= 0:
            raise ValueError(f"num_group must be > 0, got {self.num_group}")
        if self.group_size <= 0:
            raise ValueError(f"group_size must be > 0, got {self.group_size}")
        if self.sorting_mode not in {"nearest", "none"}:
            raise ValueError(
                f"Unsupported sorting_mode={sorting_mode!r}. Expected one of ['nearest', 'none']."
            )

    @staticmethod
    def _nearest_path_order(center: torch.Tensor) -> torch.Tensor:
        """Greedy nearest-neighbor path over patch centers (batch-wise)."""
        bsz, num_group, _ = center.shape
        device = center.device

        dist = torch.cdist(center.to(torch.float32), center.to(torch.float32))
        inf = torch.tensor(float("inf"), device=device, dtype=dist.dtype)
        diag = torch.arange(num_group, device=device)
        dist[:, diag, diag] = inf

        order = torch.zeros((bsz, num_group), dtype=torch.long, device=device)
        visited = torch.zeros((bsz, num_group), dtype=torch.bool, device=device)
        visited[:, 0] = True

        batch_idx = torch.arange(bsz, device=device)
        for i in range(1, num_group):
            last = order[:, i - 1]
            d = dist[batch_idx, last]
            d = d.masked_fill(visited, inf)
            nxt = d.argmin(dim=-1)
            order[:, i] = nxt
            visited[batch_idx, nxt] = True

        return order

    def forward(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

        if self.sorting_mode == "nearest":
            order = self._nearest_path_order(center)
            neighborhood = neighborhood.gather(
                1,
                order.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.group_size, 3),
            )
            center = center.gather(1, order.unsqueeze(-1).expand(-1, -1, 3))

        return neighborhood.contiguous(), center.contiguous()


class EncoderLarge(nn.Module):
    """Patch encoder used by PointGPT-B/L variants."""

    def __init__(self, encoder_channel: int) -> None:
        super().__init__()
        self.encoder_channel = int(encoder_channel)

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 1024, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(2048, 2048, 1),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Conv1d(2048, self.encoder_channel, 1),
        )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        # point_groups: (B, G, S, 3)
        bsz, num_group, group_size, _ = point_groups.shape
        x = point_groups.reshape(bsz * num_group, group_size, 3)

        feat = self.first_conv(x.transpose(2, 1))
        global_feat = feat.max(dim=2, keepdim=True).values
        feat = torch.cat([global_feat.expand(-1, -1, group_size), feat], dim=1)
        feat = self.second_conv(feat)
        global_feat = feat.max(dim=2).values

        return global_feat.reshape(bsz, num_group, self.encoder_channel)


class EncoderSmall(nn.Module):
    """Patch encoder used by PointGPT-S variant."""

    def __init__(self, encoder_channel: int) -> None:
        super().__init__()
        self.encoder_channel = int(encoder_channel)

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        # point_groups: (B, G, S, 3)
        bsz, num_group, group_size, _ = point_groups.shape
        x = point_groups.reshape(bsz * num_group, group_size, 3)

        feat = self.first_conv(x.transpose(2, 1))
        global_feat = feat.max(dim=2, keepdim=True).values
        feat = torch.cat([global_feat.expand(-1, -1, group_size), feat], dim=1)
        feat = self.second_conv(feat)
        global_feat = feat.max(dim=2).values

        return global_feat.reshape(bsz, num_group, self.encoder_channel)


class PositionEmbeddingCoordsSine(nn.Module):
    """Continuous sinusoidal positional embedding for 3D coordinates."""

    def __init__(
        self,
        n_dim: int = 3,
        d_model: int = 384,
        temperature: float = 10000.0,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_dim = int(n_dim)
        self.num_pos_feats = (int(d_model) // self.n_dim // 2) * 2
        self.temperature = float(temperature)
        self.padding = int(d_model) - self.num_pos_feats * self.n_dim
        self.scale = float(scale) * 2.0 * math.pi

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.shape[-1] != self.n_dim:
            raise ValueError(
                f"Expected coordinates with last dim {self.n_dim}, got {tuple(xyz.shape)}"
            )

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2.0 * torch.div(dim_t, 2, rounding_mode="trunc") / self.num_pos_feats)

        xyz_scaled = xyz.to(torch.float32) * self.scale
        pos_div = xyz_scaled.unsqueeze(-1) / dim_t
        pos_sin = pos_div[..., 0::2].sin()
        pos_cos = pos_div[..., 1::2].cos()
        pos = torch.stack([pos_sin, pos_cos], dim=-1).reshape(*xyz.shape[:-1], -1)

        if self.padding > 0:
            pos = nn.functional.pad(pos, (0, self.padding))

        return pos.to(dtype=xyz.dtype)


class GPTBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        y = self.ln_1(x)
        attn_out, _ = self.attn(y, y, y, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTExtractor(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.sos = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.sos, std=0.02)

        self.layers = nn.ModuleList(
            [GPTBlock(self.embed_dim, int(num_heads)) for _ in range(int(num_layers))]
        )
        self.ln_f = nn.LayerNorm(self.embed_dim)

    def forward(self, h: torch.Tensor, pos: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # h/pos: (B, L, C), attn_mask: (L, L)
        bsz, _, _ = h.shape
        h = h.transpose(0, 1)
        pos = pos.transpose(0, 1)

        sos = self.sos.expand(-1, bsz, -1)
        h = torch.cat([sos, h[:-1, :, :]], dim=0)

        for layer in self.layers:
            h = layer(h + pos, attn_mask)

        h = self.ln_f(h)
        return h.transpose(0, 1)


class GPTGenerator(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, group_size: int) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.group_size = int(group_size)

        self.layers = nn.ModuleList(
            [GPTBlock(self.embed_dim, int(num_heads)) for _ in range(int(num_layers))]
        )
        self.ln_f = nn.LayerNorm(self.embed_dim)
        self.increase_dim = nn.Conv1d(self.embed_dim, 3 * self.group_size, kernel_size=1)

    def forward(self, h: torch.Tensor, pos: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # h/pos: (B, L, C), attn_mask: (L, L)
        bsz, length, _ = h.shape

        h = h.transpose(0, 1)
        pos = pos.transpose(0, 1)

        for layer in self.layers:
            h = layer(h + pos, attn_mask)

        h = self.ln_f(h)
        points = self.increase_dim(h.transpose(0, 1).transpose(1, 2))
        points = points.transpose(1, 2).reshape(bsz * length, self.group_size, 3)
        return points


class PointGPTBackbone(nn.Module):
    """PyTorch-only PointGPT pretraining backbone."""

    def __init__(
        self,
        *,
        num_group: int,
        group_size: int,
        mask_ratio: float,
        keep_attend: int,
        encoder_dims: int,
        trans_dim: int,
        depth: int,
        decoder_depth: int,
        num_heads: int,
        deterministic_fps: bool,
        sorting_mode: str,
    ) -> None:
        super().__init__()

        self.num_group = int(num_group)
        self.group_size = int(group_size)
        self.mask_ratio = float(mask_ratio)
        self.keep_attend = int(keep_attend)
        self.encoder_dims = int(encoder_dims)
        self.trans_dim = int(trans_dim)

        if self.trans_dim <= 0:
            raise ValueError(f"trans_dim must be > 0, got {self.trans_dim}")
        if self.encoder_dims <= 0:
            raise ValueError(f"encoder_dims must be > 0, got {self.encoder_dims}")

        self.group_divider = Group(
            num_group=self.num_group,
            group_size=self.group_size,
            deterministic_fps=bool(deterministic_fps),
            sorting_mode=sorting_mode,
        )

        if self.encoder_dims == 384:
            self.encoder = EncoderSmall(self.encoder_dims)
        else:
            self.encoder = EncoderLarge(self.encoder_dims)

        self.input_proj = (
            nn.Identity()
            if self.encoder_dims == self.trans_dim
            else nn.Linear(self.encoder_dims, self.trans_dim)
        )

        self.pos_embed = PositionEmbeddingCoordsSine(n_dim=3, d_model=self.trans_dim, scale=1.0)
        self.sos_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        self.extractor = GPTExtractor(
            embed_dim=self.trans_dim,
            num_heads=int(num_heads),
            num_layers=int(depth),
        )
        self.generator = GPTGenerator(
            embed_dim=self.trans_dim,
            num_heads=int(num_heads),
            num_layers=int(decoder_depth),
            group_size=self.group_size,
        )

    def _build_attn_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        causal = torch.triu(
            torch.ones((seq_len, seq_len), device=device, dtype=torch.bool),
            diagonal=1,
        )

        keep = min(max(self.keep_attend, 0), seq_len)
        tail = max(seq_len - keep, 0)
        ratio = min(max(self.mask_ratio, 0.0), 1.0)
        num_mask = int(tail * ratio)

        if num_mask <= 0:
            return causal

        selected = torch.randperm(tail, device=device)[:num_mask] + keep
        overall = torch.zeros((seq_len,), dtype=torch.bool, device=device)
        overall[selected] = True

        column_mask = overall.unsqueeze(0).expand(seq_len, -1)
        eye = torch.eye(seq_len, device=device, dtype=torch.bool)
        return causal | (column_mask & ~eye)

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # points: (B, N, 3)
        neighborhood, center = self.group_divider(points)
        tokens = self.encoder(neighborhood)
        tokens = self.input_proj(tokens)

        bsz, seq_len, _ = tokens.shape

        if seq_len > 1:
            relative = center[:, 1:, :] - center[:, :-1, :]
            relative_norm = relative.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            relative_direction = relative / relative_norm
            position = torch.cat([center[:, :1, :], relative_direction], dim=1)
        else:
            position = center[:, :1, :]

        pos_relative = self.pos_embed(position)

        if seq_len > 1:
            pos_absolute = self.pos_embed(center[:, :-1, :])
        else:
            pos_absolute = center.new_zeros((bsz, 0, self.trans_dim))
        sos_pos = self.sos_pos.expand(bsz, -1, -1)
        pos_absolute = torch.cat([sos_pos, pos_absolute], dim=1)

        attn_mask = self._build_attn_mask(seq_len, points.device)

        encoded = self.extractor(tokens, pos_absolute, attn_mask)
        pred = self.generator(encoded, pos_relative, attn_mask)
        target = neighborhood.reshape(bsz * seq_len, self.group_size, 3)

        global_latent = encoded.mean(dim=1)
        return pred, target, global_latent


class PointGPTModule(pl.LightningModule):
    """Lightning wrapper for PointGPT-style autoregressive point reconstruction."""

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        pg = getattr(cfg, "pointgpt", None)

        def _pg_get(name: str, default):
            if pg is not None and hasattr(pg, name):
                return getattr(pg, name)
            return getattr(cfg, f"pointgpt_{name}", default)

        self.model = PointGPTBackbone(
            num_group=int(_pg_get("num_group", 64)),
            group_size=int(_pg_get("group_size", 32)),
            mask_ratio=float(_pg_get("mask_ratio", 0.7)),
            keep_attend=int(_pg_get("keep_attend", 10)),
            encoder_dims=int(_pg_get("encoder_dims", 384)),
            trans_dim=int(_pg_get("trans_dim", 384)),
            depth=int(_pg_get("depth", 12)),
            decoder_depth=int(_pg_get("decoder_depth", 4)),
            num_heads=int(_pg_get("num_heads", 6)),
            deterministic_fps=bool(_pg_get("deterministic_fps", False)),
            sorting_mode=str(_pg_get("sorting_mode", "nearest")),
        )

        self.crop_mode = str(_pg_get("crop_mode", "random"))
        self.chamfer_l1_weight = float(_pg_get("chamfer_l1_weight", 1.0))
        self.chamfer_l2_weight = float(_pg_get("chamfer_l2_weight", 1.0))

        if self.chamfer_l1_weight < 0 or self.chamfer_l2_weight < 0:
            raise ValueError("chamfer_l1_weight and chamfer_l2_weight must be non-negative")
        if self.chamfer_l1_weight == 0 and self.chamfer_l2_weight == 0:
            raise ValueError("At least one of chamfer_l1_weight or chamfer_l2_weight must be > 0")

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

    def _prepare_points(self, batch_points: torch.Tensor) -> torch.Tensor:
        points = _to_bn3(batch_points)
        points = points.to(device=self.device, dtype=self.dtype, non_blocking=True)

        if self.model_points is not None:
            points = crop_to_num_points(points, self.model_points, mode=self.crop_mode)

        points = points - points.mean(dim=1, keepdim=True)
        return points

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(points)

    def _step(self, batch, stage: str) -> torch.Tensor:
        points_raw, meta = self._unpack_batch(batch)
        points = self._prepare_points(points_raw)
        pred, target, latent = self(points)

        should_cache_stage = stage in self._supervised_cache and (
            stage != "train" or self.cache_train_supervised_metrics
        )
        if should_cache_stage and latent is not None:
            self._cache_supervised_batch(stage, latent, meta, encoder_features=latent)

        pred_f32 = pred.to(torch.float32)
        target_f32 = target.to(torch.float32)

        loss_l1, _ = chamfer_distance(pred_f32, target_f32, point_reduction="mean")
        loss_l2 = _chamfer_l2_squared(pred_f32, target_f32)
        total_loss = self.chamfer_l1_weight * loss_l1 + self.chamfer_l2_weight * loss_l2

        with torch.no_grad():
            latent_std = latent.to(torch.float32).std(dim=0).mean()

        self._log_metric(stage, "loss", total_loss, prog_bar=True, batch_size=points.shape[0])
        self._log_metric(stage, "chamfer_l1", loss_l1, batch_size=points.shape[0])
        self._log_metric(stage, "chamfer_l2", loss_l2, batch_size=points.shape[0])
        self._log_metric(stage, "latent_std", latent_std, batch_size=points.shape[0])

        return total_loss

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


__all__ = ["PointGPTModule"]
