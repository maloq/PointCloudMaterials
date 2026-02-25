from __future__ import annotations

import copy
import math

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
from src.training_methods.pointgpt.pointgpt_module import (
    EncoderLarge,
    EncoderSmall,
    Group,
    PositionEmbeddingCoordsSine,
)
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


def _gather_tokens(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def _gather_attn_bias(attn_bias: torch.Tensor, q_idx: torch.Tensor, k_idx: torch.Tensor | None = None) -> torch.Tensor:
    if k_idx is None:
        k_idx = q_idx

    batch_size, n_heads, _, _ = attn_bias.shape
    q_len = q_idx.shape[1]
    k_len = k_idx.shape[1]

    q_index = q_idx.unsqueeze(1).unsqueeze(-1).expand(-1, n_heads, -1, attn_bias.shape[-1])
    selected = attn_bias.gather(2, q_index)

    k_index = k_idx.unsqueeze(1).unsqueeze(2).expand(-1, n_heads, q_len, -1)
    return selected.gather(3, k_index).reshape(batch_size, n_heads, q_len, k_len)


class RIAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})")

        self.head_dim = self.embed_dim // self.num_heads
        self.scale = 1.0 / math.sqrt(float(self.head_dim))

        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.attn_drop = nn.Dropout(float(dropout))
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_drop = nn.Dropout(float(dropout))

        hidden_dim = int(float(mlp_ratio) * self.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.embed_dim),
            nn.Dropout(float(dropout)),
        )

    def _attention(self, x: torch.Tensor, attn_bias: torch.Tensor | None) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            attn = attn + attn_bias.to(dtype=attn.dtype)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, channels)
        out = self.proj_drop(self.proj(out))
        return out

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self._attention(self.norm1(x), attn_bias)
        x = x + self.mlp(self.norm2(x))
        return x


class RITransformer(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        depth: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                RIAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(int(embed_dim))

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.layers:
            x = block(x, attn_bias)
        return self.norm(x)


class RIMAEBackbone(nn.Module):
    def __init__(
        self,
        *,
        num_group: int,
        group_size: int,
        encoder_dims: int,
        trans_dim: int,
        depth: int,
        predictor_depth: int,
        num_heads: int,
        mask_ratio: float,
        ema_decay: float,
        mlp_ratio: float,
        dropout: float,
        deterministic_fps: bool,
        sorting_mode: str,
    ) -> None:
        super().__init__()
        self.num_group = int(num_group)
        self.group_size = int(group_size)
        self.encoder_dims = int(encoder_dims)
        self.trans_dim = int(trans_dim)
        self.num_heads = int(num_heads)
        self.mask_ratio = float(mask_ratio)
        self.ema_decay = float(ema_decay)

        if not (0.0 <= self.mask_ratio < 1.0):
            raise ValueError(f"mask_ratio must be in [0, 1), got {self.mask_ratio}")
        if not (0.0 < self.ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in (0, 1), got {self.ema_decay}")

        self.group_divider = Group(
            num_group=self.num_group,
            group_size=self.group_size,
            deterministic_fps=bool(deterministic_fps),
            sorting_mode=sorting_mode,
        )

        if self.encoder_dims == 384:
            self.patch_encoder = EncoderSmall(self.encoder_dims)
        else:
            self.patch_encoder = EncoderLarge(self.encoder_dims)

        self.input_proj = (
            nn.Identity()
            if self.encoder_dims == self.trans_dim
            else nn.Linear(self.encoder_dims, self.trans_dim)
        )
        self.pos_embed = PositionEmbeddingCoordsSine(n_dim=3, d_model=self.trans_dim, scale=1.0)

        orient_hidden = max(64, self.trans_dim // 2)
        self.orientation_mlp = nn.Sequential(
            nn.Linear(9, orient_hidden),
            nn.GELU(),
            nn.Linear(orient_hidden, self.num_heads),
        )
        self.orientation_scale = nn.Parameter(torch.tensor(1.0))

        self.student_encoder = RITransformer(
            embed_dim=self.trans_dim,
            num_heads=self.num_heads,
            depth=int(depth),
            mlp_ratio=float(mlp_ratio),
            dropout=float(dropout),
        )
        self.teacher_patch_encoder = copy.deepcopy(self.patch_encoder)
        self.teacher_input_proj = copy.deepcopy(self.input_proj)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        for module in (self.teacher_patch_encoder, self.teacher_input_proj, self.teacher_encoder):
            for param in module.parameters():
                param.requires_grad_(False)
            module.eval()

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.predictor = RITransformer(
            embed_dim=self.trans_dim,
            num_heads=self.num_heads,
            depth=int(predictor_depth),
            mlp_ratio=float(mlp_ratio),
            dropout=float(dropout),
        )

    @staticmethod
    def _estimate_patch_frames(neighborhood: torch.Tensor) -> torch.Tensor:
        # neighborhood: (B, G, S, 3), already centered around patch center.
        batch_size, num_group, group_size, _ = neighborhood.shape
        patches = neighborhood.reshape(batch_size * num_group, group_size, 3).to(torch.float32)

        cov = torch.matmul(patches.transpose(1, 2), patches) / float(max(group_size, 1))
        eigvals, eigvecs = torch.linalg.eigh(cov)

        order = torch.argsort(eigvals, dim=-1, descending=True)
        order_exp = order.unsqueeze(1).expand(-1, 3, -1)
        basis = torch.gather(eigvecs, 2, order_exp)

        coords = torch.matmul(patches, basis)
        signs = torch.ones((basis.shape[0], 3), dtype=basis.dtype, device=basis.device)
        batch_idx = torch.arange(basis.shape[0], device=basis.device)
        for axis in range(3):
            vals = coords[:, :, axis]
            max_idx = vals.abs().argmax(dim=1)
            picked = vals[batch_idx, max_idx]
            signs[:, axis] = torch.where(
                picked >= 0,
                torch.ones_like(picked),
                -torch.ones_like(picked),
            )

        basis = basis * signs.unsqueeze(1)
        det = torch.det(basis)
        handedness = torch.where(
            det < 0,
            -torch.ones_like(det),
            torch.ones_like(det),
        )
        basis[:, :, 2] = basis[:, :, 2] * handedness.unsqueeze(-1)

        return basis.reshape(batch_size, num_group, 3, 3)

    def _build_orientation_bias(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B, G, 3, 3), interpreted as global->canonical transforms.
        rel = torch.matmul(frames.unsqueeze(2).transpose(-1, -2), frames.unsqueeze(1))
        rel_flat = rel.reshape(rel.shape[0], rel.shape[1], rel.shape[2], 9)
        bias = self.orientation_mlp(rel_flat).permute(0, 3, 1, 2).contiguous()
        bias = bias - bias.mean(dim=-1, keepdim=True)
        return bias * self.orientation_scale.to(dtype=bias.dtype)

    @staticmethod
    def _mask_indices(batch_size: int, seq_len: int, mask_ratio: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len <= 1 or mask_ratio <= 0.0:
            all_idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            empty = all_idx[:, :0]
            return all_idx, empty

        num_mask = int(round(mask_ratio * seq_len))
        num_mask = max(1, min(seq_len - 1, num_mask))
        num_keep = seq_len - num_mask

        noise = torch.rand(batch_size, seq_len, device=device)
        ids_shuffle = noise.argsort(dim=1)
        ids_keep = ids_shuffle[:, :num_keep]
        ids_mask = ids_shuffle[:, num_keep:]
        return ids_keep, ids_mask

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # points: (B, N, 3)
        neighborhood, center = self.group_divider(points)
        with torch.no_grad():
            frames = self._estimate_patch_frames(neighborhood)

        frames = frames.to(dtype=points.dtype, device=points.device)
        canonical = torch.einsum("bgsc,bgcd->bgsd", neighborhood, frames)

        patch_tokens = self.patch_encoder(canonical)
        patch_tokens = self.input_proj(patch_tokens)

        ri_pos = torch.einsum("bgc,bgcd->bgd", center, frames)
        pos_tokens = self.pos_embed(ri_pos)
        encoder_input = patch_tokens + pos_tokens

        attn_bias_full = self._build_orientation_bias(frames).to(dtype=encoder_input.dtype)
        ids_keep, ids_mask = self._mask_indices(
            batch_size=encoder_input.shape[0],
            seq_len=encoder_input.shape[1],
            mask_ratio=self.mask_ratio,
            device=encoder_input.device,
        )

        visible_input = _gather_tokens(encoder_input, ids_keep)
        visible_bias = _gather_attn_bias(attn_bias_full, ids_keep)
        visible_latent = self.student_encoder(visible_input, visible_bias)

        pred_input = self.mask_token.expand(encoder_input.shape[0], encoder_input.shape[1], -1).clone()
        pred_input = pred_input.scatter(
            1,
            ids_keep.unsqueeze(-1).expand(-1, -1, pred_input.shape[-1]),
            visible_latent,
        )
        pred_input = pred_input + pos_tokens
        pred_all = self.predictor(pred_input, attn_bias_full)

        with torch.no_grad():
            teacher_tokens = self.teacher_patch_encoder(canonical.detach())
            teacher_tokens = self.teacher_input_proj(teacher_tokens)
            teacher_input = teacher_tokens + pos_tokens.detach()
            target_all = self.teacher_encoder(teacher_input, attn_bias_full.detach())

        if ids_mask.shape[1] == 0:
            loss = F.mse_loss(pred_all.to(torch.float32), target_all.to(torch.float32))
        else:
            pred_masked = _gather_tokens(pred_all, ids_mask)
            target_masked = _gather_tokens(target_all, ids_mask)
            loss = F.mse_loss(pred_masked.to(torch.float32), target_masked.to(torch.float32))

        global_latent = target_all.mean(dim=1)
        mask_fraction = torch.tensor(
            float(ids_mask.shape[1]) / float(max(1, encoder_input.shape[1])),
            dtype=loss.dtype,
            device=loss.device,
        )
        return loss, global_latent, mask_fraction, center

    @torch.no_grad()
    def forward_eval(self, points: torch.Tensor, *, use_teacher: bool = True) -> torch.Tensor:
        """Extract deterministic no-mask RI features for downstream linear probing."""
        neighborhood, center = self.group_divider(points)
        frames = self._estimate_patch_frames(neighborhood).to(dtype=points.dtype, device=points.device)
        canonical = torch.einsum("bgsc,bgcd->bgsd", neighborhood, frames)

        patch_encoder = self.teacher_patch_encoder if use_teacher else self.patch_encoder
        input_proj = self.teacher_input_proj if use_teacher else self.input_proj
        encoder = self.teacher_encoder if use_teacher else self.student_encoder

        patch_tokens = input_proj(patch_encoder(canonical))
        ri_pos = torch.einsum("bgc,bgcd->bgd", center, frames)
        pos_tokens = self.pos_embed(ri_pos)
        encoder_input = patch_tokens + pos_tokens
        attn_bias_full = self._build_orientation_bias(frames).to(dtype=encoder_input.dtype)
        tokens = encoder(encoder_input, attn_bias_full)

        max_features = tokens.max(dim=1).values
        mean_features = tokens.mean(dim=1)
        return torch.cat([max_features, mean_features], dim=-1)

    @torch.no_grad()
    def ema_update_teacher(self) -> None:
        decay = self.ema_decay
        self._ema_update_module(self.teacher_patch_encoder, self.patch_encoder, decay)
        self._ema_update_module(self.teacher_input_proj, self.input_proj, decay)
        self._ema_update_module(self.teacher_encoder, self.student_encoder, decay)

    @staticmethod
    @torch.no_grad()
    def _ema_update_module(teacher: nn.Module, student: nn.Module, decay: float) -> None:
        for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
            teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)
        for teacher_buf, student_buf in zip(teacher.buffers(), student.buffers()):
            if not teacher_buf.dtype.is_floating_point:
                teacher_buf.data.copy_(student_buf.data)
                continue
            teacher_buf.data.mul_(decay).add_(student_buf.data, alpha=1.0 - decay)


class RIMAEModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        ri_cfg = getattr(cfg, "ri_mae", None)

        def _ri_get(name: str, default):
            if ri_cfg is not None and hasattr(ri_cfg, name):
                return getattr(ri_cfg, name)
            return getattr(cfg, f"ri_mae_{name}", default)

        self.model = RIMAEBackbone(
            num_group=int(_ri_get("num_group", 64)),
            group_size=int(_ri_get("group_size", 32)),
            encoder_dims=int(_ri_get("encoder_dims", 384)),
            trans_dim=int(_ri_get("trans_dim", 384)),
            depth=int(_ri_get("depth", 8)),
            predictor_depth=int(_ri_get("predictor_depth", 2)),
            num_heads=int(_ri_get("num_heads", 6)),
            mask_ratio=float(_ri_get("mask_ratio", 0.75)),
            ema_decay=float(_ri_get("ema_decay", 0.996)),
            mlp_ratio=float(_ri_get("mlp_ratio", 4.0)),
            dropout=float(_ri_get("dropout", 0.0)),
            deterministic_fps=bool(_ri_get("deterministic_fps", False)),
            sorting_mode=str(_ri_get("sorting_mode", "nearest")),
        )

        legacy_crop_mode = str(_ri_get("crop_mode", "random")).strip()
        self.crop_mode_train = str(_ri_get("crop_mode_train", legacy_crop_mode)).strip()
        self.crop_mode_eval = str(_ri_get("crop_mode_eval", legacy_crop_mode)).strip()
        if not self.crop_mode_train:
            raise ValueError("ri_mae crop_mode_train must be a non-empty string.")
        if not self.crop_mode_eval:
            raise ValueError("ri_mae crop_mode_eval must be a non-empty string.")
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
        points = points - points.mean(dim=1, keepdim=True)
        return points

    @torch.no_grad()
    def _extract_supervised_features_from_batch(self, batch):
        """Extract no-mask RI features + class ids for split-level SVM metrics."""
        points_raw, meta = self._unpack_batch(batch)
        class_id = meta.get("class_id")
        if class_id is None:
            return None, None
        points = self._prepare_points(points_raw, stage="test")
        features = self.model.forward_eval(points, use_teacher=True)
        return features.detach().to(torch.float32), class_id

    def forward(self, batch_points: torch.Tensor, *, use_teacher: bool = True):
        """Return invariant features in contrastive-compatible tuple format."""
        points = self._prepare_points(batch_points)
        features = self.model.forward_eval(points, use_teacher=use_teacher)
        if not torch.is_tensor(features):
            raise RuntimeError(
                "RI-MAE forward_eval returned non-tensor features; "
                f"got type={type(features)}."
            )
        if features.dim() != 2:
            raise RuntimeError(
                "RI-MAE forward_eval must return a 2D feature tensor (B, D), "
                f"got shape={tuple(features.shape)}."
            )
        return features, features, None

    def _step(self, batch, stage: str) -> torch.Tensor:
        points_raw, meta = self._unpack_batch(batch)
        points = self._prepare_points(points_raw, stage=stage)

        loss, global_latent, mask_fraction, patch_centroids = self.model(points)

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
                        cached_features = self.model.forward_eval(points, use_teacher=True)
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

    def on_before_zero_grad(self, optimizer) -> None:
        if self.training:
            self.model.ema_update_teacher()

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


__all__ = ["RIMAEModule"]
