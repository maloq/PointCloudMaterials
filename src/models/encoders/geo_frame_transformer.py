from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Encoder
from .registry import register_encoder
from .ri_mae_encoder import (
    Group,
    PositionEmbeddingCoordsSine,
    RIMAEBackbone,
    RITransformer,
    _to_bn3,
)


class PatchPointEncoder(nn.Module):
    """Batch-independent point MLP for one patch scale."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        hidden_dim = max(64, int(output_dim) // 2)
        self.point_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, int(output_dim)),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * int(output_dim), int(output_dim)),
            nn.LayerNorm(int(output_dim)),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        point_features = self.point_mlp(patches)
        pooled = torch.cat(
            [point_features.max(dim=2).values, point_features.mean(dim=2)],
            dim=-1,
        )
        return self.fusion(pooled)


class InvariantPatchEncoder(nn.Module):
    """Rotation-invariant point encoder using within-patch distance geometry."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        hidden_dim = max(64, int(output_dim) // 2)
        self.point_mlp = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, int(output_dim)),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * int(output_dim), int(output_dim)),
            nn.LayerNorm(int(output_dim)),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        patches_float = patches.float()
        radius_squared = patches_float.square().sum(dim=-1)
        gram = patches_float @ patches_float.transpose(-1, -2)
        distance_squared = (
            radius_squared.unsqueeze(-1)
            + radius_squared.unsqueeze(-2)
            - 2.0 * gram
        ).clamp_min(0.0)
        distance = torch.sqrt(distance_squared + 1.0e-12)
        patch_size = int(patches.shape[2])
        diagonal = torch.eye(
            patch_size,
            dtype=torch.bool,
            device=patches.device,
        ).view(1, 1, patch_size, patch_size)
        nonself_distance = distance.masked_fill(diagonal, torch.inf)

        distance_mean = distance.sum(dim=-1) / float(max(1, patch_size - 1))
        centered_distance = distance.masked_fill(diagonal, 0.0) - distance_mean.unsqueeze(-1)
        distance_std = torch.sqrt(
            centered_distance.square().masked_fill(diagonal, 0.0).sum(dim=-1)
            / float(max(1, patch_size - 1))
        )
        nearest = nonself_distance.min(dim=-1).values
        farthest = distance.max(dim=-1).values
        centroid = patches_float.mean(dim=2)
        centroid_alignment = torch.einsum("bgsc,bgc->bgs", patches_float, centroid)
        covariance = patches_float.transpose(-1, -2) @ patches_float / float(patch_size)
        covariance_energy = torch.einsum(
            "bgsc,bgcd,bgsd->bgs",
            patches_float,
            covariance,
            patches_float,
        )
        point_invariants = torch.stack(
            [
                torch.sqrt(radius_squared + 1.0e-12),
                radius_squared,
                distance_mean,
                distance_std,
                nearest,
                farthest,
                centroid_alignment,
                covariance_energy,
            ],
            dim=-1,
        ).to(dtype=patches.dtype)
        point_features = self.point_mlp(point_invariants)
        pooled = torch.cat(
            [point_features.max(dim=2).values, point_features.mean(dim=2)],
            dim=-1,
        )
        return self.fusion(pooled)


class PairwisePatchGeometryBias(nn.Module):
    """Rotation-invariant attention bias from relative patch geometry."""

    def __init__(self, *, num_heads: int, hidden_dim: int, num_rbf: int = 16) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.num_rbf = int(num_rbf)
        if self.num_rbf < 2:
            raise ValueError(f"GeoFrameTransformer num_rbf must be >= 2, got {self.num_rbf}.")
        self.register_buffer(
            "rbf_centers",
            torch.linspace(0.0, 3.0, self.num_rbf),
            persistent=True,
        )
        feature_dim = self.num_rbf + 6 + 9 + 3
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.num_heads),
        )
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        centers: torch.Tensor,
        frames: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        delta = centers.unsqueeze(1) - centers.unsqueeze(2)
        distance = torch.linalg.vector_norm(delta.float(), dim=-1)
        cloud_scale = torch.sqrt(centers.float().square().sum(dim=-1).mean(dim=1, keepdim=True))
        normalized_distance = distance / cloud_scale.clamp_min(1.0e-6).unsqueeze(-1)
        rbf_width = self.rbf_centers[1] - self.rbf_centers[0]
        rbf = torch.exp(
            -0.5
            * (
                (normalized_distance.unsqueeze(-1) - self.rbf_centers)
                / rbf_width.clamp_min(1.0e-6)
            ).square()
        )

        unit_delta = delta.float() / distance.clamp_min(1.0e-6).unsqueeze(-1)
        direction_i = torch.einsum("bijc,bicd->bijd", unit_delta, frames.float())
        direction_j = torch.einsum("bijc,bjcd->bijd", unit_delta, frames.float())
        confidence_i = confidence.float().unsqueeze(2)
        confidence_j = confidence.float().unsqueeze(1)
        pair_confidence = torch.sqrt((confidence_i * confidence_j).clamp_min(0.0))

        relative_frame = torch.matmul(
            frames.float().unsqueeze(2).transpose(-1, -2),
            frames.float().unsqueeze(1),
        ).reshape(*delta.shape[:3], 9)
        geometry = torch.cat(
            [
                rbf,
                direction_i * confidence_i.unsqueeze(-1),
                direction_j * confidence_j.unsqueeze(-1),
                relative_frame * pair_confidence.unsqueeze(-1),
                confidence_i.expand_as(pair_confidence).unsqueeze(-1),
                confidence_j.expand_as(pair_confidence).unsqueeze(-1),
                pair_confidence.unsqueeze(-1),
            ],
            dim=-1,
        )
        geometry = geometry.to(dtype=self.mlp[0].weight.dtype)
        bias = self.mlp(geometry).permute(0, 3, 1, 2).contiguous()
        bias = bias - bias.mean(dim=-1, keepdim=True)
        return bias * self.scale.to(dtype=bias.dtype)


class RelativeFrameOrientationBias(nn.Module):
    """Minimal attention bias from relative local-frame orientation only."""

    def __init__(self, *, num_heads: int, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(9, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(num_heads)),
        )
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        centers: torch.Tensor,
        frames: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        del centers, confidence
        relative_frame = torch.matmul(
            frames.float().unsqueeze(2).transpose(-1, -2),
            frames.float().unsqueeze(1),
        ).reshape(frames.shape[0], frames.shape[1], frames.shape[1], 9)
        relative_frame = relative_frame.to(dtype=self.mlp[0].weight.dtype)
        bias = self.mlp(relative_frame).permute(0, 3, 1, 2).contiguous()
        bias = bias - bias.mean(dim=-1, keepdim=True)
        return bias * self.scale.to(dtype=bias.dtype)


class MultiQueryAttentionPool(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        output_dim: int,
        num_heads: int,
        num_queries: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        self.token_dim = int(token_dim)
        if self.num_queries < 1:
            raise ValueError(
                f"GeoFrameTransformer pool_queries must be >= 1, got {self.num_queries}."
            )
        self.queries = nn.Parameter(torch.empty(1, self.num_queries, self.token_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm = nn.LayerNorm(self.token_dim)
        self.output = nn.Linear(self.num_queries * self.token_dim, int(output_dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        query = self.queries.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attention(query=query, key=tokens, value=tokens, need_weights=False)
        return self.output(self.norm(pooled).flatten(1))


class GeoFrameTokenEncoder(nn.Module):
    """Multi-scale, confidence-gated frame transformer over local point patches."""

    def __init__(
        self,
        *,
        num_group: int,
        patch_sizes: tuple[int, ...],
        encoder_dims: int,
        trans_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        deterministic_fps: bool,
        sorting_mode: str,
        group_sampling: str,
        frame_builder: str,
        frame_eps: float,
        use_frame_gating: bool,
        frame_confidence_floor: float,
        geometry_bias_mode: str,
        num_rbf: int,
        use_gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.num_group = int(num_group)
        self.patch_sizes = tuple(int(size) for size in patch_sizes)
        self.trans_dim = int(trans_dim)
        self.frame_builder = str(frame_builder).strip().lower()
        self.frame_eps = float(frame_eps)
        self.use_frame_gating = bool(use_frame_gating)
        self.frame_confidence_floor = float(frame_confidence_floor)
        self.geometry_bias_mode = str(geometry_bias_mode).strip().lower()
        if not self.patch_sizes or any(size < 3 for size in self.patch_sizes):
            raise ValueError(
                "GeoFrameTransformer patch_sizes must contain integers >= 3, "
                f"got {self.patch_sizes}."
            )
        if tuple(sorted(set(self.patch_sizes))) != self.patch_sizes:
            raise ValueError(
                "GeoFrameTransformer patch_sizes must be strictly increasing with no duplicates, "
                f"got {self.patch_sizes}."
            )
        if self.frame_builder not in {"triad", "pca"}:
            raise ValueError(
                "GeoFrameTransformer frame_builder must be 'triad' or 'pca', "
                f"got {self.frame_builder!r}."
            )
        if self.frame_eps <= 0.0:
            raise ValueError(
                f"GeoFrameTransformer frame_eps must be > 0, got {self.frame_eps}."
            )
        if not (0.0 <= self.frame_confidence_floor <= 1.0):
            raise ValueError(
                "GeoFrameTransformer frame_confidence_floor must be in [0, 1], "
                f"got {self.frame_confidence_floor}."
            )
        if self.geometry_bias_mode not in {"pairwise", "orientation", "none"}:
            raise ValueError(
                "GeoFrameTransformer geometry_bias_mode must be 'pairwise', "
                f"'orientation', or 'none', got {self.geometry_bias_mode!r}."
            )

        self.group_divider = Group(
            num_group=self.num_group,
            group_size=max(self.patch_sizes),
            deterministic_fps=bool(deterministic_fps),
            sorting_mode=str(sorting_mode),
            group_sampling=str(group_sampling),
        )
        self.patch_encoders = nn.ModuleList(
            [PatchPointEncoder(int(encoder_dims)) for _ in self.patch_sizes]
        )
        self.patch_projections = nn.ModuleList(
            [nn.Linear(int(encoder_dims), self.trans_dim) for _ in self.patch_sizes]
        )
        self.invariant_patch_encoders = nn.ModuleList(
            [InvariantPatchEncoder(self.trans_dim) for _ in self.patch_sizes]
            if self.use_frame_gating
            else []
        )
        if len(self.patch_sizes) > 1:
            self.scale_embeddings = nn.Parameter(
                torch.empty(len(self.patch_sizes), 1, 1, self.trans_dim)
            )
            nn.init.trunc_normal_(self.scale_embeddings, std=0.02)
            self.scale_fusion = nn.Sequential(
                nn.Linear(len(self.patch_sizes) * self.trans_dim, self.trans_dim),
                nn.GELU(),
                nn.Linear(self.trans_dim, self.trans_dim),
            )
        else:
            self.register_parameter("scale_embeddings", None)
            self.scale_fusion = nn.Identity()
        self.pos_embed = PositionEmbeddingCoordsSine(
            n_dim=3,
            d_model=self.trans_dim,
            scale=1.0,
        )
        self.radial_position = nn.Sequential(
            nn.Linear(1, self.trans_dim),
            nn.GELU(),
            nn.Linear(self.trans_dim, self.trans_dim),
        )
        if self.geometry_bias_mode == "pairwise":
            self.geometry_bias = PairwisePatchGeometryBias(
                num_heads=int(num_heads),
                hidden_dim=max(64, self.trans_dim // 2),
                num_rbf=int(num_rbf),
            )
        elif self.geometry_bias_mode == "orientation":
            self.geometry_bias = RelativeFrameOrientationBias(
                num_heads=int(num_heads),
                hidden_dim=max(64, self.trans_dim // 2),
            )
        else:
            self.geometry_bias = None
        self.transformer = RITransformer(
            embed_dim=self.trans_dim,
            num_heads=int(num_heads),
            depth=int(depth),
            mlp_ratio=float(mlp_ratio),
            dropout=float(dropout),
            use_gradient_checkpointing=bool(use_gradient_checkpointing),
        )

    def group_points(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.group_divider(points)

    def _frames_and_confidence(
        self,
        neighborhood: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad(), torch.autocast(device_type=neighborhood.device.type, enabled=False):
            frames = RIMAEBackbone._estimate_patch_frames(
                neighborhood.float(),
                frame_builder=self.frame_builder,
                frame_eps=self.frame_eps,
                validate=not self.training,
            )
            batch_size, num_group, group_size, _ = neighborhood.shape
            if self.use_frame_gating:
                patches = neighborhood.float().reshape(batch_size * num_group, group_size, 3)
                covariance = patches.transpose(1, 2) @ patches / float(group_size)
                eigenvalues, _ = RIMAEBackbone._stable_patch_eigh(covariance)
                eigenvalues = eigenvalues.clamp_min(0.0)
                low, middle, high = eigenvalues.unbind(dim=-1)
                high_safe = high.clamp_min(self.frame_eps)
                primary_gap = (high - middle) / high_safe
                secondary_gap = (middle - low) / high_safe
                confidence = torch.sqrt(
                    (
                        primary_gap.clamp(0.0, 1.0)
                        * secondary_gap.clamp(0.0, 1.0)
                    ).clamp_min(0.0)
                ).reshape(batch_size, num_group)
            else:
                confidence = torch.ones(
                    (batch_size, num_group),
                    dtype=torch.float32,
                    device=neighborhood.device,
                )
        return (
            frames.to(device=neighborhood.device, dtype=neighborhood.dtype),
            confidence.to(device=neighborhood.device, dtype=neighborhood.dtype),
        )

    def prepare_tokens(
        self,
        neighborhood: torch.Tensor,
        centers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        frames, confidence = self._frames_and_confidence(neighborhood)
        frame_weight = (
            self.frame_confidence_floor
            + (1.0 - self.frame_confidence_floor) * confidence
            if self.use_frame_gating
            else torch.ones_like(confidence)
        )
        frame_weight_token = frame_weight.unsqueeze(-1)
        scale_tokens = []
        for scale_index, patch_size in enumerate(self.patch_sizes):
            patch = neighborhood[:, :, :patch_size]
            canonical = torch.einsum("bgsc,bgcd->bgsd", patch, frames)
            canonical_token = self.patch_projections[scale_index](
                self.patch_encoders[scale_index](canonical)
            )
            if self.use_frame_gating:
                invariant_token = self.invariant_patch_encoders[scale_index](patch)
                token = (
                    frame_weight_token * canonical_token
                    + (1.0 - frame_weight_token) * invariant_token
                )
            else:
                token = canonical_token
            if self.scale_embeddings is not None:
                token = token + self.scale_embeddings[scale_index]
            scale_tokens.append(token)
        patch_tokens = self.scale_fusion(torch.cat(scale_tokens, dim=-1))

        local_centers = torch.einsum("bgc,bgcd->bgd", centers, frames)
        frame_position = self.pos_embed(local_centers)
        center_radius = torch.linalg.vector_norm(centers.float(), dim=-1, keepdim=True).to(
            dtype=centers.dtype
        )
        radial_position_tokens = self.radial_position(center_radius)
        position_tokens = (
            frame_weight_token * frame_position
            + (1.0 - frame_weight_token) * radial_position_tokens
        )
        encoder_input = patch_tokens + position_tokens
        attention_bias = self.build_attention_bias(centers, frames, confidence)
        if attention_bias is not None:
            attention_bias = attention_bias.to(dtype=encoder_input.dtype)
        state = {
            "centers": centers,
            "frames": frames,
            "confidence": confidence,
            "frame_weight": frame_weight,
            "position_tokens": position_tokens,
            "radial_position_tokens": radial_position_tokens,
        }
        return encoder_input, attention_bias, state

    def build_attention_bias(
        self,
        centers: torch.Tensor,
        frames: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.geometry_bias is None:
            return None
        return self.geometry_bias(centers, frames, confidence)

    def encode_grouped(
        self,
        neighborhood: torch.Tensor,
        centers: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoder_input, attention_bias, state = self.prepare_tokens(neighborhood, centers)
        tokens = self.transformer(encoder_input, attention_bias)
        state["tokens"] = tokens
        return tokens, state


@register_encoder("GeoFrameTransformer")
class GeoFrameTransformerEncoder(Encoder):
    """Geometry-aware point transformer for compact, directional local manifolds.

    RI_MAE_Invariant remains registered separately and is intentionally not modified.
    """

    expects_channel_first = False

    def __init__(
        self,
        *,
        latent_size: int = 256,
        num_group: int = 24,
        patch_sizes: tuple[int, ...] | list[int] = (12, 24),
        encoder_dims: int = 256,
        trans_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        deterministic_fps: bool = False,
        sorting_mode: str = "none",
        group_sampling: str = "random",
        center_input: bool = True,
        frame_builder: str = "triad",
        frame_eps: float = 1.0e-6,
        use_frame_gating: bool = True,
        frame_confidence_floor: float = 0.5,
        geometry_bias_mode: str = "pairwise",
        num_rbf: int = 16,
        pool_queries: int = 2,
        pooling_mode: str = "attention",
        ray_feature_dim: int = 64,
        mask_ratio: float = 0.6,
        mask_predictor_depth: int = 2,
        ema_decay: float = 0.996,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.latent_size = int(latent_size)
        self.invariant_dim = self.latent_size
        self.output_dim = self.latent_size
        self.trans_dim = int(trans_dim)
        self.center_input = bool(center_input)
        self.directional_feature_dim = int(ray_feature_dim)
        self.mask_ratio = float(mask_ratio)
        self.ema_decay = float(ema_decay)
        self.pooling_mode = str(pooling_mode).strip().lower()
        if self.latent_size <= 0:
            raise ValueError(
                f"GeoFrameTransformer latent_size must be > 0, got {self.latent_size}."
            )
        if self.trans_dim <= 0 or self.trans_dim % int(num_heads) != 0:
            raise ValueError(
                "GeoFrameTransformer trans_dim must be positive and divisible by num_heads. "
                f"Got trans_dim={self.trans_dim}, num_heads={int(num_heads)}."
            )
        if self.directional_feature_dim <= 0:
            raise ValueError(
                "GeoFrameTransformer ray_feature_dim must be > 0, "
                f"got {self.directional_feature_dim}."
            )
        if not (0.0 < self.mask_ratio < 1.0):
            raise ValueError(
                f"GeoFrameTransformer mask_ratio must be in (0, 1), got {self.mask_ratio}."
            )
        if not (0.0 <= self.ema_decay < 1.0):
            raise ValueError(
                f"GeoFrameTransformer ema_decay must be in [0, 1), got {self.ema_decay}."
            )
        if self.pooling_mode not in {"attention", "max_mean"}:
            raise ValueError(
                "GeoFrameTransformer pooling_mode must be 'attention' or 'max_mean', "
                f"got {self.pooling_mode!r}."
            )
        if self.pooling_mode == "max_mean" and self.latent_size != self.trans_dim:
            raise ValueError(
                "GeoFrameTransformer pooling_mode='max_mean' requires latent_size == trans_dim. "
                f"Got latent_size={self.latent_size}, trans_dim={self.trans_dim}."
            )

        token_encoder_kwargs = dict(
            num_group=int(num_group),
            patch_sizes=tuple(int(size) for size in patch_sizes),
            encoder_dims=int(encoder_dims),
            trans_dim=self.trans_dim,
            depth=int(depth),
            num_heads=int(num_heads),
            mlp_ratio=float(mlp_ratio),
            dropout=float(dropout),
            deterministic_fps=bool(deterministic_fps),
            sorting_mode=str(sorting_mode),
            group_sampling=str(group_sampling),
            frame_builder=str(frame_builder),
            frame_eps=float(frame_eps),
            use_frame_gating=bool(use_frame_gating),
            frame_confidence_floor=float(frame_confidence_floor),
            geometry_bias_mode=str(geometry_bias_mode),
            num_rbf=int(num_rbf),
            use_gradient_checkpointing=bool(use_gradient_checkpointing),
        )
        self.token_encoder = GeoFrameTokenEncoder(**token_encoder_kwargs)
        self.pool = (
            MultiQueryAttentionPool(
                token_dim=self.trans_dim,
                output_dim=self.latent_size,
                num_heads=int(num_heads),
                num_queries=int(pool_queries),
                dropout=float(dropout),
            )
            if self.pooling_mode == "attention"
            else None
        )

        self.ray_token_mlp = nn.Sequential(
            nn.Linear(9, self.trans_dim),
            nn.GELU(),
            nn.Linear(self.trans_dim, self.trans_dim),
        )
        self.ray_query = nn.Parameter(torch.empty(1, 1, self.trans_dim))
        nn.init.trunc_normal_(self.ray_query, std=0.02)
        self.ray_attention = nn.MultiheadAttention(
            embed_dim=self.trans_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.ray_output = nn.Sequential(
            nn.LayerNorm(self.trans_dim),
            nn.Linear(self.trans_dim, self.directional_feature_dim),
        )

        self.mask_token = nn.Parameter(torch.empty(1, 1, self.trans_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.mask_predictor = RITransformer(
            embed_dim=self.trans_dim,
            num_heads=int(num_heads),
            depth=int(mask_predictor_depth),
            mlp_ratio=float(mlp_ratio),
            dropout=float(dropout),
            use_gradient_checkpointing=bool(use_gradient_checkpointing),
        )
        self.mask_prediction_head = nn.Linear(self.trans_dim, self.trans_dim)

        self.mask_teacher = copy.deepcopy(self.token_encoder)
        for parameter in self.mask_teacher.parameters():
            parameter.requires_grad_(False)
        self.mask_teacher.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.enforce_frozen_teacher()
        return self

    def enforce_frozen_teacher(self) -> None:
        for parameter in self.mask_teacher.parameters():
            parameter.requires_grad_(False)
        self.mask_teacher.eval()

    @torch.no_grad()
    def reset_mask_teacher_from_student(self) -> None:
        self.mask_teacher.load_state_dict(self.token_encoder.state_dict(), strict=True)
        self.enforce_frozen_teacher()

    def _center_points(self, points: torch.Tensor) -> torch.Tensor:
        points = _to_bn3(points)
        if self.center_input:
            points = points - points.mean(dim=1, keepdim=True)
        return points

    def forward(self, points: torch.Tensor):
        centered = self._center_points(points)
        neighborhood, centers = self.token_encoder.group_points(centered)
        tokens, state = self.token_encoder.encode_grouped(neighborhood, centers)
        if self.pool is None:
            features = 0.5 * (tokens.max(dim=1).values + tokens.mean(dim=1))
        else:
            features = self.pool(tokens)
        return features, features, None, state

    def directional_features_from_geometry(
        self,
        state: dict[str, torch.Tensor],
        ray_direction: torch.Tensor,
    ) -> torch.Tensor:
        required = {"tokens", "centers", "frames", "confidence"}
        missing = sorted(required.difference(state))
        if missing:
            raise ValueError(
                "GeoFrameTransformer directional state is missing tensors: "
                f"{missing}. Available keys={sorted(state)}."
            )
        tokens = state["tokens"]
        centers = state["centers"]
        frames = state["frames"]
        confidence = state["confidence"]
        if ray_direction.shape != (tokens.shape[0], 3):
            raise ValueError(
                "GeoFrameTransformer ray_direction must have shape (B, 3) aligned with tokens. "
                f"Got ray_direction={tuple(ray_direction.shape)}, tokens={tuple(tokens.shape)}."
            )
        ray = F.normalize(ray_direction.float(), dim=-1, eps=1.0e-6)
        ray_local = torch.einsum("bc,bgcd->bgd", ray, frames.float())
        longitudinal = torch.einsum("bgc,bc->bg", centers.float(), ray)
        center_scale = torch.sqrt(centers.float().square().sum(dim=-1).mean(dim=1, keepdim=True))
        longitudinal = longitudinal / center_scale.clamp_min(1.0e-6)
        center_radius_sq = centers.float().square().sum(dim=-1)
        transverse = torch.sqrt(
            (center_radius_sq - longitudinal.square() * center_scale.square()).clamp_min(0.0)
        ) / center_scale.clamp_min(1.0e-6)
        ray_geometry = torch.cat(
            [
                ray_local * confidence.float().unsqueeze(-1),
                ray_local.abs(),
                longitudinal.unsqueeze(-1),
                transverse.unsqueeze(-1),
                confidence.float().unsqueeze(-1),
            ],
            dim=-1,
        ).to(dtype=tokens.dtype)
        conditioned_tokens = tokens + self.ray_token_mlp(ray_geometry)
        query = self.ray_query.expand(tokens.shape[0], -1, -1)
        attended, _ = self.ray_attention(
            query=query,
            key=conditioned_tokens,
            value=conditioned_tokens,
            need_weights=False,
        )
        return self.ray_output(attended.squeeze(1))

    def masked_token_loss(self, points: torch.Tensor) -> torch.Tensor:
        centered = self._center_points(points)
        neighborhood, centers = self.token_encoder.group_points(centered)
        student_input, _, student_state = self.token_encoder.prepare_tokens(
            neighborhood,
            centers,
        )
        with torch.no_grad():
            teacher_tokens, _ = self.mask_teacher.encode_grouped(neighborhood, centers)

        batch_size, num_group, _ = student_input.shape
        mask_count = max(1, min(num_group - 1, int(round(num_group * self.mask_ratio))))
        mask_scores = torch.rand((batch_size, num_group), device=student_input.device)
        mask_indices = mask_scores.topk(mask_count, dim=1, largest=True, sorted=False).indices
        mask = torch.zeros(
            (batch_size, num_group),
            dtype=torch.bool,
            device=student_input.device,
        )
        mask.scatter_(1, mask_indices, True)
        masked_input = torch.where(
            mask.unsqueeze(-1),
            self.mask_token.to(dtype=student_input.dtype)
            + student_state["radial_position_tokens"],
            student_input,
        )
        masked_confidence = student_state["confidence"].masked_fill(mask, 0.0)
        masked_attention_bias = self.token_encoder.build_attention_bias(
            centers,
            student_state["frames"],
            masked_confidence,
        )
        if masked_attention_bias is not None:
            masked_attention_bias = masked_attention_bias.to(dtype=student_input.dtype)
        predicted_tokens = self.mask_prediction_head(
            self.mask_predictor(masked_input, masked_attention_bias)
        )
        predicted_masked = F.normalize(predicted_tokens[mask].float(), dim=-1, eps=1.0e-6)
        teacher_masked = F.normalize(teacher_tokens[mask].float(), dim=-1, eps=1.0e-6)
        return (2.0 - 2.0 * (predicted_masked * teacher_masked).sum(dim=-1)).mean()

    @torch.no_grad()
    def update_mask_teacher(self) -> None:
        student_parameters = dict(self.token_encoder.named_parameters())
        teacher_parameters = dict(self.mask_teacher.named_parameters())
        if student_parameters.keys() != teacher_parameters.keys():
            raise RuntimeError(
                "GeoFrameTransformer EMA teacher parameter structure no longer matches the student."
            )
        for name, teacher_parameter in teacher_parameters.items():
            teacher_parameter.mul_(self.ema_decay).add_(
                student_parameters[name],
                alpha=1.0 - self.ema_decay,
            )

        student_buffers = dict(self.token_encoder.named_buffers())
        teacher_buffers = dict(self.mask_teacher.named_buffers())
        if student_buffers.keys() != teacher_buffers.keys():
            raise RuntimeError(
                "GeoFrameTransformer EMA teacher buffer structure no longer matches the student."
            )
        for name, teacher_buffer in teacher_buffers.items():
            teacher_buffer.copy_(student_buffers[name])


__all__ = ["GeoFrameTransformerEncoder"]
