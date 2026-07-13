from __future__ import annotations

import copy
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Encoder
from .registry import register_encoder


# ---------------------------------------------------------------------------
# Point-cloud grouping helpers (originally from pointgpt)
# ---------------------------------------------------------------------------

def _to_bn3(points: torch.Tensor) -> torch.Tensor:
    if points.dim() != 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected point cloud with shape (B, N, 3), got {tuple(points.shape)}")
    return points


def _farthest_point_sample(xyz: torch.Tensor, npoint: int, *, deterministic: bool = False) -> torch.Tensor:
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


def _index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    points = _to_bn3(points)
    bsz = points.shape[0]
    idx_flat = idx.reshape(bsz, -1)
    gathered = points.gather(1, idx_flat.unsqueeze(-1).expand(-1, -1, points.shape[-1]))
    return gathered.view(bsz, *idx.shape[1:], points.shape[-1])


def _knn_point(k: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    xyz = _to_bn3(xyz)
    new_xyz = _to_bn3(new_xyz)
    if k <= 0 or k > xyz.shape[1]:
        raise ValueError(
            f"k must satisfy 1 <= k <= number of points, got k={k}, points={xyz.shape[1]}."
        )
    dist = torch.cdist(new_xyz.to(torch.float32), xyz.to(torch.float32))
    return dist.topk(k=k, dim=-1, largest=False).indices


# ---------------------------------------------------------------------------
# Group / Patch encoders / Positional embedding (originally from pointgpt)
# ---------------------------------------------------------------------------

class Group(nn.Module):
    def __init__(
        self,
        num_group: int,
        group_size: int,
        *,
        deterministic_fps: bool = False,
        sorting_mode: str = "nearest",
        group_sampling: str = "fps",
    ) -> None:
        super().__init__()
        self.num_group = int(num_group)
        self.group_size = int(group_size)
        self.deterministic_fps = bool(deterministic_fps)
        self.sorting_mode = sorting_mode
        if self.sorting_mode not in {"nearest", "none"}:
            raise ValueError(f"Unsupported sorting_mode={sorting_mode!r}.")
        # #9: opt-in random group-center selection during training. FPS
        # iterates `num_group` times and dispatches many small CUDA kernels
        # even for small `num_points` (80); `random` replaces the loop with
        # a single vectorised `topk` of random scores. In eval mode we
        # always fall back to FPS for reproducibility. `fps_always` keeps
        # today's behaviour; `random` switches the training path only.
        self.group_sampling = group_sampling
        if self.group_sampling not in {"fps", "random"}:
            raise ValueError(
                "Group sampling mode must be one of {'fps', 'random'}, "
                f"got {group_sampling!r}."
            )

    @staticmethod
    def _random_group_indices(xyz: torch.Tensor, num_group: int) -> torch.Tensor:
        bsz, n_pts, _ = xyz.shape
        if num_group > n_pts:
            raise ValueError(
                f"num_group ({num_group}) cannot exceed number of points ({n_pts}) "
                "when using random group sampling."
            )
        scores = torch.rand((bsz, n_pts), device=xyz.device, dtype=torch.float32)
        return scores.topk(num_group, dim=1, largest=True, sorted=False).indices

    @staticmethod
    def _nearest_path_order(center: torch.Tensor) -> torch.Tensor:
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
        if self.group_sampling == "random" and self.training:
            center_idx = self._random_group_indices(xyz, self.num_group)
        else:
            center_idx = _farthest_point_sample(xyz, self.num_group, deterministic=self.deterministic_fps)
        center = _index_points(xyz, center_idx)
        group_idx = _knn_point(self.group_size, xyz, center)
        neighborhood = _index_points(xyz, group_idx)
        neighborhood = neighborhood - center.unsqueeze(2)
        if self.sorting_mode == "nearest":
            order = self._nearest_path_order(center)
            neighborhood = neighborhood.gather(1, order.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.group_size, 3))
            center = center.gather(1, order.unsqueeze(-1).expand(-1, -1, 3))
        return neighborhood.contiguous(), center.contiguous()


class EncoderLarge(nn.Module):
    def __init__(self, encoder_channel: int) -> None:
        super().__init__()
        self.encoder_channel = int(encoder_channel)
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 256, 1), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, 1024, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(2048, 2048, 1), nn.BatchNorm1d(2048), nn.ReLU(inplace=True),
            nn.Conv1d(2048, self.encoder_channel, 1),
        )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        bsz, num_group, group_size, _ = point_groups.shape
        x = point_groups.reshape(bsz * num_group, group_size, 3)
        feat = self.first_conv(x.transpose(2, 1))
        global_feat = feat.max(dim=2, keepdim=True).values
        feat = torch.cat([global_feat.expand(-1, -1, group_size), feat], dim=1)
        feat = self.second_conv(feat)
        global_feat = feat.max(dim=2).values
        return global_feat.reshape(bsz, num_group, self.encoder_channel)


class EncoderSmall(nn.Module):
    def __init__(self, encoder_channel: int) -> None:
        super().__init__()
        self.encoder_channel = int(encoder_channel)
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        bsz, num_group, group_size, _ = point_groups.shape
        x = point_groups.reshape(bsz * num_group, group_size, 3)
        feat = self.first_conv(x.transpose(2, 1))
        global_feat = feat.max(dim=2, keepdim=True).values
        feat = torch.cat([global_feat.expand(-1, -1, group_size), feat], dim=1)
        feat = self.second_conv(feat)
        global_feat = feat.max(dim=2).values
        return global_feat.reshape(bsz, num_group, self.encoder_channel)


class PositionEmbeddingCoordsSine(nn.Module):
    def __init__(self, n_dim: int = 3, d_model: int = 384, temperature: float = 10000.0, scale: float = 1.0) -> None:
        super().__init__()
        self.n_dim = int(n_dim)
        self.num_pos_feats = (int(d_model) // self.n_dim // 2) * 2
        self.temperature = float(temperature)
        self.padding = int(d_model) - self.num_pos_feats * self.n_dim
        self.scale = float(scale) * 2.0 * math.pi

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2.0 * torch.div(dim_t, 2, rounding_mode="trunc") / self.num_pos_feats)
        xyz_scaled = xyz.to(torch.float32) * self.scale
        pos_div = xyz_scaled.unsqueeze(-1) / dim_t
        pos_sin = pos_div[..., 0::2].sin()
        pos_cos = pos_div[..., 1::2].cos()
        pos = torch.stack([pos_sin, pos_cos], dim=-1).reshape(*xyz.shape[:-1], -1)
        if self.padding > 0:
            pos = F.pad(pos, (0, self.padding))
        return pos.to(dtype=xyz.dtype)


# ---------------------------------------------------------------------------
# RI-MAE transformer blocks
# ---------------------------------------------------------------------------

class RIAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, *, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}.")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                "RIAttentionBlock requires embed_dim to be divisible by num_heads. "
                f"Got embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
                f"remainder={self.embed_dim % self.num_heads}."
            )
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
            nn.Linear(self.embed_dim, hidden_dim), nn.GELU(), nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.embed_dim), nn.Dropout(float(dropout)),
        )

    def _attention(self, x: torch.Tensor, attn_bias: torch.Tensor | None) -> torch.Tensor:
        # Fused SDPA path (#6): delegates to FlashAttention / mem-efficient
        # attention kernels when the inputs satisfy their shape and dtype
        # preconditions. Falls back to the math backend automatically when
        # they do not, so we get a speedup without losing correctness on
        # unsupported shapes (e.g. small seq_len on CPU). Dropout inside
        # SDPA is no-op in eval mode.
        batch_size, seq_len, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        attn_mask = None
        if attn_bias is not None:
            attn_mask = attn_bias.to(dtype=q.dtype)
        dropout_p = float(self.attn_drop.p) if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )
        out = out.transpose(1, 2).reshape(batch_size, seq_len, channels)
        return self.proj_drop(self.proj(out))

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
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            RIAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(int(depth))
        ])
        self.norm = nn.LayerNorm(int(embed_dim))
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        # Per-layer activation checkpointing (#5) re-runs each attention
        # block in backward instead of holding its intermediate activations.
        # Off by default; opt in via the encoder kwarg on memory-tight runs.
        if self.use_gradient_checkpointing and self.training and x.requires_grad:
            for block in self.layers:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, attn_bias, use_reentrant=False,
                )
            return self.norm(x)
        for block in self.layers:
            x = block(x, attn_bias)
        return self.norm(x)


# ---------------------------------------------------------------------------
# RI-MAE backbone
# ---------------------------------------------------------------------------

class RIMAEBackbone(nn.Module):
    _CUDA_EIGH_CHUNK_SIZE = 16384

    def __init__(
        self, *, num_group: int, group_size: int, encoder_dims: int, trans_dim: int,
        depth: int, predictor_depth: int, num_heads: int, mask_ratio: float,
        ema_decay: float, mlp_ratio: float, dropout: float,
        deterministic_fps: bool, sorting_mode: str,
        frame_builder: str = "triad",
        frame_eps: float = 1.0e-6,
        use_gradient_checkpointing: bool = False,
        group_sampling: str = "fps",
    ) -> None:
        super().__init__()
        self.num_group = int(num_group)
        self.group_size = int(group_size)
        self.encoder_dims = int(encoder_dims)
        self.trans_dim = int(trans_dim)
        self.num_heads = int(num_heads)
        self.mask_ratio = float(mask_ratio)
        self.ema_decay = float(ema_decay)
        self.frame_builder = frame_builder
        self.frame_eps = float(frame_eps)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.group_sampling = group_sampling
        if self.frame_builder not in {"triad", "pca"}:
            raise ValueError(
                "RIMAEBackbone frame_builder must be 'triad' or 'pca', "
                f"got {frame_builder!r}."
            )
        if self.frame_eps <= 0.0:
            raise ValueError(f"frame_eps must be > 0, got {self.frame_eps}.")

        self.group_divider = Group(
            num_group=self.num_group,
            group_size=self.group_size,
            deterministic_fps=bool(deterministic_fps),
            sorting_mode=sorting_mode,
            group_sampling=self.group_sampling,
        )
        self.patch_encoder = EncoderSmall(self.encoder_dims) if self.encoder_dims == 384 else EncoderLarge(self.encoder_dims)
        self.input_proj = nn.Identity() if self.encoder_dims == self.trans_dim else nn.Linear(self.encoder_dims, self.trans_dim)
        self.pos_embed = PositionEmbeddingCoordsSine(n_dim=3, d_model=self.trans_dim, scale=1.0)

        orient_hidden = max(64, self.trans_dim // 2)
        self.orientation_mlp = nn.Sequential(nn.Linear(9, orient_hidden), nn.GELU(), nn.Linear(orient_hidden, self.num_heads))
        self.orientation_scale = nn.Parameter(torch.tensor(1.0))

        self.student_encoder = RITransformer(
            embed_dim=self.trans_dim,
            num_heads=self.num_heads,
            depth=int(depth),
            mlp_ratio=float(mlp_ratio),
            dropout=float(dropout),
            use_gradient_checkpointing=self.use_gradient_checkpointing,
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
        self.predictor = RITransformer(embed_dim=self.trans_dim, num_heads=self.num_heads,
                                       depth=int(predictor_depth), mlp_ratio=float(mlp_ratio), dropout=float(dropout))

    @staticmethod
    def _stable_patch_eigh(cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if cov.device.type != "cuda" or cov.shape[0] <= RIMAEBackbone._CUDA_EIGH_CHUNK_SIZE:
            return torch.linalg.eigh(cov)

        eigvals_chunks = []
        eigvecs_chunks = []
        chunk_size = RIMAEBackbone._CUDA_EIGH_CHUNK_SIZE

        # Torch 2.11 can fail in cuSOLVER for large batches of tiny 3x3 matrices
        # even when the covariances are fully finite, so we solve them in chunks.
        for start in range(0, cov.shape[0], chunk_size):
            stop = min(start + chunk_size, cov.shape[0])
            cov_chunk = cov[start:stop]
            eigvals_chunk, eigvecs_chunk = torch.linalg.eigh(cov_chunk)
            eigvals_chunks.append(eigvals_chunk)
            eigvecs_chunks.append(eigvecs_chunk)

        return torch.cat(eigvals_chunks, dim=0), torch.cat(eigvecs_chunks, dim=0)

    @staticmethod
    def _normalize_frame_vectors(
        vectors: torch.Tensor,
        *,
        eps: float,
    ) -> torch.Tensor:
        norms = torch.linalg.vector_norm(vectors, dim=-1)
        return vectors / norms.clamp_min(eps).unsqueeze(-1)

    @staticmethod
    def _canonical_axis_like(
        vectors: torch.Tensor,
        axis_index: int,
    ) -> torch.Tensor:
        axis = torch.zeros_like(vectors)
        axis[:, axis_index] = 1.0
        return axis

    @staticmethod
    def _orthogonal_axis_completion(
        axis: torch.Tensor,
        *,
        eps: float,
    ) -> torch.Tensor:
        # A collinear patch determines its longitudinal axis but has no preferred
        # roll around it. Complete the frame with the Cartesian basis direction
        # least aligned with that axis.
        basis_index = axis.abs().argmin(dim=-1)
        basis = torch.eye(3, dtype=axis.dtype, device=axis.device)[basis_index]
        orthogonal = basis - (basis * axis).sum(dim=-1, keepdim=True) * axis
        return RIMAEBackbone._normalize_frame_vectors(orthogonal, eps=eps)

    @staticmethod
    def _replace_unorientable_vectors(
        vectors: torch.Tensor,
        completion: torch.Tensor,
        *,
        eps: float,
    ) -> torch.Tensor:
        unorientable = torch.linalg.vector_norm(
            vectors, dim=-1, keepdim=True
        ) <= eps
        return torch.where(unorientable, completion, vectors)

    @staticmethod
    def _apply_axis_sign_convention(patches: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
        projections = torch.einsum("bpc,bc->bp", patches, axis)
        batch_idx = torch.arange(axis.shape[0], device=axis.device)
        pivot_idx = projections.abs().argmax(dim=1)
        pivot_val = projections[batch_idx, pivot_idx]
        signs = torch.where(
            pivot_val >= 0,
            torch.ones_like(pivot_val),
            -torch.ones_like(pivot_val),
        )
        return axis * signs.unsqueeze(-1)

    @staticmethod
    def _estimate_patch_frames_pca(
        neighborhood: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_group, group_size, _ = neighborhood.shape
        with torch.autocast(device_type=neighborhood.device.type, enabled=False):
            patches = neighborhood.reshape(
                batch_size * num_group, group_size, 3
            ).float()
            covariance = patches.transpose(1, 2) @ patches / float(group_size)

        eigenvalues, eigenvectors = RIMAEBackbone._stable_patch_eigh(covariance)
        descending = eigenvalues.argsort(dim=-1, descending=True)
        basis = eigenvectors.gather(
            2,
            descending.unsqueeze(1).expand(-1, 3, -1),
        )

        coordinates = patches @ basis
        pivot_indices = coordinates.abs().argmax(dim=1)
        pivot_values = coordinates.gather(
            1,
            pivot_indices.unsqueeze(1),
        ).squeeze(1)
        signs = torch.where(
            pivot_values >= 0,
            torch.ones_like(pivot_values),
            -torch.ones_like(pivot_values),
        )
        basis = basis * signs.unsqueeze(1)

        handedness = torch.where(
            torch.det(basis) < 0,
            -torch.ones_like(signs[:, 2]),
            torch.ones_like(signs[:, 2]),
        )
        basis[:, :, 2] = basis[:, :, 2] * handedness.unsqueeze(-1)
        return basis.reshape(batch_size, num_group, 3, 3)

    @staticmethod
    def _estimate_patch_frames_triad(
        neighborhood: torch.Tensor,
        *,
        frame_eps: float,
    ) -> torch.Tensor:
        batch_size, num_group, group_size, _ = neighborhood.shape
        patches = neighborhood.reshape(batch_size * num_group, group_size, 3).contiguous()
        batch_idx = torch.arange(patches.shape[0], device=patches.device)

        radial_sq = torch.einsum("bpc,bpc->bp", patches, patches)
        primary_idx = radial_sq.argmax(dim=1)
        axis1_raw = patches[batch_idx, primary_idx]
        axis1_raw = RIMAEBackbone._replace_unorientable_vectors(
            axis1_raw,
            RIMAEBackbone._canonical_axis_like(axis1_raw, axis_index=0),
            eps=frame_eps,
        )
        axis1 = RIMAEBackbone._normalize_frame_vectors(axis1_raw, eps=frame_eps)
        axis1 = RIMAEBackbone._apply_axis_sign_convention(patches, axis1)

        axis1_proj = torch.einsum("bpc,bc->bp", patches, axis1).unsqueeze(-1)
        residual = patches - axis1_proj * axis1.unsqueeze(1)
        residual_sq = torch.einsum("bpc,bpc->bp", residual, residual)
        secondary_idx = residual_sq.argmax(dim=1)
        axis2_raw = residual[batch_idx, secondary_idx]
        axis2_raw = RIMAEBackbone._replace_unorientable_vectors(
            axis2_raw,
            RIMAEBackbone._orthogonal_axis_completion(axis1, eps=frame_eps),
            eps=frame_eps,
        )
        axis2 = RIMAEBackbone._normalize_frame_vectors(axis2_raw, eps=frame_eps)
        axis2 = RIMAEBackbone._apply_axis_sign_convention(patches, axis2)

        axis3_raw = torch.cross(axis1, axis2, dim=-1)
        axis3 = RIMAEBackbone._normalize_frame_vectors(axis3_raw, eps=frame_eps)

        axis2 = torch.cross(axis3, axis1, dim=-1)
        axis2 = RIMAEBackbone._normalize_frame_vectors(axis2, eps=frame_eps)
        axis2 = RIMAEBackbone._apply_axis_sign_convention(patches, axis2)

        axis3 = torch.cross(axis1, axis2, dim=-1)
        axis3 = RIMAEBackbone._normalize_frame_vectors(axis3, eps=frame_eps)
        basis = torch.stack([axis1, axis2, axis3], dim=-1)
        return basis.reshape(batch_size, num_group, 3, 3)

    @staticmethod
    def _estimate_patch_frames(
        neighborhood: torch.Tensor,
        *,
        frame_builder: str,
        frame_eps: float,
    ) -> torch.Tensor:
        patch_radii = torch.linalg.vector_norm(neighborhood, dim=-1).amax(dim=-1)
        fully_collapsed = patch_radii <= frame_eps
        if bool(fully_collapsed.any().item()):
            collapsed_count = int(fully_collapsed.sum().item())
            warnings.warn(
                "Patch-frame construction encountered "
                f"{collapsed_count} fully collapsed patch(es), where every neighbor "
                f"is within frame_eps={frame_eps:.3e} of its group center. "
                "Continuing with an arbitrary valid orientation for those patches.",
                RuntimeWarning,
                stacklevel=2,
            )
        if frame_builder == "triad":
            return RIMAEBackbone._estimate_patch_frames_triad(
                neighborhood,
                frame_eps=frame_eps,
            )
        if frame_builder == "pca":
            return RIMAEBackbone._estimate_patch_frames_pca(neighborhood)
        raise ValueError(
            f"frame_builder must be 'triad' or 'pca', got {frame_builder!r}."
        )


# ---------------------------------------------------------------------------
# Contrastive adapter
# ---------------------------------------------------------------------------

class RIMAEInvariantEncoderForContrastive(nn.Module):
    def __init__(self, backbone, *, center_input: bool) -> None:
        super().__init__()
        self.group_divider = backbone.group_divider
        self.patch_encoder = backbone.patch_encoder
        self.input_proj = backbone.input_proj
        self.pos_embed = backbone.pos_embed
        self.orientation_mlp = backbone.orientation_mlp
        self.orientation_scale = backbone.orientation_scale
        self.student_encoder = backbone.student_encoder
        self.frame_builder = backbone.frame_builder
        self.frame_eps = backbone.frame_eps
        self.center_input = bool(center_input)

    def _build_orientation_bias(self, frames: torch.Tensor) -> torch.Tensor:
        rel = torch.matmul(frames.unsqueeze(2).transpose(-1, -2), frames.unsqueeze(1))
        rel_flat = rel.reshape(rel.shape[0], rel.shape[1], rel.shape[2], 9)
        bias = self.orientation_mlp(rel_flat).permute(0, 3, 1, 2).contiguous()
        bias = bias - bias.mean(dim=-1, keepdim=True)
        return bias * self.orientation_scale.to(dtype=bias.dtype)

    def forward(self, points: torch.Tensor):
        bn3_points = _to_bn3(points)
        if self.center_input:
            bn3_points = bn3_points - bn3_points.mean(dim=1, keepdim=True)
        neighborhood, center = self.group_divider(bn3_points)
        with torch.no_grad():
            frames = RIMAEBackbone._estimate_patch_frames(
                neighborhood,
                frame_builder=self.frame_builder,
                frame_eps=self.frame_eps,
            )
        frames = frames.to(dtype=bn3_points.dtype, device=bn3_points.device)
        canonical = torch.einsum("bgsc,bgcd->bgsd", neighborhood, frames)
        patch_tokens = self.input_proj(self.patch_encoder(canonical))
        ri_pos = torch.einsum("bgc,bgcd->bgd", center, frames)
        pos_tokens = self.pos_embed(ri_pos)
        encoder_input = patch_tokens + pos_tokens
        attn_bias_full = self._build_orientation_bias(frames).to(dtype=encoder_input.dtype)
        tokens = self.student_encoder(encoder_input, attn_bias_full)
        max_features = tokens.max(dim=1).values
        mean_features = tokens.mean(dim=1)
        return torch.cat([max_features, mean_features], dim=-1)


# ---------------------------------------------------------------------------
# Registry-exposed encoder
# ---------------------------------------------------------------------------

@register_encoder("RI_MAE_Invariant")
class RIMAEInvariantEncoder(Encoder):
    """RI-MAE invariant transformer exposed through the encoder registry."""

    output_contract = "invariant"

    def __init__(
        self,
        *,
        latent_size: int,
        num_group: int = 64,
        group_size: int = 32,
        encoder_dims: int = 384,
        trans_dim: int = 384,
        depth: int = 8,
        predictor_depth: int = 2,
        num_heads: int = 6,
        mask_ratio: float = 0.75,
        ema_decay: float = 0.996,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        deterministic_fps: bool = False,
        sorting_mode: str = "nearest",
        center_input: bool = True,
        frame_builder: str = "triad",
        frame_eps: float = 1.0e-6,
        use_gradient_checkpointing: bool = False,
        group_sampling: str = "fps",
    ) -> None:
        super().__init__()
        output_dim = int(2 * int(trans_dim))
        if int(latent_size) != output_dim:
            raise ValueError(
                "RI_MAE_Invariant latent_size must match 2 * trans_dim. "
                f"Got latent_size={int(latent_size)}, trans_dim={int(trans_dim)}, "
                f"expected latent_size={output_dim}."
            )

        self.latent_size = output_dim
        self.invariant_dim = output_dim
        backbone = RIMAEBackbone(
            num_group=int(num_group),
            group_size=int(group_size),
            encoder_dims=int(encoder_dims),
            trans_dim=int(trans_dim),
            depth=int(depth),
            predictor_depth=int(predictor_depth),
            num_heads=int(num_heads),
            mask_ratio=float(mask_ratio),
            ema_decay=float(ema_decay),
            mlp_ratio=float(mlp_ratio),
            dropout=float(dropout),
            deterministic_fps=bool(deterministic_fps),
            sorting_mode=sorting_mode,
            frame_builder=frame_builder,
            frame_eps=float(frame_eps),
            use_gradient_checkpointing=bool(use_gradient_checkpointing),
            group_sampling=group_sampling,
        )
        self.encoder = RIMAEInvariantEncoderForContrastive(
            backbone,
            center_input=bool(center_input),
        )

    def forward(self, x):
        return self.encoder(x)
