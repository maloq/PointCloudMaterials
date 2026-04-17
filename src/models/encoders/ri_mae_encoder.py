from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Encoder
from .registry import register_encoder


# ---------------------------------------------------------------------------
# Point-cloud grouping helpers (originally from pointgpt)
# ---------------------------------------------------------------------------

def _to_bn3(points: torch.Tensor) -> torch.Tensor:
    if points.dim() != 3:
        raise ValueError(f"Expected point cloud with shape (B, N, 3) or (B, 3, N), got {tuple(points.shape)}")
    if points.shape[-1] == 3:
        return points
    if points.shape[1] == 3:
        return points.transpose(1, 2).contiguous()
    raise ValueError(f"Expected point cloud with 3 coordinates, got {tuple(points.shape)}")


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
    k_eff = min(int(k), int(xyz.shape[1]))
    if k_eff <= 0:
        raise ValueError(f"k must be positive, got {k}")
    dist = torch.cdist(new_xyz.to(torch.float32), xyz.to(torch.float32))
    return dist.topk(k=k_eff, dim=-1, largest=False).indices


# ---------------------------------------------------------------------------
# Group / Patch encoders / Positional embedding (originally from pointgpt)
# ---------------------------------------------------------------------------

class Group(nn.Module):
    def __init__(self, num_group: int, group_size: int, *, deterministic_fps: bool = False, sorting_mode: str = "nearest") -> None:
        super().__init__()
        self.num_group = int(num_group)
        self.group_size = int(group_size)
        self.deterministic_fps = bool(deterministic_fps)
        self.sorting_mode = str(sorting_mode).lower()
        if self.sorting_mode not in {"nearest", "none"}:
            raise ValueError(f"Unsupported sorting_mode={sorting_mode!r}.")

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
        fps_idx = _farthest_point_sample(xyz, self.num_group, deterministic=self.deterministic_fps)
        center = _index_points(xyz, fps_idx)
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
        batch_size, seq_len, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            attn = attn + attn_bias.to(dtype=attn.dtype)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, seq_len, channels)
        return self.proj_drop(self.proj(out))

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self._attention(self.norm1(x), attn_bias)
        x = x + self.mlp(self.norm2(x))
        return x


class RITransformer(nn.Module):
    def __init__(self, *, embed_dim: int, num_heads: int, depth: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            RIAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(int(depth))
        ])
        self.norm = nn.LayerNorm(int(embed_dim))

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
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
        frame_builder: str = "triad", frame_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_group = int(num_group)
        self.group_size = int(group_size)
        self.encoder_dims = int(encoder_dims)
        self.trans_dim = int(trans_dim)
        self.num_heads = int(num_heads)
        self.mask_ratio = float(mask_ratio)
        self.ema_decay = float(ema_decay)
        self.frame_builder = str(frame_builder).strip().lower()
        self.frame_eps = float(frame_eps)
        if self.frame_builder not in {"triad", "pca"}:
            raise ValueError(
                "RIMAEBackbone frame_builder must be one of {'triad', 'pca'}, "
                f"got {frame_builder!r}."
            )
        if self.frame_eps <= 0.0:
            raise ValueError(f"RIMAEBackbone frame_eps must be > 0, got {self.frame_eps}.")

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size,
                                   deterministic_fps=bool(deterministic_fps), sorting_mode=sorting_mode)
        self.patch_encoder = EncoderSmall(self.encoder_dims) if self.encoder_dims == 384 else EncoderLarge(self.encoder_dims)
        self.input_proj = nn.Identity() if self.encoder_dims == self.trans_dim else nn.Linear(self.encoder_dims, self.trans_dim)
        self.pos_embed = PositionEmbeddingCoordsSine(n_dim=3, d_model=self.trans_dim, scale=1.0)

        orient_hidden = max(64, self.trans_dim // 2)
        self.orientation_mlp = nn.Sequential(nn.Linear(9, orient_hidden), nn.GELU(), nn.Linear(orient_hidden, self.num_heads))
        self.orientation_scale = nn.Parameter(torch.tensor(1.0))

        self.student_encoder = RITransformer(embed_dim=self.trans_dim, num_heads=self.num_heads,
                                             depth=int(depth), mlp_ratio=float(mlp_ratio), dropout=float(dropout))
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
            try:
                eigvals_chunk, eigvecs_chunk = torch.linalg.eigh(cov_chunk)
            except RuntimeError as exc:
                nonfinite = int((~torch.isfinite(cov_chunk)).sum().item())
                if nonfinite > 0:
                    raise RuntimeError(
                        "Patch-frame covariance contains non-finite values before eigendecomposition. "
                        f"cov_shape={tuple(cov.shape)}, chunk_start={start}, chunk_stop={stop}, "
                        f"chunk_shape={tuple(cov_chunk.shape)}, nonfinite_values={nonfinite}, "
                        f"device={cov.device}, dtype={cov.dtype}."
                    ) from exc
                try:
                    eigvals_cpu, eigvecs_cpu = torch.linalg.eigh(cov_chunk.cpu())
                except RuntimeError as cpu_exc:
                    raise RuntimeError(
                        "Patch-frame eigendecomposition failed on both CUDA and CPU for finite covariances. "
                        f"cov_shape={tuple(cov.shape)}, chunk_start={start}, chunk_stop={stop}, "
                        f"chunk_shape={tuple(cov_chunk.shape)}, device={cov.device}, dtype={cov.dtype}."
                    ) from cpu_exc
                eigvals_chunk = eigvals_cpu.to(device=cov.device, dtype=cov.dtype)
                eigvecs_chunk = eigvecs_cpu.to(device=cov.device, dtype=cov.dtype)
            eigvals_chunks.append(eigvals_chunk)
            eigvecs_chunks.append(eigvecs_chunk)

        return torch.cat(eigvals_chunks, dim=0), torch.cat(eigvecs_chunks, dim=0)

    @staticmethod
    def _normalize_frame_vectors(
        vectors: torch.Tensor,
        *,
        eps: float,
        context: str,
    ) -> torch.Tensor:
        norms = torch.linalg.vector_norm(vectors, dim=-1)
        # Loud validation is only possible in eager mode: the `.item()` reads
        # below force a CPU sync that torch.compile cannot trace, so we skip
        # them while Dynamo is capturing the graph. Eager-mode paths
        # (sanity check, validation, pre-compile forwards, torch.compile
        # disabled) still raise on degenerate / non-finite inputs. In the
        # compiled path we rely on the clamp_min(eps) safety net below, so a
        # genuinely degenerate norm produces a large but finite vector
        # instead of silently propagating NaN/inf through the transformer.
        if not torch.compiler.is_compiling():
            if not bool(torch.isfinite(norms).all().item()):
                nonfinite = int((~torch.isfinite(norms)).sum().item())
                raise RuntimeError(
                    f"RI-MAE frame builder produced non-finite vector norms for {context}. "
                    f"vectors_shape={tuple(vectors.shape)}, nonfinite_norms={nonfinite}, "
                    f"dtype={vectors.dtype}, device={vectors.device}."
                )
            if bool((norms <= eps).any().item()):
                degenerate = int((norms <= eps).sum().item())
                raise RuntimeError(
                    f"RI-MAE frame builder encountered degenerate vectors for {context}. "
                    f"vectors_shape={tuple(vectors.shape)}, degenerate_count={degenerate}, "
                    f"min_norm={float(norms.min().item()):.6e}, eps={float(eps):.6e}, "
                    f"dtype={vectors.dtype}, device={vectors.device}."
                )
        return vectors / norms.clamp_min(eps).unsqueeze(-1)

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
    def _estimate_patch_frames_pca(neighborhood: torch.Tensor) -> torch.Tensor:
        batch_size, num_group, group_size, _ = neighborhood.shape
        with torch.autocast(device_type=neighborhood.device.type, enabled=False):
            patches = neighborhood.reshape(batch_size * num_group, group_size, 3).to(torch.float32)
            cov = torch.matmul(patches.transpose(1, 2), patches) / float(max(group_size, 1))
        eigvals, eigvecs = RIMAEBackbone._stable_patch_eigh(cov)
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
            signs[:, axis] = torch.where(picked >= 0, torch.ones_like(picked), -torch.ones_like(picked))
        basis = basis * signs.unsqueeze(1)
        det = torch.det(basis)
        handedness = torch.where(det < 0, -torch.ones_like(det), torch.ones_like(det))
        basis[:, :, 2] = basis[:, :, 2] * handedness.unsqueeze(-1)
        return basis.reshape(batch_size, num_group, 3, 3)

    @staticmethod
    def _estimate_patch_frames_triad(neighborhood: torch.Tensor, *, frame_eps: float) -> torch.Tensor:
        batch_size, num_group, group_size, _ = neighborhood.shape
        patches = neighborhood.reshape(batch_size * num_group, group_size, 3).contiguous()
        batch_idx = torch.arange(patches.shape[0], device=patches.device)

        radial_sq = torch.einsum("bpc,bpc->bp", patches, patches)
        primary_idx = radial_sq.argmax(dim=1)
        axis1_raw = patches[batch_idx, primary_idx]
        axis1 = RIMAEBackbone._normalize_frame_vectors(
            axis1_raw,
            eps=frame_eps,
            context="primary RI-MAE triad axis",
        )
        axis1 = RIMAEBackbone._apply_axis_sign_convention(patches, axis1)

        axis1_proj = torch.einsum("bpc,bc->bp", patches, axis1).unsqueeze(-1)
        residual = patches - axis1_proj * axis1.unsqueeze(1)
        residual_sq = torch.einsum("bpc,bpc->bp", residual, residual)
        secondary_idx = residual_sq.argmax(dim=1)
        axis2_raw = residual[batch_idx, secondary_idx]
        axis2 = RIMAEBackbone._normalize_frame_vectors(
            axis2_raw,
            eps=frame_eps,
            context="secondary RI-MAE triad axis",
        )
        axis2 = RIMAEBackbone._apply_axis_sign_convention(patches, axis2)

        axis3_raw = torch.cross(axis1, axis2, dim=-1)
        axis3 = RIMAEBackbone._normalize_frame_vectors(
            axis3_raw,
            eps=frame_eps,
            context="tertiary RI-MAE triad axis",
        )

        axis2 = torch.cross(axis3, axis1, dim=-1)
        axis2 = RIMAEBackbone._normalize_frame_vectors(
            axis2,
            eps=frame_eps,
            context="re-orthogonalized secondary RI-MAE triad axis",
        )
        axis2 = RIMAEBackbone._apply_axis_sign_convention(patches, axis2)

        axis3 = torch.cross(axis1, axis2, dim=-1)
        axis3 = RIMAEBackbone._normalize_frame_vectors(
            axis3,
            eps=frame_eps,
            context="final tertiary RI-MAE triad axis",
        )
        basis = torch.stack([axis1, axis2, axis3], dim=-1)
        return basis.reshape(batch_size, num_group, 3, 3)

    @staticmethod
    def _estimate_patch_frames(
        neighborhood: torch.Tensor,
        *,
        frame_builder: str = "triad",
        frame_eps: float = 1e-6,
    ) -> torch.Tensor:
        resolved_builder = str(frame_builder).strip().lower()
        if resolved_builder == "triad":
            return RIMAEBackbone._estimate_patch_frames_triad(neighborhood, frame_eps=float(frame_eps))
        if resolved_builder == "pca":
            return RIMAEBackbone._estimate_patch_frames_pca(neighborhood)
        raise ValueError(
            "RI-MAE frame builder must be one of {'triad', 'pca'}, "
            f"got {frame_builder!r}."
        )


# ---------------------------------------------------------------------------
# Contrastive adapter
# ---------------------------------------------------------------------------

class RIMAEInvariantEncoderForContrastive(nn.Module):
    expects_channel_first = False

    def __init__(self, backbone, *, center_input: bool, output_dim: int) -> None:
        super().__init__()
        required = ("group_divider", "patch_encoder", "input_proj", "pos_embed",
                     "orientation_mlp", "orientation_scale", "student_encoder")
        missing = [name for name in required if not hasattr(backbone, name)]
        if missing:
            raise TypeError(f"backbone is missing required attributes: {missing}.")

        self.group_divider = backbone.group_divider
        self.patch_encoder = backbone.patch_encoder
        self.input_proj = backbone.input_proj
        self.pos_embed = backbone.pos_embed
        self.orientation_mlp = backbone.orientation_mlp
        self.orientation_scale = backbone.orientation_scale
        self.student_encoder = backbone.student_encoder
        self.frame_builder = str(getattr(backbone, "frame_builder", "triad"))
        self.frame_eps = float(getattr(backbone, "frame_eps", 1e-6))
        self.center_input = bool(center_input)
        self.output_dim = int(output_dim)

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
        features = torch.cat([max_features, mean_features], dim=-1)
        return features, features, None


# ---------------------------------------------------------------------------
# Registry-exposed encoder
# ---------------------------------------------------------------------------

@register_encoder("RI_MAE_Invariant")
class RIMAEInvariantEncoder(Encoder):
    """
    RI-MAE invariant transformer encoder exposed through the common encoder registry.

    Returns tuple compatible with existing contrastive code:
    (inv_latent_net, inv_latent_net, eq_z=None)
    """

    expects_channel_first = False

    def __init__(
        self,
        *,
        latent_size: int | None = None,
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
        frame_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        output_dim = int(2 * int(trans_dim))
        if latent_size is not None and int(latent_size) != output_dim:
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
            sorting_mode=str(sorting_mode),
            frame_builder=str(frame_builder),
            frame_eps=float(frame_eps),
        )
        self.encoder = RIMAEInvariantEncoderForContrastive(
            backbone,
            center_input=bool(center_input),
            output_dim=output_dim,
        )

    def forward(self, x):
        return self.encoder(x)
