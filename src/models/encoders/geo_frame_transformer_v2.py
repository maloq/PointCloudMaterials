from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Encoder
from .geo_frame_transformer import (
    InvariantPatchEncoder,
    MultiQueryAttentionPool,
    PatchPointEncoder,
    _relative_frame_products,
)
from .registry import register_encoder
from .ri_mae_encoder import Group, PositionEmbeddingCoordsSine, RIMAEBackbone, _to_bn3


_LOCAL_SHAPE_DIM = 8
LOCAL_SHAPE_FEATURE_NAMES = (
    "normalized_eigenvalue_low",
    "normalized_eigenvalue_middle",
    "normalized_eigenvalue_high",
    "primary_eigengap",
    "secondary_eigengap",
    "normalized_rms_radius",
    "log_normalized_density",
    "frame_confidence",
)


def _parity_sign(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Reflection in the third coordinate of a right-handed local frame."""
    return torch.tensor((1.0, 1.0, -1.0), device=device, dtype=dtype)


def _symmetric_3x3_eigenvalues(
    matrices: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Closed-form ordered eigenvalues for the V2 covariance matrices.

    The repository produces symmetric positive-semidefinite ``(M, 3, 3)``
    patch covariances here.  The trigonometric symmetric-matrix solution avoids
    launching batched cuSOLVER eigensystems (and computing unused eigenvectors)
    for hundreds of thousands of tiny matrices.
    """
    mean_eigenvalue = matrices.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / 3.0
    identity = torch.eye(3, dtype=matrices.dtype, device=matrices.device)
    centered = matrices - mean_eigenvalue.unsqueeze(-1).unsqueeze(-1) * identity
    spread = torch.sqrt(centered.square().sum(dim=(-2, -1)) / 6.0)
    normalized = centered / spread.clamp_min(float(eps)).unsqueeze(-1).unsqueeze(-1)

    determinant = (
        normalized[:, 0, 0]
        * (
            normalized[:, 1, 1] * normalized[:, 2, 2]
            - normalized[:, 1, 2] * normalized[:, 2, 1]
        )
        - normalized[:, 0, 1]
        * (
            normalized[:, 1, 0] * normalized[:, 2, 2]
            - normalized[:, 1, 2] * normalized[:, 2, 0]
        )
        + normalized[:, 0, 2]
        * (
            normalized[:, 1, 0] * normalized[:, 2, 1]
            - normalized[:, 1, 1] * normalized[:, 2, 0]
        )
    )
    angle = torch.acos((0.5 * determinant).clamp(-1.0, 1.0)) / 3.0
    largest = mean_eigenvalue + 2.0 * spread * torch.cos(angle)
    smallest = mean_eigenvalue + 2.0 * spread * torch.cos(
        angle + 2.0 * math.pi / 3.0
    )
    middle = 3.0 * mean_eigenvalue - smallest - largest
    return torch.stack((smallest, middle, largest), dim=-1)


class RichPairwiseGeometry(nn.Module):
    """Encode ordered, frame-relative patch-pair geometry.

    The edge from group ``i`` to group ``j`` contains the center displacement in
    both endpoint frames, ``F_i.T @ F_j``, a radial basis, frame confidence, the
    local covariance spectrum at each endpoint, and an optional signed local
    chirality.  One consolidated stream handles parity-invariant inputs and a
    second shared stream handles all oriented inputs.  Only the oriented stream
    is evaluated on the reflected geometry in parity-invariant mode.

    ``parity_mode='sensitive'`` is invariant to proper rotations (SO(3)) but can
    distinguish mirror images.  ``parity_mode='invariant'`` averages every edge
    with its reflected local-coordinate counterpart and is O(3)-invariant under
    the right-handed frame convention used by :class:`RIMAEBackbone`.
    """

    def __init__(
        self,
        *,
        edge_dim: int,
        num_rbf: int,
        rbf_max_distance: float,
        parity_mode: str,
        use_signed_chirality: bool,
    ) -> None:
        super().__init__()
        self.edge_dim = int(edge_dim)
        self.num_rbf = int(num_rbf)
        self.rbf_max_distance = float(rbf_max_distance)
        self.parity_mode = str(parity_mode)
        self.use_signed_chirality = bool(use_signed_chirality)
        if self.edge_dim <= 0:
            raise ValueError(
                f"GeoFrameTransformerV2 edge_dim must be > 0, got {self.edge_dim}."
            )
        if self.num_rbf < 2:
            raise ValueError(
                f"GeoFrameTransformerV2 num_rbf must be >= 2, got {self.num_rbf}."
            )
        if self.rbf_max_distance <= 0.0:
            raise ValueError(
                "GeoFrameTransformerV2 rbf_max_distance must be > 0, "
                f"got {self.rbf_max_distance}."
            )
        if self.parity_mode not in {"invariant", "sensitive"}:
            raise ValueError(
                "GeoFrameTransformerV2 parity_mode must be 'invariant' or "
                f"'sensitive', got {self.parity_mode!r}."
            )

        self.register_buffer(
            "rbf_centers",
            torch.linspace(0.0, self.rbf_max_distance, self.num_rbf),
            persistent=True,
        )
        hidden_dim = max(32, self.edge_dim)

        def geometry_encoder(input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.edge_dim),
            )

        invariant_dim = self.num_rbf + 2 * _LOCAL_SHAPE_DIM + 3
        oriented_dim = 6 + 9 + (3 if self.use_signed_chirality else 0)
        self.invariant_encoder = geometry_encoder(invariant_dim)
        self.oriented_encoder = geometry_encoder(oriented_dim)
        self.output_norm = nn.LayerNorm(self.edge_dim)

    def _oriented_features(
        self,
        *,
        direction_i: torch.Tensor,
        direction_j: torch.Tensor,
        relative_frame: torch.Tensor,
        signed_chirality: torch.Tensor,
    ) -> torch.Tensor:
        features = [
            direction_i,
            direction_j,
            relative_frame.flatten(start_dim=-2),
        ]
        if self.use_signed_chirality:
            num_group = signed_chirality.shape[1]
            chirality_i = signed_chirality.unsqueeze(2)
            chirality_j = signed_chirality.unsqueeze(1)
            features.append(
                torch.stack(
                    (
                        chirality_i.expand(-1, -1, num_group),
                        chirality_j.expand(-1, num_group, -1),
                        chirality_i * chirality_j,
                    ),
                    dim=-1,
                )
            )
        return torch.cat(features, dim=-1)

    @staticmethod
    def _reflected_oriented_features(
        *,
        direction_i: torch.Tensor,
        direction_j: torch.Tensor,
        relative_frame: torch.Tensor,
        signed_chirality: torch.Tensor,
        use_signed_chirality: bool,
    ) -> torch.Tensor:
        sign = _parity_sign(device=direction_i.device, dtype=direction_i.dtype)
        reflected_features = [
            direction_i * sign,
            direction_j * sign,
            (
                relative_frame
                * sign.view(1, 1, 1, 3, 1)
                * sign.view(1, 1, 1, 1, 3)
            ).flatten(start_dim=-2),
        ]
        if use_signed_chirality:
            num_group = signed_chirality.shape[1]
            chirality_i = -signed_chirality.unsqueeze(2)
            chirality_j = -signed_chirality.unsqueeze(1)
            reflected_features.append(
                torch.stack(
                    (
                        chirality_i.expand(-1, -1, num_group),
                        chirality_j.expand(-1, num_group, -1),
                        chirality_i * chirality_j,
                    ),
                    dim=-1,
                )
            )
        return torch.cat(reflected_features, dim=-1)

    def forward(
        self,
        centers: torch.Tensor,
        frames: torch.Tensor,
        frame_confidence: torch.Tensor,
        node_shape: torch.Tensor,
        signed_chirality: torch.Tensor,
        *,
        return_state: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        centers_float = centers.float()
        frames_float = frames.float()
        confidence_float = frame_confidence.float()
        node_shape_float = node_shape.float()
        chirality_float = signed_chirality.float()

        # Ordered edge i -> j: c_j - c_i.
        delta = centers_float.unsqueeze(1) - centers_float.unsqueeze(2)
        distance = torch.linalg.vector_norm(delta, dim=-1)
        cloud_scale = torch.sqrt(
            centers_float.square().sum(dim=-1).mean(dim=1, keepdim=True)
        ).clamp_min(1.0e-6)
        normalized_distance = distance / cloud_scale.unsqueeze(-1)
        rbf_width = self.rbf_centers[1] - self.rbf_centers[0]
        rbf = torch.exp(
            -0.5
            * (
                (normalized_distance.unsqueeze(-1) - self.rbf_centers)
                / rbf_width
            ).square()
        )

        unit_delta = delta / distance.clamp_min(1.0e-6).unsqueeze(-1)
        direction_i = torch.einsum("bijc,bicd->bijd", unit_delta, frames_float)
        # At the target endpoint use the reverse displacement, c_i - c_j.
        direction_j = -torch.einsum("bijc,bjcd->bijd", unit_delta, frames_float)
        relative_frame = _relative_frame_products(frames_float)

        confidence_i = confidence_float.unsqueeze(2)
        confidence_j = confidence_float.unsqueeze(1)
        pair_confidence = torch.sqrt(
            (confidence_i * confidence_j).clamp_min(0.0)
        )
        confidence_features = torch.stack(
            (
                confidence_i.expand_as(pair_confidence),
                confidence_j.expand_as(pair_confidence),
                pair_confidence,
            ),
            dim=-1,
        )

        parameter_dtype = self.invariant_encoder[0].weight.dtype
        num_group = centers.shape[1]
        invariant_features = torch.cat(
            (
                rbf,
                node_shape_float.unsqueeze(2).expand(-1, -1, num_group, -1),
                node_shape_float.unsqueeze(1).expand(-1, num_group, -1, -1),
                confidence_features,
            ),
            dim=-1,
        ).to(dtype=parameter_dtype)
        invariant_embedding = self.invariant_encoder(invariant_features)

        direction_i_parameter = direction_i.to(dtype=parameter_dtype)
        direction_j_parameter = direction_j.to(dtype=parameter_dtype)
        relative_frame_parameter = relative_frame.to(dtype=parameter_dtype)
        chirality_parameter = chirality_float.to(dtype=parameter_dtype)
        oriented_features = self._oriented_features(
            direction_i=direction_i_parameter,
            direction_j=direction_j_parameter,
            relative_frame=relative_frame_parameter,
            signed_chirality=chirality_parameter,
        )
        if self.parity_mode == "invariant":
            reflected_features = self._reflected_oriented_features(
                direction_i=direction_i_parameter,
                direction_j=direction_j_parameter,
                relative_frame=relative_frame_parameter,
                signed_chirality=chirality_parameter,
                use_signed_chirality=self.use_signed_chirality,
            )
            oriented_batch = torch.cat((oriented_features, reflected_features), dim=0)
            oriented_original, oriented_reflected = self.oriented_encoder(
                oriented_batch
            ).chunk(2, dim=0)
            oriented_embedding = 0.5 * (oriented_original + oriented_reflected)
        else:
            oriented_embedding = self.oriented_encoder(oriented_features)

        edge = self.output_norm(invariant_embedding + oriented_embedding)
        if not return_state:
            return edge, {}
        state = {
            "distance": distance,
            "normalized_distance": normalized_distance,
            "direction_i": direction_i,
            "direction_j": direction_j,
            "relative_frame": relative_frame,
            "pair_confidence": pair_confidence,
            "node_shape": node_shape_float,
            "signed_chirality": chirality_float,
        }
        return edge, state


class GeometryConditionedAttentionBlock(nn.Module):
    """Self-attention with edge-conditioned logits and values.

    Rich ordered edges are projected to a shared per-head attention bias once
    before the transformer.  Each layer learns only a per-head scale and a
    meaningful self-edge offset.  Pair edges ending at value token ``j`` are
    likewise summarized once before producing layer-specific value gates and
    low-rank geometric vectors.  This keeps values pair-geometry-conditioned
    while retaining fused scaled-dot-product attention.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        edge_dim: int,
        edge_value_rank: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.edge_value_rank = int(edge_value_rank)
        if self.num_heads <= 0 or self.embed_dim % self.num_heads != 0:
            raise ValueError(
                "GeoFrameTransformerV2 attention requires embed_dim divisible by "
                f"num_heads, got embed_dim={self.embed_dim}, num_heads={self.num_heads}."
            )
        if self.edge_value_rank <= 0:
            raise ValueError(
                "GeoFrameTransformerV2 edge_value_rank must be > 0, "
                f"got {self.edge_value_rank}."
            )
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = 1.0 / math.sqrt(float(self.head_dim))
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.edge_bias_scale = nn.Parameter(torch.ones(self.num_heads))
        self.self_edge_bias = nn.Parameter(torch.zeros(self.num_heads))
        self.edge_value_gate = nn.Linear(int(edge_dim), self.num_heads)
        self.edge_value_coefficients = nn.Linear(
            int(edge_dim), self.edge_value_rank
        )
        self.edge_value_basis = nn.Parameter(
            torch.empty(self.num_heads, self.edge_value_rank, self.head_dim)
        )
        nn.init.trunc_normal_(self.edge_value_basis, std=0.02)
        self.attention_dropout = nn.Dropout(float(dropout))
        self.output_projection = nn.Linear(self.embed_dim, self.embed_dim)
        self.output_dropout = nn.Dropout(float(dropout))
        hidden_dim = int(float(mlp_ratio) * self.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.embed_dim),
            nn.Dropout(float(dropout)),
        )

    def geometry_modulation(
        self,
        shared_attention_bias: torch.Tensor,
        value_geometry: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_group = shared_attention_bias.shape[-1]
        self_edge = torch.eye(
            num_group,
            device=shared_attention_bias.device,
            dtype=shared_attention_bias.dtype,
        ).view(1, 1, num_group, num_group)
        logit_bias = (
            shared_attention_bias * self.edge_bias_scale.view(1, -1, 1, 1)
            + self_edge * self.self_edge_bias.view(1, -1, 1, 1)
        )
        value_gate = self.edge_value_gate(value_geometry).permute(0, 2, 1)
        value_coefficients = self.edge_value_coefficients(value_geometry)
        return logit_bias, value_gate, value_coefficients

    def _attention(
        self,
        tokens: torch.Tensor,
        shared_attention_bias: torch.Tensor,
        value_geometry: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_group, _ = tokens.shape
        qkv = self.qkv(tokens).reshape(
            batch_size,
            num_group,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)
        logit_bias, value_gate, value_coefficients = self.geometry_modulation(
            shared_attention_bias,
            value_geometry,
        )
        geometry_value = torch.einsum(
            "bjr,hrd->bhjd",
            value_coefficients.to(dtype=value.dtype),
            self.edge_value_basis.to(dtype=value.dtype),
        )
        # A factor in (0, 2) keeps the initial scale close to ordinary values
        # while allowing incoming pair geometry to suppress or amplify V_j.
        conditioned_value = (
            value
            * (2.0 * torch.sigmoid(value_gate.to(dtype=value.dtype))).unsqueeze(-1)
            + geometry_value
        )
        dropout_p = float(self.attention_dropout.p) if self.training else 0.0
        output = F.scaled_dot_product_attention(
            query,
            key,
            conditioned_value,
            attn_mask=logit_bias.to(dtype=query.dtype),
            dropout_p=dropout_p,
            scale=self.scale,
        ).transpose(1, 2).reshape(
            batch_size, num_group, self.embed_dim
        )
        return self.output_dropout(self.output_projection(output))

    def forward(
        self,
        tokens: torch.Tensor,
        shared_attention_bias: torch.Tensor,
        value_geometry: torch.Tensor,
    ) -> torch.Tensor:
        tokens = tokens + self._attention(
            self.norm1(tokens), shared_attention_bias, value_geometry
        )
        return tokens + self.mlp(self.norm2(tokens))


class GeometryConditionedTransformer(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        depth: int,
        edge_dim: int,
        edge_value_rank: int,
        mlp_ratio: float,
        dropout: float,
        use_gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GeometryConditionedAttentionBlock(
                    embed_dim=int(embed_dim),
                    num_heads=int(num_heads),
                    edge_dim=int(edge_dim),
                    edge_value_rank=int(edge_value_rank),
                    mlp_ratio=float(mlp_ratio),
                    dropout=float(dropout),
                )
                for _ in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(int(embed_dim))
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

    def forward(
        self,
        tokens: torch.Tensor,
        shared_attention_bias: torch.Tensor,
        value_geometry: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_gradient_checkpointing and self.training and tokens.requires_grad:
            for layer in self.layers:
                tokens = torch.utils.checkpoint.checkpoint(
                    layer,
                    tokens,
                    shared_attention_bias,
                    value_geometry,
                    use_reentrant=False,
                )
        else:
            for layer in self.layers:
                tokens = layer(tokens, shared_attention_bias, value_geometry)
        return self.norm(tokens)


class GeoFrameTokenEncoderV2(nn.Module):
    """Multi-scale local tokens coupled by rich pairwise geometry."""

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
        num_rbf: int,
        rbf_max_distance: float,
        edge_dim: int,
        edge_value_rank: int,
        parity_mode: str,
        use_signed_chirality: bool,
        use_gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.num_group = int(num_group)
        self.patch_sizes = tuple(int(size) for size in patch_sizes)
        self.trans_dim = int(trans_dim)
        self.frame_builder = str(frame_builder)
        self.frame_eps = float(frame_eps)
        self.use_frame_gating = bool(use_frame_gating)
        self.frame_confidence_floor = float(frame_confidence_floor)
        self.parity_mode = str(parity_mode)
        if not self.patch_sizes or any(size < 3 for size in self.patch_sizes):
            raise ValueError(
                "GeoFrameTransformerV2 patch_sizes must contain integers >= 3, "
                f"got {self.patch_sizes}."
            )
        if tuple(sorted(set(self.patch_sizes))) != self.patch_sizes:
            raise ValueError(
                "GeoFrameTransformerV2 patch_sizes must be strictly increasing, "
                f"got {self.patch_sizes}."
            )
        if self.frame_builder not in {"triad", "pca"}:
            raise ValueError(
                "GeoFrameTransformerV2 frame_builder must be 'triad' or 'pca', "
                f"got {self.frame_builder!r}."
            )
        if self.frame_eps <= 0.0:
            raise ValueError(
                f"GeoFrameTransformerV2 frame_eps must be > 0, got {self.frame_eps}."
            )
        if not (0.0 <= self.frame_confidence_floor <= 1.0):
            raise ValueError(
                "GeoFrameTransformerV2 frame_confidence_floor must be in [0, 1], "
                f"got {self.frame_confidence_floor}."
            )
        if self.parity_mode not in {"invariant", "sensitive"}:
            raise ValueError(
                "GeoFrameTransformerV2 parity_mode must be 'invariant' or "
                f"'sensitive', got {self.parity_mode!r}."
            )

        self.group_divider = Group(
            num_group=self.num_group,
            group_size=max(self.patch_sizes),
            deterministic_fps=bool(deterministic_fps),
            sorting_mode=sorting_mode,
            group_sampling=group_sampling,
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
            n_dim=3, d_model=self.trans_dim, scale=1.0
        )
        self.radial_position = (
            nn.Sequential(
                nn.Linear(1, self.trans_dim),
                nn.GELU(),
                nn.Linear(self.trans_dim, self.trans_dim),
            )
            if self.use_frame_gating
            else None
        )
        self.pair_geometry = RichPairwiseGeometry(
            edge_dim=int(edge_dim),
            num_rbf=int(num_rbf),
            rbf_max_distance=float(rbf_max_distance),
            parity_mode=self.parity_mode,
            use_signed_chirality=bool(use_signed_chirality),
        )
        self.shared_edge_logit = nn.Linear(int(edge_dim), int(num_heads))
        self.transformer = GeometryConditionedTransformer(
            embed_dim=self.trans_dim,
            num_heads=int(num_heads),
            depth=int(depth),
            edge_dim=int(edge_dim),
            edge_value_rank=int(edge_value_rank),
            mlp_ratio=float(mlp_ratio),
            dropout=float(dropout),
            use_gradient_checkpointing=bool(use_gradient_checkpointing),
        )

    def group_points(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.group_divider(points)

    def _frames_shape_and_chirality(
        self,
        neighborhood: torch.Tensor,
        centers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad(), torch.autocast(
            device_type=neighborhood.device.type, enabled=False
        ):
            neighborhood_float = neighborhood.float()
            frames = RIMAEBackbone._estimate_patch_frames(
                neighborhood_float,
                frame_builder=self.frame_builder,
                frame_eps=self.frame_eps,
            )
            batch_size, num_group, group_size, _ = neighborhood.shape
            patches = neighborhood_float.reshape(
                batch_size * num_group, group_size, 3
            )
            covariance = patches.transpose(1, 2) @ patches / float(group_size)
            eigenvalues = _symmetric_3x3_eigenvalues(
                covariance,
                eps=self.frame_eps,
            )
            eigenvalues = eigenvalues.clamp_min(0.0)
            trace = eigenvalues.sum(dim=-1, keepdim=True).clamp_min(self.frame_eps)
            normalized_eigenvalues = eigenvalues / trace
            low, middle, high = eigenvalues.unbind(dim=-1)
            high_safe = high.clamp_min(self.frame_eps)
            primary_gap = ((high - middle) / high_safe).clamp(0.0, 1.0)
            secondary_gap = ((middle - low) / high_safe).clamp(0.0, 1.0)
            confidence = torch.sqrt(
                (primary_gap * secondary_gap).clamp_min(0.0)
            )

            radius_squared = patches.square().sum(dim=-1)
            rms_radius = torch.sqrt(radius_squared.mean(dim=-1).clamp_min(self.frame_eps))
            centers_float = centers.float()
            cloud_scale = torch.sqrt(
                centers_float.square().sum(dim=-1).mean(dim=1, keepdim=True)
            ).clamp_min(self.frame_eps)
            normalized_radius = rms_radius.reshape(batch_size, num_group) / cloud_scale
            normalized_density = float(group_size) / (
                (4.0 * math.pi / 3.0) * normalized_radius.pow(3).clamp_min(self.frame_eps)
            )
            node_shape = torch.cat(
                (
                    normalized_eigenvalues,
                    primary_gap.unsqueeze(-1),
                    secondary_gap.unsqueeze(-1),
                    normalized_radius.reshape(-1, 1),
                    torch.log1p(normalized_density).reshape(-1, 1),
                    confidence.unsqueeze(-1),
                ),
                dim=-1,
            ).reshape(batch_size, num_group, _LOCAL_SHAPE_DIM)

            normalized_patches = patches / rms_radius.clamp_min(
                self.frame_eps
            ).view(-1, 1, 1)
            normalized_radius_squared = normalized_patches.square().sum(dim=-1)
            moment_1 = (
                normalized_patches * normalized_radius_squared.unsqueeze(-1)
            ).mean(dim=1)
            moment_2 = (
                normalized_patches * normalized_radius_squared.square().unsqueeze(-1)
            ).mean(dim=1)
            moment_3 = (
                normalized_patches
                * normalized_radius_squared.pow(3).unsqueeze(-1)
            ).mean(dim=1)
            triple_product = (
                moment_1 * torch.cross(moment_2, moment_3, dim=-1)
            ).sum(dim=-1)
            moment_scale = (
                torch.linalg.vector_norm(moment_1, dim=-1)
                * torch.linalg.vector_norm(moment_2, dim=-1)
                * torch.linalg.vector_norm(moment_3, dim=-1)
            ).clamp_min(self.frame_eps)
            signed_chirality = (triple_product / moment_scale).clamp(-1.0, 1.0)

        output_dtype = neighborhood.dtype
        return (
            frames.to(device=neighborhood.device, dtype=output_dtype),
            confidence.reshape(batch_size, num_group).to(
                device=neighborhood.device, dtype=output_dtype
            ),
            node_shape.to(device=neighborhood.device, dtype=output_dtype),
            signed_chirality.reshape(batch_size, num_group).to(
                device=neighborhood.device, dtype=output_dtype
            ),
        )

    def _canonical_patch_token(
        self,
        *,
        scale_index: int,
        canonical_patch: torch.Tensor,
    ) -> torch.Tensor:
        canonical_features = self.patch_encoders[scale_index](canonical_patch)
        if self.parity_mode == "invariant":
            sign = _parity_sign(
                device=canonical_patch.device, dtype=canonical_patch.dtype
            )
            reflected_features = self.patch_encoders[scale_index](
                canonical_patch * sign
            )
            canonical_features = 0.5 * (canonical_features + reflected_features)
        return self.patch_projections[scale_index](canonical_features)

    def _prepare_tokens(
        self,
        neighborhood: torch.Tensor,
        centers: torch.Tensor,
        *,
        return_state: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        frames, confidence, node_shape, signed_chirality = (
            self._frames_shape_and_chirality(neighborhood, centers)
        )
        canonical_neighborhood = torch.einsum(
            "bgsc,bgcd->bgsd", neighborhood, frames
        )
        frame_weight = self.frame_confidence_floor + (
            1.0 - self.frame_confidence_floor
        ) * confidence
        frame_weight_token = frame_weight.unsqueeze(-1)

        scale_tokens = []
        for scale_index, patch_size in enumerate(self.patch_sizes):
            canonical_patch = canonical_neighborhood[:, :, :patch_size]
            canonical_token = self._canonical_patch_token(
                scale_index=scale_index,
                canonical_patch=canonical_patch,
            )
            if self.use_frame_gating:
                invariant_token = self.invariant_patch_encoders[scale_index](
                    neighborhood[:, :, :patch_size]
                )
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
        if self.parity_mode == "invariant":
            sign = _parity_sign(device=centers.device, dtype=centers.dtype)
            reflected_position = self.pos_embed(local_centers * sign)
            frame_position = 0.5 * (frame_position + reflected_position)
        radial_position = None
        if self.use_frame_gating:
            center_radius = torch.linalg.vector_norm(
                centers.float(), dim=-1, keepdim=True
            ).to(dtype=centers.dtype)
            radial_position = self.radial_position(center_radius)
        if self.use_frame_gating:
            position_tokens = (
                frame_weight_token * frame_position
                + (1.0 - frame_weight_token) * radial_position
            )
        else:
            position_tokens = frame_position

        edge_embedding, geometry_state = self.pair_geometry(
            centers,
            frames,
            confidence,
            node_shape,
            signed_chirality,
            return_state=return_state,
        )
        shared_attention_bias, value_geometry = self.project_pairwise_geometry(
            edge_embedding
        )
        encoder_input = patch_tokens + position_tokens
        if not return_state:
            return encoder_input, shared_attention_bias, value_geometry, {}
        state = {
            "centers": centers,
            "frames": frames,
            "frame_confidence": confidence,
            "confidence": confidence,
            "frame_weight": frame_weight,
            "shape_descriptor": node_shape,
            "signed_chirality": signed_chirality,
            "position_tokens": position_tokens,
            "edge_embedding": edge_embedding,
            "shared_attention_bias": shared_attention_bias,
            "value_geometry": value_geometry,
        }
        if radial_position is not None:
            state["radial_position_tokens"] = radial_position
        for name, value in geometry_state.items():
            state[f"geometry_{name}"] = value
        return encoder_input, shared_attention_bias, value_geometry, state

    def prepare_tokens(
        self,
        neighborhood: torch.Tensor,
        centers: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        return self._prepare_tokens(neighborhood, centers, return_state=True)

    def project_pairwise_geometry(
        self,
        edge_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared_attention_bias = self.shared_edge_logit(edge_embedding).permute(
            0, 3, 1, 2
        )
        # edge_embedding[:, i, j] is the ordered edge i -> j.  Incoming edges
        # are summarized once per destination/value token j for all layers.
        value_geometry = edge_embedding.mean(dim=1)
        return shared_attention_bias, value_geometry

    def build_pairwise_geometry(
        self,
        centers: torch.Tensor,
        frames: torch.Tensor,
        frame_confidence: torch.Tensor,
        node_shape: torch.Tensor,
        signed_chirality: torch.Tensor,
        *,
        return_state: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.pair_geometry(
            centers,
            frames,
            frame_confidence,
            node_shape,
            signed_chirality,
            return_state=return_state,
        )

    def encode_grouped(
        self,
        neighborhood: torch.Tensor,
        centers: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoder_input, shared_attention_bias, value_geometry, state = (
            self.prepare_tokens(neighborhood, centers)
        )
        tokens = self.transformer(
            encoder_input, shared_attention_bias, value_geometry
        )
        state["tokens"] = tokens
        return tokens, state

    def encode_grouped_features(
        self,
        neighborhood: torch.Tensor,
        centers: torch.Tensor,
    ) -> torch.Tensor:
        encoder_input, shared_attention_bias, value_geometry, _ = self._prepare_tokens(
            neighborhood, centers, return_state=False
        )
        return self.transformer(
            encoder_input, shared_attention_bias, value_geometry
        )


@register_encoder("GeoFrameTransformerV2")
class GeoFrameTransformerV2Encoder(Encoder):
    """VICReg GeoFrame encoder with rich, parity-explicit edge messages."""

    output_contract = "invariant_aux"

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
        use_frame_gating: bool = False,
        frame_confidence_floor: float = 0.25,
        num_rbf: int = 16,
        rbf_max_distance: float = 3.0,
        edge_dim: int = 32,
        edge_value_rank: int = 4,
        parity_mode: str = "invariant",
        use_signed_chirality: bool = True,
        pool_queries: int = 2,
        pooling_mode: str = "max_mean",
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.latent_size = int(latent_size)
        self.invariant_dim = self.latent_size
        self.output_dim = self.latent_size
        self.trans_dim = int(trans_dim)
        self.center_input = bool(center_input)
        self.pooling_mode = str(pooling_mode)
        self.parity_mode = str(parity_mode)
        if self.latent_size <= 0:
            raise ValueError(
                f"GeoFrameTransformerV2 latent_size must be > 0, got {self.latent_size}."
            )
        if self.trans_dim <= 0 or self.trans_dim % int(num_heads) != 0:
            raise ValueError(
                "GeoFrameTransformerV2 trans_dim must be positive and divisible by "
                f"num_heads, got trans_dim={self.trans_dim}, num_heads={int(num_heads)}."
            )
        if self.pooling_mode not in {"attention", "max_mean"}:
            raise ValueError(
                "GeoFrameTransformerV2 pooling_mode must be 'attention' or "
                f"'max_mean', got {self.pooling_mode!r}."
            )
        if self.pooling_mode == "max_mean" and self.latent_size != self.trans_dim:
            raise ValueError(
                "GeoFrameTransformerV2 pooling_mode='max_mean' requires "
                f"latent_size == trans_dim, got {self.latent_size} and {self.trans_dim}."
            )

        self.token_encoder = GeoFrameTokenEncoderV2(
            num_group=int(num_group),
            patch_sizes=tuple(int(size) for size in patch_sizes),
            encoder_dims=int(encoder_dims),
            trans_dim=self.trans_dim,
            depth=int(depth),
            num_heads=int(num_heads),
            mlp_ratio=float(mlp_ratio),
            dropout=float(dropout),
            deterministic_fps=bool(deterministic_fps),
            sorting_mode=sorting_mode,
            group_sampling=group_sampling,
            frame_builder=frame_builder,
            frame_eps=float(frame_eps),
            use_frame_gating=bool(use_frame_gating),
            frame_confidence_floor=float(frame_confidence_floor),
            num_rbf=int(num_rbf),
            rbf_max_distance=float(rbf_max_distance),
            edge_dim=int(edge_dim),
            edge_value_rank=int(edge_value_rank),
            parity_mode=self.parity_mode,
            use_signed_chirality=bool(use_signed_chirality),
            use_gradient_checkpointing=bool(use_gradient_checkpointing),
        )
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

    def _center_points(self, points: torch.Tensor) -> torch.Tensor:
        points = _to_bn3(points)
        if self.center_input:
            points = points - points.mean(dim=1, keepdim=True)
        return points

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.pool is None:
            return 0.5 * (tokens.max(dim=1).values + tokens.mean(dim=1))
        return self.pool(tokens)

    def forward_features(self, points: torch.Tensor) -> torch.Tensor:
        centered = self._center_points(points)
        neighborhood, centers = self.token_encoder.group_points(centered)
        tokens = self.token_encoder.encode_grouped_features(neighborhood, centers)
        return self._pool_tokens(tokens)

    def forward_with_state(
        self,
        points: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        centered = self._center_points(points)
        neighborhood, centers = self.token_encoder.group_points(centered)
        tokens, state = self.token_encoder.encode_grouped(neighborhood, centers)
        return self._pool_tokens(tokens), state

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.forward_features(points), {}


__all__ = [
    "GeoFrameTransformerV2Encoder",
    "GeoFrameTokenEncoderV2",
    "GeometryConditionedAttentionBlock",
    "LOCAL_SHAPE_FEATURE_NAMES",
    "RichPairwiseGeometry",
]
