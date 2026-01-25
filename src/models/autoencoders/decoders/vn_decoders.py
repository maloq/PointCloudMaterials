from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.init as init

from ..base import Decoder
from ..registry import register_decoder
from ..encoders.vn_encoders import VNLinear, VNLinearLeakyReLU, VNBatchNorm, VNResBlock


@register_decoder("VN_Equivariant")
class VNEquivariantDecoder(Decoder):
    use_invariant_latent = False
    """
    Vector Neuron Equivariant Decoder.

    Takes equivariant latent representation (B, latent_size, 3) and generates
    a point cloud (B, num_points, 3) while preserving SO(3) equivariance.

    Architecture:
    - Progressively expands VN channels through multiple layers
    - Uses VN layers to maintain equivariance throughout
    - Final layer projects to individual point coordinates
    """

    def __init__(
        self,
        num_points: int,
        latent_size: int,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        use_batchnorm: bool = True,
        negative_slope: float = 0.1,
        output_scale: float = 1.0,  # Scale factor for output points
        learnable_scale: bool = True,  # NEW: learn the output scale
        num_res_blocks: int = 0,  # Optional VN residual blocks for more capacity
        center_output: bool = False,  # Center output points around origin
    ):
        super().__init__()
        self._n = num_points
        self.output_scale = output_scale
        self.center_output = center_output

        # Input eq_z has shape (B, latent_size, 3) where latent_size is the number of VN channels
        hidden_dims = tuple(hidden_dims)
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one entry")
        c_in = latent_size
        hidden_channels = [max(1, dim // 3) for dim in hidden_dims]

        # Expansion layers with VN
        layers = []
        for h in hidden_channels:
            layers.append(
                VNLinearLeakyReLU(
                    c_in,
                    h,
                    dim=3,
                    negative_slope=negative_slope,
                    use_batchnorm=use_batchnorm,
                )
            )
            c_in = h
        self.vn_layers = nn.Sequential(*layers)

        self.res_blocks = None
        if num_res_blocks > 0:
            self.res_blocks = nn.Sequential(
                *[
                    VNResBlock(
                        c_in,
                        dim=3,
                        negative_slope=negative_slope,
                        use_batchnorm=use_batchnorm,
                    )
                    for _ in range(num_res_blocks)
                ]
            )

        # Final projection to num_points
        # Each VN channel will produce one 3D point
        self.vn_final = VNLinear(c_in, num_points)
        
        # Better initialization for the final layer - increase gain significantly
        # Standard kaiming init leads to very small outputs in VN networks
        nn.init.xavier_uniform_(self.vn_final.map_to_feat.weight, gain=2.0)
        
        # Learnable output scale - starts at 1.0, can learn to amplify output
        if learnable_scale:
            self.scale_param = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('scale_param', torch.tensor(1.0))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Equivariant latent (B, latent_size, 3) where latent_size is the number of VN channels

        Returns:
            Point cloud (B, num_points, 3)
        """
        # z shape: (B, latent_size, 3) - latent_size VN channels, each being a 3D vector
        # Progressively expand through VN layers
        x = self.vn_layers(z)
        if self.res_blocks is not None:
            x = self.res_blocks(x)

        # Final projection to num_points
        x = self.vn_final(x)  # (B, num_points, 3)
        
        # Apply output scaling (fixed + learnable)
        x = x * self.output_scale * self.scale_param

        if self.center_output:
            x = x - x.mean(dim=1, keepdim=True)

        return x



# ---------------------------------------------------------------------------
# REVNET-inspired anchor + VN-transformer + invariant fine decoder
# (includes an optional invariant->equivariant anchor-position head for stability)
# ---------------------------------------------------------------------------

_EPS = 1e-6


class VNZCALayerNorm(nn.Module):
    """ZCA-whitening LayerNorm for VN features (B, C, 3, ...)."""

    def __init__(self, channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        else:
            self.register_buffer("gamma", torch.ones(1, channels, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 3:
            raise ValueError(f"Expected VN tensor with >=3 dims, got shape {tuple(x.shape)}")
        B, C = x.shape[:2]
        if x.shape[2] != 3:
            raise ValueError(f"VNZCALayerNorm expects vector-dim=3, got {x.shape[2]}")

        x_flat = x.reshape(B, C, 3, -1).permute(0, 2, 1, 3).reshape(B, 3, -1)
        # Whitening in float32 avoids bf16 eigh/rsqrt issues and improves stability.
        x_flat_f = x_flat.to(torch.float32)
        mu = x_flat_f.mean(dim=-1, keepdim=True)
        xc = x_flat_f - mu

        M = xc.shape[-1]
        cov = (xc @ xc.transpose(1, 2)) / (float(M) + _EPS)
        cov = cov + self.eps * torch.eye(3, device=x.device, dtype=xc.dtype).unsqueeze(0)

        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = eigvals.clamp_min(self.eps)
        inv_sqrt = torch.rsqrt(eigvals)
        W = eigvecs @ torch.diag_embed(inv_sqrt) @ eigvecs.transpose(1, 2)
        xw = W @ xc

        xw = xw.reshape(B, 3, C, -1).permute(0, 2, 1, 3).reshape_as(x).to(x.dtype)
        gamma = self.gamma
        while gamma.dim() < xw.dim():
            gamma = gamma.unsqueeze(-1)
        return xw * gamma.to(xw.dtype)


class VNChannelWiseSubtractionAttention(nn.Module):
    """Channel-wise subtraction attention (CWSA) over anchor tokens."""

    def __init__(self, channels: int, hidden: int = 64, use_pos: bool = True):
        super().__init__()
        self.channels = channels
        self.use_pos = use_pos
        self.to_q = VNLinear(channels, channels)
        self.to_k = VNLinear(channels, channels)
        self.to_v = VNLinear(channels, channels)
        self.to_out = VNLinear(channels, channels)

        in_mlp = 2 if use_pos else 1
        self.score_mlp = nn.Sequential(
            nn.Linear(in_mlp, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, C, 3, K), pos: (B, K, 3)
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        qk = q.permute(0, 3, 1, 2)  # (B,K,C,3)
        kk = k.permute(0, 3, 1, 2)
        vv = v.permute(0, 3, 1, 2)

        rel = qk[:, :, None, :, :] - kk[:, None, :, :, :]     # (B,Kq,Kk,C,3)
        rel_norm = torch.norm(rel, dim=-1)                    # (B,Kq,Kk,C)
        rel_mean = rel_norm.mean(dim=-1, keepdim=True)        # (B,Kq,Kk,1)

        if self.use_pos:
            if pos is None:
                raise ValueError("pos must be provided when use_pos=True")
            rel_pos = pos[:, :, None, :] - pos[:, None, :, :]  # (B,Kq,Kk,3)
            rel_pos_norm = torch.norm(rel_pos, dim=-1, keepdim=True)
            score_in = torch.cat([rel_mean, rel_pos_norm], dim=-1)
        else:
            score_in = rel_mean

        scores = self.score_mlp(score_in)                     # (B,Kq,Kk,C)
        attn = torch.softmax(scores, dim=2)

        out = torch.einsum("bijk,bjkd->bikd", attn, vv)        # (B,K,C,3)
        out = out.permute(0, 2, 3, 1).contiguous()             # (B,C,3,K)
        return self.to_out(out)


class VNAnchorTransformerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        mlp_ratio: int = 2,
        attn_hidden: int = 64,
        use_zca_norm: bool = True,
        use_pos: bool = True,
        negative_slope: float = 0.1,
        use_batchnorm: bool = False,
    ):
        super().__init__()
        self.attn = VNChannelWiseSubtractionAttention(channels, hidden=attn_hidden, use_pos=use_pos)
        self.norm1 = VNZCALayerNorm(channels) if use_zca_norm else VNBatchNorm(channels, dim=4)
        self.ffn = nn.Sequential(
            VNLinearLeakyReLU(
                channels,
                channels * mlp_ratio,
                dim=4,
                negative_slope=negative_slope,
                use_batchnorm=use_batchnorm,
            ),
            VNLinear(channels * mlp_ratio, channels),
        )
        self.norm2 = VNZCALayerNorm(channels) if use_zca_norm else VNBatchNorm(channels, dim=4)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attn(x, pos))
        x = self.norm2(x + self.ffn(x))
        return x


@register_decoder("VN_REVNET_Anchor")
class VNRevnetAnchorDecoder(Decoder):
    use_invariant_latent = False

    def __init__(
        self,
        num_points: int,
        latent_size: int,
        num_anchors: int = 64,
        anchor_channels: int = 48,
        transformer_depth: int = 2,
        attn_hidden: int = 128,
        mlp_ratio: int = 2,
        point_query_dim: int = 32,
        offset_mlp_hidden: int = 256,
        offset_scale: float = 0.25,
        output_scale: float = 1.0,
        learnable_scale: bool = True,
        center_output: bool = True,
        use_zca_norm: bool = True,
        use_invariant_anchor_pos: bool = True,
        pos_mlp_hidden: int = 256,
    ):
        super().__init__()
        self._n = num_points
        self.num_anchors = num_anchors
        self.anchor_channels = anchor_channels
        self.offset_scale = offset_scale
        self.output_scale = output_scale
        self.center_output = center_output
        self.use_invariant_anchor_pos = use_invariant_anchor_pos

        # NOTE: import VNStdFeature locally to avoid changing your existing top-level imports
        from ..encoders.vn_encoders import VNStdFeature
        self._VNStdFeature = VNStdFeature

        if use_invariant_anchor_pos:
            # Invariant->equivariant anchor position head (REVNET idea):
            # build a canonical frame from z, predict anchors in that frame, rotate back.
            self.z_frame = self._VNStdFeature(
                in_channels=latent_size,
                dim=3,
                normalize_frame=True,
                negative_slope=0.0,
                use_batchnorm=False,
            )
            self.pos_mlp = nn.Sequential(
                nn.Linear(latent_size * 3, pos_mlp_hidden),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(pos_mlp_hidden, pos_mlp_hidden),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(pos_mlp_hidden, num_anchors * 3),
            )
        else:
            # Purely equivariant linear projection to anchor positions
            self.anchor_pos = VNLinear(latent_size, num_anchors)
            nn.init.xavier_uniform_(self.anchor_pos.map_to_feat.weight, gain=2.0)

        # Anchor feature projection (equivariant)
        self.anchor_feat = VNLinearLeakyReLU(
            latent_size,
            num_anchors * anchor_channels,
            dim=3,
            negative_slope=0.1,
            use_batchnorm=False,
        )

        self.blocks = nn.ModuleList(
            [
                VNAnchorTransformerBlock(
                    channels=anchor_channels,
                    mlp_ratio=mlp_ratio,
                    attn_hidden=attn_hidden,
                    use_zca_norm=use_zca_norm,
                    use_pos=True,
                )
                for _ in range(transformer_depth)
            ]
        )

        # Canonicalization per-anchor to feed an invariant MLP for local offsets
        self.anchor_frame = self._VNStdFeature(
            in_channels=anchor_channels * 2,
            dim=4,
            normalize_frame=True,
            negative_slope=0.0,
            use_batchnorm=False,
        )

        self.points_per_anchor = int(math.ceil(num_points / float(num_anchors)))
        self.total_points = self.points_per_anchor * num_anchors
        self.point_queries = nn.Parameter(torch.randn(self.points_per_anchor, point_query_dim) * 0.02)

        inv_dim = anchor_channels * 3
        self.offset_mlp = nn.Sequential(
            nn.Linear(inv_dim + point_query_dim, offset_mlp_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(offset_mlp_hidden, offset_mlp_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(offset_mlp_hidden, 3),
        )

        if learnable_scale:
            self.scale_param = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("scale_param", torch.tensor(1.0))

    def _predict_anchor_pos(self, z: torch.Tensor) -> torch.Tensor:
        if not self.use_invariant_anchor_pos:
            return self.anchor_pos(z)  # (B,K,3)

        z_std, R = self.z_frame(z)                     # z_std: (B,C,3), R: (B,3,3)
        inv = z_std.reshape(z.shape[0], -1)            # (B, C*3) canonical -> rotation-invariant
        pos_local = self.pos_mlp(inv).view(z.shape[0], self.num_anchors, 3)  # (B,K,3)
        pos_global = torch.einsum("bki,bij->bkj", pos_local, R.transpose(1, 2))
        return pos_global

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]

        pos = self._predict_anchor_pos(z)  # (B,K,3)

        feat_flat = self.anchor_feat(z)  # (B,K*C,3)
        feat = feat_flat.view(B, self.num_anchors, self.anchor_channels, 3).permute(0, 2, 3, 1).contiguous()

        for blk in self.blocks:
            feat = blk(feat, pos)

        feat_mean = feat.mean(dim=-1, keepdim=True)
        feat_cat = torch.cat([feat, feat_mean.expand_as(feat)], dim=1)  # (B,2C,3,K)
        feat_std, R = self.anchor_frame(feat_cat)                       # R: (B,3,3,K)

        inv = feat_std[:, : self.anchor_channels]                        # (B,C,3,K)
        inv = inv.permute(0, 3, 1, 2).reshape(B, self.num_anchors, -1)  # (B,K,C*3)

        q = self.point_queries.unsqueeze(0).unsqueeze(0).expand(B, self.num_anchors, -1, -1)
        inv_exp = inv.unsqueeze(2).expand(B, self.num_anchors, self.points_per_anchor, inv.shape[-1])
        offsets_local = self.offset_mlp(torch.cat([inv_exp, q], dim=-1)) * self.offset_scale

        Rt = R.permute(0, 3, 2, 1).contiguous()                          # (B,K,3,3) canonical->global
        offsets_global = torch.einsum("bkpi,bkij->bkpj", offsets_local, Rt)

        points = (pos.unsqueeze(2) + offsets_global).reshape(B, self.total_points, 3)
        if self.total_points != self._n:
            points = points[:, : self._n]

        points = points * self.output_scale * self.scale_param
        if self.center_output:
            points = points - points.mean(dim=1, keepdim=True)
        return points


__all__ = [
    "VNEquivariantDecoder",
    "VNRevnetAnchorDecoder",
]
