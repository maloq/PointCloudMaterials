import torch
import torch.nn as nn


class RIMAEInvariantEncoderForContrastive(nn.Module):
    """
    Adapter that exposes RI-MAE invariant transformer features through the
    contrastive encoder interface.

    Output format matches existing encoders:
    (inv_latent_net, inv_latent_net, eq_z=None)
    """

    expects_channel_first = False

    def __init__(
        self,
        backbone,
        *,
        center_input: bool,
        output_dim: int,
    ) -> None:
        super().__init__()
        required = (
            "group_divider",
            "patch_encoder",
            "input_proj",
            "pos_embed",
            "orientation_mlp",
            "orientation_scale",
            "student_encoder",
        )
        missing = [name for name in required if not hasattr(backbone, name)]
        if missing:
            raise TypeError(
                "backbone is missing required RI-MAE encoder-path attributes: "
                f"{missing}. Got type={type(backbone)}."
            )

        # Keep only the RI encoder path modules; MAE predictor/teacher branches
        # are intentionally excluded for VICReg use.
        self.group_divider = backbone.group_divider
        self.patch_encoder = backbone.patch_encoder
        self.input_proj = backbone.input_proj
        self.pos_embed = backbone.pos_embed
        self.orientation_mlp = backbone.orientation_mlp
        self.orientation_scale = backbone.orientation_scale
        self.student_encoder = backbone.student_encoder
        estimate_patch_frames = getattr(type(backbone), "_estimate_patch_frames", None)
        if not callable(estimate_patch_frames):
            raise TypeError(
                "backbone type must provide callable static method _estimate_patch_frames(neighborhood). "
                f"Got type={type(backbone)}."
            )
        self._estimate_patch_frames = estimate_patch_frames

        self.center_input = bool(center_input)
        self.output_dim = int(output_dim)
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be > 0, got {self.output_dim}")

    @staticmethod
    def _to_bn3(points: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(points):
            raise TypeError(f"Expected points to be torch.Tensor, got {type(points)}")
        if points.dim() != 3:
            raise ValueError(
                f"Expected point cloud with shape (B, N, 3) or (B, 3, N), got {tuple(points.shape)}"
            )
        if points.shape[-1] == 3:
            return points
        if points.shape[1] == 3:
            return points.transpose(1, 2).contiguous()
        raise ValueError(f"Expected point cloud with 3 coordinates, got {tuple(points.shape)}")

    def _build_orientation_bias(self, frames: torch.Tensor) -> torch.Tensor:
        rel = torch.matmul(frames.unsqueeze(2).transpose(-1, -2), frames.unsqueeze(1))
        rel_flat = rel.reshape(rel.shape[0], rel.shape[1], rel.shape[2], 9)
        bias = self.orientation_mlp(rel_flat).permute(0, 3, 1, 2).contiguous()
        bias = bias - bias.mean(dim=-1, keepdim=True)
        return bias * self.orientation_scale.to(dtype=bias.dtype)

    def forward(self, points: torch.Tensor):
        bn3_points = self._to_bn3(points)
        if self.center_input:
            bn3_points = bn3_points - bn3_points.mean(dim=1, keepdim=True)

        neighborhood, center = self.group_divider(bn3_points)
        with torch.no_grad():
            frames = self._estimate_patch_frames(neighborhood)
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

        if not torch.is_tensor(features):
            raise RuntimeError(
                "RI-MAE encoder path returned non-tensor features in contrastive encoder adapter; "
                f"got type={type(features)}."
            )
        if features.dim() != 2:
            raise RuntimeError(
                "RI-MAE encoder path must return 2D features (B, D) in contrastive encoder adapter, "
                f"got shape={tuple(features.shape)}."
            )
        if features.shape[1] != self.output_dim:
            raise RuntimeError(
                "RI-MAE contrastive feature dim mismatch: "
                f"got {features.shape[1]}, expected {self.output_dim}."
            )

        # inv_latent_net is already invariant; eq_z is intentionally unavailable.
        return features, features, None


__all__ = ["RIMAEInvariantEncoderForContrastive"]
