from __future__ import annotations

from ..base import Encoder
from ..registry import register_encoder
from src.training_methods.ri_mae.contrastive_backbone import RIMAEInvariantEncoderForContrastive


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
        try:
            from src.training_methods.ri_mae.ri_mae_module import RIMAEBackbone
        except ImportError as exc:
            raise ImportError(
                "RI_MAE_Invariant encoder could not import RIMAEBackbone. "
                "Ensure RI-MAE dependencies are installed and src.training_methods.ri_mae.ri_mae_module is available."
            ) from exc

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
        )
        self.encoder = RIMAEInvariantEncoderForContrastive(
            backbone,
            center_input=bool(center_input),
            output_dim=output_dim,
        )

    def forward(self, x):
        return self.encoder(x)
