from __future__ import annotations

from src.models.encoders.geo_frame_transformer_v2 import GeoFrameTransformerV2Encoder
from src.temporal_vamp.temporal_encoder_pretraining import (
    configure_geoframe_tail_trainable,
)


def test_configured_geoframe_tail_is_the_only_trainable_encoder_region() -> None:
    encoder = GeoFrameTransformerV2Encoder(
        latent_size=32,
        num_group=4,
        patch_sizes=(4, 8),
        encoder_dims=32,
        trans_dim=32,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
        deterministic_fps=True,
        num_rbf=4,
        edge_dim=8,
        edge_value_rank=2,
        parity_mode="sensitive",
        pooling_mode="max_mean",
    )
    names = configure_geoframe_tail_trainable(encoder, trainable_tail_blocks=2)
    assert names
    assert all(
        name.startswith("token_encoder.transformer.layers.2.")
        or name.startswith("token_encoder.transformer.layers.3.")
        or name.startswith("token_encoder.transformer.norm.")
        for name in names
    )
    assert not any(
        parameter.requires_grad
        for name, parameter in encoder.named_parameters()
        if name not in names
    )
