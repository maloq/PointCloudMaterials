from __future__ import annotations

from src.models.encoders.geo_frame_transformer_v2 import GeoFrameTransformerV2Encoder
from src.temporal_vamp.shooting_encoder_finetune import (
    configure_last_geoframe_block_trainable,
)


def test_only_final_geoframe_block_and_norm_are_trainable() -> None:
    encoder = GeoFrameTransformerV2Encoder(
        latent_size=32,
        num_group=4,
        patch_sizes=(4, 8),
        encoder_dims=32,
        trans_dim=32,
        depth=3,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        deterministic_fps=True,
        sorting_mode="none",
        group_sampling="random",
        center_input=True,
        frame_builder="triad",
        frame_eps=1.0e-6,
        use_frame_gating=False,
        frame_confidence_floor=0.25,
        num_rbf=4,
        rbf_max_distance=3.0,
        edge_dim=8,
        edge_value_rank=2,
        parity_mode="sensitive",
        use_signed_chirality=True,
        pooling_mode="max_mean",
    )
    names = configure_last_geoframe_block_trainable(encoder)
    assert names
    assert all(
        name.startswith("token_encoder.transformer.layers.2.")
        or name.startswith("token_encoder.transformer.norm.")
        for name in names
    )
    assert not any(
        parameter.requires_grad
        for name, parameter in encoder.named_parameters()
        if name not in names
    )

