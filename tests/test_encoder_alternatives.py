from __future__ import annotations

import torch
import pytest

from src.models import (
    available_encoder_names,
    build_encoder,
    encode_point_clouds,
)


def test_registry_contains_paper_encoder_alternatives() -> None:
    expected = {
        "DGCNN",
        "PnE_L",
        "PnE_S",
        "PnE_VN",
        "RI_MAE_Invariant",
        "VN_DGCNN",
        "VN_REVNET_Atomic",
        "VN_REVNET_Backbone",
    }
    assert expected.issubset(available_encoder_names())


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        (
            "VN_DGCNN",
            {
                "latent_size": 12,
                "n_knn": 4,
                "feature_dims": (8, 8, 12, 16, 20),
                "global_mlp_dims": (20, 16),
                "global_dropout": 0.0,
                "use_batchnorm": False,
                "use_cross_product": True,
            },
        ),
        (
            "PnE_VN",
            {
                "latent_size": 12,
                "n_knn": 4,
                "feature_transform": False,
                "hidden_dim1": 12,
                "hidden_dim2": 20,
                "use_batchnorm": False,
            },
        ),
        (
            "DGCNN",
            {
                "latent_size": 12,
                "n_knn": 4,
                "feature_dims": (8, 8, 12, 16),
                "emb_dims": 24,
                "dropout_rate": 0.0,
                "use_batchnorm": False,
            },
        ),
        (
            "PnE_S",
            {
                "latent_size": 12,
                "feature_transform": False,
                "dropout_rate": 0.0,
            },
        ),
    ],
)
def test_notebook_encoder_alternatives_forward(name: str, kwargs: dict) -> None:
    output = encode_point_clouds(
        build_encoder(name, **kwargs).eval(),
        torch.randn(2, 24, 3),
    )

    assert output.invariant.shape == (2, 12)


def test_runtime_adapts_regular_and_vn_encoder_contracts() -> None:
    points = torch.randn(2, 24, 3)
    regular = build_encoder(
        "DGCNN",
        latent_size=12,
        n_knn=4,
        feature_dims=(8, 8, 12, 16),
        emb_dims=24,
        dropout_rate=0.0,
        use_batchnorm=False,
    )
    vn = build_encoder(
        "VN_REVNET_Atomic",
        latent_size=12,
        k_embed=4,
        k_list=(4, 6),
        embed_channels=8,
        hidden_channels=(12, 18),
        geom_k=4,
        geom_dim=6,
        use_batchnorm=False,
    )

    regular_output = encode_point_clouds(regular, points)
    vn_output = encode_point_clouds(vn, points)

    assert regular_output.invariant.shape == (2, 12)
    assert regular_output.equivariant is None
    assert vn_output.invariant.shape == (2, 18)
    assert vn_output.equivariant.shape == (2, 12, 3)


def test_atomic_vn_invariant_is_rotation_invariant() -> None:
    torch.manual_seed(3)
    encoder = build_encoder(
        "VN_REVNET_Atomic",
        latent_size=12,
        k_embed=4,
        k_list=(4, 6),
        embed_channels=8,
        hidden_channels=(12, 18),
        geom_k=4,
        geom_dim=6,
        use_batchnorm=False,
    ).eval()
    points = torch.randn(3, 24, 3)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1

    invariant = encode_point_clouds(encoder, points).invariant
    rotated = encode_point_clouds(encoder, points @ rotation.T).invariant

    torch.testing.assert_close(rotated, invariant, atol=2e-5, rtol=2e-5)
