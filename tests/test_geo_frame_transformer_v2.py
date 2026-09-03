from __future__ import annotations

import os

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.models.encoders.factory import available_encoder_names
from src.models.encoders.geo_frame_transformer import (
    GeoFrameTransformerEncoder,
    RelativeFrameOrientationBias,
)
from src.models.encoders.geo_frame_transformer_v2 import (
    GeoFrameTransformerV2Encoder,
    _symmetric_3x3_eigenvalues,
)
from src.training_methods.contrastive_learning.vicreg_module import VICRegModule


def _proper_rotation() -> torch.Tensor:
    matrix, _ = torch.linalg.qr(torch.tensor(
        [
            [0.31, -0.73, 0.19],
            [0.52, 0.11, -0.64],
            [-0.27, 0.43, 0.79],
        ],
        dtype=torch.float32,
    ))
    if torch.det(matrix) < 0:
        matrix[:, 0] *= -1
    return matrix


def _asymmetric_points(*, batch_size: int = 2, num_points: int = 16) -> torch.Tensor:
    generator = torch.Generator().manual_seed(8241)
    points = torch.randn(batch_size, num_points, 3, generator=generator)
    point_index = torch.arange(num_points, dtype=torch.float32)
    points[..., 0] += 0.013 * point_index.square()
    points[..., 1] += 0.037 * point_index
    points[..., 2] += 0.021 * torch.sin(point_index)
    return points


def _small_v2(
    *,
    parity_mode: str = "invariant",
    use_signed_chirality: bool = True,
    num_group: int = 4,
    patch_sizes: tuple[int, ...] = (4, 8),
) -> GeoFrameTransformerV2Encoder:
    return GeoFrameTransformerV2Encoder(
        latent_size=24,
        num_group=num_group,
        patch_sizes=patch_sizes,
        encoder_dims=24,
        trans_dim=24,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        deterministic_fps=True,
        sorting_mode="none",
        group_sampling="fps",
        center_input=True,
        frame_builder="triad",
        frame_eps=1.0e-6,
        use_frame_gating=True,
        frame_confidence_floor=0.0,
        num_rbf=8,
        rbf_max_distance=3.0,
        edge_dim=12,
        edge_value_rank=2,
        parity_mode=parity_mode,
        use_signed_chirality=use_signed_chirality,
        pooling_mode="max_mean",
        use_gradient_checkpointing=False,
    )


def _forward_with_state(
    encoder: GeoFrameTransformerV2Encoder,
    points: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    features, state = encoder.forward_with_state(points)
    expected_keys = {
        "centers",
        "frames",
        "frame_confidence",
        "shape_descriptor",
        "signed_chirality",
        "edge_embedding",
        "tokens",
    }
    assert expected_keys <= state.keys()
    return features, state


def test_v2_closed_form_covariance_eigenvalues_match_eigvalsh() -> None:
    generator = torch.Generator().manual_seed(101)
    patches = torch.randn(128, 16, 3, generator=generator)
    covariance = patches.transpose(1, 2) @ patches / float(patches.shape[1])
    covariance[0] = 2.0 * torch.eye(3)

    expected = torch.linalg.eigvalsh(covariance)
    actual = _symmetric_3x3_eigenvalues(covariance, eps=1.0e-6)

    torch.testing.assert_close(actual, expected, atol=8.0e-6, rtol=2.0e-5)


def test_v2_is_se3_invariant_and_exposes_finite_geometry_state() -> None:
    torch.manual_seed(17)
    encoder = _small_v2(parity_mode="sensitive").eval()
    points = _asymmetric_points()
    rotation = _proper_rotation()
    translation = torch.tensor([2.3, -0.7, 1.1])

    with torch.no_grad():
        features, state = _forward_with_state(encoder, points)
        transformed_features, transformed_state = _forward_with_state(
            encoder,
            points @ rotation + translation,
        )

    assert features.shape == (2, 24)
    assert state["centers"].shape == (2, 4, 3)
    assert state["frames"].shape == (2, 4, 3, 3)
    assert state["frame_confidence"].shape == (2, 4)
    assert state["edge_embedding"].shape == (2, 4, 4, 12)
    assert state["tokens"].shape == (2, 4, 24)
    assert all(torch.isfinite(value).all() for value in state.values())
    torch.testing.assert_close(features, transformed_features, atol=8.0e-5, rtol=8.0e-5)
    torch.testing.assert_close(
        state["edge_embedding"],
        transformed_state["edge_embedding"],
        atol=8.0e-5,
        rtol=8.0e-5,
    )


def test_v2_invariant_parity_mode_symmetrizes_mirror_images() -> None:
    torch.manual_seed(29)
    encoder = _small_v2(parity_mode="invariant", use_signed_chirality=True).eval()
    points = _asymmetric_points()
    reflection = torch.tensor([-1.0, 1.0, 1.0])

    with torch.no_grad():
        features, state = _forward_with_state(encoder, points)
        reflected_features, reflected_state = _forward_with_state(
            encoder,
            points * reflection,
        )

    torch.testing.assert_close(features, reflected_features, atol=8.0e-5, rtol=8.0e-5)
    torch.testing.assert_close(
        state["edge_embedding"],
        reflected_state["edge_embedding"],
        atol=8.0e-5,
        rtol=8.0e-5,
    )


def test_v2_sensitive_parity_mode_retains_signed_chirality() -> None:
    torch.manual_seed(31)
    encoder = _small_v2(parity_mode="sensitive", use_signed_chirality=True).eval()
    points = _asymmetric_points()
    reflection = torch.tensor([-1.0, 1.0, 1.0])

    with torch.no_grad():
        features, state = _forward_with_state(encoder, points)
        reflected_features, reflected_state = _forward_with_state(
            encoder,
            points * reflection,
        )

    chirality = state["signed_chirality"]
    reflected_chirality = reflected_state["signed_chirality"]
    assert chirality.abs().max() > 1.0e-5
    torch.testing.assert_close(chirality, -reflected_chirality, atol=8.0e-5, rtol=8.0e-5)
    assert not torch.allclose(state["edge_embedding"], reflected_state["edge_embedding"])
    assert not torch.allclose(features, reflected_features)


def test_v2_frame_confidence_detects_degenerate_local_geometry() -> None:
    encoder = _small_v2(
        parity_mode="invariant",
        num_group=2,
        patch_sizes=(8,),
    ).eval()
    cube = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    anisotropic = cube * torch.tensor([3.0, 2.0, 1.0])
    neighborhood = torch.stack([cube, anisotropic], dim=0).unsqueeze(0)
    centers = torch.zeros(1, 2, 3)

    with torch.no_grad():
        tokens, attention_bias, value_geometry, state = (
            encoder.token_encoder.prepare_tokens(neighborhood, centers)
        )

    confidence = state["frame_confidence"]
    assert confidence.shape == (1, 2)
    assert confidence[0, 0] < 1.0e-5
    assert confidence[0, 1] > 0.3
    assert torch.isfinite(tokens).all()
    assert torch.isfinite(attention_bias).all()
    assert torch.isfinite(value_geometry).all()
    assert torch.isfinite(state["shape_descriptor"]).all()


def test_v2_edge_basis_is_richer_than_v1_orientation_bias() -> None:
    torch.manual_seed(41)
    encoder = _small_v2(
        parity_mode="sensitive",
        use_signed_chirality=True,
        num_group=3,
        patch_sizes=(4,),
    ).eval()
    frames = torch.eye(3).reshape(1, 1, 3, 3).expand(1, 3, 3, 3).clone()
    confidence = torch.ones(1, 3)
    centers_x = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    centers_y = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]])
    shape_x = torch.zeros(1, 3, 8)
    shape_y = shape_x.clone()
    shape_y[..., 0] = torch.tensor([[0.1, 0.4, 0.8]])
    chirality_x = torch.zeros(1, 3)
    chirality_y = torch.tensor([[0.2, -0.3, 0.5]])

    v1_bias = RelativeFrameOrientationBias(num_heads=4, hidden_dim=16).eval()
    with torch.no_grad():
        v1_x = v1_bias(centers_x, frames, confidence)
        v1_y = v1_bias(centers_y, frames, confidence)
        edge_x, _ = encoder.token_encoder.build_pairwise_geometry(
            centers_x,
            frames,
            confidence,
            shape_x,
            chirality_x,
            return_state=True,
        )
        edge_y, _ = encoder.token_encoder.build_pairwise_geometry(
            centers_y,
            frames,
            confidence,
            shape_y,
            chirality_y,
            return_state=True,
        )
        attention_x, value_geometry_x = (
            encoder.token_encoder.project_pairwise_geometry(edge_x)
        )
        attention_y, value_geometry_y = (
            encoder.token_encoder.project_pairwise_geometry(edge_y)
        )
        logit_x, gate_x, coefficients_x = (
            encoder.token_encoder.transformer.layers[0].geometry_modulation(
                attention_x,
                value_geometry_x,
            )
        )
        logit_y, gate_y, coefficients_y = (
            encoder.token_encoder.transformer.layers[0].geometry_modulation(
                attention_y,
                value_geometry_y,
            )
        )
        seed_tokens = torch.randn(1, 3, 24)
        message_x = encoder.token_encoder.transformer(
            seed_tokens,
            attention_x,
            value_geometry_x,
        )
        message_y = encoder.token_encoder.transformer(
            seed_tokens,
            attention_y,
            value_geometry_y,
        )

    torch.testing.assert_close(v1_x, v1_y, atol=0.0, rtol=0.0)
    assert edge_x.shape == (1, 3, 3, 12)
    assert attention_x.shape == (1, 4, 3, 3)
    assert value_geometry_x.shape == (1, 3, 12)
    assert not torch.allclose(edge_x, edge_y)
    assert not torch.allclose(logit_x, logit_y)
    assert not torch.allclose(gate_x, gate_y)
    assert not torch.allclose(coefficients_x, coefficients_y)
    assert not torch.allclose(message_x, message_y)


def test_v2_edge_and_value_paths_receive_finite_gradients() -> None:
    torch.manual_seed(53)
    encoder = _small_v2(parity_mode="sensitive").train()
    points = _asymmetric_points()
    features, state = _forward_with_state(encoder, points)
    loss = features.square().mean() + 0.01 * state["tokens"].square().mean()
    loss.backward()

    geometry_gradients = {
        name: parameter.grad
        for name, parameter in encoder.named_parameters()
        if "pair_geometry" in name
        or "edge_logit" in name
        or "edge_value" in name
    }
    assert geometry_gradients
    assert all(gradient is not None for gradient in geometry_gradients.values())
    assert all(torch.isfinite(gradient).all() for gradient in geometry_gradients.values())
    assert any(gradient.abs().max() > 0.0 for gradient in geometry_gradients.values())


def test_v2_compile_fullgraph_backward_smoke() -> None:
    torch.manual_seed(67)
    encoder = _small_v2(parity_mode="invariant").train()
    compiled_forward = torch.compile(
        encoder.forward_features,
        backend="eager",
        fullgraph=True,
    )
    features = compiled_forward(_asymmetric_points())
    loss = features.square().mean()
    loss.backward()

    assert features.shape == (2, 24)
    assert torch.isfinite(features).all()
    trainable_gradients = [
        parameter.grad
        for parameter in encoder.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert trainable_gradients
    assert all(torch.isfinite(gradient).all() for gradient in trainable_gradients)


def test_v1_and_v2_are_distinct_registered_encoders() -> None:
    names = available_encoder_names()
    assert "GeoFrameTransformer" in names
    assert "GeoFrameTransformerV2" in names

    v1 = GeoFrameTransformerEncoder(
        latent_size=24,
        num_group=4,
        patch_sizes=(4, 8),
        encoder_dims=24,
        trans_dim=24,
        depth=1,
        num_heads=4,
        deterministic_fps=True,
        group_sampling="fps",
        pooling_mode="max_mean",
        enable_ray_head=False,
        enable_masked_token_objective=False,
    )
    v2 = _small_v2()
    assert type(v1) is GeoFrameTransformerEncoder
    assert type(v2) is GeoFrameTransformerV2Encoder
    assert not any("edge_value_gate" in name for name in v1.state_dict())
    assert any("edge_value_gate" in name for name in v2.state_dict())


def test_v2_active_vicreg_config_builds_the_new_encoder() -> None:
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("configs")):
        cfg = compose(config_name="vicreg_geo_frame_transformer_v2")
    module = VICRegModule(cfg)
    uncompiled_encoder = getattr(module.encoder, "_orig_mod", module.encoder)

    assert cfg.encoder.name == "GeoFrameTransformerV2"
    assert cfg.compile_encoder
    assert cfg.encoder_compile_mode == "default"
    assert cfg.encoder_compile_fullgraph
    assert cfg.encoder.kwargs.parity_mode == "sensitive"
    assert cfg.encoder.kwargs.use_signed_chirality
    assert isinstance(uncompiled_encoder, GeoFrameTransformerV2Encoder)
    assert uncompiled_encoder.invariant_dim == 128


def test_v2_o3_config_preserves_full_encoder_configuration() -> None:
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("configs")):
        cfg = compose(config_name="vicreg_geo_frame_transformer_v2_o3")

    assert cfg.encoder.name == "GeoFrameTransformerV2"
    assert cfg.encoder.kwargs.parity_mode == "invariant"
    assert cfg.encoder.kwargs.patch_sizes == [8, 16]
    assert cfg.encoder.kwargs.edge_dim == 16
    assert cfg.encoder.kwargs.edge_value_rank == 2
