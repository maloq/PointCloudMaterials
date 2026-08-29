from __future__ import annotations

import os

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.models.encoders.factory import available_encoder_names
from src.models.encoders.geo_frame_transformer import GeoFrameTransformerEncoder
from src.models.encoders.ri_mae_encoder import RIMAEBackbone
from src.training_methods.contrastive_learning.vicreg_module import VICRegModule


def _small_encoder(frame_builder: str = "triad") -> GeoFrameTransformerEncoder:
    return GeoFrameTransformerEncoder(
        latent_size=64,
        num_group=8,
        patch_sizes=(6, 12),
        encoder_dims=32,
        trans_dim=32,
        depth=2,
        num_heads=4,
        enable_ray_head=True,
        ray_feature_dim=16,
        enable_masked_token_objective=True,
        mask_predictor_depth=1,
        deterministic_fps=True,
        group_sampling="fps",
        frame_builder=frame_builder,
    )


def _rotation() -> torch.Tensor:
    matrix, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(matrix) < 0:
        matrix[:, 0] *= -1
    return matrix


def test_new_and_ablation_encoders_are_both_registered() -> None:
    names = available_encoder_names()
    assert "GeoFrameTransformer" in names
    assert "RI_MAE_Invariant" in names


@pytest.mark.parametrize("frame_builder", ["triad", "pca"])
def test_geo_frame_features_and_ray_conditioning_are_rotation_invariant(
    frame_builder: str,
) -> None:
    torch.manual_seed(7)
    encoder = _small_encoder(frame_builder).eval()
    points = torch.randn(4, 24, 3)
    rays = torch.randn(4, 3)
    rotation = _rotation()

    with torch.no_grad():
        output = encoder(points)
        rotated_output = encoder(points @ rotation)
        directional = encoder.directional_features_from_geometry(output[1], rays)
        rotated_directional = encoder.directional_features_from_geometry(
            rotated_output[1],
            rays @ rotation,
        )

    assert output[0].shape == (4, 64)
    assert output[1]["tokens"].shape == (4, 8, 32)
    torch.testing.assert_close(output[0], rotated_output[0], atol=3.0e-5, rtol=3.0e-5)
    torch.testing.assert_close(directional, rotated_directional, atol=3.0e-5, rtol=3.0e-5)


def test_triad_frame_completion_handles_collapsed_and_collinear_patches() -> None:
    collapsed = torch.zeros(1, 1, 6, 3)
    collinear = torch.zeros(1, 1, 6, 3)
    collinear[0, 0, :, 0] = torch.linspace(-2.0, 2.0, 6)
    neighborhoods = torch.cat([collapsed, collinear], dim=1)

    compiled_frame_builder = torch.compile(
        RIMAEBackbone._estimate_patch_frames,
        backend="eager",
        fullgraph=True,
    )
    frames = compiled_frame_builder(
        neighborhoods,
        frame_builder="triad",
        frame_eps=1.0e-6,
    )
    gram = frames.transpose(-1, -2) @ frames

    assert torch.isfinite(frames).all()
    torch.testing.assert_close(
        gram,
        torch.eye(3).expand_as(gram),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        torch.det(frames),
        torch.ones(1, 2),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_masked_token_objective_has_gradients_and_updates_ema_teacher() -> None:
    torch.manual_seed(11)
    encoder = _small_encoder().train()
    points = torch.randn(4, 24, 3)
    loss = encoder.masked_token_loss(points)
    loss.backward()

    assert torch.isfinite(loss)
    assert encoder.mask_token.grad is not None
    assert encoder.mask_prediction_head.weight.grad is not None

    student_parameter = next(encoder.token_encoder.parameters())
    teacher_parameter = next(encoder.mask_teacher.parameters())
    with torch.no_grad():
        student_parameter.add_(0.25)
        teacher_before = teacher_parameter.clone()
    encoder.update_mask_teacher()
    assert not torch.equal(teacher_parameter, teacher_before)

    encoder.reset_mask_teacher_from_student()
    for student, teacher in zip(
        encoder.token_encoder.parameters(),
        encoder.mask_teacher.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(student, teacher)


def test_disabled_auxiliary_heads_allocate_no_parameters_and_fail_loudly() -> None:
    encoder = GeoFrameTransformerEncoder(
        latent_size=32,
        num_group=8,
        patch_sizes=(6, 12),
        encoder_dims=32,
        trans_dim=32,
        depth=2,
        num_heads=4,
        enable_ray_head=False,
        enable_masked_token_objective=False,
    )

    parameter_names = set(dict(encoder.named_parameters()))
    assert not any("ray_" in name for name in parameter_names)
    assert not any("mask_" in name for name in parameter_names)
    assert encoder.ray_token_mlp is None
    assert encoder.ray_attention is None
    assert encoder.ray_output is None
    assert encoder.mask_predictor is None
    assert encoder.mask_prediction_head is None
    assert encoder.mask_teacher is None

    with pytest.raises(RuntimeError, match="ray head is disabled"):
        encoder.directional_features_from_geometry({}, torch.randn(2, 3))
    with pytest.raises(RuntimeError, match="masked-token objective is disabled"):
        encoder.masked_token_loss(torch.randn(2, 24, 3))
    with pytest.raises(RuntimeError, match="EMA teacher is disabled"):
        encoder.update_mask_teacher()


def test_geo_frame_vicreg_regularizes_exported_representation_directly() -> None:
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("configs")):
        cfg = compose(config_name="vicreg_geo_frame_multi")
    module = VICRegModule(cfg)

    assert isinstance(module.vicreg.projector, torch.nn.Identity)
    assert module.vicreg.embed_dim == module.encoder.invariant_dim == 64


def test_vicreg_paper_multiscale_config() -> None:
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("configs")):
        cfg = compose(config_name="vicreg_geo_frame_multiscale_8_16_l128")
    module = VICRegModule(cfg)

    assert cfg.compile_encoder
    assert cfg.encoder_compile_mode == "reduce-overhead"
    assert cfg.encoder_compile_fullgraph
    assert not cfg.encoder_compile_dynamic
    assert isinstance(module.encoder, torch._dynamo.eval_frame.OptimizedModule)
    assert cfg.latent_size == 128
    assert module.encoder.invariant_dim == 128
    assert module.encoder.token_encoder.patch_sizes == (8, 16)
    assert module.encoder.token_encoder.scale_embeddings is not None
    assert not module.encoder.token_encoder.use_frame_gating
    assert not module.encoder.enable_ray_head
    assert not module.encoder.enable_masked_token_objective
    assert module.encoder.ray_token_mlp is None
    assert module.encoder.mask_teacher is None
    assert isinstance(module.vicreg.projector, torch.nn.Sequential)
    assert module.vicreg.embed_dim == 128
    assert module.vicreg.sim_coeff == 25.0
    assert module.vicreg.std_coeff == 25.0
    assert module.vicreg.cov_coeff == 1.0
    assert cfg.decay_rate == 0.04
    assert cfg.epochs == 100
    assert not cfg.ddp_find_unused_parameters
    assert sum(parameter.numel() for parameter in module.parameters()) == 1_703_689
    assert all(parameter.requires_grad for parameter in module.parameters())
