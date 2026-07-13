from __future__ import annotations

import os

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.models.encoders.factory import available_encoder_names
from src.models.encoders.geo_frame_transformer import GeoFrameTransformerEncoder
from src.models.encoders.ri_mae_encoder import RIMAEBackbone
from src.training_methods.contrastive_learning.vicreg_module import VICRegModule
from src.training_methods.line_jepa.line_jepa_module import LineJEPAModule


def _small_encoder(frame_builder: str = "triad") -> GeoFrameTransformerEncoder:
    return GeoFrameTransformerEncoder(
        latent_size=64,
        num_group=8,
        patch_sizes=(6, 12),
        encoder_dims=32,
        trans_dim=32,
        depth=2,
        num_heads=4,
        ray_feature_dim=16,
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

    with pytest.warns(RuntimeWarning, match="1 fully collapsed patch"):
        frames = RIMAEBackbone._estimate_patch_frames(
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


def test_geo_frame_vicreg_regularizes_exported_representation_directly() -> None:
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("configs")):
        cfg = compose(config_name="vicreg_geo_frame_multi")
    module = VICRegModule(cfg)

    assert isinstance(module.vicreg.projector, torch.nn.Identity)
    assert module.vicreg.embed_dim == module.encoder.invariant_dim == 64


def test_geo_frame_multiscale_vicreg_config_enables_two_scales_without_gating() -> None:
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("configs")):
        cfg = compose(config_name="vicreg_geo_frame_multiscale")
    module = VICRegModule(cfg)

    assert module.encoder.token_encoder.patch_sizes == (12, 24)
    assert module.encoder.token_encoder.scale_embeddings is not None
    assert not module.encoder.token_encoder.use_frame_gating
    assert module.vicreg.enabled
    assert isinstance(module.vicreg.projector, torch.nn.Identity)


def test_line_jepa_multiscale_config_matches_pretraining_architecture() -> None:
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("configs")):
        cfg = compose(config_name="line_jepa_geo_frame_multiscale")
    module = LineJEPAModule(cfg)

    assert module.encoder.token_encoder.patch_sizes == (12, 24)
    assert module.encoder.token_encoder.scale_embeddings is not None
    assert not module.encoder.token_encoder.use_frame_gating
    assert str(cfg.init_from_checkpoint).endswith(
        "VICREG_GEOFRAME_MULTISCALE_12_24_l64_N160_M80_"
        "GeoFrameTransformer-epoch=49.ckpt"
    )


def test_vicreg_paper_multiscale_config() -> None:
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("configs")):
        cfg = compose(config_name="vicreg_geo_frame_multiscale_8_12_l128")
    module = VICRegModule(cfg)

    assert cfg.latent_size == 128
    assert module.encoder.invariant_dim == 128
    assert module.encoder.token_encoder.patch_sizes == (8, 12)
    assert module.encoder.token_encoder.scale_embeddings is not None
    assert not module.encoder.token_encoder.use_frame_gating
    assert isinstance(module.vicreg.projector, torch.nn.Sequential)
    assert module.vicreg.embed_dim == 128
    assert module.vicreg.sim_coeff == 25.0
    assert module.vicreg.std_coeff == 25.0
    assert module.vicreg.cov_coeff == 1.0
    assert cfg.decay_rate == 0.04
    assert cfg.epochs == 200


def test_line_jepa_uses_paper_vicreg_l128_encoder() -> None:
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("configs")):
        cfg = compose(config_name="line_jepa_geo_frame_multiscale_8_12_l128")
    module = LineJEPAModule(cfg)

    assert cfg.latent_size == 128
    assert module.encoder.invariant_dim == 128
    assert module.encoder.token_encoder.patch_sizes == (8, 12)
    assert module.semantic_dim == 128
    assert isinstance(module.vicreg.projector, torch.nn.Sequential)
    assert module.vicreg.embed_dim == 128
    assert module.vicreg.cov_coeff == 1.0
    assert str(cfg.init_from_checkpoint).endswith(
        "VICREG_GEOFRAME_MULTISCALE_8_12_PAPER_l128_N160_M80_"
        "GeoFrameTransformer-epoch=199.ckpt"
    )


def test_line_jepa_consumes_cached_patch_geometry() -> None:
    cfg = OmegaConf.load("configs/line_jepa_geo_frame.yaml")
    cfg.compile_encoder = False
    cfg.latent_size = 64
    cfg.encoder.kwargs.latent_size = 64
    cfg.encoder.kwargs.num_group = 8
    cfg.encoder.kwargs.patch_sizes = [6, 12]
    cfg.encoder.kwargs.encoder_dims = 32
    cfg.encoder.kwargs.trans_dim = 32
    cfg.encoder.kwargs.depth = 2
    cfg.encoder.kwargs.num_heads = 4
    cfg.encoder.kwargs.ray_feature_dim = 16
    cfg.encoder.kwargs.mask_predictor_depth = 1
    cfg.encoder.kwargs.deterministic_fps = True
    cfg.encoder.kwargs.group_sampling = "fps"
    cfg.line_jepa_target_encoder = "online"
    cfg.line_jepa_semantic_dim = 16
    cfg.line_jepa_semantic_anchor_coeff = 0.0
    cfg.line_jepa_semantic_relation_coeff = 0.0
    cfg.line_jepa_prototype_coeff = 0.0
    cfg.data.num_points = 24
    cfg.data.model_points = 24

    module = LineJEPAModule(cfg)
    batch_size = 2
    points = torch.randn(batch_size * module.line_atoms, 24, 3)
    feature_blocks, geometry_blocks = module._encode_prepared_feature_blocks([points])
    rays = torch.randn(batch_size * 2, 3)
    directional = module._encoder_directional_features_from_line_state(
        line_state=geometry_blocks[0],
        batch_size=batch_size,
        target_indices=[0, module.line_atoms - 1],
        task_ray_direction=rays,
    )

    assert feature_blocks[0].shape == (batch_size * module.line_atoms, 64)
    assert directional.shape == (batch_size * 2, module.line_atoms - 1, 16)
