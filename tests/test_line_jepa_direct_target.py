from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from src.training_methods.line_jepa.line_jepa import LineJEPALoss
from src.training_methods.line_jepa.line_jepa_module import (
    CompactSemanticProjector,
    LineContextPredictor,
    LineJEPAModule,
)
from src.training_methods.contrastive_learning.supervised_cache import _compute_probe_metrics


def _cosine_loss() -> LineJEPALoss:
    return LineJEPALoss(
        weight=1.0,
        prediction_coeff=1.0,
        sigreg_coeff=0.0,
        std_coeff=0.0,
        cov_coeff=0.0,
        std_eps=1.0e-4,
        std_target=1.0,
        start_epoch=0,
        prediction_loss="cosine",
        num_slices=8,
        integration_min=-5.0,
        integration_max=5.0,
        integration_points=17,
    )


def test_cosine_prediction_loss_has_expected_scale() -> None:
    objective = _cosine_loss()
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    matching, _ = objective.compute_loss(
        prediction=target,
        target=target,
        regularized_embeddings={},
        current_epoch=0,
        global_step=0,
    )
    opposite, _ = objective.compute_loss(
        prediction=-target,
        target=target,
        regularized_embeddings={},
        current_epoch=0,
        global_step=0,
    )

    torch.testing.assert_close(matching, torch.tensor(0.0))
    torch.testing.assert_close(opposite, torch.tensor(2.0))


def test_direct_target_prediction_inputs_are_unit_normalized() -> None:
    module = SimpleNamespace(prediction_normalization="l2")
    prediction = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    target = torch.tensor([[0.0, 7.0], [5.0, 0.0]])

    normalized_prediction, normalized_target = LineJEPAModule._prediction_loss_inputs(
        module,
        prediction=prediction,
        target=target,
    )

    torch.testing.assert_close(normalized_prediction.norm(dim=-1), torch.ones(2))
    torch.testing.assert_close(normalized_target.norm(dim=-1), torch.ones(2))


def test_shuffle_indices_preserve_task_and_requested_source_group() -> None:
    batch_size, task_count = 6, 2
    source_groups = ["Al", "Al", "Mg", "Mg", "Ta", "Ta"]
    indices = LineJEPAModule._deterministic_context_shuffle_indices(
        batch_size=batch_size,
        task_count=task_count,
        seed=17,
        source_groups=source_groups,
        device=torch.device("cpu"),
    )

    rows = torch.arange(batch_size * task_count)
    assert torch.all(indices != rows)
    torch.testing.assert_close(indices % task_count, rows % task_count)
    row_groups = [source_groups[int(row) // task_count] for row in rows]
    shuffled_groups = [source_groups[int(row) // task_count] for row in indices]
    assert shuffled_groups == row_groups


def test_endpoint_prediction_indices_cover_both_endpoints() -> None:
    module = SimpleNamespace(
        prediction_positions="endpoints",
        target_line_index=2,
        line_atoms=5,
    )
    assert LineJEPAModule._prediction_target_index_values(module, batch_idx=0) == [0, 4]


def test_context_matching_prefers_aligned_targets() -> None:
    objective = _cosine_loss()
    objective.context_match_coeff = 1.0
    objective.context_match_temperature = 0.1
    objective.context_match_negative_count = 3
    objective.context_match_negative_max_target_cosine = 0.99
    target = torch.eye(4)

    aligned, aligned_metrics = objective._context_match_loss(
        prediction=target,
        target=target,
    )
    shifted, _ = objective._context_match_loss(
        prediction=target.roll(1, dims=0),
        target=target,
    )

    assert float(aligned) < float(shifted)
    torch.testing.assert_close(
        aligned_metrics["jepa/sim/top1"], torch.tensor(1.0)
    )


def test_active_line_jepa_metrics_use_compact_groups() -> None:
    objective = _cosine_loss()
    objective.context_match_coeff = 0.25
    objective.context_match_negative_count = 3
    target = torch.eye(4)

    _, metrics = objective.compute_loss(
        prediction=target,
        target=target,
        regularized_embeddings={},
        current_epoch=0,
        global_step=0,
    )

    assert set(metrics) == {
        "jepa/pred/loss",
        "jepa/sim/loss",
        "jepa/sim/top1",
    }


def test_wandb_metric_paths_group_by_semantic_category() -> None:
    assert LineJEPAModule._jepa_metric_path("train", "jepa/pred/loss") == (
        "prediction/train_loss"
    )
    assert LineJEPAModule._jepa_metric_path("val", "jepa/sim/top1") == (
        "similarity/val_top1"
    )
    assert LineJEPAModule._jepa_metric_path("val", "jepa/context/local_gain") == (
        "context/val_local_gain"
    )
    assert LineJEPAModule._jepa_metric_path("test", "jepa/reg/std") == (
        "regularization/test_std"
    )
    assert LineJEPAModule._jepa_metric_path("val", "jepa/manifold/anchor") == (
        "manifold/val_anchor"
    )
    assert LineJEPAModule._jepa_metric_path(
        "val", "jepa/cluster/assignment_agreement"
    ) == "clustering/val_assignment_agreement"


def test_compact_semantic_projector_uses_teacher_pca_basis() -> None:
    generator = torch.Generator().manual_seed(4)
    dominant = torch.randn(256, 2, generator=generator)
    mixing = torch.randn(2, 16, generator=generator)
    teacher = dominant @ mixing + 0.001 * torch.randn(256, 16, generator=generator)
    projector = CompactSemanticProjector(input_dim=16, output_dim=4)

    projector.initialize_from_teacher(teacher)
    semantic = projector(teacher)

    assert bool(projector.initialized.item())
    assert semantic.shape == (256, 4)
    torch.testing.assert_close(semantic.norm(dim=-1), torch.ones(256), atol=1e-5, rtol=1e-5)
    gram = projector.linear.weight.float() @ projector.linear.weight.float().T
    torch.testing.assert_close(gram, torch.eye(4), atol=1e-5, rtol=1e-5)


def test_context_predictor_can_emit_compact_semantic_dimension() -> None:
    predictor = LineContextPredictor(
        input_dim=64,
        output_dim=32,
        hidden_dim=64,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
    )
    output = predictor(
        torch.randn(5, 4, 64),
        torch.randn(5, 4, 2),
        torch.randn(5, 2),
    )
    assert output.shape == (5, 32)


def test_semantic_geometry_loss_is_zero_for_preserved_manifold() -> None:
    module = SimpleNamespace(semantic_relation_samples=32)
    features = torch.randn(32, 8)

    anchor, relation = LineJEPAModule._semantic_geometry_losses(
        module,
        online=features,
        teacher=features,
    )

    torch.testing.assert_close(anchor, torch.tensor(0.0), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(relation, torch.tensor(0.0), atol=1e-6, rtol=0.0)


def test_semantic_geometry_large_batch_subsampling_is_bfloat16_safe() -> None:
    module = SimpleNamespace(semantic_relation_samples=1024)
    features = torch.randn(4800, 8, dtype=torch.bfloat16)

    anchor, relation = LineJEPAModule._semantic_geometry_losses(
        module,
        online=features,
        teacher=features,
    )

    torch.testing.assert_close(anchor, torch.tensor(0.0), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(relation, torch.tensor(0.0), atol=1e-6, rtol=0.0)


def test_balanced_sinkhorn_uses_every_prototype_equally() -> None:
    logits = torch.randn(120, 6, generator=torch.Generator().manual_seed(9))
    assignments = LineJEPAModule._balanced_sinkhorn(
        logits,
        epsilon=0.1,
        iterations=20,
    )

    torch.testing.assert_close(assignments.sum(dim=1), torch.ones(120), atol=2e-3, rtol=0.0)
    torch.testing.assert_close(
        assignments.mean(dim=0),
        torch.full((6,), 1.0 / 6.0),
        atol=1.1e-2,
        rtol=0.0,
    )


def test_probe_pca_uses_natural_variance_without_coordinate_standardization() -> None:
    rng = np.random.default_rng(12)
    latents = rng.normal(size=(600, 12)).astype(np.float32)
    latents[:, 0] *= 100.0
    module = SimpleNamespace(
        probe_seed=0,
        probe_max_samples=600,
        probe_k=6,
        probe_pca_max_components=12,
        probe_silhouette_samples=300,
        probe_kmeans_seed_count=2,
        probe_kmeans_n_init=2,
        probe_eps=1e-12,
        probe_best_k_min=2,
        probe_best_k_max=8,
    )

    metrics, _ = _compute_probe_metrics(module, latents, None, "val")

    assert metrics["pca95"] == 1.0
    assert metrics["effective_rank"] < 1.1


def test_compact_model_disables_preinitialization_summary_forward() -> None:
    cfg = OmegaConf.load("configs/line_jepa_static_pure.yaml")
    cfg.compile_encoder = False

    model = LineJEPAModule(cfg)

    assert model.example_input_array is None
    assert not bool(model.semantic_projector.initialized.item())


def test_continuous_line_jepa_keeps_vicreg_without_discrete_separation_losses() -> None:
    cfg = OmegaConf.load("configs/line_jepa_geo_frame_simple.yaml")
    cfg.compile_encoder = False

    model = LineJEPAModule(cfg)

    assert not model.freeze_encoder
    assert model.freeze_semantic_projector
    assert all(not parameter.requires_grad for parameter in model.semantic_projector.parameters())
    assert model.line_jepa.context_match_coeff == 0.0
    assert model.masked_token_coeff == 0.0
    assert model.semantic_anchor_coeff > 0.0
    assert model.semantic_relation_coeff > 0.0
    assert model.prototype_coeff == 0.0
    assert len(model.semantic_prototypes) == 0
    assert bool(model.semantic_prototypes_initialized.item())
    assert not model.regularizer_enabled
    assert isinstance(model.regularizer_projector, torch.nn.Identity)
    assert model.vicreg.enabled
    assert cfg.enable_probe_metrics
    assert cfg.checkpoint_monitor == "clustering/val_silhouette"


def test_directional_moments_are_rotation_consistent_and_polar() -> None:
    points = torch.tensor(
        [[[[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [-0.2, 0.0, 0.0]]]],
        dtype=torch.float32,
    )
    direction = torch.tensor([[1.0, 0.0, 0.0]])
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    original = LineJEPAModule._directional_moment_features(points, direction)
    rotated = LineJEPAModule._directional_moment_features(
        points @ rotation.T,
        direction @ rotation.T,
    )
    reversed_ray = LineJEPAModule._directional_moment_features(points, -direction)

    torch.testing.assert_close(original, rotated)
    torch.testing.assert_close(original[..., 0], -reversed_ray[..., 0])
    torch.testing.assert_close(original[..., 3], -reversed_ray[..., 3])
    torch.testing.assert_close(original[..., 6], reversed_ray[..., 6])
