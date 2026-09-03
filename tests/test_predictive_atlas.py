from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.temporal_vamp.predictive_atlas import (
    FittedPredictiveAtlas,
    PredictiveAtlas,
    compute_pullback_spectrum,
    prepare_joint_path_target_data,
    prepare_joint_path_target_data_from_kernel,
    random_fourier_path_features,
    save_path_kernel,
)
from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache


def _joint_process_cache() -> ShootingEmbeddingCache:
    parent_count = 6
    branches_per_parent = 4
    parent_local = np.zeros((parent_count, 1, 1), dtype=np.float32)
    future = np.empty(
        (parent_count * branches_per_parent, 2, 1, 1), dtype=np.float32
    )
    branch_parent = np.repeat(np.arange(parent_count), branches_per_parent)
    correlated = np.asarray([[-1.0, -1.0], [1.0, 1.0]] * 2)
    anticorrelated = np.asarray([[-1.0, 1.0], [1.0, -1.0]] * 2)
    for parent_index in range(parent_count):
        paths = correlated if parent_index % 2 == 0 else anticorrelated
        rows = np.flatnonzero(branch_parent == parent_index)
        future[rows, :, 0, 0] = paths
    parents = []
    for parent_index in range(parent_count):
        if parent_index < 2:
            source_split = "train"
            source_seed = parent_index + 1
        elif parent_index < 4:
            source_split = "train"
            source_seed = 99
        else:
            source_split = "validation"
            source_seed = parent_index + 1
        parents.append(
            {
                "parent_id": f"parent_{parent_index}",
                "source_run_id": f"source_{parent_index}",
                "source_split": source_split,
                "source_velocity_seed": source_seed,
                "temperature_K": 400.0,
                "phase": "boundary",
                "source_crystalline_fraction": 0.01,
            }
        )
    return ShootingEmbeddingCache(
        path=Path("."),
        manifest={"snapshot": {"parents": parents}},
        parent_z=parent_local.copy(),
        parent_local_z=parent_local,
        parent_coords=np.zeros((parent_count, 1, 3), dtype=np.float32),
        future_z=future,
        branch_parent_index=branch_parent,
        atom_ids=np.asarray([1], dtype=np.int64),
        horizons_ps=np.asarray([1.0, 2.0]),
    )


def test_joint_path_kernel_detects_dependence_with_identical_marginals() -> None:
    targets = prepare_joint_path_target_data(
        _joint_process_cache(),
        horizons_ps=[1.0, 2.0],
        horizon_weights=[1.0, 1.0],
        rff_features_per_bandwidth=4096,
        bandwidth_multipliers=[1.0],
        selection_source_velocity_seeds=[99],
        seed=7,
        rff_device="cpu",
        rff_batch_size=32,
    )
    validation = targets.parent_splits["validation"]
    left = targets.empirical_mean_embedding[int(validation[0])]
    right = targets.empirical_mean_embedding[int(validation[1])]
    joint_distance = float(np.linalg.norm(left - right))
    assert joint_distance > 0.1
    assert (
        targets.diagnostics["same_branch_alignment"]
        ["validation_mean_true_vs_shuffled_embedding_distance"]
        > 0.05
    )

    future = np.asarray(_joint_process_cache().future_z)[:, :, 0, 0]
    branch_parent = np.repeat(np.arange(6), 4)
    for horizon_index in range(2):
        correlated_values = np.sort(future[branch_parent == 4, horizon_index])
        anticorrelated_values = np.sort(future[branch_parent == 5, horizon_index])
        np.testing.assert_array_equal(correlated_values, anticorrelated_values)


def test_random_features_approximate_rbf_mean_embedding_distance() -> None:
    rng = np.random.default_rng(11)
    left = rng.normal(loc=-0.3, size=(80, 3)).astype(np.float32)
    right = rng.normal(loc=0.4, size=(90, 3)).astype(np.float32)
    bandwidth = 1.2
    feature_count = 8192
    frequencies = rng.normal(
        scale=1.0 / bandwidth, size=(1, 3, feature_count)
    ).astype(np.float32)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(1, feature_count)).astype(
        np.float32
    )
    features = random_fourier_path_features(
        np.concatenate([left, right]),
        frequencies,
        phases,
        device="cpu",
        batch_size=64,
    )
    approximate = float(
        np.sum((features[: left.shape[0]].mean(0) - features[left.shape[0] :].mean(0)) ** 2)
    )

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        squared = np.sum((a[:, None] - b[None]) ** 2, axis=-1)
        return np.exp(-squared / (2.0 * bandwidth**2))

    exact = float(
        kernel(left, left).mean()
        + kernel(right, right).mean()
        - 2.0 * kernel(left, right).mean()
    )
    assert abs(approximate - exact) / exact < 0.08


def test_fitted_path_kernel_can_be_reused_for_additional_centers(
    tmp_path: Path,
) -> None:
    cache = _joint_process_cache()
    fitted = prepare_joint_path_target_data(
        cache,
        horizons_ps=[1.0, 2.0],
        horizon_weights=[1.0, 1.0],
        rff_features_per_bandwidth=128,
        bandwidth_multipliers=[0.5, 1.0, 2.0],
        selection_source_velocity_seeds=[99],
        seed=19,
        rff_device="cpu",
        rff_batch_size=32,
    )
    kernel_path = tmp_path / "path_kernel.npz"
    save_path_kernel(fitted, kernel_path)
    reapplied = prepare_joint_path_target_data_from_kernel(
        cache,
        kernel_path=kernel_path,
        selection_source_velocity_seeds=[99],
        rff_device="cpu",
        rff_batch_size=32,
    )
    np.testing.assert_allclose(
        reapplied.empirical_mean_embedding,
        fitted.empirical_mean_embedding,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        reapplied.target_modes, fitted.target_modes, rtol=1.0e-6, atol=1.0e-6
    )


def test_predictive_atlas_is_invariant_to_rotated_token_offsets() -> None:
    torch.manual_seed(13)
    model = PredictiveAtlas(
        embedding_dim=5,
        descriptor_dim=2,
        conditioning_dim=1,
        hidden_dim=16,
        heads=4,
        blocks=2,
        rbf_dim=8,
        maximum_radius=4.0,
        latent_dim=3,
        decoder_hidden_dim=12,
        target_dim=7,
        dropout=0.0,
    ).eval()
    embeddings = torch.randn(4, 6, 5)
    descriptors = torch.randn(4, 6, 2)
    offsets = torch.randn(4, 6, 3)
    conditioning = torch.randn(4, 1)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    with torch.no_grad():
        original = model(embeddings, descriptors, offsets, conditioning)
        rotated = model(embeddings, descriptors, offsets @ rotation, conditioning)
    torch.testing.assert_close(original[0], rotated[0], rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(original[1], rotated[1], rtol=1.0e-5, atol=1.0e-6)


def test_history_conditioned_atlas_uses_ordered_past_token_embeddings() -> None:
    torch.manual_seed(17)
    model = PredictiveAtlas(
        embedding_dim=5,
        descriptor_dim=2,
        conditioning_dim=1,
        hidden_dim=16,
        heads=4,
        blocks=1,
        rbf_dim=8,
        maximum_radius=4.0,
        latent_dim=3,
        decoder_hidden_dim=12,
        target_dim=7,
        dropout=0.0,
        history_lag_count=2,
    ).eval()
    embeddings = torch.randn(4, 6, 5)
    descriptors = torch.randn(4, 6, 2)
    offsets = torch.randn(4, 6, 3)
    conditioning = torch.randn(4, 1)
    history = torch.randn(4, 2, 6, 5)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    with torch.no_grad():
        original = model(
            embeddings, descriptors, offsets, conditioning, history
        )
        rotated = model(
            embeddings, descriptors, offsets @ rotation, conditioning, history
        )
        reversed_history = model(
            embeddings,
            descriptors,
            offsets,
            conditioning,
            torch.flip(history, dims=(1,)),
        )
    torch.testing.assert_close(original[0], rotated[0], rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(original[1], rotated[1], rtol=1.0e-5, atol=1.0e-6)
    assert not torch.allclose(original[1], reversed_history[1])


def test_pullback_spectrum_has_one_value_per_latent_dimension() -> None:
    model = PredictiveAtlas(
        embedding_dim=3,
        descriptor_dim=1,
        conditioning_dim=1,
        hidden_dim=8,
        heads=2,
        blocks=1,
        rbf_dim=4,
        maximum_radius=2.0,
        latent_dim=2,
        decoder_hidden_dim=6,
        target_dim=5,
        dropout=0.0,
    )
    representations = np.asarray(
        [[-0.5, 0.2], [0.1, -0.4], [0.7, 0.8]], dtype=np.float32
    )
    fitted = FittedPredictiveAtlas(
        model=model,
        embedding_mean=np.zeros(3),
        embedding_scale=np.ones(3),
        descriptor_mean=np.zeros(1),
        descriptor_scale=np.ones(1),
        conditioning_mean=np.zeros(1),
        conditioning_scale=np.ones(1),
        seed=3,
        histories={},
        seed_metrics={},
        predictions_by_seed={3: np.zeros((3, 5), dtype=np.float32)},
        representations_by_seed={3: representations},
    )
    eigenvalues, rank = compute_pullback_spectrum(
        fitted,
        rows=np.arange(3),
        device="cpu",
        batch_size=2,
        relative_eigenvalue_cutoff=1.0e-5,
    )
    assert eigenvalues.shape == (3, 2)
    assert rank.shape == (3,)
    assert np.isfinite(eigenvalues).all()
    assert np.all(eigenvalues >= -1.0e-7)
