from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.data_utils.shooting_dataset import (
    build_periodic_environment_batch,
    load_shooting_campaign_snapshot,
    load_shooting_campaigns_snapshot,
)
from src.data_utils.shooting_text_conversion import (
    load_lammps_shooting_frames_for_conversion,
)
from src.temporal_vamp.shooting_embeddings import (
    ShootingEmbeddingCache,
    extract_shooting_embedding_cache,
)
from src.temporal_vamp.shooting_ablation import compute_dynamic_future_targets
from src.temporal_vamp.shooting_context import ShootingContextTokenCache
from src.temporal_vamp.shooting_distribution import prepare_distributional_target_data
from src.temporal_vamp.shooting_predictor import (
    evaluate_shooting_predictor,
    fit_shooting_predictive_bottleneck,
)
from src.temporal_vamp.shooting_multiscale import build_multiscale_feature_variants
from src.temporal_vamp.shooting_spatial import SpatialContextTransformer
from scripts.migrate_lammps_shooting_float32 import _migrate_campaign


def _write_shooting_dump(path: Path) -> None:
    frames = []
    for timestep, shift in ((0, 0.0), (100, 0.1)):
        atom_three_z = 1.0 if timestep == 0 else -3.0e-7
        frames.extend(
            [
                "ITEM: TIMESTEP",
                str(timestep),
                "ITEM: NUMBER OF ATOMS",
                "4",
                "ITEM: BOX BOUNDS pp pp pp",
                "0 10",
                "0 10",
                "0 10",
                "ITEM: ATOMS id type x y z vx vy vz",
                f"3 1 {9.8 + shift} 1 {atom_three_z} 0 0 3",
                f"1 1 {0.2 + shift} 1 1 1 0 0",
                f"4 1 {5.0 + shift} 5 5 0 0 4",
                f"2 1 {1.2 + shift} 1 1 0 2 0",
            ]
        )
    path.write_text("\n".join(frames) + "\n", encoding="ascii")


def _write_complete_branch(
    root: Path,
    branch: dict[str, object],
    *,
    frame_count: int = 2,
) -> None:
    branch_dir = root / str(branch["branch_dir"])
    branch_dir.mkdir(parents=True)
    trajectory = branch_dir / "trajectory.lammpstrj"
    _write_shooting_dump(trajectory)
    restart = branch_dir / "final.restart.bin"
    restart.write_bytes(b"restart")
    outcome = {
        **branch,
        "state": "complete",
        "frame_count": frame_count,
        "first_timestep": 0,
        "last_timestep": 100,
        "trajectory_size_bytes": trajectory.stat().st_size,
        "restart_size_bytes": restart.stat().st_size,
    }
    (branch_dir / "outcome.json").write_text(json.dumps(outcome), encoding="utf-8")


def _small_campaign(root: Path, *, branch_seed_offset: int = 0) -> None:
    parents = []
    branches = []
    for parent_index, split in enumerate(("train", "validation")):
        parent_id = f"parent_{parent_index}"
        parents.append(
            {
                "parent_index": parent_index,
                "parent_id": parent_id,
                "source_index": parent_index,
                "source_run_id": f"source_{parent_index}",
                "source_split": split,
                "source_velocity_seed": 10 + parent_index,
                "temperature_K": 400.0,
                "phase": "pre_nucleation_3ps",
                "nucleation_time_ps": 10.0,
                "parent_offset_ps": -3.0,
                "source_frame_index": 7,
                "source_frame_step": 700,
                "source_frame_time_ps": 7.0,
                "source_crystalline_fraction": 0.1,
                "source_largest_crystalline_cluster_atoms": 4,
                "data_sha256": f"sha256-{parent_index}",
                "data_file": f"parents/{parent_id}/parent.lammps.data",
            }
        )
        for shot_index in range(2):
            branch_index = len(branches)
            branch_id = f"branch_{branch_index}"
            branches.append(
                {
                    "branch_index": branch_index,
                    "branch_id": branch_id,
                    "branch_dir": f"branches/{branch_id}",
                    "parent_index": parent_index,
                    "parent_id": parent_id,
                    "source_run_id": f"source_{parent_index}",
                    "source_split": split,
                    "source_velocity_seed": 10 + parent_index,
                    "temperature_K": 400.0,
                    "phase": "pre_nucleation_3ps",
                    "shot_index": shot_index,
                    "velocity_seed": 100 + branch_seed_offset + branch_index,
                    "thermostat_seed": 200 + branch_seed_offset + branch_index,
                }
            )
    manifest = {
        "campaign_type": "position_conditioned_langevin_nvt_shooting",
        "atom_count": 4,
        "counts": {"parents": 2, "branches": 4},
        "protocol": {
            "dump_columns": ["id", "type", "x", "y", "z", "vx", "vy", "vz"],
            "expected_frame_count": 2,
            "run_steps": 100,
            "sample_interval_steps": 100,
            "timestep_fs": 3.0,
        },
        "parents": parents,
        "branches": branches,
    }
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for branch in branches:
        _write_complete_branch(root, branch)
    (root / "branches" / "branch_0" / "interrupted_attempt_job_utc").mkdir()


def test_shooting_snapshot_accepts_only_strict_complete_parent_ensembles(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    _small_campaign(root)
    snapshot = load_shooting_campaign_snapshot(
        root,
        temperatures_K=[400.0],
        minimum_complete_branches_per_parent=2,
    )
    assert len(snapshot.parents) == 2
    assert len(snapshot.branches) == 4
    assert snapshot.complete_outcome_count == 4
    assert {parent["source_split"] for parent in snapshot.parents} == {
        "train",
        "validation",
    }
    running_path = root / "branches" / "branch_3" / "outcome.json"
    running = json.loads(running_path.read_text(encoding="utf-8"))
    running["state"] = "running"
    running_path.write_text(json.dumps(running), encoding="utf-8")
    partial_snapshot = load_shooting_campaign_snapshot(
        root,
        temperatures_K=[400.0],
        minimum_complete_branches_per_parent=1,
    )
    assert len(partial_snapshot.branches) == 3
    assert partial_snapshot.complete_outcome_count == 3
    assert partial_snapshot.ignored_incomplete_count == 1


def test_shooting_snapshot_merges_matching_independent_campaigns(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _small_campaign(first)
    _small_campaign(second, branch_seed_offset=1000)
    _migrate_campaign(first)
    _migrate_campaign(second)
    snapshot = load_shooting_campaigns_snapshot(
        [first, second],
        temperatures_K=[400.0],
        minimum_complete_branches_per_parent=4,
    )
    assert len(snapshot.campaign_roots) == 2
    assert len(snapshot.parents) == 2
    assert len(snapshot.branches) == 8
    assert len({branch["branch_uid"] for branch in snapshot.branches}) == 8
    assert {
        Path(branch["campaign_root"]) for branch in snapshot.branches
    } == {first.resolve(), second.resolve()}

    class FakeEncoder:
        output_dim = 2
        repeats = 1
        seed = 5
        representation_source = "encoder"
        device = torch.device("cpu")

        def __init__(self, checkpoint_path: Path) -> None:
            self.checkpoint_path = checkpoint_path

        @staticmethod
        def encode(points: torch.Tensor) -> torch.Tensor:
            radii = torch.linalg.vector_norm(points, dim=-1)
            return torch.stack((radii.mean(dim=1), radii.amax(dim=1)), dim=1)

    checkpoint = tmp_path / "fake.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    cache = extract_shooting_embedding_cache(
        snapshot,
        encoder=FakeEncoder(checkpoint),
        cache_path=tmp_path / "merged_embeddings",
        horizons_ps=[0.3],
        center_atom_count=2,
        center_selection_seed=3,
        num_points=3,
        radius=2.0,
        spatial_context_center_count=1,
        spatial_context_aggregation="mean_std",
        point_cloud_batch_size=2,
        environment_batch_size=2,
        environment_num_workers=2,
        force_recompute=False,
    )
    assert cache.parent_z.shape == (2, 2, 6)
    assert cache.future_z.shape == (8, 1, 2, 2)
    assert len(list((tmp_path / "merged_embeddings_shards" / "branches").glob("*.npz"))) == 8


def test_selected_shooting_frames_preserve_velocities_and_pbc(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.lammpstrj"
    _write_shooting_dump(path)
    frames = load_lammps_shooting_frames_for_conversion(
        path, timesteps=[0, 100], atom_count=4
    )
    assert list(frames) == [0, 100]
    np.testing.assert_allclose(frames[0].velocities[2], [0.0, 0.0, 3.0])
    assert 0.0 <= frames[100].positions[2, 2] < frames[100].box_lengths[2]
    environments = build_periodic_environment_batch(
        frames[0],
        center_atom_ids=np.asarray([1]),
        num_points=3,
        radius=2.0,
        spatial_context_center_count=1,
    )
    assert environments.points.shape == (1, 3, 3)
    assert environments.context_points is not None
    assert environments.context_points.shape == (1, 1, 3, 3)
    assert environments.context_center_offsets is not None
    assert environments.context_center_offsets.shape == (1, 1, 3)
    assert environments.context_center_atom_ids is not None
    assert environments.context_center_atom_ids.shape == (1, 1)
    distances = np.sort(np.linalg.norm(environments.points.numpy()[0], axis=1))
    np.testing.assert_allclose(distances, [0.0, 0.2, 0.5], atol=1.0e-6)


def test_predictive_bottleneck_learns_multi_future_conditional_mean(
    tmp_path: Path,
) -> None:
    torch.set_num_threads(1)
    rng = np.random.default_rng(5)
    parent_count = 12
    center_count = 16
    horizon_count = 2
    branches_per_parent = 4
    input_dim = 12
    future_dim = 8
    latent = rng.normal(size=(parent_count, center_count, 3))
    input_mix = rng.normal(size=(3, input_dim))
    future_mix = rng.normal(size=(horizon_count, 3, future_dim))
    parent_z = latent @ input_mix + 0.02 * rng.normal(
        size=(parent_count, center_count, input_dim)
    )
    future_z = np.empty(
        (
            parent_count * branches_per_parent,
            horizon_count,
            center_count,
            future_dim,
        ),
        dtype=np.float32,
    )
    branch_parent_index = np.repeat(np.arange(parent_count), branches_per_parent)
    for branch_index, parent_index in enumerate(branch_parent_index):
        for horizon_index in range(horizon_count):
            future_z[branch_index, horizon_index] = (
                latent[parent_index] @ future_mix[horizon_index]
                + 0.10 * rng.normal(size=(center_count, future_dim))
            )
    parents = []
    for parent_index in range(parent_count):
        if parent_index < 8:
            split = "train"
            seed = parent_index
        else:
            split = "validation"
            seed = parent_index
        parents.append(
            {
                "parent_id": f"p{parent_index}",
                "source_run_id": f"r{parent_index}",
                "source_split": split,
                "source_velocity_seed": seed,
                "temperature_K": 400.0,
                "phase": "pre_nucleation_3ps",
            }
        )
    cache = ShootingEmbeddingCache(
        path=tmp_path,
        manifest={"snapshot": {"parents": parents}},
        parent_z=parent_z.astype(np.float32),
        parent_local_z=parent_z[..., :future_dim].astype(np.float32),
        parent_coords=np.zeros((parent_count, center_count, 3), dtype=np.float32),
        future_z=future_z,
        branch_parent_index=branch_parent_index.astype(np.int32),
        atom_ids=np.arange(1, center_count + 1, dtype=np.int64),
        horizons_ps=np.asarray([6.0, 12.0]),
    )
    fitted = fit_shooting_predictive_bottleneck(
        cache,
        device="cpu",
        hidden_dim=32,
        bottleneck_dim=4,
        target_pca_dim=4,
        input_pca_dim=4,
        dropout=0.0,
        learning_rate=3.0e-3,
        weight_decay=1.0e-5,
        geometry_weight=0.1,
        batch_size=64,
        maximum_epochs=200,
        patience=30,
        seeds=[7],
        selection_source_velocity_seeds=[7],
    )
    assert fitted.seed == 7
    assert fitted.seed_metrics[7]["validation_prediction_mse"] < 0.20
    metrics, _ = evaluate_shooting_predictor(
        fitted,
        cache,
        device="cpu",
        ridge_alphas=[1.0],
        neighbors=1,
        seed=7,
    )
    assert set(metrics["prediction"]["neural"]["by_horizon"]["validation"]) == {
        "6ps",
        "12ps",
    }
    assert metrics["ensemble_future_variance"]["6ps"]["between_fraction"] > 0.95


def test_dynamic_future_target_removes_linear_static_persistence() -> None:
    rng = np.random.default_rng(81)
    parent_count = 9
    center_count = 12
    embedding_dim = 5
    horizon_count = 2
    current = rng.normal(size=(parent_count, center_count, embedding_dim))
    transforms = rng.normal(size=(horizon_count, embedding_dim, embedding_dim))
    future = np.stack(
        [current @ transforms[horizon] for horizon in range(horizon_count)],
        axis=1,
    )
    optimization_rows = np.arange(0, 6 * center_count, dtype=np.int64)
    selection_rows = np.arange(6 * center_count, 7 * center_count, dtype=np.int64)
    targets, diagnostics = compute_dynamic_future_targets(
        current,
        future,
        optimization_rows=optimization_rows,
        selection_rows=selection_rows,
        ridge_alphas=[1.0e-8, 1.0],
    )
    assert targets["mean_delta"].shape == (
        parent_count * center_count,
        horizon_count,
        embedding_dim,
    )
    assert targets["linear_residual"].shape == targets["mean_delta"].shape
    assert np.mean(targets["linear_residual"] ** 2) < 1.0e-12
    assert {
        values["selected_alpha"]
        for values in diagnostics["ridge_by_horizon"].values()
    } == {1.0e-8}


def test_multiscale_context_features_are_radial_and_rotation_invariant(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(29)
    parent_count, center_count, context_count, embedding_dim = 2, 3, 2, 4
    local = rng.normal(
        size=(parent_count, center_count, embedding_dim)
    ).astype(np.float32)
    base = ShootingEmbeddingCache(
        path=tmp_path / "base",
        manifest={},
        parent_z=np.concatenate([local, local, np.zeros_like(local)], axis=-1),
        parent_local_z=local,
        parent_coords=np.zeros((parent_count, center_count, 3), dtype=np.float32),
        future_z=np.zeros(
            (parent_count, 1, center_count, embedding_dim), dtype=np.float32
        ),
        branch_parent_index=np.arange(parent_count, dtype=np.int32),
        atom_ids=np.arange(1, center_count + 1, dtype=np.int64),
        horizons_ps=np.asarray([6.0]),
    )
    satellite_z = rng.normal(
        size=(parent_count, center_count, context_count, embedding_dim)
    ).astype(np.float32)
    offsets = rng.normal(
        size=(parent_count, center_count, context_count, 3)
    ).astype(np.float32)
    central_descriptors = rng.normal(
        size=(parent_count, center_count, 3)
    ).astype(np.float32)
    satellite_descriptors = rng.normal(
        size=(parent_count, center_count, context_count, 3)
    ).astype(np.float32)
    context = ShootingContextTokenCache(
        path=tmp_path / "context",
        manifest={},
        satellite_z=satellite_z,
        satellite_offsets=offsets,
        central_descriptors=central_descriptors,
        satellite_descriptors=satellite_descriptors,
    )
    features = build_multiscale_feature_variants(
        base, context, radial_scales_angstrom=[1.0, 2.0]
    )
    assert features["multiscale_context"].shape == (
        parent_count * center_count,
        41,
    )
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotated_context = ShootingContextTokenCache(
        path=context.path,
        manifest={},
        satellite_z=satellite_z,
        satellite_offsets=(offsets @ rotation).astype(np.float32),
        central_descriptors=central_descriptors,
        satellite_descriptors=satellite_descriptors,
    )
    rotated = build_multiscale_feature_variants(
        base, rotated_context, radial_scales_angstrom=[1.0, 2.0]
    )
    np.testing.assert_allclose(
        features["multiscale_context"],
        rotated["multiscale_context"],
        atol=1.0e-6,
    )


def test_spatial_context_transformer_is_rotation_and_satellite_order_invariant() -> None:
    torch.manual_seed(37)
    model = SpatialContextTransformer(
        embedding_dim=5,
        descriptor_dim=3,
        hidden_dim=16,
        heads=4,
        blocks=2,
        rbf_dim=8,
        maximum_radius=4.0,
        representation_dim=6,
        target_dim=4,
        dropout=0.0,
    ).eval()
    embeddings = torch.randn(3, 7, 5)
    descriptors = torch.randn(3, 7, 3)
    offsets = torch.randn(3, 7, 3)
    offsets[:, 0] = 0.0
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    with torch.no_grad():
        reference = model.encode(embeddings, descriptors, offsets)
        rotated = model.encode(embeddings, descriptors, offsets @ rotation)
        permutation = torch.tensor([0, 4, 2, 6, 1, 5, 3])
        permuted = model.encode(
            embeddings[:, permutation],
            descriptors[:, permutation],
            offsets[:, permutation],
        )
    torch.testing.assert_close(reference, rotated, atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(reference, permuted, atol=1.0e-5, rtol=1.0e-5)


def test_distributional_target_distinguishes_equal_mean_future_spreads(
    tmp_path: Path,
) -> None:
    parent_count, center_count, embedding_dim, branches_per_parent = 6, 3, 4, 4
    parents = [
        {
            "parent_id": f"parent_{index}",
            "source_run_id": f"run_{index}",
            "source_split": "train" if index < 3 else "validation",
            "source_velocity_seed": 99 if index == 2 else index,
            "temperature_K": 400.0,
            "phase": "pre_nucleation_3ps",
        }
        for index in range(parent_count)
    ]
    branch_parent = np.repeat(
        np.arange(parent_count, dtype=np.int32), branches_per_parent
    )
    future = np.zeros(
        (
            parent_count * branches_per_parent,
            1,
            center_count,
            embedding_dim,
        ),
        dtype=np.float32,
    )
    signs = np.asarray([-1.0, 1.0, -1.0, 1.0], dtype=np.float32)
    for branch_index, parent_index in enumerate(branch_parent.tolist()):
        local_shot = branch_index % branches_per_parent
        future[branch_index, 0, :, 0] = (
            signs[local_shot] * (1.0 + 0.3 * parent_index)
        )
        future[branch_index, 0, :, 1] = signs[local_shot] * np.arange(
            1, center_count + 1
        )
    current = np.zeros(
        (parent_count, center_count, embedding_dim), dtype=np.float32
    )
    cache = ShootingEmbeddingCache(
        path=tmp_path,
        manifest={"snapshot": {"parents": parents}},
        parent_z=np.concatenate([current, current, current], axis=-1),
        parent_local_z=current,
        parent_coords=np.zeros((parent_count, center_count, 3), dtype=np.float32),
        future_z=future,
        branch_parent_index=branch_parent,
        atom_ids=np.arange(1, center_count + 1, dtype=np.int64),
        horizons_ps=np.asarray([6.0]),
    )
    targets = prepare_distributional_target_data(
        cache,
        horizons_ps=[6.0],
        change_pca_dim=2,
        rff_features_per_bandwidth=8,
        bandwidth_multipliers=[0.5, 1.0],
        selection_source_velocity_seeds=[99],
        seed=13,
    )
    signatures = targets.distribution_signature.reshape(
        parent_count, center_count, 1, -1
    )
    assert signatures.shape[-1] == 16
    assert np.linalg.norm(signatures[0] - signatures[1]) > 0.05
