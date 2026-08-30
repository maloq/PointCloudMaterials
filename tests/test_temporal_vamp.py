from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data_utils.temporal_lammps_dataset import TemporalLAMMPSDumpDataset
from src.temporal_vamp.data import (
    TemporalPairDataset,
    contiguous_temporal_split,
    event_aligned_frame_interval,
    resolve_lag_frame_offset,
)


def test_event_aligned_frame_interval_includes_bounds_and_clips() -> None:
    times = np.arange(0.0, 603.0, 3.0)
    assert event_aligned_frame_interval(
        times,
        event_time_ps=87.0,
        start_offset_ps=-24.0,
        stop_offset_ps=96.0,
        clip_to_trajectory=True,
    ) == (21, 62)
    assert event_aligned_frame_interval(
        times,
        event_time_ps=570.0,
        start_offset_ps=-24.0,
        stop_offset_ps=96.0,
        clip_to_trajectory=True,
    ) == (182, 201)
from src.temporal_vamp.linear_vamp import LinearVAMP
from src.temporal_vamp.embeddings import (
    EmbeddingCache,
    encode_spatial_context_state,
    extract_embedding_cache,
)
from src.temporal_vamp.evaluation import (
    FutureNeighborCandidateFilter,
    _filtered_neighbor_indices,
    future_neighbor_consistency,
)


def _write_small_lammps_dump(path: Path) -> None:
    frames = []
    atom_ids = np.asarray([3, 1, 4, 2], dtype=np.int64)
    initial = {
        1: np.asarray([1.0, 1.0, 1.0]),
        2: np.asarray([2.0, 1.0, 1.0]),
        3: np.asarray([1.0, 2.0, 1.0]),
        4: np.asarray([9.7, 1.0, 1.0]),
    }
    for frame_index in range(5):
        lines = [
            "ITEM: TIMESTEP",
            str(frame_index * 10),
            "ITEM: NUMBER OF ATOMS",
            "4",
            "ITEM: BOX BOUNDS pp pp pp",
            "0 10",
            "0 10",
            "0 10",
            "ITEM: ATOMS id type x y z",
        ]
        for atom_id in atom_ids.tolist():
            xyz = initial[atom_id] + np.asarray([0.1 * frame_index, 0.0, 0.0])
            xyz %= 10.0
            lines.append(
                f"{atom_id} 1 {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}"
            )
        frames.append("\n".join(lines))
    path.write_text("\n".join(frames) + "\n", encoding="utf-8")


def test_temporal_pair_tracks_same_atom_and_requested_lag(tmp_path: Path) -> None:
    dump_path = tmp_path / "trajectory.lammpstrj"
    _write_small_lammps_dump(dump_path)
    base = TemporalLAMMPSDumpDataset(
        dump_file=dump_path,
        cache_dir=tmp_path / "cache",
        sequence_length=2,
        num_points=3,
        radius=3.0,
        frame_stride=2,
        anchor_frame_indices=[0, 1],
        center_selection_mode="atom_ids",
        center_atom_ids=[2],
        normalize=True,
        center_neighborhoods=True,
        selection_method="closest",
        precompute_neighbor_indices=False,
    )
    pairs = TemporalPairDataset(base, run_id="test_run")

    first = pairs[0]
    assert int(first["atom_id"]) == 2
    assert int(first["frame0"]) == 0
    assert int(first["frame1"]) == 2
    assert int(first["timestep1"] - first["timestep0"]) == 20
    assert first["run_id"] == "test_run"
    np.testing.assert_allclose(first["points0"][0].numpy(), 0.0, atol=1.0e-6)
    np.testing.assert_allclose(first["points1"][0].numpy(), 0.0, atol=1.0e-6)


def test_temporal_pair_builds_deterministic_satellite_context(tmp_path: Path) -> None:
    dump_path = tmp_path / "trajectory.lammpstrj"
    _write_small_lammps_dump(dump_path)
    base = TemporalLAMMPSDumpDataset(
        dump_file=dump_path,
        cache_dir=tmp_path / "cache",
        sequence_length=2,
        num_points=4,
        radius=3.0,
        frame_stride=1,
        anchor_frame_indices=[0],
        center_selection_mode="atom_ids",
        center_atom_ids=[2],
        normalize=True,
        center_neighborhoods=True,
        selection_method="closest",
        precompute_neighbor_indices=False,
        spatial_context_center_count=2,
    )
    pairs = TemporalPairDataset(base, run_id="test_context")
    first = pairs[0]
    repeated = pairs[0]
    assert first["context_points0"].shape == (2, 4, 3)
    assert first["context_offsets0"].shape == (2, 3)
    assert 2 not in first["context_atom_ids0"].tolist()
    torch.testing.assert_close(first["context_points0"], repeated["context_points0"])
    torch.testing.assert_close(first["context_offsets0"], repeated["context_offsets0"])


def test_context_embedding_appends_permutation_invariant_mean_and_std() -> None:
    class FakeEncoder:
        @staticmethod
        def encode(points: torch.Tensor) -> torch.Tensor:
            radii = torch.linalg.vector_norm(points, dim=-1)
            return torch.stack([radii.mean(dim=1), radii.amax(dim=1)], dim=1)

    points = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float32
    )
    context = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    contextual, local = encode_spatial_context_state(
        FakeEncoder(),
        points,
        context_points=context,
        aggregation="mean_std",
        point_cloud_batch_size=1,
    )
    assert contextual.shape == (1, 6)
    torch.testing.assert_close(contextual[:, :2], local)
    permutation = torch.tensor([1, 0])
    permuted, _ = encode_spatial_context_state(
        FakeEncoder(),
        points,
        context_points=context[:, permutation],
        aggregation="mean_std",
        point_cloud_batch_size=2,
    )
    torch.testing.assert_close(contextual, permuted)


def test_context_embedding_cache_preserves_local_baseline(tmp_path: Path) -> None:
    class FakeEncoder:
        device = torch.device("cpu")
        output_dim = 2

        @staticmethod
        def encode(points: torch.Tensor) -> torch.Tensor:
            radii = torch.linalg.vector_norm(points, dim=-1)
            return torch.stack([radii.mean(dim=1), radii.amax(dim=1)], dim=1)

    dump_path = tmp_path / "trajectory.lammpstrj"
    _write_small_lammps_dump(dump_path)
    base = TemporalLAMMPSDumpDataset(
        dump_file=dump_path,
        cache_dir=tmp_path / "trajectory_cache",
        sequence_length=2,
        num_points=4,
        radius=3.0,
        frame_stride=1,
        anchor_frame_indices=[0],
        center_selection_mode="atom_ids",
        center_atom_ids=[2],
        normalize=True,
        center_neighborhoods=True,
        selection_method="closest",
        spatial_context_center_count=2,
    )
    cache = extract_embedding_cache(
        [TemporalPairDataset(base, run_id="context_cache")],
        encoder=FakeEncoder(),
        cache_path=tmp_path / "embeddings",
        cache_spec={"test": "context"},
        batch_size=1,
        point_cloud_batch_size=2,
        num_workers=0,
        force_recompute=False,
        spatial_context_aggregation="mean_std",
    )
    assert cache.z0.shape == (1, 6)
    assert cache.local_z0 is not None and cache.local_z0.shape == (1, 2)
    assert cache.local_z1 is not None and cache.local_z1.shape == (1, 2)
    np.testing.assert_allclose(cache.z0[:, :2], cache.local_z0)
    assert cache.manifest["spatial_context_center_count"] == 2


def test_contiguous_temporal_split_prevents_cross_boundary_pairs() -> None:
    split = contiguous_temporal_split(
        frame_count=100,
        lag_frames=7,
        train_ratio=0.8,
        boundary_gap_frames=2,
    )
    assert int((split.train + 7).max()) < split.boundary_frame - 2
    assert int(split.validation.min()) >= split.boundary_frame + 2
    assert not np.intersect1d(split.train, split.validation).size


def test_physical_lag_requires_exact_uniform_alignment() -> None:
    timesteps = np.arange(0, 100, 10, dtype=np.int64)
    assert resolve_lag_frame_offset(timesteps, lag_timesteps=30) == 3


def test_linear_vamp_recovers_slow_subspace_and_handles_redundancy(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    sample_count = 16000
    slow = np.empty(sample_count + 1, dtype=np.float64)
    fast = np.empty(sample_count + 1, dtype=np.float64)
    slow[0] = rng.normal()
    fast[0] = rng.normal()
    for index in range(sample_count):
        slow[index + 1] = 0.985 * slow[index] + np.sqrt(1.0 - 0.985**2) * rng.normal()
        fast[index + 1] = 0.20 * fast[index] + np.sqrt(1.0 - 0.20**2) * rng.normal()

    mixing = rng.normal(size=(2, 10))
    z0 = np.stack([slow[:-1], fast[:-1]], axis=1) @ mixing
    z1 = np.stack([slow[1:], fast[1:]], axis=1) @ mixing
    z0 += 0.08 * rng.normal(size=z0.shape)
    z1 += 0.08 * rng.normal(size=z1.shape)
    z0 = np.column_stack([z0, z0[:, 0] + z0[:, 1]])
    z1 = np.column_stack([z1, z1[:, 0] + z1[:, 1]])

    model = LinearVAMP(regularization=1.0e-6, eigenvalue_cutoff=1.0e-9).fit(z0, z1)
    dominant = model.left_singular_functions(z0, dimension=1)[:, 0]
    correlation = abs(np.corrcoef(dominant, slow[:-1])[0, 1])
    assert correlation > 0.9
    assert model.singular_values_[0] > 0.9
    assert model.singular_values_[0] > model.singular_values_[1] + 0.4
    assert model.whitening0_.shape[1] < z0.shape[1]

    path = tmp_path / "vamp.npz"
    model.save(path)
    restored = LinearVAMP.load(path)
    np.testing.assert_allclose(
        restored.transform(z0[:100], dimension=2),
        model.transform(z0[:100], dimension=2),
    )


def _matched_neighbor_test_cache(tmp_path: Path) -> EmbeddingCache:
    run_index = np.repeat(np.arange(4, dtype=np.int64), 2)
    relative_times = np.tile(np.asarray([-9.0, 0.0]), 4)
    nucleation_times = np.asarray([100.0, 200.0, 300.0, 400.0])
    time_ps0 = relative_times + nucleation_times[run_index]
    crystallinity = np.asarray([0.10, 0.20, 0.11, 0.50, 0.70, 0.80, 0.71, 0.40])
    values = np.arange(8, dtype=np.float32)[:, None]
    return EmbeddingCache(
        path=tmp_path,
        manifest={
            "run_ids": ["run0", "run1", "run2", "run3"],
            "run_metadata": [
                {"nucleation_time_ps": float(value)} for value in nucleation_times
            ],
        },
        z0=values,
        z1=values,
        atom_id=np.arange(8, dtype=np.int64),
        run_index=run_index,
        frame0=np.tile(np.arange(2, dtype=np.int64), 4),
        frame1=np.tile(np.arange(2, dtype=np.int64) + 1, 4),
        timestep0=np.arange(8, dtype=np.int64),
        timestep1=np.arange(8, dtype=np.int64) + 1,
        coords0=np.zeros((8, 3), dtype=np.float32),
        coords1=np.zeros((8, 3), dtype=np.float32),
        time_ps0=time_ps0,
        time_ps1=time_ps0 + 3.0,
        temperature_K=np.repeat(np.asarray([400.0, 400.0, 450.0, 450.0]), 2),
        crystalline_fraction0=crystallinity,
        crystalline_fraction1=crystallinity,
    )


def test_filtered_neighbors_match_cross_run_temperature_event_time_and_crystallinity(
    tmp_path: Path,
) -> None:
    cache = _matched_neighbor_test_cache(tmp_path)
    candidate_filter = FutureNeighborCandidateFilter(
        exclude_same_run=True,
        match_temperature=True,
        relative_time_tolerance_ps=1.0,
        crystalline_fraction_tolerance=0.05,
    )
    queries, neighbors, random_references, candidate_counts = _filtered_neighbor_indices(
        np.asarray(cache.z0),
        cache,
        query_indices=np.asarray([0, 1], dtype=np.int64),
        neighbors=1,
        candidate_filter=candidate_filter,
        seed=7,
    )
    np.testing.assert_array_equal(queries, np.asarray([0]))
    np.testing.assert_array_equal(neighbors, np.asarray([[2]]))
    np.testing.assert_array_equal(random_references, np.asarray([[2]]))
    np.testing.assert_array_equal(candidate_counts, np.asarray([1, 0]))


def test_matched_future_neighbor_metric_reports_query_coverage(tmp_path: Path) -> None:
    cache = _matched_neighbor_test_cache(tmp_path)
    metrics = future_neighbor_consistency(
        np.asarray(cache.z0),
        np.asarray(cache.z1),
        cache,
        neighbors=1,
        max_queries=0,
        exclude_same_atom=True,
        seed=7,
        candidate_filter=FutureNeighborCandidateFilter(
            exclude_same_run=True,
            match_temperature=True,
            relative_time_tolerance_ps=1.0,
            crystalline_fraction_tolerance=0.05,
        ),
    )
    assert metrics["requested_queries"] == 8
    assert metrics["queries"] == 4
    assert metrics["query_coverage"] == 0.5
