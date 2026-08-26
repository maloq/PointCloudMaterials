from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data_utils.temporal_lammps_dataset import TemporalLAMMPSDumpDataset
from src.temporal_vamp.data import (
    TemporalPairDataset,
    contiguous_temporal_split,
    resolve_lag_frame_offset,
)
from src.temporal_vamp.linear_vamp import LinearVAMP


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
