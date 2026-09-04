from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data_utils.temporal_lammps_binary import (
    TemporalLAMMPSBinaryTrajectory,
    binary_path_for_dump,
    write_temporal_lammps_binary,
)
from src.data_utils.temporal_lammps_dataset import TemporalLAMMPSDumpDataset


def test_binary_replaces_missing_text_path_for_dataset(tmp_path: Path) -> None:
    source = tmp_path / "trajectory.lammpstrj"
    source.write_text("migration provenance only\n", encoding="utf-8")
    source_stat = source.stat()
    positions = np.asarray(
        [
            [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [9.7, 1.0, 1.0]],
            [[1.1, 1.0, 1.0], [2.1, 1.0, 1.0], [1.1, 2.0, 1.0], [9.8, 1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    target = binary_path_for_dump(source)
    binary = write_temporal_lammps_binary(
        target,
        positions=positions,
        timesteps=np.asarray([5, 15], dtype=np.int64),
        box_low=np.zeros((2, 3), dtype=np.float32),
        box_high=np.full((2, 3), 10.0, dtype=np.float32),
        atom_ids=np.arange(1, 5, dtype=np.int64),
        atom_types=np.ones(4, dtype=np.int32),
        atom_columns=("id", "type", "x", "y", "z"),
        source={
            "trajectory_lammpstrj": {
                "path": str(source.resolve()),
                "size_bytes": source_stat.st_size,
                "mtime_ns": source_stat.st_mtime_ns,
                "sha256": "0" * 64,
                "deleted": False,
                "deleted_at": None,
            },
            "coordinate_archive": {
                "path": str((tmp_path / "trajectory.npz").resolve()),
                "size_bytes": 0,
                "sha256": "1" * 64,
                "positions_float32_sha256": "2" * 64,
            },
        },
        provenance={"test": True},
    )
    assert binary.verify_checksums()["positions"] == binary.manifest["arrays"][
        "positions"
    ]["sha256"]
    source.unlink()

    scan = TemporalLAMMPSDumpDataset.scan_dump_file(source)
    assert scan.frame_count == 2
    assert scan.num_atoms == 4
    assert scan.timesteps.tolist() == [5, 15]
    frame, box_lengths, timestep = TemporalLAMMPSDumpDataset.load_dump_frame_positions(
        source, frame_index=1
    )
    np.testing.assert_array_equal(frame, positions[1])
    np.testing.assert_array_equal(box_lengths, np.full(3, 10.0, dtype=np.float32))
    assert timestep == 15

    dataset = TemporalLAMMPSDumpDataset(
        dump_file=source,
        cache_dir=tmp_path / "neighbor_cache",
        sequence_length=2,
        num_points=3,
        radius=3.0,
        center_selection_mode="atom_ids",
        center_atom_ids=[2],
        normalize=True,
        center_neighborhoods=True,
        precompute_neighbor_indices=False,
    )
    assert isinstance(dataset._binary_trajectory, TemporalLAMMPSBinaryTrajectory)
    assert dataset.frame_count == 2
    assert dataset.num_atoms == 4
    np.testing.assert_array_equal(dataset.positions, positions)


def test_float16_binary_decodes_to_valid_periodic_float32_positions(tmp_path: Path) -> None:
    source = tmp_path / "trajectory.lammpstrj"
    source.write_text("conversion provenance only\n", encoding="utf-8")
    positions = np.asarray(
        [
            [[1.001, 1.0, 1.0], [2.003, 1.0, 1.0], [1.0, 2.005, 1.0], [9.999, 1.0, 1.0]],
            [[1.101, 1.0, 1.0], [2.103, 1.0, 1.0], [1.1, 2.105, 1.0], [9.998, 1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    target = binary_path_for_dump(source, storage_dtype="float16")
    binary = write_temporal_lammps_binary(
        target,
        positions=positions,
        timesteps=np.asarray([5, 15], dtype=np.int64),
        box_low=np.zeros((2, 3), dtype=np.float32),
        box_high=np.full((2, 3), 10.0, dtype=np.float32),
        atom_ids=np.arange(1, 5, dtype=np.int64),
        atom_types=np.ones(4, dtype=np.int32),
        atom_columns=("id", "type", "x", "y", "z"),
        source={"trajectory_lammpstrj": str(source.resolve())},
        provenance={"test": True},
        storage_dtype="float16",
    )
    assert binary.positions.dtype == np.dtype("float16")
    assert binary.manifest["storage_dtype"] == "float16"
    assert binary.manifest["quantization"]["maximum_absolute_error_A"] > 0.0
    binary.verify_checksums()

    frame, box_lengths, timestep = TemporalLAMMPSDumpDataset.load_dump_frame_positions(
        target, frame_index=0
    )
    assert frame.dtype == np.dtype("float32")
    assert np.all(frame >= 0.0)
    assert np.all(frame < box_lengths[None, :])
    assert timestep == 5
