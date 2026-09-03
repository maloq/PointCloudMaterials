from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data_utils.shooting_binary import (
    ShootingBinaryTrajectory,
    binary_directory_sizes,
    compose_shooting_binary_trajectories,
    convert_shooting_trajectory,
)
from src.data_utils.shooting_binary_dataset import (
    ShootingBinaryEnvironmentDataset,
    make_shooting_environment_loader,
)
from src.data_utils.shooting_dataset import (
    load_shooting_campaign_snapshot,
    resolve_shooting_trajectory_path,
)
from src.data_utils.shooting_text_conversion import (
    load_lammps_shooting_frames_for_conversion,
)
from scripts.migrate_lammps_shooting_float32 import _migrate_campaign


def _write_shooting_dump(
    path: Path, frame_specs: tuple[tuple[int, float], ...] = ((0, 0.0), (100, 0.1))
) -> None:
    frames: list[str] = []
    for timestep, shift in frame_specs:
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
                f"3 1 {9.8 + shift} 1 1 0 0 3",
                f"1 1 {0.2 + shift} 1 1 1 0 0",
                f"4 1 {5.0 + shift} 5 5 0 0 4",
                f"2 1 {1.2 + shift} 1 1 0 2 0",
            ]
        )
    path.write_text("\n".join(frames) + "\n", encoding="ascii")


def test_shooting_binary_round_trip_and_float16_decode(tmp_path: Path) -> None:
    source = tmp_path / "trajectory.lammpstrj"
    _write_shooting_dump(source)
    reference = load_lammps_shooting_frames_for_conversion(
        source, timesteps=[0, 100], atom_count=4
    )

    float32_path = tmp_path / "trajectory_float32"
    float16_path = tmp_path / "trajectory_float16"
    float32 = convert_shooting_trajectory(
        source,
        float32_path,
        timesteps=[0, 100],
        atom_count=4,
        storage_dtype="float32",
        provenance={"test": "round_trip"},
    )
    float16 = convert_shooting_trajectory(
        source,
        float16_path,
        timesteps=[0, 100],
        atom_count=4,
        storage_dtype="float16",
        provenance={"test": "round_trip"},
    )

    assert isinstance(float32, ShootingBinaryTrajectory)
    assert float32.positions.dtype == np.float32
    assert float16.positions.dtype == np.float16
    assert float32.manifest["source"]["size_bytes"] == source.stat().st_size
    assert binary_directory_sizes(float32_path)["apparent_bytes"] > 0
    for binary, tolerance in ((float32, 0.0), (float16, 4.0e-3)):
        observed = binary.load_frames([0, 100])
        for timestep in (0, 100):
            np.testing.assert_array_equal(
                observed[timestep].atom_ids, reference[timestep].atom_ids
            )
            np.testing.assert_array_equal(
                observed[timestep].atom_types, reference[timestep].atom_types
            )
            np.testing.assert_allclose(
                observed[timestep].positions,
                reference[timestep].positions,
                rtol=0.0,
                atol=tolerance,
            )
            np.testing.assert_allclose(
                observed[timestep].velocities,
                reference[timestep].velocities,
                rtol=0.0,
                atol=tolerance,
            )
            assert observed[timestep].positions.dtype == np.float32
            assert observed[timestep].velocities.dtype == np.float32

    with pytest.raises(RuntimeError, match="absent from binary shooting trajectory"):
        float32.load_frames([50])
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        convert_shooting_trajectory(
            source,
            float32_path,
            timesteps=[0, 100],
            atom_count=4,
            storage_dtype="float32",
            provenance={},
        )


def test_shooting_binary_composition_prefers_original_restart_frame(
    tmp_path: Path,
) -> None:
    original_text = tmp_path / "original.lammpstrj"
    continuation_text = tmp_path / "continuation.lammpstrj"
    _write_shooting_dump(original_text, ((0, 0.0), (100, 0.1)))
    _write_shooting_dump(continuation_text, ((100, 0.5), (200, 0.2)))
    original = convert_shooting_trajectory(
        original_text,
        tmp_path / "original_binary",
        timesteps=[0, 100],
        atom_count=4,
        storage_dtype="float32",
        provenance={"segment": "original"},
    )
    continuation = convert_shooting_trajectory(
        continuation_text,
        tmp_path / "continuation_binary",
        timesteps=[100, 200],
        atom_count=4,
        storage_dtype="float16",
        provenance={"segment": "continuation"},
    )

    composed = compose_shooting_binary_trajectories(
        [original, continuation],
        tmp_path / "composed",
        timesteps=[0, 100, 200],
        storage_dtype="float16",
        provenance={"test": "restart_composition"},
    )

    assert composed.storage_dtype == np.dtype("float16")
    np.testing.assert_array_equal(composed.timesteps, [0, 100, 200])
    np.testing.assert_allclose(
        composed.load_frames([100])[100].positions,
        original.load_frames([100])[100].positions,
        rtol=0.0,
        atol=4.0e-3,
    )
    composed.verify_checksums()


def test_complete_campaign_migration_deletes_only_branch_root_text(
    tmp_path: Path,
) -> None:
    parents: list[dict[str, object]] = []
    branches: list[dict[str, object]] = []
    for index, split in enumerate(("train", "validation")):
        parent = {
            "parent_index": index,
            "parent_id": f"parent_{index}",
            "source_index": index,
            "source_run_id": f"source_{index}",
            "source_split": split,
            "source_velocity_seed": 10 + index,
            "temperature_K": 400.0,
            "phase": "pre_nucleation_3ps",
            "nucleation_time_ps": 10.0,
            "parent_offset_ps": -3.0,
            "source_frame_index": 7,
            "source_frame_step": 700,
            "source_frame_time_ps": 7.0,
            "source_crystalline_fraction": 0.1,
            "source_largest_crystalline_cluster_atoms": 4,
            "data_sha256": f"sha256-{index}",
            "data_file": f"parents/parent_{index}/parent.lammps.data",
        }
        branch = {
            **parent,
            "branch_index": index,
            "branch_id": f"branch_{index}",
            "branch_dir": f"branches/branch_{index}",
            "shot_index": 0,
            "velocity_seed": 100 + index,
            "thermostat_seed": 200 + index,
        }
        parents.append(parent)
        branches.append(branch)
    manifest = {
        "campaign_type": "position_conditioned_langevin_nvt_shooting",
        "atom_count": 4,
        "counts": {"parents": 2, "branches": 2},
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
    (tmp_path / "branches").mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for branch in branches:
        branch_dir = tmp_path / str(branch["branch_dir"])
        branch_dir.mkdir()
        trajectory = branch_dir / "trajectory.lammpstrj"
        _write_shooting_dump(trajectory)
        restart = branch_dir / "final.restart.bin"
        restart.write_bytes(b"restart")
        outcome = {
            **branch,
            "state": "complete",
            "completed_at": "2026-09-01T00:00:00+00:00",
            "elapsed_seconds": 1.0,
            "frame_count": 2,
            "first_timestep": 0,
            "last_timestep": 100,
            "trajectory_size_bytes": trajectory.stat().st_size,
            "restart_size_bytes": restart.stat().st_size,
        }
        (branch_dir / "outcome.json").write_text(
            json.dumps(outcome), encoding="utf-8"
        )
    interrupted = (
        tmp_path / "branches" / "branch_0" / "interrupted_attempt_test"
    )
    interrupted.mkdir()
    archived_partial = interrupted / "trajectory.lammpstrj"
    archived_partial.write_bytes(b"partial archive must remain")

    with pytest.raises(RuntimeError, match="has not been migrated"):
        resolve_shooting_trajectory_path(tmp_path, branches[0])
    report = _migrate_campaign(tmp_path, workers=2)
    assert report["migrated_complete_branch_count"] == 2
    assert report["incomplete_branch_count"] == 0
    for branch in branches:
        branch_dir = tmp_path / str(branch["branch_dir"])
        assert not (branch_dir / "trajectory.lammpstrj").exists()
        binary = ShootingBinaryTrajectory.load(
            branch_dir / "trajectory_binary_float32"
        )
        binary.verify_checksums()
        frames = binary.load_frames([0, 100])
        assert tuple(frames) == (0, 100)
        outcome = json.loads(
            (branch_dir / "outcome.json").read_text(encoding="utf-8")
        )
        assert outcome["trajectory_artifact"]["source_lammpstrj"]["deleted"]
    assert archived_partial.read_bytes() == b"partial archive must remain"

    snapshot = load_shooting_campaign_snapshot(
        tmp_path,
        temperatures_K=[400.0],
        minimum_complete_branches_per_parent=1,
    )
    assert len(snapshot.branches) == 2
    environment_dataset = ShootingBinaryEnvironmentDataset(
        snapshot,
        branches=snapshot.branches,
        timesteps=[0, 100],
        center_atom_ids=np.asarray([1, 3], dtype=np.int64),
        num_points=3,
        radius=2.0,
        spatial_context_center_count=1,
    )
    environment_loader = make_shooting_environment_loader(
        environment_dataset,
        batch_size=2,
        num_workers=2,
        pin_memory=False,
    )
    environment_batch = next(iter(environment_loader))
    assert environment_batch["points"].shape == (2, 2, 2, 3, 3)
    assert environment_batch["context_points"].shape == (2, 2, 2, 1, 3, 3)
    assert environment_batch["center_positions"].shape == (2, 2, 2, 3)
    assert environment_batch["box_lengths"].shape == (2, 2, 3)
    assert environment_batch["context_center_offsets"].shape == (2, 2, 2, 1, 3)
    assert environment_batch["context_center_atom_ids"].shape == (2, 2, 2, 1)
    assert environment_batch["branch_id"] == ["branch_0", "branch_1"]
    assert environment_batch["source_split"] == ["train", "validation"]
    assert environment_batch["temperature_K"].tolist() == [400.0, 400.0]
    np.testing.assert_array_equal(
        environment_batch["timesteps"].numpy(),
        np.asarray([[0, 100], [0, 100]], dtype=np.int64),
    )
    repeated = environment_dataset[0]
    torch.testing.assert_close(environment_batch["points"][0], repeated["points"])
    repeated = _migrate_campaign(tmp_path)
    assert repeated["migrated_complete_branch_count"] == 2
    assert repeated["migrated_now_count"] == 0
