from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.trajectories.lammps import (
    TemporalLAMMPSBinaryTrajectory,
    binary_path_for_dump,
    write_temporal_lammps_binary,
)
from src.data.temporal import TemporalLAMMPSDumpDataset


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


@pytest.mark.parametrize("num_workers", [0, 1])
def test_temporal_datamodule_lifecycle_and_batched_identity(
    tmp_path, num_workers
):
    import torch
    from omegaconf import OmegaConf

    from src.data.data_modules.temporal_lammps import (
        TemporalLAMMPSDataModule,
    )

    source = tmp_path / "tiny.lammpstrj"
    frames = []
    for frame in range(6):
        rows = "\n".join(
            f"{atom + 1} 1 {1 + atom * 0.3 + frame * 0.01} 1 1"
            for atom in range(4)
        )
        frames.append(
            f"ITEM: TIMESTEP\n{frame * 10}\nITEM: NUMBER OF ATOMS\n4\n"
            "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
            f"ITEM: ATOMS id type x y z\n{rows}\n"
        )
    source.write_text("".join(frames))
    cfg = OmegaConf.create({
        "batch_size": 3, "num_workers": num_workers, "max_samples": 1,
        "data": {
            "kind": "temporal_lammps", "dump_file": str(source),
            "cache_dir": str(tmp_path / "cache"), "sequence_length": 2,
            "num_points": 3, "radius": 2.0, "train_ratio": 0.6,
            "split_seed": 42, "center_selection_mode": "atom_ids",
            "center_atom_ids": [1, 3],
        },
    })
    dm = TemporalLAMMPSDataModule(cfg)
    dm.setup("fit")
    train = dm.train_dataset
    val = dm.val_dataset
    for stage in ("fit", "validate", "test"):
        dm.setup(stage)
        assert dm.train_dataset is train
        assert dm.val_dataset is val
        assert dm.test_dataset is val
    indices = [3, 0, 2, 0]
    batch = val.__getitems__(indices)
    for key, value in batch.items():
        if key == "source_path":
            assert value == [val[i][key] for i in indices]
            continue
        expected = torch.stack([val[i][key] for i in indices])
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    loader = dm.val_dataloader()
    if num_workers:
        loader.multiprocessing_context = "spawn"
    batches = list(loader)
    assert [len(batch["points"]) for batch in batches] == [3, 1]
    for key, value in batches[0].items():
        if key == "source_path":
            assert value == [val[i][key] for i in range(3)]
            continue
        expected = torch.stack([val[i][key] for i in range(3)])
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert dm.state_dict() == {}
