from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data_utils.temporal_binary_context_dataset import TemporalBinaryContextDataset
from src.data_utils.temporal_lammps_binary import write_temporal_lammps_binary
from src.temporal_vamp.simulation_catalog import CatalogEntry, SimulationMetadata


def _metadata(*, seed: int, frame_count: int) -> SimulationMetadata:
    return SimulationMetadata(
        campaign_name="test",
        replica_name=f"seed_{seed}",
        atom_count=64,
        temperature_K=450.0,
        pressure_GPa=0.0,
        timestep_fs=1000.0,
        equilibration_steps=0,
        measurement_steps=frame_count - 1,
        sample_interval_steps=1,
        sample_interval_ps=1.0,
        frame_count=frame_count,
        first_dump_timestep=0,
        last_dump_timestep=frame_count - 1,
        velocity_seed=seed,
        crystal_seed=None,
        boundary_conditions=("p", "p", "p"),
        ensemble="npt",
        potential_name="test",
        potential_library_sha256="a" * 64,
        potential_parameter_sha256="b" * 64,
        prepared_liquid_sha256=None,
        nucleation_observed=False,
        nucleation_time_ps=None,
        initial_crystalline_fraction=0.0,
        final_crystalline_fraction=0.0,
        progress_steps=tuple(range(frame_count)),
        progress_times_ps=tuple(float(value) for value in range(frame_count)),
        structure_names=(),
        structure_fractions=tuple(() for _ in range(frame_count)),
        crystalline_fraction=tuple(0.0 for _ in range(frame_count)),
        crystalline_cluster_count=tuple(0 for _ in range(frame_count)),
        largest_crystalline_cluster_atoms=tuple(0 for _ in range(frame_count)),
    )


def _entry(tmp_path: Path, *, seed: int) -> CatalogEntry:
    frame_count = 6
    axis = 0.5 + 1.5 * np.arange(4, dtype=np.float32)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    positions = np.stack(
        [np.mod(grid + np.float32(frame) * 0.03, 7.0) for frame in range(frame_count)]
    ).astype(np.float32)
    binary = write_temporal_lammps_binary(
        tmp_path / f"seed_{seed}" / "trajectory_binary_float32",
        positions=positions,
        timesteps=np.arange(frame_count, dtype=np.int64),
        box_low=np.zeros((frame_count, 3), dtype=np.float32),
        box_high=np.full((frame_count, 3), 7.0, dtype=np.float32),
        atom_ids=np.arange(1, 65, dtype=np.int64),
        atom_types=np.ones(64, dtype=np.int32),
        atom_columns=("id", "type", "x", "y", "z"),
        source={"test": True},
        provenance={"test": True},
    )
    return CatalogEntry(
        trajectory_path=binary.root,
        run_id=f"seed_{seed}",
        cache_dir=tmp_path / "cache" / f"seed_{seed}",
        metadata=_metadata(seed=seed, frame_count=frame_count),
    )


def test_temporal_binary_context_pairs_same_atoms_at_all_horizons(tmp_path: Path) -> None:
    entries = (_entry(tmp_path, seed=11), _entry(tmp_path, seed=22))
    dataset = TemporalBinaryContextDataset(
        entries,
        center_atom_ids=[1, 10],
        horizons_ps=[1.0, 2.0],
        anchor_stride_frames=2,
        num_points=32,
        radius=6.0,
        context_center_count=3,
        steinhardt_shell_min_neighbors=8,
        steinhardt_shell_max_neighbors=24,
    )
    assert len(dataset) == 4
    sample = dataset[0]
    assert sample["anchor_frame"] == 0
    np.testing.assert_array_equal(sample["future_frames"], [1, 2])
    assert tuple(sample["token_points"].shape) == (2, 4, 32, 3)
    assert tuple(sample["future_points"].shape) == (2, 2, 32, 3)
    assert tuple(sample["token_descriptors"].shape) == (2, 4, 3)
    np.testing.assert_array_equal(sample["token_offsets"][:, 0], 0.0)
    assert sample["velocity_seed"] == 11


def test_temporal_binary_context_supports_central_only_tokens(tmp_path: Path) -> None:
    dataset = TemporalBinaryContextDataset(
        (_entry(tmp_path, seed=11),),
        center_atom_ids=[1, 10],
        horizons_ps=[1.0, 2.0],
        anchor_stride_frames=2,
        num_points=32,
        radius=6.0,
        context_center_count=0,
        steinhardt_shell_min_neighbors=8,
        steinhardt_shell_max_neighbors=24,
    )
    sample = dataset[0]
    assert tuple(sample["token_points"].shape) == (2, 1, 32, 3)
    assert tuple(sample["token_descriptors"].shape) == (2, 1, 3)
    np.testing.assert_array_equal(sample["token_offsets"], 0.0)
