from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.data_utils.temporal_lammps_binary import (
    binary_path_for_dump,
    write_temporal_lammps_binary,
)
from src.temporal_vamp.simulation_catalog import (
    discover_simulation_catalog,
    load_simulation_metadata,
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_simulation(campaign: Path, *, analysis_velocity_seed: int = 12345) -> Path:
    replica = campaign / "replicas" / "replica_000_velocity_12345"
    replica.mkdir(parents=True)
    _write_json(
        campaign / "manifest.json",
        {
            "atom_count": 4,
            "crystal_seed": None,
            "shared_liquid_source": {"prepared_liquid_sha256": "liquid-hash"},
            "potential": {
                "name": "test 2NN-MEAM",
                "library_sha256": "library-hash",
                "parameter_sha256": "parameter-hash",
            },
            "protocol": {
                "temperature_K": 500.0,
                "pressure_GPa": 0.0,
                "timestep_fs": 1.0,
                "equilibration_steps": 5,
                "measurement_steps": 20,
                "sample_interval_steps": 10,
                "sample_interval_ps": 0.01,
            },
        },
    )
    _write_json(
        replica / "analysis.json",
        {
            "replica_name": replica.name,
            "velocity_random_seed": analysis_velocity_seed,
            "crystal_seed": None,
            "initial_crystalline_fraction": 0.0,
            "final_crystalline_fraction": 0.8,
            "nucleation_observed": True,
            "nucleation_time_ps": 0.01,
        },
    )
    (replica / "in.lammps").write_text(
        "\n".join(
            [
                "units metal",
                "boundary p p p",
                "timestep 0.001",
                "velocity all create 500 12345 mom yes rot no dist gaussian",
                "fix ensemble all npt temp 500 500 0.3 iso 0 0 3",
                "run 5",
                "dump trajectory all custom 10 trajectory.lammpstrj id type x y z",
                "run 20",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    np.savez(
        replica / "crystallization_progress.npz",
        step=np.asarray([0, 10, 20], dtype=np.int64),
        time_ps=np.asarray([0.0, 0.01, 0.02]),
        structure_names=np.asarray(["OTHER", "FCC"]),
        structure_fractions=np.asarray([[1.0, 0.0], [0.5, 0.5], [0.2, 0.8]]),
        crystalline_fraction=np.asarray([0.0, 0.5, 0.8]),
        crystalline_cluster_count=np.asarray([0, 1, 1], dtype=np.int64),
        largest_crystalline_cluster_atoms=np.asarray([0, 2, 3], dtype=np.int64),
    )
    trajectory = replica / "trajectory.lammpstrj"
    trajectory.write_text("not scanned by catalog discovery\n", encoding="utf-8")
    return trajectory


def test_catalog_loads_physical_and_structural_metadata(tmp_path: Path) -> None:
    root = tmp_path / "synthetic_data"
    campaign = root / "al_meam_campaign"
    trajectory = _write_simulation(campaign)

    metadata = load_simulation_metadata(trajectory, campaign)
    assert metadata.atom_count == 4
    assert metadata.temperature_K == 500.0
    assert metadata.sample_interval_ps == 0.01
    assert metadata.velocity_seed == 12345
    assert metadata.boundary_conditions == ("p", "p", "p")
    assert metadata.progress_times_ps == (0.0, 0.01, 0.02)
    assert metadata.crystalline_fraction == (0.0, 0.5, 0.8)
    assert metadata.largest_crystalline_cluster_atoms == (0, 2, 3)

    entries = discover_simulation_catalog(
        root,
        campaign_globs=["al_meam_*"],
        cache_root=tmp_path / "cache",
        required_atom_count=4,
        required_potential_parameter_sha256="parameter-hash",
        required_crystal_seed=None,
        require_periodic=True,
    )
    assert len(entries) == 1
    assert entries[0].trajectory_path == trajectory.resolve()
    assert entries[0].run_id == "al_meam_campaign/replicas/replica_000_velocity_12345"


def test_catalog_rejects_manifest_lammps_disagreement(tmp_path: Path) -> None:
    campaign = tmp_path / "synthetic_data" / "al_meam_campaign"
    trajectory = _write_simulation(campaign, analysis_velocity_seed=99999)
    with pytest.raises(ValueError, match="velocity_seed"):
        load_simulation_metadata(trajectory, campaign)


def test_catalog_discovers_binary_replacement_without_text(tmp_path: Path) -> None:
    root = tmp_path / "synthetic_data"
    campaign = root / "al_meam_campaign"
    trajectory = _write_simulation(campaign)
    stat = trajectory.stat()
    write_temporal_lammps_binary(
        binary_path_for_dump(trajectory),
        positions=np.zeros((3, 4, 3), dtype=np.float32),
        timesteps=np.asarray([5, 15, 25], dtype=np.int64),
        box_low=np.zeros((3, 3), dtype=np.float32),
        box_high=np.ones((3, 3), dtype=np.float32),
        atom_ids=np.arange(1, 5, dtype=np.int64),
        atom_types=np.ones(4, dtype=np.int32),
        atom_columns=("id", "type", "x", "y", "z"),
        source={
            "trajectory_lammpstrj": {
                "path": str(trajectory.resolve()),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": "0" * 64,
                "deleted": False,
                "deleted_at": None,
            },
            "coordinate_archive": {
                "path": str((trajectory.parent / "trajectory.npz").resolve()),
                "size_bytes": 0,
                "sha256": "1" * 64,
                "positions_float32_sha256": "2" * 64,
            },
        },
        provenance={"test": True},
    )
    trajectory.unlink()

    entries = discover_simulation_catalog(
        root,
        campaign_globs=["al_meam_*"],
        cache_root=tmp_path / "cache",
        required_atom_count=4,
        required_potential_parameter_sha256="parameter-hash",
        required_crystal_seed=None,
        require_periodic=True,
    )
    assert len(entries) == 1
    assert entries[0].trajectory_path.name == "trajectory_binary_float32"
