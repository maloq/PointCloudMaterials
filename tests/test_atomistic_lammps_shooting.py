from __future__ import annotations

import numpy as np

from src.data_utils.synthetic.atomistic.lammps_shooting import (
    _lammps_command,
    _materialize_missing_branch_input,
    branch_random_seeds,
    render_lammps_input,
    select_parent_frame_indices,
)


def test_lammps_command_uses_slurm_pmi_without_hydra(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(
        "src.data_utils.synthetic.atomistic.lammps_shooting.shutil.which",
        lambda executable: "/usr/bin/srun" if executable == "srun" else None,
    )
    command = _lammps_command(mpi_ranks=24, launcher="srun_pmi2")
    assert command[:4] == [
        "/usr/bin/srun",
        "--mpi=pmi2",
        "--nodes=1",
        "--ntasks=24",
    ]
    assert "--cpu-bind=cores" in command
    assert all("mpiexec" not in argument for argument in command)


def test_missing_branch_input_is_reconstructed_from_manifest(tmp_path) -> None:
    branch = {
        "branch_dir": "branches/branch_0002",
        "branch_id": "branch_0002",
        "parent_id": "parent_000",
        "temperature_K": 450.0,
        "velocity_seed": 123,
        "thermostat_seed": 456,
    }
    manifest = {
        "protocol": {
            "timestep_fs": 3.0,
            "thermostat_time_fs": 300.0,
            "sample_interval_steps": 100,
            "run_steps": 16000,
        }
    }
    (tmp_path / "branches").mkdir()
    branch_dir = _materialize_missing_branch_input(tmp_path, manifest, branch)
    assert branch_dir == tmp_path / "branches" / "branch_0002"
    assert "run 16000" in (branch_dir / "in.lammps").read_text(encoding="utf-8")
    assert '"velocity_seed": 123' in (branch_dir / "metadata.json").read_text(
        encoding="utf-8"
    )


def test_parent_selection_uses_physical_times_before_nucleation() -> None:
    times_ps = np.arange(0.0, 201.0 * 3.0, 3.0)
    indices = select_parent_frame_indices(
        times_ps,
        nucleation_time_ps=87.0,
        offsets_ps=(-12.0, -3.0),
    )
    assert indices == (25, 28)
    assert times_ps[list(indices)].tolist() == [75.0, 84.0]


def test_branch_seeds_are_deterministic_distinct_and_lammps_valid() -> None:
    observed = {
        branch_random_seeds(20260830, parent_index, shot_index)
        for parent_index in range(40)
        for shot_index in range(8)
    }
    assert len(observed) == 320
    assert all(0 < velocity < 900_000_000 for velocity, _ in observed)
    assert all(0 < thermostat < 900_000_000 for _, thermostat in observed)
    assert all(velocity != thermostat for velocity, thermostat in observed)


def test_lammps_input_encodes_branchable_fixed_cell_protocol() -> None:
    text = render_lammps_input(
        parent_id="parent_000",
        branch_id="branch_0000",
        temperature_K=450.0,
        velocity_seed=123,
        thermostat_seed=456,
        timestep_fs=3.0,
        thermostat_time_fs=300.0,
        sample_interval_steps=100,
        run_steps=16000,
    )
    assert "fix integrate all nve" in text
    assert "fix thermostat all langevin 450 450 0.3 456 zero yes" in text
    assert "fix ensemble all npt" not in text
    assert "velocity all create 450 123" in text
    assert "id type x y z vx vy vz" in text
    assert "run 16000" in text
