from __future__ import annotations

from pathlib import Path

from scripts.run_lammps_nested_fixed_horizon_compatibility import (
    FIXED_DURATION_STEPS,
    FIXED_HORIZONS_PS,
    FIXED_TIMESTEPS,
    SAMPLE_INTERVAL_STEPS,
    STORAGE_DTYPE,
    _continuation_seed,
    render_continuation_input,
)


def test_nested_continuation_retains_velocity_and_reaches_24ps() -> None:
    text = render_continuation_input(
        source_restart=Path("/source/final.restart.bin"),
        temperature_K=450.0,
        thermostat_seed=12345,
        first_step=300,
    )

    assert "read_restart /source/final.restart.bin" in text
    assert "velocity all create" not in text
    assert "fix thermostat all langevin 450 450 0.3 12345 zero yes" in text
    assert "dump continuation all custom 100 continuation.lammpstrj" in text
    assert "run 7700" in text
    assert "write_restart final.restart.bin" in text


def test_fixed_horizon_contract_contains_exact_training_targets() -> None:
    assert FIXED_DURATION_STEPS == 8000
    assert FIXED_TIMESTEPS == tuple(range(0, 8001, SAMPLE_INTERVAL_STEPS))
    assert tuple(round(value * 1000.0 / 3.0) for value in FIXED_HORIZONS_PS) == (
        2000,
        4000,
        8000,
    )
    assert STORAGE_DTYPE == "float16"
    assert len({_continuation_seed(index) for index in range(144)}) == 144


def test_compatibility_worker_does_not_delete_open_binary_memmaps() -> None:
    source = Path(
        "scripts/run_lammps_nested_fixed_horizon_compatibility.py"
    ).read_text(encoding="utf-8")

    assert 'shutil.rmtree(branch_dir / "continuation_binary_float16")' not in source
