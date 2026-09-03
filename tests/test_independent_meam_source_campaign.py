from __future__ import annotations

from collections import Counter

from scripts.run_lammps_independent_meam_source_campaign import (
    MEASUREMENT_STEPS,
    RUNS_PER_TEMPERATURE,
    SAMPLE_INTERVAL_STEPS,
    STORAGE_DTYPE,
    TEMPERATURES_K,
    render_melt_input,
    render_source_input,
    source_run_specs,
)


def test_source_specs_are_structurally_independent_and_split_before_screening() -> None:
    specs = source_run_specs()

    assert len(specs) == RUNS_PER_TEMPERATURE * len(TEMPERATURES_K) == 90
    assert Counter(spec["temperature_K"] for spec in specs) == Counter(
        {temperature: 30 for temperature in TEMPERATURES_K}
    )
    for temperature in TEMPERATURES_K:
        selected = [spec for spec in specs if spec["temperature_K"] == temperature]
        assert Counter(spec["source_split"] for spec in selected) == Counter(
            optimization=18, model_selection=6, final_validation=6
        )
    all_seeds = [
        spec[key]
        for spec in specs
        for key in ("preparation_seed", "velocity_seed")
    ]
    assert len(all_seeds) == len(set(all_seeds))


def test_source_inputs_melt_each_history_and_save_dense_combined_frames() -> None:
    spec = source_run_specs()[0]
    melt = render_melt_input(spec)
    source = render_source_input(spec)

    assert f"velocity all create 1325 {spec['preparation_seed']}" in melt
    assert "write_data prepared_liquid.lammps.data" in melt
    assert f"velocity all create 400 {spec['velocity_seed']}" in source
    assert source.index("run 5000") < source.index("reset_timestep 0")
    assert (
        f"dump trajectory all custom {SAMPLE_INTERVAL_STEPS} trajectory.lammpstrj "
        "id type x y z vx vy vz"
    ) in source
    assert f"run {MEASUREMENT_STEPS}" in source
    assert source.index("undump trajectory") < source.index("write_restart final.restart.bin")
    assert STORAGE_DTYPE == "float16"
