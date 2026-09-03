from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from src.data_utils.synthetic.atomistic.lammps_nested_shooting import (
    _write_schedule,
    _source_splits,
    load_nested_shooting_config,
    multirate_output_steps,
    nested_random_seeds,
    render_nested_lammps_input,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_nested_random_seeds_share_momentum_only_within_momentum_group() -> None:
    first = nested_random_seeds(20260902, 7, 2, 0)
    second = nested_random_seeds(20260902, 7, 2, 1)
    other_momentum = nested_random_seeds(20260902, 7, 3, 0)

    assert first[0] == second[0]
    assert first[1] != second[1]
    assert other_momentum[0] != first[0]


def test_multirate_schedule_contains_every_online_monitor_frame() -> None:
    steps = multirate_output_steps(timestep_fs=3.0, maximum_duration_ps=72.0)

    assert len(steps) == 268
    assert steps[:14] == (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 133, 167, 200)
    assert steps[-1] == 24_000
    assert set(range(0, 24_001, 100)).issubset(steps)


def test_lammps_schedule_duplicates_block_boundary_control_reads(tmp_path: Path) -> None:
    path = tmp_path / "steps.txt"
    _write_schedule(
        path,
        (0, 10, 20, 100, 133, 167, 200, 233, 267, 300),
        sentinel=400,
        monitor_interval_steps=100,
    )

    assert [int(value) for value in path.read_text().splitlines()] == [
        10,
        10,
        20,
        100,
        133,
        133,
        167,
        200,
        233,
        233,
        267,
        300,
        400,
    ]


def test_repository_nested_pilot_configuration_is_checksum_bound() -> None:
    config = load_nested_shooting_config(
        REPOSITORY_ROOT
        / "configs/simulation/atomistic/al/"
        "meam_nested_shooting_pilot_70304_20260902.yaml"
    )

    assert [value.expected_source_count for value in config.temperatures] == [9, 9, 11]
    assert [value.expected_basin_a_max_cluster_atoms for value in config.temperatures] == [
        19,
        20,
        16,
    ]
    assert config.momentum_samples_per_parent == 2
    assert config.thermostat_futures_per_momentum == 2
    assert config.monitor_interval_steps == 100


def test_source_splits_approximate_70_15_15_before_parent_selection() -> None:
    sources = tuple(
        (
            f"T{temperature:g}_source_{index:02d}",
            SimpleNamespace(metadata=SimpleNamespace(temperature_K=temperature)),
        )
        for temperature, count in ((400.0, 9), (450.0, 9), (500.0, 11))
        for index in range(count)
    )

    splits = _source_splits(sources, 20260902)

    assert Counter(splits.values()) == Counter(
        optimization=20, model_selection=4, final_validation=5
    )
    assert Counter(
        splits[run_id] for run_id, entry in sources if entry.metadata.temperature_K == 400.0
    ) == Counter(optimization=6, model_selection=1, final_validation=2)
    assert Counter(
        splits[run_id] for run_id, entry in sources if entry.metadata.temperature_K == 450.0
    ) == Counter(optimization=6, model_selection=2, final_validation=1)


def test_nested_lammps_input_uses_continuous_loop_and_external_ptm_monitor() -> None:
    text = render_nested_lammps_input(
        parent_id="parent_000",
        branch_id="branch_0000",
        temperature_K=450.0,
        momentum_seed=123,
        thermostat_seed=456,
        timestep_fs=3.0,
        thermostat_time_fs=300.0,
        monitor_interval_steps=100,
        maximum_steps=16_000,
    )

    assert "velocity all create 450 123" in text
    assert "fix thermostat all langevin 450 450 0.3 456 zero yes" in text
    assert text.index("write_dump all custom initial_state.lammpstrj") < text.index(
        "fix thermostat all langevin"
    )
    assert "variable monitor_iteration loop 160" in text
    assert "run 100" in text
    assert "monitor-frame --branch-dir ." in text
    assert text.index("shell /bin/rm -f monitor_decision.txt") < text.index(
        "monitor-frame --branch-dir ."
    )
    assert 'if "${stop_decision} == 1" then "jump SELF basin_reached"' in text
    assert "variable continuation_steps equal 16000-step" in text
    assert 'if "${continuation_steps} > 0" then "run ${continuation_steps}"' in text
    assert text.index("NESTED_SHOOTING_BASIN_REACHED") < text.index(
        "variable continuation_steps equal 16000-step"
    )
    assert text.index("unfix sampled_temperature") < text.index(
        "write_restart final.restart.bin"
    )
    assert text.index("undump trajectory") < text.index(
        "write_restart final.restart.bin"
    )
    assert "read_restart" not in text
