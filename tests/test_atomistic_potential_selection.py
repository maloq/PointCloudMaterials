from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.data_utils.synthetic.atomistic.config import (
    CONFIGURED_MACE_IMPLEMENTATION_CLASS,
    load_config,
    potential_calculator_settings,
)
from src.data_utils.synthetic.atomistic.potential_performance import (
    PotentialPerformanceConfig,
    _numerical_parity,
    load_potential_performance_config,
    summarize_block_timings,
)
from src.data_utils.synthetic.atomistic.potential_selection import (
    PotentialSelectionConfig,
    load_potential_selection_config,
    select_potential,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = (
    REPOSITORY_ROOT
    / "configs/simulation/atomistic/al/phase_context_70304_mpa.yaml"
)
CANDIDATE_CONFIG = (
    REPOSITORY_ROOT
    / "configs/simulation/atomistic/al/phase_context_70304_mh1.yaml"
)
PRODUCTION_PERFORMANCE_CONFIG = (
    REPOSITORY_ROOT
    / "configs/simulation/atomistic/al/potential_performance.yaml"
)
PRODUCTION_SELECTION_CONFIG = (
    REPOSITORY_ROOT
    / "configs/simulation/atomistic/al/potential_selection.yaml"
)


def test_performance_summary_uses_end_to_end_elapsed_time() -> None:
    summary = summarize_block_timings(
        np.asarray([2.0, 4.0], dtype=np.float64), steps_per_block=10
    )

    assert summary["measurement_steps"] == 20
    assert summary["total_seconds"] == 6.0
    assert summary["mean_seconds_per_step"] == pytest.approx(0.3)
    assert summary["median_seconds_per_step"] == pytest.approx(0.3)
    assert summary["maximum_seconds_per_step"] == pytest.approx(0.4)
    assert summary["steps_per_second"] == pytest.approx(20.0 / 6.0)


def test_production_performance_gate_uses_model_specific_sources_and_long_blocks() -> None:
    config = load_potential_performance_config(PRODUCTION_PERFORMANCE_CONFIG)
    assert config.warmup_steps == 50
    assert config.measurement_blocks == 8
    assert config.steps_per_block == 50
    assert config.measurement_blocks * config.steps_per_block == 400
    assert len(config.model_configs) == 2
    assert len(config.reference_model_configs) == 2
    assert len(config.initial_homogeneous_configs) == 2
    assert {load_config(path).potential.model_name for path in config.model_configs} == {
        "mace-mpa-0-medium",
        "mace-mh-1-omat-pbe",
    }


def test_production_selection_records_two_worker_runtime_projection() -> None:
    config = load_potential_selection_config(PRODUCTION_SELECTION_CONFIG)

    assert config.workers == 2
    assert config.makespan_safety_factor >= 1.0
    assert config.baseline_homogeneous_config.name == "homogeneous_16384_mpa.yaml"
    assert config.candidate_homogeneous_config.name == "homogeneous_16384_mh1.yaml"


def test_compiled_reference_parity_uses_per_atom_energy_and_component_rmse(
    tmp_path: Path,
) -> None:
    config = PotentialPerformanceConfig(
        model_configs=(),
        reference_model_configs=(),
        initial_homogeneous_configs=(),
        temperature_K=500.0,
        pressure_GPa=0.0,
        timestep_fs=1.0,
        thermostat_time_fs=100.0,
        barostat_time_fs=500.0,
        warmup_steps=1,
        measurement_blocks=1,
        steps_per_block=1,
        random_seed=1,
        maximum_parity_energy_difference_meV_per_atom=0.6,
        maximum_parity_force_rmse_eV_per_A=0.01,
        maximum_parity_stress_difference_GPa=0.01,
        output_json=tmp_path / "performance.json",
        config_path=tmp_path / "performance.yaml",
    )
    reference = {
        "energy_eV": -10.0,
        "forces_eV_per_A": np.zeros((2, 3), dtype=np.float64),
        "stress_GPa": np.zeros((3, 3), dtype=np.float64),
    }
    production = {
        "energy_eV": -9.999,
        "forces_eV_per_A": np.full((2, 3), 0.005, dtype=np.float64),
        "stress_GPa": np.full((3, 3), 0.002, dtype=np.float64),
    }

    result = _numerical_parity(
        reference,
        production,
        atom_count=2,
        config=config,
    )

    assert result["passed"] is True
    assert result["metrics"]["energy_difference_meV_per_atom"] == pytest.approx(0.5)
    assert result["metrics"]["force_rmse_eV_per_A"] == pytest.approx(0.005)
    assert result["metrics"]["maximum_stress_difference_GPa"] == pytest.approx(
        0.002
    )


def _scientific_result(
    generator_config_path: Path,
    *,
    qualified: bool,
    metric_scale: float,
) -> tuple[str, dict[str, object], dict[str, object]]:
    generator = load_config(generator_config_path)
    potential = generator.potential
    model_result: dict[str, object] = {
        "identity": {
            "model_name": potential.model_name,
            "sha256": potential.sha256,
            "head": potential.head,
            "implementation_class": CONFIGURED_MACE_IMPLEMENTATION_CLASS,
            "calculator_settings": potential_calculator_settings(potential),
        },
        "scientifically_qualified": qualified,
        "qualification_failures": [] if qualified else ["missing independent evidence"],
        "dft_reference_errors": {
            "by_state": {
                "solid_bulk": {
                    "energy_rmse_meV_per_atom_after_global_constant_offset": (
                        8.0 * metric_scale
                    ),
                    "force_rmse_eV_per_A": 0.12 * metric_scale,
                    "stress_rmse_GPa": 0.4 * metric_scale,
                }
            }
        },
        "nve": {
            "solid_bulk": [
                {
                    "drift_meV_per_atom_ps": -0.4 * metric_scale,
                    "maximum_excursion_meV_per_atom": 1.0 * metric_scale,
                    "detrended_rms_meV_per_atom": 0.25 * metric_scale,
                }
            ]
        },
    }
    melting_result: dict[str, object] = {
        "reference": {"temperature_K": 933.45},
        "interpolated_temperatures_K": [
            933.45 + 20.0 * metric_scale,
            933.45 - 10.0 * metric_scale,
        ],
    }
    return potential.model_name, model_result, melting_result


def _performance_result(
    generator_config_path: Path,
    homogeneous_config_path: Path,
    *,
    steps_per_second: float,
    parity_passed: bool = True,
) -> tuple[str, dict[str, object]]:
    generator = load_config(generator_config_path)
    potential = generator.potential
    homogeneous_raw = json.loads(
        homogeneous_config_path.read_text(encoding="utf-8")
    )
    source_root = Path(homogeneous_raw["source_dataset"]).resolve()
    source_environment = str(homogeneous_raw["source_environment"])
    source_directory = source_root / source_environment
    return potential.model_name, {
        "steps_per_second": steps_per_second,
        "maximum_seconds_per_step": 1.0 / steps_per_second,
        "calculator_initialization_seconds": 2.0,
        "warmup_seconds": 3.0,
        "generator_config": str(generator.config_path),
        "generator_config_sha256": hashlib.sha256(
            generator.config_path.read_bytes()
        ).hexdigest(),
        "calculator": {
            "implementation_class": CONFIGURED_MACE_IMPLEMENTATION_CLASS,
            "model_name": potential.model_name,
            "model_sha256": potential.sha256,
            "head": potential.head,
            "settings": potential_calculator_settings(potential),
        },
        "numerical_parity_passed": parity_passed,
        "numerical_parity": {
            "passed": parity_passed,
            "production_evaluation_seconds": 5.0,
            "metrics": {},
            "thresholds": {},
            "failures": [] if parity_passed else ["test parity failure"],
            "reference": {
                "generator_config": str(generator.config_path),
                "generator_config_sha256": hashlib.sha256(
                    generator.config_path.read_bytes()
                ).hexdigest(),
                "calculator": {
                    "implementation_class": CONFIGURED_MACE_IMPLEMENTATION_CLASS,
                    "model_name": potential.model_name,
                    "model_sha256": potential.sha256,
                    "head": potential.head,
                    "settings": potential_calculator_settings(potential),
                },
            },
        },
        "initial_source": {
            "homogeneous_config": str(homogeneous_config_path),
            "homogeneous_config_sha256": hashlib.sha256(
                homogeneous_config_path.read_bytes()
            ).hexdigest(),
            "source_generator_config": str(generator.config_path),
            "source_generator_config_sha256": hashlib.sha256(
                generator.config_path.read_bytes()
            ).hexdigest(),
            "source_dataset": str(source_root),
            "source_environment": source_environment,
            "source_frame_step": homogeneous_raw["source_frame_step"],
            "manifest_sha256": hashlib.sha256(
                (source_root / "manifest.json").read_bytes()
            ).hexdigest(),
            "metadata_sha256": hashlib.sha256(
                (source_directory / "metadata.json").read_bytes()
            ).hexdigest(),
            "atom_table_sha256": hashlib.sha256(
                (source_directory / "atoms_full.npy").read_bytes()
            ).hexdigest(),
            "trajectory_sha256": hashlib.sha256(
                (source_directory / "trajectory.npz").read_bytes()
            ).hexdigest(),
        },
    }


def _write_homogeneous_workload(
    path: Path,
    *,
    generator_config_path: Path,
    replica_count: int = 4,
    equilibration_steps: int = 1000,
    measurement_steps: int = 9000,
) -> None:
    source_root = path.parent / f"{path.stem}_source"
    source_directory = source_root / "replica_000_bulk_liquid"
    source_directory.mkdir(parents=True)
    (source_root / "manifest.json").write_bytes(b"test manifest\n")
    (source_directory / "metadata.json").write_bytes(b"test metadata\n")
    (source_directory / "atoms_full.npy").write_bytes(b"test atom table\n")
    (source_directory / "trajectory.npz").write_bytes(b"test trajectory\n")
    path.write_text(
        json.dumps(
            {
                "dataset_name": f"test_{path.stem}",
                "source_generator_config": str(generator_config_path),
                "source_dataset": str(source_root),
                "source_environment": "replica_000_bulk_liquid",
                "source_frame_step": 3000,
                "random_seeds": list(range(100, 100 + replica_count)),
                "temperature_K": 500.0,
                "equilibration_steps": equilibration_steps,
                "steps": measurement_steps,
                "sample_interval": 1000,
                "analysis": {
                    "ptm_rmsd_cutoff": 0.1,
                    "crystalline_cluster_cutoff_A": 3.5,
                    "nucleus_size_threshold_atoms": 100,
                    "threshold_persistence_frames": 3,
                    "rdf_cutoff_A": 8.0,
                    "rdf_bins": 160,
                },
                "output": {
                    "root_dir": str(path.parent / f"{path.stem}_output"),
                    "overwrite": False,
                    "save_extxyz": False,
                    "create_visualizations": False,
                },
            }
        ),
        encoding="utf-8",
    )


def _run_selection(
    tmp_path: Path,
    *,
    baseline_qualified: bool = True,
    candidate_qualified: bool,
    candidate_metric_scale: float,
    candidate_speed: float,
    baseline_speed: float = 1.0,
    candidate_parity_passed: bool = True,
    null_unqualified_melting: bool = True,
    workers: int = 2,
    makespan_safety_factor: float = 1.0,
    tamper_candidate_workload_after_performance: bool = False,
    tamper_candidate_source_after_performance: bool = False,
) -> dict[str, object]:
    baseline_homogeneous_path = tmp_path / "baseline_homogeneous.yaml"
    candidate_homogeneous_path = tmp_path / "candidate_homogeneous.yaml"
    _write_homogeneous_workload(
        baseline_homogeneous_path,
        generator_config_path=BASELINE_CONFIG,
    )
    _write_homogeneous_workload(
        candidate_homogeneous_path,
        generator_config_path=CANDIDATE_CONFIG,
    )
    baseline_name, baseline_science, baseline_melting = _scientific_result(
        BASELINE_CONFIG, qualified=baseline_qualified, metric_scale=1.0
    )
    candidate_name, candidate_science, candidate_melting = _scientific_result(
        CANDIDATE_CONFIG,
        qualified=candidate_qualified,
        metric_scale=candidate_metric_scale,
    )
    scientific_path = tmp_path / "scientific.json"
    scientific_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "report_type": "al_crystallization_mlip_benchmark",
                "benchmark_config": {
                    "model_configs": [str(BASELINE_CONFIG), str(CANDIDATE_CONFIG)]
                },
                "models": {
                    baseline_name: baseline_science,
                    candidate_name: candidate_science,
                },
                "melting_scans": {
                    baseline_name: (
                        None
                        if null_unqualified_melting and not baseline_qualified
                        else baseline_melting
                    ),
                    candidate_name: (
                        None
                        if null_unqualified_melting and not candidate_qualified
                        else candidate_melting
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    baseline_performance_name, baseline_performance = _performance_result(
        BASELINE_CONFIG,
        baseline_homogeneous_path,
        steps_per_second=baseline_speed,
    )
    candidate_performance_name, candidate_performance = _performance_result(
        CANDIDATE_CONFIG,
        candidate_homogeneous_path,
        steps_per_second=candidate_speed,
        parity_passed=candidate_parity_passed,
    )
    performance_path = tmp_path / "performance.json"
    performance_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "report_type": "al_crystallization_mlip_performance",
                "models": {
                    baseline_performance_name: baseline_performance,
                    candidate_performance_name: candidate_performance,
                },
            }
        ),
        encoding="utf-8",
    )
    if tamper_candidate_workload_after_performance:
        with candidate_homogeneous_path.open("a", encoding="utf-8") as handle:
            handle.write("\n# changed after timing\n")
    if tamper_candidate_source_after_performance:
        candidate_source = (
            tmp_path
            / "candidate_homogeneous_source"
            / "replica_000_bulk_liquid"
            / "trajectory.npz"
        )
        with candidate_source.open("ab") as handle:
            handle.write(b"changed after timing\n")
    selection_config_path = tmp_path / "selection.yaml"
    selection_config_path.write_text("test selection config\n", encoding="utf-8")
    return select_potential(
        PotentialSelectionConfig(
            scientific_report=scientific_path,
            performance_report=performance_path,
            baseline_generator_config=BASELINE_CONFIG,
            candidate_generator_config=CANDIDATE_CONFIG,
            baseline_homogeneous_config=baseline_homogeneous_path,
            candidate_homogeneous_config=candidate_homogeneous_path,
            workers=workers,
            makespan_safety_factor=makespan_safety_factor,
            comparison_absolute_tolerance=1.0e-12,
            output_json=tmp_path / "selection.json",
            config_path=selection_config_path,
        )
    )


def test_unqualified_mh1_never_replaces_baseline_even_when_faster(tmp_path: Path) -> None:
    result = _run_selection(
        tmp_path,
        candidate_qualified=False,
        candidate_metric_scale=0.5,
        candidate_speed=2.0,
    )

    assert result["candidate_selected"] is False
    assert result["selected_generator_config"] == str(BASELINE_CONFIG)
    assert "not scientifically qualified" in result["selection_reasons"][0]


def test_null_melting_evidence_is_valid_while_both_models_are_unqualified(
    tmp_path: Path,
) -> None:
    result = _run_selection(
        tmp_path,
        baseline_qualified=False,
        candidate_qualified=False,
        candidate_metric_scale=0.5,
        candidate_speed=2.0,
    )

    assert result["candidate_selected"] is False
    assert result["selected_generator_config"] == str(BASELINE_CONFIG)
    assert result["selection_basis"] == (
        "explicit_exploratory_mpa_fallback_both_models_unqualified"
    )
    assert result["selection_is_exploratory"] is True
    assert result["scientific_preference"]["policy_preferred_model_name"] is None


def test_qualified_pareto_better_and_faster_mh1_is_selected(tmp_path: Path) -> None:
    result = _run_selection(
        tmp_path,
        candidate_qualified=True,
        candidate_metric_scale=0.5,
        candidate_speed=1.25,
    )

    assert result["candidate_selected"] is True
    assert result["selected_generator_config"] == str(CANDIDATE_CONFIG)
    assert result["selection_reasons"] == []
    assert result["schema_version"] == 4
    assert result["policy_version"] == (
        "scientific_quality_runtime_advisory_v3"
    )
    projection = result["runtime_projection"]
    assert projection["is_selection_or_launch_gate"] is False


def test_scientifically_better_but_slower_mh1_is_selected(
    tmp_path: Path,
) -> None:
    result = _run_selection(
        tmp_path,
        candidate_qualified=True,
        candidate_metric_scale=0.5,
        candidate_speed=0.5,
    )

    assert result["candidate_selected"] is True
    assert result["performance"]["candidate_speedup"] == pytest.approx(0.5)
    assert result["performance"]["relative_speed_is_selection_threshold"] is False
    projection = result["runtime_projection"]
    assert projection["candidate"]["projected_makespan_seconds"] == pytest.approx(
        40010.0
    )
    assert projection["candidate"]["persistent_worker_startup_compile"][
        "seconds_per_worker"
    ] == pytest.approx(10.0)
    assert len(projection["candidate"]["homogeneous_config_sha256"]) == 64


def test_scientifically_preferred_candidate_is_selected_despite_longer_runtime(
    tmp_path: Path,
) -> None:
    result = _run_selection(
        tmp_path,
        candidate_qualified=True,
        candidate_metric_scale=0.5,
        candidate_speed=0.3,
    )

    assert result["candidate_selected"] is True
    assert result["selected_generator_config"] == str(CANDIDATE_CONFIG)
    assert result["runtime_projection"]["candidate"][
        "projected_makespan_seconds"
    ] > result["runtime_projection"]["baseline"]["projected_makespan_seconds"]
    assert result["selection_basis"] == "scientifically_preferred_candidate"
    assert result["selection_reasons"] == []


def test_runtime_projection_never_blocks_scientific_selection(
    tmp_path: Path,
) -> None:
    result = _run_selection(
        tmp_path,
        candidate_qualified=True,
        candidate_metric_scale=0.5,
        baseline_speed=0.3,
        candidate_speed=0.2,
    )

    assert result["candidate_selected"] is True
    assert result["selected_generator_config"] == str(CANDIDATE_CONFIG)
    projection = result["runtime_projection"]
    assert projection["is_selection_or_launch_gate"] is False
    assert projection["candidate"]["projected_makespan_seconds"] > 0
    assert (tmp_path / "selection.json").is_file()


def test_projection_rejects_workload_changed_after_performance_timing(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        RuntimeError,
        match="Homogeneous workload config changed after performance timing",
    ):
        _run_selection(
            tmp_path,
            candidate_qualified=True,
            candidate_metric_scale=0.5,
            candidate_speed=1.0,
            tamper_candidate_workload_after_performance=True,
        )


def test_projection_rejects_source_changed_after_performance_timing(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="source artifact changed"):
        _run_selection(
            tmp_path,
            candidate_qualified=True,
            candidate_metric_scale=0.5,
            candidate_speed=1.0,
            tamper_candidate_source_after_performance=True,
        )


def test_scientific_regression_keeps_baseline(tmp_path: Path) -> None:
    result = _run_selection(
        tmp_path,
        candidate_qualified=True,
        candidate_metric_scale=1.1,
        candidate_speed=1.25,
    )

    assert result["candidate_selected"] is False
    assert any("candidate regresses" in reason for reason in result["selection_reasons"])


def test_qualified_candidate_replaces_unqualified_baseline(tmp_path: Path) -> None:
    result = _run_selection(
        tmp_path,
        baseline_qualified=False,
        candidate_qualified=True,
        candidate_metric_scale=1.1,
        candidate_speed=1.1,
    )

    assert result["candidate_selected"] is True
    assert result["scientific_qualification"]["candidate_is_eligibility_upgrade"] is True


def test_candidate_compiled_parity_failure_keeps_baseline(tmp_path: Path) -> None:
    result = _run_selection(
        tmp_path,
        candidate_qualified=True,
        candidate_metric_scale=0.5,
        candidate_speed=1.25,
        candidate_parity_passed=False,
    )

    assert result["candidate_selected"] is False
    assert result["selected_generator_config"] == str(BASELINE_CONFIG)
    assert any(
        "failed exact compiled/reference numerical parity" in reason
        for reason in result["selection_reasons"]
    )
