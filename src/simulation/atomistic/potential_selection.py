from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import (
    CONFIGURED_MACE_IMPLEMENTATION_CLASS,
    GeneratorConfig,
    REPOSITORY_ROOT,
    load_config,
    potential_calculator_settings,
)
from .homogeneous_config import load_homogeneous_crystallization_config


DFT_METRICS = (
    "energy_rmse_meV_per_atom_after_global_constant_offset",
    "force_rmse_eV_per_A",
    "stress_rmse_GPa",
)

POTENTIAL_SELECTION_SCHEMA_VERSION = 4
POTENTIAL_SELECTION_POLICY_VERSION = "scientific_quality_runtime_advisory_v3"


@dataclass(frozen=True)
class PotentialSelectionConfig:
    scientific_report: Path
    performance_report: Path
    baseline_generator_config: Path
    candidate_generator_config: Path
    baseline_homogeneous_config: Path
    candidate_homogeneous_config: Path
    workers: int
    makespan_safety_factor: float
    comparison_absolute_tolerance: float
    output_json: Path
    config_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }


def _repo_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def load_potential_selection_config(path: str | Path) -> PotentialSelectionConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"{config_path}: root must be a mapping.")
    expected = {
        "scientific_report",
        "performance_report",
        "baseline_generator_config",
        "candidate_generator_config",
        "baseline_homogeneous_config",
        "candidate_homogeneous_config",
        "workers",
        "makespan_safety_factor",
        "comparison_absolute_tolerance",
        "output_json",
    }
    if set(raw) != expected:
        raise KeyError(
            f"{config_path}: selection keys must be exactly {sorted(expected)}; "
            f"observed={sorted(raw)}."
        )
    workers = raw["workers"]
    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise TypeError(
            f"{config_path}: workers must be a positive integer, got {workers!r}."
        )
    makespan_safety_factor = float(raw["makespan_safety_factor"])
    if not np.isfinite(makespan_safety_factor) or makespan_safety_factor < 1.0:
        raise ValueError(
            f"{config_path}: makespan_safety_factor must be finite and >= 1, "
            f"got {makespan_safety_factor}."
        )
    tolerance = float(raw["comparison_absolute_tolerance"])
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            f"{config_path}: comparison_absolute_tolerance must be finite and >= 0, "
            f"got {tolerance}."
        )
    return PotentialSelectionConfig(
        scientific_report=_repo_path(raw["scientific_report"]),
        performance_report=_repo_path(raw["performance_report"]),
        baseline_generator_config=_repo_path(raw["baseline_generator_config"]),
        candidate_generator_config=_repo_path(raw["candidate_generator_config"]),
        baseline_homogeneous_config=_repo_path(raw["baseline_homogeneous_config"]),
        candidate_homogeneous_config=_repo_path(raw["candidate_homogeneous_config"]),
        workers=workers,
        makespan_safety_factor=makespan_safety_factor,
        comparison_absolute_tolerance=tolerance,
        output_json=_repo_path(raw["output_json"]),
        config_path=config_path,
    )


def _read_json(path: Path, *, context: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{context} does not exist: {path}.")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: {context} root must be a mapping.")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_metric(value: object, *, context: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{context} must be numeric, got {value!r}.")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{context} must be finite and nonnegative, got {result}.")
    return result


def _positive_metric(value: object, *, context: str) -> float:
    result = _finite_metric(value, context=context)
    if result <= 0.0:
        raise ValueError(f"{context} must be positive, got {result}.")
    return result


def _runtime_projection(
    *,
    model_name: str,
    generator_config: GeneratorConfig,
    homogeneous_config_path: Path,
    performance_result: dict[str, Any],
    selection_config: PotentialSelectionConfig,
) -> dict[str, Any]:
    homogeneous = load_homogeneous_crystallization_config(
        homogeneous_config_path
    )
    if homogeneous.generator.config_path != generator_config.config_path:
        raise RuntimeError(
            f"{homogeneous.config_path}: source_generator_config must be the exact "
            f"selection generator config {generator_config.config_path} for "
            f"{model_name!r}, got {homogeneous.generator.config_path}."
        )
    if homogeneous.generator.potential.model_name != model_name:
        raise RuntimeError(
            f"{homogeneous.config_path}: workload model is "
            f"{homogeneous.generator.potential.model_name!r}, expected {model_name!r}."
        )

    source_evidence = performance_result.get("initial_source")
    if not isinstance(source_evidence, dict):
        raise TypeError(
            f"Performance result for {model_name!r} must bind initial_source to the "
            "exact model-specific homogeneous workload."
        )
    performance_homogeneous_value = source_evidence.get("homogeneous_config")
    performance_homogeneous_sha256 = source_evidence.get(
        "homogeneous_config_sha256"
    )
    performance_generator_value = source_evidence.get("source_generator_config")
    performance_generator_sha256 = source_evidence.get(
        "source_generator_config_sha256"
    )
    if not all(
        isinstance(value, str)
        for value in (
            performance_homogeneous_value,
            performance_homogeneous_sha256,
            performance_generator_value,
            performance_generator_sha256,
        )
    ):
        raise TypeError(
            f"Performance initial_source for {model_name!r} must contain exact "
            "homogeneous/source-generator paths and SHA-256 digests."
        )
    performance_homogeneous_path = _repo_path(performance_homogeneous_value)
    if performance_homogeneous_path != homogeneous.config_path:
        raise RuntimeError(
            f"Performance timing for {model_name!r} used homogeneous config "
            f"{performance_homogeneous_path}, but the runtime projection requires "
            f"{homogeneous.config_path}."
        )
    homogeneous_sha256 = _sha256(homogeneous.config_path)
    if performance_homogeneous_sha256 != homogeneous_sha256:
        raise RuntimeError(
            f"Homogeneous workload config changed after performance timing: "
            f"{homogeneous.config_path}."
        )
    performance_generator_path = _repo_path(performance_generator_value)
    if performance_generator_path != generator_config.config_path:
        raise RuntimeError(
            f"Performance source for {model_name!r} used generator config "
            f"{performance_generator_path}, expected {generator_config.config_path}."
        )
    generator_sha256 = _sha256(generator_config.config_path)
    if performance_generator_sha256 != generator_sha256:
        raise RuntimeError(
            f"Performance source generator config changed after timing: "
            f"{generator_config.config_path}."
        )

    performance_source_dataset = source_evidence.get("source_dataset")
    performance_source_environment = source_evidence.get("source_environment")
    performance_source_frame_step = source_evidence.get("source_frame_step")
    if (
        not isinstance(performance_source_dataset, str)
        or _repo_path(performance_source_dataset) != homogeneous.source_dataset
        or performance_source_environment != homogeneous.source_environment
        or performance_source_frame_step != homogeneous.source_frame_step
    ):
        raise RuntimeError(
            f"Performance timing for {model_name!r} is not bound to the exact "
            "homogeneous source dataset/environment/frame: performance="
            f"{performance_source_dataset!r}/{performance_source_environment!r}/"
            f"{performance_source_frame_step!r}, workload="
            f"{homogeneous.source_dataset}/{homogeneous.source_environment}/"
            f"{homogeneous.source_frame_step}."
        )
    source_directory = (
        homogeneous.source_dataset / homogeneous.source_environment
    )
    source_artifact_spec = {
        "manifest": (
            homogeneous.source_dataset / "manifest.json",
            "manifest_sha256",
        ),
        "metadata": (source_directory / "metadata.json", "metadata_sha256"),
        "atom_table": (
            source_directory / "atoms_full.npy",
            "atom_table_sha256",
        ),
        "trajectory": (
            source_directory / "trajectory.npz",
            "trajectory_sha256",
        ),
    }
    source_artifacts: dict[str, dict[str, str]] = {}
    for artifact_name, (artifact_path, digest_field) in source_artifact_spec.items():
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Performance-timed source artifact is missing for {model_name!r}: "
                f"{artifact_path}."
            )
        recorded_sha256 = source_evidence.get(digest_field)
        observed_sha256 = _sha256(artifact_path)
        if recorded_sha256 != observed_sha256:
            raise RuntimeError(
                f"Performance-timed source artifact changed for {model_name!r}: "
                f"path={artifact_path}, recorded_sha256={recorded_sha256!r}, "
                f"observed_sha256={observed_sha256}."
            )
        source_artifacts[artifact_name] = {
            "path": str(artifact_path.resolve()),
            "sha256": observed_sha256,
        }

    measured_steps_per_second = _positive_metric(
        performance_result.get("steps_per_second"),
        context=f"{model_name}.steps_per_second",
    )
    conservative_seconds_per_step = _positive_metric(
        performance_result.get("maximum_seconds_per_step"),
        context=f"{model_name}.maximum_seconds_per_step",
    )
    calculator_initialization_seconds = _finite_metric(
        performance_result.get("calculator_initialization_seconds"),
        context=f"{model_name}.calculator_initialization_seconds",
    )
    warmup_seconds = _finite_metric(
        performance_result.get("warmup_seconds"),
        context=f"{model_name}.warmup_seconds",
    )
    parity_record = performance_result.get("numerical_parity")
    if not isinstance(parity_record, dict):
        raise TypeError(
            f"Performance result for {model_name!r} must contain numerical_parity."
        )
    first_compiled_evaluation_seconds = _finite_metric(
        parity_record.get("production_evaluation_seconds"),
        context=f"{model_name}.numerical_parity.production_evaluation_seconds",
    )
    persistent_worker_startup_compile_seconds = (
        calculator_initialization_seconds
        + first_compiled_evaluation_seconds
        + warmup_seconds
    )

    replica_count = len(homogeneous.random_seeds)
    active_workers = min(selection_config.workers, replica_count)
    maximum_replicas_per_worker = (
        replica_count + active_workers - 1
    ) // active_workers
    steps_per_replica = homogeneous.equilibration_steps + homogeneous.steps
    total_md_steps = replica_count * steps_per_replica
    critical_path_md_steps = maximum_replicas_per_worker * steps_per_replica
    steady_state_critical_path_seconds = (
        critical_path_md_steps * conservative_seconds_per_step
    )
    unsafetied_makespan_seconds = (
        persistent_worker_startup_compile_seconds
        + steady_state_critical_path_seconds
    )
    projected_makespan_seconds = (
        selection_config.makespan_safety_factor * unsafetied_makespan_seconds
    )
    return {
        "model_name": model_name,
        "homogeneous_config": str(homogeneous.config_path),
        "homogeneous_config_sha256": homogeneous_sha256,
        "source_generator_config": str(generator_config.config_path),
        "source_generator_config_sha256": generator_sha256,
        "initial_source_artifacts": source_artifacts,
        "replica_count": replica_count,
        "configured_workers": selection_config.workers,
        "active_workers": active_workers,
        "maximum_replicas_per_worker": maximum_replicas_per_worker,
        "equilibration_steps_per_replica": homogeneous.equilibration_steps,
        "measurement_steps_per_replica": homogeneous.steps,
        "full_duration_md_steps_per_replica": steps_per_replica,
        "total_md_steps": total_md_steps,
        "critical_path_md_steps": critical_path_md_steps,
        "measured_steps_per_second": measured_steps_per_second,
        "conservative_seconds_per_step": conservative_seconds_per_step,
        "conservative_steps_per_second": 1.0 / conservative_seconds_per_step,
        "persistent_worker_startup_compile": {
            "calculator_initialization_seconds": calculator_initialization_seconds,
            "first_compiled_evaluation_seconds": (
                first_compiled_evaluation_seconds
            ),
            "warmup_seconds": warmup_seconds,
            "seconds_per_worker": persistent_worker_startup_compile_seconds,
            "total_worker_seconds": (
                active_workers * persistent_worker_startup_compile_seconds
            ),
            "critical_path_seconds": persistent_worker_startup_compile_seconds,
        },
        "steady_state_critical_path_seconds": steady_state_critical_path_seconds,
        "unsafetied_makespan_seconds": unsafetied_makespan_seconds,
        "makespan_safety_factor": selection_config.makespan_safety_factor,
        "projected_makespan_seconds": projected_makespan_seconds,
        "projected_makespan_hours": projected_makespan_seconds / 3600.0,
        "projection_method": (
            "slowest measured NPT timing-block seconds/step times the exact full "
            "equilibration+measurement critical-path steps, plus one measured "
            "initialization/first-compiled-evaluation/warmup overhead per persistent "
            "worker process; the critical path is multiplied by the configured "
            "safety factor"
        ),
    }


def _scientific_metrics(
    model_name: str,
    model_result: dict[str, Any],
    melting_result: dict[str, Any],
) -> dict[str, float]:
    dft = model_result.get("dft_reference_errors")
    if not isinstance(dft, dict):
        raise RuntimeError(
            f"Scientific result for {model_name!r} has no DFT reference errors; "
            "the candidate cannot be compared or selected."
        )
    by_state = dft.get("by_state")
    if not isinstance(by_state, dict) or not by_state:
        raise RuntimeError(
            f"Scientific result for {model_name!r} has no per-state DFT metrics."
        )
    metrics: dict[str, float] = {}
    for state, state_result in sorted(by_state.items()):
        if not isinstance(state_result, dict):
            raise TypeError(f"DFT state {state!r} for {model_name!r} must be a mapping.")
        for metric in DFT_METRICS:
            metrics[f"dft.{state}.{metric}"] = _finite_metric(
                state_result.get(metric),
                context=f"{model_name}.dft.{state}.{metric}",
            )
    nve = model_result.get("nve")
    if not isinstance(nve, dict) or not nve:
        raise RuntimeError(f"Scientific result for {model_name!r} has no NVE metrics.")
    absolute_drifts: list[float] = []
    drift_differences: list[float] = []
    excursions: list[float] = []
    detrended_rms: list[float] = []
    for state, runs in nve.items():
        if not isinstance(runs, list) or not runs:
            raise TypeError(f"{model_name}.nve.{state} must be a non-empty list.")
        state_drifts: list[float] = []
        for run_index, run in enumerate(runs):
            if not isinstance(run, dict):
                raise TypeError(
                    f"{model_name}.nve.{state}[{run_index}] must be a mapping."
                )
            drift = _finite_signed_metric(
                run.get("drift_meV_per_atom_ps"),
                context=(
                    f"{model_name}.nve.{state}[{run_index}]."
                    "drift_meV_per_atom_ps"
                )
            )
            state_drifts.append(drift)
            absolute_drifts.append(abs(drift))
            excursions.append(
                _finite_metric(
                    run.get("maximum_excursion_meV_per_atom"),
                    context=(
                        f"{model_name}.nve.{state}[{run_index}]."
                        "maximum_excursion_meV_per_atom"
                    ),
                )
            )
            detrended_rms.append(
                _finite_metric(
                    run.get("detrended_rms_meV_per_atom"),
                    context=(
                        f"{model_name}.nve.{state}[{run_index}]."
                        "detrended_rms_meV_per_atom"
                    ),
                )
            )
        drift_differences.append(max(state_drifts) - min(state_drifts))
    metrics["nve.maximum_absolute_drift_meV_per_atom_ps"] = max(absolute_drifts)
    metrics["nve.maximum_timestep_drift_difference_meV_per_atom_ps"] = max(
        drift_differences
    )
    metrics["nve.maximum_excursion_meV_per_atom"] = max(excursions)
    metrics["nve.maximum_detrended_rms_meV_per_atom"] = max(detrended_rms)
    reference = melting_result.get("reference")
    temperatures = melting_result.get("interpolated_temperatures_K")
    if not isinstance(reference, dict) or not isinstance(temperatures, list) or not temperatures:
        raise RuntimeError(
            f"Scientific result for {model_name!r} lacks resolved melting temperatures."
        )
    reference_temperature = _finite_metric(
        reference.get("temperature_K"),
        context=f"{model_name}.melting.reference.temperature_K",
    )
    errors = [
        abs(
            _finite_metric(
                temperature,
                context=f"{model_name}.melting.interpolated_temperatures_K",
            )
            - reference_temperature
        )
        for temperature in temperatures
    ]
    metrics["melting.mean_absolute_error_K"] = float(np.mean(errors))
    metrics["melting.maximum_absolute_error_K"] = max(errors)
    return metrics


def _finite_signed_metric(value: object, *, context: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{context} must be numeric, got {value!r}.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{context} must be finite, got {result}.")
    return result


def select_potential(config: PotentialSelectionConfig) -> dict[str, Any]:
    if (
        not isinstance(config.workers, int)
        or isinstance(config.workers, bool)
        or config.workers <= 0
    ):
        raise TypeError(f"workers must be a positive integer, got {config.workers!r}.")
    if (
        not np.isfinite(config.makespan_safety_factor)
        or config.makespan_safety_factor < 1.0
    ):
        raise ValueError(
            "makespan_safety_factor must be finite and >= 1, got "
            f"{config.makespan_safety_factor}."
        )
    if config.output_json.exists():
        raise FileExistsError(
            f"Selection output already exists: {config.output_json}. Remove it explicitly "
            "or choose a new output path."
        )
    scientific = _read_json(config.scientific_report, context="scientific report")
    performance = _read_json(config.performance_report, context="performance report")
    if (
        scientific.get("schema_version") != 1
        or scientific.get("report_type")
        != "al_crystallization_mlip_benchmark"
    ):
        raise ValueError(
            f"{config.scientific_report}: expected schema_version=1 and "
            "report_type='al_crystallization_mlip_benchmark'."
        )
    if (
        performance.get("schema_version") != 1
        or performance.get("report_type") != "al_crystallization_mlip_performance"
    ):
        raise ValueError(
            f"{config.performance_report}: expected schema_version=1 and "
            "report_type='al_crystallization_mlip_performance'."
        )
    baseline_config = load_config(config.baseline_generator_config)
    candidate_config = load_config(config.candidate_generator_config)
    baseline_name = baseline_config.potential.model_name
    candidate_name = candidate_config.potential.model_name
    if baseline_name == candidate_name:
        raise ValueError(
            f"Baseline and candidate model names must differ, both are {baseline_name!r}."
        )
    workload_config_by_name = {
        baseline_name: config.baseline_homogeneous_config,
        candidate_name: config.candidate_homogeneous_config,
    }
    if config.baseline_homogeneous_config == config.candidate_homogeneous_config:
        raise ValueError(
            "Baseline and candidate must bind distinct model-specific homogeneous "
            f"workload configs, both are {config.baseline_homogeneous_config}."
        )
    scientific_models = scientific.get("models")
    melting_scans = scientific.get("melting_scans")
    performance_models = performance.get("models")
    if not all(
        isinstance(value, dict)
        for value in (scientific_models, melting_scans, performance_models)
    ):
        raise TypeError("Scientific/performance model collections must be mappings.")
    scientific_benchmark_config = scientific.get("benchmark_config")
    if not isinstance(scientific_benchmark_config, dict):
        raise TypeError("Scientific report benchmark_config must be a mapping.")
    scientific_model_config_values = scientific_benchmark_config.get("model_configs")
    if not isinstance(scientific_model_config_values, list):
        raise TypeError(
            "Scientific report benchmark_config.model_configs must be a list."
        )
    scientific_generator_configs = [
        load_config(_repo_path(value)) for value in scientific_model_config_values
    ]
    scientific_config_by_name = {
        item.potential.model_name: item for item in scientific_generator_configs
    }
    if len(scientific_config_by_name) != len(scientific_generator_configs):
        raise RuntimeError(
            "Scientific benchmark model_configs contain duplicate model_name values."
        )
    parity_by_model: dict[str, bool] = {}
    runtime_by_model: dict[str, dict[str, Any]] = {}
    for generator_config in (baseline_config, candidate_config):
        potential = generator_config.potential
        if (
            potential.model_name not in scientific_models
            or potential.model_name not in melting_scans
            or potential.model_name not in performance_models
        ):
            raise KeyError(
                "Scientific model, melting-scan, and performance collections must all "
                f"contain configured model {potential.model_name!r}."
            )
        scientific_result = scientific_models[potential.model_name]
        melting_result = melting_scans[potential.model_name]
        performance_result = performance_models[potential.model_name]
        if not isinstance(scientific_result, dict) or not isinstance(
            performance_result, dict
        ):
            raise TypeError(
                f"Scientific and performance entries for {potential.model_name!r} must "
                "be mappings. Melting evidence may be null only while that model is "
                "scientifically unqualified."
            )
        scientific_identity = scientific_result.get("identity")
        performance_calculator = performance_result.get("calculator")
        if not isinstance(scientific_identity, dict) or not isinstance(
            performance_calculator, dict
        ):
            raise TypeError(f"Missing model identity for {potential.model_name!r}.")
        if (
            scientific_identity.get("model_name") != potential.model_name
            or scientific_identity.get("sha256") != potential.sha256
            or scientific_identity.get("head") != potential.head
        ):
            raise RuntimeError(
                f"Scientific identity for {potential.model_name!r} does not match "
                f"configured SHA/head {potential.sha256}/{potential.head}."
            )
        if (
            performance_calculator.get("model_name") != potential.model_name
            or performance_calculator.get("model_sha256") != potential.sha256
            or performance_calculator.get("head") != potential.head
        ):
            raise RuntimeError(
                f"Performance identity for {potential.model_name!r} does not match "
                f"configured SHA/head {potential.sha256}/{potential.head}."
            )
        performance_config_value = performance_result.get("generator_config")
        performance_config_sha256 = performance_result.get(
            "generator_config_sha256"
        )
        if not isinstance(performance_config_value, str) or not isinstance(
            performance_config_sha256, str
        ):
            raise TypeError(
                f"Performance result for {potential.model_name!r} must bind its exact "
                "generator_config path and SHA-256."
            )
        performance_config_path = _repo_path(performance_config_value)
        if performance_config_path != generator_config.config_path:
            raise RuntimeError(
                f"Performance result for {potential.model_name!r} benchmarked "
                f"{performance_config_path}, but selection config requires the exact "
                f"production config {generator_config.config_path}."
            )
        if _sha256(performance_config_path) != performance_config_sha256:
            raise RuntimeError(
                f"Performance generator config changed after benchmarking: "
                f"{performance_config_path}."
            )
        if performance_calculator.get("settings") != potential_calculator_settings(
            potential
        ):
            raise RuntimeError(
                f"Performance calculator settings for {potential.model_name!r} do not "
                "match the exact selected production generator config."
            )
        parity_passed = performance_result.get("numerical_parity_passed")
        parity_record = performance_result.get("numerical_parity")
        if type(parity_passed) is not bool or not isinstance(parity_record, dict):
            raise TypeError(
                f"Performance result for {potential.model_name!r} must contain an exact "
                "numerical_parity_passed boolean and numerical_parity mapping."
            )
        if parity_record.get("passed") is not parity_passed:
            raise RuntimeError(
                f"Performance result for {potential.model_name!r} has inconsistent "
                "compiled/reference numerical-parity status."
            )
        scientific_generator = scientific_config_by_name.get(potential.model_name)
        if scientific_generator is None:
            raise KeyError(
                f"Scientific benchmark_config.model_configs does not contain "
                f"{potential.model_name!r}."
            )
        reference_evidence = parity_record.get("reference")
        if not isinstance(reference_evidence, dict):
            raise TypeError(
                f"Performance parity for {potential.model_name!r} must bind its "
                "uncompiled reference generator and calculator."
            )
        reference_config_value = reference_evidence.get("generator_config")
        reference_config_sha256 = reference_evidence.get(
            "generator_config_sha256"
        )
        reference_calculator = reference_evidence.get("calculator")
        if (
            not isinstance(reference_config_value, str)
            or not isinstance(reference_config_sha256, str)
            or not isinstance(reference_calculator, dict)
        ):
            raise TypeError(
                f"Performance parity reference for {potential.model_name!r} must "
                "contain generator path/SHA and calculator identity."
            )
        reference_config_path = _repo_path(reference_config_value)
        if reference_config_path != scientific_generator.config_path:
            raise RuntimeError(
                f"Scientific benchmark for {potential.model_name!r} used "
                f"{scientific_generator.config_path}, but compiled parity used reference "
                f"{reference_config_path}."
            )
        if _sha256(reference_config_path) != reference_config_sha256:
            raise RuntimeError(
                f"Numerical-parity reference config changed after benchmarking: "
                f"{reference_config_path}."
            )
        expected_reference_settings = potential_calculator_settings(
            scientific_generator.potential
        )
        if (
            scientific_identity.get("implementation_class")
            != CONFIGURED_MACE_IMPLEMENTATION_CLASS
            or scientific_identity.get("calculator_settings")
            != expected_reference_settings
            or reference_calculator.get("implementation_class")
            != CONFIGURED_MACE_IMPLEMENTATION_CLASS
            or reference_calculator.get("model_name") != potential.model_name
            or reference_calculator.get("model_sha256") != potential.sha256
            or reference_calculator.get("head") != potential.head
            or reference_calculator.get("settings") != expected_reference_settings
        ):
            raise RuntimeError(
                f"Scientific and numerical-parity reference calculator settings for "
                f"{potential.model_name!r} do not match the exact hashed uncompiled "
                "generator config."
            )
        parity_by_model[potential.model_name] = parity_passed
        runtime_by_model[potential.model_name] = _runtime_projection(
            model_name=potential.model_name,
            generator_config=generator_config,
            homogeneous_config_path=workload_config_by_name[potential.model_name],
            performance_result=performance_result,
            selection_config=config,
        )
    if not parity_by_model[baseline_name]:
        raise RuntimeError(
            f"Baseline production path {baseline_name!r} failed exact compiled/reference "
            "numerical parity. Refusing to select either production workflow until the "
            "backend discrepancy is resolved."
        )
    candidate_result = scientific_models[candidate_name]
    baseline_result = scientific_models[baseline_name]
    baseline_qualified = baseline_result.get("scientifically_qualified") is True
    candidate_qualified = candidate_result.get("scientifically_qualified") is True
    reasons: list[str] = []
    comparisons: dict[str, Any] = {}
    candidate_scientifically_preferred = False
    candidate_pareto_non_regressing: bool | None = None
    candidate_strictly_better: bool | None = None
    scientific_preference_status: str
    policy_preferred_model_name: str | None
    if not parity_by_model[candidate_name]:
        reasons.append("candidate failed exact compiled/reference numerical parity")
    if not candidate_qualified:
        failures = candidate_result.get("qualification_failures")
        reasons.append(f"candidate is not scientifically qualified: {failures}")
        if baseline_qualified:
            scientific_preference_status = (
                "qualified_baseline_over_unqualified_candidate"
            )
            policy_preferred_model_name = baseline_name
        else:
            scientific_preference_status = (
                "both_unqualified_no_scientific_preference"
            )
            policy_preferred_model_name = None
            reasons.append(
                "both models are scientifically unqualified; retaining MPA only as "
                "an explicit exploratory fallback and asserting no scientific "
                "preference"
            )
    else:
        candidate_melting = melting_scans[candidate_name]
        if not isinstance(candidate_melting, dict):
            raise RuntimeError(
                f"Qualified candidate {candidate_name!r} has no resolved melting evidence."
            )
        candidate_metrics = _scientific_metrics(
            candidate_name,
            candidate_result,
            candidate_melting,
        )
        if not baseline_qualified:
            comparisons = {
                metric: {
                    "baseline": None,
                    "candidate": value,
                    "candidate_minus_baseline": None,
                }
                for metric, value in sorted(candidate_metrics.items())
            }
            candidate_scientifically_preferred = True
            scientific_preference_status = "qualified_candidate_upgrade"
            policy_preferred_model_name = candidate_name
        else:
            baseline_melting = melting_scans[baseline_name]
            if not isinstance(baseline_melting, dict):
                raise RuntimeError(
                    f"Qualified baseline {baseline_name!r} has no resolved melting evidence."
                )
            baseline_metrics = _scientific_metrics(
                baseline_name,
                baseline_result,
                baseline_melting,
            )
            if baseline_metrics.keys() != candidate_metrics.keys():
                raise RuntimeError(
                    "Baseline and candidate scientific metrics differ: "
                    f"baseline={sorted(baseline_metrics)}, "
                    f"candidate={sorted(candidate_metrics)}."
                )
            candidate_strictly_better = False
            candidate_pareto_non_regressing = True
            for metric in sorted(baseline_metrics):
                baseline_value = baseline_metrics[metric]
                candidate_value = candidate_metrics[metric]
                delta = candidate_value - baseline_value
                comparisons[metric] = {
                    "baseline": baseline_value,
                    "candidate": candidate_value,
                    "candidate_minus_baseline": delta,
                }
                if delta > config.comparison_absolute_tolerance:
                    candidate_pareto_non_regressing = False
                    reasons.append(
                        f"candidate regresses {metric}: {candidate_value:.12g} > "
                        f"{baseline_value:.12g}"
                    )
                if delta < -config.comparison_absolute_tolerance:
                    candidate_strictly_better = True
            if not candidate_strictly_better:
                reasons.append(
                    "candidate is not strictly better on any scientific metric"
                )
            candidate_scientifically_preferred = bool(
                candidate_pareto_non_regressing and candidate_strictly_better
            )
            if candidate_scientifically_preferred:
                scientific_preference_status = "candidate_pareto_better"
                policy_preferred_model_name = candidate_name
            elif candidate_pareto_non_regressing:
                scientific_preference_status = (
                    "qualified_baseline_retained_without_strict_candidate_improvement"
                )
                policy_preferred_model_name = baseline_name
            else:
                scientific_preference_status = (
                    "qualified_baseline_retained_after_candidate_regression"
                )
                policy_preferred_model_name = baseline_name

    baseline_projection = runtime_by_model[baseline_name]
    candidate_projection = runtime_by_model[candidate_name]
    candidate_selected = bool(
        candidate_scientifically_preferred
        and parity_by_model[candidate_name]
    )
    if candidate_selected:
        selected_config = config.candidate_generator_config
        selection_basis = "scientifically_preferred_candidate"
    else:
        selected_config = config.baseline_generator_config
        if candidate_scientifically_preferred and not parity_by_model[candidate_name]:
            selection_basis = (
                "scientifically_preferred_candidate_failed_parity_baseline_fallback"
            )
        elif not baseline_qualified and not candidate_qualified:
            selection_basis = (
                "explicit_exploratory_mpa_fallback_both_models_unqualified"
            )
        else:
            selection_basis = "scientific_policy_retains_baseline"

    baseline_speed = float(baseline_projection["measured_steps_per_second"])
    candidate_speed = float(candidate_projection["measured_steps_per_second"])
    speedup = candidate_speed / baseline_speed
    report = {
        "schema_version": POTENTIAL_SELECTION_SCHEMA_VERSION,
        "report_type": "al_crystallization_mlip_selection",
        "policy_version": POTENTIAL_SELECTION_POLICY_VERSION,
        "policy": (
            "Prefer a scientifically qualified MH-1 candidate when it is a qualified "
            "upgrade over unqualified MPA or is Pareto non-regressing over every "
            "configured DFT/NVE/melting metric with a strict improvement in at least "
            "one. Full-duration runtime projections are descriptive scheduling evidence, "
            "never a model-selection or launch gate. If both "
            "models are unqualified, retain MPA explicitly as an exploratory fallback "
            "without asserting a scientific preference."
        ),
        "baseline_model_name": baseline_name,
        "candidate_model_name": candidate_name,
        "candidate_selected": candidate_selected,
        "selected_model_name": candidate_name if candidate_selected else baseline_name,
        "selected_generator_config": str(selected_config),
        "selection_basis": selection_basis,
        "selected_model_scientifically_qualified": (
            candidate_qualified if candidate_selected else baseline_qualified
        ),
        "selection_is_exploratory": not (
            candidate_qualified if candidate_selected else baseline_qualified
        ),
        "selection_reasons": reasons,
        "scientific_qualification": {
            "baseline": baseline_qualified,
            "candidate": candidate_qualified,
            "candidate_is_eligibility_upgrade": (
                candidate_qualified and not baseline_qualified
            ),
        },
        "scientific_preference": {
            "status": scientific_preference_status,
            "policy_preferred_model_name": policy_preferred_model_name,
            "candidate_scientifically_preferred": (
                candidate_scientifically_preferred
            ),
            "candidate_pareto_non_regressing": candidate_pareto_non_regressing,
            "candidate_strictly_better": candidate_strictly_better,
        },
        "scientific_metric_comparison": comparisons,
        "compiled_reference_numerical_parity": {
            "baseline": parity_by_model[baseline_name],
            "candidate": parity_by_model[candidate_name],
        },
        "performance": {
            "baseline_steps_per_second": baseline_speed,
            "candidate_steps_per_second": candidate_speed,
            "candidate_speedup": speedup,
            "relative_speed_is_selection_threshold": False,
        },
        "runtime_projection": {
            "is_selection_or_launch_gate": False,
            "workers": config.workers,
            "makespan_safety_factor": config.makespan_safety_factor,
            "baseline": baseline_projection,
            "candidate": candidate_projection,
        },
        "inputs": {
            "selection_config": str(config.config_path),
            "selection_config_sha256": _sha256(config.config_path),
            "scientific_report": str(config.scientific_report),
            "scientific_report_sha256": _sha256(config.scientific_report),
            "performance_report": str(config.performance_report),
            "performance_report_sha256": _sha256(config.performance_report),
            "baseline_generator_config": str(config.baseline_generator_config),
            "baseline_generator_config_sha256": _sha256(
                config.baseline_generator_config
            ),
            "candidate_generator_config": str(config.candidate_generator_config),
            "candidate_generator_config_sha256": _sha256(
                config.candidate_generator_config
            ),
            "baseline_homogeneous_config": str(
                config.baseline_homogeneous_config
            ),
            "baseline_homogeneous_config_sha256": _sha256(
                config.baseline_homogeneous_config
            ),
            "candidate_homogeneous_config": str(
                config.candidate_homogeneous_config
            ),
            "candidate_homogeneous_config_sha256": _sha256(
                config.candidate_homogeneous_config
            ),
        },
    }
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.output_json.with_suffix(config.output_json.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(config.output_json)
    return report
