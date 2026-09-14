from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import write

from .artifacts import (
    PHASE_TO_ID,
    EnvironmentLabels,
    build_atom_table,
    label_interface,
)
from .checkpoints import CheckpointStore
from .config import validate_potential_qualification
from .generator import select_calculator
from .provenance import ExecutionProvenance, validate_configured_source_manifest
from .simulation import ThermodynamicTrace, build_initial_solid, run_npt
from .transition_analysis import (
    STRUCTURE_NAMES,
    PhaseRdfAnalysis,
    TransitionAnalysis,
    analyze_phase_rdf,
    analyze_transition,
    phase_rdf_metadata,
    write_phase_rdf_archive,
    write_phase_rdf_overview,
    write_phase_rdf_visualization,
    write_structure_slice_overview,
    write_structure_slice_visualization,
    write_transition_visualization,
)
from .transition_config import (
    TransitionBranchConfig,
    TransitionConfig,
)
from .validation import SystemDiagnostics, diagnose_system


@dataclass(frozen=True)
class PreparedInterface:
    atoms: Atoms
    labels: EnvironmentLabels
    slab_bounds_fractional: tuple[float, float]


@dataclass(frozen=True)
class TransitionBranchResult:
    branch: TransitionBranchConfig
    replica_index: int
    configured_replica_seed: int
    simulation_seed: int
    atoms: Atoms
    equilibration_trace: ThermodynamicTrace
    trace: ThermodynamicTrace
    analysis: TransitionAnalysis
    phase_rdf: PhaseRdfAnalysis
    diagnostics: SystemDiagnostics


@dataclass(frozen=True)
class TransitionGenerationResult:
    output_root: Path
    branch_dirs: tuple[Path, ...]
    branches: dict[str, TransitionBranchResult]


def _load_prepared_interface(config: TransitionConfig) -> PreparedInterface:
    source_root = config.source_dataset
    source_manifest_path = source_root / "manifest.json"
    interface_dir = source_root / config.source_interface_environment
    interface_metadata_path = interface_dir / "metadata.json"
    trajectory_path = interface_dir / "trajectory.npz"
    for path in (
        source_manifest_path,
        interface_metadata_path,
        trajectory_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Transition source dataset is missing required file: {path}.")

    with source_manifest_path.open("r", encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    validate_configured_source_manifest(
        source_manifest,
        config=config.generator,
        manifest_path=source_manifest_path,
    )
    with interface_metadata_path.open("r", encoding="utf-8") as handle:
        interface_metadata = json.load(handle)
    region_definition = interface_metadata["intermediate_regions"][0]["definition"]
    slab_values = region_definition["slab_bounds_fractional"]
    slab_bounds = (float(slab_values[0]), float(slab_values[1]))
    if not 0.0 < slab_bounds[0] < slab_bounds[1] < 1.0:
        raise RuntimeError(
            f"{interface_metadata_path}: slab_bounds_fractional must satisfy "
            f"0 < lower < upper < 1 for the repository's central liquid slab, got "
            f"{slab_bounds}."
        )
    with np.load(trajectory_path) as trajectory:
        matching_frames = np.flatnonzero(trajectory["step"] == config.source_frame_step)
        if len(matching_frames) != 1:
            raise RuntimeError(
                f"{trajectory_path}: expected exactly one frame at step "
                f"{config.source_frame_step}, found indices={matching_frames.tolist()}."
            )
        frame_index = int(matching_frames[0])
        positions_A = np.asarray(trajectory["positions_A"][frame_index], dtype=np.float64)
        cell_A = np.asarray(trajectory["cell_vectors_A"][frame_index], dtype=np.float64)
        if "volume_A3" not in trajectory.files:
            raise RuntimeError(
                f"{trajectory_path}: source trajectory has no volume_A3 array and cannot "
                "verify the selected cell. Regenerate the legacy source dataset."
            )
        stored_volume_A3 = float(trajectory["volume_A3"][frame_index])
    cell_volume_A3 = float(abs(np.linalg.det(cell_A)))
    if not np.isclose(cell_volume_A3, stored_volume_A3, rtol=1.0e-10, atol=1.0e-8):
        raise RuntimeError(
            f"{trajectory_path}: selected source frame step={config.source_frame_step} has "
            f"det(cell)={cell_volume_A3:.12g} A^3 but stored volume="
            f"{stored_volume_A3:.12g} A^3. Regenerate the stale/corrupted source trajectory."
        )
    expected_atom_count = len(build_initial_solid(config.generator))
    if len(positions_A) != expected_atom_count:
        raise RuntimeError(
            f"Transition source contains {len(positions_A)} atoms but the source generator "
            f"configuration declares {expected_atom_count}."
        )
    numbers = np.full(
        expected_atom_count,
        atomic_numbers[config.generator.system.chemical_symbol],
        dtype=np.int32,
    )
    atoms = Atoms(numbers=numbers, positions=positions_A, cell=cell_A, pbc=True)
    labels = label_interface(
        atoms,
        slab_bounds,
        config.generator.system.interface_half_width_A,
    )
    return PreparedInterface(
        atoms=atoms,
        labels=labels,
        slab_bounds_fractional=slab_bounds,
    )


def _runtime_generator_config(
    config: TransitionConfig, branch: TransitionBranchConfig
):
    return replace(
        config.generator,
        dynamics=replace(
            config.generator.dynamics,
            target_temperature_K=branch.temperature_K,
            sample_interval=config.sample_interval,
        ),
    )


def _slice_trace(
    trace: ThermodynamicTrace,
    frame_mask: np.ndarray,
    *,
    step_offset: int,
) -> ThermodynamicTrace:
    return ThermodynamicTrace(
        step=trace.step[frame_mask].copy() - step_offset,
        temperature_K=trace.temperature_K[frame_mask].copy(),
        pressure_GPa=trace.pressure_GPa[frame_mask].copy(),
        volume_A3=trace.volume_A3[frame_mask].copy(),
        potential_energy_eV_per_atom=(
            trace.potential_energy_eV_per_atom[frame_mask].copy()
        ),
        positions_A=trace.positions_A[frame_mask].copy(),
        cell_vectors_A=trace.cell_vectors_A[frame_mask].copy(),
    )


def _simulate_branch(
    prepared: PreparedInterface,
    *,
    config: TransitionConfig,
    branch: TransitionBranchConfig,
    random_seed: int,
    calculator: object,
    checkpoints: CheckpointStore,
    progress: Callable[[str], None],
) -> tuple[Atoms, ThermodynamicTrace, ThermodynamicTrace]:
    continuous_stage = f"{branch.name}.replica_{random_seed}.continuous_npt"
    checkpoint = checkpoints.load(continuous_stage)
    if checkpoint is None:
        atoms = prepared.atoms.copy()
        atoms.calc = calculator
        runtime_config = _runtime_generator_config(config, branch)
        continuous_trace = run_npt(
            atoms,
            config=runtime_config,
            temperature_K=branch.temperature_K,
            steps=branch.equilibration_steps + branch.production_steps,
            stage=f"transition.{branch.name}.continuous_equilibration_and_production",
            initialize_velocities=True,
            rng=np.random.default_rng(random_seed),
            progress=progress,
        )
        atoms.wrap()
        checkpoints.save(continuous_stage, atoms, continuous_trace)
    else:
        progress(
            f"{branch.name}: loaded replica seed {random_seed} continuous NPT checkpoint "
            f"from {checkpoints.directory}"
        )
        atoms = checkpoint.atoms
        atoms.calc = calculator
        continuous_trace = checkpoint.trace
    equilibration_mask = continuous_trace.step <= branch.equilibration_steps
    production_mask = continuous_trace.step >= branch.equilibration_steps
    if not np.any(continuous_trace.step == branch.equilibration_steps):
        raise RuntimeError(
            f"{branch.name}: continuous NPT trace has no frame at equilibration boundary step "
            f"{branch.equilibration_steps}; sample_interval={config.sample_interval} must divide "
            "equilibration_steps exactly."
        )
    equilibration_trace = _slice_trace(
        continuous_trace, equilibration_mask, step_offset=0
    )
    production_trace = _slice_trace(
        continuous_trace,
        production_mask,
        step_offset=branch.equilibration_steps,
    )
    return atoms, equilibration_trace, production_trace


def _write_branch(
    directory: Path,
    *,
    prepared: PreparedInterface,
    result: TransitionBranchResult,
    config: TransitionConfig,
    execution_provenance: ExecutionProvenance,
) -> None:
    directory.mkdir(parents=True)
    atom_table = build_atom_table(result.atoms, prepared.labels)
    np.save(directory / "atoms.npy", atom_table["position"])
    np.save(directory / "atoms_full.npy", atom_table)
    with (directory / "trajectory.npz").open("wb") as handle:
        np.savez(
            handle,
            step=result.trace.step,
            positions_A=result.trace.positions_A,
            cell_vectors_A=result.trace.cell_vectors_A,
            temperature_K=result.trace.temperature_K,
            pressure_GPa=result.trace.pressure_GPa,
            volume_A3=result.trace.volume_A3,
            potential_energy_eV_per_atom=result.trace.potential_energy_eV_per_atom,
        )
    with (directory / "equilibration_trajectory.npz").open("wb") as handle:
        np.savez(
            handle,
            step=result.equilibration_trace.step,
            positions_A=result.equilibration_trace.positions_A,
            cell_vectors_A=result.equilibration_trace.cell_vectors_A,
            temperature_K=result.equilibration_trace.temperature_K,
            pressure_GPa=result.equilibration_trace.pressure_GPa,
            volume_A3=result.equilibration_trace.volume_A3,
            potential_energy_eV_per_atom=(
                result.equilibration_trace.potential_energy_eV_per_atom
            ),
        )
    with (directory / "transition_progress.npz").open("wb") as handle:
        np.savez(
            handle,
            step=result.analysis.step,
            time_ps=result.analysis.time_ps,
            structure_names=np.asarray(STRUCTURE_NAMES),
            structure_fractions=result.analysis.structure_fractions,
            crystalline_fraction=result.analysis.crystalline_fraction,
            prepared_liquid_slab_crystalline_fraction=(
                result.analysis.prepared_liquid_slab_crystalline_fraction
            ),
            prepared_solid_region_crystalline_fraction=(
                result.analysis.prepared_solid_region_crystalline_fraction
            ),
            crystalline_profile=result.analysis.crystalline_profile,
            smoothed_crystalline_profile=(
                result.analysis.smoothed_crystalline_profile
            ),
            profile_bin_centers_fractional=(
                result.analysis.profile_bin_centers_fractional
            ),
            profile_threshold=np.asarray(result.analysis.profile_threshold),
            liquid_profile_baseline=np.asarray(
                result.analysis.liquid_profile_baseline
            ),
            solid_profile_baseline=np.asarray(
                result.analysis.solid_profile_baseline
            ),
            profile_contrast=result.analysis.profile_contrast,
            interface_positions_fractional=(
                result.analysis.interface_positions_fractional
            ),
            signed_interface_advance_A=(
                result.analysis.signed_interface_advance_A
            ),
            mean_interface_advance_A=result.analysis.mean_interface_advance_A,
            fitted_interface_velocity_m_per_s=np.asarray(
                result.analysis.fitted_interface_velocity_m_per_s
            ),
            individual_interface_velocities_m_per_s=(
                result.analysis.individual_interface_velocities_m_per_s
            ),
            individual_interface_fit_r_squared=(
                result.analysis.individual_interface_fit_r_squared
            ),
            velocity_fit_r_squared=np.asarray(
                result.analysis.velocity_fit_r_squared
            ),
            velocity_fit_ols_standard_error_m_per_s=np.asarray(
                result.analysis.velocity_fit_ols_standard_error_m_per_s
            ),
            velocity_fit_residual_rms_A=np.asarray(
                result.analysis.velocity_fit_residual_rms_A
            ),
        )
    write_phase_rdf_archive(directory / "phase_rdf.npz", result.phase_rdf)
    if config.output.save_extxyz:
        extxyz_atoms = result.atoms.copy()
        extxyz_atoms.arrays["initial_phase_id"] = atom_table["phase_id"]
        write(directory / "structure.extxyz", extxyz_atoms)
    metadata = {
        "schema_version": 2,
        "branch": asdict(result.branch),
        "replica": {
            "index": result.replica_index,
            "configured_replica_seed": result.configured_replica_seed,
            "simulation_seed": result.simulation_seed,
        },
        "source": {
            "dataset": str(config.source_dataset),
            "environment": config.source_interface_environment,
            "frame_step": config.source_frame_step,
            "slab_bounds_fractional": list(prepared.slab_bounds_fractional),
        },
        "physics": {
            "method": "planar direct solid-liquid coexistence",
            "ensemble": "isothermal-isobaric (MTK)",
            "pressure_GPa": config.generator.dynamics.pressure_GPa,
            "timestep_fs": config.generator.dynamics.timestep_fs,
            "calculator": execution_provenance.calculator.to_dict(),
            "two_periodic_fronts": True,
            "equilibration_excluded_from_production": True,
            "continuous_integrator_across_equilibration_boundary": True,
            "interface_coordinate": (
                "PTM order-profile threshold crossing in the fractional cell; signed advance "
                "uses the initial interface-normal cell height and removes affine barostat "
                "strain. Positive velocity denotes crystal growth."
            ),
            "ptm_normalized_rmsd_cutoff": config.analysis.ptm_rmsd_cutoff,
            "phase_audit": (
                "PTM is used only to audit phase-front motion. Per-atom phase_id values retain "
                "the initial prepared-region provenance and are not PTM labels."
            ),
        },
        "transition": {
            "net_crystalline_fraction_change": (
                result.analysis.net_crystalline_fraction_change
            ),
            "net_mean_interface_advance_A": (
                result.analysis.net_mean_interface_advance_A
            ),
            "fitted_interface_velocity_m_per_s": (
                result.analysis.fitted_interface_velocity_m_per_s
            ),
            "individual_interface_velocities_m_per_s": (
                result.analysis.individual_interface_velocities_m_per_s.tolist()
            ),
            "individual_interface_fit_r_squared": (
                result.analysis.individual_interface_fit_r_squared.tolist()
            ),
            "velocity_fit_r_squared": result.analysis.velocity_fit_r_squared,
            "velocity_fit_ols_standard_error_m_per_s": (
                result.analysis.velocity_fit_ols_standard_error_m_per_s
            ),
            "velocity_fit_residual_rms_A": (
                result.analysis.velocity_fit_residual_rms_A
            ),
            "velocity_fit_step_interval": [
                result.analysis.velocity_fit_start_step,
                result.analysis.velocity_fit_end_step,
            ],
            "profile_threshold": result.analysis.profile_threshold,
            "liquid_profile_baseline": result.analysis.liquid_profile_baseline,
            "solid_profile_baseline": result.analysis.solid_profile_baseline,
            "minimum_steady_profile_contrast": float(
                np.min(
                    result.analysis.profile_contrast[
                        (result.analysis.step >= result.analysis.velocity_fit_start_step)
                        & (result.analysis.step <= result.analysis.velocity_fit_end_step)
                    ]
                )
            ),
            "thermodynamic_stationarity": asdict(result.analysis.stationarity),
            "initial_crystalline_fraction": float(result.analysis.crystalline_fraction[0]),
            "final_crystalline_fraction": float(result.analysis.crystalline_fraction[-1]),
        },
        "rdf": phase_rdf_metadata(
            result.phase_rdf,
            cutoff_A=config.analysis.rdf_cutoff_A,
            bins=config.analysis.rdf_bins,
        ),
        "endpoint_diagnostics": result.diagnostics.to_dict(),
    }
    with (directory / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    with (directory / "phase_mapping.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "name_to_id": PHASE_TO_ID,
                "id_to_name": {str(value): key for key, value in PHASE_TO_ID.items()},
            },
            handle,
            indent=2,
        )
    if config.output.create_visualizations:
        visualization_dir = directory / "visualizations"
        visualization_dir.mkdir()
        write_transition_visualization(
            visualization_dir / "transition_progress.png",
            trace=result.trace,
            analysis=result.analysis,
            branch=result.branch,
            pressure_GPa=config.generator.dynamics.pressure_GPa,
        )
        write_phase_rdf_visualization(
            visualization_dir / "phase_rdf.png",
            analysis=result.phase_rdf,
            branch=result.branch,
        )
        write_structure_slice_visualization(
            visualization_dir / "structure_slice.png",
            trace=result.trace,
            chemical_symbol=config.generator.system.chemical_symbol,
            timestep_fs=config.generator.dynamics.timestep_fs,
            reference_planes_fractional=prepared.slab_bounds_fractional,
            simulation_name=result.branch.name,
            temperature_K=result.branch.temperature_K,
            ptm_rmsd_cutoff=config.analysis.ptm_rmsd_cutoff,
        )


def _resolve_zero_velocity(
    temperature_summaries: list[dict[str, object]],
) -> dict[str, object]:
    overlapping = [
        item
        for item in temperature_summaries
        if item["confidence_interval_95_m_per_s"][0] <= 0.0
        <= item["confidence_interval_95_m_per_s"][1]
    ]
    robust_pairs: list[tuple[dict[str, object], dict[str, object]]] = []
    reverse_sign_pairs: list[tuple[dict[str, object], dict[str, object]]] = []
    for lower, upper in zip(temperature_summaries[:-1], temperature_summaries[1:]):
        lower_interval = lower["confidence_interval_95_m_per_s"]
        upper_interval = upper["confidence_interval_95_m_per_s"]
        if lower_interval[0] > 0.0 and upper_interval[1] < 0.0:
            robust_pairs.append((lower, upper))
        if lower_interval[1] < 0.0 and upper_interval[0] > 0.0:
            reverse_sign_pairs.append((lower, upper))
    if reverse_sign_pairs:
        zero_velocity: dict[str, object] = {
            "status": "unresolved",
            "reason": (
                "At least one adjacent higher-temperature point reverses robustly from "
                "negative melting velocity to positive growth velocity. The resolved "
                "velocity-temperature response is non-monotonic, so no unique physical "
                "zero-velocity bracket is reported."
            ),
            "reverse_sign_temperature_pairs_K": [
                [lower["temperature_K"], upper["temperature_K"]]
                for lower, upper in reverse_sign_pairs
            ],
            "candidate_positive_to_negative_brackets_K": [
                [lower["temperature_K"], upper["temperature_K"]]
                for lower, upper in robust_pairs
            ],
            "interpolated_temperature_K": None,
        }
    elif not robust_pairs:
        overlap_detail = ""
        if overlapping:
            overlap_detail = (
                " The following sampled temperatures have 95% confidence intervals that "
                "overlap zero: "
                f"{[item['temperature_K'] for item in overlapping]}."
            )
        zero_velocity = {
            "status": "unresolved",
            "reason": (
                "No adjacent temperature pair has a positive lower-temperature velocity and "
                "negative higher-temperature velocity with both 95% confidence intervals "
                "excluding zero. Add replicas, extend stationary trajectories, and/or refine "
                f"the temperature grid.{overlap_detail}"
            ),
            "temperatures_with_zero_overlapping_confidence_interval_K": [
                item["temperature_K"] for item in overlapping
            ],
            "interpolated_temperature_K": None,
        }
    elif len(robust_pairs) > 1:
        zero_velocity = {
            "status": "unresolved",
            "reason": (
                "Multiple adjacent temperature pairs show robust positive-to-negative sign "
                "changes. The velocity-temperature response is non-monotonic at the current "
                "sampling precision, so a unique zero-velocity bracket is not defined."
            ),
            "candidate_bracket_temperature_K": [
                [lower["temperature_K"], upper["temperature_K"]]
                for lower, upper in robust_pairs
            ],
            "interpolated_temperature_K": None,
        }
    else:
        lower, upper = robust_pairs[0]
        lower_temperature = float(lower["temperature_K"])
        upper_temperature = float(upper["temperature_K"])
        lower_velocity = float(lower["mean_velocity_m_per_s"])
        upper_velocity = float(upper["mean_velocity_m_per_s"])
        velocity_difference = upper_velocity - lower_velocity
        temperature_difference = upper_temperature - lower_temperature
        interpolated_temperature = lower_temperature - (
            lower_velocity * temperature_difference / velocity_difference
        )
        lower_derivative = -temperature_difference * upper_velocity / (
            velocity_difference**2
        )
        upper_derivative = temperature_difference * lower_velocity / (
            velocity_difference**2
        )
        interpolation_standard_error = float(
            np.sqrt(
                (lower_derivative * float(lower["standard_error_m_per_s"])) ** 2
                + (upper_derivative * float(upper["standard_error_m_per_s"])) ** 2
            )
        )
        zero_velocity = {
            "status": "resolved_for_this_finite_protocol",
            "reason": (
                "Adjacent replica-mean velocities change sign and both 95% confidence "
                "intervals exclude zero. The interpolation is descriptive and does not include "
                "independent interface-preparation, finite-size, orientation, duration, "
                "order-parameter, or MLIP model error."
            ),
            "bracket_temperature_K": [lower_temperature, upper_temperature],
            "interpolated_temperature_K": interpolated_temperature,
            "propagated_replica_standard_error_K": interpolation_standard_error,
            "method": "linear interpolation of adjacent replica-mean spatial front velocities",
            "other_temperatures_with_zero_overlapping_confidence_interval_K": [
                item["temperature_K"] for item in overlapping
            ],
        }
    return zero_velocity


def _velocity_summary(
    config: TransitionConfig,
    results: dict[str, TransitionBranchResult],
    execution_provenance: ExecutionProvenance,
    prepared: PreparedInterface,
    staging_root: Path,
) -> dict[str, object]:
    try:
        from scipy.stats import t as student_t
    except ImportError as exc:
        raise ImportError(
            "Direct-coexistence replica confidence intervals require scipy in the pointnet "
            "environment."
        ) from exc

    temperature_summaries: list[dict[str, object]] = []
    for branch in config.temperature_runs:
        branch_results = [
            (run_name, result)
            for run_name, result in results.items()
            if result.branch.name == branch.name
        ]
        velocities = np.asarray(
            [
                result.analysis.fitted_interface_velocity_m_per_s
                for _, result in branch_results
            ],
            dtype=np.float64,
        )
        if not np.isfinite(velocities).all():
            raise FloatingPointError(
                f"{branch.name}: replica interface velocities are non-finite: "
                f"{velocities.tolist()}."
            )
        standard_deviation = float(np.std(velocities, ddof=1))
        standard_error = standard_deviation / np.sqrt(len(velocities))
        critical_value = float(student_t.ppf(0.975, df=len(velocities) - 1))
        half_width = critical_value * standard_error
        mean_velocity = float(np.mean(velocities))
        temperature_summaries.append(
            {
                "name": branch.name,
                "temperature_K": branch.temperature_K,
                "replica_count": len(velocities),
                "mean_velocity_m_per_s": mean_velocity,
                "sample_standard_deviation_m_per_s": standard_deviation,
                "standard_error_m_per_s": standard_error,
                "confidence_interval_95_m_per_s": [
                    mean_velocity - half_width,
                    mean_velocity + half_width,
                ],
                "runs": [
                    {
                        "run_name": run_name,
                        "replica_index": result.replica_index,
                        "configured_replica_seed": result.configured_replica_seed,
                        "simulation_seed": result.simulation_seed,
                        "velocity_m_per_s": (
                            result.analysis.fitted_interface_velocity_m_per_s
                        ),
                        "individual_front_velocities_m_per_s": (
                            result.analysis.individual_interface_velocities_m_per_s.tolist()
                        ),
                        "individual_front_fit_r_squared": (
                            result.analysis.individual_interface_fit_r_squared.tolist()
                        ),
                        "fit_r_squared": result.analysis.velocity_fit_r_squared,
                        "fit_ols_standard_error_m_per_s": (
                            result.analysis.velocity_fit_ols_standard_error_m_per_s
                        ),
                        "fit_residual_rms_A": (
                            result.analysis.velocity_fit_residual_rms_A
                        ),
                        "minimum_fit_profile_contrast": float(
                            np.min(
                                result.analysis.profile_contrast[
                                    (
                                        result.analysis.step
                                        >= result.analysis.velocity_fit_start_step
                                    )
                                    & (
                                        result.analysis.step
                                        <= result.analysis.velocity_fit_end_step
                                    )
                                ]
                            )
                        ),
                        "fit_step_interval": [
                            result.analysis.velocity_fit_start_step,
                            result.analysis.velocity_fit_end_step,
                        ],
                        "production_initial_cell_vectors_A": (
                            result.trace.cell_vectors_A[0].tolist()
                        ),
                        "artifacts": {
                            filename: {
                                "path": str(
                                    config.output.root_dir / run_name / filename
                                ),
                                "sha256": hashlib.sha256(
                                    (staging_root / run_name / filename).read_bytes()
                                ).hexdigest(),
                            }
                            for filename in (
                                "trajectory.npz",
                                "equilibration_trajectory.npz",
                                "transition_progress.npz",
                                "metadata.json",
                            )
                        },
                    }
                    for run_name, result in branch_results
                ],
            }
        )

    zero_velocity = _resolve_zero_velocity(temperature_summaries)
    serialized_config = json.dumps(
        config.to_dict(), sort_keys=True, separators=(",", ":")
    )
    first_result = next(iter(results.values()))
    return {
        "schema_version": 2,
        "calculator": {
            "identity": execution_provenance.calculator.identity,
            "implementation_class": (
                execution_provenance.calculator.implementation_class
            ),
            "model_name": execution_provenance.calculator.model_name,
            "family": execution_provenance.calculator.family,
            "model_sha256": execution_provenance.calculator.model_sha256,
            "head": execution_provenance.calculator.head,
            "source_url": execution_provenance.calculator.source_url,
            "license_identifier": (
                execution_provenance.calculator.license_identifier
            ),
            "usage_mode": execution_provenance.calculator.usage_mode,
            "validation_report_sha256": (
                execution_provenance.calculator.validation_report_sha256
            ),
            "scientifically_qualified": (
                execution_provenance.calculator.scientifically_qualified
            ),
            "settings": execution_provenance.calculator.settings,
        },
        "protocol": {
            "config_sha256": hashlib.sha256(
                serialized_config.encode("utf-8")
            ).hexdigest(),
            "config_file": str(config.config_path),
            "config_file_sha256": hashlib.sha256(
                config.config_path.read_bytes()
            ).hexdigest(),
            "chemical_symbol": config.generator.system.chemical_symbol,
            "pressure_GPa": config.generator.dynamics.pressure_GPa,
            "timestep_fs": config.generator.dynamics.timestep_fs,
            "atom_count": len(first_result.atoms),
            "conventional_cell_repetitions": list(
                config.generator.system.repetitions
            ),
            "initial_lattice_constant_A": (
                config.generator.system.initial_lattice_constant_A
            ),
            "prepared_source_cell_vectors_A": np.asarray(
                prepared.atoms.cell, dtype=np.float64
            ).tolist(),
            "interface_normal_fractional_cell_axis": [0, 0, 1],
            "interface_normal_crystal_direction": "[001]",
            "source": {
                "generator_config": str(config.generator.config_path),
                "generator_config_sha256": hashlib.sha256(
                    config.generator.config_path.read_bytes()
                ).hexdigest(),
                "dataset": str(config.source_dataset),
                "dataset_manifest_sha256": hashlib.sha256(
                    (config.source_dataset / "manifest.json").read_bytes()
                ).hexdigest(),
                "environment": config.source_interface_environment,
                "frame_step": config.source_frame_step,
            },
            "analysis": asdict(config.analysis),
            "sample_interval_steps": config.sample_interval,
            "replica_seeds": list(config.random_seeds),
            "temperature_runs": [
                {
                    "name": branch.name,
                    "temperature_K": branch.temperature_K,
                    "equilibration_steps": branch.equilibration_steps,
                    "production_steps": branch.production_steps,
                    "steady_state_fit_step_interval": [
                        branch.steady_state_start_step,
                        branch.steady_state_end_step,
                    ],
                }
                for branch in config.temperature_runs
            ],
        },
        "velocity_sign_convention": "positive is crystal growth; negative is melting",
        "uncertainty_scope": (
            "Student-t intervals quantify independent-velocity trajectory variation "
            "conditional on one shared prepared interface configuration. They do not include "
            "independent interface-preparation, size, orientation, order-parameter, or MLIP "
            "uncertainty."
        ),
        "temperatures": temperature_summaries,
        "zero_velocity_bracket_K": zero_velocity.get("bracket_temperature_K"),
        "zero_velocity": zero_velocity,
    }


def _write_velocity_summary_visualization(
    path: Path, summary: dict[str, object]
) -> None:
    temperatures = summary["temperatures"]
    temperature_K = np.asarray(
        [item["temperature_K"] for item in temperatures], dtype=np.float64
    )
    mean_velocity = np.asarray(
        [item["mean_velocity_m_per_s"] for item in temperatures], dtype=np.float64
    )
    confidence_interval = np.asarray(
        [item["confidence_interval_95_m_per_s"] for item in temperatures],
        dtype=np.float64,
    )
    errors = np.vstack(
        (mean_velocity - confidence_interval[:, 0], confidence_interval[:, 1] - mean_velocity)
    )
    figure, axis = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)
    for item in temperatures:
        run_temperatures = np.full(len(item["runs"]), item["temperature_K"])
        run_velocities = [run["velocity_m_per_s"] for run in item["runs"]]
        axis.scatter(run_temperatures, run_velocities, color="#8d99ae", alpha=0.7)
    axis.errorbar(
        temperature_K,
        mean_velocity,
        yerr=errors,
        marker="o",
        color="#264653",
        capsize=5.0,
        label="replica mean and 95% CI",
    )
    axis.axhline(0.0, color="black", linewidth=1.0)
    zero_velocity = summary["zero_velocity"]
    if zero_velocity["interpolated_temperature_K"] is not None:
        axis.axvline(
            zero_velocity["interpolated_temperature_K"],
            color="#e76f51",
            linestyle="--",
            label="finite-protocol zero-velocity interpolation",
        )
    axis.set(
        xlabel="temperature (K)",
        ylabel="spatial interface velocity (m/s)",
        title=f"Direct-coexistence temperature grid: {zero_velocity['status']}",
    )
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_dataset(
    config: TransitionConfig,
    prepared: PreparedInterface,
    results: dict[str, TransitionBranchResult],
    execution_provenance: ExecutionProvenance,
) -> tuple[Path, ...]:
    output_root = config.output.root_dir
    if output_root.exists() and not config.output.overwrite:
        raise FileExistsError(
            f"Transition output already exists: {output_root}. Remove it or explicitly set "
            "output.overwrite=true."
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
    )
    try:
        for branch_name, result in results.items():
            _write_branch(
                staging_root / branch_name,
                prepared=prepared,
                result=result,
                config=config,
                execution_provenance=execution_provenance,
            )
        velocity_summary = _velocity_summary(
            config, results, execution_provenance, prepared, staging_root
        )
        with (staging_root / "velocity_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(velocity_summary, handle, indent=2)
        if config.output.create_visualizations:
            _write_velocity_summary_visualization(
                staging_root / "transition_overview.png", velocity_summary
            )
            first_replica_results = {
                branch.name: f"{branch.name}/replica_000"
                for branch in config.temperature_runs
            }
            write_phase_rdf_overview(
                staging_root / "phase_rdf_overview.png",
                {
                    branch_name: staging_root
                    / relative_path
                    / "visualizations"
                    / "phase_rdf.png"
                    for branch_name, relative_path in first_replica_results.items()
                },
            )
            write_structure_slice_overview(
                staging_root / "structure_slice_overview.png",
                {
                    branch_name: staging_root
                    / relative_path
                    / "visualizations"
                    / "structure_slice.png"
                    for branch_name, relative_path in first_replica_results.items()
                },
            )
        manifest = {
            "schema_version": 2,
            "dataset_name": config.dataset_name,
            "run_dirs": list(results),
            "config": config.to_dict(),
            "execution_provenance": execution_provenance.to_dict(),
            "potential_sha256": execution_provenance.calculator.model_sha256,
            "potential_usage_mode": execution_provenance.calculator.usage_mode,
            "scientifically_qualified_potential": (
                execution_provenance.calculator.scientifically_qualified
            ),
            "velocity_summary": "velocity_summary.json",
            "scientific_scope": {
                "supported_claim": (
                    "Replica statistics for spatially tracked seeded planar-interface velocities "
                    "under the selected calculator, temperature grid, pressure, equilibration, "
                    "and finite steady fitting windows."
                ),
                "unsupported_claim": (
                    "Homogeneous nucleation rates or potential-independent kinetics. A resolved "
                    "zero-velocity interpolation remains conditional on this cell size, "
                    "orientation, trajectory duration, PTM order coordinate, and MLIP. "
                    "Quantitative real-Al thermodynamics or kinetics remain unsupported while "
                    "the selected potential is marked exploratory rather than scientifically "
                    "qualified for this protocol."
                ),
            },
        }
        with (staging_root / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        if output_root.exists():
            shutil.rmtree(output_root)
        staging_root.replace(output_root)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return tuple(output_root / name for name in results)


def generate_transition_dataset(
    config: TransitionConfig,
    *,
    progress: Callable[[str], None] = print,
    calculator: object | None = None,
    injected_calculator_identity: str | None = None,
) -> TransitionGenerationResult:
    if config.output.root_dir.exists() and not config.output.overwrite:
        raise FileExistsError(
            f"Transition output already exists: {config.output.root_dir}. Remove it or "
            "explicitly set output.overwrite=true. This check is performed before loading "
            "the source, constructing the calculator, or running MD."
        )
    transition_temperatures_K = tuple(
        branch.temperature_K for branch in config.temperature_runs
    )
    validate_potential_qualification(
        config.generator,
        chemical_symbol=config.generator.system.chemical_symbol,
        pressure_GPa=config.generator.dynamics.pressure_GPa,
        timestep_fs=config.generator.dynamics.timestep_fs,
        state_temperatures_K={
            "solid_bulk": transition_temperatures_K,
            "liquid_bulk": transition_temperatures_K,
            "interface": transition_temperatures_K,
        },
        context=f"direct-coexistence generation {config.dataset_name!r}",
        required_claim="equilibrium_thermodynamics",
    )
    prepared = _load_prepared_interface(config)
    selected_calculator, execution_provenance = select_calculator(
        config.generator,
        calculator=calculator,
        injected_calculator_identity=injected_calculator_identity,
    )
    progress(
        f"Generating {config.dataset_name!r}: {len(prepared.atoms)} atoms, "
        f"{len(config.temperature_runs)} temperatures x {len(config.random_seeds)} "
        "direct-coexistence replicas"
    )
    checkpoints = CheckpointStore(config, execution_provenance)
    prepared_phase_ids = np.fromiter(
        (PHASE_TO_ID[str(name)] for name in prepared.labels.phase_names),
        dtype=np.int64,
        count=len(prepared.atoms),
    )
    results: dict[str, TransitionBranchResult] = {}
    for branch_index, branch in enumerate(config.temperature_runs):
        for replica_index, random_seed in enumerate(config.random_seeds):
            simulation_seed = int(
                np.random.SeedSequence([random_seed, branch_index]).generate_state(1)[0]
            )
            run_name = f"{branch.name}/replica_{replica_index:03d}"
            atoms, equilibration_trace, trace = _simulate_branch(
                prepared,
                config=config,
                branch=branch,
                random_seed=simulation_seed,
                calculator=selected_calculator,
                checkpoints=checkpoints,
                progress=progress,
            )
            analysis = analyze_transition(
                trace,
                equilibration_trace=equilibration_trace,
                chemical_symbol=config.generator.system.chemical_symbol,
                timestep_fs=config.generator.dynamics.timestep_fs,
                slab_bounds_fractional=prepared.slab_bounds_fractional,
                profile_bins=config.analysis.profile_bins,
                profile_smoothing_bins=config.analysis.profile_smoothing_bins,
                ptm_rmsd_cutoff=config.analysis.ptm_rmsd_cutoff,
                minimum_profile_contrast=config.analysis.minimum_profile_contrast,
                minimum_velocity_fit_r_squared=(
                    config.analysis.minimum_velocity_fit_r_squared
                ),
                target_pressure_GPa=config.generator.dynamics.pressure_GPa,
                maximum_temperature_error_K=(
                    config.generator.validation.maximum_temperature_error_K
                ),
                maximum_pressure_error_GPa=(
                    config.generator.validation.maximum_pressure_error_GPa
                ),
                branch=branch,
                progress=progress,
            )
            phase_rdf = analyze_phase_rdf(
                trace,
                chemical_symbol=config.generator.system.chemical_symbol,
                prepared_phase_ids=prepared_phase_ids,
                timestep_fs=config.generator.dynamics.timestep_fs,
                cutoff_A=config.analysis.rdf_cutoff_A,
                bins=config.analysis.rdf_bins,
                branch_name=run_name,
                progress=progress,
            )
            runtime_config = _runtime_generator_config(config, branch)
            diagnostics = diagnose_system(
                atoms,
                trace,
                runtime_config,
                name=run_name,
                require_pressure_convergence=True,
            )
            results[run_name] = TransitionBranchResult(
                branch=branch,
                replica_index=replica_index,
                configured_replica_seed=random_seed,
                simulation_seed=simulation_seed,
                atoms=atoms,
                equilibration_trace=equilibration_trace,
                trace=trace,
                analysis=analysis,
                phase_rdf=phase_rdf,
                diagnostics=diagnostics,
            )
    branch_dirs = _write_dataset(
        config, prepared, results, execution_provenance
    )
    progress(f"Wrote {len(branch_dirs)} transition runs to {config.output.root_dir}")
    return TransitionGenerationResult(
        output_root=config.output.root_dir,
        branch_dirs=branch_dirs,
        branches=results,
    )
