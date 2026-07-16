from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, replace
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
    build_atom_table,
    label_bulk,
)
from .checkpoints import CheckpointStore
from .config import validate_potential_qualification
from .generator import select_calculator
from .homogeneous_analysis import (
    HomogeneousCrystallizationAnalysis,
    HomogeneousSurvivalAnalysis,
    ReplicaObservation,
    analyze_homogeneous_crystallization,
    analyze_replica_survival,
    write_homogeneous_progress_visualization,
    write_homogeneous_rdf_visualization,
)
from .homogeneous_config import HomogeneousCrystallizationConfig
from .provenance import ExecutionProvenance, validate_configured_source_manifest
from .simulation import ThermodynamicTrace, build_initial_solid, run_npt
from .transition_analysis import STRUCTURE_NAMES, write_structure_slice_visualization
from .validation import SystemDiagnostics, diagnose_system


REPLICA_DIRECTORY_FORMAT = "replica_{index:03d}"


@dataclass(frozen=True)
class SourceLiquid:
    atoms: Atoms
    temperature_K: float
    pressure_GPa: float
    volume_A3: float
    crystalline_fraction: float


@dataclass(frozen=True)
class HomogeneousCrystallizationReplicaResult:
    replica_name: str
    random_seed: int
    run_dir: Path
    atoms: Atoms
    equilibration_trace: ThermodynamicTrace
    trace: ThermodynamicTrace
    analysis: HomogeneousCrystallizationAnalysis
    diagnostics: SystemDiagnostics


@dataclass(frozen=True)
class HomogeneousCrystallizationResult:
    output_root: Path
    replicas: tuple[HomogeneousCrystallizationReplicaResult, ...]
    survival: HomogeneousSurvivalAnalysis


def _load_source_liquid(config: HomogeneousCrystallizationConfig) -> SourceLiquid:
    source_root = config.source_dataset
    source_dir = source_root / config.source_environment
    manifest_path = source_root / "manifest.json"
    metadata_path = source_dir / "metadata.json"
    atom_table_path = source_dir / "atoms_full.npy"
    trajectory_path = source_dir / "trajectory.npz"
    for path in (manifest_path, metadata_path, atom_table_path, trajectory_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Homogeneous crystallization source is missing required file: {path}."
            )

    with manifest_path.open("r", encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    validate_configured_source_manifest(
        source_manifest,
        config=config.generator,
        manifest_path=manifest_path,
    )

    atom_table = np.load(atom_table_path, mmap_mode="r")
    liquid_phase_id = PHASE_TO_ID["liquid_bulk"]
    non_liquid_atoms = np.flatnonzero(atom_table["phase_id"] != liquid_phase_id)
    if len(non_liquid_atoms):
        raise RuntimeError(
            f"{atom_table_path}: homogeneous nucleation must start from the repository's "
            f"bulk-liquid environment, but {len(non_liquid_atoms)} atoms do not have "
            f"phase_id={liquid_phase_id}."
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        source_metadata = json.load(handle)
    ptm = source_metadata["diagnostics"]["ptm_structure_fractions"]
    crystalline_fraction = float(ptm["fcc"] + ptm["hcp"] + ptm["bcc"])
    maximum_source_crystalline_fraction = (
        config.generator.validation.maximum_liquid_crystalline_fraction
    )
    if crystalline_fraction > maximum_source_crystalline_fraction:
        raise RuntimeError(
            f"{metadata_path}: source crystalline fraction is {crystalline_fraction:.6f}, "
            "above validation.maximum_liquid_crystalline_fraction="
            f"{maximum_source_crystalline_fraction:.6f}; this is not a validated liquid."
        )

    with np.load(trajectory_path) as trajectory:
        matching_frames = np.flatnonzero(
            trajectory["step"] == config.source_frame_step
        )
        if len(matching_frames) != 1:
            raise RuntimeError(
                f"{trajectory_path}: expected exactly one frame at step "
                f"{config.source_frame_step}, found indices={matching_frames.tolist()}."
            )
        frame_index = int(matching_frames[0])
        positions_A = np.asarray(
            trajectory["positions_A"][frame_index], dtype=np.float64
        )
        cell_A = np.asarray(
            trajectory["cell_vectors_A"][frame_index], dtype=np.float64
        )
        temperature_K = float(trajectory["temperature_K"][frame_index])
        pressure_GPa = float(trajectory["pressure_GPa"][frame_index])
        volume_A3 = float(trajectory["volume_A3"][frame_index])
    cell_volume_A3 = float(np.linalg.det(cell_A))
    if cell_volume_A3 <= 0.0 or not np.isclose(
        cell_volume_A3, volume_A3, rtol=1.0e-10, atol=1.0e-6
    ):
        raise RuntimeError(
            f"{trajectory_path}: source frame step={config.source_frame_step} has an "
            "inconsistent periodic cell and stored volume: det(cell)="
            f"{cell_volume_A3:.12f} A^3, volume_A3={volume_A3:.12f}. This indicates a "
            "stale or corrupted phase-context artifact; regenerate the source dataset."
        )

    expected_atom_count = len(build_initial_solid(config.generator))
    if len(positions_A) != expected_atom_count:
        raise RuntimeError(
            f"Source liquid contains {len(positions_A)} atoms but its generator "
            f"configuration declares {expected_atom_count}."
        )
    numbers = np.full(
        expected_atom_count,
        atomic_numbers[config.generator.system.chemical_symbol],
        dtype=np.int32,
    )
    return SourceLiquid(
        atoms=Atoms(numbers=numbers, positions=positions_A, cell=cell_A, pbc=True),
        temperature_K=temperature_K,
        pressure_GPa=pressure_GPa,
        volume_A3=volume_A3,
        crystalline_fraction=crystalline_fraction,
    )


def _runtime_generator_config(config: HomogeneousCrystallizationConfig):
    return replace(
        config.generator,
        dynamics=replace(
            config.generator.dynamics,
            target_temperature_K=config.temperature_K,
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


def _simulate(
    source: SourceLiquid,
    *,
    config: HomogeneousCrystallizationConfig,
    replica_name: str,
    random_seed: int,
    calculator: object,
    checkpoints: CheckpointStore,
    progress: Callable[[str], None],
) -> tuple[Atoms, ThermodynamicTrace, ThermodynamicTrace]:
    continuous_stage = f"{replica_name}_continuous_equilibration_and_measurement_npt"
    checkpoint = checkpoints.load(continuous_stage)
    if checkpoint is None:
        atoms = source.atoms.copy()
        atoms.calc = calculator
        continuous_trace = run_npt(
            atoms,
            config=_runtime_generator_config(config),
            temperature_K=config.temperature_K,
            steps=config.equilibration_steps + config.steps,
            stage=continuous_stage,
            initialize_velocities=True,
            rng=np.random.default_rng(random_seed),
            progress=progress,
        )
        atoms.wrap()
        checkpoints.save(
            continuous_stage,
            atoms,
            continuous_trace,
            metadata={
                "source_dataset": str(config.source_dataset),
                "source_environment": config.source_environment,
                "source_frame_step": config.source_frame_step,
                "replica_name": replica_name,
                "random_seed": random_seed,
                "equilibration_steps_excluded_from_waiting_time": (
                    config.equilibration_steps
                ),
                "measurement_steps": config.steps,
                "integrator_continuous_across_measurement_origin": True,
            },
        )
    else:
        progress(
            f"{continuous_stage}: loaded checkpoint from {checkpoints.directory}"
        )
        atoms = checkpoint.atoms
        atoms.calc = calculator
        continuous_trace = checkpoint.trace
    boundary_frames = np.flatnonzero(
        continuous_trace.step == config.equilibration_steps
    )
    if len(boundary_frames) != 1:
        raise RuntimeError(
            f"{continuous_stage}: expected exactly one saved frame at the equilibration/"
            f"measurement boundary step={config.equilibration_steps}, found indices="
            f"{boundary_frames.tolist()}. sample_interval={config.sample_interval} must "
            "divide equilibration_steps exactly."
        )
    equilibration_mask = continuous_trace.step <= config.equilibration_steps
    measurement_mask = continuous_trace.step >= config.equilibration_steps
    equilibration_trace = _slice_trace(
        continuous_trace,
        equilibration_mask,
        step_offset=0,
    )
    measurement_trace = _slice_trace(
        continuous_trace,
        measurement_mask,
        step_offset=config.equilibration_steps,
    )
    return atoms, equilibration_trace, measurement_trace


def _write_run(
    directory: Path,
    *,
    replica_name: str,
    random_seed: int,
    source: SourceLiquid,
    atoms: Atoms,
    equilibration_trace: ThermodynamicTrace,
    trace: ThermodynamicTrace,
    analysis: HomogeneousCrystallizationAnalysis,
    diagnostics: SystemDiagnostics,
    config: HomogeneousCrystallizationConfig,
    execution_provenance: ExecutionProvenance,
) -> None:
    directory.mkdir(parents=True)
    labels = label_bulk(len(atoms), "liquid_bulk", grain_id=0)
    atom_table = build_atom_table(atoms, labels)
    np.save(directory / "atoms.npy", atom_table["position"])
    np.save(directory / "atoms_full.npy", atom_table)
    with (directory / "equilibration_trajectory.npz").open("wb") as handle:
        np.savez(
            handle,
            step=equilibration_trace.step,
            positions_A=equilibration_trace.positions_A,
            cell_vectors_A=equilibration_trace.cell_vectors_A,
            temperature_K=equilibration_trace.temperature_K,
            pressure_GPa=equilibration_trace.pressure_GPa,
            volume_A3=equilibration_trace.volume_A3,
            potential_energy_eV_per_atom=(
                equilibration_trace.potential_energy_eV_per_atom
            ),
        )
    with (directory / "trajectory.npz").open("wb") as handle:
        np.savez(
            handle,
            step=trace.step,
            positions_A=trace.positions_A,
            cell_vectors_A=trace.cell_vectors_A,
            temperature_K=trace.temperature_K,
            pressure_GPa=trace.pressure_GPa,
            volume_A3=trace.volume_A3,
            potential_energy_eV_per_atom=trace.potential_energy_eV_per_atom,
        )
    with (directory / "crystallization_progress.npz").open("wb") as handle:
        np.savez(
            handle,
            step=analysis.step,
            time_ps=analysis.time_ps,
            structure_names=np.asarray(STRUCTURE_NAMES),
            structure_fractions=analysis.structure_fractions,
            crystalline_fraction=analysis.crystalline_fraction,
            crystalline_cluster_count=analysis.crystalline_cluster_count,
            largest_crystalline_cluster_atoms=(
                analysis.largest_crystalline_cluster_atoms
            ),
        )
    with (directory / "total_rdf.npz").open("wb") as handle:
        np.savez(
            handle,
            step=analysis.step,
            time_ps=analysis.time_ps,
            distance_A=analysis.rdf_distance_A,
            g_r=analysis.rdf_g_r,
        )
    if config.output.save_extxyz:
        extxyz_atoms = atoms.copy()
        extxyz_atoms.arrays["initial_phase_id"] = atom_table["phase_id"]
        write(directory / "structure.extxyz", extxyz_atoms)

    threshold_event = {
        "observed": analysis.nucleation_observed,
        "definition": (
            "onset of the first run of threshold_persistence_frames consecutive saved "
            "measurement frames containing at least nucleus_size_threshold_atoms in one "
            "connected PTM FCC/HCP/BCC cluster"
        ),
        "observable_name": "persistent_crystalline_cluster_threshold_event",
        "ptm_normalized_rmsd_cutoff": analysis.ptm_rmsd_cutoff,
        "nucleus_size_threshold_atoms": analysis.nucleus_size_threshold_atoms,
        "threshold_persistence_frames": analysis.threshold_persistence_frames,
        "threshold_persistence_span_ps": (
            (analysis.threshold_persistence_frames - 1)
            * config.sample_interval
            * config.generator.dynamics.timestep_fs
            / 1000.0
        ),
        "cluster_neighbor_cutoff_A": (
            config.analysis.crystalline_cluster_cutoff_A
        ),
        "maximum_cluster_atoms": int(
            np.max(analysis.largest_crystalline_cluster_atoms)
        ),
        "final_cluster_atoms": int(
            analysis.largest_crystalline_cluster_atoms[-1]
        ),
        "initial_crystalline_fraction": float(analysis.crystalline_fraction[0]),
        "final_crystalline_fraction": float(analysis.crystalline_fraction[-1]),
    }
    if analysis.nucleation_observed:
        threshold_event["onset_step"] = analysis.nucleation_step
        threshold_event["onset_time_ps"] = analysis.nucleation_time_ps
        threshold_event["confirmation_step"] = analysis.confirmation_step
        threshold_event["confirmation_time_ps"] = analysis.confirmation_time_ps
    metadata = {
        "schema_version": 2,
        "replica": {
            "name": replica_name,
            "random_seed": random_seed,
            "statistically_independent_velocity_initialization": True,
        },
        "source": {
            "dataset": str(config.source_dataset),
            "environment": config.source_environment,
            "frame_step": config.source_frame_step,
            "temperature_K": source.temperature_K,
            "pressure_GPa": source.pressure_GPa,
            "volume_A3": source.volume_A3,
            "ptm_crystalline_fraction": source.crystalline_fraction,
        },
        "physics": {
            "method": "homogeneous unseeded crystallization from supercooled liquid",
            "ensemble": "isothermal-isobaric (MTK)",
            "integrator_continuous_across_measurement_origin": True,
            "target_temperature_equilibration": {
                "from_source_temperature_K": source.temperature_K,
                "to_temperature_K": config.temperature_K,
                "steps": config.equilibration_steps,
                "duration_ps": (
                    config.equilibration_steps
                    * config.generator.dynamics.timestep_fs
                    / 1000.0
                ),
                "excluded_from_waiting_time_analysis": True,
                "trajectory_file": "equilibration_trajectory.npz",
            },
            "pressure_GPa": config.generator.dynamics.pressure_GPa,
            "timestep_fs": config.generator.dynamics.timestep_fs,
            "measurement_duration_ps": (
                config.steps * config.generator.dynamics.timestep_fs / 1000.0
            ),
            "calculator": execution_provenance.calculator.to_dict(),
            "seeded": False,
            "phase_audit": (
                "PTM and its connected FCC/HCP/BCC clusters are analysis observables only. "
                "Per-atom phase_id retains the initial liquid provenance. The persistent "
                "threshold event is not identified with a committor-derived critical nucleus."
            ),
        },
        "threshold_event": threshold_event,
        "rdf": {
            "definition": "whole-cell total Al radial distribution function",
            "backend": "OVITO compiled coordination analysis",
            "cutoff_A": config.analysis.rdf_cutoff_A,
            "bins": config.analysis.rdf_bins,
            "frames": len(analysis.step),
        },
        "endpoint_diagnostics": diagnostics.to_dict(),
    }
    with (directory / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    with (directory / "phase_mapping.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "name_to_id": PHASE_TO_ID,
                "id_to_name": {
                    str(value): key for key, value in PHASE_TO_ID.items()
                },
            },
            handle,
            indent=2,
        )

    if config.output.create_visualizations:
        visualization_dir = directory / "visualizations"
        visualization_dir.mkdir()
        write_homogeneous_progress_visualization(
            visualization_dir / "crystallization_progress.png",
            trace=trace,
            analysis=analysis,
            temperature_K=config.temperature_K,
            pressure_GPa=config.generator.dynamics.pressure_GPa,
        )
        write_homogeneous_rdf_visualization(
            visualization_dir / "total_rdf.png",
            analysis=analysis,
            temperature_K=config.temperature_K,
        )
        write_structure_slice_visualization(
            visualization_dir / "structure_slice.png",
            trace=trace,
            chemical_symbol=config.generator.system.chemical_symbol,
            timestep_fs=config.generator.dynamics.timestep_fs,
            reference_planes_fractional=(),
            simulation_name="homogeneous crystallization",
            temperature_K=config.temperature_K,
            ptm_rmsd_cutoff=config.analysis.ptm_rmsd_cutoff,
        )


def _write_overview(path: Path, run_dir: Path, replica_name: str) -> None:
    images = (
        ("crystallization progress", "crystallization_progress.png"),
        ("structure slices", "structure_slice.png"),
        ("total RDF", "total_rdf.png"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(22.0, 6.5), constrained_layout=True)
    for axis, (title, filename) in zip(axes, images):
        axis.imshow(plt.imread(run_dir / "visualizations" / filename))
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(
        f"MACE homogeneous crystallization from supercooled liquid: {replica_name}"
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_dataset(
    config: HomogeneousCrystallizationConfig,
    *,
    source: SourceLiquid,
    replicas: tuple[HomogeneousCrystallizationReplicaResult, ...],
    survival: HomogeneousSurvivalAnalysis,
    execution_provenance: ExecutionProvenance,
) -> None:
    output_root = config.output.root_dir
    if output_root.exists() and not config.output.overwrite:
        raise FileExistsError(
            f"Homogeneous crystallization output already exists: {output_root}. Remove it "
            "or explicitly set output.overwrite=true."
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.staging-", dir=output_root.parent
        )
    )
    try:
        for replica in replicas:
            run_dir = staging_root / replica.replica_name
            _write_run(
                run_dir,
                replica_name=replica.replica_name,
                random_seed=replica.random_seed,
                source=source,
                atoms=replica.atoms,
                equilibration_trace=replica.equilibration_trace,
                trace=replica.trace,
                analysis=replica.analysis,
                diagnostics=replica.diagnostics,
                config=config,
                execution_provenance=execution_provenance,
            )
            if config.output.create_visualizations:
                _write_overview(
                    staging_root / f"{replica.replica_name}_overview.png",
                    run_dir,
                    replica.replica_name,
                )
        with (staging_root / "survival_summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(survival.to_dict(), handle, indent=2)
        with (staging_root / "survival_curve.npz").open("wb") as handle:
            np.savez(
                handle,
                time_ps=survival.time_ps,
                replicas_at_risk=survival.replicas_at_risk,
                events=survival.events,
                right_censored=survival.censored,
                survival_probability=survival.survival_probability,
            )
        manifest = {
            "schema_version": 2,
            "dataset_name": config.dataset_name,
            "replica_dirs": [replica.replica_name for replica in replicas],
            "config": config.to_dict(),
            "execution_provenance": execution_provenance.to_dict(),
            "potential_sha256": execution_provenance.calculator.model_sha256,
            "potential_usage_mode": execution_provenance.calculator.usage_mode,
            "scientifically_qualified_potential": (
                execution_provenance.calculator.scientifically_qualified
            ),
            "scientific_scope": {
                "supported_claim": (
                    "Replica-level waiting times or right-censoring for a sustained, "
                    "explicitly configured connected PTM-crystalline cluster threshold "
                    f"during finite unseeded {config.temperature_K:.0f} K trajectories "
                    "under the selected MACE Hamiltonian, plus their non-parametric "
                    "Kaplan-Meier survival curve."
                ),
                "unsupported_claim": (
                    "A homogeneous nucleation rate, critical nucleus size, equilibrium "
                    "melting temperature, committor, or potential-independent kinetics. "
                    "The output deliberately does not fit a rate without separate tests of "
                    "stationarity, model-specific undercooling, and finite-size convergence. "
                    "Quantitative claims about real aluminium are unsupported while the "
                    "configured potential remains exploratory and lacks a qualifying "
                    "solid/liquid/interface validation report."
                ),
            },
            "method_references": [
                {
                    "title": (
                        "Crystal nucleation and growth dynamics of aluminum via "
                        "quantum-accurate MD simulations"
                    ),
                    "url": (
                        "https://www.sciencedirect.com/science/article/abs/pii/"
                        "S1359645425005324"
                    ),
                    "relevance": (
                        "Reports spontaneous Al crystallization at 500-540 K using an ML "
                        "interatomic potential."
                    ),
                },
                {
                    "title": (
                        "Molecular simulation of the crystallization of aluminum from the "
                        "supercooled liquid"
                    ),
                    "url": "https://pubmed.ncbi.nlm.nih.gov/17935411/",
                    "relevance": (
                        "Reports complete Al crystallization at 1 atm and 15-20% below the "
                        "model melting temperature using hybrid Monte Carlo."
                    ),
                },
            ],
        }
        with (staging_root / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        if output_root.exists():
            shutil.rmtree(output_root)
        staging_root.replace(output_root)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


def generate_homogeneous_crystallization_dataset(
    config: HomogeneousCrystallizationConfig,
    *,
    progress: Callable[[str], None] = print,
    calculator: object | None = None,
    injected_calculator_identity: str | None = None,
) -> HomogeneousCrystallizationResult:
    if config.output.root_dir.exists() and not config.output.overwrite:
        raise FileExistsError(
            f"Homogeneous crystallization output already exists: "
            f"{config.output.root_dir}. Remove it or explicitly set "
            "output.overwrite=true. This check is performed before loading the source, "
            "constructing the calculator, or running MD."
        )
    homogeneous_temperatures_K = (config.temperature_K,)
    validate_potential_qualification(
        config.generator,
        chemical_symbol=config.generator.system.chemical_symbol,
        pressure_GPa=config.generator.dynamics.pressure_GPa,
        timestep_fs=config.generator.dynamics.timestep_fs,
        state_temperatures_K={
            "liquid_bulk": homogeneous_temperatures_K,
            "interface": homogeneous_temperatures_K,
            "nucleus": homogeneous_temperatures_K,
        },
        context=f"homogeneous crystallization generation {config.dataset_name!r}",
        required_claim="kinetics",
    )
    source = _load_source_liquid(config)
    selected_calculator, execution_provenance = select_calculator(
        config.generator,
        calculator=calculator,
        injected_calculator_identity=injected_calculator_identity,
    )
    progress(
        f"Generating {config.dataset_name!r}: {len(config.random_seeds)} independent "
        f"replicas of {len(source.atoms)} atoms at {config.temperature_K:.0f} K; "
        f"equilibration={config.equilibration_steps * config.generator.dynamics.timestep_fs / 1000.0:.1f} ps, "
        f"measurement={config.steps * config.generator.dynamics.timestep_fs / 1000.0:.1f} ps"
    )
    checkpoints = CheckpointStore(config, execution_provenance)
    runtime_config = _runtime_generator_config(config)
    replicas: list[HomogeneousCrystallizationReplicaResult] = []
    observations: list[ReplicaObservation] = []
    for replica_index, random_seed in enumerate(config.random_seeds):
        replica_name = REPLICA_DIRECTORY_FORMAT.format(index=replica_index)
        progress(f"{replica_name}: random_seed={random_seed}")
        atoms, equilibration_trace, trace = _simulate(
            source,
            config=config,
            replica_name=replica_name,
            random_seed=random_seed,
            calculator=selected_calculator,
            checkpoints=checkpoints,
            progress=progress,
        )
        analysis = analyze_homogeneous_crystallization(
            trace,
            chemical_symbol=config.generator.system.chemical_symbol,
            timestep_fs=config.generator.dynamics.timestep_fs,
            ptm_rmsd_cutoff=config.analysis.ptm_rmsd_cutoff,
            crystalline_cluster_cutoff_A=(
                config.analysis.crystalline_cluster_cutoff_A
            ),
            nucleus_size_threshold_atoms=(
                config.analysis.nucleus_size_threshold_atoms
            ),
            threshold_persistence_frames=(
                config.analysis.threshold_persistence_frames
            ),
            rdf_cutoff_A=config.analysis.rdf_cutoff_A,
            rdf_bins=config.analysis.rdf_bins,
            progress=progress,
        )
        maximum_initial_crystalline_fraction = (
            config.generator.validation.maximum_liquid_crystalline_fraction
        )
        if analysis.crystalline_fraction[0] > maximum_initial_crystalline_fraction:
            raise RuntimeError(
                f"{replica_name}: the measured trajectory does not start from a validated "
                "metastable liquid after target-temperature equilibration: frame-zero PTM "
                f"crystalline fraction={analysis.crystalline_fraction[0]:.6f}, maximum="
                f"{maximum_initial_crystalline_fraction:.6f}."
            )
        if analysis.nucleation_observed and analysis.nucleation_step == 0:
            raise RuntimeError(
                f"{replica_name}: a persistent threshold-sized crystalline cluster is "
                "already present in the first measured frame, so its waiting time is "
                "left-censored and cannot enter the Kaplan-Meier analysis. Use a condition "
                "that remains metastable through equilibration or shorten the explicitly "
                "reported equilibration interval."
            )
        diagnostics = diagnose_system(
            atoms,
            trace,
            runtime_config,
            name=f"{replica_name}_homogeneous_crystallization",
            require_pressure_convergence=True,
        )
        replicas.append(
            HomogeneousCrystallizationReplicaResult(
                replica_name=replica_name,
                random_seed=random_seed,
                run_dir=config.output.root_dir / replica_name,
                atoms=atoms,
                equilibration_trace=equilibration_trace,
                trace=trace,
                analysis=analysis,
                diagnostics=diagnostics,
            )
        )
        observation_time_ps = (
            analysis.nucleation_time_ps
            if analysis.nucleation_observed
            else float(analysis.time_ps[-1])
        )
        if observation_time_ps is None:
            raise RuntimeError(
                f"{replica_name}: event is marked observed but has no onset time."
            )
        observations.append(
            ReplicaObservation(
                replica_name=replica_name,
                random_seed=random_seed,
                event_observed=analysis.nucleation_observed,
                observation_time_ps=observation_time_ps,
            )
        )

    survival = analyze_replica_survival(tuple(observations))
    replica_results = tuple(replicas)
    _write_dataset(
        config,
        source=source,
        replicas=replica_results,
        survival=survival,
        execution_provenance=execution_provenance,
    )
    progress(f"Wrote homogeneous crystallization dataset to {config.output.root_dir}")
    return HomogeneousCrystallizationResult(
        output_root=config.output.root_dir,
        replicas=replica_results,
        survival=survival,
    )
