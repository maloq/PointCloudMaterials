from __future__ import annotations

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
    sha256_file,
)
from .checkpoints import CheckpointStore
from .generator import build_calculator
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
    solid_number_density_per_A3: float


@dataclass(frozen=True)
class TransitionBranchResult:
    branch: TransitionBranchConfig
    atoms: Atoms
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
    solid_metadata_path = source_root / config.source_solid_environment / "metadata.json"
    interface_metadata_path = interface_dir / "metadata.json"
    trajectory_path = interface_dir / "trajectory.npz"
    for path in (
        source_manifest_path,
        solid_metadata_path,
        interface_metadata_path,
        trajectory_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Transition source dataset is missing required file: {path}.")

    with source_manifest_path.open("r", encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    observed_sha256 = str(source_manifest["potential_sha256"])
    expected_sha256 = config.generator.potential.sha256
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            "Transition source and configured MACE potential differ: "
            f"source sha256={observed_sha256}, configured sha256={expected_sha256}."
        )
    with interface_metadata_path.open("r", encoding="utf-8") as handle:
        interface_metadata = json.load(handle)
    region_definition = interface_metadata["intermediate_regions"][0]["definition"]
    slab_values = region_definition["slab_bounds_fractional"]
    slab_bounds = (float(slab_values[0]), float(slab_values[1]))
    with solid_metadata_path.open("r", encoding="utf-8") as handle:
        solid_metadata = json.load(handle)
    solid_density = float(solid_metadata["diagnostics"]["number_density_per_A3"])

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
        solid_number_density_per_A3=solid_density,
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


def _simulate_branch(
    prepared: PreparedInterface,
    *,
    config: TransitionConfig,
    branch: TransitionBranchConfig,
    random_seed: int,
    calculator: object,
    checkpoints: CheckpointStore,
    progress: Callable[[str], None],
) -> tuple[Atoms, ThermodynamicTrace]:
    checkpoint = checkpoints.load(branch.name)
    if checkpoint is not None:
        progress(f"{branch.name}: loaded checkpoint from {checkpoints.directory}")
        atoms = checkpoint.atoms
        atoms.calc = calculator
        return atoms, checkpoint.trace

    atoms = prepared.atoms.copy()
    atoms.calc = calculator
    runtime_config = _runtime_generator_config(config, branch)
    trace = run_npt(
        atoms,
        config=runtime_config,
        temperature_K=branch.temperature_K,
        steps=branch.steps,
        stage=f"transition.{branch.name}",
        initialize_velocities=True,
        rng=np.random.default_rng(random_seed),
        progress=progress,
    )
    atoms.wrap()
    checkpoints.save(branch.name, atoms, trace)
    return atoms, trace


def _write_branch(
    directory: Path,
    *,
    prepared: PreparedInterface,
    result: TransitionBranchResult,
    config: TransitionConfig,
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
            profile_bin_centers_fractional=(
                result.analysis.profile_bin_centers_fractional
            ),
            front_displacement_A=result.analysis.front_displacement_A,
        )
    write_phase_rdf_archive(directory / "phase_rdf.npz", result.phase_rdf)
    if config.output.save_extxyz:
        extxyz_atoms = result.atoms.copy()
        extxyz_atoms.arrays["initial_phase_id"] = atom_table["phase_id"]
        write(directory / "structure.extxyz", extxyz_atoms)
    metadata = {
        "schema_version": 1,
        "branch": asdict(result.branch),
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
            "potential_sha256": config.generator.potential.sha256,
            "two_periodic_fronts": True,
            "phase_audit": (
                "PTM is used only to audit phase-front motion. Per-atom phase_id values retain "
                "the initial prepared-region provenance and are not PTM labels."
            ),
        },
        "transition": {
            "net_crystalline_fraction_change": (
                result.analysis.net_crystalline_fraction_change
            ),
            "net_front_displacement_A": result.analysis.net_front_displacement_A,
            "average_front_velocity_m_per_s": (
                result.analysis.average_front_velocity_m_per_s
            ),
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
        )


def _write_dataset(
    config: TransitionConfig,
    prepared: PreparedInterface,
    results: dict[str, TransitionBranchResult],
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
            )
        if config.output.create_visualizations:
            figure, axes = plt.subplots(1, 2, figsize=(15.0, 6.0), constrained_layout=True)
            for axis, branch_name in zip(axes, results):
                image = plt.imread(
                    staging_root
                    / branch_name
                    / "visualizations"
                    / "transition_progress.png"
                )
                axis.imshow(image)
                axis.set_title(branch_name)
                axis.axis("off")
            figure.suptitle("MACE direct-coexistence phase transformations")
            figure.savefig(staging_root / "transition_overview.png", dpi=160)
            plt.close(figure)
            write_phase_rdf_overview(
                staging_root / "phase_rdf_overview.png",
                {
                    branch_name: staging_root
                    / branch_name
                    / "visualizations"
                    / "phase_rdf.png"
                    for branch_name in results
                },
            )
            write_structure_slice_overview(
                staging_root / "structure_slice_overview.png",
                {
                    branch_name: staging_root
                    / branch_name
                    / "visualizations"
                    / "structure_slice.png"
                    for branch_name in results
                },
            )
        manifest = {
            "schema_version": 1,
            "dataset_name": config.dataset_name,
            "branch_dirs": list(results),
            "config": config.to_dict(),
            "potential_sha256": sha256_file(config.generator.potential.model_path),
            "scientific_scope": {
                "supported_claim": (
                    "Observation of seeded planar crystal growth and melting under the selected "
                    "MACE Hamiltonian, pressure, temperatures, and finite simulation time."
                ),
                "unsupported_claim": (
                    "Homogeneous nucleation rates, equilibrium melting temperature, or "
                    "potential-independent phase kinetics from these two trajectories alone."
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
) -> TransitionGenerationResult:
    prepared = _load_prepared_interface(config)
    selected_calculator = (
        calculator if calculator is not None else build_calculator(config.generator)
    )
    progress(
        f"Generating {config.dataset_name!r}: {len(prepared.atoms)} atoms, "
        "direct-coexistence crystallization and melting branches"
    )
    checkpoints = CheckpointStore(config)
    seeds = np.random.SeedSequence(config.random_seed).spawn(2)
    prepared_phase_ids = np.fromiter(
        (PHASE_TO_ID[str(name)] for name in prepared.labels.phase_names),
        dtype=np.int64,
        count=len(prepared.atoms),
    )
    results: dict[str, TransitionBranchResult] = {}
    for branch, seed in zip((config.crystallization, config.melting), seeds):
        atoms, trace = _simulate_branch(
            prepared,
            config=config,
            branch=branch,
            random_seed=int(seed.generate_state(1)[0]),
            calculator=selected_calculator,
            checkpoints=checkpoints,
            progress=progress,
        )
        analysis = analyze_transition(
            trace,
            chemical_symbol=config.generator.system.chemical_symbol,
            timestep_fs=config.generator.dynamics.timestep_fs,
            slab_bounds_fractional=prepared.slab_bounds_fractional,
            solid_number_density_per_A3=prepared.solid_number_density_per_A3,
            profile_bins=config.analysis.profile_bins,
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
            branch_name=branch.name,
            progress=progress,
        )
        runtime_config = _runtime_generator_config(config, branch)
        diagnostics = diagnose_system(
            atoms,
            trace,
            runtime_config,
            name=branch.name,
            require_pressure_convergence=True,
        )
        results[branch.name] = TransitionBranchResult(
            branch=branch,
            atoms=atoms,
            trace=trace,
            analysis=analysis,
            phase_rdf=phase_rdf,
            diagnostics=diagnostics,
        )
    branch_dirs = _write_dataset(config, prepared, results)
    progress(f"Wrote {len(branch_dirs)} transition branches to {config.output.root_dir}")
    return TransitionGenerationResult(
        output_root=config.output.root_dir,
        branch_dirs=branch_dirs,
        branches=results,
    )
