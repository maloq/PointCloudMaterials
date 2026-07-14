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
    sha256_file,
)
from .checkpoints import CheckpointStore
from .generator import build_calculator
from .homogeneous_analysis import (
    HomogeneousCrystallizationAnalysis,
    analyze_homogeneous_crystallization,
    write_homogeneous_progress_visualization,
    write_homogeneous_rdf_visualization,
)
from .homogeneous_config import HomogeneousCrystallizationConfig
from .simulation import ThermodynamicTrace, build_initial_solid, run_npt
from .transition_analysis import STRUCTURE_NAMES, write_structure_slice_visualization
from .validation import SystemDiagnostics, diagnose_system


RUN_DIRECTORY_NAME = "homogeneous_crystallization"


@dataclass(frozen=True)
class SourceLiquid:
    atoms: Atoms
    temperature_K: float
    pressure_GPa: float
    volume_A3: float
    crystalline_fraction: float


@dataclass(frozen=True)
class HomogeneousCrystallizationResult:
    output_root: Path
    run_dir: Path
    atoms: Atoms
    trace: ThermodynamicTrace
    analysis: HomogeneousCrystallizationAnalysis
    diagnostics: SystemDiagnostics


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
    observed_sha256 = str(source_manifest["potential_sha256"])
    expected_sha256 = config.generator.potential.sha256
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            "Source liquid and configured MACE potential differ: "
            f"source sha256={observed_sha256}, configured sha256={expected_sha256}."
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


def _simulate(
    source: SourceLiquid,
    *,
    config: HomogeneousCrystallizationConfig,
    calculator: object,
    checkpoints: CheckpointStore,
    progress: Callable[[str], None],
) -> tuple[Atoms, ThermodynamicTrace]:
    stage = "homogeneous_crystallization"
    checkpoint = checkpoints.load(stage)
    if checkpoint is not None:
        progress(f"{stage}: loaded checkpoint from {checkpoints.directory}")
        atoms = checkpoint.atoms
        atoms.calc = calculator
        return atoms, checkpoint.trace

    atoms = source.atoms.copy()
    atoms.calc = calculator
    trace = run_npt(
        atoms,
        config=_runtime_generator_config(config),
        temperature_K=config.temperature_K,
        steps=config.steps,
        stage=stage,
        initialize_velocities=True,
        rng=np.random.default_rng(config.random_seed),
        progress=progress,
    )
    atoms.wrap()
    checkpoints.save(
        stage,
        atoms,
        trace,
        metadata={
            "source_dataset": str(config.source_dataset),
            "source_environment": config.source_environment,
            "source_frame_step": config.source_frame_step,
        },
    )
    return atoms, trace


def _write_run(
    directory: Path,
    *,
    source: SourceLiquid,
    atoms: Atoms,
    trace: ThermodynamicTrace,
    analysis: HomogeneousCrystallizationAnalysis,
    diagnostics: SystemDiagnostics,
    config: HomogeneousCrystallizationConfig,
) -> None:
    directory.mkdir(parents=True)
    labels = label_bulk(len(atoms), "liquid_bulk", grain_id=0)
    atom_table = build_atom_table(atoms, labels)
    np.save(directory / "atoms.npy", atom_table["position"])
    np.save(directory / "atoms_full.npy", atom_table)
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

    nucleation = {
        "observed": analysis.nucleation_observed,
        "definition": (
            "first saved frame containing at least nucleus_size_threshold_atoms in one "
            "connected FCC/HCP/BCC cluster"
        ),
        "nucleus_size_threshold_atoms": analysis.nucleus_size_threshold_atoms,
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
        nucleation["onset_step"] = analysis.nucleation_step
        nucleation["onset_time_ps"] = analysis.nucleation_time_ps
    metadata = {
        "schema_version": 1,
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
            "instantaneous_velocity_quench": {
                "from_source_temperature_K": source.temperature_K,
                "to_temperature_K": config.temperature_K,
            },
            "pressure_GPa": config.generator.dynamics.pressure_GPa,
            "timestep_fs": config.generator.dynamics.timestep_fs,
            "duration_ps": (
                config.steps * config.generator.dynamics.timestep_fs / 1000.0
            ),
            "potential_sha256": config.generator.potential.sha256,
            "seeded": False,
            "phase_audit": (
                "PTM and its connected FCC/HCP/BCC clusters are analysis observables only. "
                "Per-atom phase_id retains the initial liquid provenance."
            ),
        },
        "nucleation": nucleation,
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
        )


def _write_overview(output_root: Path, run_dir: Path) -> None:
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
    figure.suptitle("MACE homogeneous crystallization from supercooled liquid")
    figure.savefig(output_root / "crystallization_overview.png", dpi=160)
    plt.close(figure)


def _write_dataset(
    config: HomogeneousCrystallizationConfig,
    *,
    source: SourceLiquid,
    atoms: Atoms,
    trace: ThermodynamicTrace,
    analysis: HomogeneousCrystallizationAnalysis,
    diagnostics: SystemDiagnostics,
) -> Path:
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
        run_dir = staging_root / RUN_DIRECTORY_NAME
        _write_run(
            run_dir,
            source=source,
            atoms=atoms,
            trace=trace,
            analysis=analysis,
            diagnostics=diagnostics,
            config=config,
        )
        if config.output.create_visualizations:
            _write_overview(staging_root, run_dir)
        manifest = {
            "schema_version": 1,
            "dataset_name": config.dataset_name,
            "run_dir": RUN_DIRECTORY_NAME,
            "config": config.to_dict(),
            "potential_sha256": sha256_file(config.generator.potential.model_path),
            "scientific_scope": {
                "supported_claim": (
                    "Whether a threshold-sized connected PTM-crystalline cluster appears "
                    "during this finite, unseeded 500 K trajectory under the selected MACE "
                    "Hamiltonian."
                ),
                "unsupported_claim": (
                    "A homogeneous nucleation rate, critical nucleus size, equilibrium "
                    "melting temperature, or potential-independent kinetics from one 10 ps "
                    "trajectory."
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
    return output_root / RUN_DIRECTORY_NAME


def generate_homogeneous_crystallization_dataset(
    config: HomogeneousCrystallizationConfig,
    *,
    progress: Callable[[str], None] = print,
    calculator: object | None = None,
) -> HomogeneousCrystallizationResult:
    source = _load_source_liquid(config)
    selected_calculator = (
        calculator if calculator is not None else build_calculator(config.generator)
    )
    progress(
        f"Generating {config.dataset_name!r}: {len(source.atoms)} atoms, unseeded "
        f"{config.temperature_K:.0f} K crystallization for "
        f"{config.steps * config.generator.dynamics.timestep_fs / 1000.0:.1f} ps"
    )
    checkpoints = CheckpointStore(config)
    atoms, trace = _simulate(
        source,
        config=config,
        calculator=selected_calculator,
        checkpoints=checkpoints,
        progress=progress,
    )
    analysis = analyze_homogeneous_crystallization(
        trace,
        chemical_symbol=config.generator.system.chemical_symbol,
        timestep_fs=config.generator.dynamics.timestep_fs,
        crystalline_cluster_cutoff_A=(
            config.analysis.crystalline_cluster_cutoff_A
        ),
        nucleus_size_threshold_atoms=(
            config.analysis.nucleus_size_threshold_atoms
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
            "The simulated trajectory does not start from a validated liquid: frame-zero "
            f"PTM crystalline fraction={analysis.crystalline_fraction[0]:.6f}, maximum="
            f"{maximum_initial_crystalline_fraction:.6f}."
        )
    if (
        analysis.largest_crystalline_cluster_atoms[0]
        >= config.analysis.nucleus_size_threshold_atoms
    ):
        raise RuntimeError(
            "The source frame already contains a threshold-sized crystalline cluster: "
            f"largest={analysis.largest_crystalline_cluster_atoms[0]} atoms, threshold="
            f"{config.analysis.nucleus_size_threshold_atoms}. Select an earlier validated "
            "bulk-liquid frame."
        )
    runtime_config = _runtime_generator_config(config)
    diagnostics = diagnose_system(
        atoms,
        trace,
        runtime_config,
        name="homogeneous_crystallization",
        require_pressure_convergence=True,
    )
    run_dir = _write_dataset(
        config,
        source=source,
        atoms=atoms,
        trace=trace,
        analysis=analysis,
        diagnostics=diagnostics,
    )
    progress(f"Wrote homogeneous crystallization dataset to {config.output.root_dir}")
    return HomogeneousCrystallizationResult(
        output_root=config.output.root_dir,
        run_dir=run_dir,
        atoms=atoms,
        trace=trace,
        analysis=analysis,
        diagnostics=diagnostics,
    )
