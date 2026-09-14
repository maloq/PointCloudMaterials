"""Repository-bound production assembly for jumpy FFS."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers

from .artifacts import PHASE_TO_ID, sha256_file
from .config import validate_potential_qualification
from .generator import select_calculator
from .jumpy_ffs import JumpyFFSResult, run_jumpy_ffs
from .jumpy_ffs_config import JumpyFFSRunConfig
from .jumpy_ffs_engine import (
    LangevinNVTShotEngine,
    PTMLargestCrystallineClusterCV,
)
from .provenance import validate_configured_source_manifest
from .simulation import build_initial_solid


def _canonical_sha256(value: object) -> str:
    serialized = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(contiguous.shape, separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _load_source_atoms(
    config: JumpyFFSRunConfig,
) -> tuple[Atoms, dict[str, object]]:
    observed_selection_sha256 = sha256_file(config.potential_selection_report)
    if observed_selection_sha256 != config.potential_selection_report_sha256:
        raise RuntimeError(
            "Potential-selection report changed after jFFS configuration load: path="
            f"{config.potential_selection_report}, loaded_sha256="
            f"{config.potential_selection_report_sha256}, observed_sha256="
            f"{observed_selection_sha256}. Reload the configuration and revalidate the "
            "selected generator before running."
        )
    observed_generator_sha256 = sha256_file(config.generator.config_path)
    if observed_generator_sha256 != config.selected_generator_config_sha256:
        raise RuntimeError(
            "Selected source generator config changed after jFFS configuration load: "
            f"path={config.generator.config_path}, selected_sha256="
            f"{config.selected_generator_config_sha256}, observed_sha256="
            f"{observed_generator_sha256}."
        )
    source_root = config.source_dataset
    source_directory = source_root / config.source_environment
    manifest_path = source_root / "manifest.json"
    atom_table_path = source_directory / "atoms_full.npy"
    trajectory_path = source_directory / "trajectory.npz"
    metadata_path = source_directory / "metadata.json"
    for path in (manifest_path, atom_table_path, trajectory_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"jFFS source is missing required repository artifact: {path}."
            )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_configured_source_manifest(
        manifest,
        config=config.generator,
        manifest_path=manifest_path,
    )
    if manifest.get("source_kind") != "immutable_homogeneous_liquid_only":
        raise RuntimeError(
            f"{manifest_path}: jFFS requires source_kind="
            "'immutable_homogeneous_liquid_only', got "
            f"{manifest.get('source_kind')!r}. Generate a dedicated 500 K NPT liquid "
            "source; do not couple rare-event shots to an active phase-context run."
        )
    if manifest.get("interface_preparation_performed") is not False:
        raise RuntimeError(
            f"{manifest_path}: immutable liquid source must explicitly record "
            "interface_preparation_performed=false."
        )
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    structure_fractions = metadata["diagnostics"]["ptm_structure_fractions"]
    crystalline_fraction = float(
        sum(structure_fractions[name] for name in ("fcc", "hcp", "bcc"))
    )
    maximum_crystalline_fraction = (
        config.generator.validation.maximum_liquid_crystalline_fraction
    )
    if crystalline_fraction > maximum_crystalline_fraction:
        raise RuntimeError(
            f"{metadata_path}: source crystalline fraction={crystalline_fraction:.6f} "
            "exceeds validation.maximum_liquid_crystalline_fraction="
            f"{maximum_crystalline_fraction:.6f}."
        )
    atom_table = np.load(atom_table_path, mmap_mode="r")
    liquid_phase_id = PHASE_TO_ID["liquid_bulk"]
    non_liquid = np.flatnonzero(atom_table["phase_id"] != liquid_phase_id)
    if len(non_liquid):
        raise RuntimeError(
            f"{atom_table_path}: jFFS homogeneous nucleation requires a bulk-liquid "
            f"source, but {len(non_liquid)} atoms do not have phase_id="
            f"{liquid_phase_id}."
        )
    with np.load(trajectory_path) as trajectory:
        frame_indices = np.flatnonzero(
            trajectory["step"] == config.source_frame_step
        )
        if len(frame_indices) != 1:
            raise RuntimeError(
                f"{trajectory_path}: expected exactly one source frame at step="
                f"{config.source_frame_step}, found indices={frame_indices.tolist()}."
            )
        frame_index = int(frame_indices[0])
        stored_positions_A = np.asarray(
            trajectory["positions_A"][frame_index]
        ).copy()
        positions_A = np.asarray(
            stored_positions_A, dtype=np.float64
        )
        cell_A = np.asarray(
            trajectory["cell_vectors_A"][frame_index], dtype=np.float64
        )
        recorded_volume_A3 = float(trajectory["volume_A3"][frame_index])
        source_temperature_K = float(trajectory["temperature_K"][frame_index])
        source_pressure_GPa = float(trajectory["pressure_GPa"][frame_index])
    if atom_table["position"].shape != stored_positions_A.shape:
        raise RuntimeError(
            f"{atom_table_path}: position shape={atom_table['position'].shape} does not "
            f"match {trajectory_path} endpoint shape={stored_positions_A.shape}."
        )
    if not np.array_equal(atom_table["position"], stored_positions_A):
        maximum_difference_A = float(
            np.max(
                np.abs(
                    np.asarray(atom_table["position"], dtype=np.float64)
                    - positions_A
                )
            )
        )
        raise RuntimeError(
            f"{atom_table_path}: final atom-table positions do not exactly match source "
            f"trajectory endpoint step={config.source_frame_step}; "
            f"maximum_absolute_difference_A={maximum_difference_A:.12g}."
        )
    cell_volume_A3 = float(np.linalg.det(cell_A))
    if not np.isclose(
        cell_volume_A3, recorded_volume_A3, rtol=1.0e-10, atol=1.0e-8
    ):
        raise RuntimeError(
            f"{trajectory_path}: source frame cell determinant={cell_volume_A3:.12g} "
            f"A^3 differs from recorded volume={recorded_volume_A3:.12g} A^3."
        )
    expected_count = len(build_initial_solid(config.generator))
    if positions_A.shape != (expected_count, 3):
        raise RuntimeError(
            f"{trajectory_path}: source positions have shape={positions_A.shape}, "
            f"expected {(expected_count, 3)} from the repository generator."
        )
    numbers = np.full(
        expected_count,
        atomic_numbers[config.generator.system.chemical_symbol],
        dtype=np.int32,
    )
    atoms = Atoms(numbers=numbers, positions=positions_A, cell=cell_A, pbc=True)
    source_evidence = {
        "potential_selection_report": {
            "path": str(config.potential_selection_report),
            "sha256": config.potential_selection_report_sha256,
        },
        "source_dataset": str(source_root),
        "source_environment": config.source_environment,
        "source_frame_step": config.source_frame_step,
        "endpoint": {
            "atom_count": expected_count,
            "volume_A3": recorded_volume_A3,
            "number_density_per_A3": expected_count / recorded_volume_A3,
            "instantaneous_temperature_K": source_temperature_K,
            "instantaneous_pressure_GPa": source_pressure_GPa,
            "positions_sha256": _array_sha256(positions_A),
            "cell_sha256": _array_sha256(cell_A),
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "config_sha256": _canonical_sha256(manifest["config"]),
        },
        "trajectory": {
            "path": str(trajectory_path),
            "sha256": sha256_file(trajectory_path),
        },
        "atom_table": {
            "path": str(atom_table_path),
            "sha256": sha256_file(atom_table_path),
        },
        "environment_metadata": {
            "path": str(metadata_path),
            "sha256": sha256_file(metadata_path),
            "ptm_crystalline_fraction": crystalline_fraction,
        },
        "source_generator_config": {
            "path": str(config.generator.config_path),
            "sha256": config.selected_generator_config_sha256,
            "parsed_config_sha256": _canonical_sha256(config.generator.to_dict()),
        },
        "jumpy_ffs_config": {
            "path": str(config.config_path),
            "sha256": sha256_file(config.config_path),
            "parsed_config_sha256": _canonical_sha256(config.to_dict()),
        },
    }
    return atoms, source_evidence


def generate_jumpy_ffs(
    config: JumpyFFSRunConfig,
    *,
    resume: bool,
    progress: Callable[[str], None] = print,
    calculator: object | None = None,
    injected_calculator_identity: str | None = None,
) -> JumpyFFSResult:
    """Run jFFS with a manifest-verified source and calculator identity."""

    source_atoms, source_evidence = _load_source_atoms(config)
    shot_generator = replace(
        config.generator,
        potential=replace(
            config.generator.potential,
            md_property_mode=config.shot_md_property_mode,
        ),
    )
    validate_potential_qualification(
        shot_generator,
        chemical_symbol=shot_generator.system.chemical_symbol,
        pressure_GPa=shot_generator.dynamics.pressure_GPa,
        timestep_fs=config.timestep_fs,
        state_temperatures_K={
            "liquid_bulk": (config.temperature_K,),
            "nucleus": (config.temperature_K,),
        },
        context=f"jumpy FFS force-only shot calculator {config.dataset_name!r}",
        required_claim="kinetics",
    )
    selected_calculator, execution_provenance = select_calculator(
        shot_generator,
        calculator=calculator,
        injected_calculator_identity=injected_calculator_identity,
    )
    execution_identity = execution_provenance.to_dict()
    if calculator is None:
        property_mode_setter = getattr(
            selected_calculator, "set_md_property_mode", None
        )
        if not callable(property_mode_setter):
            raise TypeError(
                "Configured jFFS production calculator lacks the audited "
                "set_md_property_mode('forces') fast path. Fixed-cell NVT must not "
                "silently compute unused stress every MD step."
            )
        property_mode_setter(config.shot_md_property_mode)
        observed_property_mode = getattr(
            selected_calculator, "md_property_mode", None
        )
        if observed_property_mode != config.shot_md_property_mode:
            raise RuntimeError(
                "Configured jFFS calculator rejected the required forces-only MD mode: "
                f"observed md_property_mode={observed_property_mode!r}."
            )
        execution_identity["jumpy_ffs_runtime"] = {
            "md_property_mode": config.shot_md_property_mode,
            "stress_evaluated_during_nvt_steps": False,
            "reason": "fixed-cell Langevin-NVT requires forces but not virial stress",
        }
    else:
        execution_identity["jumpy_ffs_runtime"] = {
            "md_property_mode": "ASE_requested_properties",
            "stress_evaluated_during_nvt_steps": None,
            "reason": "injected test/controlled-study calculator; generic ASE contract",
        }
    engine = LangevinNVTShotEngine(
        selected_calculator,
        execution_provenance=execution_identity,
        temperature_K=config.temperature_K,
        timestep_fs=config.timestep_fs,
        friction_time_fs=config.friction_time_fs,
    )
    velocity_seed = int(
        np.random.SeedSequence(
            [config.algorithm.random_seed, 0x56454C]
        ).generate_state(1, dtype=np.uint64)[0]
    )
    initial_state = engine.initialize_state(source_atoms, random_seed=velocity_seed)
    collective_variable = PTMLargestCrystallineClusterCV(
        ptm_rmsd_cutoff=config.cv.ptm_rmsd_cutoff,
        cluster_cutoff_A=config.cv.cluster_cutoff_A,
    )
    qualification = shot_generator.potential.qualification
    potential_is_qualified = bool(
        shot_generator.potential.scientifically_qualified
        and qualification is not None
        and qualification.md_property_mode == "forces"
    )
    scientific_scope = {
        "claim_status": (
            "method_validation_required_with_qualified_potential"
            if potential_is_qualified
            else "exploratory_unqualified_kinetics"
        ),
        "scientifically_qualified_potential": potential_is_qualified,
        "potential_usage_mode": config.generator.potential.usage_mode,
        "supported_claim": (
            "A weighted jFFS rate for this exact finite periodic cell, fixed-volume "
            "Langevin-NVT dynamics, integer connected-PTM cluster coordinate, interface "
            "set, CV cadence, and manifest-bound MLIP."
        ),
        "required_before_physical_aluminium_rate_claim": (
            "Validate interface placement and overlap, progress-coordinate committors, "
            "basin stationarity, CV-cadence convergence, shot-count convergence, "
            "finite-size convergence, fixed-volume sensitivity, and MLIP melting/transport/"
            "solid-liquid-interface kinetics."
        ),
        "uncertainty_method": (
            "Paired circular moving-block bootstrap of consecutive basin inter-exit "
            "times and complete per-exit descendant-tree success probabilities; the "
            "configured block length is measured in basin exits."
        ),
        "boundary_uncertainty_policy": (
            "If every observed descendant tree fails or every tree succeeds, retain the "
            "weighted point estimate but report confidence interval, standard error, and "
            "confidence level as undefined. A degenerate all-zero/all-one bootstrap is "
            "not evidence of zero uncertainty."
        ),
        "restart_checkpoint_policy": (
            "The connected-PTM CV is evaluated at every configured CV interval. Exact "
            "outward-crossing states are committed immediately; ordinary basin progress "
            "is committed only at basin_checkpoint_interval_steps. Equilibration and "
            "active shots follow their own explicit checkpoint intervals, with phase "
            "boundaries, trial starts, returns, and interface landings committed "
            "immediately. A crash discards all later volatile observations and restarts "
            "from the complete phase-space and PCG64 Markov state in the last durable "
            "checkpoint. Statistical correctness does not assume that CUDA/CuEq force "
            "kernels reproduce the discarded numerical path bit-for-bit."
        ),
        "warning": (
            None
            if potential_is_qualified
            else "The selected potential has no qualifying kinetics validation report; "
            "the numerical jFFS rate is exploratory and must not be presented as a "
            "quantitative aluminium nucleation rate."
        ),
    }
    progress(
        f"jFFS {config.dataset_name!r}: atoms={len(source_atoms)}, "
        f"volume={source_atoms.get_volume():.6g} A^3, temperature="
        f"{config.temperature_K:.1f} K, interfaces="
        f"{list(config.algorithm.interfaces_atoms)}, CV cadence="
        f"{config.algorithm.cv_interval_steps * config.timestep_fs:.1f} fs, "
        "checkpoint cadences (equilibration/basin/shot)="
        f"{config.algorithm.equilibration_checkpoint_interval_steps * config.timestep_fs:.1f}/"
        f"{config.algorithm.basin_checkpoint_interval_steps * config.timestep_fs:.1f}/"
        f"{config.algorithm.shot_checkpoint_interval_steps * config.timestep_fs:.1f} fs"
    )
    return run_jumpy_ffs(
        initial_state,
        algorithm=config.algorithm,
        engine=engine,
        collective_variable=collective_variable,
        output_root=config.output_root,
        resume=resume,
        source_evidence=source_evidence,
        scientific_scope=scientific_scope,
        progress=progress,
    )
