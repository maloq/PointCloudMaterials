from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
from ase import Atoms

from .artifacts import _write_environment, label_bulk
from .checkpoints import CheckpointStore
from .config import GeneratorConfig, validate_potential_qualification
from .generator import select_calculator
from .provenance import (
    ExecutionProvenance,
    homogeneous_liquid_source_producer_code_provenance,
)
from .simulation import ThermodynamicTrace, build_initial_solid, run_npt
from .validation import SystemDiagnostics, diagnose_system


LIQUID_ENVIRONMENT_FORMAT = "replica_{index:03d}_bulk_liquid"


@dataclass(frozen=True)
class HomogeneousLiquidSourceResult:
    output_root: Path
    environment_dirs: tuple[Path, ...]
    diagnostics: dict[str, SystemDiagnostics]


def _quench_liquid(
    atoms: Atoms,
    *,
    config: GeneratorConfig,
    rng: np.random.Generator,
    replica_name: str,
    progress: Callable[[str], None],
) -> None:
    total_steps = config.dynamics.quench_steps
    if total_steps == 0:
        return
    stage_count = min(config.dynamics.quench_stages, total_steps)
    base_steps, remainder = divmod(total_steps, stage_count)
    temperatures_K = np.linspace(
        config.dynamics.melt_temperature_K,
        config.dynamics.target_temperature_K,
        stage_count + 1,
        dtype=np.float64,
    )[1:]
    for stage_index, temperature_K in enumerate(temperatures_K):
        stage_steps = base_steps + (1 if stage_index < remainder else 0)
        run_npt(
            atoms,
            config=config,
            temperature_K=float(temperature_K),
            steps=stage_steps,
            stage=f"{replica_name}.liquid.quench_{stage_index + 1:02d}",
            initialize_velocities=False,
            rng=rng,
            progress=progress,
        )


def _simulate_liquid(
    config: GeneratorConfig,
    *,
    calculator: object,
    execution_provenance: ExecutionProvenance,
    replica_name: str,
    random_seed: int,
    progress: Callable[[str], None],
) -> tuple[Atoms, ThermodynamicTrace]:
    checkpoints = CheckpointStore(config, execution_provenance)
    solid_seed, liquid_seed = np.random.SeedSequence(random_seed).spawn(2)
    solid_rng = np.random.default_rng(solid_seed)
    liquid_rng = np.random.default_rng(liquid_seed)

    solid_stage = f"{replica_name}.homogeneous_source_solid"
    solid_checkpoint = checkpoints.load(solid_stage)
    if solid_checkpoint is None:
        solid = build_initial_solid(config)
        solid.calc = calculator
        solid_trace = run_npt(
            solid,
            config=config,
            temperature_K=config.dynamics.target_temperature_K,
            steps=config.dynamics.solid_equilibration_steps,
            stage=f"{replica_name}.solid.equilibrate",
            initialize_velocities=True,
            rng=solid_rng,
            progress=progress,
        )
        solid.wrap()
        checkpoints.save(
            solid_stage,
            solid,
            solid_trace,
            metadata={"purpose": "liquid-source parent solid"},
        )
    else:
        progress(f"{solid_stage}: loaded checkpoint from {checkpoints.directory}")
        solid = solid_checkpoint.atoms
        solid.calc = calculator

    liquid_stage = f"{replica_name}.homogeneous_source_liquid"
    liquid_checkpoint = checkpoints.load(liquid_stage)
    if liquid_checkpoint is not None:
        progress(f"{liquid_stage}: loaded checkpoint from {checkpoints.directory}")
        liquid = liquid_checkpoint.atoms
        liquid.calc = calculator
        return liquid, liquid_checkpoint.trace

    liquid = solid.copy()
    liquid.calc = calculator
    run_npt(
        liquid,
        config=config,
        temperature_K=config.dynamics.melt_temperature_K,
        steps=config.dynamics.melt_steps,
        stage=f"{replica_name}.liquid.melt",
        initialize_velocities=True,
        rng=liquid_rng,
        progress=progress,
    )
    _quench_liquid(
        liquid,
        config=config,
        rng=liquid_rng,
        replica_name=replica_name,
        progress=progress,
    )
    liquid_trace = run_npt(
        liquid,
        config=config,
        temperature_K=config.dynamics.target_temperature_K,
        steps=config.dynamics.target_equilibration_steps,
        stage=f"{replica_name}.liquid.equilibrate",
        initialize_velocities=False,
        rng=liquid_rng,
        progress=progress,
    )
    liquid.wrap()
    checkpoints.save(
        liquid_stage,
        liquid,
        liquid_trace,
        metadata={
            "purpose": "immutable homogeneous-crystallization liquid source",
            "interface_preparation_performed": False,
        },
    )
    return liquid, liquid_trace


def _write_liquid_source(
    config: GeneratorConfig,
    *,
    replicas: tuple[
        tuple[str, int, Atoms, ThermodynamicTrace, SystemDiagnostics], ...
    ],
    execution_provenance: ExecutionProvenance,
) -> tuple[Path, ...]:
    output_root = config.output.root_dir
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.liquid-source-staging-",
            dir=output_root.parent,
        )
    )
    try:
        for environment_name, random_seed, atoms, trace, diagnostics in replicas:
            _write_environment(
                staging_root / environment_name,
                name=environment_name,
                atoms=atoms,
                labels=label_bulk(len(atoms), "liquid_bulk", grain_id=1),
                trace=trace,
                diagnostics=diagnostics,
                config=config,
                execution_provenance=execution_provenance,
                random_seed=random_seed,
            )
        manifest = {
            "schema_version": 4,
            "dataset_name": config.dataset_name,
            "source_kind": "immutable_homogeneous_liquid_only",
            "interface_preparation_performed": False,
            "environment_dirs": [item[0] for item in replicas],
            "config": config.to_dict(),
            "potential_sha256": execution_provenance.calculator.model_sha256,
            "execution_provenance": execution_provenance.to_dict(),
            "potential_usage_mode": execution_provenance.calculator.usage_mode,
            "scientifically_qualified_potential": (
                execution_provenance.calculator.scientifically_qualified
            ),
            "random_seeds": [item[1] for item in replicas],
            "repository_reference_number_densities_per_A3": None,
            "scientific_scope": {
                "supported_claim": (
                    "A reusable pressure- and temperature-equilibrated bulk-liquid starting "
                    "configuration for independent homogeneous-crystallization replicas under "
                    "the exactly recorded Hamiltonian and preparation protocol."
                ),
                "unsupported_claim": (
                    "An equilibrium melting point, a nucleation rate, or any solid-liquid "
                    "interface property. This liquid-only producer deliberately performs no "
                    "interface preparation."
                ),
            },
        }
        with (staging_root / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        staging_root.replace(output_root)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return tuple(output_root / item[0] for item in replicas)


def generate_homogeneous_liquid_source(
    config: GeneratorConfig,
    *,
    progress: Callable[[str], None] = print,
    calculator: object | None = None,
    injected_calculator_identity: str | None = None,
) -> HomogeneousLiquidSourceResult:
    """Generate only the reusable liquid artifact required by homogeneous runs."""
    if config.output.overwrite:
        raise ValueError(
            "Homogeneous liquid sources are immutable: output.overwrite must be false. "
            "Select a new output.root_dir when the model or preparation protocol changes."
        )
    if config.output.root_dir.exists():
        raise FileExistsError(
            f"Immutable homogeneous liquid source already exists: {config.output.root_dir}. "
            "Reuse that path or select a new path; this producer never replaces it."
        )
    validate_potential_qualification(
        config,
        chemical_symbol=config.system.chemical_symbol,
        pressure_GPa=config.dynamics.pressure_GPa,
        timestep_fs=config.dynamics.timestep_fs,
        state_temperatures_K={
            "liquid_bulk": (
                config.dynamics.target_temperature_K,
                config.dynamics.melt_temperature_K,
            )
        },
        context=f"homogeneous liquid source {config.dataset_name!r}",
        required_claim="kinetics",
    )
    selected_calculator, execution_provenance = select_calculator(
        config,
        calculator=calculator,
        injected_calculator_identity=injected_calculator_identity,
    )
    execution_provenance = replace(
        execution_provenance,
        producer_code=homogeneous_liquid_source_producer_code_provenance(),
    )
    prepared: list[
        tuple[str, int, Atoms, ThermodynamicTrace, SystemDiagnostics]
    ] = []
    diagnostics_by_name: dict[str, SystemDiagnostics] = {}
    for replica_index, random_seed in enumerate(config.random_seeds):
        replica_name = f"replica_{replica_index:03d}"
        environment_name = LIQUID_ENVIRONMENT_FORMAT.format(index=replica_index)
        progress(
            f"{environment_name}: preparing liquid-only source with random_seed={random_seed}"
        )
        liquid, liquid_trace = _simulate_liquid(
            config,
            calculator=selected_calculator,
            execution_provenance=execution_provenance,
            replica_name=replica_name,
            random_seed=random_seed,
            progress=progress,
        )
        diagnostics = diagnose_system(
            liquid,
            liquid_trace,
            config,
            name=environment_name,
            require_pressure_convergence=True,
        )
        crystalline_fraction = sum(
            diagnostics.ptm_structure_fractions[name]
            for name in ("fcc", "hcp", "bcc")
        )
        maximum_fraction = config.validation.maximum_liquid_crystalline_fraction
        if crystalline_fraction > maximum_fraction:
            raise RuntimeError(
                f"{environment_name}: PTM recognizes crystalline fraction="
                f"{crystalline_fraction:.6f}, above "
                "validation.maximum_liquid_crystalline_fraction="
                f"{maximum_fraction:.6f}. Increase melting or liquid equilibration before "
                "using this state for homogeneous nucleation."
            )
        prepared.append(
            (environment_name, random_seed, liquid, liquid_trace, diagnostics)
        )
        diagnostics_by_name[environment_name] = diagnostics
    environment_dirs = _write_liquid_source(
        config,
        replicas=tuple(prepared),
        execution_provenance=execution_provenance,
    )
    progress(
        f"Wrote {len(environment_dirs)} immutable liquid-only environments to "
        f"{config.output.root_dir}"
    )
    return HomogeneousLiquidSourceResult(
        output_root=config.output.root_dir,
        environment_dirs=environment_dirs,
        diagnostics=diagnostics_by_name,
    )
