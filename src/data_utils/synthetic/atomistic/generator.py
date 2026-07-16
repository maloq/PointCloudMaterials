from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from .artifacts import sha256_file, write_dataset
from .config import (
    GeneratorConfig,
    mace_kernel_backend,
    validate_potential_qualification,
)
from .provenance import (
    ExecutionProvenance,
    build_execution_provenance,
    configured_mace_provenance,
    injected_calculator_provenance,
)
from .simulation import build_initial_solid, simulate_systems
from .validation import SystemDiagnostics, validate_systems


@dataclass(frozen=True)
class GenerationResult:
    output_root: Path
    environment_dirs: tuple[Path, ...]
    diagnostics: dict[str, SystemDiagnostics]
    reference_densities_per_A3: dict[str, float] | None


def build_calculator(config: GeneratorConfig) -> object:
    if not config.potential.model_path.is_file():
        raise FileNotFoundError(
            f"Configured potential model is missing: {config.potential.model_path}. "
            "Place the explicitly selected model below datasets/; generation never downloads "
            "a checkpoint or substitutes another potential."
        )
    observed_sha256 = sha256_file(config.potential.model_path)
    if observed_sha256 != config.potential.sha256:
        raise RuntimeError(
            f"Potential checksum mismatch for {config.potential.model_path}: configured "
            f"sha256={config.potential.sha256}, observed sha256={observed_sha256}. "
            "Do not run a scientific benchmark with an unidentified checkpoint."
        )
    if config.potential.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"potential.device={config.potential.device!r} requires CUDA, but torch reports "
            "that CUDA is unavailable. Select an available device explicitly."
        )
    try:
        from .calculator import VerletSkinMACECalculator
    except ImportError as exc:
        raise ImportError(
            "The force-driven generator requires mace-torch in the pointnet environment."
        ) from exc
    if config.potential.compile_mode is not None:
        produced_atom_count = len(build_initial_solid(config))
        if config.potential.pad_num_atoms != produced_atom_count:
            raise ValueError(
                f"Compiled fixed-shape config {config.config_path} produces "
                f"{produced_atom_count} atoms from system.repetitions, but "
                f"potential.pad_num_atoms={config.potential.pad_num_atoms}. Use the exact "
                "repository-produced atom count; cross-workload padding belongs in a "
                "separate config."
            )
    calculator_kwargs: dict[str, object] = {}
    if config.potential.head is not None:
        calculator_kwargs["head"] = config.potential.head
    calculator = VerletSkinMACECalculator(
        model_paths=str(config.potential.model_path),
        device=config.potential.device,
        default_dtype=config.potential.default_dtype,
        enable_cueq=config.potential.enable_cueq,
        enable_oeq=config.potential.enable_oeq,
        compile_mode=config.potential.compile_mode,
        fullgraph=config.potential.compile_fullgraph,
        pad_num_atoms=config.potential.pad_num_atoms,
        pad_num_edges=config.potential.pad_num_edges,
        md_property_mode=config.potential.md_property_mode,
        neighbor_skin_A=config.potential.neighbor_skin_A,
        **calculator_kwargs,
    )
    available_heads = tuple(str(head) for head in calculator.available_heads)
    configured_head = config.potential.head
    if configured_head is None:
        if len(available_heads) != 1:
            raise RuntimeError(
                f"Configured MACE model {config.potential.model_path} exposes heads="
                f"{list(available_heads)}, but potential.head is null. Select one head "
                "explicitly; multi-head energies and forces are different Hamiltonians."
            )
        expected_head = available_heads[0]
    else:
        if configured_head not in available_heads:
            raise RuntimeError(
                f"Configured potential.head={configured_head!r} does not exist in MACE model "
                f"{config.potential.model_path}; available_heads={list(available_heads)}. "
                "MACE may warn and fall back to a different head, which this generator forbids."
            )
        expected_head = configured_head
    selected_head = str(calculator.head)
    if selected_head != expected_head:
        raise RuntimeError(
            f"MACE selected head={selected_head!r} for {config.potential.model_path}, but the "
            f"validated expected head is {expected_head!r}; available_heads="
            f"{list(available_heads)}."
        )
    expected_backend = mace_kernel_backend(
        enable_cueq=config.potential.enable_cueq,
        enable_oeq=config.potential.enable_oeq,
    )
    if calculator.kernel_backend != expected_backend:
        raise RuntimeError(
            f"MACE calculator initialized kernel_backend={calculator.kernel_backend!r}, "
            f"but config requires {expected_backend!r}. Backend substitution is forbidden."
        )
    if bool(calculator.use_compile) != (config.potential.compile_mode is not None):
        raise RuntimeError(
            f"MACE calculator use_compile={calculator.use_compile!r} does not match "
            f"configured compile_mode={config.potential.compile_mode!r}."
        )
    return calculator


def select_calculator(
    config: GeneratorConfig,
    *,
    calculator: object | None,
    injected_calculator_identity: str | None,
) -> tuple[object, ExecutionProvenance]:
    if calculator is None:
        if injected_calculator_identity is not None:
            raise ValueError(
                "injected_calculator_identity was provided without an injected calculator."
            )
        selected_calculator = build_calculator(config)
        calculator_provenance = configured_mace_provenance(
            config, selected_calculator
        )
    else:
        if injected_calculator_identity is None:
            raise ValueError(
                "An injected calculator requires injected_calculator_identity. This prevents "
                "test or alternate potentials from being persisted as the configured MACE "
                "Hamiltonian."
            )
        if config.potential.usage_mode == "quantitative":
            raise RuntimeError(
                "A quantitative qualification report is bound to the configured MACE model, "
                "head, and calculator settings. An injected calculator cannot inherit that "
                "qualification; use an explicit exploratory configuration for controlled "
                "alternate-calculator studies."
            )
        selected_calculator = calculator
        calculator_provenance = injected_calculator_provenance(
            calculator,
            identity=injected_calculator_identity,
        )
    return selected_calculator, build_execution_provenance(calculator_provenance)


def generate_dataset(
    config: GeneratorConfig,
    *,
    progress: Callable[[str], None] = print,
    calculator: object | None = None,
    injected_calculator_identity: str | None = None,
) -> GenerationResult:
    """Generate and persist the complete force-driven benchmark.

    ``calculator`` exists for deterministic unit tests and controlled potential
    studies. Injected calculators require a unique explicit identity and are
    recorded as injected rather than as the config-selected MACE model.
    """

    if config.output.root_dir.exists() and not config.output.overwrite:
        raise FileExistsError(
            f"Output directory already exists: {config.output.root_dir}. Set "
            "output.overwrite=true only when replacement is intended. This check is "
            "performed before calculator construction or MD so an existing dataset cannot "
            "waste a production run."
        )

    validate_potential_qualification(
        config,
        chemical_symbol=config.system.chemical_symbol,
        pressure_GPa=config.dynamics.pressure_GPa,
        timestep_fs=config.dynamics.timestep_fs,
        state_temperatures_K={
            "solid_bulk": (config.dynamics.target_temperature_K,),
            "liquid_bulk": (
                config.dynamics.target_temperature_K,
                config.dynamics.melt_temperature_K,
            ),
            "interface": (
                config.dynamics.target_temperature_K,
                config.dynamics.melt_temperature_K,
            ),
        },
        context=f"base phase generation {config.dataset_name!r}",
        required_claim="phase_context_structure",
    )
    selected_calculator, execution_provenance = select_calculator(
        config,
        calculator=calculator,
        injected_calculator_identity=injected_calculator_identity,
    )
    atom_count = len(build_initial_solid(config))
    progress(
        f"Generating {config.dataset_name!r}: {len(config.random_seeds)} replicas, "
        f"{atom_count} {config.system.chemical_symbol} atoms each, "
        f"calculator={execution_provenance.calculator.identity!r}"
    )
    replicas = {}
    diagnostics = {}
    reference_densities = None
    for replica_index, random_seed in enumerate(config.random_seeds):
        replica_name = f"replica_{replica_index:03d}"
        progress(f"{replica_name}: random_seed={random_seed}")
        systems = simulate_systems(
            config,
            selected_calculator,
            random_seed=random_seed,
            checkpoint_prefix=replica_name,
            execution_provenance=execution_provenance,
            progress=progress,
        )
        replica_diagnostics, observed_reference = validate_systems(systems, config)
        replicas[replica_name] = (random_seed, systems)
        diagnostics.update(
            {
                f"{replica_name}_{environment_name}": value
                for environment_name, value in replica_diagnostics.items()
            }
        )
        if reference_densities is None:
            reference_densities = observed_reference
        elif observed_reference != reference_densities:
            raise RuntimeError(
                "Repository density references changed while validating replicas: "
                f"first={reference_densities}, {replica_name}={observed_reference}."
            )
    environment_dirs = write_dataset(
        replicas,
        diagnostics,
        reference_densities,
        config,
        execution_provenance=execution_provenance,
    )
    progress(f"Wrote {len(environment_dirs)} environments to {config.output.root_dir}")
    return GenerationResult(
        output_root=config.output.root_dir,
        environment_dirs=environment_dirs,
        diagnostics=diagnostics,
        reference_densities_per_A3=reference_densities,
    )
