from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from .artifacts import sha256_file, write_dataset
from .config import GeneratorConfig
from .simulation import build_initial_solid, simulate_systems
from .validation import SystemDiagnostics, validate_systems


@dataclass(frozen=True)
class GenerationResult:
    output_root: Path
    environment_dirs: tuple[Path, ...]
    diagnostics: dict[str, SystemDiagnostics]
    reference_densities_per_A3: dict[str, float] | None


def _build_calculator(config: GeneratorConfig) -> object:
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
    return VerletSkinMACECalculator(
        model_paths=str(config.potential.model_path),
        device=config.potential.device,
        default_dtype=config.potential.default_dtype,
        enable_cueq=config.potential.enable_cueq,
        neighbor_skin_A=config.potential.neighbor_skin_A,
    )


def generate_dataset(
    config: GeneratorConfig,
    *,
    progress: Callable[[str], None] = print,
    calculator: object | None = None,
) -> GenerationResult:
    """Generate and persist the complete force-driven benchmark.

    ``calculator`` exists for deterministic unit tests and controlled potential
    studies. Production callers should leave it unset so the config-selected,
    hash-recorded MACE model is used.
    """

    selected_calculator = calculator if calculator is not None else _build_calculator(config)
    atom_count = len(build_initial_solid(config))
    progress(
        f"Generating {config.dataset_name!r}: {len(config.random_seeds)} replicas, "
        f"{atom_count} {config.system.chemical_symbol} atoms each, "
        f"cuEquivariance={config.potential.enable_cueq}, "
        f"neighbor_skin={config.potential.neighbor_skin_A:g} A"
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
    )
    progress(f"Wrote {len(environment_dirs)} environments to {config.output.root_dir}")
    return GenerationResult(
        output_root=config.output.root_dir,
        environment_dirs=environment_dirs,
        diagnostics=diagnostics,
        reference_densities_per_A3=reference_densities,
    )
