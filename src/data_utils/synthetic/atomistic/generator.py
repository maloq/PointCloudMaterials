from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from .artifacts import sha256_file, write_dataset
from .config import GeneratorConfig
from .simulation import simulate_systems
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
        from mace.calculators import MACECalculator
    except ImportError as exc:
        raise ImportError(
            "The force-driven generator requires mace-torch in the pointnet environment."
        ) from exc
    return MACECalculator(
        model_paths=str(config.potential.model_path),
        device=config.potential.device,
        default_dtype=config.potential.default_dtype,
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
    progress(
        f"Generating {config.dataset_name!r} with {config.system.chemical_symbol}, "
        f"{config.system.repetitions} conventional cells"
    )
    systems = simulate_systems(config, selected_calculator, progress=progress)
    diagnostics, reference_densities = validate_systems(systems, config)
    environment_dirs = write_dataset(
        systems,
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
