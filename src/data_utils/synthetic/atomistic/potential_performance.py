from __future__ import annotations

import gc
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import yaml
from ase import Atoms, units
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from .config import REPOSITORY_ROOT, load_config, potential_calculator_settings
from .generator import select_calculator
from .homogeneous_config import load_homogeneous_crystallization_config
from .homogeneous_generator import _load_source_liquid
from .simulation import build_initial_solid

if TYPE_CHECKING:
    from .config import GeneratorConfig


@dataclass(frozen=True)
class PotentialPerformanceConfig:
    model_configs: tuple[Path, ...]
    reference_model_configs: tuple[Path, ...]
    initial_homogeneous_configs: tuple[Path, ...]
    temperature_K: float
    pressure_GPa: float
    timestep_fs: float
    thermostat_time_fs: float
    barostat_time_fs: float
    warmup_steps: int
    measurement_blocks: int
    steps_per_block: int
    random_seed: int
    maximum_parity_energy_difference_meV_per_atom: float
    maximum_parity_force_rmse_eV_per_A: float
    maximum_parity_stress_difference_GPa: float
    output_json: Path
    config_path: Path

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        return _serialize(result)


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    return value


def _repo_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _positive_float(value: object, *, name: str, path: Path) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{path}: {name} must be finite and > 0, got {result}.")
    return result


def _positive_int(value: object, *, name: str, path: Path) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TypeError(f"{path}: {name} must be a positive integer, got {value!r}.")
    return value


def load_potential_performance_config(
    path: str | Path,
) -> PotentialPerformanceConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"{config_path}: root must be a mapping.")
    expected_keys = {
        "model_configs",
        "reference_model_configs",
        "initial_homogeneous_configs",
        "temperature_K",
        "pressure_GPa",
        "timestep_fs",
        "thermostat_time_fs",
        "barostat_time_fs",
        "warmup_steps",
        "measurement_blocks",
        "steps_per_block",
        "random_seed",
        "maximum_parity_energy_difference_meV_per_atom",
        "maximum_parity_force_rmse_eV_per_A",
        "maximum_parity_stress_difference_GPa",
        "output_json",
    }
    if set(raw) != expected_keys:
        raise KeyError(
            f"{config_path}: performance config keys must be exactly "
            f"{sorted(expected_keys)}; observed={sorted(raw)}."
        )
    model_configs_raw = raw["model_configs"]
    if not isinstance(model_configs_raw, list) or len(model_configs_raw) < 2:
        raise TypeError(
            f"{config_path}: model_configs must contain at least two config paths."
        )
    model_configs = tuple(_repo_path(value) for value in model_configs_raw)
    for model_config in model_configs:
        if not model_config.is_file():
            raise FileNotFoundError(
                f"{config_path}: model config does not exist: {model_config}."
            )
    reference_model_configs_raw = raw["reference_model_configs"]
    if (
        not isinstance(reference_model_configs_raw, list)
        or len(reference_model_configs_raw) != len(model_configs)
    ):
        raise TypeError(
            f"{config_path}: reference_model_configs must contain exactly one "
            f"uncompiled reference for each of the {len(model_configs)} production "
            "model configs."
        )
    reference_model_configs = tuple(
        _repo_path(value) for value in reference_model_configs_raw
    )
    for reference_model_config in reference_model_configs:
        if not reference_model_config.is_file():
            raise FileNotFoundError(
                f"{config_path}: reference model config does not exist: "
                f"{reference_model_config}."
            )
    initial_homogeneous_configs_raw = raw["initial_homogeneous_configs"]
    if (
        not isinstance(initial_homogeneous_configs_raw, list)
        or len(initial_homogeneous_configs_raw) != len(model_configs)
    ):
        raise TypeError(
            f"{config_path}: initial_homogeneous_configs must contain exactly one "
            f"model-specific immutable source config for each of the "
            f"{len(model_configs)} production model configs."
        )
    initial_homogeneous_configs = tuple(
        _repo_path(value) for value in initial_homogeneous_configs_raw
    )
    for initial_homogeneous_config in initial_homogeneous_configs:
        if not initial_homogeneous_config.is_file():
            raise FileNotFoundError(
                f"{config_path}: initial homogeneous config does not exist: "
                f"{initial_homogeneous_config}."
            )
    pressure_GPa = float(raw["pressure_GPa"])
    if not np.isfinite(pressure_GPa):
        raise ValueError(
            f"{config_path}: pressure_GPa must be finite, got {pressure_GPa}."
        )
    random_seed = raw["random_seed"]
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise TypeError(
            f"{config_path}: random_seed must be an integer, got {random_seed!r}."
        )
    return PotentialPerformanceConfig(
        model_configs=model_configs,
        reference_model_configs=reference_model_configs,
        initial_homogeneous_configs=initial_homogeneous_configs,
        temperature_K=_positive_float(
            raw["temperature_K"], name="temperature_K", path=config_path
        ),
        pressure_GPa=pressure_GPa,
        timestep_fs=_positive_float(
            raw["timestep_fs"], name="timestep_fs", path=config_path
        ),
        thermostat_time_fs=_positive_float(
            raw["thermostat_time_fs"], name="thermostat_time_fs", path=config_path
        ),
        barostat_time_fs=_positive_float(
            raw["barostat_time_fs"], name="barostat_time_fs", path=config_path
        ),
        warmup_steps=_positive_int(
            raw["warmup_steps"], name="warmup_steps", path=config_path
        ),
        measurement_blocks=_positive_int(
            raw["measurement_blocks"], name="measurement_blocks", path=config_path
        ),
        steps_per_block=_positive_int(
            raw["steps_per_block"], name="steps_per_block", path=config_path
        ),
        random_seed=random_seed,
        maximum_parity_energy_difference_meV_per_atom=_positive_float(
            raw["maximum_parity_energy_difference_meV_per_atom"],
            name="maximum_parity_energy_difference_meV_per_atom",
            path=config_path,
        ),
        maximum_parity_force_rmse_eV_per_A=_positive_float(
            raw["maximum_parity_force_rmse_eV_per_A"],
            name="maximum_parity_force_rmse_eV_per_A",
            path=config_path,
        ),
        maximum_parity_stress_difference_GPa=_positive_float(
            raw["maximum_parity_stress_difference_GPa"],
            name="maximum_parity_stress_difference_GPa",
            path=config_path,
        ),
        output_json=_repo_path(raw["output_json"]),
        config_path=config_path,
    )


def summarize_block_timings(
    block_seconds: np.ndarray, *, steps_per_block: int
) -> dict[str, Any]:
    values = np.asarray(block_seconds, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError(
            "block_seconds must be a finite, non-empty one-dimensional array; "
            f"shape={values.shape}, values={values.tolist()}."
        )
    if np.any(values <= 0.0):
        raise ValueError(f"Every block duration must be positive, got {values.tolist()}.")
    if not isinstance(steps_per_block, int) or steps_per_block <= 0:
        raise TypeError(
            f"steps_per_block must be a positive integer, got {steps_per_block!r}."
        )
    seconds_per_step = values / steps_per_block
    total_steps = int(len(values) * steps_per_block)
    total_seconds = float(values.sum())
    return {
        "block_seconds": values.tolist(),
        "measurement_steps": total_steps,
        "total_seconds": total_seconds,
        "mean_seconds_per_step": float(seconds_per_step.mean()),
        "median_seconds_per_step": float(np.median(seconds_per_step)),
        "maximum_seconds_per_step": float(seconds_per_step.max()),
        "steps_per_second": float(total_steps / total_seconds),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synchronize_cuda(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Performance config requests device='cuda', but torch.cuda.is_available() "
            "is false. Run this benchmark on the intended production GPU."
        )
    torch.cuda.synchronize(torch.device(device))


def _release_cuda_memory(device: str) -> None:
    """Release a completed reference model before loading its production twin."""

    gc.collect()
    if not device.startswith("cuda"):
        return
    import torch

    _synchronize_cuda(device)
    torch.cuda.empty_cache()


def _evaluate_reference_state(
    initial_atoms: Atoms,
    *,
    generator_config: GeneratorConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    atoms = initial_atoms.copy()
    device = generator_config.potential.device
    _synchronize_cuda(device)
    initialization_start = time.perf_counter()
    calculator, provenance = select_calculator(
        generator_config,
        calculator=None,
        injected_calculator_identity=None,
    )
    _synchronize_cuda(device)
    initialization_seconds = time.perf_counter() - initialization_start
    atoms.calc = calculator
    evaluation_start = time.perf_counter()
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    stress_GPa = (
        np.asarray(atoms.get_stress(voigt=False), dtype=np.float64) / units.GPa
    )
    energy_eV = float(atoms.get_potential_energy())
    _synchronize_cuda(device)
    evaluation_seconds = time.perf_counter() - evaluation_start
    values = {
        "energy_eV": energy_eV,
        "forces_eV_per_A": forces,
        "stress_GPa": stress_GPa,
    }
    evidence = {
        "generator_config": str(generator_config.config_path),
        "generator_config_sha256": _sha256(generator_config.config_path),
        "calculator_initialization_seconds": initialization_seconds,
        "evaluation_seconds": evaluation_seconds,
        "calculator": provenance.calculator.to_dict(),
        "execution_provenance": provenance.to_dict(),
    }
    atoms.calc = None
    del calculator
    _release_cuda_memory(device)
    return values, evidence


def _numerical_parity(
    reference: dict[str, Any],
    production: dict[str, Any],
    *,
    atom_count: int,
    config: PotentialPerformanceConfig,
) -> dict[str, Any]:
    energy_difference = (
        1000.0
        * abs(float(production["energy_eV"]) - float(reference["energy_eV"]))
        / atom_count
    )
    force_difference = np.asarray(
        production["forces_eV_per_A"], dtype=np.float64
    ) - np.asarray(reference["forces_eV_per_A"], dtype=np.float64)
    stress_difference = np.asarray(
        production["stress_GPa"], dtype=np.float64
    ) - np.asarray(reference["stress_GPa"], dtype=np.float64)
    metrics = {
        "energy_difference_meV_per_atom": float(energy_difference),
        "force_rmse_eV_per_A": float(
            np.sqrt(np.mean(np.square(force_difference)))
        ),
        "maximum_stress_difference_GPa": float(
            np.max(np.abs(stress_difference))
        ),
    }
    if not np.isfinite(list(metrics.values())).all():
        raise FloatingPointError(
            f"Compiled/reference numerical parity produced non-finite metrics: {metrics}."
        )
    thresholds = {
        "energy_difference_meV_per_atom": (
            config.maximum_parity_energy_difference_meV_per_atom
        ),
        "force_rmse_eV_per_A": config.maximum_parity_force_rmse_eV_per_A,
        "maximum_stress_difference_GPa": (
            config.maximum_parity_stress_difference_GPa
        ),
    }
    failures = [
        f"{metric}={metrics[metric]:.12g} exceeds {threshold:.12g}"
        for metric, threshold in thresholds.items()
        if metrics[metric] > threshold
    ]
    return {
        "passed": not failures,
        "metrics": metrics,
        "thresholds": thresholds,
        "failures": failures,
    }


def _benchmark_model(
    initial_atoms: Atoms,
    *,
    generator_config: GeneratorConfig,
    reference_values: dict[str, Any],
    reference_evidence: dict[str, Any],
    config: PotentialPerformanceConfig,
) -> dict[str, Any]:
    atoms = initial_atoms.copy()
    _synchronize_cuda(generator_config.potential.device)
    initialization_start = time.perf_counter()
    calculator, provenance = select_calculator(
        generator_config,
        calculator=None,
        injected_calculator_identity=None,
    )
    _synchronize_cuda(generator_config.potential.device)
    initialization_seconds = time.perf_counter() - initialization_start
    atoms.calc = calculator
    parity_evaluation_start = time.perf_counter()
    production_values = {
        "forces_eV_per_A": np.asarray(atoms.get_forces(), dtype=np.float64),
        "stress_GPa": (
            np.asarray(atoms.get_stress(voigt=False), dtype=np.float64) / units.GPa
        ),
        "energy_eV": float(atoms.get_potential_energy()),
    }
    _synchronize_cuda(generator_config.potential.device)
    parity_evaluation_seconds = time.perf_counter() - parity_evaluation_start
    parity = _numerical_parity(
        reference_values,
        production_values,
        atom_count=len(atoms),
        config=config,
    )
    parity["reference"] = reference_evidence
    parity["production_evaluation_seconds"] = parity_evaluation_seconds
    rng = np.random.default_rng(config.random_seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=config.temperature_K, rng=rng)
    Stationary(atoms, preserve_temperature=True)
    dynamics = IsotropicMTKNPT(
        atoms,
        timestep=config.timestep_fs * units.fs,
        temperature_K=config.temperature_K,
        pressure_au=config.pressure_GPa * units.GPa,
        tdamp=config.thermostat_time_fs * units.fs,
        pdamp=config.barostat_time_fs * units.fs,
    )
    warmup_start = time.perf_counter()
    dynamics.run(config.warmup_steps)
    _synchronize_cuda(generator_config.potential.device)
    warmup_seconds = time.perf_counter() - warmup_start
    block_seconds: list[float] = []
    for _block in range(config.measurement_blocks):
        start = time.perf_counter()
        dynamics.run(config.steps_per_block)
        _synchronize_cuda(generator_config.potential.device)
        block_seconds.append(time.perf_counter() - start)
    timing = summarize_block_timings(
        np.asarray(block_seconds), steps_per_block=config.steps_per_block
    )
    timing.update(
        {
            "warmup_steps": config.warmup_steps,
            "calculator_initialization_seconds": initialization_seconds,
            "warmup_seconds": warmup_seconds,
            "atom_count": len(atoms),
            "final_temperature_K": float(atoms.get_temperature()),
            "final_pressure_GPa": float(
                -np.trace(atoms.get_stress(voigt=False, include_ideal_gas=True))
                / 3.0
                / units.GPa
            ),
            "graph_rebuild_count": int(calculator.graph_rebuild_count),
            "graph_reuse_count": int(calculator.graph_reuse_count),
            "graph_cache_metrics": calculator.graph_cache_metrics(),
            "calculator": provenance.calculator.to_dict(),
            "execution_provenance": provenance.to_dict(),
            "numerical_parity": parity,
            "numerical_parity_passed": parity["passed"],
        }
    )
    atoms.calc = None
    del dynamics
    del calculator
    _release_cuda_memory(generator_config.potential.device)
    return timing


def run_potential_performance_benchmark(
    config: PotentialPerformanceConfig,
    *,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    if config.output_json.exists():
        raise FileExistsError(
            f"Performance output already exists: {config.output_json}. Remove it "
            "explicitly or choose a new output path."
        )
    generator_configs = [load_config(path) for path in config.model_configs]
    reference_generator_configs = [
        load_config(path) for path in config.reference_model_configs
    ]
    model_names = [item.potential.model_name for item in generator_configs]
    if len(set(model_names)) != len(model_names):
        raise ValueError(f"Performance model names must be unique, got {model_names}.")
    reference_by_name = {
        item.potential.model_name: item for item in reference_generator_configs
    }
    if len(reference_by_name) != len(reference_generator_configs) or set(
        reference_by_name
    ) != set(model_names):
        raise ValueError(
            "reference_model_configs must contain the same unique model names as "
            f"model_configs; production={model_names}, reference="
            f"{[item.potential.model_name for item in reference_generator_configs]}."
        )
    production_by_name = {
        item.potential.model_name: item for item in generator_configs
    }
    initial_homogeneous_configs = [
        load_homogeneous_crystallization_config(path)
        for path in config.initial_homogeneous_configs
    ]
    initial_by_name = {
        item.generator.potential.model_name: item
        for item in initial_homogeneous_configs
    }
    if len(initial_by_name) != len(initial_homogeneous_configs) or set(
        initial_by_name
    ) != set(model_names):
        raise ValueError(
            "initial_homogeneous_configs must contain the same unique model names as "
            f"model_configs; production={model_names}, initial="
            f"{[item.generator.potential.model_name for item in initial_homogeneous_configs]}."
        )
    sources_by_name: dict[str, tuple[Any, dict[str, Any]]] = {}
    for model_name, initial_homogeneous in initial_by_name.items():
        production_generator = production_by_name[model_name]
        if (
            initial_homogeneous.generator.config_path
            != production_generator.config_path
        ):
            raise RuntimeError(
                f"{initial_homogeneous.config_path}: source_generator_config must be the "
                f"exact production model config {production_generator.config_path}, got "
                f"{initial_homogeneous.generator.config_path}."
            )
        source = _load_source_liquid(initial_homogeneous)
        source_manifest_path = initial_homogeneous.source_dataset / "manifest.json"
        with source_manifest_path.open("r", encoding="utf-8") as handle:
            source_manifest = json.load(handle)
        if (
            not isinstance(source_manifest, dict)
            or source_manifest.get("source_kind")
            != "immutable_homogeneous_liquid_only"
            or source_manifest.get("interface_preparation_performed") is not False
        ):
            raise RuntimeError(
                f"{source_manifest_path}: performance timing requires the dedicated "
                "immutable liquid-only producer with "
                "interface_preparation_performed=false."
            )
        initial_atoms = source.atoms
        if not isinstance(initial_atoms, Atoms) or not np.all(initial_atoms.pbc):
            raise ValueError(
                f"{initial_homogeneous.config_path}: expected one fully periodic ASE "
                "Atoms source frame."
            )
        source_directory = (
            initial_homogeneous.source_dataset
            / initial_homogeneous.source_environment
        )
        source_evidence = {
            "homogeneous_config": str(initial_homogeneous.config_path),
            "homogeneous_config_sha256": _sha256(
                initial_homogeneous.config_path
            ),
            "source_generator_config": str(
                initial_homogeneous.generator.config_path
            ),
            "source_generator_config_sha256": _sha256(
                initial_homogeneous.generator.config_path
            ),
            "source_dataset": str(initial_homogeneous.source_dataset),
            "source_environment": initial_homogeneous.source_environment,
            "source_frame_step": initial_homogeneous.source_frame_step,
            "manifest_sha256": _sha256(source_manifest_path),
            "metadata_sha256": _sha256(source_directory / "metadata.json"),
            "atom_table_sha256": _sha256(source_directory / "atoms_full.npy"),
            "trajectory_sha256": _sha256(source_directory / "trajectory.npz"),
            "temperature_K": source.temperature_K,
            "pressure_GPa": source.pressure_GPa,
            "volume_A3": source.volume_A3,
            "crystalline_fraction": source.crystalline_fraction,
        }
        sources_by_name[model_name] = (source, source_evidence)
    for generator_config in generator_configs:
        initial_homogeneous = initial_by_name[
            generator_config.potential.model_name
        ]
        source, _source_evidence = sources_by_name[
            generator_config.potential.model_name
        ]
        initial_atoms = source.atoms
        reference_system = build_initial_solid(generator_config)
        expected_atom_count = len(reference_system)
        expected_atomic_number = reference_system.numbers[0]
        if len(initial_atoms) != expected_atom_count or not np.all(
            initial_atoms.numbers == expected_atomic_number
        ):
            raise RuntimeError(
                f"{initial_homogeneous.config_path}: model "
                f"{generator_config.potential.model_name!r} "
                f"requires exactly {expected_atom_count} "
                f"{generator_config.system.chemical_symbol} atoms, observed "
                f"atom_count={len(initial_atoms)} and atomic_numbers="
                f"{sorted(set(initial_atoms.numbers.tolist()))}."
            )
        dynamics = generator_config.dynamics
        protocol_values = {
            "temperature_K": (config.temperature_K, dynamics.target_temperature_K),
            "pressure_GPa": (config.pressure_GPa, dynamics.pressure_GPa),
            "timestep_fs": (config.timestep_fs, dynamics.timestep_fs),
            "thermostat_time_fs": (
                config.thermostat_time_fs,
                dynamics.thermostat_time_fs,
            ),
            "barostat_time_fs": (
                config.barostat_time_fs,
                dynamics.barostat_time_fs,
            ),
        }
        mismatches = {
            name: {"performance": observed, "generator": expected}
            for name, (observed, expected) in protocol_values.items()
            if not np.isclose(observed, expected, rtol=0.0, atol=0.0)
        }
        if mismatches:
            raise RuntimeError(
                f"Performance protocol must match the exact generator dynamics for "
                f"{generator_config.potential.model_name!r}; mismatches={mismatches}."
            )
        reference_config = reference_by_name[
            generator_config.potential.model_name
        ]
        production_potential = generator_config.potential
        reference_potential = reference_config.potential
        identity_fields = (
            "model_name",
            "family",
            "model_path",
            "sha256",
            "head",
            "source_url",
            "license_identifier",
        )
        identity_mismatches = {
            field: {
                "production": getattr(production_potential, field),
                "reference": getattr(reference_potential, field),
            }
            for field in identity_fields
            if getattr(production_potential, field)
            != getattr(reference_potential, field)
        }
        production_settings = potential_calculator_settings(production_potential)
        reference_settings = potential_calculator_settings(reference_potential)
        numerical_setting_keys = {
            "device",
            "default_dtype",
            "kernel_backend",
            "enable_cueq",
            "enable_oeq",
            "md_property_mode",
            "neighbor_skin_A",
        }
        setting_mismatches = {
            key: {
                "production": production_settings[key],
                "reference": reference_settings[key],
            }
            for key in numerical_setting_keys
            if production_settings[key] != reference_settings[key]
        }
        if identity_mismatches or setting_mismatches:
            raise RuntimeError(
                f"Production/reference configs for {production_potential.model_name!r} "
                "may differ only by compilation and padding; "
                f"identity_mismatches={identity_mismatches}, "
                f"calculator_setting_mismatches={setting_mismatches}."
            )
        if (
            reference_potential.compile_mode is not None
            or reference_potential.pad_num_atoms != 0
            or reference_potential.pad_num_edges != 0
        ):
            raise RuntimeError(
                f"Reference config {reference_config.config_path} must be uncompiled and "
                "unpadded for an independent numerical-parity baseline."
            )
        if (
            production_potential.compile_mode is None
            or production_potential.pad_num_atoms != len(initial_atoms)
            or production_potential.pad_num_edges <= 0
        ):
            raise RuntimeError(
                f"Production config {generator_config.config_path} must use compiled "
                f"execution with pad_num_atoms={len(initial_atoms)} and a positive fixed "
                "edge budget."
            )
    results: dict[str, Any] = {}
    for generator_config in generator_configs:
        model_name = generator_config.potential.model_name
        source, source_evidence = sources_by_name[model_name]
        initial_atoms = source.atoms
        reference_config = reference_by_name[model_name]
        progress(f"Evaluating uncompiled numerical reference for {model_name}")
        reference_values, reference_evidence = _evaluate_reference_state(
            initial_atoms,
            generator_config=reference_config,
        )
        progress(f"Timing production NPT path for {model_name}")
        result = _benchmark_model(
            initial_atoms,
            generator_config=generator_config,
            reference_values=reference_values,
            reference_evidence=reference_evidence,
            config=config,
        )
        result["generator_config"] = str(generator_config.config_path)
        result["generator_config_sha256"] = _sha256(
            generator_config.config_path
        )
        result["initial_source"] = source_evidence
        results[model_name] = result
    report = {
        "schema_version": 1,
        "report_type": "al_crystallization_mlip_performance",
        "benchmark_config": config.to_dict(),
        "benchmark_config_file_sha256": _sha256(config.config_path),
        "models": results,
    }
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.output_json.with_suffix(config.output_json.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(config.output_json)
    progress(f"Wrote potential performance report to {config.output_json}")
    return report
