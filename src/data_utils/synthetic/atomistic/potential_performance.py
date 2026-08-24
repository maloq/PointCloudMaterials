from __future__ import annotations

import gc
import hashlib
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import yaml
from ase import Atoms, units
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from .config import (
    REPOSITORY_ROOT,
    SUPPORTED_COMPILE_MODES,
    load_config,
    potential_calculator_settings,
)
from .generator import select_calculator
from .homogeneous_config import load_homogeneous_crystallization_config
from .homogeneous_generator import _load_source_liquid
from .simulation import build_initial_solid

if TYPE_CHECKING:
    from .config import GeneratorConfig
    from .homogeneous_config import HomogeneousCrystallizationConfig


RUNTIME_KERNEL_BACKENDS = {
    "cueq": (True, False),
    "oeq": (False, True),
    "hybrid_cueq_oeq": (True, True),
}
REFERENCE_KERNEL_BACKENDS = {
    "e3nn": (False, False),
    **RUNTIME_KERNEL_BACKENDS,
}
_GRAPH_COUNTER_KEYS = (
    "requests",
    "rebuilds",
    "compiled_buffer_refills",
    "reuses",
    "model_evaluations",
    "force_evaluations",
    "stress_evaluations",
)
_GRAPH_STATE_KEYS = (
    "real_edge_count",
    "maximum_real_edge_count",
    "maximum_edge_budget_fraction",
    "pad_num_edges",
    "neighbor_skin_A",
)


@dataclass(frozen=True)
class PotentialRuntimeVariant:
    """One explicit runtime choice for an identical MD workload."""

    name: str
    kernel_backend: str
    compile_mode: str | None
    pad_num_atoms: int
    pad_num_edges: int
    neighbor_skin_A: float

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))

    @property
    def canonical_sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class PotentialRuntimeSweep:
    model_config: Path
    initial_homogeneous_config: Path
    reference_kernel_backend: str
    baseline_variant: str
    variants: tuple[PotentialRuntimeVariant, ...]


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
    runtime_sweep: PotentialRuntimeSweep | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.runtime_sweep is None:
            result.pop("runtime_sweep")
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


def _nonnegative_int(value: object, *, name: str, path: Path) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TypeError(f"{path}: {name} must be a nonnegative integer, got {value!r}.")
    return value


def _parse_runtime_sweep(
    value: object, *, config_path: Path
) -> PotentialRuntimeSweep | None:
    if value is None:
        return None
    sweep_keys = {
        "model_config",
        "initial_homogeneous_config",
        "reference_kernel_backend",
        "baseline_variant",
        "variants",
    }
    if not isinstance(value, dict) or set(value) != sweep_keys:
        raise KeyError(
            f"{config_path}: runtime_sweep keys must be exactly {sorted(sweep_keys)}."
        )
    raw_variants = value["variants"]
    if not isinstance(raw_variants, list) or not raw_variants:
        raise TypeError(
            f"{config_path}: runtime_sweep.variants must be a non-empty list."
        )
    variant_keys = {
        "name",
        "kernel_backend",
        "compile_mode",
        "pad_num_atoms",
        "pad_num_edges",
        "neighbor_skin_A",
    }
    variants: list[PotentialRuntimeVariant] = []
    for index, raw_variant in enumerate(raw_variants):
        context = f"runtime_sweep.variants[{index}]"
        if not isinstance(raw_variant, dict) or set(raw_variant) != variant_keys:
            raise KeyError(
                f"{config_path}: {context} keys must be exactly "
                f"{sorted(variant_keys)}."
            )
        name = raw_variant["name"]
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"{config_path}: {context}.name must be a non-empty string."
            )
        kernel_backend = raw_variant["kernel_backend"]
        if not isinstance(kernel_backend, str) or (
            kernel_backend not in RUNTIME_KERNEL_BACKENDS
        ):
            raise ValueError(
                f"{config_path}: {context}.kernel_backend must be one of "
                f"{sorted(RUNTIME_KERNEL_BACKENDS)}, got {kernel_backend!r}."
            )
        compile_mode = raw_variant["compile_mode"]
        if compile_mode is not None and (
            not isinstance(compile_mode, str)
            or compile_mode not in SUPPORTED_COMPILE_MODES
        ):
            raise ValueError(
                f"{config_path}: {context}.compile_mode must be null or one of "
                f"{sorted(SUPPORTED_COMPILE_MODES)}, got {compile_mode!r}."
            )
        pad_num_atoms = _nonnegative_int(
            raw_variant["pad_num_atoms"],
            name=f"{context}.pad_num_atoms",
            path=config_path,
        )
        pad_num_edges = _nonnegative_int(
            raw_variant["pad_num_edges"],
            name=f"{context}.pad_num_edges",
            path=config_path,
        )
        if compile_mode is None and (pad_num_atoms != 0 or pad_num_edges != 0):
            raise ValueError(
                f"{config_path}: uncompiled {context} requires zero atom/edge padding, "
                f"got pad_num_atoms={pad_num_atoms}, pad_num_edges={pad_num_edges}."
            )
        if compile_mode is not None and (
            pad_num_atoms == 0 or pad_num_edges == 0
        ):
            raise ValueError(
                f"{config_path}: compiled {context} requires positive atom/edge "
                f"padding, got pad_num_atoms={pad_num_atoms}, "
                f"pad_num_edges={pad_num_edges}."
            )
        variants.append(
            PotentialRuntimeVariant(
                name=name,
                kernel_backend=kernel_backend,
                compile_mode=compile_mode,
                pad_num_atoms=pad_num_atoms,
                pad_num_edges=pad_num_edges,
                neighbor_skin_A=_positive_float(
                    raw_variant["neighbor_skin_A"],
                    name=f"{context}.neighbor_skin_A",
                    path=config_path,
                ),
            )
        )
    names = [variant.name for variant in variants]
    if len(set(names)) != len(names):
        raise ValueError(
            f"{config_path}: runtime variant names must be unique, got {names}."
        )
    baseline_variant = value["baseline_variant"]
    if baseline_variant not in names:
        raise ValueError(
            f"{config_path}: runtime_sweep.baseline_variant={baseline_variant!r} "
            f"is not one of {names}."
        )
    settings = [
        (
            variant.kernel_backend,
            variant.compile_mode,
            variant.pad_num_atoms,
            variant.pad_num_edges,
            variant.neighbor_skin_A,
        )
        for variant in variants
    ]
    if len(set(settings)) != len(settings):
        raise ValueError(
            f"{config_path}: runtime variants contain duplicate effective controls: "
            f"{settings}."
        )
    model_config = _repo_path(value["model_config"])
    homogeneous_config = _repo_path(value["initial_homogeneous_config"])
    reference_kernel_backend = value["reference_kernel_backend"]
    if (
        not isinstance(reference_kernel_backend, str)
        or reference_kernel_backend not in REFERENCE_KERNEL_BACKENDS
    ):
        raise ValueError(
            f"{config_path}: runtime_sweep.reference_kernel_backend must be one of "
            f"{sorted(REFERENCE_KERNEL_BACKENDS)}, got "
            f"{reference_kernel_backend!r}."
        )
    for path in (model_config, homogeneous_config):
        if not path.is_file():
            raise FileNotFoundError(
                f"{config_path}: runtime sweep file is missing: {path}."
            )
    return PotentialRuntimeSweep(
        model_config=model_config,
        initial_homogeneous_config=homogeneous_config,
        reference_kernel_backend=reference_kernel_backend,
        baseline_variant=baseline_variant,
        variants=tuple(variants),
    )


def load_potential_performance_config(
    path: str | Path,
) -> PotentialPerformanceConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"{config_path}: root must be a mapping.")
    required_keys = {
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
    allowed_keys = required_keys | {"runtime_sweep"}
    if not required_keys.issubset(raw) or set(raw) - allowed_keys:
        raise KeyError(
            f"{config_path}: performance config requires keys={sorted(required_keys)} "
            f"and optionally runtime_sweep; observed={sorted(raw)}."
        )
    runtime_sweep = _parse_runtime_sweep(
        raw.get("runtime_sweep"), config_path=config_path
    )
    model_configs_raw = raw["model_configs"]
    if not isinstance(model_configs_raw, list) or (
        runtime_sweep is None and len(model_configs_raw) < 2
    ):
        raise TypeError(
            f"{config_path}: model_configs must be a list containing at least two "
            "config paths for the model-selection benchmark, or it may be empty when "
            "runtime_sweep is configured."
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
        runtime_sweep=runtime_sweep,
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


def _disable_tf32() -> dict[str, Any]:
    """Use the repository's required IEEE FP32 matmul path for every variant."""

    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "tf32_enabled": False,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def _reset_cuda_peak_memory(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    _synchronize_cuda(device)
    torch.cuda.reset_peak_memory_stats(torch.device(device))


def _cuda_memory_peaks(device: str) -> dict[str, int] | None:
    if not device.startswith("cuda"):
        return None
    import torch

    _synchronize_cuda(device)
    cuda_device = torch.device(device)
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(cuda_device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(cuda_device)),
    }


def _combined_cuda_peaks(
    before_measurement: dict[str, int] | None,
    measurement: dict[str, int] | None,
) -> dict[str, int] | None:
    if before_measurement is None:
        return None
    if measurement is None:
        raise RuntimeError("CUDA measurement peak memory is missing.")
    return {
        key: max(before_measurement[key], measurement[key])
        for key in ("peak_allocated_bytes", "peak_reserved_bytes")
    }


def graph_cache_metrics_delta(
    before: dict[str, float | int], after: dict[str, float | int]
) -> dict[str, float | int]:
    """Return counters attributable only to one warmup or measurement interval."""

    deltas = {
        key: int(after[key]) - int(before[key]) for key in _GRAPH_COUNTER_KEYS
    }
    if any(value < 0 for value in deltas.values()):
        raise RuntimeError(f"Graph cache counters decreased: deltas={deltas}.")
    requests = deltas["requests"]
    result: dict[str, float | int] = {
        **deltas,
        "reuse_fraction": (
            float(deltas["reuses"] / requests) if requests else 0.0
        ),
    }
    result.update({key: after[key] for key in _GRAPH_STATE_KEYS})
    result["final_real_edge_count"] = result.pop("real_edge_count")
    return result


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

    if not device.startswith("cuda"):
        gc.collect()
        return
    import torch

    _synchronize_cuda(device)
    torch.compiler.reset()
    gc.collect()
    torch.cuda.empty_cache()
    _synchronize_cuda(device)


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
    reference_energy = float(reference["energy_eV"])
    production_energy = float(production["energy_eV"])
    reference_forces = np.asarray(reference["forces_eV_per_A"], dtype=np.float64)
    production_forces = np.asarray(production["forces_eV_per_A"], dtype=np.float64)
    reference_stress = np.asarray(reference["stress_GPa"], dtype=np.float64)
    production_stress = np.asarray(production["stress_GPa"], dtype=np.float64)
    expected_force_shape = (atom_count, 3)
    if (
        reference_forces.shape != expected_force_shape
        or production_forces.shape != expected_force_shape
    ):
        raise ValueError(
            "Numerical parity requires exact force arrays with shape="
            f"{expected_force_shape}; reference={reference_forces.shape}, "
            f"production={production_forces.shape}."
        )
    if reference_stress.shape != (3, 3) or production_stress.shape != (3, 3):
        raise ValueError(
            "Numerical parity requires exact full stress tensors with shape=(3, 3); "
            f"reference={reference_stress.shape}, production={production_stress.shape}."
        )
    if not (
        np.isfinite(reference_energy)
        and np.isfinite(production_energy)
        and np.isfinite(reference_forces).all()
        and np.isfinite(production_forces).all()
        and np.isfinite(reference_stress).all()
        and np.isfinite(production_stress).all()
    ):
        raise FloatingPointError(
            "Reference/production energy, force, and stress values must all be finite "
            "before numerical parity is evaluated."
        )
    energy_difference = (
        1000.0
        * abs(production_energy - reference_energy)
        / atom_count
    )
    force_difference = production_forces - reference_forces
    stress_difference = production_stress - reference_stress
    metrics = {
        "energy_difference_meV_per_atom": float(energy_difference),
        "force_rmse_eV_per_A": float(
            np.sqrt(np.mean(np.square(force_difference)))
        ),
        "maximum_force_difference_eV_per_A": float(
            np.max(np.abs(force_difference))
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


def require_numerical_parity(
    parity: dict[str, Any], *, runtime_name: str
) -> None:
    if parity["passed"]:
        return
    raise RuntimeError(
        f"Numerical parity failed for {runtime_name!r}; "
        + "; ".join(parity["failures"])
    )


def _benchmark_model(
    initial_atoms: Atoms,
    *,
    generator_config: GeneratorConfig,
    reference_values: dict[str, Any],
    reference_evidence: dict[str, Any],
    config: PotentialPerformanceConfig,
    runtime_name: str,
) -> dict[str, Any]:
    atoms = initial_atoms.copy()
    device = generator_config.potential.device
    _reset_cuda_peak_memory(device)
    initialization_start = time.perf_counter()
    calculator, provenance = select_calculator(
        generator_config,
        calculator=None,
        injected_calculator_identity=None,
    )
    _synchronize_cuda(device)
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
    _synchronize_cuda(device)
    parity_evaluation_seconds = time.perf_counter() - parity_evaluation_start
    parity = _numerical_parity(
        reference_values,
        production_values,
        atom_count=len(atoms),
        config=config,
    )
    parity["reference"] = reference_evidence
    parity["production_evaluation_seconds"] = parity_evaluation_seconds
    require_numerical_parity(parity, runtime_name=runtime_name)
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
    graph_before_warmup = calculator.graph_cache_metrics()
    warmup_start = time.perf_counter()
    dynamics.run(config.warmup_steps)
    _synchronize_cuda(device)
    warmup_seconds = time.perf_counter() - warmup_start
    graph_after_warmup = calculator.graph_cache_metrics()
    memory_before_measurement = _cuda_memory_peaks(device)
    _reset_cuda_peak_memory(device)
    graph_before_measurement = calculator.graph_cache_metrics()
    block_seconds: list[float] = []
    for _block in range(config.measurement_blocks):
        start = time.perf_counter()
        dynamics.run(config.steps_per_block)
        _synchronize_cuda(device)
        block_seconds.append(time.perf_counter() - start)
    graph_after_measurement = calculator.graph_cache_metrics()
    measurement_memory = _cuda_memory_peaks(device)
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
            "graph_cache_metrics": graph_after_measurement,
            "warmup_graph_cache_metrics": graph_cache_metrics_delta(
                graph_before_warmup, graph_after_warmup
            ),
            "measurement_graph_cache_metrics": graph_cache_metrics_delta(
                graph_before_measurement, graph_after_measurement
            ),
            "cuda_memory": {
                "before_measurement_peaks": memory_before_measurement,
                "measurement_peaks": measurement_memory,
                "overall_peaks": _combined_cuda_peaks(
                    memory_before_measurement, measurement_memory
                ),
            },
            "calculator": provenance.calculator.to_dict(),
            "execution_provenance": provenance.to_dict(),
            "numerical_parity": parity,
            "numerical_parity_passed": parity["passed"],
        }
    )
    atoms.calc = None
    del dynamics
    del calculator
    _release_cuda_memory(device)
    return timing


def _runtime_generator_config(
    base_config: GeneratorConfig,
    variant: PotentialRuntimeVariant,
    *,
    numerical_reference: bool,
    reference_kernel_backend: str = "e3nn",
) -> GeneratorConfig:
    if numerical_reference:
        try:
            enable_cueq, enable_oeq = REFERENCE_KERNEL_BACKENDS[
                reference_kernel_backend
            ]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported runtime reference kernel backend="
                f"{reference_kernel_backend!r}; expected one of "
                f"{sorted(REFERENCE_KERNEL_BACKENDS)}."
            ) from exc
        compile_mode = None
        pad_num_atoms = 0
        pad_num_edges = 0
    else:
        enable_cueq, enable_oeq = RUNTIME_KERNEL_BACKENDS[
            variant.kernel_backend
        ]
        compile_mode = variant.compile_mode
        pad_num_atoms = variant.pad_num_atoms
        pad_num_edges = variant.pad_num_edges
    potential = replace(
        base_config.potential,
        usage_mode="exploratory",
        validation_report=None,
        validation_report_sha256=None,
        scientifically_qualified=False,
        qualification=None,
        enable_cueq=enable_cueq,
        enable_oeq=enable_oeq,
        compile_mode=compile_mode,
        compile_fullgraph=False,
        pad_num_atoms=pad_num_atoms,
        pad_num_edges=pad_num_edges,
        neighbor_skin_A=variant.neighbor_skin_A,
    )
    return replace(base_config, potential=potential)


def _load_performance_source(
    homogeneous: HomogeneousCrystallizationConfig,
    expected_generator: GeneratorConfig,
) -> tuple[Any, dict[str, Any]]:
    if homogeneous.generator.config_path != expected_generator.config_path:
        raise RuntimeError(
            f"{homogeneous.config_path}: source_generator_config must be the exact "
            f"performance model config {expected_generator.config_path}, got "
            f"{homogeneous.generator.config_path}."
        )
    source = _load_source_liquid(homogeneous)
    source_manifest_path = homogeneous.source_dataset / "manifest.json"
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
    if not isinstance(source.atoms, Atoms) or not np.all(source.atoms.pbc):
        raise ValueError(
            f"{homogeneous.config_path}: expected one fully periodic ASE Atoms source "
            "frame."
        )
    source_directory = homogeneous.source_dataset / homogeneous.source_environment
    evidence = {
        "homogeneous_config": str(homogeneous.config_path),
        "homogeneous_config_sha256": _sha256(homogeneous.config_path),
        "source_generator_config": str(homogeneous.generator.config_path),
        "source_generator_config_sha256": _sha256(
            homogeneous.generator.config_path
        ),
        "source_dataset": str(homogeneous.source_dataset),
        "source_environment": homogeneous.source_environment,
        "source_frame_step": homogeneous.source_frame_step,
        "manifest_sha256": _sha256(source_manifest_path),
        "metadata_sha256": _sha256(source_directory / "metadata.json"),
        "atom_table_sha256": _sha256(source_directory / "atoms_full.npy"),
        "trajectory_sha256": _sha256(source_directory / "trajectory.npz"),
        "temperature_K": source.temperature_K,
        "pressure_GPa": source.pressure_GPa,
        "volume_A3": source.volume_A3,
        "crystalline_fraction": source.crystalline_fraction,
    }
    return source, evidence


def _validate_performance_workload(
    generator_config: GeneratorConfig,
    initial_atoms: Atoms,
    config: PotentialPerformanceConfig,
) -> None:
    expected_atoms = build_initial_solid(generator_config)
    if len(initial_atoms) != len(expected_atoms) or not np.array_equal(
        initial_atoms.numbers, expected_atoms.numbers
    ):
        raise RuntimeError(
            f"{generator_config.config_path}: expected the exact repository-produced "
            f"system with {len(expected_atoms)} atoms, observed {len(initial_atoms)}."
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
            "Performance protocol must match the exact generator dynamics; "
            f"model={generator_config.potential.model_name!r}, mismatches={mismatches}."
        )


def _runtime_variant_comparison(
    variants: tuple[PotentialRuntimeVariant, ...],
    baseline_name: str,
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    baseline_speed = float(results[baseline_name]["steps_per_second"])
    ranking = [
        {
            "name": variant.name,
            "steps_per_second": float(results[variant.name]["steps_per_second"]),
            "speedup_vs_baseline": float(
                results[variant.name]["steps_per_second"] / baseline_speed
            ),
            **results[variant.name]["cuda_memory"]["overall_peaks"],
            "measurement_graph_reuse_fraction": float(
                results[variant.name]["measurement_graph_cache_metrics"][
                    "reuse_fraction"
                ]
            ),
            "maximum_real_edge_count": int(
                results[variant.name]["graph_cache_metrics"][
                    "maximum_real_edge_count"
                ]
            ),
        }
        for variant in variants
    ]
    ranking.sort(key=lambda row: (-float(row["steps_per_second"]), str(row["name"])))
    return {
        "baseline_variant": baseline_name,
        "fastest_parity_passing_variant": ranking[0]["name"],
        "ranking": ranking,
    }


def _run_runtime_variants(
    config: PotentialPerformanceConfig,
    *,
    progress: Callable[[str], None],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    sweep = config.runtime_sweep
    if sweep is None:
        return {}, {}
    variants = sweep.variants
    base_config = load_config(sweep.model_config)
    homogeneous = load_homogeneous_crystallization_config(
        sweep.initial_homogeneous_config
    )
    source, source_evidence = _load_performance_source(homogeneous, base_config)
    initial_atoms = source.atoms
    _validate_performance_workload(base_config, initial_atoms, config)
    bad_atom_budgets = {
        variant.name: variant.pad_num_atoms
        for variant in variants
        if variant.compile_mode is not None
        and variant.pad_num_atoms != len(initial_atoms)
    }
    if bad_atom_budgets:
        raise RuntimeError(
            "Compiled runtime variants require the exact repository-produced "
            f"pad_num_atoms={len(initial_atoms)}, got {bad_atom_budgets}."
        )
    reference_cache: dict[float, tuple[dict[str, Any], dict[str, Any]]] = {}
    results: dict[str, dict[str, Any]] = {}
    for variant in variants:
        if variant.neighbor_skin_A not in reference_cache:
            reference_config = _runtime_generator_config(
                base_config,
                variant,
                numerical_reference=True,
                reference_kernel_backend=sweep.reference_kernel_backend,
            )
            progress(
                "Evaluating common uncompiled "
                f"{sweep.reference_kernel_backend} numerical reference for "
                f"neighbor_skin_A={variant.neighbor_skin_A:g}"
            )
            reference_cache[variant.neighbor_skin_A] = _evaluate_reference_state(
                initial_atoms,
                generator_config=reference_config,
            )
        reference_values, reference_evidence = reference_cache[
            variant.neighbor_skin_A
        ]
        variant_config = _runtime_generator_config(
            base_config, variant, numerical_reference=False
        )
        progress(
            f"Timing runtime variant {variant.name}: backend="
            f"{variant.kernel_backend}, compile_mode={variant.compile_mode}, "
            f"neighbor_skin_A={variant.neighbor_skin_A:g}, "
            f"pad_num_edges={variant.pad_num_edges}"
        )
        result = _benchmark_model(
            initial_atoms,
            generator_config=variant_config,
            reference_values=reference_values,
            reference_evidence=reference_evidence,
            config=config,
            runtime_name=variant.name,
        )
        result.update(
            {
                "variant": variant.to_dict(),
                "variant_canonical_sha256": variant.canonical_sha256,
                "base_generator_config": str(base_config.config_path),
                "base_generator_config_sha256": _sha256(base_config.config_path),
                "effective_calculator_settings": potential_calculator_settings(
                    variant_config.potential
                ),
                "initial_source": source_evidence,
            }
        )
        results[variant.name] = result
    return results, _runtime_variant_comparison(
        variants, sweep.baseline_variant, results
    )


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
    cuda_math = _disable_tf32()
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
        sources_by_name[model_name] = _load_performance_source(
            initial_homogeneous, production_generator
        )
    for generator_config in generator_configs:
        source, _source_evidence = sources_by_name[
            generator_config.potential.model_name
        ]
        initial_atoms = source.atoms
        _validate_performance_workload(generator_config, initial_atoms, config)
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
            runtime_name=model_name,
        )
        result["generator_config"] = str(generator_config.config_path)
        result["generator_config_sha256"] = _sha256(
            generator_config.config_path
        )
        result["initial_source"] = source_evidence
        results[model_name] = result
    runtime_variants, runtime_comparison = _run_runtime_variants(
        config, progress=progress
    )
    report = {
        "schema_version": 1,
        "report_type": "al_crystallization_mlip_performance",
        "benchmark_config": config.to_dict(),
        "benchmark_config_file_sha256": _sha256(config.config_path),
        "runtime_controls": {"cuda_math": cuda_math},
        "models": results,
    }
    if runtime_variants:
        report["runtime_variants"] = runtime_variants
        report["runtime_variant_comparison"] = runtime_comparison
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.output_json.with_suffix(config.output_json.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(config.output_json)
    progress(f"Wrote potential performance report to {config.output_json}")
    return report
