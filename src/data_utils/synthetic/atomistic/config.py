from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class PotentialConfig:
    model_path: Path
    sha256: str
    device: str
    default_dtype: str
    enable_cueq: bool
    neighbor_skin_A: float


@dataclass(frozen=True)
class DynamicsConfig:
    pressure_GPa: float
    target_temperature_K: float
    melt_temperature_K: float
    timestep_fs: float
    thermostat_time_fs: float
    barostat_time_fs: float
    solid_equilibration_steps: int
    melt_steps: int
    quench_steps: int
    quench_stages: int
    target_equilibration_steps: int
    interface_evolution_steps: int
    sample_interval: int


@dataclass(frozen=True)
class SystemConfig:
    chemical_symbol: str
    crystal_structure: str
    initial_lattice_constant_A: float
    repetitions: tuple[int, int, int]
    liquid_slab_fraction: float
    interface_half_width_A: float


@dataclass(frozen=True)
class ValidationConfig:
    maximum_force_eV_per_A: float
    maximum_pressure_error_GPa: float
    maximum_temperature_error_K: float
    minimum_pair_distance_A: float
    reference_density_cache: Path | None
    maximum_relative_density_error: float | None
    minimum_solid_fcc_fraction: float
    maximum_liquid_crystalline_fraction: float
    minimum_interface_crystalline_fraction: float
    maximum_interface_crystalline_fraction: float


@dataclass(frozen=True)
class OutputConfig:
    root_dir: Path
    overwrite: bool
    save_extxyz: bool
    create_visualizations: bool


@dataclass(frozen=True)
class GeneratorConfig:
    dataset_name: str
    random_seeds: tuple[int, ...]
    potential: PotentialConfig
    dynamics: DynamicsConfig
    system: SystemConfig
    validation: ValidationConfig
    output: OutputConfig
    config_path: Path

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        return _serialize(result)


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    return value


def _mapping(parent: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: {key!r} must be a mapping, got {type(value).__name__}.")
    return value


def _required(mapping: dict[str, Any], key: str, context: str, path: Path) -> Any:
    if key not in mapping:
        raise KeyError(f"{path}: missing required {context}.{key}.")
    return mapping[key]


def _reject_unknown_keys(
    mapping: dict[str, Any], allowed: set[str], context: str, path: Path
) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise KeyError(
            f"{path}: unsupported keys in {context}: {unknown}. "
            "Remove obsolete/ad hoc controls instead of relying on ignored configuration."
        )


def _positive_float(value: Any, context: str, path: Path) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{path}: {context} must be > 0, got {result}.")
    return result


def _nonnegative_int(value: Any, context: str, path: Path) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{path}: {context} must be an integer, got {value!r}.")
    result = value
    if result < 0:
        raise ValueError(f"{path}: {context} must be >= 0, got {result}.")
    return result


def _positive_int(value: Any, context: str, path: Path) -> int:
    result = _nonnegative_int(value, context, path)
    if result == 0:
        raise ValueError(f"{path}: {context} must be > 0, got 0.")
    return result


def _resolve_repo_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _reject_density_controls(value: Any, *, path: Path, location: str = "root") -> None:
    forbidden = {"density_target", "rho_target", "target_density", "avg_nn_dist"}
    if isinstance(value, dict):
        for key, item in value.items():
            child_location = f"{location}.{key}"
            if str(key) in forbidden:
                raise ValueError(
                    f"{path}: {child_location} is not accepted. Number density is an NPT "
                    "simulation result, not a generator input."
                )
            _reject_density_controls(item, path=path, location=child_location)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_density_controls(item, path=path, location=f"{location}[{index}]")


def load_config(path: str | Path) -> GeneratorConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"{config_path}: root must be a mapping, got {type(raw).__name__}.")
    _reject_density_controls(raw, path=config_path)
    _reject_unknown_keys(
        raw,
        {
            "kind",
            "data_path",
            "num_points",
            "model_points",
            "radius",
            "sample_type",
            "overlap_fraction",
            "n_samples",
            "dataset_max_samples",
            "train_ratio",
            "drop_edge_samples",
            "pre_normalize",
            "normalize",
            "auto_cutoff",
            "synthetic",
            "dataset_name",
            "random_seeds",
            "potential",
            "dynamics",
            "system",
            "validation",
            "output",
        },
        "root",
        config_path,
    )

    potential_raw = _mapping(raw, "potential", config_path)
    dynamics_raw = _mapping(raw, "dynamics", config_path)
    system_raw = _mapping(raw, "system", config_path)
    validation_raw = _mapping(raw, "validation", config_path)
    output_raw = _mapping(raw, "output", config_path)
    _reject_unknown_keys(
        potential_raw,
        {
            "model_path",
            "sha256",
            "device",
            "default_dtype",
            "enable_cueq",
            "neighbor_skin_A",
        },
        "potential",
        config_path,
    )
    _reject_unknown_keys(
        dynamics_raw,
        {
            "pressure_GPa",
            "target_temperature_K",
            "melt_temperature_K",
            "timestep_fs",
            "thermostat_time_fs",
            "barostat_time_fs",
            "solid_equilibration_steps",
            "melt_steps",
            "quench_steps",
            "quench_stages",
            "target_equilibration_steps",
            "interface_evolution_steps",
            "sample_interval",
        },
        "dynamics",
        config_path,
    )
    _reject_unknown_keys(
        system_raw,
        {
            "chemical_symbol",
            "crystal_structure",
            "initial_lattice_constant_A",
            "repetitions",
            "liquid_slab_fraction",
            "interface_half_width_A",
        },
        "system",
        config_path,
    )
    _reject_unknown_keys(
        validation_raw,
        {
            "maximum_force_eV_per_A",
            "maximum_pressure_error_GPa",
            "maximum_temperature_error_K",
            "minimum_pair_distance_A",
            "reference_density_cache",
            "maximum_relative_density_error",
            "minimum_solid_fcc_fraction",
            "maximum_liquid_crystalline_fraction",
            "minimum_interface_crystalline_fraction",
            "maximum_interface_crystalline_fraction",
        },
        "validation",
        config_path,
    )
    _reject_unknown_keys(
        output_raw,
        {"root_dir", "overwrite", "save_extxyz", "create_visualizations"},
        "output",
        config_path,
    )

    model_path = _resolve_repo_path(
        _required(potential_raw, "model_path", "potential", config_path)
    )
    datasets_root = (REPOSITORY_ROOT / "datasets").resolve()
    if not model_path.is_relative_to(datasets_root):
        raise ValueError(
            f"{config_path}: potential.model_path must be repository-owned data below "
            f"{datasets_root}, got {model_path}."
        )
    repetitions_raw = _required(system_raw, "repetitions", "system", config_path)
    if not isinstance(repetitions_raw, list) or len(repetitions_raw) != 3:
        raise ValueError(
            f"{config_path}: system.repetitions must be a three-item list, got {repetitions_raw!r}."
        )
    if any(not isinstance(value, int) or isinstance(value, bool) for value in repetitions_raw):
        raise TypeError(
            f"{config_path}: system.repetitions entries must be integers, got "
            f"{repetitions_raw!r}."
        )
    repetitions = tuple(repetitions_raw)
    if min(repetitions) < 2:
        raise ValueError(
            f"{config_path}: every system.repetitions entry must be >= 2, got {repetitions}."
        )

    random_seeds_raw = _required(raw, "random_seeds", "root", config_path)
    if not isinstance(random_seeds_raw, list) or not random_seeds_raw:
        raise TypeError(
            f"{config_path}: random_seeds must be a non-empty list of integers, "
            f"got {random_seeds_raw!r}."
        )
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in random_seeds_raw):
        raise TypeError(
            f"{config_path}: every random_seeds entry must be an integer, "
            f"got {random_seeds_raw!r}."
        )
    if len(set(random_seeds_raw)) != len(random_seeds_raw):
        raise ValueError(
            f"{config_path}: random_seeds must be unique, got {random_seeds_raw!r}."
        )

    target_temperature = _positive_float(
        _required(dynamics_raw, "target_temperature_K", "dynamics", config_path),
        "dynamics.target_temperature_K",
        config_path,
    )
    melt_temperature = _positive_float(
        _required(dynamics_raw, "melt_temperature_K", "dynamics", config_path),
        "dynamics.melt_temperature_K",
        config_path,
    )
    if melt_temperature <= target_temperature:
        raise ValueError(
            f"{config_path}: dynamics.melt_temperature_K={melt_temperature} must exceed "
            f"target_temperature_K={target_temperature}."
        )

    slab_fraction = float(_required(system_raw, "liquid_slab_fraction", "system", config_path))
    if not 0.2 <= slab_fraction <= 0.8:
        raise ValueError(
            f"{config_path}: system.liquid_slab_fraction must be within [0.2, 0.8], "
            f"got {slab_fraction}."
        )

    density_cache_value = validation_raw.get("reference_density_cache")
    density_cache = (
        None
        if density_cache_value is None
        else _resolve_repo_path(density_cache_value)
    )
    if density_cache is not None and not density_cache.is_dir():
        raise FileNotFoundError(
            f"{config_path}: validation.reference_density_cache is not a directory: {density_cache}."
        )
    density_error_value = validation_raw.get("maximum_relative_density_error")
    density_error = None if density_error_value is None else float(density_error_value)
    if density_error is not None and not 0.0 < density_error < 1.0:
        raise ValueError(
            f"{config_path}: validation.maximum_relative_density_error must be in (0, 1), "
            f"got {density_error}."
        )
    minimum_solid_fcc_fraction = float(
        _required(
            validation_raw, "minimum_solid_fcc_fraction", "validation", config_path
        )
    )
    maximum_liquid_crystalline_fraction = float(
        _required(
            validation_raw,
            "maximum_liquid_crystalline_fraction",
            "validation",
            config_path,
        )
    )
    minimum_interface_crystalline_fraction = float(
        _required(
            validation_raw,
            "minimum_interface_crystalline_fraction",
            "validation",
            config_path,
        )
    )
    maximum_interface_crystalline_fraction = float(
        _required(
            validation_raw,
            "maximum_interface_crystalline_fraction",
            "validation",
            config_path,
        )
    )
    for context, value in (
        ("validation.minimum_solid_fcc_fraction", minimum_solid_fcc_fraction),
        (
            "validation.maximum_liquid_crystalline_fraction",
            maximum_liquid_crystalline_fraction,
        ),
        (
            "validation.minimum_interface_crystalline_fraction",
            minimum_interface_crystalline_fraction,
        ),
        (
            "validation.maximum_interface_crystalline_fraction",
            maximum_interface_crystalline_fraction,
        ),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{config_path}: {context} must be in [0, 1], got {value}.")
    if minimum_interface_crystalline_fraction >= maximum_interface_crystalline_fraction:
        raise ValueError(
            f"{config_path}: validation.minimum_interface_crystalline_fraction must be less "
            "than validation.maximum_interface_crystalline_fraction."
        )

    default_dtype = str(potential_raw.get("default_dtype", "float32"))
    if default_dtype not in {"float32", "float64"}:
        raise ValueError(
            f"{config_path}: potential.default_dtype must be 'float32' or 'float64', "
            f"got {default_dtype!r}."
        )
    enable_cueq = _required(potential_raw, "enable_cueq", "potential", config_path)
    if not isinstance(enable_cueq, bool):
        raise TypeError(
            f"{config_path}: potential.enable_cueq must be a boolean, "
            f"got {enable_cueq!r}."
        )
    neighbor_skin_A = _positive_float(
        _required(potential_raw, "neighbor_skin_A", "potential", config_path),
        "potential.neighbor_skin_A",
        config_path,
    )
    potential_sha256 = str(
        _required(potential_raw, "sha256", "potential", config_path)
    ).lower()
    if len(potential_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in potential_sha256
    ):
        raise ValueError(
            f"{config_path}: potential.sha256 must be 64 lowercase hexadecimal characters, "
            f"got {potential_sha256!r}."
        )

    return GeneratorConfig(
        dataset_name=str(_required(raw, "dataset_name", "root", config_path)),
        random_seeds=tuple(random_seeds_raw),
        potential=PotentialConfig(
            model_path=model_path,
            sha256=potential_sha256,
            device=str(potential_raw.get("device", "cuda")),
            default_dtype=default_dtype,
            enable_cueq=enable_cueq,
            neighbor_skin_A=neighbor_skin_A,
        ),
        dynamics=DynamicsConfig(
            pressure_GPa=float(_required(dynamics_raw, "pressure_GPa", "dynamics", config_path)),
            target_temperature_K=target_temperature,
            melt_temperature_K=melt_temperature,
            timestep_fs=_positive_float(
                _required(dynamics_raw, "timestep_fs", "dynamics", config_path),
                "dynamics.timestep_fs",
                config_path,
            ),
            thermostat_time_fs=_positive_float(
                _required(dynamics_raw, "thermostat_time_fs", "dynamics", config_path),
                "dynamics.thermostat_time_fs",
                config_path,
            ),
            barostat_time_fs=_positive_float(
                _required(dynamics_raw, "barostat_time_fs", "dynamics", config_path),
                "dynamics.barostat_time_fs",
                config_path,
            ),
            solid_equilibration_steps=_nonnegative_int(
                _required(dynamics_raw, "solid_equilibration_steps", "dynamics", config_path),
                "dynamics.solid_equilibration_steps",
                config_path,
            ),
            melt_steps=_nonnegative_int(
                _required(dynamics_raw, "melt_steps", "dynamics", config_path),
                "dynamics.melt_steps",
                config_path,
            ),
            quench_steps=_nonnegative_int(
                _required(dynamics_raw, "quench_steps", "dynamics", config_path),
                "dynamics.quench_steps",
                config_path,
            ),
            quench_stages=_positive_int(
                _required(dynamics_raw, "quench_stages", "dynamics", config_path),
                "dynamics.quench_stages",
                config_path,
            ),
            target_equilibration_steps=_nonnegative_int(
                _required(dynamics_raw, "target_equilibration_steps", "dynamics", config_path),
                "dynamics.target_equilibration_steps",
                config_path,
            ),
            interface_evolution_steps=_nonnegative_int(
                _required(
                    dynamics_raw, "interface_evolution_steps", "dynamics", config_path
                ),
                "dynamics.interface_evolution_steps",
                config_path,
            ),
            sample_interval=_positive_int(
                _required(dynamics_raw, "sample_interval", "dynamics", config_path),
                "dynamics.sample_interval",
                config_path,
            ),
        ),
        system=SystemConfig(
            chemical_symbol=str(_required(system_raw, "chemical_symbol", "system", config_path)),
            crystal_structure=str(
                _required(system_raw, "crystal_structure", "system", config_path)
            ).lower(),
            initial_lattice_constant_A=_positive_float(
                _required(system_raw, "initial_lattice_constant_A", "system", config_path),
                "system.initial_lattice_constant_A",
                config_path,
            ),
            repetitions=repetitions,
            liquid_slab_fraction=slab_fraction,
            interface_half_width_A=_positive_float(
                _required(system_raw, "interface_half_width_A", "system", config_path),
                "system.interface_half_width_A",
                config_path,
            ),
        ),
        validation=ValidationConfig(
            maximum_force_eV_per_A=_positive_float(
                _required(validation_raw, "maximum_force_eV_per_A", "validation", config_path),
                "validation.maximum_force_eV_per_A",
                config_path,
            ),
            maximum_pressure_error_GPa=_positive_float(
                _required(validation_raw, "maximum_pressure_error_GPa", "validation", config_path),
                "validation.maximum_pressure_error_GPa",
                config_path,
            ),
            maximum_temperature_error_K=_positive_float(
                _required(
                    validation_raw,
                    "maximum_temperature_error_K",
                    "validation",
                    config_path,
                ),
                "validation.maximum_temperature_error_K",
                config_path,
            ),
            minimum_pair_distance_A=_positive_float(
                _required(validation_raw, "minimum_pair_distance_A", "validation", config_path),
                "validation.minimum_pair_distance_A",
                config_path,
            ),
            reference_density_cache=density_cache,
            maximum_relative_density_error=density_error,
            minimum_solid_fcc_fraction=minimum_solid_fcc_fraction,
            maximum_liquid_crystalline_fraction=maximum_liquid_crystalline_fraction,
            minimum_interface_crystalline_fraction=minimum_interface_crystalline_fraction,
            maximum_interface_crystalline_fraction=maximum_interface_crystalline_fraction,
        ),
        output=OutputConfig(
            root_dir=_resolve_repo_path(
                _required(output_raw, "root_dir", "output", config_path)
            ),
            overwrite=bool(output_raw.get("overwrite", False)),
            save_extxyz=bool(output_raw.get("save_extxyz", True)),
            create_visualizations=bool(output_raw.get("create_visualizations", True)),
        ),
        config_path=config_path,
    )
