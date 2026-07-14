from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import GeneratorConfig, REPOSITORY_ROOT, load_config


@dataclass(frozen=True)
class TransitionBranchConfig:
    name: str
    expected_direction: str
    temperature_K: float
    steps: int
    minimum_crystalline_fraction_change: float


@dataclass(frozen=True)
class TransitionOutputConfig:
    root_dir: Path
    overwrite: bool
    save_extxyz: bool
    create_visualizations: bool


@dataclass(frozen=True)
class TransitionAnalysisConfig:
    profile_bins: int
    rdf_cutoff_A: float
    rdf_bins: int


@dataclass(frozen=True)
class TransitionConfig:
    dataset_name: str
    source_dataset: Path
    source_interface_environment: str
    source_solid_environment: str
    source_frame_step: int
    random_seed: int
    sample_interval: int
    analysis: TransitionAnalysisConfig
    crystallization: TransitionBranchConfig
    melting: TransitionBranchConfig
    output: TransitionOutputConfig
    generator: GeneratorConfig
    config_path: Path

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["generator"] = self.generator.to_dict()
        return _serialize(value)


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def _repo_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _mapping(raw: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: {key} must be a mapping, got {type(value).__name__}.")
    return value


def _reject_unknown(
    raw: dict[str, Any], allowed: set[str], context: str, path: Path
) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise KeyError(f"{path}: unsupported keys in {context}: {unknown}.")


def _positive_float(value: Any, context: str, path: Path) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{path}: {context} must be > 0, got {result}.")
    return result


def _nonnegative_float(value: Any, context: str, path: Path) -> float:
    result = float(value)
    if result < 0.0:
        raise ValueError(f"{path}: {context} must be >= 0, got {result}.")
    return result


def _positive_int(value: Any, context: str, path: Path) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{path}: {context} must be a positive integer, got {value!r}.")
    return value


def _branch(
    raw: dict[str, Any],
    *,
    name: str,
    expected_direction: str,
    path: Path,
) -> TransitionBranchConfig:
    _reject_unknown(
        raw,
        {"temperature_K", "steps", "minimum_crystalline_fraction_change"},
        name,
        path,
    )
    return TransitionBranchConfig(
        name=name,
        expected_direction=expected_direction,
        temperature_K=_positive_float(raw["temperature_K"], f"{name}.temperature_K", path),
        steps=_positive_int(raw["steps"], f"{name}.steps", path),
        minimum_crystalline_fraction_change=_nonnegative_float(
            raw["minimum_crystalline_fraction_change"],
            f"{name}.minimum_crystalline_fraction_change",
            path,
        ),
    )


def load_transition_config(path: str | Path) -> TransitionConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"{config_path}: root must be a mapping.")
    _reject_unknown(
        raw,
        {
            "dataset_name",
            "source_generator_config",
            "source_dataset",
            "source_interface_environment",
            "source_solid_environment",
            "source_frame_step",
            "random_seed",
            "sample_interval",
            "analysis",
            "crystallization",
            "melting",
            "output",
        },
        "root",
        config_path,
    )
    generator = load_config(_repo_path(raw["source_generator_config"]))
    crystallization = _branch(
        _mapping(raw, "crystallization", config_path),
        name="crystallization",
        expected_direction="growth",
        path=config_path,
    )
    melting = _branch(
        _mapping(raw, "melting", config_path),
        name="melting",
        expected_direction="melting",
        path=config_path,
    )
    if crystallization.temperature_K >= melting.temperature_K:
        raise ValueError(
            f"{config_path}: crystallization.temperature_K must be below "
            f"melting.temperature_K, got {crystallization.temperature_K} and "
            f"{melting.temperature_K}."
        )
    source_frame_step = raw["source_frame_step"]
    if not isinstance(source_frame_step, int) or isinstance(source_frame_step, bool):
        raise TypeError(
            f"{config_path}: source_frame_step must be an integer, got "
            f"{source_frame_step!r}."
        )
    random_seed = raw["random_seed"]
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise TypeError(f"{config_path}: random_seed must be an integer, got {random_seed!r}.")
    output_raw = _mapping(raw, "output", config_path)
    _reject_unknown(
        output_raw,
        {"root_dir", "overwrite", "save_extxyz", "create_visualizations"},
        "output",
        config_path,
    )
    analysis_raw = _mapping(raw, "analysis", config_path)
    _reject_unknown(
        analysis_raw,
        {"profile_bins", "rdf_cutoff_A", "rdf_bins"},
        "analysis",
        config_path,
    )
    return TransitionConfig(
        dataset_name=str(raw["dataset_name"]),
        source_dataset=_repo_path(raw["source_dataset"]),
        source_interface_environment=str(raw["source_interface_environment"]),
        source_solid_environment=str(raw["source_solid_environment"]),
        source_frame_step=source_frame_step,
        random_seed=random_seed,
        sample_interval=_positive_int(
            raw["sample_interval"], "sample_interval", config_path
        ),
        analysis=TransitionAnalysisConfig(
            profile_bins=_positive_int(
                analysis_raw["profile_bins"], "analysis.profile_bins", config_path
            ),
            rdf_cutoff_A=_positive_float(
                analysis_raw["rdf_cutoff_A"], "analysis.rdf_cutoff_A", config_path
            ),
            rdf_bins=_positive_int(
                analysis_raw["rdf_bins"], "analysis.rdf_bins", config_path
            ),
        ),
        crystallization=crystallization,
        melting=melting,
        output=TransitionOutputConfig(
            root_dir=_repo_path(output_raw["root_dir"]),
            overwrite=bool(output_raw["overwrite"]),
            save_extxyz=bool(output_raw["save_extxyz"]),
            create_visualizations=bool(output_raw["create_visualizations"]),
        ),
        generator=generator,
        config_path=config_path,
    )
