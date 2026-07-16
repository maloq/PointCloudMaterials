from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import (
    GeneratorConfig,
    REPOSITORY_ROOT,
    load_config,
    validate_potential_qualification,
)


@dataclass(frozen=True)
class HomogeneousAnalysisConfig:
    ptm_rmsd_cutoff: float
    crystalline_cluster_cutoff_A: float
    nucleus_size_threshold_atoms: int
    threshold_persistence_frames: int
    rdf_cutoff_A: float
    rdf_bins: int


@dataclass(frozen=True)
class HomogeneousOutputConfig:
    root_dir: Path
    overwrite: bool
    save_extxyz: bool
    create_visualizations: bool


@dataclass(frozen=True)
class HomogeneousCrystallizationConfig:
    dataset_name: str
    source_dataset: Path
    source_environment: str
    source_frame_step: int
    random_seeds: tuple[int, ...]
    temperature_K: float
    equilibration_steps: int
    steps: int
    sample_interval: int
    analysis: HomogeneousAnalysisConfig
    output: HomogeneousOutputConfig
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


def _positive_int(value: Any, context: str, path: Path) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{path}: {context} must be a positive integer, got {value!r}.")
    return value


def _positive_float(value: Any, context: str, path: Path) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{path}: {context} must be > 0, got {result}.")
    return result


def load_homogeneous_crystallization_config(
    path: str | Path,
) -> HomogeneousCrystallizationConfig:
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
            "source_environment",
            "source_frame_step",
            "random_seeds",
            "temperature_K",
            "equilibration_steps",
            "steps",
            "sample_interval",
            "analysis",
            "output",
        },
        "root",
        config_path,
    )
    analysis_raw = _mapping(raw, "analysis", config_path)
    _reject_unknown(
        analysis_raw,
        {
            "ptm_rmsd_cutoff",
            "crystalline_cluster_cutoff_A",
            "nucleus_size_threshold_atoms",
            "threshold_persistence_frames",
            "rdf_cutoff_A",
            "rdf_bins",
        },
        "analysis",
        config_path,
    )
    output_raw = _mapping(raw, "output", config_path)
    _reject_unknown(
        output_raw,
        {"root_dir", "overwrite", "save_extxyz", "create_visualizations"},
        "output",
        config_path,
    )
    source_frame_step = raw["source_frame_step"]
    if not isinstance(source_frame_step, int) or isinstance(source_frame_step, bool):
        raise TypeError(
            f"{config_path}: source_frame_step must be an integer, got "
            f"{source_frame_step!r}."
        )
    random_seeds = raw["random_seeds"]
    if not isinstance(random_seeds, list) or not random_seeds:
        raise TypeError(
            f"{config_path}: random_seeds must be a non-empty list of unique integers, "
            f"got {random_seeds!r}."
        )
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in random_seeds):
        raise TypeError(
            f"{config_path}: every random_seeds entry must be an integer, got "
            f"{random_seeds!r}."
        )
    if len(set(random_seeds)) != len(random_seeds):
        raise ValueError(
            f"{config_path}: random_seeds must be unique so replicas are independent, "
            f"got {random_seeds!r}."
        )
    temperature_K = _positive_float(raw["temperature_K"], "temperature_K", config_path)
    equilibration_steps = _positive_int(
        raw["equilibration_steps"], "equilibration_steps", config_path
    )
    steps = _positive_int(raw["steps"], "steps", config_path)
    sample_interval = _positive_int(
        raw["sample_interval"], "sample_interval", config_path
    )
    if steps % sample_interval != 0:
        raise ValueError(
            f"{config_path}: steps={steps} must be divisible by sample_interval="
            f"{sample_interval} so the measured trace includes its endpoint."
        )
    if equilibration_steps % sample_interval != 0:
        raise ValueError(
            f"{config_path}: equilibration_steps={equilibration_steps} must be divisible "
            f"by sample_interval={sample_interval} so one continuous MTK-NPT trace contains "
            "the exact equilibration/measurement boundary."
        )
    ptm_rmsd_cutoff = _positive_float(
        analysis_raw["ptm_rmsd_cutoff"],
        "analysis.ptm_rmsd_cutoff",
        config_path,
    )
    if ptm_rmsd_cutoff > 1.0:
        raise ValueError(
            f"{config_path}: analysis.ptm_rmsd_cutoff is a dimensionless normalized "
            f"RMSD and must be <= 1, got {ptm_rmsd_cutoff}."
        )
    threshold_persistence_frames = _positive_int(
        analysis_raw["threshold_persistence_frames"],
        "analysis.threshold_persistence_frames",
        config_path,
    )
    measured_frame_count = steps // sample_interval + 1
    if threshold_persistence_frames > measured_frame_count:
        raise ValueError(
            f"{config_path}: analysis.threshold_persistence_frames="
            f"{threshold_persistence_frames} exceeds the {measured_frame_count} saved "
            "measurement frames."
        )

    generator = load_config(_repo_path(raw["source_generator_config"]))
    config = HomogeneousCrystallizationConfig(
        dataset_name=str(raw["dataset_name"]),
        source_dataset=_repo_path(raw["source_dataset"]),
        source_environment=str(raw["source_environment"]),
        source_frame_step=source_frame_step,
        random_seeds=tuple(random_seeds),
        temperature_K=temperature_K,
        equilibration_steps=equilibration_steps,
        steps=steps,
        sample_interval=sample_interval,
        analysis=HomogeneousAnalysisConfig(
            ptm_rmsd_cutoff=ptm_rmsd_cutoff,
            crystalline_cluster_cutoff_A=_positive_float(
                analysis_raw["crystalline_cluster_cutoff_A"],
                "analysis.crystalline_cluster_cutoff_A",
                config_path,
            ),
            nucleus_size_threshold_atoms=_positive_int(
                analysis_raw["nucleus_size_threshold_atoms"],
                "analysis.nucleus_size_threshold_atoms",
                config_path,
            ),
            threshold_persistence_frames=threshold_persistence_frames,
            rdf_cutoff_A=_positive_float(
                analysis_raw["rdf_cutoff_A"], "analysis.rdf_cutoff_A", config_path
            ),
            rdf_bins=_positive_int(
                analysis_raw["rdf_bins"], "analysis.rdf_bins", config_path
            ),
        ),
        output=HomogeneousOutputConfig(
            root_dir=_repo_path(output_raw["root_dir"]),
            overwrite=bool(output_raw["overwrite"]),
            save_extxyz=bool(output_raw["save_extxyz"]),
            create_visualizations=bool(output_raw["create_visualizations"]),
        ),
        generator=generator,
        config_path=config_path,
    )
    homogeneous_temperatures_K = (config.temperature_K,)
    validate_potential_qualification(
        config.generator,
        chemical_symbol=config.generator.system.chemical_symbol,
        pressure_GPa=config.generator.dynamics.pressure_GPa,
        timestep_fs=config.generator.dynamics.timestep_fs,
        state_temperatures_K={
            "liquid_bulk": homogeneous_temperatures_K,
            "interface": homogeneous_temperatures_K,
            "nucleus": homogeneous_temperatures_K,
        },
        context=f"homogeneous crystallization configuration {config_path}",
        required_claim="kinetics",
    )
    return config
