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
class TransitionBranchConfig:
    name: str
    expected_direction: str
    temperature_K: float
    equilibration_steps: int
    production_steps: int
    steady_state_start_step: int
    steady_state_end_step: int
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
    profile_smoothing_bins: int
    ptm_rmsd_cutoff: float
    minimum_profile_contrast: float
    minimum_velocity_fit_r_squared: float
    rdf_cutoff_A: float
    rdf_bins: int


@dataclass(frozen=True)
class TransitionConfig:
    dataset_name: str
    source_dataset: Path
    source_interface_environment: str
    source_frame_step: int
    random_seeds: tuple[int, ...]
    sample_interval: int
    analysis: TransitionAnalysisConfig
    temperature_runs: tuple[TransitionBranchConfig, ...]
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


def _boolean(value: Any, context: str, path: Path) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{path}: {context} must be a boolean, got {value!r}.")
    return value


def _branch(
    raw: dict[str, Any],
    *,
    index: int,
    path: Path,
) -> TransitionBranchConfig:
    _reject_unknown(
        raw,
        {
            "temperature_K",
            "name",
            "expected_direction",
            "equilibration_steps",
            "production_steps",
            "steady_state_start_step",
            "steady_state_end_step",
            "minimum_crystalline_fraction_change",
        },
        f"temperature_runs[{index}]",
        path,
    )
    context = f"temperature_runs[{index}]"
    name = str(raw["name"])
    if not name or not all(character.isalnum() or character in "_-" for character in name):
        raise ValueError(
            f"{path}: {context}.name must be a non-empty filesystem-safe identifier, "
            f"got {name!r}."
        )
    expected_direction = str(raw["expected_direction"])
    if expected_direction not in {"growth", "melting", "unconstrained"}:
        raise ValueError(
            f"{path}: {context}.expected_direction must be 'growth', 'melting', or "
            f"'unconstrained', got {expected_direction!r}."
        )
    equilibration_steps = _positive_int(
        raw["equilibration_steps"], f"{context}.equilibration_steps", path
    )
    production_steps = _positive_int(
        raw["production_steps"], f"{context}.production_steps", path
    )
    steady_state_start_step = _positive_int(
        raw["steady_state_start_step"], f"{context}.steady_state_start_step", path
    )
    steady_state_end_step = _positive_int(
        raw["steady_state_end_step"], f"{context}.steady_state_end_step", path
    )
    if steady_state_start_step >= steady_state_end_step:
        raise ValueError(
            f"{path}: {context}.steady_state_start_step must be below "
            f"{context}.steady_state_end_step, got {steady_state_start_step} and "
            f"{steady_state_end_step}."
        )
    if steady_state_end_step > production_steps:
        raise ValueError(
            f"{path}: {context}.steady_state_end_step={steady_state_end_step} exceeds "
            f"{context}.production_steps={production_steps}."
        )
    minimum_fraction_change = _nonnegative_float(
        raw["minimum_crystalline_fraction_change"],
        f"{context}.minimum_crystalline_fraction_change",
        path,
    )
    if expected_direction == "unconstrained" and minimum_fraction_change != 0.0:
        raise ValueError(
            f"{path}: {context}.minimum_crystalline_fraction_change must be 0 when "
            "expected_direction='unconstrained', got "
            f"{minimum_fraction_change}."
        )
    return TransitionBranchConfig(
        name=name,
        expected_direction=expected_direction,
        temperature_K=_positive_float(raw["temperature_K"], f"{context}.temperature_K", path),
        equilibration_steps=equilibration_steps,
        production_steps=production_steps,
        steady_state_start_step=steady_state_start_step,
        steady_state_end_step=steady_state_end_step,
        minimum_crystalline_fraction_change=minimum_fraction_change,
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
            "source_frame_step",
            "random_seeds",
            "sample_interval",
            "analysis",
            "temperature_runs",
            "output",
        },
        "root",
        config_path,
    )
    generator = load_config(_repo_path(raw["source_generator_config"]))
    temperature_runs_raw = raw["temperature_runs"]
    if not isinstance(temperature_runs_raw, list) or len(temperature_runs_raw) < 2:
        raise TypeError(
            f"{config_path}: temperature_runs must be a list with at least two mappings, "
            f"got {temperature_runs_raw!r}."
        )
    if any(not isinstance(item, dict) for item in temperature_runs_raw):
        raise TypeError(
            f"{config_path}: every temperature_runs entry must be a mapping."
        )
    temperature_runs = tuple(
        _branch(item, index=index, path=config_path)
        for index, item in enumerate(temperature_runs_raw)
    )
    names = [branch.name for branch in temperature_runs]
    temperatures = [branch.temperature_K for branch in temperature_runs]
    if len(set(names)) != len(names):
        raise ValueError(f"{config_path}: temperature_runs names must be unique, got {names}.")
    if len(set(temperatures)) != len(temperatures):
        raise ValueError(
            f"{config_path}: temperature_runs temperatures must be unique, got {temperatures}."
        )
    if temperatures != sorted(temperatures):
        raise ValueError(
            f"{config_path}: temperature_runs must be sorted by temperature_K, got "
            f"{temperatures}."
        )
    constrained_directions = {branch.expected_direction for branch in temperature_runs}
    if "growth" not in constrained_directions or "melting" not in constrained_directions:
        raise ValueError(
            f"{config_path}: temperature_runs must include at least one expected growth run "
            "and one expected melting run to verify that the grid brackets a direction change."
        )
    highest_growth_temperature = max(
        branch.temperature_K
        for branch in temperature_runs
        if branch.expected_direction == "growth"
    )
    lowest_melting_temperature = min(
        branch.temperature_K
        for branch in temperature_runs
        if branch.expected_direction == "melting"
    )
    if highest_growth_temperature >= lowest_melting_temperature:
        raise ValueError(
            f"{config_path}: every expected growth temperature must be below every expected "
            f"melting temperature, got highest growth={highest_growth_temperature} K and "
            f"lowest melting={lowest_melting_temperature} K."
        )
    validate_potential_qualification(
        generator,
        chemical_symbol=generator.system.chemical_symbol,
        pressure_GPa=generator.dynamics.pressure_GPa,
        timestep_fs=generator.dynamics.timestep_fs,
        state_temperatures_K={
            "interface": tuple(branch.temperature_K for branch in temperature_runs)
        },
        context=f"{config_path}: direct-coexistence temperature grid",
        required_claim="equilibrium_thermodynamics",
    )
    source_frame_step = raw["source_frame_step"]
    if not isinstance(source_frame_step, int) or isinstance(source_frame_step, bool):
        raise TypeError(
            f"{config_path}: source_frame_step must be an integer, got "
            f"{source_frame_step!r}."
        )
    if source_frame_step < 0:
        raise ValueError(
            f"{config_path}: source_frame_step must be >= 0, got {source_frame_step}."
        )
    random_seeds_raw = raw["random_seeds"]
    if not isinstance(random_seeds_raw, list) or len(random_seeds_raw) < 2:
        raise TypeError(
            f"{config_path}: random_seeds must contain at least two independent replicas so "
            "front-velocity uncertainty is defined, got "
            f"{random_seeds_raw!r}."
        )
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in random_seeds_raw):
        raise TypeError(
            f"{config_path}: every random_seeds entry must be an integer, got "
            f"{random_seeds_raw!r}."
        )
    if len(set(random_seeds_raw)) != len(random_seeds_raw):
        raise ValueError(
            f"{config_path}: random_seeds must be unique, got {random_seeds_raw!r}."
        )
    sample_interval = _positive_int(
        raw["sample_interval"], "sample_interval", config_path
    )
    for branch in temperature_runs:
        for field_name, step in (
            ("equilibration_steps", branch.equilibration_steps),
            ("steady_state_start_step", branch.steady_state_start_step),
            ("steady_state_end_step", branch.steady_state_end_step),
        ):
            if step % sample_interval:
                raise ValueError(
                    f"{config_path}: {branch.name}.{field_name}={step} must be divisible by "
                    f"sample_interval={sample_interval} so the equilibration boundary and "
                    "velocity-fit endpoints are stored exactly."
                )
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
        {
            "profile_bins",
            "profile_smoothing_bins",
            "ptm_rmsd_cutoff",
            "minimum_profile_contrast",
            "minimum_velocity_fit_r_squared",
            "rdf_cutoff_A",
            "rdf_bins",
        },
        "analysis",
        config_path,
    )
    profile_bins = _positive_int(
        analysis_raw["profile_bins"], "analysis.profile_bins", config_path
    )
    profile_smoothing_bins = _positive_int(
        analysis_raw["profile_smoothing_bins"],
        "analysis.profile_smoothing_bins",
        config_path,
    )
    if profile_smoothing_bins % 2 == 0:
        raise ValueError(
            f"{config_path}: analysis.profile_smoothing_bins must be odd, got "
            f"{profile_smoothing_bins}."
        )
    if profile_smoothing_bins > profile_bins:
        raise ValueError(
            f"{config_path}: analysis.profile_smoothing_bins={profile_smoothing_bins} "
            f"exceeds analysis.profile_bins={profile_bins}."
        )
    minimum_profile_contrast = _positive_float(
        analysis_raw["minimum_profile_contrast"],
        "analysis.minimum_profile_contrast",
        config_path,
    )
    if minimum_profile_contrast >= 1.0:
        raise ValueError(
            f"{config_path}: analysis.minimum_profile_contrast must be below 1, got "
            f"{minimum_profile_contrast}."
        )
    ptm_rmsd_cutoff = _positive_float(
        analysis_raw["ptm_rmsd_cutoff"],
        "analysis.ptm_rmsd_cutoff",
        config_path,
    )
    if ptm_rmsd_cutoff > 1.0:
        raise ValueError(
            f"{config_path}: analysis.ptm_rmsd_cutoff is a dimensionless normalized RMSD "
            f"and must be <= 1, got {ptm_rmsd_cutoff}."
        )
    minimum_velocity_fit_r_squared = _nonnegative_float(
        analysis_raw["minimum_velocity_fit_r_squared"],
        "analysis.minimum_velocity_fit_r_squared",
        config_path,
    )
    if minimum_velocity_fit_r_squared > 1.0:
        raise ValueError(
            f"{config_path}: analysis.minimum_velocity_fit_r_squared must be <= 1, got "
            f"{minimum_velocity_fit_r_squared}."
        )
    config = TransitionConfig(
        dataset_name=str(raw["dataset_name"]),
        source_dataset=_repo_path(raw["source_dataset"]),
        source_interface_environment=str(raw["source_interface_environment"]),
        source_frame_step=source_frame_step,
        random_seeds=tuple(random_seeds_raw),
        sample_interval=sample_interval,
        analysis=TransitionAnalysisConfig(
            profile_bins=profile_bins,
            profile_smoothing_bins=profile_smoothing_bins,
            ptm_rmsd_cutoff=ptm_rmsd_cutoff,
            minimum_profile_contrast=minimum_profile_contrast,
            minimum_velocity_fit_r_squared=minimum_velocity_fit_r_squared,
            rdf_cutoff_A=_positive_float(
                analysis_raw["rdf_cutoff_A"], "analysis.rdf_cutoff_A", config_path
            ),
            rdf_bins=_positive_int(
                analysis_raw["rdf_bins"], "analysis.rdf_bins", config_path
            ),
        ),
        temperature_runs=temperature_runs,
        output=TransitionOutputConfig(
            root_dir=_repo_path(output_raw["root_dir"]),
            overwrite=_boolean(output_raw["overwrite"], "output.overwrite", config_path),
            save_extxyz=_boolean(
                output_raw["save_extxyz"], "output.save_extxyz", config_path
            ),
            create_visualizations=_boolean(
                output_raw["create_visualizations"],
                "output.create_visualizations",
                config_path,
            ),
        ),
        generator=generator,
        config_path=config_path,
    )
    transition_temperatures_K = tuple(
        branch.temperature_K for branch in config.temperature_runs
    )
    validate_potential_qualification(
        config.generator,
        chemical_symbol=config.generator.system.chemical_symbol,
        pressure_GPa=config.generator.dynamics.pressure_GPa,
        timestep_fs=config.generator.dynamics.timestep_fs,
        state_temperatures_K={
            "solid_bulk": transition_temperatures_K,
            "liquid_bulk": transition_temperatures_K,
            "interface": transition_temperatures_K,
        },
        context=f"direct-coexistence configuration {config_path}",
        required_claim="equilibrium_thermodynamics",
    )
    return config
