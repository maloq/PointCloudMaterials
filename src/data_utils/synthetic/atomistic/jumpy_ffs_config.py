"""Strict YAML configuration for homogeneous-crystallization jumpy FFS."""

from __future__ import annotations

import hashlib
import json
import math
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
from .jumpy_ffs import JumpyFFSAlgorithmConfig, _json_value
from .potential_selection import (
    POTENTIAL_SELECTION_POLICY_VERSION,
    POTENTIAL_SELECTION_SCHEMA_VERSION,
)


@dataclass(frozen=True)
class JumpyFFSCVConfig:
    ptm_rmsd_cutoff: float
    cluster_cutoff_A: float


@dataclass(frozen=True)
class JumpyFFSRunConfig:
    dataset_name: str
    source_dataset: Path
    source_environment: str
    source_frame_step: int
    temperature_K: float
    timestep_fs: float
    friction_time_fs: float
    shot_md_property_mode: str
    cv: JumpyFFSCVConfig
    algorithm: JumpyFFSAlgorithmConfig
    output_root: Path
    potential_selection_report: Path
    potential_selection_report_sha256: str
    selected_generator_config_sha256: str
    generator: GeneratorConfig
    config_path: Path

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["generator"] = self.generator.to_dict()
        return _json_value(value)


def _repo_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _mapping(parent: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: {key} must be a mapping, got {type(value).__name__}.")
    return value


def _reject_unknown(
    value: dict[str, Any], allowed: set[str], *, context: str, path: Path
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise KeyError(f"{path}: unsupported keys in {context}: {unknown}.")


def _positive_float(value: Any, *, context: str, path: Path) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{path}: {context} must be an explicit number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{path}: {context} must be finite and > 0, got {result}.")
    return result


def _integer(value: Any, *, context: str, path: Path, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{path}: {context} must be an integer, got {value!r}.")
    if value < minimum:
        raise ValueError(f"{path}: {context} must be >= {minimum}, got {value}.")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_potential_selection_report(
    value: Any,
    *,
    generator: GeneratorConfig,
    config_path: Path,
) -> tuple[Path, str, str]:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(
            f"{config_path}: potential_selection_report must be a non-empty path "
            "string. jFFS will not run either MPA-0 or MH-1 without an explicit "
            "model-selection result."
        )
    report_path = _repo_path(value)
    if not report_path.is_file():
        raise FileNotFoundError(
            f"{config_path}: potential_selection_report does not exist: {report_path}. "
            "Run the checksum-bound potential-selection workflow first; jFFS will not "
            "fall back to its configured model silently."
        )
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise TypeError(f"{report_path}: potential-selection report must be a mapping.")
    if (
        report.get("schema_version") != POTENTIAL_SELECTION_SCHEMA_VERSION
        or report.get("report_type") != "al_crystallization_mlip_selection"
        or report.get("policy_version") != POTENTIAL_SELECTION_POLICY_VERSION
    ):
        raise RuntimeError(
            f"{report_path}: expected schema_version="
            f"{POTENTIAL_SELECTION_SCHEMA_VERSION}, report_type="
            "'al_crystallization_mlip_selection', and policy_version="
            f"{POTENTIAL_SELECTION_POLICY_VERSION!r}."
        )
    selected_config_value = report.get("selected_generator_config")
    if not isinstance(selected_config_value, str) or not selected_config_value:
        raise TypeError(
            f"{report_path}: selected_generator_config must be a non-empty path string."
        )
    selected_config_path = _repo_path(selected_config_value)
    if selected_config_path != generator.config_path:
        raise RuntimeError(
            f"{config_path}: selection report chose generator config "
            f"{selected_config_path}, but source_generator_config resolves to "
            f"{generator.config_path}. jFFS source preparation, fixed-shape budgets, "
            "model, and head must come from the exact selected production config."
        )
    if not selected_config_path.is_file():
        raise FileNotFoundError(
            f"{report_path}: selected_generator_config does not exist: "
            f"{selected_config_path}."
        )
    inputs = report.get("inputs")
    if not isinstance(inputs, dict):
        raise TypeError(f"{report_path}: inputs must be a mapping.")
    selected_checksum: str | None = None
    selected_role: str | None = None
    for role in ("baseline", "candidate"):
        role_config_value = inputs.get(f"{role}_generator_config")
        if not isinstance(role_config_value, str) or not role_config_value:
            continue
        if _repo_path(role_config_value) == selected_config_path:
            selected_role = role
            checksum = inputs.get(f"{role}_generator_config_sha256")
            if (
                not isinstance(checksum, str)
                or len(checksum) != 64
                or any(character not in "0123456789abcdef" for character in checksum)
            ):
                raise TypeError(
                    f"{report_path}: inputs.{role}_generator_config_sha256 must be "
                    "64 lowercase hexadecimal characters."
                )
            selected_checksum = checksum
            break
    if selected_checksum is None or selected_role is None:
        raise RuntimeError(
            f"{report_path}: selected_generator_config={selected_config_path} is not one "
            "of the report's SHA-bound baseline/candidate generator inputs."
        )
    observed_config_sha256 = _sha256(selected_config_path)
    if observed_config_sha256 != selected_checksum:
        raise RuntimeError(
            f"{report_path}: selected {selected_role} generator config changed after "
            f"selection: recorded SHA-256={selected_checksum}, observed="
            f"{observed_config_sha256}."
        )
    selected_generator = load_config(selected_config_path)
    selected_potential = selected_generator.potential
    configured_potential = generator.potential
    identity_mismatches = {
        field: {
            "selected": getattr(selected_potential, field),
            "configured": getattr(configured_potential, field),
        }
        for field in ("model_name", "sha256", "head", "family")
        if getattr(selected_potential, field) != getattr(configured_potential, field)
    }
    if identity_mismatches:
        raise RuntimeError(
            f"{config_path}: source_generator_config does not use the potential selected "
            f"by {report_path}: mismatches={identity_mismatches}."
        )
    if report.get("selected_model_name") != configured_potential.model_name:
        raise RuntimeError(
            f"{report_path}: selected_model_name="
            f"{report.get('selected_model_name')!r} differs from jFFS model_name="
            f"{configured_potential.model_name!r}."
        )
    return report_path, _sha256(report_path), selected_checksum


def load_jumpy_ffs_config(path: str | Path) -> JumpyFFSRunConfig:
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
            "potential_selection_report",
            "ensemble",
            "temperature_K",
            "timestep_fs",
            "friction_time_fs",
            "shot_calculator",
            "random_seed",
            "equilibration_steps",
            "equilibration_checkpoint_interval_steps",
            "cv",
            "basin",
            "shooting",
            "uncertainty",
            "output",
        },
        context="root",
        path=config_path,
    )
    ensemble = raw.get("ensemble")
    if ensemble != "langevin_nvt":
        raise RuntimeError(
            f"{config_path}: ensemble must be exactly 'langevin_nvt', got "
            f"{ensemble!r}. MTK-NPT branch checkpoints in this repository do not serialize "
            "thermostat-chain and barostat state."
        )
    cv_raw = _mapping(raw, "cv", config_path)
    basin_raw = _mapping(raw, "basin", config_path)
    shooting_raw = _mapping(raw, "shooting", config_path)
    uncertainty_raw = _mapping(raw, "uncertainty", config_path)
    output_raw = _mapping(raw, "output", config_path)
    shot_calculator_raw = _mapping(raw, "shot_calculator", config_path)
    _reject_unknown(
        cv_raw,
        {"type", "ptm_rmsd_cutoff", "cluster_cutoff_A", "interval_steps", "interfaces_atoms"},
        context="cv",
        path=config_path,
    )
    _reject_unknown(
        basin_raw,
        {"target_crossings", "max_steps", "checkpoint_interval_steps"},
        context="basin",
        path=config_path,
    )
    _reject_unknown(
        shooting_raw,
        {"trials_per_state", "max_steps", "checkpoint_interval_steps"},
        context="shooting",
        path=config_path,
    )
    _reject_unknown(
        uncertainty_raw,
        {"bootstrap_samples", "bootstrap_block_crossings"},
        context="uncertainty",
        path=config_path,
    )
    _reject_unknown(output_raw, {"root_dir"}, context="output", path=config_path)
    _reject_unknown(
        shot_calculator_raw,
        {"md_property_mode"},
        context="shot_calculator",
        path=config_path,
    )
    if shot_calculator_raw.get("md_property_mode") != "forces":
        raise ValueError(
            f"{config_path}: shot_calculator.md_property_mode must be exactly 'forces' "
            "for fixed-cell Langevin-NVT; stress is unused and must not be evaluated."
        )
    if cv_raw.get("type") != "ptm_largest_crystalline_cluster_atoms":
        raise ValueError(
            f"{config_path}: cv.type must be "
            "'ptm_largest_crystalline_cluster_atoms'; got "
            f"{cv_raw.get('type')!r}."
        )
    interfaces_raw = cv_raw.get("interfaces_atoms")
    if not isinstance(interfaces_raw, list):
        raise TypeError(
            f"{config_path}: cv.interfaces_atoms must be a list of integer cluster "
            f"sizes, got {interfaces_raw!r}."
        )
    interfaces = tuple(
        _integer(
            value,
            context=f"cv.interfaces_atoms[{index}]",
            path=config_path,
            minimum=1,
        )
        for index, value in enumerate(interfaces_raw)
    )
    source_frame_step = _integer(
        raw.get("source_frame_step"),
        context="source_frame_step",
        path=config_path,
        minimum=0,
    )
    timestep_fs = _positive_float(
        raw.get("timestep_fs"), context="timestep_fs", path=config_path
    )
    generator = load_config(_repo_path(raw.get("source_generator_config")))
    selection_report, selection_report_sha256, selected_generator_sha256 = (
        _validate_potential_selection_report(
            raw.get("potential_selection_report"),
            generator=generator,
            config_path=config_path,
        )
    )
    if timestep_fs != generator.dynamics.timestep_fs:
        raise ValueError(
            f"{config_path}: timestep_fs={timestep_fs} must match the source generator's "
            f"qualified timestep_fs={generator.dynamics.timestep_fs}; jFFS cannot silently "
            "change integration stability assumptions."
        )
    algorithm = JumpyFFSAlgorithmConfig(
        interfaces_atoms=interfaces,
        equilibration_steps=_integer(
            raw.get("equilibration_steps"),
            context="equilibration_steps",
            path=config_path,
            minimum=0,
        ),
        equilibration_checkpoint_interval_steps=_integer(
            raw.get("equilibration_checkpoint_interval_steps"),
            context="equilibration_checkpoint_interval_steps",
            path=config_path,
            minimum=1,
        ),
        basin_target_crossings=_integer(
            basin_raw.get("target_crossings"),
            context="basin.target_crossings",
            path=config_path,
            minimum=1,
        ),
        basin_max_steps=_integer(
            basin_raw.get("max_steps"),
            context="basin.max_steps",
            path=config_path,
            minimum=1,
        ),
        basin_checkpoint_interval_steps=_integer(
            basin_raw.get("checkpoint_interval_steps"),
            context="basin.checkpoint_interval_steps",
            path=config_path,
            minimum=1,
        ),
        cv_interval_steps=_integer(
            cv_raw.get("interval_steps"),
            context="cv.interval_steps",
            path=config_path,
            minimum=1,
        ),
        trials_per_state=_integer(
            shooting_raw.get("trials_per_state"),
            context="shooting.trials_per_state",
            path=config_path,
            minimum=1,
        ),
        shot_max_steps=_integer(
            shooting_raw.get("max_steps"),
            context="shooting.max_steps",
            path=config_path,
            minimum=1,
        ),
        shot_checkpoint_interval_steps=_integer(
            shooting_raw.get("checkpoint_interval_steps"),
            context="shooting.checkpoint_interval_steps",
            path=config_path,
            minimum=1,
        ),
        bootstrap_samples=_integer(
            uncertainty_raw.get("bootstrap_samples"),
            context="uncertainty.bootstrap_samples",
            path=config_path,
            minimum=2,
        ),
        bootstrap_block_crossings=_integer(
            uncertainty_raw.get("bootstrap_block_crossings"),
            context="uncertainty.bootstrap_block_crossings",
            path=config_path,
            minimum=1,
        ),
        random_seed=_integer(
            raw.get("random_seed"),
            context="random_seed",
            path=config_path,
            minimum=0,
        ),
    )
    algorithm.validate()
    ptm_rmsd_cutoff = _positive_float(
        cv_raw.get("ptm_rmsd_cutoff"),
        context="cv.ptm_rmsd_cutoff",
        path=config_path,
    )
    if ptm_rmsd_cutoff > 1.0:
        raise ValueError(
            f"{config_path}: cv.ptm_rmsd_cutoff is normalized and must be <= 1, got "
            f"{ptm_rmsd_cutoff}."
        )
    temperature_K = _positive_float(
        raw.get("temperature_K"), context="temperature_K", path=config_path
    )
    config = JumpyFFSRunConfig(
        dataset_name=str(raw.get("dataset_name")),
        source_dataset=_repo_path(raw.get("source_dataset")),
        source_environment=str(raw.get("source_environment")),
        source_frame_step=source_frame_step,
        temperature_K=temperature_K,
        timestep_fs=timestep_fs,
        friction_time_fs=_positive_float(
            raw.get("friction_time_fs"),
            context="friction_time_fs",
            path=config_path,
        ),
        shot_md_property_mode="forces",
        cv=JumpyFFSCVConfig(
            ptm_rmsd_cutoff=ptm_rmsd_cutoff,
            cluster_cutoff_A=_positive_float(
                cv_raw.get("cluster_cutoff_A"),
                context="cv.cluster_cutoff_A",
                path=config_path,
            ),
        ),
        algorithm=algorithm,
        output_root=_repo_path(output_raw.get("root_dir")),
        potential_selection_report=selection_report,
        potential_selection_report_sha256=selection_report_sha256,
        selected_generator_config_sha256=selected_generator_sha256,
        generator=generator,
        config_path=config_path,
    )
    if not config.dataset_name or config.dataset_name == "None":
        raise ValueError(f"{config_path}: dataset_name must be a non-empty string.")
    if not config.source_environment or config.source_environment == "None":
        raise ValueError(f"{config_path}: source_environment must be a non-empty string.")
    if config.generator.dynamics.target_temperature_K != config.temperature_K:
        raise ValueError(
            f"{config_path}: source generator target_temperature_K="
            f"{config.generator.dynamics.target_temperature_K} does not match jFFS "
            f"temperature_K={config.temperature_K}. Fixed-volume NVT shots must start from "
            "a liquid whose cell was NPT-equilibrated at the same temperature and pressure; "
            "do not cool a differently equilibrated cell at fixed volume."
        )
    if (
        config.source_frame_step
        != config.generator.dynamics.target_equilibration_steps
    ):
        raise ValueError(
            f"{config_path}: source_frame_step={config.source_frame_step} must select the "
            "NPT liquid equilibration endpoint at target_equilibration_steps="
            f"{config.generator.dynamics.target_equilibration_steps}. An earlier frame "
            "does not provide the final pressure-equilibrated fixed volume."
        )
    validate_potential_qualification(
        config.generator,
        chemical_symbol=config.generator.system.chemical_symbol,
        pressure_GPa=config.generator.dynamics.pressure_GPa,
        timestep_fs=config.timestep_fs,
        state_temperatures_K={
            "liquid_bulk": (config.temperature_K,),
            "nucleus": (config.temperature_K,),
        },
        context=f"jumpy FFS configuration {config_path}",
        required_claim="kinetics",
    )
    return config
