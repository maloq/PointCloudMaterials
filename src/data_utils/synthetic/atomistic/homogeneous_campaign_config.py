from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import REPOSITORY_ROOT, load_config
from .homogeneous_config import (
    HomogeneousCrystallizationConfig,
    load_homogeneous_crystallization_config,
)
from .potential_selection import (
    POTENTIAL_SELECTION_POLICY_VERSION,
    POTENTIAL_SELECTION_SCHEMA_VERSION,
)


ANALYSIS_MODES = ("asynchronous", "deferred")


def campaign_config_matches_after_path_relocation(
    observed: object,
    expected: dict[str, Any],
) -> bool:
    """Accept only repository config-file location changes."""

    if observed == expected:
        return True
    if not isinstance(observed, dict) or set(observed) != set(expected):
        return False
    relocated = deepcopy(observed)
    try:
        relocated["config_path"] = expected["config_path"]
        relocated["homogeneous_config"] = expected["homogeneous_config"]
        relocated_homogeneous = relocated["homogeneous"]
        expected_homogeneous = expected["homogeneous"]
        if not isinstance(relocated_homogeneous, dict) or not isinstance(
            expected_homogeneous, dict
        ):
            return False
        relocated_homogeneous["config_path"] = expected_homogeneous["config_path"]
        relocated_generator = relocated_homogeneous["generator"]
        expected_generator = expected_homogeneous["generator"]
        if not isinstance(relocated_generator, dict) or not isinstance(
            expected_generator, dict
        ):
            return False
        relocated_generator["config_path"] = expected_generator["config_path"]
    except KeyError:
        return False
    return relocated == expected


@dataclass(frozen=True)
class HomogeneousCampaignExecutionConfig:
    chunk_steps: int
    event_check_interval: int
    online_persistence_frames: int
    stop_on_event: bool
    post_event_steps: int
    analysis_mode: str
    analysis_workers: int
    checkpoint_retention: int


@dataclass(frozen=True)
class HomogeneousCampaignConfig:
    homogeneous: HomogeneousCrystallizationConfig
    output_root: Path
    execution: HomogeneousCampaignExecutionConfig
    potential_selection_report: Path | None
    potential_selection_report_sha256: str | None
    potential_selection_runtime_controls: dict[str, int | float | bool] | None
    source_evidence: dict[str, dict[str, str]]
    config_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "homogeneous_config": str(self.homogeneous.config_path),
            "homogeneous": self.homogeneous.to_dict(),
            "output_root": str(self.output_root),
            "execution": asdict(self.execution),
            "potential_selection_report": (
                None
                if self.potential_selection_report is None
                else str(self.potential_selection_report)
            ),
            "potential_selection_report_sha256": (
                self.potential_selection_report_sha256
            ),
            "potential_selection_runtime_controls": (
                self.potential_selection_runtime_controls
            ),
            "source_evidence": self.source_evidence,
            "config_path": str(self.config_path),
        }


def _repo_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _integer(value: Any, *, name: str, path: Path, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        comparator = ">= 0" if minimum == 0 else "> 0"
        raise ValueError(
            f"{path}: execution.{name} must be an integer {comparator}, got {value!r}."
        )
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
    homogeneous: HomogeneousCrystallizationConfig,
    source_evidence: dict[str, dict[str, str]],
    config_path: Path,
) -> tuple[Path | None, str | None, dict[str, int | float | bool] | None]:
    if value is None:
        return None, None, None
    report_path = _repo_path(value)
    if not report_path.is_file():
        raise FileNotFoundError(
            f"{config_path}: requested potential_selection_report does not exist: "
            f"{report_path}. Run the explicit potential-selection command first; the "
            "campaign will not fall back to its generator model silently."
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
        if inputs.get(f"{role}_generator_config") == str(selected_config_path):
            checksum = inputs.get(f"{role}_generator_config_sha256")
            if not isinstance(checksum, str) or len(checksum) != 64:
                raise TypeError(
                    f"{report_path}: inputs.{role}_generator_config_sha256 must be a "
                    "64-character SHA-256 string."
                )
            selected_checksum = checksum
            selected_role = role
            break
    if selected_checksum is None or selected_role is None:
        raise RuntimeError(
            f"{report_path}: selected_generator_config={selected_config_path} is not one "
            "of the report's hashed baseline/candidate generator inputs."
        )
    observed_selected_checksum = _sha256(selected_config_path)
    if observed_selected_checksum != selected_checksum:
        raise RuntimeError(
            f"{report_path}: selected generator config changed after selection: recorded "
            f"SHA-256={selected_checksum}, observed={observed_selected_checksum}."
        )
    if selected_config_path != homogeneous.generator.config_path:
        raise RuntimeError(
            f"{config_path}: selection report chose generator config "
            f"{selected_config_path}, but homogeneous source_generator_config resolves to "
            f"{homogeneous.generator.config_path}. Campaign size, thermodynamic protocol, "
            "fixed-shape budgets, and model settings must come from the exact selected "
            "production config, not merely a config using the same model."
        )
    selected_config = load_config(selected_config_path)
    campaign_potential = homogeneous.generator.potential
    selected_potential = selected_config.potential
    identity_mismatches = {
        field: {
            "selected": getattr(selected_potential, field),
            "campaign": getattr(campaign_potential, field),
        }
        for field in ("model_name", "sha256", "head", "family")
        if getattr(selected_potential, field) != getattr(campaign_potential, field)
    }
    if identity_mismatches:
        raise RuntimeError(
            f"{config_path}: campaign source_generator_config does not use the potential "
            f"selected by {report_path}: mismatches={identity_mismatches}. Create the "
            "liquid-source/homogeneous configs for the selected model explicitly."
        )
    if report.get("selected_model_name") != campaign_potential.model_name:
        raise RuntimeError(
            f"{report_path}: selected_model_name={report.get('selected_model_name')!r} "
            f"differs from campaign model_name={campaign_potential.model_name!r}."
        )

    projection = report.get("runtime_projection")
    if not isinstance(projection, dict):
        raise TypeError(
            f"{report_path}: runtime_projection must be a mapping."
        )
    if projection.get("is_selection_or_launch_gate") is not False:
        raise RuntimeError(
            f"{report_path}: runtime_projection.is_selection_or_launch_gate must be "
            "false."
        )
    workers = projection.get("workers")
    makespan_safety_factor = projection.get("makespan_safety_factor")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise TypeError(
            f"{report_path}: runtime_projection.workers must be a "
            f"positive integer, got {workers!r}."
        )
    if (
        not isinstance(makespan_safety_factor, (int, float))
        or isinstance(makespan_safety_factor, bool)
        or not math.isfinite(float(makespan_safety_factor))
        or float(makespan_safety_factor) < 1.0
    ):
        raise ValueError(
            f"{report_path}: makespan_safety_factor must be finite and >= 1, got "
            f"{makespan_safety_factor!r}."
        )

    selected_projection = projection.get(selected_role)
    if not isinstance(selected_projection, dict):
        raise TypeError(
            f"{report_path}: runtime_projection.{selected_role} must be a mapping."
        )
    projected_model_name = selected_projection.get("model_name")
    projected_homogeneous_value = selected_projection.get("homogeneous_config")
    projected_homogeneous_sha256 = selected_projection.get(
        "homogeneous_config_sha256"
    )
    projected_makespan_seconds = selected_projection.get(
        "projected_makespan_seconds"
    )
    projected_source_evidence = selected_projection.get(
        "initial_source_artifacts"
    )
    if projected_model_name != campaign_potential.model_name:
        raise RuntimeError(
            f"{report_path}: selected {selected_role} runtime projection is for "
            f"model={projected_model_name!r}, expected "
            f"{campaign_potential.model_name!r}."
        )
    if not isinstance(projected_homogeneous_value, str):
        raise TypeError(
            f"{report_path}: selected homogeneous_config must be a path string."
        )
    projected_homogeneous_path = _repo_path(projected_homogeneous_value)
    if projected_homogeneous_path != homogeneous.config_path:
        raise RuntimeError(
            f"{config_path}: runtime was projected for homogeneous "
            f"config {projected_homogeneous_path}, but this campaign uses "
            f"{homogeneous.config_path}."
        )
    if (
        not isinstance(projected_homogeneous_sha256, str)
        or projected_homogeneous_sha256 != _sha256(homogeneous.config_path)
    ):
        raise RuntimeError(
            f"{report_path}: selected homogeneous workload changed after its "
            f"runtime projection: {homogeneous.config_path}."
        )
    if (
        not isinstance(projected_makespan_seconds, (int, float))
        or isinstance(projected_makespan_seconds, bool)
        or not math.isfinite(float(projected_makespan_seconds))
        or float(projected_makespan_seconds) < 0.0
    ):
        raise ValueError(
            f"{report_path}: selected projected_makespan_seconds must be finite and "
            f"nonnegative, got {projected_makespan_seconds!r}."
        )
    if projected_source_evidence != source_evidence:
        raise RuntimeError(
            f"{report_path}: selected runtime projection source artifacts differ "
            f"from this campaign's immutable source: projected="
            f"{projected_source_evidence!r}, campaign={source_evidence!r}."
        )
    runtime_controls: dict[str, int | float | bool] = {
        "workers": workers,
        "makespan_safety_factor": float(makespan_safety_factor),
        "projected_makespan_seconds": float(projected_makespan_seconds),
        "is_selection_or_launch_gate": False,
    }
    return report_path, _sha256(report_path), runtime_controls


def _bind_source_evidence(
    homogeneous: HomogeneousCrystallizationConfig,
    *,
    config_path: Path,
) -> dict[str, dict[str, str]]:
    source_directory = homogeneous.source_dataset / homogeneous.source_environment
    paths = {
        "manifest": homogeneous.source_dataset / "manifest.json",
        "metadata": source_directory / "metadata.json",
        "atom_table": source_directory / "atoms_full.npy",
        "trajectory": source_directory / "trajectory.npz",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{config_path}: immutable homogeneous source evidence is incomplete; "
            f"missing={missing}. Generate the configured liquid source before loading the "
            "campaign. No path-only or unverified source fallback is permitted."
        )
    manifest_path = paths["manifest"]
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise TypeError(f"{manifest_path}: immutable source manifest must be a mapping.")
    if manifest.get("source_kind") != "immutable_homogeneous_liquid_only":
        raise RuntimeError(
            f"{manifest_path}: optimized campaign requires source_kind="
            "'immutable_homogeneous_liquid_only', got "
            f"{manifest.get('source_kind')!r}. Generate the dedicated liquid source; "
            "a general phase-context artifact is not the configured reusable source."
        )
    if manifest.get("interface_preparation_performed") is not False:
        raise RuntimeError(
            f"{manifest_path}: immutable liquid source must explicitly record "
            "interface_preparation_performed=false."
        )
    return {
        name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for name, path in paths.items()
    }


def load_homogeneous_campaign_config(
    path: str | Path,
) -> HomogeneousCampaignConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"{config_path}: root must be a mapping.")
    expected_root_keys = {
        "homogeneous_config",
        "output_root",
        "execution",
        "potential_selection_report",
    }
    unknown_root = sorted(set(raw) - expected_root_keys)
    missing_root = sorted(expected_root_keys - set(raw))
    if unknown_root or missing_root:
        raise KeyError(
            f"{config_path}: campaign root keys must be exactly "
            f"{sorted(expected_root_keys)}; missing={missing_root}, unsupported={unknown_root}."
        )
    execution_raw = raw["execution"]
    if not isinstance(execution_raw, dict):
        raise TypeError(
            f"{config_path}: execution must be a mapping, got "
            f"{type(execution_raw).__name__}."
        )
    expected_execution_keys = {
        "chunk_steps",
        "event_check_interval",
        "stop_on_event",
        "post_event_steps",
        "analysis_mode",
        "analysis_workers",
        "checkpoint_retention",
    }
    unknown_execution = sorted(set(execution_raw) - expected_execution_keys)
    missing_execution = sorted(expected_execution_keys - set(execution_raw))
    if unknown_execution or missing_execution:
        raise KeyError(
            f"{config_path}: execution keys must be exactly "
            f"{sorted(expected_execution_keys)}; missing={missing_execution}, "
            f"unsupported={unknown_execution}."
        )
    homogeneous = load_homogeneous_crystallization_config(
        _repo_path(raw["homogeneous_config"])
    )
    source_evidence = _bind_source_evidence(
        homogeneous, config_path=config_path
    )
    (
        selection_report,
        selection_report_sha256,
        selection_runtime_controls,
    ) = _validate_potential_selection_report(
        raw["potential_selection_report"],
        homogeneous=homogeneous,
        source_evidence=source_evidence,
        config_path=config_path,
    )
    chunk_steps = _integer(
        execution_raw["chunk_steps"],
        name="chunk_steps",
        path=config_path,
        minimum=1,
    )
    event_check_interval = _integer(
        execution_raw["event_check_interval"],
        name="event_check_interval",
        path=config_path,
        minimum=1,
    )
    post_event_steps = _integer(
        execution_raw["post_event_steps"],
        name="post_event_steps",
        path=config_path,
        minimum=0,
    )
    analysis_workers = _integer(
        execution_raw["analysis_workers"],
        name="analysis_workers",
        path=config_path,
        minimum=0,
    )
    checkpoint_retention = _integer(
        execution_raw["checkpoint_retention"],
        name="checkpoint_retention",
        path=config_path,
        minimum=1,
    )
    stop_on_event = execution_raw["stop_on_event"]
    if not isinstance(stop_on_event, bool):
        raise TypeError(
            f"{config_path}: execution.stop_on_event must be true or false, got "
            f"{stop_on_event!r}."
        )
    if not stop_on_event and post_event_steps:
        raise ValueError(
            f"{config_path}: execution.post_event_steps={post_event_steps} requires "
            "execution.stop_on_event=true; a full-duration trajectory already contains "
            "all configured post-event growth."
        )
    analysis_mode = execution_raw["analysis_mode"]
    if analysis_mode not in ANALYSIS_MODES:
        raise ValueError(
            f"{config_path}: execution.analysis_mode must be one of {ANALYSIS_MODES}, "
            f"got {analysis_mode!r}."
        )
    if analysis_mode == "asynchronous" and analysis_workers == 0:
        raise ValueError(
            f"{config_path}: asynchronous analysis requires analysis_workers > 0."
        )
    if analysis_mode == "deferred" and analysis_workers != 0:
        raise ValueError(
            f"{config_path}: deferred analysis requires analysis_workers=0; run the "
            "separate analyze command with an explicit worker count later."
        )
    if homogeneous.equilibration_steps % event_check_interval:
        raise ValueError(
            f"{config_path}: equilibration_steps={homogeneous.equilibration_steps} must be "
            f"divisible by event_check_interval={event_check_interval} so the online event "
            "series contains the exact waiting-time origin."
        )
    if homogeneous.steps % event_check_interval:
        raise ValueError(
            f"{config_path}: steps={homogeneous.steps} must be divisible by "
            f"event_check_interval={event_check_interval} so a no-event trajectory is "
            "right-censored at an observed online frame."
        )
    if homogeneous.sample_interval % event_check_interval:
        raise ValueError(
            f"{config_path}: sample_interval={homogeneous.sample_interval} must be "
            f"divisible by event_check_interval={event_check_interval}. Online control "
            "must contain every saved scientific-analysis frame plus optional denser "
            "frames; it may not skip or shift the configured event observations."
        )
    # Dense checks are monitoring/control data only. Event persistence is evaluated on
    # the original saved-frame cadence, so neither a sub-cadence dip nor an apparent
    # between-frame onset changes the repository's event definition.
    online_persistence_frames = homogeneous.analysis.threshold_persistence_frames
    if post_event_steps > homogeneous.steps:
        raise ValueError(
            f"{config_path}: post_event_steps={post_event_steps} exceeds the configured "
            f"measurement duration steps={homogeneous.steps}."
        )
    output_root = _repo_path(raw["output_root"])
    if output_root == homogeneous.source_dataset:
        raise ValueError(
            f"{config_path}: output_root and immutable source_dataset both resolve to "
            f"{output_root}; campaign output must never mutate its source."
        )
    return HomogeneousCampaignConfig(
        homogeneous=homogeneous,
        output_root=output_root,
        execution=HomogeneousCampaignExecutionConfig(
            chunk_steps=chunk_steps,
            event_check_interval=event_check_interval,
            online_persistence_frames=online_persistence_frames,
            stop_on_event=stop_on_event,
            post_event_steps=post_event_steps,
            analysis_mode=str(analysis_mode),
            analysis_workers=analysis_workers,
            checkpoint_retention=checkpoint_retention,
        ),
        potential_selection_report=selection_report,
        potential_selection_report_sha256=selection_report_sha256,
        potential_selection_runtime_controls=selection_runtime_controls,
        source_evidence=source_evidence,
        config_path=config_path,
    )
