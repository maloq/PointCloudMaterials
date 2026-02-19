"""Experiment plan schema: parsing, validation, and matrix expansion."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Schema dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SlurmConfig:
    partition: str = "L40S"
    gpus: int = 1
    cpus: int = 8
    mem: str = "120G"
    time: str = "24:00:00"
    extra_sbatch: List[str] = field(default_factory=list)

    # Environment setup executed at the top of every sbatch script.
    conda_env: str = "pointnet"
    conda_sh: str = "/home/infres/vmorozov/miniconda3/etc/profile.d/conda.sh"


@dataclass
class MetricBestSpec:
    name: str
    mode: str  # "min" or "max"

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError(
                f"MetricBestSpec.mode must be 'min' or 'max', got {self.mode!r} "
                f"for metric {self.name!r}."
            )


@dataclass
class MetricsConfig:
    final: List[str] = field(default_factory=list)
    best: List[MetricBestSpec] = field(default_factory=list)


@dataclass
class Experiment:
    """A single concrete experiment (one training run)."""

    name: str
    overrides: List[str] = field(default_factory=list)
    stage: str = "default"


@dataclass
class StageSpec:
    """One stage in a multi-stage pipeline."""

    name: str
    config_name: Optional[str] = None
    train_script: Optional[str] = None
    base_overrides: List[str] = field(default_factory=list)
    experiments: List[Experiment] = field(default_factory=list)
    depends_on: Optional[str] = None
    inherit_checkpoint: bool = False
    slurm: Optional[SlurmConfig] = None


@dataclass
class ExperimentPlan:
    """Top-level experiment plan parsed from YAML."""

    name: str
    train_script: str
    config_name: Optional[str] = None
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    base_overrides: List[str] = field(default_factory=list)
    stages: List[StageSpec] = field(default_factory=list)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    output_root: str = "output/experiments"

    # Populated after expansion.
    all_experiments: List[Experiment] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_slurm(raw: Dict[str, Any], defaults: Optional[SlurmConfig] = None) -> SlurmConfig:
    base = defaults or SlurmConfig()
    return SlurmConfig(
        partition=raw.get("partition", base.partition),
        gpus=int(raw.get("gpus", base.gpus)),
        cpus=int(raw.get("cpus", base.cpus)),
        mem=str(raw.get("mem", base.mem)),
        time=str(raw.get("time", base.time)),
        extra_sbatch=list(raw.get("extra_sbatch", base.extra_sbatch)),
        conda_env=str(raw.get("conda_env", base.conda_env)),
        conda_sh=str(raw.get("conda_sh", base.conda_sh)),
    )


def _parse_metrics(raw: Dict[str, Any]) -> MetricsConfig:
    final = list(raw.get("final", []))
    best_raw = raw.get("best", [])
    best = []
    for entry in best_raw:
        if not isinstance(entry, dict) or "name" not in entry or "mode" not in entry:
            raise ValueError(
                f"Each metrics.best entry must be a dict with 'name' and 'mode', got {entry!r}."
            )
        best.append(MetricBestSpec(name=entry["name"], mode=entry["mode"]))
    return MetricsConfig(final=final, best=best)


def _parse_experiments(raw_list: Sequence[Dict[str, Any]], stage_name: str) -> List[Experiment]:
    experiments = []
    for entry in raw_list:
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError(
                f"Each experiment must have a 'name' key, got {entry!r} in stage {stage_name!r}."
            )
        experiments.append(Experiment(
            name=entry["name"],
            overrides=list(entry.get("overrides", [])),
            stage=stage_name,
        ))
    return experiments


def _expand_matrix(matrix: Dict[str, List[Any]], stage_name: str) -> List[Experiment]:
    """Expand a matrix dict into a list of Experiments (cartesian product)."""
    if not matrix:
        return []
    keys = sorted(matrix.keys())
    for key in keys:
        values = matrix[key]
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"Matrix key {key!r} must map to a non-empty list, got {values!r}."
            )

    value_lists = [matrix[k] for k in keys]
    experiments = []
    for combo in itertools.product(*value_lists):
        parts = []
        overrides = []
        for key, val in zip(keys, combo):
            overrides.append(f"{key}={_format_override_value(val)}")
            parts.append(f"{_short_key(key)}={_format_name_value(val)}")
        name = "__".join(parts)
        experiments.append(Experiment(name=name, overrides=overrides, stage=stage_name))
    return experiments


def _format_override_value(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, float):
        return f"{val:.10g}"
    return str(val)


def _format_name_value(val: Any) -> str:
    if isinstance(val, bool):
        return "T" if val else "F"
    if isinstance(val, float):
        return f"{val:.6g}".replace(".", "p").replace("-", "m")
    return str(val)


def _short_key(key: str) -> str:
    """Shorten a dotted key for use in experiment names."""
    parts = key.split(".")
    return parts[-1]


# ---------------------------------------------------------------------------
# Main loading / validation
# ---------------------------------------------------------------------------

def load_plan(path: Path) -> ExperimentPlan:
    """Load and validate an experiment plan from a YAML file."""
    if not path.exists():
        raise FileNotFoundError(f"Plan file not found: {path}")

    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"Plan file must be a YAML mapping, got {type(raw).__name__}.")

    name = raw.get("name")
    if not name:
        raise ValueError("Plan must have a 'name' field.")

    train_script = raw.get("train_script")
    if not train_script:
        raise ValueError("Plan must have a 'train_script' field.")

    config_name = raw.get("config_name")
    slurm_cfg = _parse_slurm(raw.get("slurm", {}))
    base_overrides = list(raw.get("base_overrides", []))
    output_root = raw.get("output_root", "output/experiments")

    metrics_raw = raw.get("metrics", {})
    if not metrics_raw:
        raise ValueError("Plan must have a 'metrics' section with at least 'final' metric names.")
    metrics = _parse_metrics(metrics_raw)
    if not metrics.final:
        raise ValueError("metrics.final must list at least one metric name.")

    plan = ExperimentPlan(
        name=name,
        train_script=train_script,
        config_name=config_name,
        slurm=slurm_cfg,
        base_overrides=base_overrides,
        metrics=metrics,
        output_root=output_root,
    )

    # ---- Parse stages vs flat experiments/matrix ----
    raw_stages = raw.get("stages")
    raw_experiments = raw.get("experiments")
    raw_matrix = raw.get("matrix")

    if raw_stages:
        plan.stages = _parse_stages(raw_stages, plan)
    else:
        # Single implicit stage built from top-level experiments/matrix.
        if config_name is None:
            raise ValueError(
                "Plan must set 'config_name' at the top level when not using 'stages'."
            )
        stage = StageSpec(name="default", config_name=config_name, base_overrides=base_overrides)
        explicit = _parse_experiments(raw_experiments or [], "default")
        from_matrix = _expand_matrix(raw_matrix or {}, "default")
        stage.experiments = _merge_experiments(explicit, from_matrix)
        if not stage.experiments:
            raise ValueError(
                "Plan must define at least one experiment via 'experiments' or 'matrix'."
            )
        plan.stages = [stage]

    _validate_stage_graph(plan.stages)

    plan.all_experiments = []
    for stage in plan.stages:
        plan.all_experiments.extend(stage.experiments)

    return plan


def _parse_stages(
    raw_stages: List[Dict[str, Any]], plan: ExperimentPlan
) -> List[StageSpec]:
    stages = []
    for raw_stage in raw_stages:
        if not isinstance(raw_stage, dict) or "name" not in raw_stage:
            raise ValueError(f"Each stage must have a 'name', got {raw_stage!r}.")

        stage_name = raw_stage["name"]
        slurm_override = raw_stage.get("slurm")
        stage_slurm = _parse_slurm(slurm_override, plan.slurm) if slurm_override else None

        stage = StageSpec(
            name=stage_name,
            config_name=raw_stage.get("config_name", plan.config_name),
            train_script=raw_stage.get("train_script"),
            base_overrides=list(raw_stage.get("base_overrides", plan.base_overrides)),
            depends_on=raw_stage.get("depends_on"),
            inherit_checkpoint=bool(raw_stage.get("inherit_checkpoint", False)),
            slurm=stage_slurm,
        )

        if stage.config_name is None:
            raise ValueError(
                f"Stage {stage_name!r} has no config_name and none is set at the top level."
            )

        explicit = _parse_experiments(raw_stage.get("experiments", []), stage_name)
        from_matrix = _expand_matrix(raw_stage.get("matrix", {}), stage_name)
        stage.experiments = _merge_experiments(explicit, from_matrix)

        if not stage.experiments:
            raise ValueError(f"Stage {stage_name!r} has no experiments after expansion.")
        stages.append(stage)

    return stages


def _merge_experiments(
    explicit: List[Experiment], from_matrix: List[Experiment]
) -> List[Experiment]:
    """Merge explicit experiments with matrix-generated ones.

    If both are present, each explicit experiment is combined with every matrix
    entry (cartesian product of overrides). If only one source is present,
    return that list directly.
    """
    if not explicit and not from_matrix:
        return []
    if not explicit:
        return from_matrix
    if not from_matrix:
        return explicit

    merged = []
    for exp in explicit:
        for mat in from_matrix:
            merged.append(Experiment(
                name=f"{exp.name}__{mat.name}",
                overrides=exp.overrides + mat.overrides,
                stage=exp.stage,
            ))
    return merged


def _validate_stage_graph(stages: List[StageSpec]) -> None:
    """Check that depends_on references valid prior stages and there are no cycles."""
    known: set[str] = set()
    for stage in stages:
        if stage.depends_on is not None:
            if stage.depends_on not in known:
                raise ValueError(
                    f"Stage {stage.name!r} depends on {stage.depends_on!r}, "
                    f"which must appear earlier. Known stages: {sorted(known)}."
                )
            if stage.inherit_checkpoint:
                dep_stage = next(s for s in stages if s.name == stage.depends_on)
                dep_names = {e.name for e in dep_stage.experiments}
                cur_names = {e.name for e in stage.experiments}
                missing = cur_names - dep_names
                if missing:
                    raise ValueError(
                        f"Stage {stage.name!r} has inherit_checkpoint=true but experiments "
                        f"{sorted(missing)} have no matching name in stage "
                        f"{stage.depends_on!r} (available: {sorted(dep_names)})."
                    )
        known.add(stage.name)
