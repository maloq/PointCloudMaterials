"""Result collection, aggregation, and reporting."""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from dataclasses import dataclass

from .plan import ExperimentPlan, MetricBestSpec, MetricsConfig

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
FAILURE_LOG_PATTERNS = (
    re.compile(r"Traceback \(most recent call last\):"),
    re.compile(r"^Could not override ", re.MULTILINE),
    re.compile(r"Error executing job with overrides", re.MULTILINE),
    re.compile(r": error: ", re.MULTILINE),
)
SUCCESS_JOB_STATUSES = {"completed", "dry_run", "COMPLETED"}


# ---------------------------------------------------------------------------
# Metric discovery from run directories
# ---------------------------------------------------------------------------

def find_best_checkpoint(run_dir: Path) -> Optional[Path]:
    """Find the best checkpoint in a run directory (Lightning convention)."""
    candidates: List[Path] = []

    # Standard Lightning checkpoints/ subfolder.
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.exists():
        candidates.extend(ckpt_dir.glob("*.ckpt"))

    # Checkpoints saved directly in the run dir.
    candidates.extend(run_dir.glob("*.ckpt"))

    if not candidates:
        return None

    # Prefer files that are NOT "last.ckpt".
    best = [c for c in candidates if "last" not in c.stem.lower()]
    if not best:
        best = candidates

    # Sort by name (Lightning embeds metric values in filenames) then pick first.
    return sorted(best, key=lambda p: p.name)[0]


def _parse_metric_from_log(log_path: Path, metric_name: str) -> Optional[float]:
    """Parse the last occurrence of a metric from a training log file."""
    if not log_path.exists():
        return None
    text = ANSI_RE.sub("", log_path.read_text(errors="replace"))
    pattern = re.compile(
        rf"{re.escape(metric_name)}(?:\s*[:=]\s*|\s+)([+-]?\d+(?:\.\d*)?(?:e[+-]?\d+)?)",
        re.IGNORECASE,
    )
    matches = pattern.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def _read_checkpoint_metric(run_dir: Path, metric_name: str) -> Optional[float]:
    """Read a monitored metric value from checkpoint callback state."""
    try:
        import torch
    except ImportError:
        return None

    ckpt_paths = sorted(run_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    for ckpt_path in ckpt_paths:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        callbacks = ckpt.get("callbacks")
        if not isinstance(callbacks, dict):
            continue
        for _cb_name, state in callbacks.items():
            if not isinstance(state, dict):
                continue
            monitor = str(state.get("monitor", "")).strip()
            if monitor != metric_name:
                continue
            score = state.get("best_model_score") or state.get("current_score")
            if score is None:
                continue
            if hasattr(score, "item"):
                return float(score.item())
            return float(score)
    return None


def _read_metrics_json(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Read a final_metrics.json if present (from run_ablation.py style runs)."""
    path = run_dir / "final_metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def _find_slurm_logs(
    run_dir: Path,
    plan_name: str = "",
    *,
    job_id: str | None = None,
    suffixes: Sequence[str] = (".out",),
) -> List[Path]:
    """Locate SLURM .out log files that belong to this run directory.

    The runner places SLURM logs in ``<output_dir>/slurm_logs/`` with names
    ``{plan_name}_{stage}_{experiment}_{jobid}.out``.  From a run dir like
    ``<output_dir>/<stage>/<experiment>/`` we walk up to find the
    ``slurm_logs/`` sibling and match on the exact job-name prefix.
    """
    # run_dir is <output_dir>/<stage>/<exp_name>
    output_dir = run_dir.parent.parent
    slurm_dir = output_dir / "slurm_logs"
    if not slurm_dir.is_dir():
        return []

    matches: List[Path] = []
    if job_id is not None:
        for suffix in suffixes:
            matches.extend(slurm_dir.glob(f"*_{job_id}{suffix}"))
        if matches:
            return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)

    exp_name = run_dir.name
    stage_name = run_dir.parent.name

    # The sbatch job name is "{plan_name}_{stage}_{experiment}" (see slurm.py).
    # The SLURM log file is "{job_name}_{jobid}.out".
    # We need the full prefix to avoid "pointnet" matching "vn_pointnet".
    if plan_name:
        prefix = f"{plan_name}_{stage_name}_{exp_name}_"
    else:
        prefix = f"{stage_name}_{exp_name}_"

    for f in slurm_dir.iterdir():
        if f.suffix not in suffixes:
            continue
        # With plan_name: exact prefix match (e.g. "vicreg_encoders_default_pointnet_")
        # Without: fall back to substring but require trailing underscore + digits
        if plan_name and f.name.startswith(prefix):
            matches.append(f)
        elif not plan_name and f"_{stage_name}_{exp_name}_" in f.name:
            matches.append(f)

    return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)


def collect_metrics_for_run(
    run_dir: Path,
    metric_names: Sequence[str],
    plan_name: str = "",
    job_id: str | None = None,
) -> Dict[str, Optional[float]]:
    """Collect metric values for a single experiment run directory.

    Tries multiple sources in order:
      1. final_metrics.json (from ablation-style runs)
      2. Log files inside the run dir
      3. SLURM stdout logs (in the sibling slurm_logs/ directory)
      4. Checkpoint callback state
    """
    result: Dict[str, Optional[float]] = {m: None for m in metric_names}

    # Source 1: JSON file from ablation-style runs.
    json_metrics = _read_metrics_json(run_dir)
    if json_metrics:
        for name in metric_names:
            if name in json_metrics and json_metrics[name] is not None:
                try:
                    result[name] = float(json_metrics[name])
                except (TypeError, ValueError):
                    pass

    # Source 2: Parse training logs inside the run directory.
    log_candidates: List[Path] = [
        run_dir / "train.log",
        run_dir / "train_contrastive.log",
        run_dir / "sweep_driver.log",
    ]
    # Source 3: SLURM stdout logs (contain Lightning progress bars + test results).
    log_candidates.extend(
        _find_slurm_logs(run_dir, plan_name=plan_name, job_id=job_id, suffixes=(".out",))
    )

    for name in metric_names:
        if result[name] is not None:
            continue
        for log_path in log_candidates:
            val = _parse_metric_from_log(log_path, name)
            if val is not None:
                result[name] = val
                break

    # Source 4: Checkpoint callback state (only stores monitored metrics).
    for name in metric_names:
        if result[name] is not None:
            continue
        val = _read_checkpoint_metric(run_dir, name)
        if val is not None:
            result[name] = val

    return result


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt_metric(value: Optional[float]) -> str:
    """Format a metric value for display, with smart rounding."""
    if value is None:
        return "-"
    try:
        d = Decimal(str(value))
    except (ArithmeticError, ValueError):
        return str(value)

    if d.is_nan():
        return "nan"
    if d.is_infinite():
        return "inf" if d > 0 else "-inf"
    if d == 0:
        return "0"

    abs_d = abs(d)
    if abs_d >= 1:
        quantize_unit = Decimal("0.001")
    else:
        sig_digits = 2
        exponent = abs_d.adjusted() - sig_digits + 1
        quantize_unit = Decimal(f"1e{exponent}")

    rounded = d.quantize(quantize_unit, rounding=ROUND_HALF_UP)
    if rounded == 0:
        return "0"
    formatted = format(rounded.normalize(), "f")
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted or "0"


def _sanitize_col(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _normalize_saved_status(status: str | None) -> str | None:
    if status is None:
        return None
    cleaned = status.strip()
    if cleaned == "":
        return None
    if cleaned in SUCCESS_JOB_STATUSES:
        return "completed"
    return cleaned.lower().replace(" ", "_")


def _load_saved_job_metadata(output_dir: Path) -> Dict[tuple[str, str], "SavedJobMetadata"]:
    from .state import RunState

    state_path = output_dir / "state.json"
    if not state_path.exists():
        return {}

    state = RunState.load(output_dir)
    saved_jobs: Dict[tuple[str, str], SavedJobMetadata] = {}
    for stage_name, stage_state in state.stages.items():
        for exp_name, job in stage_state.jobs.items():
            saved_jobs[(stage_name, exp_name)] = SavedJobMetadata(
                status=job.status,
                job_id=job.job_id,
            )
    return saved_jobs


def _detect_failed_run_status(
    run_dir: Path,
    *,
    plan_name: str = "",
    job_id: str | None = None,
) -> str | None:
    from .slurm import read_job_status_file

    job_status = read_job_status_file(run_dir)
    if job_status is not None:
        normalized = _normalize_saved_status(job_status["state"])
        if normalized != "completed":
            return normalized

    failure_logs: List[Path] = [
        run_dir / "train.log",
        run_dir / "train_contrastive.log",
        run_dir / "sweep_driver.log",
    ]
    failure_logs.extend(
        _find_slurm_logs(
            run_dir,
            plan_name=plan_name,
            job_id=job_id,
            suffixes=(".err", ".out"),
        )
    )
    for log_path in failure_logs:
        if _log_contains_failure(log_path):
            return "failed"
    return None


def _log_contains_failure(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    text = ANSI_RE.sub("", log_path.read_text(errors="replace"))
    return any(pattern.search(text) for pattern in FAILURE_LOG_PATTERNS)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    name: str
    base_name: str
    stage: str
    run_dir: str
    status: str
    repeat_index: int | None
    final_metrics: Dict[str, Optional[float]]
    best_metrics: Dict[str, Optional[Dict[str, Any]]]


@dataclass(frozen=True)
class SavedJobMetadata:
    status: str | None
    job_id: str | None


@dataclass(frozen=True)
class MetricAggregate:
    count: int
    mean: float | None
    std: float | None
    ci95_low: float | None
    ci95_high: float | None
    ci95_half_width: float | None


@dataclass
class AggregatedExperimentResult:
    name: str
    stage: str
    status: str
    successful_repeats: int
    scheduled_repeats: int
    final_metrics: Dict[str, MetricAggregate]
    best_metrics: Dict[str, MetricAggregate]


def collect_all_results(
    output_dir: Path,
    plan: ExperimentPlan,
) -> List[ExperimentResult]:
    """Walk the output directory and collect metrics for every experiment."""
    all_metric_names = list(plan.metrics.final)
    for spec in plan.metrics.best:
        if spec.name not in all_metric_names:
            all_metric_names.append(spec.name)

    saved_jobs = _load_saved_job_metadata(output_dir)
    results: List[ExperimentResult] = []
    for stage in plan.stages:
        for exp in stage.experiments:
            saved_job = saved_jobs.get((stage.name, exp.name))
            saved_status = _normalize_saved_status(saved_job.status if saved_job else None)
            job_id = saved_job.job_id if saved_job else None
            run_dir = output_dir / stage.name / exp.name
            base_name = exp.repeat_group or exp.name
            if not run_dir.exists():
                results.append(ExperimentResult(
                    name=exp.name,
                    base_name=base_name,
                    stage=stage.name,
                    run_dir=str(run_dir), status="missing",
                    repeat_index=exp.repeat_index,
                    final_metrics={m: None for m in plan.metrics.final},
                    best_metrics={},
                ))
                continue

            final = collect_metrics_for_run(
                run_dir,
                plan.metrics.final,
                plan_name=plan.name,
                job_id=job_id,
            )
            best: Dict[str, Optional[Dict[str, Any]]] = {}
            for spec in plan.metrics.best:
                val = collect_metrics_for_run(
                    run_dir,
                    [spec.name],
                    plan_name=plan.name,
                    job_id=job_id,
                ).get(spec.name)
                if val is not None:
                    best[spec.name] = {"value": val}
                else:
                    best[spec.name] = None

            has_any = any(v is not None for v in final.values())
            detected_failure = _detect_failed_run_status(
                run_dir,
                plan_name=plan.name,
                job_id=job_id,
            )
            if has_any:
                status = "completed"
            elif detected_failure is not None:
                status = detected_failure
            elif saved_status is not None and saved_status != "completed":
                status = saved_status
            else:
                status = "no_metrics"

            results.append(ExperimentResult(
                name=exp.name,
                base_name=base_name,
                stage=stage.name,
                run_dir=str(run_dir), status=status,
                repeat_index=exp.repeat_index,
                final_metrics=final, best_metrics=best,
            ))

    return results


_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.16,
    14: 2.145,
    15: 2.131,
    16: 2.12,
    17: 2.11,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.08,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.06,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def has_repeated_experiments(results: Sequence[ExperimentResult]) -> bool:
    return any(result.repeat_index is not None for result in results)


def aggregate_repeated_results(
    results: Sequence[ExperimentResult],
    plan: ExperimentPlan,
) -> List[AggregatedExperimentResult]:
    grouped: Dict[tuple[str, str], List[ExperimentResult]] = defaultdict(list)
    for result in results:
        grouped[(result.stage, result.base_name)].append(result)

    aggregated_results: List[AggregatedExperimentResult] = []
    for (stage_name, base_name), group_results in sorted(grouped.items()):
        successful_runs = [result for result in group_results if result.status == "completed"]
        final_metrics = {
            metric_name: _aggregate_values(
                [
                    float(result.final_metrics[metric_name])
                    for result in successful_runs
                    if result.final_metrics.get(metric_name) is not None
                ]
            )
            for metric_name in plan.metrics.final
        }
        best_metrics = {
            spec.name: _aggregate_values(
                [
                    float(result.best_metrics[spec.name]["value"])
                    for result in successful_runs
                    if result.best_metrics.get(spec.name) is not None
                    and result.best_metrics[spec.name].get("value") is not None
                ]
            )
            for spec in plan.metrics.best
        }
        aggregated_results.append(
            AggregatedExperimentResult(
                name=base_name,
                stage=stage_name,
                status=_aggregate_group_status(group_results),
                successful_repeats=len(successful_runs),
                scheduled_repeats=len(group_results),
                final_metrics=final_metrics,
                best_metrics=best_metrics,
            )
        )

    return aggregated_results


def _aggregate_group_status(group_results: Sequence[ExperimentResult]) -> str:
    statuses = [result.status for result in group_results]
    completed_count = sum(1 for status in statuses if status == "completed")
    if completed_count == len(statuses):
        return "completed"
    if completed_count > 0:
        return "partial_failures"

    distinct = {status for status in statuses}
    if len(distinct) == 1:
        return statuses[0]
    return "failed"


def _aggregate_values(values: Sequence[float]) -> MetricAggregate:
    if not values:
        return MetricAggregate(
            count=0,
            mean=None,
            std=None,
            ci95_low=None,
            ci95_high=None,
            ci95_half_width=None,
        )

    mean_value = statistics.mean(values)
    std_value = _safe_stdev(values)
    half_width = _ci95_half_width(values)
    return MetricAggregate(
        count=len(values),
        mean=mean_value,
        std=std_value,
        ci95_low=mean_value - half_width,
        ci95_high=mean_value + half_width,
        ci95_half_width=half_width,
    )


def _ci95_half_width(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    critical = _t_critical_95(len(values))
    return critical * statistics.stdev(values) / (len(values) ** 0.5)


def _t_critical_95(sample_size: int) -> float:
    if sample_size < 2:
        return 0.0
    degrees_of_freedom = sample_size - 1
    return _T_CRITICAL_95.get(degrees_of_freedom, 1.96)


def write_csv_tables(
    results: List[ExperimentResult],
    output_dir: Path,
    metrics_cfg: MetricsConfig,
) -> List[Path]:
    """Write aggregated CSV tables. Returns paths of files written."""
    written: List[Path] = []
    ordered_groups = _ordered_result_groups(results)

    # --- Final metrics table ---
    final_path = output_dir / "final_metrics.csv"
    rows: List[Dict[str, Any]] = []
    for group_results in ordered_groups:
        for r in group_results:
            row: Dict[str, Any] = {"experiment": r.name, "stage": r.stage, "status": r.status}
            for m in metrics_cfg.final:
                row[_sanitize_col(m)] = fmt_metric(r.final_metrics.get(m))
            rows.append(row)
        rows.extend(_build_group_metric_summary_rows(group_results, metrics_cfg.final))

    # Add global mean +/- std summary row.
    summary: Dict[str, Any] = {"experiment": "global__mean_std", "stage": "", "status": ""}
    for m in metrics_cfg.final:
        vals = [r.final_metrics[m] for r in results if r.final_metrics.get(m) is not None]
        summary[_sanitize_col(m)] = _format_values_mean_std(vals)
    rows.append(summary)

    _write_csv_file(final_path, rows)
    written.append(final_path)

    # --- Best metrics tables (one per spec) ---
    for spec in metrics_cfg.best:
        col = _sanitize_col(spec.name)
        best_path = output_dir / f"best_{col}.csv"
        best_rows: List[Dict[str, Any]] = []
        vals_for_stats: List[float] = []
        for group_results in ordered_groups:
            for r in group_results:
                entry = r.best_metrics.get(spec.name)
                val = entry["value"] if entry else None
                if val is not None:
                    vals_for_stats.append(float(val))
                best_rows.append({
                    "experiment": r.name,
                    "stage": r.stage,
                    f"{col}_value": fmt_metric(val),
                })
            best_rows.extend(_build_group_best_summary_rows(group_results, spec.name, col))
        best_rows.append({
            "experiment": "global__mean_std",
            "stage": "",
            f"{col}_value": _format_values_mean_std(vals_for_stats),
        })
        _write_csv_file(best_path, best_rows)
        written.append(best_path)

    return written


def write_grouped_csv_tables(
    aggregated_results: Sequence[AggregatedExperimentResult],
    output_dir: Path,
    metrics_cfg: MetricsConfig,
) -> List[Path]:
    """Write grouped repeat summaries with paper-style mean +/- std cells."""
    if not aggregated_results:
        return []

    written: List[Path] = []

    final_path = output_dir / "final_metrics_grouped.csv"
    final_rows: List[Dict[str, Any]] = []
    for result in aggregated_results:
        row: Dict[str, Any] = {
            "experiment": result.name,
            "stage": result.stage,
            "status": result.status,
            "successful_repeats": result.successful_repeats,
            "scheduled_repeats": result.scheduled_repeats,
        }
        for metric_name in metrics_cfg.final:
            stats = result.final_metrics[metric_name]
            col = _sanitize_col(metric_name)
            row[col] = _format_aggregate_mean_std(stats)
        final_rows.append(row)
    _write_csv_file(final_path, final_rows)
    written.append(final_path)

    for spec in metrics_cfg.best:
        col = _sanitize_col(spec.name)
        best_path = output_dir / f"best_{col}_grouped.csv"
        best_rows: List[Dict[str, Any]] = []
        for result in aggregated_results:
            stats = result.best_metrics[spec.name]
            best_rows.append(
                {
                    "experiment": result.name,
                    "stage": result.stage,
                    "status": result.status,
                    "successful_repeats": result.successful_repeats,
                    "scheduled_repeats": result.scheduled_repeats,
                    col: _format_aggregate_mean_std(stats),
                }
            )
        _write_csv_file(best_path, best_rows)
        written.append(best_path)

    return written


def write_summary_json(
    results: List[ExperimentResult],
    output_dir: Path,
    plan: ExperimentPlan,
    aggregated_results: Sequence[AggregatedExperimentResult] | None = None,
) -> Path:
    """Write a JSON summary of all results."""
    path = output_dir / "summary.json"
    payload = {
        "plan_name": plan.name,
        "generated_at": datetime.now().isoformat(),
        "repeat_each": plan.repeat_each,
        "results": [
            {
                "name": r.name,
                "base_name": r.base_name,
                "stage": r.stage,
                "run_dir": r.run_dir,
                "status": r.status,
                "repeat_index": r.repeat_index,
                "final_metrics": {
                    k: v for k, v in r.final_metrics.items()
                },
                "best_metrics": r.best_metrics,
            }
            for r in results
        ],
    }
    if aggregated_results:
        payload["grouped_results"] = [
            {
                "name": result.name,
                "stage": result.stage,
                "status": result.status,
                "successful_repeats": result.successful_repeats,
                "scheduled_repeats": result.scheduled_repeats,
                "final_metrics": {
                    metric_name: _metric_aggregate_to_dict(stats)
                    for metric_name, stats in result.final_metrics.items()
                },
                "best_metrics": {
                    metric_name: _metric_aggregate_to_dict(stats)
                    for metric_name, stats in result.best_metrics.items()
                },
            }
            for result in aggregated_results
        ]
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def print_summary_table(
    results: List[ExperimentResult],
    metrics_cfg: MetricsConfig,
) -> None:
    """Print a human-readable summary table to stdout."""
    if not results:
        print("No results to display.")
        return

    metric_names = list(metrics_cfg.final)
    # Build columns: experiment | stage | status | metric1 | metric2 ...
    headers = ["experiment", "stage", "status"] + [_sanitize_col(m) for m in metric_names]
    col_widths = {h: len(h) for h in headers}

    rows_data: List[Dict[str, str]] = []
    for group_results in _ordered_result_groups(results):
        for r in group_results:
            row = {
                "experiment": r.name,
                "stage": r.stage,
                "status": r.status,
            }
            for m in metric_names:
                row[_sanitize_col(m)] = fmt_metric(r.final_metrics.get(m))
            rows_data.append(row)
            for h in headers:
                col_widths[h] = max(col_widths[h], len(row.get(h, "")))
        for row in _build_group_metric_summary_rows(group_results, metric_names):
            rows_data.append(row)
            for h in headers:
                col_widths[h] = max(col_widths[h], len(row.get(h, "")))

    # Global summary row.
    row = {"experiment": "global__mean_std", "stage": "", "status": ""}
    for m in metric_names:
        vals = [r.final_metrics[m] for r in results if r.final_metrics.get(m) is not None]
        row[_sanitize_col(m)] = _format_values_mean_std(vals)
    rows_data.append(row)
    for h in headers:
        col_widths[h] = max(col_widths[h], len(row.get(h, "")))

    sep = "+" + "+".join("-" * (col_widths[h] + 2) for h in headers) + "+"
    header_line = "|" + "|".join(f" {h:<{col_widths[h]}} " for h in headers) + "|"

    print("\n" + sep)
    print(header_line)
    print(sep)
    for row in rows_data:
        line = "|" + "|".join(
            f" {row.get(h, ''):<{col_widths[h]}} " for h in headers
        ) + "|"
        print(line)
    print(sep + "\n")


def print_grouped_summary_table(
    aggregated_results: Sequence[AggregatedExperimentResult],
    metrics_cfg: MetricsConfig,
) -> None:
    """Print grouped repeat summaries with mean +/- std."""
    if not aggregated_results:
        return

    metric_names = list(metrics_cfg.final)
    headers = ["experiment", "stage", "status", "repeats"] + [_sanitize_col(m) for m in metric_names]
    col_widths = {h: len(h) for h in headers}

    rows_data: List[Dict[str, str]] = []
    for result in aggregated_results:
        row = {
            "experiment": result.name,
            "stage": result.stage,
            "status": result.status,
            "repeats": f"{result.successful_repeats}/{result.scheduled_repeats}",
        }
        for metric_name in metric_names:
            row[_sanitize_col(metric_name)] = _format_aggregate_for_table(
                result.final_metrics[metric_name]
            )
        rows_data.append(row)
        for header in headers:
            col_widths[header] = max(col_widths[header], len(row.get(header, "")))

    sep = "+" + "+".join("-" * (col_widths[h] + 2) for h in headers) + "+"
    header_line = "|" + "|".join(f" {h:<{col_widths[h]}} " for h in headers) + "|"

    print("\nGrouped repeat summary (mean +/- std):")
    print(sep)
    print(header_line)
    print(sep)
    for row in rows_data:
        line = "|" + "|".join(
            f" {row.get(h, ''):<{col_widths[h]}} " for h in headers
        ) + "|"
        print(line)
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_aggregate_for_table(stats: MetricAggregate) -> str:
    return _format_aggregate_mean_std(stats)


def _metric_aggregate_to_dict(stats: MetricAggregate) -> Dict[str, float | int | None]:
    return {
        "count": stats.count,
        "mean": stats.mean,
        "std": stats.std,
        "ci95_low": stats.ci95_low,
        "ci95_high": stats.ci95_high,
        "ci95_half_width": stats.ci95_half_width,
    }


def _safe_stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _ordered_result_groups(
    results: Sequence[ExperimentResult],
) -> List[List[ExperimentResult]]:
    groups: Dict[tuple[str, str], List[ExperimentResult]] = {}
    for result in results:
        groups.setdefault((result.stage, result.base_name), []).append(result)
    return list(groups.values())


def _has_group_summary_rows(group_results: Sequence[ExperimentResult]) -> bool:
    return len(group_results) > 1 or any(result.repeat_index is not None for result in group_results)


def _build_group_metric_summary_rows(
    group_results: Sequence[ExperimentResult],
    metric_names: Sequence[str],
) -> List[Dict[str, str]]:
    if not _has_group_summary_rows(group_results):
        return []

    base_name = group_results[0].base_name
    stage_name = group_results[0].stage
    row = {
        "experiment": f"{base_name}__mean_std",
        "stage": stage_name,
        "status": _aggregate_group_status(group_results),
    }
    for metric_name in metric_names:
        values = [
            result.final_metrics[metric_name]
            for result in group_results
            if result.final_metrics.get(metric_name) is not None
        ]
        row[_sanitize_col(metric_name)] = _format_values_mean_std(values)
    return [row]


def _build_group_best_summary_rows(
    group_results: Sequence[ExperimentResult],
    metric_name: str,
    column_name: str,
) -> List[Dict[str, str]]:
    if not _has_group_summary_rows(group_results):
        return []

    base_name = group_results[0].base_name
    stage_name = group_results[0].stage
    values = [
        float(result.best_metrics[metric_name]["value"])
        for result in group_results
        if result.best_metrics.get(metric_name) is not None
        and result.best_metrics[metric_name].get("value") is not None
    ]

    return [{
        "experiment": f"{base_name}__mean_std",
        "stage": stage_name,
        f"{column_name}_value": _format_values_mean_std(values),
    }]


def _format_mean_std(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None:
        return "-"
    if std is None:
        return fmt_metric(mean)
    return f"{fmt_metric(mean)} +/- {fmt_metric(std)}"


def _format_values_mean_std(values: Sequence[float]) -> str:
    if not values:
        return "-"
    return _format_mean_std(statistics.mean(values), _safe_stdev(values))


def _format_aggregate_mean_std(stats: MetricAggregate) -> str:
    return _format_mean_std(stats.mean, stats.std)


def _write_csv_file(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.touch()
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
