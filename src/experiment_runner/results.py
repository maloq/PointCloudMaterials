"""Result collection, aggregation, and reporting."""

from __future__ import annotations

import csv
import json
import re
import statistics
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from dataclasses import dataclass

from .plan import ExperimentPlan, MetricBestSpec, MetricsConfig

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


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


def _find_slurm_logs(run_dir: Path, plan_name: str = "") -> List[Path]:
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

    exp_name = run_dir.name
    stage_name = run_dir.parent.name

    # The sbatch job name is "{plan_name}_{stage}_{experiment}" (see slurm.py).
    # The SLURM log file is "{job_name}_{jobid}.out".
    # We need the full prefix to avoid "pointnet" matching "vn_pointnet".
    if plan_name:
        prefix = f"{plan_name}_{stage_name}_{exp_name}_"
    else:
        prefix = f"{stage_name}_{exp_name}_"

    matches: List[Path] = []
    for f in slurm_dir.iterdir():
        if f.suffix != ".out":
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
    log_candidates.extend(_find_slurm_logs(run_dir, plan_name=plan_name))

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


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    name: str
    stage: str
    run_dir: str
    status: str
    final_metrics: Dict[str, Optional[float]]
    best_metrics: Dict[str, Optional[Dict[str, Any]]]


def collect_all_results(
    output_dir: Path,
    plan: ExperimentPlan,
) -> List[ExperimentResult]:
    """Walk the output directory and collect metrics for every experiment."""
    all_metric_names = list(plan.metrics.final)
    for spec in plan.metrics.best:
        if spec.name not in all_metric_names:
            all_metric_names.append(spec.name)

    results: List[ExperimentResult] = []
    for stage in plan.stages:
        for exp in stage.experiments:
            run_dir = output_dir / stage.name / exp.name
            if not run_dir.exists():
                results.append(ExperimentResult(
                    name=exp.name, stage=stage.name,
                    run_dir=str(run_dir), status="missing",
                    final_metrics={m: None for m in plan.metrics.final},
                    best_metrics={},
                ))
                continue

            final = collect_metrics_for_run(run_dir, plan.metrics.final, plan_name=plan.name)
            best: Dict[str, Optional[Dict[str, Any]]] = {}
            for spec in plan.metrics.best:
                val = collect_metrics_for_run(run_dir, [spec.name], plan_name=plan.name).get(spec.name)
                if val is not None:
                    best[spec.name] = {"value": val}
                else:
                    best[spec.name] = None

            has_any = any(v is not None for v in final.values())
            status = "completed" if has_any else "no_metrics"

            results.append(ExperimentResult(
                name=exp.name, stage=stage.name,
                run_dir=str(run_dir), status=status,
                final_metrics=final, best_metrics=best,
            ))

    return results


def write_csv_tables(
    results: List[ExperimentResult],
    output_dir: Path,
    metrics_cfg: MetricsConfig,
) -> List[Path]:
    """Write aggregated CSV tables. Returns paths of files written."""
    written: List[Path] = []

    # --- Final metrics table ---
    final_path = output_dir / "final_metrics.csv"
    rows: List[Dict[str, Any]] = []
    for r in results:
        row: Dict[str, Any] = {"experiment": r.name, "stage": r.stage, "status": r.status}
        for m in metrics_cfg.final:
            row[_sanitize_col(m)] = fmt_metric(r.final_metrics.get(m))
        rows.append(row)

    # Add mean / std summary rows.
    for stat_name, stat_fn in [("Mean", statistics.mean), ("Std", _safe_stdev)]:
        summary: Dict[str, Any] = {"experiment": stat_name, "stage": "", "status": ""}
        for m in metrics_cfg.final:
            vals = [r.final_metrics[m] for r in results if r.final_metrics.get(m) is not None]
            if vals:
                summary[_sanitize_col(m)] = fmt_metric(stat_fn(vals))
            else:
                summary[_sanitize_col(m)] = "-"
        rows.append(summary)

    _write_csv_file(final_path, rows)
    written.append(final_path)

    # --- Best metrics tables (one per spec) ---
    for spec in metrics_cfg.best:
        col = _sanitize_col(spec.name)
        best_path = output_dir / f"best_{col}.csv"
        best_rows: List[Dict[str, Any]] = []
        vals_for_stats: List[float] = []
        for r in results:
            entry = r.best_metrics.get(spec.name)
            val = entry["value"] if entry else None
            if val is not None:
                vals_for_stats.append(float(val))
            best_rows.append({
                "experiment": r.name,
                "stage": r.stage,
                f"{col}_value": fmt_metric(val),
            })
        for stat_name, stat_fn in [("Mean", statistics.mean), ("Std", _safe_stdev)]:
            best_rows.append({
                "experiment": stat_name,
                "stage": "",
                f"{col}_value": fmt_metric(stat_fn(vals_for_stats)) if vals_for_stats else "-",
            })
        _write_csv_file(best_path, best_rows)
        written.append(best_path)

    return written


def write_summary_json(
    results: List[ExperimentResult],
    output_dir: Path,
    plan: ExperimentPlan,
) -> Path:
    """Write a JSON summary of all results."""
    path = output_dir / "summary.json"
    payload = {
        "plan_name": plan.name,
        "generated_at": datetime.now().isoformat(),
        "results": [
            {
                "name": r.name,
                "stage": r.stage,
                "run_dir": r.run_dir,
                "status": r.status,
                "final_metrics": {
                    k: v for k, v in r.final_metrics.items()
                },
                "best_metrics": r.best_metrics,
            }
            for r in results
        ],
    }
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
    for r in results:
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

    # Summary rows.
    for stat_label, stat_fn in [("Mean", statistics.mean), ("Std", _safe_stdev)]:
        row = {"experiment": stat_label, "stage": "", "status": ""}
        for m in metric_names:
            vals = [r.final_metrics[m] for r in results if r.final_metrics.get(m) is not None]
            row[_sanitize_col(m)] = fmt_metric(stat_fn(vals)) if vals else "-"
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _write_csv_file(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.touch()
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
