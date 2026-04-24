#!/usr/bin/env python3
"""Render publication-style metric sweep plots from an experiment summary JSON."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    from plotting_common import (
        apply_axis_formatting,
        compute_y_limits_from_bounds,
        compute_y_ticks,
        extract_x_value,
        paper_rcparams,
        parse_finite_float,
        save_figure,
        slugify,
    )
except ModuleNotFoundError as exc:
    if exc.name != "plotting_common":
        raise
    from scripts.plotting_common import (
        apply_axis_formatting,
        compute_y_limits_from_bounds,
        compute_y_ticks,
        extract_x_value,
        paper_rcparams,
        parse_finite_float,
        save_figure,
        slugify,
    )


DEFAULT_METRICS = [
    "acc_kmeans_plusplus_hungarian",
    "nmi",
    "ari",
]


@dataclass(frozen=True)
class SweepRun:
    name: str
    x_value: float
    metrics: dict[str, float]
    metric_std: dict[str, float] = field(default_factory=dict)
    metric_ci95_half_width: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricStyle:
    label: str
    color: str


METRIC_STYLES: dict[str, MetricStyle] = {
    "acc_kmeans_plusplus_hungarian": MetricStyle(
        label="ACC (KMeans++ + Hungarian)",
        color="#1557a6",
    ),
    "nmi": MetricStyle(
        label="NMI",
        color="#c94a36",
    ),
    "ari": MetricStyle(
        label="ARI",
        color="#3f9142",
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create paper-style metric sweep plots from an experiment runner "
            "summary.json file."
        )
    )
    parser.add_argument(
        "summary_path",
        type=Path,
        help="Path to an experiment summary.json file.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help=(
            "Metric names to plot. Bare names are resolved against --metric-scope "
            "or by unique suffix matching."
        ),
    )
    parser.add_argument(
        "--metric-scope",
        default="test/class",
        help=(
            "Prefix used to resolve bare metric names, for example 'test/class'. "
            "Set to an empty string to disable scoped resolution."
        ),
    )
    parser.add_argument(
        "--x-pattern",
        default=None,
        help=(
            "Regex used to extract the numeric sweep value from each result name. "
            "If omitted, a plan-specific default is inferred when possible. "
            "If the regex has a capture group, the first group is used."
        ),
    )
    parser.add_argument(
        "--x-label",
        default=None,
        help="X-axis label. If omitted, a label is inferred from the plan name when possible.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where plot files will be written. Defaults to <summary_dir>/summary_plots.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Filename prefix for the combined figure. Defaults to metrics_vs_<inferred-x>.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf", "svg"],
        help="Output formats to write. Example: --formats png pdf svg",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster DPI used for PNG output.",
    )
    parser.add_argument(
        "--title",
        default='',
        help="Optional figure title for the combined plot.",
    )
    parser.add_argument(
        "--series-source",
        choices=["auto", "raw", "grouped"],
        default="auto",
        help=(
            "Which summary series to plot. 'auto' uses grouped_results when present, "
            "otherwise raw per-run results."
        ),
    )
    parser.add_argument(
        "--error-bars",
        choices=["none", "std", "ci95"],
        default="none",
        help="Error-bar mode. For grouped repeat summaries, choose std or ci95.",
    )
    parser.add_argument(
        "--panel-titles",
        action="store_true",
        help="Show per-panel metric titles. Disabled by default.",
    )
    parser.add_argument(
        "--no-individual-panels",
        action="store_true",
        help="Skip writing one file per metric and only save the combined figure.",
    )
    return parser.parse_args(argv)


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Summary file does not exist: {path}")
    if path.name != "summary.json":
        raise ValueError(
            f"Expected a summary.json file, got {path.name!r} at {path}."
        )
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected {path} to contain a JSON object, got {type(data).__name__}."
        )
    results = data.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError(
            f"Expected non-empty 'results' list in {path}, got {type(results).__name__}."
        )
    return data


def _infer_x_label(summary: dict[str, Any]) -> str:
    plan_name = str(summary.get("plan_name", "")).lower()
    if "dataset_fraction" in plan_name:
        return "Dataset fraction (%)"
    if "model_points" in plan_name:
        return "Model points"
    if "num_points" in plan_name or plan_name.endswith("_points"):
        return "Number of points"
    if "batch_size" in plan_name:
        return "Batch size"
    return "Sweep value"


def _infer_x_pattern(summary: dict[str, Any]) -> str:
    plan_name = str(summary.get("plan_name", "")).lower()
    if "dataset_fraction" in plan_name:
        return r"(\d+)(?=pct)"
    if "model_points" in plan_name:
        return r"mp_(\d+)"
    return r"(\d+)(?!.*\d)"


def _resolve_metric_key(
    requested: str,
    available_keys: set[str],
    metric_scope: str,
) -> str:
    if requested in available_keys:
        return requested
    if "/" not in requested and metric_scope:
        scoped_key = f"{metric_scope.rstrip('/')}/{requested}"
        if scoped_key in available_keys:
            return scoped_key
    if "/" not in requested:
        suffix_matches = sorted(
            key for key in available_keys if key.endswith(f"/{requested}")
        )
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        if len(suffix_matches) > 1:
            raise ValueError(
                f"Metric name {requested!r} is ambiguous; matching keys: {suffix_matches}."
            )
    raise KeyError(
        f"Metric {requested!r} was not found. Available metrics: {sorted(available_keys)}."
    )


def _validate_metric_value(
    value: Any,
    *,
    metric_key: str,
    result_name: str,
) -> float:
    return parse_finite_float(value, context=f"metric {metric_key!r} for result {result_name!r}")


def _validate_error_value(
    value: Any,
    *,
    metric_key: str,
    result_name: str,
    field_name: str,
) -> float:
    if value is None:
        return 0.0
    numeric = parse_finite_float(
        value,
        context=(
            f"aggregate field {field_name!r} for metric {metric_key!r} "
            f"in result {result_name!r}"
        )
    )
    if numeric < 0.0:
        raise ValueError(
            f"Aggregate field {field_name!r} for metric {metric_key!r} in result "
            f"{result_name!r} must be >= 0, got {numeric!r}."
        )
    return numeric


def _select_result_entries(
    summary: dict[str, Any],
    *,
    series_source: str,
) -> tuple[list[dict[str, Any]], bool]:
    raw_results = summary.get("results")
    if not isinstance(raw_results, list) or not raw_results:
        raise ValueError(
            f"Expected non-empty 'results' list in summary, got {type(raw_results).__name__}."
        )
    grouped_results = summary.get("grouped_results")
    has_grouped = isinstance(grouped_results, list) and len(grouped_results) > 0

    if series_source == "raw":
        return raw_results, False
    if series_source == "grouped":
        if not has_grouped:
            raise ValueError(
                "Requested --series-source grouped, but summary.json has no grouped_results."
            )
        return grouped_results, True

    if has_grouped:
        return grouped_results, True
    return raw_results, False


def _extract_grouped_metric_value(
    metric_payload: Any,
    *,
    metric_key: str,
    result_name: str,
) -> tuple[float, float, float]:
    if not isinstance(metric_payload, dict):
        raise TypeError(
            f"Grouped metric {metric_key!r} for result {result_name!r} must be an object, "
            f"got {type(metric_payload).__name__}."
        )
    mean_value = _validate_metric_value(
        metric_payload.get("mean"),
        metric_key=metric_key,
        result_name=result_name,
    )
    std_value = _validate_error_value(
        metric_payload.get("std"),
        metric_key=metric_key,
        result_name=result_name,
        field_name="std",
    )
    ci95_value = _validate_error_value(
        metric_payload.get("ci95_half_width"),
        metric_key=metric_key,
        result_name=result_name,
        field_name="ci95_half_width",
    )
    return mean_value, std_value, ci95_value


def collect_runs(
    summary: dict[str, Any],
    *,
    metric_names: Sequence[str],
    metric_scope: str,
    x_pattern: str,
    series_source: str,
) -> tuple[list[SweepRun], list[str]]:
    if series_source not in {"auto", "raw", "grouped"}:
        raise ValueError(
            f"series_source must be one of ['auto', 'raw', 'grouped'], got {series_source!r}."
        )
    results, use_grouped = _select_result_entries(summary, series_source=series_source)
    incomplete = [
        str(entry.get("name", "<unnamed>"))
        for entry in results
        if entry.get("status") != "completed"
    ]
    if incomplete:
        raise RuntimeError(
            "Refusing to plot an incomplete sweep. Non-completed results: "
            f"{incomplete}."
        )

    metric_key_order: list[str] | None = None
    runs: list[SweepRun] = []
    x_seen: dict[float, str] = {}
    for idx, entry in enumerate(results):
        if not isinstance(entry, dict):
            raise TypeError(
                f"Result entry {idx} must be a JSON object, got {type(entry).__name__}."
            )
        result_name = entry.get("name")
        if not isinstance(result_name, str) or not result_name:
            raise ValueError(
                f"Each result requires a non-empty string 'name', got {result_name!r}."
            )
        x_value = extract_x_value(
            result_name,
            pattern=x_pattern,
            context=f"result name {result_name!r}",
            numeric_direct=False,
        )
        if x_value in x_seen:
            raise ValueError(
                f"Duplicate x-axis value {x_value} extracted from results "
                f"{x_seen[x_value]!r} and {result_name!r}."
            )
        final_metrics = entry.get("final_metrics")
        if not isinstance(final_metrics, dict):
            raise TypeError(
                f"Result {result_name!r} is missing a 'final_metrics' object."
            )
        available_keys = set(final_metrics)
        resolved_keys = [
            _resolve_metric_key(name, available_keys, metric_scope)
            for name in metric_names
        ]
        if metric_key_order is None:
            metric_key_order = resolved_keys
        elif resolved_keys != metric_key_order:
            raise ValueError(
                "Resolved metric keys changed across results. "
                f"Expected {metric_key_order}, got {resolved_keys} for result {result_name!r}."
            )
        metric_values: dict[str, float] = {}
        metric_std: dict[str, float] = {}
        metric_ci95: dict[str, float] = {}
        for metric_key in resolved_keys:
            metric_payload = final_metrics.get(metric_key)
            if use_grouped:
                mean_value, std_value, ci95_value = _extract_grouped_metric_value(
                    metric_payload,
                    metric_key=metric_key,
                    result_name=result_name,
                )
                metric_values[metric_key] = mean_value
                metric_std[metric_key] = std_value
                metric_ci95[metric_key] = ci95_value
            else:
                metric_values[metric_key] = _validate_metric_value(
                    metric_payload,
                    metric_key=metric_key,
                    result_name=result_name,
                )
                metric_std[metric_key] = 0.0
                metric_ci95[metric_key] = 0.0
        runs.append(
            SweepRun(
                name=result_name,
                x_value=x_value,
                metrics=metric_values,
                metric_std=metric_std,
                metric_ci95_half_width=metric_ci95,
            )
        )
        x_seen[x_value] = result_name

    runs.sort(key=lambda run: run.x_value)
    assert metric_key_order is not None
    return runs, metric_key_order


def _metric_short_name(metric_key: str) -> str:
    return metric_key.rsplit("/", 1)[-1]


def _metric_style(metric_key: str) -> MetricStyle:
    short_name = _metric_short_name(metric_key)
    if short_name in METRIC_STYLES:
        return METRIC_STYLES[short_name]
    return MetricStyle(
        label=short_name.replace("_", " ").upper(),
        color="#1557a6",
    )


def _resolve_metric_error(
    run: SweepRun,
    metric_key: str,
    *,
    error_bars: str,
) -> float:
    if error_bars == "none":
        return 0.0
    if error_bars == "std":
        return float(run.metric_std.get(metric_key, 0.0))
    if error_bars == "ci95":
        return float(run.metric_ci95_half_width.get(metric_key, 0.0))
    raise ValueError(
        f"error_bars must be one of ['none', 'std', 'ci95'], got {error_bars!r}."
    )


def _compute_shared_y_limits(
    runs: Sequence[SweepRun],
    metric_keys: Sequence[str],
    *,
    error_bars: str,
) -> tuple[float, float]:
    lower_bounds = np.asarray(
        [
            run.metrics[key] - _resolve_metric_error(run, key, error_bars=error_bars)
            for run in runs
            for key in metric_keys
        ],
        dtype=np.float64,
    )
    upper_bounds = np.asarray(
        [
            run.metrics[key] + _resolve_metric_error(run, key, error_bars=error_bars)
            for run in runs
            for key in metric_keys
        ],
        dtype=np.float64,
    )
    return compute_y_limits_from_bounds(lower_bounds, upper_bounds)


def _plot_metric_series(
    ax: plt.Axes,
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    y_errors: np.ndarray | None,
    style: MetricStyle,
    panel_label: str | None,
    show_panel_title: bool,
) -> None:
    if y_errors is not None and np.any(y_errors > 0.0):
        ax.errorbar(
            x_values,
            y_values,
            yerr=y_errors,
            fmt="none",
            ecolor=style.color,
            elinewidth=1.3,
            capsize=3.5,
            capthick=1.3,
            alpha=0.75,
            zorder=2,
        )
    ax.plot(
        x_values,
        y_values,
        color=style.color,
        linewidth=2.3,
        marker="o",
        markersize=7.0,
        markerfacecolor="white",
        markeredgecolor=style.color,
        markeredgewidth=1.8,
        zorder=3,
    )
    if panel_label:
        ax.text(
            0.02,
            0.98,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            color="#111111",
        )
    if show_panel_title:
        ax.set_title(style.label, pad=8)


def _plot_combined_figure(
    runs: Sequence[SweepRun],
    metric_keys: Sequence[str],
    *,
    x_label: str,
    title: str | None,
    error_bars: str,
    show_panel_titles: bool,
    formats: Sequence[str],
    dpi: int,
    output_prefix: Path,
) -> list[Path]:
    x_values = np.asarray([run.x_value for run in runs], dtype=np.float64)
    y_limits = _compute_shared_y_limits(runs, metric_keys, error_bars=error_bars)
    y_ticks = compute_y_ticks(y_limits)

    fig, axes = plt.subplots(
        1,
        len(metric_keys),
        figsize=(4.1 * len(metric_keys), 3.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(metric_keys) == 1:
        axes = np.asarray([axes])

    for idx, (ax, metric_key) in enumerate(zip(axes, metric_keys, strict=True)):
        y_values = np.asarray([run.metrics[metric_key] for run in runs], dtype=np.float64)
        y_errors = np.asarray(
            [_resolve_metric_error(run, metric_key, error_bars=error_bars) for run in runs],
            dtype=np.float64,
        )
        style = _metric_style(metric_key)
        _plot_metric_series(
            ax,
            x_values=x_values,
            y_values=y_values,
            y_errors=y_errors,
            style=style,
            panel_label=f"({chr(ord('a') + idx)})",
            show_panel_title=show_panel_titles,
        )
        apply_axis_formatting(
            ax,
            x_values=x_values,
            x_label="",
            y_label=None,
            y_limits=y_limits,
            y_ticks=y_ticks,
        )

    fig.supxlabel(x_label, fontsize=12)
    fig.supylabel("Score", fontsize=12)
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    return save_figure(fig, output_prefix, formats=formats, dpi=dpi)


def _plot_individual_figures(
    runs: Sequence[SweepRun],
    metric_keys: Sequence[str],
    *,
    x_label: str,
    error_bars: str,
    show_panel_titles: bool,
    formats: Sequence[str],
    dpi: int,
    output_dir: Path,
) -> list[Path]:
    x_values = np.asarray([run.x_value for run in runs], dtype=np.float64)
    all_written: list[Path] = []
    for metric_key in metric_keys:
        y_values = np.asarray([run.metrics[metric_key] for run in runs], dtype=np.float64)
        y_errors = np.asarray(
            [_resolve_metric_error(run, metric_key, error_bars=error_bars) for run in runs],
            dtype=np.float64,
        )
        y_limits = _compute_shared_y_limits(
            runs=[
                SweepRun(
                    name=run.name,
                    x_value=run.x_value,
                    metrics={metric_key: run.metrics[metric_key]},
                    metric_std={metric_key: run.metric_std.get(metric_key, 0.0)},
                    metric_ci95_half_width={
                        metric_key: run.metric_ci95_half_width.get(metric_key, 0.0)
                    },
                )
                for run in runs
            ],
            metric_keys=[metric_key],
            error_bars=error_bars,
        )
        y_ticks = compute_y_ticks(y_limits)
        style = _metric_style(metric_key)

        fig, ax = plt.subplots(figsize=(4.5, 3.8), constrained_layout=True)
        _plot_metric_series(
            ax,
            x_values=x_values,
            y_values=y_values,
            y_errors=y_errors,
            style=style,
            panel_label=None,
            show_panel_title=show_panel_titles,
        )
        apply_axis_formatting(
            ax,
            x_values=x_values,
            x_label=x_label,
            y_label="Score",
            y_limits=y_limits,
            y_ticks=y_ticks,
        )
        out_prefix = output_dir / f"{slugify(_metric_short_name(metric_key))}_vs_{slugify(x_label)}"
        all_written.extend(save_figure(fig, out_prefix, formats=formats, dpi=dpi))
        plt.close(fig)
    return all_written


def render_summary_plots(
    summary_path: Path,
    *,
    metrics: Sequence[str] | None = None,
    metric_scope: str = "test/class",
    x_pattern: str | None = None,
    x_label: str | None = None,
    output_dir: Path | None = None,
    prefix: str | None = None,
    formats: Sequence[str] = ("png", "pdf", "svg"),
    dpi: int = 300,
    title: str = "",
    series_source: str = "auto",
    error_bars: str = "none",
    show_panel_titles: bool = False,
    individual_panels: bool = True,
) -> list[Path]:
    summary_path = summary_path.resolve()
    resolved_output_dir = (
        output_dir.resolve()
        if output_dir is not None
        else summary_path.parent / "summary_plots"
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_summary(summary_path)
    resolved_x_label = x_label or _infer_x_label(summary)
    resolved_x_pattern = x_pattern or _infer_x_pattern(summary)
    metric_names = list(metrics) if metrics is not None else list(DEFAULT_METRICS)
    runs, metric_keys = collect_runs(
        summary,
        metric_names=metric_names,
        metric_scope=metric_scope.strip(),
        x_pattern=resolved_x_pattern,
        series_source=series_source,
    )

    inferred_prefix = f"metrics_vs_{slugify(resolved_x_label)}"
    combined_prefix = resolved_output_dir / (prefix or inferred_prefix)

    with plt.rc_context(paper_rcparams()):
        written_paths = _plot_combined_figure(
            runs,
            metric_keys,
            x_label=resolved_x_label,
            title=title,
            error_bars=error_bars,
            show_panel_titles=show_panel_titles,
            formats=formats,
            dpi=dpi,
            output_prefix=combined_prefix,
        )
        plt.close("all")
        if individual_panels:
            written_paths.extend(
                _plot_individual_figures(
                    runs,
                    metric_keys,
                    x_label=resolved_x_label,
                    error_bars=error_bars,
                    show_panel_titles=show_panel_titles,
                    formats=formats,
                    dpi=dpi,
                    output_dir=resolved_output_dir,
                )
            )
    return written_paths


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    written_paths = render_summary_plots(
        args.summary_path,
        metrics=args.metrics,
        metric_scope=args.metric_scope,
        x_pattern=args.x_pattern,
        x_label=args.x_label,
        output_dir=args.output_dir,
        prefix=args.prefix,
        formats=args.formats,
        dpi=args.dpi,
        title=args.title,
        series_source=args.series_source,
        error_bars=args.error_bars,
        show_panel_titles=args.panel_titles,
        individual_panels=not args.no_individual_panels,
    )

    for path in written_paths:
        print(path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
