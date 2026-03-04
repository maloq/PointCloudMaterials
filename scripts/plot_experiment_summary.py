#!/usr/bin/env python3
"""Render publication-style metric sweep plots from an experiment summary JSON."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FormatStrFormatter


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
        default=r"(\d+)(?!.*\d)",
        help=(
            "Regex used to extract the numeric sweep value from each result name. "
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
        default=None,
        help="Optional figure title for the combined plot.",
    )
    parser.add_argument(
        "--no-individual-panels",
        action="store_true",
        help="Skip writing one file per metric and only save the combined figure.",
    )
    return parser.parse_args(argv)


def _paper_rcparams() -> dict[str, Any]:
    return {
        "font.family": "DejaVu Sans",
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.1,
        "axes.labelsize": 12,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "grid.linewidth": 0.8,
        "grid.alpha": 0.32,
        "grid.color": "#6c7a89",
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }


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
    if "num_points" in plan_name or plan_name.endswith("_points"):
        return "Number of points"
    if "batch_size" in plan_name:
        return "Batch size"
    return "Sweep value"


def _slugify(text: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return out or "plot"


def _parse_numeric_text(text: str, *, context: str) -> float:
    stripped = text.strip()
    if not stripped:
        raise ValueError(f"Expected numeric text for {context}, got an empty string.")
    try:
        value = float(stripped)
    except ValueError as exc:
        raise ValueError(
            f"Failed to parse numeric value {stripped!r} for {context}."
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"Non-finite numeric value {value!r} for {context}.")
    return value


def _extract_x_value(result_name: str, pattern: str) -> float:
    if not isinstance(result_name, str) or not result_name:
        raise ValueError(
            f"Each result requires a non-empty string 'name', got {result_name!r}."
        )
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"Invalid --x-pattern regex {pattern!r}: {exc}") from exc
    match = regex.search(result_name)
    if match is None:
        raise ValueError(
            "Failed to extract x-axis value from result name "
            f"{result_name!r} using pattern {pattern!r}. "
            "Pass a different --x-pattern if the sweep value is encoded differently."
        )
    token = match.group(1) if match.lastindex else match.group(0)
    return _parse_numeric_text(token, context=f"result name {result_name!r}")


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
    if value is None:
        raise ValueError(
            f"Metric {metric_key!r} is missing for result {result_name!r}."
        )
    if isinstance(value, bool):
        raise TypeError(
            f"Metric {metric_key!r} for result {result_name!r} must be numeric, got bool."
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Metric {metric_key!r} for result {result_name!r} must be numeric, got {value!r}."
        ) from exc
    if not math.isfinite(numeric):
        raise ValueError(
            f"Metric {metric_key!r} for result {result_name!r} is not finite: {numeric!r}."
        )
    return numeric


def _collect_runs(
    summary: dict[str, Any],
    *,
    metric_names: Sequence[str],
    metric_scope: str,
    x_pattern: str,
) -> tuple[list[SweepRun], list[str]]:
    results = summary["results"]
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
        x_value = _extract_x_value(result_name, x_pattern)
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
        metric_values = {
            metric_key: _validate_metric_value(
                final_metrics.get(metric_key),
                metric_key=metric_key,
                result_name=result_name,
            )
            for metric_key in resolved_keys
        }
        runs.append(SweepRun(name=result_name, x_value=x_value, metrics=metric_values))
        x_seen[x_value] = result_name

    runs.sort(key=lambda run: run.x_value)
    assert metric_key_order is not None
    return runs, metric_key_order


def _format_x_tick(value: float) -> str:
    rounded = round(value)
    if math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-9):
        return str(int(rounded))
    return f"{value:g}"


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


def _compute_shared_y_limits(
    runs: Sequence[SweepRun],
    metric_keys: Sequence[str],
) -> tuple[float, float]:
    values = np.asarray(
        [run.metrics[key] for run in runs for key in metric_keys],
        dtype=np.float64,
    )
    if values.size == 0:
        raise ValueError("Cannot compute y-axis limits with no metric values.")
    val_min = float(np.min(values))
    val_max = float(np.max(values))
    if 0.0 <= val_min and val_max <= 1.0:
        lower = max(0.0, math.floor((val_min - 0.03) / 0.05) * 0.05)
        upper = min(1.0, math.ceil((val_max + 0.02) / 0.05) * 0.05)
        if upper - lower < 0.2:
            center = 0.5 * (upper + lower)
            half_span = 0.1
            lower = max(0.0, center - half_span)
            upper = min(1.0, center + half_span)
        return lower, upper
    span = max(val_max - val_min, 1e-6)
    padding = 0.12 * span
    return val_min - padding, val_max + padding


def _compute_y_ticks(y_min: float, y_max: float) -> np.ndarray:
    span = y_max - y_min
    if span <= 0:
        raise ValueError(f"Invalid y-axis limits: y_min={y_min}, y_max={y_max}.")
    if 0.0 <= y_min and y_max <= 1.0:
        step = 0.05 if span <= 0.3 else 0.1
        start = math.ceil(y_min / step) * step
        stop = math.floor(y_max / step) * step
        ticks = np.arange(start, stop + 0.5 * step, step, dtype=np.float64)
        if ticks.size >= 3:
            return ticks
    return np.linspace(y_min, y_max, num=5, dtype=np.float64)


def _apply_axis_formatting(
    ax: plt.Axes,
    *,
    x_values: np.ndarray,
    x_label: str,
    y_label: str | None,
    y_limits: tuple[float, float],
    y_ticks: np.ndarray,
) -> None:
    x_span = float(np.ptp(x_values))
    x_margin = 0.04 * x_span if x_span > 0.0 else 1.0
    ax.set_xlim(
        float(np.min(x_values)) - x_margin,
        float(np.max(x_values)) + x_margin,
    )
    ax.set_ylim(*y_limits)
    ax.xaxis.set_major_locator(FixedLocator(x_values))
    ax.set_xticklabels([_format_x_tick(v) for v in x_values])
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)


def _plot_metric_series(
    ax: plt.Axes,
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    style: MetricStyle,
    panel_label: str | None,
) -> None:
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
    ax.set_title(style.label, pad=8)


def _save_figure(fig: plt.Figure, path_without_suffix: Path, formats: Sequence[str], dpi: int) -> list[Path]:
    written: list[Path] = []
    for fmt in formats:
        normalized = fmt.lower().lstrip(".")
        if normalized not in {"png", "pdf", "svg"}:
            raise ValueError(
                f"Unsupported output format {fmt!r}. Supported formats: png, pdf, svg."
            )
        out_path = path_without_suffix.with_suffix(f".{normalized}")
        save_kwargs: dict[str, Any] = {}
        if normalized == "png":
            save_kwargs["dpi"] = int(dpi)
        fig.savefig(out_path, **save_kwargs)
        written.append(out_path)
    return written


def _plot_combined_figure(
    runs: Sequence[SweepRun],
    metric_keys: Sequence[str],
    *,
    x_label: str,
    title: str | None,
    formats: Sequence[str],
    dpi: int,
    output_prefix: Path,
) -> list[Path]:
    x_values = np.asarray([run.x_value for run in runs], dtype=np.float64)
    y_limits = _compute_shared_y_limits(runs, metric_keys)
    y_ticks = _compute_y_ticks(*y_limits)

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
        style = _metric_style(metric_key)
        _plot_metric_series(
            ax,
            x_values=x_values,
            y_values=y_values,
            style=style,
            panel_label=f"({chr(ord('a') + idx)})",
        )
        _apply_axis_formatting(
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
    return _save_figure(fig, output_prefix, formats=formats, dpi=dpi)


def _plot_individual_figures(
    runs: Sequence[SweepRun],
    metric_keys: Sequence[str],
    *,
    x_label: str,
    formats: Sequence[str],
    dpi: int,
    output_dir: Path,
) -> list[Path]:
    x_values = np.asarray([run.x_value for run in runs], dtype=np.float64)
    all_written: list[Path] = []
    for metric_key in metric_keys:
        y_values = np.asarray([run.metrics[metric_key] for run in runs], dtype=np.float64)
        y_limits = _compute_shared_y_limits(
            runs=[SweepRun(name=run.name, x_value=run.x_value, metrics={metric_key: run.metrics[metric_key]}) for run in runs],
            metric_keys=[metric_key],
        )
        y_ticks = _compute_y_ticks(*y_limits)
        style = _metric_style(metric_key)

        fig, ax = plt.subplots(figsize=(4.5, 3.8), constrained_layout=True)
        _plot_metric_series(
            ax,
            x_values=x_values,
            y_values=y_values,
            style=style,
            panel_label=None,
        )
        _apply_axis_formatting(
            ax,
            x_values=x_values,
            x_label=x_label,
            y_label="Score",
            y_limits=y_limits,
            y_ticks=y_ticks,
        )
        out_prefix = output_dir / f"{_slugify(_metric_short_name(metric_key))}_vs_{_slugify(x_label)}"
        all_written.extend(_save_figure(fig, out_prefix, formats=formats, dpi=dpi))
        plt.close(fig)
    return all_written


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary_path = args.summary_path.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else summary_path.parent / "summary_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_summary(summary_path)
    x_label = args.x_label or _infer_x_label(summary)
    runs, metric_keys = _collect_runs(
        summary,
        metric_names=args.metrics,
        metric_scope=args.metric_scope.strip(),
        x_pattern=args.x_pattern,
    )

    inferred_prefix = f"metrics_vs_{_slugify(x_label)}"
    combined_prefix = output_dir / (args.prefix or inferred_prefix)

    with plt.rc_context(_paper_rcparams()):
        written_paths = _plot_combined_figure(
            runs,
            metric_keys,
            x_label=x_label,
            title=args.title,
            formats=args.formats,
            dpi=args.dpi,
            output_prefix=combined_prefix,
        )
        plt.close("all")
        if not args.no_individual_panels:
            written_paths.extend(
                _plot_individual_figures(
                    runs,
                    metric_keys,
                    x_label=x_label,
                    formats=args.formats,
                    dpi=args.dpi,
                    output_dir=output_dir,
                )
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
