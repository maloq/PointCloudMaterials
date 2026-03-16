#!/usr/bin/env python3
"""Render a publication-style metric sweep plot from a grouped CSV file."""

from __future__ import annotations

import argparse
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
import pandas as pd
from matplotlib.ticker import FixedLocator, FormatStrFormatter


DEFAULT_X_PATTERN = r"(\d+)(?!.*\d)"


@dataclass(frozen=True)
class SweepPoint:
    name: str
    x_value: float
    mean: float
    std: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a paper-style metric sweep plot from a grouped CSV file."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the grouped CSV file.",
    )
    parser.add_argument(
        "--metric-column",
        required=True,
        help="CSV column containing the metric to plot, for example test_class_acc_kmeans_plusplus_hungarian.",
    )
    parser.add_argument(
        "--x-column",
        default="experiment",
        help="CSV column used to define the sweep x-axis.",
    )
    parser.add_argument(
        "--x-pattern",
        default=DEFAULT_X_PATTERN,
        help=(
            "Regex used to extract the numeric x value from the x-column text. "
            "If the regex has a capture group, the first group is used."
        ),
    )
    parser.add_argument(
        "--x-label",
        default="Sweep value",
        help="X-axis label.",
    )
    parser.add_argument(
        "--y-label",
        default=None,
        help="Y-axis label. Defaults to a readable label inferred from --metric-column.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help=(
            "Output path without suffix. Defaults to "
            "<csv_dir>/<metric-column>_vs_<x-label>."
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf"],
        help="Output formats to write. Supported: pdf, png, svg.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster DPI used for PNG output.",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional plot title. Leave empty for no title.",
    )
    parser.add_argument(
        "--line-color",
        default="#1557a6",
        help="Primary line color.",
    )
    return parser.parse_args(argv)


def _paper_rcparams() -> dict[str, Any]:
    return {
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
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


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "plot"


def _parse_float(value: Any, *, context: str) -> float:
    if value is None:
        raise ValueError(f"Missing numeric value for {context}.")
    if isinstance(value, bool):
        raise TypeError(f"Expected a numeric value for {context}, got bool.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Expected a numeric value for {context}, got {value!r}.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"Non-finite numeric value {numeric!r} for {context}.")
    return numeric


def _parse_mean_std_text(value: Any, *, context: str) -> tuple[float, float]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"Expected a numeric metric cell for {context}, got an empty string.")
        match = re.fullmatch(
            r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(?:\+/-|±)\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
            text,
        )
        if match is not None:
            mean = _parse_float(match.group(1), context=f"{context} mean")
            std = _parse_float(match.group(2), context=f"{context} std")
            if std < 0.0:
                raise ValueError(f"Expected a non-negative std for {context}, got {std!r}.")
            return mean, std
        return _parse_float(text, context=context), 0.0

    mean = _parse_float(value, context=context)
    return mean, 0.0


def _extract_x_value(value: Any, *, x_pattern: str, context: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"Expected a numeric or string x-axis value for {context}, got bool.")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return _parse_float(value, context=context)
    if not isinstance(value, str):
        raise TypeError(
            f"Expected a numeric or string x-axis value for {context}, got {type(value).__name__}."
        )

    text = value.strip()
    if not text:
        raise ValueError(f"Expected a non-empty x-axis value for {context}.")

    try:
        return _parse_float(text, context=context)
    except (TypeError, ValueError):
        pass

    try:
        regex = re.compile(x_pattern)
    except re.error as exc:
        raise ValueError(f"Invalid --x-pattern regex {x_pattern!r}: {exc}") from exc
    match = regex.search(text)
    if match is None:
        raise ValueError(
            f"Failed to extract a numeric x value from {text!r} for {context} using pattern {x_pattern!r}."
        )
    token = match.group(1) if match.lastindex else match.group(0)
    return _parse_float(token, context=context)


def _default_y_label(metric_column: str) -> str:
    known = {
        "test_class_acc_kmeans_plusplus_hungarian": "Test ACC (KMeans++ + Hungarian)",
        "test_class_nmi": "Test NMI",
        "test_class_ari": "Test ARI",
        "test_loss": "Test loss",
    }
    return known.get(metric_column, metric_column.replace("_", " "))


def _load_points(
    csv_path: Path,
    *,
    x_column: str,
    metric_column: str,
    x_pattern: str,
) -> list[SweepPoint]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Grouped CSV file does not exist: {csv_path}")

    table = pd.read_csv(csv_path)
    required_columns = {x_column, metric_column}
    missing = sorted(required_columns.difference(table.columns))
    if missing:
        raise KeyError(
            f"Grouped CSV {csv_path} is missing required columns {missing}. "
            f"Available columns: {list(table.columns)}"
        )

    points: list[SweepPoint] = []
    seen_x_values: dict[float, str] = {}
    for row_index, row in table.iterrows():
        name = str(row[x_column])
        x_value = _extract_x_value(
            row[x_column],
            x_pattern=x_pattern,
            context=f"row {row_index} column {x_column!r}",
        )
        if x_value in seen_x_values:
            raise ValueError(
                f"Duplicate x-axis value {x_value} extracted from {seen_x_values[x_value]!r} and {name!r}."
            )
        mean, std = _parse_mean_std_text(
            row[metric_column],
            context=f"row {row_index} column {metric_column!r}",
        )
        points.append(SweepPoint(name=name, x_value=x_value, mean=mean, std=std))
        seen_x_values[x_value] = name

    if not points:
        raise ValueError(f"Grouped CSV {csv_path} is empty; nothing to plot.")

    points.sort(key=lambda point: point.x_value)
    return points


def _format_x_tick(value: float) -> str:
    rounded = round(value)
    if math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-9):
        return str(int(rounded))
    return f"{value:g}"


def _compute_y_limits(values: np.ndarray, errors: np.ndarray) -> tuple[float, float]:
    lower_bounds = values - errors
    upper_bounds = values + errors
    value_min = float(np.min(lower_bounds))
    value_max = float(np.max(upper_bounds))

    if 0.0 <= value_min and value_max <= 1.0:
        lower = max(0.0, math.floor((value_min - 0.03) / 0.05) * 0.05)
        upper = min(1.0, math.ceil((value_max + 0.02) / 0.05) * 0.05)
        if upper - lower < 0.2:
            center = 0.5 * (upper + lower)
            lower = max(0.0, center - 0.1)
            upper = min(1.0, center + 0.1)
        return lower, upper

    span = max(value_max - value_min, 1e-6)
    padding = 0.12 * span
    return value_min - padding, value_max + padding


def _compute_y_ticks(y_limits: tuple[float, float]) -> np.ndarray:
    y_min, y_max = y_limits
    span = y_max - y_min
    if span <= 0.0:
        raise ValueError(f"Invalid y-axis limits: {y_limits}.")
    if 0.0 <= y_min and y_max <= 1.0:
        step = 0.05 if span <= 0.3 else 0.1
        start = math.ceil(y_min / step) * step
        stop = math.floor(y_max / step) * step
        ticks = np.arange(start, stop + 0.5 * step, step, dtype=np.float64)
        if ticks.size >= 3:
            return ticks
    return np.linspace(y_min, y_max, num=5, dtype=np.float64)


def _resolve_output_prefix(
    csv_path: Path,
    *,
    output_prefix: Path | None,
    metric_column: str,
    x_label: str,
) -> Path:
    if output_prefix is not None:
        return output_prefix.resolve()
    return (csv_path.resolve().parent / f"{_slugify(metric_column)}_vs_{_slugify(x_label)}").resolve()


def _save_figure(
    fig: plt.Figure,
    *,
    output_prefix: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    written_paths: list[Path] = []
    for fmt in formats:
        normalized = fmt.lower().lstrip(".")
        if normalized not in {"pdf", "png", "svg"}:
            raise ValueError(
                f"Unsupported output format {fmt!r}. Supported formats: pdf, png, svg."
            )
        output_path = output_prefix.with_suffix(f".{normalized}")
        save_kwargs: dict[str, Any] = {}
        if normalized == "png":
            save_kwargs["dpi"] = int(dpi)
        fig.savefig(output_path, **save_kwargs)
        written_paths.append(output_path)
    return written_paths


def render_grouped_metric_plot(
    csv_path: Path,
    *,
    metric_column: str,
    x_column: str = "experiment",
    x_pattern: str = DEFAULT_X_PATTERN,
    x_label: str = "Sweep value",
    y_label: str | None = None,
    output_prefix: Path | None = None,
    formats: Sequence[str] = ("pdf",),
    dpi: int = 300,
    title: str = "",
    line_color: str = "#1557a6",
) -> list[Path]:
    points = _load_points(
        csv_path.resolve(),
        x_column=x_column,
        metric_column=metric_column,
        x_pattern=x_pattern,
    )
    x_values = np.asarray([point.x_value for point in points], dtype=np.float64)
    y_values = np.asarray([point.mean for point in points], dtype=np.float64)
    y_errors = np.asarray([point.std for point in points], dtype=np.float64)
    y_limits = _compute_y_limits(y_values, y_errors)
    y_ticks = _compute_y_ticks(y_limits)
    resolved_output_prefix = _resolve_output_prefix(
        csv_path,
        output_prefix=output_prefix,
        metric_column=metric_column,
        x_label=x_label,
    )
    resolved_output_prefix.parent.mkdir(parents=True, exist_ok=True)
    resolved_y_label = y_label or _default_y_label(metric_column)

    with plt.rc_context(_paper_rcparams()):
        fig, ax = plt.subplots(figsize=(4.6, 3.5), constrained_layout=True)
        ax.errorbar(
            x_values,
            y_values,
            yerr=y_errors,
            fmt="none",
            ecolor=line_color,
            elinewidth=1.3,
            capsize=3.5,
            capthick=1.3,
            alpha=0.75,
            zorder=2,
        )
        ax.plot(
            x_values,
            y_values,
            color=line_color,
            linewidth=2.3,
            marker="o",
            markersize=7.0,
            markerfacecolor="white",
            markeredgecolor=line_color,
            markeredgewidth=1.8,
            zorder=3,
        )

        x_span = float(np.ptp(x_values))
        x_margin = 0.04 * x_span if x_span > 0.0 else 1.0
        ax.set_xlim(float(np.min(x_values)) - x_margin, float(np.max(x_values)) + x_margin)
        ax.set_ylim(*y_limits)
        ax.xaxis.set_major_locator(FixedLocator(x_values))
        ax.set_xticklabels([_format_x_tick(value) for value in x_values])
        ax.yaxis.set_major_locator(FixedLocator(y_ticks))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_xlabel(x_label)
        ax.set_ylabel(resolved_y_label)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        if title:
            ax.set_title(title, pad=8)

        written_paths = _save_figure(
            fig,
            output_prefix=resolved_output_prefix,
            formats=formats,
            dpi=dpi,
        )
        plt.close(fig)
    return written_paths


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    written_paths = render_grouped_metric_plot(
        args.csv_path,
        metric_column=args.metric_column,
        x_column=args.x_column,
        x_pattern=args.x_pattern,
        x_label=args.x_label,
        y_label=args.y_label,
        output_prefix=args.output_prefix,
        formats=args.formats,
        dpi=args.dpi,
        title=args.title,
        line_color=args.line_color,
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
