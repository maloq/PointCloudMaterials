#!/usr/bin/env python3
"""Render a publication-style metric sweep plot from a grouped CSV file."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from plotting_common import (
        apply_axis_formatting,
        compute_y_limits,
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
        compute_y_limits,
        compute_y_ticks,
        extract_x_value,
        paper_rcparams,
        parse_finite_float,
        save_figure,
        slugify,
    )


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


def _parse_mean_std_text(value, *, context: str) -> tuple[float, float]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"Expected a numeric metric cell for {context}, got an empty string.")
        match = re.fullmatch(
            r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(?:\+/-|±)\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
            text,
        )
        if match is not None:
            mean = parse_finite_float(match.group(1), context=f"{context} mean")
            std = parse_finite_float(match.group(2), context=f"{context} std")
            if std < 0.0:
                raise ValueError(f"Expected a non-negative std for {context}, got {std!r}.")
            return mean, std
        return parse_finite_float(text, context=context), 0.0

    mean = parse_finite_float(value, context=context)
    return mean, 0.0


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
        x_value = extract_x_value(
            row[x_column],
            pattern=x_pattern,
            context=f"row {row_index} column {x_column!r}",
            numeric_direct=True,
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


def _resolve_output_prefix(
    csv_path: Path,
    *,
    output_prefix: Path | None,
    metric_column: str,
    x_label: str,
) -> Path:
    if output_prefix is not None:
        return output_prefix.resolve()
    return (csv_path.resolve().parent / f"{slugify(metric_column)}_vs_{slugify(x_label)}").resolve()


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
    y_limits = compute_y_limits(y_values, y_errors)
    y_ticks = compute_y_ticks(y_limits)
    resolved_output_prefix = _resolve_output_prefix(
        csv_path,
        output_prefix=output_prefix,
        metric_column=metric_column,
        x_label=x_label,
    )
    resolved_output_prefix.parent.mkdir(parents=True, exist_ok=True)
    resolved_y_label = y_label or _default_y_label(metric_column)

    with plt.rc_context(paper_rcparams()):
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

        apply_axis_formatting(
            ax,
            x_values=x_values,
            x_label=x_label,
            y_label=resolved_y_label,
            y_limits=y_limits,
            y_ticks=y_ticks,
        )
        if title:
            ax.set_title(title, pad=8)

        written_paths = save_figure(
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
