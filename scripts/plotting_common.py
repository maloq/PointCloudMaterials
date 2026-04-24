from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FormatStrFormatter


def paper_rcparams() -> dict[str, Any]:
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


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return slug or "plot"


def parse_finite_float(value: Any, *, context: str) -> float:
    if value is None:
        raise ValueError(f"Missing numeric value for {context}.")
    if isinstance(value, bool):
        raise TypeError(f"Expected a numeric value for {context}, got bool.")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"Expected numeric text for {context}, got an empty string.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Expected a numeric value for {context}, got {value!r}.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"Non-finite numeric value {numeric!r} for {context}.")
    return numeric


def extract_x_value(
    value: Any,
    *,
    pattern: str,
    context: str,
    numeric_direct: bool,
) -> float:
    if isinstance(value, bool):
        raise TypeError(f"Expected a numeric or string x-axis value for {context}, got bool.")
    if numeric_direct and isinstance(value, (int, float, np.integer, np.floating)):
        return parse_finite_float(value, context=context)
    if not isinstance(value, str):
        raise TypeError(
            f"Expected a numeric or string x-axis value for {context}, got {type(value).__name__}."
        )

    text = value.strip()
    if not text:
        raise ValueError(f"Expected a non-empty x-axis value for {context}.")

    if numeric_direct:
        try:
            return parse_finite_float(text, context=context)
        except (TypeError, ValueError):
            pass

    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"Invalid --x-pattern regex {pattern!r}: {exc}") from exc
    match = regex.search(text)
    if match is None:
        raise ValueError(
            f"Failed to extract a numeric x value from {text!r} for {context} using pattern {pattern!r}."
        )
    token = match.group(1) if match.lastindex else match.group(0)
    return parse_finite_float(token, context=context)


def format_x_tick(value: float) -> str:
    rounded = round(value)
    if math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-9):
        return str(int(rounded))
    return f"{value:g}"


def compute_y_limits_from_bounds(
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> tuple[float, float]:
    if lower_bounds.size == 0 or upper_bounds.size == 0:
        raise ValueError("Cannot compute y-axis limits with no metric values.")
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


def compute_y_limits(values: np.ndarray, errors: np.ndarray) -> tuple[float, float]:
    return compute_y_limits_from_bounds(values - errors, values + errors)


def compute_y_ticks(y_limits: tuple[float, float]) -> np.ndarray:
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


def apply_axis_formatting(
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
    ax.set_xticklabels([format_x_tick(value) for value in x_values])
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)


def save_figure(
    fig: plt.Figure,
    output_prefix: Path,
    *,
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
