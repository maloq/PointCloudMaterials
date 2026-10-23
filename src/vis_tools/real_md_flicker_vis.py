"""Render temporal cluster diagnostics from tables, without model inference."""

from pathlib import Path
from typing import Any
import json

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_flicker_figure(fig: Any, out_file: Path) -> str:
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
    finally:
        plt.close(fig)
    print(f"[analysis][savefig] {out_path.resolve()}")
    return str(out_path)

def _flicker_table_x_axis(table, *, time_column, fallback_column, fallback_label):
    time = table[time_column].to_numpy(dtype=float)
    if np.isfinite(time).all():
        units = table['time_unit'].dropna().unique()
        unit = units[0] if len(units) == 1 else ''
        return time, f'time ({unit})' if unit else 'time'
    return table[fallback_column].to_numpy(dtype=float), fallback_label

def _style_flicker_axes(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.20, linewidth=0.7)
    ax.tick_params(axis="both", labelsize=9)

def _cluster_display_labels_for_flicker(cluster_ids, cluster_display_map):
    return [cluster_display_map[cid] for cid in cluster_ids]

def _cluster_colors_for_flicker(cluster_ids, cluster_color_map):
    return {cid: cluster_color_map[cid] for cid in cluster_ids}

def _rgba_with_alpha(color: str, alpha: float) -> tuple[float, float, float, float]:
    red, green, blue, _ = mcolors.to_rgba(color)
    return (float(red), float(green), float(blue), float(alpha))

def _weighted_metric_mean(
    table: pd.DataFrame,
    *,
    value_column: str,
    weight_column: str,
) -> float:
    values = table[value_column].to_numpy(dtype=np.float64)
    weights = table[weight_column].to_numpy(dtype=np.float64)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(mask):
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))

def _save_flicker_churn_timeseries_plot(
    pair_table: pd.DataFrame,
    out_file: Path,
) -> str:
    x, x_label = _flicker_table_x_axis(
        pair_table,
        time_column="time_to",
        fallback_column="pair_index",
        fallback_label="frame pair",
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.0, 7.0),
        dpi=220,
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0]},
    )
    high_rate_specs = [
        ("label_churn_rate", "label churn", "#111827", 2.4),
        ("excess_churn_rate", "excess churn", "#6b7280", 2.1),
        ("reciprocal_pair_flicker_rate", "reciprocal pair flicker", "#0f766e", 2.1),
    ]
    high_rate_values: list[np.ndarray] = []
    for column, label, color, linewidth in high_rate_specs:
        if column not in pair_table.columns:
            raise KeyError(f"Missing required flicker column for churn plot: {column!r}.")
        y = pair_table[column].to_numpy(dtype=np.float64)
        high_rate_values.append(y)
        axes[0].plot(x, y, label=label, color=color, linewidth=linewidth)
    stacked_high = np.concatenate([values[np.isfinite(values)] for values in high_rate_values])
    if stacked_high.size > 0:
        y_min = float(np.min(stacked_high))
        y_max = float(np.max(stacked_high))
        pad = max(0.01, 0.12 * (y_max - y_min))
        axes[0].set_ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))
    axes[0].set_title("Frame-to-frame flicker rates")
    axes[0].set_ylabel("fraction changed")
    _style_flicker_axes(axes[0])
    axes[0].legend(frameon=False, ncol=3)

    if "net_population_tv" not in pair_table.columns:
        raise KeyError("Missing required flicker column for churn plot: 'net_population_tv'.")
    adjacent_tv = pair_table["net_population_tv"].to_numpy(dtype=np.float64)
    axes[1].plot(x, adjacent_tv, label="adjacent population TV", color="#64748b", linewidth=2.1)
    if "population_tv_from_first" in pair_table.columns:
        cumulative_tv = pair_table["population_tv_from_first"].to_numpy(dtype=np.float64)
        axes[1].plot(
            x,
            cumulative_tv,
            label="population TV from first frame",
            color="#be123c",
            linewidth=2.4,
        )
    axes[1].set_title("Population drift: adjacent step vs accumulated phase change")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("population TV")
    axes[1].set_ylim(0.0, 1.0)
    _style_flicker_axes(axes[1])
    axes[1].legend(frameon=False, ncol=2)
    return _save_flicker_figure(fig, out_file)

def _save_flicker_spatial_coherence_plot(
    pair_table: pd.DataFrame,
    out_file: Path,
) -> str:
    x, x_label = _flicker_table_x_axis(
        pair_table,
        time_column="time_to",
        fallback_column="pair_index",
        fallback_label="frame pair",
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.0, 7.0),
        dpi=220,
        sharex=True,
        gridspec_kw={"height_ratios": [1.05, 1.0]},
    )
    trend_specs = [
        (
            "spatial_neighbor_changed_fraction_mean",
            "neighbors that changed",
            "#111827",
            axes[0],
        ),
        (
            "spatial_same_transition_fraction_mean",
            "neighbors with same transition",
            "#475569",
            axes[0],
        ),
        (
            "spatial_isolated_change_rate",
            "isolated changes",
            "#991b1b",
            axes[1],
        ),
        (
            "spatial_coherent_change_rate",
            "coherent changes",
            "#0f766e",
            axes[1],
        ),
    ]
    for column, label, color, ax in trend_specs:
        if column not in pair_table.columns:
            raise KeyError(f"Missing required flicker column for spatial plot: {column!r}.")
        y = pair_table[column].to_numpy(dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[finite], y[finite], s=7.0, color=color, alpha=0.16, linewidths=0.0)
        if np.count_nonzero(finite) > 0:
            window = max(5, min(51, 2 * (int(np.count_nonzero(finite)) // 40) + 1))
            trend = (
                pd.Series(y)
                .rolling(window=window, center=True, min_periods=max(3, window // 3))
                .median()
                .to_numpy(dtype=np.float64)
            )
            ax.plot(x, trend, label=f"{label} ({window}-pair median)", color=color, linewidth=2.4)
    axes[0].set_title("Spatial neighborhood context of label changes")
    axes[0].set_ylabel("neighbor fraction")
    axes[0].set_ylim(0.0, 1.0)
    _style_flicker_axes(axes[0])
    axes[0].legend(frameon=False, ncol=2)

    axes[1].set_title("Fraction of changed structures passing isolation/coherence thresholds")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("changed-structure fraction")
    axes[1].set_ylim(0.0, 1.0)
    _style_flicker_axes(axes[1])
    axes[1].legend(frameon=False, ncol=2)
    return _save_flicker_figure(fig, out_file)

def _save_flicker_reciprocal_heatmap_plot(
    reciprocal_table: pd.DataFrame,
    *,
    cluster_ids: list[int],
    cluster_colors: dict[int, str],
    cluster_display_labels: list[str],
    out_file: Path,
) -> str:
    matrix = np.full((len(cluster_ids), len(cluster_ids)), np.nan, dtype=np.float64)
    cluster_to_pos = {int(cluster_id): pos for pos, cluster_id in enumerate(cluster_ids)}
    grouped = reciprocal_table.groupby(["cluster_a", "cluster_b"], dropna=False)["reciprocal_rate"].mean()
    for (cluster_a, cluster_b), value in grouped.items():
        left = int(cluster_a)
        right = int(cluster_b)
        if left not in cluster_to_pos or right not in cluster_to_pos:
            raise ValueError(
                "Reciprocal flicker table contains a cluster not present in cluster_ids: "
                f"cluster_a={left}, cluster_b={right}, cluster_ids={cluster_ids}."
            )
        i = cluster_to_pos[left]
        j = cluster_to_pos[right]
        matrix[i, j] = float(value)
        matrix[j, i] = float(value)
    np.fill_diagonal(matrix, np.nan)

    finite_values = matrix[np.isfinite(matrix)]
    vmax = float(np.max(finite_values)) if finite_values.size > 0 and np.max(finite_values) > 0.0 else 1.0
    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad("#f8fafc")
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=220)
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_title("Mean reciprocal flicker by cluster pair")
    ax.set_xticks(np.arange(len(cluster_ids)))
    ax.set_yticks(np.arange(len(cluster_ids)))
    ax.set_xticklabels(cluster_display_labels, rotation=45, ha="right")
    ax.set_yticklabels(cluster_display_labels)
    for pos, cluster_id in enumerate(cluster_ids):
        ax.get_xticklabels()[pos].set_color(cluster_colors[int(cluster_id)])
        ax.get_yticklabels()[pos].set_color(cluster_colors[int(cluster_id)])
        ax.add_patch(
            plt.Rectangle(
                (pos - 0.5, pos - 0.5),
                1.0,
                1.0,
                facecolor="#e5e7eb",
                edgecolor="#94a3b8",
                hatch="//",
                linewidth=0.45,
                zorder=3,
            )
        )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if i == j or not np.isfinite(value) or value <= 0.0:
                continue
            text_color = "#f8fafc" if value > 0.55 * vmax else "#111827"
            ax.text(j, i, f"{100.0 * value:.1f}%", ha="center", va="center", fontsize=8, color=text_color)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("mean reciprocal rate")
    ax.set_xlabel("cluster")
    ax.set_ylabel("cluster")
    return _save_flicker_figure(fig, out_file)

def _save_flicker_transition_popularity_plot(
    transition_table: pd.DataFrame,
    *,
    cluster_ids: list[int],
    cluster_colors: dict[int, str],
    cluster_display_labels: list[str],
    top_n: int,
    out_file: Path,
) -> str:
    required_columns = {
        "source_cluster_id",
        "target_cluster_id",
        "transition_count",
        "fraction_of_changed_transitions",
    }
    missing = sorted(required_columns.difference(transition_table.columns))
    if missing:
        raise KeyError(f"Missing required transition popularity columns: {missing}.")
    label_by_cluster = {
        int(cluster_id): str(display_label)
        for cluster_id, display_label in zip(cluster_ids, cluster_display_labels, strict=True)
    }
    table = transition_table.copy()
    table["source_cluster_id"] = table["source_cluster_id"].astype(int)
    table["target_cluster_id"] = table["target_cluster_id"].astype(int)
    table["transition_count"] = table["transition_count"].astype(np.int64)
    table["fraction_of_changed_transitions"] = table["fraction_of_changed_transitions"].astype(float)
    table = table[
        (table["source_cluster_id"] != table["target_cluster_id"])
        & (table["transition_count"] > 0)
    ].sort_values(
        ["transition_count", "source_cluster_id", "target_cluster_id"],
        ascending=[False, True, True],
    )
    if table.empty:
        raise ValueError("Cannot write transition popularity plot because no off-diagonal transitions were observed.")

    top = table.head(int(top_n)).iloc[::-1].reset_index(drop=True)
    y = np.arange(len(top), dtype=np.float64)
    fractions = top["fraction_of_changed_transitions"].to_numpy(dtype=np.float64) * 100.0
    counts = top["transition_count"].to_numpy(dtype=np.int64)
    labels = [
        f"{label_by_cluster.get(int(row.source_cluster_id), f'C{int(row.source_cluster_id)}')}"
        f" -> {label_by_cluster.get(int(row.target_cluster_id), f'C{int(row.target_cluster_id)}')}"
        for row in top.itertuples(index=False)
    ]
    source_colors = [
        cluster_colors.get(int(source_id), "#64748b")
        for source_id in top["source_cluster_id"].to_numpy(dtype=np.int64)
    ]
    target_colors = [
        cluster_colors.get(int(target_id), "#111827")
        for target_id in top["target_cluster_id"].to_numpy(dtype=np.int64)
    ]

    fig, ax = plt.subplots(figsize=(9.8, 5.7), dpi=220)
    bars = ax.barh(
        y,
        fractions,
        color=[_rgba_with_alpha(color, 0.72) for color in source_colors],
        edgecolor=target_colors,
        linewidth=1.6,
    )
    x_max = max(1.0, float(np.nanmax(fractions)) * 1.18)
    ax.set_xlim(0.0, x_max)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("share of all changed adjacent-frame observations (%)")
    ax.set_ylabel("directed cluster change")
    ax.set_title(f"Top {min(int(top_n), len(table))} cluster-cluster transitions over the whole simulation")
    total_changed = int(np.sum(table["transition_count"].to_numpy(dtype=np.int64)))
    ax.text(
        0.995,
        0.02,
        f"total changed observations: {total_changed:,}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#475569",
    )
    for bar, fraction, count in zip(bars, fractions, counts, strict=True):
        width = float(bar.get_width())
        label = f"{fraction:.1f}%  ({int(count):,})"
        ax.text(
            min(width + 0.015 * x_max, 0.985 * x_max),
            bar.get_y() + bar.get_height() / 2.0,
            label,
            va="center",
            ha="left" if width < 0.84 * x_max else "right",
            fontsize=8.5,
            color="#111827",
        )
    _style_flicker_axes(ax)
    return _save_flicker_figure(fig, out_file)

def _directed_transition_counts_by_pair(
    reciprocal_table: pd.DataFrame,
    *,
    source_cluster_id: int,
    target_cluster_id: int,
    pair_indices: np.ndarray,
) -> np.ndarray:
    if int(source_cluster_id) == int(target_cluster_id):
        raise ValueError(
            "Directed transition count series requires different source and target clusters, "
            f"got {source_cluster_id} -> {target_cluster_id}."
        )
    required_columns = {"pair_index", "cluster_a", "cluster_b", "a_to_b_count", "b_to_a_count"}
    missing = sorted(required_columns.difference(reciprocal_table.columns))
    if missing:
        raise KeyError(f"Missing required reciprocal transition columns: {missing}.")
    table = reciprocal_table.copy()
    table["pair_index"] = table["pair_index"].astype(np.int64)
    table["cluster_a"] = table["cluster_a"].astype(int)
    table["cluster_b"] = table["cluster_b"].astype(int)
    table["a_to_b_count"] = table["a_to_b_count"].astype(np.int64)
    table["b_to_a_count"] = table["b_to_a_count"].astype(np.int64)
    pair_index_arr = np.asarray(pair_indices, dtype=np.int64).reshape(-1)
    pair_to_pos = {int(pair_index): pos for pos, pair_index in enumerate(pair_index_arr.tolist())}
    counts = np.zeros(pair_index_arr.shape[0], dtype=np.int64)
    seen_pairs: set[int] = set()
    forward = table[
        (table["cluster_a"] == int(source_cluster_id))
        & (table["cluster_b"] == int(target_cluster_id))
    ]
    reverse = table[
        (table["cluster_a"] == int(target_cluster_id))
        & (table["cluster_b"] == int(source_cluster_id))
    ]
    for rows, count_column in ((forward, "a_to_b_count"), (reverse, "b_to_a_count")):
        for row in rows.itertuples(index=False):
            pair_index = int(row.pair_index)
            if pair_index not in pair_to_pos:
                continue
            if pair_index in seen_pairs:
                raise ValueError(
                    "Duplicate reciprocal rows while building directed transition time series: "
                    f"transition={source_cluster_id}->{target_cluster_id}, pair_index={pair_index}."
                )
            counts[pair_to_pos[pair_index]] = int(getattr(row, count_column))
            seen_pairs.add(pair_index)
    missing_pair_indices = sorted(set(pair_to_pos.keys()).difference(seen_pairs))
    if missing_pair_indices:
        preview = missing_pair_indices[:5]
        raise ValueError(
            "Missing reciprocal rows while building directed transition time series: "
            f"transition={source_cluster_id}->{target_cluster_id}, "
            f"missing_pair_count={len(missing_pair_indices)}, first_missing={preview}."
        )
    return counts

def _save_flicker_transition_popularity_timeseries_plot(
    reciprocal_table: pd.DataFrame,
    transition_table: pd.DataFrame,
    pair_table: pd.DataFrame,
    *,
    cluster_ids: list[int],
    cluster_colors: dict[int, str],
    cluster_display_labels: list[str],
    top_n: int,
    window_frames: int,
    out_file: Path,
) -> str:
    if "pair_index" not in pair_table.columns:
        raise KeyError("Missing required frame-pair column for transition time-series plot: 'pair_index'.")
    window_int = int(window_frames)
    if window_int <= 0:
        raise ValueError(f"transition popularity time window must be positive, got {window_frames}.")
    label_by_cluster = {
        int(cluster_id): str(display_label)
        for cluster_id, display_label in zip(cluster_ids, cluster_display_labels, strict=True)
    }
    pair_order = np.argsort(pair_table["pair_index"].to_numpy(dtype=np.int64))
    sorted_pair_table = pair_table.iloc[pair_order].reset_index(drop=True)
    pair_indices = sorted_pair_table["pair_index"].to_numpy(dtype=np.int64)
    x_raw, x_label = _flicker_table_x_axis(
        sorted_pair_table,
        time_column="time_to",
        fallback_column="pair_index",
        fallback_label="frame pair",
    )
    if x_raw.size != pair_indices.size:
        raise RuntimeError(
            "Transition time-series x-axis length mismatch: "
            f"x_len={x_raw.size}, pair_count={pair_indices.size}."
        )
    if pair_indices.size == 0:
        raise ValueError("Cannot write transition time-series plot because frame_pair_flicker_metrics is empty.")
    effective_window = min(window_int, int(pair_indices.size))
    selected = transition_table.copy()
    selected["source_cluster_id"] = selected["source_cluster_id"].astype(int)
    selected["target_cluster_id"] = selected["target_cluster_id"].astype(int)
    selected["transition_count"] = selected["transition_count"].astype(np.int64)
    selected = selected[
        (selected["source_cluster_id"] != selected["target_cluster_id"])
        & (selected["transition_count"] > 0)
    ].sort_values(
        ["transition_count", "source_cluster_id", "target_cluster_id"],
        ascending=[False, True, True],
    ).head(int(top_n))
    if selected.empty:
        raise ValueError("Cannot write transition time-series plot because no off-diagonal transitions were observed.")

    kernel = np.ones(int(effective_window), dtype=np.float64)
    x_window = np.convolve(np.asarray(x_raw, dtype=np.float64), kernel / float(effective_window), mode="valid")
    unit_text = ""
    if "delta_time" in sorted_pair_table.columns:
        deltas = sorted_pair_table["delta_time"].to_numpy(dtype=np.float64)
        finite_deltas = deltas[np.isfinite(deltas) & (deltas > 0.0)]
        if finite_deltas.size > 0:
            units = [
                str(value).strip()
                for value in sorted_pair_table.get("time_unit", pd.Series(dtype=str)).dropna().unique().tolist()
                if str(value).strip()
            ]
            unit = str(units[0]) if len(set(units)) == 1 else "time units"
            unit_text = f", about {float(np.median(finite_deltas) * effective_window):g} {unit}"

    fig, ax = plt.subplots(figsize=(11.2, 6.4), dpi=220)
    palette = [
        "#111827",
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#f97316",
        "#0891b2",
    ]
    max_count = 0.0
    for color_pos, row in enumerate(selected.itertuples(index=False)):
        source_cluster_id = int(row.source_cluster_id)
        target_cluster_id = int(row.target_cluster_id)
        counts = _directed_transition_counts_by_pair(
            reciprocal_table,
            source_cluster_id=source_cluster_id,
            target_cluster_id=target_cluster_id,
            pair_indices=pair_indices,
        )
        window_counts = np.convolve(counts.astype(np.float64), kernel, mode="valid")
        window_sum_squares = np.convolve(counts.astype(np.float64) ** 2, kernel, mode="valid")
        window_mean = window_counts / float(effective_window)
        window_variance = np.maximum(
            (window_sum_squares / float(effective_window)) - (window_mean ** 2),
            0.0,
        )
        window_std = np.sqrt(window_variance) * np.sqrt(float(effective_window))
        line_color = palette[color_pos % len(palette)]
        transition_label = (
            f"{label_by_cluster.get(source_cluster_id, f'C{source_cluster_id}')}"
            f" -> {label_by_cluster.get(target_cluster_id, f'C{target_cluster_id}')}"
            f" ({int(row.transition_count):,})"
        )
        ax.fill_between(
            x_window,
            np.maximum(0.0, window_counts - window_std),
            window_counts + window_std,
            color=line_color,
            alpha=0.10,
            linewidth=0.0,
        )
        ax.plot(
            x_window,
            window_counts,
            color=line_color,
            linewidth=2.2,
            label=transition_label,
        )
        if window_counts.size > 0:
            max_count = max(max_count, float(np.nanmax(window_counts + window_std)))
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"transition count per {effective_window}-pair window")
    ax.set_ylim(0.0, max(1.0, max_count * 1.12))
    _style_flicker_axes(ax)
    ax.legend(
        title="directed transition (whole-run count)",
        frameon=False,
        ncol=2,
        loc="upper right",
        fontsize=8.4,
        title_fontsize=8.8,
    )
    fig.suptitle(
        f"Top {len(selected)} cluster-cluster transitions over time "
        f"({effective_window} adjacent-pair window{unit_text}; fade = +/- 1 window std)",
        y=1.01,
    )
    return _save_flicker_figure(fig, out_file)

def _save_flicker_dwell_summary_plot(
    dwell_summary_table: pd.DataFrame,
    *,
    cluster_ids: list[int],
    cluster_colors: dict[int, str],
    cluster_display_labels: list[str],
    out_file: Path,
) -> str:
    x = np.arange(len(cluster_ids), dtype=np.float64)
    rows_by_cluster = {
        int(row["cluster_id"]): row
        for _, row in dwell_summary_table.iterrows()
    }
    colors = [cluster_colors[int(cluster_id)] for cluster_id in cluster_ids]
    frame_spacing_text = ""
    if {"mean_dwell_time", "mean_dwell_frames", "time_unit"}.issubset(dwell_summary_table.columns):
        spacings: list[float] = []
        for _, row in dwell_summary_table.iterrows():
            frames = float(row["mean_dwell_frames"])
            time = float(row["mean_dwell_time"])
            if np.isfinite(frames) and np.isfinite(time) and frames > 0.0:
                spacings.append(time / frames)
        if spacings:
            units = [
                str(value).strip()
                for value in dwell_summary_table["time_unit"].dropna().unique().tolist()
                if str(value).strip()
            ]
            unit = str(units[0]) if len(set(units)) == 1 else "time units"
            frame_spacing_text = f"; 1 frame = {float(np.median(spacings)):.3g} {unit}"
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), dpi=220, sharex=True)
    metrics = [
        ("median_dwell_frames", "typical episode length (median)", "consecutive frames"),
        ("mean_dwell_frames", "average episode length (mean)", "consecutive frames"),
        ("p90_dwell_frames", "long episode threshold (90th percentile)", "consecutive frames"),
        ("fraction_runs_le_2_frames", "short episodes (<= 2 frames)", "episode fraction"),
    ]
    for ax, (column, title, y_label) in zip(axes.ravel(), metrics, strict=True):
        if column not in dwell_summary_table.columns:
            raise KeyError(f"Missing required dwell summary column for plot: {column!r}.")
        values = np.asarray(
            [
                float(rows_by_cluster[int(cluster_id)][column])
                if int(cluster_id) in rows_by_cluster and pd.notna(rows_by_cluster[int(cluster_id)][column])
                else np.nan
                for cluster_id in cluster_ids
            ],
            dtype=np.float64,
        )
        ax.bar(x, values, color=colors, alpha=0.88, edgecolor="#111827", linewidth=0.35)
        ax.set_title(title)
        ax.set_ylabel(y_label)
        if y_label == "episode fraction":
            ax.set_ylim(0.0, 1.0)
        _style_flicker_axes(ax)
    for ax in axes[-1, :]:
        ax.set_xticks(x)
        ax.set_xticklabels(cluster_display_labels)
        for tick, cluster_id in zip(ax.get_xticklabels(), cluster_ids, strict=True):
            tick.set_color(cluster_colors[int(cluster_id)])
    fig.suptitle(
        "Cluster-label episodes by source cluster "
        f"(episode = consecutive frames with the same label{frame_spacing_text})",
        y=1.02,
    )
    return _save_flicker_figure(fig, out_file)

def _save_flicker_dwell_distribution_plot(
    dwell_hist_table: pd.DataFrame,
    *,
    cluster_ids: list[int],
    cluster_colors: dict[int, str],
    cluster_display_labels: list[str],
    out_file: Path,
) -> str:
    all_lengths = dwell_hist_table["dwell_frames"].to_numpy(dtype=np.float64)
    all_counts = dwell_hist_table["run_count"].to_numpy(dtype=np.float64)
    positive = np.isfinite(all_lengths) & np.isfinite(all_counts) & (all_counts > 0.0)
    if not np.any(positive):
        raise ValueError("Cannot write dwell distribution plot because all episode counts are zero or invalid.")
    order = np.argsort(all_lengths[positive])
    sorted_lengths = all_lengths[positive][order]
    sorted_counts = all_counts[positive][order]
    cumulative_counts = np.cumsum(sorted_counts)
    p95_index = int(np.searchsorted(cumulative_counts, 0.95 * float(cumulative_counts[-1]), side="left"))
    short_limit = int(min(float(sorted_lengths[-1]), max(10.0, float(sorted_lengths[p95_index]))))
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.0), dpi=220, sharex=False)
    for cluster_id, display_label in zip(cluster_ids, cluster_display_labels, strict=True):
        sub = dwell_hist_table[dwell_hist_table["cluster_id"].astype(int) == int(cluster_id)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("dwell_frames")
        lengths = sub["dwell_frames"].to_numpy(dtype=np.float64)
        counts = sub["run_count"].to_numpy(dtype=np.float64)
        if np.sum(counts) <= 0.0:
            continue
        cumulative = np.cumsum(counts) / float(np.sum(counts))
        survival = np.cumsum(counts[::-1])[::-1] / float(np.sum(counts))
        axes[0].step(
            lengths,
            cumulative,
            where="post",
            color=cluster_colors[int(cluster_id)],
            linewidth=2.0,
            label=display_label,
        )
        axes[1].step(
            lengths,
            survival,
            where="post",
            color=cluster_colors[int(cluster_id)],
            linewidth=2.0,
            label=display_label,
        )
    axes[0].set_title("How quickly cluster-label episodes end")
    axes[0].set_xlabel("episode length threshold (consecutive frames)")
    axes[0].set_ylabel("fraction of episodes with length <= threshold")
    axes[0].set_xlim(1.0, max(1.0, float(short_limit)))
    axes[0].set_ylim(0.0, 1.0)
    _style_flicker_axes(axes[0])
    axes[0].legend(title="cluster", frameon=False, ncol=min(4, max(1, len(cluster_ids))))

    axes[1].set_title("Rare long cluster-label episodes")
    axes[1].set_xlabel("episode length threshold (consecutive frames)")
    axes[1].set_ylabel("fraction of episodes with length >= threshold")
    axes[1].set_yscale("log")
    axes[1].set_ylim(1.0e-4, 1.05)
    _style_flicker_axes(axes[1])
    axes[1].legend(title="cluster", frameon=False, ncol=min(4, max(1, len(cluster_ids))))
    fig.suptitle("Episode length distribution (episode = uninterrupted stay in one cluster)", y=1.02)
    return _save_flicker_figure(fig, out_file)

def _save_flicker_recrossing_by_lag_plot(
    recrossing_table: pd.DataFrame,
    out_file: Path,
) -> str:
    if "lag_frames" in recrossing_table.columns:
        order = np.argsort(
            recrossing_table["lag_frames"].to_numpy(dtype=np.float64)
        )
    else:
        x_raw, _x_label = _flicker_table_x_axis(
            recrossing_table,
            time_column="lag_time",
            fallback_column="lag_frames",
            fallback_label="lag (frames)",
        )
        order = np.argsort(x_raw)
    persistent = recrossing_table["persistent_fraction"].to_numpy(dtype=np.float64)[order]
    recross = recrossing_table["recross_fraction"].to_numpy(dtype=np.float64)[order]
    other = recrossing_table["other_fraction"].to_numpy(dtype=np.float64)[order]
    lag_frames = (
        recrossing_table["lag_frames"].to_numpy(dtype=np.int64)[order]
        if "lag_frames" in recrossing_table.columns
        else np.arange(1, len(recrossing_table) + 1, dtype=np.int64)
    )
    lag_time = (
        recrossing_table["lag_time"].to_numpy(dtype=np.float64)[order]
        if "lag_time" in recrossing_table.columns
        else np.full(lag_frames.shape, np.nan, dtype=np.float64)
    )
    units = [
        str(value).strip()
        for value in recrossing_table.get("time_unit", pd.Series(dtype=str)).dropna().unique().tolist()
        if str(value).strip()
    ]
    unit = str(units[0]) if len(set(units)) == 1 else ""
    tick_labels = []
    for lag_frame, time_value in zip(lag_frames, lag_time, strict=True):
        frame_word = "frame" if int(lag_frame) == 1 else "frames"
        if np.isfinite(time_value) and unit:
            tick_labels.append(f"{int(lag_frame)} {frame_word}\n{float(time_value):g} {unit}")
        else:
            tick_labels.append(f"{int(lag_frame)} {frame_word}")
    x = np.arange(len(lag_frames), dtype=np.float64)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.6, 6.8),
        dpi=220,
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0]},
    )
    outcome_specs = [
        (persistent, "still in target cluster B", "#334155"),
        (recross, "back to source cluster A", "#0f766e"),
        (other, "neither A nor B", "#b45309"),
    ]
    for values, label, color in outcome_specs:
        axes[0].plot(x, values, marker="o", markersize=5.0, linewidth=2.3, color=color, label=label)
    y_max = max(0.1, float(np.nanmax([np.nanmax(persistent), np.nanmax(recross), np.nanmax(other)])) + 0.06)
    axes[0].set_title("Outcome after an adjacent label change A -> B")
    axes[0].set_ylabel("fraction of changed transitions")
    axes[0].set_ylim(0.0, min(1.0, y_max))
    _style_flicker_axes(axes[0])
    axes[0].legend(frameon=False, ncol=3)

    for values, label, color in outcome_specs:
        delta_pp = 100.0 * (values - values[0])
        axes[1].plot(x, delta_pp, marker="o", markersize=4.5, linewidth=2.1, color=color, label=label)
    axes[1].axhline(0.0, color="#94a3b8", linewidth=1.0)
    all_delta = np.concatenate([100.0 * (values - values[0]) for values, _label, _color in outcome_specs])
    finite_delta = all_delta[np.isfinite(all_delta)]
    if finite_delta.size > 0:
        bound = max(2.0, 1.18 * float(np.max(np.abs(finite_delta))))
        axes[1].set_ylim(-bound, bound)
    axes[1].set_title("Change relative to the shortest follow-up lag")
    axes[1].set_ylabel("percentage points")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tick_labels)
    axes[1].set_xlabel("follow-up lag after the A -> B change")
    _style_flicker_axes(axes[1])
    return _save_flicker_figure(fig, out_file)

def _save_flicker_recrossing_by_source_plot(
    recrossing_by_source_table: pd.DataFrame,
    *,
    cluster_ids: list[int],
    cluster_colors: dict[int, str],
    cluster_display_labels: list[str],
    out_file: Path,
) -> str:
    table = recrossing_by_source_table.copy()
    table["source_cluster_id"] = table["source_cluster_id"].astype(int)
    if "lag_frames" in table.columns:
        lag_order = np.sort(table["lag_frames"].unique().astype(int))
    else:
        lag_order = np.arange(1, len(table["lag_time"].unique()) + 1, dtype=int)
    units = [
        str(value).strip()
        for value in table.get("time_unit", pd.Series(dtype=str)).dropna().unique().tolist()
        if str(value).strip()
    ]
    unit = str(units[0]) if len(set(units)) == 1 else ""
    lag_labels: list[str] = []
    for lag in lag_order:
        sub_lag = table[table["lag_frames"].astype(int) == int(lag)] if "lag_frames" in table.columns else table.iloc[0:0]
        lag_time_value = (
            float(sub_lag["lag_time"].dropna().iloc[0])
            if (not sub_lag.empty and "lag_time" in sub_lag.columns and sub_lag["lag_time"].notna().any())
            else np.nan
        )
        frame_word = "frame" if int(lag) == 1 else "frames"
        if np.isfinite(lag_time_value) and unit:
            lag_labels.append(f"{int(lag)} {frame_word}\n{lag_time_value:g} {unit}")
        else:
            lag_labels.append(f"{int(lag)} {frame_word}")

    recross_matrix = np.full((len(cluster_ids), len(lag_order)), np.nan, dtype=np.float64)
    longest_lag = int(lag_order[-1])
    persistent_long = np.full(len(cluster_ids), np.nan, dtype=np.float64)
    recross_long = np.full(len(cluster_ids), np.nan, dtype=np.float64)
    other_long = np.full(len(cluster_ids), np.nan, dtype=np.float64)
    for row_pos, cluster_id in enumerate(cluster_ids):
        sub = table[table["source_cluster_id"] == int(cluster_id)]
        for lag_pos, lag in enumerate(lag_order):
            lag_sub = sub[sub["lag_frames"].astype(int) == int(lag)]
            if lag_sub.empty:
                continue
            recross_matrix[row_pos, lag_pos] = float(lag_sub["recross_fraction"].iloc[0])
        long_sub = sub[sub["lag_frames"].astype(int) == longest_lag]
        if not long_sub.empty:
            persistent_long[row_pos] = float(long_sub["persistent_fraction"].iloc[0])
            recross_long[row_pos] = float(long_sub["recross_fraction"].iloc[0])
            other_long[row_pos] = float(long_sub["other_fraction"].iloc[0])

    fig, axes = plt.subplots(2, 1, figsize=(9.8, 7.4), dpi=220, gridspec_kw={"height_ratios": [1.0, 1.08]})
    cmap = plt.cm.YlOrBr.copy()
    cmap.set_bad("#f8fafc")
    image = axes[0].imshow(np.ma.masked_invalid(recross_matrix), cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    axes[0].set_title("Back-to-source fraction after a label change A -> B")
    axes[0].set_xticks(np.arange(len(lag_order)))
    axes[0].set_xticklabels(lag_labels)
    axes[0].set_yticks(np.arange(len(cluster_ids)))
    axes[0].set_yticklabels(cluster_display_labels)
    for tick, cluster_id in zip(axes[0].get_yticklabels(), cluster_ids, strict=True):
        tick.set_color(cluster_colors[int(cluster_id)])
    for i in range(recross_matrix.shape[0]):
        for j in range(recross_matrix.shape[1]):
            value = recross_matrix[i, j]
            if np.isfinite(value):
                axes[0].text(j, i, f"{100.0 * value:.0f}%", ha="center", va="center", fontsize=8)
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.035, pad=0.02)
    colorbar.set_label("fraction back in source A")
    axes[0].set_ylabel("source cluster A")

    y = np.arange(len(cluster_ids), dtype=np.float64)
    axes[1].barh(y, persistent_long, color="#334155", label="still in target B")
    axes[1].barh(y, recross_long, left=persistent_long, color="#0f766e", label="back to source A")
    axes[1].barh(y, other_long, left=persistent_long + recross_long, color="#b45309", label="neither A nor B")
    axes[1].set_title(f"Outcome at longest shown lag ({lag_labels[-1].replace(chr(10), ', ')})")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(cluster_display_labels)
    for tick, cluster_id in zip(axes[1].get_yticklabels(), cluster_ids, strict=True):
        tick.set_color(cluster_colors[int(cluster_id)])
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_xlabel("fraction of changed transitions")
    axes[1].set_ylabel("source cluster A")
    _style_flicker_axes(axes[1])
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(
        "Recrossing: after A -> B, a recrossing means the later label is back to A",
        y=1.02,
    )
    return _save_flicker_figure(fig, out_file)

def _save_flicker_cluster_margin_plot(
    margin_table: pd.DataFrame,
    *,
    cluster_ids: list[int],
    cluster_colors: dict[int, str],
    cluster_display_labels: list[str],
    out_file: Path,
) -> str:
    changed_margin: list[float] = []
    stable_margin: list[float] = []
    changed_fraction: list[float] = []
    target_was_second: list[float] = []
    for cluster_id in cluster_ids:
        sub = margin_table[margin_table["source_cluster_id"].astype(int) == int(cluster_id)]
        changed_margin.append(
            _weighted_metric_mean(
                sub,
                value_column="changed_margin_before_mean",
                weight_column="changed_margin_before_count",
            )
        )
        stable_margin.append(
            _weighted_metric_mean(
                sub,
                value_column="stable_margin_before_mean",
                weight_column="stable_margin_before_count",
            )
        )
        source_count = sub["source_count"].to_numpy(dtype=np.float64)
        changed_count = sub["changed_count"].to_numpy(dtype=np.float64)
        finite_counts = np.isfinite(source_count) & np.isfinite(changed_count) & (source_count >= 0.0) & (changed_count >= 0.0)
        changed_fraction.append(
            float(np.sum(changed_count[finite_counts]) / np.sum(source_count[finite_counts]))
            if np.any(finite_counts) and np.sum(source_count[finite_counts]) > 0.0
            else np.nan
        )
        target_was_second.append(
            _weighted_metric_mean(
                sub,
                value_column="runner_up_is_target_before_rate",
                weight_column="changed_count",
            )
        )

    colors = [cluster_colors[int(cluster_id)] for cluster_id in cluster_ids]
    y = np.arange(len(cluster_ids), dtype=np.float64)
    height = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.8), dpi=220, gridspec_kw={"width_ratios": [1.05, 1.0]})
    axes[0].barh(
        y - height / 2.0,
        stable_margin,
        height=height,
        color=[_rgba_with_alpha(color, 0.32) for color in colors],
        edgecolor=colors,
        linewidth=1.0,
        label="stayed in same label",
    )
    axes[0].barh(
        y + height / 2.0,
        changed_margin,
        height=height,
        color=colors,
        alpha=0.88,
        label="changed next frame",
    )
    axes[0].set_title("Assignment confidence before next frame")
    axes[0].set_xlabel("score gap to 2nd-best cluster")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(cluster_display_labels)
    for tick, cluster_id in zip(axes[0].get_yticklabels(), cluster_ids, strict=True):
        tick.set_color(cluster_colors[int(cluster_id)])
    _style_flicker_axes(axes[0])
    axes[0].legend(frameon=False)

    axes[1].barh(y - height / 2.0, changed_fraction, height=height, color=colors, alpha=0.88, label="changed next frame")
    axes[1].barh(
        y + height / 2.0,
        target_was_second,
        height=height,
        color=[_rgba_with_alpha(color, 0.34) for color in colors],
        edgecolor=colors,
        linewidth=1.0,
        hatch="//",
        label="next label was already 2nd-best",
    )
    axes[1].set_title("How boundary-like are changes?")
    axes[1].set_xlabel("fraction")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(cluster_display_labels)
    for tick, cluster_id in zip(axes[1].get_yticklabels(), cluster_ids, strict=True):
        tick.set_color(cluster_colors[int(cluster_id)])
    _style_flicker_axes(axes[1])
    axes[1].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.22))
    fig.suptitle(
        "Cluster-assignment margin: assigned score minus 2nd-best score; smaller gap = more ambiguous",
        y=1.02,
    )
    return _save_flicker_figure(fig, out_file)

def render_flicker_metrics(
    *,
    flicker_dir: Path,
    cluster_ids: list[int],
    cluster_color_map: dict[int, str] | None,
    cluster_display_map: dict[int, str] | None,
    pair_table: pd.DataFrame,
    reciprocal_table: pd.DataFrame,
    transition_table: pd.DataFrame,
    dwell_hist_table: pd.DataFrame,
    dwell_summary_table: pd.DataFrame,
    recrossing_table: pd.DataFrame,
    recrossing_by_source_table: pd.DataFrame,
    margin_table: pd.DataFrame | None,
    transition_popularity_window_frames: int,
) -> dict[str, str]:

    ordered_cluster_ids = [int(cluster_id) for cluster_id in cluster_ids]
    cluster_colors = _cluster_colors_for_flicker(ordered_cluster_ids, cluster_color_map)
    cluster_display_labels = _cluster_display_labels_for_flicker(
        ordered_cluster_ids,
        cluster_display_map,
    )
    flicker_path = Path(flicker_dir)
    artifacts = {
        "churn_timeseries_png": _save_flicker_churn_timeseries_plot(
            pair_table,
            flicker_path / "flicker_churn_timeseries.png",
        ),
        "spatial_coherence_timeseries_png": _save_flicker_spatial_coherence_plot(
            pair_table,
            flicker_path / "flicker_spatial_coherence_timeseries.png",
        ),
        "reciprocal_pair_heatmap_png": _save_flicker_reciprocal_heatmap_plot(
            reciprocal_table,
            cluster_ids=ordered_cluster_ids,
            cluster_colors=cluster_colors,
            cluster_display_labels=cluster_display_labels,
            out_file=flicker_path / "flicker_reciprocal_pair_heatmap.png",
        ),
        "cluster_transition_popularity_png": _save_flicker_transition_popularity_plot(
            transition_table,
            cluster_ids=ordered_cluster_ids,
            cluster_colors=cluster_colors,
            cluster_display_labels=cluster_display_labels,
            top_n=7,
            out_file=flicker_path / "flicker_cluster_transition_popularity.png",
        ),
        "cluster_transition_popularity_timeseries_png": _save_flicker_transition_popularity_timeseries_plot(
            reciprocal_table,
            transition_table,
            pair_table,
            cluster_ids=ordered_cluster_ids,
            cluster_colors=cluster_colors,
            cluster_display_labels=cluster_display_labels,
            top_n=7,
            window_frames=int(transition_popularity_window_frames),
            out_file=flicker_path / "flicker_cluster_transition_popularity_timeseries.png",
        ),
        "dwell_time_summary_png": _save_flicker_dwell_summary_plot(
            dwell_summary_table,
            cluster_ids=ordered_cluster_ids,
            cluster_colors=cluster_colors,
            cluster_display_labels=cluster_display_labels,
            out_file=flicker_path / "flicker_dwell_time_summary.png",
        ),
        "dwell_time_distribution_png": _save_flicker_dwell_distribution_plot(
            dwell_hist_table,
            cluster_ids=ordered_cluster_ids,
            cluster_colors=cluster_colors,
            cluster_display_labels=cluster_display_labels,
            out_file=flicker_path / "flicker_dwell_time_distribution.png",
        ),
        "recrossing_by_lag_png": _save_flicker_recrossing_by_lag_plot(
            recrossing_table,
            flicker_path / "flicker_recrossing_by_lag.png",
        ),
        "recrossing_by_source_cluster_png": _save_flicker_recrossing_by_source_plot(
            recrossing_by_source_table,
            cluster_ids=ordered_cluster_ids,
            cluster_colors=cluster_colors,
            cluster_display_labels=cluster_display_labels,
            out_file=flicker_path / "flicker_recrossing_by_source_cluster.png",
        ),
    }
    if margin_table is not None:
        artifacts["cluster_margin_flicker_png"] = _save_flicker_cluster_margin_plot(
            margin_table,
            cluster_ids=ordered_cluster_ids,
            cluster_colors=cluster_colors,
            cluster_display_labels=cluster_display_labels,
            out_file=flicker_path / "flicker_cluster_margin_summary.png",
        )
    return artifacts


def render_saved_flicker_metrics(
    flicker_dir, *, cluster_color_map=None, cluster_display_map=None,
):
    """Regenerate figures from an exported ``flicker/`` directory.

    Example::

        render_saved_flicker_metrics("output/my_analysis/flicker")

    Supply the original color/display maps to reproduce its styling, or new
    maps to change figures. Tables and scientific summaries remain unchanged.
    """
    flicker_dir = Path(flicker_dir)
    summary = json.loads((flicker_dir / "flicker_summary.json").read_text())
    cluster_ids = summary["cluster_ids"]
    if cluster_color_map is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        cluster_color_map = {cid: colors[i % len(colors)] for i, cid in enumerate(cluster_ids)}
    if cluster_display_map is None:
        cluster_display_map = {cid: str(cid) for cid in cluster_ids}
    table_files = {
        "pair_table": "frame_pair_flicker_metrics.csv",
        "reciprocal_table": "reciprocal_pair_flicker.csv",
        "transition_table": "cluster_transition_popularity.csv",
        "dwell_hist_table": "dwell_time_histogram.csv",
        "dwell_summary_table": "dwell_time_summary.csv",
        "recrossing_table": "recrossing_metrics.csv",
        "recrossing_by_source_table": "recrossing_by_source_cluster.csv",
    }
    if summary["cluster_margin_flicker"]["available"]:
        table_files["margin_table"] = "cluster_margin_flicker.csv"
    tables = {name: pd.read_csv(flicker_dir / filename, float_precision="round_trip")
              for name, filename in table_files.items()}
    tables.setdefault("margin_table", None)
    return render_flicker_metrics(
        flicker_dir=flicker_dir, cluster_ids=cluster_ids,
        cluster_color_map=cluster_color_map, cluster_display_map=cluster_display_map,
        transition_popularity_window_frames=summary["transition_popularity_window_frames"],
        **tables,
    )
