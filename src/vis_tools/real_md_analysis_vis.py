from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.training_methods.contrastive_learning._cluster_colors import _boost_saturation
from src.training_methods.contrastive_learning._cluster_geometry import (
    _draw_cube_wireframe,
    _sample_indices_stratified,
    _set_equal_axes_3d,
)


def _log_saved_figure(path: Path | str) -> None:
    print(f"[analysis][savefig] {Path(path).resolve()}")


def _sorted_cluster_ids(values: np.ndarray) -> list[int]:
    arr = np.asarray(values, dtype=int).reshape(-1)
    return sorted(int(v) for v in np.unique(arr) if int(v) >= 0)


def _cluster_palette_from_map(
    cluster_ids: list[int],
    cluster_color_map: dict[int, str] | None,
) -> list[str]:
    default_palette = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors)
    colors: list[str] = []
    for pos, cluster_id in enumerate(cluster_ids):
        if cluster_color_map is not None and int(cluster_id) in cluster_color_map:
            colors.append(str(cluster_color_map[int(cluster_id)]))
            continue
        colors.append(mcolors.to_hex(default_palette[pos % len(default_palette)]))
    return colors


def _style_paper_axes(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10, width=0.8, length=3.0)
    ax.grid(True, axis="y", alpha=0.18, linewidth=0.6)


def save_cluster_proportion_plots(
    frame_labels: list[str],
    counts: np.ndarray,
    cluster_ids: list[int],
    out_dir: Path,
    *,
    cluster_color_map: dict[int, str] | None = None,
    save_paper_svg: bool = False,
) -> dict[str, Any]:
    counts_arr = np.asarray(counts, dtype=np.float64)
    if counts_arr.ndim != 2:
        raise ValueError(f"counts must have shape (num_frames, num_clusters), got {counts_arr.shape}.")
    if counts_arr.shape != (len(frame_labels), len(cluster_ids)):
        raise ValueError(
            "counts shape does not match frame_labels/cluster_ids: "
            f"counts={counts_arr.shape}, frames={len(frame_labels)}, clusters={len(cluster_ids)}."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = _cluster_palette_from_map(cluster_ids, cluster_color_map)
    x = np.arange(len(frame_labels), dtype=np.float64)
    totals = counts_arr.sum(axis=1, keepdims=True)
    fractions = np.divide(
        counts_arr,
        totals,
        out=np.zeros_like(counts_arr, dtype=np.float64),
        where=totals > 0.0,
    )

    stacked_area_path = out_dir / "cluster_proportions_stacked_area.png"
    fig, ax = plt.subplots(figsize=(11, 5), dpi=220)
    ax.stackplot(x, fractions.T, colors=colors, alpha=0.94, linewidth=0.0)
    if len(x) > 1:
        ax.set_xlim(x[0], x[-1])
    else:
        ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(frame_labels, rotation=35, ha="right")
    ax.set_ylabel("Fraction of local environments")
    ax.set_title("Cluster proportions across time")
    handles = [
        plt.Line2D([0], [0], color=colors[pos], linewidth=7)
        for pos, _ in enumerate(cluster_ids)
    ]
    ax.legend(handles, [f"C{cluster_id}" for cluster_id in cluster_ids], title="cluster", ncol=2)
    ax.grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(stacked_area_path, bbox_inches="tight")
    plt.close(fig)
    _log_saved_figure(stacked_area_path)

    stacked_bar_path = out_dir / "cluster_proportions_stacked_bar.png"
    fig, ax = plt.subplots(figsize=(11, 5), dpi=220)
    bottom = np.zeros((len(frame_labels),), dtype=np.float64)
    for pos, cluster_id in enumerate(cluster_ids):
        ax.bar(
            x,
            fractions[:, pos],
            bottom=bottom,
            color=colors[pos],
            width=0.82,
            label=f"C{cluster_id}",
        )
        bottom += fractions[:, pos]
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(frame_labels, rotation=35, ha="right")
    ax.set_ylabel("Fraction of local environments")
    ax.set_title("Cluster proportions across time")
    ax.legend(title="cluster", ncol=2)
    ax.grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(stacked_bar_path, bbox_inches="tight")
    plt.close(fig)
    _log_saved_figure(stacked_bar_path)

    counts_bar_path = out_dir / "cluster_counts_stacked_bar.png"
    fig, ax = plt.subplots(figsize=(11, 5), dpi=220)
    bottom = np.zeros((len(frame_labels),), dtype=np.float64)
    for pos, cluster_id in enumerate(cluster_ids):
        ax.bar(
            x,
            counts_arr[:, pos],
            bottom=bottom,
            color=colors[pos],
            width=0.82,
            label=f"C{cluster_id}",
        )
        bottom += counts_arr[:, pos]
    ax.set_xticks(x)
    ax.set_xticklabels(frame_labels, rotation=35, ha="right")
    ax.set_ylabel("Number of local environments")
    ax.set_title("Cluster counts across time")
    ax.legend(title="cluster", ncol=2)
    ax.grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(counts_bar_path, bbox_inches="tight")
    plt.close(fig)
    _log_saved_figure(counts_bar_path)

    outputs: dict[str, Any] = {
        "stacked_area": str(stacked_area_path),
        "stacked_bar_fraction": str(stacked_bar_path),
        "stacked_bar_count": str(counts_bar_path),
    }
    if save_paper_svg:
        paper_outputs: dict[str, str] = {}

        stacked_area_paper_path = out_dir / "cluster_proportions_stacked_area_paper.svg"
        fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=220)
        ax.stackplot(x, fractions.T, colors=colors, alpha=0.96, linewidth=0.0)
        if len(x) > 1:
            ax.set_xlim(x[0], x[-1])
        else:
            ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(frame_labels, rotation=0, ha="center")
        ax.set_ylabel("Fraction")
        _style_paper_axes(ax)
        handles = [
            plt.Line2D([0], [0], color=colors[pos], linewidth=5.5)
            for pos, _ in enumerate(cluster_ids)
        ]
        ax.legend(
            handles,
            [f"C{cluster_id}" for cluster_id in cluster_ids],
            title="Cluster",
            ncol=min(3, max(1, len(cluster_ids))),
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            frameon=False,
            fontsize=9,
            title_fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(stacked_area_paper_path, bbox_inches="tight", transparent=True)
        plt.close(fig)
        _log_saved_figure(stacked_area_paper_path)
        paper_outputs["stacked_area_svg"] = str(stacked_area_paper_path)

        stacked_bar_paper_path = out_dir / "cluster_proportions_stacked_bar_paper.svg"
        fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=220)
        bottom = np.zeros((len(frame_labels),), dtype=np.float64)
        for pos, cluster_id in enumerate(cluster_ids):
            ax.bar(
                x,
                fractions[:, pos],
                bottom=bottom,
                color=colors[pos],
                width=0.78,
                label=f"C{cluster_id}",
            )
            bottom += fractions[:, pos]
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(frame_labels, rotation=0, ha="center")
        ax.set_ylabel("Fraction")
        _style_paper_axes(ax)
        ax.legend(
            title="Cluster",
            ncol=min(3, max(1, len(cluster_ids))),
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            frameon=False,
            fontsize=9,
            title_fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(stacked_bar_paper_path, bbox_inches="tight", transparent=True)
        plt.close(fig)
        _log_saved_figure(stacked_bar_paper_path)
        paper_outputs["stacked_bar_fraction_svg"] = str(stacked_bar_paper_path)

        counts_bar_paper_path = out_dir / "cluster_counts_stacked_bar_paper.svg"
        fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=220)
        bottom = np.zeros((len(frame_labels),), dtype=np.float64)
        for pos, cluster_id in enumerate(cluster_ids):
            ax.bar(
                x,
                counts_arr[:, pos],
                bottom=bottom,
                color=colors[pos],
                width=0.78,
                label=f"C{cluster_id}",
            )
            bottom += counts_arr[:, pos]
        ax.set_xticks(x)
        ax.set_xticklabels(frame_labels, rotation=0, ha="center")
        ax.set_ylabel("Count")
        _style_paper_axes(ax)
        ax.legend(
            title="Cluster",
            ncol=min(3, max(1, len(cluster_ids))),
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            frameon=False,
            fontsize=9,
            title_fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(counts_bar_paper_path, bbox_inches="tight", transparent=True)
        plt.close(fig)
        _log_saved_figure(counts_bar_paper_path)
        paper_outputs["stacked_bar_count_svg"] = str(counts_bar_paper_path)
        outputs["paper"] = paper_outputs
    return outputs


def save_spatial_cluster_view(
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    out_file: Path,
    *,
    title: str,
    cluster_color_map: dict[int, str] | None = None,
    visible_cluster_ids: list[int] | None = None,
    bounds: np.ndarray | None = None,
    max_points: int | None = None,
    point_size: float = 5.6,
    alpha: float = 0.62,
    saturation_boost: float = 1.18,
    view_elev: float = 24.0,
    view_azim: float = 35.0,
) -> dict[str, Any]:
    coords_arr = np.asarray(coords, dtype=np.float32)
    labels = np.asarray(cluster_labels, dtype=int).reshape(-1)
    if coords_arr.ndim != 2 or coords_arr.shape[1] < 3:
        raise ValueError(f"coords must have shape (N, >=3), got {coords_arr.shape}.")
    if labels.size != coords_arr.shape[0]:
        raise ValueError(
            "coords and cluster_labels length mismatch: "
            f"{coords_arr.shape[0]} vs {labels.size}."
        )

    coords_xyz = coords_arr[:, :3]
    mask = labels >= 0
    if visible_cluster_ids is not None:
        visible_ids = sorted(set(int(v) for v in visible_cluster_ids))
        if not visible_ids:
            raise ValueError("visible_cluster_ids resolved to an empty set.")
        mask &= np.isin(labels, np.asarray(visible_ids, dtype=int))
    else:
        visible_ids = None

    view_bounds: np.ndarray | None = None
    if bounds is not None:
        view_bounds = np.asarray(bounds, dtype=np.float32)
        if view_bounds.shape != (2, 3):
            raise ValueError(
                f"bounds must have shape (2, 3) with [[xmin,ymin,zmin],[xmax,ymax,zmax]], got {view_bounds.shape}."
            )
        mins = np.minimum(view_bounds[0], view_bounds[1])
        maxs = np.maximum(view_bounds[0], view_bounds[1])
        bounds_mask = np.all((coords_xyz >= mins[None, :]) & (coords_xyz <= maxs[None, :]), axis=1)
        mask &= bounds_mask
        view_bounds = np.stack([mins, maxs], axis=0)

    if not np.any(mask):
        raise ValueError("No points remained after applying spatial/cluster filters.")

    coords_visible = coords_xyz[mask]
    labels_visible = labels[mask]
    sample_idx = _sample_indices_stratified(labels_visible, max_points, random_seed=0)
    coords_plot = coords_visible[sample_idx]
    labels_plot = labels_visible[sample_idx]
    cluster_ids = _sorted_cluster_ids(labels_plot)
    colors = _cluster_palette_from_map(cluster_ids, cluster_color_map)
    color_lookup = {cluster_id: colors[pos] for pos, cluster_id in enumerate(cluster_ids)}

    fig = plt.figure(figsize=(7.8, 7.8), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for cluster_id in cluster_ids:
        cluster_mask = labels_plot == int(cluster_id)
        if not np.any(cluster_mask):
            continue
        cluster_points = coords_plot[cluster_mask]
        base_color = np.asarray(mcolors.to_rgb(color_lookup[int(cluster_id)]), dtype=np.float32)
        point_colors = np.repeat(base_color[None, :], cluster_points.shape[0], axis=0)
        point_colors = _boost_saturation(point_colors, float(saturation_boost))
        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            cluster_points[:, 2],
            c=point_colors,
            s=float(point_size),
            alpha=float(alpha),
            linewidths=0.0,
            depthshade=False,
        )

    if view_bounds is None:
        _set_equal_axes_3d(ax, coords_visible)
        wire_mins = np.min(coords_visible, axis=0)
        wire_maxs = np.max(coords_visible, axis=0)
    else:
        wire_mins = view_bounds[0]
        wire_maxs = view_bounds[1]
        center = 0.5 * (wire_mins + wire_maxs)
        span = float(np.max(wire_maxs - wire_mins))
        span = max(span, 1e-3)
        half = 0.5 * span
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect((1.0, 1.0, 1.0))
    _draw_cube_wireframe(ax, wire_mins, wire_maxs, linewidth=1.2)
    ax.view_init(elev=float(view_elev), azim=float(view_azim))
    ax.set_title(title, fontsize=13, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        if hasattr(axis, "pane"):
            axis.pane.fill = False
            axis.pane.set_edgecolor("white")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.95)

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    _log_saved_figure(out_file)
    return {
        "out_file": str(out_file),
        "num_points_total": int(coords_xyz.shape[0]),
        "num_points_visible": int(coords_visible.shape[0]),
        "num_points_rendered": int(coords_plot.shape[0]),
        "clusters_rendered": [int(v) for v in cluster_ids],
        "visible_cluster_ids": None if visible_ids is None else [int(v) for v in visible_ids],
        "bounds": None if view_bounds is None else view_bounds.astype(float).tolist(),
        "view_elev": float(view_elev),
        "view_azim": float(view_azim),
    }


def save_embedding_discrete_plot(
    embedding_2d: np.ndarray,
    labels: np.ndarray,
    out_file: Path,
    *,
    title: str,
    cluster_color_map: dict[int, str] | None = None,
    label_prefix: str = "C",
    point_size: float = 9.0,
    alpha: float = 0.72,
) -> dict[str, Any]:
    coords = np.asarray(embedding_2d, dtype=np.float32)
    values = np.asarray(labels)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"embedding_2d must have shape (N, 2), got {coords.shape}.")
    if values.shape[0] != coords.shape[0]:
        raise ValueError(
            "embedding_2d and labels length mismatch: "
            f"{coords.shape[0]} vs {values.shape[0]}."
        )
    cluster_ids = _sorted_cluster_ids(values)
    colors = _cluster_palette_from_map(cluster_ids, cluster_color_map)
    fig, ax = plt.subplots(figsize=(7.4, 6.4), dpi=220)
    for pos, cluster_id in enumerate(cluster_ids):
        mask = values == int(cluster_id)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=float(point_size),
            alpha=float(alpha),
            color=colors[pos],
            linewidths=0.0,
            label=f"{label_prefix}{cluster_id}",
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if len(cluster_ids) <= 16:
        ax.legend(title="label", markerscale=1.4, fontsize=8)
    fig.tight_layout()
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    _log_saved_figure(out_file)
    return {
        "out_file": str(out_file),
        "num_points": int(coords.shape[0]),
        "labels": [int(v) for v in cluster_ids],
    }


def save_embedding_continuous_plot(
    embedding_2d: np.ndarray,
    values: np.ndarray,
    out_file: Path,
    *,
    title: str,
    colorbar_label: str,
    point_size: float = 9.0,
    alpha: float = 0.80,
    cmap: str = "viridis",
) -> dict[str, Any]:
    coords = np.asarray(embedding_2d, dtype=np.float32)
    scalar = np.asarray(values, dtype=np.float32).reshape(-1)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"embedding_2d must have shape (N, 2), got {coords.shape}.")
    if scalar.shape[0] != coords.shape[0]:
        raise ValueError(
            "embedding_2d and values length mismatch: "
            f"{coords.shape[0]} vs {scalar.shape[0]}."
        )
    finite_mask = np.isfinite(scalar)
    if not finite_mask.any():
        raise ValueError("No finite scalar values available for continuous embedding plot.")

    fig, ax = plt.subplots(figsize=(7.4, 6.4), dpi=220)
    sc = ax.scatter(
        coords[finite_mask, 0],
        coords[finite_mask, 1],
        s=float(point_size),
        c=scalar[finite_mask],
        cmap=str(cmap),
        alpha=float(alpha),
        linewidths=0.0,
    )
    if (~finite_mask).any():
        ax.scatter(
            coords[~finite_mask, 0],
            coords[~finite_mask, 1],
            s=float(point_size),
            color="#d6d6d6",
            alpha=0.25,
            linewidths=0.0,
        )
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.03)
    cb.set_label(colorbar_label)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    _log_saved_figure(out_file)
    return {
        "out_file": str(out_file),
        "num_points": int(coords.shape[0]),
        "finite_fraction": float(np.mean(finite_mask)),
    }


def save_descriptor_violin_grid(
    descriptor_table: pd.DataFrame,
    *,
    cluster_column: str,
    scalar_columns: list[str],
    out_file: Path,
    cluster_color_map: dict[int, str] | None = None,
) -> dict[str, Any]:
    if descriptor_table.empty:
        raise ValueError("descriptor_table is empty.")
    if cluster_column not in descriptor_table.columns:
        raise KeyError(f"descriptor_table is missing required cluster column {cluster_column!r}.")
    if not scalar_columns:
        raise ValueError("scalar_columns must contain at least one descriptor name.")

    cluster_ids = _sorted_cluster_ids(descriptor_table[cluster_column].to_numpy())
    palette_colors = _cluster_palette_from_map(cluster_ids, cluster_color_map)
    palette = {
        int(cluster_id): str(palette_colors[pos])
        for pos, cluster_id in enumerate(cluster_ids)
    }
    n_cols = min(3, len(scalar_columns))
    n_rows = int(np.ceil(len(scalar_columns) / max(1, n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.6 * n_rows), dpi=220)
    axes_arr = np.asarray(axes).reshape(-1)
    for pos, column in enumerate(scalar_columns):
        ax = axes_arr[pos]
        if column not in descriptor_table.columns:
            raise KeyError(f"descriptor_table is missing scalar descriptor column {column!r}.")
        plot_df = descriptor_table[[cluster_column, column]].dropna().copy()
        if plot_df.empty:
            ax.axis("off")
            continue
        plot_df[cluster_column] = pd.to_numeric(plot_df[cluster_column], errors="raise").astype(int)
        sns.violinplot(
            data=plot_df,
            x=cluster_column,
            y=column,
            hue=cluster_column,
            ax=ax,
            order=cluster_ids,
            hue_order=cluster_ids,
            palette=palette,
            dodge=False,
            legend=False,
            cut=0,
            inner="quartile",
            linewidth=0.8,
        )
        ax.set_title(column)
        ax.set_xlabel("cluster")
        ax.grid(True, axis="y", alpha=0.18)
    for ax in axes_arr[len(scalar_columns):]:
        ax.axis("off")
    fig.tight_layout()
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    _log_saved_figure(out_file)
    return {
        "out_file": str(out_file),
        "scalar_columns": list(scalar_columns),
        "num_clusters": int(len(cluster_ids)),
    }


def save_cluster_feature_heatmap(
    feature_table: pd.DataFrame,
    out_file: Path,
    *,
    title: str,
    center: float | None = 0.0,
    annotate: bool = True,
    fmt: str = ".2f",
) -> dict[str, Any]:
    if feature_table.empty:
        raise ValueError("feature_table is empty.")
    fig, ax = plt.subplots(
        figsize=(1.0 * max(6, feature_table.shape[1]) + 1.5, 0.55 * max(4, feature_table.shape[0]) + 1.5),
        dpi=220,
    )
    sns.heatmap(
        feature_table,
        ax=ax,
        cmap="coolwarm",
        center=center,
        annot=annotate,
        fmt=fmt,
        cbar_kws={"shrink": 0.65},
    )
    ax.set_title(title)
    fig.tight_layout()
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    _log_saved_figure(out_file)
    return {
        "out_file": str(out_file),
        "shape": [int(feature_table.shape[0]), int(feature_table.shape[1])],
    }


def save_transition_heatmap(
    matrix: np.ndarray,
    out_file: Path,
    *,
    title: str,
    row_labels: list[str],
    col_labels: list[str] | None = None,
    fmt: str = ".2f",
) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"matrix must have shape (R, C), got {values.shape}.")
    if col_labels is None:
        col_labels = list(row_labels)
    if values.shape != (len(row_labels), len(col_labels)):
        raise ValueError(
            "matrix shape does not match row/col labels: "
            f"matrix={values.shape}, rows={len(row_labels)}, cols={len(col_labels)}."
        )
    fig, ax = plt.subplots(figsize=(0.95 * len(col_labels) + 2.2, 0.85 * len(row_labels) + 2.0), dpi=220)
    sns.heatmap(
        values,
        ax=ax,
        cmap="mako",
        annot=True,
        fmt=fmt,
        xticklabels=col_labels,
        yticklabels=row_labels,
        cbar_kws={"shrink": 0.72},
    )
    ax.set_title(title)
    ax.set_xlabel("to")
    ax.set_ylabel("from")
    fig.tight_layout()
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    _log_saved_figure(out_file)
    return {
        "out_file": str(out_file),
        "shape": [int(values.shape[0]), int(values.shape[1])],
    }


__all__ = [
    "save_cluster_proportion_plots",
    "save_spatial_cluster_view",
    "save_embedding_discrete_plot",
    "save_embedding_continuous_plot",
    "save_descriptor_violin_grid",
    "save_cluster_feature_heatmap",
    "save_transition_heatmap",
]
