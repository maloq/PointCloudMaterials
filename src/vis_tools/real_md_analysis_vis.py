from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
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


def _cluster_ids_from_labels(labels: list[str]) -> list[int]:
    cluster_ids: list[int] = []
    for pos, label in enumerate(labels):
        text = str(label).strip()
        if text.startswith("C") and text[1:].lstrip("+-").isdigit():
            cluster_ids.append(int(text[1:]))
        else:
            cluster_ids.append(int(pos))
    return cluster_ids


def _format_percent_share(value: float) -> str:
    share = float(value)
    if share < 0.0:
        raise ValueError(f"share must be non-negative, got {share}.")
    if share <= 0.0:
        return "0%"
    if share < 0.01:
        return "<1%"
    if share >= 0.10:
        return f"{100.0 * share:.0f}%"
    return f"{100.0 * share:.1f}%"


def _format_cna_plot_label(column_name: str) -> str:
    text = str(column_name)
    if text == "cna_other":
        return "other"
    if not text.startswith("cna_"):
        return text
    signature = text[len("cna_") :]
    return f"[{signature.replace('-', ' ')}]"


def save_cluster_proportion_plots(
    frame_labels: list[str],
    counts: np.ndarray,
    cluster_ids: list[int],
    out_dir: Path,
    *,
    cluster_color_map: dict[int, str] | None = None,
    cluster_display_labels: list[str] | None = None,
    save_paper_svg: bool = False,
    stack_alpha: float = 0.78,
    bar_alpha: float = 0.82,
    paper_alpha: float = 0.72,
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
    if cluster_display_labels is None:
        display_labels = [f"C{int(cluster_id)}" for cluster_id in cluster_ids]
    else:
        display_labels = [str(label) for label in cluster_display_labels]
        if len(display_labels) != len(cluster_ids):
            raise ValueError(
                "cluster_display_labels length does not match cluster_ids: "
                f"{len(display_labels)} vs {len(cluster_ids)}."
            )
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
    ax.stackplot(x, fractions.T, colors=colors, alpha=float(stack_alpha), linewidth=0.0)
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
    ax.legend(handles, display_labels, title="cluster", ncol=2)
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
            alpha=float(bar_alpha),
            edgecolor="white",
            linewidth=0.45,
            width=0.82,
            label=display_labels[pos],
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
            alpha=float(bar_alpha),
            edgecolor="white",
            linewidth=0.45,
            width=0.82,
            label=display_labels[pos],
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
        ax.stackplot(x, fractions.T, colors=colors, alpha=float(paper_alpha), linewidth=0.0)
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
            display_labels,
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
                alpha=float(paper_alpha),
                edgecolor="white",
                linewidth=0.35,
                width=0.78,
                label=display_labels[pos],
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
                alpha=float(paper_alpha),
                edgecolor="white",
                linewidth=0.35,
                width=0.78,
                label=display_labels[pos],
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


def _prepare_spatial_cluster_view_data(
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    visible_cluster_ids: list[int] | None = None,
    bounds: np.ndarray | None = None,
    max_points: int | None = None,
    allow_empty: bool = False,
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
        if not allow_empty:
            raise ValueError("No points remained after applying spatial/cluster filters.")
        if view_bounds is None:
            if coords_xyz.shape[0] == 0:
                raise ValueError("coords is empty; cannot prepare an empty spatial panel.")
            mins = np.min(coords_xyz, axis=0)
            maxs = np.max(coords_xyz, axis=0)
            view_bounds = np.stack([mins, maxs], axis=0)
        return {
            "coords_xyz": coords_xyz,
            "coords_visible": np.zeros((0, 3), dtype=np.float32),
            "labels_visible": np.zeros((0,), dtype=int),
            "coords_plot": np.zeros((0, 3), dtype=np.float32),
            "labels_plot": np.zeros((0,), dtype=int),
            "cluster_ids": [],
            "color_lookup": {},
            "view_bounds": view_bounds.astype(np.float32, copy=False),
            "visible_ids": None if visible_ids is None else [int(v) for v in visible_ids],
            "num_points_total": int(coords_xyz.shape[0]),
            "num_points_visible": 0,
            "num_points_rendered": 0,
            "empty": True,
        }

    coords_visible = coords_xyz[mask]
    labels_visible = labels[mask]
    sample_idx = _sample_indices_stratified(labels_visible, max_points, random_seed=0)
    coords_plot = coords_visible[sample_idx]
    labels_plot = labels_visible[sample_idx]
    cluster_ids = _sorted_cluster_ids(labels_plot)

    return {
        "coords_xyz": coords_xyz,
        "coords_visible": coords_visible,
        "labels_visible": labels_visible,
        "coords_plot": coords_plot,
        "labels_plot": labels_plot,
        "cluster_ids": [int(v) for v in cluster_ids],
        "view_bounds": None if view_bounds is None else view_bounds.astype(np.float32, copy=False),
        "visible_ids": None if visible_ids is None else [int(v) for v in visible_ids],
        "num_points_total": int(coords_xyz.shape[0]),
        "num_points_visible": int(coords_visible.shape[0]),
        "num_points_rendered": int(coords_plot.shape[0]),
        "empty": False,
    }


def _draw_spatial_cluster_view(
    ax: Any,
    view_data: dict[str, Any],
    *,
    cluster_color_map: dict[int, str] | None,
    point_size: float,
    alpha: float,
    saturation_boost: float,
    view_elev: float,
    view_azim: float,
    title: str | None,
) -> list[int]:
    cluster_ids = [int(v) for v in view_data["cluster_ids"]]
    colors = _cluster_palette_from_map(cluster_ids, cluster_color_map)
    color_lookup = {int(cluster_id): str(colors[pos]) for pos, cluster_id in enumerate(cluster_ids)}

    ax.set_facecolor("white")
    coords_plot = np.asarray(view_data["coords_plot"], dtype=np.float32)
    labels_plot = np.asarray(view_data["labels_plot"], dtype=int)
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

    view_bounds = view_data["view_bounds"]
    coords_visible = np.asarray(view_data["coords_visible"], dtype=np.float32)
    if view_bounds is None:
        _set_equal_axes_3d(ax, coords_visible)
        wire_mins = np.min(coords_visible, axis=0)
        wire_maxs = np.max(coords_visible, axis=0)
    else:
        wire_mins = np.asarray(view_bounds[0], dtype=np.float32)
        wire_maxs = np.asarray(view_bounds[1], dtype=np.float32)
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

    if bool(view_data["empty"]):
        ax.text2D(
            0.50,
            0.50,
            "no points",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#6f6f6f",
        )

    ax.view_init(elev=float(view_elev), azim=float(view_azim))
    if title:
        ax.set_title(str(title), fontsize=13, pad=6)
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
    return cluster_ids


def save_spatial_cluster_view(
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    out_file: Path,
    *,
    title: str | None,
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
    view_data = _prepare_spatial_cluster_view_data(
        coords,
        cluster_labels,
        visible_cluster_ids=visible_cluster_ids,
        bounds=bounds,
        max_points=max_points,
        allow_empty=False,
    )

    fig = plt.figure(figsize=(7.8, 7.8), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    cluster_ids = _draw_spatial_cluster_view(
        ax,
        view_data,
        cluster_color_map=cluster_color_map,
        point_size=float(point_size),
        alpha=float(alpha),
        saturation_boost=float(saturation_boost),
        view_elev=float(view_elev),
        view_azim=float(view_azim),
        title=title,
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.95)

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    _log_saved_figure(out_file)
    return {
        "out_file": str(out_file),
        "num_points_total": int(view_data["num_points_total"]),
        "num_points_visible": int(view_data["num_points_visible"]),
        "num_points_rendered": int(view_data["num_points_rendered"]),
        "clusters_rendered": [int(v) for v in cluster_ids],
        "visible_cluster_ids": None if view_data["visible_ids"] is None else [int(v) for v in view_data["visible_ids"]],
        "bounds": None if view_data["view_bounds"] is None else np.asarray(view_data["view_bounds"], dtype=np.float32).astype(float).tolist(),
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
    display_label_map: dict[int, str] | None = None,
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
            label=(
                str(display_label_map[int(cluster_id)])
                if display_label_map is not None and int(cluster_id) in display_label_map
                else f"{label_prefix}{cluster_id}"
            ),
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
    background_alpha: float = 0.08,
    cmap: str = "viridis",
    value_limits: tuple[float, float] | None = None,
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
    finite_values = scalar[finite_mask]
    if value_limits is None:
        if finite_values.size >= 8:
            vmin, vmax = np.nanpercentile(finite_values, [2.0, 98.0])
        else:
            vmin = float(np.nanmin(finite_values))
            vmax = float(np.nanmax(finite_values))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
            vmin = float(np.nanmin(finite_values))
            vmax = float(np.nanmax(finite_values))
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1e-6
    else:
        vmin = float(value_limits[0])
        vmax = float(value_limits[1])

    fig, ax = plt.subplots(figsize=(7.4, 6.4), dpi=220)
    if (~finite_mask).any():
        ax.scatter(
            coords[~finite_mask, 0],
            coords[~finite_mask, 1],
            s=float(point_size),
            color="#d6d6d6",
            alpha=float(background_alpha),
            linewidths=0.0,
        )
    sc = ax.scatter(
        coords[finite_mask, 0],
        coords[finite_mask, 1],
        s=float(point_size),
        c=scalar[finite_mask],
        cmap=str(cmap),
        alpha=float(alpha),
        linewidths=0.0,
        vmin=float(vmin),
        vmax=float(vmax),
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
    cluster_label_map: dict[int, str] | None = None,
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
        if cluster_label_map is not None:
            ax.set_xticks(np.arange(len(cluster_ids)))
            ax.set_xticklabels([str(cluster_label_map[int(cluster_id)]) for cluster_id in cluster_ids])
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


def save_cna_cluster_signature_stacked_bar(
    signature_table: pd.DataFrame,
    *,
    out_file: Path,
    cluster_color_map: dict[int, str] | None = None,
    cluster_label_map: dict[int, str] | None = None,
    save_svg: bool = False,
) -> dict[str, Any]:
    if signature_table.empty:
        raise ValueError("signature_table is empty.")
    if "cluster_id" not in signature_table.columns:
        raise KeyError("signature_table must contain 'cluster_id'.")
    signature_columns = [
        str(column)
        for column in signature_table.columns
        if column != "cluster_id"
    ]
    if not signature_columns:
        raise ValueError("signature_table must contain at least one CNA signature column.")

    table = signature_table.copy()
    table["cluster_id"] = pd.to_numeric(table["cluster_id"], errors="raise").astype(int)
    cluster_ids = _sorted_cluster_ids(table["cluster_id"].to_numpy())
    grouped = (
        table.groupby("cluster_id", sort=True)[signature_columns]
        .mean(numeric_only=True)
        .reindex(cluster_ids)
        .fillna(0.0)
    )
    values = grouped.to_numpy(dtype=np.float64)
    row_sums = values.sum(axis=1, keepdims=True)
    values = np.divide(
        values,
        row_sums,
        out=np.zeros_like(values),
        where=row_sums > 0.0,
    )

    colors = _cluster_palette_from_map(list(range(len(signature_columns))), None)
    signature_color_map = {
        str(column): ("#c8c8c8" if str(column) == "cna_other" else str(colors[pos]))
        for pos, column in enumerate(signature_columns)
    }
    x = np.arange(len(cluster_ids), dtype=np.float64)
    display_labels = [
        str(cluster_label_map[int(cluster_id)])
        if cluster_label_map is not None and int(cluster_id) in cluster_label_map
        else f"C{int(cluster_id)}"
        for cluster_id in cluster_ids
    ]

    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=220)
    bottom = np.zeros((len(cluster_ids),), dtype=np.float64)
    for col_idx, column in enumerate(signature_columns):
        ax.bar(
            x,
            values[:, col_idx],
            bottom=bottom,
            color=signature_color_map[str(column)],
            edgecolor="white",
            linewidth=0.4,
            width=0.80,
            label=_format_cna_plot_label(str(column)),
        )
        bottom += values[:, col_idx]
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean CNA fraction")
    ax.set_title("CNA signature composition by cluster")
    ax.grid(True, axis="y", alpha=0.18)
    ax.legend(
        title="signature",
        ncol=min(4, max(1, len(signature_columns))),
        fontsize=8,
        title_fontsize=9,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
    )
    fig.tight_layout()
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    _log_saved_figure(out_file)

    result = {
        "out_file": str(out_file),
        "cluster_ids": [int(v) for v in cluster_ids],
        "signature_columns": list(signature_columns),
    }
    if save_svg:
        svg_path = out_file.with_suffix(".svg")
        fig, ax = plt.subplots(figsize=(7.0, 3.1), dpi=220)
        bottom = np.zeros((len(cluster_ids),), dtype=np.float64)
        for col_idx, column in enumerate(signature_columns):
            ax.bar(
                x,
                values[:, col_idx],
                bottom=bottom,
                color=signature_color_map[str(column)],
                edgecolor="white",
                linewidth=0.35,
                width=0.76,
                label=_format_cna_plot_label(str(column)),
            )
            bottom += values[:, col_idx]
        ax.set_xticks(x)
        ax.set_xticklabels(display_labels)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Fraction")
        _style_paper_axes(ax)
        ax.legend(
            title="CNA",
            ncol=min(4, max(1, len(signature_columns))),
            fontsize=8,
            title_fontsize=9,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
        )
        fig.tight_layout()
        fig.savefig(svg_path, bbox_inches="tight", transparent=True)
        plt.close(fig)
        _log_saved_figure(svg_path)
        result["svg"] = str(svg_path)
    return result


def save_cna_signature_time_series(
    frame_labels: list[str],
    signature_matrix: np.ndarray,
    *,
    signature_labels: list[str],
    out_file: Path,
    save_svg: bool = False,
) -> dict[str, Any]:
    values = np.asarray(signature_matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(
            "signature_matrix must have shape (num_frames, num_signatures), "
            f"got {values.shape}."
        )
    if values.shape != (len(frame_labels), len(signature_labels)):
        raise ValueError(
            "signature_matrix shape does not match labels: "
            f"matrix={values.shape}, frames={len(frame_labels)}, signatures={len(signature_labels)}."
        )

    row_sums = values.sum(axis=1, keepdims=True)
    fractions = np.divide(
        values,
        row_sums,
        out=np.zeros_like(values),
        where=row_sums > 0.0,
    )
    colors = _cluster_palette_from_map(list(range(len(signature_labels))), None)
    signature_colors = [
        "#c8c8c8" if str(label) == "other" else str(colors[pos])
        for pos, label in enumerate(signature_labels)
    ]
    display_signature_labels = [
        _format_cna_plot_label(
            "cna_other"
            if str(label) == "other"
            else str(label)
            if str(label).startswith("cna_")
            else f"cna_{label}"
        )
        for label in signature_labels
    ]
    x = np.arange(len(frame_labels), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10.6, 4.6), dpi=220)
    ax.stackplot(x, fractions.T, colors=signature_colors, alpha=0.82, linewidth=0.0)
    ax.set_xlim(x[0] if len(x) > 1 else -0.5, x[-1] if len(x) > 1 else 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(frame_labels, rotation=35, ha="right")
    ax.set_ylabel("Mean CNA fraction")
    ax.set_title("CNA signatures across time")
    ax.grid(True, axis="y", alpha=0.18)
    handles = [
        plt.Line2D([0], [0], color=signature_colors[pos], linewidth=6)
        for pos in range(len(signature_labels))
    ]
    ax.legend(
        handles,
        display_signature_labels,
        title="signature",
        ncol=min(4, max(1, len(signature_labels))),
        fontsize=8,
        title_fontsize=9,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
    )
    fig.tight_layout()
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    _log_saved_figure(out_file)

    result = {
        "out_file": str(out_file),
        "frame_labels": [str(v) for v in frame_labels],
        "signature_labels": [str(v) for v in signature_labels],
    }
    if save_svg:
        svg_path = out_file.with_suffix(".svg")
        fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=220)
        ax.stackplot(x, fractions.T, colors=signature_colors, alpha=0.74, linewidth=0.0)
        ax.set_xlim(x[0] if len(x) > 1 else -0.5, x[-1] if len(x) > 1 else 0.5)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(frame_labels, rotation=0, ha="center")
        ax.set_ylabel("Fraction")
        _style_paper_axes(ax)
        handles = [
            plt.Line2D([0], [0], color=signature_colors[pos], linewidth=5.5)
            for pos in range(len(signature_labels))
        ]
        ax.legend(
            handles,
            display_signature_labels,
            title="CNA",
            ncol=min(4, max(1, len(signature_labels))),
            fontsize=8,
            title_fontsize=9,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
        )
        fig.tight_layout()
        fig.savefig(svg_path, bbox_inches="tight", transparent=True)
        plt.close(fig)
        _log_saved_figure(svg_path)
        result["svg"] = str(svg_path)
    return result


def _allocate_flow_spans(
    weights: np.ndarray,
    *,
    gap: float,
) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.ndim != 1:
        raise ValueError(f"weights must be 1D, got {values.shape}.")
    positive = values > 0.0
    spans = np.full((values.size, 2), np.nan, dtype=np.float64)
    positive_count = int(np.sum(positive))
    if positive_count == 0:
        return spans
    total = float(np.sum(values[positive]))
    usable_height = 1.0 - gap * max(0, positive_count - 1)
    if usable_height <= 0.0:
        raise ValueError(
            "Transition flow gap is too large for the number of active clusters: "
            f"gap={gap}, active_clusters={positive_count}."
        )
    scale = usable_height / total
    y_top = 1.0
    for idx, value in enumerate(values):
        if value <= 0.0:
            continue
        height = float(value) * scale
        y_bottom = y_top - height
        spans[idx] = [y_bottom, y_top]
        y_top = y_bottom - gap
    return spans


def save_transition_flow_plot(
    matrix: np.ndarray,
    out_file: Path,
    *,
    title: str | None = None,
    row_labels: list[str],
    col_labels: list[str] | None = None,
    cluster_color_map: dict[int, str] | None = None,
    cluster_ids_for_palette: list[int] | None = None,
    mute_diagonal: bool = True,
    min_draw_fraction: float = 0.0,
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
    total_flow = float(np.sum(values))
    if total_flow <= 0.0:
        raise ValueError("Transition flow plot requires a matrix with positive total weight.")

    row_weights = np.sum(values, axis=1)
    col_weights = np.sum(values, axis=0)
    gap = 0.028
    row_spans = _allocate_flow_spans(row_weights, gap=gap)
    col_spans = _allocate_flow_spans(col_weights, gap=gap)
    row_cursor = row_spans[:, 1].copy()
    col_cursor = col_spans[:, 1].copy()

    palette_cluster_ids = (
        [int(v) for v in cluster_ids_for_palette]
        if cluster_ids_for_palette is not None
        else _cluster_ids_from_labels(row_labels)
    )
    colors = _cluster_palette_from_map(palette_cluster_ids, cluster_color_map)
    color_lookup = {
        int(idx): str(colors[idx])
        for idx in range(len(cluster_ids_for_palette))
    }

    fig, ax = plt.subplots(figsize=(8.4, 6.2), dpi=220)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(top=0.95, bottom=0.10)
    ax.set_facecolor("white")
    x_left0, x_left1 = 0.10, 0.20
    x_right0, x_right1 = 0.80, 0.90
    x_curve_left = 0.34
    x_curve_right = 0.66

    for idx, (y0, y1) in enumerate(row_spans):
        if not np.isfinite(y0):
            continue
        color = color_lookup[int(idx)]
        ax.add_patch(
            Rectangle(
                (x_left0, y0),
                x_left1 - x_left0,
                y1 - y0,
                facecolor=color,
                edgecolor="white",
                linewidth=0.9,
                alpha=0.95,
            )
        )
        label_text = f"{row_labels[idx]}  {_format_percent_share(row_weights[idx] / total_flow)}"
        ax.text(x_left0 - 0.02, 0.5 * (y0 + y1), label_text, ha="right", va="center", fontsize=10)

    for idx, (y0, y1) in enumerate(col_spans):
        if not np.isfinite(y0):
            continue
        color = color_lookup[int(idx)]
        ax.add_patch(
            Rectangle(
                (x_right0, y0),
                x_right1 - x_right0,
                y1 - y0,
                facecolor=color,
                edgecolor="white",
                linewidth=0.9,
                alpha=0.95,
            )
        )
        label_text = f"{col_labels[idx]}  {_format_percent_share(col_weights[idx] / total_flow)}"
        ax.text(x_right1 + 0.02, 0.5 * (y0 + y1), label_text, ha="left", va="center", fontsize=10)

    min_draw_value = float(min_draw_fraction) * total_flow
    positive_weights = values[values > 0.0]
    max_weight = float(np.max(positive_weights)) if positive_weights.size else 1.0
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            weight = float(values[row_idx, col_idx])
            if weight <= 0.0:
                continue
            if weight < min_draw_value:
                continue
            left_height = (row_spans[row_idx, 1] - row_spans[row_idx, 0]) * (weight / max(row_weights[row_idx], 1e-12))
            right_height = (col_spans[col_idx, 1] - col_spans[col_idx, 0]) * (weight / max(col_weights[col_idx], 1e-12))
            left_y1 = row_cursor[row_idx]
            left_y0 = left_y1 - left_height
            row_cursor[row_idx] = left_y0
            right_y1 = col_cursor[col_idx]
            right_y0 = right_y1 - right_height
            col_cursor[col_idx] = right_y0

            flow_color = color_lookup[int(row_idx)]
            weight_norm = np.sqrt(weight / max(max_weight, 1e-12))
            if row_idx == col_idx:
                alpha = 0.16 + 0.26 * weight_norm if mute_diagonal else 0.24 + 0.28 * weight_norm
                edge_alpha = 0.03 + 0.08 * weight_norm
            else:
                alpha = 0.06 + 0.82 * weight_norm
                edge_alpha = 0.04 + 0.18 * weight_norm

            path = MplPath(
                [
                    (x_left1, left_y0),
                    (x_curve_left, left_y0),
                    (x_curve_right, right_y0),
                    (x_right0, right_y0),
                    (x_right0, right_y1),
                    (x_curve_right, right_y1),
                    (x_curve_left, left_y1),
                    (x_left1, left_y1),
                    (x_left1, left_y0),
                ],
                [
                    MplPath.MOVETO,
                    MplPath.CURVE4,
                    MplPath.CURVE4,
                    MplPath.CURVE4,
                    MplPath.LINETO,
                    MplPath.CURVE4,
                    MplPath.CURVE4,
                    MplPath.CURVE4,
                    MplPath.CLOSEPOLY,
                ],
            )
            ax.add_patch(
                PathPatch(
                    path,
                    facecolor=flow_color,
                    edgecolor=mcolors.to_rgba(flow_color, alpha=edge_alpha),
                    linewidth=0.5,
                    alpha=alpha,
                )
            )
    if title:
        fig.suptitle(str(title), fontsize=15, y=0.975)
    ax.text(x_left0, 1.008, "from", ha="left", va="bottom", fontsize=10, weight="bold")
    ax.text(x_right1, 1.008, "to", ha="right", va="bottom", fontsize=10, weight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_file,
        bbox_inches="tight",
        transparent=out_file.suffix.lower() == ".svg",
    )
    plt.close(fig)
    _log_saved_figure(out_file)
    return {
        "out_file": str(out_file),
        "shape": [int(values.shape[0]), int(values.shape[1])],
        "total_weight": float(total_flow),
        "mute_diagonal": bool(mute_diagonal),
    }


__all__ = [
    "save_cluster_proportion_plots",
    "save_spatial_cluster_view",
    "save_embedding_discrete_plot",
    "save_embedding_continuous_plot",
    "save_descriptor_violin_grid",
    "save_cna_cluster_signature_stacked_bar",
    "save_cna_signature_time_series",
    "save_transition_flow_plot",
]
