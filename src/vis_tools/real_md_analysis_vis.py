from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis.cluster_colors import _boost_saturation
from src.analysis.cluster_geometry import (
    _draw_cube_wireframe,
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

    outputs: dict[str, Any] = {
        "stacked_area": str(stacked_area_path),
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

        outputs["paper"] = paper_outputs
    return outputs


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




_DIAGONAL_DIRECTION = np.asarray([1.0, 1.0, 1.0], dtype=np.float32) / np.sqrt(3.0)


def _render_figure_to_rgb_array(fig: Any) -> np.ndarray:
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    return image[..., :3].copy()


def _set_3d_scatter_offsets(collection: Any, coords: np.ndarray) -> None:
    coords_arr = np.asarray(coords, dtype=np.float32)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError(
            "3D scatter updates require coords with shape (N, 3), "
            f"got {coords_arr.shape}."
        )
    # Matplotlib's 3D scatter API exposes coordinate updates through this
    # private attribute rather than a public setter.
    collection._offsets3d = (
        coords_arr[:, 0],
        coords_arr[:, 1],
        coords_arr[:, 2],
    )


def _save_rgb_frames_as_gif(
    frames_rgb: list[np.ndarray],
    out_file: Path,
    *,
    frame_duration_ms: int | None = None,
    total_duration_seconds: float | None = None,
) -> None:
    if not frames_rgb:
        raise ValueError("frames_rgb must be a non-empty list.")
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Temporal animation export requires Pillow. Install the project requirements."
        ) from exc
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(len(frames_rgb))
    if total_duration_seconds is not None:
        total_seconds = float(total_duration_seconds)
        if total_seconds <= 0.0:
            raise ValueError(
                f"total_duration_seconds must be > 0, got {total_seconds}."
            )
        total_centiseconds = int(round(total_seconds * 100.0))
        if total_centiseconds < frame_count:
            raise ValueError(
                "total_duration_seconds is too short to encode all frames while keeping every frame. "
                f"total_duration_seconds={total_seconds}, frame_count={frame_count}."
            )
        base_centiseconds = total_centiseconds // frame_count
        remainder_centiseconds = total_centiseconds % frame_count
        durations = [
            10 * (base_centiseconds + (1 if idx < remainder_centiseconds else 0))
            for idx in range(frame_count)
        ]
    else:
        if frame_duration_ms is None:
            raise ValueError(
                "Either frame_duration_ms or total_duration_seconds must be provided."
            )
        if int(frame_duration_ms) <= 0:
            raise ValueError(f"frame_duration_ms must be > 0, got {frame_duration_ms}.")
        durations = [int(frame_duration_ms)] * frame_count
    images = [Image.fromarray(frame_rgb) for frame_rgb in frames_rgb]
    images[0].save(
        out_file,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
    )
    _log_saved_figure(out_file)


def _resolve_equalized_bounds(bounds: np.ndarray) -> np.ndarray:
    bounds_arr = np.asarray(bounds, dtype=np.float32)
    if bounds_arr.shape != (2, 3):
        raise ValueError(f"bounds must have shape (2, 3), got {bounds_arr.shape}.")
    mins = np.minimum(bounds_arr[0], bounds_arr[1])
    maxs = np.maximum(bounds_arr[0], bounds_arr[1])
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    span = max(span, 1e-6)
    half = 0.5 * span
    return np.stack([center - half, center + half], axis=0).astype(np.float32, copy=False)


def _resolve_global_spatial_bounds(frame_records: list[dict[str, Any]]) -> np.ndarray:
    all_coords = [
        np.asarray(record["coords"], dtype=np.float32)
        for record in frame_records
        if np.asarray(record["coords"]).size > 0
    ]
    if not all_coords:
        raise ValueError("Temporal spatial animation requires at least one non-empty frame.")
    stacked = np.concatenate(all_coords, axis=0)
    mins = np.min(stacked[:, :3], axis=0)
    maxs = np.max(stacked[:, :3], axis=0)
    return _resolve_equalized_bounds(np.stack([mins, maxs], axis=0))


def _plane_box_intersection_polygon(
    *,
    bounds: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    bounds_arr = np.asarray(bounds, dtype=np.float32)
    mins = np.asarray(bounds_arr[0], dtype=np.float32)
    maxs = np.asarray(bounds_arr[1], dtype=np.float32)
    vertices = np.asarray(
        [
            [mins[0], mins[1], mins[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], maxs[1], mins[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], maxs[2]],
            [mins[0], maxs[1], maxs[2]],
        ],
        dtype=np.float32,
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    signed = (vertices - plane_point[None, :]) @ plane_normal
    points: list[np.ndarray] = []
    for start, stop in edges:
        value_a = float(signed[start])
        value_b = float(signed[stop])
        point_a = vertices[start]
        point_b = vertices[stop]
        if abs(value_a) < 1e-6:
            points.append(point_a)
        if abs(value_b) < 1e-6:
            points.append(point_b)
        if value_a * value_b < 0.0:
            alpha = value_a / (value_a - value_b)
            points.append(point_a + alpha * (point_b - point_a))
    unique_points: list[np.ndarray] = []
    for point in points:
        if not any(np.linalg.norm(point - existing) < 1e-5 for existing in unique_points):
            unique_points.append(np.asarray(point, dtype=np.float32))
    if len(unique_points) < 3:
        return np.zeros((0, 3), dtype=np.float32)
    polygon = np.asarray(unique_points, dtype=np.float32)
    centroid = np.mean(polygon, axis=0)
    normal = np.asarray(plane_normal, dtype=np.float32)
    normal = normal / np.linalg.norm(normal)
    tangent_a = np.asarray([1.0, -1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(tangent_a, normal))) > 0.95:
        tangent_a = np.asarray([1.0, 0.0, -1.0], dtype=np.float32)
    tangent_a = tangent_a - normal * float(np.dot(tangent_a, normal))
    tangent_a = tangent_a / np.linalg.norm(tangent_a)
    tangent_b = np.cross(normal, tangent_a).astype(np.float32)
    rel = polygon - centroid[None, :]
    u = rel @ tangent_a
    v = rel @ tangent_b
    order = np.argsort(np.arctan2(v, u))
    return polygon[order]


def _draw_diagonal_cut_plane(ax: Any, *, bounds: np.ndarray) -> None:
    bounds_arr = np.asarray(bounds, dtype=np.float32)
    center = 0.5 * (bounds_arr[0] + bounds_arr[1])
    polygon = _plane_box_intersection_polygon(
        bounds=bounds_arr,
        plane_point=center,
        plane_normal=_DIAGONAL_DIRECTION,
    )
    if polygon.shape[0] < 3:
        return
    poly = Poly3DCollection(
        [polygon],
        facecolor="#CBD5E1",
        edgecolor="#475569",
        linewidths=1.0,
        alpha=0.08,
    )
    ax.add_collection3d(poly)
    wrapped = np.vstack([polygon, polygon[0]])
    ax.plot(
        wrapped[:, 0],
        wrapped[:, 1],
        wrapped[:, 2],
        color="#475569",
        linewidth=1.1,
        alpha=0.50,
    )


def _select_diagonal_cut_points(
    coords: np.ndarray,
    labels: np.ndarray,
    *,
    bounds: np.ndarray,
    visible_depth_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    coords_arr = np.asarray(coords, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=int).reshape(-1)
    if coords_arr.ndim != 2 or coords_arr.shape[1] < 3:
        raise ValueError(f"coords must have shape (N, >=3), got {coords_arr.shape}.")
    if coords_arr.shape[0] != labels_arr.shape[0]:
        raise ValueError(
            "coords and labels length mismatch for diagonal cut selection: "
            f"{coords_arr.shape[0]} vs {labels_arr.shape[0]}."
        )
    if coords_arr.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=int)
    if visible_depth_fraction <= 0.0:
        raise ValueError(
            f"visible_depth_fraction must be positive, got {visible_depth_fraction}."
        )
    bounds_arr = np.asarray(bounds, dtype=np.float32)
    center = 0.5 * (bounds_arr[0] + bounds_arr[1])
    span = np.maximum(bounds_arr[1] - bounds_arr[0], 1e-6)
    normalized = (coords_arr[:, :3] - center[None, :]) / span[None, :]
    signed = normalized @ _DIAGONAL_DIRECTION
    mask = signed <= 0.0
    near_plane = signed >= -float(visible_depth_fraction)
    if int(np.count_nonzero(mask & near_plane)) >= min(2000, coords_arr.shape[0]):
        mask &= near_plane
    return coords_arr[mask, :3].astype(np.float32, copy=False), labels_arr[mask].astype(int, copy=False)


def save_temporal_spatial_cluster_animation(
    frame_records: list[dict[str, Any]],
    out_file: Path,
    *,
    cluster_color_map: dict[int, str] | None = None,
    cluster_display_map: dict[int, str] | None = None,
    point_size: float = 5.6,
    alpha: float = 0.62,
    saturation_boost: float = 1.18,
    view_elev: float = 24.0,
    view_azim: float = 35.0,
    diagonal_visible_depth_fraction: float = 0.10,
    frame_duration_ms: int = 450,
    total_duration_seconds: float | None = None,
) -> dict[str, Any]:
    if not frame_records:
        raise ValueError("frame_records must be a non-empty list.")

    global_bounds = _resolve_global_spatial_bounds(frame_records)
    cluster_ids = _sorted_cluster_ids(
        np.concatenate(
            [
                np.asarray(record["labels"], dtype=int).reshape(-1)
                for record in frame_records
            ]
        )
    )
    colors = _cluster_palette_from_map(cluster_ids, cluster_color_map)
    color_lookup = {
        int(cluster_id): str(colors[pos])
        for pos, cluster_id in enumerate(cluster_ids)
    }
    display_labels = {
        int(cluster_id): (
            str(cluster_display_map[int(cluster_id)])
            if cluster_display_map is not None and int(cluster_id) in cluster_display_map
            else f"C{int(cluster_id)}"
        )
        for cluster_id in cluster_ids
    }

    boosted_color_lookup: dict[int, tuple[float, float, float]] = {}
    for cluster_id in cluster_ids:
        boosted = _boost_saturation(
            np.asarray([mcolors.to_rgb(color_lookup[int(cluster_id)])], dtype=np.float32),
            float(saturation_boost),
        )[0]
        boosted_color_lookup[int(cluster_id)] = (
            float(boosted[0]),
            float(boosted[1]),
            float(boosted[2]),
        )

    empty_xyz = np.zeros((0, 3), dtype=np.float32)
    selected_frame_records: list[dict[str, Any]] = []
    for record in frame_records:
        frame_coords, frame_labels = _select_diagonal_cut_points(
            np.asarray(record["coords"], dtype=np.float32),
            np.asarray(record["labels"], dtype=int),
            bounds=global_bounds,
            visible_depth_fraction=float(diagonal_visible_depth_fraction),
        )
        selected_frame_records.append(
            {
                "frame_label": str(record["frame_label"]),
                "coords": np.asarray(frame_coords, dtype=np.float32),
                "labels": np.asarray(frame_labels, dtype=int),
            }
        )

    frames_rgb: list[np.ndarray] = []
    fig = plt.figure(figsize=(9.6, 7.2), dpi=180)
    try:
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0.04, 0.08, 0.66, 0.84], projection="3d")
        legend_ax = fig.add_axes([0.74, 0.16, 0.24, 0.64])
        legend_ax.axis("off")
        ax.set_facecolor("white")

        scatter_by_cluster: dict[int, Any] = {}
        for cluster_id in cluster_ids:
            scatter_by_cluster[int(cluster_id)] = ax.scatter(
                empty_xyz[:, 0],
                empty_xyz[:, 1],
                empty_xyz[:, 2],
                color=boosted_color_lookup[int(cluster_id)],
                s=float(point_size),
                alpha=float(alpha),
                linewidths=0.0,
                depthshade=False,
            )

        no_points_text = ax.text2D(
            0.50,
            0.50,
            "no points",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#6f6f6f",
        )
        no_points_text.set_visible(False)

        _draw_cube_wireframe(ax, global_bounds[0], global_bounds[1], linewidth=1.0, alpha=0.30, color="#475569")
        _draw_diagonal_cut_plane(ax, bounds=global_bounds)
        center = 0.5 * (global_bounds[0] + global_bounds[1])
        span = float(np.max(global_bounds[1] - global_bounds[0]))
        half = 0.5 * max(span, 1e-6)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.view_init(elev=float(view_elev), azim=float(view_azim))
        ax.grid(False)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xticks(np.linspace(center[0] - half, center[0] + half, num=3))
        ax.set_yticks(np.linspace(center[1] - half, center[1] + half, num=3))
        ax.set_zticks(np.linspace(center[2] - half, center[2] + half, num=3))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            if hasattr(axis, "pane"):
                axis.pane.fill = False
                axis.pane.set_edgecolor("white")
        legend_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=color_lookup[int(cluster_id)],
                markerfacecolor=color_lookup[int(cluster_id)],
                markersize=7,
                label=display_labels[int(cluster_id)],
            )
            for cluster_id in cluster_ids
        ]
        legend_ax.legend(
            legend_handles,
            [display_labels[int(cluster_id)] for cluster_id in cluster_ids],
            title="cluster",
            loc="upper left",
            frameon=False,
        )
        title_artist = fig.suptitle("", fontsize=13)

        for record in selected_frame_records:
            frame_coords = np.asarray(record["coords"], dtype=np.float32)
            frame_labels = np.asarray(record["labels"], dtype=int)
            for cluster_id in cluster_ids:
                cluster_points = frame_coords[frame_labels == int(cluster_id)]
                _set_3d_scatter_offsets(
                    scatter_by_cluster[int(cluster_id)],
                    cluster_points if cluster_points.size > 0 else empty_xyz,
                )
            no_points_text.set_visible(frame_coords.shape[0] == 0)
            title_artist.set_text(f"MD-space clusters | {record['frame_label']}")
            frames_rgb.append(_render_figure_to_rgb_array(fig))
    finally:
        plt.close(fig)

    out_file = Path(out_file)
    _save_rgb_frames_as_gif(
        frames_rgb,
        out_file,
        frame_duration_ms=int(frame_duration_ms),
        total_duration_seconds=total_duration_seconds,
    )
    return {
        "out_file": str(out_file),
        "frame_count": int(len(frame_records)),
        "cluster_ids": [int(v) for v in cluster_ids],
        "bounds": global_bounds.astype(float).tolist(),
    }


def save_temporal_embedding_cluster_animation(
    frame_records: list[dict[str, Any]],
    out_file: Path,
    *,
    cluster_color_map: dict[int, str] | None = None,
    cluster_display_map: dict[int, str] | None = None,
    point_size: float = 8.0,
    alpha: float = 0.74,
    frame_duration_ms: int = 450,
    total_duration_seconds: float | None = None,
    title: str = "Latent cluster evolution",
    xlabel: str = "UMAP 1",
    ylabel: str = "UMAP 2",
) -> dict[str, Any]:
    if not frame_records:
        raise ValueError("frame_records must be a non-empty list.")

    all_embeddings = np.concatenate(
        [
            np.asarray(record["embedding"], dtype=np.float32)
            for record in frame_records
            if np.asarray(record["embedding"]).size > 0
        ],
        axis=0,
    )
    if all_embeddings.size == 0:
        raise ValueError("Temporal embedding animation requires at least one non-empty frame.")
    if all_embeddings.ndim != 2 or all_embeddings.shape[1] != 2:
        raise ValueError(
            "Temporal embedding animation expects 2D embeddings for every frame, "
            f"got stacked shape {all_embeddings.shape}."
        )
    cluster_ids = _sorted_cluster_ids(
        np.concatenate(
            [
                np.asarray(record["labels"], dtype=int).reshape(-1)
                for record in frame_records
            ]
        )
    )
    colors = _cluster_palette_from_map(cluster_ids, cluster_color_map)
    color_lookup = {
        int(cluster_id): str(colors[pos])
        for pos, cluster_id in enumerate(cluster_ids)
    }
    display_labels = {
        int(cluster_id): (
            str(cluster_display_map[int(cluster_id)])
            if cluster_display_map is not None and int(cluster_id) in cluster_display_map
            else f"C{int(cluster_id)}"
        )
        for cluster_id in cluster_ids
    }
    mins = np.min(all_embeddings, axis=0)
    maxs = np.max(all_embeddings, axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = 0.06 * span
    x_limits = (float(mins[0] - pad[0]), float(maxs[0] + pad[0]))
    y_limits = (float(mins[1] - pad[1]), float(maxs[1] + pad[1]))

    empty_xy = np.zeros((0, 2), dtype=np.float32)
    frames_rgb: list[np.ndarray] = []
    fig = plt.figure(figsize=(9.0, 6.8), dpi=180)
    try:
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0.08, 0.12, 0.64, 0.78])
        legend_ax = fig.add_axes([0.76, 0.18, 0.22, 0.60])
        legend_ax.axis("off")

        scatter_by_cluster: dict[int, Any] = {}
        for cluster_id in cluster_ids:
            scatter_by_cluster[int(cluster_id)] = ax.scatter(
                empty_xy[:, 0],
                empty_xy[:, 1],
                s=float(point_size),
                alpha=float(alpha),
                color=color_lookup[int(cluster_id)],
                linewidths=0.0,
            )

        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_xlabel(str(xlabel))
        ax.set_ylabel(str(ylabel))
        ax.grid(True, alpha=0.18, linewidth=0.6)
        legend_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=color_lookup[int(cluster_id)],
                markerfacecolor=color_lookup[int(cluster_id)],
                markersize=7,
                label=display_labels[int(cluster_id)],
            )
            for cluster_id in cluster_ids
        ]
        legend_ax.legend(
            legend_handles,
            [display_labels[int(cluster_id)] for cluster_id in cluster_ids],
            title="cluster",
            loc="upper left",
            frameon=False,
        )
        title_artist = fig.suptitle("", fontsize=13)

        for record in frame_records:
            embedding = np.asarray(record["embedding"], dtype=np.float32)
            labels = np.asarray(record["labels"], dtype=int).reshape(-1)
            if embedding.shape[0] != labels.shape[0]:
                raise ValueError(
                    "embedding and labels length mismatch for temporal embedding animation: "
                    f"{embedding.shape[0]} vs {labels.shape[0]}."
                )
            for cluster_id in cluster_ids:
                cluster_offsets = embedding[labels == int(cluster_id)]
                scatter_by_cluster[int(cluster_id)].set_offsets(
                    cluster_offsets if cluster_offsets.size > 0 else empty_xy
                )
            title_artist.set_text(f"{title} | {record['frame_label']}")
            frames_rgb.append(_render_figure_to_rgb_array(fig))
    finally:
        plt.close(fig)

    out_file = Path(out_file)
    _save_rgb_frames_as_gif(
        frames_rgb,
        out_file,
        frame_duration_ms=int(frame_duration_ms),
        total_duration_seconds=total_duration_seconds,
    )
    return {
        "out_file": str(out_file),
        "frame_count": int(len(frame_records)),
        "cluster_ids": [int(v) for v in cluster_ids],
        "x_limits": [float(x_limits[0]), float(x_limits[1])],
        "y_limits": [float(y_limits[0]), float(y_limits[1])],
    }


def save_temporal_embedding_trajectory_animation(
    frame_records: list[dict[str, Any]],
    out_file: Path,
    *,
    cluster_color_map: dict[int, str] | None = None,
    cluster_display_map: dict[int, str] | None = None,
    line_width: float = 0.8,
    line_alpha: float = 0.22,
    history_steps: int | None = 8,
    fade_min_alpha_fraction: float = 0.18,
    fade_power: float = 1.0,
    directional_subsegments: int = 6,
    directional_start_alpha_fraction: float = 0.32,
    directional_start_width_fraction: float = 0.60,
    directional_end_width_fraction: float = 1.35,
    endpoint_point_size: float = 3.0,
    endpoint_point_alpha: float = 0.95,
    frame_duration_ms: int = 450,
    total_duration_seconds: float | None = None,
    title: str = "Latent trajectory evolution",
    xlabel: str = "UMAP 1",
    ylabel: str = "UMAP 2",
) -> dict[str, Any]:
    if not frame_records:
        raise ValueError("frame_records must be a non-empty list.")

    embeddings_by_frame: list[np.ndarray] = []
    labels_by_frame: list[np.ndarray] = []
    instance_ids_ref: np.ndarray | None = None
    for frame_idx, record in enumerate(frame_records):
        embedding = np.asarray(record["embedding"], dtype=np.float32)
        labels = np.asarray(record["labels"], dtype=int).reshape(-1)
        instance_ids = np.asarray(record["instance_ids"], dtype=np.int64).reshape(-1)
        if embedding.ndim != 2 or embedding.shape[1] != 2:
            raise ValueError(
                "Temporal trajectory animation expects 2D embeddings for every frame, "
                f"got frame_idx={frame_idx}, shape={embedding.shape}."
            )
        if embedding.shape[0] != labels.shape[0] or embedding.shape[0] != instance_ids.shape[0]:
            raise ValueError(
                "Temporal trajectory animation requires matching embedding/labels/instance_ids lengths. "
                f"frame_idx={frame_idx}, embedding={embedding.shape[0]}, "
                f"labels={labels.shape[0]}, instance_ids={instance_ids.shape[0]}."
            )
        if instance_ids_ref is None:
            instance_ids_ref = instance_ids.copy()
        elif instance_ids.shape != instance_ids_ref.shape or not np.array_equal(instance_ids, instance_ids_ref):
            raise ValueError(
                "Temporal trajectory animation requires identical ordered instance_ids in every frame. "
                f"frame_idx={frame_idx}, expected_shape={instance_ids_ref.shape}, got_shape={instance_ids.shape}."
            )
        embeddings_by_frame.append(embedding)
        labels_by_frame.append(labels)

    assert instance_ids_ref is not None
    if fade_min_alpha_fraction <= 0.0 or fade_min_alpha_fraction > 1.0:
        raise ValueError(
            "fade_min_alpha_fraction must be in (0, 1], "
            f"got {fade_min_alpha_fraction}."
        )
    if history_steps is not None and int(history_steps) <= 0:
        raise ValueError(f"history_steps must be > 0 when provided, got {history_steps}.")
    if fade_power <= 0.0:
        raise ValueError(f"fade_power must be > 0, got {fade_power}.")
    if int(directional_subsegments) <= 0:
        raise ValueError(
            f"directional_subsegments must be a positive integer, got {directional_subsegments}."
        )
    if directional_start_alpha_fraction <= 0.0 or directional_start_alpha_fraction > 1.0:
        raise ValueError(
            "directional_start_alpha_fraction must be in (0, 1], "
            f"got {directional_start_alpha_fraction}."
        )
    if directional_start_width_fraction <= 0.0:
        raise ValueError(
            "directional_start_width_fraction must be > 0, "
            f"got {directional_start_width_fraction}."
        )
    if directional_end_width_fraction <= 0.0:
        raise ValueError(
            "directional_end_width_fraction must be > 0, "
            f"got {directional_end_width_fraction}."
        )
    if endpoint_point_size < 0.0:
        raise ValueError(f"endpoint_point_size must be >= 0, got {endpoint_point_size}.")
    if endpoint_point_alpha < 0.0 or endpoint_point_alpha > 1.0:
        raise ValueError(
            "endpoint_point_alpha must be in [0, 1], "
            f"got {endpoint_point_alpha}."
        )
    all_embeddings = np.concatenate(embeddings_by_frame, axis=0)
    cluster_ids = _sorted_cluster_ids(np.concatenate(labels_by_frame, axis=0))
    colors = _cluster_palette_from_map(cluster_ids, cluster_color_map)
    color_lookup = {
        int(cluster_id): str(colors[pos])
        for pos, cluster_id in enumerate(cluster_ids)
    }
    display_labels = {
        int(cluster_id): (
            str(cluster_display_map[int(cluster_id)])
            if cluster_display_map is not None and int(cluster_id) in cluster_display_map
            else f"C{int(cluster_id)}"
        )
        for cluster_id in cluster_ids
    }
    mins = np.min(all_embeddings, axis=0)
    maxs = np.max(all_embeddings, axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = 0.06 * span
    x_limits = (float(mins[0] - pad[0]), float(maxs[0] + pad[0]))
    y_limits = (float(mins[1] - pad[1]), float(maxs[1] + pad[1]))

    empty_xy = np.zeros((0, 2), dtype=np.float32)
    empty_segments = np.zeros((0, 2, 2), dtype=np.float32)
    empty_rgba = np.zeros((0, 4), dtype=np.float32)
    empty_widths = np.zeros((0,), dtype=np.float32)
    trajectory_payloads: list[dict[str, np.ndarray]] = []
    interpolation = np.linspace(
        0.0,
        1.0,
        num=int(directional_subsegments) + 1,
        dtype=np.float32,
    )
    segment_alpha_profile = (
        float(directional_start_alpha_fraction)
        + (1.0 - float(directional_start_alpha_fraction)) * interpolation[1:]
    ).astype(np.float32, copy=False)
    segment_width_profile = (
        float(directional_start_width_fraction)
        + (
            float(directional_end_width_fraction)
            - float(directional_start_width_fraction)
        )
        * interpolation[1:]
    ).astype(np.float32, copy=False)
    base_rgba_by_frame = [
        np.asarray(
            [mcolors.to_rgba(color_lookup[int(label)]) for label in frame_labels],
            dtype=np.float32,
        )
        for frame_labels in labels_by_frame
    ]
    for frame_idx in range(len(frame_records)):
        start_step_idx = 1
        if history_steps is not None:
            start_step_idx = max(1, frame_idx - int(history_steps) + 1)
        visible_step_count = max(1, frame_idx - start_step_idx + 1)
        frame_segments: list[np.ndarray] = []
        frame_colors: list[np.ndarray] = []
        frame_widths: list[np.ndarray] = []
        for step_idx in range(start_step_idx, frame_idx + 1):
            prev_embedding = embeddings_by_frame[step_idx - 1]
            curr_embedding = embeddings_by_frame[step_idx]
            curr_base_rgba = base_rgba_by_frame[step_idx]
            recency = float(step_idx - start_step_idx + 1) / float(visible_step_count)
            fade_weight = float(fade_min_alpha_fraction) + (
                1.0 - float(fade_min_alpha_fraction)
            ) * (recency ** float(fade_power))
            start_points = (
                prev_embedding[:, None, :] * (1.0 - interpolation[:-1][None, :, None])
                + curr_embedding[:, None, :] * interpolation[:-1][None, :, None]
            )
            end_points = (
                prev_embedding[:, None, :] * (1.0 - interpolation[1:][None, :, None])
                + curr_embedding[:, None, :] * interpolation[1:][None, :, None]
            )
            frame_segments.append(
                np.stack([start_points, end_points], axis=2).reshape(-1, 2, 2)
            )
            segment_rgba = np.repeat(
                curr_base_rgba[:, None, :],
                int(directional_subsegments),
                axis=1,
            )
            segment_rgba[:, :, 3] = (
                float(line_alpha) * fade_weight * segment_alpha_profile[None, :]
            )
            frame_colors.append(segment_rgba.reshape(-1, 4).astype(np.float32, copy=False))
            frame_widths.append(
                (
                    float(line_width) * segment_width_profile[None, :]
                ).repeat(curr_embedding.shape[0], axis=0).reshape(-1).astype(np.float32, copy=False)
            )
        trajectory_payloads.append(
            {
                "segments": (
                    np.concatenate(frame_segments, axis=0)
                    if frame_segments
                    else empty_segments
                ),
                "colors": (
                    np.concatenate(frame_colors, axis=0)
                    if frame_colors
                    else empty_rgba
                ),
                "widths": (
                    np.concatenate(frame_widths, axis=0)
                    if frame_widths
                    else empty_widths
                ),
            }
        )

    frames_rgb: list[np.ndarray] = []
    num_instances = int(instance_ids_ref.shape[0])
    fig = plt.figure(figsize=(9.0, 6.8), dpi=180)
    try:
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0.08, 0.12, 0.64, 0.78])
        legend_ax = fig.add_axes([0.76, 0.18, 0.22, 0.60])
        legend_ax.axis("off")
        trajectory_collection = LineCollection(
            empty_segments,
            colors=empty_rgba,
            linewidths=empty_widths,
            capstyle="round",
            joinstyle="round",
            zorder=1,
        )
        ax.add_collection(trajectory_collection)
        endpoint_scatter = None
        if float(endpoint_point_size) > 0.0:
            endpoint_scatter = ax.scatter(
                empty_xy[:, 0],
                empty_xy[:, 1],
                s=float(endpoint_point_size),
                alpha=float(endpoint_point_alpha),
                color="#000000",
                linewidths=0.0,
                zorder=3,
            )

        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_xlabel(str(xlabel))
        ax.set_ylabel(str(ylabel))
        ax.grid(True, alpha=0.18, linewidth=0.6)
        legend_handles = [
            plt.Line2D(
                [0],
                [0],
                linestyle="-",
                color=color_lookup[int(cluster_id)],
                linewidth=2.2,
                label=display_labels[int(cluster_id)],
            )
            for cluster_id in cluster_ids
        ]
        legend_ax.legend(
            legend_handles,
            [display_labels[int(cluster_id)] for cluster_id in cluster_ids],
            title="cluster",
            loc="upper left",
            frameon=False,
        )
        title_artist = fig.suptitle("", fontsize=13)

        for frame_idx, record in enumerate(frame_records):
            payload = trajectory_payloads[frame_idx]
            if payload["segments"].shape[0] == 0:
                trajectory_collection.set_segments([])
                trajectory_collection.set_color([])
                trajectory_collection.set_linewidth([])
            else:
                trajectory_collection.set_segments(payload["segments"])
                trajectory_collection.set_color(payload["colors"])
                trajectory_collection.set_linewidth(payload["widths"])
            if endpoint_scatter is not None:
                endpoint_scatter.set_offsets(
                    embeddings_by_frame[frame_idx]
                    if embeddings_by_frame[frame_idx].size > 0
                    else empty_xy
                )
            title_artist.set_text(f"{title} | {record['frame_label']}")
            frames_rgb.append(_render_figure_to_rgb_array(fig))
    finally:
        plt.close(fig)

    out_file = Path(out_file)
    _save_rgb_frames_as_gif(
        frames_rgb,
        out_file,
        frame_duration_ms=int(frame_duration_ms),
        total_duration_seconds=total_duration_seconds,
    )
    return {
        "out_file": str(out_file),
        "frame_count": int(len(frame_records)),
        "num_instances": int(num_instances),
        "cluster_ids": [int(v) for v in cluster_ids],
        "x_limits": [float(x_limits[0]), float(x_limits[1])],
        "y_limits": [float(y_limits[0]), float(y_limits[1])],
    }


def save_temporal_transition_flow_animation(
    pair_records: list[dict[str, Any]],
    out_file: Path,
    *,
    row_labels: list[str],
    col_labels: list[str] | None = None,
    cluster_color_map: dict[int, str] | None = None,
    cluster_ids_for_palette: list[int] | None = None,
    mute_diagonal: bool = True,
    min_draw_fraction: float = 0.0,
    frame_duration_ms: int = 450,
    total_duration_seconds: float | None = None,
    title: str = "Cluster transition flow",
) -> dict[str, Any]:
    if not pair_records:
        raise ValueError("pair_records must be a non-empty list.")

    frames_rgb: list[np.ndarray] = []
    for pair_idx, record in enumerate(pair_records):
        pair_title = str(
            record.get(
                "title",
                f"{title} | {record.get('frame_from_label', record.get('frame_from', '?'))} -> "
                f"{record.get('frame_to_label', record.get('frame_to', '?'))}",
            )
        )
        fig, _ = _build_transition_flow_figure(
            np.asarray(record["counts"], dtype=np.float64),
            title=pair_title,
            row_labels=row_labels,
            col_labels=col_labels,
            cluster_color_map=cluster_color_map,
            cluster_ids_for_palette=cluster_ids_for_palette,
            mute_diagonal=bool(mute_diagonal),
            min_draw_fraction=float(min_draw_fraction),
        )
        frames_rgb.append(_render_figure_to_rgb_array(fig))
        plt.close(fig)

    out_file = Path(out_file)
    _save_rgb_frames_as_gif(
        frames_rgb,
        out_file,
        frame_duration_ms=int(frame_duration_ms),
        total_duration_seconds=total_duration_seconds,
    )
    return {
        "out_file": str(out_file),
        "frame_count": int(len(pair_records)),
        "pair_titles": [
            str(
                record.get(
                    "title",
                    f"{record.get('frame_from_label', record.get('frame_from', '?'))} -> "
                    f"{record.get('frame_to_label', record.get('frame_to', '?'))}",
                )
            )
            for record in pair_records
        ],
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


def _build_transition_flow_figure(
    matrix: np.ndarray,
    *,
    title: str | None,
    row_labels: list[str],
    col_labels: list[str] | None,
    cluster_color_map: dict[int, str] | None,
    cluster_ids_for_palette: list[int] | None,
    mute_diagonal: bool,
    min_draw_fraction: float,
) -> tuple[Any, dict[str, Any]]:
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

    return fig, {
        "shape": [int(values.shape[0]), int(values.shape[1])],
        "total_weight": float(total_flow),
        "mute_diagonal": bool(mute_diagonal),
    }


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
    fig, summary = _build_transition_flow_figure(
        matrix,
        title=title,
        row_labels=row_labels,
        col_labels=col_labels,
        cluster_color_map=cluster_color_map,
        cluster_ids_for_palette=cluster_ids_for_palette,
        mute_diagonal=bool(mute_diagonal),
        min_draw_fraction=float(min_draw_fraction),
    )

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
        **summary,
    }


__all__ = [
    "save_cluster_proportion_plots",
    "save_embedding_discrete_plot",
    "save_descriptor_violin_grid",
    "save_cna_signature_time_series",
    "save_temporal_embedding_cluster_animation",
    "save_temporal_embedding_trajectory_animation",
    "save_temporal_spatial_cluster_animation",
    "save_temporal_transition_flow_animation",
    "save_transition_flow_plot",
]
