"""Coordinate-space (MD) cluster visualization outputs.

Handles saving static PNG plots, interactive HTML plots (plotly),
per-snapshot breakdown, and HDBSCAN MD outputs.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .clustering import HDBSCANResult
from src.vis_tools.latent_analysis_vis import (
    save_local_structure_assignments,
    save_md_space_clusters_plot,
)
from src.vis_tools.md_cluster_plot import save_interactive_md_plot


def _ordered_md_k_values(
    k_values_order: list[int],
    *,
    primary_k: int,
) -> list[int]:
    ordered = [int(primary_k)] + [int(k) for k in k_values_order if int(k) != int(primary_k)]
    return list(dict.fromkeys(ordered))


def _build_md_plot_title(
    labels_for_k: np.ndarray,
    *,
    k_value: int,
    sample_count: int,
    source_name: str | None = None,
) -> str:
    labels_arr = np.asarray(labels_for_k, dtype=int).reshape(-1)
    visible_clusters = int(len(np.unique(labels_arr)))
    parts = [f"k={int(k_value)}", f"n={int(sample_count)}"]
    if visible_clusters != int(k_value):
        parts.append(f"visible={visible_clusters}")
    if source_name is not None:
        parts.append(f"snapshot={source_name}")
    return f"MD local-structure clusters ({', '.join(parts)})"


def _save_md_cluster_outputs(
    coords: np.ndarray,
    labels_by_k: dict[int, np.ndarray],
    out_dir: Path,
    *,
    k_values_order: list[int],
    primary_k: int,
    shared_cluster_color_maps_by_k: dict[int, dict[int, str]],
    max_points: int | None,
    source_name: str | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered_k_values = _ordered_md_k_values(k_values_order, primary_k=int(primary_k))
    if not ordered_k_values:
        raise ValueError("Cannot save MD cluster outputs: no k values were provided.")

    coords_arr = np.asarray(coords, dtype=np.float32)
    if coords_arr.ndim != 2 or coords_arr.shape[0] < 2 or coords_arr.shape[1] < 3:
        raise ValueError(
            "Cannot save MD cluster outputs: expected coords with shape (n, d), "
            f"n >= 2, d >= 3, got shape={coords_arr.shape}."
        )

    stale_patterns = (
        "md_space_clusters.png",
        "md_space_clusters.html",
        "md_space_clusters_k*.png",
        "md_space_clusters_k*.html",
    )
    stale_paths: set[Path] = set()
    for pattern in stale_patterns:
        stale_paths.update(out_dir.glob(pattern))
    for stale_path in stale_paths:
        stale_path.unlink()

    static_pngs: dict[int, str] = {}
    interactive_htmls: dict[int, str] = {}
    for k_value in ordered_k_values:
        if int(k_value) not in labels_by_k:
            raise KeyError(
                "Cannot save MD cluster outputs because labels are missing for "
                f"k={int(k_value)}. Available k values: {sorted(int(k) for k in labels_by_k)}."
            )
        if int(k_value) not in shared_cluster_color_maps_by_k:
            raise KeyError(
                "Cannot save MD cluster outputs because the shared color map is missing for "
                f"k={int(k_value)}. Available k values: "
                f"{sorted(int(k) for k in shared_cluster_color_maps_by_k)}."
            )
        labels_arr = np.asarray(labels_by_k[int(k_value)], dtype=int).reshape(-1)
        if labels_arr.shape[0] != coords_arr.shape[0]:
            raise ValueError(
                "Cannot save MD cluster outputs because coords/labels lengths do not match. "
                f"k={int(k_value)}, coords_shape={coords_arr.shape}, labels_shape={labels_arr.shape}."
            )
        title = _build_md_plot_title(
            labels_arr,
            k_value=int(k_value),
            sample_count=int(coords_arr.shape[0]),
            source_name=source_name,
        )
        static_path = out_dir / f"md_space_clusters_k{int(k_value)}.png"
        save_md_space_clusters_plot(
            coords_arr,
            labels_arr,
            static_path,
            cluster_color_map=shared_cluster_color_maps_by_k[int(k_value)],
            max_points=max_points,
            title=title,
        )
        static_pngs[int(k_value)] = str(static_path)

    primary_static_path = out_dir / f"md_space_clusters_k{int(primary_k)}.png"
    primary_static_alias = out_dir / "md_space_clusters.png"
    shutil.copyfile(primary_static_path, primary_static_alias)

    results: dict[str, Any] = {
        "static_png": str(primary_static_alias),
        "static_pngs": static_pngs,
    }

    try:
        for k_value in ordered_k_values:
            labels_arr = np.asarray(labels_by_k[int(k_value)], dtype=int).reshape(-1)
            title = _build_md_plot_title(
                labels_arr,
                k_value=int(k_value),
                sample_count=int(coords_arr.shape[0]),
                source_name=source_name,
            )
            interactive_path = out_dir / f"md_space_clusters_k{int(k_value)}.html"
            save_interactive_md_plot(
                coords_arr,
                labels_arr,
                interactive_path,
                palette="tab10",
                cluster_color_map=shared_cluster_color_maps_by_k[int(k_value)],
                max_points=max_points,
                marker_size=3.0,
                marker_line_width=0.0,
                title=title,
                aspect_mode="cube",
            )
            interactive_htmls[int(k_value)] = str(interactive_path)
        primary_interactive_path = out_dir / f"md_space_clusters_k{int(primary_k)}.html"
        primary_interactive_alias = out_dir / "md_space_clusters.html"
        shutil.copyfile(primary_interactive_path, primary_interactive_alias)
        results["interactive_html"] = str(primary_interactive_alias)
        results["interactive_htmls"] = interactive_htmls
    except ImportError:
        context = "full dataset" if source_name is None else f"snapshot {source_name}"
        print(f"Plotly not installed; skipping interactive MD plot for {context}.")

    return results


def _save_hdbscan_md_outputs(
    coords: np.ndarray,
    hdbscan_labels: np.ndarray | None,
    out_dir: Path,
    *,
    hdbscan_color_map: dict[int, str] | None,
    max_points: int | None,
    title: str,
    context_label: str,
) -> dict[str, Any]:
    if hdbscan_labels is None:
        return {}

    outputs: dict[str, Any] = {}
    hdbscan_coord_files = save_local_structure_assignments(
        coords,
        hdbscan_labels,
        out_dir,
        prefix="local_structure_hdbscan",
    )
    if hdbscan_coord_files:
        outputs["hdbscan_coords_files"] = hdbscan_coord_files
    try:
        hdbscan_path = Path(out_dir) / "md_space_clusters_hdbscan.html"
        n_hdb_clusters = int(len(np.unique(hdbscan_labels[hdbscan_labels >= 0])))
        save_interactive_md_plot(
            coords,
            hdbscan_labels,
            hdbscan_path,
            palette="tab10",
            cluster_color_map=hdbscan_color_map,
            max_points=max_points,
            marker_size=3.0,
            marker_line_width=0.0,
            title=f"{title}, k={n_hdb_clusters})",
            label_prefix="HDBSCAN",
            aspect_mode="cube",
        )
        outputs["hdbscan_interactive_html"] = str(hdbscan_path)
    except ImportError:
        print(
            "Plotly not installed; skipping HDBSCAN interactive MD plot "
            f"for {context_label}."
        )
    return outputs


def _save_snapshot_md_outputs(
    *,
    out_dir: Path,
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_labels_by_k: dict[int, np.ndarray],
    configured_k_values: list[int],
    primary_k: int,
    snapshot_source_groups: list[tuple[str, np.ndarray]],
    snapshot_output_names: dict[str, str],
    shared_cluster_color_maps_by_k: dict[int, dict[int, str]],
    interactive_max_points: int | None,
    hdbscan_result: HDBSCANResult,
) -> dict[str, Any]:
    snapshot_root = Path(out_dir) / "md_clusters_by_snapshot"
    summary: dict[str, Any] = {
        "root_dir": str(snapshot_root),
        "primary_k": int(primary_k),
        "k_values_used": [int(k) for k in configured_k_values],
        "snapshots": [],
    }
    for source_name, indices in snapshot_source_groups:
        snapshot_dirname = snapshot_output_names[str(source_name)]
        snapshot_dir = snapshot_root / snapshot_dirname
        coords_subset = coords[indices]
        labels_subset = cluster_labels[indices]
        coord_files = save_local_structure_assignments(
            coords_subset,
            labels_subset,
            snapshot_dir,
        )
        if not coord_files:
            raise RuntimeError(
                "Failed to save per-snapshot MD cluster assignments. "
                f"snapshot={source_name}, coords_shape={coords_subset.shape}, "
                f"labels_shape={labels_subset.shape}, out_dir={snapshot_dir}."
            )
        snapshot_record: dict[str, Any] = {
            "source_name": str(source_name),
            "output_name": str(snapshot_dirname),
            "sample_count": int(indices.size),
            "coords_files": coord_files,
        }
        snapshot_record.update(
            _save_md_cluster_outputs(
                coords_subset,
                {
                    int(k_value): cluster_labels_by_k[int(k_value)][indices]
                    for k_value in configured_k_values
                },
                snapshot_dir,
                k_values_order=[int(k_value) for k_value in configured_k_values],
                primary_k=int(primary_k),
                shared_cluster_color_maps_by_k=shared_cluster_color_maps_by_k,
                max_points=interactive_max_points,
                source_name=str(source_name),
            )
        )
        if hdbscan_result.info is not None:
            snapshot_record["hdbscan"] = hdbscan_result.info
        if hdbscan_result.labels is not None:
            snapshot_record.update(
                _save_hdbscan_md_outputs(
                    coords_subset,
                    hdbscan_result.labels[indices],
                    snapshot_dir,
                    hdbscan_color_map=hdbscan_result.color_map,
                    max_points=interactive_max_points,
                    title=(
                        "MD local-structure clusters "
                        f"(HDBSCAN, snapshot={source_name}, n={len(coords_subset)}"
                    ),
                    context_label=f"snapshot {source_name}",
                )
            )
        summary["snapshots"].append(snapshot_record)
    return summary


def build_md_metrics(
    *,
    out_dir: Path,
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_labels_by_k: dict[int, np.ndarray],
    configured_k_values: list[int],
    primary_k: int,
    shared_cluster_color_maps_by_k: dict[int, dict[int, str]],
    interactive_max_points: int | None,
    multi_snapshot_real: bool,
    snapshot_source_groups: list[tuple[str, np.ndarray]],
    snapshot_output_names: dict[str, str],
    hdbscan_result: HDBSCANResult,
) -> dict[str, Any]:
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    metrics: dict[str, Any] = {
        "primary_k": int(primary_k),
        "k_values_used": [int(k) for k in configured_k_values],
        "n_clusters": int(len(unique_labels)),
        "cluster_counts": {int(k): int(v) for k, v in zip(unique_labels, counts)},
    }
    if hdbscan_result.info is not None:
        metrics["hdbscan"] = hdbscan_result.info

    if multi_snapshot_real:
        metrics["by_snapshot"] = _save_snapshot_md_outputs(
            out_dir=out_dir,
            coords=coords,
            cluster_labels=cluster_labels,
            cluster_labels_by_k=cluster_labels_by_k,
            configured_k_values=configured_k_values,
            primary_k=primary_k,
            snapshot_source_groups=snapshot_source_groups,
            snapshot_output_names=snapshot_output_names,
            shared_cluster_color_maps_by_k=shared_cluster_color_maps_by_k,
            interactive_max_points=interactive_max_points,
            hdbscan_result=hdbscan_result,
        )
        return metrics

    coord_files = save_local_structure_assignments(
        coords,
        cluster_labels,
        out_dir,
    )
    if not coord_files:
        raise RuntimeError(
            "Failed to save MD cluster assignments. "
            f"coords_shape={coords.shape}, labels_shape={cluster_labels.shape}, "
            f"out_dir={out_dir}."
        )
    metrics["coords_files"] = coord_files
    metrics.update(
        _save_md_cluster_outputs(
            coords,
            {
                int(k_value): cluster_labels_by_k[int(k_value)]
                for k_value in configured_k_values
            },
            out_dir,
            k_values_order=[int(k_value) for k_value in configured_k_values],
            primary_k=int(primary_k),
            shared_cluster_color_maps_by_k=shared_cluster_color_maps_by_k,
            max_points=interactive_max_points,
        )
    )
    if hdbscan_result.labels is not None:
        metrics.update(
            _save_hdbscan_md_outputs(
                coords,
                hdbscan_result.labels,
                out_dir,
                hdbscan_color_map=hdbscan_result.color_map,
                max_points=interactive_max_points,
                title=f"MD local-structure clusters (HDBSCAN, n={len(coords)}",
                context_label="full dataset",
            )
        )
    return metrics
