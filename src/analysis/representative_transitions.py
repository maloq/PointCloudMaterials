"""Interactive real-structure sequences along connected cluster directions."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib.colors as mcolors
import numpy as np

from .cluster_colors import _compute_center_to_edge_colors
from .cluster_geometry import (
    _build_local_coordination_edges,
    _compute_cluster_representative_indices,
    _compute_pca_orientation_basis,
    _load_points_from_dataset,
    _orient_points_for_crystal_view,
)


def select_transition_representative_rows(
    pair_latents: np.ndarray,
    pair_labels: np.ndarray,
    *,
    cluster_a: int,
    cluster_b: int,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select distinct real samples nearest a cluster-centroid line."""
    latents = np.asarray(pair_latents, dtype=np.float64)
    labels = np.asarray(pair_labels, dtype=np.int64).reshape(-1)
    if latents.ndim != 2 or len(latents) != len(labels):
        raise ValueError(
            "Transition representative selection expects latents (N,D) and labels (N,), "
            f"got {latents.shape} and {labels.shape}."
        )
    if int(steps) < 3:
        raise ValueError(f"Transition representative steps must be >= 3, got {steps}.")
    mask_a = labels == int(cluster_a)
    mask_b = labels == int(cluster_b)
    if not np.any(mask_a) or not np.any(mask_b):
        raise ValueError(
            "Transition representative selection requires samples from both clusters. "
            f"clusters=({cluster_a}, {cluster_b}), counts=({mask_a.sum()}, {mask_b.sum()})."
        )
    centroid_a = latents[mask_a].mean(axis=0)
    centroid_b = latents[mask_b].mean(axis=0)
    direction = centroid_b - centroid_a
    direction_norm_sq = float(np.dot(direction, direction))
    if direction_norm_sq <= 1.0e-15:
        raise ValueError(
            f"Clusters {cluster_a} and {cluster_b} have coincident latent centroids."
        )

    target_fractions = np.linspace(0.0, 1.0, int(steps), dtype=np.float64)
    selected: list[int] = []
    available = np.ones((len(latents),), dtype=bool)
    for fraction in target_fractions:
        target = centroid_a + float(fraction) * direction
        distances = np.sum((latents - target[None, :]) ** 2, axis=1)
        distances[~available] = np.inf
        row = int(np.argmin(distances))
        if not np.isfinite(distances[row]):
            raise RuntimeError(
                "Transition representative selection ran out of distinct real samples. "
                f"requested_steps={steps}, pair_sample_count={len(latents)}."
            )
        selected.append(row)
        available[row] = False

    projection = ((latents - centroid_a[None, :]) @ direction) / direction_norm_sq
    return (
        np.asarray(selected, dtype=np.int64),
        target_fractions.astype(np.float32),
        projection.astype(np.float32),
    )


def select_within_cluster_representative_rows(
    cluster_latents: np.ndarray,
    *,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select distinct real samples along a cluster's dominant latent direction."""
    latents = np.asarray(cluster_latents, dtype=np.float64)
    if latents.ndim != 2:
        raise ValueError(
            "Within-cluster transition selection expects latents (N,D), "
            f"got {latents.shape}."
        )
    if int(steps) < 3:
        raise ValueError(f"Transition representative steps must be >= 3, got {steps}.")
    if len(latents) < int(steps):
        raise ValueError(
            "Within-cluster transition needs at least one distinct real sample per step. "
            f"requested_steps={steps}, cluster_sample_count={len(latents)}."
        )

    centroid = latents.mean(axis=0)
    centered = latents - centroid[None, :]
    _left, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    if singular_values.size == 0 or float(singular_values[0]) <= 1.0e-12:
        raise ValueError(
            "Within-cluster transition cannot define a direction because the cluster "
            "has no measurable latent variation."
        )
    direction = np.asarray(right[0], dtype=np.float64)
    largest_component = int(np.argmax(np.abs(direction)))
    if direction[largest_component] < 0.0:
        direction *= -1.0

    scalar_projection = centered @ direction
    lower, upper = np.quantile(scalar_projection, [0.01, 0.99])
    span = float(upper - lower)
    if span <= 1.0e-12:
        raise ValueError(
            "Within-cluster transition has a degenerate 1st-to-99th percentile range "
            f"along its dominant latent direction: span={span}."
        )
    target_fractions = np.linspace(0.0, 1.0, int(steps), dtype=np.float64)
    target_scalars = float(lower) + target_fractions * span
    selected: list[int] = []
    available = np.ones((len(latents),), dtype=bool)
    for scalar in target_scalars:
        target = centroid + float(scalar) * direction
        distances = np.sum((latents - target[None, :]) ** 2, axis=1)
        distances[~available] = np.inf
        row = int(np.argmin(distances))
        if not np.isfinite(distances[row]):
            raise RuntimeError(
                "Within-cluster transition ran out of distinct real samples. "
                f"requested_steps={steps}, cluster_sample_count={len(latents)}."
            )
        selected.append(row)
        available[row] = False

    projection_fraction = (scalar_projection - float(lower)) / span
    return (
        np.asarray(selected, dtype=np.int64),
        target_fractions.astype(np.float32),
        projection_fraction.astype(np.float32),
    )


def _prepare_local_structure(
    dataset: Any,
    sample_index: int,
    *,
    point_scale: float,
    target_points: int,
) -> np.ndarray:
    points = _load_points_from_dataset(
        dataset,
        int(sample_index),
        point_scale=float(point_scale),
    )
    center_index = int(np.argmin(np.linalg.norm(points, axis=1)))
    centered = points - points[center_index]
    order = np.argsort(np.linalg.norm(centered, axis=1), kind="mergesort")
    keep = order[: min(int(target_points), len(order))]
    local = np.asarray(centered[keep], dtype=np.float32)
    if local.shape[0] < 2:
        raise ValueError(
            "Interactive transition representative needs at least two local atoms. "
            f"sample_index={sample_index}, retained_shape={local.shape}."
        )
    return local


def _plotly_rgb_strings(colors: np.ndarray) -> list[str]:
    rgb = np.clip(np.asarray(colors, dtype=np.float64), 0.0, 1.0)
    return [mcolors.to_hex(row, keep_alpha=False) for row in rgb]


def _edge_coordinates(
    points: np.ndarray,
    edges: Sequence[tuple[int, int]],
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    for first, second in edges:
        x.extend([float(points[first, 0]), float(points[second, 0]), None])
        y.extend([float(points[first, 1]), float(points[second, 1]), None])
        z.extend([float(points[first, 2]), float(points[second, 2]), None])
    return x, y, z


def _plotly_edge_trace(points: np.ndarray) -> tuple[Any, dict[str, Any]]:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("Interactive representative edges require Plotly.") from exc
    edges, edge_info = _build_local_coordination_edges(
        points,
        min_shell_neighbors=3,
        max_shell_neighbors=6,
        shell_gap_ratio=1.22,
        edge_mode="coordination_shell_mutual",
    )
    x, y, z = _edge_coordinates(points, edges)
    trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        showlegend=False,
        hoverinfo="skip",
        opacity=0.28,
        line={"color": "#666666", "width": 1.0},
    )
    return trace, edge_info


def _frame_title(record: dict[str, Any], *, cluster_a: int, cluster_b: int) -> str:
    return (
        f"C{cluster_a + 1} → C{cluster_b + 1}: real structures along the latent centerline | "
        f"step {record['step_index'] + 1}/{record['step_count']}, "
        f"target t={record['target_fraction']:.2f}, actual t={record['actual_fraction']:.2f}, "
        f"sample C{record['cluster_id'] + 1}"
        "<br><sup>"
        f"sample index={record['sample_index']}, source={record['source_name'] or 'unknown'}, "
        f"order parameter={record['order_parameter']:.2f}, "
        f"other-cluster neighbor fraction={record['local_other_cluster_fraction']:.2f}"
        "</sup>"
    )


def _within_cluster_frame_title(record: dict[str, Any], *, cluster_id: int) -> str:
    return (
        f"C{cluster_id + 1}: real structures along the dominant within-cluster direction | "
        f"step {record['step_index'] + 1}/{record['step_count']}, "
        f"target t={record['target_fraction']:.2f}, actual t={record['actual_fraction']:.2f}"
        "<br><sup>"
        f"sample index={record['sample_index']}, source={record['source_name'] or 'unknown'}"
        "</sup>"
    )


def _render_transition_sequence_3d(
    *,
    dataset: Any,
    labels: np.ndarray,
    sample_indices: np.ndarray,
    selected_rows: np.ndarray,
    target_fractions: np.ndarray,
    path_projection: np.ndarray,
    cluster_color_map: dict[int, str],
    out_file: Path,
    point_scale: float,
    target_points: int,
    title_builder: Callable[[dict[str, Any]], str],
    cluster_ids: Sequence[int],
    transition_kind: str,
    direction_definition: str,
    source_names: Sequence[str] | None,
    order_parameter: np.ndarray | None = None,
    local_other_fraction: np.ndarray | None = None,
    include_plotlyjs: bool | str = "cdn",
) -> dict[str, Any]:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("Interactive real-structure transitions require Plotly.") from exc

    local_structures = [
        _prepare_local_structure(
            dataset,
            int(sample_indices[row]),
            point_scale=float(point_scale),
            target_points=int(target_points),
        )
        for row in selected_rows
    ]
    _, shared_basis, _, _, _ = _compute_pca_orientation_basis(local_structures[0])
    oriented_structures = [
        np.asarray(points @ shared_basis, dtype=np.float32) for points in local_structures
    ]
    axis_limit = max(float(np.max(np.abs(points))) for points in oriented_structures) * 1.08
    axis_limit = max(axis_limit, 1.0e-4)

    frame_records: list[dict[str, Any]] = []
    frame_payloads: list[tuple[Any, Any]] = []
    step_count = len(selected_rows)
    for step_index, (row_raw, target_fraction, points) in enumerate(
        zip(selected_rows, target_fractions, oriented_structures, strict=True)
    ):
        row = int(row_raw)
        cluster_id = int(labels[row])
        marker_colors = _plotly_rgb_strings(
            _compute_center_to_edge_colors(
                points, str(cluster_color_map[cluster_id])
            )
        )
        radii = np.linalg.norm(points, axis=1)
        source_name = "" if source_names is None else str(source_names[row])
        record = {
            "step_index": int(step_index),
            "step_count": int(step_count),
            "target_fraction": float(target_fraction),
            "actual_fraction": float(path_projection[row]),
            "cluster_id": cluster_id,
            "cluster_label": f"C{cluster_id + 1}",
            "sample_index": int(sample_indices[row]),
            "source_name": source_name,
            "num_points": int(len(points)),
        }
        if order_parameter is not None:
            record["order_parameter"] = float(order_parameter[row])
        if local_other_fraction is not None:
            record["local_other_cluster_fraction"] = float(local_other_fraction[row])
        edge_trace, edge_info = _plotly_edge_trace(points)
        record["edge_count"] = int(edge_info["num_edges"])
        frame_records.append(record)
        frame_payloads.append(
            (
                edge_trace,
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="markers",
                    name=f"C{cluster_id + 1} atoms",
                    customdata=np.column_stack(
                        (np.arange(len(points), dtype=np.int64), radii)
                    ),
                    marker={
                        "size": 8.5,
                        "color": marker_colors,
                        "opacity": 0.98,
                        "line": {"color": "#222222", "width": 1.0},
                    },
                    hovertemplate=(
                        "atom row=%{customdata[0]:.0f}<br>radius=%{customdata[1]:.4f}"
                        "<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>"
                    ),
                ),
            )
        )

    first = frame_records[0]
    figure = go.Figure(data=list(frame_payloads[0]))
    figure.frames = [
        go.Frame(
            name=str(record["step_index"]),
            data=list(payload),
            traces=[0, 1],
            layout=go.Layout(title={"text": title_builder(record)}),
        )
        for record, payload in zip(frame_records, frame_payloads, strict=True)
    ]
    slider_steps = [
        {
            "label": f"{record['target_fraction']:.2f} · C{record['cluster_id'] + 1}",
            "method": "animate",
            "args": [
                [str(record["step_index"])],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        }
        for record in frame_records
    ]
    figure.update_layout(
        title={"text": title_builder(first)},
        template=None,
        scene={
            "xaxis": {"title": "PCA x", "range": [-axis_limit, axis_limit]},
            "yaxis": {"title": "PCA y", "range": [-axis_limit, axis_limit]},
            "zaxis": {"title": "PCA z", "range": [-axis_limit, axis_limit]},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.15}},
        },
        margin={"l": 0, "r": 0, "t": 70, "b": 20},
        showlegend=False,
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "latent path t = "},
                "pad": {"t": 45},
                "steps": slider_steps,
            }
        ],
    )
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(out_file), include_plotlyjs=include_plotlyjs)
    return {
        "out_file": str(out_file),
        "cluster_ids": [int(cluster_id) for cluster_id in cluster_ids],
        "transition_kind": str(transition_kind),
        "direction_definition": str(direction_definition),
        "shared_orientation": "PCA basis fitted to the first real local structure",
        "steps": frame_records,
    }


def render_connected_pair_transition_3d(
    *,
    dataset: Any,
    pair_latents: np.ndarray,
    pair_labels: np.ndarray,
    pair_sample_indices: np.ndarray,
    order_parameter: np.ndarray,
    local_other_fraction: np.ndarray,
    cluster_a: int,
    cluster_b: int,
    cluster_color_map: dict[int, str],
    out_file: Path,
    steps: int,
    point_scale: float,
    target_points: int,
    source_names: Sequence[str] | None = None,
    include_plotlyjs: bool | str = "cdn",
) -> dict[str, Any]:
    """Render real local structures along one connected-pair latent direction."""
    labels = np.asarray(pair_labels, dtype=np.int64).reshape(-1)
    sample_indices = np.asarray(pair_sample_indices, dtype=np.int64).reshape(-1)
    order = np.asarray(order_parameter, dtype=np.float64).reshape(-1)
    mixing = np.asarray(local_other_fraction, dtype=np.float64).reshape(-1)
    if not (
        len(labels)
        == len(sample_indices)
        == len(order)
        == len(mixing)
        == len(pair_latents)
    ):
        raise ValueError(
            "Connected transition arrays must have the same row count. "
            f"latents={len(pair_latents)}, labels={len(labels)}, samples={len(sample_indices)}, "
            f"order={len(order)}, mixing={len(mixing)}."
        )
    if int(target_points) < 2:
        raise ValueError(f"target_points must be >= 2, got {target_points}.")

    selected_rows, target_fractions, centerline_projection = (
        select_transition_representative_rows(
            pair_latents,
            labels,
            cluster_a=int(cluster_a),
            cluster_b=int(cluster_b),
            steps=int(steps),
        )
    )
    return _render_transition_sequence_3d(
        dataset=dataset,
        labels=labels,
        sample_indices=sample_indices,
        selected_rows=selected_rows,
        target_fractions=target_fractions,
        path_projection=centerline_projection,
        cluster_color_map=cluster_color_map,
        out_file=out_file,
        point_scale=float(point_scale),
        target_points=int(target_points),
        title_builder=lambda record: _frame_title(
            record, cluster_a=int(cluster_a), cluster_b=int(cluster_b)
        ),
        cluster_ids=[int(cluster_a), int(cluster_b)],
        transition_kind="connected_pair",
        direction_definition=(
            "straight line between cluster centroids in original latent space"
        ),
        source_names=source_names,
        order_parameter=order,
        local_other_fraction=mixing,
        include_plotlyjs=include_plotlyjs,
    )


def render_within_cluster_transition_3d(
    *,
    dataset: Any,
    cluster_latents: np.ndarray,
    cluster_sample_indices: np.ndarray,
    cluster_id: int,
    cluster_color_map: dict[int, str],
    out_file: Path,
    steps: int,
    point_scale: float,
    target_points: int,
    source_names: Sequence[str] | None = None,
    include_plotlyjs: bool | str = "cdn",
) -> dict[str, Any]:
    """Render real structures across an isolated cluster's dominant latent direction."""
    latents = np.asarray(cluster_latents, dtype=np.float64)
    sample_indices = np.asarray(cluster_sample_indices, dtype=np.int64).reshape(-1)
    if latents.ndim != 2 or len(latents) != len(sample_indices):
        raise ValueError(
            "Within-cluster transition expects latents (N,D) aligned with sample indices "
            f"(N,), got {latents.shape} and {sample_indices.shape}."
        )
    if int(cluster_id) not in cluster_color_map:
        raise KeyError(
            f"Within-cluster transition is missing a color for cluster {cluster_id}."
        )
    if source_names is not None and len(source_names) != len(latents):
        raise ValueError(
            "Within-cluster transition source names must align with latent rows, "
            f"got names={len(source_names)}, latents={len(latents)}."
        )
    selected_rows, target_fractions, direction_projection = (
        select_within_cluster_representative_rows(latents, steps=int(steps))
    )
    labels = np.full(len(latents), int(cluster_id), dtype=np.int64)
    return _render_transition_sequence_3d(
        dataset=dataset,
        labels=labels,
        sample_indices=sample_indices,
        selected_rows=selected_rows,
        target_fractions=target_fractions,
        path_projection=direction_projection,
        cluster_color_map=cluster_color_map,
        out_file=out_file,
        point_scale=float(point_scale),
        target_points=int(target_points),
        title_builder=lambda record: _within_cluster_frame_title(
            record, cluster_id=int(cluster_id)
        ),
        cluster_ids=[int(cluster_id)],
        transition_kind="within_cluster",
        direction_definition=(
            "1st-to-99th percentile centerline along the cluster's dominant PCA "
            "direction in original latent space"
        ),
        source_names=source_names,
        include_plotlyjs=include_plotlyjs,
    )


def render_cluster_representatives_3d(
    *,
    dataset: Any,
    latents: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_color_map: dict[int, str],
    out_file: Path,
    point_scale: float,
    target_points: int,
    selection_features: np.ndarray | None = None,
    source_names_by_sample: Sequence[str] | None = None,
    include_plotlyjs: bool | str = "cdn",
) -> dict[str, Any]:
    """Render one rotatable 3D local structure for every cluster on one screen."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError("Interactive cluster representatives require Plotly.") from exc

    labels = np.asarray(cluster_labels, dtype=np.int64).reshape(-1)
    latent_array = np.asarray(latents, dtype=np.float32)
    if latent_array.ndim != 2 or len(latent_array) != len(labels):
        raise ValueError(
            "Interactive cluster gallery expects latents (N,D) aligned with labels (N,), "
            f"got {latent_array.shape} and {labels.shape}."
        )
    representatives = _compute_cluster_representative_indices(
        latent_array,
        labels,
        selection_features=selection_features,
    )
    cluster_ids = sorted(representatives)
    missing_colors = [cluster_id for cluster_id in cluster_ids if cluster_id not in cluster_color_map]
    if missing_colors:
        raise KeyError(
            "Interactive cluster gallery is missing colors for clusters "
            f"{missing_colors}."
        )

    records: list[dict[str, Any]] = []
    structures: list[np.ndarray] = []
    for cluster_id in cluster_ids:
        sample_index = int(representatives[cluster_id])
        local = _prepare_local_structure(
            dataset,
            sample_index,
            point_scale=float(point_scale),
            target_points=int(target_points),
        )
        oriented, orientation_info = _orient_points_for_crystal_view(local, method="pca")
        source_name = ""
        if source_names_by_sample is not None:
            source_name = str(source_names_by_sample[sample_index])
        records.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_label": f"C{cluster_id + 1}",
                "sample_index": sample_index,
                "source_name": source_name,
                "num_points": int(len(oriented)),
                "orientation": orientation_info,
            }
        )
        structures.append(np.asarray(oriented, dtype=np.float32))

    column_count = min(3, len(cluster_ids))
    row_count = int(np.ceil(len(cluster_ids) / column_count))
    specs: list[list[dict[str, str] | None]] = []
    titles: list[str] = []
    for row in range(row_count):
        spec_row: list[dict[str, str] | None] = []
        for column in range(column_count):
            index = row * column_count + column
            if index < len(cluster_ids):
                spec_row.append({"type": "scene"})
                record = records[index]
                source_text = f" · {record['source_name']}" if record["source_name"] else ""
                titles.append(
                    f"{record['cluster_label']} · sample {record['sample_index']}{source_text}"
                )
            else:
                spec_row.append(None)
        specs.append(spec_row)
    figure = make_subplots(
        rows=row_count,
        cols=column_count,
        specs=specs,
        subplot_titles=titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.06,
    )
    axis_limit = max(float(np.max(np.abs(points))) for points in structures) * 1.08
    axis_limit = max(axis_limit, 1.0e-4)
    scene_layout = {
        "xaxis": {"title": "PCA x", "range": [-axis_limit, axis_limit]},
        "yaxis": {"title": "PCA y", "range": [-axis_limit, axis_limit]},
        "zaxis": {"title": "PCA z", "range": [-axis_limit, axis_limit]},
        "aspectmode": "cube",
        "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.15}},
    }
    for index, (record, points) in enumerate(zip(records, structures, strict=True)):
        edge_trace, edge_info = _plotly_edge_trace(points)
        record["edge_count"] = int(edge_info["num_edges"])
        colors = _plotly_rgb_strings(
            _compute_center_to_edge_colors(
                points, str(cluster_color_map[int(record["cluster_id"])])
            )
        )
        radii = np.linalg.norm(points, axis=1)
        figure.add_trace(
            edge_trace,
            row=index // column_count + 1,
            col=index % column_count + 1,
        )
        figure.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                showlegend=False,
                customdata=np.column_stack(
                    (np.arange(len(points), dtype=np.int64), radii)
                ),
                marker={
                    "size": 8.5,
                    "color": colors,
                    "opacity": 0.98,
                    "line": {"color": "#222222", "width": 1.0},
                },
                hovertemplate=(
                    f"{record['cluster_label']} · sample {record['sample_index']}"
                    "<br>atom row=%{customdata[0]:.0f}<br>radius=%{customdata[1]:.4f}"
                    "<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>"
                ),
            ),
            row=index // column_count + 1,
            col=index % column_count + 1,
        )
    layout_updates: dict[str, Any] = {
        "title": {"text": "Interactive 3D cluster representatives"},
        "template": None,
        "height": 450 * row_count,
        "margin": {"l": 5, "r": 5, "t": 75, "b": 10},
    }
    for index in range(len(records)):
        scene_name = "scene" if index == 0 else f"scene{index + 1}"
        layout_updates[scene_name] = dict(scene_layout)
    figure.update_layout(**layout_updates)
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(out_file), include_plotlyjs=include_plotlyjs)
    return {
        "out_file": str(out_file),
        "selection": "nearest real sample to each cluster center",
        "orientation": "independent deterministic PCA orientation per local structure",
        "representatives": records,
    }


def write_transition_representatives_index(
    out_file: Path,
    transition_summaries: Sequence[dict[str, Any]],
    *,
    cluster_gallery_path: Path | None = None,
) -> None:
    entries = []
    for summary in transition_summaries:
        path = Path(str(summary["out_file"]))
        cluster_ids = [int(value) for value in summary["cluster_ids"]]
        if summary["transition_kind"] == "connected_pair":
            if len(cluster_ids) != 2:
                raise ValueError(
                    "Connected-pair transition summary must contain two cluster IDs, "
                    f"got {cluster_ids}."
                )
            label = f"C{cluster_ids[0] + 1} → C{cluster_ids[1] + 1}"
        elif summary["transition_kind"] == "within_cluster":
            if len(cluster_ids) != 1:
                raise ValueError(
                    "Within-cluster transition summary must contain one cluster ID, "
                    f"got {cluster_ids}."
                )
            label = f"C{cluster_ids[0] + 1} within-cluster path"
        else:
            raise ValueError(
                "Unknown transition summary kind "
                f"{summary['transition_kind']!r} for {path}."
            )
        entries.append(
            f'<li><a href="{escape(path.name)}">{escape(label)}</a> '
            f'({len(summary["steps"])} real structures)</li>'
        )
    gallery_entry = ""
    if cluster_gallery_path is not None:
        gallery_entry = (
            "<h2>All clusters</h2>"
            f'<p><a href="{escape(Path(cluster_gallery_path).name)}">'
            "Interactive 3D representative for every cluster</a></p>"
        )
    Path(out_file).write_text(
        """<!doctype html><meta charset="utf-8"><title>Connected-regime 3D representatives</title>
<style>body{font:16px sans-serif;max-width:900px;margin:2.5rem auto;line-height:1.5}
li{margin:.8rem 0}</style>
<h1>Real-structure paths in latent space</h1>
<p>Use each manual slider to inspect real sampled structures along a latent-space direction.
Connected-pair paths run from one cluster center through their boundary to the other center.
Clusters outside every connected pair receive a path along their dominant internal PCA direction.
Atom coordinates are never interpolated.</p>
GALLERY<h2>Structure transitions</h2><ul>ENTRIES</ul>""".replace(
            "GALLERY", gallery_entry
        ).replace("ENTRIES", "".join(entries)),
        encoding="utf-8",
    )


__all__ = [
    "render_connected_pair_transition_3d",
    "render_cluster_representatives_3d",
    "render_within_cluster_transition_3d",
    "select_transition_representative_rows",
    "select_within_cluster_representative_rows",
    "write_transition_representatives_index",
]
