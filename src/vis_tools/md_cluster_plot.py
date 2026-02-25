from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def load_coords_clusters(analysis_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    analysis_dir = Path(analysis_dir)
    npz_path = analysis_dir / "local_structure_coords_clusters.npz"
    csv_path = analysis_dir / "local_structure_coords_clusters.csv"

    if npz_path.exists():
        data = np.load(npz_path)
        coords = data["coords"]
        clusters = data["clusters"]
        return coords, clusters

    if csv_path.exists():
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        coords = data[:, 1:4]
        clusters = data[:, 4].astype(int)
        return coords, clusters

    raise FileNotFoundError(
        f"Missing local structure files under {analysis_dir}. Expected "
        f"{npz_path.name} or {csv_path.name}."
    )


def _resolve_palette(palette: str | None):
    try:
        import plotly.express as px
    except ImportError as exc:
        raise ImportError("Plotly is required for interactive MD plots.") from exc

    tab10 = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    palette_norm = "" if palette is None else str(palette).strip().lower()
    if palette_norm in {"tab10", "t10"}:
        return tab10
    if palette and hasattr(px.colors.qualitative, palette):
        return list(getattr(px.colors.qualitative, palette))
    if palette and str(palette).strip().lower() == "accent":
        return [
            "#7fc97f",
            "#beaed4",
            "#fdc086",
            "#ffff99",
            "#386cb0",
            "#f0027f",
            "#bf5b17",
            "#666666",
        ]
    return tab10


def _normalize_sizes(
    values: np.ndarray,
    *,
    size_range: Tuple[float, float] = (3.0, 8.0),
    clip_percentiles: Tuple[float, float] = (5.0, 95.0),
) -> np.ndarray:
    values = np.asarray(values)
    sizes = np.full(values.shape, size_range[0], dtype=np.float32)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return sizes

    v = values[finite_mask].astype(np.float32)
    v_min = np.percentile(v, clip_percentiles[0])
    v_max = np.percentile(v, clip_percentiles[1])
    if not np.isfinite(v_min) or not np.isfinite(v_max) or v_max <= v_min:
        sizes[finite_mask] = size_range[0]
        return sizes
    scaled = (v - v_min) / (v_max - v_min)
    scaled = np.clip(scaled, 0.0, 1.0)
    sizes[finite_mask] = size_range[0] + scaled * (size_range[1] - size_range[0])
    return sizes


def save_interactive_md_plot(
    coords: np.ndarray,
    clusters: np.ndarray,
    out_file: Path,
    *,
    palette: str | None = "tab10",
    max_points: int | None = None,
    marker_size: float = 3.0,
    marker_line_width: float = 0.0,
    title: str | None = None,
    label_prefix: str = "Cluster",
    hover_values: np.ndarray | None = None,
    hover_label: str = "value",
    aspect_mode: str = "cube",
    legend_max_items: int = 60,
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("Plotly is required for interactive MD plots.") from exc

    if coords.size == 0 or clusters.size != len(coords):
        raise ValueError("Invalid coords/clusters for interactive plot.")

    coords_plot = coords
    clusters_plot = clusters
    hover_values_plot = hover_values
    if max_points is not None and len(coords) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(coords), size=max_points, replace=False)
        coords_plot = coords[idx]
        clusters_plot = clusters[idx]
        if hover_values is not None:
            hover_values_plot = hover_values[idx]

    # Reduce payload size in HTML/JSON while preserving visual structure.
    coords_plot = np.round(np.asarray(coords_plot, dtype=np.float32), 4)
    clusters_plot = np.asarray(clusters_plot)
    if hover_values_plot is not None:
        hover_values_plot = np.round(np.asarray(hover_values_plot, dtype=np.float32), 4)

    palette_colors = _resolve_palette(palette)
    unique_labels = np.unique(clusters_plot)
    show_legend = len(unique_labels) <= max(0, int(legend_max_items))

    fig = go.Figure()
    for i, label in enumerate(unique_labels):
        mask = clusters_plot == label
        color = palette_colors[i % len(palette_colors)]
        customdata = None
        hovertemplate = (
            f"{label_prefix} {int(label)}<br>"
            "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}"
        )
        if hover_values_plot is not None:
            customdata = np.asarray(hover_values_plot)[mask].reshape(-1, 1)
            hovertemplate += f"<br>{hover_label}=%{{customdata[0]:.3f}}"
        hovertemplate += "<extra></extra>"
        fig.add_trace(
            go.Scatter3d(
                x=coords_plot[mask, 0],
                y=coords_plot[mask, 1],
                z=coords_plot[mask, 2],
                mode="markers",
                name=f"{label_prefix} {int(label)}",
                marker=dict(
                    size=marker_size,
                    color=color,
                    line=dict(color="black", width=marker_line_width),
                    opacity=0.8,
                ),
                hovertemplate=hovertemplate,
                customdata=customdata,
                legendgroup=str(int(label)),
                showlegend=show_legend,
            )
        )

    fig.update_layout(
        title=title
        or f"MD local-structure clusters (n={len(coords_plot)}, k={len(unique_labels)})",
        legend_title_text=label_prefix.lower(),
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode=aspect_mode,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_file), include_plotlyjs="cdn")


def render_interactive_md_clusters(
    analysis_dir: Path,
    *,
    out_file: Path | None = None,
    palette: str | None = "tab10",
    max_points: int | None = None,
    marker_size: float = 3.0,
    marker_line_width: float = 0.0,
    aspect_mode: str = "cube",
) -> Path:
    coords, clusters = load_coords_clusters(analysis_dir)
    if out_file is None:
        out_file = Path(analysis_dir) / "md_space_clusters.html"
    save_interactive_md_plot(
        coords,
        clusters,
        out_file,
        palette=palette,
        max_points=max_points,
        marker_size=marker_size,
        marker_line_width=marker_line_width,
        aspect_mode=aspect_mode,
    )
    return Path(out_file)


def save_interactive_md_continuous_plot(
    coords: np.ndarray,
    values: np.ndarray,
    out_file: Path,
    *,
    max_points: int | None = None,
    marker_size: float = 3.5,
    colorscale: str = "Viridis",
    title: str | None = None,
    value_label: str = "value",
    opacity: float = 0.85,
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("Plotly is required for interactive MD plots.") from exc

    if coords.size == 0 or values.size != len(coords):
        raise ValueError("Invalid coords/values for continuous MD plot.")

    coords_plot = coords
    values_plot = values
    if max_points is not None and len(coords) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(coords), size=max_points, replace=False)
        coords_plot = coords[idx]
        values_plot = values[idx]

    values_plot = np.asarray(values_plot)
    finite_mask = np.isfinite(values_plot)
    if not finite_mask.any():
        raise ValueError("No finite values for continuous MD plot.")

    vmin = float(np.nanmin(values_plot))
    vmax = float(np.nanmax(values_plot))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=coords_plot[finite_mask, 0],
            y=coords_plot[finite_mask, 1],
            z=coords_plot[finite_mask, 2],
            mode="markers",
            name=value_label,
            marker=dict(
                size=marker_size,
                color=values_plot[finite_mask],
                colorscale=colorscale,
                cmin=vmin,
                cmax=vmax,
                colorbar=dict(title=value_label),
                opacity=opacity,
            ),
            hovertemplate=(
                "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}"
                f"<br>{value_label}=%{{marker.color:.3f}}<extra></extra>"
            ),
        )
    )

    if (~finite_mask).any():
        fig.add_trace(
            go.Scatter3d(
                x=coords_plot[~finite_mask, 0],
                y=coords_plot[~finite_mask, 1],
                z=coords_plot[~finite_mask, 2],
                mode="markers",
                name="missing",
                marker=dict(size=marker_size, color="lightgray", opacity=0.4),
                hovertemplate=(
                    "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<br>"
                    "value=nan<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title or f"MD values ({value_label})",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_file), include_plotlyjs="cdn")


def save_interactive_md_two_layer_plot(
    coords: np.ndarray,
    clusters: np.ndarray,
    values: np.ndarray,
    out_file: Path,
    *,
    palette: str | None = "tab10",
    max_points: int | None = None,
    base_marker_size: float = 2.5,
    base_opacity: float = 0.35,
    overlay_marker_size: float = 4.0,
    overlay_opacity: float = 0.85,
    overlay_sizes: np.ndarray | None = None,
    overlay_size_range: Tuple[float, float] = (3.0, 8.0),
    colorscale: str = "Viridis",
    title: str | None = None,
    value_label: str = "value",
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("Plotly is required for interactive MD plots.") from exc

    if coords.size == 0 or clusters.size != len(coords) or values.size != len(coords):
        raise ValueError("Invalid coords/clusters/values for two-layer MD plot.")

    coords_plot = coords
    clusters_plot = clusters
    values_plot = values
    overlay_sizes_plot = overlay_sizes
    if max_points is not None and len(coords) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(coords), size=max_points, replace=False)
        coords_plot = coords[idx]
        clusters_plot = clusters[idx]
        values_plot = values[idx]
        if overlay_sizes is not None:
            overlay_sizes_plot = overlay_sizes[idx]

    palette_colors = _resolve_palette(palette)
    unique_labels = np.unique(clusters_plot)

    fig = go.Figure()
    for i, label in enumerate(unique_labels):
        mask = clusters_plot == label
        color = palette_colors[i % len(palette_colors)]
        fig.add_trace(
            go.Scatter3d(
                x=coords_plot[mask, 0],
                y=coords_plot[mask, 1],
                z=coords_plot[mask, 2],
                mode="markers",
                name=f"Cluster {int(label)}",
                marker=dict(
                    size=base_marker_size,
                    color=color,
                    opacity=base_opacity,
                ),
                hovertemplate=(
                    "cluster=%{text}<br>"
                    "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"
                ),
                text=[str(int(label))] * int(mask.sum()),
                legendgroup=str(int(label)),
                showlegend=True,
            )
        )

    values_plot = np.asarray(values_plot)
    finite_mask = np.isfinite(values_plot)
    if finite_mask.any():
        if overlay_sizes_plot is not None:
            sizes = _normalize_sizes(
                np.asarray(overlay_sizes_plot),
                size_range=overlay_size_range,
            )
            sizes = sizes[finite_mask]
        else:
            sizes = overlay_marker_size
        vmin = float(np.nanmin(values_plot))
        vmax = float(np.nanmax(values_plot))
        fig.add_trace(
            go.Scatter3d(
                x=coords_plot[finite_mask, 0],
                y=coords_plot[finite_mask, 1],
                z=coords_plot[finite_mask, 2],
                mode="markers",
                name=value_label,
                marker=dict(
                    size=sizes,
                    color=values_plot[finite_mask],
                    colorscale=colorscale,
                    cmin=vmin,
                    cmax=vmax,
                    colorbar=dict(title=value_label),
                    opacity=overlay_opacity,
                ),
                hovertemplate=(
                    "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}"
                    f"<br>{value_label}=%{{marker.color:.3f}}<extra></extra>"
                ),
                showlegend=True,
            )
        )

    fig.update_layout(
        title=title or "MD clusters + overlay",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_file), include_plotlyjs="cdn")
