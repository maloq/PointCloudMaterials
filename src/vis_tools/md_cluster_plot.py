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

    if palette and hasattr(px.colors.qualitative, palette):
        return getattr(px.colors.qualitative, palette)
    return px.colors.qualitative.Set3


def save_interactive_md_plot(
    coords: np.ndarray,
    clusters: np.ndarray,
    out_file: Path,
    *,
    palette: str | None = "Set3",
    max_points: int | None = None,
    marker_size: float = 3.0,
    marker_line_width: float = 0.0,
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("Plotly is required for interactive MD plots.") from exc

    if coords.size == 0 or clusters.size != len(coords):
        raise ValueError("Invalid coords/clusters for interactive plot.")

    coords_plot = coords
    clusters_plot = clusters
    if max_points is not None and len(coords) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(coords), size=max_points, replace=False)
        coords_plot = coords[idx]
        clusters_plot = clusters[idx]

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
                    size=marker_size,
                    color=color,
                    line=dict(color="black", width=marker_line_width),
                    opacity=0.8,
                ),
                hovertemplate=(
                    "cluster=%{text}<br>"
                    "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"
                ),
                text=[str(int(label))] * int(mask.sum()),
                legendgroup=str(int(label)),
            )
        )

    fig.update_layout(
        title=f"MD local-structure clusters (n={len(coords_plot)}, k={len(unique_labels)})",
        legend_title_text="cluster",
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


def render_interactive_md_clusters(
    analysis_dir: Path,
    *,
    out_file: Path | None = None,
    palette: str | None = "Set3",
    max_points: int | None = None,
    marker_size: float = 3.0,
    marker_line_width: float = 0.0,
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
    )
    return Path(out_file)
