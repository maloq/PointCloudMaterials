from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go


_TAB10 = [
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


def load_coords_clusters(analysis_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the NPZ written by ``save_local_structure_assignments``."""
    npz_path = Path(analysis_dir) / "local_structure_coords_clusters.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(
            "Missing repository MD-cluster output "
            f"{npz_path}. Run the clustering analysis first."
        )
    data = np.load(npz_path)
    return data["coords"], data["clusters"]


def _resolve_palette(palette: str) -> list[str]:
    if palette == "tab10":
        return _TAB10
    if not hasattr(px.colors.qualitative, palette):
        available = sorted(
            name
            for name in vars(px.colors.qualitative)
            if not name.startswith("_")
        )
        raise ValueError(
            f"Unknown Plotly qualitative palette {palette!r}; available={available}."
        )
    return list(getattr(px.colors.qualitative, palette))


def save_interactive_md_plot(
    coords: np.ndarray,
    clusters: np.ndarray,
    out_file: Path,
    *,
    palette: str = "tab10",
    cluster_color_map: dict[int, str] | None = None,
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
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape (N, 3), got {coords.shape}.")
    if clusters.ndim != 1 or clusters.shape[0] != coords.shape[0]:
        raise ValueError(
            "clusters must have shape (N,) matching coords, "
            f"got coords={coords.shape}, clusters={clusters.shape}."
        )
    if hover_values is not None and (
        hover_values.ndim != 1 or hover_values.shape[0] != coords.shape[0]
    ):
        raise ValueError(
            "hover_values must have shape (N,) matching coords, "
            f"got coords={coords.shape}, hover_values={hover_values.shape}."
        )
    if coords.shape[0] == 0:
        raise ValueError("Cannot render an empty MD point set.")

    coords_plot = coords
    clusters_plot = clusters
    hover_values_plot = hover_values
    if max_points is not None and coords.shape[0] > max_points:
        indices = np.random.default_rng(0).choice(
            coords.shape[0],
            size=max_points,
            replace=False,
        )
        coords_plot = coords[indices]
        clusters_plot = clusters[indices]
        if hover_values is not None:
            hover_values_plot = hover_values[indices]

    coords_plot = np.round(coords_plot.astype(np.float32, copy=False), 4)
    if hover_values_plot is not None:
        hover_values_plot = np.round(
            hover_values_plot.astype(np.float32, copy=False),
            4,
        )

    palette_colors = _resolve_palette(palette)
    unique_labels = np.unique(clusters_plot)
    show_legend = unique_labels.size <= legend_max_items
    figure = go.Figure()
    for palette_index, label in enumerate(unique_labels):
        mask = clusters_plot == label
        label_id = int(label)
        color = (
            cluster_color_map[label_id]
            if cluster_color_map is not None and label_id in cluster_color_map
            else palette_colors[palette_index % len(palette_colors)]
        )
        custom_data = None
        hover_template = (
            f"{label_prefix} {label_id}<br>"
            "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}"
        )
        if hover_values_plot is not None:
            custom_data = hover_values_plot[mask, None]
            hover_template += f"<br>{hover_label}=%{{customdata[0]:.3f}}"
        hover_template += "<extra></extra>"
        figure.add_trace(
            go.Scatter3d(
                x=coords_plot[mask, 0],
                y=coords_plot[mask, 1],
                z=coords_plot[mask, 2],
                mode="markers",
                name=f"{label_prefix} {label_id}",
                marker={
                    "size": marker_size,
                    "color": color,
                    "line": {"color": "black", "width": marker_line_width},
                    "opacity": 0.8,
                },
                hovertemplate=hover_template,
                customdata=custom_data,
                legendgroup=str(label_id),
                showlegend=show_legend,
            )
        )

    figure.update_layout(
        title=title
        or f"MD local-structure clusters (n={coords_plot.shape[0]}, k={unique_labels.size})",
        legend_title_text=label_prefix.lower(),
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
            "aspectmode": aspect_mode,
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    output = Path(out_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output), include_plotlyjs="cdn")


def render_interactive_md_clusters(
    analysis_dir: Path,
    *,
    out_file: Path | None = None,
    palette: str = "tab10",
    max_points: int | None = None,
    marker_size: float = 3.0,
    marker_line_width: float = 0.0,
    aspect_mode: str = "cube",
) -> Path:
    coords, clusters = load_coords_clusters(analysis_dir)
    output = Path(analysis_dir) / "md_space_clusters.html" if out_file is None else out_file
    save_interactive_md_plot(
        coords,
        clusters,
        output,
        palette=palette,
        max_points=max_points,
        marker_size=marker_size,
        marker_line_width=marker_line_width,
        aspect_mode=aspect_mode,
    )
    return Path(output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an interactive MD cluster plot from an analysis directory."
    )
    parser.add_argument(
        "analysis_dir",
        type=Path,
        help="Directory containing local_structure_coords_clusters.npz.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output HTML file (default: <analysis_dir>/md_space_clusters.html).",
    )
    parser.add_argument(
        "--palette",
        default="Set3",
        help="Plotly qualitative palette name (default: Set3).",
    )
    parser.add_argument(
        "--max-points",
        "--max_points",
        dest="max_points",
        type=int,
        default=None,
        help="Optional cap on points in the interactive plot.",
    )
    parser.add_argument(
        "--marker-size",
        "--marker_size",
        dest="marker_size",
        type=float,
        default=3.0,
        help="Marker size (default: 3.0).",
    )
    parser.add_argument(
        "--marker-line-width",
        "--marker_line_width",
        dest="marker_line_width",
        type=float,
        default=0.0,
        help="Marker outline width in pixels (default: 0.0).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = render_interactive_md_clusters(
        args.analysis_dir,
        out_file=args.out,
        palette=args.palette,
        max_points=args.max_points,
        marker_size=args.marker_size,
        marker_line_width=args.marker_line_width,
    )
    print(f"Saved interactive plot to {output}")


if __name__ == "__main__":
    main()
