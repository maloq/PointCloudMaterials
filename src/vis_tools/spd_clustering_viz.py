from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt


def extract_ground_truth_labels(atoms: np.ndarray, metadata: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract ground truth phase and grain labels for all atoms."""
    num_atoms = len(atoms)
    phase_labels = np.full(num_atoms, "unknown", dtype=object)
    grain_labels = np.full(num_atoms, -1, dtype=int)

    for grain in metadata.get("grains", []):
        indices = np.array(grain.get("atom_indices", []), dtype=int)
        valid = indices[(indices >= 0) & (indices < num_atoms)]
        phase_labels[valid] = grain["base_phase_id"]
        grain_labels[valid] = int(grain["grain_id"])

    for region in metadata.get("intermediate_regions", []):
        indices = np.array(region.get("atom_indices", []), dtype=int)
        valid = indices[(indices >= 0) & (indices < num_atoms)]
        phase_labels[valid] = region.get("intermediate_phase_id", "intermediate")
        grain_labels[valid] = -1

    return phase_labels, grain_labels


def create_visualization(
    atoms: np.ndarray,
    metadata: Dict[str, Any],
    sample_coords: np.ndarray,
    phase_labels_sample: np.ndarray,
    grain_labels_sample: np.ndarray,
    kmeans_labels: np.ndarray,
    box_size: float,
    output_path: Path,
    max_points_per_panel: int = 6000,
    dbscan_labels: Optional[np.ndarray] = None,
    hdbscan_labels: Optional[np.ndarray] = None,
) -> None:
    """Create visualization comparing ground truth and predictions."""
    atom_phase_labels, atom_grain_labels = extract_ground_truth_labels(atoms, metadata)

    n_atoms = len(atoms)
    if n_atoms > max_points_per_panel:
        atom_sample_indices = np.random.choice(n_atoms, max_points_per_panel, replace=False)
    else:
        atom_sample_indices = np.arange(n_atoms)

    atoms_sample = atoms[atom_sample_indices]
    atom_phase_sample = atom_phase_labels[atom_sample_indices]
    atom_grain_sample = atom_grain_labels[atom_sample_indices]

    n_samples = len(kmeans_labels)
    if n_samples > max_points_per_panel:
        pred_sample_indices = np.random.choice(n_samples, max_points_per_panel, replace=False)
    else:
        pred_sample_indices = np.arange(n_samples)

    coords_sample = sample_coords[pred_sample_indices]
    kmeans_sample = kmeans_labels[pred_sample_indices]
    dbscan_sample = dbscan_labels[pred_sample_indices] if dbscan_labels is not None else None
    hdbscan_sample = hdbscan_labels[pred_sample_indices] if hdbscan_labels is not None else None
    phase_sample = phase_labels_sample[pred_sample_indices]
    grain_sample = grain_labels_sample[pred_sample_indices]

    payload = {
        "atoms_sample": atoms_sample,
        "atom_phase_sample": atom_phase_sample,
        "atom_grain_sample": atom_grain_sample,
        "coords_sample": coords_sample,
        "kmeans_sample": kmeans_sample,
        "dbscan_sample": dbscan_sample,
        "hdbscan_sample": hdbscan_sample,
        "phase_sample": phase_sample,
        "grain_sample": grain_sample,
    }

    view_presets = [
        {"label": "Default View", "elev": None, "azim": None, "diagonal_cut": False},
        {"label": "High-Left View", "elev": 55, "azim": -35, "diagonal_cut": False},
        {"label": "Low-Right View", "elev": 15, "azim": 135, "diagonal_cut": False},
        {"label": "Diagonal Cut View", "elev": 30, "azim": 45, "diagonal_cut": True, "cut_ratio": 0.65},
    ]

    n_views = len(view_presets)
    include_dbscan = dbscan_sample is not None
    include_hdbscan = hdbscan_sample is not None
    n_panels = 4 + int(include_dbscan) + int(include_hdbscan)  # Removed phase agreement panel
    fig = plt.figure(figsize=(4 * n_panels, 8 * n_views))
    fig.patch.set_facecolor("white")
    fig.patch.set_edgecolor("black")
    border_width = 72.0 / fig.dpi
    fig.patch.set_linewidth(border_width)

    for row_idx, preset in enumerate(view_presets):
        row_axes: List[Any] = []
        for col_idx in range(n_panels):
            ax = fig.add_subplot(n_views, n_panels, row_idx * n_panels + col_idx + 1, projection="3d")
            row_axes.append(ax)

        _populate_clustering_panels(
            axes=row_axes,
            payload=payload,
            box_size=box_size,
            include_dbscan=include_dbscan,
            include_hdbscan=include_hdbscan,
            diagonal_cut=preset["diagonal_cut"],
            cut_ratio=preset.get("cut_ratio", 0.65),
            title_suffix=f" ({preset['label']})" if preset["label"] else "",
            border_width=border_width,
        )
        _apply_camera_view(row_axes, elev=preset["elev"], azim=preset["azim"])

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined visualization to {output_path}")


def _populate_clustering_panels(
    axes: List[Any],
    payload: Dict[str, Any],
    box_size: float,
    include_dbscan: bool,
    include_hdbscan: bool,
    diagonal_cut: bool = False,
    cut_ratio: float = 0.65,
    title_suffix: str = "",
    border_width: float = 1.0,
) -> None:
    """Populate clustering panels into the provided axes."""
    atoms_sample = payload["atoms_sample"]
    atom_phase_sample = payload["atom_phase_sample"]
    atom_grain_sample = payload["atom_grain_sample"]
    coords_sample = payload["coords_sample"]
    kmeans_sample = payload["kmeans_sample"]
    dbscan_sample = payload.get("dbscan_sample")
    hdbscan_sample = payload.get("hdbscan_sample")
    phase_sample = payload["phase_sample"]
    grain_sample = payload["grain_sample"]

    expected_axes = 4 + int(include_dbscan) + int(include_hdbscan)  # Removed phase agreement panel
    if len(axes) != expected_axes:
        raise ValueError(f"Expected {expected_axes} axes, received {len(axes)}")

    if diagonal_cut:
        atom_mask = _diagonal_cut_mask(atoms_sample, keep_ratio=cut_ratio)
        if atom_mask.size and np.any(atom_mask):
            atoms_sample = atoms_sample[atom_mask]
            atom_phase_sample = atom_phase_sample[atom_mask]
            atom_grain_sample = atom_grain_sample[atom_mask]

        coords_mask = _diagonal_cut_mask(coords_sample, keep_ratio=cut_ratio)
        if coords_mask.size and np.any(coords_mask):
            coords_sample = coords_sample[coords_mask]
            kmeans_sample = kmeans_sample[coords_mask]
            if include_dbscan and dbscan_sample is not None:
                dbscan_sample = dbscan_sample[coords_mask]
            if include_hdbscan and hdbscan_sample is not None:
                hdbscan_sample = hdbscan_sample[coords_mask]
            phase_sample = phase_sample[coords_mask]
            grain_sample = grain_sample[coords_mask]

    axis_idx = 0

    ax1 = axes[axis_idx]
    axis_idx += 1
    unique_phases = np.unique(atom_phase_sample)
    phase_colors = _build_color_map(unique_phases, "tab20")

    for phase in unique_phases:
        mask = atom_phase_sample == phase
        if not np.any(mask):
            continue
        points = atoms_sample[mask]
        ax1.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            color=phase_colors[phase],
            s=12,
            depthshade=True,
            edgecolors="black",
            linewidths=0.2,
        )
    ax1.set_title(f"Ground Truth: Phases{title_suffix}")
    _set_cube_axes(ax1, box_size)
    _add_axes_border(ax1, linewidth=border_width)

    ax2 = axes[axis_idx]
    axis_idx += 1
    unique_grains = np.unique(atom_grain_sample[atom_grain_sample >= 0])
    grain_colors = _build_color_map(unique_grains, "gist_ncar")

    for grain in unique_grains:
        mask = atom_grain_sample == grain
        if not np.any(mask):
            continue
        points = atoms_sample[mask]
        ax2.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            color=grain_colors[grain],
            s=12,
            depthshade=True,
            edgecolors="black",
            linewidths=0.2,
        )

    boundary_mask = atom_grain_sample == -1
    if np.any(boundary_mask):
        boundary_points = atoms_sample[boundary_mask]
        ax2.scatter(
            boundary_points[:, 0], boundary_points[:, 1], boundary_points[:, 2],
            color="purple",
            s=14,
            depthshade=True,
            edgecolors="black",
            linewidths=0.25,
            alpha=0.8,
        )
    ax2.set_title(f"Ground Truth: Grains{title_suffix}")
    _set_cube_axes(ax2, box_size)
    _add_axes_border(ax2, linewidth=border_width)

    ax3 = axes[axis_idx]
    axis_idx += 1
    if np.any(boundary_mask):
        boundary_points = atoms_sample[boundary_mask]
        ax3.scatter(
            boundary_points[:, 0], boundary_points[:, 1], boundary_points[:, 2],
            color="purple",
            s=14,
            depthshade=True,
            edgecolors="black",
            linewidths=0.25,
            alpha=0.8,
        )
    else:
        intermediate_mask = np.array([str(p).startswith("intermediate") for p in atom_phase_sample])
        if np.any(intermediate_mask):
            inter_points = atoms_sample[intermediate_mask]
            ax3.scatter(
                inter_points[:, 0], inter_points[:, 1], inter_points[:, 2],
                color="purple",
                s=14,
                depthshade=True,
                edgecolors="black",
                linewidths=0.25,
                alpha=0.8,
            )
    ax3.set_title(f"Ground Truth: Boundaries{title_suffix}")
    _set_cube_axes(ax3, box_size)
    _add_axes_border(ax3, linewidth=border_width)

    ax4 = axes[axis_idx]
    axis_idx += 1
    unique_kmeans = np.unique(kmeans_sample)
    kmeans_colors = _build_color_map(unique_kmeans, "tab20")

    for cluster in unique_kmeans:
        mask = kmeans_sample == cluster
        if not np.any(mask):
            continue
        points = coords_sample[mask]
        ax4.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            color=kmeans_colors[cluster],
            s=12,
            depthshade=True,
            edgecolors="black",
            linewidths=0.2,
        )
    ax4.set_title(f"KMeans Clusters (k={len(unique_kmeans)}){title_suffix}")
    _set_cube_axes(ax4, box_size)
    _add_axes_border(ax4, linewidth=border_width)

    if include_dbscan and dbscan_sample is not None:
        ax_dbscan = axes[axis_idx]
        axis_idx += 1
        _plot_density_clusters(
            ax_dbscan,
            coords_sample,
            dbscan_sample,
            method_label="DBSCAN",
            box_size=box_size,
            title_suffix=title_suffix,
            border_width=border_width,
        )

    if include_hdbscan and hdbscan_sample is not None:
        ax_hdbscan = axes[axis_idx]
        axis_idx += 1
        _plot_density_clusters(
            ax_hdbscan,
            coords_sample,
            hdbscan_sample,
            method_label="HDBSCAN",
            box_size=box_size,
            title_suffix=title_suffix,
            border_width=border_width,
        )


def _plot_density_clusters(
    ax: Any,
    coords_sample: np.ndarray,
    labels: np.ndarray,
    method_label: str,
    box_size: float,
    title_suffix: str,
    border_width: float,
) -> None:
    """Scatter plot for density-based clustering predictions."""
    if labels is None or len(labels) == 0:
        ax.text(0.5, 0.5, "No predictions", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(f"{method_label} Clusters{title_suffix} (no data)")
        _set_cube_axes(ax, box_size)
        _add_axes_border(ax, linewidth=border_width)
        return

    unique_labels = np.unique(labels)
    label_colors = _build_color_map(unique_labels, "tab20")

    for cluster in unique_labels:
        mask = labels == cluster
        if not np.any(mask):
            continue
        points = coords_sample[mask]
        color = "lightgray" if cluster == -1 else label_colors.get(cluster, "black")
        alpha = 0.7 if cluster == -1 else 1.0
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            color=color,
            s=12,
            depthshade=True,
            edgecolors="black",
            linewidths=0.2,
            alpha=alpha,
        )

    n_clusters = len(unique_labels[unique_labels >= 0])
    noise_points = int(np.sum(labels == -1))
    if noise_points > 0:
        title = f"{method_label} Clusters ({n_clusters} + noise){title_suffix}"
    else:
        title = f"{method_label} Clusters (k={n_clusters}){title_suffix}"
    ax.set_title(title)
    _set_cube_axes(ax, box_size)
    _add_axes_border(ax, linewidth=border_width)


def _apply_camera_view(axes: List[Any], elev: float | None, azim: float | None) -> None:
    """Apply a consistent camera view across axes."""
    for ax in axes:
        current_elev = getattr(ax, "elev", 30)
        current_azim = getattr(ax, "azim", -60)
        ax.view_init(
            elev=current_elev if elev is None else elev,
            azim=current_azim if azim is None else azim,
        )


def _diagonal_cut_mask(points: np.ndarray, keep_ratio: float = 0.65) -> np.ndarray:
    """Return mask that keeps points closer to the origin along the main diagonal."""
    if points.size == 0:
        return np.zeros(len(points), dtype=bool)
    diag_vals = points.sum(axis=1)
    diag_min = diag_vals.min()
    diag_max = diag_vals.max()
    if np.isclose(diag_min, diag_max):
        return np.ones(len(points), dtype=bool)

    ratio = float(np.clip(keep_ratio, 0.0, 1.0))
    cutoff = diag_min + (diag_max - diag_min) * ratio
    mask = diag_vals <= cutoff

    if not np.any(mask):
        cutoff = np.quantile(diag_vals, 0.75)
        mask = diag_vals <= cutoff

    return mask


def _build_color_map(unique_values: np.ndarray, cmap_name: str = "tab20") -> Dict[Any, Any]:
    """Build a color map for unique values."""
    unique_sorted = sorted(unique_values)
    if not unique_sorted:
        return {}

    cmap = cm.get_cmap(cmap_name)
    denom = max(1, len(unique_sorted) - 1)
    return {
        val: tuple(map(float, cmap(i / denom)))
        for i, val in enumerate(unique_sorted)
    }


def _set_cube_axes(ax: Any, box_size: float) -> None:
    """Set axes limits for a cubic box."""
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_zlim(0, box_size)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def _add_axes_border(ax: Any, linewidth: float = 1.0, color: str = "black") -> None:
    """Add border to 3D axes."""
    try:
        ax.patch.set_edgecolor(color)
        ax.patch.set_linewidth(linewidth)
    except Exception:
        pass
    for spine in getattr(ax, "spines", {}).values():
        spine.set_linewidth(linewidth)
        spine.set_color(color)


__all__ = [
    "create_visualization",
]
