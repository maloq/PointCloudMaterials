"""
Visualization utilities for synthetic atomistic datasets.

Provides routines for rendering the global multi-panel figure and local
neighborhood galleries with optional separation of intermediate phases.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import proj3d
from scipy.spatial import cKDTree

_DIAGONAL_DIRECTION = np.array([1.0, 1.0, 1.0], dtype=float) / np.sqrt(3.0)
_DEFAULT_GRAY_RGBA = tuple(mcolors.to_rgba("gray"))


def generate_visualizations(
    global_cfg: Any,
    grains: Sequence[Dict[str, Any]],
    atoms: Sequence[Dict[str, Any]],
    metadata: Dict[str, Any],
    rng: np.random.Generator,
    output_dir: pathlib.Path,
) -> None:
    """
    Create diagnostic visualizations for the generated dataset.

    Args:
        global_cfg: GlobalConfig dataclass with box parameters.
        grains: List of grain dictionaries (with seed positions, rotations, etc.).
        atoms: Final atom records (from `_finalize_atoms`).
        metadata: Metadata dictionary populated during generation.
        rng: Random generator for reproducible sampling.
        output_dir: Directory to write figures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    atoms_array = np.asarray([atom["position"] for atom in atoms], dtype=float)
    phases = np.asarray([atom["phase_id"] for atom in atoms], dtype=str)
    grain_ids = np.array(
        [(-1 if atom.get("grain_id") is None else int(atom["grain_id"])) for atom in atoms],
        dtype=int,
    )
    color_map = _build_phase_color_map(phases)
    grain_color_map = _build_grain_color_map(grains)

    _render_global_structure(
        global_cfg,
        grains,
        atoms,
        atoms_array,
        phases,
        grain_ids,
        metadata,
        grain_color_map,
        color_map,
        output_dir / "figure_global.png",
        rng,
    )
    _render_global_structure_diagonal_cut(
        global_cfg,
        grains,
        atoms,
        atoms_array,
        phases,
        grain_ids,
        metadata,
        grain_color_map,
        color_map,
        output_dir / "figure_global_diagonal_cut.png",
        rng,
        view_angles=_view_from_vector(-_DIAGONAL_DIRECTION),
    )
    _render_closeup_view(global_cfg, atoms, atoms_array, phases, color_map, output_dir / "figure_closeup.png", rng)
    _render_local_galleries(global_cfg, atoms, atoms_array, phases, metadata, output_dir, rng)


def _render_global_structure(
    global_cfg: Any,
    grains: Sequence[Dict[str, Any]],
    atoms: Sequence[Dict[str, Any]],
    atom_positions: np.ndarray,
    phases: Sequence[str],
    grain_ids: np.ndarray,
    metadata: Dict[str, Any],
    grain_color_map: Dict[int, Any],
    color_map: Dict[str, Any],
    output_path: pathlib.Path,
    rng: np.random.Generator,
    view_angles: Optional[Tuple[float, float]] = None,
) -> None:
    _render_global_structure_core(
        global_cfg=global_cfg,
        grains=grains,
        atoms=atoms,
        atom_positions=atom_positions,
        phases=phases,
        grain_ids=grain_ids,
        metadata=metadata,
        grain_color_map=grain_color_map,
        color_map=color_map,
        output_path=output_path,
        rng=rng,
        original_indices=np.arange(len(atoms), dtype=int),
        view_angles=view_angles,
    )


def _render_global_structure_diagonal_cut(
    global_cfg: Any,
    grains: Sequence[Dict[str, Any]],
    atoms: Sequence[Dict[str, Any]],
    atom_positions: np.ndarray,
    phases: Sequence[str],
    grain_ids: np.ndarray,
    metadata: Dict[str, Any],
    grain_color_map: Dict[int, Any],
    color_map: Dict[str, Any],
    output_path: pathlib.Path,
    rng: np.random.Generator,
    view_angles: Optional[Tuple[float, float]] = None,
) -> None:
    if len(atoms) == 0:
        return

    mask = _diagonal_cut_mask(atom_positions, global_cfg.L)
    if not np.any(mask):
        return

    kept_indices = np.flatnonzero(mask)
    filtered_atoms = [atoms[int(i)] for i in kept_indices]
    filtered_positions = atom_positions[mask]
    filtered_phases = [phases[int(i)] for i in kept_indices]
    filtered_grain_ids = grain_ids[mask]

    _render_global_structure_core(
        global_cfg=global_cfg,
        grains=grains,
        atoms=filtered_atoms,
        atom_positions=filtered_positions,
        phases=filtered_phases,
        grain_ids=filtered_grain_ids,
        metadata=metadata,
        grain_color_map=grain_color_map,
        color_map=color_map,
        output_path=output_path,
        rng=rng,
        original_indices=kept_indices,
        view_angles=view_angles,
    )


def _render_global_structure_core(
    global_cfg: Any,
    grains: Sequence[Dict[str, Any]],
    atoms: Sequence[Dict[str, Any]],
    atom_positions: np.ndarray,
    phases: Sequence[str],
    grain_ids: np.ndarray,
    metadata: Dict[str, Any],
    grain_color_map: Dict[int, Any],
    color_map: Dict[str, Any],
    output_path: pathlib.Path,
    rng: np.random.Generator,
    original_indices: np.ndarray,
    view_angles: Optional[Tuple[float, float]],
) -> None:
    total_atoms = len(atoms)
    if total_atoms == 0:
        return

    original_indices = np.asarray(original_indices, dtype=int)
    orig_to_local = {orig_idx: local_idx for local_idx, orig_idx in enumerate(original_indices)}

    phase_array = np.asarray(phases, dtype=object)
    grain_ids = np.asarray(grain_ids, dtype=int)

    phase_sample_limit = 6000
    if total_atoms <= phase_sample_limit:
        phase_indices = np.arange(total_atoms, dtype=int)
    else:
        phase_indices = rng.choice(total_atoms, size=phase_sample_limit, replace=False)
    phase_indices = np.asarray(phase_indices, dtype=int)
    phase_positions = _ensure_point_array(atom_positions[phase_indices])
    phase_ids = phase_array[phase_indices]

    fig = plt.figure(figsize=(16, 10))
    border_width = 72.0 / fig.dpi
    fig.patch.set_facecolor("white")
    fig.patch.set_edgecolor("black")
    fig.patch.set_linewidth(border_width)

    # Phases overview
    ax1 = fig.add_subplot(231, projection="3d")
    for phase in color_map:
        mask = phase_ids == phase
        if not np.any(mask):
            continue
        points = phase_positions[mask]
        ax1.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=color_map[phase],
            s=12,
            depthshade=True,
            edgecolors="black",
            linewidths=0.2,
        )
    ax1.set_title("Phases")
    _set_cube_axes(ax1, global_cfg.L)
    _add_axes_border(ax1, linewidth=border_width)

    # Intermediate phases (full point set)
    ax2 = fig.add_subplot(232, projection="3d")
    for phase in color_map:
        if not str(phase).startswith("intermediate_"):
            continue
        mask = phase_array == phase
        if not np.any(mask):
            continue
        points = _ensure_point_array(atom_positions[mask])
        ax2.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=color_map.get(phase, _DEFAULT_GRAY_RGBA),
            s=12,
            depthshade=True,
            edgecolors="black",
            linewidths=0.2,
        )
    ax2.set_title("Intermediate phases")
    _set_cube_axes(ax2, global_cfg.L)
    _add_axes_border(ax2, linewidth=border_width)

    # Grain boundaries (full point set)
    ax3 = fig.add_subplot(233, projection="3d")
    intermediate_mask = np.fromiter(
        (str(phase).startswith("intermediate_") for phase in phase_array),
        dtype=bool,
        count=len(phase_array),
    )
    if np.any(intermediate_mask):
        inter_positions = _ensure_point_array(atom_positions[intermediate_mask])
        inter_phases = phase_array[intermediate_mask]
        inter_colors = [color_map.get(phase, _DEFAULT_GRAY_RGBA) for phase in inter_phases]
        ax3.scatter(
            inter_positions[:, 0],
            inter_positions[:, 1],
            inter_positions[:, 2],
            c=inter_colors,
            s=12,
            depthshade=True,
            edgecolors="black",
            linewidths=0.2,
        )

    boundary_mask, _ = _compute_boundary_indices(atom_positions, grain_ids, grains)
    if np.any(boundary_mask):
        boundary_positions = _ensure_point_array(atom_positions[boundary_mask])
        boundary_grains = grain_ids[boundary_mask]
        boundary_colors = [grain_color_map.get(int(g), _DEFAULT_GRAY_RGBA) for g in boundary_grains]
        ax3.scatter(
            boundary_positions[:, 0],
            boundary_positions[:, 1],
            boundary_positions[:, 2],
            c=boundary_colors,
            s=14,
            depthshade=True,
            edgecolors="black",
            linewidths=0.25,
        )
    ax3.set_title("Grain boundaries")
    _set_cube_axes(ax3, global_cfg.L)
    _add_axes_border(ax3, linewidth=border_width)

    # Grain orientations
    ax4 = fig.add_subplot(234, projection="3d")
    for grain in grains:
        seed = grain["seed_position"]
        rotation = grain["base_rotation"]
        axes = rotation @ np.eye(3) * global_cfg.avg_nn_dist
        ax4.quiver(
            np.full(3, seed[0]),
            np.full(3, seed[1]),
            np.full(3, seed[2]),
            axes[0, :],
            axes[1, :],
            axes[2, :],
            color=["r", "g", "b"],
            length=global_cfg.avg_nn_dist,
            normalize=False,
        )
    ax4.set_title("Grain orientations")
    _set_cube_axes(ax4, global_cfg.L)
    _add_axes_border(ax4, linewidth=border_width)

    # Perturbation overview
    ax5 = fig.add_subplot(235, projection="3d")
    perturb = metadata.get("perturbations", {})
    for bubble in perturb.get("rotation_bubbles", []):
        center = np.array(bubble["center"])
        ax5.scatter(
            *center,
            color="orange",
            s=60,
            marker="^",
            edgecolors="black",
            linewidths=0.5,
        )
    for bubble in perturb.get("density_bubbles", []):
        center = np.array(bubble["center"])
        color = "blue" if bubble["alpha"] > 0 else "red"
        ax5.scatter(
            *center,
            color=color,
            s=60,
            marker="o",
            edgecolors="black",
            linewidths=0.5,
        )
    for event in perturb.get("dropouts", {}).get("events", []):
        idx = event.get("removed_atom_final_index")
        if idx is None:
            continue
        local_idx = orig_to_local.get(int(idx))
        if local_idx is None or local_idx >= len(atom_positions):
            continue
        position = atom_positions[local_idx]
        ax5.scatter(
            *position,
            color="black",
            marker="x",
            s=50,
            linewidths=0.5,
        )
    ax5.set_title("Perturbations")
    _set_cube_axes(ax5, global_cfg.L)
    _add_axes_border(ax5, linewidth=border_width)

    fig.tight_layout()
    if view_angles is not None:
        for axis in fig.axes:
            if hasattr(axis, "view_init"):
                axis.view_init(elev=view_angles[0], azim=view_angles[1])

    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def _render_closeup_view(
    global_cfg: Any,
    atoms: Sequence[Dict[str, Any]],
    atom_positions: np.ndarray,
    phases: Sequence[str],
    color_map: Dict[str, Any],
    output_path: pathlib.Path,
    rng: np.random.Generator,
) -> None:
    """Render a close-up view showing 1/16 of the volume with full point density,
    plus a highlighted diagonal slice on the same picture.
    If 1/16 has more than 62500 points, also render a 1/32 closeup.
    """
    total_atoms = len(atoms)
    if total_atoms == 0:
        return

    # Select 1/16 of the volume (corner region)
    L = global_cfg.L
    corner_min = np.array([0, 0, 0])
    corner_max = np.array([L / 2, L / 2, L / 4])  # 1/16 volume

    # Filter atoms in this region
    mask = np.all((atom_positions >= corner_min) & (atom_positions <= corner_max), axis=1)
    closeup_indices = np.flatnonzero(mask)

    if closeup_indices.size == 0:
        return

    closeup_positions = _ensure_point_array(atom_positions[closeup_indices])
    closeup_phases = np.asarray(phases)[closeup_indices]

    # Render 1/16 closeup
    _render_single_closeup(
        closeup_positions=closeup_positions,
        closeup_phases=closeup_phases,
        color_map=color_map,
        corner_min=corner_min,
        corner_max=corner_max,
        output_path=output_path,
        num_points=len(closeup_indices),
        volume_fraction="1/16"
    )

    # If 1/16 has more than 62500 points, render 1/32 closeup
    if len(closeup_indices) > 62500:
        corner_min_32 = np.array([0, 0, 0])
        corner_max_32 = np.array([L / 2, L / 2, L / 8])  # 1/32 volume
        
        # Filter atoms in the smaller region
        mask_32 = np.all((atom_positions >= corner_min_32) & (atom_positions <= corner_max_32), axis=1)
        closeup_indices_32 = np.flatnonzero(mask_32)
        
        if closeup_indices_32.size > 0:
            closeup_positions_32 = _ensure_point_array(atom_positions[closeup_indices_32])
            closeup_phases_32 = np.asarray(phases)[closeup_indices_32]
            
            # Generate output path for 1/32 closeup
            output_path_32 = output_path.parent / output_path.name.replace("closeup.png", "closeup_32.png")
            
            _render_single_closeup(
                closeup_positions=closeup_positions_32,
                closeup_phases=closeup_phases_32,
                color_map=color_map,
                corner_min=corner_min_32,
                corner_max=corner_max_32,
                output_path=output_path_32,
                num_points=len(closeup_indices_32),
                volume_fraction="1/32"
            )


def _render_single_closeup(
    closeup_positions: np.ndarray,
    closeup_phases: np.ndarray,
    color_map: Dict[str, Any],
    corner_min: np.ndarray,
    corner_max: np.ndarray,
    output_path: pathlib.Path,
    num_points: int,
    volume_fraction: str,
) -> None:
    """Helper function to render a single closeup view with diagonal slice overlay."""
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")

    # --- original rendering of all points (unchanged) ---
    for phase in np.unique(closeup_phases):
        phase_mask = closeup_phases == phase
        points = _ensure_point_array(closeup_positions[phase_mask])
        if not points.size:
            continue
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=color_map.get(phase, _DEFAULT_GRAY_RGBA),
            s=15,
            depthshade=True,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
        )

    # --- diagonal slice overlay (added) ---
    # points within a small distance of the long diagonal get highlighted
    diag_thickness_frac = 0.02  # ~2% of the long-diagonal length; adjust as you like

    diag_p0 = corner_min.astype(float)
    diag_v = corner_max.astype(float) - corner_min.astype(float)
    diag_len = np.linalg.norm(diag_v)
    if diag_len > 0:
        d_hat = diag_v / diag_len
        V = closeup_positions - diag_p0[None, :]
        # distance from point to line in 3D: || (p - p0) x d_hat ||
        dist = np.linalg.norm(np.cross(V, d_hat[None, :]), axis=1)
        slab_thickness = diag_thickness_frac * diag_len
        keep = dist <= slab_thickness

        if np.any(keep):
            slice_pos = _ensure_point_array(closeup_positions[keep])
            slice_phases = closeup_phases[keep]
            slice_colors = [color_map.get(p, _DEFAULT_GRAY_RGBA) for p in slice_phases]

            # draw the diagonal line for orientation
            ax.plot(
                [corner_min[0], corner_max[0]],
                [corner_min[1], corner_max[1]],
                [corner_min[2], corner_max[2]],
                color='k', linewidth=1.0, alpha=0.6
            )

            # overlay highlighted slice points (slightly bigger, fully opaque, light edge)
            ax.scatter(
                slice_pos[:, 0], slice_pos[:, 1], slice_pos[:, 2],
                c=slice_colors, s=28, depthshade=True, alpha=1.0,
                edgecolors="black", linewidths=0.6
            )

    ax.set_title(f"Close-up ({volume_fraction} volume, {num_points} points)")
    ax.set_xlim(corner_min[0], corner_max[0])
    ax.set_ylim(corner_min[1], corner_max[1])
    ax.set_zlim(corner_min[2], corner_max[2])

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)





def _render_local_galleries(
    global_cfg: Any,
    atoms: Sequence[Dict[str, Any]],
    atom_positions: np.ndarray,
    phases: Sequence[str],
    metadata: Dict[str, Any],
    output_dir: pathlib.Path,
    rng: np.random.Generator,
) -> None:
    base_phases = sorted({phase for phase in phases if not phase.startswith("intermediate_")})
    intermediate_phases = sorted({phase for phase in phases if phase.startswith("intermediate_")})
    positions_tree = cKDTree(atom_positions)

    _render_local_gallery(
        title="Base phases",
        filename=output_dir / "figure_local_base.png",
        selected_phases=base_phases,
        atoms=atoms,
        atom_positions=atom_positions,
        positions_tree=positions_tree,
        global_cfg=global_cfg,
        rng=rng,
    )
    _render_local_gallery(
        title="Intermediate phases",
        filename=output_dir / "figure_local_intermediate.png",
        selected_phases=intermediate_phases,
        atoms=atoms,
        atom_positions=atom_positions,
        positions_tree=positions_tree,
        global_cfg=global_cfg,
        rng=rng,
    )


def _render_local_gallery(
    title: str,
    filename: pathlib.Path,
    selected_phases: Sequence[str],
    atoms: Sequence[Dict[str, Any]],
    atom_positions: np.ndarray,
    positions_tree: cKDTree,
    global_cfg: Any,
    rng: np.random.Generator,
) -> None:
    if not selected_phases:
        return

    phase_to_atoms: Dict[str, List[Dict[str, Any]]] = {phase: [] for phase in selected_phases}
    for atom in atoms:
        phase = atom["phase_id"]
        if phase in phase_to_atoms:
            phase_to_atoms[phase].append(atom)

    phase_list = [phase for phase in selected_phases if phase_to_atoms[phase]]
    if not phase_list:
        return

    rows = 4
    cols = len(phase_list)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), subplot_kw={"projection": "3d"})
    axes = np.array(axes, dtype=object)
    if axes.ndim == 1:
        axes = axes[:, np.newaxis]

    for col, phase in enumerate(phase_list):
        phase_atoms = phase_to_atoms[phase]
        num_samples = min(rows, len(phase_atoms))
        sample_indices = rng.choice(len(phase_atoms), size=num_samples, replace=False)
        for row in range(rows):
            ax = axes[row][col]
            if row >= num_samples:
                ax.axis("off")
                continue
            center_atom = phase_atoms[int(sample_indices[row])]
            _plot_local_neighborhood(
                ax=ax,
                center_atom=center_atom,
                atoms=atoms,
                atom_positions=atom_positions,
                positions_tree=positions_tree,
                global_cfg=global_cfg,
                target_count=48,
            )
            ax.set_title(f"{phase}\ncenter #{center_atom.get('final_index', '')}")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def _plot_local_neighborhood(
    ax: Any,
    center_atom: Dict[str, Any],
    atoms: Sequence[Dict[str, Any]],
    atom_positions: np.ndarray,
    positions_tree: cKDTree,
    global_cfg: Any,
    target_count: int,
) -> None:
    radius = 2.0 * global_cfg.avg_nn_dist
    center_pos = center_atom["position"]

    idxs = positions_tree.query_ball_point(center_pos, r=radius)
    coords = _ensure_point_array(atom_positions[idxs]) - center_pos

    # If we have fewer than target_count points, fall back to nearest-neighbour query
    if len(coords) < target_count:
        _, nn_indices = positions_tree.query(center_pos, k=min(target_count, len(atom_positions)))
        coords = _ensure_point_array(atom_positions[np.atleast_1d(nn_indices)]) - center_pos

    if len(coords) > target_count:
        coords = coords[:target_count]

    if not center_atom["phase_id"].startswith("amorphous"):
        rotation = center_atom["orientation"]
        coords = (rotation.T @ coords.T).T

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        s=25,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.3,
    )

    if coords.shape[0] > 1:
        neighbor_tree = cKDTree(coords)
        drawn_pairs = set()
        k = min(4, coords.shape[0])
        for idx, point in enumerate(coords):
            _, neighbor_indices = neighbor_tree.query(point, k=k)
            if k == 1:
                continue
            if np.isscalar(neighbor_indices):
                neighbor_indices = [neighbor_indices]
            else:
                neighbor_indices = neighbor_indices.tolist()
            for neighbor_idx in neighbor_indices[1:4]:
                if neighbor_idx == idx:
                    continue
                pair = tuple(sorted((idx, neighbor_idx)))
                if pair in drawn_pairs:
                    continue
                drawn_pairs.add(pair)
                neighbor_point = coords[neighbor_idx]
                ax.plot(
                    [point[0], neighbor_point[0]],
                    [point[1], neighbor_point[1]],
                    [point[2], neighbor_point[2]],
                    color="gray",
                    linewidth=0.6,
                    alpha=0.6,
                )

    limit = radius
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)


def _build_phase_color_map(phases: Sequence[str]) -> Dict[str, Any]:
    unique_phases = sorted(set(map(str, phases)))
    if not unique_phases:
        return {}
    denom = max(1, len(unique_phases) - 1)
    return {phase: tuple(map(float, cm.tab20(i / denom))) for i, phase in enumerate(unique_phases)}


def _build_grain_color_map(grains: Sequence[Dict[str, Any]]) -> Dict[int, Any]:
    grain_ids = sorted(
        {
            int(grain["grain_id"])
            for grain in grains
            if grain.get("grain_id") is not None
        }
    )
    if not grain_ids:
        return {}
    cmap = cm.get_cmap("gist_ncar")
    denom = max(1, len(grain_ids) - 1)
    return {gid: tuple(map(float, cmap(i / denom))) for i, gid in enumerate(grain_ids)}


def _compute_boundary_indices(
    atom_positions: np.ndarray,
    grain_ids: np.ndarray,
    grains: Sequence[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    if atom_positions.size == 0 or grain_ids.size == 0:
        return np.zeros(len(atom_positions), dtype=bool), np.full(len(atom_positions), -1, dtype=int)

    grain_positions = np.asarray([grain["seed_position"] for grain in grains], dtype=float)
    num_grains = grain_positions.shape[0]
    if num_grains < 2:
        mask = grain_ids < 0
        return mask, np.full(len(atom_positions), -1, dtype=int)

    tree = cKDTree(grain_positions)
    _, neighbor_indices = tree.query(atom_positions, k=2)
    neighbor_indices = np.asarray(neighbor_indices, dtype=int)
    if neighbor_indices.ndim == 1:
        neighbor_indices = neighbor_indices[np.newaxis, :]

    second_nearest = neighbor_indices[:, -1]
    mask = (grain_ids < 0) | (grain_ids >= num_grains)
    valid = ~mask
    mask[valid] |= second_nearest[valid] != grain_ids[valid]
    return mask, second_nearest


def _view_from_vector(direction: np.ndarray) -> Tuple[float, float]:
    vec = np.asarray(direction, dtype=float)
    if vec.shape != (3,):
        raise ValueError("direction must be a 3-element vector")
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return 0.0, 0.0
    vec = vec / norm
    azim = float(np.degrees(np.arctan2(vec[1], vec[0])))
    xy_hypot = float(np.hypot(vec[0], vec[1]))
    elev = float(np.degrees(np.arctan2(vec[2], xy_hypot)))
    return elev, azim


def _ensure_point_array(points: Any) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim == 1:
        if arr.size == 0:
            return arr.reshape(0, 3)
        if arr.size % 3 != 0:
            raise ValueError("Point array must have multiples of three components")
        return arr.reshape(-1, 3)
    return arr


def _add_axes_border(ax: Any, linewidth: float = 1.0, color: str = "black") -> None:
    """Ensure a visible border around 3D axes."""
    try:
        ax.patch.set_edgecolor(color)
        ax.patch.set_linewidth(linewidth)
    except Exception:
        pass
    for spine in getattr(ax, "spines", {}).values():
        spine.set_linewidth(linewidth)
        spine.set_color(color)


def _set_cube_axes(ax: Any, box_size: float) -> None:
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_zlim(0, box_size)


def _diagonal_cut_mask(atom_positions: np.ndarray, box_size: float) -> np.ndarray:
    """Keep half of the volume by slicing with a plane perpendicular to the long diagonal."""
    if atom_positions.size == 0:
        return np.zeros(len(atom_positions), dtype=bool)
    center = np.full(3, box_size / 2.0)
    relative = atom_positions - center
    projection = relative @ _DIAGONAL_DIRECTION
    return projection <= 0.0


__all__ = ["generate_visualizations"]
