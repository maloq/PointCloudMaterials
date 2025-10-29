"""
Visualization utilities for synthetic atomistic datasets.

Provides routines for rendering the global multi-panel figure and local
neighborhood galleries with optional separation of intermediate phases.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.spatial import KDTree, ConvexHull, Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _compute_alpha_shape_mesh(points: np.ndarray, alpha: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute an alpha shape (concave hull) mesh from a point cloud.

    Args:
        points: Nx3 array of 3D points
        alpha: Alpha parameter controlling mesh tightness (smaller = tighter fit)

    Returns:
        vertices: Mx3 array of mesh vertices
        faces: Kx3 array of triangle face indices
    """
    if len(points) < 4:
        return points, np.array([])

    try:
        # Compute Delaunay triangulation
        tri = Delaunay(points)

        # Filter tetrahedra based on circumradius (alpha shape criterion)
        tetras = tri.simplices
        valid_faces = []

        for tetra in tetras:
            # Get the four vertices of the tetrahedron
            tetra_points = points[tetra]

            # Compute circumradius
            # For simplicity, we use a rough approximation
            distances = []
            for i in range(4):
                for j in range(i + 1, 4):
                    distances.append(np.linalg.norm(tetra_points[i] - tetra_points[j]))

            max_edge = max(distances)

            # If circumradius is small enough, include the boundary faces
            if max_edge < 1.0 / alpha:
                # Add the four faces of the tetrahedron
                faces = [
                    [tetra[0], tetra[1], tetra[2]],
                    [tetra[0], tetra[1], tetra[3]],
                    [tetra[0], tetra[2], tetra[3]],
                    [tetra[1], tetra[2], tetra[3]],
                ]
                valid_faces.extend(faces)

        if not valid_faces:
            # Fallback to convex hull
            return _compute_convex_hull_mesh(points)

        # Remove duplicate faces (interior faces appear twice)
        face_set = {}
        for face in valid_faces:
            sorted_face = tuple(sorted(face))
            face_set[sorted_face] = face_set.get(sorted_face, 0) + 1

        # Keep only boundary faces (appear once)
        boundary_faces = [face for face, count in face_set.items() if count == 1]

        if not boundary_faces:
            return _compute_convex_hull_mesh(points)

        return points, np.array([list(face) for face in boundary_faces])

    except Exception:
        # Fallback to convex hull if alpha shape fails
        return _compute_convex_hull_mesh(points)


def _compute_convex_hull_mesh(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a convex hull mesh from a point cloud.

    Args:
        points: Nx3 array of 3D points

    Returns:
        vertices: Mx3 array of mesh vertices
        faces: Kx3 array of triangle face indices
    """
    if len(points) < 4:
        return points, np.array([])

    try:
        hull = ConvexHull(points)
        return points, hull.simplices
    except Exception:
        return points, np.array([])


def _render_mesh_surface(ax: Any, vertices: np.ndarray, faces: np.ndarray,
                         color: Any = 'cyan', alpha: float = 0.3,
                         edgecolor: str = 'gray', linewidth: float = 0.5) -> None:
    """
    Render a triangular mesh surface on a 3D axis.

    Args:
        ax: Matplotlib 3D axis
        vertices: Nx3 array of mesh vertices
        faces: Kx3 array of triangle face indices
        color: Face color
        alpha: Transparency
        edgecolor: Edge color
        linewidth: Edge line width
    """
    if len(faces) == 0:
        return

    # Create the 3D polygon collection
    mesh = [[vertices[face[0]], vertices[face[1]], vertices[face[2]]] for face in faces]
    collection = Poly3DCollection(mesh, alpha=alpha, facecolor=color,
                                  edgecolor=edgecolor, linewidth=linewidth)
    ax.add_collection3d(collection)


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
    atoms_array = np.array([atom["position"] for atom in atoms])
    phases = [atom["phase_id"] for atom in atoms]

    _render_global_structure(global_cfg, grains, atoms, atoms_array, phases, metadata, output_dir / "figure_global.png", rng)
    _render_global_structure_mesh(global_cfg, grains, atoms, atoms_array, phases, metadata, output_dir / "figure_global_mesh.png", rng)
    _render_closeup_view(global_cfg, atoms, atoms_array, phases, output_dir / "figure_closeup.png", rng)
    _render_local_galleries(global_cfg, atoms, atoms_array, phases, metadata, output_dir, rng)


def _render_global_structure(
    global_cfg: Any,
    grains: Sequence[Dict[str, Any]],
    atoms: Sequence[Dict[str, Any]],
    atom_positions: np.ndarray,
    phases: Sequence[str],
    metadata: Dict[str, Any],
    output_path: pathlib.Path,
    rng: np.random.Generator,
) -> None:
    total_atoms = len(atoms)
    if total_atoms == 0:
        return
    sample_size = min(total_atoms, 3000)
    if sample_size < total_atoms:
        sample_indices = rng.choice(total_atoms, size=sample_size, replace=False)
    else:
        sample_indices = np.arange(total_atoms)

    sampled_atoms = [atoms[i] for i in sample_indices]
    sampled_positions = atom_positions[sample_indices]
    sampled_phases = [phases[i] for i in sample_indices]

    unique_phases = sorted(set(phases))
    color_map = {phase: cm.tab20(i / max(1, len(unique_phases) - 1)) for i, phase in enumerate(unique_phases)}

    fig = plt.figure(figsize=(16, 10))

    # Phases overview with shadows
    ax1 = fig.add_subplot(231, projection="3d")
    for pos, phase in zip(sampled_positions, sampled_phases):
        ax1.scatter(*pos, color=color_map[phase], s=12, depthshade=True)
    # Add shadow projections on bottom plane
    ax1.scatter(sampled_positions[:, 0], sampled_positions[:, 1],
                np.zeros(len(sampled_positions)), color='gray', s=2, alpha=0.2)
    ax1.set_title("Phases")
    _set_cube_axes(ax1, global_cfg.L)

    # Intermediate phases
    ax2 = fig.add_subplot(232, projection="3d")
    for atom in sampled_atoms:
        if atom["phase_id"].startswith("intermediate_"):
            ax2.scatter(*atom["position"], color=color_map[atom["phase_id"]], s=12)
    ax2.set_title("Intermediate phases")
    _set_cube_axes(ax2, global_cfg.L)

    # Grain boundaries
    ax3 = fig.add_subplot(233, projection="3d")
    boundary_atoms = [
        atom for atom in atoms if atom["phase_id"].startswith("intermediate_") or _is_boundary_atom(atom, grains, global_cfg.L)
    ]
    if boundary_atoms:
        boundary_sample_size = min(len(boundary_atoms), 4000)
        if boundary_sample_size == len(boundary_atoms):
            boundary_indices = range(len(boundary_atoms))
        else:
            boundary_indices = rng.choice(len(boundary_atoms), size=boundary_sample_size, replace=False)
        for idx in boundary_indices:
            atom = boundary_atoms[int(idx)]
            ax3.scatter(*atom["position"], color="gray", s=12)
    ax3.set_title("Grain boundaries")
    _set_cube_axes(ax3, global_cfg.L)

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

    # Perturbation overview
    ax5 = fig.add_subplot(235, projection="3d")
    perturb = metadata.get("perturbations", {})
    for bubble in perturb.get("rotation_bubbles", []):
        center = np.array(bubble["center"])
        ax5.scatter(*center, color="orange", s=60, marker="^")
    for bubble in perturb.get("density_bubbles", []):
        center = np.array(bubble["center"])
        color = "blue" if bubble["alpha"] > 0 else "red"
        ax5.scatter(*center, color=color, s=60, marker="o")
    for event in perturb.get("dropouts", {}).get("events", []):
        idx = event.get("removed_atom_final_index")
        if idx is None or idx >= len(atom_positions):
            continue
        position = atom_positions[idx]
        ax5.scatter(*position, color="black", marker="x", s=50)
    ax5.set_title("Perturbations")
    _set_cube_axes(ax5, global_cfg.L)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _render_global_structure_mesh(
    global_cfg: Any,
    grains: Sequence[Dict[str, Any]],
    atoms: Sequence[Dict[str, Any]],
    atom_positions: np.ndarray,
    phases: Sequence[str],
    metadata: Dict[str, Any],
    output_path: pathlib.Path,
    rng: np.random.Generator,
) -> None:
    """Render global structure with mesh-based visualizations for boundaries and phases."""
    total_atoms = len(atoms)
    if total_atoms == 0:
        return

    unique_phases = sorted(set(phases))
    color_map = {phase: cm.tab20(i / max(1, len(unique_phases) - 1)) for i, phase in enumerate(unique_phases)}

    fig = plt.figure(figsize=(16, 10))

    # Phases as mesh
    ax1 = fig.add_subplot(231, projection="3d")
    phase_to_points = {}
    for i, phase in enumerate(phases):
        if phase not in phase_to_points:
            phase_to_points[phase] = []
        phase_to_points[phase].append(atom_positions[i])

    for phase, points in phase_to_points.items():
        if len(points) >= 4:
            points_array = np.array(points)
            # Sample if too many points for mesh computation
            if len(points_array) > 500:
                sample_indices = rng.choice(len(points_array), size=500, replace=False)
                points_array = points_array[sample_indices]

            vertices, faces = _compute_convex_hull_mesh(points_array)
            if len(faces) > 0:
                _render_mesh_surface(ax1, vertices, faces, color=color_map[phase],
                                    alpha=0.3, edgecolor='darkgray', linewidth=0.3)
    ax1.set_title("Phases (Mesh)")
    _set_cube_axes(ax1, global_cfg.L)

    # Intermediate phases as mesh
    ax2 = fig.add_subplot(232, projection="3d")
    intermediate_points = []
    for atom in atoms:
        if atom["phase_id"].startswith("intermediate_"):
            intermediate_points.append(atom["position"])

    if len(intermediate_points) >= 4:
        intermediate_array = np.array(intermediate_points)
        if len(intermediate_array) > 800:
            sample_indices = rng.choice(len(intermediate_array), size=800, replace=False)
            intermediate_array = intermediate_array[sample_indices]

        vertices, faces = _compute_convex_hull_mesh(intermediate_array)
        if len(faces) > 0:
            _render_mesh_surface(ax2, vertices, faces, color='purple',
                                alpha=0.4, edgecolor='darkviolet', linewidth=0.5)
    ax2.set_title("Intermediate phases (Mesh)")
    _set_cube_axes(ax2, global_cfg.L)

    # Grain boundaries as mesh
    ax3 = fig.add_subplot(233, projection="3d")
    boundary_atoms = [
        atom for atom in atoms if atom["phase_id"].startswith("intermediate_") or _is_boundary_atom(atom, grains, global_cfg.L)
    ]
    if len(boundary_atoms) >= 4:
        boundary_positions = np.array([atom["position"] for atom in boundary_atoms])
        boundary_sample_size = min(len(boundary_positions), 1000)
        if boundary_sample_size < len(boundary_positions):
            boundary_indices = rng.choice(len(boundary_positions), size=boundary_sample_size, replace=False)
            boundary_positions = boundary_positions[boundary_indices]

        vertices, faces = _compute_convex_hull_mesh(boundary_positions)
        if len(faces) > 0:
            _render_mesh_surface(ax3, vertices, faces, color='gray',
                                alpha=0.5, edgecolor='black', linewidth=0.4)
    ax3.set_title("Grain boundaries (Mesh)")
    _set_cube_axes(ax3, global_cfg.L)

    # Phase boundaries as mesh (boundaries between different phases)
    ax4 = fig.add_subplot(234, projection="3d")
    # Build KDTree for neighbor search
    tree = KDTree(atom_positions)
    phase_boundary_points = []
    sample_size = min(total_atoms, 5000)
    sample_indices = rng.choice(total_atoms, size=sample_size, replace=False)

    for idx in sample_indices:
        atom = atoms[idx]
        phase = atom["phase_id"]
        pos = atom["position"]

        # Query nearest neighbors
        neighbors_idx = tree.query_ball_point(pos, r=global_cfg.avg_nn_dist * 1.5)
        neighbor_phases = [phases[n] for n in neighbors_idx if n != idx]

        # If any neighbor has a different phase, this is a phase boundary point
        if any(np != phase for np in neighbor_phases):
            phase_boundary_points.append(pos)

    if len(phase_boundary_points) >= 4:
        phase_boundary_array = np.array(phase_boundary_points)
        if len(phase_boundary_array) > 800:
            sample_indices = rng.choice(len(phase_boundary_array), size=800, replace=False)
            phase_boundary_array = phase_boundary_array[sample_indices]

        vertices, faces = _compute_convex_hull_mesh(phase_boundary_array)
        if len(faces) > 0:
            _render_mesh_surface(ax4, vertices, faces, color='cyan',
                                alpha=0.4, edgecolor='teal', linewidth=0.4)
    ax4.set_title("Phase boundaries (Mesh)")
    _set_cube_axes(ax4, global_cfg.L)

    # Grain orientations (same as before)
    ax5 = fig.add_subplot(235, projection="3d")
    for grain in grains:
        seed = grain["seed_position"]
        rotation = grain["base_rotation"]
        axes = rotation @ np.eye(3) * global_cfg.avg_nn_dist
        ax5.quiver(
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
    ax5.set_title("Grain orientations")
    _set_cube_axes(ax5, global_cfg.L)

    # Perturbation overview (same as before)
    ax6 = fig.add_subplot(236, projection="3d")
    perturb = metadata.get("perturbations", {})
    for bubble in perturb.get("rotation_bubbles", []):
        center = np.array(bubble["center"])
        ax6.scatter(*center, color="orange", s=60, marker="^")
    for bubble in perturb.get("density_bubbles", []):
        center = np.array(bubble["center"])
        color = "blue" if bubble["alpha"] > 0 else "red"
        ax6.scatter(*center, color=color, s=60, marker="o")
    for event in perturb.get("dropouts", {}).get("events", []):
        idx = event.get("removed_atom_final_index")
        if idx is None or idx >= len(atom_positions):
            continue
        position = atom_positions[idx]
        ax6.scatter(*position, color="black", marker="x", s=50)
    ax6.set_title("Perturbations")
    _set_cube_axes(ax6, global_cfg.L)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _render_closeup_view(
    global_cfg: Any,
    atoms: Sequence[Dict[str, Any]],
    atom_positions: np.ndarray,
    phases: Sequence[str],
    output_path: pathlib.Path,
    rng: np.random.Generator,
) -> None:
    """Render a close-up view showing 1/8 of the volume with full point density."""
    total_atoms = len(atoms)
    if total_atoms == 0:
        return

    # Select 1/8 of the volume (corner region)
    L = global_cfg.L
    corner_min = np.array([0, 0, 0])
    corner_max = np.array([L / 2, L / 2, L / 2])

    # Filter atoms in this region
    closeup_indices = []
    for i, pos in enumerate(atom_positions):
        if np.all(pos >= corner_min) and np.all(pos <= corner_max):
            closeup_indices.append(i)

    if len(closeup_indices) == 0:
        return

    closeup_atoms = [atoms[i] for i in closeup_indices]
    closeup_positions = atom_positions[closeup_indices]
    closeup_phases = [phases[i] for i in closeup_indices]

    unique_phases = sorted(set(phases))
    color_map = {phase: cm.tab20(i / max(1, len(unique_phases) - 1)) for i, phase in enumerate(unique_phases)}

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")

    # Render all points with shadows
    for pos, phase in zip(closeup_positions, closeup_phases):
        ax.scatter(*pos, color=color_map[phase], s=15, depthshade=True, alpha=0.8)

    # Add shadow projections on bottom plane
    ax.scatter(closeup_positions[:, 0], closeup_positions[:, 1],
               np.full(len(closeup_positions), corner_min[2]),
               color='gray', s=3, alpha=0.15)

    # Add shadow projections on back wall (Y-Z plane)
    ax.scatter(np.full(len(closeup_positions), corner_min[0]),
               closeup_positions[:, 1], closeup_positions[:, 2],
               color='gray', s=3, alpha=0.15)

    # Add shadow projections on side wall (X-Z plane)
    ax.scatter(closeup_positions[:, 0],
               np.full(len(closeup_positions), corner_min[1]),
               closeup_positions[:, 2],
               color='gray', s=3, alpha=0.15)

    ax.set_title(f"Close-up view (1/8 volume, {len(closeup_indices)} atoms)")
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
    positions_tree = KDTree(atom_positions)

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
    positions_tree: KDTree,
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
    positions_tree: KDTree,
    global_cfg: Any,
    target_count: int,
) -> None:
    radius = 2.0 * global_cfg.avg_nn_dist
    center_pos = center_atom["position"]

    idxs = positions_tree.query_ball_point(center_pos, r=radius)
    coords = atom_positions[idxs] - center_pos

    # If we have fewer than target_count points, fall back to nearest-neighbour query
    if len(coords) < target_count:
        _, nn_indices = positions_tree.query(center_pos, k=min(target_count, len(atom_positions)))
        coords = atom_positions[nn_indices] - center_pos

    if len(coords) > target_count:
        coords = coords[:target_count]

    if not center_atom["phase_id"].startswith("amorphous"):
        rotation = center_atom["orientation"]
        coords = (rotation.T @ coords.T).T

    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=25, alpha=0.9)

    if coords.shape[0] > 1:
        neighbor_tree = KDTree(coords)
        drawn_pairs = set()
        k = min(4, coords.shape[0])
        for idx, point in enumerate(coords):
            distances, neighbor_indices = neighbor_tree.query(point, k=k)
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


def _set_cube_axes(ax: Any, box_size: float) -> None:
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_zlim(0, box_size)


def _is_boundary_atom(atom: Dict[str, Any], grains: Sequence[Dict[str, Any]], box_size: float) -> bool:
    if atom["grain_id"] is None:
        return True
    grain_positions = np.array([grain["seed_position"] for grain in grains])
    dists = np.linalg.norm(grain_positions - atom["position"], axis=1)
    if len(dists) < 2:
        return False
    nearest_indices = np.argsort(dists)[:2]
    return nearest_indices[1] != atom["grain_id"]


__all__ = ["generate_visualizations"]
