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
from scipy.spatial import KDTree


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

    # Phases overview
    ax1 = fig.add_subplot(231, projection="3d")
    for pos, phase in zip(sampled_positions, sampled_phases):
        ax1.scatter(*pos, color=color_map[phase], s=12)
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
