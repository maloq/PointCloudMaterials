"""
Prediction and visualization pipeline for SPD models with synthetic data.

This script:
1. Loads a trained SPD model from checkpoint
2. Loads the synthetic training data from the model's config
3. Extracts latent representations with ground truth labels
4. Performs clustering analysis (KMeans and HDBSCAN)
5. Creates comprehensive visualizations comparing:
   - Ground truth phases (from metadata)
   - Ground truth grains (from metadata)
   - Ground truth grain boundaries (from metadata)
   - Predicted clusters (KMeans)
   - Predicted clusters (HDBSCAN)

All ground truth visualization uses the actual synthetic atomistic data with
the same visual style as src/data_utils/synthetic/visualization.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hdbscan
import numpy as np
import torch
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.training_methods.spd.eval_spd import load_spd_model
from src.data_utils.data_load import SyntheticPointCloudDataset
from src.data_utils.prepare_data import get_regular_samples, get_random_samples


def load_synthetic_environment_data(env_dir: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load synthetic environment data and metadata."""
    env_path = Path(env_dir)
    atoms_path = env_path / "atoms.npy"
    metadata_path = env_path / "metadata.json"

    if not atoms_path.exists():
        raise FileNotFoundError(f"atoms.npy not found in {env_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {env_dir}")

    atoms = np.load(atoms_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return atoms, metadata


def extract_ground_truth_labels(
    atoms: np.ndarray,
    metadata: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract ground truth phase and grain labels for all atoms."""
    num_atoms = len(atoms)
    phase_labels = np.full(num_atoms, "unknown", dtype=object)
    grain_labels = np.full(num_atoms, -1, dtype=int)

    # Assign grain and phase labels
    for grain in metadata.get("grains", []):
        grain_id = int(grain["grain_id"])
        phase_id = grain["base_phase_id"]
        atom_indices = np.array(grain.get("atom_indices", []), dtype=int)

        valid_mask = (atom_indices >= 0) & (atom_indices < num_atoms)
        valid_indices = atom_indices[valid_mask]

        phase_labels[valid_indices] = phase_id
        grain_labels[valid_indices] = grain_id

    # Assign intermediate region labels (override grain labels for boundaries)
    for region in metadata.get("intermediate_regions", []):
        phase_id = region.get("intermediate_phase_id", "intermediate")
        atom_indices = np.array(region.get("atom_indices", []), dtype=int)

        valid_mask = (atom_indices >= 0) & (atom_indices < num_atoms)
        valid_indices = atom_indices[valid_mask]

        phase_labels[valid_indices] = phase_id
        grain_labels[valid_indices] = -1  # Mark as boundary

    return phase_labels, grain_labels


def create_synthetic_dataset_with_coords(
    env_dirs: List[str],
    cfg: Any,
    max_samples: int = None,
) -> Tuple[SyntheticPointCloudDataset, np.ndarray]:
    """Create synthetic dataset and extract sample center coordinates."""
    all_sample_coords = []

    for env_dir in env_dirs:
        atoms, _ = load_synthetic_environment_data(env_dir)

        sample_type = cfg.data.get("sample_type", "regular")

        if sample_type == "regular":
            max_samples_per_env = cfg.data.get("n_samples", 1000) if cfg.data.get("n_samples", 0) > 0 else int(2e9)
            samples = get_regular_samples(
                atoms,
                size=cfg.data.radius,
                overlap_fraction=cfg.data.get("overlap_fraction", 0.0),
                return_coords=True,
                n_points=cfg.data.num_points,
                max_samples=max_samples_per_env,
                drop_edge_samples=True,
            )
        elif sample_type == "random":
            if cfg.data.get("n_samples", 0) <= 0:
                raise ValueError("n_samples must be > 0 for random sampling")
            samples = get_random_samples(
                atoms,
                n_samples=cfg.data.n_samples,
                size=cfg.data.radius,
                n_points=cfg.data.num_points,
                return_coords=True,
            )
        else:
            raise ValueError(f"Invalid sample type: {sample_type}")

        # Extract centers
        centers = [c for _, c in samples]
        all_sample_coords.extend(centers)

        if max_samples is not None and len(all_sample_coords) >= max_samples:
            all_sample_coords = all_sample_coords[:max_samples]
            break

    sample_coords = np.array(all_sample_coords, dtype=np.float32)

    # Create the dataset
    dataset = SyntheticPointCloudDataset(
        env_dirs=env_dirs,
        radius=cfg.data.radius,
        sample_type=cfg.data.get("sample_type", "regular"),
        overlap_fraction=cfg.data.get("overlap_fraction", 0.0),
        n_samples=cfg.data.get("n_samples", 1000),
        num_points=cfg.data.num_points,
        pre_normalize=True,
        normalize=True,
        max_samples=max_samples,
    )

    # Verify sizes match
    if len(dataset) != len(sample_coords):
        print(f"Warning: Dataset size ({len(dataset)}) doesn't match coords ({len(sample_coords)})")
        sample_coords = sample_coords[:len(dataset)]

    return dataset, sample_coords


def extract_latents_with_ground_truth(
    model: torch.nn.Module,
    dataset: SyntheticPointCloudDataset,
    sample_coords: np.ndarray,
    device: str = "cpu",
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract latent representations with ground truth labels."""
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    latents_list = []
    phase_list = []
    grain_list = []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Extracting latents"):
            # SyntheticPointCloudDataset returns: (points, phase, grain, orientation, quaternion)
            points, phase, grain, _, _ = batch

            points = points.to(device)
            inv_z, _, _, _ = model(points)

            latents_list.append(inv_z.detach().cpu().numpy())
            phase_list.append(phase.detach().cpu().numpy())
            grain_list.append(grain.detach().cpu().numpy())

    latents = np.concatenate(latents_list, axis=0)
    phase_labels = np.concatenate(phase_list, axis=0)
    grain_labels = np.concatenate(grain_list, axis=0)

    return latents, sample_coords, phase_labels, grain_labels


def predict_and_cache(
    checkpoint_path: str,
    cuda_device: int = 0,
    cache_dir: str = "output/spd_analysis/predictions_cache",
    force_recompute: bool = False,
    max_samples: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any, Any, np.ndarray, Dict]:
    """Load model, extract predictions from synthetic data, and cache results."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    ckpt_name = Path(checkpoint_path).stem
    cache_file = cache_path / f"{ckpt_name}_synth_predictions.npz"

    # Load model and config
    print(f"Loading model from {checkpoint_path}")
    model, cfg, device = load_spd_model(checkpoint_path, cuda_device=cuda_device)

    # Get synthetic data paths from config
    # Handle different config formats
    env_dirs = None

    if hasattr(cfg.data, 'env_dirs'):
        env_dirs = cfg.data.env_dirs
    elif hasattr(cfg.data, 'data_path'):
        env_dirs = cfg.data.data_path
    elif hasattr(cfg.data, 'synthetic') and hasattr(cfg.data.synthetic, 'data_dir'):
        env_dirs = cfg.data.synthetic.data_dir

    if env_dirs is None:
        raise ValueError(
            "Config does not contain synthetic data path. "
            "Expected one of: data.env_dirs, data.data_path, or data.synthetic.data_dir"
        )

    if isinstance(env_dirs, str):
        env_dirs = [env_dirs]

    print(f"Loading synthetic data from: {env_dirs}")

    # Load atoms and metadata for visualization
    atoms, metadata = load_synthetic_environment_data(env_dirs[0])
    print(f"Loaded {len(atoms)} atoms from {env_dirs[0]}")

    # Try to load from cache
    if cache_file.exists() and not force_recompute:
        print(f"Loading predictions from cache: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        latents = data["latents"]
        sample_coords = data["sample_coords"]
        phase_labels = data["phase_labels"]
        grain_labels = data["grain_labels"]

        return latents, sample_coords, phase_labels, grain_labels, model, cfg, atoms, metadata

    # Compute predictions
    print(f"Computing predictions...")

    dataset, sample_coords = create_synthetic_dataset_with_coords(env_dirs, cfg, max_samples)
    print(f"Created dataset with {len(dataset)} samples")

    batch_size = getattr(cfg, "batch_size", 32)
    latents, coords, phase_labels, grain_labels = extract_latents_with_ground_truth(
        model, dataset, sample_coords, device, batch_size
    )

    # Save to cache
    print(f"Saving predictions to cache: {cache_file}")
    np.savez_compressed(
        cache_file,
        latents=latents,
        sample_coords=coords,
        phase_labels=phase_labels,
        grain_labels=grain_labels,
    )

    return latents, coords, phase_labels, grain_labels, model, cfg, atoms, metadata


def perform_clustering(
    latents: np.ndarray,
    n_phases: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform clustering using KMeans and HDBSCAN."""
    print(f"Clustering {len(latents)} samples...")

    # KMeans clustering
    print(f"  KMeans with k={n_phases}...")
    kmeans = KMeans(n_clusters=n_phases, n_init=10, random_state=0)
    kmeans_labels = kmeans.fit_predict(latents)

    # HDBSCAN clustering
    print("  HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(10, len(latents) // 100),
        min_samples=5,
        cluster_selection_epsilon=0.0,
    )
    hdbscan_labels = clusterer.fit_predict(latents)

    n_hdbscan = len(np.unique(hdbscan_labels[hdbscan_labels >= 0]))
    print(f"  KMeans found {n_phases} clusters (by design)")
    print(f"  HDBSCAN found {n_hdbscan} clusters ({np.sum(hdbscan_labels == -1)} noise points)")

    return kmeans_labels, hdbscan_labels


def create_visualization(
    atoms: np.ndarray,
    metadata: Dict[str, Any],
    sample_coords: np.ndarray,
    phase_labels_sample: np.ndarray,
    grain_labels_sample: np.ndarray,
    kmeans_labels: np.ndarray,
    hdbscan_labels: np.ndarray,
    box_size: float,
    output_path: Path,
    max_points_per_panel: int = 6000,
) -> None:
    """Create 7-panel visualization comparing ground truth and predictions."""
    # Extract ground truth from atoms
    atom_phase_labels, atom_grain_labels = extract_ground_truth_labels(atoms, metadata)

    # Sample atoms for ground truth visualization
    n_atoms = len(atoms)
    if n_atoms > max_points_per_panel:
        atom_sample_indices = np.random.choice(n_atoms, max_points_per_panel, replace=False)
    else:
        atom_sample_indices = np.arange(n_atoms)

    atoms_sample = atoms[atom_sample_indices]
    atom_phase_sample = atom_phase_labels[atom_sample_indices]
    atom_grain_sample = atom_grain_labels[atom_sample_indices]

    # Sample predictions
    n_samples = len(kmeans_labels)
    if n_samples > max_points_per_panel:
        pred_sample_indices = np.random.choice(n_samples, max_points_per_panel, replace=False)
    else:
        pred_sample_indices = np.arange(n_samples)

    coords_sample = sample_coords[pred_sample_indices]
    kmeans_sample = kmeans_labels[pred_sample_indices]
    hdbscan_sample = hdbscan_labels[pred_sample_indices]
    phase_sample = phase_labels_sample[pred_sample_indices]
    grain_sample = grain_labels_sample[pred_sample_indices]

    fig = plt.figure(figsize=(28, 8))
    border_width = 72.0 / fig.dpi
    fig.patch.set_facecolor("white")
    fig.patch.set_edgecolor("black")
    fig.patch.set_linewidth(border_width)

    # Panel 1: Ground truth phases
    ax1 = fig.add_subplot(1, 7, 1, projection="3d")
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
    ax1.set_title("Ground Truth: Phases")
    _set_cube_axes(ax1, box_size)
    _add_axes_border(ax1, linewidth=border_width)

    # Panel 2: Ground truth grains
    ax2 = fig.add_subplot(1, 7, 2, projection="3d")
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

    # Show boundaries in different color
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
    ax2.set_title("Ground Truth: Grains")
    _set_cube_axes(ax2, box_size)
    _add_axes_border(ax2, linewidth=border_width)

    # Panel 3: Ground truth boundaries only
    ax3 = fig.add_subplot(1, 7, 3, projection="3d")
    if np.any(boundary_mask):
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
        # If no boundaries, show intermediate phases
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
    ax3.set_title("Ground Truth: Boundaries")
    _set_cube_axes(ax3, box_size)
    _add_axes_border(ax3, linewidth=border_width)

    # Panel 4: KMeans predictions
    ax4 = fig.add_subplot(1, 7, 4, projection="3d")
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
    ax4.set_title(f"KMeans Clusters (k={len(unique_kmeans)})")
    _set_cube_axes(ax4, box_size)
    _add_axes_border(ax4, linewidth=border_width)

    # Panel 5: HDBSCAN predictions
    ax5 = fig.add_subplot(1, 7, 5, projection="3d")
    unique_hdbscan = np.unique(hdbscan_sample)
    hdbscan_colors = _build_color_map(unique_hdbscan, "tab20")

    for cluster in unique_hdbscan:
        mask = hdbscan_sample == cluster
        if not np.any(mask):
            continue
        points = coords_sample[mask]
        color = "lightgray" if cluster == -1 else hdbscan_colors[cluster]
        ax5.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            color=color,
            s=12,
            depthshade=True,
            edgecolors="black",
            linewidths=0.2,
            alpha=0.7 if cluster == -1 else 1.0,
        )
    n_clusters = len(unique_hdbscan[unique_hdbscan >= 0])
    ax5.set_title(f"HDBSCAN Clusters ({n_clusters} + noise)")
    _set_cube_axes(ax5, box_size)
    _add_axes_border(ax5, linewidth=border_width)

    # Panel 6: KMeans-Phase Agreement
    # Map each KMeans cluster to most common ground truth phase
    ax6 = fig.add_subplot(1, 7, 6, projection="3d")
    cluster_to_phase = {}
    for cluster in np.unique(kmeans_sample):
        mask = kmeans_sample == cluster
        if np.any(mask):
            # Find most common phase in this cluster
            phases_in_cluster = phase_sample[mask]
            unique, counts = np.unique(phases_in_cluster, return_counts=True)
            cluster_to_phase[cluster] = unique[np.argmax(counts)]

    # Color by correctness: green if cluster's dominant phase matches sample's true phase
    correct_mask = np.array([
        cluster_to_phase.get(kmeans_sample[i], None) == phase_sample[i]
        for i in range(len(kmeans_sample))
    ])

    # Plot correct predictions in green, incorrect in red
    if np.any(correct_mask):
        correct_points = coords_sample[correct_mask]
        ax6.scatter(
            correct_points[:, 0], correct_points[:, 1], correct_points[:, 2],
            color="green",
            s=12,
            depthshade=True,
            edgecolors="black",
            linewidths=0.2,
            alpha=0.8,
            label="Correct",
        )

    if np.any(~correct_mask):
        incorrect_points = coords_sample[~correct_mask]
        ax6.scatter(
            incorrect_points[:, 0], incorrect_points[:, 1], incorrect_points[:, 2],
            color="red",
            s=12,
            depthshade=True,
            edgecolors="black",
            linewidths=0.2,
            alpha=0.8,
            label="Incorrect",
        )

    accuracy = np.sum(correct_mask) / len(correct_mask) * 100
    ax6.set_title(f"KMeans-Phase Agreement\n({accuracy:.1f}% accuracy)")
    ax6.legend(loc="upper right", fontsize=8)
    _set_cube_axes(ax6, box_size)
    _add_axes_border(ax6, linewidth=border_width)

    # Panel 7: Boundary Detection
    # Color by whether sample is near a grain boundary (grain_label == -1 or multiple grains nearby)
    ax7 = fig.add_subplot(1, 7, 7, projection="3d")

    # Ground truth: samples with grain_label == -1 are boundaries
    is_boundary_gt = grain_sample == -1

    # Predicted boundaries: use HDBSCAN noise points as boundary candidates
    is_boundary_pred = hdbscan_sample == -1

    # True positives: correctly identified boundaries
    true_positive = is_boundary_gt & is_boundary_pred
    # True negatives: correctly identified non-boundaries
    true_negative = (~is_boundary_gt) & (~is_boundary_pred)
    # False positives: incorrectly marked as boundary
    false_positive = (~is_boundary_gt) & is_boundary_pred
    # False negatives: missed boundaries
    false_negative = is_boundary_gt & (~is_boundary_pred)

    if np.any(true_positive):
        ax7.scatter(
            coords_sample[true_positive, 0],
            coords_sample[true_positive, 1],
            coords_sample[true_positive, 2],
            color="green", s=14, depthshade=True,
            edgecolors="black", linewidths=0.25,
            alpha=0.8, label="True Boundary",
        )

    if np.any(false_positive):
        ax7.scatter(
            coords_sample[false_positive, 0],
            coords_sample[false_positive, 1],
            coords_sample[false_positive, 2],
            color="orange", s=10, depthshade=True,
            edgecolors="black", linewidths=0.2,
            alpha=0.6, label="False Boundary",
        )

    if np.any(false_negative):
        ax7.scatter(
            coords_sample[false_negative, 0],
            coords_sample[false_negative, 1],
            coords_sample[false_negative, 2],
            color="red", s=14, depthshade=True,
            edgecolors="black", linewidths=0.25,
            alpha=0.8, label="Missed Boundary",
        )

    if np.any(true_negative):
        # Only show a subset of true negatives to avoid clutter
        tn_indices = np.where(true_negative)[0]
        if len(tn_indices) > 1000:
            tn_indices = np.random.choice(tn_indices, 1000, replace=False)
        ax7.scatter(
            coords_sample[tn_indices, 0],
            coords_sample[tn_indices, 1],
            coords_sample[tn_indices, 2],
            color="lightgray", s=8, depthshade=True,
            edgecolors="none", alpha=0.3,
            label="Bulk (subset)",
        )

    boundary_recall = np.sum(true_positive) / np.sum(is_boundary_gt) * 100 if np.sum(is_boundary_gt) > 0 else 0
    ax7.set_title(f"Boundary Detection\n({boundary_recall:.1f}% recall)")
    ax7.legend(loc="upper right", fontsize=8)
    _set_cube_axes(ax7, box_size)
    _add_axes_border(ax7, linewidth=border_width)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to {output_path}")


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


def generate_reference_point_cloud(
    structure_def: Dict[str, Any],
    target_count: int = 64,
    box_size: float = 10.0,
) -> np.ndarray:
    """
    Generate a small point cloud sample from a reference structure definition.

    For crystalline structures, generates atoms in a small box, then centers and normalizes.
    For amorphous structures, generates random points and normalizes.

    The output is centered (mean at origin) and normalized by box_size to match
    the preprocessing in PointCloudDataset/SyntheticPointCloudDataset.

    Args:
        structure_def: Structure definition from reference_structures.npy
        target_count: Target number of atoms to generate
        box_size: Size of the region to fill with atoms (used for normalization)

    Returns:
        Point cloud array of shape (N, 3), centered and normalized
    """
    phase_type = structure_def.get("phase_type")

    if phase_type in ["crystal_fcc", "crystal_bcc", "amorphous_repeat"]:
        # For structured phases, tile the lattice/motif
        if phase_type == "amorphous_repeat":
            cell_vectors = np.array(structure_def["tile_vectors"], dtype=float)
            motif = np.array(structure_def["motif"], dtype=float)
            motif_fractional = False
        else:
            cell_vectors = np.array(structure_def["lattice_vectors"], dtype=float)
            motif = np.array(structure_def["motif"], dtype=float)
            motif_fractional = True

        # Convert motif to Cartesian coordinates
        if motif_fractional:
            motif_coords = motif @ cell_vectors
        else:
            motif_coords = motif

        # Determine how many cells needed
        cell_size = np.linalg.norm(cell_vectors[0])
        n_cells_per_dim = max(1, int(np.ceil(box_size / cell_size)))

        positions = []
        for i in range(-n_cells_per_dim, n_cells_per_dim + 1):
            for j in range(-n_cells_per_dim, n_cells_per_dim + 1):
                for k in range(-n_cells_per_dim, n_cells_per_dim + 1):
                    lattice_origin = i * cell_vectors[0] + j * cell_vectors[1] + k * cell_vectors[2]
                    for motif_offset in motif_coords:
                        pos = lattice_origin + motif_offset
                        # Keep atoms within box
                        if np.all(np.abs(pos) <= box_size / 2):
                            positions.append(pos)

        positions = np.array(positions, dtype=float) if positions else np.zeros((0, 3), dtype=float)

        # Center and subsample if needed
        if len(positions) > 0:
            positions = positions - positions.mean(axis=0)
            if len(positions) > target_count:
                # Keep central atoms
                dists = np.linalg.norm(positions, axis=1)
                indices = np.argsort(dists)[:target_count]
                positions = positions[indices]

        # Normalize by box_size to match PointCloudDataset behavior
        if len(positions) > 0:
            positions = positions / box_size

        return positions

    elif phase_type in ["amorphous_random", "amorphous_mixed"]:
        # For amorphous, generate random points with minimum separation
        min_pair_dist = structure_def.get("min_pair_dist", 1.0)
        positions = []
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility

        max_attempts = target_count * 100
        attempts = 0

        while len(positions) < target_count and attempts < max_attempts:
            candidate = rng.uniform(-box_size / 2, box_size / 2, size=3)
            attempts += 1

            # Check minimum distance
            if len(positions) == 0:
                positions.append(candidate)
                continue

            dists = np.linalg.norm(np.array(positions) - candidate, axis=1)
            if np.min(dists) >= min_pair_dist:
                positions.append(candidate)

        positions_array = np.array(positions, dtype=float) if positions else np.zeros((0, 3), dtype=float)

        # Normalize by box_size to match PointCloudDataset behavior
        if len(positions_array) > 0:
            positions_array = positions_array / box_size

        return positions_array

    else:
        # Unknown or intermediate phase type
        return np.zeros((0, 3), dtype=float)


def visualize_reference_structures(
    checkpoint_path: str,
    reference_structures_path: str,
    output_path: Path,
    cuda_device: int = 0,
    target_atoms: int = 64,
    box_size: float = 10.0,
) -> None:
    """
    Create visualization of reference structures, their reconstructions, and latent space.

    Args:
        checkpoint_path: Path to trained SPD model checkpoint
        reference_structures_path: Path to reference_structures.npy file
        output_path: Path to save visualization
        cuda_device: CUDA device ID
        target_atoms: Number of atoms to generate for each structure
        box_size: Size of region to generate atoms in
    """
    from sklearn.decomposition import PCA

    print(f"Loading checkpoint from {checkpoint_path}")
    model, cfg, device = load_spd_model(checkpoint_path, cuda_device=cuda_device)
    model.eval()

    print(f"Loading reference structures from {reference_structures_path}")
    ref_structures = np.load(reference_structures_path, allow_pickle=True).item()

    # Filter out intermediate phases (only visualize base phases)
    base_structures = {
        name: struct for name, struct in ref_structures.items()
        if not name.startswith("intermediate_")
    }

    print(f"Found {len(base_structures)} base phase structures")

    # Generate point clouds and pass through model
    structure_names = sorted(base_structures.keys())
    point_clouds = []
    latents_list = []
    reconstructions_list = []

    # Use the device from load_spd_model
    with torch.no_grad():
        for name in structure_names:
            struct_def = base_structures[name]
            print(f"  Processing {name}...")

            # Generate point cloud
            pc = generate_reference_point_cloud(struct_def, target_atoms, box_size)

            if len(pc) == 0:
                print(f"    Warning: Could not generate point cloud for {name}, skipping")
                continue

            point_clouds.append(pc)

            # Convert to tensor and add batch dimension
            pc_tensor = torch.from_numpy(pc).float().unsqueeze(0).to(device)

            # Get latent and reconstruction
            # SPD model forward returns: (inv_z, recon, cano, rot)
            try:
                inv_z, recon, cano, rot = model(pc_tensor)

                latents_list.append(inv_z.cpu().numpy()[0])
                reconstructions_list.append(recon.cpu().numpy()[0])
            except Exception as e:
                print(f"    Warning: Model inference failed for {name}: {e}")
                continue

    if len(point_clouds) == 0:
        print("No valid structures to visualize")
        return

    print(f"Successfully processed {len(point_clouds)} structures")

    # Create visualization
    n_structures = len(point_clouds)
    fig = plt.figure(figsize=(4 * n_structures, 12))

    border_width = 72.0 / fig.dpi
    fig.patch.set_facecolor("white")
    fig.patch.set_edgecolor("black")
    fig.patch.set_linewidth(border_width)

    # Row 1: Original structures
    # Note: points are normalized by box_size, so they're in range ~[-0.5, 0.5]
    viz_limit = 0.6  # Slightly larger than 0.5 for better visualization

    for col, (name, pc) in enumerate(zip(structure_names[:len(point_clouds)], point_clouds)):
        ax = fig.add_subplot(3, n_structures, col + 1, projection="3d")

        ax.scatter(
            pc[:, 0], pc[:, 1], pc[:, 2],
            s=25, alpha=0.9,
            edgecolors="black",
            linewidths=0.3,
        )

        ax.set_title(f"{name}\n(Original)")
        ax.set_xlim(-viz_limit, viz_limit)
        ax.set_ylim(-viz_limit, viz_limit)
        ax.set_zlim(-viz_limit, viz_limit)

    # Row 2: Reconstructions
    for col, (name, recon) in enumerate(zip(structure_names[:len(reconstructions_list)], reconstructions_list)):
        ax = fig.add_subplot(3, n_structures, n_structures + col + 1, projection="3d")

        ax.scatter(
            recon[:, 0], recon[:, 1], recon[:, 2],
            s=25, alpha=0.9,
            edgecolors="black",
            linewidths=0.3,
            c="orange",
        )

        ax.set_title(f"{name}\n(Reconstruction)")
        ax.set_xlim(-viz_limit, viz_limit)
        ax.set_ylim(-viz_limit, viz_limit)
        ax.set_zlim(-viz_limit, viz_limit)

    # Row 3: Latent space visualization (PCA to 3D)
    if len(latents_list) > 1:
        latents_array = np.array(latents_list)

        # Apply PCA to reduce to 3D
        pca = PCA(n_components=min(3, latents_array.shape[1]))
        latents_3d = pca.fit_transform(latents_array)

        # If we have fewer than 3 components, pad with zeros
        if latents_3d.shape[1] < 3:
            latents_3d = np.pad(
                latents_3d,
                ((0, 0), (0, 3 - latents_3d.shape[1])),
                mode='constant'
            )

        # Create a single plot for latent space
        ax = fig.add_subplot(3, n_structures, 2 * n_structures + n_structures // 2 + 1, projection="3d")

        # Use different colors for each structure
        colors = cm.tab20(np.linspace(0, 1, len(structure_names[:len(latents_3d)])))

        for i, (name, latent_pt, color) in enumerate(zip(structure_names[:len(latents_3d)], latents_3d, colors)):
            ax.scatter(
                latent_pt[0], latent_pt[1], latent_pt[2],
                s=200, alpha=0.9,
                edgecolors="black",
                linewidths=2,
                c=[color],
                label=name,
            )

        ax.set_title("Latent Space (PCA)")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})" if len(pca.explained_variance_ratio_) > 1 else "PC2")
        ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})" if len(pca.explained_variance_ratio_) > 2 else "PC3")

    fig.suptitle("Reference Structure Analysis", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict and visualize SPD model latent space clustering on synthetic data"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--n-phases",
        type=int,
        default=3,
        help="Number of phases for KMeans clustering (default: 3)",
    )
    parser.add_argument(
        "--box-size",
        type=float,
        default=None,
        help="Size of simulation box (default: inferred from config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/spd_predictions",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="output/spd_analysis/predictions_cache",
        help="Directory for caching predictions",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation even if cache exists",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="CUDA device ID (default: 0)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process",
    )
    parser.add_argument(
        "--reference-structures",
        type=str,
        default=None,
        help="Path to reference_structures.npy (optional, for reference structure visualization)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load predictions (or compute and cache)
    print("\n=== Step 1: Loading/Computing Predictions ===")
    (latents, sample_coords, phase_labels, grain_labels,
     model, cfg, atoms, metadata) = predict_and_cache(
        checkpoint_path=args.checkpoint,
        cuda_device=args.cuda_device,
        cache_dir=args.cache_dir,
        force_recompute=args.force_recompute,
        max_samples=args.max_samples,
    )

    print(f"Latents shape: {latents.shape}")
    print(f"Sample coords shape: {sample_coords.shape}")
    print(f"Phase labels shape: {phase_labels.shape}")

    # Step 2: Perform clustering
    print("\n=== Step 2: Clustering ===")
    kmeans_labels, hdbscan_labels = perform_clustering(latents, args.n_phases)

    # Save cluster assignments
    cluster_file = output_dir / "cluster_assignments.npz"
    np.savez_compressed(
        cluster_file,
        kmeans=kmeans_labels,
        hdbscan=hdbscan_labels,
    )
    print(f"Saved cluster assignments to {cluster_file}")

    # Step 3: Create visualization
    print("\n=== Step 3: Creating Visualization ===")
    box_size = args.box_size
    if box_size is None:
        box_size = metadata.get("box_size", getattr(cfg.data, "L", 100.0))

    viz_file = output_dir / "clustering_visualization.png"
    create_visualization(
        atoms=atoms,
        metadata=metadata,
        sample_coords=sample_coords,
        phase_labels_sample=phase_labels,
        grain_labels_sample=grain_labels,
        kmeans_labels=kmeans_labels,
        hdbscan_labels=hdbscan_labels,
        box_size=box_size,
        output_path=viz_file,
    )

    # Step 4: Reference structures visualization (if provided)
    if args.reference_structures:
        print("\n=== Step 4: Creating Reference Structures Visualization ===")
        visualize_reference_structures(
            checkpoint_path=args.checkpoint,
            reference_structures_path=args.reference_structures,
            output_path=output_dir / "reference_structures_analysis.png",
            cuda_device=args.cuda_device,
            target_atoms=64,
            box_size=10.0,
        )

    print("\n=== Done! ===")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
