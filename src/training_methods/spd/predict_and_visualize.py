"""
Prediction and visualization pipeline for SPD models with synthetic data.

This script demonstrates how to:
1. Load a trained SPD model from checkpoint
2. Extract latent representations from synthetic training data
3. Perform clustering analysis (KMeans and HDBSCAN)
4. Create visualizations comparing ground truth and predictions
5. Compare reference structures with actual dataset samples

The script automatically loads synthetic data from the model's config.

Features:
- Compare reference structures (from reference_point_clouds.npy) with actual dataset samples
- Visualize preprocessing at each step with diagnostic output
- 4-row visualization: Originals, Canonicals, Reconstructions, Latent space
- Color-coded: Blue=Reference, Green=Dataset, Purple=Canonical, Orange=Reconstruction

Use this to investigate preprocessing inconsistencies or reference structure issues.
"""

from __future__ import annotations

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
from src.data_utils.data_load import SyntheticPointCloudDataset, pc_normalize
from src.data_utils.prepare_data import get_regular_samples, get_random_samples


def extract_actual_dataset_samples(
    dataset: SyntheticPointCloudDataset,
    num_samples_per_phase: int = 3,
) -> Dict[str, List[np.ndarray]]:
    """Extract actual samples from the synthetic dataset, organized by phase.

    Returns:
        Dictionary mapping phase names to lists of point clouds (as numpy arrays).
    """
    phase_samples: Dict[str, List[np.ndarray]] = {}

    # Get phase index to name mapping
    idx_to_phase = {idx: name for name, idx in dataset._phase_to_idx.items()}

    # Track samples collected per phase
    phase_counts: Dict[str, int] = {name: 0 for name in idx_to_phase.values()}

    for i in range(len(dataset)):
        # Check if we've collected enough samples for all phases
        if all(count >= num_samples_per_phase for count in phase_counts.values()):
            break

        phase_idx = dataset._phase_labels[i]
        phase_name = idx_to_phase.get(phase_idx, f"unknown_{phase_idx}")

        if phase_counts[phase_name] < num_samples_per_phase:
            # Get the raw tensor (already normalized by dataset)
            pc_tensor = dataset.samples[i]
            pc_numpy = pc_tensor.numpy()

            if phase_name not in phase_samples:
                phase_samples[phase_name] = []

            phase_samples[phase_name].append(pc_numpy)
            phase_counts[phase_name] += 1

    return phase_samples


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
    """
    Create 6-panel visualization comparing ground truth and predictions.
    All requested views are stacked into a single figure.
    """
    # Extract ground truth from atoms
    atom_phase_labels, atom_grain_labels = extract_ground_truth_labels(atoms, metadata)

    # Sample atoms for ground truth visualization once so every view reuses the same subset
    n_atoms = len(atoms)
    if n_atoms > max_points_per_panel:
        atom_sample_indices = np.random.choice(n_atoms, max_points_per_panel, replace=False)
    else:
        atom_sample_indices = np.arange(n_atoms)

    atoms_sample = atoms[atom_sample_indices]
    atom_phase_sample = atom_phase_labels[atom_sample_indices]
    atom_grain_sample = atom_grain_labels[atom_sample_indices]

    # Sample predictions (latents)
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

    payload = {
        "atoms_sample": atoms_sample,
        "atom_phase_sample": atom_phase_sample,
        "atom_grain_sample": atom_grain_sample,
        "coords_sample": coords_sample,
        "kmeans_sample": kmeans_sample,
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
    n_panels = 6
    fig = plt.figure(figsize=(24, 8 * n_views))
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
    payload: Dict[str, np.ndarray],
    box_size: float,
    diagonal_cut: bool = False,
    cut_ratio: float = 0.65,
    title_suffix: str = "",
    border_width: float = 1.0,
) -> None:
    """Populate the standard clustering panels into the provided axes."""
    atoms_sample = payload["atoms_sample"]
    atom_phase_sample = payload["atom_phase_sample"]
    atom_grain_sample = payload["atom_grain_sample"]
    coords_sample = payload["coords_sample"]
    kmeans_sample = payload["kmeans_sample"]
    hdbscan_sample = payload["hdbscan_sample"]
    phase_sample = payload["phase_sample"]
    grain_sample = payload["grain_sample"]

    if len(axes) != 6:
        raise ValueError("Expected exactly 6 axes to populate clustering panels")

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
            hdbscan_sample = hdbscan_sample[coords_mask]
            phase_sample = phase_sample[coords_mask]
            grain_sample = grain_sample[coords_mask]

    # Panel 1: Ground truth phases
    ax1 = axes[0]
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

    # Panel 2: Ground truth grains
    ax2 = axes[1]
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

    # Panel 3: Ground truth boundaries only
    ax3 = axes[2]
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

    # Panel 4: KMeans predictions
    ax4 = axes[3]
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

    # Panel 5: HDBSCAN predictions
    ax5 = axes[4]
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
    ax5.set_title(f"HDBSCAN Clusters ({n_clusters} + noise){title_suffix}")
    _set_cube_axes(ax5, box_size)
    _add_axes_border(ax5, linewidth=border_width)

    # Panel 6: KMeans-Phase Agreement
    ax6 = axes[5]
    cluster_to_phase = {}
    for cluster in np.unique(kmeans_sample):
        mask = kmeans_sample == cluster
        if np.any(mask):
            phases_in_cluster = phase_sample[mask]
            unique, counts = np.unique(phases_in_cluster, return_counts=True)
            cluster_to_phase[cluster] = unique[np.argmax(counts)]

    correct_mask = np.array([
        cluster_to_phase.get(kmeans_sample[i], None) == phase_sample[i]
        for i in range(len(kmeans_sample))
    ]) if len(kmeans_sample) else np.array([], dtype=bool)

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

    accuracy = (np.sum(correct_mask) / len(correct_mask) * 100) if len(correct_mask) else 0.0
    ax6.set_title(f"KMeans-Phase Agreement{title_suffix}\n({accuracy:.1f}% accuracy)")
    if len(correct_mask):
        ax6.legend(loc="upper right", fontsize=8)
    _set_cube_axes(ax6, box_size)
    _add_axes_border(ax6, linewidth=border_width)


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


def visualize_reference_structures(
    checkpoint_path: str,
    reference_structures_path: str,
    output_path: Path,
    cuda_device: int = 0,
    dataset = None,
    compare_with_dataset: bool = True,
) -> None:
    """
    Create visualization of reference structures, their reconstructions, and latent space.
    Optionally compare with actual dataset samples.

    Args:
        checkpoint_path: Path to trained SPD model checkpoint
        reference_structures_path: Path to reference_point_clouds.npy file
        output_path: Path to save visualization
        cuda_device: CUDA device ID
        dataset: Optional dataset to extract actual samples for comparison
        compare_with_dataset: Whether to add dataset samples to visualization
    """
    from sklearn.decomposition import PCA

    print(f"Loading checkpoint from {checkpoint_path}")
    model, cfg, device = load_spd_model(checkpoint_path, cuda_device=cuda_device)
    model.eval()

    # Get the radius used for dataset normalization - CRITICAL for correct preprocessing!
    radius = float(cfg.data.radius) if hasattr(cfg.data, 'radius') else None
    print(f"Model training radius: {radius}")
    if radius is None:
        print("WARNING: Could not determine radius from config. Using max_norm normalization.")

    print(f"Loading reference point clouds from {reference_structures_path}")
    ref_point_clouds = np.load(reference_structures_path, allow_pickle=True).item()

    if not isinstance(ref_point_clouds, dict) or len(ref_point_clouds) == 0:
        print("No reference point clouds found in the provided file")
        return

    structure_names: List[str] = []
    point_clouds: List[np.ndarray] = []
    source_types: List[str] = []  # Track whether from reference or dataset

    # Load reference structures
    print("\n=== Processing Reference Structures ===")
    print("APPLYING FIX: Normalizing by radius (not max_norm) to match dataset preprocessing")
    for name in sorted(ref_point_clouds.keys()):
        if name.startswith("intermediate_"):
            continue

        pc = np.asarray(ref_point_clouds[name], dtype=float)
        if pc.ndim != 2 or pc.shape[1] != 3:
            print(f"    Warning: Invalid point cloud shape for {name}, skipping")
            continue
        if len(pc) == 0:
            print(f"    Warning: Empty point cloud for {name}, skipping")
            continue

        # DIAGNOSTIC: Check original state
        mean_before = pc.mean(axis=0)
        max_norm_before = np.max(np.linalg.norm(pc, axis=1))
        print(f"  {name} [BEFORE]: mean={mean_before}, max_norm={max_norm_before:.4f}")

        # FIX: Apply the SAME preprocessing as dataset samples
        # Reference structures are normalized to max_norm=1.0, but the model was trained
        # on dataset samples normalized by RADIUS. This mismatch causes reconstruction issues!

        # 1. Center (re-center to ensure mean=0)
        pc_centered = pc - pc.mean(axis=0, keepdims=True)

        # 2. Apply radius normalization (like dataset samples do via pc_normalize)
        if radius is not None:
            pc_normalized = pc_normalize(pc_centered, radius)
        else:
            # Fallback to max_norm normalization if radius not available
            pc_normalized = pc_normalize(pc_centered, None)

        # DIAGNOSTIC: Check after preprocessing
        mean_after = pc_normalized.mean(axis=0)
        max_norm_after = np.max(np.linalg.norm(pc_normalized, axis=1))
        scale_factor = max_norm_before / max_norm_after if max_norm_after > 0 else 1.0
        print(f"  {name} [AFTER]:  mean={mean_after}, max_norm={max_norm_after:.4f}, scale_factor={scale_factor:.4f}")

        point_clouds.append(pc_normalized)
        structure_names.append(f"ref_{name}")
        source_types.append("reference")

    if len(point_clouds) == 0:
        print("No valid structures to visualize")
        return

    print(f"\nLoaded {len(structure_names)} reference structures")

    # Add actual dataset samples if requested
    if compare_with_dataset and dataset is not None:
        print("\n=== Extracting Actual Dataset Samples ===")
        dataset_samples = extract_actual_dataset_samples(dataset, num_samples_per_phase=2)

        for phase_name, samples_list in sorted(dataset_samples.items()):
            for idx, sample_pc in enumerate(samples_list):
                # Dataset samples are already preprocessed (centered and normalized)
                mean_ds = sample_pc.mean(axis=0)
                max_norm_ds = np.max(np.linalg.norm(sample_pc, axis=1))
                print(f"  {phase_name}_sample{idx}: mean={mean_ds}, max_norm={max_norm_ds:.4f}")

                point_clouds.append(sample_pc)
                structure_names.append(f"ds_{phase_name}_{idx}")
                source_types.append("dataset")

        print(f"Added {sum(len(v) for v in dataset_samples.values())} dataset samples")

    print(f"\nTotal structures to process: {len(structure_names)}")

    latents_list: List[np.ndarray] = []
    reconstructions_list: List[np.ndarray] = []
    canonicals_list: List[np.ndarray] = []
    originals_list: List[np.ndarray] = []  # Store successfully processed originals
    recon_names: List[str] = []
    recon_source_types: List[str] = []

    # Use the device from load_spd_model
    print("\n=== Running Model Inference ===")
    with torch.no_grad():
        for name, pc, source_type in zip(structure_names, point_clouds, source_types):
            print(f"  Processing {name} ({source_type})...")

            # DIAGNOSTIC: Check preprocessing before model
            mean_pre = pc.mean(axis=0)
            max_norm_pre = np.max(np.linalg.norm(pc, axis=1))
            print(f"    Before model: mean={mean_pre}, max_norm={max_norm_pre:.4f}")

            # Convert to tensor and add batch dimension
            pc_tensor = torch.from_numpy(pc).float().unsqueeze(0).to(device)

            # Get latent and reconstruction
            # SPD model forward returns: (inv_z, recon, cano, rot)
            try:
                inv_z, recon, cano, rot = model(pc_tensor)

                recon_np = recon.cpu().numpy()[0]
                cano_np = cano.cpu().numpy()[0]

                # DIAGNOSTIC: Check reconstruction statistics
                mean_recon = recon_np.mean(axis=0)
                max_norm_recon = np.max(np.linalg.norm(recon_np, axis=1))
                print(f"    Reconstruction: mean={mean_recon}, max_norm={max_norm_recon:.4f}")

                latents_list.append(inv_z.cpu().numpy()[0])
                reconstructions_list.append(recon_np)
                canonicals_list.append(cano_np)
                originals_list.append(pc)  # Store the original that was successfully processed
                recon_names.append(name)
                recon_source_types.append(source_type)
            except Exception as e:
                print(f"    Warning: Model inference failed for {name}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"\nSuccessfully processed {len(recon_names)} structures")

    # Create visualization
    n_structures = len(recon_names)
    if n_structures == 0:
        print("No structures to visualize")
        return

    # Use 4 rows: originals, canonicals, reconstructions, latent space
    fig = plt.figure(figsize=(4 * n_structures, 16))

    border_width = 72.0 / fig.dpi
    fig.patch.set_facecolor("white")
    fig.patch.set_edgecolor("black")
    fig.patch.set_linewidth(border_width)

    # Determine visualization limits
    all_points = originals_list + reconstructions_list + canonicals_list
    if all_points:
        max_extent = max(np.max(np.abs(pc)) for pc in all_points)
        viz_limit = max(0.6, float(max_extent) * 1.1)
    else:
        viz_limit = 0.6

    # Color coding: blue for reference, green for dataset
    color_map = {"reference": "blue", "dataset": "green"}

    # Row 1: Original structures
    for col, (name, pc, source_type) in enumerate(zip(recon_names, originals_list, recon_source_types)):
        ax = fig.add_subplot(4, n_structures, col + 1, projection="3d")

        color = color_map.get(source_type, "gray")
        ax.scatter(
            pc[:, 0], pc[:, 1], pc[:, 2],
            s=25, alpha=0.9,
            edgecolors="black",
            linewidths=0.3,
            c=color,
        )

        # Shorten name for display
        display_name = name.replace("ref_", "").replace("ds_", "")
        source_label = "REF" if source_type == "reference" else "DS"
        ax.set_title(f"[{source_label}] {display_name}\n(Original)", fontsize=8)
        ax.set_xlim(-viz_limit, viz_limit)
        ax.set_ylim(-viz_limit, viz_limit)
        ax.set_zlim(-viz_limit, viz_limit)

    # Row 2: Canonical (before rotation)
    for col, (name, cano, source_type) in enumerate(zip(recon_names, canonicals_list, recon_source_types)):
        ax = fig.add_subplot(4, n_structures, n_structures + col + 1, projection="3d")

        ax.scatter(
            cano[:, 0], cano[:, 1], cano[:, 2],
            s=25, alpha=0.9,
            edgecolors="black",
            linewidths=0.3,
            c="purple",
        )

        display_name = name.replace("ref_", "").replace("ds_", "")
        source_label = "REF" if source_type == "reference" else "DS"
        ax.set_title(f"[{source_label}] {display_name}\n(Canonical)", fontsize=8)
        ax.set_xlim(-viz_limit, viz_limit)
        ax.set_ylim(-viz_limit, viz_limit)
        ax.set_zlim(-viz_limit, viz_limit)

    # Row 3: Reconstructions
    for col, (name, recon, source_type) in enumerate(zip(recon_names, reconstructions_list, recon_source_types)):
        ax = fig.add_subplot(4, n_structures, 2 * n_structures + col + 1, projection="3d")

        ax.scatter(
            recon[:, 0], recon[:, 1], recon[:, 2],
            s=25, alpha=0.9,
            edgecolors="black",
            linewidths=0.3,
            c="orange",
        )

        display_name = name.replace("ref_", "").replace("ds_", "")
        source_label = "REF" if source_type == "reference" else "DS"
        ax.set_title(f"[{source_label}] {display_name}\n(Reconstruction)", fontsize=8)
        ax.set_xlim(-viz_limit, viz_limit)
        ax.set_ylim(-viz_limit, viz_limit)
        ax.set_zlim(-viz_limit, viz_limit)

    # Row 4: Latent space visualization (PCA to 3D)
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

        # Create a single plot for latent space centered in the bottom row
        ax = fig.add_subplot(4, n_structures, 3 * n_structures + n_structures // 2 + 1, projection="3d")

        # Color by source type: blue for reference, green for dataset
        for i, (name, latent_pt, source_type) in enumerate(zip(recon_names[:len(latents_3d)], latents_3d, recon_source_types)):
            color = color_map.get(source_type, "gray")
            marker = 'o' if source_type == "reference" else '^'
            display_name = name.replace("ref_", "").replace("ds_", "")

            ax.scatter(
                latent_pt[0], latent_pt[1], latent_pt[2],
                s=200, alpha=0.9,
                edgecolors="black",
                linewidths=2,
                c=color,
                marker=marker,
                label=display_name,
            )

        ax.set_title("Latent Space (PCA)\nBlue=Reference, Green=Dataset", fontsize=10)
        ax.legend(loc="upper left", fontsize=6, ncol=2)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})" if len(pca.explained_variance_ratio_) > 1 else "PC2")
        ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})" if len(pca.explained_variance_ratio_) > 2 else "PC3")

    title = "Reference Structure Analysis"
    if compare_with_dataset and dataset is not None:
        title += " (with Dataset Samples)"
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nVisualization saved to {output_path}")
    print("Legend: REF=Reference structure, DS=Dataset sample")
    print("Colors: Blue=Reference, Green=Dataset, Purple=Canonical, Orange=Reconstruction")


def main():
    # ========================================================================
    # CONFIGURATION - Edit these paths to match your setup
    # ========================================================================

    # Path to your trained model checkpoint
    checkpoint_path = "output/2025-11-14/04-09-11/synth_SPD_VN_EQ_l96_P80_sinkhorn_512-epoch=74.ckpt"

    # Expected number of phases for KMeans clustering
    n_phases = 3

    # Output directory for all visualizations
    output_dir = "output/spd_analysis"
    cache_dir = "output/spd_analysis/predictions_cache"

    # Number of samples to process (None = all samples)
    max_samples = None  # Set to e.g., 1000 for faster testing

    # Path to reference point clouds (set to None to skip reference analysis)
    reference_structures_path = "output/synthetic_data/baseline_box_no_perturb/reference_point_clouds.npy"

    # ========================================================================
    # COMPARISON SETTINGS - For investigating preprocessing issues
    # ========================================================================

    # Compare reference structures with actual dataset samples
    # This helps identify preprocessing inconsistencies
    compare_with_dataset = True  # Set to False for faster execution

    # Limit dataset samples for comparison (to avoid memory issues)
    max_samples_for_comparison = 5000

    # ========================================================================
    # PIPELINE EXECUTION
    # ========================================================================

    # Step 1: Extract latents (or load from cache)
    print("\n=== Step 1: Extracting latents from synthetic data ===")
    (latents, sample_coords, phase_labels, grain_labels,
     model, cfg, atoms, metadata) = predict_and_cache(
        checkpoint_path=checkpoint_path,
        cuda_device=0,
        cache_dir=cache_dir,
        force_recompute=False,  # Set to True to recompute
        max_samples=max_samples,
    )

    print(f"Latents shape: {latents.shape}")
    print(f"Sample coords shape: {sample_coords.shape}")
    print(f"Phase labels shape: {phase_labels.shape}")
    print(f"Atoms shape: {atoms.shape}")

    # Step 2: Perform clustering
    print("\n=== Step 2: Performing clustering ===")
    kmeans_labels, hdbscan_labels = perform_clustering(latents, n_phases)

    # Step 3: Create visualization
    print("\n=== Step 3: Creating visualization ===")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Infer box size from metadata
    box_size = metadata.get("box_size", 160.0)

    create_visualization(
        atoms=atoms,
        metadata=metadata,
        sample_coords=sample_coords,
        phase_labels_sample=phase_labels,
        grain_labels_sample=grain_labels,
        kmeans_labels=kmeans_labels,
        hdbscan_labels=hdbscan_labels,
        box_size=box_size,
        output_path=output_path / "clustering_visualization.png",
    )

    print(f"\nClustering visualization saved to {output_path / 'clustering_visualization.png'}")

    # Step 4: Create reference structures visualization (optional)
    if reference_structures_path:
        print("\n=== Step 4: Creating reference structures visualization ===")

        # Create dataset for comparison if requested
        dataset_for_viz = None
        if compare_with_dataset:
            print(f"Creating dataset for comparison (max {max_samples_for_comparison} samples)...")
            # Get environment directories from the loaded config
            env_dirs = None
            if hasattr(cfg.data, 'env_dirs'):
                env_dirs = cfg.data.env_dirs
            elif hasattr(cfg.data, 'data_path'):
                env_dirs = [cfg.data.data_path] if isinstance(cfg.data.data_path, str) else cfg.data.data_path

            if env_dirs:
                dataset_for_viz, _ = create_synthetic_dataset_with_coords(
                    env_dirs, cfg, max_samples=max_samples_for_comparison
                )
                print(f"Dataset created with {len(dataset_for_viz)} samples")
            else:
                print("Warning: Could not determine environment directories for dataset comparison")

        visualize_reference_structures(
            checkpoint_path=checkpoint_path,
            reference_structures_path=reference_structures_path,
            output_path=output_path / "reference_structures_analysis.png",
            cuda_device=0,
            dataset=dataset_for_viz,
            compare_with_dataset=compare_with_dataset,
        )

        comparison_note = " (with dataset comparison)" if compare_with_dataset else ""
        print(f"Reference structures visualization{comparison_note} saved to {output_path / 'reference_structures_analysis.png'}")

    print(f"\n=== Done! ===")
    print(f"All outputs saved to {output_path}")
    print("\nVisualization includes:")
    print("  - Clustering visualization: Ground truth vs KMeans vs HDBSCAN")
    if reference_structures_path:
        if compare_with_dataset:
            print("  - Reference structures analysis: Reference (blue) vs Dataset samples (green)")
            print("    Shows: Originals, Canonicals, Reconstructions, and Latent space")
        else:
            print("  - Reference structures analysis: Reference structures only")


if __name__ == "__main__":
    main()
