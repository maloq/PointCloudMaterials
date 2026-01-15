"""Prediction and visualization pipeline for SPD models with synthetic data."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.data_utils.data_load import SyntheticPointCloudDataset
from src.data_utils.prepare_data import get_regular_samples
from src.training_methods.spd.eval_spd import load_spd_model
from src.vis_tools.spd_clustering_viz import create_visualization
from src.vis_tools.spd_reference_viz import visualize_dataset_samples, visualize_latent_space


# =============================================================================
# RDF AND ADF COMPUTATION (Per-Phase)
# =============================================================================

def estimate_nn_distance(positions: np.ndarray, k: int = 6) -> float:
    """Estimate average nearest neighbor distance in a point cloud."""
    if len(positions) < 2:
        return 0.1
    tree = cKDTree(positions)
    k_actual = min(k + 1, len(positions))
    dists, _ = tree.query(positions, k=k_actual)
    # Average distance to nearest neighbor (excluding self at index 0)
    nn_dists = dists[:, 1] if dists.shape[1] > 1 else dists[:, 0]
    return float(np.mean(nn_dists))


def compute_rdf(
    positions: np.ndarray,
    r_max: Optional[float] = None,
    n_bins: int = 50,
    sample_size: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute radial distribution function g(r) for a point cloud.
    
    Args:
        positions: (N, 3) point positions (normalized)
        r_max: Maximum radius for RDF (auto-computed if None)
        n_bins: Number of histogram bins
        sample_size: Number of atoms to sample for computation
        rng: Random number generator
        
    Returns:
        r_centers: Bin centers
        g_r: RDF values
    """
    n = len(positions)
    if n < 2:
        r_max = r_max or 1.0
        return np.linspace(0, r_max, n_bins), np.zeros(n_bins)
    
    # Auto-compute r_max based on nearest neighbor distance
    if r_max is None:
        nn_dist = estimate_nn_distance(positions)
        r_max = nn_dist * 5.0  # Go out to ~5x nearest neighbor distance
    
    rng = rng or np.random.default_rng(42)
    tree = cKDTree(positions)
    
    actual_sample = min(sample_size, n)
    sample_indices = rng.choice(n, actual_sample, replace=False)
    
    r_bins = np.linspace(0, r_max, n_bins + 1)
    dr = r_bins[1] - r_bins[0]
    hist = np.zeros(n_bins)
    
    for i in sample_indices:
        neighbors = tree.query_ball_point(positions[i], r_max)
        for j in neighbors:
            if j != i:
                dist = np.linalg.norm(positions[j] - positions[i])
                if 0 < dist < r_max:
                    bin_idx = int(dist / dr)
                    if 0 <= bin_idx < n_bins:
                        hist[bin_idx] += 1
    
    r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
    shell_volumes = 4 * np.pi * r_centers**2 * dr
    
    # Estimate density from bounding box
    pos_range = positions.max(axis=0) - positions.min(axis=0)
    box_vol = np.prod(np.maximum(pos_range, 1e-6))
    rho = n / box_vol
    expected = actual_sample * rho * shell_volumes
    
    g_r = np.divide(hist, expected, where=expected > 0, out=np.ones_like(hist))
    
    return r_centers, g_r


def compute_bond_angles(
    positions: np.ndarray,
    cutoff: Optional[float] = None,
    sample_size: int = 300,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Compute bond angle distribution for a point cloud.
    
    Args:
        positions: (N, 3) point positions
        cutoff: Neighbor cutoff distance (auto-computed if None)
        sample_size: Number of atoms to sample
        rng: Random number generator
        
    Returns:
        angles: Array of bond angles in degrees
    """
    n = len(positions)
    if n < 3:
        return np.array([])
    
    # Auto-compute cutoff based on nearest neighbor distance
    if cutoff is None:
        nn_dist = estimate_nn_distance(positions)
        cutoff = nn_dist * 1.5  # Include first coordination shell
    
    rng = rng or np.random.default_rng(42)
    tree = cKDTree(positions)
    
    actual_sample = min(sample_size, n)
    sample_indices = rng.choice(n, actual_sample, replace=False)
    
    angles = []
    for i in sample_indices:
        neighbors = tree.query_ball_point(positions[i], cutoff)
        neighbors = [j for j in neighbors if j != i]
        
        if len(neighbors) < 2:
            continue
        
        center = positions[i]
        for k, j1 in enumerate(neighbors[:6]):
            for j2 in neighbors[k+1:7]:
                v1 = positions[j1] - center
                v2 = positions[j2] - center
                
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 < 1e-10 or norm2 < 1e-10:
                    continue
                    
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.degrees(np.arccos(cos_angle))
                angles.append(angle)
    
    return np.array(angles)


def compute_rdf_adf_comparison(
    model: torch.nn.Module,
    dataset: SyntheticPointCloudDataset,
    device: str,
    output_path: Path,
    num_samples_per_phase: int = 50,
    n_bins_rdf: int = 50,
    n_bins_adf: int = 36,
) -> None:
    """Compute and plot RDF and ADF comparison between original and reconstructed samples, per phase.
    
    Args:
        model: SPD model
        dataset: Synthetic dataset with phase labels
        device: Device for inference
        output_path: Path to save the figure
        num_samples_per_phase: Number of samples per phase to analyze
        n_bins_rdf: Number of RDF bins
        n_bins_adf: Number of ADF bins
    """
    print("Computing RDF and ADF comparison per phase...")
    
    # Get phase mapping
    idx_to_phase = {idx: name for name, idx in dataset._phase_to_idx.items()}
    unique_phases = sorted(idx_to_phase.keys())
    n_phases = len(unique_phases)
    
    # Collect samples per phase
    phase_samples: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {p: [] for p in unique_phases}
    
    model.eval()
    rng = np.random.default_rng(42)
    
    # Group indices by phase
    phase_indices: Dict[int, List[int]] = {p: [] for p in unique_phases}
    for i in range(len(dataset)):
        phase_idx = dataset._phase_labels[i]
        if phase_idx in phase_indices:
            phase_indices[phase_idx].append(i)
    
    # Sample and process
    with torch.inference_mode():
        for phase_idx in unique_phases:
            indices = phase_indices[phase_idx]
            if len(indices) == 0:
                continue
            
            sample_indices = rng.choice(
                indices, 
                size=min(num_samples_per_phase, len(indices)), 
                replace=False
            )
            
            for idx in sample_indices:
                pc = dataset.samples[idx].numpy()
                pc_tensor = torch.from_numpy(pc).float().unsqueeze(0).to(device)
                
                _, recon, _, _, _ = model(pc_tensor)
                recon_np = recon.cpu().numpy()[0]
                
                phase_samples[phase_idx].append((pc, recon_np))
    
    # Create figure: n_phases rows x 2 columns (RDF, ADF)
    fig, axes = plt.subplots(n_phases, 2, figsize=(12, 4 * n_phases))
    if n_phases == 1:
        axes = axes.reshape(1, -1)
    
    colors_orig = '#2ecc71'  # green
    colors_recon = '#e74c3c'  # red
    
    for row, phase_idx in enumerate(unique_phases):
        phase_name = idx_to_phase.get(phase_idx, f"Phase {phase_idx}")
        samples = phase_samples[phase_idx]
        
        if len(samples) == 0:
            axes[row, 0].text(0.5, 0.5, f"No samples for {phase_name}", 
                            ha='center', va='center', transform=axes[row, 0].transAxes)
            axes[row, 1].text(0.5, 0.5, f"No samples for {phase_name}",
                            ha='center', va='center', transform=axes[row, 1].transAxes)
            continue
        
        # Aggregate RDF and ADF across samples
        all_rdf_orig = []
        all_rdf_recon = []
        all_adf_orig = []
        all_adf_recon = []
        
        for orig, recon in samples:
            # Auto-compute r_max and cutoff based on data
            r_centers, rdf_o = compute_rdf(orig, n_bins=n_bins_rdf)
            _, rdf_r = compute_rdf(recon, n_bins=n_bins_rdf)
            all_rdf_orig.append(rdf_o)
            all_rdf_recon.append(rdf_r)
            
            adf_o = compute_bond_angles(orig)
            adf_r = compute_bond_angles(recon)
            all_adf_orig.extend(adf_o)
            all_adf_recon.extend(adf_r)
        
        # Average RDF
        mean_rdf_orig = np.mean(all_rdf_orig, axis=0)
        std_rdf_orig = np.std(all_rdf_orig, axis=0)
        mean_rdf_recon = np.mean(all_rdf_recon, axis=0)
        std_rdf_recon = np.std(all_rdf_recon, axis=0)
        
        # Plot RDF
        ax_rdf = axes[row, 0]
        ax_rdf.plot(r_centers, mean_rdf_orig, color=colors_orig, label='Original', linewidth=2)
        ax_rdf.fill_between(r_centers, mean_rdf_orig - std_rdf_orig, mean_rdf_orig + std_rdf_orig,
                           color=colors_orig, alpha=0.2)
        ax_rdf.plot(r_centers, mean_rdf_recon, color=colors_recon, label='Reconstructed', 
                   linewidth=2, linestyle='--')
        ax_rdf.fill_between(r_centers, mean_rdf_recon - std_rdf_recon, mean_rdf_recon + std_rdf_recon,
                           color=colors_recon, alpha=0.2)
        ax_rdf.set_xlabel('r (normalized)')
        ax_rdf.set_ylabel('g(r)')
        ax_rdf.set_title(f'{phase_name} - Radial Distribution Function')
        ax_rdf.legend()
        ax_rdf.grid(True, alpha=0.3)
        
        # Plot ADF
        ax_adf = axes[row, 1]
        if len(all_adf_orig) > 0:
            ax_adf.hist(all_adf_orig, bins=n_bins_adf, range=(0, 180), 
                       color=colors_orig, alpha=0.5, label='Original', density=True)
        if len(all_adf_recon) > 0:
            ax_adf.hist(all_adf_recon, bins=n_bins_adf, range=(0, 180),
                       color=colors_recon, alpha=0.5, label='Reconstructed', density=True)
        
        # Mark characteristic angles
        ax_adf.axvline(60, color='blue', linestyle=':', alpha=0.5, label='60° (FCC/ICO)')
        ax_adf.axvline(90, color='purple', linestyle=':', alpha=0.5, label='90° (BCC/FCC)')
        ax_adf.axvline(109.5, color='orange', linestyle=':', alpha=0.5, label='109.5° (tet)')
        
        ax_adf.set_xlabel('Bond Angle (degrees)')
        ax_adf.set_ylabel('Probability Density')
        ax_adf.set_title(f'{phase_name} - Angular Distribution Function')
        ax_adf.legend(loc='upper right', fontsize=8)
        ax_adf.set_xlim(0, 180)
        ax_adf.grid(True, alpha=0.3)
    
    fig.suptitle('RDF and ADF: Original vs Reconstructed (Per Phase)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved RDF/ADF comparison to {output_path}")


def load_synthetic_environment_data(env_dir: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load synthetic environment data and metadata."""
    env_path = Path(env_dir)
    atoms = np.load(env_path / "atoms.npy")
    with open(env_path / "metadata.json") as f:
        metadata = json.load(f)
    return atoms, metadata


def create_synthetic_dataset_with_coords(
    env_dirs: List[str], cfg: Any, max_samples: int = None
) -> Tuple[SyntheticPointCloudDataset, np.ndarray]:
    """Create synthetic dataset and extract sample center coordinates."""
    all_coords = []

    for env_dir in env_dirs:
        atoms, _ = load_synthetic_environment_data(env_dir)
        max_per_env = cfg.data.get("n_samples", 1000) if cfg.data.get("n_samples", 0) > 0 else int(2e9)

        samples = get_regular_samples(
            atoms,
            size=cfg.data.radius,
            overlap_fraction=cfg.data.get("overlap_fraction", 0.0),
            return_coords=True,
            n_points=cfg.data.num_points,
            max_samples=max_per_env,
            drop_edge_samples=True,
        )
        all_coords.extend([c for _, c in samples])

        if max_samples and len(all_coords) >= max_samples:
            all_coords = all_coords[:max_samples]
            break

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

    coords = np.array(all_coords[:len(dataset)], dtype=np.float32)
    return dataset, coords


def run_memory_safe_dbscan(
    latents: np.ndarray,
    eps: float,
    min_samples: int,
    max_samples: Optional[int] = None,
    random_state: int = 0,
) -> Tuple[np.ndarray, Optional[int]]:
    """Run DBSCAN while optionally sub-sampling to keep memory usage bounded."""

    n_points = len(latents)
    if not max_samples or max_samples <= 0 or n_points <= max_samples:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(latents)
        return labels, None

    max_samples = min(max_samples, n_points)
    rng = np.random.default_rng(random_state)
    subset_idx = rng.choice(n_points, size=max_samples, replace=False)
    subset_latents = latents[subset_idx]

    subset_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(subset_latents)
    full_labels = np.full(n_points, -1, dtype=int)
    full_labels[subset_idx] = subset_labels

    rest_mask = np.ones(n_points, dtype=bool)
    rest_mask[subset_idx] = False
    rest_idx = np.where(rest_mask)[0]

    core_mask = subset_labels >= 0
    if rest_idx.size > 0 and core_mask.any():
        core_latents = subset_latents[core_mask]
        core_labels = subset_labels[core_mask]
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(core_latents)
        dists, nbr_idx = nn.kneighbors(latents[rest_idx], return_distance=True)
        within_eps = dists[:, 0] <= eps
        if within_eps.any():
            assigned_idx = rest_idx[within_eps]
            assigned_labels = core_labels[nbr_idx[within_eps, 0]]
            full_labels[assigned_idx] = assigned_labels

    return full_labels, subset_idx.size


def extract_latents_with_ground_truth(
    model: torch.nn.Module,
    dataset: SyntheticPointCloudDataset,
    sample_coords: np.ndarray,
    device: str = "cpu",
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract latent representations with ground truth labels."""
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latents_list, phase_list, grain_list = [], [], []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Extracting latents"):
            points, phase, grain, _, _ = batch
            inv_z, _, _, _, _ = model(points.to(device))
            latents_list.append(inv_z.detach().cpu().numpy())
            phase_list.append(phase.detach().cpu().numpy())
            grain_list.append(grain.detach().cpu().numpy())

    return (
        np.concatenate(latents_list),
        sample_coords,
        np.concatenate(phase_list),
        np.concatenate(grain_list),
    )


def get_env_dirs(cfg: Any) -> List[str]:
    """Extract environment directories from config."""
    if hasattr(cfg.data, "env_dirs"):
        dirs = cfg.data.env_dirs
    elif hasattr(cfg.data, "data_path"):
        dirs = cfg.data.data_path
    elif hasattr(cfg.data, "synthetic") and hasattr(cfg.data.synthetic, "data_dir"):
        dirs = cfg.data.synthetic.data_dir
    else:
        raise ValueError("Config missing synthetic data path (env_dirs/data_path/synthetic.data_dir)")
    return [dirs] if isinstance(dirs, str) else dirs


def predict_and_cache(
    checkpoint_path: str,
    cuda_device: int = 0,
    cache_dir: str = "output/spd_analysis/predictions_cache",
    force_recompute: bool = False,
    max_samples: int = None,
    cfg: DictConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any, Any, np.ndarray, Dict]:
    """Load model, extract predictions from synthetic data, and cache results."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / f"{Path(checkpoint_path).stem}_synth_predictions.npz"

    print(f"Loading model from {checkpoint_path}")
    model, cfg, device = load_spd_model(checkpoint_path, cuda_device=cuda_device, cfg=cfg)
    env_dirs = get_env_dirs(cfg)

    print(f"Loading synthetic data from: {env_dirs}")
    atoms, metadata = load_synthetic_environment_data(env_dirs[0])
    print(f"Loaded {len(atoms)} atoms")

    if cache_file.exists() and not force_recompute:
        print(f"Loading from cache: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        return (data["latents"], data["sample_coords"], data["phase_labels"],
                data["grain_labels"], model, cfg, atoms, metadata)

    print("Computing predictions...")
    dataset, coords = create_synthetic_dataset_with_coords(env_dirs, cfg, max_samples)
    print(f"Created dataset with {len(dataset)} samples")

    latents, coords, phases, grains = extract_latents_with_ground_truth(
        model, dataset, coords, device, getattr(cfg, "batch_size", 32)
    )

    print(f"Saving to cache: {cache_file}")
    np.savez_compressed(cache_file, latents=latents, sample_coords=coords,
                       phase_labels=phases, grain_labels=grains)

    return latents, coords, phases, grains, model, cfg, atoms, metadata


def perform_clustering(
    latents: np.ndarray,
    n_phases: int,
    run_dbscan: bool = False,
    run_hdbscan: bool = False,
    k_range: Optional[range] = None,
    dbscan_eps: Optional[float] = None,
    dbscan_max_samples: Optional[int] = 2000,
    random_state: int = 0,
) -> Dict[str, Any]:
    """Perform clustering using KMeans (multiple k values), and optionally DBSCAN/HDBSCAN.
    
    Args:
        latents: Latent representations (N, D)
        n_phases: Default number of phases (used if k_range is None)
        run_dbscan: Whether to run DBSCAN clustering
        run_hdbscan: Whether to run HDBSCAN clustering
        k_range: Range of k values for KMeans (e.g., range(2, 7))
        dbscan_eps: DBSCAN epsilon parameter (auto-computed if None)
        dbscan_max_samples: Max samples for DBSCAN memory efficiency
        random_state: Random seed
        
    Returns:
        Dictionary with clustering results for each method and k value
    """
    print(f"Clustering {len(latents)} samples...")
    results: Dict[str, Any] = {"kmeans": {}}
    
    # Use provided k_range or default to single n_phases value
    if k_range is None:
        k_range = range(n_phases, n_phases + 1)
    
    # KMeans for each k value
    for k in k_range:
        print(f"  KMeans (k={k})...")
        labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(latents)
        results["kmeans"][k] = labels

    # DBSCAN (optional)
    if run_dbscan:
        print("  DBSCAN...")
        if dbscan_eps is None:
            norms = np.linalg.norm(latents - latents.mean(axis=0, keepdims=True), axis=1)
            eps = float(np.median(norms) * 0.3) if np.isfinite(norms).all() else 0.5
            eps = max(eps, 0.5) if eps > 0 else 0.5
        else:
            eps = float(dbscan_eps)

        min_samples = max(4, latents.shape[1] // 2)
        dbscan_labels, subset_size = run_memory_safe_dbscan(
            latents, eps=eps, min_samples=min_samples, max_samples=dbscan_max_samples, random_state=random_state
        )
        results["dbscan"] = dbscan_labels
        if subset_size:
            print(f"    Used subset of {subset_size} samples (out of {len(latents)}) to limit memory usage")
        print(f"    Found {len(np.unique(dbscan_labels[dbscan_labels >= 0]))} clusters "
              f"({(dbscan_labels == -1).sum()} noise, eps={eps:.3f})")
    else:
        results["dbscan"] = None

    # HDBSCAN (optional)
    if run_hdbscan:
        print("  HDBSCAN...")
        hdbscan_labels = hdbscan.HDBSCAN(
            min_cluster_size=max(10, len(latents) // 100), min_samples=5
        ).fit_predict(latents)
        results["hdbscan"] = hdbscan_labels
        print(f"    Found {len(np.unique(hdbscan_labels[hdbscan_labels >= 0]))} clusters "
              f"({(hdbscan_labels == -1).sum()} noise)")
    else:
        results["hdbscan"] = None

    return results


    
    return metrics


def compute_best_mapping_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Compute confusion matrix with best cluster-to-class assignment.
    
    Uses the Hungarian algorithm (linear sum assignment) to find the best mapping
    between predicted clusters and ground truth classes.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted cluster labels
        
    Returns:
        conf_mat: Reordered confusion matrix (rows=truth, cols=predicted_mapped)
        mapping: List of (row_ind, col_ind) mapping
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix
    
    # Compute standard confusion matrix
    # rows = true classes, cols = predicted clusters
    cm = confusion_matrix(y_true, y_pred)
    
    # We want to maximize the diagonal elements (matches)
    # linear_sum_assignment minimizes cost, so we use negative counts
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    # Reorder columns (predictions) to match rows (ground truth)
    # If standard cm is cm[i, j], finding mapping means class i maps to cluster j
    # We want to rearrange so that new_cm[i, i] is the count for class i mapped to its best cluster
    
    # Create a mapping dictionary: true_class -> best_cluster
    mapping = list(zip(row_ind, col_ind))
    
    # Reorder the confusion matrix columns
    # We need to construct a new matrix where column i corresponds to the cluster mapped to true class i
    new_cm = np.zeros_like(cm)
    
    # For each true class i (row index), find which cluster j it maps to (col_ind[i])
    # Then take that column j from the original matrix and place it in column i of the new matrix
    # Wait, simple way:
    # If mapping is (0, 2), (1, 0), (2, 1) means:
    # True class 0 corresponds to Cluster 2
    # True class 1 corresponds to Cluster 0
    # True class 2 corresponds to Cluster 1
    
    # So the confusion matrix we want to display should have:
    # Row 0: Distribution of True Class 0. Col 0 should be "Predicted as Class 0's cluster" (i.e., Cluster 2)
    # So new_cm[:, i] = cm[:, col_ind[i]] ??
    # Let's verify.
    # new_cm[0, 0] should be count where True=0 and Pred=2. Original cm[0, 2].
    # new_cm[1, 0] should be count where True=1 and Pred=2. Original cm[1, 2].
    # YES. We permute columns.
    
    # But wait, linear_sum_assignment returns row_ind and col_ind such that cost is minimized.
    # row_ind is usually just 0, 1, 2... if matrix is square.
    # If rectangular, it might skip some.
    
    n_classes = cm.shape[0]
    n_clusters = cm.shape[1]
    
    if n_classes == n_clusters:
        # Permute columns
        new_cm = cm[:, col_ind]
        # Also need to make sure row_ind is sorted 0..N
        # linear_sum_assignment documentation says: "The row indices will be sorted; in the case of a square cost matrix they will be equal to numpy.arange(cost_matrix.shape[0])."
    else:
        # Handle non-square case just in case
        new_cm = np.zeros_like(cm)
        # We can only map min(n_classes, n_clusters)
        # For simplicity, let's just use the pairs found
        for r, c in zip(row_ind, col_ind):
            # Move column c to column r (if possible, but what if r != destination index?)
            # Actually, standardizing on a "Best Match Confusion Matrix" usually implies
            # we want the diagonal to be the matches.
            pass
        # Fallback to just returning original if sizes differ too much or logic is complex
        # But for our specific requirement "best class assignment to clusters", usually K=N_classes.
        if n_classes <= n_clusters:
             # We have more clusters than classes, or equal.
             # We map each class to a unique cluster.
             # New matrix should be N_classes x N_classes (showing only the mapped clusters?)
             # Or N_classes x N_clusters (just reordered columns?)
             # Let's reorder columns so the "diagonal" (or as close as possible) is maximized.
             
             # Create a permutation array for columns
             perm = np.zeros(n_clusters, dtype=int)
             used_cols = set(col_ind)
             unused_cols = [c for c in range(n_clusters) if c not in used_cols]
             
             # Map row_ind (class) to col_ind (cluster)
             # We want column `r` of new matrix to be column `c` of old matrix
             for r, c in zip(row_ind, col_ind):
                 perm[r] = c
                 
             # Fill the rest
             curr = len(row_ind)
             for c in unused_cols:
                 if curr < n_clusters:
                     perm[curr] = c
                     curr += 1
            
             new_cm = cm[:, perm]
             
             # But wait, if we have K=5 and N=3.
             # Classes 0,1,2 map to clusters 4,0,1.
             # We want new col 0 to be cluster 4.
             # new col 1 to be cluster 0.
             # new col 2 to be cluster 1.
             # Co that new_cm[0,0] is T0,P4.
             # new_cm[1,1] is T1,P0.
             # new_cm[2,2] is T2,P1.
             # This aligns the main matches to diagonal.
        else:
             # More classes than clusters.
             # We map each cluster to a unique class? 
             # LSA on -cm gives assignment.
             # row_ind (classes) -> col_ind (clusters).
             # Not all classes will be assigned.
             # Just return original for safety if dimensions don't match perfectly or logic is fuzzy.
             return cm, mapping

    return new_cm, mapping

    return new_cm, mapping


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] | None = None,
    output_path: Path | str = "confusion_matrix.png",
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save confusion matrix as an image.
    
    Args:
        cm: Confusion matrix (rows=true, cols=pred)
        class_names: List of class names
        output_path: Path to save the image
        title: Plot title
    """
    try:
        import seaborn as sns
    except ImportError:
        sns = None
    
    plt.figure(figsize=(10, 8))
    
    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]
        
    if sns is not None:
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names
        )
    else:
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Annotation
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
    
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label (Aligned to Truth)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

def save_clustering_metrics(
    latents: np.ndarray,
    phase_labels: np.ndarray,
    clustering_results: Dict[str, Any],
    output_path: Path,
    max_samples_silhouette: int = 5000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Compute and save clustering metrics to JSON.
    
    Args:
        latents: Latent representations (N, D)
        phase_labels: Ground truth phase labels (N,)
        clustering_results: Results from perform_clustering()
        output_path: Path to save JSON metrics
        max_samples_silhouette: Max samples for silhouette score (for speed)
        random_state: Random seed for subsampling
        
    Returns:
        Dictionary with all computed metrics
    """
    print(f"Computing clustering metrics (silhouette on max {max_samples_silhouette} samples)...")
    metrics: Dict[str, Any] = {
        "per_k_metrics": {},
        "optimal_k": None,
        "best_silhouette": -1.0,
        "n_samples_total": len(latents),
        "n_samples_silhouette": min(max_samples_silhouette, len(latents)),
    }
    
    # Subsample for silhouette score computation (O(n^2) complexity)
    n_samples = len(latents)
    if n_samples > max_samples_silhouette:
        rng = np.random.default_rng(random_state)
        sil_idx = rng.choice(n_samples, size=max_samples_silhouette, replace=False)
        latents_sil = latents[sil_idx]
    else:
        sil_idx = np.arange(n_samples)
        latents_sil = latents
    
    kmeans_results = clustering_results.get("kmeans", {})
    
    unique_phases = np.unique(phase_labels)
    n_phases = len(unique_phases)
    
    for k, labels in kmeans_results.items():
        k_str = str(k)
        
        # Silhouette score on subsample (only valid for k >= 2)
        labels_sil = labels[sil_idx]
        if k >= 2 and len(np.unique(labels_sil)) >= 2:
            sil = float(silhouette_score(latents_sil, labels_sil))
        else:
            sil = 0.0
        
        # ARI and NMI vs ground truth (fast, use all samples)
        ari = float(adjusted_rand_score(phase_labels, labels))
        nmi = float(normalized_mutual_info_score(phase_labels, labels))
        
        metrics["per_k_metrics"][k_str] = {
            "silhouette": round(sil, 4),
            "ari": round(ari, 4),
            "nmi": round(nmi, 4),
            "n_clusters": int(len(np.unique(labels))),
        }
        
        print(f"  k={k}: silhouette={sil:.4f}, ARI={ari:.4f}, NMI={nmi:.4f}")
        
        if k == n_phases:
            print(f"  -> Matching n_clusters ({k}) with n_phases ({n_phases}): Computing Confusion Matrix")
            try:
                cm, mapping = compute_best_mapping_confusion_matrix(phase_labels, labels)
                print("  Aligned Confusion Matrix (Rows=True, Cols=Pred(Aligned)):")
                print(cm)
                
                # Calculate accuracy from aligned diagonal
                total = np.sum(cm)
                diag = np.trace(cm)
                acc = diag / total if total > 0 else 0
                print(f"  Aligned Accuracy: {acc:.4f}")
                
                metrics["per_k_metrics"][k_str]["confusion_matrix"] = cm.tolist()
                metrics["per_k_metrics"][k_str]["aligned_accuracy"] = round(acc, 4)
                
                # Plot and save confusion matrix
                plot_filename = output_path.parent / f"confusion_matrix_k{k}.png"
                plot_confusion_matrix(
                    cm, 
                    class_names=[f"Phase {i}" for i in range(n_phases)], 
                    output_path=plot_filename,
                    title=f"Aligned Confusion Matrix (k={k}, Acc={acc:.2%})"
                )
                print(f"  Saved confusion matrix plot to {plot_filename}")
                
            except Exception as e:
                print(f"  Error computing/plotting confusion matrix: {e}")
                import traceback
                traceback.print_exc()
        
        if sil > metrics["best_silhouette"]:
            metrics["best_silhouette"] = round(sil, 4)
            metrics["optimal_k"] = k




    
    # Add DBSCAN metrics if available
    if clustering_results.get("dbscan") is not None:
        dbscan_labels = clustering_results["dbscan"]
        dbscan_labels_sil = dbscan_labels[sil_idx]
        valid_mask = dbscan_labels_sil >= 0
        if valid_mask.sum() > 1 and len(np.unique(dbscan_labels_sil[valid_mask])) >= 2:
            sil_db = float(silhouette_score(latents_sil[valid_mask], dbscan_labels_sil[valid_mask]))
        else:
            sil_db = 0.0
        ari_db = float(adjusted_rand_score(phase_labels, dbscan_labels))
        nmi_db = float(normalized_mutual_info_score(phase_labels, dbscan_labels))
        
        metrics["dbscan"] = {
            "silhouette": round(sil_db, 4),
            "ari": round(ari_db, 4),
            "nmi": round(nmi_db, 4),
            "n_clusters": int(len(np.unique(dbscan_labels[dbscan_labels >= 0]))),
            "noise_points": int((dbscan_labels == -1).sum()),
        }
        print(f"  DBSCAN: silhouette={sil_db:.4f}, ARI={ari_db:.4f}, NMI={nmi_db:.4f}")
    
    # Add HDBSCAN metrics if available
    if clustering_results.get("hdbscan") is not None:
        hdbscan_labels = clustering_results["hdbscan"]
        hdbscan_labels_sil = hdbscan_labels[sil_idx]
        valid_mask = hdbscan_labels_sil >= 0
        if valid_mask.sum() > 1 and len(np.unique(hdbscan_labels_sil[valid_mask])) >= 2:
            sil_hdb = float(silhouette_score(latents_sil[valid_mask], hdbscan_labels_sil[valid_mask]))
        else:
            sil_hdb = 0.0
        ari_hdb = float(adjusted_rand_score(phase_labels, hdbscan_labels))
        nmi_hdb = float(normalized_mutual_info_score(phase_labels, hdbscan_labels))
        
        metrics["hdbscan"] = {
            "silhouette": round(sil_hdb, 4),
            "ari": round(ari_hdb, 4),
            "nmi": round(nmi_hdb, 4),
            "n_clusters": int(len(np.unique(hdbscan_labels[hdbscan_labels >= 0]))),
            "noise_points": int((hdbscan_labels == -1).sum()),
        }
        print(f"  HDBSCAN: silhouette={sil_hdb:.4f}, ARI={ari_hdb:.4f}, NMI={nmi_hdb:.4f}")
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Optimal k by silhouette: {metrics['optimal_k']} (score={metrics['best_silhouette']:.4f})")
    print(f"Saved metrics to {output_path}")
    
    return metrics


def run_post_training_analysis(
    checkpoint_path: str,
    output_dir: str,
    cuda_device: int = 0,
    max_samples: Optional[int] = 5000,
    k_range: range = range(2, 7),
    run_dbscan: bool = False,
    run_hdbscan: bool = False,
    force_recompute: bool = True,
    cfg: DictConfig | None = None,
) -> None:
    """Run full post-training analysis pipeline.
    
    This function is designed to be called from train_spd.py after training completes.
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        output_dir: Base directory to save analysis outputs (checkpoint subfolder will be created)
        cuda_device: GPU device to use
        max_samples: Maximum number of samples to analyze
        k_range: Range of k values for KMeans clustering
        run_dbscan: Whether to run DBSCAN
        run_hdbscan: Whether to run HDBSCAN
        force_recompute: Whether to force recomputation (ignore cache)
        cfg: Optional Hydra configuration (to avoid reloading/re-initializing)
    """
    print("\n" + "=" * 60)
    print("POST-TRAINING ANALYSIS")
    print("=" * 60)
    
    # Create checkpoint-specific output folder
    checkpoint_name = Path(checkpoint_path).stem
    output_path = Path(output_dir) / checkpoint_name
    output_path.mkdir(parents=True, exist_ok=True)
    cache_dir = str(output_path / "predictions_cache")
    
    print(f"Output folder: {output_path}")
    
    # Step 1: Extract latents
    print("\n=== Step 1: Extracting latents ===")
    latents, coords, phases, grains, model, cfg, atoms, metadata = predict_and_cache(
        checkpoint_path, 
        cuda_device=cuda_device, 
        cache_dir=cache_dir, 
        max_samples=max_samples,
        force_recompute=force_recompute,
        cfg=cfg,
    )
    print(f"Shapes - Latents: {latents.shape}, Coords: {coords.shape}, Atoms: {atoms.shape}")
    
    # Step 2: Clustering with multiple k values
    print("\n=== Step 2: Clustering (k in {})===".format(list(k_range)))
    n_phases = len(np.unique(phases))
    clustering = perform_clustering(
        latents,
        n_phases=n_phases,
        run_dbscan=run_dbscan,
        run_hdbscan=run_hdbscan,
        k_range=k_range,
    )
    
    # Step 3: Save clustering metrics
    print("\n=== Step 3: Clustering metrics ===")
    metrics = save_clustering_metrics(
        latents, phases, clustering, 
        output_path / "clustering_metrics.json"
    )
    
    # Step 4: Generate visualizations for each k value
    print("\n=== Step 4: Clustering visualizations ===")
    for k, kmeans_labels in clustering["kmeans"].items():
        viz_filename = f"clustering_visualization_k{k}.png"
        create_visualization(
            atoms, metadata, coords, phases, grains,
            kmeans_labels,
            cfg.data["global"].L,
            output_path / viz_filename,
            dbscan_labels=clustering["dbscan"],
            hdbscan_labels=clustering["hdbscan"],
        )
        print(f"Saved: {output_path / viz_filename}")
    
    # Step 5: Dataset samples visualization with edges
    print("\n=== Step 5: Dataset samples visualization ===")
    env_dirs = get_env_dirs(cfg)
    dataset_viz, _ = create_synthetic_dataset_with_coords(env_dirs, cfg, min(max_samples or 5000, 5000))
    print(f"Created dataset with {len(dataset_viz)} samples")
    
    visualize_dataset_samples(
        checkpoint_path,
        output_path / "dataset_samples_analysis.png",
        dataset=dataset_viz,
        cuda_device=cuda_device,
        edge_type='knn',
    )
    print(f"Saved: {output_path / 'dataset_samples_analysis.png'}")
    
    # Step 6: Latent space visualization
    print("\n=== Step 6: Latent space visualization ===")
    visualize_latent_space(
        latents=latents,
        phase_labels=phases,
        output_path=output_path / "latent_space_visualization.png",
        tsne_max_samples=min(20000, len(latents)),
    )
    print(f"Saved: {output_path / 'latent_space_visualization.png'}")
    
    # Step 7: RDF and ADF comparison per phase
    print("\n=== Step 7: RDF and ADF analysis ===")
    compute_rdf_adf_comparison(
        model=model,
        dataset=dataset_viz,
        device=f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu",
        output_path=output_path / "rdf_adf_analysis.png",
        num_samples_per_phase=50,
    )
    
    print("\n" + "=" * 60)
    print(f"ANALYSIS COMPLETE - All outputs saved to: {output_path}")
    print("=" * 60)


def main():
    """Main entry point for standalone analysis."""
    # Configuration
    checkpoint_path = "output/2025-11-27/18-28-09/synth_SPD_VN_Equivariant_l513_P80_chamfer+sinkhorn_512-epoch=15.ckpt"
    output_dir = "output/spd_analysis"
    max_samples = None
    k_range = range(2, 7)  # k from 2 to 6
    run_dbscan = False
    run_hdbscan = False
    
    run_post_training_analysis(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        cuda_device=0,
        max_samples=max_samples,
        k_range=k_range,
        run_dbscan=run_dbscan,
        run_hdbscan=run_hdbscan,
        force_recompute=False,
    )


if __name__ == "__main__":
    main()
