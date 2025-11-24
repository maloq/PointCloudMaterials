"""Prediction and visualization pipeline for SPD models with synthetic data."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hdbscan
import numpy as np
import torch
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.data_utils.data_load import SyntheticPointCloudDataset
from src.data_utils.prepare_data import get_regular_samples
from src.training_methods.spd.eval_spd import load_spd_model
from src.vis_tools.spd_clustering_viz import create_visualization
from src.vis_tools.spd_reference_viz import visualize_dataset_samples, visualize_latent_space


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any, Any, np.ndarray, Dict]:
    """Load model, extract predictions from synthetic data, and cache results."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / f"{Path(checkpoint_path).stem}_synth_predictions.npz"

    print(f"Loading model from {checkpoint_path}")
    model, cfg, device = load_spd_model(checkpoint_path, cuda_device=cuda_device)
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
    run_hdbscan: bool = True,
    dbscan_eps: Optional[float] = None,
    dbscan_max_samples: Optional[int] = 20000,
    random_state: int = 0,
) -> Dict[str, Optional[np.ndarray]]:
    """Perform clustering using KMeans, DBSCAN, and optionally HDBSCAN."""
    print(f"Clustering {len(latents)} samples...")
    results = {}

    # KMeans
    print(f"  KMeans (k={n_phases})...")
    results["kmeans"] = KMeans(n_clusters=n_phases, n_init=10, random_state=0).fit_predict(latents)

    # DBSCAN
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

    # HDBSCAN
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


def main():
    # Configuration
    checkpoint_path = "output/2025-11-24/19-57-52/synth_SPD_VQMoE_l240_P280_sinkhorn+chamfer_256-epoch=59.ckpt"
    n_phases = 3
    output_dir = "output/spd_analysis"
    cache_dir = "output/spd_analysis/predictions_cache"
    max_samples = None
    run_hdbscan = False
    dbscan_max_samples = 100000  # Prevent DBSCAN from loading all samples into memory
    max_samples_for_visualization = 5000

    # Extract latents
    print("\n=== Step 1: Extracting latents ===")
    latents, coords, phases, grains, model, cfg, atoms, metadata = predict_and_cache(
        checkpoint_path, cuda_device=0, cache_dir=cache_dir, max_samples=max_samples
    )
    print(f"Shapes - Latents: {latents.shape}, Coords: {coords.shape}, Atoms: {atoms.shape}")

    # Clustering
    print("\n=== Step 2: Clustering ===")
    clustering = perform_clustering(
        latents,
        n_phases,
        run_hdbscan=run_hdbscan,
        dbscan_max_samples=dbscan_max_samples,
    )

    # Visualization
    print("\n=== Step 3: Visualization ===")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    create_visualization(
        atoms, metadata, coords, phases, grains,
        clustering["kmeans"],
        metadata.get("box_size", 160.0),
        output_path / "clustering_visualization.png",
        dbscan_labels=clustering["dbscan"],
        hdbscan_labels=clustering["hdbscan"],
    )
    print(f"Saved: {output_path / 'clustering_visualization.png'}")

    # Dataset samples visualization
    print("\n=== Step 4: Dataset samples visualization ===")
    print(f"Creating dataset for visualization ({max_samples_for_visualization} samples)...")
    dataset_viz, _ = create_synthetic_dataset_with_coords(
        get_env_dirs(cfg), cfg, max_samples_for_visualization
    )
    print(f"Created dataset with {len(dataset_viz)} samples")

    visualize_dataset_samples(
        checkpoint_path,
        output_path / "dataset_samples_analysis.png",
        dataset=dataset_viz,
        cuda_device=0,
    )
    print(f"Saved: {output_path / 'dataset_samples_analysis.png'}")

    # Latent space visualization (PCA + t-SNE)
    print("\n=== Step 5: Latent space visualization ===")
    visualize_latent_space(
        latents=latents,
        phase_labels=phases,
        output_path=output_path / "latent_space_visualization.png",
        tsne_max_samples=20000,
    )
    print(f"Saved: {output_path / 'latent_space_visualization.png'}")

    print(f"\n=== Done! All outputs in {output_path} ===")


if __name__ == "__main__":
    main()
