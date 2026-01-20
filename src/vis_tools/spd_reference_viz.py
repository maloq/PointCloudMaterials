from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay, cKDTree
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.training_methods.spd.eval_spd import load_spd_model
from src.data_utils.data_load import SyntheticPointCloudDataset


# Color palette for phases
PHASE_COLORS = {
    0: "#e41a1c",  # red
    1: "#377eb8",  # blue
    2: "#4daf4a",  # green
    3: "#984ea3",  # purple
    4: "#ff7f00",  # orange
    5: "#ffff33",  # yellow
    6: "#a65628",  # brown
    7: "#f781bf",  # pink
}

PHASE_MARKERS = {
    0: "o",  # circle
    1: "s",  # square
    2: "^",  # triangle up
    3: "D",  # diamond
    4: "v",  # triangle down
    5: "p",  # pentagon
    6: "h",  # hexagon
    7: "*",  # star
}


def extract_actual_dataset_samples(
    dataset: SyntheticPointCloudDataset,
    num_samples_per_class: int = 3,
) -> Dict[str, List[np.ndarray]]:
    """Extract actual samples from the synthetic dataset, organized by class."""
    class_samples: Dict[str, List[np.ndarray]] = {}
    idx_to_class = dataset.class_names  # {id: name}
    class_counts: Dict[str, int] = {name: 0 for name in idx_to_class.values()}

    for i in range(len(dataset)):
        if all(count >= num_samples_per_class for count in class_counts.values()):
            break

        class_idx = dataset._class_ids[i]
        class_name = idx_to_class.get(class_idx, f"unknown_{class_idx}")

        if class_counts[class_name] < num_samples_per_class:
            pc_tensor = dataset.samples[i]
            class_samples.setdefault(class_name, []).append(pc_tensor.numpy())
            class_counts[class_name] += 1

    return class_samples


def draw_edges_on_ax(
    ax,
    coords: np.ndarray,
    edge_type: Literal['delaunay', 'knn', None] = 'delaunay',
    knn_k: int = 4,
    edge_color: str = '#555555',
    edge_alpha: float = 0.3,
    edge_linewidth: float = 0.4,
) -> None:
    """Draw edges on a 3D matplotlib axis.
    
    Args:
        ax: Matplotlib 3D axis
        coords: (N, 3) point coordinates
        edge_type: 'delaunay', 'knn', or None
        knn_k: Number of neighbors for KNN edges
        edge_color: Edge line color
        edge_alpha: Edge transparency
        edge_linewidth: Edge line width
    """
    if edge_type is None or len(coords) < 4:
        return
    
    drawn = set()
    
    if edge_type == 'delaunay':
        try:
            tri = Delaunay(coords)
            for simplex in tri.simplices:
                for i in range(4):
                    for j in range(i + 1, 4):
                        edge = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
                        if edge not in drawn:
                            drawn.add(edge)
                            p1, p2 = coords[edge[0]], coords[edge[1]]
                            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                                   color=edge_color, linewidth=edge_linewidth, alpha=edge_alpha)
        except Exception:
            # Fallback to KNN if Delaunay fails
            edge_type = 'knn'
    
    if edge_type == 'knn':
        tree = cKDTree(coords)
        k = min(knn_k + 1, len(coords))
        
        for i, point in enumerate(coords):
            _, indices = tree.query(point, k=k)
            indices = np.atleast_1d(indices)
            for j in indices:
                if j != i:
                    edge = (min(i, j), max(i, j))
                    if edge not in drawn:
                        drawn.add(edge)
                        p1, p2 = coords[i], coords[j]
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                               color=edge_color, linewidth=edge_linewidth, alpha=edge_alpha)


def visualize_dataset_samples(
    checkpoint_path: str,
    output_path: Path,
    dataset: SyntheticPointCloudDataset,
    cuda_device: int = 0,
    num_samples_per_phase: int = 2,
    edge_type: Optional[Literal['delaunay', 'knn']] = None,
    knn_k: int = 4,
) -> None:
    """
    Create visualization of dataset samples and their reconstructions.
    
    Args:
        checkpoint_path: Path to model checkpoint
        output_path: Path to save visualization
        dataset: Synthetic dataset
        cuda_device: GPU device index
        num_samples_per_phase: Number of samples per phase to visualize
        edge_type: Edge type for visualization ('delaunay', 'knn', or None)
        knn_k: Number of neighbors for KNN edges
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    model, cfg, device = load_spd_model(checkpoint_path, cuda_device=cuda_device)
    model.eval()

    structure_names: List[str] = []
    point_clouds: List[np.ndarray] = []
    phase_indices: List[int] = []

    print("\n=== Extracting Dataset Samples ===")
    dataset_samples = extract_actual_dataset_samples(dataset, num_samples_per_phase=num_samples_per_phase)

    # Get phase name to index mapping
    phase_name_to_idx = {name: idx for idx, name in enumerate(sorted(dataset_samples.keys()))}

    for phase_name, samples_list in sorted(dataset_samples.items()):
        for idx, sample_pc in enumerate(samples_list):
            mean_ds = sample_pc.mean(axis=0)
            max_norm_ds = np.max(np.linalg.norm(sample_pc, axis=1))
            print(f"  {phase_name}_sample{idx}: mean={mean_ds}, max_norm={max_norm_ds:.4f}")

            point_clouds.append(sample_pc)
            structure_names.append(f"{phase_name}_{idx}")
            phase_indices.append(phase_name_to_idx[phase_name])

    print(f"Extracted {len(point_clouds)} dataset samples")

    if len(point_clouds) == 0:
        print("No valid structures to visualize")
        return

    print(f"\nTotal structures to process: {len(structure_names)}")

    reconstructions_list: List[np.ndarray] = []
    canonicals_list: List[np.ndarray] = []
    originals_list: List[np.ndarray] = []
    recon_names: List[str] = []
    recon_phase_indices: List[int] = []

    print("\n=== Running Model Inference ===")
    with torch.no_grad():
        for name, pc, phase_idx in zip(structure_names, point_clouds, phase_indices):
            print(f"  Processing {name}...")

            mean_pre = pc.mean(axis=0)
            max_norm_pre = np.max(np.linalg.norm(pc, axis=1))
            print(f"    Before model: mean={mean_pre}, max_norm={max_norm_pre:.4f}")

            pc_tensor = torch.from_numpy(pc).float().unsqueeze(0).to(device)

            try:
                inv_z, recon, cano, rot, _ = model(pc_tensor)

                recon_np = recon.cpu().numpy()[0]
                cano_np = cano.cpu().numpy()[0]

                mean_recon = recon_np.mean(axis=0)
                max_norm_recon = np.max(np.linalg.norm(recon_np, axis=1))
                print(f"    Reconstruction: mean={mean_recon}, max_norm={max_norm_recon:.4f}")

                reconstructions_list.append(recon_np)
                canonicals_list.append(cano_np)
                originals_list.append(pc)
                recon_names.append(name)
                recon_phase_indices.append(phase_idx)
            except Exception as e:
                print(f"    Warning: Model inference failed for {name}: {e}")
                import traceback

                traceback.print_exc()
                continue

    print(f"\nSuccessfully processed {len(recon_names)} structures")

    n_structures = len(recon_names)
    if n_structures == 0:
        print("No structures to visualize")
        return

    fig = plt.figure(figsize=(4 * n_structures, 12))

    border_width = 72.0 / fig.dpi
    fig.patch.set_facecolor("white")
    fig.patch.set_edgecolor("black")
    fig.patch.set_linewidth(border_width)

    all_points = originals_list + reconstructions_list + canonicals_list
    if all_points:
        max_extent = max(np.max(np.abs(pc)) for pc in all_points)
        viz_limit = max(0.6, float(max_extent) * 1.1)
    else:
        viz_limit = 0.6

    for col, (name, pc, phase_idx) in enumerate(zip(recon_names, originals_list, recon_phase_indices)):
        ax = fig.add_subplot(3, n_structures, col + 1, projection="3d")
        color = PHASE_COLORS.get(phase_idx, "gray")

        # Draw edges first (behind points)
        if edge_type:
            draw_edges_on_ax(ax, pc, edge_type=edge_type, knn_k=knn_k, 
                           edge_color='#333333', edge_alpha=0.25)

        ax.scatter(
            pc[:, 0], pc[:, 1], pc[:, 2],
            s=25, alpha=0.9,
            edgecolors="black",
            linewidths=0.3,
            c=color,
        )
        ax.set_title(f"{name}\n(Original)", fontsize=8)
        ax.set_xlim(-viz_limit, viz_limit)
        ax.set_ylim(-viz_limit, viz_limit)
        ax.set_zlim(-viz_limit, viz_limit)

    for col, (name, cano) in enumerate(zip(recon_names, canonicals_list)):
        ax = fig.add_subplot(3, n_structures, n_structures + col + 1, projection="3d")

        # Draw edges first (behind points)
        if edge_type:
            draw_edges_on_ax(ax, cano, edge_type=edge_type, knn_k=knn_k,
                           edge_color='#663399', edge_alpha=0.25)

        ax.scatter(
            cano[:, 0], cano[:, 1], cano[:, 2],
            s=25, alpha=0.9,
            edgecolors="black",
            linewidths=0.3,
            c="purple",
        )

        ax.set_title(f"{name}\n(Canonical)", fontsize=8)
        ax.set_xlim(-viz_limit, viz_limit)
        ax.set_ylim(-viz_limit, viz_limit)
        ax.set_zlim(-viz_limit, viz_limit)

    for col, (name, recon) in enumerate(zip(recon_names, reconstructions_list)):
        ax = fig.add_subplot(3, n_structures, 2 * n_structures + col + 1, projection="3d")

        # Draw edges first (behind points)
        if edge_type:
            draw_edges_on_ax(ax, recon, edge_type=edge_type, knn_k=knn_k,
                           edge_color='#cc5500', edge_alpha=0.25)

        ax.scatter(
            recon[:, 0], recon[:, 1], recon[:, 2],
            s=25, alpha=0.9,
            edgecolors="black",
            linewidths=0.3,
            c="orange",
        )

        ax.set_title(f"{name}\n(Reconstruction)", fontsize=8)
        ax.set_xlim(-viz_limit, viz_limit)
        ax.set_ylim(-viz_limit, viz_limit)
        ax.set_zlim(-viz_limit, viz_limit)

    fig.suptitle("Dataset Samples Analysis", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nVisualization saved to {output_path}")
    print("Colors: Phase-colored=Original, Purple=Canonical, Orange=Reconstruction")


def visualize_latent_space(
    latents: np.ndarray,
    phase_labels: np.ndarray,
    output_path: Path,
    phase_names: Optional[Dict[int, str]] = None,
    tsne_max_samples: int = 20000,
    random_state: int = 42,
) -> None:
    """
    Create PCA and t-SNE visualizations of the latent space.
    
    Args:
        latents: Latent representations (N, D)
        phase_labels: Phase labels for each sample (N,)
        output_path: Path to save the visualization
        phase_names: Optional mapping from phase index to name
        tsne_max_samples: Maximum samples for t-SNE (default 20k)
        random_state: Random seed for reproducibility
    """
    print(f"\n=== Visualizing Latent Space ===")
    print(f"Total samples: {len(latents)}, Latent dim: {latents.shape[1]}")
    
    unique_phases = np.unique(phase_labels)
    n_phases = len(unique_phases)
    print(f"Found {n_phases} unique phases: {unique_phases}")
    
    if phase_names is None:
        phase_names = {i: f"Phase {i}" for i in unique_phases}
    
    # PCA on full dataset
    print("Computing PCA...")
    pca = PCA(n_components=min(3, latents.shape[1]))
    latents_pca = pca.fit_transform(latents)
    
    if latents_pca.shape[1] < 2:
        latents_pca = np.pad(latents_pca, ((0, 0), (0, 2 - latents_pca.shape[1])), mode="constant")
    
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    
    # t-SNE on subset
    n_samples = len(latents)
    if n_samples > tsne_max_samples:
        print(f"Subsampling to {tsne_max_samples} samples for t-SNE...")
        rng = np.random.default_rng(random_state)
        subset_idx = rng.choice(n_samples, size=tsne_max_samples, replace=False)
        latents_subset = latents[subset_idx]
        phase_labels_subset = phase_labels[subset_idx]
    else:
        subset_idx = np.arange(n_samples)
        latents_subset = latents
        phase_labels_subset = phase_labels
    
    print(f"Computing t-SNE on {len(latents_subset)} samples...")
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=min(30, len(latents_subset) - 1))
    latents_tsne = tsne.fit_transform(latents_subset)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # PCA plot (2D)
    ax_pca = axes[0]
    for phase_idx in unique_phases:
        mask = phase_labels == phase_idx
        color = PHASE_COLORS.get(phase_idx % len(PHASE_COLORS), "gray")
        marker = PHASE_MARKERS.get(phase_idx % len(PHASE_MARKERS), "o")
        label = phase_names.get(phase_idx, f"Phase {phase_idx}")
        
        ax_pca.scatter(
            latents_pca[mask, 0],
            latents_pca[mask, 1],
            c=color,
            marker=marker,
            s=15,
            alpha=0.6,
            label=label,
            edgecolors="none",
        )
    
    ax_pca.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax_pca.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})" if len(pca.explained_variance_ratio_) > 1 else "PC2")
    ax_pca.set_title(f"PCA of Latent Space\n({len(latents)} samples)", fontsize=12, fontweight="bold")
    ax_pca.legend(loc="best", fontsize=9)
    ax_pca.grid(True, alpha=0.3)
    
    # t-SNE plot
    ax_tsne = axes[1]
    for phase_idx in unique_phases:
        mask = phase_labels_subset == phase_idx
        color = PHASE_COLORS.get(phase_idx % len(PHASE_COLORS), "gray")
        marker = PHASE_MARKERS.get(phase_idx % len(PHASE_MARKERS), "o")
        label = phase_names.get(phase_idx, f"Phase {phase_idx}")
        
        ax_tsne.scatter(
            latents_tsne[mask, 0],
            latents_tsne[mask, 1],
            c=color,
            marker=marker,
            s=15,
            alpha=0.6,
            label=label,
            edgecolors="none",
        )
    
    ax_tsne.set_xlabel("t-SNE 1")
    ax_tsne.set_ylabel("t-SNE 2")
    ax_tsne.set_title(f"t-SNE of Latent Space\n({len(latents_subset)} samples)", fontsize=12, fontweight="bold")
    ax_tsne.legend(loc="best", fontsize=9)
    ax_tsne.grid(True, alpha=0.3)
    
    fig.suptitle("Latent Space Visualization", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Latent space visualization saved to {output_path}")


__all__ = ["visualize_dataset_samples", "visualize_latent_space"]
