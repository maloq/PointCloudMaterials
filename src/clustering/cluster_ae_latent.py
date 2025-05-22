import sys
import os
import torch
import numpy as np
from typing import List
sys.path.append(os.getcwd())

from hydra import compose, initialize
from omegaconf import DictConfig
from src.autoencoder.autoencoder_module import PointNetAutoencoder
from src.models.autoencoders_nn.pointnet_autoencoder import PointNetVAEBase
from src.autoencoder.eval_autoencoder import create_autoencoder_dataloader

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def get_latents_from_dataloader(model, dataloader, device: str = 'cpu') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract latent representations from dataloader batches."""
    if len(dataloader) == 0:
        raise ValueError("Dataloader is empty - no data to process")
        
    latents, point_clouds, originals = [], [], []

    for batch in dataloader:
        points = batch[0].to(device).transpose(2, 1)
        
        with torch.no_grad():
            if isinstance(model, PointNetAutoencoder):
                point_cloud, latent, _ = model(points)
            elif isinstance(model, PointNetVAEBase):
                point_cloud, latent, _, _ = model(points)
            else:
                raise ValueError(f"Unknown model class: {type(model)}")
        
        latents.append(latent.cpu().numpy())
        point_clouds.append(point_cloud.cpu().numpy())
        originals.append(points.cpu().numpy())

    return (np.concatenate(latents, axis=0), 
            np.concatenate(point_clouds, axis=0), 
            np.concatenate(originals, axis=0))


def _process_files(cfg, model, file_paths: List[str], label: str, device: str, max_samples: int = None):
    """Helper function to process a list of files and extract latents."""
    if not file_paths:
        raise ValueError(f"No {label} file paths provided")
        
    print(f"Processing {label} datasets...")
    all_latents, all_point_clouds, all_originals = [], [], []
    
    for i, file_path in enumerate(file_paths):
        print(f"Processing {label} file {i+1}/{len(file_paths)}: {file_path}")
        dataloader = create_autoencoder_dataloader(cfg, file_path, shuffle=True, max_samples=max_samples)
        latents, point_clouds, originals = get_latents_from_dataloader(model, dataloader, device)
        
        all_latents.append(latents)
        all_point_clouds.append(point_clouds)
        all_originals.append(originals)
        print(f"  {len(latents)} samples from {file_path}")
    
    if not all_latents:
        raise ValueError(f"No data was successfully loaded from any {label} files")
        
    return (np.concatenate(all_latents, axis=0),
            np.concatenate(all_point_clouds, axis=0), 
            np.concatenate(all_originals, axis=0))


def predict_and_save_latent(cfg: str,
                            model,
                            liquid_file_paths: List[str],
                            crystal_file_paths: List[str],
                            device: str = 'cpu',
                            save_folder: str = 'output',
                            model_class: str = None,
                            max_samples: int = None):

    if not liquid_file_paths and not crystal_file_paths:
        raise ValueError("At least one of liquid_file_paths or crystal_file_paths must be provided")

    model.to(device).eval()
    
    # Split max_samples if specified
    if max_samples:
        max_samples_per_type = max_samples // 2
        max_samples_l = max_samples_per_type
        max_samples_c = max_samples - max_samples_per_type
    else:
        max_samples_l = max_samples_c = None

    # Process files - these will raise errors if no data is found
    latents_l, point_clouds_l, originals_l = _process_files(cfg, model, liquid_file_paths, "liquid", device, max_samples_l)
    latents_c, point_clouds_c, originals_c = _process_files(cfg, model, crystal_file_paths, "crystal", device, max_samples_c)
    
    # Create labels and combine data
    labels = np.array(["liquid"] * len(latents_l) + ["crystal"] * len(latents_c))
    all_latents = np.concatenate((latents_l, latents_c), axis=0)
    all_points = np.concatenate((point_clouds_l, point_clouds_c), axis=0)
    all_originals = np.concatenate((originals_l, originals_c), axis=0)

    print(f"Total samples: {len(all_latents)} (liquid: {len(latents_l)}, crystal: {len(latents_c)})")

    # Save combined data
    output_path = os.path.join(save_folder, "latent_data.npz")
    os.makedirs(save_folder, exist_ok=True)
    
    np.savez_compressed(output_path, 
                        latents=all_latents, 
                        points=all_points, 
                        originals=all_originals, 
                        labels=labels)

    print(f"Saved combined data to {output_path}")


def visualize_latents(npy_file="latent_label_pairs.npy",
                      perplexity=30, n_iter=1000,
                      random_state=42,
                      output_image="latent_tsne.png"):
    """
    Loads latent representations and their labels from a numpy file,
    computes a 2D t-SNE projection, and visualizes them with distinct colors.

    Args:
        npy_file (str): Path to the numpy file containing latent-label pairs.
        perplexity (float): The perplexity parameter for t-SNE.
        n_iter (int): Number of iterations for t-SNE optimization.
        random_state (int): Random seed for reproducibility.
        output_image (str): Filename for saving the plot.
    """
    # Load the latent-label pairs.
    print("Loading latent-label pairs from:", npy_file)
    latent_label_pairs = np.load(npy_file, allow_pickle=True)

    # Separate the latent vectors and labels.
    latents = []
    labels = []
    for latent, label in latent_label_pairs:
        latents.append(latent)
        labels.append(label)
    
    latents = np.stack(latents, axis=0)
    labels = np.array(labels)

    # Compute t-SNE projection.
    print("Performing t-SNE on latent representations...")
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=random_state)
    tsne_results = tsne.fit_transform(latents)

    # Determine unique labels and map them to colors.
    unique_labels = sorted(set(labels))
    # Define a set of colors (expandable if needed).
    color_list = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    color_map = {label: color_list[i % len(color_list)] for i, label in enumerate(unique_labels)}

    # Create scatter plot.
    plt.figure(figsize=(8, 6))
    for label in unique_labels:
        indices = np.where(labels == label)
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                    color=color_map[label], label=label,
                    alpha=0.3, edgecolors='w', s=12)

    plt.title("t-SNE Visualization of Latent Space")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print("Saved t-SNE plot to:", output_image)
    plt.show()



def cluster_latents(latents, points, original_points, labels, n_clusters=2, 
                   random_state=42, save_centers=False, label=None, 
                   output_file="output/cluster_centers.npy"):
    """Clusters latent representations and finds samples closest to each cluster center."""
    from sklearn.cluster import KMeans
    
    print(f"K-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(latents)
    
    # Find closest sample to each cluster center
    rep_indices = []
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        distances = np.linalg.norm(latents[cluster_mask] - kmeans.cluster_centers_[i], axis=1)
        rep_indices.append(np.where(cluster_mask)[0][np.argmin(distances)])
        print(f"Cluster {i}: sample {rep_indices[-1]}, label={labels[rep_indices[-1]]}")
    
    # Extract representative data
    rep_samples = latents[rep_indices]
    rep_points = points[rep_indices]
    rep_originals = original_points[rep_indices]
    rep_labels = labels[rep_indices]
    
    if save_centers:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.save(output_file, {
            'cluster_centers': kmeans.cluster_centers_,
            'representative_samples': rep_samples,
            'representative_points': rep_points,
            'representative_labels': rep_labels,
            'cluster_labels': cluster_labels
        })
        print(f"Saved to {output_file}")
    
    return kmeans.cluster_centers_, rep_samples, rep_points, rep_originals, rep_labels


if __name__ == '__main__':

    save_path = 'output/latent_label_pairs.npy'
    checkpoint_path = 'output/2025-02-05/23-02-04/pointnet-epoch=289-val_loss=0.37.ckpt'
    liquid_file_paths = ['datasets/Al/inherent_configurations_off/166ps.off']
    crystal_file_paths = ['datasets/Al/inherent_configurations_off/240ps.off']

    if not os.path.exists(save_path):
        predict_and_save_latent('autoencoder',
                                checkpoint_path,
                                liquid_file_paths,
                                crystal_file_paths,
                                device='cuda')