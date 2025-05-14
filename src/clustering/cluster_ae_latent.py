import sys
import os
import torch
import numpy as np
sys.path.append(os.getcwd())

from hydra import compose, initialize
from omegaconf import DictConfig
from src.autoencoder.autoencoder_module import PointNetAutoencoder
from src.autoencoder.eval_autoencoder import create_autoencoder_dataloader

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def get_latents_from_dataloader(model, dataloader, device: str = 'cpu') -> np.ndarray:
    """
    Runs the autoencoder on each batch and extracts the latent representations.
    
    Args:
        model: The loaded autoencoder model (PointNetAutoencoder or AutoencoderSeq2Seq).
        dataloader: Torch DataLoader yielding point cloud batches.
        device: Device to run the inference on.
    
    Returns:
        A numpy array of shape (num_samples, latent_dim) with latent codes.
    """
    latents_list = []
    point_clouds_list = []
    original_points_list = []

    for batch in dataloader:
        points = batch[0]
        points = points.to(device)
        
        # Check if the model is a PointNetAutoencoder or AutoencoderSeq2Seq
        if isinstance(model, PointNetAutoencoder):
            points = points.transpose(2, 1)
        else:
            raise ValueError(f"Unknown model class: {type(model)}")
        
        with torch.no_grad():
            print(points.shape)
            point_cloud, latent, _ = model(points)

        latents_list.append(latent.cpu().numpy())
        point_clouds_list.append(point_cloud.cpu().numpy())
        original_points_list.append(points.cpu().numpy())

    latents = np.concatenate(latents_list, axis=0)
    point_clouds = np.concatenate(point_clouds_list, axis=0)
    original_points = np.concatenate(original_points_list, axis=0)
    
    return latents, point_clouds, original_points


def predict_and_save_latent(cfg: str,
                            model,
                            liquid_file_path: str,
                            crystal_file_path: str,
                            device: str = 'cpu',
                            save_folder: str = 'output',
                            model_class: str = None,
                            max_samples: int = None):

    # Determine which model class to use based on the checkpoint
    if max_samples:
        # Ensure balanced samples if max_samples is set
        max_samples_c = max_samples // 2
        max_samples_l = max_samples - max_samples_c
    else:
        # Handle case where max_samples is None
        max_samples_l = None
        max_samples_c = None
        

    print("Loaded PointNetAutoencoder model")
    liquid_loader = create_autoencoder_dataloader(cfg, liquid_file_path, shuffle=True, max_samples=max_samples_l)
    crystal_loader = create_autoencoder_dataloader(cfg, crystal_file_path, shuffle=True, max_samples=max_samples_c)

    model.to(device)
    model.eval()
    
    print("Processing liquid dataset ...")
    latents_l, point_clouds_l, original_points_l = get_latents_from_dataloader(model, liquid_loader, device)
    labels_l = ["liquid"] * len(latents_l) # Create list of labels
    print(f"{len(latents_l)} liquid latents") # Corrected print statement

    print("Processing crystal dataset ...")
    latents_c, point_clouds_c, original_points_c = get_latents_from_dataloader(model, crystal_loader, device)
    labels_c = ["crystal"] * len(latents_c) # Corrected label calculation
    print(f"{len(latents_c)} crystal latents")

    # Concatenate numpy arrays correctly
    all_latents = np.concatenate((latents_l, latents_c), axis=0)
    all_points = np.concatenate((point_clouds_l, point_clouds_c), axis=0)
    all_originals = np.concatenate((original_points_l, original_points_c), axis=0)
    all_labels = np.array(labels_l + labels_c) # Combine lists and convert to numpy array

    print(f"Created combined arrays. Total samples: {len(all_latents)}")

    # Define the path for the single output file
    output_path = os.path.join(save_folder, "latent_data.npz")
    os.makedirs(save_folder, exist_ok=True) # Ensure save folder exists

    # Save all arrays into a single compressed .npz file
    np.savez_compressed(output_path, 
                          latents=all_latents, 
                          points=all_points, 
                          originals=all_originals, 
                          labels=all_labels)

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


def cluster_latents(npy_file="output/latent_label_pairs.npy",
                   n_clusters=2,
                   random_state=42,
                   save_centers=False,
                   label=None,
                   output_file="output/cluster_centers.npy"):
    """
    Performs clustering on latent representations and finds samples closest to each cluster center.
    
    Args:
        npy_file (str): Path to the numpy file containing latent-label pairs.
        n_clusters (int): Number of clusters to create.
        random_state (int): Random seed for reproducibility.
        save_centers (bool): Whether to save the cluster centers and their closest samples.
        output_file (str): Path to save the cluster centers and representative samples.
        
    Returns:
        tuple: (cluster_centers, representative_samples, representative_points, representative_labels)
        - cluster_centers: Array of cluster centers
        - representative_samples: Latent vectors closest to each center
        - representative_points: Point clouds closest to each center
        - representative_labels: Labels of the representative samples
    """
    from sklearn.cluster import KMeans
    
    # Load the latent-label pairs
    print(f"Loading latent-label pairs from {npy_file}...")
    latent_label_pairs = np.load(npy_file, allow_pickle=True)
    
    if label:
        latent_label_pairs = latent_label_pairs[latent_label_pairs[:, 3] == label]
    print(f"Found {len(latent_label_pairs)} latent-label pairs for label {label}")

    latents = []
    points = []
    labels = []
    for latent, point, original_point, label in latent_label_pairs:
        latents.append(latent)
        points.append(point)
        labels.append(label)
    
    latents = np.stack(latents, axis=0)
    points = np.stack(points, axis=0)
    labels = np.array(labels)
    
    # Perform K-means clustering
    print(f"Performing K-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(latents)
    cluster_centers = kmeans.cluster_centers_
    
    # Find the sample closest to each cluster center
    representative_indices = []
    for i in range(n_clusters):
        # Get indices of points in this cluster
        cluster_indices = np.where(cluster_labels == i)[0]
        
        # Calculate distance from each point in cluster to the center
        cluster_points = latents[cluster_indices]
        distances = np.linalg.norm(cluster_points - cluster_centers[i], axis=1)
        
        # Find the point with minimum distance to center
        min_distance_idx = np.argmin(distances)
        representative_idx = cluster_indices[min_distance_idx]
        representative_indices.append(representative_idx)
        
        print(f"Cluster {i}: Found representative sample with index {representative_idx}, "
              f"label={labels[representative_idx]}, "
              f"distance={distances[min_distance_idx]:.4f}")
    
    # Extract the representative samples
    representative_samples = latents[representative_indices]
    representative_points = points[representative_indices]
    representative_labels = labels[representative_indices]
    
    if save_centers:
        # Save the cluster centers and representative samples
        output_data = {
            'cluster_centers': cluster_centers,
            'representative_samples': representative_samples,
            'representative_points': representative_points,
            'representative_labels': representative_labels,
            'cluster_labels': cluster_labels  # Also save the cluster assignments
        }
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.save(output_file, output_data)
        print(f"Saved cluster centers and representative samples to {output_file}")
    
    return cluster_centers, representative_samples, representative_points, representative_labels


if __name__ == '__main__':

    save_path = 'output/latent_label_pairs.npy'
    checkpoint_path = 'output/2025-02-05/23-02-04/pointnet-epoch=289-val_loss=0.37.ckpt'
    liquid_file_path = 'datasets/Al/inherent_configurations_off/166ps.off'
    crystal_file_path = 'datasets/Al/inherent_configurations_off/240ps.off'

    if not os.path.exists(save_path):
        predict_and_save_latent('autoencoder',
                                checkpoint_path,
                                liquid_file_path,
                                crystal_file_path,
                                device='cuda')