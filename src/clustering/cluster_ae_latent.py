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


def get_latents_from_dataloader(model: PointNetAutoencoder, dataloader, device: str = 'cpu') -> np.ndarray:
    """
    Runs the autoencoder on each batch and extracts the latent representations.
    
    Args:
        model: The loaded PointNetAutoencoder model.
        dataloader: Torch DataLoader yielding point cloud batches.
        device: Device to run the inference on.
    
    Returns:
        A numpy array of shape (num_samples, latent_dim) with latent codes.
    """
    latents = []
    for batch in dataloader:
        points = batch[0]
        points = points.to(device)
        points = points.transpose(2, 1)
        with torch.no_grad():
            latent = model(points, return_latent=True)[0]
            print(latent.shape)
        latents.append(latent.cpu().numpy())
    latents = np.concatenate(latents, axis=0)
    return latents


def predict_and_save_latent(checkpoint_path,
                            liquid_file_path,
                            crystal_file_path,
                            device: str = 'cpu',
                            save_path: str = 'output/latent_label_pairs.npy'):
    with initialize(version_base=None, config_path="../../configs"):
        cfg: DictConfig = compose(config_name="Al_autoencoder")
    
    model = PointNetAutoencoder.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    
    latent_label_pairs = []
    
    print("Processing liquid dataset ...")
    liquid_loader = create_autoencoder_dataloader(cfg, liquid_file_path, shuffle=False)
    liquid_latents = get_latents_from_dataloader(model, liquid_loader, device)
    for latent_vec in liquid_latents:
        latent_label_pairs.append((latent_vec, "liquid"))
    
    print("Processing crystal dataset ...")
    crystal_loader = create_autoencoder_dataloader(cfg, crystal_file_path, shuffle=False)
    crystal_latents = get_latents_from_dataloader(model, crystal_loader, device)
    for latent_vec in crystal_latents:
        latent_label_pairs.append((latent_vec, "crystal"))
    
    latent_label_pairs_np = np.array(latent_label_pairs, dtype=object)
    print(f"Created latent-label pairs tensor with shape: {latent_label_pairs_np.shape}")
    
    np.save(save_path, latent_label_pairs_np)
    print(f"Saved latent-label pairs to {save_path}.")



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

if __name__ == '__main__':

    save_path = 'output/latent_label_pairs.npy'
    checkpoint_path = 'output/2025-02-05/23-02-04/pointnet-epoch=289-val_loss=0.37.ckpt'
    liquid_file_path = 'datasets/Al/inherent_configurations_off/166ps.off'
    crystal_file_path = 'datasets/Al/inherent_configurations_off/240ps.off'

    if not os.path.exists(save_path):
        predict_and_save_latent(checkpoint_path,
                                liquid_file_path,
                                crystal_file_path,
                                device='cuda')
    
    # visualize_latents(npy_file='output/latent_label_pairs.npy',
    #                   perplexity=30, n_iter=1000,
    #                   random_state=42,
    #                   output_image="latent_tsne.png")