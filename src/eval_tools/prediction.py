import sys,os
import numpy as np
import plotly.graph_objects as go
import torch
sys.path.append(os.getcwd())
from src.data_utils.prepare_data import read_off_file
from sklearn.cluster import KMeans

from torch.utils.data import Dataset, DataLoader
from src.data_utils.data_load import RegularDataset
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset


def create_dataloader(cfg: DictConfig, file_paths: list[str], shuffle: bool = False) -> DataLoader:
    """
    Create a dataloader from a list of file paths.
    """
    datasets = []
    for file_path in file_paths:
        points = read_off_file(file_path)
        dataset = RegularDataset(points,
                                 sample_shape=cfg.data.sample_shape,
                                 size=cfg.data.cube_size if cfg.data.sample_shape == 'cubic' else cfg.data.radius,
                                 n_points=cfg.data.num_points,
                                 overlap_fraction=cfg.data.overlap_fraction)
        datasets.append(dataset)

    dataset = ConcatDataset(datasets)
    print(f"Number of samples in {cfg.data.sample_shape} dataset: {len(dataset)}")
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle)


def predict_phases(model, dataloader: DataLoader, device: str = 'cpu', return_probablitity=False) -> np.ndarray:
    """Make predictions for all cubes in the dataloader.
    
    Args:
        model: Trained PointNetClassifier model
        dataloader: DataLoader containing cube samples
        device: Device to run predictions on ('cpu' or 'cuda')
    Returns:
        Nx4 numpy array where each row contains (x, y, z, prediction)
    """
    predictions = []
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for points, coords in dataloader:
            points = points.transpose(2, 1).to(device)
            pred, _ = model(points)
            if return_probablitity:
                probs = torch.nn.functional.softmax(pred, dim=1)
                # Get probability of first class (index 0)
                class_0_probs = probs[:, 0]
                coords = np.array(coords)
                coords = np.moveaxis(coords, 0, 1)
                for coord, prob in zip(coords, class_0_probs.cpu().numpy()):
                    predictions.append([coord[0], coord[1], coord[2], prob])
            else:
                pred_choice = pred.data.max(1)[1]
                coords = np.array(coords)
                coords = np.moveaxis(coords, 0, 1)
                for coord, label in zip(coords, pred_choice.cpu().numpy()):
                    predictions.append([coord[0], coord[1], coord[2], label.item()])
                
    return np.array(predictions)


def predict_clusters(autoencoder_model,
                     train_dataloader: DataLoader,
                     eval_dataloader: DataLoader,
                     n_clusters: int,
                     device: str = 'cpu',
                     kmeans_random_state: int = 42) -> np.ndarray:
    """
    Fits KMeans clustering on latent vectors from train_dataloader and predicts
    cluster assignments for samples from eval_dataloader.
    
    Args:
        autoencoder_model: Trained autoencoder model. Its forward pass should return
                           latent vectors (e.g., as the second element if a tuple is returned,
                           or directly if only latents are returned).
        train_dataloader: DataLoader for training samples used to fit KMeans clustering.
        eval_dataloader: DataLoader for evaluation samples to predict cluster assignments.
                        Both dataloaders yield (points, coords) for each sample.
                        'points' are expected to be (batch_size, num_points, features),
                        and 'coords' are expected to be (batch_size, 3).
        n_clusters: The number of clusters to form using KMeans.
        device: Device to run autoencoder inference on ('cpu' or 'cuda').
        kmeans_random_state: Random state for KMeans for reproducibility.

    Returns:
        Nx4 numpy array where each row contains (x, y, z, cluster_id) for eval samples.
    """
    
    def extract_latents_and_coords(dataloader):
        """Helper function to extract latent vectors and coordinates from a dataloader."""
        latents_list = []
        coords_list = []
        
        with torch.no_grad():
            for points_batch, coords_batch in dataloader:
                points_batch = points_batch.to(device)
                
                # Transpose points for PointNet-like models: (B, N, Features) -> (B, Features, N)
                if points_batch.dim() == 3:
                     points_batch = points_batch.transpose(1, 2)
                
                model_outputs = autoencoder_model(points_batch)
                
                # Extract latent_batch
                if isinstance(model_outputs, tuple):
                    latent_batch = model_outputs[1]
                else:
                    latent_batch = model_outputs

                latents_list.append(latent_batch.cpu().numpy())
                coords_list.append(np.array(coords_batch))
        
        return latents_list, coords_list
    
    autoencoder_model.eval()
    autoencoder_model.to(device)
    
    # Extract latent vectors from train dataloader
    print("Extracting latent vectors from train dataloader...")
    train_latents_list, _ = extract_latents_and_coords(train_dataloader)
    
    if not train_latents_list:
        return np.empty((0, 4))
    
    # Concatenate all train latent vectors
    train_latents = np.concatenate(train_latents_list, axis=0)
    
    # Fit KMeans on train latent vectors
    print(f"Fitting KMeans with {n_clusters} clusters on {train_latents.shape[0]} train samples...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=kmeans_random_state, n_init=10)
    kmeans.fit(train_latents)
    
    # Extract latent vectors from eval dataloader
    print("Extracting latent vectors from eval dataloader...")
    eval_latents_list, eval_coords_list = extract_latents_and_coords(eval_dataloader)
    
    if not eval_latents_list:
        return np.empty((0, 4))
    
    # Concatenate all eval latent vectors and coordinates
    eval_latents = np.concatenate(eval_latents_list, axis=0)
    eval_coords_temp = np.concatenate(eval_coords_list, axis=1)
    eval_coords = eval_coords_temp.T  # Transpose to get (total_samples, 3)
    
    # Predict cluster assignments for eval samples
    print(f"Predicting cluster assignments for {eval_latents.shape[0]} eval samples...")
    cluster_labels = kmeans.predict(eval_latents)
    
    # Combine coordinates and cluster labels for eval samples
    output_data = []
    for i in range(eval_coords.shape[0]):
        coord = eval_coords[i]  # [x, y, z]
        label = cluster_labels[i]  # cluster_id (integer)
        output_data.append([coord[0], coord[1], coord[2], label])

    return np.array(output_data)



