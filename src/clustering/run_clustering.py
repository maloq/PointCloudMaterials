import sys
import os
import numpy as np
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix, silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from itertools import product
from typing import List

import random
import torch
import warnings
warnings.filterwarnings("ignore")
print(f'Running from {os.getcwd()}')



def find_optimal_clusters(data, range_n_clusters=range(2, 11), random_state=66):
    """
    Use silhouette score to find the optimal number of clusters
    """
    silhouette_scores = []
    
    for n_clusters in range_n_clusters:
        # Initialize the clusterer
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        cluster_labels = kmeans.fit_predict(data)
        
        # Calculate silhouette score
        if n_clusters > 1:  # Silhouette requires at least 2 clusters
            silhouette_avg = silhouette_score(data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            print(f"For n_clusters = {n_clusters}, the silhouette score is {silhouette_avg:.4f}")
        else:
            silhouette_scores.append(0)
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(list(range_n_clusters), silhouette_scores, 'o-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.grid(True)
    # plt.savefig('silhouette_scores.png')
    plt.show()
    
    # Return the optimal number of clusters (maximum silhouette score)
    optimal_clusters = list(range_n_clusters)[np.argmax(silhouette_scores)]
    print(f"\nOptimal number of clusters based on silhouette score: {optimal_clusters}")
    
    return optimal_clusters, silhouette_scores


def evaluate_clustering(cluster_labels, true_labels):
    """
    Evaluate clustering performance using multiple metrics
    and find the best cluster-to-class assignment
    """
    # Encode true labels if they're not already numeric
    le = LabelEncoder()
    true_labels_encoded = le.fit_transform(true_labels)
    n_clusters = len(np.unique(cluster_labels))
    
    # Calculate ARI and NMI
    ari = adjusted_rand_score(true_labels_encoded, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels_encoded, cluster_labels)
    
    # Find the best cluster-to-class assignment
    best_accuracy = 0
    best_labels = None
    best_assignment = None
    
    # Try all possible binary assignments for each cluster
    for assignment in product([0, 1], repeat=n_clusters):
        # Skip trivial assignments
        if sum(assignment) == 0 or sum(assignment) == n_clusters:
            continue
            
        # Create mapped labels
        mapped_labels = np.array([assignment[lbl] for lbl in cluster_labels])
        
        # Compute confusion matrix and accuracy
        cm = confusion_matrix(true_labels_encoded, mapped_labels)
        
        # Find optimal assignment
        row_ind, col_ind = linear_sum_assignment(-cm)
        accuracy = cm[row_ind, col_ind].sum() / cm.sum()
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_labels = mapped_labels.copy()
            best_assignment = assignment
    
    # Swap classes if needed for better correspondence
    final_labels = best_labels.copy()
    cm_final = confusion_matrix(true_labels_encoded, final_labels)
    if cm_final[0][1] + cm_final[1][0] > cm_final[0][0] + cm_final[1][1]:
        final_labels = 1 - final_labels
    
    return {
        'accuracy': best_accuracy,
        'ari': ari,
        'nmi': nmi,
        'cluster_mapping': best_assignment,
        'mapped_labels': final_labels
    }

def cluster_and_evaluate(latents, labels, random_state=66):
    """
    Main function to perform clustering evaluation with train/validation split
    """
    # Split data into training and validation sets (50/50 split)
    split_idx = len(latents) // 2
    train_latents = latents[:split_idx]
    train_labels = labels[:split_idx]
    val_latents = latents[split_idx:]
    val_labels = labels[split_idx:]
    
    print("Finding optimal number of clusters using silhouette score...")
    optimal_clusters, silhouette_scores = find_optimal_clusters(train_latents, random_state=random_state)
    
    print(f"\nTraining KMeans with {optimal_clusters} clusters...")
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=random_state)
    kmeans.fit(train_latents)
    
    # Get cluster assignments
    train_cluster_labels = kmeans.predict(train_latents)
    val_cluster_labels = kmeans.predict(val_latents)
    
    # Evaluate on training set
    print("\nEvaluating on training set...")
    train_results = evaluate_clustering(train_cluster_labels, train_labels)
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_results = evaluate_clustering(val_cluster_labels, val_labels)
    
    # Print results
    print("\n===== CLUSTERING RESULTS WITH OPTIMAL CLUSTERS =====")
    print(f"Optimal number of clusters: {optimal_clusters}")
    print(f"Training Accuracy: {train_results['accuracy']:.4f}")
    print(f"Training ARI: {train_results['ari']:.4f}")
    print(f"Training NMI: {train_results['nmi']:.4f}")
    print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
    print(f"Validation ARI: {val_results['ari']:.4f}")
    print(f"Validation NMI: {val_results['nmi']:.4f}")
    
    # Apply the model to the full dataset for the final result
    print("\nApplying to full dataset...")
    full_cluster_labels = kmeans.predict(latents)
    full_results = evaluate_clustering(full_cluster_labels, labels)
    
    print("\n===== FINAL RESULTS ON FULL DATASET =====")
    print(f"Full Dataset Accuracy: {full_results['accuracy']:.4f}")
    print(f"Full Dataset ARI: {full_results['ari']:.4f}")
    print(f"Full Dataset NMI: {full_results['nmi']:.4f}")
    
    return {
        'optimal_clusters': optimal_clusters,
        'silhouette_scores': silhouette_scores,
        'kmeans_model': kmeans,
        'train_results': train_results,
        'val_results': val_results,
        'full_results': full_results
    }



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




def predict_clusters(model,
                     train_dataloader,
                     eval_dataloader,
                     n_clusters: int = None,
                     clustering_method: str = 'kmeans',
                     device: str = 'cpu',
                     kmeans_random_state: int = 42,
                     hdbscan_min_cluster_size: int = 1000,
                     hdbscan_min_samples: int = 500,
                     hdbscan_cluster_selection_epsilon: float = 0.0, 
                     subsample_rate: int = 1
                     ) -> np.ndarray:
    """
    Fits clustering (KMeans or HDBSCAN) on latent vectors from train_dataloader and predicts
    cluster assignments for samples from eval_dataloader.
    
    Args:
        model: Trained model (e.g. PointNetAutoencoder, ShapePoseDisentanglement, ...).
               Its forward pass should yield latent codes either directly or inside a tuple.
        train_dataloader: DataLoader for training samples used to fit clustering.
        eval_dataloader: DataLoader for evaluation samples to predict cluster assignments.
                        Both dataloaders yield (points, coords) for each sample.
                        'points' are expected to be (batch_size, num_points, features),
                        and 'coords' are expected to be (batch_size, 3).
        n_clusters: The number of clusters to form using KMeans. Ignored for HDBSCAN.
        clustering_method: Clustering method to use ('kmeans' or 'hdbscan').
        device: Device to run autoencoder inference on ('cpu' or 'cuda').
        kmeans_random_state: Random state for KMeans for reproducibility.
        hdbscan_min_cluster_size: Minimum size of clusters for HDBSCAN.
        hdbscan_min_samples: Number of samples in a neighborhood for a point to be considered core.
        hdbscan_cluster_selection_epsilon: Distance threshold for cluster selection in HDBSCAN.

    Returns:
        Nx4 numpy array where each row contains (x, y, z, cluster_id) for eval samples.
        Note: HDBSCAN may assign noise points cluster_id = -1.
    """
    
    def extract_latents_and_coords(dataloader):
        """Helper function to extract latent vectors and coordinates from a dataloader."""
        latents_list = []
        coords_list = []
        
        # Local imports to avoid heavy dependencies at module import time
        from src.training_methods.spd.spd_module import ShapePoseDisentanglement
        from src.training_methods.spd.eval_spd import spd_predict_latent
        from src.training_methods.autoencoder.autoencoder_module import PointNetAutoencoder

        with torch.no_grad():
            for batch in dataloader:
                # ------------------------------------------------------
                # Unpack batch (points[, coords])
                # ------------------------------------------------------
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    points_batch, coords_batch = batch
                else:
                    # When dataloader returns only points
                    points_batch, coords_batch = batch, None

                points_batch = points_batch.to(device)

                # Optionally subsample batch dimension
                if subsample_rate > 1:
                    points_batch = points_batch[::subsample_rate]
                    if coords_batch is not None:
                        coords_batch = np.array(coords_batch)[::subsample_rate]

                # Ensure shape (B, N, 3) for further processing
                if points_batch.dim() == 2:
                    points_batch = points_batch.unsqueeze(0)

                if points_batch.dim() == 3 and points_batch.shape[1] == 3:
                    # (B, 3, N) → (B, N, 3)
                    points_batch = points_batch.permute(0, 2, 1)

                # ------------------------------------------------------
                # Extract latent representations depending on model type
                # ------------------------------------------------------
                if isinstance(model, ShapePoseDisentanglement):
                    latent_np = spd_predict_latent(points_batch, model, device=device)
                elif isinstance(model, PointNetAutoencoder):
                    # PointNetAutoencoder expects transposed input
                    points_transposed = points_batch.transpose(1, 2)
                    recon, latent_tensor, _ = model(points_transposed)
                    latent_np = latent_tensor.cpu().numpy()
                else:
                    # Generic fallback – try to interpret model output
                    model_outputs = model(points_batch)
                    if isinstance(model_outputs, tuple):
                        latent_tensor = model_outputs[0]
                    else:
                        latent_tensor = model_outputs
                    latent_np = latent_tensor.detach().cpu().numpy()

                # Ensure 2-D array with shape (B, D)
                if latent_np.ndim == 1:
                    latent_np = latent_np[None, :]
                latents_list.append(latent_np)

                # Handle coordinates if provided
                if coords_batch is not None:
                    coords_np = np.array(coords_batch)
                    # Make sure coords have shape (B, 3)
                    if coords_np.ndim == 1:
                        coords_np = coords_np[None, :]
                    coords_list.append(coords_np)
                else:
                    # Fallback to dummy zeros if coords are missing
                    coords_list.append(np.zeros((points_batch.shape[0], 3)))
        
        return latents_list, coords_list
    
    if clustering_method not in ['kmeans', 'hdbscan']:
        raise ValueError("clustering_method must be 'kmeans' or 'hdbscan'")
    
    model.eval()
    model.to(device)
    
    # Extract latent vectors from train dataloader
    print("Extracting latent vectors from train dataloader...")
    train_latents_list, _ = extract_latents_and_coords(train_dataloader)
    
    if not train_latents_list:
        return np.empty((0, 4))
    
    # Concatenate all train latent vectors
    train_latents = np.concatenate(train_latents_list, axis=0)
    
    # Fit clustering model on train latent vectors
    if clustering_method == 'kmeans':
        from sklearn.cluster import KMeans
        print(f"Fitting KMeans with {n_clusters} clusters on {train_latents.shape[0]} train samples...")
        clusterer = KMeans(n_clusters=n_clusters, random_state=kmeans_random_state, n_init=10)
        clusterer.fit(train_latents)
    elif clustering_method == 'hdbscan':
        import hdbscan
        print(f"Fitting HDBSCAN on {train_latents.shape[0]} train samples...")
        print(f"HDBSCAN parameters: min_cluster_size={hdbscan_min_cluster_size}, "
            f"min_samples={hdbscan_min_samples}, "
            f"cluster_selection_epsilon={hdbscan_cluster_selection_epsilon}")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
            prediction_data=True           # <- keep data needed for approximate_predict
        )
        clusterer.fit(train_latents)
    
    # Extract latent vectors from eval dataloader
    print("Extracting latent vectors from eval dataloader...")
    eval_latents_list, eval_coords_list = extract_latents_and_coords(eval_dataloader)
    
    if not eval_latents_list:
        return np.empty((0, 4))
    
    # Concatenate all eval latent vectors and coordinates
    eval_latents = np.concatenate(eval_latents_list, axis=0)
    eval_coords = np.concatenate(eval_coords_list, axis=0)
    
    # Predict cluster assignments for eval samples
    if clustering_method == 'kmeans':
        print(f"Predicting cluster assignments for {eval_latents.shape[0]} eval samples...")
        cluster_labels = clusterer.predict(eval_latents)
    elif clustering_method == 'hdbscan':
        print(f"Predicting cluster assignments for {eval_latents.shape[0]} eval samples using HDBSCAN...")
        # For HDBSCAN, we need to use approximate_predict for new data
        cluster_labels, _ = hdbscan.approximate_predict(clusterer, eval_latents)
        
        # Report clustering statistics
        unique_labels = np.unique(cluster_labels)
        n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(cluster_labels == -1)
        print(f"HDBSCAN found {n_clusters_found} clusters with {n_noise} noise points")
    
    # Combine coordinates and cluster labels for eval samples
    output_data = []
    for i in range(eval_coords.shape[0]):
        coord = eval_coords[i]  # [x, y, z]
        label = cluster_labels[i]  # cluster_id (integer, -1 for noise in HDBSCAN)
        output_data.append([coord[0], coord[1], coord[2], label])

    return np.array(output_data)
