import sys
import os
import numpy as np
sys.path.append(os.getcwd())
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from src.clustering.cluster_ae_latent import predict_and_save_latent
from src.autoencoder.autoencoder_module import PointNetAutoencoder
from src.utils.model_utils import load_model_from_checkpoint
from hydra import compose, initialize
from typing import List

import random
import torch
import warnings
warnings.filterwarnings("ignore")
print(f'Running from {os.getcwd()}')

def get_config_path_from_checkpoint(checkpoint_path, config_name):
    config_path_chekpoint = os.path.join(*checkpoint_path.split('/')[:-1], '.hydra')
    if not os.path.exists(config_path_chekpoint):
        config_path = '../../configs' if 'src' in os.getcwd() else 'configs' 
        print(f"Config in {config_path_chekpoint} not found, using default location {config_path}")
    else:
        config_path =  config_path_chekpoint
        config_name = 'config'

    return config_path, config_name


# Encapsulate the main logic into a function
def run_clustering_pipeline(checkpoint_path: str,
                            save_folder: str,
                            liquid_file_paths: List[str],
                            crystal_file_paths: List[str],
                            model_class: str,
                            cuda_device: int = 0,
                            max_samples: int = None,
                            add_parent_dir=False,
                            config_name: str = 'autoencoder_e3nn_64'):
    """
    Runs the clustering pipeline: loads model, generates/saves latents.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        save_folder (str): Path to save the latent-label pairs (.npy).
        liquid_file_paths (List[str]): List of paths to liquid dataset files.
        crystal_file_paths (List[str]): List of paths to crystal dataset files.
        model_class (str): Type of model ('Autoencoder' or 'Seq2Seq').
        cuda_device (int): GPU device index to use.
        max_samples (int, optional): Maximum number of samples to process. Defaults to None.
    """
    config_path, config_name = get_config_path_from_checkpoint(checkpoint_path, config_name)

    with initialize(version_base=None, config_path='../../' + config_path ):
        print(f"Loading config from {config_path}/{config_name}")
        cfg = compose(config_name=config_name)
    
    device = f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = None
    if model_class == 'Autoencoder':
        model = load_model_from_checkpoint(checkpoint_path, cfg, device=device, module=PointNetAutoencoder)
    else:
        raise ValueError(f"Unknown model_class: {model_class}. Use 'Autoencoder' or ...")

    if model is None:
        raise RuntimeError("Failed to load the model.")

    # Predict and save latent vectors
    predict_and_save_latent(cfg=cfg,  # Pass the loaded hydra config
                            model=model,
                            liquid_file_paths=liquid_file_paths,
                            crystal_file_paths=crystal_file_paths,
                            device=device,
                            save_folder=save_folder,
                            model_class=model_class, # Pass model_class to predict_and_save_latent
                            max_samples=max_samples) # Pass max_samples
    return model



from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix, silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

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

#   results = cluster_and_evaluate(latents, labels)


# Example usage (optional, can be removed or put under if __name__ == '__main__')
if __name__ == '__main__':
    checkpoint_path = 'output/2025-04-15/05-09-11/pointnet-epoch=7999-val_loss=0.04.ckpt'
    save_folder = 'output'
    liquid_file_paths = ['datasets/Al/inherent_configurations_off/166ps.off']
    crystal_file_paths = ['datasets/Al/inherent_configurations_off/240ps.off']
    model_class = 'Seq2Seq' # Or 'Autoencoder'

    run_clustering_pipeline(checkpoint_path=checkpoint_path,
                            save_folder=save_folder,
                            liquid_file_paths=liquid_file_paths,
                            crystal_file_paths=crystal_file_paths,
                            model_class=model_class,
                            cuda_device=0,
                            max_samples=1000,
                            add_parent_dir=True) 