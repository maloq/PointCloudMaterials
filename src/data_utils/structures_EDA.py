import sys,os
sys.path.append(os.getcwd())
from pytorch3d.loss import chamfer_distance 
import torch

import numpy as np
from scipy.spatial.distance import cdist
import random
from src.cls.prediction import create_dataloader
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from warnings import filterwarnings
filterwarnings("ignore")


N = 100  # Number of random pairs to average over

def calculate_chamfer_distances(dataloader1, dataloader2, N=5):
    """
    Calculate Chamfer distances between crystal and liquid point clouds.
    
    Args:
        dataloader1: Dataloader for liquid point clouds
        dataloader2: Dataloader for crystal point clouds 
        N: Number of random pairs to average over
        
    Returns:
        dict: Statistics (mean, std) for each comparison type
    """
    # Initialize lists to store distances
    distances = {
        'Crystal-Liquid': [],
        'Liquid-Liquid': [], 
        'Crystal-Crystal': []
    }

    # Calculate distances for N random pairs
    for _ in range(N):
        # Get random point clouds
        points_l1, _ = random.choice(list(dataloader1))
        points_l2, _ = random.choice(list(dataloader1)) 
        points_c1, _ = random.choice(list(dataloader2))
        points_c2, _ = random.choice(list(dataloader2))
        
        # Convert to torch tensors
        points = {
            'l1': torch.tensor(points_l1[0].numpy()).unsqueeze(0),
            'l2': torch.tensor(points_l2[0].numpy()).unsqueeze(0),
            'c1': torch.tensor(points_c1[0].numpy()).unsqueeze(0),
            'c2': torch.tensor(points_c2[0].numpy()).unsqueeze(0)
        }
        
        # Calculate and store distances
        distances['Crystal-Liquid'].append(chamfer_distance(points['l1'], points['c1'])[0].item())
        distances['Liquid-Liquid'].append(chamfer_distance(points['l1'], points['l2'])[0].item())
        distances['Crystal-Crystal'].append(chamfer_distance(points['c1'], points['c2'])[0].item())

    # Calculate statistics
    stats = {}
    for key in distances:
        mean = sum(distances[key]) / N
        std = (sum((x - mean) ** 2 for x in distances[key]) / N) ** 0.5
        stats[key] = (mean, std)

    # Print results
    print(f"Mean Chamfer distances over {N} random pairs:")
    for key, (mean, std) in stats.items():
        print(f"{key}: {mean:.4f} ± {std:.4f}")
        
    return stats


def compute_sorted_distances(points, k):
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    sorted_dists = np.sort(dists, axis=1)
    return sorted_dists[:, 1:k+1]


def compare(point_set1, point_set2, k=100, metric='chebyshev'):
    """
    Calculate both PDD and AMD between two point sets.
    """
    sorted_dists1 = compute_sorted_distances(point_set1, k)
    sorted_dists2 = compute_sorted_distances(point_set2, k)
    
    pdd = np.mean(cdist(sorted_dists1, sorted_dists2, metric=metric))
    amd1 = np.mean(sorted_dists1, axis=1)
    amd2 = np.mean(sorted_dists2, axis=1)
    amd = cdist([amd1], [amd2], metric=metric)[0, 0]
    
    return pdd, amd

def compare_multiple_samples(dataloader1, dataloader2, N=5, k=100, metric='chebyshev'):
    """
    Compare N random pairs of samples and calculate statistics for both PDD and AMD.
    """
    comparisons = {
        'Crystal-Liquid': [],
        'Liquid-Liquid': [],
        'Crystal-Crystal': []
    }
    stats = {
        'PDD': {key: [] for key in comparisons},
        'AMD': {key: [] for key in comparisons}
    }

    for _ in range(N):
        points_l1 = random.choice(list(dataloader1))[0][0].numpy()
        points_l2 = random.choice(list(dataloader1))[0][0].numpy()
        points_c1 = random.choice(list(dataloader2))[0][0].numpy()
        points_c2 = random.choice(list(dataloader2))[0][0].numpy()
        
        sample_pairs = {
            'Crystal-Liquid': (points_l1, points_c1),
            'Liquid-Liquid': (points_l1, points_l2),
            'Crystal-Crystal': (points_c1, points_c2)
        }
        
        for key, (p1, p2) in sample_pairs.items():
            pdd, amd = compare(p1, p2, k=k, metric=metric)
            stats['PDD'][key].append(pdd)
            stats['AMD'][key].append(amd)
    
    for metric_type in stats:
        for key in stats[metric_type]:
            mean = np.mean(stats[metric_type][key])
            std = np.std(stats[metric_type][key])
            stats[metric_type][key] = (mean, std)
    
    return stats


if __name__ == "__main__":    

    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="Al_classification.yaml")
    print(OmegaConf.to_yaml(cfg))

    cfg.data.point_size = 64
    cfg.data.radius = 7.9
    cfg.data.overlap_fraction = 0.5

    file_path = 'datasets/Al/inherent_configurations_off/166ps.off'
    dataloader1 = create_dataloader(cfg, file_path)
    file_path = 'datasets/Al/inherent_configurations_off/240ps.off'
    dataloader2 = create_dataloader(cfg, file_path)
    calculate_chamfer_distances(dataloader1, dataloader2, N=5)
          
    N = 100
    stats = compare_multiple_samples(dataloader1, dataloader2, N=N)

    print(f"\nResults over {N} random pairs:")
    for metric in ['PDD', 'AMD']:
        print(f"\n{metric} distances:")
        for comparison, (mean, std) in stats[metric].items():
            print(f"{comparison}: {mean:.4f} ± {std:.4f}")
