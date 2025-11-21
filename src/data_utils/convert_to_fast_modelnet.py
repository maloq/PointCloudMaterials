import os
import sys
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# from src.data_utils.prepare_data import read_off_file, drop_points_fps

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def read_off_file(filename: str, verbose=True, cache=True) -> np.ndarray:
    """Read points from OFF file and return as numpy array."""
    if cache:
        base, _ = os.path.splitext(filename)
        cache_filename = base + '.npy'
        if os.path.exists(cache_filename):
            # if verbose:
            #     print(f"Loading cached file from {cache_filename}")
            points = np.load(cache_filename)
            return points

    # Read the OFF file
    with open(filename, 'r') as f:
        header = f.readline().strip()
        if header != 'OFF':
            raise ValueError("Invalid OFF file format")
        n_vertices, n_faces, n_edges = map(int, f.readline().split())
        points = []
        for _ in range(n_vertices):
            x, y, z = map(float, f.readline().split())
            points.append([x, y, z])
            
    points = np.array(points)

    # Cache the data to disk for faster future loading
    if cache:
        # if verbose:
        #     print(f"Caching file to disk at {cache_filename}")
        np.save(cache_filename, points)
    return points

def drop_points_fps(points: np.ndarray, n_points: int) -> np.ndarray:
    """Adjust points to have exactly n_points using Farthest Point Sampling."""
    if len(points) == n_points:
        return points
    elif len(points) > n_points:
        return farthest_point_sample(points, n_points)
    else:
        # Upsample
        center = np.mean(points, axis=0)
        num_to_add = n_points - len(points)
        idx = np.random.choice(len(points), num_to_add, replace=True)
        new_points = 2 * center - points[idx]
        return np.vstack((points, new_points))

def convert_modelnet(
    root_dir="datasets/ModelNet40",
    metadata_file="datasets/metadata_modelnet40.csv",
    output_dir="datasets/ModelNet40_fast",
    n_points=2048,
    classes=None,
    use_fps=False
):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading metadata from {metadata_file}")
    df = pd.read_csv(metadata_file)
    
    # Get unique classes and splits
    all_classes = sorted(df['class'].unique())
    if classes is not None:
        classes = [c for c in classes if c in all_classes]
        print(f"Filtering to {len(classes)} classes: {classes}")
    else:
        classes = all_classes
    
    splits = ['train', 'test']
    
    print(f"Found {len(classes)} classes.")
    print(f"Output directory: {output_dir}")
    print(f"Target points per sample: {n_points}")
    if use_fps:
        print(f"Using FPS (SLOW - this will take hours!)")
    else:
        print(f"Using random sampling (FAST)")
    print()
    
    import time
    overall_start = time.time()
    
    for class_idx, class_name in enumerate(classes, 1):
        print(f"\n[{class_idx}/{len(classes)}] Processing class: {class_name}")
        class_start = time.time()
        
        for split in splits:
            subset = df[(df['class'] == class_name) & (df['split'] == split)]
            
            if len(subset) == 0:
                continue
            
            print(f"  {split}: {len(subset)} samples", end='', flush=True)
            split_start = time.time()
                
            all_points = []
            
            for sample_idx, (_, row) in enumerate(subset.iterrows(), 1):
                if sample_idx % 10 == 0 or sample_idx == len(subset):
                    print(f".", end='', flush=True)
                    
                rel_path = row['object_path']
                full_path = os.path.join(root_dir, rel_path)
                
                try:
                    # Load points
                    points = read_off_file(full_path, verbose=False, cache=True)
                    
                    # Resample - use random sampling by default (FAST) or FPS if requested (SLOW)
                    if use_fps:
                        points = drop_points_fps(points, n_points)
                    else:
                        # Fast random sampling
                        if len(points) >= n_points:
                            idx = np.random.choice(len(points), n_points, replace=False)
                        else:
                            idx = np.random.choice(len(points), n_points, replace=True)
                        points = points[idx]
                    
                    # Normalize
                    points = points - np.mean(points, axis=0)
                    dist = np.max(np.linalg.norm(points, axis=1))
                    if dist > 0:
                        points = points / dist
                        
                    all_points.append(points)
                    
                except Exception as e:
                    print(f"\n  ERROR processing {full_path}: {e}")
                    
            split_time = time.time() - split_start
            print(f" done ({split_time:.1f}s)")
                    
            if not all_points:
                continue
                
            # Stack
            points_tensor = torch.tensor(np.array(all_points), dtype=torch.float32)
            
            # Save
            save_path = os.path.join(output_dir, f"{class_name}_{split}.pt")
            torch.save({
                'points': points_tensor,
                'class_name': class_name,
                'n_samples': len(all_points)
            }, save_path)
            
    print("Conversion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="datasets/ModelNet40")
    parser.add_argument("--metadata_file", default="datasets/metadata_modelnet40.csv")
    parser.add_argument("--output_dir", default="datasets/ModelNet40_fast")
    parser.add_argument("--n_points", type=int, default=2048)
    parser.add_argument("--classes", nargs="+", help="Specific classes to process")
    parser.add_argument("--use-fps", action="store_true", 
                        help="Use FPS instead of random sampling (MUCH slower but better quality)")
    
    args = parser.parse_args()
    
    convert_modelnet(
        root_dir=args.root_dir,
        metadata_file=args.metadata_file,
        output_dir=args.output_dir,
        n_points=args.n_points,
        classes=args.classes,
        use_fps=args.use_fps
    )
