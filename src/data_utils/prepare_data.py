import sys,os
sys.path.append(os.getcwd())
from src.utils.logging_config import setup_logging
logger = setup_logging()
import numpy as np
from typing import List, Tuple
from scipy.spatial import KDTree
from typing import Iterator, Tuple
import os
import logging
from src.utils.logging_config import setup_logging
logger = setup_logging()



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
    """Read points from OFF file and return as numpy array.
    
    Optionally caches the file on disk in a faster .npy format.
    
    Args:
        filename: Path to the OFF file.
        verbose: If True, prints additional information.
        cache: If True, will attempt to load a cached npy file if available,
               and will save to cache after parsing.
        
    Returns:
        A numpy array of point coordinates (shape: [N, 3]).
    """
    if cache:
        base, _ = os.path.splitext(filename)
        cache_filename = base + '.npy'
        if os.path.exists(cache_filename):
            if verbose:
                print(f"Loading cached file from {cache_filename}")
            points = np.load(cache_filename)
            return points

    # Read the OFF file
    with open(filename, 'r') as f:
        # Read and verify OFF header
        header = f.readline().strip()
        if header != 'OFF':
            raise ValueError("Invalid OFF file format")
        n_vertices, n_faces, n_edges = map(int, f.readline().split())
        points = []
        for _ in range(n_vertices):
            x, y, z = map(float, f.readline().split())
            points.append([x, y, z])
            
    points = np.array(points)

    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    space_size = max_coords - min_coords
    if verbose: 
        logger.print(f"Read {len(points)} points")
        logger.print(f"Size of space: {space_size}")
        logger.print(f"Min coords: {min_coords}")
        logger.print(f"Max coords: {max_coords}")

    # Cache the data to disk for faster future loading
    if cache:
        if verbose:
            print(f"Caching file to disk at {cache_filename}")
        np.save(cache_filename, points)
    return points


def drop_points_random(points: np.ndarray, n_points: int) -> np.ndarray:

    """Drop points from point cloud to have exactly n_points."""
    if len(points) > n_points:
        idx = np.random.choice(len(points), n_points, replace=False)
        return points[idx]
    elif len(points) < n_points:
        idx = np.random.choice(len(points), n_points, replace=True)
        return points[idx]
    return points


def drop_points_farthest(points: np.ndarray, n_points: int) -> np.ndarray:
    """Adjust points to have exactly n_points.

    If len(points) > n_points, drop the farthest points from the center.
    If len(points) < n_points, add points symmetrically with respect to the center.
    """
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    if len(points) == n_points:
        return points
    elif len(points) > n_points:
        indices = np.argsort(distances)[:n_points]
        return points[indices]
    else:
        num_to_add = n_points - len(points)
        idx = np.random.choice(len(points), num_to_add, replace=True)
        new_points = 2 * center - points[idx]
        return np.vstack((points, new_points))
  

def get_min_max_coords(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    return min_coords, max_coords


def calculate_stride(size: float, sample_type: str, overlap_fraction: float) -> float:
    multiplier = 2 if sample_type == 'spheric' else 1
    stride = size * multiplier * (1 - overlap_fraction)
    if stride <= 0:
        raise ValueError("overlap_fraction must be less than 1.")
    return stride


def compute_dimensions(min_coords: np.ndarray, max_coords: np.ndarray, stride: float) -> np.ndarray:
    dims = np.ceil((max_coords - min_coords) / stride).astype(int)
    return dims


def calculate_center(min_coords, stride, i, j, k, size, sample_type):
    center = min_coords + np.array([i, j, k]) * stride
    if sample_type == 'spheric':
        center += size
    else:
        center += size / 2
    return center


def process_sample(points, tree, center, size, sample_type, n_points):
    if sample_type == 'cubic':
        radius = (size * np.sqrt(3)) / 2
    else:
        radius = size
    indices = tree.query_ball_point(center, radius)
    if not indices:
        return None, 0, 0
    sample_points = points[indices]
    if sample_type == 'cubic':
        min_corner = center - size / 2
        max_corner = center + size / 2
        mask = np.all((sample_points >= min_corner) & (sample_points <= max_corner), axis=1)
        sample_points = sample_points[mask]
        drop_func = drop_points_random
    else:
        distances = np.linalg.norm(sample_points - center, axis=1)
        mask = distances <= size
        sample_points = sample_points[mask]
        drop_func = drop_points_farthest
    if len(sample_points) > n_points:
        dropped = len(sample_points) - n_points
    else:
        dropped = 0
    added = max(n_points - len(sample_points), 0)
    sample_points = drop_func(sample_points, n_points)
    return sample_points, added, dropped


def generate_samples(
    points: np.ndarray,
    tree: KDTree,
    min_coords: np.ndarray,
    stride: float,
    size: float,
    sample_type: str,
    dims: np.ndarray,
    n_points: int,
    return_coords: bool,
    max_samples: int
) -> Tuple[List, int, int]:
    samples = []
    added_points = dropped_points = 0
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                # Compute the grid center as before.
                computed_center = calculate_center(min_coords, stride, i, j, k, size, sample_type)
                # Adjust the center so that it matches an atom by snapping to the nearest input point.
                _, nearest_index = tree.query(computed_center)
                center = points[nearest_index]
                # Proceed as before using the adjusted center.
                sample_points, add, drop = process_sample(points, tree, center, size, sample_type, n_points)
                added_points += add
                dropped_points += drop
                if sample_points is not None:
                    if return_coords:
                        samples.append((sample_points, tuple(center)))
                    else:
                        samples.append(sample_points)
                    if len(samples) >= max_samples:
                        return samples, added_points, dropped_points
    return samples, added_points, dropped_points


def get_random_samples(
    points: np.ndarray,
    n_samples: int,
    size: float,
    n_points: int,
    sample_shape: str = 'cubic'
) -> List[np.ndarray]:
    """Same as get_regular_samples but with random center points.
    
    Args:
        points: Nx3 array of points (more than 10^5 points)
        n_samples: Number of samples to extract
        size: Size of the sample (cube_size for cubic, radius for spherical)
        n_points: Number of points per sample
        sample_shape: 'cubic' or 'spheric'
    Returns:
        List of arrays containing points within each sample
    """
    
    if sample_shape not in ['cubic', 'spheric']:
        raise ValueError("sample_shape must be 'cubic' or 'spheric'")
    
    samples = []
    tree = KDTree(points)
    min_coords, max_coords = get_min_max_coords(points)
    dropped_points = 0
    added_points = 0

    while len(samples) < n_samples:
        # Random center point for sample
        center = np.random.uniform(
            low=min_coords + (size / 2 if sample_shape == 'cubic' else size),
            high=max_coords - (size / 2 if sample_shape == 'cubic' else size)
        )
        
        sample_points, add, drop = process_sample(points, tree, center, size, sample_shape, n_points)
        if sample_points is not None:
            samples.append(sample_points)
            added_points += add
            dropped_points += drop

    logger.print(f"Avg added {round(added_points/len(samples), 2)} points, avg dropped {round(dropped_points/len(samples), 2)} points")
    return samples


def get_regular_samples(
    points: np.ndarray,
    size: float,
    sample_shape: str = 'cubic',
    overlap_fraction: float = 0.0,
    return_coords: bool = False,
    n_points: int = 100,
    max_samples: int = 2e32
) -> List[Tuple[np.ndarray, Tuple[float, float, float]]]:
    """
    Divide point cloud into regular samples covering the entire data space with optional overlap.

    This function partitions the input point cloud into a grid of regularly spaced sampling regions 
    (either cubic or spheric) based on the provided size and overlap fraction. For cubic samples, the 
    regions are cubes with edge length 'size', while for spheric samples, the regions are spheres with 
    radius 'size'. To ensure uniformity, each sample is adjusted to contain exactly n_points by 
    appropriately dropping or adding points. Only regions that fully fit within the adjusted data space 
    (after applying the necessary padding) are considered, thus avoiding partial samples along the edges. 
    A KDTree is employed for efficient neighborhood queries, and the function logs the average number of 
    points added and dropped per sample during this adjustment.

    Parameters:
    -----------
    points : np.ndarray
        The input point cloud as an array of shape (N, 3).
    size : float
        For cubic samples, the side length of the cube; for spheric samples, the radius of the sphere.
    sample_shape : str, optional
        The shape of the sample window; must be either 'cubic' (default) or 'spheric'.
    overlap_fraction : float, optional
        The fractional overlap between adjacent sample regions. This value should be less than 1.
    return_coords : bool, optional
        If True, each sample is returned as a tuple (sample_points, sample_center), where sample_center 
        is a tuple (x, y, z); otherwise, only the sample_points array is returned.
    n_points : int, optional
        The desired number of points in each sample. Points in a sample will be dropped or added 
        to meet this exact count.
    max_samples : int, optional
        The maximum number of samples to generate. The function stops sampling once this number is reached,
        even if additional samples could be produced.

    Returns:
    --------
    List[Tuple[np.ndarray, Tuple[float, float, float]]]
        A list of samples extracted from the point cloud. Each entry in the list is:
          - an np.ndarray of shape (n_points, 3) if return_coords is False, or 
          - a tuple (sample_points, sample_center) if return_coords is True,
        where sample_center is a tuple containing the (x, y, z) coordinates of the sample's center.
    """
    tree = KDTree(points)
    min_coords, max_coords = get_min_max_coords(points)
    stride = calculate_stride(size, sample_shape, overlap_fraction)
    
    padding = size / 2 if sample_shape == 'cubic' else size
    min_center = min_coords + padding
    max_center = max_coords - padding

    dims = compute_dimensions(min_center, max_center, stride)
    samples, added_points, dropped_points = generate_samples(
        points, tree, min_center, stride, size, sample_shape, dims, 
        n_points, return_coords, max_samples
    )
    logger.print(f"Avg added {round(added_points/len(samples), 2)} points, avg dropped {round(dropped_points/len(samples), 2)} points")
    return samples




if __name__ == "__main__":
    points = read_off_file("datasets/Al/inherent_configurations_off/240ps.off")

    samples = get_random_samples(points, sample_shape='cubic', n_samples=8000, size=10, n_points=100) 
    std = np.std([len(sample) for sample in samples])
    logger.print(f"Found {len(samples)} cubic non-empty samples with std={std}")

    samples = get_random_samples(points, sample_shape='spheric', n_samples=8000, size=8.1, n_points=128) 
    std = np.std([len(sample) for sample in samples])
    logger.print(f"Found {len(samples)} spheric non-empty samples with std={std}")

    samples = get_regular_samples(points, sample_shape='cubic', max_samples=10000, size=10, n_points=100, overlap_fraction=0.3) 
    std = np.std([len(sample) for sample in samples])
    logger.print(f"Found {len(samples)} regular cubic non-empty samples with std={std}")

    samples = get_regular_samples(points, sample_shape='spheric', max_samples=10000, size=8.0, n_points=128, overlap_fraction=0.3)
    std = np.std([len(sample) for sample in samples])
    logger.print(f"Found {len(samples)} regular spheric non-empty samples with std={std}")
