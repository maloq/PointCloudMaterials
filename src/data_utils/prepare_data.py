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

    If len(points) > n_points, drop the farthest points from the centroid.
    If len(points) < n_points, add points symmetrically with respect to the centroid.
    """
    n = len(points)
    if n == n_points:
        return points
    center = np.mean(points, axis=0)
    if n > n_points:
        diff = points - center
        dist_sq = np.einsum('ij,ij->i', diff, diff)
        indices = np.argpartition(dist_sq, n_points)[:n_points]
        return points[indices]
    else:
        num_to_add = n_points - n
        idx = np.random.choice(n, num_to_add, replace=True)
        new_points = 2 * center - points[idx]
        return np.vstack((points, new_points))


def drop_points_fps(points: np.ndarray, n_points: int) -> np.ndarray:
    """Adjust points to have exactly n_points using Farthest Point Sampling.

    If len(points) > n_points, use FPS to select a subset that preserves shape.
    If len(points) < n_points, add points symmetrically with respect to the center.
    """
    if len(points) == n_points:
        return points
    elif len(points) > n_points:
        return farthest_point_sample(points, n_points)
    else:
        # Same upsampling strategy as drop_points_farthest
        center = np.mean(points, axis=0)
        num_to_add = n_points - len(points)
        idx = np.random.choice(len(points), num_to_add, replace=True)
        new_points = 2 * center - points[idx]
        return np.vstack((points, new_points))
  

def get_min_max_coords(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    return min_coords, max_coords


def calculate_stride(size: float, overlap_fraction: float) -> float:
    stride = size * (1 - overlap_fraction)
    if stride <= 0:
        raise ValueError("overlap_fraction must be less than 1.")
    return stride


def compute_dimensions(min_coords: np.ndarray, max_coords: np.ndarray, stride: float) -> np.ndarray:
    dims = np.ceil((max_coords - min_coords) / stride).astype(int)
    return dims


def calculate_center(min_coords, stride, i, j, k, size):
    center = min_coords + np.array([i, j, k]) * stride
    center += size
    return center


def process_sample(points, tree, center, size, n_points, sampling_method="drop_farthest"):
 
    radius = size
    indices = tree.query_ball_point(center, radius)
    if not indices:
        return None, 0, 0
    sample_points = points[indices]
   
    distances = np.linalg.norm(sample_points - center, axis=1)
    mask = distances <= size
    sample_points = sample_points[mask]
    
    if sampling_method == "fps":
        drop_func = drop_points_fps
    elif sampling_method == "drop_farthest":
        drop_func = drop_points_farthest
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")

    if len(sample_points) > n_points:
        dropped = len(sample_points) - n_points
    else:
        dropped = 0
    added = max(n_points - len(sample_points), 0)
    sample_points = drop_func(sample_points, n_points)
    # Ensure the snapped center atom is always present at the origin.
    if sample_points.size > 0:
        dists_to_center = np.linalg.norm(sample_points - center, axis=1)
        if not np.any(dists_to_center < 1e-8):
            farthest_idx = int(np.argmax(dists_to_center))
            sample_points[farthest_idx] = center
    return sample_points, added, dropped


def _resolve_drop_func(sampling_method: str):
    if sampling_method == "fps":
        return drop_points_fps
    if sampling_method == "drop_farthest":
        return drop_points_farthest
    raise ValueError(f"Unknown sampling method: {sampling_method!r}")


def generate_samples(
    points: np.ndarray,
    tree: KDTree,
    min_coords: np.ndarray,
    stride: float,
    size: float,
    dims: np.ndarray,
    n_points: int,
    return_coords: bool,
    max_samples: int,
    drop_edge_samples: bool = True,
    sampling_method: str = "drop_farthest"
) -> Tuple[List, int, int]:
    drop_func = _resolve_drop_func(sampling_method)

    if drop_edge_samples:
        ranges = [(1, int(d) - 1) if d >= 3 else (0, 0) for d in dims]
    else:
        ranges = [(0, int(d)) for d in dims]

    if any(s >= e for s, e in ranges):
        return [], 0, 0

    # Vectorized grid center computation via meshgrid
    grid_arrays = [np.arange(s, e) for s, e in ranges]
    mesh = np.meshgrid(*grid_arrays, indexing='ij')
    grid_ijk = np.column_stack([m.ravel() for m in mesh]).astype(np.float64)

    M = min(len(grid_ijk), int(max_samples))
    grid_ijk = grid_ijk[:M]

    computed_centers = min_coords + grid_ijk * stride + size

    # Batch snap all grid centers to nearest atoms (one C call instead of M Python calls)
    _, nearest_indices = tree.query(computed_centers)
    snapped_centers = points[nearest_indices]

    # Batch radius query (one C call instead of M Python calls)
    neighbor_lists = tree.query_ball_point(snapped_centers, size)

    samples: list = []
    total_added = total_dropped = 0

    for center, nbrs in zip(snapped_centers, neighbor_lists):
        if not nbrs:
            continue
        sample_pts = points[nbrs]
        n_have = len(sample_pts)
        added = max(n_points - n_have, 0)
        dropped = max(n_have - n_points, 0)
        total_added += added
        total_dropped += dropped

        sample_pts = drop_func(sample_pts, n_points)

        # Ensure center atom present
        diff = sample_pts - center
        dist_sq = np.einsum('ij,ij->i', diff, diff)
        if not np.any(dist_sq < 1e-16):
            sample_pts[int(np.argmax(dist_sq))] = center

        sample_pts = sample_pts - center
        if return_coords:
            samples.append((sample_pts, np.array(center)))
        else:
            samples.append(sample_pts)
        if len(samples) >= max_samples:
            break

    return samples, total_added, total_dropped



def get_regular_samples(
    points: np.ndarray,
    size: float,
    overlap_fraction: float = 0.0,
    return_coords: bool = False,
    n_points: int = 100,
    max_samples: int = 2e32,
    drop_edge_samples: bool = True,
    sampling_method: str = "drop_farthest"
) -> List[Tuple[np.ndarray, Tuple[float, float, float]]]:
    """
    Divide point cloud into regular samples covering the entire data space with optional overlap.

    This function partitions the input point cloud into a grid of regularly spaced sampling regions 
    (spheric) based on the provided size and overlap fraction. For spheric samples, the regions are spheres with 
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
        For spheric samples, the radius of the sphere.

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
    drop_edge_samples : bool, optional
        If True, samples from the outermost layer of the grid will be dropped. This results in samples
        that are further away from the boundaries of the sampling region. Defaults to False.

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
    stride = calculate_stride(size, overlap_fraction)
    
    padding = size
    min_center = min_coords + padding
    max_center = max_coords - padding

    # Ensure min_center is less than max_center along all dimensions
    if np.any(min_center >= max_center):
        logger.print("Sampling region is too small or inverted after padding. No samples will be generated.")
        return []

    # Grid dimensions
    dims = compute_dimensions(min_center, max_center, stride)
    # For reporting: grid without padding (i.e., if edge-near centers were not dropped)
    dims_no_padding = compute_dimensions(min_coords, max_coords, stride)

    # Report what fraction is excluded by padding and by drop_edge_samples
    total_no_padding = int(np.prod(np.maximum(dims_no_padding, 0)))
    total_padded = int(np.prod(np.maximum(dims, 0)))
    if total_no_padding > 0:
        dropped_padding = max(total_no_padding - total_padded, 0)
        pct_padding = 100.0 * dropped_padding / total_no_padding
        logger.print(
            f"Edge exclusion (padding): dropped {dropped_padding}/{total_no_padding} centers "
            f"({pct_padding:.2f}%)"
        )
    else:
        logger.print("Edge exclusion (padding): insufficient grid to estimate (0 total)")
    
    # Ensure dimensions are non-negative
    if np.any(dims <= 0):
        logger.print("Calculated dimensions for sampling grid are non-positive. No samples will be generated.")
        return []

    samples, added_points, dropped_points = generate_samples(
        points, tree, min_center, stride, size, dims, 
        n_points, return_coords, max_samples, drop_edge_samples, sampling_method
    )

    if len(samples) > 0:
        avg_added = round(added_points / len(samples), 2)
        avg_dropped = round(dropped_points / len(samples), 2)
        logger.print(f"Generated {len(samples)} samples.")
        logger.print(f"Avg added {avg_added} points, avg dropped {avg_dropped} points per sample.")
        if drop_edge_samples and total_padded > 0:
            # interior dims after dropping the outer layer on each side when dims >= 3
            keep_i = dims[0] - 2 if dims[0] >= 3 else 0
            keep_j = dims[1] - 2 if dims[1] >= 3 else 0
            keep_k = dims[2] - 2 if dims[2] >= 3 else 0
            kept_interior = int(max(keep_i, 0) * max(keep_j, 0) * max(keep_k, 0))
            dropped_edges = max(total_padded - kept_interior, 0)
            pct_edges = 100.0 * dropped_edges / total_padded
            logger.print(
                f"Additional edge-layer drop (drop_edge_samples=True): dropped {dropped_edges}/{total_padded} centers "
                f"({pct_edges:.2f}%), kept interior {kept_interior}"
            )
    else:
        logger.print("No samples were generated with the given parameters.")
        
    return samples


def get_random_samples(
    points: np.ndarray,
    n_samples: int,
    size: float,
    n_points: int,
    return_coords: bool = False,
    sampling_method: str = "drop_farthest"
) -> List[np.ndarray]:
    """Same as get_regular_samples but with random center points.

    Args:
        points: Nx3 array of points (more than 10^5 points)
        n_samples: Number of samples to extract
        size: Size of the sample (radius for spherical)
        n_points: Number of points per sample
        return_coords: If True, also return the sampled center coordinates
    Returns:
        List of arrays containing points within each sample
    """
    drop_func = _resolve_drop_func(sampling_method)
    tree = KDTree(points)
    min_coords, max_coords = get_min_max_coords(points)

    # Generate all random centers in one batch (with extra to account for empty ones)
    n_generate = int(n_samples * 1.05) + 64
    random_centers = np.random.uniform(
        low=min_coords + size,
        high=max_coords - size,
        size=(n_generate, 3),
    )

    # Batch snap to nearest atoms
    _, nearest_indices = tree.query(random_centers)
    snapped_centers = points[nearest_indices]

    # Batch radius query
    neighbor_lists = tree.query_ball_point(snapped_centers, size)

    samples: list = []
    total_added = total_dropped = 0

    for center, nbrs in zip(snapped_centers, neighbor_lists):
        if not nbrs:
            continue
        sample_pts = points[nbrs]
        n_have = len(sample_pts)
        added = max(n_points - n_have, 0)
        dropped = max(n_have - n_points, 0)
        total_added += added
        total_dropped += dropped

        sample_pts = drop_func(sample_pts, n_points)

        diff = sample_pts - center
        dist_sq = np.einsum('ij,ij->i', diff, diff)
        if not np.any(dist_sq < 1e-16):
            sample_pts[int(np.argmax(dist_sq))] = center

        sample_pts = sample_pts - center
        if return_coords:
            samples.append((sample_pts, center.astype(np.float64)))
        else:
            samples.append(sample_pts)
        if len(samples) >= n_samples:
            break

    # Rare fallback: if batch didn't produce enough, generate individually
    while len(samples) < n_samples:
        center = np.random.uniform(low=min_coords + size, high=max_coords - size)
        _, nearest_index = tree.query(center)
        center = points[nearest_index]
        nbrs = tree.query_ball_point(center, size)
        if not nbrs:
            continue
        sample_pts = points[nbrs]
        n_have = len(sample_pts)
        total_added += max(n_points - n_have, 0)
        total_dropped += max(n_have - n_points, 0)
        sample_pts = drop_func(sample_pts, n_points)
        diff = sample_pts - center
        dist_sq = np.einsum('ij,ij->i', diff, diff)
        if not np.any(dist_sq < 1e-16):
            sample_pts[int(np.argmax(dist_sq))] = center
        sample_pts = sample_pts - center
        if return_coords:
            samples.append((sample_pts, center.astype(np.float64)))
        else:
            samples.append(sample_pts)

    avg_added = round(total_added / len(samples), 2)
    avg_dropped = round(total_dropped / len(samples), 2)
    logger.print(f"Avg added {avg_added} points, avg dropped {avg_dropped} points")
    return samples


if __name__ == "__main__":
    points = read_off_file("datasets/Al/inherent_configurations_off/240ps.npy")

  

    samples = get_random_samples(points, n_samples=8000, size=8.1, n_points=128) 
    std = np.std([len(sample) for sample in samples])
    logger.print(f"Found {len(samples)} spheric non-empty samples with std={std}")


    samples = get_regular_samples(points, max_samples=10000, size=8.0, n_points=128, overlap_fraction=0.3)
    std = np.std([len(sample) for sample in samples])
    logger.print(f"Found {len(samples)} regular spheric non-empty samples with std={std}")


    samples_spheric_no_edges = get_regular_samples(points, max_samples=10000, size=8.0, n_points=128, overlap_fraction=0.3, drop_edge_samples=True)
    if samples_spheric_no_edges:
        std_spheric_no_edges = np.std([len(sample) for sample in samples_spheric_no_edges])
        logger.print(f"Found {len(samples_spheric_no_edges)} regular spheric non-empty samples (edges dropped) with std={std_spheric_no_edges}")
    else:
        logger.print("Found 0 regular spheric non-empty samples (edges dropped)")
