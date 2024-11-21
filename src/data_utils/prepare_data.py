import numpy as np
from typing import List, Tuple
from scipy.spatial import KDTree


def read_off_file(filename: str, verbose=True) -> np.ndarray:
    """Read points from OFF file and return as numpy array."""
    with open(filename, 'r') as f:
        # Skip OFF header
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
        print(f"Read {len(points)} points")
        print(f"Size of space: {space_size}")
        print(f"Min coords: {min_coords}")
        print(f"Max coords: {max_coords}")

    return np.array(points)


def get_cubic_samples(points: np.ndarray, n_samples: int, cube_size: float) -> List[np.ndarray]:
    """Get N random cubic samples from point cloud using KDTree for faster querying.
    
    Args:
        points: Nx3 array of points (more than 10^5 points)
        n_samples: Number of cubic samples to extract
        cube_size: Size of cubic sample
    Returns:
        List of arrays containing points within each cube
    """
    
    samples = []
    tree = KDTree(points)
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    
    while len(samples) < n_samples:
        # Random center point for cube
        center = np.random.uniform(
            low=min_coords + cube_size/2,
            high=max_coords - cube_size/2
        )
        
        # Query points within the cube's bounding sphere
        # Using diagonal/2 as radius ensures we cover the entire cube
        radius = (cube_size * np.sqrt(3)) / 2
        indices = tree.query_ball_point(center, radius)
        
        if indices:
            # Filter points to exact cube shape
            cube_points = points[indices]
            mask = np.all(
                (cube_points >= center - cube_size/2) & 
                (cube_points <= center + cube_size/2),
                axis=1
            )
            cube_points = cube_points[mask]
            
            if len(cube_points) > 0:
                samples.append(cube_points)
    
    return samples


if __name__ == "__main__":
    points = read_off_file("datasets/Al/inherent_configurations_off/240ps.off")
    samples = get_cubic_samples(points, n_samples=8000, cube_size=12) 
    
    print(f"Found {len(samples)} non-empty samples")
    for i, sample in enumerate(samples):
        print(f"Sample {i} has {len(sample)}")