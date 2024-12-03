import numpy as np
from typing import List, Tuple
from scipy.spatial import KDTree
from typing import Iterator, Tuple


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
  


def get_cubic_samples(points: np.ndarray, n_samples: int, cube_size: float, n_points: int) -> List[np.ndarray]:
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
    n = 0  
    dropped_points = 0
    added_points = 0

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
            if len(cube_points) > n_points:
                dropped_points += len(cube_points) - n_points
            else:
                added_points += n_points - len(cube_points)
            cube_points = drop_points_random(cube_points, n_points)

            if len(cube_points) > 0:
                samples.append(cube_points)
                n += 1

    print(f"Avg added {round(added_points/len(samples), 2)} points, avg dropped {dropped_points/n} points")
    return samples



def get_spheric_samples(points: np.ndarray, n_samples: int, radius: float, n_points: int) -> List[np.ndarray]:
    """Get N random spherical samples from point cloud using KDTree for faster querying.
    
    Args:
        points: Nx3 array of points (more than 10^5 points)
        n_samples: Number of spherical samples to extract
        radius: Radius of the spherical sample
    Returns:
        List of arrays containing points within each sphere
    """
    
    samples = []
    tree = KDTree(points)
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    dropped_points = 0
    added_points = 0

    while len(samples) < n_samples:
        # Random center point for sphere
        center = np.random.uniform(
            low=min_coords + radius,
            high=max_coords - radius
        )
        # Query points within the sphere
        indices = tree.query_ball_point(center, radius)
        
        if indices:
            sphere_points = points[indices]
            # Filter points to exact sphere shape
            distances = np.linalg.norm(sphere_points - center, axis=1)
            mask = distances <= radius
            sphere_points = sphere_points[mask]
            if len(sphere_points) > n_points:
                dropped_points += len(sphere_points) - n_points
            else:
                added_points += n_points - len(sphere_points)

            sphere_points = drop_points_farthest(sphere_points, n_points)
            if len(sphere_points) > 0:
                samples.append(sphere_points)

    print(f"Avg added {round(added_points/len(samples), 2)} points, avg dropped {dropped_points/len(samples)} points")
    return samples


def get_regular_cubic_samples(points: np.ndarray, cube_size: float, return_coords=False, n_points=100, max_samples=100000) -> List[Tuple[np.ndarray, Tuple[float, float, float]]]:
    """Divide point cloud into regular cubic samples covering the entire data space.
    
    Args:
        points: Nx3 array of points
        cube_size: Size of each cubic sample
    Returns:
        List of tuples containing points in cube and their center coordinates if return_coords is True
    """
    # Create KDTree for efficient point queries
    tree = KDTree(points)
    
    # Calculate grid dimensions
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    
    # Calculate number of cubes in each dimension
    dims = np.ceil((max_coords - min_coords) / cube_size).astype(int)
    dropped_points = 0
    added_points = 0
    samples = []
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):

                center = min_coords + np.array([i + 0.5, j + 0.5, k + 0.5]) * cube_size
                min_corner = center - cube_size / 2
                max_corner = center + cube_size / 2
                
                radius = (cube_size * np.sqrt(3)) / 2
                indices = tree.query_ball_point(center, radius)
                
                if indices:
                    cube_points = points[indices]
                    mask = np.all(
                        (cube_points >= min_corner) & 
                        (cube_points <= max_corner),
                        axis=1
                    )
                    cube_points = cube_points[mask]
                    if len(cube_points) > n_points:
                        dropped_points += len(cube_points) - n_points
                    else:
                        added_points += n_points - len(cube_points)
                    cube_points = drop_points_random(cube_points, n_points)

                    if len(cube_points) > 0:
                        if return_coords:
                            samples.append((cube_points, (center[0], center[1], center[2])))
                        else:
                            samples.append(cube_points)
                    if len(samples) >= max_samples:
                        break

    print(f"Avg added {round(added_points/len(samples), 2)} points, avg dropped {dropped_points/len(samples)} points")              
    return samples


def get_regular_spheric_samples(points: np.ndarray, radius: float, return_coords=False, n_points=100, max_samples=100000) -> List[Tuple[np.ndarray, Tuple[float, float, float]]]:
    """Divide point cloud into regular spherical samples covering the entire data space.

    Args:
        points: Nx3 array of points
        radius: Radius of each spherical sample
    Returns:
        List of arrays containing points within each sphere, optionally with their center coordinates.
    """
    tree = KDTree(points)
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)

    # Calculate grid dimensions
    dims = np.ceil((max_coords - min_coords) / (2 * radius)).astype(int)
    added_points = 0
    dropped_points = 0
    samples = []
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                center = min_coords + np.array([i + 0.5, j + 0.5, k + 0.5]) * (2 * radius)
                indices = tree.query_ball_point(center, radius)
                if indices:
                    sphere_points = points[indices]
                    distances = np.linalg.norm(sphere_points - center, axis=1)
                    mask = distances <= radius
                    sphere_points = sphere_points[mask]
                    if len(sphere_points) > n_points:
                        dropped_points += len(sphere_points) - n_points
                    else:
                        added_points += n_points - len(sphere_points)
                    sphere_points = drop_points_farthest(sphere_points, n_points)

                    if len(sphere_points) > 0:
                        if return_coords:
                            samples.append((sphere_points, (center[0], center[1], center[2])))
                        else:
                            samples.append(sphere_points)
                        if len(samples) >= max_samples:
                            break
    print(f"Avg added {round(added_points/len(samples), 2)} points, avg dropped {round(dropped_points/len(samples), 2)} points")
    return samples

if __name__ == "__main__":
    points = read_off_file("datasets/Al/inherent_configurations_off/240ps.off")

    samples = get_cubic_samples(points, n_samples=8000, cube_size=10, n_points=100) 
    std = np.std([len(sample) for sample in samples])
    print(f"Found {len(samples)} cubic non-empty samples with std={std}")

    samples = get_spheric_samples(points, n_samples=8000, radius=8.1, n_points=128) 
    std = np.std([len(sample) for sample in samples])
    print(f"Found {len(samples)} spheric non-empty samples with std={std}")

    samples = get_regular_cubic_samples(points, max_samples=8000, cube_size=10, n_points=100) 
    std = np.std([len(sample) for sample in samples])
    print(f"Found {len(samples)} regular cubic non-empty samples with std={std}")

    samples = get_regular_spheric_samples(points, max_samples=8000, radius=8.0, n_points=128)
    std = np.std([len(sample) for sample in samples])
    print(f"Found {len(samples)} regular spheric non-empty samples with std={std}")
