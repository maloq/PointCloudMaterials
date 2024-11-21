import numpy as np
from typing import Iterator, Tuple
from scipy.spatial import KDTree
import sys,os
sys.path.append(os.getcwd())
from src.data_utils.prepare_data import read_off_file
from src.data_utils.data_load import pc_normalize
import torch
from torch.utils.data import Dataset, DataLoader
import plotly.graph_objects as go

# Looks like this fuction outputs only near diagonal samples from the data. I need samples from all the data space, like regular grid

def get_regular_cubic_samples_faster(points: np.ndarray, cube_size: float) -> Iterator[Tuple[np.ndarray, Tuple[float, float, float]]]:
    """Divide point cloud into regular cubic samples covering the entire data space.
    
    Args:
        points: Nx3 array of points
        cube_size: Size of each cubic sample
    Yields:
        Tuple of (points in cube, (x, y, z) cube center coordinates)
    """
    # Create KDTree for efficient point queries
    tree = KDTree(points)
    
    # Calculate grid dimensions
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    
    # Calculate number of cubes in each dimension
    dims = np.ceil((max_coords - min_coords) / cube_size).astype(int)
    
    # Iterate through all grid positions
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                # Calculate cube center
                center = min_coords + np.array([i + 0.5, j + 0.5, k + 0.5]) * cube_size
                
                # Define cube boundaries
                min_corner = center - cube_size / 2
                max_corner = center + cube_size / 2
                
                # Query points within the cube's bounding sphere
                radius = (cube_size * np.sqrt(3)) / 2
                indices = tree.query_ball_point(center, radius)
                
                if indices:
                    # Filter points to exact cube shape
                    cube_points = points[indices]
                    mask = np.all(
                        (cube_points >= min_corner) & 
                        (cube_points <= max_corner),
                        axis=1
                    )
                    cube_points = cube_points[mask]
                    
                    if len(cube_points) > 0:
                        yield cube_points, (center[0], center[1], center[2])


def get_regular_cubic_samples(points: np.ndarray, cube_size: float) -> Iterator[Tuple[np.ndarray, Tuple[float, float, float]]]:
    """Create regular cubic samples in a grid pattern covering the entire data space.
    
    Args:
        points: Nx3 array of points
        cube_size: Size of each cubic sample
    Yields:
        Tuple of (points in cube, (x, y, z) cube center coordinates)
    """
    # Calculate grid boundaries
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    
    # Add small padding to ensure all points are included
    padding = cube_size * 0.1
    min_coords -= padding
    max_coords += padding
    
    # Calculate number of cubes in each dimension
    dims = np.ceil((max_coords - min_coords) / cube_size).astype(int)
    
    # Create regular grid
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                # Calculate cube center
                center = min_coords + np.array([i + 0.5, j + 0.5, k + 0.5]) * cube_size
                
                # Define cube boundaries
                min_corner = center - cube_size / 2
                max_corner = center + cube_size / 2
                
                # Find points within cube bounds using vectorized operations
                mask = np.all(
                    (points >= min_corner) & 
                    (points <= max_corner),
                    axis=1
                )
                cube_points = points[mask]
                
                # Yield cube even if empty (for regular grid coverage)
                yield cube_points, (center[0], center[1], center[2])


class CubeDataset(Dataset):
    def __init__(self, points: np.ndarray, cube_size: float):
        self.samples = list(get_regular_cubic_samples_faster(points, cube_size))
        self.npoints = 100

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cube_points, coords = self.samples[idx]        
        if len(cube_points) > self.npoints:
            idx = np.random.choice(len(cube_points), self.npoints, replace=False)
            cube_points = cube_points[idx]
        elif len(cube_points) < self.npoints:
            idx = np.random.choice(len(cube_points), self.npoints, replace=True)
            cube_points = cube_points[idx]
            
        cube_points = pc_normalize(cube_points).astype(np.float32)
        return torch.tensor(cube_points, dtype=torch.float32), coords


def create_dataloader(file_path: str, cube_size: int, batch_size: int, shuffle: bool = False) -> DataLoader:
    points = read_off_file(file_path)
    dataset = CubeDataset(points, cube_size)
    print(f"Number of samples: {len(dataset)}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def predict_phases(model, dataloader: DataLoader, device: str = 'cpu') -> np.ndarray:
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
            pred_choice = pred.data.max(1)[1]

            coords = np.array(coords)
            coords = np.moveaxis(coords, 0, 1)
            for coord, label in zip(coords, pred_choice.cpu().numpy()):
                predictions.append([coord[0], coord[1], coord[2], label.item()])
    return np.array(predictions)



def visualize_predictions(predictions, title: str = "Phase Predictions"):
    """Visualize 3D predictions using plotly.
    
    Args:
        predictions: Nx4 array where each row is (x, y, z, prediction)
        title: Plot title
    """
    fig = go.Figure()
    
    for phase in np.unique(predictions[:, 3]):
        mask = predictions[:, 3] == phase
        points = predictions[mask]
        
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(size=5),
            name=f'Phase {int(phase)}'
        ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    
    fig.show()


if __name__ == '__main__':
    from src.cls.lightning_module import PointNetClassifier
    model = PointNetClassifier.load_from_checkpoint('/home/teshbek/Work/PhD/PointCloudMaterials/output/2024-11-21/23-25-55/pointnet-epoch=33-val_acc=0.92.ckpt')
    dataloader = create_dataloader('datasets/Al/inherent_configurations_off/166ps.off', 12, 32)
    predictions = predict_phases(model, dataloader, 'cpu')
    print(predictions[0])


