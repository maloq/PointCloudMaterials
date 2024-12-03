
import sys,os
import numpy as np
import plotly.graph_objects as go
import torch
sys.path.append(os.getcwd())
from src.data_utils.prepare_data import read_off_file
from src.data_utils.data_load import pc_normalize
from torch.utils.data import Dataset, DataLoader
from typing import Iterator, Tuple
from scipy.spatial import KDTree
from src.data_utils.prepare_data import get_regular_cubic_samples, get_regular_spheric_samples
from omegaconf import DictConfig, OmegaConf



class CubeDataset(Dataset):
    def __init__(self, points: np.ndarray, cube_size: float, n_points: int = 128):
        self.samples = get_regular_cubic_samples(points, cube_size, n_points=n_points, return_coords=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cube_points, coords = self.samples[idx]        
        cube_points = pc_normalize(cube_points).astype(np.float32)
        return torch.tensor(cube_points, dtype=torch.float32), coords


class SphericDataset(Dataset):
    def __init__(self, points: np.ndarray, radius: float, n_points: int = 128):
        self.samples = get_regular_spheric_samples(points, radius=radius, n_points=n_points, return_coords=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        spheres, coords = self.samples[idx]        
        spheres = pc_normalize(spheres).astype(np.float32)
        return torch.tensor(spheres, dtype=torch.float32), coords
    

def create_dataloader(cfg: DictConfig, file_path: str, shuffle: bool = False) -> DataLoader:

    points = read_off_file(file_path)
    if cfg.data.sample_shape == 'cubic':
        dataset = CubeDataset(points,
                              cube_size=cfg.data.cube_size,
                              n_points=cfg.data.num_points)
        print(f"Number of samples in cubic dataset: {len(dataset)}")
    elif cfg.data.sample_shape == 'spheric':
        dataset = SphericDataset(points,
                                 radius=cfg.data.radius,
                                 n_points=cfg.data.num_points)
        print(f"Number of samples in spheric dataset: {len(dataset)}")
    else:
        raise ValueError(f"Invalid sample type: {cfg.data.sample_shape}")
    
    return DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=shuffle)


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




if __name__ == '__main__':
    from src.cls.lightning_module import PointNetClassifier
    model = PointNetClassifier.load_from_checkpoint('/home/teshbek/Work/PhD/PointCloudMaterials/output/2024-11-21/23-25-55/pointnet-epoch=33-val_acc=0.92.ckpt')
    dataloader = create_dataloader('datasets/Al/inherent_configurations_off/166ps.off', 12, 32)
    predictions = predict_phases(model, dataloader, 'cpu')
    print(predictions[0])


