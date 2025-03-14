
import sys,os
import numpy as np
import plotly.graph_objects as go
import torch
sys.path.append(os.getcwd())
from src.data_utils.prepare_data import read_off_file

from torch.utils.data import Dataset, DataLoader
from src.data_utils.data_load import RegularDataset
from omegaconf import DictConfig, OmegaConf



def create_dataloader(cfg: DictConfig, file_path: str, shuffle: bool = False) -> DataLoader:

    points = read_off_file(file_path)
    dataset = RegularDataset(points,
                             sample_shape=cfg.data.sample_shape,
                             size=cfg.data.cube_size if cfg.data.sample_shape == 'cubic' else cfg.data.radius,
                             n_points=cfg.data.num_points,
                             overlap_fraction=cfg.data.overlap_fraction)
    print(f"Number of samples in {cfg.data.sample_shape} dataset: {len(dataset)}")
    
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle)


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

