import sys, os
import numpy as np
import torch
sys.path.append(os.getcwd())
from src.data_utils.prepare_data import read_off_file
from torch.utils.data import DataLoader
from src.data_utils.data_load import RegularDataset
from omegaconf import DictConfig
from src.autoencoder.autoencoder_module import PointNetAutoencoder
from omegaconf import OmegaConf
from hydra import compose, initialize
from torch.utils.data import Subset


def create_autoencoder_dataloader(cfg: DictConfig, file_path, shuffle: bool = False, max_samples = None) -> DataLoader:
    """Create a dataloader for autoencoder inference from OFF file(s).
    
    Args:
        cfg: Configuration dictionary 
        file_path: Path to the OFF file or list of paths to OFF files
        shuffle: Whether to shuffle the samples
        max_samples: Maximum number of samples to include
    Returns:
        DataLoader containing point cloud samples
    """
    # Handle both single path and list of paths
    if isinstance(file_path, str):
        file_paths = [file_path]
    elif isinstance(file_path, (list, tuple)):
        file_paths = file_path
    else:
        raise ValueError("file_path must be a string or a list/tuple of strings")
    
    # Read and combine points from all files
    all_points = []
    for path in file_paths:
        points = read_off_file(path)
        all_points.append(points)
        print(f"Loaded {len(points)} points from {path}")
    
    # Concatenate all points
    combined_points = np.concatenate(all_points, axis=0)
    print(f"Total combined points: {len(combined_points)}")
    
    dataset = RegularDataset(combined_points,
                             sample_shape=cfg.data.sample_shape,
                             size= cfg.data.cube_size if cfg.data.sample_shape == 'cubic' else cfg.data.radius,
                             n_points=cfg.data.num_points,
                             overlap_fraction=cfg.data.overlap_fraction)
    print(f"Number of samples in {cfg.data.sample_shape} dataset: {len(dataset)}")
    
    if max_samples:
        dataset = Subset(dataset, list(range(max_samples)))
        print(f"Dataset limited to {len(dataset)}")
    
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle)



def get_batch_reconstructions(model: PointNetAutoencoder,
                             points: torch.Tensor,
                             n_points: int,
                             device: str = 'cpu') -> tuple[np.ndarray, np.ndarray]:
    original_points = []
    reconstructed_points = []
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        # Prepare input
        points = points.to(device)
        points_transposed = points.transpose(2, 1)  # (B, N, 3) -> (B, 3, N)
        
        # Get reconstruction
        reconstructed, _ = model(points_transposed)
        # reconstructed is (B, 3, N), we need to handle this shape
        
        print(f"Input points shape: {points.shape}")
        print(f"Reconstructed shape: {reconstructed.shape}")
        
        points_np = points.cpu().numpy()
        reconstructed_np = reconstructed.cpu().numpy()
        original_points = []
        reconstructed_points = []
        
        for idx, (orig, recon) in enumerate(zip(points_np, reconstructed_np)):
            orig_sample = []
            recon_sample = []
            
            for point in orig:
                orig_sample.append([point[0], point[1], point[2], idx])
                
            for point in recon:  # Transpose here to get correct point ordering
                recon_sample.append([point[0], point[1], point[2], idx])
            
            original_points.append(orig_sample)
            reconstructed_points.append(recon_sample)
    
    result_orig = np.array(original_points)
    result_recon = np.array(reconstructed_points)
    print(f"Original points shape after processing: {result_orig.shape}")
    print(f"Reconstructed points shape after processing: {result_recon.shape}")
            
    return result_orig, result_recon
    


if __name__ == '__main__':
    model = PointNetAutoencoder.load_from_checkpoint('output/2025-01-28/23-56-42/pointnet-epoch=149-val_loss=0.26.ckpt')
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="Al_autoencoder")
    file_path = 'datasets/Al/inherent_configurations_off/240ps.off'

    dataloader = create_autoencoder_dataloader(cfg, file_path)
    points = next(iter(dataloader))[0]
    original_points, reconstructed_points = get_batch_reconstructions(model, points, cfg.data.num_points)
    print(original_points.shape, reconstructed_points.shape)

