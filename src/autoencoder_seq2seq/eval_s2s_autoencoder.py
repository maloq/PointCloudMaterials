import sys, os
import numpy as np
import torch

# Ensure the current working directory is in the sys.path.
sys.path.append(os.getcwd())

from src.data_utils.prepare_data import read_off_file
from torch.utils.data import DataLoader
from src.data_utils.data_load import RegularSequenceDataset
from omegaconf import DictConfig
from src.autoencoder_seq2seq.autoencoder_s2s_module import AutoencoderSeq2Seq
from omegaconf import OmegaConf
from hydra import compose, initialize


def create_autoencoder_dataloader_s2s(cfg: DictConfig, file_path: str, shuffle: bool = False) -> DataLoader:
    """
    Create a dataloader for autoencoder inference from an OFF file
    for the seq2seq autoencoder.

    Args:
        cfg: Configuration dictionary.
        file_path: Path to the OFF file.
        shuffle: Whether to shuffle the dataset samples.

    Returns:
        DataLoader containing point cloud samples.
    """
    points = read_off_file(file_path)
    dataset = RegularSequenceDataset(points,
                             sample_shape=cfg.data.sample_shape,
                             size=cfg.data.cube_size if cfg.data.sample_shape == 'cubic' else cfg.data.radius,
                             n_points=cfg.data.num_points,
                             overlap_fraction=cfg.data.overlap_fraction)
    print(f"Number of samples in {cfg.data.sample_shape} dataset: {len(dataset)}")

    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle)


def get_batch_reconstructions(model: AutoencoderSeq2Seq,
                              points: torch.Tensor,
                              n_points: int,
                              device: str = 'cpu') -> tuple[np.ndarray, np.ndarray]:
    """
    Generate original and reconstructed point clouds for a batch using the seq2seq autoencoder.

    Args:
        model: An instance of PointNetAutoencoderSeq2Seq.
        points: A batch of input point clouds of shape (B, N, 3).
        n_points: The number of points per sample.
        device: The device on which to perform inference (e.g., 'cpu' or 'cuda').

    Returns:
        A tuple (original_points, reconstructed_points) where each is a numpy array.
    """
    model.eval()
    model.to(device)
    print(f"Device: {device}")
    with torch.no_grad():
        # For the seq2seq autoencoder, the input is expected as (B, N, 3) so no transpose is needed.
        points = points.to(device)
        reconstructed, _ = model(points)
        
        print(f"Input points shape: {points.shape}")
        print(f"Reconstructed shape: {reconstructed.shape}")

        points_np = points.cpu().numpy()
        reconstructed_np = reconstructed.cpu().numpy()

        original_points = []
        reconstructed_points = []
        
        # Append an index to each point for downstream processing
        for idx, (orig, recon) in enumerate(zip(points_np, reconstructed_np)):
            orig_sample = []
            recon_sample = []
            
            for point in orig:
                orig_sample.append([point[0], point[1], point[2], idx])
            for point in recon:
                recon_sample.append([point[0], point[1], point[2], idx])
                
            original_points.append(orig_sample)
            reconstructed_points.append(recon_sample)
        
        result_orig = np.array(original_points)
        result_recon = np.array(reconstructed_points)
        print(f"Original points shape after processing: {result_orig.shape}")
        print(f"Reconstructed points shape after processing: {result_recon.shape}")

    return result_orig, result_recon


if __name__ == '__main__':
    model = AutoencoderSeq2Seq.load_from_checkpoint('output/2025-02-24/00-37-30/pointnet-epoch=59-val_loss=0.19.ckpt')
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="s2s_autoencoder")
        
    file_path = 'datasets/Al/inherent_configurations_off/240ps.off'

    dataloader = create_autoencoder_dataloader(cfg, file_path)
    points = next(iter(dataloader))[0]
    
    original_points, reconstructed_points = get_batch_reconstructions(model, points, cfg.data.num_points)
    print(original_points.shape, reconstructed_points.shape)
