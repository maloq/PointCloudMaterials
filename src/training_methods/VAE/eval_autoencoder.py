import sys, os
import numpy as np
import torch
sys.path.append(os.getcwd())
from src.data_utils.prepare_data import read_off_file
from torch.utils.data import DataLoader
from src.data_utils.data_load import PointCloudDataset
from omegaconf import DictConfig
from src.training_methods.VAE.autoencoder_module import PointNetAutoencoder
from src.utils.model_utils import load_model_from_checkpoint
from hydra import compose, initialize
from torch.utils.data import Subset


def create_autoencoder_dataloader(cfg: DictConfig, file_path, shuffle: bool = False, max_samples = None, return_coords: bool = False) -> DataLoader:
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
    
    dataset = PointCloudDataset(
                 root=cfg.data.data_path,
                 data_files=file_paths,
                 return_coords=return_coords,
                 sample_type='regular',
                 radius=cfg.data.radius,
                 overlap_fraction=cfg.data.overlap_fraction,
                 n_samples=cfg.data.n_samples,
                 num_points=cfg.data.num_points,
                 pre_normalize=True,
                 normalize=True)
    
    print(f"Number of samples in dataset: {len(dataset)}")
    
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
    

def get_config_path_from_checkpoint(checkpoint_path):

    config_path_chekpoint = os.path.join(*checkpoint_path.split('/')[:-1], '.hydra')
    config_name = checkpoint_path.split('/')[-1].split('.')[0]
    
    if not os.path.exists(config_path_chekpoint):
        config_path = '../../configs' if 'src' in os.getcwd() else 'configs' 
        print(f"Config in {config_path_chekpoint} not found, using default location {config_path}")
    else:
        config_path =  config_path_chekpoint
        config_name = 'config'

    return config_path, config_name



def load_model_and_config(checkpoint_path: str,
                          cuda_device: int = 0,
                          fallback_config_path: str = None):
    """Load Hydra config, restore the model from *checkpoint_path* and return
    the model instance together with the resolved *cfg* object and chosen
    *device* string.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        cuda_device (int): GPU index to place the model on.
        config_name (str): Name of the Hydra config file (without extension).

    Returns:
        Tuple[torch.nn.Module, omegaconf.DictConfig, str]:
            Restored model, Hydra configuration and device string.
    """

    if fallback_config_path is None:
        config_path, resolved_config_name = get_config_path_from_checkpoint(checkpoint_path)
    else:
        config_path = os.path.join(*fallback_config_path.split('/')[:-1])
        resolved_config_name = fallback_config_path.split('/')[-1].split('.')[0]

    # Load Hydra config
    with initialize(version_base=None, config_path='../../' + config_path):
        print(f"Loading config from {config_path}/{resolved_config_name}")
        cfg = compose(config_name=resolved_config_name)

    # Select device
    device = f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = load_model_from_checkpoint(checkpoint_path, cfg, device=device, module=PointNetAutoencoder)

    if model is None:
        raise RuntimeError("Failed to load the model.")

    return model, cfg, device


def autoencoder_predict_from_dataloader(model, dataloader, device: str = 'cpu', dataloader_returns_coords: bool = False):
    if len(dataloader) == 0:
        raise ValueError("Dataloader is empty - no data to process")
        
    latents, point_clouds, originals = [], [], []
    for batch in dataloader:
        points = batch[0].to(device).transpose(2, 1)
        
        with torch.no_grad():
            point_cloud, latent, _ = model(points)
        
        latents.append(latent.cpu().numpy())
        point_clouds.append(point_cloud.cpu().numpy())
        originals.append(points.cpu().numpy())

    return (np.concatenate(latents, axis=0), 
            np.concatenate(point_clouds, axis=0), 
            np.concatenate(originals, axis=0))


def autoencoder_predict_latent(points, model, device: str = 'cpu'):
    model.eval()
    model.to(device)
    if isinstance(points, torch.Tensor):
        points = points.to(device)
    else:
        points = torch.tensor(points, dtype=torch.float32, device=device)
    
    # Add batch dimension if it's a single point cloud
    if points.dim() == 2:
        points = points.unsqueeze(0)

    points_transposed = points.transpose(1, 2)
    with torch.no_grad():
        _, latent, _ = model(points_transposed)
    
    if latent.size(0) == 1:
        latent = latent.squeeze(0)

    return latent.cpu().numpy()


if __name__ == '__main__':
    model = PointNetAutoencoder.load_from_checkpoint('output/2025-01-28/23-56-42/pointnet-epoch=149-val_loss=0.26.ckpt')
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="Al_autoencoder")
    file_path = 'datasets/Al/inherent_configurations_off/240ps.off'

    dataloader = create_autoencoder_dataloader(cfg, file_path)
    points = next(iter(dataloader))[0]
    original_points, reconstructed_points = get_batch_reconstructions(model, points, cfg.data.num_points)
    print(original_points.shape, reconstructed_points.shape)

