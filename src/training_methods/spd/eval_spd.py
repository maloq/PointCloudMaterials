import sys, os
import numpy as np
import torch
sys.path.append(os.getcwd())
from src.data_utils.prepare_data import read_off_file
from torch.utils.data import DataLoader
from src.data_utils.data_load import PointCloudDataset
from omegaconf import DictConfig
from src.training_methods.spd.spd_module import ShapePoseDisentanglement
from src.training_methods.autoencoder.eval_autoencoder import create_autoencoder_dataloader
from src.utils.model_utils import load_model_from_checkpoint
from hydra import compose, initialize
from torch.utils.data import Subset


def get_batch_reconstructions(model: ShapePoseDisentanglement,
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

        # Ensure the tensor has a batch dimension.  The dataloader already
        # provides (B, N, 3) but if the caller passes a single sample tensor
        # shaped (N, 3) we add the missing batch dimension so the model sees
        # (1, N, 3).
        if points.dim() == 2:
            points = points.unsqueeze(0)
        inv_z, recon, cano, rot = model(points)
        # reconstructed is (B, 3, N), we need to handle this shape
        
        print(f"Input points shape: {points.shape}")
        print(f"Reconstructed shape: {recon.shape}")
        
        points_np = points.cpu().numpy()
        reconstructed_np = recon.cpu().numpy()
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



def load_spd_model_and_config(checkpoint_path: str,
                          cuda_device: int = 0,
                          model_class: str = 'Autoencoder',
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
    with initialize(version_base=None, config_path='../../../' + config_path):
        print(f"Loading config from {config_path}/{resolved_config_name}")
        cfg = compose(config_name=resolved_config_name)

    # Select device
    device = f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = load_model_from_checkpoint(checkpoint_path, cfg, device=device, module=ShapePoseDisentanglement)

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


def spd_predict_latent(points, model, device: str = 'cpu'):
    """Return the *invariant* latent vector (inv_z) predicted by the SPD model.

    The Shape-Pose Disentanglement model expects point clouds shaped
    (B, N, 3).  For convenience this helper will also accept the common
    (B, 3, N) layout and transpose it automatically.

    Args:
        points: A single point cloud (N, 3) or batch of point clouds in one
            of the following shapes: (B, N, 3), (N, 3), (B, 3, N) or (3, N).
        model: A ``ShapePoseDisentanglement`` instance.
        device: Device string on which to run the prediction.

    Returns:
        np.ndarray: The predicted ``inv_z`` latent code.  If a single sample
        is provided the batch dimension is squeezed.
    """
    # Prepare model
    model.eval()
    model.to(device)

    # Convert input to ``torch.Tensor`` and move to the chosen device.
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32)
    points = points.to(device)

    # Ensure a batch dimension is present.
    if points.dim() == 2:
        points = points.unsqueeze(0)  # (1, N, 3) or (1, 3, N)

    # If points are provided in (B, 3, N) format, convert to (B, N, 3).
    if points.dim() == 3 and points.shape[1] == 3:
        points = points.permute(0, 2, 1)  # (B, 3, N) -> (B, N, 3)

    # Forward pass through the model.
    with torch.no_grad():
        inv_z, _, _, _ = model(points)

    # Squeeze batch dimension when processing a single sample.
    if inv_z.size(0) == 1:
        inv_z = inv_z.squeeze(0)

    return inv_z.cpu().numpy()
 

if __name__ == '__main__':
    model = ShapePoseDisentanglement.load_from_checkpoint('output/2025-07-28/16-47-38/SPD_FoldingSphereAttn_l72_P80_Sinkhorn_512-epoch=09-val_loss=0.02.ckpt')
    with initialize(version_base=None, config_path="../../../configs"):
        cfg = compose(config_name="spd")
    file_path = '240ps.off'

    dataloader = create_autoencoder_dataloader(cfg, file_path)
    points = next(iter(dataloader))[0]
    original_points, reconstructed_points = get_batch_reconstructions(model, points, cfg.data.num_points)
    print(original_points.shape, reconstructed_points.shape)
    latent = spd_predict_latent(points, model)
    print(latent.shape)

