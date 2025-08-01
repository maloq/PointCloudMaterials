import torch
import os
from typing import Tuple


def resolve_config_path(checkpoint_path: str) -> Tuple[str, str]:
    """Return *(config_dir, config_name)* guessed from *checkpoint_path*."""
    hydra_dir = os.path.join(os.path.dirname(checkpoint_path), ".hydra")
    if os.path.isdir(hydra_dir):
        return hydra_dir, "config"
    else:
        print(f"No hydra directory found in {checkpoint_path}, using default config")
        return None, None


def load_model_from_checkpoint(checkpoint_path, cfg, device='cpu', module=None):
    """
    Load a PointNetAutoencoder model from a checkpoint.
    
    This function handles the proper loading of a PointNetAutoencoder model from a checkpoint.
    It creates a new model instance, and then loads the state dictionary from the checkpoint.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        device (str): Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        PointNetAutoencoder: The loaded model
    """
    
    if module is None:
        from src.training_methods.autoencoder.autoencoder_module import PointNetAutoencoder as module
    model = None
    try:
        model = module(cfg)
    except Exception:
        model = module()

    # With PyTorch >=2.6 the default for `weights_only` became `True`,
    # which breaks loading checkpoints that contain objects beyond plain
    # tensors (e.g., Hydra configurations stored by Lightning).
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint


    stripped_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[6:] if k.startswith('model.') else k
        stripped_state_dict[new_key] = v

    model.load_state_dict(stripped_state_dict, strict=False)
    print("✅ Loaded checkpoint by manually restoring state_dict")

    model = model.to(device)
    model.eval()

    try:
        print(f"Model loaded successfully from {checkpoint_path}")
        if hasattr(cfg, 'type'):
            print(f"Model type: {cfg.type}")
        if hasattr(cfg, 'latent_size'):
            print(f"Latent size: {cfg.latent_size}")
    except Exception:
        # cfg might be a plain dict or have no such attributes – ignore.
        pass

    return model