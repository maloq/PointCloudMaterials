import torch
import os
from typing import Tuple


def resolve_config_path(checkpoint_path: str) -> Tuple[str, str]:
    """Return *(config_dir, config_name)* guessed from *checkpoint_path*."""
    if "eval_results" in checkpoint_path:
        checkpoint_path = os.path.join(*checkpoint_path.split("/")[:-1])
        print(f"Eval results checkpoint path, using {checkpoint_path} as config path")
        assert os.path.isdir(checkpoint_path), f"Config path {checkpoint_path} is not a directory"
        assert os.path.isfile(os.path.join(checkpoint_path, "eval_results.yaml")), f"Config file not found in {checkpoint_path}"
        return checkpoint_path,  "eval_results"
    else:
        hydra_dir = os.path.join(os.path.dirname(checkpoint_path), ".hydra")
        if os.path.isdir(hydra_dir):
            return hydra_dir, "config"
        else:
            raise ValueError(f"No hydra directory found in {checkpoint_path}, put default config in .hydra/config.yaml")


def load_model_from_checkpoint(checkpoint_path, cfg, device='cpu', module=None):
    """
    Load a ShapePoseDisentanglement model from a checkpoint.
    
    This function handles the proper loading of a ShapePoseDisentanglement model from a checkpoint.
    It creates a new model instance, and then loads the state dictionary from the checkpoint.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        device (str): Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        ShapePoseDisentanglement: The loaded model
    """
    
    if module is None:
        from src.training_methods.spd.spd_module import ShapePoseDisentanglement as module
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