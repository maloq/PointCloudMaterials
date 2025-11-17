import torch
import os
from typing import Tuple, Optional
from omegaconf import DictConfig, ListConfig
from omegaconf.base import ContainerMetadata


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


def load_supervised_checkpoint(checkpoint_path: str, encoder, rot_net=None):
    """
    Load pretrained supervised encoder and rotation network from checkpoint.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        encoder: Encoder model to load weights into
        rot_net: Optional rotation network to load weights into
        
    Returns:
        dict: The loaded checkpoint dictionary
    """
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Supervised checkpoint not found at {checkpoint_path}")
        return None

    print(f"Loading supervised checkpoint from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = None
    # Prefer safe loading with allowlisted OmegaConf globals (PyTorch >= 2.6)
    safe_ctx = getattr(getattr(torch, "serialization", object), "safe_globals", None)
    if callable(safe_ctx) and (DictConfig is not None or ListConfig is not None or ContainerMetadata is not None):
        allowlist = [c for c in (DictConfig, ListConfig, ContainerMetadata) if c is not None]
        try:
            with torch.serialization.safe_globals(allowlist):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except Exception as e:
            print(f"Safe checkpoint load failed ({e}). Falling back to full unpickling.")

    if checkpoint is None:
        # Last resort: allow full unpickling. Only do this for trusted checkpoints.
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except TypeError:
            # Older torch without weights_only argument
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract encoder state dict
    encoder_state = {}
    for key, value in checkpoint['state_dict'].items():
        if key.startswith('encoder.'):
            new_key = key.replace('encoder.', '')
            encoder_state[new_key] = value

    # Load encoder weights
    if encoder_state:
        missing_keys, unexpected_keys = encoder.load_state_dict(encoder_state, strict=False)
        print(f"Loaded encoder weights from supervised checkpoint")
        if missing_keys:
            print(f"  Missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
        if unexpected_keys:
            print(f"  Unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
    else:
        print("Warning: No encoder weights found in checkpoint")

    # Extract rotation network state dict (if using rot_head mode)
    if rot_net is not None:
        rot_net_state = {}
        for key, value in checkpoint['state_dict'].items():
            if key.startswith('rot_net.'):
                new_key = key.replace('rot_net.', '')
                rot_net_state[new_key] = value

        # Load rotation network weights
        if rot_net_state:
            missing_keys, unexpected_keys = rot_net.load_state_dict(rot_net_state, strict=False)
            print(f"Loaded rotation network weights from supervised checkpoint")
            if missing_keys:
                print(f"  Missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                print(f"  Unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
        else:
            print("Warning: No rotation network weights found in checkpoint")

    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        print(f"Checkpoint hyperparameters:")
        print(f"  Encoder: {hparams.get('encoder', {}).get('name', 'unknown')}")
        print(f"  Latent size: {hparams.get('latent_size', 'unknown')}")
        print(f"  Learning rate: {hparams.get('learning_rate', 'unknown')}")

    return checkpoint


def find_best_supervised_checkpoint(cfg) -> Optional[str]:
    """
    Auto-discover best supervised checkpoint from lightning_logs directory.

    Args:
        cfg: Configuration object (unused but kept for compatibility)

    Returns:
        Path to the most recent checkpoint, or None if not found
    """
    # Try to find checkpoints directory
    base_dirs = ['lightning_logs', 'outputs/supervised_encoder']

    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue

        # Find all checkpoint files
        checkpoint_files = []
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.ckpt'):
                    checkpoint_files.append(os.path.join(root, file))

        if not checkpoint_files:
            continue

        # Sort by modification time and return the most recent
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        print(f"Auto-discovered checkpoint: {checkpoint_files[0]}")
        return checkpoint_files[0]

    return None