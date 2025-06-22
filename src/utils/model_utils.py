import torch


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
    
    # ------------------------------------------------------------------
    # 1. Determine the module class to instantiate
    # ------------------------------------------------------------------
    if module is None:
        # Fallback to PointNetAutoencoder if no specific module is provided
        from src.autoencoder.autoencoder_module import PointNetAutoencoder as module
    model = None
    # Instantiate the module *manually*
    try:
        model = module(cfg)
    except Exception:
        # In case the ctor doesn't accept cfg, try without it.
        model = module()

    # With PyTorch >=2.6 the default for `weights_only` became `True`,
    # which breaks loading checkpoints that contain objects beyond plain
    # tensors (e.g., Hydra configurations stored by Lightning).
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Determine the right key that contains the actual weights.
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # It might *just* be a state-dict already.
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Some checkpoints prefix parameters with "model.". Remove that
    # prefix so that it matches our module's keys.
    stripped_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[6:] if k.startswith('model.') else k
        stripped_state_dict[new_key] = v

    # Finally, load the weights (allowing missing keys for flexibility).
    model.load_state_dict(stripped_state_dict, strict=False)
    print("✅ Loaded checkpoint by manually restoring state_dict")

    # ------------------------------------------------------------------
    # 4. Device placement & evaluation mode
    # ------------------------------------------------------------------
    model = model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # 5. Friendly logging
    # ------------------------------------------------------------------
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