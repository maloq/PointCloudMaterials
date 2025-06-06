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
    
    # Create a new model instance with the configuration
    if module is None:
        from src.autoencoder.autoencoder_module import PointNetAutoencoder
        model = PointNetAutoencoder(cfg)
    else:
        model = module(cfg)
    
    # Load the state dictionary from the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    
    # Move model to the specified device
    model = model.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    print(f"Model loaded successfully from {checkpoint_path}")
    print(f"Model type: {cfg.type}")
    print(f"Latent size: {cfg.latent_size}")
    return model