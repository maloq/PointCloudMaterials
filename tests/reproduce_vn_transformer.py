import torch
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.models.autoencoders.encoders.vn_encoders import VNTransformerEncoder

def test_vn_transformer_large():
    print("Testing VNTransformerEncoder with 'large' preset...")
    
    # Instantiate model
    try:
        model = VNTransformerEncoder(
            latent_size=513, # Divisible by 3
            num_points=80,
            size_preset='large'
        )
        print("Model instantiated successfully.")
    except Exception as e:
        print(f"Failed to instantiate model: {e}")
        return

    # Create dummy input
    batch_size = 4
    num_points = 80
    x = torch.randn(batch_size, num_points, 3) # (B, N, 3)
    
    # Forward pass
    try:
        z_inv, z_eq, crystallinity = model(x)
        print("Forward pass successful.")
        print(f"z_inv shape: {z_inv.shape}")
        print(f"z_eq shape: {z_eq.shape}")
        print(f"crystallinity shape: {crystallinity.shape}")
        
        assert z_inv.shape == (batch_size, 513)
        assert z_eq.shape == (batch_size, 513 // 3, 3)
        assert crystallinity.shape == (batch_size, 1)
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vn_transformer_large()
