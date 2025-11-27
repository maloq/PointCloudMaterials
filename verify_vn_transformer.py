
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from models.autoencoders.encoders.vn_encoders import VNTransformerEncoder

def test_vn_transformer():
    print("Testing VNTransformerEncoder...")
    
    # Test default
    print("Testing default hidden_dim=64")
    model = VNTransformerEncoder(latent_size=128, num_points=80, n_knn=20)
    x = torch.randn(2, 3, 80) # B, 3, N
    z_inv, z_eq, cryst = model(x)
    print(f"Output shapes: z_inv={z_inv.shape}, z_eq={z_eq.shape}, cryst={cryst.shape}")
    assert z_inv.shape == (2, 128)
    assert z_eq.shape == (2, 128//3, 3) # 42, 3
    
    # Test custom hidden_dim
    print("Testing custom hidden_dim=32")
    model = VNTransformerEncoder(latent_size=128, num_points=80, n_knn=20, hidden_dim=32)
    z_inv, z_eq, cryst = model(x)
    print(f"Output shapes: z_inv={z_inv.shape}, z_eq={z_eq.shape}, cryst={cryst.shape}")
    assert z_inv.shape == (2, 128)
    
    print("Testing custom hidden_dim=128")
    model = VNTransformerEncoder(latent_size=128, num_points=80, n_knn=20, hidden_dim=128)
    z_inv, z_eq, cryst = model(x)
    print(f"Output shapes: z_inv={z_inv.shape}, z_eq={z_eq.shape}, cryst={cryst.shape}")
    assert z_inv.shape == (2, 128)

    print("All tests passed!")

if __name__ == "__main__":
    test_vn_transformer()
