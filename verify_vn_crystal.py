import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.models.autoencoders.decoders.vn_crystal import VNCrystalDecoder
from src.training_methods.equivariant_autoencoder.eq_ae_module import EquivariantAutoencoder
from omegaconf import OmegaConf

def get_random_rotation():
    """Generates a random 3x3 rotation matrix."""
    q = torch.randn(4)
    q = q / torch.norm(q)
    w, x, y, z = q
    R = torch.tensor([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    return R

def verify_decoder():
    print("Verifying VNCrystalDecoder...")
    
    B = 4
    latent_size = 60 # VN channels
    num_points = 100
    
    decoder = VNCrystalDecoder(num_points=num_points, latent_size=latent_size)
    
    # Create random equivariant latent (B, C, 3)
    z_eq = torch.randn(B, latent_size, 3)
    
    # Forward pass
    pts = decoder(z_eq)
    print(f"Output shape: {pts.shape}")
    assert pts.shape == (B, num_points, 3)
    
    # Equivariance Test
    print("Testing Equivariance...")
    R = get_random_rotation()
    
    # Rotate input latent: (B, C, 3) @ R^T -> (B, C, 3)
    # Each channel is a vector that rotates with the frame
    z_eq_rot = z_eq @ R.T
    
    # Pass rotated latent
    pts_rot_pred = decoder(z_eq_rot)
    
    # Rotate original output
    pts_rot_gt = pts @ R.T
    
    # Check difference
    diff = torch.abs(pts_rot_pred - pts_rot_gt).max()
    print(f"Max difference (Equivariance Error): {diff.item()}")
    
    if diff < 1e-5:
        print("Equivariance Test PASSED!")
    else:
        print("Equivariance Test FAILED!")

def verify_integration():
    print("\nVerifying Integration into EquivariantAutoencoder...")
    
    cfg = OmegaConf.create({
        "encoder": {
            "name": "PnE_VN",
            "kwargs": {
                "latent_size": 60,
                "n_knn": 20
            }
        },
        "decoder": {
            "name": "VN_Crystal",
            "kwargs": {
                "num_points": 100,
                "latent_size": 60
            }
        },
        "loss": "chamfer",
        "kl_latent_loss_scale": 0.0,
        "sinkhorn_blur_schedule": {
            "start": 0.1,
            "end": 0.001,
            "steps": 10,
            "start_epoch": 0,
            "end_epoch": 10,
            "duration_epochs": 10,
            "enable": True
        }
    })
    
    model = EquivariantAutoencoder(cfg)
    
    # Dummy input
    pc = torch.randn(4, 100, 3)
    
    # Forward pass
    inv_z, recon, eq_z, diff_loss = model(pc)
    
    print(f"Recon shape: {recon.shape}")
    assert recon.shape == (4, 100, 3)
    print("Integration Test PASSED!")

if __name__ == "__main__":
    verify_decoder()
    verify_integration()
