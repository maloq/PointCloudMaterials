import torch
import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.models.autoencoders.encoders.vn_encoders import VNTransformerEncoder

def get_random_rotation():
    """Generates a random 3x3 rotation matrix."""
    q, r = torch.linalg.qr(torch.randn(3, 3))
    return q

def check_equivariance():
    print("Checking equivariance of VNTransformerEncoder...")
    
    # Instantiate model
    model = VNTransformerEncoder(
        latent_size=128,
        num_points=80,
        size_preset='small' # Use small for speed, logic is same
    )
    model.eval()
    
    # Create dummy input
    batch_size = 4
    num_points = 80
    x = torch.randn(batch_size, num_points, 3) # (B, N, 3)
    
    # Generate random rotation
    R = get_random_rotation().to(x.device) # (3, 3)
    
    # Rotate input: (B, N, 3) @ (3, 3) -> (B, N, 3)
    x_rotated = x @ R.T
    
    # Forward pass on original
    with torch.no_grad():
        z_inv, z_eq, crystallinity = model(x)
        
    # Forward pass on rotated
    with torch.no_grad():
        z_inv_rot, z_eq_rot, crystallinity_rot = model(x_rotated)
        
    # Check Invariance of z_inv
    # z_inv should be identical
    diff_inv = (z_inv - z_inv_rot).abs().max().item()
    print(f"Invariant Feature Difference (Max Abs): {diff_inv:.6f}")
    
    # Check Invariance of crystallinity
    diff_cryst = (crystallinity - crystallinity_rot).abs().max().item()
    print(f"Crystallinity Difference (Max Abs): {diff_cryst:.6f}")
    
    # Check Equivariance of z_eq
    # z_eq: (B, L, 3)
    # z_eq_rot should be z_eq @ R.T
    z_eq_rotated_expected = z_eq @ R.T
    diff_eq = (z_eq_rotated_expected - z_eq_rot).abs().max().item()
    print(f"Equivariant Feature Difference (Max Abs): {diff_eq:.6f}")
    
    # Thresholds
    tol = 1e-4
    if diff_inv < tol and diff_cryst < tol and diff_eq < tol:
        print("\nSUCCESS: Model is equivariant/invariant within tolerance.")
    else:
        print("\nFAILURE: Model violates equivariance/invariance.")

if __name__ == "__main__":
    check_equivariance()
