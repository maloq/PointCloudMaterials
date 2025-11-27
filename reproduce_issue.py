
import torch
import math
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.models.autoencoders.decoders.vn_crystal import VNCrystalDecoder

def check_planarity(points):
    # points: (N, 3)
    # Center points
    points = points - points.mean(dim=0)
    # SVD
    u, s, v = torch.linalg.svd(points)
    # If smallest singular value is close to 0, it's planar
    print(f"Singular values: {s}")
    return s[-1] < 1e-4

def test_decoder():
    num_points = 80
    latent_size = 513 # From config
    
    decoder = VNCrystalDecoder(
        num_points=num_points,
        latent_size=latent_size,
        hidden_dims=(1024, 512, 256),
        correction_mode='none'
    )
    
    # Check grid
    grid = decoder.base_grid
    print(f"Grid shape: {grid.shape}")
    print(f"Grid x values: {grid[:, 0].unique()}")
    print(f"Grid y values: {grid[:, 1].unique()}")
    print(f"Grid z values: {grid[:, 2].unique()}")
    
    if len(grid[:, 0].unique()) == 1:
        print("ISSUE: Grid x is constant!")
        
    # Check grid isotropy
    grid_centered = grid - grid.mean(dim=0)
    _, s_grid, _ = torch.linalg.svd(grid_centered)
    print(f"Grid singular values: {s_grid}")
    
    # Random input
    B = 2
    z = torch.randn(B, latent_size//3, 3)
    
    # Forward
    pts = decoder(z)
    
    print(f"Output shape: {pts.shape}")
    
    is_planar = check_planarity(pts[0])
    if is_planar:
        print("ISSUE: Output is planar!")
    else:
        print("Output is 3D.")

if __name__ == "__main__":
    test_decoder()
