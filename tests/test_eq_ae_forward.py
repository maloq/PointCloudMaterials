"""
Test script for Equivariant Autoencoder forward pass and overfitting capability.

Run with:
    python tests/test_eq_ae_forward.py

This tests:
1. Model instantiation with different decoder types
2. Forward pass works correctly
3. Model can overfit on a small batch (reconstruction capability)
"""

from __future__ import annotations

import sys
import os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf


def create_test_config(decoder_name: str = "VN_Equivariant", latent_size: int = 96, num_points: int = 128):
    """Create a minimal test configuration."""
    
    decoder_configs = {
        "VN_Equivariant": {
            "name": "VN_Equivariant",
            "kwargs": {
                "num_points": num_points,
                "latent_size": latent_size,
                "hidden_dims": [512, 256, 128],
                "use_batchnorm": False,  # Disable BN to allow larger outputs
                "negative_slope": 0.1,
                "output_scale": 10.0,  # Scale up outputs
                "learnable_scale": True,
            }
        },
        "VN_Snowflake": {
            "name": "VN_Snowflake",
            "kwargs": {
                "num_points": num_points,
                "latent_size": latent_size,
                "num_seeds": 8,  # 8 * 2^4 = 128 points
                "hidden_channels": 32,
                "k": 6,
                "use_batchnorm": True,
                "negative_slope": 0.1,
                "temperature": 1.0,
                "residual_scale": 0.5,
                "disp_scale": 1.0,
                "feature_clamp_max": 500.0,
                "final_disp_scale": 1.0,
                "use_tanh_displacement": False,
                "progressive_disp_scale": False,
            }
        },
        "MLPLarge": {
            "name": "MLPLarge",
            "kwargs": {
                "num_points": num_points,
                "latent_size": latent_size,
                "hidden_dims": [512, 512, 256],
                "use_batchnorm": True,
                "dropout_rate": 0.0,
            }
        },
    }
    
    cfg_dict = {
        "project_name": "test",
        "experiment_name": f"test_{decoder_name}",
        "model_type": "spd",
        "latent_size": latent_size,
        "encoder": {
            "name": "VN_DGCNN_Refined",
            "kwargs": {
                "latent_size": latent_size,
                "feature_dims": [64, 64, 128, 256, 256],
                "pooling": "max",
            }
        },
        "decoder": decoder_configs[decoder_name],
        "loss": "chamfer",
        "loss_params": {
            "chamfer": {
                "point_reduction": "mean",
            }
        },
        "learning_rate": 0.001,
        "batch_size": 4,
        "epochs": 100,
        "kl_latent_loss_scale": 0.0,
        "scheduler_name": "Cosine",
        "scheduler_gamma": 0.5,
        "gradient_clip_val": 1.0,
        "sinkhorn_blur": 0.1,
        "sinkhorn_blur_schedule": {
            "enable": False,
            "start": 0.1,
            "end": 0.01,
            "start_epoch": 0,
            "duration_epochs": 100,
        },
        "data": {
            "num_points": num_points,
        },
    }
    
    return OmegaConf.create(cfg_dict)


def create_random_point_cloud(batch_size: int, num_points: int, device: str = "cpu"):
    """Create random point clouds normalized to unit sphere."""
    # Random points in a sphere
    points = torch.randn(batch_size, num_points, 3, device=device)
    # Normalize to unit sphere
    norms = torch.norm(points, dim=-1, keepdim=True)
    points = points / norms.clamp(min=1e-6)
    # Scale randomly
    scales = torch.rand(batch_size, 1, 1, device=device) * 0.5 + 0.5
    points = points * scales
    return points


def test_forward_pass(decoder_name: str, device: str = "cpu"):
    """Test that forward pass works without errors."""
    from src.training_methods.equivariant_autoencoder.eq_ae_module import EquivariantAutoencoder
    
    print(f"\n{'='*60}")
    print(f"Testing forward pass with {decoder_name} decoder")
    print('='*60)
    
    cfg = create_test_config(decoder_name)
    model = EquivariantAutoencoder(cfg).to(device)
    model.eval()
    
    batch_size = 4
    num_points = cfg.data.num_points
    pc = create_random_point_cloud(batch_size, num_points, device)
    
    with torch.no_grad():
        inv_z, recon, eq_z = model(pc)
    
    print(f"  Input shape: {pc.shape}")
    print(f"  Invariant latent shape: {inv_z.shape}")
    print(f"  Reconstruction shape: {recon.shape}")
    if eq_z is not None:
        print(f"  Equivariant latent shape: {eq_z.shape}")
    
    # Verify shapes
    assert inv_z.shape == (batch_size, cfg.latent_size), f"Expected inv_z shape {(batch_size, cfg.latent_size)}, got {inv_z.shape}"
    assert recon.shape == (batch_size, num_points, 3), f"Expected recon shape {(batch_size, num_points, 3)}, got {recon.shape}"
    
    # Check for NaN/Inf
    assert not torch.isnan(inv_z).any(), "NaN in invariant latent"
    assert not torch.isnan(recon).any(), "NaN in reconstruction"
    assert not torch.isinf(recon).any(), "Inf in reconstruction"
    
    # Check reconstruction range (should be roughly in [-2, 2] for normalized data)
    recon_max = recon.abs().max().item()
    print(f"  Max reconstruction magnitude: {recon_max:.4f}")
    if recon_max > 10:
        print(f"  WARNING: Reconstruction magnitude is very large ({recon_max:.2f})")
    
    print(f"  PASSED: Forward pass works correctly")
    return True


def test_overfit_single_batch(decoder_name: str, device: str = "cpu", num_steps: int = 500):
    """Test that model can overfit on a single batch."""
    from src.training_methods.equivariant_autoencoder.eq_ae_module import EquivariantAutoencoder
    from src.loss.reconstruction_loss import chamfer_distance
    
    print(f"\n{'='*60}")
    print(f"Testing overfitting with {decoder_name} decoder")
    print('='*60)
    
    cfg = create_test_config(decoder_name)
    model = EquivariantAutoencoder(cfg).to(device)
    model.train()
    
    # Create a fixed batch to overfit on
    batch_size = 2
    num_points = cfg.data.num_points
    pc = create_random_point_cloud(batch_size, num_points, device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    initial_loss = None
    final_loss = None
    losses = []
    
    for step in range(num_steps):
        optimizer.zero_grad()
        inv_z, recon, eq_z = model(pc)
        loss, _ = chamfer_distance(recon, pc, point_reduction="mean")
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        losses.append(loss.item())
        
        if step == 0:
            initial_loss = loss.item()
        if step == num_steps - 1:
            final_loss = loss.item()
        
        if step % 100 == 0:
            print(f"  Step {step:4d}: loss = {loss.item():.6f}")
    
    # Compute final non-squared CD for evaluation
    with torch.no_grad():
        _, final_recon, _ = model(pc)
        final_cd, _ = chamfer_distance(final_recon, pc, point_reduction="mean")
        final_cd = final_cd.item()
    
    target_cd = 0.02  # Target CD for overfitting
    print(f"  Final CD (target ~{target_cd}): {final_cd:.6f}")
    
    # Check if model learned - target is CD < 0.05 for this quick test
    # (Full training should reach ~0.02)
    if final_cd < 0.10:
        print(f"  PASSED: Model learning (CD={final_cd:.4f})")
        return True, final_cd
    else:
        print(f"  WARNING: CD={final_cd:.4f} is far from target {target_cd}")
        return False, final_cd


def main():
    """Run all tests."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running tests on {device}")
    
    # Test all decoder types
    decoders_to_test = ["VN_Equivariant", "VN_Snowflake", "MLPLarge"]
    
    results = {}
    
    for decoder_name in decoders_to_test:
        try:
            # Test forward pass
            forward_ok = test_forward_pass(decoder_name, device)
            
            # Test overfitting
            overfit_ok, final_loss = test_overfit_single_batch(decoder_name, device)
            
            results[decoder_name] = {
                "forward_pass": forward_ok,
                "overfit": overfit_ok,
                "final_loss": final_loss,
            }
        except Exception as e:
            print(f"\n  FAILED: {decoder_name} - {e}")
            import traceback
            traceback.print_exc()
            results[decoder_name] = {"error": str(e)}
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for decoder_name, result in results.items():
        if "error" in result:
            print(f"  {decoder_name}: FAILED - {result['error']}")
        else:
            status = "OK" if result["overfit"] else "WEAK"
            print(f"  {decoder_name}: {status} (final loss: {result['final_loss']:.6f})")


if __name__ == "__main__":
    main()
