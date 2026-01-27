"""
Debug script for Equivariant Autoencoder to identify Chamfer Distance floor issues.

This script adds detailed instrumentation to:
1. Verify input/output statistics (mean, std, min/max, RMS radius)
2. Check pairwise distance distributions
3. Verify encoder output stability
4. Validate Chamfer Distance implementation
5. Log per-batch diagnostics

Usage:
    conda activate pointnet
    cd /home/infres/vmorozov/PointCloudMaterials
    python src/training_methods/equivariant_autoencoder/debug_eq_ae.py
"""

import os
import sys
import hashlib
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.getcwd())

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback
from datetime import datetime

from src.utils.logging_config import setup_logging
from src.training_methods.equivariant_autoencoder.eq_ae_module import EquivariantAutoencoder
from src.data_utils.data_module import SyntheticPointCloudDataModule
from src.loss.reconstruction_loss import chamfer_distance

torch.set_float32_matmul_precision('high')
logger = setup_logging()


def compute_pc_stats(pc: torch.Tensor, name: str = "pc") -> dict:
    """Compute comprehensive point cloud statistics."""
    # pc: (B, N, 3) or (N, 3)
    if pc.dim() == 2:
        pc = pc.unsqueeze(0)
    
    B, N, _ = pc.shape
    
    # Per-point norms (radius from origin)
    radii = pc.norm(dim=-1)  # (B, N)
    
    # Centroid
    centroid = pc.mean(dim=1)  # (B, 3)
    centered_pc = pc - centroid.unsqueeze(1)
    centered_radii = centered_pc.norm(dim=-1)
    
    # Pairwise distances
    pairwise = torch.cdist(pc, pc)  # (B, N, N)
    # Get upper triangle (excluding diagonal)
    mask = torch.triu(torch.ones(N, N, device=pc.device, dtype=torch.bool), diagonal=1)
    pairwise_upper = pairwise[:, mask]  # (B, N*(N-1)/2)
    
    stats = {
        f"{name}_mean": pc.mean().item(),
        f"{name}_std": pc.std().item(),
        f"{name}_min": pc.min().item(),
        f"{name}_max": pc.max().item(),
        f"{name}_rms_radius": radii.pow(2).mean().sqrt().item(),
        f"{name}_centroid_norm": centroid.norm(dim=-1).mean().item(),
        f"{name}_centered_rms": centered_radii.pow(2).mean().sqrt().item(),
        f"{name}_pairwise_mean": pairwise_upper.mean().item(),
        f"{name}_pairwise_median": pairwise_upper.median().item(),
        f"{name}_pairwise_min": pairwise_upper.min().item(),
        f"{name}_pairwise_max": pairwise_upper.max().item(),
    }
    return stats


def compute_fingerprint(pc: torch.Tensor) -> str:
    """Compute a deterministic fingerprint for a point cloud."""
    # Use first 100 floats or all if less
    flat = pc.flatten()[:100].cpu().numpy()
    fingerprint = hashlib.md5(flat.tobytes()).hexdigest()[:8]
    checksum = flat.sum()
    return f"{fingerprint}_{checksum:.4f}"


def validate_chamfer_distance():
    """Validate Chamfer Distance implementation correctness."""
    print("\n" + "=" * 60)
    print("VALIDATING CHAMFER DISTANCE IMPLEMENTATION")
    print("=" * 60)
    
    # Test 1: CD(pred, pred) should be ~0
    pc = torch.randn(2, 80, 3)
    cd_self, _ = chamfer_distance(pc, pc)
    print(f"CD(pred, pred) = {cd_self.item():.8f} (should be ~0)")
    
    # Test 2: Symmetry CD(A, B) ≈ CD(B, A)
    pc_a = torch.randn(2, 80, 3)
    pc_b = torch.randn(2, 80, 3)
    cd_ab, _ = chamfer_distance(pc_a, pc_b)
    cd_ba, _ = chamfer_distance(pc_b, pc_a)
    print(f"CD(A, B) = {cd_ab.item():.6f}")
    print(f"CD(B, A) = {cd_ba.item():.6f}")
    print(f"Symmetry diff = {abs(cd_ab.item() - cd_ba.item()):.8f}")
    
    # Test 3: Scale sensitivity
    pc_unit = torch.randn(2, 80, 3)
    pc_unit = pc_unit / pc_unit.norm(dim=-1, keepdim=True).mean(dim=1, keepdim=True)  # normalize to unit sphere
    
    # Small perturbation
    noise_small = torch.randn_like(pc_unit) * 0.01
    cd_small, _ = chamfer_distance(pc_unit, pc_unit + noise_small)
    
    # Large perturbation
    noise_large = torch.randn_like(pc_unit) * 0.1
    cd_large, _ = chamfer_distance(pc_unit, pc_unit + noise_large)
    
    print(f"\nScale sensitivity test (unit sphere data):")
    print(f"  Noise std=0.01: CD = {cd_small.item():.6f}")
    print(f"  Noise std=0.10: CD = {cd_large.item():.6f}")
    print(f"  Ratio: {cd_large.item() / cd_small.item():.2f}x (should be ~10x)")
    
    # Test 4: Compare with manual cdist-based computation
    pc_test = torch.randn(2, 80, 3)
    pc_target = torch.randn(2, 80, 3)
    cd_impl, _ = chamfer_distance(pc_test, pc_target)
    
    # Manual computation
    dists = torch.cdist(pc_test, pc_target)  # (B, N, M)
    min_pred2gt = dists.min(dim=2)[0]  # (B, N)
    min_gt2pred = dists.min(dim=1)[0]  # (B, M)
    cd_manual = (min_pred2gt.mean(dim=1) + min_gt2pred.mean(dim=1)).mean()
    
    print(f"\nImplementation validation:")
    print(f"  Our CD: {cd_impl.item():.6f}")
    print(f"  Manual CD: {cd_manual.item():.6f}")
    print(f"  Diff: {abs(cd_impl.item() - cd_manual.item()):.8f}")
    
    # Test 5: What CD floor would correspond to what displacement?
    print(f"\nCD floor interpretation (for 80-point unit-sphere data):")
    for noise_std in [0.1, 0.2, 0.3, 0.4, 0.5, 1.0]:
        noise = torch.randn_like(pc_unit) * noise_std
        cd, _ = chamfer_distance(pc_unit, pc_unit + noise)
        print(f"  Noise std={noise_std:.1f} -> CD = {cd.item():.4f}")
    
    print("\n" + "=" * 60)


class DebugCallback(Callback):
    """Callback for detailed debugging of training dynamics."""
    
    def __init__(self, log_every_n_steps: int = 1):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.prev_fingerprint = None
        self.step = 0
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.step % self.log_every_n_steps != 0:
            self.step += 1
            return
            
        self.step += 1
        
        with torch.no_grad():
            # Unpack batch
            if isinstance(batch, dict):
                pc = batch["points"]
            else:
                pc = batch[0] if isinstance(batch, (tuple, list)) else batch
            
            pc = pc.to(pl_module.device)
            
            # Check determinism
            fingerprint = compute_fingerprint(pc)
            if self.prev_fingerprint is not None and fingerprint != self.prev_fingerprint:
                logger.print(f"[DEBUG] Data fingerprint CHANGED: {self.prev_fingerprint} -> {fingerprint}")
            self.prev_fingerprint = fingerprint
            
            # Forward pass for diagnostics
            inv_z, recon, eq_z = pl_module(pc)
            
            # Input/output statistics
            input_stats = compute_pc_stats(pc, "input")
            recon_stats = compute_pc_stats(recon, "recon")
            
            # Chamfer distance
            cd, _ = chamfer_distance(recon, pc)
            
            # Latent statistics
            if eq_z is not None:
                eq_z_norm = eq_z.norm(dim=-1)  # (B, latent_size)
                eq_z_stats = {
                    "eq_z_norm_mean": eq_z_norm.mean().item(),
                    "eq_z_norm_std": eq_z_norm.std().item(),
                    "eq_z_norm_min": eq_z_norm.min().item(),
                    "eq_z_norm_max": eq_z_norm.max().item(),
                }
            else:
                eq_z_stats = {}
            
            if inv_z is not None:
                inv_z_stats = {
                    "inv_z_mean": inv_z.mean().item(),
                    "inv_z_std": inv_z.std().item(),
                    "inv_z_norm": inv_z.norm(dim=-1).mean().item(),
                }
            else:
                inv_z_stats = {}
            
            # Log summary
            logger.print(f"\n[DEBUG Step {self.step}] CD={cd.item():.6f}")
            logger.print(f"  INPUT:  rms={input_stats['input_rms_radius']:.4f}, centered_rms={input_stats['input_centered_rms']:.4f}, "
                        f"pairwise_mean={input_stats['input_pairwise_mean']:.4f}, centroid_norm={input_stats['input_centroid_norm']:.4f}")
            logger.print(f"  RECON:  rms={recon_stats['recon_rms_radius']:.4f}, centered_rms={recon_stats['recon_centered_rms']:.4f}, "
                        f"pairwise_mean={recon_stats['recon_pairwise_mean']:.4f}, centroid_norm={recon_stats['recon_centroid_norm']:.4f}")
            
            if eq_z_stats:
                logger.print(f"  EQ_Z:   norm_mean={eq_z_stats['eq_z_norm_mean']:.4f}, norm_std={eq_z_stats['eq_z_norm_std']:.4f}")
            
            # Check for scale mismatch
            scale_ratio = recon_stats['recon_rms_radius'] / (input_stats['input_rms_radius'] + 1e-8)
            if abs(scale_ratio - 1.0) > 0.2:
                logger.print(f"  WARNING: Scale mismatch! recon_rms/input_rms = {scale_ratio:.3f}")
            
            pairwise_ratio = recon_stats['recon_pairwise_mean'] / (input_stats['input_pairwise_mean'] + 1e-8)
            if abs(pairwise_ratio - 1.0) > 0.2:
                logger.print(f"  WARNING: Pairwise distance mismatch! ratio = {pairwise_ratio:.3f}")
            
            # Check for collapse
            if recon_stats['recon_centered_rms'] < 0.1:
                logger.print(f"  WARNING: Reconstruction may be collapsing! centered_rms = {recon_stats['recon_centered_rms']:.4f}")
            
            if recon_stats['recon_pairwise_min'] < 0.01:
                logger.print(f"  WARNING: Points collapsing together! pairwise_min = {recon_stats['recon_pairwise_min']:.6f}")


def debug_model_forward(model, sample_pc):
    """Debug a single forward pass through the model."""
    print("\n" + "=" * 60)
    print("DEBUGGING SINGLE FORWARD PASS")
    print("=" * 60)
    
    model.eval()
    with torch.no_grad():
        # Input stats
        input_stats = compute_pc_stats(sample_pc, "input")
        print(f"\nInput point cloud stats:")
        for k, v in input_stats.items():
            print(f"  {k}: {v:.6f}")
        
        # Encoder forward
        enc_input = model._prepare_encoder_input(sample_pc)
        print(f"\nEncoder input shape: {enc_input.shape}")
        
        enc_out = model.encoder(enc_input)
        inv_z, eq_z = model._split_encoder_output(enc_out)
        
        print(f"\nEncoder output:")
        if inv_z is not None:
            print(f"  inv_z shape: {inv_z.shape}")
            print(f"  inv_z mean: {inv_z.mean():.6f}, std: {inv_z.std():.6f}")
            print(f"  inv_z min: {inv_z.min():.6f}, max: {inv_z.max():.6f}")
        
        if eq_z is not None:
            print(f"  eq_z shape: {eq_z.shape}")
            eq_z_norms = eq_z.norm(dim=-1)
            print(f"  eq_z vector norms - mean: {eq_z_norms.mean():.6f}, std: {eq_z_norms.std():.6f}")
            print(f"  eq_z vector norms - min: {eq_z_norms.min():.6f}, max: {eq_z_norms.max():.6f}")
        
        # Decoder forward
        decoder_input = inv_z if model.use_invariant_latent else eq_z
        print(f"\nDecoder input shape: {decoder_input.shape}")
        
        recon = model.decoder(decoder_input)
        if isinstance(recon, tuple):
            recon = recon[0]
        
        print(f"Decoder output shape: {recon.shape}")
        
        recon_stats = compute_pc_stats(recon, "recon")
        print(f"\nReconstruction stats:")
        for k, v in recon_stats.items():
            print(f"  {k}: {v:.6f}")
        
        # Chamfer distance
        cd, _ = chamfer_distance(recon, sample_pc)
        print(f"\nChamfer Distance: {cd.item():.6f}")
        
        # Check decoder internal state if possible
        if hasattr(model.decoder, 'scale_param'):
            print(f"\nDecoder scale_param: {model.decoder.scale_param.item():.6f}")
        if hasattr(model.decoder, 'output_scale'):
            print(f"Decoder output_scale: {model.decoder.output_scale}")
        if hasattr(model.decoder, 'offset_scale'):
            print(f"Decoder offset_scale: {model.decoder.offset_scale}")
    
    print("\n" + "=" * 60)
    return recon


def apply_debug_overrides(cfg: DictConfig) -> None:
    """Apply minimal overrides for debugging."""
    try:
        OmegaConf.set_readonly(cfg, False)
        OmegaConf.set_struct(cfg, False)
    except Exception:
        pass
    
    # Use a small batch for detailed logging
    cfg.batch_size = 1
    cfg.max_samples = 1
    cfg.num_workers = 0
    cfg.precision = "32-true"
    
    # Disable regularizers
    cfg.kl_latent_loss_scale = 0.0
    
    # Remove augmentation
    if hasattr(cfg, "data") and hasattr(cfg.data, "augmentation"):
        cfg.data.augmentation.rotation_scale = 0.0
        cfg.data.augmentation.noise_scale = 0.0
        cfg.data.augmentation.jitter_scale = 0.0
        cfg.data.augmentation.scaling_range = 0.0
    
    # Disable barlow
    cfg.barlow_enabled = False
    cfg.barlow_weight = 0.0
    
    # Disable curriculum
    if hasattr(cfg, "curriculum_learning"):
        cfg.curriculum_learning.enable = False
    
    # Higher learning rate for overfitting
    cfg.learning_rate = 0.01
    
    # Disable gradient clipping
    cfg.gradient_clip_val = 0.0
    
    logger.print("Applied debug overrides for single-sample overfitting")


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), 'configs'), config_name='eq_ae_vn_molecular.yaml')
def main(cfg: DictConfig):
    # Setup
    run_dir = f"output/debug_eq_ae/{datetime.now():%Y-%m-%d_%H-%M-%S}"
    os.makedirs(run_dir, exist_ok=True)
    
    logger.print(f"\n{'=' * 60}")
    logger.print("EQUIVARIANT AUTOENCODER DEBUG SESSION")
    logger.print(f"{'=' * 60}")
    
    # Step 1: Validate Chamfer Distance
    validate_chamfer_distance()
    
    # Step 2: Apply overrides and setup
    apply_debug_overrides(cfg)
    
    # Step 3: Create data module and model
    dm = SyntheticPointCloudDataModule(cfg)
    dm.setup()
    
    model = EquivariantAutoencoder(cfg)
    model.cuda()
    
    # Step 4: Get a sample
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    if isinstance(batch, dict):
        sample_pc = batch["points"].cuda()
    else:
        sample_pc = batch[0].cuda() if isinstance(batch, (tuple, list)) else batch.cuda()
    
    logger.print(f"\nSample point cloud shape: {sample_pc.shape}")
    logger.print(f"Sample fingerprint: {compute_fingerprint(sample_pc)}")
    
    # Step 5: Debug single forward pass
    recon = debug_model_forward(model, sample_pc)
    
    # Step 6: Run short overfit training with debug callback
    logger.print("\n" + "=" * 60)
    logger.print("RUNNING OVERFIT TRAINING WITH DEBUG LOGGING")
    logger.print("=" * 60)
    
    debug_callback = DebugCallback(log_every_n_steps=5)
    
    trainer = pl.Trainer(
        default_root_dir=run_dir,
        max_epochs=50,
        accelerator='gpu',
        devices=[0],
        callbacks=[debug_callback],
        log_every_n_steps=1,
        precision="32-true",
        overfit_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    
    trainer.fit(model, dm)
    
    # Step 7: Final diagnostic
    logger.print("\n" + "=" * 60)
    logger.print("FINAL DIAGNOSTIC AFTER OVERFIT TRAINING")
    logger.print("=" * 60)
    
    recon_final = debug_model_forward(model, sample_pc)
    
    # Compute final CD
    model.eval()
    with torch.no_grad():
        cd_final, _ = chamfer_distance(recon_final, sample_pc)
        logger.print(f"\n FINAL Chamfer Distance: {cd_final.item():.6f}")
        
        if cd_final.item() > 0.1:
            logger.print("\n WARNING: CD floor detected! Investigating potential causes...")
            
            # Check scale
            input_rms = sample_pc.norm(dim=-1).mean().item()
            recon_rms = recon_final.norm(dim=-1).mean().item()
            logger.print(f"  Input RMS: {input_rms:.4f}, Recon RMS: {recon_rms:.4f}, Ratio: {recon_rms/input_rms:.3f}")
            
            # Check centroid
            input_centroid = sample_pc.mean(dim=1).norm(dim=-1).mean().item()
            recon_centroid = recon_final.mean(dim=1).norm(dim=-1).mean().item()
            logger.print(f"  Input centroid norm: {input_centroid:.4f}, Recon centroid norm: {recon_centroid:.4f}")
            
            # Check pairwise distances
            input_pair = torch.cdist(sample_pc, sample_pc)
            recon_pair = torch.cdist(recon_final, recon_final)
            mask = torch.triu(torch.ones_like(input_pair[0], dtype=bool), diagonal=1)
            
            input_pair_mean = input_pair[:, mask].mean().item()
            recon_pair_mean = recon_pair[:, mask].mean().item()
            logger.print(f"  Input pairwise mean: {input_pair_mean:.4f}, Recon pairwise mean: {recon_pair_mean:.4f}")


if __name__ == "__main__":
    if not any(arg.startswith("hydra.run.dir=") for arg in sys.argv):
        sys.argv.append('hydra.run.dir=output/debug_eq_ae/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    main()
