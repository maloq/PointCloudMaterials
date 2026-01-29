import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

from src.training_methods.equivariant_autoencoder.predict_and_visualize import (
    load_eq_model,
    build_datamodule,
    gather_inference_batches,
)
from src.utils.spd_utils import to_float32

def compute_rdf_hist(points, n_bins=100, r_max=None, r_min=0.0, density=True):
    """
    Compute Radial Distribution Function (RDF) histogram for a batch of point clouds.
    In this context (matching rdf_loss), RDF is the distribution of distances from the origin.
    
    Args:
        points: (B, N, 3) numpy array
        n_bins: Number of bins
        r_max: Max radius. If None, computed from data.
        r_min: Min radius.
        density: Whether to normalize histogram as a probability density.
        
    Returns:
        hist: (n_bins,) array - the histogram
        bin_centers: (n_bins,) array
    """
    # Calculate norms (distances from origin)
    # points: (B, N, 3) -> norms: (B, N)
    norms = np.linalg.norm(points, axis=-1).flatten()
    
    if r_max is None:
        r_max = np.max(norms)
        
    hist, bin_edges = np.histogram(norms, bins=n_bins, range=(r_min, r_max), density=density)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return hist, bin_centers, r_max

def plot_rdf_comparison(
    originals, 
    reconstructions, 
    out_dir, 
    n_bins=100,
    r_max=None
):
    """
    Plot RDF comparison between original and reconstructed point clouds.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute histograms
    # Use same r_max for both for fair comparison
    if r_max is None:
        # Compute global max for both
        max_orig = np.max(np.linalg.norm(originals, axis=-1))
        max_recon = np.max(np.linalg.norm(reconstructions, axis=-1))
        r_max = max(max_orig, max_recon)
        
    hist_orig, bins_orig, _ = compute_rdf_hist(originals, n_bins=n_bins, r_max=r_max)
    hist_recon, bins_recon, _ = compute_rdf_hist(reconstructions, n_bins=n_bins, r_max=r_max)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    ax.plot(bins_orig, hist_orig, label='Original', color='#2c3e50', linewidth=2, alpha=0.8)
    ax.plot(bins_recon, hist_recon, label='Reconstructed', color='#e74c3c', linewidth=2, alpha=0.8, linestyle='--')
    
    ax.set_xlabel('Distance from Origin (r)')
    ax.set_ylabel('Density P(r)')
    ax.set_title('Radial Distribution Function (RDF) Comparison\nDistribution of distances from origin')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    out_file = out_dir / "rdf_comparison.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved RDF plot to {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Calculate and plot RDF for Equivariant Autoencoder")
    parser.add_argument("--checkpoint", type=str, default="output/2026-01-28/18-41-16/EQ_AE_l120_N120_M80_chamfer+rdf_VN_REVNET_Anchor-epoch=148.ckpt", help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--n_bins", type=int, default=100, help="Number of bins for histogram")
    parser.add_argument("--max_batches", type=int, default=10, help="Max batches to process")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.checkpoint}...")
    model, cfg, device_str = load_eq_model(args.checkpoint, cuda_device=args.device)
    
    print("Setting up data module...")
    dm = build_datamodule(cfg)
    test_loader = dm.test_dataloader()
    
    print(f"Gathering inference results (max_batches={args.max_batches})...")
    results = gather_inference_batches(model, test_loader, device_str, max_batches=args.max_batches)
    
    originals = results["originals"]
    reconstructions = results["reconstructions"]
    
    if len(originals) == 0:
        print("No data collected!")
        return
        
    print(f"Collected {len(originals)} samples.")
    print("Computing and plotting RDF...")
    
    # Determine output directory (same as checkpoint dir)
    checkpoint_path = Path(args.checkpoint)
    out_dir = checkpoint_path.parent / "eval_plots"
    
    plot_rdf_comparison(originals, reconstructions, out_dir, n_bins=args.n_bins)
    
if __name__ == "__main__":
    main()
