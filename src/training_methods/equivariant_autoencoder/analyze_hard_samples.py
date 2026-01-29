"""
Per-sample reconstruction loss analysis.

This script analyzes which samples in the molecular dataset are hard vs easy to reconstruct.
It computes per-sample chamfer distance and groups results by phase to identify patterns.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig

sys.path.append(os.getcwd())

from src.loss.reconstruction_loss import chamfer_distance
from src.training_methods.equivariant_autoencoder.predict_and_visualize import (
    load_eq_model,
    build_datamodule,
    _extract_pc_and_phase,
)
from src.vis_tools.vis_utils import set_axes_equal


def compute_per_sample_metrics(
    model, dataloader, device: str, max_samples: int = None
) -> Dict[str, np.ndarray]:
    """Compute per-sample chamfer distance and other metrics.
    
    Returns:
        Dict with keys: 'chamfer', 'phases', 'originals', 'reconstructions',
                        'inv_latents', 'latent_norms'
    """
    results = {
        'chamfer': [],
        'phases': [],
        'originals': [],
        'reconstructions': [],
        'inv_latents': [],
        'latent_norms': [],
        'instance_ids': [],
    }
    
    total_samples = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            pc, phase = _extract_pc_and_phase(batch)
            pc = pc.to(device)
            batch_size = pc.shape[0]
            
            # Get instance_id if available
            instance_ids = batch.get('instance_id', None) if isinstance(batch, dict) else None
            
            # Model forward pass
            pc_for_model = model._prepare_model_input(pc)
            inv_z, recon, eq_z = model(pc_for_model)
            
            # Compute per-sample chamfer distance
            for i in range(batch_size):
                cd, _ = chamfer_distance(
                    recon[i:i+1].float(),
                    pc_for_model[i:i+1].float(),
                    point_reduction='mean'
                )
                results['chamfer'].append(float(cd.item()))
            
            results['originals'].append(pc_for_model.detach().cpu().numpy())
            results['reconstructions'].append(recon.detach().cpu().numpy())
            results['inv_latents'].append(inv_z.detach().cpu().numpy())
            results['latent_norms'].extend(
                np.linalg.norm(inv_z.detach().cpu().numpy(), axis=1).tolist()
            )
            
            if phase is not None:
                results['phases'].extend(phase.view(-1).cpu().numpy().tolist())
            
            if instance_ids is not None:
                results['instance_ids'].extend(instance_ids.cpu().numpy().tolist())
            
            total_samples += batch_size
            if max_samples is not None and total_samples >= max_samples:
                break
    
    # Convert to numpy arrays
    results['chamfer'] = np.array(results['chamfer'])
    results['phases'] = np.array(results['phases'])
    results['originals'] = np.concatenate(results['originals'], axis=0)
    results['reconstructions'] = np.concatenate(results['reconstructions'], axis=0)
    results['inv_latents'] = np.concatenate(results['inv_latents'], axis=0)
    results['latent_norms'] = np.array(results['latent_norms'])
    results['instance_ids'] = np.array(results['instance_ids']) if results['instance_ids'] else None
    
    return results


def analyze_by_phase(results: Dict[str, np.ndarray], phase_names: Dict[int, str] = None) -> Dict:
    """Analyze reconstruction quality per phase."""
    phases = results['phases']
    chamfer = results['chamfer']
    
    unique_phases = np.unique(phases)
    phase_stats = {}
    
    for p in unique_phases:
        mask = phases == p
        cd_values = chamfer[mask]
        phase_name = phase_names.get(int(p), f"Phase {p}") if phase_names else f"Phase {p}"
        
        phase_stats[int(p)] = {
            'name': phase_name,
            'count': int(mask.sum()),
            'mean_chamfer': float(np.mean(cd_values)),
            'std_chamfer': float(np.std(cd_values)),
            'min_chamfer': float(np.min(cd_values)),
            'max_chamfer': float(np.max(cd_values)),
            'median_chamfer': float(np.median(cd_values)),
            'p90_chamfer': float(np.percentile(cd_values, 90)),
            'p99_chamfer': float(np.percentile(cd_values, 99)),
        }
    
    return phase_stats


def get_hardest_and_easiest(
    results: Dict[str, np.ndarray], n: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Get indices of hardest and easiest samples to reconstruct."""
    chamfer = results['chamfer']
    sorted_idx = np.argsort(chamfer)
    
    easiest_idx = sorted_idx[:n]
    hardest_idx = sorted_idx[-n:][::-1]  # Descending order
    
    return easiest_idx, hardest_idx


def visualize_sample_comparison(
    results: Dict[str, np.ndarray],
    indices: np.ndarray,
    title_prefix: str,
    out_file: Path,
    phase_names: Dict[int, str] = None,
    max_show: int = 6,
):
    """Visualize original vs reconstructed for given sample indices."""
    n_show = min(len(indices), max_show)
    
    fig = plt.figure(figsize=(12, 4 * n_show), dpi=120)
    
    for i, idx in enumerate(indices[:n_show]):
        orig = results['originals'][idx]
        reco = results['reconstructions'][idx]
        cd = results['chamfer'][idx]
        phase = results['phases'][idx] if len(results['phases']) > idx else None
        phase_name = phase_names.get(int(phase), f"Phase {phase}") if phase_names and phase is not None else ""
        
        # Original
        ax = fig.add_subplot(n_show, 2, i * 2 + 1, projection='3d')
        ax.scatter(orig[:, 0], orig[:, 1], orig[:, 2], s=10, alpha=0.8, c='#2c3e50')
        ax.set_title(f"Original ({phase_name})")
        ax.axis('off')
        set_axes_equal(ax)
        
        # Reconstruction
        ax = fig.add_subplot(n_show, 2, i * 2 + 2, projection='3d')
        ax.scatter(reco[:, 0], reco[:, 1], reco[:, 2], s=10, alpha=0.8, c='#e74c3c')
        ax.set_title(f"Reconstruction • CD={cd:.4f}")
        ax.axis('off')
        set_axes_equal(ax)
    
    fig.suptitle(f"{title_prefix} Samples", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_chamfer_distribution(
    results: Dict[str, np.ndarray],
    phase_names: Dict[int, str],
    out_file: Path,
):
    """Plot chamfer distance distribution overall and per phase."""
    chamfer = results['chamfer']
    phases = results['phases']
    unique_phases = np.unique(phases)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
    
    # Overall distribution
    axes[0].hist(chamfer, bins=50, alpha=0.7, color='#3498db', edgecolor='black')
    axes[0].axvline(np.mean(chamfer), color='red', linestyle='--', label=f'Mean: {np.mean(chamfer):.4f}')
    axes[0].axvline(np.median(chamfer), color='green', linestyle='--', label=f'Median: {np.median(chamfer):.4f}')
    axes[0].set_xlabel('Chamfer Distance')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Overall Distribution (n={len(chamfer)})')
    axes[0].legend()
    
    # Per-phase distribution (overlaid)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_phases)))
    for i, p in enumerate(unique_phases):
        mask = phases == p
        phase_name = phase_names.get(int(p), f"Phase {p}")
        axes[1].hist(chamfer[mask], bins=30, alpha=0.5, color=colors[i], 
                    label=f'{phase_name} (μ={np.mean(chamfer[mask]):.4f})')
    axes[1].set_xlabel('Chamfer Distance')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution by Phase')
    axes[1].legend(fontsize=8)
    
    # Box plot per phase
    phase_data = [chamfer[phases == p] for p in unique_phases]
    phase_labels = [phase_names.get(int(p), f"P{p}") for p in unique_phases]
    bp = axes[2].boxplot(phase_data, labels=phase_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[2].set_xlabel('Phase')
    axes[2].set_ylabel('Chamfer Distance')
    axes[2].set_title('Chamfer by Phase')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_latent_vs_chamfer(
    results: Dict[str, np.ndarray],
    phase_names: Dict[int, str],
    out_file: Path,
):
    """Plot relationship between latent norm and reconstruction error."""
    chamfer = results['chamfer']
    latent_norms = results['latent_norms']
    phases = results['phases']
    unique_phases = np.unique(phases)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)
    
    # Scatter: latent norm vs chamfer
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_phases)))
    for i, p in enumerate(unique_phases):
        mask = phases == p
        phase_name = phase_names.get(int(p), f"Phase {p}")
        axes[0].scatter(latent_norms[mask], chamfer[mask], s=10, alpha=0.5, 
                       color=colors[i], label=phase_name)
    axes[0].set_xlabel('Latent Norm ||z||')
    axes[0].set_ylabel('Chamfer Distance')
    axes[0].set_title('Latent Norm vs Reconstruction Error')
    axes[0].legend(fontsize=8)
    
    # Correlation
    corr = np.corrcoef(latent_norms, chamfer)[0, 1]
    axes[0].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[0].transAxes,
                fontsize=12, verticalalignment='top')
    
    # Latent norm distribution by phase
    for i, p in enumerate(unique_phases):
        mask = phases == p
        phase_name = phase_names.get(int(p), f"Phase {p}")
        axes[1].hist(latent_norms[mask], bins=30, alpha=0.5, color=colors[i], 
                    label=f'{phase_name} (μ={np.mean(latent_norms[mask]):.2f})')
    axes[1].set_xlabel('Latent Norm ||z||')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Latent Norm by Phase')
    axes[1].legend(fontsize=8)
    
    plt.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved: {out_file}")


def run_analysis(
    checkpoint_path: str,
    output_dir: str,
    cuda_device: int = 0,
    cfg: DictConfig = None,
    max_samples: int = None,
    use_train_data: bool = True,
):
    """Run full per-sample analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Per-Sample Reconstruction Analysis")
    print("=" * 60)
    
    # Load model
    print("\n1. Loading model...")
    model, cfg, device = load_eq_model(checkpoint_path, cuda_device, cfg)
    
    # Build datamodule
    print("\n2. Building datamodule...")
    dm = build_datamodule(cfg)
    dataloader = dm.train_dataloader() if use_train_data else dm.test_dataloader()
    
    # Load phase names if available
    phase_names = {}
    data_path = getattr(cfg.data, 'data_path', None)
    if data_path:
        phase_mapping_path = Path(data_path) / 'phase_mapping.json'
        if phase_mapping_path.exists():
            with open(phase_mapping_path) as f:
                pm = json.load(f)
                # Handle nested structure with 'id_to_name' key
                if 'id_to_name' in pm:
                    phase_names = {int(k): v for k, v in pm['id_to_name'].items()}
                else:
                    # Fallback: assume flat dict with name->id mapping
                    phase_names = {int(v): k for k, v in pm.items()}
            print(f"   Loaded phase names: {phase_names}")
    
    # Compute per-sample metrics
    print(f"\n3. Computing per-sample metrics (max_samples={max_samples})...")
    results = compute_per_sample_metrics(model, dataloader, device, max_samples)
    
    print(f"   Total samples analyzed: {len(results['chamfer'])}")
    print(f"   Mean Chamfer: {np.mean(results['chamfer']):.6f}")
    print(f"   Median Chamfer: {np.median(results['chamfer']):.6f}")
    print(f"   Std Chamfer: {np.std(results['chamfer']):.6f}")
    
    # Analyze by phase
    print("\n4. Analyzing by phase...")
    phase_stats = analyze_by_phase(results, phase_names)
    
    print("\n   Phase-wise Statistics:")
    print("   " + "-" * 80)
    print(f"   {'Phase':<30} {'Count':>8} {'Mean CD':>10} {'Std CD':>10} {'Max CD':>10}")
    print("   " + "-" * 80)
    for p, stats in sorted(phase_stats.items(), key=lambda x: x[1]['mean_chamfer'], reverse=True):
        print(f"   {stats['name']:<30} {stats['count']:>8} {stats['mean_chamfer']:>10.6f} "
              f"{stats['std_chamfer']:>10.6f} {stats['max_chamfer']:>10.6f}")
    
    # Save statistics
    stats_file = output_dir / 'phase_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(phase_stats, f, indent=2)
    print(f"\n   Saved: {stats_file}")
    
    # Get hardest and easiest samples
    print("\n5. Identifying hardest and easiest samples...")
    easiest_idx, hardest_idx = get_hardest_and_easiest(results, n=10)
    
    print("\n   Top 10 Hardest Samples:")
    for i, idx in enumerate(hardest_idx):
        phase = results['phases'][idx]
        phase_name = phase_names.get(int(phase), f"Phase {phase}")
        cd = results['chamfer'][idx]
        print(f"   {i+1:3d}. Sample {idx}: CD={cd:.6f} ({phase_name})")
    
    print("\n   Top 10 Easiest Samples:")
    for i, idx in enumerate(easiest_idx):
        phase = results['phases'][idx]
        phase_name = phase_names.get(int(phase), f"Phase {phase}")
        cd = results['chamfer'][idx]
        print(f"   {i+1:3d}. Sample {idx}: CD={cd:.6f} ({phase_name})")
    
    # Visualizations
    print("\n6. Generating visualizations...")
    
    # Distribution plots
    plot_chamfer_distribution(results, phase_names, output_dir / 'chamfer_distribution.png')
    
    # Latent vs chamfer
    plot_latent_vs_chamfer(results, phase_names, output_dir / 'latent_vs_chamfer.png')
    
    # Hardest samples visualization
    visualize_sample_comparison(
        results, hardest_idx, "Hardest", 
        output_dir / 'hardest_samples.png', phase_names
    )
    
    # Easiest samples visualization
    visualize_sample_comparison(
        results, easiest_idx, "Easiest",
        output_dir / 'easiest_samples.png', phase_names
    )
    
    # Per-phase sample visualizations (hardest in each phase)
    print("\n7. Per-phase analysis...")
    for p in np.unique(results['phases']):
        mask = results['phases'] == p
        phase_chamfer = results['chamfer'][mask]
        phase_indices = np.where(mask)[0]
        
        # Get hardest 3 samples from this phase
        sorted_in_phase = np.argsort(phase_chamfer)[::-1][:3]
        hardest_in_phase = phase_indices[sorted_in_phase]
        
        phase_name = phase_names.get(int(p), f"phase_{p}")
        safe_name = phase_name.replace(' ', '_').replace('/', '_')
        visualize_sample_comparison(
            results, hardest_in_phase, f"Hardest in {phase_name}",
            output_dir / f'hardest_{safe_name}.png', phase_names, max_show=3
        )
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)
    
    return results, phase_stats


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze per-sample reconstruction quality')
    parser.add_argument('--checkpoint',  default="output/2026-01-28/20-52-49/EQ_AE_l120_N120_M80_pdist+rdf_VN_REVNET_Anchor-epoch=33.ckpt",   help='Path to model checkpoint')
    parser.add_argument('--output-dir', '-o', default=None, help='Output directory (default: checkpoint_dir/sample_analysis)')
    parser.add_argument('--cuda-device', '-d', type=int, default=0, help='CUDA device')
    parser.add_argument('--max-samples', '-n', type=int, default=None, help='Max samples to analyze')
    parser.add_argument('--use-test-data', action='store_true', help='Use test data instead of train')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = str(Path(args.checkpoint).parent / 'sample_analysis')
    
    run_analysis(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        cuda_device=args.cuda_device,
        max_samples=args.max_samples,
        use_train_data=not args.use_test_data,
    )
