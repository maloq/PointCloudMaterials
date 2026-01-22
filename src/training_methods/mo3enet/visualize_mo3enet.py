"""
Visualization and analysis tools for Mo3ENet.
Generates qualitative and quantitative diagnostics similar to equivariant autoencoder analysis.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig

sys.path.append(os.getcwd())

from src.loss.reconstruction_loss import chamfer_distance
from src.utils.model_utils import load_model_from_checkpoint, resolve_config_path
from src.vis_tools.tsne_vis import compute_tsne, save_tsne_plot
from src.vis_tools.vis_utils import set_axes_equal


def load_mo3enet_model(
    checkpoint_path: str, cuda_device: int = 0, cfg: DictConfig | None = None
) -> Tuple[Any, DictConfig, str]:
    """Load Mo3ENet model from checkpoint."""
    from hydra import compose, initialize
    from src.training_methods.mo3enet.mo3enet_module import Mo3ENetModule

    if cfg is None:
        config_dir, config_name = resolve_config_path(checkpoint_path)
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parents[2]
        absolute_config_dir = project_root / config_dir
        relative_config_dir = os.path.relpath(absolute_config_dir, current_dir)
        with initialize(version_base=None, config_path=relative_config_dir):
            cfg = compose(config_name=config_name)

    device = f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu"
    model: Mo3ENetModule = load_model_from_checkpoint(
        checkpoint_path, cfg, device=device, module=Mo3ENetModule
    )
    model.to(device).eval()
    return model, cfg, device


def build_datamodule(cfg: DictConfig):
    """Instantiate and setup the matching datamodule."""
    from src.data_utils.data_module import (
        RealPointCloudDataModule,
        SyntheticPointCloudDataModule,
    )

    if getattr(cfg, "data", None) is None:
        raise ValueError("Config missing data section")
    if getattr(cfg.data, "kind", None) == "synthetic":
        dm = SyntheticPointCloudDataModule(cfg)
    else:
        dm = RealPointCloudDataModule(cfg)
    dm.setup(stage="test")
    return dm


def _extract_from_batch(batch: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Extract point cloud, types, mask, and class_id from batch."""
    if isinstance(batch, dict):
        pc = batch["points"]
        types = batch.get("types", None)
        node_mask = batch.get("node_mask", batch.get("mask", None))
        class_id = batch.get("class_id", None)
    elif isinstance(batch, (tuple, list)):
        pc = batch[0]
        types = None
        node_mask = None
        class_id = batch[1] if len(batch) > 1 else None
    else:
        pc = batch
        types = None
        node_mask = None
        class_id = None
    
    if class_id is not None and not torch.is_tensor(class_id):
        class_id = torch.as_tensor(class_id)
    
    return pc, types, node_mask, class_id


def _downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    idx = np.random.default_rng(0).choice(len(points), size=max_points, replace=False)
    return points[idx]


def gather_inference_batches(
    model: Any,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    max_batches: int = 4,
) -> Dict[str, np.ndarray]:
    """Collect originals, swarm outputs, and latents from batches."""
    originals = []
    swarm_outputs = []
    eq_latents = []  # g: (B, K, 3) - equivariant
    inv_latents = []  # ||g||: (B, K) - invariant norms
    class_ids = []
    weight_logits_all = []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            
            pc, types, node_mask, class_id = _extract_from_batch(batch)
            pc = pc.to(device)
            if types is not None:
                types = types.to(device)
            if node_mask is not None:
                node_mask = node_mask.to(device)

            # Forward pass
            g, y, type_logits, weight_logits = model(pc, types, node_mask)
            
            originals.append(pc.detach().cpu())
            swarm_outputs.append(y.detach().cpu())
            eq_latents.append(g.detach().cpu())
            
            # Invariant latent = norms of equivariant vectors
            inv_z = torch.linalg.norm(g, dim=-1)  # (B, K)
            inv_latents.append(inv_z.detach().cpu())
            weight_logits_all.append(weight_logits.detach().cpu())
            
            if class_id is not None:
                class_ids.append(class_id.detach().view(-1).cpu())

    def _cat(tensors):
        return torch.cat(tensors, dim=0).numpy() if tensors else np.empty((0,))

    return {
        "originals": _cat(originals),
        "swarm_outputs": _cat(swarm_outputs),
        "eq_latents": _cat(eq_latents),
        "inv_latents": _cat(inv_latents),
        "class_ids": _cat(class_ids),
        "weight_logits": _cat(weight_logits_all),
    }


def _chamfer(orig: np.ndarray, reco: np.ndarray) -> float:
    a = torch.tensor(orig, dtype=torch.float32).unsqueeze(0)
    b = torch.tensor(reco, dtype=torch.float32).unsqueeze(0)
    val, _ = chamfer_distance(a, b, squared=False, point_reduction="mean")
    return float(val.item())


def save_reconstruction_grid(
    originals: np.ndarray,
    swarm_outputs: np.ndarray,
    class_ids: np.ndarray,
    out_file: Path,
    max_examples: int = 6,
    max_points: int = 2048,
) -> None:
    """Save side-by-side visualization of original vs swarm output."""
    if originals.size == 0 or swarm_outputs.size == 0:
        return

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    count = int(min(len(originals), max_examples))

    fig = plt.figure(figsize=(10, 4 * count), dpi=150)
    for i in range(count):
        orig = _downsample(originals[i], max_points)
        swarm = _downsample(swarm_outputs[i], max_points)
        cd = _chamfer(orig, swarm)
        class_label = int(class_ids[i]) if class_ids.size == len(originals) else None

        # Original
        ax = fig.add_subplot(count, 2, i * 2 + 1, projection="3d")
        ax.scatter(orig[:, 0], orig[:, 1], orig[:, 2], s=3, alpha=0.7, c="#2c3e50")
        title = f"Original ({len(originals[i])} pts)"
        if class_label is not None:
            title += f" • class {class_label}"
        ax.set_title(title)
        ax.axis("off")
        set_axes_equal(ax)

        # Swarm output
        ax = fig.add_subplot(count, 2, i * 2 + 2, projection="3d")
        ax.scatter(swarm[:, 0], swarm[:, 1], swarm[:, 2], s=1, alpha=0.5, c="#e74c3c")
        ax.set_title(f"Swarm ({len(swarm_outputs[i])} pts) • CD={cd:.4f}")
        ax.axis("off")
        set_axes_equal(ax)

    plt.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved reconstruction grid to {out_file}")


def save_latent_tsne(
    inv_latents: np.ndarray,
    class_ids: np.ndarray,
    out_file: Path,
    max_samples: int = 4000,
) -> None:
    """Save t-SNE visualization of invariant latent space."""
    if inv_latents.size == 0 or len(inv_latents) < 2:
        return

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    latents = inv_latents
    labels = class_ids if class_ids.size == len(latents) else np.zeros(len(latents), dtype=int)

    if len(latents) > max_samples:
        idx = np.random.default_rng(0).choice(len(latents), size=max_samples, replace=False)
        latents = latents[idx]
        labels = labels[idx]

    tsne_coords = compute_tsne(latents)
    save_tsne_plot(tsne_coords, labels, out_file=str(out_file), title="Mo3ENet Invariant Latent (||g||)")
    print(f"Saved t-SNE plot to {out_file}")


def evaluate_equivariance(
    model: Any,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    max_batches: int = 2,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate O(3) equivariance of Mo3ENet."""
    from src.models.mo3enet import random_rotation_matrix
    
    latent_errors = []
    swarm_errors = []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            
            pc, types, node_mask, _ = _extract_from_batch(batch)
            pc = pc.to(device)
            if types is not None:
                types = types.to(device)
            if node_mask is not None:
                node_mask = node_mask.to(device)
            
            batch_size = pc.shape[0]

            # Generate random rotations
            rots = torch.stack([
                random_rotation_matrix(device, pc.dtype)
                for _ in range(batch_size)
            ])
            pc_rot = torch.einsum("bij,bnj->bni", rots, pc)

            # Forward on original and rotated
            g, y, _, _ = model(pc, types, node_mask)
            g_rot, y_rot, _, _ = model(pc_rot, types, node_mask)

            # Expected equivariant latent: g_rot should equal R @ g
            expected_g = torch.einsum("bij,bkj->bki", rots, g)
            latent_err = torch.linalg.norm(g_rot - expected_g, dim=-1) / torch.linalg.norm(expected_g, dim=-1).clamp_min(1e-6)
            latent_errors.extend(latent_err.mean(dim=1).detach().cpu().numpy().tolist())

            # Expected equivariant swarm: y_rot should equal R @ y
            expected_y = torch.einsum("bij,bmj->bmi", rots, y)
            swarm_err = torch.linalg.norm(y_rot - expected_y, dim=-1) / torch.linalg.norm(expected_y, dim=-1).clamp_min(1e-6)
            swarm_errors.extend(swarm_err.mean(dim=1).detach().cpu().numpy().tolist())

    latent_arr = np.asarray(latent_errors)
    swarm_arr = np.asarray(swarm_errors)
    
    metrics = {
        "latent_equiv_rel_error_mean": float(latent_arr.mean()) if latent_arr.size else float("nan"),
        "latent_equiv_rel_error_median": float(np.median(latent_arr)) if latent_arr.size else float("nan"),
        "swarm_equiv_rel_error_mean": float(swarm_arr.mean()) if swarm_arr.size else float("nan"),
        "swarm_equiv_rel_error_median": float(np.median(swarm_arr)) if swarm_arr.size else float("nan"),
        "num_samples": len(latent_errors),
    }
    return metrics, latent_arr, swarm_arr


def save_equivariance_plot(
    latent_errors: np.ndarray,
    swarm_errors: np.ndarray,
    out_file: Path,
) -> None:
    """Save histogram of equivariance errors."""
    if latent_errors.size == 0 and swarm_errors.size == 0:
        return

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    if latent_errors.size:
        axes[0].hist(latent_errors, bins=30, color="#2980b9", alpha=0.8)
        axes[0].axvline(np.median(latent_errors), color='red', linestyle='--', label=f'median={np.median(latent_errors):.4f}')
        axes[0].set_title("Latent Equivariance Error")
        axes[0].set_xlabel("||g(Rx) - Rg(x)|| / ||Rg(x)||")
        axes[0].set_ylabel("count")
        axes[0].legend()
    else:
        axes[0].axis("off")
        axes[0].set_title("No latent errors collected")

    if swarm_errors.size:
        axes[1].hist(swarm_errors, bins=30, color="#c0392b", alpha=0.8)
        axes[1].axvline(np.median(swarm_errors), color='red', linestyle='--', label=f'median={np.median(swarm_errors):.4f}')
        axes[1].set_title("Swarm Equivariance Error")
        axes[1].set_xlabel("||y(Rx) - Ry(x)|| / ||Ry(x)||")
        axes[1].set_ylabel("count")
        axes[1].legend()
    else:
        axes[1].axis("off")
        axes[1].set_title("No swarm errors collected")

    plt.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved equivariance plot to {out_file}")


def save_swarm_weight_distribution(
    weight_logits: np.ndarray,
    out_file: Path,
    num_examples: int = 4,
) -> None:
    """Visualize swarm weight distributions."""
    if weight_logits.size == 0:
        return

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert logits to weights
    weights = torch.softmax(torch.tensor(weight_logits), dim=-1).numpy()
    
    fig, axes = plt.subplots(1, min(num_examples, len(weights)), figsize=(4 * min(num_examples, len(weights)), 3), dpi=150)
    if num_examples == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes[:len(weights)]):
        w = weights[i]
        ax.bar(range(len(w)), np.sort(w)[::-1], alpha=0.7, color="#3498db")
        ax.set_title(f"Sample {i+1}")
        ax.set_xlabel("Sorted swarm point index")
        ax.set_ylabel("Weight")
        ax.set_ylim(0, max(w) * 1.1)
    
    plt.suptitle("Swarm Point Weight Distribution (sorted)", y=1.02)
    plt.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved weight distribution to {out_file}")


def run_mo3enet_analysis(
    checkpoint_path: str,
    output_dir: str,
    cuda_device: int = 0,
    cfg: Optional[DictConfig] = None,
    max_batches: int = 4,
) -> None:
    """
    Run comprehensive post-training analysis for Mo3ENet.
    
    Generates:
    - Reconstruction grid (original vs swarm)
    - Latent space t-SNE
    - Equivariance evaluation plots
    - Swarm weight distribution
    - Metrics JSON
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {checkpoint_path}")
    model, cfg, device = load_mo3enet_model(checkpoint_path, cuda_device=cuda_device, cfg=cfg)
    
    print("Building datamodule...")
    dm = build_datamodule(cfg)
    dl = dm.test_dataloader()

    print("Gathering inference batches...")
    cache = gather_inference_batches(model, dl, device, max_batches=max_batches)

    print("Saving reconstruction grid...")
    save_reconstruction_grid(
        cache["originals"],
        cache["swarm_outputs"],
        cache["class_ids"],
        out_dir / "reconstruction_grid.png",
    )

    print("Saving latent t-SNE...")
    save_latent_tsne(cache["inv_latents"], cache["class_ids"], out_dir / "latent_tsne.png")

    print("Evaluating equivariance...")
    metrics, latent_err, swarm_err = evaluate_equivariance(model, dl, device, max_batches=2)
    save_equivariance_plot(latent_err, swarm_err, out_dir / "equivariance.png")

    print("Saving weight distribution...")
    save_swarm_weight_distribution(cache["weight_logits"], out_dir / "swarm_weights.png")

    # Compute additional metrics
    print("Computing reconstruction metrics...")
    chamfer_distances = []
    for i in range(min(len(cache["originals"]), 100)):
        cd = _chamfer(cache["originals"][i], cache["swarm_outputs"][i])
        chamfer_distances.append(cd)
    
    metrics["chamfer_mean"] = float(np.mean(chamfer_distances)) if chamfer_distances else float("nan")
    metrics["chamfer_std"] = float(np.std(chamfer_distances)) if chamfer_distances else float("nan")
    metrics["num_reconstruction_samples"] = len(chamfer_distances)

    # Save metrics
    metrics_path = out_dir / "analysis_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    print(f"\n{'='*60}")
    print("Mo3ENet Analysis Complete")
    print(f"{'='*60}")
    print(f"Output directory: {out_dir}")
    print(f"\nKey metrics:")
    print(f"  Chamfer distance: {metrics['chamfer_mean']:.4f} ± {metrics['chamfer_std']:.4f}")
    print(f"  Latent equivariance error: {metrics['latent_equiv_rel_error_mean']:.6f}")
    print(f"  Swarm equivariance error: {metrics['swarm_equiv_rel_error_mean']:.6f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mo3ENet post-training analysis")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output_dir", type=str, default="./mo3enet_analysis", help="Output directory")
    parser.add_argument("--cuda_device", type=int, default=0, help="CUDA device")
    parser.add_argument("--max_batches", type=int, default=4, help="Max batches to process")
    args = parser.parse_args()

    run_mo3enet_analysis(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        cuda_device=args.cuda_device,
        max_batches=args.max_batches,
    )
