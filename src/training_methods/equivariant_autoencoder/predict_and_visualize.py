import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import DictConfig

sys.path.append(os.getcwd())

from src.data_utils.data_module import (
    RealPointCloudDataModule,
    SyntheticPointCloudDataModule,
)
from src.loss.reconstruction_loss import chamfer_distance
from src.training_methods.equivariant_autoencoder.eq_ae_module import (
    EquivariantAutoencoder,
)
from src.utils.model_utils import load_model_from_checkpoint, resolve_config_path
from src.utils.spd_metrics import random_rotation_matrix
from src.vis_tools.tsne_vis import compute_tsne, save_tsne_plot
from src.vis_tools.vis_utils import set_axes_equal


def load_eq_model(
    checkpoint_path: str, cuda_device: int = 0, cfg: DictConfig | None = None
) -> Tuple[EquivariantAutoencoder, DictConfig, str]:
    """Restore the Equivariant AE together with its Hydra cfg and device string."""
    if cfg is None:
        config_dir, config_name = resolve_config_path(checkpoint_path)
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parents[2]
        absolute_config_dir = project_root / config_dir
        relative_config_dir = os.path.relpath(absolute_config_dir, current_dir)
        with initialize(version_base=None, config_path=relative_config_dir):
            cfg = compose(config_name=config_name)

    device = f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu"
    model: EquivariantAutoencoder = load_model_from_checkpoint(
        checkpoint_path, cfg, device=device, module=EquivariantAutoencoder
    )
    model.to(device).eval()
    return model, cfg, device


def build_datamodule(cfg: DictConfig):
    """Instantiate and setup the matching datamodule."""
    if getattr(cfg, "data", None) is None:
        raise ValueError("Config missing data section")
    if getattr(cfg.data, "kind", None) == "synthetic":
        dm = SyntheticPointCloudDataModule(cfg)
    else:
        dm = RealPointCloudDataModule(cfg)
    dm.setup(stage="test")
    return dm


def _extract_pc_and_phase(batch: Any) -> Tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(batch, (tuple, list)):
        pc = batch[0]
        phase = batch[1] if len(batch) > 1 else None
    else:
        pc = batch
        phase = None
    if phase is not None and not torch.is_tensor(phase):
        phase = torch.as_tensor(phase)
    return pc, phase


def _downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    idx = np.random.default_rng(0).choice(len(points), size=max_points, replace=False)
    return points[idx]


def gather_inference_batches(
    model: EquivariantAutoencoder,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    max_batches: int = 4,
) -> Dict[str, np.ndarray]:
    """Collect originals, reconstructions, and latents from a few batches."""
    originals, reconstructions, inv_latents, eq_latents, phases = [], [], [], [], []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            pc, phase = _extract_pc_and_phase(batch)
            pc = pc.to(device)

            inv_z, recon, eq_z, _ = model(pc)
            originals.append(pc.detach().cpu())
            reconstructions.append(recon.detach().cpu())
            inv_latents.append(inv_z.detach().cpu())
            if eq_z is not None:
                eq_latents.append(eq_z.detach().cpu())
            if phase is not None:
                phases.append(phase.detach().view(-1).cpu())

    def _cat(tensors):
        return torch.cat(tensors, dim=0).numpy() if tensors else np.empty((0,))

    return {
        "originals": _cat(originals),
        "reconstructions": _cat(reconstructions),
        "inv_latents": _cat(inv_latents),
        "eq_latents": _cat(eq_latents),
        "phases": _cat(phases),
    }


def _chamfer(orig: np.ndarray, reco: np.ndarray) -> float:
    a = torch.tensor(orig, dtype=torch.float32).unsqueeze(0)
    b = torch.tensor(reco, dtype=torch.float32).unsqueeze(0)
    val, _ = chamfer_distance(a, b, squared=False, point_reduction="mean")
    return float(val.item())


def save_reconstruction_grid(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    phases: np.ndarray,
    out_file: Path,
    max_examples: int = 6,
    max_points: int = 2048,
) -> None:
    if originals.size == 0 or reconstructions.size == 0:
        return

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    count = int(min(len(originals), max_examples))

    fig = plt.figure(figsize=(8, 3 * count), dpi=150)
    for i in range(count):
        orig = _downsample(originals[i], max_points)
        reco = _downsample(reconstructions[i], max_points)
        cd = _chamfer(orig, reco)
        phase_label = int(phases[i]) if phases.size == len(originals) else None

        ax = fig.add_subplot(count, 2, i * 2 + 1, projection="3d")
        ax.scatter(orig[:, 0], orig[:, 1], orig[:, 2], s=2, alpha=0.7, c="#2c3e50")
        ax.set_title(f"Original{'' if phase_label is None else f' (phase {phase_label})'}")
        ax.axis("off")
        set_axes_equal(ax)

        ax = fig.add_subplot(count, 2, i * 2 + 2, projection="3d")
        ax.scatter(reco[:, 0], reco[:, 1], reco[:, 2], s=2, alpha=0.7, c="#e74c3c")
        ax.set_title(f"Reconstruction • CD={cd:.4f}")
        ax.axis("off")
        set_axes_equal(ax)

    plt.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def save_latent_tsne(
    inv_latents: np.ndarray,
    phases: np.ndarray,
    out_file: Path,
    max_samples: int = 4000,
) -> None:
    if inv_latents.size == 0 or len(inv_latents) < 2:
        return

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    latents = inv_latents
    labels = phases if phases.size == len(latents) else np.zeros(len(latents), dtype=int)

    if len(latents) > max_samples:
        idx = np.random.default_rng(0).choice(len(latents), size=max_samples, replace=False)
        latents = latents[idx]
        labels = labels[idx]

    tsne_coords = compute_tsne(latents)
    save_tsne_plot(tsne_coords, labels, out_file=str(out_file), title="Invariant latent space")


def evaluate_equivariance(
    model: EquivariantAutoencoder,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    max_batches: int = 2,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    eq_errors, chamfer_errors = [], []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            pc, _ = _extract_pc_and_phase(batch)
            pc = pc.to(device)
            batch_size = pc.shape[0]

            rots = torch.stack(
                [
                    torch.tensor(random_rotation_matrix(), device=device, dtype=pc.dtype)
                    for _ in range(batch_size)
                ]
            )
            pc_rot = torch.einsum("bij,bnj->bni", rots, pc)

            inv_z, recon, eq_z, _ = model(pc)
            _, recon_rot, eq_z_rot, _ = model(pc_rot)

            if eq_z is not None and eq_z_rot is not None:
                expected_eq = torch.einsum("bij,bcj->bci", rots, eq_z)
                rel = torch.linalg.norm(eq_z_rot - expected_eq, dim=-1) / torch.linalg.norm(
                    expected_eq, dim=-1
                ).clamp_min(1e-6)
                eq_errors.extend(rel.mean(dim=1).detach().cpu().numpy().tolist())

            recon_back = torch.einsum("bij,bnj->bni", rots.transpose(1, 2), recon_rot)
            for i in range(batch_size):
                cd, _ = chamfer_distance(
                    recon_back[i].unsqueeze(0).float(),
                    recon[i].unsqueeze(0).float(),
                    squared=False,
                    point_reduction="mean",
                )
                chamfer_errors.append(float(cd.detach().cpu().item()))

    eq_arr = np.asarray(eq_errors)
    cd_arr = np.asarray(chamfer_errors)
    metrics = {
        "eq_latent_rel_error_mean": float(eq_arr.mean()) if eq_arr.size else float("nan"),
        "eq_latent_rel_error_median": float(np.median(eq_arr)) if eq_arr.size else float("nan"),
        "recon_equiv_chamfer_mean": float(cd_arr.mean()) if cd_arr.size else float("nan"),
        "recon_equiv_chamfer_median": float(np.median(cd_arr)) if cd_arr.size else float("nan"),
        "num_samples": int(len(chamfer_errors)),
    }
    return metrics, eq_arr, cd_arr


def save_equivariance_plot(
    eq_errors: np.ndarray,
    chamfer_errors: np.ndarray,
    out_file: Path,
) -> None:
    if eq_errors.size == 0 and chamfer_errors.size == 0:
        return

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    if eq_errors.size:
        axes[0].hist(eq_errors, bins=30, color="#2980b9", alpha=0.8)
        axes[0].set_title("Equivariant latent relative error")
        axes[0].set_xlabel("||z_R - Rz|| / ||Rz||")
        axes[0].set_ylabel("count")
    else:
        axes[0].axis("off")
        axes[0].set_title("No equivariant latents collected")

    if chamfer_errors.size:
        axes[1].hist(chamfer_errors, bins=30, color="#c0392b", alpha=0.8)
        axes[1].set_title("Reconstruction equivariance (Chamfer)")
        axes[1].set_xlabel("CD(rot⁻¹(f(Rx)), f(x))")
        axes[1].set_ylabel("count")
    else:
        axes[1].axis("off")
        axes[1].set_title("No reconstruction pairs collected")

    plt.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def run_post_training_analysis(
    checkpoint_path: str,
    output_dir: str,
    cuda_device: int = 0,
    cfg: DictConfig | None = None,
    max_batches: int = 4,
) -> None:
    """Generate qualitative and quantitative diagnostics for the Equivariant AE."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg, device = load_eq_model(checkpoint_path, cuda_device=cuda_device, cfg=cfg)
    dm = build_datamodule(cfg)
    dl = dm.test_dataloader()

    cache = gather_inference_batches(model, dl, device, max_batches=max_batches)

    save_reconstruction_grid(
        cache["originals"],
        cache["reconstructions"],
        cache["phases"],
        out_dir / "recon_vs_prediction.png",
    )

    save_latent_tsne(cache["inv_latents"], cache["phases"], out_dir / "latent_tsne.png")

    metrics, eq_err, cd_err = evaluate_equivariance(model, dl, device, max_batches=2)
    save_equivariance_plot(eq_err, cd_err, out_dir / "equivariance.png")

    metrics_path = out_dir / "analysis_metrics.json"
    with metrics_path.open("w") as handle:
        import json

        json.dump(metrics, handle, indent=2)

    print(f"Saved reconstruction, latent, and equivariance analyses to {out_dir}")
