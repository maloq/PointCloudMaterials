from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import DictConfig

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

sys.path.append(os.getcwd())
from src.data_utils.data_load import PointCloudDataset
from src.training_methods.autoencoder.autoencoder_module import PointNetAutoencoder
from src.utils.model_utils import load_model_from_checkpoint, resolve_config_path


def load_autoencoder_model(
    checkpoint_path: str,
    cuda_device: int = 0,
    fallback_config_path: str | None = None,
) -> Tuple[PointNetAutoencoder, DictConfig, str]:
    """Restore the autoencoder together with its Hydra *cfg* and *device* string."""
    config_dir, config_name = resolve_config_path(checkpoint_path)
    if config_dir is None:
        config_dir = os.path.dirname(fallback_config_path)
        config_name = os.path.splitext(os.path.basename(fallback_config_path))[0]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    absolute_config_dir = os.path.join(project_root, config_dir)
    relative_config_dir = os.path.relpath(absolute_config_dir, current_dir)

    with initialize(version_base=None, config_path=relative_config_dir):
        cfg = compose(config_name=config_name)

    device = f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu"
    model: PointNetAutoencoder = load_model_from_checkpoint(
        checkpoint_path, cfg, device=device, module=PointNetAutoencoder
    )
    model.to(device).eval()
    return model, cfg, device

def create_autoencoder_dataloader(
    cfg: DictConfig,
    file_paths: str | List[str],
    shuffle: bool = False,
    max_samples: int | None = None,
    return_coords: bool = False,
    batch_size: int | None = None,
) -> torch.utils.data.DataLoader:
    """Build a *torch* dataloader for the given .off files.

    The parameters are intentionally identical to ``eval_spd.create_dataloader``.
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    dataset = PointCloudDataset(
        root=cfg.data.data_path,
        data_files=file_paths,
        return_coords=return_coords,
        sample_type="regular",
        radius=cfg.data.radius,
        overlap_fraction=cfg.data.overlap_fraction,
        n_samples=cfg.data.n_samples,
        num_points=cfg.data.num_points,
        pre_normalize=True,
        normalize=True,
    )

    if max_samples is not None:
        from torch.utils.data import Subset

        dataset = Subset(dataset, list(range(max_samples)))
    if batch_size is None:
        batch_size = cfg.batch_size
    
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



def _prepare_batch(points: torch.Tensor | np.ndarray, device: str) -> torch.Tensor:
    """Ensure *points* is a float32 tensor of shape (B, 3, N) located on *device*."""
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32)

    # If input is (N, 3) add batch dim; if (B, N, 3) transpose; if already (B, 3, N) keep
    if points.dim() == 2:  # (N, 3)
        points = points.unsqueeze(0)
    if points.shape[-1] == 3 and points.shape[1] != 3:  # (B, N, 3)
        points = points.permute(0, 2, 1)  # -> (B, 3, N)

    return points.to(device)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()



@torch.inference_mode()
def predict_reconstructions(
    model: PointNetAutoencoder,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return *(originals, reconstructions)* for every batch in *dataloader*."""
    originals, recos = [], []
    for batch in dataloader:
        pts = batch[0] if isinstance(batch, (list, tuple)) else batch
        pts_prepped = _prepare_batch(pts, device)
        reco, _, _ = model(pts_prepped)  # (B, 3, N)
        originals.append(_to_numpy(pts_prepped.permute(0, 2, 1)))  # -> (B, N, 3)
        recos.append(_to_numpy(reco.permute(0, 2, 1)))

    return np.concatenate(originals, axis=0), np.concatenate(recos, axis=0)


@torch.inference_mode()
def predict_latents(
    model: PointNetAutoencoder,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> np.ndarray:
    """Return latent codes for every sample in *dataloader*."""
    latents = []
    for batch in dataloader:
        pts = batch[0] if isinstance(batch, (list, tuple)) else batch
        _, latent, _ = model(_prepare_batch(pts, device))
        latents.append(_to_numpy(latent))
    return np.concatenate(latents, axis=0)


@torch.inference_mode()
def predict_single_latent(
    points: torch.Tensor | np.ndarray,
    model: PointNetAutoencoder,
    device: str = "cpu",
) -> np.ndarray:
    """Predict latent vector for one point cloud."""
    _, latent, _ = model(_prepare_batch(points, device))
    return _to_numpy(latent.squeeze(0))


@torch.inference_mode()
def predict_single_reconstruction(
    points: torch.Tensor | np.ndarray,
    model: PointNetAutoencoder,
    device: str = "cpu",
) -> np.ndarray:
    """Predict reconstruction for one point cloud (returns array of shape *(N, 3)*)."""
    reco, _, _ = model(_prepare_batch(points, device))
    return _to_numpy(reco.permute(0, 2, 1).squeeze(0))


if __name__ == "__main__":
    CKPT = "output/2025-07-31/00-06-11/PnE_L_FoldingSphereAttn_l64_P80_Sinkhorn_4096-epoch=09-val_loss=0.02.ckpt"
    FILE = "240ps.off"

    model, cfg, device = load_autoencoder_model(CKPT)
    dl = create_autoencoder_dataloader(cfg, FILE)

    orig, rec = predict_reconstructions(model, dl, device)
    print("originals:", orig.shape, "recons:", rec.shape)

    lat = predict_latents(model, dl, device)
    print("latents:", lat.shape)

