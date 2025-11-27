import os
import sys
from typing import Tuple, List

import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import DictConfig

sys.path.append(os.getcwd())

from src.data_utils.data_load import PointCloudDataset
from src.training_methods.spd.spd_module import ShapePoseDisentanglement
from src.utils.model_utils import load_model_from_checkpoint, resolve_config_path


def load_spd_model(
    checkpoint_path: str,
    cuda_device: int = 0,
    fallback_config_path: str | None = None,
    cfg: DictConfig | None = None,
) -> Tuple[ShapePoseDisentanglement, DictConfig, str]:
    """Restore the SPD model together with its Hydra *cfg* and *device* string."""
    if cfg is None:
        config_dir, config_name = resolve_config_path(checkpoint_path)

        if config_dir is None:
            config_dir = os.path.dirname(fallback_config_path)
            config_name = os.path.splitext(os.path.basename(fallback_config_path))[0]

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
        absolute_config_dir = os.path.join(project_root, config_dir)
        relative_config_dir = os.path.relpath(absolute_config_dir, current_dir)

        # Check if Hydra is already initialized to avoid errors
        from hydra.core.global_hydra import GlobalHydra
        if GlobalHydra.instance().is_initialized():
            # If already initialized, we assume the config is available or we can't easily change it.
            # However, since we are loading a specific checkpoint, we ideally want THAT checkpoint's config.
            # If we are here, it means we didn't pass 'cfg' but Hydra is initialized.
            # This might happen if we are running in a notebook or another script.
            # We'll try to compose, hoping the config path is compatible.
            # If not, we might need to rely on the user passing 'cfg'.
            try:
                cfg = compose(config_name=config_name)
            except Exception as e:
                print(f"Warning: Hydra already initialized but failed to compose {config_name}: {e}")
                print("Attempting to clear GlobalHydra and re-initialize (this might affect other parts of the application)...")
                GlobalHydra.instance().clear()
                with initialize(version_base=None, config_path=relative_config_dir):
                    cfg = compose(config_name=config_name)
        else:
            with initialize(version_base=None, config_path=relative_config_dir):
                cfg = compose(config_name=config_name)

    device = f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu"
    model: ShapePoseDisentanglement = load_model_from_checkpoint(
        checkpoint_path, cfg, device=device, module=ShapePoseDisentanglement
    )
    model.to(device).eval()
    return model, cfg, device


def create_spd_dataloader(
    cfg: DictConfig,
    file_paths: str | List[str],
    shuffle: bool = False,
    max_samples: int | None = None,
    return_coords: bool = False,
    batch_size: int | None = None,
) -> torch.utils.data.DataLoader:
    """Build a dataloader from one or more .off files for SPD inference."""
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
    """Ensure *points* is a (B, N, 3) float32 tensor located on *device*."""
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32)
    if points.dim() == 2:
        points = points.unsqueeze(0)  # (N, 3) -> (1, N, 3)
    if points.dim() == 3 and points.shape[1] == 3:
        points = points.permute(0, 2, 1)  # (B, 3, N) -> (B, N, 3)
    return points.to(device)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


@torch.inference_mode()
def predict_reconstructions(
    model: ShapePoseDisentanglement,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (originals, reconstructions) for all samples in *dataloader*."""
    originals, recos = [], []
    for batch in dataloader:
        pts = batch[0] if isinstance(batch, (list, tuple)) else batch
        pts = _prepare_batch(pts, device)
        _, reco, _, _, _ = model(pts)  # reco: (B, 3, N)
        originals.append(_to_numpy(pts))
        recos.append(_to_numpy(reco.permute(0, 2, 1)))  # -> (B, N, 3)

    return np.concatenate(originals, axis=0), np.concatenate(recos, axis=0)


@torch.inference_mode()
def predict_latents(
    model: ShapePoseDisentanglement,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> np.ndarray:
    """Return invariant latent codes for every sample in *dataloader*."""
    latents = []
    for batch in dataloader:
        pts = batch[0] if isinstance(batch, (list, tuple)) else batch
        inv_z, _, _, _, _ = model(_prepare_batch(pts, device))
        latents.append(_to_numpy(inv_z))
    return np.concatenate(latents, axis=0)


@torch.inference_mode()
def predict_single_latent(
    points: torch.Tensor | np.ndarray,
    model: ShapePoseDisentanglement,
    device: str = "cpu",
) -> np.ndarray:
    """Predict invariant latent for one point cloud."""
    inv_z = model(_prepare_batch(points, device))[0]
    return _to_numpy(inv_z.squeeze(0))


@torch.inference_mode()
def predict_single_reconstruction(
    points: torch.Tensor | np.ndarray,
    model: ShapePoseDisentanglement,
    device: str = "cpu",
) -> np.ndarray:
    """Predict reconstruction for one point cloud (shape: (N, 3))."""
    reco = model(_prepare_batch(points, device))[1]
    return _to_numpy(reco.permute(0, 2, 1).squeeze(0))



if __name__ == "__main__":
    CKPT = "output/2025-07-28/19-55-08/SPD_FoldingSphereAttn_l72_P80_Sinkhorn_1024-epoch=224-val_loss=0.02.ckpt"
    FILE = '240ps.off'

    model, cfg, device = load_spd_model(CKPT)
    dl = create_spd_dataloader(cfg, FILE)
    orig, rec = predict_reconstructions(model, dl, device)
    print("originals:", orig.shape, "recons:", rec.shape)
    lat = predict_latents(model, dl, device)
    print("latents:", lat.shape)


