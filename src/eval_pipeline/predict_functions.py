from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, List, Mapping, Tuple

import numpy as np
import torch
import sys,os
import json
import shutil
import sys,os
from omegaconf import OmegaConf
from tqdm import tqdm
sys.path.append(os.getcwd())
from src.training_methods.autoencoder.eval_autoencoder import (
    create_autoencoder_dataloader,
    load_autoencoder_model,
)
from src.training_methods.spd.eval_spd import load_spd_model, create_spd_dataloader
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class Backend:
    """Bundle of helper call‑backs required for one model family."""

    load_model_and_cfg: Callable[..., Tuple[torch.nn.Module, dict, str]]
    create_dataloader: Callable[..., torch.utils.data.DataLoader]


class ModelType(str, Enum):
    AUTOENCODER = "autoencoder"
    SPD = "spd"

    @classmethod
    def from_any(cls, value: "ModelType | str") -> "ModelType":
        if isinstance(value, cls):
            return value
        try:
            return cls(value.strip().lower())
        except ValueError:  # pragma: no cover – caught early during dev
            opts = ", ".join(m.value for m in cls)
            raise ValueError(f"Unknown model_type '{value}'. Supported: {opts}")


BACKENDS: Mapping[ModelType, Backend] = {
    ModelType.AUTOENCODER: Backend(
        load_model_and_cfg=load_autoencoder_model,
        create_dataloader=create_autoencoder_dataloader,
    ),
    ModelType.SPD: Backend(
        load_model_and_cfg=load_spd_model,
        create_dataloader=create_spd_dataloader,
    ),
}


@torch.inference_mode()
def _get_latents_from_dataloader(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    *,
    device: str = "cpu",
    return_coords: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Return *(latents, reconstructions, originals)* for every batch in *dataloader*."""

    from src.training_methods.spd.spd_module import ShapePoseDisentanglement
    from src.training_methods.autoencoder.autoencoder_module import PointNetAutoencoder

    lats, recs, origs, coords = [], [], [], []
    for batch in tqdm(dataloader, desc="Extracting latents"):
        if return_coords:
            points, coords_batch = batch
            coords_batch = coords_batch.reshape(coords_batch.shape[0], 1, 3)
            points = points.to(device)
        else:
            points = batch  # (B, N, 3)

        if isinstance(model, ShapePoseDisentanglement):
            inv_z, recon, _, _ = model(points)
            latent = inv_z
        elif isinstance(model, PointNetAutoencoder):
            recon, latent, _ = model(points.transpose(2, 1))  # AE expects (B, 3, N)
        else:  # Fallback – best‑effort guess
            out = model(points)
            latent, recon = out[:2] if isinstance(out, (tuple, list)) else (out, points)

        coords.append(coords_batch.detach().cpu().numpy())
        lats.append(latent.detach().cpu().numpy())
        recs.append(recon.detach().cpu().numpy())
        origs.append(points.detach().cpu().numpy())

    return (
        np.concatenate(lats, axis=0),
        np.concatenate(recs, axis=0),
        np.concatenate(origs, axis=0),
        np.concatenate(coords, axis=0) if coords else None,
    )


def _extract_and_save_phase(
    label: str,
    paths: Iterable[str],
    *,
    backend: Backend,
    cfg: dict,
    model: torch.nn.Module,
    device: str,
    max_samples: int | None,
    save_folder: str,
) -> None:
    latents_all, recs_all, origs_all = [], [], []
    save_dir = Path(save_folder)
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(paths):
        dl = backend.create_dataloader(
            cfg, path, shuffle=True, max_samples=max_samples, return_coords=True
        )
        l, r, o, c = _get_latents_from_dataloader(model, dl, device=device, return_coords=True)

        stem = Path(path).stem  # e.g. "240ps"
        out_file = save_dir / f"latent_data_{stem}.npz"

        data_to_save = {
            "latents": l.astype(np.float32),
            "reconstructions": r.astype(np.float32),
            "originals": o.astype(np.float32),
            "labels": np.array([label] * len(l)),
        }
        if c is not None:
            data_to_save["coords"] = c.astype(np.float32)

        print("latents shape", l.shape)
        print("coords shape", c.shape)

        np.savez_compressed(out_file, **data_to_save)

        print(f"[✓] Saved {out_file}")

        latents_all.append(l)
        recs_all.append(r)
        origs_all.append(o)
        print(f"[{label}] {i+1}/{len(list(paths))}: {len(l)} samples from {path}")


def load_model_for_inference(
    checkpoint_path: str,
    model_type: "ModelType | str",
    cuda_device: int,
) -> Tuple[torch.nn.Module, dict, str, Backend, ModelType]:
    """Loads a model for inference from a checkpoint."""
    model_type = ModelType.from_any(model_type)
    backend = BACKENDS[model_type]
    model, cfg, device = backend.load_model_and_cfg(
        checkpoint_path=checkpoint_path,
        cuda_device=cuda_device,
        fallback_config_path="autoencoder_80",
    )
    model.to(device).eval()
    return model, cfg, device, backend, model_type


def save_latents_to_file(
    *,
    model: torch.nn.Module,
    cfg: dict,
    device: str,
    backend: Backend,
    model_type: ModelType,
    save_folder: str,
    checkpoint_path: str,
    liquid_file_paths: List[str] | None = None,
    crystal_file_paths: List[str] | None = None,
    max_samples: int | None = None,
) -> torch.nn.Module:
    """Extract latents using a pre-loaded model and save the bundle for clustering.

    Parameters
    ----------
    model : torch.nn.Module
        The loaded model instance.
    cfg : dict
        Model configuration dictionary.
    device : str
        The device the model is on.
    backend : Backend
        The backend for the model.
    model_type : ModelType
        The type of the model.
    save_folder : str
    checkpoint_path : str
        Path to the model checkpoint, which will be saved along with the latent data.
    liquid_file_paths, crystal_file_paths : list[str]
        OFF/PLY files representing different phases.  You may pass further
        phases by editing the code to include additional ``*_file_paths``
        arguments.
    max_samples : int | None
        Limit the *total* number of processed samples (evenly split between
        phases).  ``None`` means "use everything".
    """

    phases: list[tuple[str, list[str]]] = [
        ("liquid", liquid_file_paths or []),
        ("crystal", crystal_file_paths or []),
    ]

    if all(len(p[1]) == 0 for p in phases):
        raise ValueError("You must provide at least one OFF file to process.")


    full_save_folder = os.path.join(
        save_folder, model_type.value, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    save_dir = Path(full_save_folder)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=4)
    shutil.copy(checkpoint_path, save_dir / "checkpoint.ckpt")

    latents, recs, origs, labels = [], [], [], []
    for label, paths in phases:
        if not paths:
            continue
        _extract_and_save_phase(
            label,
            paths,
            backend=backend,
            cfg=cfg,
            model=model,
            device=device,
            max_samples=max_samples,
            save_folder=full_save_folder,
        )   
    
    print(f"Saved to {full_save_folder}")
    return full_save_folder


def load_latents(folder: str, files: list[str]):
    """Loads and concatenates latent data from multiple .npz files."""
    if not files:
        raise ValueError("No files to load")

    data_keys = ['latents', 'reconstructions', 'originals', 'labels', 'coords']
    data_lists = {key: [] for key in data_keys}

    for file in files:
        file_path = f"{folder}/latent_data_{file}.npz"
        print(file_path)
        with np.load(file_path) as data:
            for key in data_keys:
                data_lists[key].append(data[key])

    concatenated_data = tuple(
        np.concatenate(data_lists[key], axis=0) for key in data_keys
    )

    return concatenated_data


if __name__ == "__main__":
    model, cfg, device, backend, model_type = load_model_for_inference(
        checkpoint_path="output/2025-07-31/00-06-11/PnE_L_FoldingSphereAttn_l64_P80_Sinkhorn_4096-epoch=09-val_loss=0.02.ckpt",
        model_type="autoencoder",
        cuda_device=0,
    )
    save_latents_to_file(
        model=model,
        cfg=cfg,
        device=device,
        backend=backend,
        model_type=model_type,
        save_folder="output",
        checkpoint_path="output/2025-07-31/00-06-11/PnE_L_FoldingSphereAttn_l64_P80_Sinkhorn_4096-epoch=09-val_loss=0.02.ckpt",
        liquid_file_paths=["166ps.off"],
        crystal_file_paths=["240ps.off"],
        max_samples=1000,
    )
