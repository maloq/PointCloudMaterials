"""Utilities for extracting latent representations from trained models.

This module focuses on preparing the data that will later be used for the
clustering and visualisation experiments living in
`src/clustering/run_clustering.py`.  Functions in this file are **only**
responsible for I/O:

1. Loading a trained model from a checkpoint
2. Running that model on a set of OFF files and collecting the latent vectors
3. Saving the resulting latent‒point cloud‒label bundle as a compressed
   `.npz` archive for downstream use.

Keeping these responsibilities separate from the actual clustering logic makes
the codebase easier to follow and avoids circular dependencies between the
"data preparation" and "analysis" stages.
"""

# Standard library
import os
from enum import Enum
from typing import List, Tuple

# Third-party libraries
import numpy as np
import torch

# Local application imports
from src.training_methods.autoencoder.eval_autoencoder import (
    create_autoencoder_dataloader,
    load_ae_model_and_config,
)
from src.training_methods.spd.eval_spd import load_spd_model_and_config
import warnings
warnings.filterwarnings("ignore")
print(f"Running from {os.getcwd()}")


class ModelType(str, Enum):
    """Enumeration of the supported model back-ends for latent extraction."""

    AUTOENCODER = "autoencoder"
    SPD = "spd"

    @classmethod
    def from_string(cls, value: str) -> "ModelType":
        """Parse a user-provided string value into a :class:`ModelType`."""

        try:
            return cls(value.strip().lower())
        except ValueError as exc:
            raise ValueError(
                f"Unknown model_type '{value}'. Supported values: "
                f"{', '.join([m.value for m in cls])}"
            ) from exc


# ---------------------------------------------------------------------
# Model loading helper
# ---------------------------------------------------------------------

def load_model_and_config(
    *,
    checkpoint_path: str,
    model_type: "ModelType | str" = ModelType.AUTOENCODER,
    cuda_device: int = 0,
    fallback_config_path: str = "autoencoder",
) -> Tuple[torch.nn.Module, "dict", str]:
    """Load a trained model *and* its Hydra configuration in a uniform way.

    This thin wrapper unifies the branching logic required to instantiate either
    an *autoencoder* or an *SPD* model based solely on :pydata:`model_type`.

    Parameters
    ----------
    checkpoint_path
        Filesystem path to the model checkpoint produced by *PyTorch Lightning*.
    model_type
        Identifier specifying which back-end implementation to load. Accepts
        either a :class:`ModelType` enum value or a case-insensitive string.
    cuda_device
        GPU index on which to place the model. Use ``-1`` to force CPU usage.
    fallback_config_path
        Optional Hydra config to fall back to when the checkpoint lacks the
        necessary information.

    Returns
    -------
    Tuple[torch.nn.Module, cfg, torch.device]
        The instantiated model, its Hydra config, and the device reference.
    """

    # Normalise *model_type* into an enum instance for easier comparison
    if isinstance(model_type, str):
        model_type = ModelType.from_string(model_type)

    if model_type is ModelType.AUTOENCODER:
        return load_ae_model_and_config(
            checkpoint_path=checkpoint_path,
            cuda_device=cuda_device,
            fallback_config_path=fallback_config_path,
        )
    elif model_type is ModelType.SPD:
        return load_spd_model_and_config(
            checkpoint_path=checkpoint_path,
            cuda_device=cuda_device,
            fallback_config_path=fallback_config_path,
        )
    else:  # Should be impossible due to enum validation above
        raise RuntimeError(f"Unsupported model_type: {model_type}")


def run_clustering_pipeline(
    *,
    model: torch.nn.Module,
    cfg: dict,
    device,
    save_folder: str,
    liquid_file_paths: List[str],
    crystal_file_paths: List[str],
    max_samples: int | None = None,
) -> torch.nn.Module:
    """Load a trained model, extract latents, and save them to disk.

    Parameters
    ----------
    checkpoint_path
        Path to the model checkpoint produced by *Lightning*.
    save_folder
        Destination directory for the compressed ``latent_data.npz`` file.
    liquid_file_paths, crystal_file_paths
        Lists of ``.off`` (or similarly supported) structures representing the
        *liquid* and *crystal* phases, respectively.
    cuda_device
        GPU index to place the model on.  Use ``-1`` for CPU-only execution.
    max_samples
        Optionally limit the number of processed samples **per phase**.
    model_type
        One of :pydata:`ModelType.AUTOENCODER` or :pydata:`ModelType.SPD`.  Can
        be supplied as a case-insensitive string for convenience.
    config_path
        Fallback Hydra config file in case the information cannot be recovered
        from the checkpoint.
    """

    # Predict and save latent vectors
    predict_and_save_latent(cfg=cfg,  # Pass the loaded hydra config
                            model=model,
                            liquid_file_paths=liquid_file_paths,
                            crystal_file_paths=crystal_file_paths,
                            device=device,
                            save_folder=save_folder,
                            max_samples=max_samples) # Pass max_samples
    return model


def get_latents_from_dataloader(model, dataloader, device: str = 'cpu') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract latent representations and reconstructed point clouds from a
    dataloader, picking the correct latent-prediction routine based on the
    model type.

        • ShapePoseDisentanglement → spd_predict_latent
        • PointNetAutoencoder      → autoencoder_predict_latent
        • fallback                 → best-effort parse of model output

    Returns
    -------
    latents        : (M, D)  np.ndarray
    point_clouds   : (M, N, 3) np.ndarray
    originals      : (M, N, 3) np.ndarray
    """
    if len(dataloader) == 0:
        raise ValueError("Dataloader is empty - no data to process")

    # Local imports to avoid circular dependencies at module import time
    from src.training_methods.spd.spd_module import ShapePoseDisentanglement
    from src.training_methods.spd.eval_spd import spd_predict_latent
    from src.training_methods.autoencoder.autoencoder_module import PointNetAutoencoder
    from src.training_methods.autoencoder.eval_autoencoder import autoencoder_predict_latent

    latents_list, point_clouds_list, originals_list = [], [], []

    for batch in dataloader:
        # Some datasets return (points, coords); handle both cases
        points = batch[0] if isinstance(batch, (tuple, list)) else batch
        points = points.to(device)

        # ----------------------------------------------------------
        # Choose the right latent-prediction path
        # ----------------------------------------------------------
        if isinstance(model, ShapePoseDisentanglement):
            latent_np = spd_predict_latent(points, model, device=device)

            with torch.no_grad():
                _, recon, _, _ = model(points)           # (B, N, 3)
            recon_tensor = recon

        elif isinstance(model, PointNetAutoencoder):
            latent_np = autoencoder_predict_latent(points, model, device=device)

            with torch.no_grad():
                recon_tensor, _, _ = model(points.transpose(2, 1))  # (B, N, 3)

        else:
            # Generic fallback – try to interpret model output
            with torch.no_grad():
                out = model(points)
            if isinstance(out, tuple):
                if out[0].shape[-1] == 3:      # (recon, latent, ...)
                    recon_tensor, latent_tensor = out[0], out[1]
                else:                          # (latent, recon, ...)
                    latent_tensor, recon_tensor = out[0], out[1]
            else:
                raise RuntimeError("Unable to parse model output for latent/reconstruction")
            latent_np = latent_tensor.detach().cpu().numpy()

        # ----------------------------------------------------------
        # Collect results
        # ----------------------------------------------------------
        latents_list.append(latent_np)
        point_clouds_list.append(recon_tensor.detach().cpu().numpy())
        originals_list.append(points.detach().cpu().numpy())

    return (np.concatenate(latents_list, axis=0),
            np.concatenate(point_clouds_list, axis=0),
            np.concatenate(originals_list, axis=0))


def _process_files(cfg, model, file_paths: List[str], label: str, device: str, max_samples: int = None):
    """Helper function to process a list of files and extract latents."""
    if not file_paths:
        raise ValueError(f"No {label} file paths provided")
        
    print(f"Processing {label} datasets...")
    all_latents, all_point_clouds, all_originals = [], [], []
    
    for i, file_path in enumerate(file_paths):
        print(f"Processing {label} file {i+1}/{len(file_paths)}: {file_path}")
        dataloader = create_autoencoder_dataloader(cfg, file_path, shuffle=True, max_samples=max_samples)
        latents, point_clouds, originals = get_latents_from_dataloader(model, dataloader, device)
        
        all_latents.append(latents)
        all_point_clouds.append(point_clouds)
        all_originals.append(originals)
        print(f"  {len(latents)} samples from {file_path}")
    
    if not all_latents:
        raise ValueError(f"No data was successfully loaded from any {label} files")
        
    return (np.concatenate(all_latents, axis=0),
            np.concatenate(all_point_clouds, axis=0), 
            np.concatenate(all_originals, axis=0))


def predict_and_save_latent(cfg: str,
                            model,
                            liquid_file_paths: List[str],
                            crystal_file_paths: List[str],
                            device: str = 'cpu',
                            save_folder: str = 'output',
                            max_samples: int = None):

    if not liquid_file_paths and not crystal_file_paths:
        raise ValueError("At least one of liquid_file_paths or crystal_file_paths must be provided")

    model.to(device).eval()
    
    # Split max_samples if specified
    if max_samples:
        max_samples_per_type = max_samples // 2
        max_samples_l = max_samples_per_type
        max_samples_c = max_samples - max_samples_per_type
    else:
        max_samples_l = max_samples_c = None

    # Process files - these will raise errors if no data is found
    latents_l, point_clouds_l, originals_l = _process_files(cfg, model, liquid_file_paths, "liquid", device, max_samples_l)
    latents_c, point_clouds_c, originals_c = _process_files(cfg, model, crystal_file_paths, "crystal", device, max_samples_c)
    
    # Create labels and combine data
    labels = np.array(["liquid"] * len(latents_l) + ["crystal"] * len(latents_c))
    all_latents = np.concatenate((latents_l, latents_c), axis=0)
    all_points = np.concatenate((point_clouds_l, point_clouds_c), axis=0)
    all_originals = np.concatenate((originals_l, originals_c), axis=0)

    print(f"Total samples: {len(all_latents)} (liquid: {len(latents_l)}, crystal: {len(latents_c)})")

    # Save combined data
    output_path = os.path.join(save_folder, "latent_data.npz")
    os.makedirs(save_folder, exist_ok=True)
    
    np.savez_compressed(output_path, 
                        latents=all_latents, 
                        points=all_points, 
                        originals=all_originals, 
                        labels=labels)

    print(f"Saved combined data to {output_path}")


# Example usage (optional, can be removed or put under if __name__ == '__main__')
if __name__ == '__main__':
    checkpoint_path = 'output/2025-04-15/05-09-11/pointnet-epoch=7999-val_loss=0.04.ckpt'
    save_folder = 'output'
    liquid_file_paths = ['datasets/Al/inherent_configurations_off/166ps.off']
    crystal_file_paths = ['datasets/Al/inherent_configurations_off/240ps.off']
    # Choose between "autoencoder" and "spd" depending on the checkpoint
    model_type = "autoencoder"

    run_clustering_pipeline(
        checkpoint_path=checkpoint_path,
        save_folder=save_folder,
        liquid_file_paths=liquid_file_paths,
        crystal_file_paths=crystal_file_paths,
        model_type=model_type,
        cuda_device=0,
        max_samples=1000,
    ) 