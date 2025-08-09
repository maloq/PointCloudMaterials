from __future__ import annotations

"""Model predictors used by the evaluation pipeline."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Type, Any

import numpy as np
import torch
from tqdm.auto import tqdm

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from src.training_methods.autoencoder.eval_autoencoder import create_autoencoder_dataloader
    from src.training_methods.spd.eval_spd import create_spd_dataloader


@dataclass
class PredictionBundle:
    """Container returned by :meth:`Predictor.predict`."""

    latents: np.ndarray
    reconstructions: np.ndarray | None
    originals: np.ndarray


class Predictor(ABC):
    """Abstract predictor that wraps a trained model."""

    model: Any
    device: str

    @classmethod
    @abstractmethod
    def from_checkpoint(
        cls, cfg, checkpoint_path: str, device: str
    ) -> "Predictor":
        """Restore predictor from *checkpoint_path*."""

    @abstractmethod
    def predict(self, dataloader: torch.utils.data.DataLoader) -> PredictionBundle:
        """Return latents (and optionally reconstructions) for *dataloader*."""

    @abstractmethod
    def predict_raw(self, points: np.ndarray) -> np.ndarray:
        """Predict a single latent from raw ``(N,3)`` point cloud."""


class AutoencoderPredictor(Predictor):
    """Predictor for the autoencoder model."""

    def __init__(self, model: torch.nn.Module, device: str):
        self.model = model
        self.device = device

    @classmethod
    def from_checkpoint(cls, cfg, checkpoint_path: str, device: str) -> "AutoencoderPredictor":
        from src.training_methods.autoencoder.eval_autoencoder import load_autoencoder_model

        cuda = int(device.split(":")[-1]) if device.startswith("cuda") else 0
        model, _cfg, dev = load_autoencoder_model(
            checkpoint_path, cuda_device=cuda, fallback_config_path=cfg.get("config_path", None)
        )
        return cls(model, dev)

    @torch.inference_mode()
    def predict(self, dataloader: torch.utils.data.DataLoader) -> PredictionBundle:  # type: ignore[override]
        lats, recs, origs = [], [], []
        for batch in dataloader:
            pts = batch[0] if isinstance(batch, (list, tuple)) else batch
            pts = pts.to(self.device)
            recon, latent, _ = self.model(pts.transpose(2, 1))
            lats.append(latent.detach().cpu().numpy())
            recs.append(recon.permute(0, 2, 1).detach().cpu().numpy())
            origs.append(pts.detach().cpu().numpy())
        return PredictionBundle(
            latents=np.concatenate(lats, axis=0),
            reconstructions=np.concatenate(recs, axis=0),
            originals=np.concatenate(origs, axis=0),
        )

    @torch.inference_mode()
    def predict_raw(self, points: np.ndarray) -> np.ndarray:  # type: ignore[override]
        pts = torch.tensor(points, dtype=torch.float32, device=self.device)
        if pts.ndim == 2:
            pts = pts.unsqueeze(0)
        _, latent, _ = self.model(pts.transpose(2, 1))
        return latent.squeeze(0).detach().cpu().numpy()


class SPDPredictor(Predictor):
    """Predictor for the SPD model."""

    def __init__(self, model: torch.nn.Module, device: str):
        self.model = model
        self.device = device

    @classmethod
    def from_checkpoint(cls, cfg, checkpoint_path: str, device: str) -> "SPDPredictor":
        from src.training_methods.spd.eval_spd import load_spd_model

        cuda = int(device.split(":")[-1]) if device.startswith("cuda") else 0
        model, _cfg, dev = load_spd_model(
            checkpoint_path, cuda_device=cuda, fallback_config_path=cfg.get("config_path", None)
        )
        return cls(model, dev)

    @torch.inference_mode()
    def predict(self, dataloader: torch.utils.data.DataLoader) -> PredictionBundle:  # type: ignore[override]
        lats, recs, origs = [], [], []
        for batch in dataloader:
            pts = batch[0] if isinstance(batch, (list, tuple)) else batch
            pts = pts.to(self.device)
            inv_z, recon, _, _ = self.model(pts)
            lats.append(inv_z.detach().cpu().numpy())
            recs.append(recon.permute(0, 2, 1).detach().cpu().numpy())
            origs.append(pts.detach().cpu().numpy())
        return PredictionBundle(
            latents=np.concatenate(lats, axis=0),
            reconstructions=np.concatenate(recs, axis=0),
            originals=np.concatenate(origs, axis=0),
        )

    @torch.inference_mode()
    def predict_raw(self, points: np.ndarray) -> np.ndarray:  # type: ignore[override]
        pts = torch.tensor(points, dtype=torch.float32, device=self.device)
        if pts.ndim == 2:
            pts = pts.unsqueeze(0)
        inv_z, _, _, _ = self.model(pts)
        return inv_z.squeeze(0).detach().cpu().numpy()


class SOAPPredictor(Predictor):
    """Predictor for SOAP + PCA features."""

    def __init__(self, soap, pca, species: str = "Al"):
        self.soap = soap
        self.pca = pca
        self.species = species
        self.device = "cpu"

    @classmethod
    def from_checkpoint(cls, cfg, checkpoint_path: str, device: str) -> "SOAPPredictor":
        """
        Restore a SOAP+PCA predictor.

        Two modes are supported:
        1) If checkpoint_path points to a joblib bundle with keys 'soap' and 'pca',
           load them directly.
        2) Otherwise, fall back to building SOAP and fitting PCA on-the-fly using the
           dataset described by cfg.data. This makes SOAP usable without a model
           checkpoint. The fallback limits the number of samples for speed.
        """
        # First try to load a precomputed bundle
        try:
            import joblib  # local import to avoid an unconditional dependency

            bundle = joblib.load(checkpoint_path)
            soap = bundle["soap"]
            pca = bundle["pca"]
            species = bundle.get("species", "Al")
            return cls(soap, pca, species)
        except Exception:
            # Fall back to on-the-fly fitting from data
            pass

        # Fallback path: build SOAP and fit PCA from a subset of the evaluation dataset
        # We intentionally reuse the autoencoder dataloader which yields (B, N, 3)
        from src.training_methods.autoencoder.eval_autoencoder import (
            create_autoencoder_dataloader,
        )
        from src.training_methods.SOAP.predict_soap_pca import fit_soap_pca

        # Sensible defaults for quick fitting; can be tuned later via config if needed
        max_fit_pointclouds = int(getattr(cfg, "soap_max_fit_pointclouds", 10000))
        soap_params = dict(getattr(cfg, "soap_params", {}))
        n_components = getattr(cfg, "soap_pca_components", 32)
        species = getattr(cfg, "soap_species", "Al")

        # Build a small dataloader to sample point clouds for fitting
        file_paths = cfg.data.data_files
        dl = create_autoencoder_dataloader(
            cfg,
            file_paths,
            shuffle=False,
            max_samples=max_fit_pointclouds,
            batch_size=getattr(cfg, "batch_size", 512),
        )

        # Stream point clouds from the dataloader into the fitter
        def iter_point_clouds():
            num_seen = 0
            for batch in dl:
                pts = batch[0] if isinstance(batch, (list, tuple)) else batch
                np_pts = pts.detach().cpu().numpy()
                # np_pts is (B, N, 3); yield each (N, 3)
                for xyz in np_pts:
                    yield xyz
                    num_seen += 1
                    if num_seen >= max_fit_pointclouds:
                        return

        soap, pca = fit_soap_pca(
            iter_point_clouds(),
            species=species,
            soap_params=soap_params,
            n_components=n_components,
            n_jobs=1,
            verbose=False,
        )
        return cls(soap, pca, species)

    def predict(self, dataloader: torch.utils.data.DataLoader) -> PredictionBundle:  # type: ignore[override]
        from src.training_methods.SOAP.predict_soap_pca import soap_pca_predict_latent

        lats, origs = [], []
        for batch in dataloader:
            pts = batch[0] if isinstance(batch, (list, tuple)) else batch
            np_pts = pts.detach().cpu().numpy()
            lat = soap_pca_predict_latent(np_pts, soap=self.soap, pca=self.pca, species=self.species)
            lats.append(lat)          # (B, d)
            origs.append(np_pts)      # (B, N, 3)
        return PredictionBundle(
            latents=np.concatenate(lats, axis=0),
            reconstructions=None,
            originals=np.concatenate(origs, axis=0),
        )

    def predict_raw(self, points: np.ndarray) -> np.ndarray:  # type: ignore[override]
        from src.training_methods.SOAP.predict_soap_pca import soap_pca_predict_latent

        return soap_pca_predict_latent(points, soap=self.soap, pca=self.pca, species=self.species)


PREDICTOR_REGISTRY: Dict[str, Type[Predictor]] = {
    "autoencoder": AutoencoderPredictor,
    "spd": SPDPredictor,
    "soap": SOAPPredictor,
}
