from __future__ import annotations

import numpy as np
import torch

from .base import SingleRunMetric
from src.loss.reconstruction_loss import sinkhorn_distance, sliced_wasserstein_distance


class ReconstructionEMDMetric(SingleRunMetric):
    """Compute the mean EMD (Sinkhorn) between reconstructions and originals."""
    requires_clustering = False

    def __init__(self) -> None:
        super().__init__(name="reconstruction_emd")

    def run_once(
        self,
        predictions: np.ndarray,
        *,
        reconstructions: np.ndarray | None = None,
        originals: np.ndarray | None = None,
        **_: object,
    ) -> float:  # type: ignore[override]
        if reconstructions is None or originals is None:
            return float("nan")
        if len(originals) == 0 or len(reconstructions) == 0:
            return float("nan")

        rec_t = torch.from_numpy(reconstructions).float()
        orig_t = torch.from_numpy(originals).float()

        try:
            import geomloss  # type: ignore
            dist, _ = sinkhorn_distance(rec_t, orig_t)
            return float(dist.item())
        except Exception:
            dist = sliced_wasserstein_distance(rec_t, orig_t, num_projections=64, p=2)
            return float(dist.item())
