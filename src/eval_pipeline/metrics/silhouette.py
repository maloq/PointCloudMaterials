from __future__ import annotations

"""Silhouette score metric."""

import numpy as np
from sklearn.metrics import silhouette_score

from .base import SingleRunMetric


class SilhouetteMetric(SingleRunMetric):
    """Compute the silhouette score from latent representations."""

    def __init__(self) -> None:
        super().__init__(name="silhouette")

    def run_once(
        self, predictions: np.ndarray, *, cluster_labels: np.ndarray, **_: object
    ) -> float:  # type: ignore[override]
        if len(predictions) == 0:
            return float("nan")
        return float(silhouette_score(predictions, cluster_labels))
