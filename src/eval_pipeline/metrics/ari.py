from __future__ import annotations

"""Adjusted Rand Index metric."""

import numpy as np
from sklearn.metrics import adjusted_rand_score

from .base import SingleRunMetric


class ARIMetric(SingleRunMetric):
    """Compute ARI between predicted clusters and ground truth labels."""

    def __init__(self) -> None:
        super().__init__(name="ari")

    def run_once(
        self,
        predictions: np.ndarray,
        *,
        true_labels: np.ndarray,
        cluster_labels: np.ndarray,
        **_: object,
    ) -> float:  # type: ignore[override]
        if len(predictions) == 0:
            return float("nan")
        return float(adjusted_rand_score(true_labels, cluster_labels))
