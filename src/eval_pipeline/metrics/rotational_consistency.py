from __future__ import annotations

"""Rotational consistency metric."""

from typing import Any
import numpy as np

from .base import MultiRunMetric
from ..predictors import Predictor
from src.eval_tools.rotational_stability import rotational_robustness, Predictor as _RSPredictor


class _Wrapper(_RSPredictor):
    """Adapts our :class:`Predictor` to the interface expected by
    :func:`rotational_robustness`."""

    def __init__(self, predictor: Predictor):
        self._pred = predictor

    def predict(self, pcs: np.ndarray) -> np.ndarray:  # type: ignore[override]
        return self._pred.predict_raw(pcs)


class RotationalConsistencyMetric(MultiRunMetric):
    """Evaluate rotational robustness of a predictor."""

    def __init__(self, predictor: Predictor, **params: Any):
        super().__init__(name="rotational_consistency", predictor=predictor, **params)

    def run_once(self, _predictions, *, points: np.ndarray, **_: Any) -> float:  # type: ignore[override]
        wrapper = _Wrapper(self.predictor)
        score, _ = rotational_robustness(wrapper, points, **self.params)
        return float(score)
