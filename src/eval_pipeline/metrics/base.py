from __future__ import annotations

"""Base classes for evaluation metrics."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Metric(ABC):
    """Abstract metric base class.

    Subclasses implement :meth:`run_once` which returns the metric value.
    """

    name: str

    @abstractmethod
    def run_once(self, predictions, **kwargs) -> Any:
        """Compute the metric.

        Parameters
        ----------
        predictions:
            Typically the latent representations produced by a model.
        **kwargs:
            Additional metric specific arguments.
        """


class SingleRunMetric(Metric):
    """Metric that only needs a single forward pass."""

    pass


class MultiRunMetric(Metric):
    """Metric that may query the model multiple times.

    The metric receives a :class:`~src.eval_pipeline.predictors.Predictor`
    instance which can be used to run further predictions internally.
    """

    def __init__(self, name: str, predictor, **params):
        super().__init__(name)
        self.predictor = predictor
        self.params = params
