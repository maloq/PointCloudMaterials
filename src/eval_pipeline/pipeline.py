from __future__ import annotations

"""High level evaluation pipeline."""

from typing import Any, Dict, List
import os
import sys,os
sys.path.append(os.getcwd())

import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig

from src.eval_pipeline.predictors import PREDICTOR_REGISTRY, Predictor
from src.eval_pipeline.metrics.silhouette import SilhouetteMetric
from src.eval_pipeline.metrics.ari import ARIMetric
from src.eval_pipeline.metrics.rotational_consistency import RotationalConsistencyMetric
from src.eval_pipeline.metrics.base import SingleRunMetric, MultiRunMetric
from src.eval_pipeline.config_merger import load_training_config, merge_and_save


SINGLE_METRICS = {
    "silhouette": SilhouetteMetric,
    "ari": ARIMetric,
}

MULTI_METRICS = {
    "rotational_consistency": RotationalConsistencyMetric,
}


def _build_dataloader(cfg: DictConfig, model_type: str, batch_size: int | None = None):
    file_paths = cfg.data.data_files
    if model_type == "autoencoder":
        from src.training_methods.autoencoder.eval_autoencoder import (
            create_autoencoder_dataloader,
        )

        return create_autoencoder_dataloader(cfg, file_paths, shuffle=False, batch_size=batch_size)
    if model_type == "spd":
        from src.training_methods.spd.eval_spd import create_spd_dataloader

        return create_spd_dataloader(cfg, file_paths, shuffle=False, batch_size=batch_size)
    if model_type == "soap":
        from src.training_methods.autoencoder.eval_autoencoder import (
            create_autoencoder_dataloader,
        )

        return create_autoencoder_dataloader(cfg, file_paths, shuffle=False, batch_size=batch_size)
    raise ValueError(f"Unknown model_type {model_type!r}")


def _cluster_latents(latents: np.ndarray, method: str, params: Dict[str, Any]) -> np.ndarray:
    """Cluster *latents* using *method* with given *params*.

    Supports ``kmeans``, ``gmm`` and ``hdbscan``.
    """

    method = method.lower()
    if method == "kmeans":
        from sklearn.cluster import KMeans

        model = KMeans(n_init="auto", **params)
    elif method == "gmm":
        from sklearn.mixture import GaussianMixture

        model = GaussianMixture(**params)
    elif method == "hdbscan":
        from hdbscan import HDBSCAN

        model = HDBSCAN(**params)
    else:
        raise ValueError(f"Unknown clustering method {method!r}")
    return model.fit_predict(latents)


def run(eval_config_path: str) -> Dict[str, Any]:
    """Execute the evaluation pipeline described in *eval_config_path*."""

    eval_cfg = OmegaConf.load(eval_config_path)
    checkpoint = eval_cfg.prediction.checkpoint
    train_cfg = load_training_config(checkpoint)

    # ------------------------------------------------------------------
    # Compose dataset config (training data cfg + overrides from eval cfg)
    # ------------------------------------------------------------------
    dataset_cfg = OmegaConf.load(eval_cfg.dataset.config)
    if eval_cfg.dataset.get("override"):
        dataset_cfg = OmegaConf.merge(dataset_cfg, eval_cfg.dataset.override)
    train_cfg.data = dataset_cfg

    device = (
        f"cuda:{eval_cfg.prediction.cuda_device}"
        if (eval_cfg.prediction.cuda_device is not None and torch.cuda.is_available())
        else "cpu"
    )

    PredictorCls = PREDICTOR_REGISTRY[eval_cfg.prediction.model_type]
    predictor: Predictor = PredictorCls.from_checkpoint(train_cfg, checkpoint, device)

    batch_size_override = eval_cfg.prediction.get("batch_size")
    dataloader = _build_dataloader(train_cfg, eval_cfg.prediction.model_type, batch_size=batch_size_override)
    print("Predicting...")
    preds = predictor.predict(dataloader)
    results: Dict[str, Any] = {
        "latents": preds.latents,
        "reconstructions": preds.reconstructions,
        "metrics": {},
    }
    print("Predictions done")
    metrics_cfg = eval_cfg.get("metrics", {})
    single_run_cfg = metrics_cfg.get("single_run", [])

    cluster_labels = None
    if single_run_cfg:
        clustering_method = metrics_cfg.get("clustering_method", "kmeans")
        clustering_params = metrics_cfg.get("clustering_params", {})
        cluster_labels = _cluster_latents(
            preds.latents, clustering_method, clustering_params
        )

    for m in single_run_cfg:
        print("Running single run metric:", m.name)
        cls = SINGLE_METRICS[m.name]
        metric: SingleRunMetric = cls()
        params = m.get("params", {})
        results["metrics"][m.name] = metric.run_once(
            preds.latents, cluster_labels=cluster_labels, **params
        )

    print("Single run metrics done")
    for m in metrics_cfg.get("multi_run", []):
        print("Running multi run metric:", m.name)
        cls = MULTI_METRICS[m.name]
        metric: MultiRunMetric = cls(predictor, **m.get("params", {}))
        results["metrics"][m.name] = metric.run_once(
            preds.latents, points=preds.originals, **m.get("params", {})
        )
    print("Multi run metrics done")
    merge_and_save(train_cfg, eval_cfg, results["metrics"], checkpoint_path=checkpoint)
    print("Results saved")
    return results

if __name__ == "__main__":
    run("configs/eval_configs/eval_expl.yaml")